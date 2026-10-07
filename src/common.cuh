// Shared CUDA helpers.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + #x + " -> " + cudaGetErrorString(e_)); } } while (0)

namespace q27 {

// ---- Programmatic dependent launch (PDL). Every kernel launched with launch_k may start before the previous
// kernel in the stream ends; it must call pdl_wait() before it reads or writes anything the previous kernels
// touch, and calls pdl_trigger() so the next kernel can start loading constant data (weights).
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
// Q27_PDL=0 in the environment turns PDL off (plain stream order).
bool pdl_enabled();
// One-shot: the next launch_k on this host thread uses PDL even when Q27_PDL is off (used for the GEMV after a
// cross-card sum, so it can fetch weights while the sum waits on the link).
bool& pdl_once();

template <typename... KArgs, typename... Args>
inline void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = (pdl_enabled() || pdl_once()) ? 1 : 0;
  pdl_once() = false;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, kern, std::forward<Args>(args)...));
}

// L2 bulk prefetch of [p, p + bytes) spread over `nthr` threads (index t), 64 KiB pieces (sm_90+ TMA prefetch).
__device__ __forceinline__ void l2_prefetch_range(const void* p, size_t bytes, int t, int nthr) {
  const size_t piece = 65536, npieces = (bytes + piece - 1) / piece;
  for (size_t i = t; i < npieces; i += nthr) {
    const size_t off = i * piece;
    const uint32_t sz = (uint32_t)(bytes - off < piece ? bytes - off : piece) & ~15u;
    if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"((const char*)p + off), "r"(sz) : "memory");
  }
}

// Streaming 128-bit load for weights: read-only path, no L1 allocation.
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ld_stream2(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_stream1(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// q8_1 of one value per thread; the 32 lanes of the warp hold 32 consecutive values starting at a multiple
// of 32 (index i of lane 0 .. i+31). Same numerics as quantize_q8_1 (qgemv.cu, llama.cpp).
__device__ __forceinline__ void q8_store(float y, int8_t* __restrict__ xq, float* __restrict__ xd, size_t i,
                                         float* __restrict__ xs = nullptr) {
  float amax = fabsf(y);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float d = amax / 127.0f;
  xq[i] = amax == 0.0f ? 0 : (int8_t)roundf(y / d);
  if ((threadIdx.x & 31) == 0) xd[i / 32] = __half2float(__float2half(d));
  if (xs) {  // sum of the unquantized values per 16 (for the prefill GEMM of types with a min term)
    float sum = y;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 15) == 0) xs[i / 16] = sum;
  }
}

}  // namespace q27
