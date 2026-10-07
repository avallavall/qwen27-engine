// Link limits for the cross-card sum: kernels that read / write mapped pinned host memory, run as a CUDA graph of
// many launches. Variants: load width and type, blocks, threads, read and write at the same time, both cards.
// Usage: bench_link2 [bytes=40960]
#include "common.cuh"

#include <chrono>
#include <cstdio>
#include <vector>

namespace {

__device__ __forceinline__ uint4 ld_sys16(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_vol16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cv16(const void* p) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned short ld_vol2(const void* p) {
  return *(volatile const unsigned short*)p;
}

// mode: 0 = read 16B relaxed.sys, 1 = read 16B volatile, 2 = read 16B cv, 3 = read 2B volatile, 4 = write 16B,
// 5 = half the blocks write, half read (16B relaxed.sys)
template <int MODE>
__global__ void k_link(const uint8_t* src, uint8_t* dst, int bytes, unsigned* sink) {
  unsigned acc = 0;
  int nb = gridDim.x, b = blockIdx.x;
  bool wr = MODE == 4;
  if (MODE == 5) { nb = gridDim.x / 2; wr = b >= nb; b = wr ? b - nb : b; }
  if (MODE == 3) {
    for (int i = (b * blockDim.x + threadIdx.x) * 2; i < bytes; i += nb * blockDim.x * 2) acc += ld_vol2(src + i);
  } else {
    const int n16 = bytes / 16;
    // 4 independent loads in flight per thread
    for (int i0 = b * blockDim.x + threadIdx.x; i0 < n16; i0 += nb * blockDim.x * 4) {
      if (wr) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int i = i0 + u * nb * blockDim.x;
          if (i < n16) ((uint4*)dst)[i] = make_uint4(i, 1, 2, 3);
        }
      } else {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int i = i0 + u * nb * blockDim.x;
          if (i < n16) v[u] = MODE == 1 ? ld_vol16(src + 16 * (size_t)i) : MODE == 2 ? ld_cv16(src + 16 * (size_t)i) : ld_sys16(src + 16 * (size_t)i);
          else v[u] = make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) acc += v[u].x ^ v[u].w;
      }
    }
  }
  if (wr) __threadfence_system();
  if (acc == 0x12345678u) *sink = acc;
}

typedef void (*KFn)(const uint8_t*, uint8_t*, int, unsigned*);
KFn kfn(int mode) {
  switch (mode) {
    case 0: return k_link<0>; case 1: return k_link<1>; case 2: return k_link<2>; case 3: return k_link<3>;
    case 4: return k_link<4>; default: return k_link<5>;
  }
}
const char* mname(int m) {
  static const char* n[] = {"read16 relaxed.sys", "read16 volatile", "read16 cv", "read2 volatile", "write16", "write+read16"};
  return n[m];
}

}  // namespace

int main(int argc, char** argv) try {
  const int bytes = argc > 1 ? atoi(argv[1]) : 40960;
  const int reps = 200;
  uint8_t *hr[2], *hw[2];
  for (int d = 0; d < 2; d++) {
    CK(cudaHostAlloc(&hr[d], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&hw[d], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(hr[d], 1, bytes);
  }
  cudaStream_t s[2];
  unsigned* sink[2];
  for (int d = 0; d < 2; d++) {
    CK(cudaSetDevice(d));
    CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
    CK(cudaMalloc(&sink[d], 4));
  }
  // graph of `reps` launches of one variant, per card
  auto make = [&](int d, int mode, int blocks, int threads) {
    CK(cudaSetDevice(d));
    cudaGraph_t g;
    cudaGraphExec_t e;
    CK(cudaStreamBeginCapture(s[d], cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < reps; i++) kfn(mode)<<<blocks, threads, 0, s[d]>>>(hr[d], hw[d], bytes, sink[d]);
    CK(cudaStreamEndCapture(s[d], &g));
    CK(cudaGraphInstantiate(&e, g, 0));
    CK(cudaGraphDestroy(g));
    return e;
  };
  auto time_it = [&](int ncards, int mode, int blocks, int threads) {
    cudaGraphExec_t e[2] = {};
    for (int d = 0; d < ncards; d++) e[d] = make(d, mode, blocks, threads);
    double best = 1e30;
    for (int it = 0; it < 4; it++) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int d = 0; d < ncards; d++) { CK(cudaSetDevice(d)); CK(cudaGraphLaunch(e[d], s[d])); cudaStreamQuery(s[d]); }
      for (int d = 0; d < ncards; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
      if (it) best = std::min(best, us);
    }
    for (int d = 0; d < ncards; d++) { CK(cudaSetDevice(d)); CK(cudaGraphExecDestroy(e[d])); }
    const double moved = (mode == 5 ? 2.0 : 1.0) * bytes;
    printf("cards %d %-20s blocks %3d x %4d: %7.2f us per kernel, %5.2f GB/s per card\n", ncards, mname(mode), blocks, threads, best,
           moved / best / 1e3);
  };
  printf("payload %d bytes, %d launches per graph\n", bytes, reps);
  for (int mode : {3, 0, 1, 2, 4})
    for (int blocks : {4, 8, 16, 32, 64})
      time_it(1, mode, blocks, mode == 3 ? 1024 : 256);
  for (int blocks : {8, 16, 32, 64}) time_it(1, 5, blocks, 256);
  for (int blocks : {8, 16, 32}) { time_it(2, 0, blocks, 256); time_it(2, 4, blocks, 256); time_it(2, 5, 2 * blocks, 256); }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
