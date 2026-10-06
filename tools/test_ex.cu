// Cross-card exchange test for prefill sums: each card's kernel writes its rows (BF16) to pinned host memory and
// raises a flag; the other card's copy engine waits for that flag on the GPU (cuStreamWaitValue32) and copies the
// rows into device memory (H2D). Times one full exchange on both cards, with `chunks` flags per exchange.
// Usage: test_ex [rows=512] [chunks=4] [reps=20]
#include "common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

#include <chrono>
#include <cstdio>
#include <vector>

typedef CUresult(CUDAAPI* PFN_wait32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

__global__ void send_kernel(const float* __restrict__ part, __nv_bfloat16* host, int* flags, int n, int chunks, int token) {
  // block b handles chunk b % chunks; each chunk is written by gridDim/chunks blocks, then its flag is set by
  // the last block to finish it (atomic counter in flags[chunk*32+1]).
  const int per = (n + chunks - 1) / chunks;
  const int ch = blockIdx.x % chunks, sub = blockIdx.x / chunks, nsub = gridDim.x / chunks;
  const int beg = ch * per, end = min(n, beg + per);
  for (int i = beg + (sub * blockDim.x + threadIdx.x) * 2; i < end; i += nsub * blockDim.x * 2) {
    const float2 p = *(const float2*)(part + i);
    *(__nv_bfloat162*)(host + i) = __floats2bfloat162_rn(p.x, p.y);
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int done = atomicAdd(flags + ch * 32 + 1, 1) + 1;
    if (done == nsub) { flags[ch * 32 + 1] = 0; __threadfence_system(); *(volatile int*)(flags + ch * 32) = token; }
  }
}

int main(int argc, char** argv) try {
  const int rows = argc > 1 ? atoi(argv[1]) : 512;
  const int chunks = argc > 2 ? atoi(argv[2]) : 4;
  const int reps = argc > 3 ? atoi(argv[3]) : 20;
  const int n = rows * 5120;
  PFN_wait32 wait32 = nullptr;
  cudaDriverEntryPointQueryResult q;
  CK(cudaGetDriverEntryPoint("cuStreamWaitValue32", (void**)&wait32, cudaEnableDefault, &q));
  if (!wait32) throw std::runtime_error("cuStreamWaitValue32 not found");
  __nv_bfloat16* hdata[2];
  int* hflags[2];
  for (int r = 0; r < 2; r++) {
    CK(cudaHostAlloc(&hdata[r], (size_t)n * 2, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&hflags[r], chunks * 32 * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(hflags[r], 0, chunks * 32 * 4);
  }
  struct C { cudaStream_t s, c; float* part; __nv_bfloat16* recv; cudaEvent_t ev; };
  C c[2];
  for (int r = 0; r < 2; r++) {
    CK(cudaSetDevice(r));
    int attr = 0;
    CK(cudaDeviceGetAttribute(&attr, cudaDevAttrCanUseHostPointerForRegisteredMem, r));
    CK(cudaStreamCreateWithFlags(&c[r].s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&c[r].c, cudaStreamNonBlocking));
    CK(cudaMalloc(&c[r].part, (size_t)n * 4));
    CK(cudaMemset(c[r].part, 0, (size_t)n * 4));
    CK(cudaMalloc(&c[r].recv, (size_t)n * 2));
    CK(cudaEventCreateWithFlags(&c[r].ev, cudaEventDisableTiming));
  }
  const int per = (n + chunks - 1) / chunks;
  int token = 0;
  auto exchange = [&]() {
    token++;
    for (int r = 0; r < 2; r++) {
      CK(cudaSetDevice(r));
      send_kernel<<<chunks * 4, 1024, 0, c[r].s>>>(c[r].part, hdata[r], hflags[r], n, chunks, token);
      const int o = 1 - r;
      for (int k = 0; k < chunks; k++) {
        CUdeviceptr fp;
        CK(cudaHostGetDevicePointer((void**)&fp, hflags[o] + k * 32, 0));
        if (wait32((CUstream)c[r].c, fp, (cuuint32_t)token, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS)
          throw std::runtime_error("cuStreamWaitValue32 failed");
        const int cnt = std::min(per, n - k * per);
        CK(cudaMemcpyAsync(c[r].recv + (size_t)k * per, hdata[o] + (size_t)k * per, (size_t)cnt * 2, cudaMemcpyHostToDevice, c[r].c));
      }
      CK(cudaEventRecord(c[r].ev, c[r].c));
      CK(cudaStreamWaitEvent(c[r].s, c[r].ev, 0));
      cudaStreamQuery(c[r].s); cudaStreamQuery(c[r].c);
    }
    for (int r = 0; r < 2; r++) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(c[r].s)); }
  };
  exchange();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; i++) exchange();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  printf("rows %d (%.1f MB per direction), chunks %d: %.2f ms per exchange (%.2f GB/s per direction)\n", rows, n * 2 / 1e6,
         chunks, ms, n * 2 / 1e6 / ms);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
