// PCIe link test with kernels on mapped pinned host memory: GPU writes to host, GPU reads from host, and both
// at the same time (two streams), on one card and on both cards.
// Usage: test_link [MiB=64] [blocks=16]
#include "common.cuh"

#include <chrono>
#include <cstdio>
#include <vector>

__global__ void write_host(uint4* dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = make_uint4((unsigned)i, 1, 2, 3);
}
__global__ void read_host(const uint4* src, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const volatile uint4* p = (const volatile uint4*)src + i;
    acc += p->x ^ p->w;
  }
  if (acc == 0x12345678u) *sink = acc;
}

int main(int argc, char** argv) try {
  const size_t mib = argc > 1 ? atoi(argv[1]) : 64;
  const int blocks = argc > 2 ? atoi(argv[2]) : 16;
  const size_t bytes = mib << 20, n = bytes / 16;
  uint4 *hw[2], *hr[2];
  for (int d = 0; d < 2; d++) {
    CK(cudaHostAlloc(&hw[d], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&hr[d], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(hr[d], 1, bytes);
  }
  cudaStream_t s[2][2];
  unsigned* sink[2];
  for (int d = 0; d < 2; d++) {
    CK(cudaSetDevice(d));
    CK(cudaStreamCreateWithFlags(&s[d][0], cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s[d][1], cudaStreamNonBlocking));
    CK(cudaMalloc(&sink[d], 4));
    write_host<<<blocks, 1024, 0, s[d][0]>>>(hw[d], n);  // warm-up
    read_host<<<blocks, 1024, 0, s[d][1]>>>(hr[d], n, sink[d]);
    CK(cudaDeviceSynchronize());
  }
  auto run = [&](bool w, bool r, int ncards) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int d = 0; d < ncards; d++) {
      CK(cudaSetDevice(d));
      if (w) write_host<<<blocks, 1024, 0, s[d][0]>>>(hw[d], n);
      if (r) read_host<<<blocks, 1024, 0, s[d][1]>>>(hr[d], n, sink[d]);
      cudaStreamQuery(s[d][0]); cudaStreamQuery(s[d][1]);
    }
    for (int d = 0; d < ncards; d++) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gb = (double)bytes * ((w ? 1 : 0) + (r ? 1 : 0)) * ncards / 1e9;
    printf("cards %d  write %d read %d: %.2f ms, %.2f GB/s total\n", ncards, w, r, sec * 1e3, gb / sec);
  };
  for (int nc : {1, 2}) {
    run(true, false, nc);
    run(false, true, nc);
    run(true, true, nc);
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
