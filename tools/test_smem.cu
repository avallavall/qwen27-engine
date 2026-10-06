// Shared memory check: each block fills its dynamic shared memory with a pattern, waits, reads it back.
// Usage: test_smem [device=1] [bytes=64512] [blocks=72]
#include "common.cuh"

#include <cstdio>

__global__ void smem_kernel(int words, unsigned* errors, unsigned* first_bad) {
  extern __shared__ unsigned sm[];
  const unsigned tag = blockIdx.x * 0x10000u;
  for (int r = 0; r < 50; r++) {
    for (int i = threadIdx.x; i < words; i += blockDim.x) sm[i] = tag + i + r * 7;
    __syncthreads();
    // spin a little so blocks overlap in time
    long long t0 = clock64();
    while (clock64() - t0 < 2000) {}
    for (int i = threadIdx.x; i < words; i += blockDim.x)
      if (sm[i] != tag + i + r * 7) { atomicAdd(errors, 1u); atomicMin(first_bad, (unsigned)i * 4); }
    __syncthreads();
  }
}

int main(int argc, char** argv) try {
  const int dev = argc > 1 ? atoi(argv[1]) : 1;
  const int bytes = argc > 2 ? atoi(argv[2]) : 64512;
  const int blocks = argc > 3 ? atoi(argv[3]) : 72;
  CK(cudaSetDevice(dev));
  unsigned *err, *first;
  CK(cudaMallocManaged(&err, 4));
  CK(cudaMallocManaged(&first, 4));
  *err = 0; *first = 0xffffffffu;
  CK(cudaFuncSetAttribute(smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, smem_kernel, 256, bytes));
  smem_kernel<<<blocks, 256, bytes>>>(bytes / 4, err, first);
  CK(cudaDeviceSynchronize());
  printf("device %d, %d bytes, %d blocks, occupancy %d blocks/SM: %u errors, first bad byte offset %u\n", dev, bytes, blocks, occ,
         *err, *first == 0xffffffffu ? 0 : *first);
  return *err ? 1 : 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
