// hw.cu: measure the hardware limits of this rig (2x RTX 5060 Ti, PCIe Gen3 x4, no NVLink).
// Build: bench\build.bat   Run: bench\build\hw.exe <info|bw|seq|launch|pcie|ar|all>
// Every number is printed as "key value unit" lines so it can go straight into PLAN.md.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static int g_ndev = 0;

// ---------------------------------------------------------------- info
static void info() {
  int drv, rt; CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
  printf("info.devices %d\ninfo.driver_api %d\ninfo.runtime %d\n", g_ndev, drv, rt);
  for (int d = 0; d < g_ndev; d++) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
    int clk_khz = 0, memclk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, d);
    cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, d);
    printf("info.dev%d %s cc=%d.%d sms=%d l2_kib=%d smem_per_sm_kib=%zu smem_block_optin_kib=%zu regs_per_sm=%d "
           "bus_bits=%d clk_khz=%d memclk_khz=%d pci=%02x:%02x tcc=%d async_engines=%d persist_l2_max_kib=%d "
           "access_policy_max_window_mib=%d mem_mib=%zu\n",
           d, p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1024, p.sharedMemPerMultiprocessor / 1024,
           p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor, p.memoryBusWidth, clk_khz, memclk_khz, p.pciBusID,
           p.pciDeviceID, p.tccDriver, p.asyncEngineCount, p.persistingL2CacheMaxSize / 1024,
           p.accessPolicyMaxWindowSize >> 20, p.totalGlobalMem >> 20);
  }
  for (int a = 0; a < g_ndev; a++)
    for (int b = 0; b < g_ndev; b++) {
      if (a == b) continue;
      int can = 0, acc = 0, native = 0;
      CK(cudaDeviceCanAccessPeer(&can, a, b));
      cudaDeviceGetP2PAttribute(&acc, cudaDevP2PAttrAccessSupported, a, b);
      cudaDeviceGetP2PAttribute(&native, cudaDevP2PAttrNativeAtomicSupported, a, b);
      printf("info.p2p dev%d->dev%d can_access=%d access_supported=%d native_atomic=%d\n", a, b, can, acc, native);
    }
}

// ---------------------------------------------------------------- read bandwidth
__device__ __forceinline__ int4 ldnc(const int4* p) {
  int4 v;
  asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// Grid-stride streaming read with 4 loads in flight per thread.
__global__ void read_k(const int4* __restrict__ p, size_t n, int* out) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int acc = 0;
  for (; i + 3 * stride < n; i += 4 * stride) {
    int4 a = ldnc(p + i), b = ldnc(p + i + stride), c = ldnc(p + i + 2 * stride), d = ldnc(p + i + 3 * stride);
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n; i += stride) { int4 a = ldnc(p + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  if (acc == 0x7fffffff) out[0] = acc;
}

// GEMV-like: one warp per row of row_bytes (multiple of 16). Lanes stride the row with 16-byte loads.
__global__ void rows_k(const int4* __restrict__ p, int rows, int row_v4, int* out) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  int acc = 0;
  for (int r = warp; r < rows; r += nwarps) {
    const int4* row = p + (size_t)r * row_v4;
    for (int j = lane; j < row_v4; j += 32) { int4 a = ldnc(row + j); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x7fffffff) out[0] = acc;
}

static void bw() {
  for (int d = 0; d < g_ndev; d++) {
    CK(cudaSetDevice(d));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
    size_t bytes = (size_t)4 << 30;
    void* buf; int* out; CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&out, 4));
    CK(cudaMemset(buf, 1, bytes));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double best = 0; int best_b = 0, best_t = 0;
    for (int t : {256, 512, 1024})
      for (int bm : {2, 4, 8, 16, 32}) {
        int blocks = p.multiProcessorCount * bm * 256 / t;
        if (blocks < p.multiProcessorCount) continue;
        read_k<<<blocks, t>>>((const int4*)buf, bytes / 16, out);
        std::vector<double> v;
        for (int it = 0; it < 5; it++) {
          CK(cudaEventRecord(e0)); read_k<<<blocks, t>>>((const int4*)buf, bytes / 16, out); CK(cudaEventRecord(e1));
          CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
        }
        double gbs = bytes / (median(v) * 1e6);
        if (gbs > best) { best = gbs; best_b = blocks; best_t = t; }
      }
    printf("bw.dev%d.stream_read_4GiB %.1f GB/s (blocks=%d threads=%d)\n", d, best, best_b, best_t);

    // GEMV-like rows: IQ3_S rows of 5120 weights = 2200 B (padded 2208), of 17408 weights = 7480 B (padded 7488).
    for (int row_bytes : {2208, 7488}) {
      int rows = (int)(bytes / row_bytes);
      double rb = 0; int rbb = 0;
      for (int bm : {4, 8, 16, 32}) {
        int blocks = p.multiProcessorCount * bm;
        rows_k<<<blocks, 256>>>((const int4*)buf, rows, row_bytes / 16, out);
        std::vector<double> v;
        for (int it = 0; it < 5; it++) {
          CK(cudaEventRecord(e0)); rows_k<<<blocks, 256>>>((const int4*)buf, rows, row_bytes / 16, out); CK(cudaEventRecord(e1));
          CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
        }
        double gbs = (double)rows * row_bytes / (median(v) * 1e6);
        if (gbs > rb) { rb = gbs; rbb = blocks; }
      }
      printf("bw.dev%d.rows_%dB %.1f GB/s (blocks=%d)\n", d, row_bytes, rb, rbb);
    }
    CK(cudaFree(buf)); CK(cudaFree(out));
  }
}

// ---------------------------------------------------------------- per-kernel overhead on realistic sizes
// Read ~2 GiB as a sequence of kernels of chunk_mib each (like one GEMV per weight tensor), in a stream
// and in a CUDA graph. Effective GB/s shows the cost of ramp-up, tail and launch gaps.
static void seq() {
  for (int d = 0; d < g_ndev; d++) {
    CK(cudaSetDevice(d));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
    size_t total = (size_t)2 << 30;
    char* buf; int* out; CK(cudaMalloc(&buf, total)); CK(cudaMalloc(&out, 4)); CK(cudaMemset(buf, 1, total));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int blocks = p.multiProcessorCount * 8, threads = 256;
    for (double chunk_mib : {0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0}) {
      size_t cb = (size_t)(chunk_mib * (1 << 20));
      int n = (int)(total / cb);
      auto enqueue = [&]() {
        for (int k = 0; k < n; k++) read_k<<<blocks, threads, 0, s>>>((const int4*)(buf + k * cb), cb / 16, out);
      };
      enqueue(); CK(cudaStreamSynchronize(s));
      std::vector<double> v;
      for (int it = 0; it < 5; it++) {
        CK(cudaEventRecord(e0, s)); enqueue(); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
      }
      double ms_stream = median(v);
      cudaGraph_t g; cudaGraphExec_t ge;
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); enqueue(); CK(cudaStreamEndCapture(s, &g));
      CK(cudaGraphInstantiate(&ge, g, 0));
      CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
      v.clear();
      for (int it = 0; it < 5; it++) {
        CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
      }
      double ms_graph = median(v);
      printf("seq.dev%d.chunk_%.1fMiB kernels=%d stream %.1f GB/s (%.2f us/kernel) graph %.1f GB/s (%.2f us/kernel)\n", d,
             chunk_mib, n, (double)n * cb / (ms_stream * 1e6), ms_stream * 1e3 / n, (double)n * cb / (ms_graph * 1e6),
             ms_graph * 1e3 / n);
      CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    CK(cudaFree(buf)); CK(cudaFree(out)); CK(cudaStreamDestroy(s));
  }
}

// ---------------------------------------------------------------- launch costs
__global__ void empty_k() {}

// Small dependent kernel: reads a slice and writes one value (like a tiny norm / add kernel).
__global__ void small_k(const float* __restrict__ in, float* __restrict__ outp, int n) {
  float acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += in[i];
  if (acc == 12345.f) outp[0] = acc;
}
__global__ void small_pdl_k(const float* __restrict__ in, float* __restrict__ outp, int n) {
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += in[i];
  if (acc == 12345.f) outp[0] = acc;
}

static void launch() {
  for (int d = 0; d < g_ndev; d++) {
    CK(cudaSetDevice(d));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int N = 2000;
    for (int i = 0; i < 100; i++) empty_k<<<1, 32, 0, s>>>();
    CK(cudaStreamSynchronize(s));
    // host submit cost and GPU throughput of back-to-back empty kernels
    auto t0 = clk::now();
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < N; i++) empty_k<<<1, 32, 0, s>>>();
    double submit = us_since(t0) / N;
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("launch.dev%d.empty_submit_host %.2f us/launch\n", d, submit);
    printf("launch.dev%d.empty_stream_gpu %.2f us/kernel\n", d, ms * 1e3 / N);
    // round trip: launch + synchronize
    std::vector<double> v;
    for (int i = 0; i < 500; i++) { auto t = clk::now(); empty_k<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s)); v.push_back(us_since(t)); }
    printf("launch.dev%d.launch_plus_sync_roundtrip %.1f us (median)\n", d, median(v));
    v.clear();
    for (int i = 0; i < 500; i++) { auto t = clk::now(); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); v.push_back(us_since(t)); }
    printf("launch.dev%d.event_record_plus_sync %.1f us (median)\n", d, median(v));
    // graph of N empty kernels
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < N; i++) empty_k<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    v.clear();
    for (int it = 0; it < 5; it++) {
      auto t = clk::now();
      CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); double sub = us_since(t); CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms * 1e3 / N);
      if (it == 4) printf("launch.dev%d.graph_launch_host %.1f us per graph of %d nodes\n", d, sub, N);
    }
    printf("launch.dev%d.empty_graph_gpu %.2f us/node\n", d, median(v));
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));

    // small dependent kernels (SMs blocks x 256 threads, reading 256 KiB each), stream vs graph vs PDL
    int n = 64 * 1024; float *in, *outp; CK(cudaMalloc(&in, n * 4)); CK(cudaMalloc(&outp, 4)); CK(cudaMemset(in, 0, n * 4));
    int blocks = p.multiProcessorCount;
    auto run_plain = [&]() { for (int i = 0; i < N; i++) small_k<<<blocks, 256, 0, s>>>(in, outp, n); };
    auto run_pdl = [&]() {
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      at[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(256); cfg.stream = s;
      cfg.attrs = at; cfg.numAttrs = 1;
      for (int i = 0; i < N; i++) CK(cudaLaunchKernelEx(&cfg, small_pdl_k, (const float*)in, outp, n));
    };
    auto time_it = [&](auto fn, bool graph) {
      cudaGraphExec_t ge2 = nullptr; cudaGraph_t g2 = nullptr;
      if (graph) {
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); fn(); CK(cudaStreamEndCapture(s, &g2));
        CK(cudaGraphInstantiate(&ge2, g2, 0));
      }
      std::vector<double> vv;
      for (int it = 0; it < 6; it++) {
        CK(cudaEventRecord(e0, s));
        if (graph) CK(cudaGraphLaunch(ge2, s)); else fn();
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float m; CK(cudaEventElapsedTime(&m, e0, e1)); if (it) vv.push_back(m * 1e3 / N);
      }
      if (graph) { CK(cudaGraphExecDestroy(ge2)); CK(cudaGraphDestroy(g2)); }
      return median(vv);
    };
    printf("launch.dev%d.small_kernel_stream %.2f us/kernel\n", d, time_it(run_plain, false));
    printf("launch.dev%d.small_kernel_graph %.2f us/kernel\n", d, time_it(run_plain, true));
    printf("launch.dev%d.small_kernel_pdl_stream %.2f us/kernel\n", d, time_it(run_pdl, false));
    printf("launch.dev%d.small_kernel_pdl_graph %.2f us/kernel\n", d, time_it(run_pdl, true));
    CK(cudaFree(in)); CK(cudaFree(outp)); CK(cudaStreamDestroy(s));
  }
}

// ---------------------------------------------------------------- PCIe host transfers
static void pcie() {
  const size_t sizes[] = {4096, 10240, 20480, 40960, 81920, 163840, 1 << 20, 16 << 20, 64 << 20};
  for (int d = 0; d < g_ndev; d++) {
    CK(cudaSetDevice(d));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    void *h, *dv; CK(cudaHostAlloc(&h, 64 << 20, cudaHostAllocPortable)); CK(cudaMalloc(&dv, 64 << 20));
    for (size_t sz : sizes) {
      int iters = sz >= (16 << 20) ? 20 : 300;
      std::vector<double> d2h, h2d;
      for (int i = 0; i < iters; i++) {
        auto t = clk::now(); CK(cudaMemcpyAsync(h, dv, sz, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); d2h.push_back(us_since(t));
        t = clk::now(); CK(cudaMemcpyAsync(dv, h, sz, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); h2d.push_back(us_since(t));
      }
      double a = median(d2h), b = median(h2d);
      printf("pcie.dev%d.%zuB d2h %.1f us (%.2f GB/s) h2d %.1f us (%.2f GB/s)\n", d, sz, a, sz / (a * 1e3), b, sz / (b * 1e3));
    }
    // duplex: both directions at once, 64 MiB each
    cudaStream_t s2; CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    void *h2, *dv2; CK(cudaHostAlloc(&h2, 64 << 20, cudaHostAllocPortable)); CK(cudaMalloc(&dv2, 64 << 20));
    std::vector<double> v;
    for (int i = 0; i < 10; i++) {
      auto t = clk::now();
      CK(cudaMemcpyAsync(h, dv, 64 << 20, cudaMemcpyDeviceToHost, s)); CK(cudaMemcpyAsync(dv2, h2, 64 << 20, cudaMemcpyHostToDevice, s2));
      CK(cudaStreamSynchronize(s)); CK(cudaStreamSynchronize(s2)); v.push_back(us_since(t));
    }
    printf("pcie.dev%d.duplex_64MiB_each_way %.2f GB/s total\n", d, 2.0 * (64 << 20) / (median(v) * 1e3));
    CK(cudaFreeHost(h)); CK(cudaFreeHost(h2)); CK(cudaFree(dv)); CK(cudaFree(dv2));
    CK(cudaStreamDestroy(s)); CK(cudaStreamDestroy(s2));
  }
  if (g_ndev < 2) return;
  // both cards D2H at the same time (separate CPU root ports)
  void *h0, *h1, *d0, *d1; cudaStream_t s0, s1;
  CK(cudaSetDevice(0)); CK(cudaMalloc(&d0, 64 << 20)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaSetDevice(1)); CK(cudaMalloc(&d1, 64 << 20)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaHostAlloc(&h0, 64 << 20, cudaHostAllocPortable)); CK(cudaHostAlloc(&h1, 64 << 20, cudaHostAllocPortable));
  std::vector<double> v;
  for (int i = 0; i < 10; i++) {
    auto t = clk::now();
    CK(cudaSetDevice(0)); CK(cudaMemcpyAsync(h0, d0, 64 << 20, cudaMemcpyDeviceToHost, s0));
    CK(cudaSetDevice(1)); CK(cudaMemcpyAsync(h1, d1, 64 << 20, cudaMemcpyDeviceToHost, s1));
    CK(cudaStreamSynchronize(s0)); CK(cudaStreamSynchronize(s1)); v.push_back(us_since(t));
  }
  printf("pcie.both_cards_d2h_64MiB_each %.2f GB/s total\n", 2.0 * (64 << 20) / (median(v) * 1e3));
  CK(cudaFreeHost(h0)); CK(cudaFreeHost(h1)); CK(cudaFree(d0)); CK(cudaFree(d1));
}

// ---------------------------------------------------------------- 2-GPU all-reduce emulation
__global__ void produce_k(half* x, int n, int salt) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = __float2half((float)((i + salt) & 7));
}
__global__ void add_k(const half* a, const half* b, half* o, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = __hadd(a[i], b[i]);
}

// One-shot all-reduce through mapped pinned host memory. Block b writes slice b of my partial to host,
// raises flag_mine[b] = seq, waits for flag_other[b] >= seq, reads the other slice from host and adds.
__global__ void ar_mapped_k(const half* __restrict__ mine, half* __restrict__ outp, int4* host_mine, const int4* host_other,
                            volatile unsigned* flag_mine, volatile unsigned* flag_other, unsigned seq, int n_v4,
                            unsigned* err) {
  int per = (n_v4 + gridDim.x - 1) / gridDim.x;
  int beg = blockIdx.x * per, end = min(n_v4, beg + per);
  const int4* m4 = (const int4*)mine;
  for (int i = beg + threadIdx.x; i < end; i += blockDim.x) host_mine[i] = m4[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    flag_mine[blockIdx.x] = seq;
    __threadfence_system();
    long long t0 = clock64();
    while (flag_other[blockIdx.x] < seq) {
      if (clock64() - t0 > 3000000000LL) { atomicAdd(err, 1u); break; }
    }
  }
  __syncthreads();
  int4* o4 = (int4*)outp;
  for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
    int4 a = m4[i], b = __ldcv(host_other + i);
    half2* ha = (half2*)&a; half2* hb = (half2*)&b;
    for (int k = 0; k < 4; k++) ha[k] = __hadd2(ha[k], hb[k]);
    o4[i] = a;
  }
}

static void ar() {
  if (g_ndev < 2) { printf("ar: needs 2 devices\n"); return; }
  const int K = 128, STEPS = 12;
  for (int bytes : {10240, 20480, 40960, 81920}) {
    int n = bytes / 2;
    half *dx[2], *dr[2], *dout[2]; cudaStream_t s[2]; cudaEvent_t ev[2][K], es[2], ee[2];
    for (int d = 0; d < 2; d++) {
      CK(cudaSetDevice(d));
      CK(cudaMalloc(&dx[d], bytes)); CK(cudaMalloc(&dr[d], bytes)); CK(cudaMalloc(&dout[d], bytes));
      CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
      for (int k = 0; k < K; k++) CK(cudaEventCreateWithFlags(&ev[d][k], cudaEventDisableTiming));
      CK(cudaEventCreate(&es[d])); CK(cudaEventCreate(&ee[d]));
    }
    char* hstage; CK(cudaHostAlloc((void**)&hstage, (size_t)2 * K * bytes, cudaHostAllocPortable));
    auto hb = [&](int d, int k) { return hstage + ((size_t)d * K + k) * bytes; };

    // (a) host staging with cross-device events, no host sync inside the step (like llama.cpp without RIG_AR_SYNC)
    auto staged = [&]() {
      for (int k = 0; k < K; k++) {
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          produce_k<<<8, 256, 0, s[d]>>>(dx[d], n, k);
          CK(cudaMemcpyAsync(hb(d, k), dx[d], bytes, cudaMemcpyDeviceToHost, s[d]));
          CK(cudaEventRecord(ev[d][k], s[d]));
        }
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          CK(cudaStreamWaitEvent(s[d], ev[1 - d][k], 0));
          CK(cudaMemcpyAsync(dr[d], hb(1 - d, k), bytes, cudaMemcpyHostToDevice, s[d]));
          add_k<<<8, 256, 0, s[d]>>>(dx[d], dr[d], dout[d], n);
        }
      }
    };
    // (b) same but the host waits for both D2H copies before issuing the H2D copies (like RIG_AR_SYNC=1)
    auto staged_hostsync = [&]() {
      for (int k = 0; k < K; k++) {
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          produce_k<<<8, 256, 0, s[d]>>>(dx[d], n, k);
          CK(cudaMemcpyAsync(hb(d, k), dx[d], bytes, cudaMemcpyDeviceToHost, s[d]));
        }
        CK(cudaStreamSynchronize(s[0])); CK(cudaStreamSynchronize(s[1]));
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          CK(cudaMemcpyAsync(dr[d], hb(1 - d, k), bytes, cudaMemcpyHostToDevice, s[d]));
          add_k<<<8, 256, 0, s[d]>>>(dx[d], dr[d], dout[d], n);
        }
      }
    };
    // (c) only the compute kernels, no transfer: the floor of this emulation
    auto compute_only = [&]() {
      for (int k = 0; k < K; k++)
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          produce_k<<<8, 256, 0, s[d]>>>(dx[d], n, k);
          add_k<<<8, 256, 0, s[d]>>>(dx[d], dr[d], dout[d], n);
        }
    };
    auto time_step = [&](auto fn) {
      std::vector<double> v;
      for (int it = 0; it < STEPS; it++) {
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
        auto t = clk::now();
        fn();
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
        if (it) v.push_back(us_since(t) / K);
      }
      return median(v);
    };
    double t_staged = time_step(staged), t_sync = time_step(staged_hostsync), t_comp = time_step(compute_only);

    // (d) one-shot through mapped host memory
    int4 *hm[2]; unsigned* hf[2]; unsigned* err[2];
    int NB = 4; const int NBMAX = 64;
    for (int d = 0; d < 2; d++) {
      CK(cudaHostAlloc((void**)&hm[d], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
      CK(cudaHostAlloc((void**)&hf[d], NBMAX * sizeof(unsigned), cudaHostAllocMapped | cudaHostAllocPortable));
      memset(hf[d], 0, NBMAX * sizeof(unsigned));
      CK(cudaSetDevice(d)); CK(cudaMalloc(&err[d], 4)); CK(cudaMemset(err[d], 0, 4));
    }
    unsigned seq = 0;
    auto mapped = [&]() {
      for (int k = 0; k < K; k++) {
        ++seq;
        for (int d = 0; d < 2; d++) {
          CK(cudaSetDevice(d));
          produce_k<<<8, 256, 0, s[d]>>>(dx[d], n, k);
          int4 *pm, *po; unsigned *fm, *fo;
          CK(cudaHostGetDevicePointer((void**)&pm, hm[d], 0)); CK(cudaHostGetDevicePointer((void**)&po, hm[1 - d], 0));
          CK(cudaHostGetDevicePointer((void**)&fm, hf[d], 0)); CK(cudaHostGetDevicePointer((void**)&fo, hf[1 - d], 0));
          ar_mapped_k<<<NB, 256, 0, s[d]>>>(dx[d], dout[d], pm, po, fm, fo, seq, bytes / 16, err[d]);
        }
        if ((k & 7) == 7) { cudaStreamQuery(s[0]); cudaStreamQuery(s[1]); }  // push work out of the WDDM batch queue
      }
      cudaStreamQuery(s[0]); cudaStreamQuery(s[1]);
    };
    double t_mapped = time_step(mapped);

    // (e) the same one-shot and the compute-only floor, each device's K steps captured in its own CUDA graph,
    // so host launch cost is gone. Flags are reset by the host between steps (seq restarts at 1).
    auto graph_step = [&](bool with_ar) {
      cudaGraphExec_t ge[2]; cudaGraph_t g[2];
      for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        int4 *pm, *po; unsigned *fm, *fo;
        CK(cudaHostGetDevicePointer((void**)&pm, hm[d], 0)); CK(cudaHostGetDevicePointer((void**)&po, hm[1 - d], 0));
        CK(cudaHostGetDevicePointer((void**)&fm, hf[d], 0)); CK(cudaHostGetDevicePointer((void**)&fo, hf[1 - d], 0));
        CK(cudaStreamBeginCapture(s[d], cudaStreamCaptureModeThreadLocal));
        for (int k = 0; k < K; k++) {
          produce_k<<<8, 256, 0, s[d]>>>(dx[d], n, k);
          if (with_ar) ar_mapped_k<<<NB, 256, 0, s[d]>>>(dx[d], dout[d], pm, po, fm, fo, (unsigned)(k + 1), bytes / 16, err[d]);
          else add_k<<<8, 256, 0, s[d]>>>(dx[d], dr[d], dout[d], n);
        }
        CK(cudaStreamEndCapture(s[d], &g[d])); CK(cudaGraphInstantiate(&ge[d], g[d], 0));
      }
      std::vector<double> v;
      for (int it = 0; it < STEPS; it++) {
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
        memset(hf[0], 0, NBMAX * sizeof(unsigned)); memset(hf[1], 0, NBMAX * sizeof(unsigned));
        auto t = clk::now();
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaGraphLaunch(ge[d], s[d])); cudaStreamQuery(s[d]); }
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
        if (it) v.push_back(us_since(t) / K);
      }
      for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaGraphExecDestroy(ge[d])); CK(cudaGraphDestroy(g[d])); }
      return median(v);
    };
    double t_graph_comp = graph_step(false);
    for (int nb : {16, 36, 64}) {
      NB = nb; double t = graph_step(true);
      printf("ar.%dB graph: mapped_oneshot blocks=%d %.2f us per step | compute_only %.2f us | allreduce cost ~%.2f us\n",
             bytes, nb, t, t_graph_comp, t - t_graph_comp);
    }
    NB = 4;
    double t_graph_ar = graph_step(true);  // last, so the result check below sees the all-reduce output
    printf("ar.%dB graph: mapped_oneshot blocks=4 %.2f us per step | compute_only %.2f us | allreduce cost ~%.2f us\n",
           bytes, t_graph_ar, t_graph_comp, t_graph_ar - t_graph_comp);
    unsigned e0 = 0, e1 = 0;
    CK(cudaSetDevice(0)); CK(cudaMemcpy(&e0, err[0], 4, cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(1)); CK(cudaMemcpy(&e1, err[1], 4, cudaMemcpyDeviceToHost));
    // check the result of the mapped path: out = x0 + x1 with the last salt on both cards
    std::vector<half> o0(n), o1(n);
    CK(cudaSetDevice(0)); CK(cudaMemcpy(o0.data(), dout[0], bytes, cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(1)); CK(cudaMemcpy(o1.data(), dout[1], bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; i++) {
      float want = 2.0f * (float)((i + K - 1) & 7);
      if (__half2float(o0[i]) != want || __half2float(o1[i]) != want) bad++;
    }
    printf("ar.%dB per_allreduce: staged_events %.1f us | staged_hostsync %.1f us | mapped_oneshot %.1f us (timeouts %u, bad %d) | compute_only %.1f us\n",
           bytes, t_staged, t_sync, t_mapped, e0 + e1, bad, t_comp);
    for (int d = 0; d < 2; d++) {
      CK(cudaSetDevice(d));
      CK(cudaFree(dx[d])); CK(cudaFree(dr[d])); CK(cudaFree(dout[d])); CK(cudaFree(err[d]));
      for (int k = 0; k < K; k++) CK(cudaEventDestroy(ev[d][k]));
      CK(cudaStreamDestroy(s[d])); CK(cudaFreeHost(hm[d])); CK(cudaFreeHost(hf[d]));
    }
    CK(cudaFreeHost(hstage));
  }
}

int main(int argc, char** argv) {
  CK(cudaGetDeviceCount(&g_ndev));
  std::string t = argc > 1 ? argv[1] : "all";
  if (t == "info" || t == "all") info();
  if (t == "bw" || t == "all") bw();
  if (t == "seq" || t == "all") seq();
  if (t == "launch" || t == "all") launch();
  if (t == "pcie" || t == "all") pcie();
  if (t == "ar" || t == "all") ar();
  return 0;
}
