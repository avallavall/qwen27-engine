// Attention decode kernel: check against the simple reference kernel and measure time at depth.
// Usage: bench_attn [device=1] [max_depth=180224] [kv=f16|q8]
// Random K, V, Q, gate on one card with 2 KV heads (the per-card shape with two cards).
#include "common.cuh"
#include "ops.h"

#include <chrono>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>

using namespace q27;

static uint64_t rng = 88172645463325252ull;
static float frand() {  // uniform in [-1, 1)
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return (float)((rng >> 40) * (1.0 / 8388608.0)) - 1.0f;
}

int main(int argc, char** argv) try {
  const int dev = argc > 1 ? atoi(argv[1]) : 1;
  const int max_depth = argc > 2 ? atoi(argv[2]) : 180224;
  const bool q8 = argc > 3 && std::string(argv[3]) == "q8";
  CK(cudaSetDevice(dev));
  const int nkvh = 2, nqh = 12, D = 256;
  const int n_ctx = max_depth + 8;
  const size_t kv_n = (size_t)nkvh * n_ctx * D;
  const size_t kv_bytes = kv_cache_bytes(nkvh, n_ctx, q8);
  void *kc, *vc;
  CK(cudaMalloc(&kc, kv_bytes));
  CK(cudaMalloc(&vc, kv_bytes));
  for (int w = 0; w < 2; w++) {
    std::vector<uint8_t> h(kv_bytes);
    const float amp = w ? 2.0f : 2.5f;
    if (!q8) {
      for (size_t i = 0; i < kv_n; i++) ((__half*)h.data())[i] = __float2half(amp * frand());
    } else {
      for (size_t i = 0; i < kv_n; i++) ((int8_t*)h.data())[i] = (int8_t)(127.f * frand());
      for (size_t i = 0; i < kv_n / 32; i++) ((__half*)(h.data() + kv_n))[i] = __float2half(amp / 127.f * (0.5f + 0.5f * fabsf(frand())));
    }
    CK(cudaMemcpy(w ? vc : kc, h.data(), kv_bytes, cudaMemcpyHostToDevice));
  }
  float *qn, *qg, *o, *oref;
  int* dpos;
  CK(cudaMalloc(&qn, 4 * nqh * D * 4));
  CK(cudaMalloc(&qg, 4 * nqh * 2 * D * 4));
  CK(cudaMalloc(&o, 4 * nqh * D * 4));
  CK(cudaMalloc(&oref, 4 * nqh * D * 4));
  CK(cudaMalloc(&dpos, 4));
  {
    std::vector<float> h(4 * nqh * 2 * D);
    for (auto& x : h) x = 2.5f * frand();
    CK(cudaMemcpy(qn, h.data(), 4 * nqh * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(qg, h.data(), 4 * nqh * 2 * D * 4, cudaMemcpyHostToDevice));
  }
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  const float scale = 1.0f / 16.0f;
  printf("device %d, %d chunks per KV head\n", dev, attn_chunks(nkvh));

  const bool only_prefill = argc > 4 && std::string(argv[4]) == "prefill";
  // ---- prefill attention: M query tokens at pos0 .. pos0+M-1 against the reference
  {
    const int MM = 512;
    float *qnp, *qgp, *op, *opr;
    CK(cudaMalloc(&qnp, sizeof(float) * MM * nqh * D));
    CK(cudaMalloc(&qgp, sizeof(float) * MM * nqh * 2 * D));
    CK(cudaMalloc(&op, sizeof(float) * MM * nqh * D));
    CK(cudaMalloc(&opr, sizeof(float) * MM * nqh * D));
    {
      std::vector<float> h((size_t)MM * nqh * 2 * D);
      for (auto& x : h) x = 2.5f * frand();
      CK(cudaMemcpy(qnp, h.data(), sizeof(float) * MM * nqh * D, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(qgp, h.data(), sizeof(float) * MM * nqh * 2 * D, cudaMemcpyHostToDevice));
    }
    const int cases[][2] = {{0, 37}, {0, 128}, {1000, 64}, {5000, 128}, {30000, 512}, {100000, 512}, {150000, 512}};
    for (auto& c : cases) {
      const int pos0 = c[0], M = c[1];
      if (pos0 + M > max_depth) continue;
      CK(cudaMemcpy(dpos, &pos0, 4, cudaMemcpyHostToDevice));
      double err = 0, ref = 0;
      const bool check = (size_t)M * nqh * n_ctx * 4 < ((size_t)1 << 31);  // reference score buffer
      if (check) {
        attn_prefill(qnp, qgp, kc, vc, op, dpos, M, scale, nqh, nkvh, n_ctx, q8, s);
        attn_decode_ref(qnp, qgp, kc, vc, opr, dpos, M, scale, nqh, nkvh, n_ctx, q8, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> a((size_t)M * nqh * D), b((size_t)M * nqh * D);
        CK(cudaMemcpy(a.data(), op, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), opr, b.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); i++) { err = std::max(err, (double)fabsf(a[i] - b[i])); ref += (double)b[i] * b[i]; }
        ref = sqrt(ref / a.size());
      }
      cudaEvent_t e0, e1;
      CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      attn_prefill(qnp, qgp, kc, vc, op, dpos, M, scale, nqh, nkvh, n_ctx, q8, s);
      CK(cudaEventRecord(e0, s));
      const int reps = 3;
      for (int i = 0; i < reps; i++) attn_prefill(qnp, qgp, kc, vc, op, dpos, M, scale, nqh, nkvh, n_ctx, q8, s);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      ms /= reps;
      const double flop = (double)nqh * 1024.0 * ((double)M * pos0 + (double)M * (M + 1) / 2);
      printf("prefill pos0 %6d M %4d: %8.3f ms  %6.1f TFLOPS  %.3f ms per token", pos0, M, ms, flop / ms / 1e9, ms / M);
      if (check) printf("  max err %.2e (ref rms %.3f)", err, ref);
      printf("\n");
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
  }
  if (only_prefill) return 0;

  std::vector<int> depths = {1, 7, 33, 100, 1000, 4096, 30000, 100000, 150000, 180224};
  for (int d : depths) {
    if (d > max_depth) continue;
    for (int T : {1, 4}) {
      const int pos0 = std::max(0, d - T);
      CK(cudaMemcpy(dpos, &pos0, 4, cudaMemcpyHostToDevice));
      // correctness
      double err = 0, ref = 0;
      if (d <= 150000) {
        attn_decode(qn, qg, kc, vc, o, nullptr, nullptr, dpos, T, scale, nqh, nkvh, n_ctx, q8, s);
        attn_decode_ref(qn, qg, kc, vc, oref, dpos, T, scale, nqh, nkvh, n_ctx, q8, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> a(T * nqh * D), b(T * nqh * D);
        CK(cudaMemcpy(a.data(), o, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), oref, b.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); i++) { err = std::max(err, (double)fabsf(a[i] - b[i])); ref += (double)b[i] * b[i]; }
        ref = sqrt(ref / a.size());
      }
      // speed: 20 launches in a graph, repeated
      cudaGraph_t g;
      cudaGraphExec_t ge;
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < 20; i++) attn_decode(qn, qg, kc, vc, o, nullptr, nullptr, dpos, T, scale, nqh, nkvh, n_ctx, q8, s);
      CK(cudaStreamEndCapture(s, &g));
      CK(cudaGraphInstantiate(&ge, g, 0));
      CK(cudaGraphLaunch(ge, s));
      CK(cudaStreamSynchronize(s));
      cudaEvent_t e0, e1;
      CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0, s));
      for (int i = 0; i < 5; i++) CK(cudaGraphLaunch(ge, s));
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      const double us = ms * 1000.0 / 100.0;
      const double bytes = 2.0 * kv_cache_bytes(nkvh, pos0 + T, q8);
      printf("depth %6d T=%d: %8.2f us  %6.1f GB/s  (%.4f ms per 1000 tok)", d, T, us, bytes / us / 1e3, us / 1000.0 / (d / 1000.0));
      if (d <= 150000) printf("  max err %.2e (ref rms %.3f)", err, ref);
      printf("\n");
      CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
