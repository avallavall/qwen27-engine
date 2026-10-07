// Prefill GEMM check and speed on real tensors of the model.
// Usage: bench_gemm <model.gguf> [device=1] [M=1024]
// For every (type, K, N) group: the first tensor, M random tokens (normal values, 1% outliers x5) as q8_1.
// Check: qgemm against the decode GEMV (qgemv, 4 tokens at a time) on the same q8_1 inputs; prints the
// largest difference relative to the RMS of the output. Speed: TOPS = 2 N K M / time.
// Q2_K and Q4_K show 0.1-1.0 with random inputs, and that is expected: their min term uses the sum of the float
// activations in the GEMM (llama.cpp MMQ) but the sum of the rounded int8 values in the GEMV (llama.cpp MMVQ).
// GEMM_EXACT=1 uses inputs that q8_1 holds exactly, so both sums agree: then every type must be below ~2e-3.
#include "common.cuh"
#include "gguf.h"
#include "qmat.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace q27;

int main(int argc, char** argv) try {
  if (argc < 2) { fprintf(stderr, "usage: bench_gemm <model.gguf> [device] [M]\n"); return 1; }
  const int dev = argc > 2 ? atoi(argv[2]) : 1;
  const int M = argc > 3 ? atoi(argv[3]) : 1024;
  CK(cudaSetDevice(dev));
  GGUF g(argv[1]);
  std::map<std::tuple<int, int64_t, int64_t>, const GTensor*> groups;
  for (const auto& t : g.tensors()) {
    if (t.name.rfind("blk.", 0) != 0) continue;
    if (t.n_dims != 2 || !qmat_supported(t.type) || t.ne[1] % 128 || t.ne[0] % 256) continue;
    if (const char* f = getenv("GEMM_ONLY")) if (std::string(f) != gtype_name(t.type)) continue;
    groups.emplace(std::make_tuple((int)t.type, t.ne[0], t.ne[1]), &t);
  }
  size_t scratch_bytes = 0;
  for (auto& kv : groups) scratch_bytes = std::max(scratch_bytes, (size_t)kv.second->nbytes);
  void* scratch; CK(cudaMalloc(&scratch, scratch_bytes));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int64_t kmax = 0, nmax = 0;
  for (auto& kv : groups) { kmax = std::max(kmax, std::get<1>(kv.first)); nmax = std::max(nmax, std::get<2>(kv.first)); }
  float *x, *xd, *xs, *y, *yr; int8_t* xq;
  CK(cudaMalloc(&x, sizeof(float) * M * kmax));
  CK(cudaMalloc(&xd, sizeof(float) * M * kmax / 32));
  CK(cudaMalloc(&xs, sizeof(float) * M * kmax / 16));
  CK(cudaMalloc(&xq, (size_t)M * kmax));
  CK(cudaMalloc(&y, sizeof(float) * M * nmax));
  CK(cudaMalloc(&yr, sizeof(float) * M * nmax));

  printf("%-8s %6s %6s %5s %10s %8s %12s\n", "type", "K", "N", "M", "us", "TOPS", "maxerr/rms");
  double worst = 0;
  for (auto& [key, t] : groups) {
    const GType type = (GType)std::get<0>(key);
    const int K = (int)std::get<1>(key), N = (int)std::get<2>(key);
    QMat m = qmat_upload(*t, 0, N, scratch, scratch_bytes, s);
    std::vector<float> hx((size_t)M * K);
    std::mt19937 rng(99 + K);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (auto& v : hx) { v = nd(rng); if (ud(rng) < 0.01f) v *= 5.f; }
    if (getenv("GEMM_EXACT")) {  // values k/64 with |k| <= 127 and one 127 per 32: q8_1 is exact, sum x = d * sum q
      std::uniform_int_distribution<int> ki(-127, 127);
      for (size_t i = 0; i < hx.size(); i++) hx[i] = (i % 32 == 0 ? 127 : ki(rng)) / 64.0f;
    }
    if (getenv("GEMM_DUP"))  // tokens 32..63 = tokens 0..31
      for (int t = 32; t < M; t++) for (int k = 0; k < K; k++) hx[(size_t)t * K + k] = hx[(size_t)(t % 32) * K + k];
    CK(cudaMemcpy(x, hx.data(), sizeof(float) * hx.size(), cudaMemcpyHostToDevice));
    quantize_q8_1(x, xq, xd, K, M, s, xs);
    // reference: decode GEMV 4 tokens at a time
    if (!getenv("GEMM_NOREF"))
      for (int t0 = 0; t0 + 4 <= M; t0 += 4) qgemv(m, xq + (size_t)t0 * K, xd + (size_t)t0 * (K / 32), yr + (size_t)t0 * N, 4, s);
    qgemm(m, xq, xd, xs, y, M, s);
    CK(cudaStreamSynchronize(s));
    std::vector<float> a((size_t)M * N), b((size_t)M * N);
    CK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost));
    const size_t nchk = (size_t)(M / 4 * 4) * N;
    double err = 0, rms = 0;
    for (size_t i = 0; i < nchk; i++) { err = std::max(err, (double)fabsf(a[i] - b[i])); rms += (double)b[i] * b[i]; }
    rms = sqrt(rms / nchk);
    if (getenv("GEMM_ONLY")) {
      for (int rep = 0; rep < 4; rep++) {  // more runs: same result as the first?
        if (getenv("GEMM_SYNC")) CK(cudaDeviceSynchronize());
        qgemm(m, xq, xd, xs, y, M, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> a2((size_t)M * N);
        CK(cudaMemcpy(a2.data(), y, a2.size() * 4, cudaMemcpyDeviceToHost));
        int nd2 = 0;
        for (size_t i = 0; i < a2.size(); i++) nd2 += memcmp(&a2[i], &a[i], 4) != 0;
        printf("  rerun differs at %d of %zu outputs\n", nd2, a2.size());
        if (getenv("Q27_GEMM_DBG") && atoi(getenv("Q27_GEMM_DBG")) == 3) {
          std::vector<float> hd((size_t)M * K / 32);
          CK(cudaMemcpy(hd.data(), xd, hd.size() * 4, cudaMemcpyDeviceToHost));
          for (int t : {0, 1, 31, 32, 33, 63}) {
            double ex = 0;
            for (int j = 0; j < K / 32; j++) ex += 32.0 * hd[(size_t)t * (K / 32) + j];
            printf("  token %d: expected %.5f | run2 rows 0, 127, 128, 300, %d, %d:", t, ex, N / 2, N - 1);
            for (int r : {0, 127, 128, 300, N / 2, N - 1}) printf(" %.5f", a2[(size_t)t * N + r]);
            printf("\n");
          }
        }
      }
      for (int i : {0, 1, 2, 3, N, N + 1, 5 * N + 7, (M - 1) * N + 3}) printf("  y[%d] gemm %.5f gemv %.5f\n", i, a[i], b[i]);
      for (int t = 0; t < M; t++) {
        double e = 0; int worst_n = 0;
        for (int n = 0; n < N; n++) { const double d = fabs((double)a[(size_t)t * N + n] - b[(size_t)t * N + n]); if (!(d <= e)) { e = d; worst_n = n; } }
        if (e > 1e-3) printf("  token %d: max err %.4g at row %d (gemm %.4g gemv %.4g)\n", t, e, worst_n, a[(size_t)t * N + worst_n], b[(size_t)t * N + worst_n]);
      }
    }
    worst = std::max(worst, err / rms);
    // speed
    const int rounds = 5;
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < rounds; r++) qgemm(m, xq, xd, xs, y, M, s);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1000 / rounds;
    printf("%-8s %6d %6d %5d %10.1f %8.1f %12.2e\n", gtype_name(type), K, N, M, us, 2.0 * N * K * M / us / 1e6, err / rms);
    fflush(stdout);
    qmat_free(m);
  }
  printf("worst maxerr/rms %.2e\n", worst);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
