// Measure the weight-streaming speed of a whole decode pass: every supported main-model matrix
// (blk.0..63) in model order, one GEMV each, captured in one CUDA graph. This includes the cost of
// kernel boundaries, which the per-matrix bench hides.
// Usage: bench_pass <model.gguf> [device=1] [split=1|2]
//   split=2 approximates one card of the 2-card tensor-parallel split: every matrix keeps half of
//   its rows. The real split halves K for attn_output, ssm_out and ffn_down instead; the bytes per
//   card are the same, the shapes differ.
#include "common.cuh"
#include "gguf.h"
#include "qmat.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace q27;

static bool row_parallel(const std::string& n) {
  return n.find("attn_output") != std::string::npos || n.find("ssm_out") != std::string::npos ||
         n.find("ffn_down") != std::string::npos;
}

int main(int argc, char** argv) try {
  if (argc < 2) { fprintf(stderr, "usage: bench_pass <model.gguf> [device] [split]\n"); return 1; }
  const int dev = argc > 2 ? atoi(argv[2]) : 1;
  const int split = argc > 3 ? atoi(argv[3]) : 1;
  CK(cudaSetDevice(dev));
  GGUF g(argv[1]);

  std::vector<const GTensor*> list;
  size_t skipped = 0;
  for (int L = 0; L < 64; L++) {
    const std::string p = "blk." + std::to_string(L) + ".";
    for (const auto& t : g.tensors()) {
      if (t.name.rfind(p, 0) != 0 || t.n_dims != 2 || t.type == GType::F32 || t.type == GType::BF16) continue;
      if (!qmat_supported(t.type)) { skipped += t.nbytes; continue; }
      list.push_back(&t);
    }
  }

  size_t scratch_bytes = 0;
  for (auto* t : list) scratch_bytes = std::max(scratch_bytes, (size_t)t->nbytes);
  void* scratch; CK(cudaMalloc(&scratch, scratch_bytes));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  std::vector<QMat> mats;
  size_t total = 0;
  int64_t kmax = 0, nmax = 0;
  for (auto* t : list) {
    int N = (int)t->ne[1];
    if (split == 2) N = N / 2 / 8 * 8;  // same bytes as a half split on either axis
    mats.push_back(qmat_upload(*t, 0, N, scratch, scratch_bytes, s));
    total += mats.back().bytes;
    kmax = std::max(kmax, t->ne[0]); nmax = std::max<int64_t>(nmax, N);
  }
  CK(cudaStreamSynchronize(s));
  printf("matrices %zu, bytes %.1f MiB (types not supported yet: %.1f MiB skipped), split %d\n",
         mats.size(), total / 1048576.0, skipped / 1048576.0, split);

  float *x, *xd, *y; int8_t* xq;
  CK(cudaMalloc(&x, sizeof(float) * 4 * kmax));
  CK(cudaMalloc(&xd, sizeof(float) * 4 * kmax / 32));
  CK(cudaMalloc(&xq, 4 * kmax));
  CK(cudaMalloc(&y, sizeof(float) * 4 * nmax));
  std::vector<float> hx(4 * kmax);
  std::mt19937 rng(7); std::normal_distribution<float> nd;
  for (auto& v : hx) v = nd(rng);
  CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
  quantize_q8_1(x, xq, xd, (int)kmax, 4, s);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int nc : {1, 2, 4}) {
    for (auto& m : mats) qgemv(m, xq, xd, y, nc, s);  // sizes the split workspace before capture
    CK(cudaStreamSynchronize(s));
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (auto& m : mats) qgemv(m, xq, xd, y, nc, s);
    CK(cudaStreamEndCapture(s, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<float> v;
    for (int it = 0; it < 7; it++) {
      CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    const double ms = v[v.size() / 2];
    printf("nc=%d  pass %.2f ms  %.1f GB/s  (%.1f us per kernel, %zu kernels)\n", nc, ms, total / (ms * 1e6),
           ms * 1e3 / mats.size(), mats.size());
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
