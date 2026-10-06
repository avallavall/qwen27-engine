// Bench and check the decode GEMV on real tensors of the model.
// Usage: bench_gemv <model.gguf> [device=1] [out_dir=bench/out]
// For every (type, shape) group of supported main-model tensors: upload and repack several tensors
// (more than the 32 MB L2 in total), time 1/2/4 columns, and write the first tensor's inputs and
// outputs to <out_dir>/check_<type>_<K>x<N>.bin for tools/check_gemv.py.
#include "common.cuh"
#include "gguf.h"
#include "qmat.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace q27;

int main(int argc, char** argv) try {
  if (argc < 2) { fprintf(stderr, "usage: bench_gemv <model.gguf> [device] [out_dir]\n"); return 1; }
  const int dev = argc > 2 ? atoi(argv[2]) : 1;
  const std::string out_dir = argc > 3 ? argv[3] : "bench/out";
  CK(cudaSetDevice(dev));
  GGUF g(argv[1]);

  // Optional filter: BENCH_ONLY=<type>:<K>:<N>[:<nc>], for example IQ3_S:17408:5120:1
  std::string only_type; int64_t only_k = 0, only_n = 0; int only_nc = 0;
  if (const char* f = getenv("BENCH_ONLY")) {
    char tb[32] = {0}; long long k = 0, n = 0; int nc = 0;
    if (sscanf(f, "%31[^:]:%lld:%lld:%d", tb, &k, &n, &nc) >= 3) { only_type = tb; only_k = k; only_n = n; only_nc = nc; }
  }
  std::map<std::tuple<int, int64_t, int64_t>, std::vector<const GTensor*>> groups;
  for (const auto& t : g.tensors()) {
    if (t.name.rfind("blk.", 0) != 0) continue;  // blk.64 included: it has the only Q6_K tensors
    if (t.n_dims != 2 || !qmat_supported(t.type) || t.ne[1] % 8 || t.ne[0] % 256) continue;
    if (!only_type.empty() && (only_type != gtype_name(t.type) || only_k != t.ne[0] || only_n != t.ne[1])) continue;
    groups[{(int)t.type, t.ne[0], t.ne[1]}].push_back(&t);
  }

  size_t scratch_bytes = 0;
  for (const auto& t : g.tensors()) scratch_bytes = std::max(scratch_bytes, (size_t)t.nbytes);
  scratch_bytes = std::min(scratch_bytes, (size_t)256 << 20);
  void* scratch; CK(cudaMalloc(&scratch, scratch_bytes));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  const int NCMAX = 4;
  int64_t kmax = 0, nmax = 0;
  for (auto& kv : groups) { kmax = std::max(kmax, std::get<1>(kv.first)); nmax = std::max(nmax, std::get<2>(kv.first)); }
  float *x, *xd, *y; int8_t* xq;
  CK(cudaMalloc(&x, sizeof(float) * NCMAX * kmax));
  CK(cudaMalloc(&xd, sizeof(float) * NCMAX * kmax / 32));
  CK(cudaMalloc(&xq, NCMAX * kmax));
  CK(cudaMalloc(&y, sizeof(float) * NCMAX * nmax));

  printf("%-8s %6s %6s %3s %4s %10s %9s %9s\n", "type", "K", "N", "nc", "mats", "us/matrix", "GB/s", "GB/s_gguf");
  for (auto& [key, list] : groups) {
    const GType type = (GType)std::get<0>(key);
    const int K = (int)std::get<1>(key), N = (int)std::get<2>(key);
    // enough matrices to exceed L2 several times
    std::vector<QMat> mats;
    size_t total = 0, total_gguf = 0;
    for (const GTensor* t : list) {
      mats.push_back(qmat_upload(*t, 0, N, scratch, scratch_bytes, s));
      total += mats.back().bytes;
      total_gguf += t->nbytes;
      if (total > ((size_t)160 << 20) && mats.size() >= 4) break;
    }
    CK(cudaStreamSynchronize(s));

    // activations: normal values with 1% outliers x5, fixed seed
    std::vector<float> hx((size_t)NCMAX * K);
    std::mt19937 rng(1234 + K);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (auto& v : hx) { v = nd(rng); if (ud(rng) < 0.01f) v *= 5.f; }
    CK(cudaMemcpy(x, hx.data(), sizeof(float) * hx.size(), cudaMemcpyHostToDevice));
    quantize_q8_1(x, xq, xd, K, NCMAX, s);

    for (int nc : {1, 2, 4}) {
      if (only_nc && nc != only_nc) continue;
      for (auto& m : mats) qgemv(m, xq, xd, y, nc, s);
      const int rounds = 10;
      CK(cudaEventRecord(e0, s));
      for (int r = 0; r < rounds; r++)
        for (auto& m : mats) qgemv(m, xq, xd, y, nc, s);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      const double sec = ms / 1e3 / rounds;
      printf("%-8s %6d %6d %3d %4zu %10.1f %9.1f %9.1f\n", gtype_name(type), K, N, nc, mats.size(),
             sec * 1e6 / mats.size(), total / sec / 1e9, total_gguf / sec / 1e9);
    }

    // check data for the first matrix, 4 columns
    qgemv(mats[0], xq, xd, y, 4, s);
    CK(cudaStreamSynchronize(s));
    std::vector<int8_t> hq((size_t)NCMAX * K);
    std::vector<float> hd((size_t)NCMAX * K / 32), hy((size_t)NCMAX * N);
    CK(cudaMemcpy(hq.data(), xq, hq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hd.data(), xd, hd.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
    const std::string fn = out_dir + "/check_" + gtype_name(type) + "_" + std::to_string(K) + "x" + std::to_string(N) + ".bin";
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + fn);
    const std::string& name = list[0]->name;
    int32_t hdr[4] = {(int32_t)name.size(), N, K, NCMAX};
    fwrite(hdr, 4, 4, f);
    fwrite(name.data(), 1, name.size(), f);
    fwrite(hq.data(), 1, hq.size(), f);
    fwrite(hd.data(), 4, hd.size(), f);
    fwrite(hy.data(), 4, hy.size(), f);
    fclose(f);

    for (auto& m : mats) qmat_free(m);
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
