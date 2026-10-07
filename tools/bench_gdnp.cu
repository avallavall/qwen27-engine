// GDN delta rule over a prefill half-batch (M tokens, flip mode): time per call and the difference between the
// per-column kernel (gdn_step_kernel) and the shared-memory kernel (gdn_prefill_kernel) on random inputs.
// Usage: bench_gdnp [device=1] [M=256]
#include "common.cuh"
#include "ops.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace q27;

int main(int argc, char** argv) try {
  const int dev = argc > 1 ? atoi(argv[1]) : 1;
  const int M = argc > 2 ? atoi(argv[2]) : 256;
  CK(cudaSetDevice(dev));
  const int H = 24, HK = 8, D = 128, C = 2 * HK * D + H * D, L = 48;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hconv((size_t)M * C), hg((size_t)M * H), hb((size_t)M * H), hst((size_t)H * D * D);
  for (auto& v : hconv) v = 0.1f * nd(rng);
  for (auto& v : hg) v = -0.05f * fabsf(nd(rng));
  for (auto& v : hb) v = 0.5f + 0.1f * nd(rng);
  for (auto& v : hst) v = 0.01f * nd(rng);
  std::vector<float*> planes(L);
  for (auto& p : planes) {
    CK(cudaMalloc(&p, sizeof(float) * 2 * H * D * D));
    CK(cudaMemcpy(p, hst.data(), sizeof(float) * H * D * D, cudaMemcpyHostToDevice));
  }
  float *conv, *g, *beta, *o;
  int* plane;
  CK(cudaMalloc(&conv, sizeof(float) * M * C));
  CK(cudaMalloc(&g, sizeof(float) * M * H));
  CK(cudaMalloc(&beta, sizeof(float) * M * H));
  CK(cudaMalloc(&o, sizeof(float) * M * H * D));
  CK(cudaMalloc(&plane, sizeof(int)));
  CK(cudaMemcpy(conv, hconv.data(), sizeof(float) * M * C, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g, hg.data(), sizeof(float) * M * H, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(beta, hb.data(), sizeof(float) * M * H, cudaMemcpyHostToDevice));
  CK(cudaMemset(plane, 0, sizeof(int)));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const float scale = 1.0f / sqrtf((float)D);
  auto run = [&](int l) { gdn_step(conv, conv + HK * D, conv + 2 * HK * D, C, g, beta, planes[l], plane, o, H, HK, M, scale, false, s, true); };
  for (int l = 0; l < L; l++) run(l);
  CK(cudaStreamSynchronize(s));
  std::vector<float> out((size_t)M * H * D), st((size_t)H * D * D);
  CK(cudaMemcpy(out.data(), o, out.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(st.data(), planes[L - 1] + (size_t)H * D * D, st.size() * 4, cudaMemcpyDeviceToHost));
  double so = 0, ss = 0;
  for (float v : out) so += (double)v * v;
  for (float v : st) ss += (double)v * v;
  printf("output rms %.6g, final state rms %.6g (compare between Q27_GDN_OLD=1 and 0)\n", sqrt(so / out.size()), sqrt(ss / st.size()));
  CK(cudaEventRecord(e0, s));
  for (int r = 0; r < 4; r++)
    for (int l = 0; l < L; l++) run(l);
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  printf("M %d: %.1f us per call\n", M, ms * 1000 / (4 * L));
  // dump a few values for comparison
  printf("o[0..3] %.7g %.7g %.7g %.7g | o[last] %.7g\n", out[0], out[1], out[2], out[3], out.back());
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
