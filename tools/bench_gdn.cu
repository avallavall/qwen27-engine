// Cost of the GDN state snapshots in a 4-token verify pass (idea: GDN rollback by replay): gdn_step over the
// 48 GDN layers of one card with snapshots (state after each token -> planes 0..3, what the verify pass does) and
// without (final state only, what a replay design would write), plus a replay of k accepted tokens.
// Usage: bench_gdn [device=0]
#include "common.cuh"
#include "ops.h"

#include <cstdio>
#include <vector>

using namespace q27;

int main(int argc, char** argv) try {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  CK(cudaSetDevice(dev));
  const int H = 24, HK = 8, D = 128, C = 2 * HK * D + H * D, L = 48, T = 4;
  std::vector<float*> planes(L);
  for (auto& p : planes) { CK(cudaMalloc(&p, sizeof(float) * 4 * H * D * D)); CK(cudaMemset(p, 0, sizeof(float) * 4 * H * D * D)); }
  float *conv, *g, *beta, *o;
  int* plane;
  CK(cudaMalloc(&conv, sizeof(float) * T * C));
  CK(cudaMalloc(&g, sizeof(float) * T * H));
  CK(cudaMalloc(&beta, sizeof(float) * T * H));
  CK(cudaMalloc(&o, sizeof(float) * T * H * D));
  CK(cudaMalloc(&plane, sizeof(int)));
  CK(cudaMemset(conv, 0, sizeof(float) * T * C));
  CK(cudaMemset(g, 0, sizeof(float) * T * H));
  CK(cudaMemset(beta, 0, sizeof(float) * T * H));
  CK(cudaMemset(plane, 0, sizeof(int)));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto run = [&](int t, bool snaps) {
    const int R = 50;
    for (int w = 0; w < 3; w++)
      for (int l = 0; l < L; l++) gdn_step(conv, conv + HK * D, conv + 2 * HK * D, C, g, beta, planes[l], plane, o, H, HK, t, 0.088f, snaps, s);
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < R; r++)
      for (int l = 0; l < L; l++) gdn_step(conv, conv + HK * D, conv + 2 * HK * D, C, g, beta, planes[l], plane, o, H, HK, t, 0.088f, snaps, s);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / R;
  };
  const float with = run(T, true), without = run(T, false);
  printf("48 layers, T=4: with snapshots %.3f ms, final state only %.3f ms, difference %.3f ms per verify pass\n", with, without,
         with - without);
  for (int k = 1; k <= 3; k++) printf("replay of %d accepted tokens (48 layers): %.3f ms\n", k, run(k, false));
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
