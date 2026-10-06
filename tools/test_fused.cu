// Unit test: fused kernels (ops.cu) against the unfused kernel chains, bit for bit, on random data (one card).
// Usage: test_fused [device=1]
#include "common.cuh"
#include "ops.h"
#include "qmat.h"

#include <cstring>
#include <random>
#include <vector>

using namespace q27;

template <class T>
static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
template <class T>
static int ndiff(const std::vector<T>& a, const std::vector<T>& b, const char* what) {
  int n = 0, first = -1;
  for (size_t i = 0; i < a.size(); i++)
    if (memcmp(&a[i], &b[i], sizeof(T))) { if (first < 0) first = (int)i; n++; }
  printf("  %-6s %s (%d of %zu differ", what, n ? "DIFF" : "same", n, a.size());
  if (first >= 0) printf(", first at %d: %.9g vs %.9g", first, (double)a[first], (double)b[first]);
  printf(")\n");
  return n;
}

int main(int argc, char** argv) try {
  const int dev = argc > 1 ? atoi(argv[1]) : 1;
  CK(cudaSetDevice(dev));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto upload = [&](size_t n, float scale) {
    std::vector<float> h(n);
    for (auto& v : h) v = scale * nd(rng);
    float* d; CK(cudaMalloc(&d, n * 4));
    CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
  };
  const int E = 5120, T = 4;
  float* x0 = upload(T * E, 3.f);
  float* part = upload(T * E, 1.f);
  float* w = upload(E, 0.5f);
  float *xa, *xb, *ha, *hb, *xda, *xdb;
  int8_t *qa, *qb;
  CK(cudaMalloc(&xa, T * E * 4)); CK(cudaMalloc(&xb, T * E * 4));
  CK(cudaMalloc(&ha, T * E * 4)); CK(cudaMalloc(&hb, T * E * 4));
  CK(cudaMalloc(&xda, T * E / 32 * 4)); CK(cudaMalloc(&xdb, T * E / 32 * 4));
  CK(cudaMalloc(&qa, T * E)); CK(cudaMalloc(&qb, T * E));
  int bad = 0;

  printf("sum_norm_q8 vs add + rmsnorm + quantize_q8_1:\n");
  CK(cudaMemcpy(xa, x0, T * E * 4, cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(xb, x0, T * E * 4, cudaMemcpyDeviceToDevice));
  sum_norm_q8(xa, part, E, T, nullptr, w, 1e-6f, ha, qa, xda, s);
  add(xb, part, xb, T * E, s);
  rmsnorm(xb, w, hb, E, T, 1e-6f, s);
  quantize_q8_1(hb, qb, xdb, E, T, s);
  CK(cudaStreamSynchronize(s));
  bad += ndiff(down(xa, T * E), down(xb, T * E), "x");
  bad += ndiff(down(ha, T * E), down(hb, T * E), "h");
  bad += ndiff(down(qa, T * E), down(qb, T * E), "xq");
  bad += ndiff(down(xda, T * E / 32), down(xdb, T * E / 32), "xd");

  printf("rmsnorm_q8 vs rmsnorm + quantize_q8_1:\n");
  rmsnorm_q8(x0, w, ha, qa, xda, E, T, 1e-6f, s);
  rmsnorm(x0, w, hb, E, T, 1e-6f, s);
  quantize_q8_1(hb, qb, xdb, E, T, s);
  CK(cudaStreamSynchronize(s));
  bad += ndiff(down(ha, T * E), down(hb, T * E), "h");
  bad += ndiff(down(qa, T * E), down(qb, T * E), "xq");

  printf("gdn_conv_l2 vs gdn_gates + gdn_conv + l2norm_heads:\n");
  {
    const int KH = 8, H = 24, C = 2 * KH * 128 + H * 128;
    float* qkv = upload(T * C, 1.f);
    float* cw = upload(C * 4, 0.5f);
    float* st0 = upload(4 * C * 3, 1.f);
    float* ab = upload(2 * T * H, 2.f);
    float* sa = upload(H, 1.f);
    float* dtb = upload(H, 1.f);
    float *sta, *stb, *ya, *yb, *ga, *gb, *ba, *bb;
    CK(cudaMalloc(&sta, 4 * C * 3 * 4)); CK(cudaMalloc(&stb, 4 * C * 3 * 4));
    CK(cudaMalloc(&ya, T * C * 4)); CK(cudaMalloc(&yb, T * C * 4));
    CK(cudaMalloc(&ga, T * H * 4)); CK(cudaMalloc(&gb, T * H * 4));
    CK(cudaMalloc(&ba, T * H * 4)); CK(cudaMalloc(&bb, T * H * 4));
    int* plane; CK(cudaMalloc(&plane, 4)); CK(cudaMemset(plane, 0, 4));
    CK(cudaMemcpy(sta, st0, 4 * C * 3 * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(stb, st0, 4 * C * 3 * 4, cudaMemcpyDeviceToDevice));
    gdn_conv_l2(qkv, cw, sta, plane, ya, C, T, true, 2 * KH, 1e-6f, ab, ab + T * H, sa, dtb, ga, ba, H, s);
    gdn_gates(ab, ab + T * H, sa, dtb, gb, bb, H, T, s);
    gdn_conv(qkv, cw, stb, plane, yb, C, T, true, s);
    l2norm_heads(yb, 128, 2 * KH, T, C, 1e-6f, s);
    CK(cudaStreamSynchronize(s));
    bad += ndiff(down(ya, T * C), down(yb, T * C), "y");
    bad += ndiff(down(sta, 4 * C * 3), down(stb, 4 * C * 3), "state");
    bad += ndiff(down(ga, T * H), down(gb, T * H), "g");
    bad += ndiff(down(ba, T * H), down(bb, T * H), "beta");
  }
  printf(bad ? "DIFFERENCES FOUND\n" : "all identical\n");
  return bad ? 1 : 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
