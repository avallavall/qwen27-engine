// Unit test on two cards: sum_norm_q8 with the cross-card exchange against ar_add + rmsnorm + quantize_q8_1.
// With the q8 wire (default; Q27_WIRE=bf16 for the bf16 wire) the check is that both cards end with the same x.
// Usage: test_ar [T=4]
#include "common.cuh"
#include "ops.h"
#include "qmat.h"

#include <cstring>
#include <random>
#include <vector>

using namespace q27;

int main(int argc, char** argv) try {
  const int T = argc > 1 ? atoi(argv[1]) : 4;
  const int E = 5120, n = T * E;
  __nv_bfloat16* data;
  int *flags, *err;
  CK(cudaHostAlloc(&data, sizeof(__nv_bfloat16) * 2 * 2 * 4 * E, cudaHostAllocMapped | cudaHostAllocPortable));
  CK(cudaHostAlloc(&flags, sizeof(int) * 2 * 2 * 4 * 32, cudaHostAllocMapped | cudaHostAllocPortable));
  CK(cudaHostAlloc(&err, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
  memset(flags, 0, sizeof(int) * 2 * 2 * 4 * 32);
  *err = 0;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hx(n), hw(E), hp[2];
  for (auto& v : hx) v = 3 * nd(rng);
  for (auto& v : hw) v = 0.5f * nd(rng);
  for (int r = 0; r < 2; r++) { hp[r].resize(n); for (auto& v : hp[r]) v = nd(rng); }
  struct Card { cudaStream_t s; float *x, *p, *w, *h; int8_t* q; float* d; int* step; };
  Card c[2];
  for (int r = 0; r < 2; r++) {
    CK(cudaSetDevice(r));
    CK(cudaStreamCreateWithFlags(&c[r].s, cudaStreamNonBlocking));
    CK(cudaMalloc(&c[r].x, n * 4)); CK(cudaMalloc(&c[r].p, n * 4)); CK(cudaMalloc(&c[r].w, E * 4)); CK(cudaMalloc(&c[r].h, n * 4));
    CK(cudaMalloc(&c[r].q, n)); CK(cudaMalloc(&c[r].d, n / 32 * 4)); CK(cudaMalloc(&c[r].step, 4));
    CK(cudaMemset(c[r].step, 0, 4));
    CK(cudaMemcpy(c[r].p, hp[r].data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c[r].w, hw.data(), E * 4, cudaMemcpyHostToDevice));
  }
  auto run = [&](bool fused, int index, std::vector<float>* xo, std::vector<float>* ho, std::vector<int8_t>* qo) {
    const int slot = index & 1;
    for (int r = 0; r < 2; r++) {
      CK(cudaSetDevice(r));
      CK(cudaMemcpy(c[r].x, hx.data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaDeviceSynchronize());  // pageable copies may return before the DMA ends
      __nv_bfloat16* mine = data + ((size_t)slot * 2 + r) * 4 * E;
      __nv_bfloat16* other = data + ((size_t)slot * 2 + (1 - r)) * 4 * E;
      int* fm = flags + ((size_t)slot * 2 + r) * 4 * 32;
      int* fo = flags + ((size_t)slot * 2 + (1 - r)) * 4 * 32;
      if (fused) {
        ArArgs a{mine, other, fm, fo, c[r].step, 1000, index, err};
        sum_norm_q8(c[r].x, c[r].p, E, T, &a, c[r].w, 1e-6f, c[r].h, c[r].q, c[r].d, c[r].s);
      } else {
        ar_add(c[r].x, c[r].p, n, mine, other, fm, fo, c[r].step, 1000, index, err, c[r].s);
        rmsnorm(c[r].x, c[r].w, c[r].h, E, T, 1e-6f, c[r].s);
        quantize_q8_1(c[r].h, c[r].q, c[r].d, E, T, c[r].s);
      }
      cudaStreamQuery(c[r].s);
    }
    for (int r = 0; r < 2; r++) {
      CK(cudaSetDevice(r));
      CK(cudaStreamSynchronize(c[r].s));
      xo[r].resize(n); ho[r].resize(n); qo[r].resize(n);
      CK(cudaMemcpy(xo[r].data(), c[r].x, n * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ho[r].data(), c[r].h, n * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(qo[r].data(), c[r].q, n, cudaMemcpyDeviceToHost));
    }
  };
  std::vector<float> xf[2], hf[2], xu[2], hu[2];
  std::vector<int8_t> qf[2], qu[2];
  run(true, 0, xf, hf, qf);   // warm-up: lazy module loading can exceed the 1 s exchange timeout
  run(false, 1, xu, hu, qu);
  *err = 0;
  // Each card must end with the same x (both compute x + p0 + p1 with the same roundings).
  int bad = 0;
  const int rounds = argc > 2 ? atoi(argv[2]) : 50;
  for (int it = 0; it < rounds; it++) {
    for (int r = 0; r < 2; r++) {
      for (auto& v : hp[r]) v = nd(rng);
      CK(cudaSetDevice(r));
      CK(cudaMemcpy(c[r].p, hp[r].data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaDeviceSynchronize());
    }
    run(true, 2 + 2 * it, xf, hf, qf);
    run(false, 3 + 2 * it, xu, hu, qu);
    int cx = 0, dx = 0, dh = 0, dq = 0;
    for (int r = 0; r < 2; r++)
      for (int i = 0; i < n; i++) {
        dx += memcmp(&xf[r][i], &xu[r][i], 4) != 0;
        dh += memcmp(&hf[r][i], &hu[r][i], 4) != 0;
        dq += qf[r][i] != qu[r][i];
        if (r == 0) cx += memcmp(&xf[0][i], &xf[1][i], 4) != 0;
      }
    // q8 wire: the fused kernel rounds the partials to int8, ar_add (the reference chain) keeps the bf16 wire, so only
    // the agreement of the two cards is required (both must compute the same x).
    if (wire_q8() ? cx != 0 : (cx || dx || dh || dq)) {
      printf("round %d: cards differ %d; fused vs ar_add: x %d h %d xq %d\n", it, cx, dx, dh, dq);
      bad++;
    }
  }
  printf("%d of %d rounds bad, err flag %d\n", bad, rounds, *err);
  return bad ? 1 : 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
