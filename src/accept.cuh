// MTP acceptance rules, host and device: the decoder runs them in k_accept (model.cu); tools/test_accept.cu runs the
// same code on the CPU to check that the emitted tokens follow the target distribution exactly.
//   pc[i] = target candidates for the token after s, d[0], ..., d[i-1] (i = 0..nd; row nd is the bonus row)
//   qc[i] = draft candidates that d[i] was drawn from (i = 0..nd-1)
// Random numbers: rng01(seed, counter, row, salt), the same values on both cards.
#pragma once
#include <cstdint>

#include "sampling.h"

namespace q27 {

__host__ __device__ inline float rng01(uint64_t seed, int counter, int row, int salt) {
  uint64_t z = seed ^ ((uint64_t)(uint32_t)counter << 24) ^ ((uint64_t)row << 8) ^ (uint64_t)salt;
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return (float)((z >> 40) * (1.0 / 16777216.0));
}

__host__ __device__ inline float cand_p(const CandRow& c, int id) {
  for (int i = 0; i < c.n; i++) if (c.id[i] == id) return c.p[i];
  return 0.f;
}
__host__ __device__ inline int cand_draw(const CandRow& c, float u) {
  float cum = 0.f;
  for (int i = 0; i < c.n; i++) { cum += c.p[i]; if (u < cum) return c.id[i]; }
  return c.n > 0 ? c.id[c.n - 1] : 0;
}

// Token by token, llama.cpp rule (common/sampling.cpp, PR #27694): accept draft x if q(x) > 0 and (p(x) >= q(x) or
// U < p(x)/q(x)); on reject draw from max(0, p - q) (tokens outside q keep p); after nd accepts draw a bonus token
// from row nd. Writes the emitted tokens to emit and returns how many (1..nd+1).
__host__ __device__ inline int accept_token(const int* d, int nd, const CandRow* pc, const CandRow* qc, uint64_t seed,
                                            int counter, int* emit) {
  int cnt = 0;
  for (int i = 0; i < nd; i++) {
    const int x = d[i];
    const float px = cand_p(pc[i], x), qx = cand_p(qc[i], x);
    const float u = rng01(seed, counter, i, 1);
    if (qx > 0.f && (px >= qx || u < px / qx)) { emit[cnt++] = x; continue; }
    float sum = 0.f;
    for (int k = 0; k < pc[i].n; k++) sum += fmaxf(0.f, pc[i].p[k] - cand_p(qc[i], pc[i].id[k]));
    int pick;
    if (sum > 0.f) {
      const float u2 = rng01(seed, counter, i, 2) * sum;
      float cum = 0.f;
      pick = pc[i].n > 0 ? pc[i].id[pc[i].n - 1] : 0;
      for (int k = 0; k < pc[i].n; k++) {
        cum += fmaxf(0.f, pc[i].p[k] - cand_p(qc[i], pc[i].id[k]));
        if (u2 < cum) { pick = pc[i].id[k]; break; }
      }
    } else {
      pick = cand_draw(pc[i], rng01(seed, counter, i, 3));
    }
    emit[cnt++] = pick;
    return cnt;
  }
  emit[cnt++] = cand_draw(pc[nd], rng01(seed, counter, nd, 4));
  return cnt;
}

// Sum over x of max(w p(x) - q(x), 0) (only tokens in p can be positive).
__host__ __device__ inline float resid_sum(const CandRow& p, const CandRow& q, float w) {
  float s = 0.f;
  for (int k = 0; k < p.n; k++) s += fmaxf(0.f, w * p.p[k] - cand_p(q, p.id[k]));
  return s;
}
// Draw from max(w p - q, 0) / sum; u in [0, 1).
__host__ __device__ inline int resid_draw(const CandRow& p, const CandRow& q, float w, float sum, float u) {
  if (!(sum > 0.f)) return cand_draw(p, u);  // only after rounding: the residual has no mass
  const float t = u * sum;
  float cum = 0.f;
  int last = p.n > 0 ? p.id[p.n - 1] : 0;
  for (int k = 0; k < p.n; k++) {
    const float r = fmaxf(0.f, w * p.p[k] - cand_p(q, p.id[k]));
    if (r > 0.f) last = p.id[k];
    cum += r;
    if (t < cum) return p.id[k];
  }
  return last;
}

// Block verification (Sun et al., "Block Verification Accelerates Speculative Decoding", ICLR 2025, arXiv 2403.10444,
// Algorithm 2). Exact like the token rule and never worse in expected tokens per step. With P_0 = 1:
//   P_i = min(P_{i-1} p_i(X_i) / q_i(X_i), 1)
//   h_i = R_i / (R_i + 1 - P_i) for i < nd, with R_i = sum_x max(P_i p_{i+1}(x) - q_{i+1}(x), 0);  h_nd = P_nd
//   tau = the last i with U_i <= h_i (0 if none); emit X_1..X_tau, then Y from p_{nd+1} if tau = nd, else from
//   max(P_tau p_{tau+1} - q_{tau+1}, 0) normalized.
// (Here p_{i+1} = pc[i], q_{i+1} = qc[i], X_i = d[i-1].) h = 0 when R_i = 0, so a residual without mass is never used.
// broken = true: test only, draws Y from p_{tau+1} instead of the residual (the exactness test must catch it).
__host__ __device__ inline int accept_block(const int* d, int nd, const CandRow* pc, const CandRow* qc, uint64_t seed,
                                            int counter, int* emit, bool broken = false) {
  float P = 1.f, Pt = 1.f;
  int tau = 0;
  for (int i = 1; i <= nd; i++) {
    const int x = d[i - 1];
    const float px = cand_p(pc[i - 1], x), qx = cand_p(qc[i - 1], x);
    P = qx > 0.f ? fminf(P * px / qx, 1.f) : 0.f;
    float h;
    if (i == nd) h = P;
    else {
      const float R = resid_sum(pc[i], qc[i], P);
      const float den = R + 1.f - P;
      h = R > 0.f && den > 0.f ? R / den : 0.f;
    }
    if (h > 0.f && rng01(seed, counter, i - 1, 1) <= h) { tau = i; Pt = P; }
  }
  for (int i = 0; i < tau; i++) emit[i] = d[i];
  if (tau == nd) emit[tau] = cand_draw(pc[nd], rng01(seed, counter, nd, 4));
  else if (broken) emit[tau] = cand_draw(pc[tau], rng01(seed, counter, tau, 2));
  else emit[tau] = resid_draw(pc[tau], qc[tau], Pt, resid_sum(pc[tau], qc[tau], Pt), rng01(seed, counter, tau, 2));
  return tau + 1;
}

}  // namespace q27
