// Exactness test of the MTP acceptance rules in src/accept.cuh (CPU only). Synthetic target and draft distributions
// over V tokens depend on the whole prefix; the draft has a smaller support for some prefixes (like top-k 10) and
// the target drops tokens for others (like top-p). Each sample runs speculative rounds of 3 drafts until L tokens
// are emitted. The frequency of every L-token sequence must match its exact target probability: chi-square over the
// V^L cells, and the largest |z|. A deliberately wrong rule ("broken") must fail, to show the test can see an error.
// Usage: test_accept [samples=2000000] [V=5] [L=3]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "accept.cuh"

using namespace q27;

namespace {

int V = 5, L = 3;
constexpr int ND = 3;

uint64_t ctx_hash(const std::vector<int>& ctx, uint64_t salt) {
  uint64_t h = 1469598103934665603ull ^ salt;
  for (int t : ctx) { h ^= (uint64_t)(t + 1); h *= 1099511628211ull; }
  h ^= (uint64_t)ctx.size() * 0x9e3779b97f4a7c15ull;
  return h;
}
float gauss(uint64_t h, int v) {  // deterministic normal-ish value from a hash
  float s = 0.f;
  for (int k = 0; k < 4; k++) s += rng01(h, v, k, 77);
  return (s - 2.f) * 1.7320508f;
}
// Target distribution after the prefix (as a candidate row, all V tokens, some may be dropped).
CandRow target(const std::vector<int>& ctx) {
  const uint64_t h = ctx_hash(ctx, 1);
  std::vector<float> p(V);
  float mx = -1e30f;
  for (int v = 0; v < V; v++) { p[v] = 1.8f * gauss(h, v); mx = std::max(mx, p[v]); }
  double s = 0;
  for (int v = 0; v < V; v++) { p[v] = expf(p[v] - mx); s += p[v]; }
  if (h % 3 == 0) {  // drop the least likely token (like top-p)
    int lo = 0;
    for (int v = 1; v < V; v++) if (p[v] < p[lo]) lo = v;
    s -= p[lo]; p[lo] = 0.f;
  }
  CandRow c{};
  for (int v = 0; v < V; v++) if (p[v] > 0.f) { c.id[c.n] = v; c.p[c.n] = (float)(p[v] / s); c.n++; }
  return c;
}
// Draft distribution: the target's logits plus noise, a different temperature, and for some prefixes only the
// top 3 tokens (like the draft's top-k).
CandRow draft(const std::vector<int>& ctx) {
  const uint64_t h = ctx_hash(ctx, 1), h2 = ctx_hash(ctx, 2);
  std::vector<float> lg(V);
  float mx = -1e30f;
  for (int v = 0; v < V; v++) { lg[v] = 1.8f * gauss(h, v) * 1.3f + 1.2f * gauss(h2, v); mx = std::max(mx, lg[v]); }
  std::vector<int> ord(V);
  for (int v = 0; v < V; v++) ord[v] = v;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });
  const int keep = h2 % 2 == 0 ? 3 : V;
  double s = 0;
  for (int k = 0; k < keep; k++) s += expf(lg[ord[k]] - mx);
  CandRow c{};
  for (int k = 0; k < keep; k++) { c.id[c.n] = ord[k]; c.p[c.n] = (float)(expf(lg[ord[k]] - mx) / s); c.n++; }
  return c;
}

struct Result { double chi2, max_z, tok_per_round; int worst; double worst_e; long worst_o; };

Result run(int rule, long samples, uint64_t seed) {  // rule 0 = token, 1 = block, 2 = broken block
  int cells = 1;
  for (int i = 0; i < L; i++) cells *= V;
  std::vector<long> hist(cells, 0);
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  long rounds = 0, emitted = 0;
  int counter = 0;
  for (long n = 0; n < samples; n++) {
    std::vector<int> out;
    while ((int)out.size() < L) {
      int d[ND];
      CandRow pc[ND + 1], qc[ND];
      std::vector<int> ctx = out;
      for (int i = 0; i < ND; i++) {
        qc[i] = draft(ctx);
        d[i] = cand_draw(qc[i], uni(gen));
        pc[i] = target(ctx);
        ctx.push_back(d[i]);
      }
      pc[ND] = target(ctx);
      int emit[ND + 1];
      const int cnt = rule == 0 ? accept_token(d, ND, pc, qc, seed, counter, emit)
                                : accept_block(d, ND, pc, qc, seed, counter, emit, rule == 2);
      counter++;
      rounds++; emitted += cnt;
      for (int i = 0; i < cnt; i++) out.push_back(emit[i]);
    }
    int cell = 0;
    for (int i = 0; i < L; i++) cell = cell * V + out[i];
    hist[cell]++;
  }
  Result r{0, 0, (double)emitted / rounds, -1, 0, 0};
  for (int cell = 0; cell < cells; cell++) {
    std::vector<int> seq(L);
    for (int i = L - 1, c = cell; i >= 0; i--, c /= V) seq[i] = c % V;
    double prob = 1;
    std::vector<int> ctx;
    for (int i = 0; i < L; i++) { prob *= cand_p(target(ctx), seq[i]); ctx.push_back(seq[i]); }
    const double e = prob * samples;
    if (e > 0) {
      r.chi2 += (hist[cell] - e) * (hist[cell] - e) / e;
      const double z = fabs(hist[cell] - e) / sqrt(e);
      if (z > r.max_z) { r.max_z = z; r.worst = cell; r.worst_e = e; r.worst_o = hist[cell]; }
    } else if (hist[cell] > 0) {
      r.chi2 = INFINITY;  // a sequence the target can never produce
    }
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const long samples = argc > 1 ? atol(argv[1]) : 2000000;
  if (argc > 2) V = atoi(argv[2]);
  if (argc > 3) L = atoi(argv[3]);
  int cells = 1;
  for (int i = 0; i < L; i++) cells *= V;
  // cells with probability 0 do not count; the limit uses all cells (a slightly loose bound)
  const double df = cells - 1, limit = df + 5 * sqrt(2 * df);
  printf("V=%d L=%d, %ld samples, %d cells, chi-square limit %.0f (df %.0f)\n", V, L, samples, cells, limit, df);
  const char* names[3] = {"token", "block", "broken"};
  int fails = 0;
  for (int rule = 0; rule < 3; rule++) {
    const uint64_t seed0 = getenv("ACCEPT_SEED") ? strtoull(getenv("ACCEPT_SEED"), nullptr, 10) : 12345;
    const Result r = run(rule, samples, seed0 + rule);
    const bool ok = r.chi2 < limit;
    printf("  %-6s  tokens per round %.4f  chi-square %.1f  max |z| %.2f  %s\n", names[rule], r.tok_per_round, r.chi2,
           r.max_z, rule == 2 ? (ok ? "NOT DETECTED (test too weak)" : "detected (expected)") : (ok ? "PASS" : "FAIL"));
    if (getenv("ACCEPT_WORST")) printf("          worst cell %d: expected %.1f, observed %ld\n", r.worst, r.worst_e, r.worst_o);
    if ((rule < 2) != ok) fails++;
  }
  printf(fails ? "FAILED\n" : "ALL PASS\n");
  return fails ? 1 : 0;
}
