// Test of the prompt-cache API of the Decoder (state_save / state_load, keep_from_last_step, kv_to_host /
// kv_from_host). Usage: test_cache <model.gguf> <tokens file (q27ref or llama base, e.g. bench\out\tok_200k.bin)>
//   1. restore: prefill 1500 tokens, save the state, prefill 1500 more -> logits L1; load the state at 1500,
//      prefill the same 1500 -> logits L2. Same passes, so L1 == L2 exactly.
//   2. swap: prefill 3000 tokens, save state + KV to host; feed 8 tokens -> L1. Reset, read 2000 other tokens,
//      load KV + state, feed the same 8 tokens -> L2. L1 == L2 exactly.
//   3. cut: greedy speculative decoding until a step emits >= 2 tokens; keep only the first one; feed 24 tokens
//      -> L1. Reference: the same token sequence read from scratch -> L2. Different pass kinds, so L1 ~ L2
//      (same top token, small difference). A control without the cut must differ much more.
#include "common.cuh"
#include "model.h"
#include "ops.h"
#include "reffile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using namespace q27;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Diff { double max_abs, rms; int top_a, top_b; };
static Diff diff(const std::vector<float>& a, const std::vector<float>& b) {
  Diff d{0, 0, 0, 0};
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const double x = std::fabs((double)a[i] - b[i]);
    d.max_abs = std::max(d.max_abs, x);
    s += x * x;
  }
  d.rms = std::sqrt(s / a.size());
  d.top_a = (int)(std::max_element(a.begin(), a.end()) - a.begin());
  d.top_b = (int)(std::max_element(b.begin(), b.end()) - b.begin());
  return d;
}

int main(int argc, char** argv) try {
  if (argc < 3) { fprintf(stderr, "usage: test_cache <model.gguf> <tokens.bin>\n"); return 1; }
  const RefFile ref = read_ref(argv[2]);
  const std::vector<int>& A = ref.tokens;
  if (A.size() < 8000) throw std::runtime_error("token file too short");
  Model model(argv[1], {0, 1});
  Decoder dec(model, 16384, Decoder::kv_q8_from_env());
  const int V = model.hp().vocab;
  const size_t sb = dec.state_bytes();
  printf("KV %s, state %.1f MB per card\n", dec.kv_q8() ? "q8_0" : "f16", sb / 1048576.0);
  std::vector<uint8_t*> H(2);
  for (auto& h : H) CK(cudaHostAlloc(&h, sb, cudaHostAllocPortable));
  std::vector<float> L1(V), L2(V);
  int fails = 0;
  auto check = [&](const char* name, bool ok, const Diff& d) {
    printf("%-8s %s  max|d| %.6f  rms %.6f  top %d / %d\n", name, ok ? "PASS" : "FAIL", d.max_abs, d.rms, d.top_a, d.top_b);
    if (!ok) fails++;
  };

  // 1. restore
  {
    dec.reset();
    dec.prefill(A.data(), 1500);
    auto t0 = clk::now();
    dec.state_save(H);
    dec.state_wait();
    printf("state_save + wait: %.1f ms\n", ms_since(t0));
    dec.prefill(A.data() + 1500, 1500);
    dec.logits_to_host(L1.data());
    t0 = clk::now();
    dec.state_load(H, 1500);
    printf("state_load: %.1f ms\n", ms_since(t0));
    dec.prefill(A.data() + 1500, 1500);
    dec.logits_to_host(L2.data());
    const Diff d = diff(L1, L2);
    check("restore", d.max_abs == 0.0, d);
  }
  // 2. swap
  {
    dec.reset();
    dec.prefill(A.data(), 3000);
    dec.state_save(H);
    dec.state_wait();
    const size_t kvb = Decoder::kv_bytes_per_token(model, dec.kv_q8()) * 3000;
    std::vector<std::vector<uint8_t>> KV(2, std::vector<uint8_t>(kvb));
    auto t0 = clk::now();
    dec.kv_to_host({KV[0].data(), KV[1].data()}, 0, 3000);
    const double ms_out = ms_since(t0);
    const int X[8] = {A[6000], A[6001], A[6002], A[6003], A[6004], A[6005], A[6006], A[6007]};
    dec.feed(X, 8);
    dec.logits_to_host(L1.data());
    dec.reset();
    dec.prefill(A.data() + 5000, 2000);
    t0 = clk::now();
    dec.kv_from_host({KV[0].data(), KV[1].data()}, 0, 3000);
    const double ms_in = ms_since(t0);
    dec.state_load(H, 3000);
    dec.feed(X, 8);
    dec.logits_to_host(L2.data());
    printf("KV 3000 tokens, %.1f MB per card: to host %.1f ms, from host %.1f ms\n", kvb / 1048576.0, ms_out, ms_in);
    const Diff d = diff(L1, L2);
    check("swap", d.max_abs == 0.0, d);
  }
  // 3. cut
  {
    SampleParams sp;
    sp.temp = 0.0f;
    // Greedy speculative decoding from the same prompt is deterministic; run() replays it up to the first step
    // that emits >= 2 tokens. seq = tokens in the caches before that step, pending = the token after them.
    std::vector<int> seq;
    int pending = 0, out[4], n = 0;
    auto run = [&]() {
      seq.assign(A.begin(), A.begin() + 1000);
      dec.reset();
      dec.prefill(seq.data(), (int)seq.size());
      pending = dec.start(sp);
      for (int steps = 0;; steps++) {
        n = dec.spec_step(out);
        if (n >= 2) break;
        if (steps > 200) throw std::runtime_error("no step with 2 or more tokens");
        seq.push_back(pending);
        pending = out[0];
      }
    };
    auto scratch = [&](const std::vector<int>& toks, std::vector<float>& L) {
      dec.reset();
      dec.prefill(toks.data(), (int)toks.size());
      dec.logits_to_host(L.data());
    };
    std::vector<int> X = {0, A[7000], A[7001], A[7002]};  // X[0] = the pending token after the step
    // baseline: no cut. Caches: seq, pending, out[0..n-2]; pending out[n-1].
    run();
    const int n_step = n;
    X[0] = out[n - 1];
    dec.feed(X.data(), 4);
    std::vector<float> Lb(V), Lb2(V);
    dec.logits_to_host(Lb.data());
    std::vector<int> full = seq;
    full.push_back(pending);
    for (int i = 0; i + 1 < n; i++) full.push_back(out[i]);
    full.insert(full.end(), X.begin(), X.end());
    scratch(full, Lb2);
    const Diff db = diff(Lb, Lb2);
    printf("baseline (no cut) max|d| %.6f  rms %.6f  top %d / %d\n", db.max_abs, db.rms, db.top_a, db.top_b);
    // cut: keep only out[0]. Caches: seq, pending; pending out[0].
    run();
    dec.keep_from_last_step(1);
    seq.push_back(pending);
    if (dec.position() != (int)seq.size()) throw std::runtime_error("position mismatch after the cut");
    X[0] = out[0];
    dec.feed(X.data(), 4);
    dec.logits_to_host(L1.data());
    full = seq;
    full.insert(full.end(), X.begin(), X.end());
    scratch(full, L2);
    const Diff d = diff(L1, L2);
    // control: the state with one extra token (out[0] twice) must be far from the reference
    std::vector<int> wrong = seq;
    wrong.push_back(out[0]);
    wrong.insert(wrong.end(), X.begin(), X.end());
    std::vector<float> L3(V);
    scratch(wrong, L3);
    const Diff dc = diff(L3, L2);
    printf("control (extra token) max|d| %.6f  rms %.6f\n", dc.max_abs, dc.rms);
    check("cut", d.top_a == d.top_b && d.rms < 3 * db.rms + 1e-3 && dc.rms > 4 * d.rms, d);
    printf("  (the step emitted %d tokens; kept 1)\n", n_step);
  }
  // 4. image rows and explicit positions: rows given as embeddings (the tokens' own embedding vectors, on card 1)
  //    with ids < 0, and rope3 = (p, p, p), must give exactly the logits of the plain token prefill.
  {
    const int N = 600, K = 200, S = 100;  // rows S .. S+K-1 become "image rows"
    std::vector<int> toks(A.begin() + 2000, A.begin() + 2000 + N);
    dec.reset();
    dec.prefill(toks.data(), N);
    dec.logits_to_host(L1.data());
    const int E = model.hp().n_embd;
    CK(cudaSetDevice(model.shards[1].dev));
    int* dids;
    float* emb;
    CK(cudaMalloc(&dids, sizeof(int) * K));
    CK(cudaMalloc(&emb, sizeof(float) * (size_t)K * E));
    CK(cudaMemcpy(dids, toks.data() + S, sizeof(int) * K, cudaMemcpyHostToDevice));
    get_rows_iq2_s(model.tok_embd, model.tok_embd_row_bytes, dids, K, emb, E, 0);
    CK(cudaDeviceSynchronize());
    dec.set_image(emb, K, model.shards[1].dev);
    std::vector<int> mixed = toks;
    for (int i = 0; i < K; i++) mixed[S + i] = -(i + 1);
    std::vector<int> rope(3 * (size_t)N);
    for (int i = 0; i < N; i++) rope[3 * i] = rope[3 * i + 1] = rope[3 * i + 2] = i;
    dec.reset();
    dec.prefill(mixed.data(), N, rope.data());
    dec.logits_to_host(L2.data());
    const Diff d = diff(L1, L2);
    check("img+rope", d.max_abs == 0.0, d);
    // a different position for the image rows must change the logits (the rope3 path is used)
    for (int i = 0; i < K; i++) rope[3 * (S + i) + 1] += 7;
    dec.reset();
    dec.prefill(mixed.data(), N, rope.data());
    dec.logits_to_host(L2.data());
    const Diff d2 = diff(L1, L2);
    printf("  (h positions of the image rows moved by 7: rms %.4f, must be > 0)\n", d2.rms);
    if (!(d2.rms > 0)) { printf("rope3 FAIL\n"); fails++; }
    CK(cudaSetDevice(model.shards[1].dev));
    cudaFree(dids);
    cudaFree(emb);
  }
  for (auto* h : H) cudaFreeHost(h);
  printf("%s (%d failures)\n", fails ? "FAIL" : "ALL PASS", fails);
  return fails ? 1 : 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
