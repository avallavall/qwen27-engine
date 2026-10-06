// Logit test against llama.cpp.
// Mode 1 (llama-perplexity base file, "_logits_"): reads the tokens, runs this engine over each chunk and
//   writes a base file in the same format with this engine's log-probabilities. Then
//   `llama-perplexity --kl-divergence --kl-divergence-base <out>` compares them.
// Mode 2 (bench/llama_ref file, "q27ref01"): runs the engine over the tokens and compares the scored
//   positions directly (same KLD formula as llama-perplexity); out_base.bin is not written ("-").
// Usage: q27_ppl <model.gguf> <ref.bin> <out_base.bin|-> [devices=1, or 0,1 for two cards] [max_chunks=all] [T=1|4|P]
// T = P (q27ref files only): batched prompt reading (Decoder::prefill, batch size Q27_PREFILL_BATCH, default 512).
#include "common.cuh"
#include "model.h"
#include "reffile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace q27;

// llama.cpp tools/perplexity/perplexity.cpp (MIT): 16-bit log-prob compression.
static int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i; memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}
static double log_softmax(int n_vocab, const float* logits, uint16_t* log_prob, int tok) {
  float max_logit = logits[0], min_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) { max_logit = std::max(max_logit, logits[i]); min_logit = std::min(min_logit, logits[i]); }
  min_logit = std::max(min_logit, max_logit - 16);
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
  const float log_sum_exp = (float)log(sum_exp);
  const float min_log_prob = min_logit - max_logit - log_sum_exp;
  const float scale = (max_logit - min_logit) / 65535.f;
  float* d = (float*)log_prob;
  d[0] = scale; d[1] = min_log_prob;
  log_prob += 4;
  if (scale) {
    const float inv_scale = 1 / scale;
    for (int i = 0; i < n_vocab; ++i) log_prob[i] = logits[i] > min_logit ? (uint16_t)nearest_int(inv_scale * (logits[i] - min_logit)) : 0;
  } else {
    memset(log_prob, 0, n_vocab * sizeof(uint16_t));
  }
  return max_logit + log_sum_exp - logits[tok];
}

// Same statistics as llama-perplexity --kl-divergence (KLD of the base distribution against ours).
struct Kld {
  double sum_kld = 0, sum_kld2 = 0, sum_nll = 0, sum_nll_base = 0, max_kld = 0;
  long count = 0, same_top = 0;
  void add(int n_vocab, const float* logits, const uint16_t* base, int tok) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > max_logit) { max_logit = logits[i]; imax = i; }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = (float)log(sum_exp);
    const float* d = (const float*)base;
    const float scale = d[0], min_log_prob = d[1];
    base += 4;
    sum_nll += max_logit + log_sum_exp - logits[tok];
    sum_nll_base += -(scale * base[tok] + min_log_prob);
    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float pmax = 0;
    for (int i = 0; i < n_vocab; ++i) {
      const float plb = scale * base[i] + min_log_prob;
      if (i == 0 || plb > pmax) { pmax = plb; imax_base = i; }
      if (plb > -16.f) sum += expf(plb) * (plb - logits[i] + max_logit);
    }
    sum_kld += sum; sum_kld2 += sum * sum; max_kld = std::max(max_kld, sum);
    count++;
    if (imax == imax_base) same_top++;
  }
  void print(const char* tag) const {
    const double m = sum_kld / count;
    const double sd = count > 10 ? sqrt(std::max(0.0, sum_kld2 / count - m * m) / (count - 1)) : 0;
    const double p = (double)same_top / count;
    printf("%s: %ld positions | mean KLD %.6f +- %.6f | max KLD %.4f | same top %.3f +- %.3f %% | PPL %.4f (base %.4f)\n", tag,
           count, m, sd, max_kld, 100 * p, 100 * sqrt(p * (1 - p) / count), exp(sum_nll / count), exp(sum_nll_base / count));
    fflush(stdout);
  }
};

int main(int argc, char** argv) try {
  if (argc < 4) { fprintf(stderr, "usage: q27_ppl <model.gguf> <ref.bin> <out_base.bin|-> [devices] [max_chunks] [T]\n"); return 1; }
  std::vector<int> devs;
  {
    const std::string d = argc > 4 ? argv[4] : "1";
    size_t a = 0;
    while (a <= d.size()) { size_t b = d.find(',', a); if (b == std::string::npos) b = d.size(); devs.push_back(std::stoi(d.substr(a, b - a))); a = b + 1; }
  }
  const int max_chunks = argc > 5 ? atoi(argv[5]) : 1 << 30;
  const bool prefill = argc > 6 && std::string(argv[6]) == "P";  // batched prompt reading (q27ref files)
  const int T = (argc > 6 && !prefill) ? atoi(argv[6]) : 1;

  RefFile ref = read_ref(argv[2]);
  const int n_ctx = ref.n_ctx, n_vocab = ref.n_vocab;
  const int n_chunk = std::min(ref.n_chunk, max_chunks);
  printf("reference: %s, %d tokens%s, n_vocab %d", ref.q27 ? "q27ref" : "llama-perplexity", n_ctx, ref.q27 ? "" : " per chunk", n_vocab);
  if (ref.q27) printf(", scored %d..%d, llama ubatch %d", ref.first, ref.first + ref.n_scored - 1, ref.ubatch);
  else printf(", chunks %d", n_chunk);
  printf("\n");

  auto t0 = std::chrono::steady_clock::now();
  Model model(argv[1], devs);
  printf("model loaded in %.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  for (auto& sh : model.shards) printf(", %.1f MiB on device %d", sh.vram_bytes / 1048576.0, sh.dev);
  printf("\n");
  if (model.hp().vocab != n_vocab) throw std::runtime_error("vocab mismatch");
  Decoder dec(model, n_ctx + 8, Decoder::kv_q8_from_env());
  printf("KV cache: %s\n", dec.kv_q8() ? "q8_0" : "f16");
  const int nv = 2 * ((n_vocab + 1) / 2) + 4;
  std::vector<float> logits(n_vocab);
  std::vector<uint16_t> lp(nv);

  if (ref.q27 && prefill) {
    // Batched prompt reading: passes of max_batch() tokens; the passes that hold scored rows keep all logits.
    std::ifstream in(argv[2], std::ios::binary);
    in.seekg(ref.rows_off);
    const int first = ref.first, end = ref.first + ref.n_scored;
    const int32_t* tk = ref.tokens.data();
    const int MB = dec.max_batch();
    Kld k;
    auto tr = std::chrono::steady_clock::now();
    double pf_s = 0; int pf_tok = 0;
    for (int p = 0; p < end;) {
      const int M = std::min(MB, end - p);
      auto ts = std::chrono::steady_clock::now();
      if (p + M > first) {
        dec.prefill_logits(tk + p, M);
        for (int t = 0; t < M; t++) {
          const int q = p + t;
          if (q < first) continue;
          in.read((char*)lp.data(), (size_t)nv * 2);
          if (!in) throw std::runtime_error("reference rows truncated");
          dec.prefill_logits_to_host(logits.data(), t);
          k.add(n_vocab, logits.data(), lp.data(), tk[q + 1]);
        }
      } else {
        dec.prefill(tk + p, M);
        pf_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        pf_tok += M;
      }
      p += M;
      if (p % 16384 < M) {
        printf("  %d / %d tokens, %.1f s, %.0f t/s in plain prefill passes\n", p, end,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count(), pf_tok / std::max(pf_s, 1e-9));
        fflush(stdout);
      }
    }
    k.print("result (prefill)");
    return 0;
  }

  if (ref.q27) {
    std::ifstream in(argv[2], std::ios::binary);
    in.seekg(ref.rows_off);
    const int first = ref.first, end = ref.first + ref.n_scored;
    const int32_t* tk = ref.tokens.data();
    Kld k;
    double step_ms = 0; int steps = 0;
    auto tr = std::chrono::steady_clock::now();
    for (int p = 0; p < end;) {
      const int Tp = (end - p >= T) ? T : 1;
      auto ts = std::chrono::steady_clock::now();
      dec.step(tk + p, Tp, p);
      step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count(); steps++;
      for (int t = 0; t < Tp; t++) {
        const int q = p + t;
        if (q < first) continue;
        in.read((char*)lp.data(), (size_t)nv * 2);
        if (!in) throw std::runtime_error("reference rows truncated");
        dec.logits_to_host(logits.data(), t);
        k.add(n_vocab, logits.data(), lp.data(), tk[q + 1]);
      }
      p += Tp;
      if (p % 16384 < Tp) {
        printf("  %d / %d tokens, %.1f s, %.2f ms per pass\n", p, end,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count(), step_ms / steps);
        fflush(stdout);
        step_ms = 0; steps = 0;
      }
    }
    k.print("result");
    return 0;
  }

  std::ofstream out(argv[3], std::ios::binary);
  out.write("_logits_", 8);
  const uint32_t nc = (uint32_t)n_ctx;
  out.write((const char*)&nc, 4);
  out.write((const char*)&n_vocab, 4);
  out.write((const char*)&n_chunk, 4);
  out.write((const char*)ref.tokens.data(), (size_t)n_ctx * n_chunk * 4);

  const int first = n_ctx / 2;
  double nll = 0; int count = 0;
  double step_ms = 0; int steps = 0;
  for (int c = 0; c < n_chunk; c++) {
    dec.reset();
    const int32_t* tk = ref.tokens.data() + (size_t)c * n_ctx;
    for (int p = 0; p < n_ctx - 1; p += T) {
      auto ts = std::chrono::steady_clock::now();
      dec.step(tk + p, T, p);
      step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count(); steps++;
      for (int t = 0; t < T; t++) {
        const int q = p + t;
        if (q < first || q >= n_ctx - 1) continue;
        dec.logits_to_host(logits.data(), t);
        nll += log_softmax(n_vocab, logits.data(), lp.data(), tk[q + 1]);
        count++;
        out.write((const char*)lp.data(), nv * 2);
      }
    }
    printf("[%d] ppl %.4f  (%.2f ms per step)\n", c + 1, exp(nll / count), step_ms / steps);
    fflush(stdout);
  }
  printf("final ppl %.4f over %d tokens\n", exp(nll / count), count);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
