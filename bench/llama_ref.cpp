// Long-context logit reference from llama.cpp (production build in qwen38_27\bin-parches, used read-only).
// llama-perplexity keeps every logit of a chunk in RAM (n_ctx/2 x 248320 floats), which does not fit at
// 32k+ tokens. This tool runs llama.cpp over n_tok tokens and keeps the log-probabilities only for the
// last n_scored positions, in llama-perplexity's 16-bit format.
//
// Usage: llama_ref <model.gguf> <text.txt> <out.bin> <n_tok> <n_scored> <ubatch> [split=tensor|none] [main_gpu]
// ubatch 0: tokenize only (no GPU), write a q27ref file with tokens and no rows.
// Output "q27ref01": int32 n_tok, n_vocab, first, n_scored, ubatch; int32 tokens[n_tok];
//                   then n_scored rows of nv uint16 (nv = 2*((n_vocab+1)/2) + 4).
// Row i holds the distribution after token first+i (it predicts token first+i+1).
// Build: bench\build-llama-ref.bat. Run with bin-parches on PATH and CUDA_DEVICE_ORDER=PCI_BUS_ID.
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i; memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}
// llama.cpp tools/perplexity/perplexity.cpp (MIT): 16-bit log-prob compression.
static void compress(int n_vocab, const float* logits, uint16_t* log_prob) {
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
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: llama_ref <model> <text> <out> <n_tok> <n_scored> <ubatch> [split=tensor|none] [main_gpu]\n");
    return 1;
  }
  const int n_tok = atoi(argv[4]), n_scored = atoi(argv[5]), ub = atoi(argv[6]);
  const std::string split = argc > 7 ? argv[7] : "tensor";
  const int main_gpu = argc > 8 ? atoi(argv[8]) : 0;
  const int first = n_tok - 1 - n_scored;
  if (first < 0 && ub > 0) { fprintf(stderr, "n_scored too large\n"); return 1; }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 99;
  mp.split_mode = split == "tensor" ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_NONE;
  mp.main_gpu = main_gpu;
  mp.load_mode = LLAMA_LOAD_MODE_NONE;
  mp.load_mtp = false;
  mp.vocab_only = ub == 0;  // ubatch 0: only tokenize and write the tokens
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  std::ifstream tf(argv[2], std::ios::binary);
  std::stringstream ss; ss << tf.rdbuf();
  const std::string text = ss.str();
  std::vector<llama_token> toks(text.size() + 16);
  const int nt = llama_tokenize(vocab, text.data(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), true, false);
  if (nt < n_tok) { fprintf(stderr, "text has %d tokens, need %d\n", nt, n_tok); return 1; }
  toks.resize(n_tok);
  if (ub == 0) {
    std::ofstream out(argv[3], std::ios::binary);
    out.write("q27ref01", 8);
    const int32_t hdr[5] = {n_tok, n_vocab, n_tok - 1, 0, 0};
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)toks.data(), (size_t)n_tok * 4);
    printf("wrote %d tokens (text has %d)\n", n_tok, nt);
    return 0;
  }

  const int n_batch = 512;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)((n_tok + 255) / 256 * 256);
  cp.n_batch = n_batch;
  cp.n_ubatch = ub;
  cp.n_seq_max = 1;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  cp.no_perf = true;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

  std::ofstream out(argv[3], std::ios::binary);
  out.write("q27ref01", 8);
  const int32_t hdr[5] = {n_tok, n_vocab, first, n_scored, ub};
  out.write((const char*)hdr, sizeof(hdr));
  out.write((const char*)toks.data(), (size_t)n_tok * 4);

  const int nv = 2 * ((n_vocab + 1) / 2) + 4;
  std::vector<uint16_t> lp(nv);
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  const auto t0 = std::chrono::steady_clock::now();
  for (int p = 0; p < n_tok; p += n_batch) {
    const int n = std::min(n_batch, n_tok - p);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
      batch.token[i] = toks[p + i];
      batch.pos[i] = p + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      const int q = p + i;
      batch.logits[i] = q >= first && q < first + n_scored;
    }
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed at %d\n", p); return 1; }
    for (int i = 0; i < n; i++) {
      if (!batch.logits[i]) continue;
      compress(n_vocab, llama_get_logits_ith(ctx, i), lp.data());
      out.write((const char*)lp.data(), (size_t)nv * 2);
    }
    if ((p / n_batch) % 32 == 0) {
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      fprintf(stderr, "%d / %d tokens, %.0f s\n", p + n, n_tok, s);
    }
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("done: %d tokens (ubatch %d), scored %d..%d, %.1f s\n", n_tok, ub, first, first + n_scored - 1, s);
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  return 0;
}
