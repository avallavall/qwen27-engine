// Generation test for the MTP speculative loop.
// Usage: q27_gen <model.gguf> <llama_base.bin> [devices=0,1] [prompt_tokens=400] [gen=200] [mode=all|greedy|sample]
// The prompt is the first prompt_tokens tokens of the reference file (llama-perplexity base or q27ref).
//  greedy: plain decoding (argmax, one token per pass) vs speculative decoding at temp 0. The token
//          sequences must be identical.
//  sample: speculative decoding at temp 1.0, top-k 20, top-p 0.95 (production settings); prints ms per
//          step, tokens per step and tok/s.
//  depth:  q27_gen <model> <tokens> <devices> <depths, e.g. 30000,100000,150000> <gen> depth [runs=2]
//          feeds the token file up to each depth, then generates `gen` tokens (sample settings) `runs`
//          times and prints ms per step at that depth. Generated tokens stay in the context.
#include "common.cuh"
#include "model.h"
#include "prof.h"
#include "reffile.h"
#include "tokenizer.h"

#include <sstream>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace q27;
using clk = std::chrono::steady_clock;

static std::vector<int> parse_devs(const std::string& d) {
  std::vector<int> v;
  size_t a = 0;
  while (a <= d.size()) { size_t b = d.find(',', a); if (b == std::string::npos) b = d.size(); v.push_back(std::stoi(d.substr(a, b - a))); a = b + 1; }
  return v;
}

int main(int argc, char** argv) try {
  if (argc < 3) { fprintf(stderr, "usage: q27_gen <model.gguf> <llama_base.bin> [devices] [prompt] [gen] [mode]\n"); return 1; }
  const std::vector<int> devs = parse_devs(argc > 3 ? argv[3] : "0,1");
  const int np = argc > 4 ? atoi(argv[4]) : 400;
  const int ngen = argc > 5 ? atoi(argv[5]) : 200;
  const std::string mode = argc > 6 ? argv[6] : "all";

  if (mode == "depth") {
    const RefFile ref = read_ref(argv[2]);
    std::vector<int> depths;
    for (const std::string& x : {std::string(argc > 4 ? argv[4] : "30000")}) {
      size_t a = 0;
      while (a <= x.size()) { size_t b = x.find(',', a); if (b == std::string::npos) b = x.size(); depths.push_back(std::stoi(x.substr(a, b - a))); a = b + 1; }
    }
    const int runs = argc > 7 ? atoi(argv[7]) : 2;
    const int max_d = *std::max_element(depths.begin(), depths.end());
    if (max_d > (int)ref.tokens.size()) throw std::runtime_error("token file too short");
    Model model(argv[1], devs);
    Decoder dec(model, max_d + (int)depths.size() * runs * (ngen + 16) + 64, Decoder::kv_q8_from_env());
    printf("n_ctx %d\n", dec.n_ctx());
    dec.reset();
    int src = 0;  // next token of the file to feed
    for (int D : depths) {
      const int n = D - dec.position();
      if (n > 0) {
        auto t0 = clk::now();
        dec.prefill(ref.tokens.data() + src, n);
        const double s = std::chrono::duration<double>(clk::now() - t0).count();
        printf("fed %d tokens to depth %d in %.1f s (%.0f t/s)\n", n, dec.position(), s, n / s);
        prof::report("prefill batch");
        src += n;
      }
      for (int run = 0; run < runs; run++) {
        SampleParams sp;
        sp.seed = 2000 + run;
        const int d0 = dec.position();
        std::vector<int> toks;
        toks.push_back(dec.start(sp));
        int steps = 0;
        auto t1 = clk::now();
        while ((int)toks.size() < ngen) {
          int out[4];
          const int k = dec.spec_step(out);
          for (int i = 0; i < k; i++) toks.push_back(out[i]);
          steps++;
        }
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
        const int gen = (int)toks.size() - 1;
        printf("depth %d run %d: %d tokens in %d steps: %.2f ms/step, %.2f tok/step, %.1f tok/s\n", d0, run, gen, steps,
               ms / steps, (double)gen / steps, gen / (ms / 1000.0));
        fflush(stdout);
      }
    }
    return 0;
  }
  if (mode == "accept") {
    // Draft acceptance on fixed prompts: q27_gen <model> <tokens file | text.txt> <devices> <prompt> <gen> accept [n=16]
    // n prompts of `prompt` tokens spread over the input, one fresh generation of `gen` tokens each (sample settings,
    // random counter fixed per prompt), so two settings (e.g. Q27_DRAFT_VOCAB) see the same prompts and draws.
    Model model(argv[1], devs);
    std::vector<int> toks;
    const std::string src = argv[2];
    if (src.size() > 4 && src.compare(src.size() - 4, 4, ".txt") == 0) {
      std::ifstream f(src, std::ios::binary);
      std::stringstream ss;
      ss << f.rdbuf();
      Tokenizer tok(*model.g_);
      toks = tok.encode(ss.str(), false);
    } else {
      const RefFile r = read_ref(src);
      toks.assign(r.tokens.begin(), r.tokens.end());
    }
    const int R = argc > 7 ? atoi(argv[7]) : 16;
    if ((int)toks.size() < np + R) throw std::runtime_error("input too short");
    Decoder dec(model, np + ngen + 64, Decoder::kv_q8_from_env());
    const int stride = ((int)toks.size() - np) / R;
    SampleParams sp;
    long tok_sum = 0, step_sum = 0;
    double ms_sum = 0;
    for (int i = 0; i < R; i++) {
      dec.reset();
      dec.prefill(toks.data() + (size_t)i * stride, np);
      dec.start(sp, 1000 + i * 7919);
      int n_out = 1, steps = 0;
      auto t1 = clk::now();
      while (n_out < ngen) {
        int out[4];
        n_out += dec.spec_step(out);
        steps++;
      }
      ms_sum += std::chrono::duration<double, std::milli>(clk::now() - t1).count();
      tok_sum += n_out - 1;
      step_sum += steps;
    }
    printf("accept: %d prompts x %d tokens: %.3f tok/step, %.2f ms/step, %.1f tok/s\n", R, ngen, (double)tok_sum / step_sum,
           ms_sum / step_sum, tok_sum / (ms_sum / 1000.0));
    return 0;
  }
  const RefFile ref = read_ref(argv[2]);
  if (np > (int)ref.tokens.size()) throw std::runtime_error("prompt longer than the token file");
  std::vector<int> prompt(ref.tokens.begin(), ref.tokens.begin() + np);

  Model model(argv[1], devs);
  Decoder dec(model, np + ngen + 64, Decoder::kv_q8_from_env());
  const int V = model.hp().vocab;

  if (mode == "all" || mode == "greedy") {
    // plain greedy
    std::vector<int> plain;
    std::vector<float> gap;  // top-1 minus top-2 logit at each plain step
    {
      int p = 0;
      while (p < np) {
        const int T = (np - p > 4) ? 4 : 1;
        dec.step(prompt.data() + p, T, p);
        p += T;
      }
      std::vector<float> lg(V);
      for (int i = 0; i < ngen; i++) {
        dec.logits_to_host(lg.data(), 0);
        const int t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        float second = -1e30f;
        for (int j = 0; j < V; j++) if (j != t && lg[j] > second) second = lg[j];
        gap.push_back(lg[t] - second);
        plain.push_back(t);
        dec.step(t, np + i);
      }
    }
    // speculative greedy
    std::vector<int> spec;
    SampleParams sp; sp.temp = 0.f;
    spec.push_back(dec.begin(prompt, sp));
    int steps = 0;
    while ((int)spec.size() < ngen) {
      int out[4];
      const int n = dec.spec_step(out);
      for (int i = 0; i < n; i++) spec.push_back(out[i]);
      steps++;
    }
    spec.resize(ngen);
    int first_diff = -1;
    for (int i = 0; i < ngen; i++) if (plain[i] != spec[i]) { first_diff = i; break; }
    printf("greedy: plain vs speculative over %d tokens: %s", ngen, first_diff < 0 ? "IDENTICAL" : "DIFFERENT");
    if (first_diff >= 0) {
      printf(" (first difference at %d: %d vs %d, plain top-1/top-2 logit gap %.5f", first_diff, plain[first_diff], spec[first_diff],
             gap[first_diff]);
      float mg = 1e30f;
      for (int i = 0; i < first_diff; i++) mg = std::min(mg, gap[i]);
      printf(", smallest gap before it %.5f)", mg);
    }
    printf("; speculative steps %d, tokens per step %.2f\n", steps, (double)(ngen - 1) / steps);
    printf("first tokens:");
    for (int i = 0; i < std::min(ngen, 16); i++) printf(" %d", spec[i]);
    printf("\n");
  }
  if (mode == "all" || mode == "sample") {
    SampleParams sp;  // temp 1.0, top-k 20, top-p 0.95, draft top-k 10
    for (int run = 0; run < 3; run++) {
      sp.seed = 1000 + run;
      std::vector<int> toks;
      auto t0 = clk::now();
      toks.push_back(dec.begin(prompt, sp));
      prof::report("prompt (prefill batches)");
      const double t_prompt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      int steps = 0;
      auto t1 = clk::now();
      while ((int)toks.size() < ngen) {
        int out[4];
        const int n = dec.spec_step(out);
        for (int i = 0; i < n; i++) toks.push_back(out[i]);
        steps++;
      }
      const double ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
      const int gen = (int)toks.size() - 1;
      printf("sample run %d: prompt %d tok in %.0f ms | %d tokens in %d steps: %.2f ms/step, %.2f tok/step, %.1f tok/s\n",
             run, np, t_prompt, gen, steps, ms / steps, (double)gen / steps, gen / (ms / 1000.0));
      prof::report("sample step");
    }
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
