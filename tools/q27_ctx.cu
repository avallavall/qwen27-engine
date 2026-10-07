// Context sizing check (A3). Loads the model on two cards, computes the largest context that leaves the
// reserves free, allocates the decoder, writes KV at the end of the context, and reports VRAM per card.
// Then it allocates a block of the size reserved for vision on card 1 and runs again.
// Usage: q27_ctx <model.gguf> [devices=0,1] [n_ctx=auto]
// Reserves (MiB), as q27_server: card 0: prefill 1100 + prompt-cache checkpoints 300 + desktop growth 600;
//                 card 1: prefill 1100 + checkpoints 300 + vision 1536.
#include "common.cuh"
#include "model.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <string>
#include <vector>

using namespace q27;

static std::vector<int> parse_devs(const std::string& d) {
  std::vector<int> v;
  size_t a = 0;
  while (a <= d.size()) { size_t b = d.find(',', a); if (b == std::string::npos) b = d.size(); v.push_back(std::stoi(d.substr(a, b - a))); a = b + 1; }
  return v;
}

static void report(const Model& m, const char* tag) {
  printf("%s:", tag);
  for (auto& sh : m.shards) {
    CK(cudaSetDevice(sh.dev));
    size_t f, t;
    CK(cudaMemGetInfo(&f, &t));
    printf("  card %d used %.0f MiB, free %.0f MiB", sh.dev, (t - f) / 1048576.0, f / 1048576.0);
  }
  printf("\n");
}

int main(int argc, char** argv) try {
  if (argc < 2) { fprintf(stderr, "usage: q27_ctx <model.gguf> [devices] [n_ctx]\n"); return 1; }
  const std::vector<int> devs = parse_devs(argc > 2 ? argv[2] : "0,1");
  const size_t MiB = 1 << 20;
  Model model(argv[1], devs);
  report(model, "after model load");
  const std::vector<size_t> reserve = {(1100 + 300 + 600) * MiB, (1100 + 300 + 1536) * MiB};  // as q27_server
  const int fit = Decoder::fit_ctx(model, reserve, Decoder::kv_q8_from_env());
  const int n_ctx = argc > 3 ? atoi(argv[3]) : fit;
  printf("KV %.1f KiB per token per card; decoder fixed %.0f MiB per card; fit_ctx = %d; using n_ctx = %d\n",
         Decoder::kv_bytes_per_token(model, Decoder::kv_q8_from_env()) / 1024.0, Decoder::fixed_bytes(model) / (double)MiB, fit, n_ctx);
  Decoder dec(model, n_ctx, Decoder::kv_q8_from_env());
  // touch the whole KV range: passes at the start and at the very end of the context
  const int toks[4] = {198, 271, 1032, 361};
  dec.step(toks, 4, 0);
  dec.step(toks, 4, n_ctx - 4);
  dec.step(toks, 1, n_ctx - 5);
  report(model, "after decoder + passes at the end of the context");
  // vision stand-in on card 1
  if (model.tp() == 2) {
    CK(cudaSetDevice(model.shards[1].dev));
    void* vis = nullptr;
    CK(cudaMalloc(&vis, 1536 * MiB));
    CK(cudaMemset(vis, 1, 1536 * MiB));
    dec.step(toks, 4, n_ctx - 8);
    report(model, "with 1536 MiB vision block on card 1");
    if (const char* hold = getenv("Q27_CTX_HOLD")) {  // keep the memory for an outside check (nvidia-smi)
      printf("holding %s s\n", hold);
      fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(atoi(hold)));
    }
    CK(cudaFree(vis));
  }
  printf("OK: n_ctx %d\n", n_ctx);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
