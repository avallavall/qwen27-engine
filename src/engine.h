// Generation worker of the server: one compute slot, a FIFO queue of jobs, and the prompt cache.
//
// Prompt cache. The live sequence is the token list whose KV rows and recurrent state are on the cards. A new
// prompt reuses its longest common prefix with the live sequence. The recurrent state (GDN, MTP hidden) cannot be
// cut back, so a shorter reuse needs a checkpoint: a host copy of that state at a position (Decoder::state_save).
// Checkpoints are taken while reading a prompt, at the start of the second turn (end of the system turn), at the
// start of the last message, at the start of the generation prompt, and every ckpt_every tokens. When a request
// shares less than half of the live sequence, the live sequence (KV rows + state + checkpoints) is first copied to
// host RAM; a later request can load it back. All host copies share one RAM budget (cache_ram_mb).
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "chat_parser.h"
#include "chat_template.h"
#include "model.h"
#include "tokenizer.h"
#include "vision.h"

namespace q27 {

// One image of a prompt: its rows are prompt[start .. start + plan.n_tokens).
struct ImageSpan {
  int start = 0;
  VisionPlan plan;
  std::shared_ptr<std::vector<uint8_t>> rgb;  // decoded image, plan.src_w x plan.src_h RGB8
  uint64_t hash = 0;                          // of the image file bytes
};

// Prompt id of image row i (negative, so it never equals a vocabulary token; different images give different ids).
inline int image_row_id(uint64_t hash, int i) {
  return -1 - (int)(((uint32_t)(hash ^ (hash >> 32)) + (uint32_t)i * 0x9E3779B1u) & 0x3FFFFFFFu);
}

struct GenRequest {
  std::vector<int> prompt;      // token ids; image rows have image_row_id values
  std::vector<ImageSpan> images;  // sorted by start
  std::vector<int> rope;        // [prompt.size()][3] IMRoPE positions (t, h, w) when images is not empty
  SampleParams sp;
  int rng_start = 0;            // first value of the sampler's random counter
  bool thinking = true;         // the prompt ends inside <think>
  ojson tools;                  // request tools (null = no tool-call parsing)
  int max_tokens = -1;          // -1: until the context is full
  std::vector<std::string> stop;
  std::string tag;              // for the log
};

struct GenEvent {
  enum Kind { kDelta, kDone, kError } kind = kDelta;
  ParseDelta delta;
  // kDone
  std::string finish_reason;
  std::string reasoning, content;
  std::string raw;  // generated text before parsing (end-of-generation token not included)
  ojson tool_calls;
  ojson usage, timings;
  // kError
  int code = 500;
  std::string err_type = "server_error", message;
};

class Job {
 public:
  GenRequest req;
  std::atomic<bool> cancel{false};
  void push(GenEvent e);
  // Wait up to `timeout` for events; moves them into out. Returns false on timeout.
  bool pop(std::vector<GenEvent>& out, std::chrono::milliseconds timeout);

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<GenEvent> q_;
};

struct EngineOptions {
  size_t cache_ram_mb = 8192;  // host RAM for checkpoints and swapped sequences
  int ckpt_every = 16384;      // extra checkpoint every this many prompt tokens
  int max_ckpts = 8;           // per sequence
  int min_swap_tokens = 1024;  // smaller live sequences are not copied to RAM
};

class Engine {
 public:
  // venc may be null (no image input).
  Engine(const Model& m, Decoder& dec, const Tokenizer& tok, EngineOptions opt, VisionEncoder* venc = nullptr);
  ~Engine();
  void submit(std::shared_ptr<Job> j);
  int queued();
  int capacity() const;  // largest prompt + generated length

 private:
  struct HostState {
    std::vector<uint8_t*> card;
    ~HostState();
  };
  struct Ckpt {
    int pos;
    std::shared_ptr<HostState> st;
  };
  struct Seq {
    std::vector<int> toks;
    std::vector<Ckpt> ckpts;                 // sorted by position
    std::vector<std::vector<uint8_t>> kv;    // swapped sequences: KV rows [0, toks.size()) per card
    uint64_t used = 0;
  };

  void worker();
  void run(Job& j);
  int prepare(const std::vector<int>& P);    // cache match; returns the reused prefix length
  void prefill_with_ckpts(Job& j, const GenRequest& R, int from);
  void load_image(const ImageSpan& im);  // encode it (unless it is the one in the decoder) and hand it to the decoder
  void add_ckpt(int pos);
  void swap_out_live();
  std::shared_ptr<HostState> new_state();
  size_t ram_used() const;
  void make_room(size_t bytes);
  void log(const char* fmt, ...);

  const Model& m_;
  Decoder& dec_;
  const Tokenizer& tok_;
  EngineOptions opt_;
  size_t sbytes_ = 0;        // state bytes per card
  size_t kv_tok_bytes_ = 0;  // KV bytes per token per card
  int ncards_ = 1;
  int im_start_ = -1;
  VisionEncoder* venc_ = nullptr;
  cudaStream_t vstream_ = nullptr;
  bool img_loaded_ = false;
  uint64_t img_hash_ = 0;
  Seq live_;
  std::list<Seq> swapped_;
  std::vector<HostState*> free_;  // released host states, reused by new_state
  uint64_t clock_ = 0;

  std::mutex qmu_;
  std::condition_variable qcv_;
  std::deque<std::shared_ptr<Job>> queue_;
  bool stop_ = false;
  bool busy_ = false;
  std::thread th_;
};

}  // namespace q27
