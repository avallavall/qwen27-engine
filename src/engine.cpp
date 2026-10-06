#include "engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <stdexcept>

namespace q27 {

using clk = std::chrono::steady_clock;
static double ms_between(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------- Job
void Job::push(GenEvent e) {
  {
    std::lock_guard<std::mutex> l(mu_);
    q_.push_back(std::move(e));
  }
  cv_.notify_one();
}

bool Job::pop(std::vector<GenEvent>& out, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> l(mu_);
  if (!cv_.wait_for(l, timeout, [&] { return !q_.empty(); })) return false;
  while (!q_.empty()) {
    out.push_back(std::move(q_.front()));
    q_.pop_front();
  }
  return true;
}

// ---------------------------------------------------------------- host states
static size_t g_states = 0;  // HostState objects alive (in use or pooled); the worker thread owns them

Engine::HostState::~HostState() {
  for (auto* p : card)
    if (p) cudaFreeHost(p);
  g_states--;
}

// Host states are pinned buffers (one per card); released ones go to a free list for reuse.
std::shared_ptr<Engine::HostState> Engine::new_state() {
  HostState* h = nullptr;
  if (!free_.empty()) {
    h = free_.back();
    free_.pop_back();
  } else {
    make_room(sbytes_ * ncards_);
    h = new HostState();
    g_states++;
    h->card.assign(ncards_, nullptr);
    for (int c = 0; c < ncards_; c++) {
      if (cudaHostAlloc((void**)&h->card[c], sbytes_, cudaHostAllocPortable) != cudaSuccess) {
        delete h;
        throw std::runtime_error("cudaHostAlloc failed for a checkpoint");
      }
    }
  }
  return std::shared_ptr<HostState>(h, [this](HostState* p) { free_.push_back(p); });
}

size_t Engine::ram_used() const {
  size_t b = g_states * sbytes_ * ncards_;
  for (const Seq& s : swapped_)
    for (const auto& v : s.kv) b += v.size();
  return b;
}

// Free host memory until `bytes` more fit in the budget: the least recently used swapped sequence first, then the
// oldest live checkpoints (the first one, at the end of the system turn, is kept as long as possible).
void Engine::make_room(size_t bytes) {
  const size_t limit = opt_.cache_ram_mb << 20;
  for (int guard = 0; guard < 1000 && ram_used() + bytes > limit; guard++) {
    if (!free_.empty()) {
      delete free_.back();
      free_.pop_back();
      continue;
    }
    if (!swapped_.empty()) {
      auto lru = swapped_.begin();
      for (auto it = swapped_.begin(); it != swapped_.end(); ++it)
        if (it->used < lru->used) lru = it;
      log("cache: drop a swapped sequence of %zu tokens (RAM budget)", lru->toks.size());
      swapped_.erase(lru);
      continue;
    }
    if (live_.ckpts.size() > 2) { live_.ckpts.erase(live_.ckpts.begin() + 1); continue; }
    if (!live_.ckpts.empty()) { live_.ckpts.erase(live_.ckpts.begin()); continue; }
    break;
  }
}

// ---------------------------------------------------------------- Engine
Engine::Engine(const Model& m, Decoder& dec, const Tokenizer& tok, EngineOptions opt, VisionEncoder* venc)
    : m_(m), dec_(dec), tok_(tok), opt_(opt), venc_(venc) {
  sbytes_ = dec_.state_bytes();
  ncards_ = dec_.n_cards();
  kv_tok_bytes_ = Decoder::kv_bytes_per_token(m_, dec_.kv_q8());
  im_start_ = tok_.find("<|im_start|>");
  if (venc_) {
    cudaSetDevice(venc_->device());
    if (cudaStreamCreateWithFlags(&vstream_, cudaStreamNonBlocking) != cudaSuccess)
      throw std::runtime_error("cannot create the vision stream");
  }
  th_ = std::thread([this] { worker(); });
}

void Engine::load_image(const ImageSpan& im) {
  if (img_loaded_ && img_hash_ == im.hash) return;
  if (!venc_) throw std::runtime_error("image input without a vision encoder");
  const auto t0 = clk::now();
  cudaSetDevice(venc_->device());
  const float* emb = venc_->encode(im.rgb->data(), im.plan.src_w, im.plan.src_h, im.plan, vstream_);
  if (cudaStreamSynchronize(vstream_) != cudaSuccess) throw std::runtime_error("vision encoder failed");
  const auto t1 = clk::now();
  dec_.set_image(emb, im.plan.n_tokens, venc_->device());
  img_loaded_ = true;
  img_hash_ = im.hash;
  log("image %dx%d -> %dx%d tokens: encoder %.0f ms, copy %.0f ms", im.plan.src_w, im.plan.src_h, im.plan.nx, im.plan.ny,
      ms_between(t0, t1), ms_between(t1, clk::now()));
}

Engine::~Engine() {
  {
    std::lock_guard<std::mutex> l(qmu_);
    stop_ = true;
  }
  qcv_.notify_all();
  if (th_.joinable()) th_.join();
}

int Engine::capacity() const { return dec_.n_ctx() - 2 * Decoder::kMaxT - 1; }

void Engine::submit(std::shared_ptr<Job> j) {
  {
    std::lock_guard<std::mutex> l(qmu_);
    queue_.push_back(std::move(j));
  }
  qcv_.notify_one();
}

int Engine::queued() {
  std::lock_guard<std::mutex> l(qmu_);
  return (int)queue_.size() + (busy_ ? 1 : 0);
}

void Engine::log(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::time_t t = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
  fprintf(stderr, "%s %s\n", ts, buf);
  fflush(stderr);
}

void Engine::worker() {
  for (;;) {
    std::shared_ptr<Job> j;
    {
      std::unique_lock<std::mutex> l(qmu_);
      qcv_.wait(l, [&] { return stop_ || !queue_.empty(); });
      if (stop_) break;
      j = queue_.front();
      queue_.pop_front();
      busy_ = true;
    }
    run(*j);
    {
      std::lock_guard<std::mutex> l(qmu_);
      busy_ = false;
    }
  }
  live_ = Seq{};
  swapped_.clear();
  for (HostState* h : free_) delete h;
  free_.clear();
}

void Engine::add_ckpt(int pos) {
  if (pos != dec_.position()) return;
  for (const Ckpt& c : live_.ckpts)
    if (c.pos == pos) return;
  if ((int)live_.ckpts.size() >= opt_.max_ckpts && !live_.ckpts.empty())
    live_.ckpts.erase(live_.ckpts.begin() + (live_.ckpts.size() > 1 ? 1 : 0));
  auto st = new_state();
  dec_.state_save(st->card);
  live_.ckpts.push_back({pos, st});
  std::sort(live_.ckpts.begin(), live_.ckpts.end(), [](const Ckpt& a, const Ckpt& b) { return a.pos < b.pos; });
}

// Copy the live sequence to host RAM: KV rows, the state at its end, its checkpoints (shared).
void Engine::swap_out_live() {
  const int L = (int)live_.toks.size();
  const size_t kvb = kv_tok_bytes_ * (size_t)L;
  make_room(kvb * ncards_ + sbytes_ * ncards_);
  if (ram_used() + kvb * ncards_ + sbytes_ * ncards_ > (opt_.cache_ram_mb << 20)) {
    log("cache: sequence of %d tokens does not fit the RAM budget, not saved", L);
    return;
  }
  const auto t0 = clk::now();
  Seq s;
  s.toks = live_.toks;
  s.ckpts = live_.ckpts;
  s.used = ++clock_;
  if (s.ckpts.empty() || s.ckpts.back().pos != L) {
    auto st = new_state();
    dec_.state_save(st->card);
    s.ckpts.push_back({L, st});
  }
  s.kv.resize(ncards_);
  std::vector<uint8_t*> ptr;
  for (auto& v : s.kv) {
    v.resize(kvb);
    ptr.push_back(v.data());
  }
  dec_.kv_to_host(ptr, 0, L);
  dec_.state_wait();
  swapped_.push_back(std::move(s));
  log("cache: saved a sequence of %d tokens to RAM in %.0f ms (%zu MB in use)", L, ms_between(t0, clk::now()),
      ram_used() >> 20);
}

int Engine::prepare(const std::vector<int>& P) {
  const int n = (int)P.size();
  const int limit = n - 1;  // the last prompt token is always read (its logits start the generation)
  auto lcp = [&](const std::vector<int>& a) {
    int k = 0;
    const int m = std::min((int)a.size(), limit);
    while (k < m && a[k] == P[k]) k++;
    return k;
  };
  auto best_ckpt = [](const std::vector<Ckpt>& c, int k) {
    int b = -1;
    for (int i = 0; i < (int)c.size(); i++)
      if (c[i].pos <= k) b = i;
    return b;
  };
  const int L = (int)live_.toks.size();
  const int ml = lcp(live_.toks);
  int reuse_live = 0, ci_live = -1;
  if (ml == L) reuse_live = L;
  else {
    ci_live = best_ckpt(live_.ckpts, ml);
    reuse_live = ci_live >= 0 ? live_.ckpts[ci_live].pos : 0;
  }
  auto best = swapped_.end();
  int reuse_sw = 0, ci_sw = -1;
  for (auto it = swapped_.begin(); it != swapped_.end(); ++it) {
    const int ci = best_ckpt(it->ckpts, lcp(it->toks));
    if (ci >= 0 && it->ckpts[ci].pos > reuse_sw) { reuse_sw = it->ckpts[ci].pos; best = it; ci_sw = ci; }
  }
  if (best != swapped_.end() && reuse_sw > reuse_live) {
    if (L >= opt_.min_swap_tokens) swap_out_live();
    const auto t0 = clk::now();
    Seq s = std::move(*best);
    swapped_.erase(best);
    std::vector<const uint8_t*> ptr;
    for (auto& v : s.kv) ptr.push_back(v.data());
    dec_.kv_from_host(ptr, 0, (int)s.toks.size());  // whole block: the host layout is per head, [toks.size()] rows
    dec_.state_load(s.ckpts[ci_sw].st->card, reuse_sw);
    live_ = Seq{};
    live_.toks.assign(s.toks.begin(), s.toks.begin() + reuse_sw);
    for (const Ckpt& c : s.ckpts)
      if (c.pos <= reuse_sw) live_.ckpts.push_back(c);
    log("cache: loaded a sequence from RAM, reuse %d of %zu tokens, %.0f ms", reuse_sw, s.toks.size(),
        ms_between(t0, clk::now()));
    return reuse_sw;
  }
  if (reuse_live < L) {
    if (L >= opt_.min_swap_tokens && 2 * (size_t)reuse_live < (size_t)L) swap_out_live();
    if (reuse_live > 0) dec_.state_load(live_.ckpts[ci_live].st->card, reuse_live);
    live_.toks.resize(reuse_live);
    while (!live_.ckpts.empty() && live_.ckpts.back().pos > reuse_live) live_.ckpts.pop_back();
    log("cache: prefix %d of %d live tokens matches, restart at %d", ml, L, reuse_live);
  }
  if (reuse_live == 0) {
    dec_.reset();
    live_.toks.clear();
    live_.ckpts.clear();
  }
  return reuse_live;
}

void Engine::prefill_with_ckpts(Job& j, const GenRequest& R, int from) {
  const std::vector<int>& P = R.prompt;
  const int n = (int)P.size();
  std::vector<int> bnd;  // positions of <|im_start|> (message starts)
  for (int i = 0; i < n; i++)
    if (P[i] == im_start_) bnd.push_back(i);
  std::vector<int> cps;
  if (bnd.size() >= 2) {
    cps.push_back(bnd[1]);               // end of the first turn (the system turn when there is one)
    cps.push_back(bnd[bnd.size() - 2]);  // start of the last message
  }
  if (!bnd.empty()) cps.push_back(bnd.back());  // start of the generation prompt
  for (const ImageSpan& s : R.images) cps.push_back(s.start + s.plan.n_tokens + 1);  // after <|vision_end|>: a new
                                                                                     // question on the same image
  for (int p = opt_.ckpt_every; p < n; p += opt_.ckpt_every) cps.push_back(p);
  std::sort(cps.begin(), cps.end());
  cps.erase(std::unique(cps.begin(), cps.end()), cps.end());
  constexpr int kChunk = 2048;  // cancel checks between chunks (about 3 s)
  // Text rows use position = cache row - delta; with images every row gets explicit positions (R.rope).
  if (R.images.empty() && dec_.rope_delta() != 0) dec_.set_rope_delta(0);
  std::vector<int> ids;
  int pos = from;
  size_t ci = 0;
  while (pos < n) {
    while (ci < cps.size() && cps[ci] <= pos) ci++;
    int next = std::min(n, pos + kChunk);
    if (ci < cps.size() && cps[ci] < next) next = cps[ci];
    // a segment holds the rows of at most one image (the decoder has one image buffer), with any text around it
    const ImageSpan* im = nullptr;
    for (const ImageSpan& s : R.images) {
      const int e = s.start + s.plan.n_tokens;
      if (e <= pos) continue;
      if (s.start >= next) break;
      if (!im) { im = &s; continue; }
      next = s.start;  // a second image: it starts the next segment
      break;
    }
    ids.assign(P.begin() + pos, P.begin() + next);
    if (im) {
      load_image(*im);
      const int a = std::max(pos, im->start), b = std::min(next, im->start + im->plan.n_tokens);
      for (int i = a; i < b; i++) ids[i - pos] = -1 - (i - im->start);  // decoder input: row of the image buffer
    }
    dec_.prefill(ids.data(), next - pos, R.rope.empty() ? nullptr : R.rope.data() + 3 * (size_t)pos);
    live_.toks.insert(live_.toks.end(), P.begin() + pos, P.begin() + next);
    pos = next;
    if (ci < cps.size() && cps[ci] == pos) add_ckpt(pos);
    if (j.cancel) return;
  }
  // decode positions after the prompt: cache row - delta, the last prompt row being a text row
  if (!R.rope.empty()) dec_.set_rope_delta(n - (R.rope[3 * (size_t)(n - 1)] + 1));
}

void Engine::run(Job& j) {
  GenRequest& R = j.req;
  if (j.cancel) return;
  const auto t0 = clk::now();
  const int n = (int)R.prompt.size();
  int reuse = 0;
  try {
    if (n < 1) throw std::runtime_error("empty prompt");
    if (n > capacity()) throw std::runtime_error("prompt longer than the context");
    reuse = prepare(R.prompt);
    prefill_with_ckpts(j, R, reuse);
    if (j.cancel) {
      log("%s cancelled while reading the prompt (%d of %d tokens read)", R.tag.c_str(), (int)live_.toks.size(), n);
      return;
    }
    const auto t1 = clk::now();

    StreamParser parser(tok_, R.thinking, R.tools);
    int n_gen = 0, draft_n = 0, draft_acc = 0;
    std::string finish = "length", raw;
    std::vector<int> gen_ids;
    const int max_gen = R.max_tokens < 0 ? capacity() : R.max_tokens;
    // Consume one emitted token. Returns true when generation stops at it.
    auto consume = [&](int t) -> bool {
      n_gen++;
      if (tok_.is_eog(t)) { finish = "stop"; return true; }
      gen_ids.push_back(t);
      ParseDelta d = parser.push(t);
      if (!d.empty()) j.push(GenEvent{GenEvent::kDelta, std::move(d)});
      if (!R.stop.empty()) {
        const size_t old = raw.size();
        raw += tok_.piece(t);
        for (const std::string& s : R.stop) {
          if (s.empty()) continue;
          const size_t from = old >= s.size() ? old - s.size() + 1 : 0;
          if (raw.find(s, from) != std::string::npos) { finish = "stop"; return true; }
        }
      }
      if (n_gen >= max_gen) { finish = "length"; return true; }
      return false;
    };
    bool stop = false;
    int pending = 0;
    if (max_gen > 0) {
      pending = dec_.start(R.sp, R.rng_start);
      stop = consume(pending);
    } else {
      stop = true;
    }
    while (!stop) {
      if (j.cancel) { finish = "cancel"; break; }
      if (dec_.position() + 2 * Decoder::kMaxT + 1 > dec_.n_ctx()) { finish = "length"; break; }
      int out[Decoder::kMaxT];
      const int k = dec_.spec_step(out);
      draft_n += Decoder::kDrafts;
      draft_acc += k - 1;
      int used = k;
      for (int i = 0; i < k; i++)
        if (consume(out[i])) { used = i + 1; stop = true; break; }
      if (used < k) dec_.keep_from_last_step(used);
      live_.toks.push_back(pending);
      for (int i = 0; i + 1 < used; i++) live_.toks.push_back(out[i]);
      pending = out[used - 1];
    }
    if ((int)live_.toks.size() != dec_.position())
      throw std::runtime_error("internal: token list and decoder position differ");
    ParseDelta last = parser.finish();
    if (!last.empty()) j.push(GenEvent{GenEvent::kDelta, std::move(last)});
    const auto t2 = clk::now();
    live_.used = ++clock_;

    GenEvent e;
    e.kind = GenEvent::kDone;
    e.finish_reason = finish == "stop" && !parser.tool_calls().empty() ? "tool_calls" : finish;
    if (finish == "cancel") e.finish_reason = "stop";
    e.reasoning = parser.reasoning();
    e.content = parser.content();
    e.raw = tok_.decode(gen_ids);
    e.tool_calls = parser.tool_calls();
    const double pms = ms_between(t0, t1), gms = ms_between(t1, t2);
    const int pn = n - reuse;
    e.timings = ojson{
        {"cache_n", reuse},
        {"prompt_n", pn},
        {"prompt_ms", pms},
        {"prompt_per_token_ms", pn > 0 ? pms / pn : 0.0},
        {"prompt_per_second", pms > 0 ? 1e3 * pn / pms : 0.0},
        {"predicted_n", n_gen},
        {"predicted_ms", gms},
        {"predicted_per_token_ms", n_gen > 0 ? gms / n_gen : 0.0},
        {"predicted_per_second", gms > 0 ? 1e3 * n_gen / gms : 0.0},
    };
    if (draft_n > 0) {
      e.timings["draft_n"] = draft_n;
      e.timings["draft_n_accepted"] = draft_acc;
    }
    e.usage = ojson{
        {"completion_tokens", n_gen},
        {"prompt_tokens", n},
        {"total_tokens", n + n_gen},
        {"prompt_tokens_details", ojson{{"cached_tokens", reuse}}},
    };
    log("%s prompt %d (cached %d) %.0f ms %.0f t/s | gen %d %.0f ms %.1f t/s, drafts %d/%d | %s | ctx %d", R.tag.c_str(),
        n, reuse, pms, pms > 0 ? 1e3 * pn / pms : 0.0, n_gen, gms, gms > 0 ? 1e3 * n_gen / gms : 0.0, draft_acc,
        draft_n, finish.c_str(), dec_.position());
    j.push(std::move(e));
  } catch (const std::exception& ex) {
    log("%s error: %s", R.tag.c_str(), ex.what());
    // the decoder state is unknown: drop the live sequence
    live_ = Seq{};
    try {
      dec_.reset();
    } catch (const std::exception& ex2) {
      // a sticky CUDA error (e.g. an illegal address) makes every later call fail: stop the process
      log("fatal: the GPU state cannot be recovered (%s); exiting", ex2.what());
      std::_Exit(3);
    }
    GenEvent e;
    e.kind = GenEvent::kError;
    e.code = 500;
    e.err_type = "server_error";
    e.message = ex.what();
    j.push(std::move(e));
  }
}

}  // namespace q27
