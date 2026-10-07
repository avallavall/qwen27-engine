#include "model.h"

#include <thread>
#include <cuda.h>

#include <array>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

#include "accept.cuh"
#include "common.cuh"
#include "ops.h"
#include "prefill.h"
#include "prof.h"
#include "requant.h"
#include "sampling.h"

namespace q27 {

namespace {

std::vector<int> range(int a, int n) { std::vector<int> v(n); std::iota(v.begin(), v.end(), a); return v; }
void append(std::vector<int>& v, const std::vector<int>& w) { v.insert(v.end(), w.begin(), w.end()); }

struct Loader {
  const GGUF& g;
  Shard& sh;
  const std::map<std::string, GTensor>* over = nullptr;  // replacement tensors by name
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  cudaStream_t s = nullptr;

  void* dev_copy(const void* src, size_t bytes) {
    void* d; CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice));
    sh.allocs.push_back(d);
    sh.vram_bytes += bytes;
    return d;
  }
  // F32 vector or matrix rows: rows = list of row indices (each row of `cols` floats); empty = all.
  float* f32(const std::string& name, size_t cols, const std::vector<int>& rows = {}) {
    const GTensor& t = g.tensor(name);
    if (t.type != GType::F32) throw std::runtime_error(name + ": expected F32");
    const float* src = (const float*)t.data;
    const size_t total = (size_t)(t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
    if (rows.empty()) return (float*)dev_copy(src, total * 4);
    std::vector<float> h(rows.size() * cols);
    for (size_t r = 0; r < rows.size(); r++) memcpy(&h[r * cols], src + (size_t)rows[r] * cols, cols * 4);
    return (float*)dev_copy(h.data(), h.size() * 4);
  }
  __nv_bfloat16* bf16_rows(const std::string& name, const std::vector<int>& rows) {
    const GTensor& t = g.tensor(name);
    if (t.type != GType::BF16) throw std::runtime_error(name + ": expected BF16");
    const size_t cols = (size_t)t.ne[0];
    std::vector<uint16_t> h(rows.size() * cols);
    for (size_t r = 0; r < rows.size(); r++) memcpy(&h[r * cols], (const uint16_t*)t.data + (size_t)rows[r] * cols, cols * 2);
    return (__nv_bfloat16*)dev_copy(h.data(), h.size() * 2);
  }
  QMat q(const std::string& name, const std::vector<int>& rows = {}, const std::vector<int>& blocks = {}) {
    const auto it = over ? over->find(name) : std::map<std::string, GTensor>::const_iterator{};
    const GTensor& t = over && it != over->end() ? it->second : g.tensor(name);
    QMat m = qmat_upload_shard(t, rows.empty() ? range(0, (int)t.ne[1]) : rows, blocks, scratch, scratch_bytes, s);
    sh.vram_bytes += m.bytes;
    return m;
  }
};

}  // namespace

Model::Model(const std::string& path, const std::vector<int>& devices) {
  if (devices.empty() || devices.size() > 2) throw std::runtime_error("1 or 2 devices");
  g_ = std::make_unique<GGUF>(path);
  const GGUF& g = *g_;
  if (g.get_str("general.architecture") != "qwen35") throw std::runtime_error("not a qwen35 model");
  auto expect = [&](const char* key, int64_t want) {
    const int64_t v = g.get_int(key);
    if (v != want) throw std::runtime_error(std::string(key) + " = " + std::to_string(v) + ", expected " + std::to_string(want));
  };
  expect("qwen35.block_count", 65);
  expect("qwen35.embedding_length", hp_.n_embd);
  expect("qwen35.feed_forward_length", hp_.n_ff);
  expect("qwen35.attention.head_count", hp_.n_head);
  expect("qwen35.attention.head_count_kv", hp_.n_head_kv);
  expect("qwen35.attention.key_length", hp_.head_dim);
  expect("qwen35.attention.value_length", hp_.head_dim);
  expect("qwen35.ssm.group_count", hp_.ssm_k_heads);
  expect("qwen35.ssm.time_step_rank", hp_.ssm_v_heads);
  expect("qwen35.ssm.state_size", hp_.ssm_dim);
  expect("qwen35.ssm.conv_kernel", hp_.conv_k);
  expect("qwen35.ssm.inner_size", hp_.ssm_v_heads * hp_.ssm_dim);
  expect("qwen35.full_attention_interval", hp_.full_attn_interval);
  expect("qwen35.rope.dimension_count", hp_.rope_dims);
  hp_.rope_base = (float)g.get_float("qwen35.rope.freq_base");
  hp_.eps = (float)g.get_float("qwen35.attention.layer_norm_rms_epsilon");
  hp_.ctx_train = (int)g.get_int("qwen35.context_length");

  // Embedding table (IQ2_S, 407 MB): a copy in each card's VRAM (load_shard). Q27_EMBD_HOST=1: one copy in pinned
  // host memory, mapped into both cards (the old path; each row read crosses PCIe).
  const GTensor& te = g.tensor("token_embd.weight");
  if (te.type != GType::IQ2_S) throw std::runtime_error("token_embd: expected IQ2_S");
  tok_embd_row_bytes = te.row_bytes();
  if (const char* e = getenv("Q27_EMBD_HOST"); e && e[0] == '1') {
    CK(cudaHostAlloc(&tok_embd, te.nbytes, cudaHostAllocMapped | cudaHostAllocPortable));
    memcpy(tok_embd, te.data, te.nbytes);
  }

  // MTP layer (blk.64, Q6_K in the file) in a smaller type, Q4_K by default (Q27_MTP_TYPE=q4_k|iq4_xs|q6_k; q6_k = as
  // in the file). Only the drafts read it, so this changes which tokens get proposed, never the output distribution.
  // It runs on a CPU thread while the cards load the main layers.
  {
    const char* e = getenv("Q27_MTP_TYPE");
    const GType to = parse_gtype(e ? e : "q4_k");
    if (to != GType::Q6_K) mtp_job_ = std::async(std::launch::async, [this, to] {
      const GGUF& g = *g_;
      const auto t0 = std::chrono::steady_clock::now();
      for (const char* n : {"ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output", "nextn.eh_proj"}) {
        const GTensor& t = g.tensor(std::string("blk.64.") + n + ".weight");
        over_bufs_.push_back(requant(t, to, std::max(1, (int)std::thread::hardware_concurrency() / 2)));  // half: leaves CPU to the loader
        GTensor r = t;
        r.type = to;
        r.data = over_bufs_.back().data();
        r.nbytes = over_bufs_.back().size();
        over_[t.name] = r;
      }
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      fprintf(stderr, "MTP layer re-quantized to %s in %.1f s (on a CPU thread)\n", gtype_name(to), s);
    });
  }

  const int tp = (int)devices.size();
  shards.resize(tp);
  for (int r = 0; r < tp; r++) {
    shards[r].dev = devices[r];
    shards[r].rank = r;
    load_shard(shards[r], tp);
  }
  if (mtp_job_.valid()) mtp_job_.get();  // the re-quantized MTP layer (rethrows its errors)
  for (int r = 0; r < tp; r++) load_mtp(shards[r], tp);
  over_.clear();
  over_bufs_.clear();
  over_bufs_.shrink_to_fit();
  // Q27_DRAFT_VOCAB=<file>[:n] (tools): draft vocabulary from a ranked id file (see set_draft_vocab_file)
  if (const char* e = getenv("Q27_DRAFT_VOCAB")) {
    std::string v = e;
    int n = 0;
    const size_t c = v.rfind(':');
    if (c != std::string::npos && c > 1) { n = atoi(v.c_str() + c + 1); v = v.substr(0, c); }
    if (v != "0" && !v.empty()) set_draft_vocab_file(v, n);
  }
}

void Model::load_shard(Shard& sh, int tp) {
  CK(cudaSetDevice(sh.dev));
  const int r = sh.rank;
  const Hparams& hp = hp_;
  sh.kh = hp.ssm_k_heads / tp;
  sh.vh = hp.ssm_v_heads / tp;
  sh.qh = hp.n_head / tp;
  sh.kvh = hp.n_head_kv / tp;
  sh.ff = hp.n_ff / tp;
  sh.vocab_n = hp.vocab / tp;
  sh.vocab0 = r * sh.vocab_n;

  // GDN heads of this card. V head g reads K head g % 16, so with K heads [kh0, kh0+kh) the card owns
  // V heads {g : g % 16 in [kh0, kh0+kh)}, in ascending order; local V head j then reads local K head j % kh.
  const int kh0 = r * sh.kh;
  std::vector<int> vheads;
  for (int g = 0; g < hp.ssm_v_heads; g++)
    if (g % hp.ssm_k_heads >= kh0 && g % hp.ssm_k_heads < kh0 + sh.kh) vheads.push_back(g);
  const int D = hp.ssm_dim;
  std::vector<int> qkv_rows = range(kh0 * D, sh.kh * D);                                  // q
  append(qkv_rows, range(hp.ssm_k_heads * D + kh0 * D, sh.kh * D));                       // k
  std::vector<int> z_rows, ssm_out_blocks;
  for (int g : vheads) {
    append(qkv_rows, range(2 * hp.ssm_k_heads * D + g * D, D));                           // v
    append(z_rows, range(g * D, D));
  }
  for (size_t i = 0; i < vheads.size(); i += 2) {
    if (vheads[i] % 2 || i + 1 >= vheads.size() || vheads[i + 1] != vheads[i] + 1)
      throw std::runtime_error("GDN head split does not align with 256-weight blocks");
    ssm_out_blocks.push_back(vheads[i] / 2);
  }
  // attention
  const std::vector<int> q_rows = range(r * sh.qh * 2 * hp.head_dim, sh.qh * 2 * hp.head_dim);
  const std::vector<int> kv_rows = range(r * sh.kvh * hp.head_dim, sh.kvh * hp.head_dim);
  const std::vector<int> o_blocks = range(r * sh.qh * hp.head_dim / 256, sh.qh * hp.head_dim / 256);
  // FFN
  const std::vector<int> ff_rows = range(r * sh.ff, sh.ff);
  const std::vector<int> down_blocks = range(r * sh.ff / 256, sh.ff / 256);

  Loader L{*g_, sh};
  CK(cudaStreamCreate(&L.s));
  for (const auto& t : g_->tensors())
    if (qmat_supported(t.type)) L.scratch_bytes = std::max(L.scratch_bytes, (size_t)t.nbytes);
  CK(cudaMalloc(&L.scratch, L.scratch_bytes));

  if (tok_embd) sh.embd = tok_embd;
  else { const GTensor& te = g_->tensor("token_embd.weight"); sh.embd = (const uint8_t*)L.dev_copy(te.data, te.nbytes); }

  const int E = hp.n_embd;
  sh.layers.resize(hp.n_layer);
  for (int il = 0; il < hp.n_layer; il++) {
    Layer& Ly = sh.layers[il];
    const std::string p = "blk." + std::to_string(il) + ".";
    Ly.attn = hp.is_attn(il);
    Ly.attn_norm = L.f32(p + "attn_norm.weight", E);
    Ly.post_norm = L.f32(p + "post_attention_norm.weight", E);
    Ly.ffn_gate = L.q(p + "ffn_gate.weight", ff_rows);
    Ly.ffn_up = L.q(p + "ffn_up.weight", ff_rows);
    Ly.ffn_down = L.q(p + "ffn_down.weight", {}, down_blocks);
    if (Ly.attn) {
      Ly.wq = L.q(p + "attn_q.weight", q_rows);
      Ly.wk = L.q(p + "attn_k.weight", kv_rows);
      Ly.wv = L.q(p + "attn_v.weight", kv_rows);
      Ly.wo = L.q(p + "attn_output.weight", {}, o_blocks);
      Ly.q_norm = L.f32(p + "attn_q_norm.weight", hp.head_dim);
      Ly.k_norm = L.f32(p + "attn_k_norm.weight", hp.head_dim);
    } else {
      Ly.qkv = L.q(p + "attn_qkv.weight", qkv_rows);
      Ly.gate = L.q(p + "attn_gate.weight", z_rows);
      Ly.ssm_out = L.q(p + "ssm_out.weight", {}, ssm_out_blocks);
      Ly.alpha = L.bf16_rows(p + "ssm_alpha.weight", vheads);
      Ly.beta = L.bf16_rows(p + "ssm_beta.weight", vheads);
      Ly.ssm_a = L.f32(p + "ssm_a", 1, vheads);
      Ly.dt_bias = L.f32(p + "ssm_dt.bias", 1, vheads);
      Ly.conv_w = L.f32(p + "ssm_conv1d.weight", hp.conv_k, qkv_rows);
      Ly.ssm_norm = L.f32(p + "ssm_norm.weight", D);
    }
  }
  sh.output_norm = L.f32("output_norm.weight", E);
  sh.output = L.q("output.weight", range(sh.vocab0, sh.vocab_n));
  CK(cudaFree(L.scratch));
  CK(cudaStreamDestroy(L.s));
}

// The MTP block (blk.64), split like a target attention layer; eh_proj whole on every card. Loaded after the main
// layers of both cards, so the re-quantization on the CPU thread has the whole main load to finish.
void Model::load_mtp(Shard& sh, int tp) {
  CK(cudaSetDevice(sh.dev));
  const int r = sh.rank;
  const Hparams& hp = hp_;
  const int E = hp.n_embd;
  const std::vector<int> q_rows = range(r * sh.qh * 2 * hp.head_dim, sh.qh * 2 * hp.head_dim);
  const std::vector<int> kv_rows = range(r * sh.kvh * hp.head_dim, sh.kvh * hp.head_dim);
  const std::vector<int> o_blocks = range(r * sh.qh * hp.head_dim / 256, sh.qh * hp.head_dim / 256);
  const std::vector<int> ff_rows = range(r * sh.ff, sh.ff);
  const std::vector<int> down_blocks = range(r * sh.ff / 256, sh.ff / 256);
  (void)tp;
  Loader L{*g_, sh, &over_};
  CK(cudaStreamCreate(&L.s));
  const std::string p = "blk.64.";
  for (const char* n : {"ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output", "nextn.eh_proj"}) {
    const std::string name = p + n + ".weight";
    const auto it = over_.find(name);
    L.scratch_bytes = std::max(L.scratch_bytes, (size_t)(it != over_.end() ? it->second.nbytes : g_->tensor(name).nbytes));
  }
  CK(cudaMalloc(&L.scratch, L.scratch_bytes));
  Layer& M = sh.mtp;
  M.attn = true;
  M.attn_norm = L.f32(p + "attn_norm.weight", E);
  M.post_norm = L.f32(p + "post_attention_norm.weight", E);
  M.ffn_gate = L.q(p + "ffn_gate.weight", ff_rows);
  M.ffn_up = L.q(p + "ffn_up.weight", ff_rows);
  M.ffn_down = L.q(p + "ffn_down.weight", {}, down_blocks);
  M.wq = L.q(p + "attn_q.weight", q_rows);
  M.wk = L.q(p + "attn_k.weight", kv_rows);
  M.wv = L.q(p + "attn_v.weight", kv_rows);
  M.wo = L.q(p + "attn_output.weight", {}, o_blocks);
  M.q_norm = L.f32(p + "attn_q_norm.weight", hp.head_dim);
  M.k_norm = L.f32(p + "attn_k_norm.weight", hp.head_dim);
  sh.eh_proj = L.q(p + "nextn.eh_proj.weight");
  sh.enorm = L.f32(p + "nextn.enorm.weight", E);
  sh.hnorm = L.f32(p + "nextn.hnorm.weight", E);
  sh.shared_head_norm = L.f32(p + "nextn.shared_head_norm.weight", E);
  CK(cudaFree(L.scratch));
  CK(cudaStreamDestroy(L.s));
}

void Model::set_draft_vocab(std::vector<int> ids) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  const GTensor& t = g_->tensor("output.weight");
  // The rows come from the GGUF (host memory), so any card can score any token: split the subset in equal parts
  // (the frequent ids are mostly low, which put almost all rows on card 0 when split by vocab half).
  std::vector<char> used(hp_.vocab, 0);
  for (int id : ids) used[id] = 1;
  const int nsh = (int)shards.size();
  int next_pad = 0;
  for (int r = 0; r < nsh; r++) {
    Shard& sh = shards[r];
    CK(cudaSetDevice(sh.dev));
    if (sh.draft_n) {
      qmat_free(sh.draft_out);
      CK(cudaFree(sh.draft_ids));
      sh.draft_ids = nullptr;
      sh.draft_n = 0;
    }
    if (ids.empty()) continue;
    const size_t a = ids.size() * r / nsh, b = ids.size() * (r + 1) / nsh;
    std::vector<int> rows(ids.begin() + a, ids.begin() + b);
    while (rows.size() % 128) {  // pad with the lowest ids in no card's part
      while (used[next_pad]) next_pad++;
      rows.push_back(next_pad);
      used[next_pad] = 1;
    }
    std::sort(rows.begin(), rows.end());
    const size_t scratch_bytes = (size_t)t.row_bytes() * rows.size();
    void* scratch;
    CK(cudaMalloc(&scratch, scratch_bytes));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    sh.draft_out = qmat_upload_shard(t, rows, {}, scratch, scratch_bytes, s);
    CK(cudaStreamSynchronize(s));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(scratch));
    CK(cudaMalloc(&sh.draft_ids, sizeof(int) * rows.size()));
    CK(cudaMemcpy(sh.draft_ids, rows.data(), sizeof(int) * rows.size(), cudaMemcpyHostToDevice));
    sh.draft_n = (int)rows.size();
  }
}

void Model::set_draft_vocab_file(const std::string& path, int n) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open draft vocabulary " + path);
  std::vector<int> ids;
  int v;
  while (fread(&v, sizeof(int), 1, f) == 1 && (n <= 0 || (int)ids.size() < n))
    if (v >= 0 && v < hp_.vocab) ids.push_back(v);
  fclose(f);
  set_draft_vocab(ids);
}

Model::~Model() {
  for (auto& sh : shards) {
    cudaSetDevice(sh.dev);
    if (sh.draft_n) { qmat_free(sh.draft_out); cudaFree(sh.draft_ids); }
    for (void* p : sh.allocs) cudaFree(p);
    for (auto& L : sh.layers)
      for (QMat* q : {&L.ffn_gate, &L.ffn_up, &L.ffn_down, &L.qkv, &L.gate, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo}) qmat_free(*q);
    for (QMat* q : {&sh.mtp.ffn_gate, &sh.mtp.ffn_up, &sh.mtp.ffn_down, &sh.mtp.wq, &sh.mtp.wk, &sh.mtp.wv, &sh.mtp.wo}) qmat_free(*q);
    qmat_free(sh.eh_proj);
    qmat_free(sh.output);
  }
  if (tok_embd) cudaFreeHost(tok_embd);
}

// ---------------------------------------------------------------- decode
namespace {

// Mirror of Decoder::DState: P, s, d[3], n_emit, emit[4], counter.
struct St { int P, s, d[3], n_emit, emit[4], counter; };

// MTP input rows: hrows[0] = pend_h (hidden of the token before the pass), hrows[t] = hfin[t-1];
// then pend_h = hfin[last], where last = n_emit-1 after a verify (st != null) or T-1 for a prompt pass.
__global__ void k_rows(float* hrows, float* pend_h, const float* hfin, int T, int E, const St* st) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;
  const int last = st ? st->n_emit - 1 : T - 1;
  hrows[i] = pend_h[i];
  for (int t = 1; t < T; t++) hrows[(size_t)t * E + i] = hfin[(size_t)(t - 1) * E + i];
  pend_h[i] = hfin[(size_t)last * E + i];
}
__global__ void k_copy(float* dst, const float* src, int n) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

}  // namespace

// Kernels on the speculative state (one thread).
namespace {

__global__ void k_prep_verify(St* st, int* dtok, int* dpos) {
  pdl_wait();
  pdl_trigger();
  dtok[0] = st->s; dtok[1] = st->d[0]; dtok[2] = st->d[1]; dtok[3] = st->d[2];
  *dpos = st->P;
}
// Q27_ACCEPT=token: llama.cpp's token-by-token acceptance rule; default: block verification.
bool accept_block_on() {
  static const bool v = [] { const char* e = getenv("Q27_ACCEPT"); return !(e && strcmp(e, "token") == 0); }();
  return v;
}
// Acceptance of the 3 drafts (src/accept.cuh): block verification (Sun et al., ICLR 2025) or, with
// Q27_ACCEPT=token, llama.cpp's token-by-token rule. Both keep the target distribution exactly.
__global__ void k_accept(St* st, const CandRow* pc, const CandRow* qc, uint64_t seed, int* dplane, int* host_emit,
                         int block) {
  pdl_wait();
  pdl_trigger();
  const int cnt = block ? accept_block(st->d, 3, pc, qc, seed, st->counter, st->emit)
                        : accept_token(st->d, 3, pc, qc, seed, st->counter, st->emit);
  st->n_emit = cnt;
  *dplane = cnt - 1;
  if (host_emit) {
    host_emit[0] = cnt;
    for (int i = 0; i < cnt; i++) host_emit[1 + i] = st->emit[i];
    __threadfence_system();
  }
}
// After the catch-up: move to the next verify position and prepare draft 0 (token s at P).
__global__ void k_advance(St* st, int* dtok, int* dpos) {
  pdl_wait();
  pdl_trigger();
  const int n = st->n_emit;
  st->P += n;
  st->s = st->emit[n - 1];
  dtok[0] = st->s;
  *dpos = st->P;
}
// After the prompt: the first token was drawn into st->s; emit it and prepare draft 0.
__global__ void k_first(St* st, int* dtok, int* dpos, int* host_emit) {
  pdl_wait();
  pdl_trigger();
  st->n_emit = 1;
  st->emit[0] = st->s;
  dtok[0] = st->s;
  *dpos = st->P;
  if (host_emit) { host_emit[0] = 1; host_emit[1] = st->s; __threadfence_system(); }
}
__global__ void k_next_draft(St* st, int j, int* dtok, int* dpos) {
  pdl_wait();
  pdl_trigger();
  dtok[0] = st->d[j];
  *dpos = st->P + j + 1;
}
__global__ void k_finish(St* st) {
  pdl_wait();
  pdl_trigger(); st->counter += 1; }

}  // namespace

template <class T> T* Decoder::alloc(Rank& r, size_t n) {
  T* p; CK(cudaMalloc(&p, n * sizeof(T)));
  CK(cudaMemset(p, 0, n * sizeof(T)));
  r.allocs.push_back(p);
  return p;
}

Decoder::Decoder(const Model& m, int n_ctx, bool kv_q8) : m_(m), n_ctx_(n_ctx), kv_q8_(kv_q8) {
  static_assert(sizeof(St) == sizeof(DState), "St must mirror DState");
  const Hparams& hp = m.hp();
  const int T = kMaxT;
  r_.resize(m.tp());
  for (int ri = 0; ri < m.tp(); ri++) {
    Rank& R = r_[ri];
    R.sh = &m.shards[ri];
    const Shard& sh = *R.sh;
    CK(cudaSetDevice(sh.dev));
    CK(cudaStreamCreateWithFlags(&R.s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&R.s2, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&R.ev_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&R.ev_join, cudaEventDisableTiming));
    for (int il = 0; il < hp.n_layer; il++) {
      if (hp.is_attn(il)) {
        R.kc.push_back(alloc<uint8_t>(R, kv_cache_bytes(sh.kvh, n_ctx, kv_q8)));
        R.vc.push_back(alloc<uint8_t>(R, kv_cache_bytes(sh.kvh, n_ctx, kv_q8)));
      } else {
        R.conv_st.push_back(alloc<float>(R, (size_t)4 * sh.conv_channels() * (hp.conv_k - 1)));
        R.ssm_st.push_back(alloc<float>(R, (size_t)4 * sh.vh * hp.ssm_dim * hp.ssm_dim));
      }
    }
    R.mkc = alloc<uint8_t>(R, kv_cache_bytes(sh.kvh, n_ctx, kv_q8));
    R.mvc = alloc<uint8_t>(R, kv_cache_bytes(sh.kvh, n_ctx, kv_q8));
    const int E = hp.n_embd;
    R.x = alloc<float>(R, T * E); R.h = alloc<float>(R, T * E); R.m_out = alloc<float>(R, T * E);
    R.hfin = alloc<float>(R, T * E); R.hrows = alloc<float>(R, T * E); R.mtp_h = alloc<float>(R, T * E);
    R.pend_h = alloc<float>(R, E); R.cat = alloc<float>(R, T * 2 * E);
    R.ef = alloc<float>(R, T * E);
    if (const char* e = getenv("Q27_SUMPROF"); e && e[0] == '1') R.tprof = alloc<unsigned long long>(R, (size_t)kNex * 8);
    R.qkv = alloc<float>(R, T * sh.conv_channels()); R.z = alloc<float>(R, T * sh.vh * hp.ssm_dim);
    R.ab = alloc<float>(R, 2 * T * sh.vh); R.g = alloc<float>(R, T * sh.vh); R.beta = alloc<float>(R, T * sh.vh);
    R.conv = alloc<float>(R, T * sh.conv_channels());
    R.o = alloc<float>(R, T * std::max(sh.vh * hp.ssm_dim, sh.qh * hp.head_dim));
    R.y = alloc<float>(R, T * sh.vh * hp.ssm_dim);
    R.qg = alloc<float>(R, T * 2 * sh.qh * hp.head_dim); R.k = alloc<float>(R, T * sh.kvh * hp.head_dim);
    R.v = alloc<float>(R, T * sh.kvh * hp.head_dim); R.qn = alloc<float>(R, T * sh.qh * hp.head_dim);
    R.fg = alloc<float>(R, T * sh.ff); R.fu = alloc<float>(R, T * sh.ff); R.act = alloc<float>(R, T * sh.ff);
    R.logits = alloc<float>(R, (size_t)T * sh.vocab_n);
    const int kmax = std::max({2 * E, sh.ff, sh.vh * hp.ssm_dim, sh.qh * hp.head_dim});
    R.xq = alloc<int8_t>(R, T * kmax); R.xd = alloc<float>(R, T * kmax / 32);
    R.dtok = alloc<int>(R, kMaxT);
    R.dpos = alloc<int>(R, 1);
    R.dplane = alloc<int>(R, 1);
    R.ddelta = alloc<int>(R, 1);
    R.dstep = alloc<int>(R, 1);
    R.st = alloc<DState>(R, 1);
    R.pc = alloc<CandRow>(R, 4);
    R.qc = alloc<CandRow>(R, kDrafts);
    CK(cudaHostAlloc(&R.hin, (kMaxT + 1) * sizeof(int), cudaHostAllocDefault));
  }
  CK(cudaHostAlloc(&emit_host_, 8 * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
  memset(emit_host_, 0, 8 * sizeof(int));
  if (m.tp() == 2) {
    CK(cudaHostAlloc(&ar_data_, sizeof(__nv_bfloat16) * 2 * 2 * kArMax, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&ar_flags_, sizeof(int) * 2 * 2 * 4 * 32, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&ar_err_, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
    memset(ar_flags_, 0, sizeof(int) * 2 * 2 * 4 * 32);
    *ar_err_ = 0;
    CK(cudaHostAlloc(&mb_.vals, sizeof(float) * 2 * 2 * 4 * kCand, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&mb_.ids, sizeof(int) * 2 * 2 * 4 * kCand, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&mb_.flags, sizeof(int) * 2 * 2 * 32, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(mb_.flags, 0, sizeof(int) * 2 * 2 * 32);
    mb_.err = ar_err_;
  }
  if (const char* e = getenv("Q27_PREFILL_BATCH")) pmb_ = std::max(16, atoi(e) / 16 * 16);
  prefill_alloc();
  pf2_alloc();
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    stage_.push_back((float*)alloc<uint8_t>(R, state_bytes()));
    cudaStream_t ss;
    CK(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking));
    ss_.push_back(ss);
    cudaEvent_t e1, e2;
    CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    ev_staged_.push_back(e1);
    ev_saved_.push_back(e2);
    CK(cudaEventRecord(e2, ss));
  }
}

// Q27_GAPPROF=1: host gap of the speculative step: wall time per run(kStep), GPU time of each card's graph, and the
// host time between two steps (outside run). Printed when the decoder is destroyed.
namespace {
struct GapProf {
  bool on = [] { const char* e = getenv("Q27_GAPPROF"); return e && e[0] == '1'; }();
  long n = 0;
  double wall = 0, gpu[2] = {0, 0}, between = 0, launch = 0;
  std::chrono::steady_clock::time_point last_end;
  bool have_last = false;
  cudaEvent_t e[2][2] = {};
};
GapProf& gapprof() { static GapProf g; return g; }
}  // namespace

Decoder::~Decoder() {
  if (GapProf& gp = gapprof(); gp.on && gp.n) {
    fprintf(stderr, "host gap (Q27_GAPPROF): %ld steps | wall per run %.1f us (launch calls %.1f) | GPU card 0 %.1f us, card 1 "
            "%.1f us | host time between steps %.1f us\n", gp.n, gp.wall / gp.n, gp.launch / gp.n, gp.gpu[0] / gp.n,
            gp.gpu[1] / gp.n, gp.between / (gp.n > 1 ? gp.n - 1 : 1));
    gp.n = 0; gp.wall = gp.gpu[0] = gp.gpu[1] = gp.between = gp.launch = 0; gp.have_last = false;
  }
  // Q27_SUMPROF=1: average phase times of the cross-card sums (verify = exchange indices 0..127 of the step graph).
  for (size_t ri = 0; ri < r_.size(); ri++) {
    if (!r_[ri].tprof) continue;
    cudaSetDevice(r_[ri].sh->dev);
    std::vector<unsigned long long> h((size_t)kNex * 8);
    cudaMemcpy(h.data(), r_[ri].tprof, h.size() * 8, cudaMemcpyDeviceToHost);
    auto line = [&](const char* name, int i0, int i1) {
      double s[6] = {0, 0, 0, 0, 0, 0};
      for (int i = i0; i < i1; i++) for (int k = 0; k < 6; k++) s[k] += (double)h[(size_t)i * 8 + k];
      if (s[4] == 0) return;
      fprintf(stderr, "  dev %d %-14s n %7.0f | write %5.1f  wait %5.1f  read %5.1f  total %5.1f us (to the add)\n",
              r_[ri].sh->dev, name, s[4], s[0] / s[4] / 1e3, s[1] / s[4] / 1e3, s[2] / s[4] / 1e3, s[5] / s[4] / 1e3);
    };
    fprintf(stderr, "sum phases (Q27_SUMPROF), card %zu:\n", ri);
    line("idx 0-127", 0, 128);
    for (int i = 128; i < kNex; i++) { char nm[32]; snprintf(nm, sizeof nm, "idx %d", i); line(nm, i, i + 1); }
  }
  for (size_t ri = 0; ri < ss_.size(); ri++) {
    cudaSetDevice(r_[ri].sh->dev);
    cudaStreamSynchronize(ss_[ri]);
    cudaStreamDestroy(ss_[ri]);
    cudaEventDestroy(ev_staged_[ri]);
    cudaEventDestroy(ev_saved_[ri]);
  }
  for (auto& R : r_) {
    cudaSetDevice(R.sh->dev);
    for (auto& g : R.graph) if (g) cudaGraphExecDestroy(g);
    for (void* p : R.allocs) cudaFree(p);
    if (R.hin) cudaFreeHost(R.hin);
    if (R.s) cudaStreamDestroy(R.s);
    if (R.s2) cudaStreamDestroy(R.s2);
    if (R.ev_fork) cudaEventDestroy(R.ev_fork);
    if (R.ev_join) cudaEventDestroy(R.ev_join);
  }
  for (auto& P : pr_) if (P.htok) cudaFreeHost(P.htok);
  for (auto& P : pr_) if (P.hrope) cudaFreeHost(P.hrope);
  for (int ri = 0; ri < (int)img_.size(); ri++) { cudaSetDevice(r_[ri].sh->dev); cudaFree(img_[ri]); }
  for (int ri = 0; ri < (int)pr_.size(); ri++) {
    cudaSetDevice(r_[ri].sh->dev);
    if (pr_[ri].sx) cudaStreamDestroy(pr_[ri].sx);
    for (int h = 0; h < 2; h++) { if (pr_[ri].ev_part[h]) cudaEventDestroy(pr_[ri].ev_part[h]); if (pr_[ri].ev_recv[h]) cudaEventDestroy(pr_[ri].ev_recv[h]); }
  }
  for (void* p : {(void*)pex2_data_, (void*)pex2_flags_}) if (p) cudaFreeHost(p);
  for (void* p : {(void*)ar_data_, (void*)ar_flags_, (void*)ar_err_, (void*)mb_.vals, (void*)mb_.ids, (void*)mb_.flags,
                  (void*)emit_host_, (void*)pex_data_, (void*)pex_flags_})
    if (p) cudaFreeHost(p);
}

void Decoder::reset() {
  const Hparams& hp = m_.hp();
  last_n_ = 0;
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    for (auto* p : R.conv_st) CK(cudaMemsetAsync(p, 0, sizeof(float) * 4 * R.sh->conv_channels() * (hp.conv_k - 1), R.s));
    for (auto* p : R.ssm_st) CK(cudaMemsetAsync(p, 0, sizeof(float) * 4 * R.sh->vh * hp.ssm_dim * hp.ssm_dim, R.s));
    CK(cudaMemsetAsync(R.dplane, 0, sizeof(int), R.s));
    CK(cudaMemsetAsync(R.ddelta, 0, sizeof(int), R.s));
    CK(cudaMemsetAsync(R.pend_h, 0, sizeof(float) * hp.n_embd, R.s));
    CK(cudaMemsetAsync(R.st, 0, sizeof(DState), R.s));
    CK(cudaStreamSynchronize(R.s));
  }
  hpos_ = 0;
  rope_delta_ = 0;
}

int Decoder::position() const { return hpos_; }

// L2 prefetch budget per sum (the L2 is 32 MiB; the running GEMV streams through it too).
static size_t pf_budget() {
  // 8 MB was best with the bf16 wire; with the shorter q8 wire, 4 MB (2026-10-07, benchb.py)
  static const size_t b = [] { const char* e = getenv("Q27_PF_MB"); return (size_t)(e ? atoi(e) : wire_q8() ? 4 : 8) << 20; }();
  return b;
}

// L2 prefetch also in the kernels that do not wait (GDN conv, attention prep): measured slower (2026-10-06), so off
// unless Q27_PF_MID=1. The kernels that wait for the other card (cross-card sums) always prefetch.
// Q27_PDL_SUM=1: the GEMV after each cross-card sum starts early (PDL) and fetches its weights itself; the sum kernel
// then does no L2 prefetch.
static bool pdl_sum() {
  static const bool v = [] { const char* e = getenv("Q27_PDL_SUM"); return e && e[0] == '1'; }();
  return v;
}
static bool pf_mid() {
  static const bool v = [] { const char* e = getenv("Q27_PF_MID"); return e && e[0] == '1'; }();
  return v;
}

// Debug: Q27_UNFUSED bit mask runs the unfused kernel sequence (1 sums+norms, 2 GDN input, 4 gated norm,
// 8 SwiGLU, 16 attention output).
// Q27_EF=0 turns off the error feedback of the q8 wire (for measurements).
static bool ef_off() {
  static const bool v = [] { const char* e = getenv("Q27_EF"); return e && e[0] == '0'; }();
  return v;
}
static int unfused() {
  static const int u = [] { const char* e = getenv("Q27_UNFUSED"); return e ? atoi(e) : 0; }();
  return u;
}

void Decoder::sum_norm(int ri, const float* partial, int T, int& ix, bool with_sums, const float* w, float* h, const QMat* next) {
  const void* pf = next ? next->buf : nullptr;
  const size_t pfb = next ? std::min(next->bytes, pf_budget()) : 0;
  Rank& R = r_[ri];
  const Hparams& hp = m_.hp();
  const int index = ix++;
  if (unfused() & 1) {
    const int n = T * hp.n_embd;
    if (m_.tp() == 1 || !with_sums) add(R.x, partial, R.x, n, R.s);
    else {
      const int slot = index & 1;
      ar_add(R.x, partial, n, ar_data_ + ((size_t)slot * 2 + ri) * kArMax, ar_data_ + ((size_t)slot * 2 + (1 - ri)) * kArMax,
             ar_flags_ + ((size_t)slot * 2 + ri) * 4 * 32, ar_flags_ + ((size_t)slot * 2 + (1 - ri)) * 4 * 32, R.dstep, kNex, index,
             ar_err_, R.s);
    }
    float* hh = h ? h : R.m_out;  // m_out (T x E) is free after the sum
    rmsnorm(R.x, w, hh, hp.n_embd, T, hp.eps, R.s);
    quantize_q8_1(hh, R.xq, R.xd, hp.n_embd, T, R.s);
    return;
  }
  if (m_.tp() == 1 || !with_sums) {
    sum_norm_q8(R.x, partial, hp.n_embd, T, nullptr, w, hp.eps, h, R.xq, R.xd, R.s, pf, pfb);
    return;
  }
  const int slot = index & 1;
  ArArgs a;
  a.host_mine = ar_data_ + ((size_t)slot * 2 + ri) * kArMax;
  a.host_other = ar_data_ + ((size_t)slot * 2 + (1 - ri)) * kArMax;
  a.flag_mine = ar_flags_ + ((size_t)slot * 2 + ri) * 4 * 32;
  a.flag_other = ar_flags_ + ((size_t)slot * 2 + (1 - ri)) * 4 * 32;
  a.dstep = R.dstep;
  a.n_ar = kNex;
  a.index = index;
  a.err = ar_err_;
  a.ef = ef_off() ? nullptr : R.ef;
  a.ef_first = ef_first_ ? 1 : 0;
  a.tprof = R.tprof;
  ef_first_ = false;
  sum_norm_q8(R.x, partial, hp.n_embd, T, &a, w, hp.eps, h, R.xq, R.xd, R.s, pdl_sum() ? nullptr : pf, pdl_sum() ? 0 : pfb);
  if (pdl_sum()) pdl_once() = true;
}

void Decoder::candidates(int ri, const float* logits, int rows, bool draft, CandRow* out, int& ix, bool with_sums) {
  Rank& R = r_[ri];
  const CandMailbox* mb = (m_.tp() == 2 && with_sums) ? &mb_ : nullptr;
  const bool sub = draft && R.sh->draft_n > 0;  // drafts over the draft vocabulary
  sample_candidates(logits, rows, sub ? R.sh->draft_n : R.sh->vocab_n, R.sh->vocab0, draft ? sp_.draft_top_k : sp_.top_k, draft,
                    sp_, out, mb, ri, R.dstep, kNex, ix++, R.s, sub ? R.sh->draft_ids : nullptr);
}

void Decoder::upload_inputs(Rank& R) {
  CK(cudaMemcpyAsync(R.dtok, R.hin, kMaxT * sizeof(int), cudaMemcpyHostToDevice, R.s));
  CK(cudaMemcpyAsync(R.dpos, R.hin + kMaxT, sizeof(int), cudaMemcpyHostToDevice, R.s));
}

void Decoder::attn_layer(int ri, const Layer& L, void* kc, void* vc, int T, int& ix, bool with_sums) {
  Rank& R = r_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  cudaStream_t s = R.s;
  const float theta_scale = powf(hp.rope_base, -2.0f / hp.rope_dims);
  {
    cudaStream_t b = fork(ri);
    qgemv(L.wk, R.xq, R.xd, R.k, T, b);
    qgemv(L.wv, R.xq, R.xd, R.v, T, b);
  }
  qgemv(L.wq, R.xq, R.xd, R.qg, T, s);
  join(ri);
  mk(ri, "attn.qkv");
  attn_prep(R.qg, R.k, R.v, L.q_norm, L.k_norm, R.qn, kc, vc, R.dpos, T, hp.eps, theta_scale, sh.qh, sh.kvh, n_ctx_, kv_q8_, s,
            pf_mid() ? L.wo.buf : nullptr, pf_mid() ? std::min(L.wo.bytes, pf_budget()) : 0, nullptr, R.ddelta);
  mk(ri, "attn.prep");
  if (unfused() & 16) {
    attn_decode(R.qn, R.qg, kc, vc, R.o, nullptr, nullptr, R.dpos, T, 1.0f / sqrtf((float)hp.head_dim), sh.qh, sh.kvh, n_ctx_,
                kv_q8_, s);
    quantize_q8_1(R.o, R.xq, R.xd, sh.qh * hp.head_dim, T, s);
  } else
  attn_decode(R.qn, R.qg, kc, vc, nullptr, R.xq, R.xd, R.dpos, T, 1.0f / sqrtf((float)hp.head_dim), sh.qh, sh.kvh, n_ctx_,
              kv_q8_, s);
  mk(ri, "attn.decode");
  qgemv(L.wo, R.xq, R.xd, R.m_out, T, s);
  mk(ri, "attn.o");
  sum_norm(ri, R.m_out, T, ix, with_sums, L.post_norm, nullptr, &L.ffn_gate);
  mk(ri, "sum.attn");
}

void Decoder::ffn(int ri, const Layer& L, int T, int& ix, bool with_sums, const float* next_w, float* next_h, const QMat* next) {
  Rank& R = r_[ri];
  const Shard& sh = *R.sh;
  cudaStream_t s = R.s;
  qgemv(L.ffn_up, R.xq, R.xd, R.fu, T, fork(ri));
  qgemv(L.ffn_gate, R.xq, R.xd, R.fg, T, s);
  join(ri);
  mk(ri, "ffn.gate+up");
  if (unfused() & 8) { swiglu(R.fg, R.fu, R.act, T * sh.ff, s); quantize_q8_1(R.act, R.xq, R.xd, sh.ff, T, s); }
  else swiglu_q8(R.fg, R.fu, R.xq, R.xd, T * sh.ff, s);
  mk(ri, "ffn.swiglu");
  qgemv(L.ffn_down, R.xq, R.xd, R.m_out, T, s);
  mk(ri, "ffn.down");
  sum_norm(ri, R.m_out, T, ix, with_sums, next_w, next_h, next);
  mk(ri, "sum.ffn");
}

// Target pass over T tokens (dtok, dpos). snaps: keep the GDN state after every token (verify).
// set_plane: the next pass reads the final state. Writes hfin (final normed hidden) and logits.
void Decoder::forward(int ri, int T, bool snaps, bool set_plane, int& ix, bool with_sums) {
  Rank& R = r_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd, D = hp.ssm_dim;
  const float eps = hp.eps;
  cudaStream_t s = R.s;
  get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, R.dtok, T, R.x, E, s);
  ef_first_ = true;
  if (unfused() & 1) { rmsnorm(R.x, sh.layers[0].attn_norm, R.h, E, T, eps, s); quantize_q8_1(R.h, R.xq, R.xd, E, T, s); }
  else rmsnorm_q8(R.x, sh.layers[0].attn_norm, R.h, R.xq, R.xd, E, T, eps, s);
  mk(ri, "embed");
  int ig = 0, ia = 0;
  for (int il = 0; il < hp.n_layer; il++) {
    const Layer& L = sh.layers[il];
    if (!L.attn) {
      const int C = sh.conv_channels(), H = sh.vh, KH = sh.kh;
      {
        cudaStream_t b = fork(ri);
        gemv_bf16_pair(L.alpha, L.beta, R.h, R.ab, R.ab + T * H, H, E, T, b);
        qgemv(L.gate, R.xq, R.xd, R.z, T, b);
      }
      qgemv(L.qkv, R.xq, R.xd, R.qkv, T, s);
      join(ri);
      mk(ri, "gdn.qkv+z+ab");
      if (unfused() & 2) {
        gdn_gates(R.ab, R.ab + T * H, L.ssm_a, L.dt_bias, R.g, R.beta, H, T, s);
        gdn_conv(R.qkv, L.conv_w, R.conv_st[ig], R.dplane, R.conv, C, T, snaps, s);
        l2norm_heads(R.conv, D, 2 * KH, T, C, eps, s);
      } else
      gdn_conv_l2(R.qkv, L.conv_w, R.conv_st[ig], R.dplane, R.conv, C, T, snaps, 2 * KH, eps, R.ab, R.ab + T * H, L.ssm_a,
                  L.dt_bias, R.g, R.beta, H, s, pf_mid() ? L.ssm_out.buf : nullptr, pf_mid() ? std::min(L.ssm_out.bytes, pf_budget()) : 0);
      mk(ri, "gdn.conv");
      gdn_step(R.conv, R.conv + KH * D, R.conv + 2 * KH * D, C, R.g, R.beta, R.ssm_st[ig], R.dplane, R.o, H, KH, T,
               1.0f / sqrtf((float)D), snaps, s);
      mk(ri, "gdn.step");
      if (unfused() & 4) { gated_rmsnorm(R.o, L.ssm_norm, R.z, R.y, D, T * H, eps, s); quantize_q8_1(R.y, R.xq, R.xd, H * D, T, s); }
      else gated_rmsnorm_q8(R.o, L.ssm_norm, R.z, R.xq, R.xd, T * H, eps, s);
      mk(ri, "gdn.gnorm");
      qgemv(L.ssm_out, R.xq, R.xd, R.m_out, T, s);
      mk(ri, "gdn.out");
      sum_norm(ri, R.m_out, T, ix, with_sums, L.post_norm, nullptr, &L.ffn_gate);
      mk(ri, "sum.gdn");
      ig++;
    } else {
      attn_layer(ri, L, R.kc[ia], R.vc[ia], T, ix, with_sums);
      ia++;
    }
    const bool last = il + 1 == hp.n_layer;
    const Layer* N = last ? nullptr : &sh.layers[il + 1];
    ffn(ri, L, T, ix, with_sums, last ? sh.output_norm : N->attn_norm, last ? R.hfin : (N->attn ? nullptr : R.h),
        last ? &sh.output : (N->attn ? &N->wq : &N->qkv));
  }
  if (set_plane) set_int(R.dplane, T - 1, s);
  qgemv(sh.output, R.xq, R.xd, R.logits, T, s);
  mk(ri, "output");
}

// MTP pass over T rows: tokens dtok at dpos, hidden inputs hrows. Writes mtp_h and (if logits) logits.
void Decoder::mtp_forward(int ri, int T, bool logits, int& ix, bool with_sums) {
  Rank& R = r_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  cudaStream_t s = R.s;
  get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, R.dtok, T, R.x, E, s);
  ef_first_ = true;
  rmsnorm(R.x, sh.enorm, R.cat, E, T, hp.eps, s, 2 * E);           // cat[t][0:E]   = enorm(e)
  rmsnorm(R.hrows, sh.hnorm, R.cat + E, E, T, hp.eps, s, 2 * E);   // cat[t][E:2E]  = hnorm(h)
  quantize_q8_1(R.cat, R.xq, R.xd, 2 * E, T, s);
  mk(ri, "mtp.in");
  qgemv(sh.eh_proj, R.xq, R.xd, R.x, T, s);
  mk(ri, "mtp.eh");
  rmsnorm_q8(R.x, sh.mtp.attn_norm, nullptr, R.xq, R.xd, E, T, hp.eps, s);
  mk(ri, "mtp.norm");
  attn_layer(ri, sh.mtp, R.mkc, R.mvc, T, ix, with_sums);
  const QMat& head = sh.draft_n ? sh.draft_out : sh.output;  // draft logits: the draft vocabulary if set
  ffn(ri, sh.mtp, T, ix, with_sums, sh.shared_head_norm, R.mtp_h, logits ? &head : nullptr);
  if (logits) { qgemv(head, R.xq, R.xd, R.logits, T, s); mk(ri, "mtp.head"); }
}

void Decoder::enqueue(int ri, Kind kind, bool with_sums) {
  Rank& R = r_[ri];
  const int E = m_.hp().n_embd;
  cudaStream_t s = R.s;
  St* st = (St*)R.st;
  int* host_emit = ri == 0 ? emit_host_ : nullptr;
  int ix = 0;
  ph_ = "";
  mk(ri, "start");
  auto drafts = [&]() {
    ph_ = "D.";
    launch_k(k_copy, (E + 255) / 256, 256, 0, s, R.hrows, R.pend_h, E);
    for (int j = 0; j < kDrafts; j++) {
      mtp_forward(ri, 1, true, ix, with_sums);
      candidates(ri, R.logits, 1, true, R.qc + j, ix, with_sums);
      mk(ri, "cand");
      sample_tokens(R.qc + j, 1, &st->d[j], sp_.seed, &st->counter, 100 + j, s);
      mk(ri, "sample");
      if (j + 1 < kDrafts) {
        launch_k(k_next_draft, 1, 1, 0, s, st, j, R.dtok, R.dpos);
        launch_k(k_copy, (E + 255) / 256, 256, 0, s, R.hrows, R.mtp_h, E);
      }
    }
  };
  switch (kind) {
    case kT1: case kT4: {
      const int T = kind == kT1 ? 1 : 4;
      upload_inputs(R);
      forward(ri, T, false, true, ix, with_sums);
      break;
    }
    case kM1: case kM4: {
      const int T = kind == kM1 ? 1 : 4;
      upload_inputs(R);
      launch_k(k_rows, (E + 255) / 256, 256, 0, s, R.hrows, R.pend_h, R.hfin, T, E, nullptr);
      mtp_forward(ri, T, false, ix, with_sums);
      break;
    }
    case kFirst: {
      candidates(ri, R.logits, 1, false, R.pc, ix, with_sums);
      sample_tokens(R.pc, 1, &st->s, sp_.seed, &st->counter, 200, s);
      launch_k(k_first, 1, 1, 0, s, st, R.dtok, R.dpos, host_emit);
      drafts();
      launch_k(k_finish, 1, 1, 0, s, st);
      break;
    }
    case kStep: {
      launch_k(k_prep_verify, 1, 1, 0, s, st, R.dtok, R.dpos);
      ph_ = "V.";
      forward(ri, 4, true, false, ix, with_sums);
      ph_ = "S.";
      candidates(ri, R.logits, 4, false, R.pc, ix, with_sums);
      mk(ri, "cand");
      launch_k(k_accept, 1, 1, 0, s, st, R.pc, R.qc, sp_.seed, R.dplane, host_emit, (int)accept_block_on());
      launch_k(k_rows, (E + 255) / 256, 256, 0, s, R.hrows, R.pend_h, R.hfin, 4, E, st);
      mk(ri, "accept");
      ph_ = "C.";
      mtp_forward(ri, 4, false, ix, with_sums);
      launch_k(k_advance, 1, 1, 0, s, st, R.dtok, R.dpos);
      drafts();
      launch_k(k_finish, 1, 1, 0, s, st);
      ph_ = "";
      mk(ri, "end");
      break;
    }
    default: break;
  }
  if (ix > kNex) throw std::runtime_error("too many exchanges in one graph");
  if (m_.tp() == 2 && with_sums) inc_counter(R.dstep, s);
  CK(cudaGetLastError());
}

void Decoder::warmup() {
  // One eager pass of every graph kind without cross-card exchanges sizes every lazy workspace
  // before any capture. Then the state is cleared.
  for (int k = 0; k < kKinds; k++)
    for (int ri = 0; ri < (int)r_.size(); ri++) {
      Rank& R = r_[ri];
      CK(cudaSetDevice(R.sh->dev));
      for (int i = 0; i <= kMaxT; i++) R.hin[i] = 0;
      enqueue(ri, (Kind)k, false);
      CK(cudaStreamSynchronize(R.s));
    }
  // one prefill pass without exchanges: loads the prefill kernels (lazy module loading can take longer than
  // the exchange timeout) and sizes their workspaces
  {
    std::vector<int> toks(32, 0);
    hpos_ = 0;
    prefill_batch(toks.data(), 32, true, false, false);
    hpos_ = 0;
  }
  reset();
  warm_ = true;
}


void Decoder::run(Kind k) {
  if (!warm_) warmup();
  GapProf& gp = gapprof();
  const bool gap = gp.on && k == kStep && r_.size() <= 2;
  std::chrono::steady_clock::time_point tw0, tw1;
  if (gap) {
    tw0 = std::chrono::steady_clock::now();
    if (gp.have_last) gp.between += std::chrono::duration<double, std::micro>(tw0 - gp.last_end).count();
  }
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    CK(cudaSetDevice(R.sh->dev));
    if (use_graph && !R.graph[k]) {
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(R.s, cudaStreamCaptureModeThreadLocal));
      R.prof_a[k] = prof::next_slot();
      enqueue(ri, k, true);
      R.prof_b[k] = prof::next_slot();
      CK(cudaStreamEndCapture(R.s, &g));
      CK(cudaGraphInstantiate(&R.graph[k], g, 0));
      CK(cudaGraphDestroy(g));
    }
  }
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    CK(cudaSetDevice(R.sh->dev));
    if (gap) {
      if (!gp.e[ri][0]) { CK(cudaEventCreate(&gp.e[ri][0])); CK(cudaEventCreate(&gp.e[ri][1])); }
      CK(cudaEventRecord(gp.e[ri][0], R.s));
    }
    if (R.graph[k]) CK(cudaGraphLaunch(R.graph[k], R.s));
    else enqueue(ri, k, true);
    if (gap) CK(cudaEventRecord(gp.e[ri][1], R.s));
    cudaStreamQuery(R.s);  // push the work out of the WDDM queue so the peer can meet it
  }
  if (gap) tw1 = std::chrono::steady_clock::now();
  for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); CK(cudaStreamSynchronize(R.s)); }
  if (gap) {
    const auto tw2 = std::chrono::steady_clock::now();
    gp.n++;
    gp.wall += std::chrono::duration<double, std::micro>(tw2 - tw0).count();
    gp.launch += std::chrono::duration<double, std::micro>(tw1 - tw0).count();
    for (size_t ri = 0; ri < r_.size(); ri++) {
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, gp.e[ri][0], gp.e[ri][1]));
      gp.gpu[ri] += ms * 1000.0;
    }
    gp.last_end = tw2;
    gp.have_last = true;
  }
  if (ar_err_ && *ar_err_) throw std::runtime_error("cross-card exchange timed out");
  if (prof::on() && k == kStep)
    for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); prof::collect(R.prof_a[k], R.prof_b[k]); }
}
static bool branches() {
  static const bool v = [] { const char* e = getenv("Q27_BRANCH"); return !(e && e[0] == '0'); }();
  return v;
}
cudaStream_t Decoder::fork(int ri) {
  Rank& R = r_[ri];
  if (!branches()) return R.s;
  CK(cudaEventRecord(R.ev_fork, R.s));
  CK(cudaStreamWaitEvent(R.s2, R.ev_fork, 0));
  return R.s2;
}
void Decoder::join(int ri) {
  Rank& R = r_[ri];
  if (!branches()) return;
  CK(cudaEventRecord(R.ev_join, R.s2));
  CK(cudaStreamWaitEvent(R.s, R.ev_join, 0));
}
void Decoder::mk(int ri, const char* what) {
  if (prof::on()) prof::mark(r_[ri].s, (ph_ + what).c_str());
}

void Decoder::step(const int* tokens, int T, int pos) {
  if (T != 1 && T != 4) throw std::runtime_error("step: T must be 1 or 4");
  if (pos + T > n_ctx_) throw std::runtime_error("context full");
  if (!warm_) warmup();
  for (auto& R : r_) {
    for (int i = 0; i < kMaxT; i++) R.hin[i] = i < T ? tokens[i] : 0;
    R.hin[kMaxT] = pos;
  }
  run(T == 1 ? kT1 : kT4);
}

void Decoder::logits_to_host(float* out, int t) {
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    CK(cudaMemcpy(out + R.sh->vocab0, R.logits + (size_t)t * R.sh->vocab_n, sizeof(float) * R.sh->vocab_n,
                  cudaMemcpyDeviceToHost));
  }
}

void Decoder::feed(const int* tokens, int n) {
  if (n < 1) return;
  last_n_ = 0;
  if (hpos_ + n + 2 * kMaxT > n_ctx_) throw std::runtime_error("context full");
  if (!warm_) warmup();
  // Target pass, then the MTP pass over the same tokens (each MTP row pairs token p with the target
  // hidden of p-1). The last passes have one token, so the logits row 0 belongs to the last token.
  int p = 0;
  while (p < n) {
    const int T = (n - p > kMaxT) ? kMaxT : 1;
    for (auto& R : r_) {
      for (int i = 0; i < kMaxT; i++) R.hin[i] = i < T ? tokens[p + i] : 0;
      R.hin[kMaxT] = hpos_ + p;
    }
    run(T == 1 ? kT1 : kT4);
    run(T == 1 ? kM1 : kM4);
    p += T;
  }
  hpos_ += n;
}

int Decoder::start(const SampleParams& sp, int rng_start) {
  const bool same = sp.temp == sp_.temp && sp.top_k == sp_.top_k && sp.top_p == sp_.top_p && sp.min_p == sp_.min_p &&
                    sp.draft_top_k == sp_.draft_top_k && sp.seed == sp_.seed;
  if (!same) {  // sampling parameters are baked into the captured graphs
    for (auto& R : r_)
      for (Kind k : {kFirst, kStep})
        if (R.graph[k]) { CK(cudaSetDevice(R.sh->dev)); CK(cudaGraphExecDestroy(R.graph[k])); R.graph[k] = nullptr; }
    sp_ = sp;
  }
  if (hpos_ < 1) throw std::runtime_error("start: nothing fed");
  if (hpos_ + 2 * kMaxT > n_ctx_) throw std::runtime_error("context full");
  if (!warm_) warmup();
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    DState h{};
    h.P = hpos_;
    h.counter = rng_start;
    CK(cudaMemcpy(R.st, &h, sizeof(h), cudaMemcpyHostToDevice));
  }
  run(kFirst);
  last_n_ = 1;
  return emit_host_[1];
}

int Decoder::begin(const std::vector<int>& prompt, const SampleParams& sp) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  if (!warm_) warmup();
  reset();
  prefill(prompt.data(), (int)prompt.size());
  return start(sp);
}

bool Decoder::kv_q8_from_env() {
  const char* e = getenv("Q27_KV");  // default q8_0 (user decision 2026-10-06); Q27_KV=f16 for f16
  if (!e || !*e) return true;
  std::string s(e);
  for (char& ch : s) ch = (char)tolower((unsigned char)ch);
  if (s == "q8_0" || s == "q8") return true;
  if (s == "f16" || s == "fp16") return false;
  throw std::runtime_error("Q27_KV=" + std::string(e) + ": use q8_0 (default) or f16");
}

size_t Decoder::kv_bytes_per_token(const Model& m, bool kv_q8) {
  const Hparams& hp = m.hp();
  int n_attn = 0;
  for (int il = 0; il < hp.n_layer; il++) n_attn += hp.is_attn(il);
  return (size_t)(n_attn + 1) * 2 * kv_cache_bytes(m.shards[0].kvh, 1, kv_q8);
}

size_t Decoder::fixed_bytes(const Model& m) {
  const Hparams& hp = m.hp();
  const Shard& sh = m.shards[0];
  int n_gdn = 0;
  for (int il = 0; il < hp.n_layer; il++) n_gdn += !hp.is_attn(il);
  const size_t states = (size_t)n_gdn * 4 * ((size_t)sh.conv_channels() * (hp.conv_k - 1) + (size_t)sh.vh * hp.ssm_dim * hp.ssm_dim) * 4;
  return states + ((size_t)96 << 20);  // + activations, logits, workspaces, graphs
}

int Decoder::fit_ctx(const Model& m, const std::vector<size_t>& reserve, bool kv_q8) {
  const size_t per_tok = kv_bytes_per_token(m, kv_q8), fixed = fixed_bytes(m);
  long long best = m.hp().ctx_train;
  for (int r = 0; r < m.tp(); r++) {
    CK(cudaSetDevice(m.shards[r].dev));
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const long long avail = (long long)free_b - (long long)fixed - (long long)(r < (int)reserve.size() ? reserve[r] : 0);
    best = std::min(best, avail / (long long)per_tok);
  }
  return (int)std::max(0LL, best / 256 * 256);
}

void Decoder::draft_info(float* top1, float* pdraw) {
  Rank& R = r_[0];
  CK(cudaSetDevice(R.sh->dev));
  CandRow q[kDrafts];
  DState st;
  CK(cudaMemcpy(q, R.qc, sizeof(q), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&st, R.st, sizeof(st), cudaMemcpyDeviceToHost));
  for (int j = 0; j < kDrafts; j++) {
    top1[j] = q[j].n > 0 ? q[j].p[0] : 0.f;
    pdraw[j] = 0.f;
    for (int i = 0; i < q[j].n; i++) if (q[j].id[i] == st.d[j]) pdraw[j] = q[j].p[i];
  }
}

int Decoder::spec_step(int* out) {
  if (hpos_ + 2 * kMaxT > n_ctx_) throw std::runtime_error("context full");
  run(kStep);
  const int n = emit_host_[0];
  for (int i = 0; i < n; i++) out[i] = emit_host_[1 + i];
  hpos_ += n;
  last_n_ = n;
  return n;
}


// ---------------------------------------------------------------- prefill (batched prompt reading)
void Decoder::prefill_alloc() {
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd, MB = pmb_;
  pr_.resize(r_.size());
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    const Shard& sh = *R.sh;
    CK(cudaSetDevice(sh.dev));
    P.tok = alloc<int>(R, MB);
    CK(cudaHostAlloc(&P.htok, (MB + 4) * sizeof(int), cudaHostAllocDefault));
    P.x = alloc<float>(R, (size_t)MB * E); P.h = alloc<float>(R, (size_t)MB * E); P.hfin = alloc<float>(R, (size_t)MB * E);
    P.m_out = alloc<float>(R, (size_t)MB * E); P.hrows = alloc<float>(R, (size_t)MB * E);
    P.qkv = alloc<float>(R, (size_t)MB * sh.conv_channels()); P.z = alloc<float>(R, (size_t)MB * sh.vh * hp.ssm_dim);
    P.ab = alloc<float>(R, (size_t)2 * MB * sh.vh); P.g = alloc<float>(R, (size_t)MB * sh.vh); P.beta = alloc<float>(R, (size_t)MB * sh.vh);
    P.conv = alloc<float>(R, (size_t)MB * sh.conv_channels());
    P.o = alloc<float>(R, (size_t)MB * std::max(sh.vh * hp.ssm_dim, sh.qh * hp.head_dim));
    P.fg = alloc<float>(R, (size_t)MB * sh.ff); P.fu = alloc<float>(R, (size_t)MB * sh.ff);
    P.qg = alloc<float>(R, (size_t)MB * 2 * sh.qh * hp.head_dim);
    P.k = alloc<float>(R, (size_t)MB * sh.kvh * hp.head_dim); P.v = alloc<float>(R, (size_t)MB * sh.kvh * hp.head_dim);
    P.qn = alloc<float>(R, (size_t)MB * sh.qh * hp.head_dim);
    P.cat = alloc<float>(R, (size_t)MB * 2 * E);
    const int kmax = std::max({2 * E, sh.ff, sh.vh * hp.ssm_dim, sh.qh * hp.head_dim});
    P.xq = alloc<int8_t>(R, (size_t)MB * kmax); P.xd = alloc<float>(R, (size_t)MB * kmax / 32);
    P.xs = alloc<float>(R, (size_t)MB * kmax / 16);
  }
  if (m_.tp() == 2) {
    CK(cudaHostAlloc(&pex_data_, sizeof(__nv_bfloat16) * 2 * 2 * (size_t)MB * E, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&pex_flags_, sizeof(int) * 2 * 2 * (size_t)MB * 32, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(pex_flags_, 0, sizeof(int) * 2 * 2 * (size_t)MB * 32);
  }
}

void Decoder::pf_sum(int ri, int M, bool exchange, const float* w, float* h) {
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  if (m_.tp() == 1 || !exchange) {
    sum_norm_rows(P.x, P.m_out, E, M, nullptr, w, hp.eps, h, P.xq, P.xd, P.xs, R.s);
    return;
  }
  const int token = ++P.token;  // both cards make the same sequence of exchanges
  const int slot = token & 1;
  PfExchange ex;
  ex.mine = pex_data_ + ((size_t)slot * 2 + ri) * pmb_ * E;
  ex.other = pex_data_ + ((size_t)slot * 2 + (1 - ri)) * pmb_ * E;
  ex.flag_mine = pex_flags_ + ((size_t)slot * 2 + ri) * pmb_ * 32;
  ex.flag_other = pex_flags_ + ((size_t)slot * 2 + (1 - ri)) * pmb_ * 32;
  ex.token = token;
  ex.err = ar_err_;
  sum_norm_rows(P.x, P.m_out, E, M, &ex, w, hp.eps, h, P.xq, P.xd, P.xs, R.s);
}

// One layer of the target model over M rows (x, and the q8_1 input of the layer in xq/xd/xs, are ready).
void Decoder::pf_layer(int ri, int il, int M, int& ig, int& ia, bool exchange) {
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd, D = hp.ssm_dim;
  const float eps = hp.eps;
  cudaStream_t s = R.s;
  const Layer& L = sh.layers[il];
  if (!L.attn) {
    const int C = sh.conv_channels(), H = sh.vh, KH = sh.kh;
    qgemm(L.qkv, P.xq, P.xd, P.xs, P.qkv, M, s);
    qgemm(L.gate, P.xq, P.xd, P.xs, P.z, M, s);
    bf16_pair_gemm(L.alpha, L.beta, P.h, P.ab, P.ab + (size_t)M * H, H, E, M, s);
    gdn_conv_prefill(P.qkv, L.conv_w, R.conv_st[ig], R.dplane, P.conv, C, M, 2 * KH, eps, P.ab, P.ab + (size_t)M * H, L.ssm_a,
                     L.dt_bias, P.g, P.beta, H, s);
    gdn_step(P.conv, P.conv + KH * D, P.conv + 2 * KH * D, C, P.g, P.beta, R.ssm_st[ig], R.dplane, P.o, H, KH, M,
             1.0f / sqrtf((float)D), false, s, true);
    gated_rmsnorm_q8(P.o, L.ssm_norm, P.z, P.xq, P.xd, M * H, eps, s, P.xs);
    qgemm(L.ssm_out, P.xq, P.xd, P.xs, P.m_out, M, s);
    ig++;
  } else {
    const float theta_scale = powf(hp.rope_base, -2.0f / hp.rope_dims);
    qgemm(L.wq, P.xq, P.xd, P.xs, P.qg, M, s);
    qgemm(L.wk, P.xq, P.xd, P.xs, P.k, M, s);
    qgemm(L.wv, P.xq, P.xd, P.xs, P.v, M, s);
    attn_prep(P.qg, P.k, P.v, L.q_norm, L.k_norm, P.qn, R.kc[ia], R.vc[ia], R.dpos, M, eps, theta_scale, sh.qh, sh.kvh, n_ctx_,
              kv_q8_, s, nullptr, 0, rope_dev(ri, 0), R.ddelta);
    attn_prefill(P.qn, P.qg, R.kc[ia], R.vc[ia], P.o, R.dpos, M, 1.0f / sqrtf((float)hp.head_dim), sh.qh, sh.kvh, n_ctx_, kv_q8_, s);
    quantize_q8_1(P.o, P.xq, P.xd, sh.qh * hp.head_dim, M, s, P.xs);
    qgemm(L.wo, P.xq, P.xd, P.xs, P.m_out, M, s);
    ia++;
  }
  pf_sum(ri, M, exchange, L.post_norm, nullptr);
  qgemm(L.ffn_gate, P.xq, P.xd, P.xs, P.fg, M, s);
  qgemm(L.ffn_up, P.xq, P.xd, P.xs, P.fu, M, s);
  swiglu_q8(P.fg, P.fu, P.xq, P.xd, M * sh.ff, s, P.xs);
  qgemm(L.ffn_down, P.xq, P.xd, P.xs, P.m_out, M, s);
  const bool last = il + 1 == hp.n_layer;
  const Layer* N = last ? nullptr : &sh.layers[il + 1];
  pf_sum(ri, M, exchange, last ? sh.output_norm : N->attn_norm, last ? P.hfin : (N->attn ? nullptr : P.h));
}

// MTP block over the M rows of the batch: row t pairs token t with the target hidden of the token before it.
void Decoder::pf_mtp(int ri, int M, bool exchange) {
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  const float eps = hp.eps;
  cudaStream_t s = R.s;
  CK(cudaMemcpyAsync(P.hrows, R.pend_h, sizeof(float) * E, cudaMemcpyDeviceToDevice, s));
  if (M > 1) CK(cudaMemcpyAsync(P.hrows + E, P.hfin, sizeof(float) * (size_t)(M - 1) * E, cudaMemcpyDeviceToDevice, s));
  CK(cudaMemcpyAsync(R.pend_h, P.hfin + (size_t)(M - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, s));
  get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, P.tok, M, P.x, E, s, img_.empty() ? nullptr : img_[ri]);
  rmsnorm(P.x, sh.enorm, P.cat, E, M, eps, s, 2 * E);
  rmsnorm(P.hrows, sh.hnorm, P.cat + E, E, M, eps, s, 2 * E);
  quantize_q8_1(P.cat, P.xq, P.xd, 2 * E, M, s, P.xs);
  qgemm(sh.eh_proj, P.xq, P.xd, P.xs, P.x, M, s);
  rmsnorm_q8(P.x, sh.mtp.attn_norm, nullptr, P.xq, P.xd, E, M, eps, s, P.xs);
  const Layer& L = sh.mtp;
  const float theta_scale = powf(hp.rope_base, -2.0f / hp.rope_dims);
  qgemm(L.wq, P.xq, P.xd, P.xs, P.qg, M, s);
  qgemm(L.wk, P.xq, P.xd, P.xs, P.k, M, s);
  qgemm(L.wv, P.xq, P.xd, P.xs, P.v, M, s);
  attn_prep(P.qg, P.k, P.v, L.q_norm, L.k_norm, P.qn, R.mkc, R.mvc, R.dpos, M, eps, theta_scale, sh.qh, sh.kvh, n_ctx_, kv_q8_, s,
            nullptr, 0, rope_dev(ri, 0), R.ddelta);
  attn_prefill(P.qn, P.qg, R.mkc, R.mvc, P.o, R.dpos, M, 1.0f / sqrtf((float)hp.head_dim), sh.qh, sh.kvh, n_ctx_, kv_q8_, s);
  quantize_q8_1(P.o, P.xq, P.xd, sh.qh * hp.head_dim, M, s, P.xs);
  qgemm(L.wo, P.xq, P.xd, P.xs, P.m_out, M, s);
  pf_sum(ri, M, exchange, L.post_norm, nullptr);
  qgemm(L.ffn_gate, P.xq, P.xd, P.xs, P.fg, M, s);
  qgemm(L.ffn_up, P.xq, P.xd, P.xs, P.fu, M, s);
  swiglu_q8(P.fg, P.fu, P.xq, P.xd, M * sh.ff, s, P.xs);
  qgemm(L.ffn_down, P.xq, P.xd, P.xs, P.m_out, M, s);
  pf_sum(ri, M, exchange, sh.shared_head_norm, nullptr);
}

void Decoder::prefill_batch(const int* tokens, int M, bool last, bool all_logits, bool exchange) {
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  const int nr = (int)r_.size();
  for (int ri = 0; ri < nr; ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    CK(cudaSetDevice(R.sh->dev));
    memcpy(P.htok, tokens, sizeof(int) * M);
    P.htok[M] = hpos_;
    CK(cudaMemcpyAsync(P.tok, P.htok, sizeof(int) * M, cudaMemcpyHostToDevice, R.s));
    CK(cudaMemcpyAsync(R.dpos, P.htok + M, sizeof(int), cudaMemcpyHostToDevice, R.s));
    upload_rope(ri, M);
    get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, P.tok, M, P.x, E, R.s, img_.empty() ? nullptr : img_[ri]);
    rmsnorm_q8(P.x, R.sh->layers[0].attn_norm, P.h, P.xq, P.xd, E, M, hp.eps, R.s, P.xs);
  }
  // Layer by layer on both cards, so the cross-card sums of one layer are queued on both before the next.
  std::vector<int> ig(nr, 0), ia(nr, 0);
  for (int il = 0; il < hp.n_layer; il++)
    for (int ri = 0; ri < nr; ri++) {
      CK(cudaSetDevice(r_[ri].sh->dev));
      pf_layer(ri, il, M, ig[ri], ia[ri], exchange);
      cudaStreamQuery(r_[ri].s);
    }
  for (int ri = 0; ri < nr; ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    const Shard& sh = *R.sh;
    CK(cudaSetDevice(sh.dev));
    gdn_flip_plane(R.dplane, R.s);
    if (last) qgemv(sh.output, P.xq + (size_t)(M - 1) * E, P.xd + (size_t)(M - 1) * (E / 32), R.logits, 1, R.s);
    if (all_logits) {
      if (!P.plog) {
        CK(cudaMalloc(&P.plog, sizeof(float) * (size_t)pmb_ * sh.vocab_n));
        R.allocs.push_back(P.plog);
      }
      qgemm(sh.output, P.xq, P.xd, P.xs, P.plog, M, R.s);
    }
    pf_mtp(ri, M, exchange);
    cudaStreamQuery(R.s);
  }
  for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); CK(cudaStreamSynchronize(R.s)); }
  if (ar_err_ && *ar_err_) throw std::runtime_error("cross-card exchange timed out (prefill)");
  hpos_ += M;
}

void Decoder::prefill(const int* tokens, int n, const int* rope3) {
  if (n < 1) return;
  last_n_ = 0;
  bool img_rows = false;
  for (int i = 0; i < n; i++) img_rows |= tokens[i] < 0;
  if (img_rows && img_.empty()) throw std::runtime_error("prefill: image rows without set_image");
  if (n < kPrefillMin && !rope3 && !img_rows) { feed(tokens, n); return; }
  if (hpos_ + n + 2 * kMaxT > n_ctx_) throw std::runtime_error("context full");
  if (!warm_) warmup();
  static const bool ov = [] { const char* e = getenv("Q27_PF_OVERLAP"); return !(e && e[0] == '0'); }();
  int done = 0;
  while (done < n) {
    const int M = std::min(pmb_, n - done);
    batch_rope_ = rope3 ? rope3 + 3 * (size_t)done : nullptr;
    if (ov && M >= 64) prefill_batch_ov(tokens + done, M, done + M == n, false);
    else prefill_batch(tokens + done, M, done + M == n, false, true);
    done += M;
  }
  batch_rope_ = nullptr;
}

void Decoder::upload_rope(int ri, int M) {
  PRank& P = pr_[ri];
  if (!batch_rope_) return;
  if (!P.rope) {
    P.rope = alloc<int>(r_[ri], (size_t)3 * pmb_);
    CK(cudaHostAlloc(&P.hrope, sizeof(int) * 3 * (size_t)pmb_, cudaHostAllocDefault));
  }
  memcpy(P.hrope, batch_rope_, sizeof(int) * 3 * (size_t)M);
  CK(cudaMemcpyAsync(P.rope, P.hrope, sizeof(int) * 3 * (size_t)M, cudaMemcpyHostToDevice, r_[ri].s));
}

const int* Decoder::rope_dev(int ri, int r0) const { return batch_rope_ ? pr_[ri].rope + 3 * (size_t)r0 : nullptr; }

void Decoder::set_image(const float* src, int rows, int src_dev) {
  const int E = m_.hp().n_embd;
  if (rows > img_cap_) {
    for (int ri = 0; ri < (int)img_.size(); ri++) { CK(cudaSetDevice(r_[ri].sh->dev)); CK(cudaFree(img_[ri])); }
    img_.assign(r_.size(), nullptr);
    img_cap_ = std::max(rows, 1024);
    for (int ri = 0; ri < (int)r_.size(); ri++) {
      CK(cudaSetDevice(r_[ri].sh->dev));
      CK(cudaMalloc(&img_[ri], sizeof(float) * (size_t)img_cap_ * E));
    }
  }
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    const int dev = r_[ri].sh->dev;
    CK(cudaSetDevice(dev));
    if (dev == src_dev) CK(cudaMemcpy(img_[ri], src, sizeof(float) * (size_t)rows * E, cudaMemcpyDeviceToDevice));
    else CK(cudaMemcpyPeer(img_[ri], dev, src, src_dev, sizeof(float) * (size_t)rows * E));
  }
  // the copies run on the legacy stream, which the non-blocking compute streams do not wait for
  for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); CK(cudaDeviceSynchronize()); }
}

void Decoder::set_rope_delta(int delta) {
  rope_delta_ = delta;
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    set_int(R.ddelta, delta, R.s);
    CK(cudaStreamSynchronize(R.s));
  }
}

void Decoder::prefill_logits(const int* tokens, int M) {
  if (M < 1 || M > pmb_) throw std::runtime_error("prefill_logits: bad M");
  if (hpos_ + M + 2 * kMaxT > n_ctx_) throw std::runtime_error("context full");
  if (!warm_) warmup();
  static const bool ov = [] { const char* e = getenv("Q27_PF_OVERLAP"); return !(e && e[0] == '0'); }();
  if (ov && M >= 64) prefill_batch_ov(tokens, M, true, true);
  else prefill_batch(tokens, M, true, true, true);
}

void Decoder::prefill_logits_to_host(float* out, int t) {
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    CK(cudaSetDevice(R.sh->dev));
    CK(cudaMemcpy(out + R.sh->vocab0, pr_[ri].plog + (size_t)t * R.sh->vocab_n, sizeof(float) * R.sh->vocab_n,
                  cudaMemcpyDeviceToHost));
  }
}

// ---------------------------------------------------------------- prefill with overlapped cross-card sums
// The batch is split into two halves A and B. Both run on the compute stream in the order
//   A(p), B(p), norm A(p), A(p+1), norm B(p), B(p+1), ...
// where part p ends with a row-split GEMM (a partial sum). While one half computes, the other half's exchange
// runs on a second stream that uses only the copy engine: BF16 copy to pinned host memory, a flag written by a
// stream memory operation, a wait for the peer's flag, and a copy of the peer's rows back (cuStreamWaitValue32).
// Then norm A(p) adds both partials (BF16 wire, as the decode sum) and runs the next RMSNorm + q8_1.
// GDN: half A reads the states of plane *dplane and writes plane flip(dplane); half B reads that plane
// (dplane2) and writes flip(dplane2); at the end dplane = flip(dplane2). Attention: B reads A's K/V rows.
namespace {
typedef CUresult(CUDAAPI* PFN_wait32_t)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult(CUDAAPI* PFN_write32_t)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
PFN_wait32_t p_wait32 = nullptr;
PFN_write32_t p_write32 = nullptr;
void load_memops() {
  if (p_wait32) return;
  cudaDriverEntryPointQueryResult q;
  CK(cudaGetDriverEntryPoint("cuStreamWaitValue32", (void**)&p_wait32, cudaEnableDefault, &q));
  CK(cudaGetDriverEntryPoint("cuStreamWriteValue32", (void**)&p_write32, cudaEnableDefault, &q));
  if (!p_wait32 || !p_write32) throw std::runtime_error("stream memory operations not available");
}
}  // namespace

constexpr int kPfParts = 130;
// Test switch: Q27_PF_NOEX=1 skips the cross-card exchange in prompt reading (wrong results; shows the compute time).
static bool pf_noex() {
  static const bool v = [] { const char* e = getenv("Q27_PF_NOEX"); return e && e[0] == '1'; }();
  return v;
}  // 64 layers x (block, FFN) + MTP (attention, FFN)

void Decoder::pf2_alloc() {
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd, MB = pmb_;
  gdn_idx_.assign(hp.n_layer, -1);
  attn_idx_.assign(hp.n_layer, -1);
  for (int il = 0, ig = 0, ia = 0; il < hp.n_layer; il++) (hp.is_attn(il) ? attn_idx_[il] = ia++ : gdn_idx_[il] = ig++);
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    CK(cudaSetDevice(R.sh->dev));
    CK(cudaStreamCreateWithFlags(&P.sx, cudaStreamNonBlocking));
    for (int h = 0; h < 2; h++) {
      CK(cudaEventCreateWithFlags(&P.ev_part[h], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&P.ev_recv[h], cudaEventDisableTiming));
    }
    P.sendbuf = alloc<__nv_bfloat16>(R, (size_t)MB * E);
    P.recvbuf = alloc<__nv_bfloat16>(R, (size_t)MB * E);
    if (m_.tp() == 2 && wire_q8()) P.ef = alloc<float>(R, (size_t)MB * E);
    P.dpos2 = alloc<int>(R, 2);
    P.dplane2 = alloc<int>(R, 1);
  }
  if (m_.tp() == 2) {
    // [2 halves][2 slots][2 ranks][MB/2 rows][E] and flags [2 halves][2 slots][2 ranks][32]
    CK(cudaHostAlloc(&pex2_data_, sizeof(__nv_bfloat16) * 2 * 2 * 2 * (size_t)(MB / 2) * E, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&pex2_flags_, sizeof(int) * 2 * 2 * 2 * 32, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(pex2_flags_, 0, sizeof(int) * 2 * 2 * 2 * 32);
    load_memops();
  }
}

// Part p of half hv (rows r0 .. r0+Mh-1): from the half's q8_1 input to the partial sum in m_out.
void Decoder::pf2_part(int ri, int p, const PfHalf& hv) {
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd, D = hp.ssm_dim, Mh = hv.Mh;
  const size_t r0 = hv.r0;
  const float eps = hp.eps;
  cudaStream_t s = R.s;
  int8_t* xq = hv.xq;
  float *xd = hv.xd, *xs = hv.xs;
  float* m_out = P.m_out + r0 * E;
  auto attention = [&](const Layer& L, void* kc, void* vc) {
    const float theta_scale = powf(hp.rope_base, -2.0f / hp.rope_dims);
    float* qg = P.qg + r0 * 2 * sh.qh * hp.head_dim;
    float* k = P.k + r0 * sh.kvh * hp.head_dim;
    float* v = P.v + r0 * sh.kvh * hp.head_dim;
    float* qn = P.qn + r0 * sh.qh * hp.head_dim;
    float* o = P.o + r0 * std::max(sh.vh * hp.ssm_dim, sh.qh * hp.head_dim);
    qgemm(L.wq, xq, xd, xs, qg, Mh, s);
    mk(ri, "attn.q");
    qgemm(L.wk, xq, xd, xs, k, Mh, s);
    qgemm(L.wv, xq, xd, xs, v, Mh, s);
    mk(ri, "attn.kv");
    attn_prep(qg, k, v, L.q_norm, L.k_norm, qn, kc, vc, hv.dpos, Mh, eps, theta_scale, sh.qh, sh.kvh, n_ctx_, kv_q8_, s,
              nullptr, 0, rope_dev(ri, (int)r0), R.ddelta);
    mk(ri, "attn.prep");
    attn_prefill(qn, qg, kc, vc, o, hv.dpos, Mh, 1.0f / sqrtf((float)hp.head_dim), sh.qh, sh.kvh, n_ctx_, kv_q8_, s,
                 hpos_ + (int)r0);
    mk(ri, "attn.fa");
    quantize_q8_1(o, xq, xd, sh.qh * hp.head_dim, Mh, s, xs);
    mk(ri, "attn.quant");
    qgemm(L.wo, xq, xd, xs, m_out, Mh, s);
    mk(ri, "attn.o");
  };
  auto ffn = [&](const Layer& L) {
    float* fg = P.fg + r0 * sh.ff;
    float* fu = P.fu + r0 * sh.ff;
    qgemm(L.ffn_gate, xq, xd, xs, fg, Mh, s);
    mk(ri, "ffn.gate");
    qgemm(L.ffn_up, xq, xd, xs, fu, Mh, s);
    mk(ri, "ffn.up");
    swiglu_q8(fg, fu, xq, xd, Mh * sh.ff, s, xs);
    mk(ri, "ffn.swiglu");
    qgemm(L.ffn_down, xq, xd, xs, m_out, Mh, s);
    mk(ri, "ffn.down");
  };
  if (p < 2 * hp.n_layer) {
    const int il = p / 2;
    const Layer& L = sh.layers[il];
    if (p % 2) { ffn(L); return; }
    if (L.attn) { attention(L, R.kc[attn_idx_[il]], R.vc[attn_idx_[il]]); return; }
    const int C = sh.conv_channels(), H = sh.vh, KH = sh.kh, ig = gdn_idx_[il];
    float* qkv = P.qkv + r0 * C;
    float* z = P.z + r0 * H * D;
    float* ab = P.ab + r0 * 2 * H;
    float* g = P.g + r0 * H;
    float* beta = P.beta + r0 * H;
    float* conv = P.conv + r0 * C;
    float* o = P.o + r0 * std::max(sh.vh * hp.ssm_dim, sh.qh * hp.head_dim);
    qgemm(L.qkv, xq, xd, xs, qkv, Mh, s);
    mk(ri, "gdn.qkv");
    qgemm(L.gate, xq, xd, xs, z, Mh, s);
    mk(ri, "gdn.z");
    bf16_pair_gemm(L.alpha, L.beta, P.h + r0 * E, ab, ab + (size_t)Mh * H, H, E, Mh, s);
    mk(ri, "gdn.ab");
    gdn_conv_prefill(qkv, L.conv_w, R.conv_st[ig], hv.dplane, conv, C, Mh, 2 * KH, eps, ab, ab + (size_t)Mh * H, L.ssm_a, L.dt_bias, g,
                     beta, H, s);
    mk(ri, "gdn.conv");
    gdn_step(conv, conv + KH * D, conv + 2 * KH * D, C, g, beta, R.ssm_st[ig], hv.dplane, o, H, KH, Mh, 1.0f / sqrtf((float)D),
             false, s, true);
    mk(ri, "gdn.step");
    gated_rmsnorm_q8(o, L.ssm_norm, z, xq, xd, Mh * H, eps, s, xs);
    mk(ri, "gdn.gnorm");
    qgemm(L.ssm_out, xq, xd, xs, m_out, Mh, s);
    mk(ri, "gdn.out");
    return;
  }
  if (p == 2 * hp.n_layer) {  // MTP: inputs (token embedding, target hidden of the previous token), then attention
    float* hrows = P.hrows + r0 * E;
    if (r0 == 0) {
      CK(cudaMemcpyAsync(hrows, R.pend_h, sizeof(float) * E, cudaMemcpyDeviceToDevice, s));
      if (Mh > 1) CK(cudaMemcpyAsync(hrows + E, P.hfin, sizeof(float) * (size_t)(Mh - 1) * E, cudaMemcpyDeviceToDevice, s));
    } else {
      CK(cudaMemcpyAsync(hrows, P.hfin + (r0 - 1) * E, sizeof(float) * (size_t)Mh * E, cudaMemcpyDeviceToDevice, s));
    }
    float* x = P.x + r0 * E;
    float* cat = P.cat + r0 * 2 * E;
    get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, P.tok + r0, Mh, x, E, s, img_.empty() ? nullptr : img_[ri]);
    rmsnorm(x, sh.enorm, cat, E, Mh, eps, s, 2 * E);
    rmsnorm(hrows, sh.hnorm, cat + E, E, Mh, eps, s, 2 * E);
    quantize_q8_1(cat, xq, xd, 2 * E, Mh, s, xs);
    qgemm(sh.eh_proj, xq, xd, xs, x, Mh, s);
    rmsnorm_q8(x, sh.mtp.attn_norm, nullptr, xq, xd, E, Mh, eps, s, xs);
    attention(sh.mtp, R.mkc, R.mvc);
    return;
  }
  ffn(sh.mtp);
}

// Exchange of half hv's partial (part p) on the copy-engine stream.
void Decoder::pf2_send(int ri, const PfHalf& hv, int p) {
  const bool hv_first = p == 0 || p == 2 * m_.hp().n_layer;  // first exchange of the target pass or of the MTP pass
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const int E = m_.hp().n_embd;
  const size_t r0 = hv.r0, n = (size_t)hv.Mh * E;
  CK(cudaEventRecord(P.ev_part[hv.half], R.s));
  if (m_.tp() == 1 || pf_noex()) return;
  const int token = ++P.tok2[hv.half];
  const int slot = token & 1;
  const size_t rows_cap = pmb_ / 2;
  __nv_bfloat16* mine = pex2_data_ + (((size_t)hv.half * 2 + slot) * 2 + ri) * rows_cap * E;
  __nv_bfloat16* other = pex2_data_ + (((size_t)hv.half * 2 + slot) * 2 + (1 - ri)) * rows_cap * E;
  int* fm = pex2_flags_ + (((size_t)hv.half * 2 + slot) * 2 + ri) * 32;
  int* fo = pex2_flags_ + (((size_t)hv.half * 2 + slot) * 2 + (1 - ri)) * 32;
  cudaStream_t x = P.sx;
  CK(cudaStreamWaitEvent(x, P.ev_part[hv.half], 0));
  const bool q8 = wire_q8();
  const size_t rb = q8 ? (size_t)E + 2 * E / wire_qb() : (size_t)E * 2;  // wire bytes per row
  uint8_t* sb = (uint8_t*)P.sendbuf + r0 * rb;
  if (q8) to_q8_wire(P.m_out + r0 * E, ef_off() ? nullptr : P.ef + r0 * E, hv_first, sb, E, hv.Mh, x);
  else to_bf16(P.m_out + r0 * E, P.sendbuf + r0 * E, n, x);
  CK(cudaMemcpyAsync(mine, sb, hv.Mh * rb, cudaMemcpyDeviceToHost, x));
  if (p_write32((CUstream)x, (CUdeviceptr)fm, (cuuint32_t)token, 0) != CUDA_SUCCESS) throw std::runtime_error("cuStreamWriteValue32");
  if (p_wait32((CUstream)x, (CUdeviceptr)fo, (cuuint32_t)token, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS)
    throw std::runtime_error("cuStreamWaitValue32");
  CK(cudaMemcpyAsync((uint8_t*)P.recvbuf + r0 * rb, other, hv.Mh * rb, cudaMemcpyHostToDevice, x));
  CK(cudaEventRecord(P.ev_recv[hv.half], x));
  cudaStreamQuery(x);
}

// Sum of part p for half hv (own partial + received partial), then the next RMSNorm (+ q8_1 input of part p+1).
void Decoder::pf2_norm(int ri, int p, const PfHalf& hv) {
  Rank& R = r_[ri];
  PRank& P = pr_[ri];
  const Shard& sh = *R.sh;
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  const size_t r0 = hv.r0;
  const float* w;
  float* h = nullptr;
  if (p < 2 * hp.n_layer) {
    const int il = p / 2;
    if (p % 2 == 0) w = sh.layers[il].post_norm;
    else if (il + 1 == hp.n_layer) { w = sh.output_norm; h = P.hfin + r0 * E; }
    else { w = sh.layers[il + 1].attn_norm; if (!sh.layers[il + 1].attn) h = P.h + r0 * E; }
  } else {
    w = p == 2 * hp.n_layer ? sh.mtp.post_norm : sh.shared_head_norm;
  }
  const bool ex = m_.tp() == 2 && !pf_noex();
  if (ex) CK(cudaStreamWaitEvent(R.s, P.ev_recv[hv.half], 0));
  if (ex && wire_q8()) {
    const size_t rb = (size_t)E + 2 * E / wire_qb();
    add_norm_rows_q8(P.x + r0 * E, (const uint8_t*)P.sendbuf + r0 * rb, (const uint8_t*)P.recvbuf + r0 * rb, E, hv.Mh, w, hp.eps, h,
                     hv.xq, hv.xd, hv.xs, R.s);
  } else
  add_norm_rows(P.x + r0 * E, P.m_out + r0 * E, ex ? P.recvbuf + r0 * E : nullptr, E, hv.Mh, w, hp.eps, h, hv.xq, hv.xd,
                hv.xs, R.s);
  mk(ri, "sum");
}

void Decoder::prefill_batch_ov(const int* tokens, int M, bool last, bool all_logits) {
  const Hparams& hp = m_.hp();
  const int E = hp.n_embd;
  const int nr = (int)r_.size();
  const int MA = (M / 2 + 15) / 16 * 16, MBh = M - MA;
  std::vector<std::array<PfHalf, 2>> hv(nr);
  for (int ri = 0; ri < nr; ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    const Shard& sh = *R.sh;
    CK(cudaSetDevice(sh.dev));
    const int kmax = std::max({2 * E, sh.ff, sh.vh * hp.ssm_dim, sh.qh * hp.head_dim});
    memcpy(P.htok, tokens, sizeof(int) * M);
    P.htok[M] = hpos_;
    P.htok[M + 1] = hpos_ + MA;
    CK(cudaMemcpyAsync(P.tok, P.htok, sizeof(int) * M, cudaMemcpyHostToDevice, R.s));
    CK(cudaMemcpyAsync(P.dpos2, P.htok + M, 2 * sizeof(int), cudaMemcpyHostToDevice, R.s));
    flip_plane_to(P.dplane2, R.dplane, R.s);  // half B reads the plane half A writes
    for (int h = 0; h < 2; h++) {
      PfHalf& v = hv[ri][h];
      v.half = h;
      v.r0 = h ? MA : 0;
      v.Mh = h ? MBh : MA;
      v.dpos = P.dpos2 + h;
      v.dplane = h ? P.dplane2 : R.dplane;
      const size_t off = h ? (size_t)(pmb_ / 2) : 0;
      v.xq = P.xq + off * kmax;
      v.xd = P.xd + off * kmax / 32;
      v.xs = P.xs + off * kmax / 16;
    }
    upload_rope(ri, M);
    ph_ = "P.";
    if (prof::on()) { prof::eager(true); mk(ri, "start"); }
    get_rows_iq2_s(R.sh->embd, m_.tok_embd_row_bytes, P.tok, M, P.x, E, R.s, img_.empty() ? nullptr : img_[ri]);
    for (int h = 0; h < 2; h++)
      rmsnorm_q8(P.x + (size_t)hv[ri][h].r0 * E, sh.layers[0].attn_norm, P.h + (size_t)hv[ri][h].r0 * E, hv[ri][h].xq, hv[ri][h].xd, E,
                 hv[ri][h].Mh, hp.eps, R.s, hv[ri][h].xs);
  }
  auto each = [&](auto fn) {
    for (int ri = 0; ri < nr; ri++) {
      CK(cudaSetDevice(r_[ri].sh->dev));
      fn(ri);
      cudaStreamQuery(r_[ri].s);
    }
  };
  each([&](int ri) {
    pf2_part(ri, 0, hv[ri][0]); pf2_send(ri, hv[ri][0], 0);
    pf2_part(ri, 0, hv[ri][1]); pf2_send(ri, hv[ri][1], 0);
  });
  for (int p = 0; p < kPfParts; p++) {
    each([&](int ri) {
      for (int h = 0; h < 2; h++) {
        const PfHalf& v = hv[ri][h];
        pf2_norm(ri, p, v);
        if (p == 2 * hp.n_layer - 1) {  // output norm done: logits of this half's rows
          Rank& R = r_[ri];
          PRank& P = pr_[ri];
          const Shard& sh = *R.sh;
          if (last && h == 1) qgemv(sh.output, v.xq + (size_t)(v.Mh - 1) * E, v.xd + (size_t)(v.Mh - 1) * (E / 32), R.logits, 1, R.s);
          if (all_logits) {
            if (!P.plog) { CK(cudaMalloc(&P.plog, sizeof(float) * (size_t)pmb_ * sh.vocab_n)); R.allocs.push_back(P.plog); }
            qgemm(sh.output, v.xq, v.xd, v.xs, P.plog + (size_t)v.r0 * sh.vocab_n, v.Mh, R.s);
          }
        }
        if (p + 1 < kPfParts) { pf2_part(ri, p + 1, v); pf2_send(ri, v, p + 1); }
      }
    });
  }
  for (int ri = 0; ri < nr; ri++) {
    Rank& R = r_[ri];
    PRank& P = pr_[ri];
    CK(cudaSetDevice(R.sh->dev));
    CK(cudaMemcpyAsync(R.pend_h, P.hfin + (size_t)(M - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, R.s));
    flip_plane_to(R.dplane, P.dplane2, R.s);
  }
  for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); CK(cudaStreamSynchronize(R.s)); }
  for (auto& P : pr_) CK(cudaStreamSynchronize(P.sx));
  if (prof::on())
    for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); prof::collect(prof::eager_base(), prof::next_slot()); prof::eager(false); }
  ph_ = "";
  hpos_ += M;
}

// ---------------------------------------------------------------- prompt cache support
namespace {
// dst = plane *plane of planes (plane_elems floats each)
__global__ void copy_plane_kernel(const float* __restrict__ planes, const int* __restrict__ plane, size_t plane_elems,
                                  float* __restrict__ dst) {
  const float* src = planes + (size_t)(*plane) * plane_elems;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < plane_elems; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}
}  // namespace

// Layout of a saved state (per card): for each GDN layer conv state then SSM state, then the MTP input hidden.
size_t Decoder::state_bytes() const {
  const Hparams& hp = m_.hp();
  const Shard& sh = *r_[0].sh;
  const size_t per_layer = (size_t)sh.conv_channels() * (hp.conv_k - 1) + (size_t)sh.vh * hp.ssm_dim * hp.ssm_dim;
  return (r_[0].conv_st.size() * per_layer + hp.n_embd) * sizeof(float);
}

void Decoder::state_save(const std::vector<uint8_t*>& host) {
  if (host.size() != r_.size()) throw std::runtime_error("state_save: one buffer per card");
  const Hparams& hp = m_.hp();
  const size_t sb = state_bytes();
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    CK(cudaSetDevice(R.sh->dev));
    const size_t conv_n = (size_t)R.sh->conv_channels() * (hp.conv_k - 1), ssm_n = (size_t)R.sh->vh * hp.ssm_dim * hp.ssm_dim;
    CK(cudaStreamWaitEvent(R.s, ev_saved_[ri], 0));  // the previous save has left the staging buffer
    float* p = stage_[ri];
    for (size_t ig = 0; ig < R.conv_st.size(); ig++) {
      copy_plane_kernel<<<64, 256, 0, R.s>>>(R.conv_st[ig], R.dplane, conv_n, p);
      copy_plane_kernel<<<256, 256, 0, R.s>>>(R.ssm_st[ig], R.dplane, ssm_n, p + conv_n);
      p += conv_n + ssm_n;
    }
    CK(cudaMemcpyAsync(p, R.pend_h, sizeof(float) * hp.n_embd, cudaMemcpyDeviceToDevice, R.s));
    CK(cudaGetLastError());
    CK(cudaEventRecord(ev_staged_[ri], R.s));
    CK(cudaStreamWaitEvent(ss_[ri], ev_staged_[ri], 0));
    CK(cudaMemcpyAsync(host[ri], stage_[ri], sb, cudaMemcpyDeviceToHost, ss_[ri]));
    CK(cudaEventRecord(ev_saved_[ri], ss_[ri]));
    if (stage_of_.size() != r_.size()) stage_of_.assign(r_.size(), nullptr);
    stage_of_[ri] = host[ri];
  }
}

void Decoder::state_wait() {
  for (int ri = 0; ri < (int)r_.size(); ri++) { CK(cudaSetDevice(r_[ri].sh->dev)); CK(cudaStreamSynchronize(ss_[ri])); }
}

void Decoder::state_load(const std::vector<uint8_t*>& host, int pos) {
  if (host.size() != r_.size()) throw std::runtime_error("state_load: one buffer per card");
  if (pos < 1 || pos + 2 * kMaxT > n_ctx_) throw std::runtime_error("state_load: bad position");
  if (!warm_) warmup();
  state_wait();
  const Hparams& hp = m_.hp();
  for (int ri = 0; ri < (int)r_.size(); ri++) {
    Rank& R = r_[ri];
    CK(cudaSetDevice(R.sh->dev));
    const size_t conv_n = (size_t)R.sh->conv_channels() * (hp.conv_k - 1), ssm_n = (size_t)R.sh->vh * hp.ssm_dim * hp.ssm_dim;
    // The last saved state is still in the VRAM staging buffer (the usual case in an agent loop: the next request
    // restarts at the last checkpoint): copy from there instead of from host memory (about 1 ms instead of 23 ms).
    // Q27_STAGE_HIT=0 always reads host memory.
    static const bool hit_on = [] { const char* e = getenv("Q27_STAGE_HIT"); return !(e && e[0] == '0'); }();
    const bool hit = hit_on && ri < (int)stage_of_.size() && stage_of_[ri] == host[ri];
    const float* p = hit ? stage_[ri] : (const float*)host[ri];
    const cudaMemcpyKind kind = hit ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    for (size_t ig = 0; ig < R.conv_st.size(); ig++) {
      CK(cudaMemcpyAsync(R.conv_st[ig], p, conv_n * sizeof(float), kind, R.s));
      CK(cudaMemcpyAsync(R.ssm_st[ig], p + conv_n, ssm_n * sizeof(float), kind, R.s));
      p += conv_n + ssm_n;
    }
    CK(cudaMemcpyAsync(R.pend_h, p, sizeof(float) * hp.n_embd, kind, R.s));
    set_int(R.dplane, 0, R.s);
  }
  for (auto& R : r_) { CK(cudaSetDevice(R.sh->dev)); CK(cudaStreamSynchronize(R.s)); }
  hpos_ = pos;
  last_n_ = 0;
}

void Decoder::keep_from_last_step(int k) {
  if (k < 1 || k > last_n_) throw std::runtime_error("keep_from_last_step: bad count");
  if (k == last_n_) return;
  // last_n_ > 1 only after a verify pass: the state after verify token k-1 (the old pending token, then the
  // accepted drafts) is in plane k-1, its final hidden in hfin[k-1].
  const int E = m_.hp().n_embd;
  for (auto& R : r_) {
    CK(cudaSetDevice(R.sh->dev));
    set_int(R.dplane, k - 1, R.s);
    CK(cudaMemcpyAsync(R.pend_h, R.hfin + (size_t)(k - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, R.s));
    CK(cudaStreamSynchronize(R.s));
  }
  hpos_ -= last_n_ - k;
  last_n_ = k;
}

// Host layout per card: for each cache (K and V of each attention layer, then MTP K and V) the int8 or f16 values
// [kvh][p1-p0][row], then (q8_0) the scales [kvh][p1-p0][16 bytes].
void Decoder::kv_copy(int ri, uint8_t* host, int p0, int p1, bool to_host) {
  Rank& R = r_[ri];
  const int kvh = R.sh->kvh, n = p1 - p0;
  const size_t row = kv_q8_ ? 256 : 512;
  std::vector<void*> caches;
  for (size_t ia = 0; ia < R.kc.size(); ia++) { caches.push_back(R.kc[ia]); caches.push_back(R.vc[ia]); }
  caches.push_back(R.mkc);
  caches.push_back(R.mvc);
  uint8_t* h = host;
  auto copy = [&](uint8_t* dev, size_t rb) {  // rows [p0, p1) of every head; dev = start of head 0, row bytes rb
    uint8_t* d = dev + (size_t)p0 * rb;
    if (to_host) CK(cudaMemcpy2DAsync(h, n * rb, d, (size_t)n_ctx_ * rb, n * rb, kvh, cudaMemcpyDeviceToHost, R.s));
    else CK(cudaMemcpy2DAsync(d, (size_t)n_ctx_ * rb, h, n * rb, n * rb, kvh, cudaMemcpyHostToDevice, R.s));
    h += (size_t)kvh * n * rb;
  };
  for (void* c : caches) {
    copy((uint8_t*)c, row);
    if (kv_q8_) copy((uint8_t*)c + (size_t)kvh * n_ctx_ * 256, 16);
  }
  CK(cudaStreamSynchronize(R.s));
}

// One thread per card: copies from pageable host memory are staged by the driver and block the calling thread,
// so the two cards' links are used at the same time only from two threads.
void Decoder::kv_to_host(const std::vector<uint8_t*>& host, int p0, int p1) {
  if (p0 < 0 || p1 > n_ctx_ || p0 >= p1) return;
  std::vector<std::thread> th;
  std::vector<std::string> err(r_.size());
  for (int ri = 0; ri < (int)r_.size(); ri++)
    th.emplace_back([&, ri] {
      try {
        CK(cudaSetDevice(r_[ri].sh->dev));
        kv_copy(ri, host[ri], p0, p1, true);
      } catch (const std::exception& e) {
        err[ri] = e.what();
      }
    });
  for (auto& t : th) t.join();
  for (auto& e : err)
    if (!e.empty()) throw std::runtime_error(e);
}

void Decoder::kv_from_host(const std::vector<const uint8_t*>& host, int p0, int p1) {
  if (p0 < 0 || p1 > n_ctx_ || p0 >= p1) return;
  std::vector<std::thread> th;
  std::vector<std::string> err(r_.size());
  for (int ri = 0; ri < (int)r_.size(); ri++)
    th.emplace_back([&, ri] {
      try {
        CK(cudaSetDevice(r_[ri].sh->dev));
        kv_copy(ri, const_cast<uint8_t*>(host[ri]), p0, p1, false);
      } catch (const std::exception& e) {
        err[ri] = e.what();
      }
    });
  for (auto& t : th) t.join();
  for (auto& e : err)
    if (!e.empty()) throw std::runtime_error(e);
}

}  // namespace q27
