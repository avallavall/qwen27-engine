// Qwen3.8-27B (GGUF arch qwen35) weights on one or two GPUs (tensor parallel), and the
// single-token decode pass.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "qmat.h"
#include "sampling.h"

namespace q27 {

struct Hparams {
  int n_embd = 5120, n_ff = 17408, n_layer = 64, vocab = 248320;
  int n_head = 24, n_head_kv = 4, head_dim = 256;
  int ssm_k_heads = 16, ssm_v_heads = 48, ssm_dim = 128, conv_k = 4;
  int full_attn_interval = 4;
  int rope_dims = 64;
  float rope_base = 1e7f;
  float eps = 1e-6f;
  int ctx_train = 262144;
  bool is_attn(int il) const { return (il + 1) % full_attn_interval == 0; }
};

// Weights of one layer on one card (local slice for tensor parallel).
struct Layer {
  bool attn = false;
  float* attn_norm = nullptr;
  float* post_norm = nullptr;
  QMat ffn_gate, ffn_up, ffn_down;
  // Gated DeltaNet
  QMat qkv, gate, ssm_out;
  __nv_bfloat16* alpha = nullptr;  // [vh][5120]
  __nv_bfloat16* beta = nullptr;   // [vh][5120]
  float* ssm_a = nullptr;          // [vh]
  float* dt_bias = nullptr;        // [vh]
  float* conv_w = nullptr;         // [channels][4]
  float* ssm_norm = nullptr;       // [128]
  // full attention
  QMat wq, wk, wv, wo;
  float* q_norm = nullptr;  // [256]
  float* k_norm = nullptr;  // [256]
};

// Everything one card holds. With tp = 2, card r keeps GDN K heads [8r, 8r+8) and the 24 V heads that
// read them, attention KV heads [2r, 2r+2) with their 12 query heads, FFN rows [8704r, 8704r+8704),
// and vocab rows [124160r, 124160r+124160) of the output layer. Row-parallel matrices (ssm_out,
// attn_output, ffn_down) keep the matching column blocks.
struct Shard {
  int dev = 0, rank = 0;
  int kh = 16, vh = 48;    // GDN K / V heads
  int qh = 24, kvh = 4;    // attention q / kv heads
  int ff = 17408;
  int vocab0 = 0, vocab_n = 248320;
  int conv_channels() const { return 2 * kh * 128 + vh * 128; }
  std::vector<Layer> layers;
  float* output_norm = nullptr;
  QMat output;
  // MTP block (blk.64): attention + FFN split like a target attention layer; eh_proj whole on every card.
  Layer mtp;
  QMat eh_proj;
  float* enorm = nullptr;
  float* hnorm = nullptr;
  float* shared_head_norm = nullptr;
  // Draft vocabulary (optional): the output rows of a token subset (this card's part), used by the MTP draft passes
  // only. draft_ids [draft_n] (device) = token id of each row.
  QMat draft_out;
  int* draft_ids = nullptr;
  int draft_n = 0;
  size_t vram_bytes = 0;
  std::vector<void*> allocs;
};

class Model {
 public:
  Model(const std::string& path, const std::vector<int>& devices);
  ~Model();
  const Hparams& hp() const { return hp_; }
  int tp() const { return (int)shards.size(); }
  // Draft vocabulary: the MTP drafts score only these tokens (the target still scores all of them, so the output
  // distribution does not change; drafts outside the subset are just never proposed). Call before a Decoder runs
  // (its graphs capture the choice). Empty = full vocabulary. Each card's part is padded to a multiple of 128 rows.
  void set_draft_vocab(std::vector<int> ids);
  // Reads int32 token ids from a file (bench\make_draft_vocab.py) and keeps the first n (0 = all).
  void set_draft_vocab_file(const std::string& path, int n = 0);

  Hparams hp_;
  std::unique_ptr<GGUF> g_;
  std::vector<Shard> shards;
  uint8_t* tok_embd = nullptr;  // raw IQ2_S rows in pinned, mapped host memory (read by both cards)
  int64_t tok_embd_row_bytes = 0;

 private:
  void load_shard(Shard& sh, int tp);
};

// Decode state for one sequence: KV caches and GDN states on each card, scratch, CUDA graphs per card.
class Decoder {
 public:
  static constexpr int kMaxT = 4;
  static constexpr int kDrafts = 3;
  // kv_q8: KV cache in q8_0 instead of f16 (option, changes numerics slightly).
  Decoder(const Model& m, int n_ctx, bool kv_q8 = false);
  ~Decoder();
  void reset();

  // ---- plain passes (no MTP), used by the logit test
  // Run T (1 or 4) tokens at positions pos .. pos+T-1 (pos = number of tokens already in the state).
  void step(const int* tokens, int T, int pos);
  void step(int token, int pos) { step(&token, 1, pos); }
  // Copy the logits of token t of the last pass (all vocab slices) to host memory.
  void logits_to_host(float* out, int t = 0);
  bool use_graph = true;

  // ---- sequence API (speculative generation with the MTP head)
  // Read n tokens at positions position() .. position()+n-1 into the target and MTP caches. The last
  // passes have one token, so afterwards the logits of the last token are in row 0.
  void feed(const int* tokens, int n);
  // Sample the token after the fed ones (it goes to position(), not yet in the caches) and the first
  // drafts. Returns that token.
  // rng_start: first value of the random-number counter (a new value per request gives new random draws
  // without recapturing the graphs, which have the seed baked in).
  int start(const SampleParams& sp, int rng_start = 0);
  // reset + prefill(prompt) + start.
  int begin(const std::vector<int>& prompt, const SampleParams& sp);
  // One speculative step (verify 4, accept, MTP catch-up, 3 drafts). Writes the emitted tokens
  // (1..4) to out and returns how many. The last emitted token is pending (not in the caches yet).
  int spec_step(int* out);
  int n_ctx() const { return n_ctx_; }
  // KV cache bytes per token on each card (16 attention layers + MTP layer, K and V).
  static size_t kv_bytes_per_token(const Model& m, bool kv_q8 = false);
  // Device memory the decoder takes on each card apart from the KV caches (estimate, bytes).
  static size_t fixed_bytes(const Model& m);
  // Largest context (multiple of 256, at most the trained context) that leaves reserve[r] bytes free
  // on card r. Reads the free memory now.
  static int fit_ctx(const Model& m, const std::vector<size_t>& reserve, bool kv_q8 = false);
  bool kv_q8() const { return kv_q8_; }
  // KV cache type from the environment: q8_0 by default (user decision 2026-10-06), Q27_KV=f16 selects f16.
  static bool kv_q8_from_env();
  int position() const;  // tokens in the target caches

  // ---- batched prompt reading (prefill.cu kernels, int8 tensor-core GEMMs)
  // Same contract as feed(): read n tokens at position() .. in passes of up to max_batch() tokens (the last
  // token's logits end in row 0). Short inputs (< kPrefillMin tokens) go through feed().
  // Image rows: a token id < 0 selects row (-id - 1) of the image embeddings set with set_image. rope3 (host,
  // [n][3], may be null): IMRoPE positions (t, h, w) of every row; null means position = cache row - rope_delta().
  void prefill(const int* tokens, int n, const int* rope3 = nullptr);
  // Image embeddings (f32 [rows][n_embd], device memory of card src_dev) for the next prefill calls.
  void set_image(const float* src, int rows, int src_dev);
  // Cache rows minus RoPE positions for text rows (an image of nx * ny rows advances the position by max(nx, ny)).
  // Decode passes and prefill rows without rope3 use position = cache row - delta.
  void set_rope_delta(int delta);
  int rope_delta() const { return rope_delta_; }
  static constexpr int kPrefillMin = 16;
  int max_batch() const { return pmb_; }
  // Test hook: read M tokens (M <= max_batch) as one prefill pass and keep the logits of every row.
  // Then prefill_logits_to_host(out, t) copies the logits of row t.
  void prefill_logits(const int* tokens, int M);
  void prefill_logits_to_host(float* out, int t);

  // ---- prompt cache support (server)
  // The recurrent part of the sequence state (GDN conv and SSM states of the current plane, and the MTP input
  // hidden) cannot be cut back, so the server keeps copies of it (checkpoints) in host memory. KV rows below a
  // position stay valid when the sequence is cut back to it.
  size_t state_bytes() const;  // per card
  int n_cards() const { return (int)r_.size(); }
  // Start a copy of the state at position() into host[card] (pinned, state_bytes() each): a device copy into a
  // staging buffer on the compute stream, then a device-to-host copy on a side stream. Passes may run meanwhile.
  // state_wait() waits for the copy.
  void state_save(const std::vector<uint8_t*>& host);
  void state_wait();
  // Set position() = pos and load the state saved at pos. The KV rows below pos must hold the same tokens.
  void state_load(const std::vector<uint8_t*>& host, int pos);
  // After spec_step() returned n tokens: keep only the first k (1 <= k <= n); token k becomes the pending one.
  // Uses the per-token GDN snapshots of the last verify pass, so call it before any other pass.
  void keep_from_last_step(int k);
  // KV rows [p0, p1) of all attention layers and the MTP layer to / from host memory. Per card
  // kv_bytes_per_token(m, kv_q8()) * (p1 - p0) bytes. Synchronous.
  void kv_to_host(const std::vector<uint8_t*>& host, int p0, int p1);
  void kv_from_host(const std::vector<const uint8_t*>& host, int p0, int p1);

 private:
  // Device state of the speculative loop (one copy per card; both cards keep it identical).
  struct DState {
    int P;          // position of s (first token of the next verify)
    int s;          // last emitted token, not yet in the target caches
    int d[kDrafts]; // drafts
    int n_emit;     // tokens emitted by the last step
    int emit[4];
    int counter;    // RNG counter
  };
  enum Kind { kT1 = 0, kT4, kM1, kM4, kFirst, kStep, kKinds };
  struct Rank {
    const Shard* sh = nullptr;
    cudaStream_t s = nullptr;
    std::vector<void*> allocs;
    std::vector<void*> kc, vc;            // per attention layer, see kv_cache_bytes (ops.h)
    void *mkc = nullptr, *mvc = nullptr;  // MTP KV cache
    std::vector<float*> conv_st, ssm_st;  // per GDN layer, 4 planes each
    float *x, *h, *m_out, *qkv, *z, *ab, *g, *beta, *conv, *o, *y, *qg, *k, *v, *qn, *fg, *fu, *act, *logits;
    float *hfin, *hrows, *mtp_h, *pend_h, *cat;
    int8_t* xq;
    float* xd;
    int* dtok;         // device: tokens [kMaxT]
    int* dpos;         // device: position of the first token
    int* dplane;       // device: GDN state plane to read
    int* ddelta;       // device: rope delta (cache row - position)
    int* hin;          // pinned host staging: tokens [kMaxT], pos
    int* dstep;        // device: exchanges counter (one per graph)
    DState* st;        // device: speculative state
    CandRow* pc;       // device: target candidates [4]
    CandRow* qc;       // device: draft candidates [kDrafts]
    cudaGraphExec_t graph[kKinds] = {};
  };
  // Prefill buffers per card, for up to pmb_ tokens.
  struct PRank {
    int* tok = nullptr;    // device [pmb]
    int* htok = nullptr;   // pinned staging [pmb + 1] (tokens, then the position)
    float *x, *h, *hfin, *m_out, *hrows, *qkv, *z, *ab, *g, *beta, *conv, *o, *fg, *fu, *qg, *k, *v, *qn, *cat;
    int8_t* xq;
    float *xd, *xs;
    float* plog = nullptr;  // [pmb][vocab_n] logits of every row (test hook only)
    int token = 0;          // exchange counter (advances the same way on both cards)
    // overlapped exchanges (prefill_batch_ov)
    cudaStream_t sx = nullptr;          // copy-engine stream
    cudaEvent_t ev_part[2] = {}, ev_recv[2] = {};
    __nv_bfloat16 *sendbuf = nullptr, *recvbuf = nullptr;  // [pmb][5120]
    int* dpos2 = nullptr;               // device: positions of the two halves
    int* dplane2 = nullptr;             // device: GDN plane half B reads
    int* rope = nullptr;                // device [pmb][3]: IMRoPE positions of the batch rows (when rope_on_)
    int* hrope = nullptr;               // pinned staging [pmb][3]
    int tok2[2] = {0, 0};               // exchange counters per half
  };
  // One half of a prefill batch: rows r0 .. r0+Mh-1, its positions and GDN plane, its q8_1 input buffers.
  struct PfHalf {
    int half = 0, r0 = 0, Mh = 0;
    int* dpos = nullptr;
    int* dplane = nullptr;
    int8_t* xq = nullptr;
    float *xd = nullptr, *xs = nullptr;
  };
  const Model& m_;
  int n_ctx_;
  bool kv_q8_ = false;
  std::vector<Rank> r_;
  bool warm_ = false;
  SampleParams sp_;
  // shared host memory for the cross-card exchanges
  __nv_bfloat16* ar_data_ = nullptr;  // [2 slots][2 ranks][kArMax]
  int* ar_flags_ = nullptr;           // [2 slots][2 ranks][4 blocks * 32]
  int* ar_err_ = nullptr;
  CandMailbox mb_{};
  int* emit_host_ = nullptr;          // pinned, mapped: n_emit, emit[4] (written by card 0)
  int hpos_ = 0;
  static constexpr int kArMax = kMaxT * 5120;
  // Exchanges per graph: 128 target sums + catch-up (2) + first/step sampling and drafts.
  static constexpr int kNex = 160;

  std::vector<PRank> pr_;
  int pmb_ = 512;
  int last_n_ = 0;                  // tokens emitted by the last start / spec_step (0 after other passes)
  int rope_delta_ = 0;
  const int* batch_rope_ = nullptr;  // host rope3 rows of the current prefill batch (or null)
  std::vector<float*> img_;          // per card: image embeddings [img_cap_][n_embd]
  int img_cap_ = 0;
  void upload_rope(int ri, int M);   // batch_rope_ -> PRank.rope
  const int* rope_dev(int ri, int r0) const;
  std::vector<float*> stage_;       // per card: VRAM staging buffer for state_save (state_bytes)
  std::vector<cudaStream_t> ss_;    // per card: side stream of state_save
  std::vector<cudaEvent_t> ev_staged_, ev_saved_;
  void kv_copy(int ri, uint8_t* host, int p0, int p1, bool to_host);
  __nv_bfloat16* pex_data_ = nullptr;  // [2 slots][2 ranks][pmb][5120]
  int* pex_flags_ = nullptr;           // [2 slots][2 ranks][pmb][32]
  __nv_bfloat16* pex2_data_ = nullptr;  // [2 halves][2 slots][2 ranks][pmb/2][5120]
  int* pex2_flags_ = nullptr;           // [2 halves][2 slots][2 ranks][32]
  std::vector<int> gdn_idx_, attn_idx_; // layer -> GDN state index / KV cache index
  void prefill_alloc();
  void pf2_alloc();
  void pf2_part(int ri, int p, const PfHalf& hv);
  void pf2_send(int ri, const PfHalf& hv);
  void pf2_norm(int ri, int p, const PfHalf& hv);
  void prefill_batch_ov(const int* tokens, int M, bool last, bool all_logits);
  void prefill_batch(const int* tokens, int M, bool last, bool all_logits, bool exchange);
  void pf_layer(int ri, int il, int M, int& ig, int& ia, bool exchange);
  void pf_mtp(int ri, int M, bool exchange);
  void pf_sum(int ri, int M, bool exchange, const float* w, float* h);

  template <class T> T* alloc(Rank& r, size_t n);
  void warmup();
  void run(Kind k);   // launch graph k on every card (capture on first use) and wait
  void enqueue(int ri, Kind k, bool with_sums);
  void forward(int ri, int T, bool snaps, bool set_plane, int& ix, bool with_sums);
  void mtp_forward(int ri, int T, bool logits, int& ix, bool with_sums);
  // Layer parts. Each starts from the q8_1 input (xq, xd) of its normed input and ends with the sum of its
  // output into x and the next RMSNorm (weight next_w, f32 copy to next_h if not null, q8_1 to xq, xd).
  void attn_layer(int ri, const Layer& L, void* kc, void* vc, int T, int& ix, bool with_sums);
  // next: the GEMV that follows the sum; its weights are prefetched into L2 during the cross-card wait.
  void ffn(int ri, const Layer& L, int T, int& ix, bool with_sums, const float* next_w, float* next_h, const QMat* next);
  void sum_norm(int ri, const float* partial, int T, int& ix, bool with_sums, const float* w, float* h, const QMat* next);
  void candidates(int ri, const float* logits, int rows, bool draft, CandRow* out, int& ix, bool with_sums);
  void upload_inputs(Rank& R);
};

}  // namespace q27
