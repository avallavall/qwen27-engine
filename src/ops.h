// Small kernels of the forward pass (everything except the quantized GEMV), for T = 1..4 tokens.
// Numerics follow llama.cpp's CUDA kernels where the order of float operations matters.
// Layouts: per-token rows are contiguous, [T][n].
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace q27 {

// y = x * rsqrt(mean(x^2) + eps) * w, for `rows` rows of n values (w has n values).
// Output rows are ystride values apart (0 = n).
void rmsnorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s, int ystride = 0);
// y = x + a (n values)
void add(const float* x, const float* a, float* y, int n, cudaStream_t s);
// y = silu(g) * u
void swiglu(const float* g, const float* u, float* y, int n, cudaStream_t s);
// Embedding rows of an IQ2_S table (raw GGUF blocks, device-accessible) -> f32 [T][n].
// The row ids are read from device memory, so the call can live in a CUDA graph.
// An id < 0 selects row (-id - 1) of img (f32 [rows][n], image embeddings) instead.
void get_rows_iq2_s(const uint8_t* table, int64_t row_bytes, const int* ids, int T, float* y, int n, cudaStream_t s,
                    const float* img = nullptr);
// ya[t] = Wa x[t], yb[t] = Wb x[t] for two BF16 matrices of N rows (ssm_alpha and ssm_beta), f32 x [T][K].
void gemv_bf16_pair(const __nv_bfloat16* Wa, const __nv_bfloat16* Wb, const float* x, float* ya, float* yb, int N, int K,
                    int T, cudaStream_t s);

// ---- Gated DeltaNet. States have 4 planes (state after each of up to 4 tokens of the last pass);
// *in_plane selects the input plane. With snapshots, the state after token t goes to plane t;
// without, only the final state goes to plane T-1.
// beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias), for T*H values ([T][H]).
void gdn_gates(const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g, float* beta,
               int H, int T, cudaStream_t s);
// Causal conv (4 taps) + SiLU over C channels and T tokens. planes: [4][C][3] (oldest first).
void gdn_conv(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int T, bool snapshots,
              cudaStream_t s);
// x <- rms_norm(x, eps/n) * (1/sqrt(n)) per head of n values; `heads` heads per token at the start of
// each token row of `stride` values.
void l2norm_heads(float* x, int n, int heads, int T, int stride, float eps, cudaStream_t s);
// Delta rule over T tokens for H value heads. q, k, v point into the conv output ([T][stride]).
// planes: [4][H][128 value][128 key] (transposed like llama.cpp). V head h uses K head h % HK.
// o[t][h][128] = (S^T q) * scale.
// flip (prefill, any T): no snapshots; the final state goes to plane (*in_plane == 0 ? 1 : 0).
void gdn_step(const float* q, const float* k, const float* v, int stride, const float* g, const float* beta, float* planes,
              const int* in_plane, float* o, int H, int HK, int T, float scale, bool snapshots, cudaStream_t s, bool flip = false);
// y_h = (rms_norm_128(o_h) * w) * silu(z_h), over `heads` heads (T*H).
void gated_rmsnorm(const float* o, const float* w, const float* z, float* y, int n, int heads, float eps, cudaStream_t s);
void set_int(int* p, int v, cudaStream_t s);

// ---- Attention, T query tokens at positions *pos0 + t.
// KV cache of one layer: f16 [nkvh][n_ctx][256], or q8_0 (q8 = true): int8 values [nkvh][n_ctx][256] followed by
// f16 scales [nkvh][n_ctx][8] (one per 32 values), same rounding as llama.cpp quantize_f32_q8_0_block.
size_t kv_cache_bytes(int nkvh, int n_ctx, bool q8);  // bytes of the K (or V) cache of one layer
// Per-head RMSNorm of q (nqh heads at stride 512 inside q|gate) and k (nkvh heads), partial NeoX RoPE on the
// first 64 dims, K and V written at cache rows pos0+t. qn [T][nqh][256].
// RoPE positions (IMRoPE, llama.cpp sections [11,11,10,0]: rotary pair j uses t, h, w for j % 3 = 0, 1, 2):
// rope3 [T][3] (t, h, w) when not null (image rows), else p = pos0 + t - *rdelta for all three (rdelta null = 0;
// the delta is cache rows minus positions after images).
void attn_prep(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, float* qn,
               void* kcache, void* vcache, const int* pos0, int T, float eps, float theta_scale, int nqh, int nkvh,
               int n_ctx, bool q8, cudaStream_t s, const void* pf = nullptr, size_t pf_bytes = 0,
               const int* rope3 = nullptr, const int* rdelta = nullptr);
// o[t][h] = softmax((q . K) * scale) V over positions 0..pos0+t, KV head h / 6; then o *= sigmoid(gate).
// Tensor-core split-KV kernel (src/attn.cu); T = 1..4. With xq != null the result goes to q8_1 (xq, xd)
// instead of o.
void attn_decode(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, int8_t* xq, float* xd,
                 const int* pos0, int T, float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s);
// Prefill: M query tokens at positions *pos0 .. *pos0+M-1 (their K/V already in the cache), causal.
// o [M][nqh][256] f32 (gate applied). P V accumulates in f16 with llama.cpp's max offset (as its prefill FA).
// depth_hint (host copy of *pos0, or -1): lets the int8 kernel split long position ranges over more CTAs.
void attn_prefill(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, const int* pos0, int M,
                  float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s, int depth_hint = -1);
// Same result with a simple slow kernel (tests only).
void attn_decode_ref(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, const int* pos0,
                     int T, float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s);
// Chunks of positions per KV head used by attn_decode (grid = chunks x nkvh).
int attn_chunks(int nkvh);

// ---- fused kernels that end with the q8_1 activation quantization of the next GEMV input
// Cross-card exchange arguments (see ar_add). token = (*dstep) * n_ar + index + 1; flags per row (stride 32).
struct ArArgs {
  __nv_bfloat16* host_mine = nullptr;
  const __nv_bfloat16* host_other = nullptr;
  int* flag_mine = nullptr;
  const int* flag_other = nullptr;
  const int* dstep = nullptr;
  int n_ar = 0, index = 0;
  int* err = nullptr;
  // q8 wire only: error feedback. ef [rows][n] keeps this card's rounding error of the last sum of the pass and is
  // added to the next partial before rounding, so the error in x does not grow over the layers. ef_first: first sum
  // of a pass (ef not read).
  float* ef = nullptr;
  int ef_first = 1;
  // Q27_SUMPROF=1 only: phase times of the q8 wire path, accumulated per exchange index, [n_ar][8] u64:
  // 0 quantize + write own partial + fence, 1 flag + wait for the peer, 2 read peer, 3 add + norm, 4 calls, 5 total.
  unsigned long long* tprof = nullptr;
};
// RMSNorm of `rows` rows (n = 5120) -> h (f32, may be null) and q8_1 (xq, xd).
void rmsnorm_q8(const float* x, const float* w, float* h, int8_t* xq, float* xd, int n, int rows, float eps, cudaStream_t s,
                float* xs = nullptr);
// x += partial (ar != null: summed over both cards), then RMSNorm(x) * w -> h (may be null) and q8_1. n = 5120.
// pf, pf_bytes (optional): device range (the next GEMV's weights) to prefetch into L2 while the kernel waits.
void sum_norm_q8(float* x, const float* partial, int n, int rows, const ArArgs* ar, const float* w, float eps, float* h,
                 int8_t* xq, float* xd, cudaStream_t s, const void* pf = nullptr, size_t pf_bytes = 0);
// silu(g) * u -> q8_1 (n values in total, rows contiguous).
void swiglu_q8(const float* g, const float* u, int8_t* xq, float* xd, int n, cudaStream_t s, float* xs = nullptr);
// (rms_norm_128(o_h) * w) * silu(z_h) -> q8_1, over `heads` heads (T*H).
void gated_rmsnorm_q8(const float* o, const float* w, const float* z, int8_t* xq, float* xd, int heads, float eps, cudaStream_t s,
                      float* xs = nullptr);
// gdn_conv + l2norm_heads of the first n_norm_heads heads (q and k) + gdn_gates, in one kernel.
// pf, pf_bytes (optional): weights of a later GEMV to prefetch into L2.
void gdn_conv_l2(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int T, bool snapshots,
                 int n_norm_heads, float eps, const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g,
                 float* beta, int H, cudaStream_t s, const void* pf = nullptr, size_t pf_bytes = 0);

// ---- two cards
// Wire format of the cross-card sums (Q27_WIRE): q8b16 (default: int8 + fp16 scale per 16 values, with error
// feedback, 56% of the bf16 bytes), q8 (scale per 32; one 32k test position failed), or bf16 (as llama.cpp).
bool wire_q8();
int wire_qb();  // values per scale on the q8 wire: 32 (Q27_WIRE=q8), 16 (q8b16), 0 = bf16 wire
// x += sum over both cards of `partial` (n floats), through mapped pinned host memory. Each card calls it
// with its own slot and flags; flags are per block (stride 32 ints). token = (*dstep) * n_ar + index + 1.
// err is incremented if the peer never arrives.
void ar_add(float* x, const float* partial, int n, __nv_bfloat16* host_mine, const __nv_bfloat16* host_other, int* flag_mine,
            const int* flag_other, const int* dstep, int n_ar, int index, int* err, cudaStream_t s);
void inc_counter(int* p, cudaStream_t s);

}  // namespace q27
