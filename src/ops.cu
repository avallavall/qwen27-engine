// Small kernels of the forward pass, for T = 1..4 tokens. Float order follows llama.cpp where it
// matters (llama.cpp is MIT licensed, see THIRD_PARTY_NOTICES.md).
#include "ops.h"
#include "common.cuh"
#include "quant_tables.h"

#include <cstring>
#include <vector>

namespace q27 {

bool pdl_enabled() {
  static const bool on = [] { const char* e = getenv("Q27_PDL"); return e && e[0] == '1'; }();
  return on;
}

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// Block sum for blockDim.x <= 1024; result valid in all threads.
__device__ float block_sum(float v) {
  __shared__ float sh[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) / 32;
  v = lane < nw ? sh[lane] : 0.f;
  return warp_sum(v);
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// ---------------------------------------------------------------- norms and element-wise
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n, float eps,
                               int ystride) {
  pdl_wait();
  pdl_trigger();
  x += (size_t)blockIdx.x * n;
  y += (size_t)blockIdx.x * ystride;
  float t = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; t += v * v; }
  t = block_sum(t);
  const float scale = rsqrtf(t / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = scale * x[i] * w[i];
}

__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ a, float* __restrict__ y, int n) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + a[i];
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ y, int n) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = silu(g[i]) * u[i];
}

__global__ void set_int_kernel(int* p, int v) {
  pdl_wait();
  pdl_trigger(); *p = v; }

// ---------------------------------------------------------------- fused kernels with q8_1 output
// RMSNorm of one row (block per row, 1024 threads, n = NPT * 1024) -> h (f32, optional) and q8_1.
// Values come in v[] (thread t holds indices t + k*1024). Same float order as rmsnorm_kernel. n is a run-time
// value on purpose: the mean is a run-time division as in llama.cpp (a constant would become a multiply).
template <int NPT>
__device__ __forceinline__ void norm_q8_row(const float (&v)[NPT], const float* __restrict__ w, float eps, int n, size_t row,
                                            float* __restrict__ h, int8_t* __restrict__ xq, float* __restrict__ xd,
                                            float* __restrict__ xs = nullptr) {
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < NPT; k++) t += v[k] * v[k];
  t = block_sum(t);
  const float scale = rsqrtf(t / n + eps);
#pragma unroll
  for (int k = 0; k < NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    const float y = scale * v[k] * w[i];
    if (h) h[row * n + i] = y;
    q8_store(y, xq, xd, row * n + i, xs);
  }
}

template <int NPT>
__global__ void __launch_bounds__(1024) rmsnorm_q8_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                                          float* __restrict__ h, int8_t* __restrict__ xq, float* __restrict__ xd,
                                                          int n, float* __restrict__ xs) {
  pdl_wait();
  pdl_trigger();
  const size_t row = blockIdx.x;
  float v[NPT];
#pragma unroll
  for (int k = 0; k < NPT; k++) v[k] = x[row * n + threadIdx.x + k * 1024];
  norm_q8_row<NPT>(v, w, eps, n, row, h, xq, xd, xs);
}

// x[row] += partial[row] (two cards: the sum of both cards' partials through mapped host memory, BF16 wire
// as llama.cpp), then the RMSNorm of the new x[row] with weight w -> h (optional) and q8_1. Block per row.
template <int NPT>
__global__ void __launch_bounds__(1024) sum_norm_q8_kernel(float* __restrict__ x, const float* __restrict__ partial, ArArgs ar,
                                                           bool exchange, const float* __restrict__ w, float eps,
                                                           float* __restrict__ h, int8_t* __restrict__ xq,
                                                           float* __restrict__ xd, int n, const uint8_t* pf, size_t pf_bytes) {
  pdl_wait();
  pdl_trigger();
  // Bring the next GEMV's weights into L2 while this kernel waits on the link.
  if (pf && threadIdx.x >= 32 && threadIdx.x < 64) l2_prefetch_range(pf, pf_bytes, blockIdx.x * 32 + threadIdx.x - 32, gridDim.x * 32);
  const size_t row = blockIdx.x;
  float* xr = x + row * n;
  const float* pr = partial + row * n;
  float v[NPT];
  if (exchange) {
    const int token = (*ar.dstep) * ar.n_ar + ar.index + 1;
    __nv_bfloat16* mine = ar.host_mine + row * n;
    const __nv_bfloat16* other = ar.host_other + row * n;
    float own[NPT];
#pragma unroll
    for (int k = 0; k < NPT; k++) {
      const __nv_bfloat16 b = __float2bfloat16(pr[threadIdx.x + k * 1024]);
      mine[threadIdx.x + k * 1024] = b;
      own[k] = __bfloat162float(b);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      volatile int* fm = ar.flag_mine + row * 32;
      volatile const int* fo = ar.flag_other + row * 32;
      *fm = token;
      __threadfence_system();
      const long long t0 = clock64();
      while (*fo != token) {
        if (clock64() - t0 > 3000000000LL) { atomicAdd(ar.err, 1); break; }  // about 1 s: the peer is gone
      }
    }
    __syncthreads();
    __threadfence_system();
#pragma unroll
    for (int k = 0; k < NPT; k++) {
      const int i = threadIdx.x + k * 1024;
      const float oth = __bfloat162float(((volatile const __nv_bfloat16*)other)[i]);
      v[k] = xr[i] + (own[k] + oth);
      xr[i] = v[k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < NPT; k++) {
      const int i = threadIdx.x + k * 1024;
      v[k] = xr[i] + pr[i];
      xr[i] = v[k];
    }
  }
  norm_q8_row<NPT>(v, w, eps, n, row, h, xq, xd);
}

// act = silu(g) * u -> q8_1; n % 32 == 0.
__global__ void swiglu_q8_kernel(const float* __restrict__ g, const float* __restrict__ u, int8_t* __restrict__ xq,
                                 float* __restrict__ xd, int n, float* __restrict__ xs) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;  // whole warps, n % 32 == 0
  q8_store(silu(g[i]) * u[i], xq, xd, i, xs);
}

// One block of 128 threads per (token, head): y = (rms_norm(o) * w) * silu(z) -> q8_1 (same order as
// gated_rmsnorm_kernel).
__global__ void gated_rmsnorm_q8_kernel(const float* __restrict__ o, const float* __restrict__ w, const float* __restrict__ z,
                                        int8_t* __restrict__ xq, float* __restrict__ xd, float eps, float* __restrict__ xs) {
  pdl_wait();
  pdl_trigger();
  const size_t off = (size_t)blockIdx.x * 128;
  const float v = o[off + threadIdx.x];
  const float t = block_sum(v * v);
  const float scale = rsqrtf(t / 128 + eps);
  q8_store((scale * v * w[threadIdx.x]) * silu(z[off + threadIdx.x]), xq, xd, off + threadIdx.x, xs);
}

// GDN input stage: causal conv (4 taps) + SiLU per channel over T tokens, then L2 norm of the q and k heads,
// plus the gates. Block = one head of 128 channels (q heads, k heads, v heads in conv-channel order).
// Same float order as gdn_conv_kernel, l2norm_heads_kernel and gdn_gates_kernel.
__global__ void gdn_conv_l2_kernel(const float* __restrict__ x, const float* __restrict__ cw, float* __restrict__ planes,
                                   const int* __restrict__ in_plane, float* __restrict__ y, int C, int T, bool snapshots,
                                   int n_norm_heads, float eps, const float* __restrict__ a, const float* __restrict__ b,
                                   const float* __restrict__ ssm_a, const float* __restrict__ dt_bias, float* __restrict__ g,
                                   float* __restrict__ beta, int H, const void* pf, size_t pf_bytes) {
  pdl_wait();
  pdl_trigger();
  if (pf && threadIdx.x < 32) l2_prefetch_range(pf, pf_bytes, blockIdx.x * 32 + threadIdx.x, gridDim.x * 32);
  __shared__ float sy[4][128];
  __shared__ float sscale[4];
  const int head = blockIdx.x, tid = threadIdx.x;
  const int c = head * 128 + tid;
  const float* st = planes + ((size_t)(*in_plane) * C + c) * 3;
  float s0 = st[0], s1 = st[1], s2 = st[2];
  const float w0 = cw[4 * c], w1 = cw[4 * c + 1], w2 = cw[4 * c + 2], w3 = cw[4 * c + 3];
  float out[4];
  for (int t = 0; t < T; t++) {
    const float xn = x[(size_t)t * C + c];
    float sum = 0.f;
    sum += s0 * w0;
    sum += s1 * w1;
    sum += s2 * w2;
    sum += xn * w3;
    out[t] = silu(sum);
    s0 = s1; s1 = s2; s2 = xn;
    if (snapshots || t == T - 1) {
      float* o = planes + ((size_t)t * C + c) * 3;
      o[0] = s0; o[1] = s1; o[2] = s2;
    }
  }
  if (head < n_norm_heads) {
    for (int t = 0; t < T; t++) sy[t][tid] = out[t];
    __syncthreads();
    if (tid < 32 * T) {
      const int t = tid >> 5, lane = tid & 31;
      float acc = 0.f;
      for (int i = lane; i < 128; i += 32) acc += sy[t][i] * sy[t][i];
      acc = warp_sum(acc);
      if (lane == 0) sscale[t] = rsqrtf(acc / 128 + eps / 128);
    }
    __syncthreads();
    const float post = 1.0f / sqrtf(128.f);
    for (int t = 0; t < T; t++) out[t] = (out[t] * sscale[t]) * post;
  } else if (tid < T) {
    const int hv = head - n_norm_heads;  // v head: gates of (token tid, head hv)
    const int i = tid * H + hv;
    beta[i] = sigmoid(b[i]);
    g[i] = softplus(a[i] + dt_bias[hv]) * ssm_a[hv];
  }
  for (int t = 0; t < T; t++) y[(size_t)t * C + c] = out[t];
}

// ---------------------------------------------------------------- embedding (IQ2_S rows)
// Grid (super-blocks, T), 32 threads per 256-weight super-block; thread t handles 8 weights.
// Same float order as ggml dequantize_row_iq2_s: db = d * (0.5 + s) * 0.25, y = db * grid * sign.
__global__ void get_rows_iq2_s_kernel(const uint8_t* __restrict__ table, int64_t row_bytes, const int* __restrict__ ids,
                                      const uint64_t* __restrict__ grid, float* __restrict__ y, int n,
                                      const float* __restrict__ img) {
  pdl_wait();
  pdl_trigger();
  const int tok = blockIdx.y;
  const int id = ids[tok];
  const int t = threadIdx.x;      // 0..31
  const int ib32 = t / 4, l = t % 4;
  if (id < 0) {  // image embedding row
    const size_t o = (size_t)blockIdx.x * 256 + 32 * ib32 + 8 * l;
    const float* src = img + (size_t)(-id - 1) * n + o;
    float* out = y + (size_t)tok * n + o;
#pragma unroll
    for (int j = 0; j < 8; j++) out[j] = src[j];
    return;
  }
  const uint8_t* b = table + (size_t)id * row_bytes + (size_t)blockIdx.x * 82;
  const float d = __half2float(*(const __half*)b);
  const uint8_t* qs = b + 2;
  const uint8_t* signs = b + 2 + 32;
  const uint8_t qh = b[66 + ib32];
  const uint8_t sc = b[74 + ib32];
  const float db = d * (0.5f + ((l < 2 ? sc : sc >> 4) & 0xF)) * 0.25f;
  const uint64_t gv = grid[qs[4 * ib32 + l] | ((qh << (8 - 2 * l)) & 0x300)];
  const uint8_t sg = signs[4 * ib32 + l];
  float* out = y + (size_t)tok * n + (size_t)blockIdx.x * 256 + 32 * ib32 + 8 * l;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const float g = (float)((gv >> (8 * j)) & 0xFF);
    out[j] = db * g * ((sg >> j) & 1 ? -1.f : 1.f);
  }
}

// ---------------------------------------------------------------- BF16 GEMV (ssm_alpha, ssm_beta)
// Grid (2N, T): block per output row of two matrices (rows 0..N-1 from Wa, N..2N-1 from Wb), 256 threads over K.
__global__ void gemv_bf16_pair_kernel(const __nv_bfloat16* __restrict__ Wa, const __nv_bfloat16* __restrict__ Wb,
                                      const float* __restrict__ x, float* __restrict__ ya, float* __restrict__ yb, int N, int K) {
  pdl_wait();
  pdl_trigger();
  const int row = blockIdx.x, t = blockIdx.y;
  const __nv_bfloat16* w = row < N ? Wa + (size_t)row * K : Wb + (size_t)(row - N) * K;
  const float* xt = x + (size_t)t * K;
  float acc = 0.f;
  for (int k = threadIdx.x * 2; k < K; k += blockDim.x * 2) {
    const __nv_bfloat162 v = *(const __nv_bfloat162*)(w + k);
    acc += __bfloat162float(v.x) * xt[k] + __bfloat162float(v.y) * xt[k + 1];
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) (row < N ? ya[(size_t)t * N + row] : yb[(size_t)t * N + row - N]) = acc;
}

// ---------------------------------------------------------------- Gated DeltaNet
__global__ void gdn_gates_kernel(const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g, float* beta,
                                 int H, int n) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int h = i % H;
  beta[i] = sigmoid(b[i]);
  g[i] = softplus(a[i] + dt_bias[h]) * ssm_a[h];
}

// One thread per channel; walks the T tokens. Plane layout [4][C][3], taps oldest first.
__global__ void gdn_conv_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ planes,
                                const int* __restrict__ in_plane, float* __restrict__ y, int C, int T, bool snapshots) {
  pdl_wait();
  pdl_trigger();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float* st = planes + ((size_t)(*in_plane) * C + c) * 3;
  float s0 = st[0], s1 = st[1], s2 = st[2];
  const float w0 = w[4 * c], w1 = w[4 * c + 1], w2 = w[4 * c + 2], w3 = w[4 * c + 3];
  for (int t = 0; t < T; t++) {
    const float xn = x[(size_t)t * C + c];
    float sum = 0.f;
    sum += s0 * w0;
    sum += s1 * w1;
    sum += s2 * w2;
    sum += xn * w3;
    y[(size_t)t * C + c] = silu(sum);
    s0 = s1; s1 = s2; s2 = xn;
    if (snapshots || t == T - 1) {
      float* o = planes + ((size_t)t * C + c) * 3;
      o[0] = s0; o[1] = s1; o[2] = s2;
    }
  }
}

// One warp per (token, head) of n values.
__global__ void l2norm_heads_kernel(float* x, int n, int heads, int T, int stride, float eps) {
  pdl_wait();
  pdl_trigger();
  const int hw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (hw >= heads * T) return;
  const int t = hw / heads, h = hw % heads;
  float* p = x + (size_t)t * stride + (size_t)h * n;
  float acc = 0.f;
  for (int i = lane; i < n; i += 32) acc += p[i] * p[i];
  acc = warp_sum(acc);
  const float scale = rsqrtf(acc / n + eps / n);
  const float post = 1.0f / sqrtf((float)n);
  for (int i = lane; i < n; i += 32) p[i] = (p[i] * scale) * post;
}

// llama.cpp gated_delta_net_cuda (S_v = 128): block = (head h, 4 columns), warp per value column,
// looping over the T tokens with the column of the state in registers.
// Plane layout [4][H][128 value col][128 key i] (M[col][i] = S[i][col], row col contiguous).
__global__ void gdn_step_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                int stride, const float* __restrict__ g, const float* __restrict__ beta,
                                float* __restrict__ planes, const int* __restrict__ in_plane, float* __restrict__ o, int H,
                                int HK, int T, float scale, bool snapshots, bool flip) {
  pdl_wait();
  pdl_trigger();
  const int h = blockIdx.x;
  const int lane = threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  const int kh = h % HK;
  const size_t plane_sz = (size_t)H * 128 * 128;
  const float* M = planes + (size_t)(*in_plane) * plane_sz + ((size_t)h * 128 + col) * 128;
  float s[4];
#pragma unroll
  for (int r = 0; r < 4; r++) s[r] = M[r * 32 + lane];
  for (int t = 0; t < T; t++) {
    const float* qt = q + (size_t)t * stride + kh * 128;
    const float* kt = k + (size_t)t * stride + kh * 128;
    float kr[4], qr[4];
#pragma unroll
    for (int r = 0; r < 4; r++) { kr[r] = kt[r * 32 + lane]; qr[r] = qt[r * 32 + lane]; }
    const float gv = expf(g[(size_t)t * H + h]);
    const float bv = beta[(size_t)t * H + h];
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < 4; r++) kv += s[r] * kr[r];
    kv = warp_sum(kv);
    const float delta = (v[(size_t)t * stride + h * 128 + col] - gv * kv) * bv;
    float at = 0.f;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      s[r] = gv * s[r] + kr[r] * delta;
      at += s[r] * qr[r];
    }
    at = warp_sum(at);
    if (lane == 0) o[((size_t)t * H + h) * 128 + col] = at * scale;
    if (snapshots || t == T - 1) {
      // flip (prefill, T up to thousands): the final state goes to the other of planes 0 and 1
      const int pl = flip ? (*in_plane == 0 ? 1 : 0) : t;
      float* W = planes + (size_t)pl * plane_sz + ((size_t)h * 128 + col) * 128;
#pragma unroll
      for (int r = 0; r < 4; r++) W[r * 32 + lane] = s[r];
    }
  }
}

// One block of 128 threads per head: y = (rms_norm(o) * w) * silu(z)
__global__ void gated_rmsnorm_kernel(const float* __restrict__ o, const float* __restrict__ w, const float* __restrict__ z,
                                     float* __restrict__ y, int n, float eps) {
  pdl_wait();
  pdl_trigger();
  const size_t off = (size_t)blockIdx.x * n;
  float t = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = o[off + i]; t += v * v; }
  t = block_sum(t);
  const float scale = rsqrtf(t / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[off + i] = (scale * o[off + i] * w[i]) * silu(z[off + i]);
}

// ---------------------------------------------------------------- attention
// Grid (nqh + nkvh, T), 256 threads (one per dim).
__global__ void attn_prep_kernel(const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
                                 const float* __restrict__ q_norm, const float* __restrict__ k_norm, float* __restrict__ qn,
                                 void* __restrict__ kcache, void* __restrict__ vcache, const int* __restrict__ dpos0,
                                 float eps, float theta_scale, int nqh, int nkvh, int n_ctx, bool q8, const void* pf,
                                 size_t pf_bytes, const int* __restrict__ rope3, const int* __restrict__ rdelta) {
  pdl_wait();
  pdl_trigger();
  if (pf && threadIdx.x < 32) {
    const int nb = gridDim.x * gridDim.y, b = blockIdx.y * gridDim.x + blockIdx.x;
    l2_prefetch_range(pf, pf_bytes, b * 32 + threadIdx.x, nb * 32);
  }
  const int tok = blockIdx.y;
  const int pos = *dpos0 + tok;  // cache row
  const int hb = blockIdx.x;   // 0..nqh-1 q heads, then nkvh k heads
  const int d = threadIdx.x;   // 0..255
  const bool isq = hb < nqh;
  const float* src = isq ? qg + (size_t)tok * nqh * 512 + (size_t)hb * 512 : k + ((size_t)tok * nkvh + (hb - nqh)) * 256;
  const float* w = isq ? q_norm : k_norm;
  __shared__ float buf[256];
  const float xv = src[d];
  const float t = block_sum(xv * xv);
  const float scale = rsqrtf(t / 256 + eps);
  buf[d] = scale * xv * w[d];
  __syncthreads();
  float outv = buf[d];
  if (d < 64) {
    // NeoX pair (j, j + 32), j < 32; theta = p * theta_scale^j (llama.cpp rope_multi, IMROPE): p = t, h, w for
    // j % 3 = 0, 1, 2 (sections [11,11,10,0]); text rows have t = h = w
    const int j = d & 31;
    int rp;
    if (rope3) rp = rope3[tok * 3 + j % 3];
    else rp = pos - (rdelta ? *rdelta : 0);
    const float theta = (float)rp * powf(theta_scale, (float)j);
    const float c = cosf(theta), sn = sinf(theta);
    const float x0 = buf[j], x1 = buf[j + 32];
    outv = d < 32 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
  }
  if (isq) {
    qn[((size_t)tok * nqh + hb) * 256 + d] = outv;
  } else {
    const int kh = hb - nqh;
    const size_t row = (size_t)kh * n_ctx + pos;
    const float vv = v[((size_t)tok * nkvh + kh) * 256 + d];
    if (!q8) {
      ((__half*)kcache)[row * 256 + d] = __float2half(outv);
      ((__half*)vcache)[row * 256 + d] = __float2half(vv);
    } else {
      // q8_0 per 32 values (one warp): d = amax / 127, q = round(x / d) via x * (1/d) (llama.cpp cpy-utils.cuh)
      float ak = fabsf(outv), av = fabsf(vv);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
      }
      const float dk = ak / 127.f, dv = av / 127.f;
      const float ik = dk ? 1.0f / dk : 0.0f, iv = dv ? 1.0f / dv : 0.0f;
      ((int8_t*)kcache)[row * 256 + d] = (int8_t)roundf(outv * ik);
      ((int8_t*)vcache)[row * 256 + d] = (int8_t)roundf(vv * iv);
      if ((d & 31) == 0) {
        const size_t so = (size_t)nkvh * n_ctx * 256;
        ((__half*)((uint8_t*)kcache + so))[row * 8 + d / 32] = __float2half(dk);
        ((__half*)((uint8_t*)vcache + so))[row * 8 + d / 32] = __float2half(dv);
      }
    }
  }
}

// ---------------------------------------------------------------- cross-card sum
// One-shot sum of two cards' partial vectors through mapped pinned host memory (same numerics as
// llama.cpp ggml_cuda_ar_kernel with a BF16 wire): each card writes bf16(partial) to its host slot,
// raises a per-block flag, waits for the peer's flag, then x += float(bf16(own)) + float(bf16(peer)).
// Both cards compute the same sum. token = (*dstep) * n_ar + index + 1 (monotonic, no resets).
__global__ void ar_add_kernel(float* __restrict__ x, const float* __restrict__ partial, int n, __nv_bfloat16* host_mine,
                              const __nv_bfloat16* host_other, int* flag_mine, const int* flag_other, const int* dstep,
                              int n_ar, int index, int* err) {
  pdl_wait();
  pdl_trigger();
  const int token = (*dstep) * n_ar + index + 1;
  const int per = (n + gridDim.x - 1) / gridDim.x;
  const int beg = blockIdx.x * per, end = min(n, beg + per);
  for (int i = beg + threadIdx.x; i < end; i += blockDim.x) host_mine[i] = __float2bfloat16(partial[i]);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile int* fm = flag_mine + blockIdx.x * 32;
    volatile const int* fo = flag_other + blockIdx.x * 32;
    *fm = token;
    __threadfence_system();
    const long long t0 = clock64();
    while (*fo != token) {
      if (clock64() - t0 > 3000000000LL) { atomicAdd(err, 1); break; }  // about 1 s: the peer is gone
    }
  }
  __syncthreads();
  __threadfence_system();
  for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
    const float own = __bfloat162float(__float2bfloat16(partial[i]));
    const float other = __bfloat162float(((volatile const __nv_bfloat16*)host_other)[i]);
    x[i] = x[i] + (own + other);
  }
}

__global__ void inc_kernel(int* p) {
  pdl_wait();
  pdl_trigger(); *p += 1; }

}  // namespace

void rmsnorm_q8(const float* x, const float* w, float* h, int8_t* xq, float* xd, int n, int rows, float eps, cudaStream_t s,
                float* xs) {
  if (n != 5120) throw std::runtime_error("rmsnorm_q8: n must be 5120");
  launch_k(rmsnorm_q8_kernel<5>, rows, 1024, 0, s, x, w, eps, h, xq, xd, n, xs);
}
void sum_norm_q8(float* x, const float* partial, int n, int rows, const ArArgs* ar, const float* w, float eps, float* h,
                 int8_t* xq, float* xd, cudaStream_t s, const void* pf, size_t pf_bytes) {
  if (n != 5120) throw std::runtime_error("sum_norm_q8: n must be 5120");
  ArArgs a = ar ? *ar : ArArgs{};
  launch_k(sum_norm_q8_kernel<5>, rows, 1024, 0, s, x, partial, a, ar != nullptr, w, eps, h, xq, xd, n, (const uint8_t*)pf,
           pf_bytes);
}
void swiglu_q8(const float* g, const float* u, int8_t* xq, float* xd, int n, cudaStream_t s, float* xs) {
  if (n % 32) throw std::runtime_error("swiglu_q8: n % 32");
  launch_k(swiglu_q8_kernel, (n + 255) / 256, 256, 0, s, g, u, xq, xd, n, xs);
}
void gated_rmsnorm_q8(const float* o, const float* w, const float* z, int8_t* xq, float* xd, int heads, float eps, cudaStream_t s,
                      float* xs) {
  launch_k(gated_rmsnorm_q8_kernel, heads, 128, 0, s, o, w, z, xq, xd, eps, xs);
}
void gdn_conv_l2(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int T, bool snapshots,
                 int n_norm_heads, float eps, const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g,
                 float* beta, int H, cudaStream_t s, const void* pf, size_t pf_bytes) {
  if (C % 128 || C / 128 - n_norm_heads != H) throw std::runtime_error("gdn_conv_l2: bad shape");
  launch_k(gdn_conv_l2_kernel, C / 128, 128, 0, s, x, w, planes, in_plane, y, C, T, snapshots, n_norm_heads, eps, a, b, ssm_a,
                                             dt_bias, g, beta, H, pf, pf_bytes);
}

void ar_add(float* x, const float* partial, int n, __nv_bfloat16* host_mine, const __nv_bfloat16* host_other, int* flag_mine,
            const int* flag_other, const int* dstep, int n_ar, int index, int* err, cudaStream_t s) {
  launch_k(ar_add_kernel, 4, 256, 0, s, x, partial, n, host_mine, host_other, flag_mine, flag_other, dstep, n_ar, index, err);
}
void inc_counter(int* p, cudaStream_t s) { launch_k(inc_kernel, 1, 1, 0, s, p); }
void set_int(int* p, int v, cudaStream_t s) { launch_k(set_int_kernel, 1, 1, 0, s, p, v); }

void rmsnorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s, int ystride) {
  launch_k(rmsnorm_kernel, rows, n >= 1024 ? 1024 : 256, 0, s, x, w, y, n, eps, ystride ? ystride : n);
}
void add(const float* x, const float* a, float* y, int n, cudaStream_t s) {
  launch_k(add_kernel, (n + 255) / 256, 256, 0, s, x, a, y, n);
}
void swiglu(const float* g, const float* u, float* y, int n, cudaStream_t s) {
  launch_k(swiglu_kernel, (n + 255) / 256, 256, 0, s, g, u, y, n);
}

void get_rows_iq2_s(const uint8_t* table, int64_t row_bytes, const int* ids, int T, float* y, int n, cudaStream_t s,
                    const float* img) {
  static const uint64_t* grid_dev[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!grid_dev[dev]) {
    uint64_t* g; CK(cudaMalloc(&g, sizeof(iq2s_grid)));
    CK(cudaMemcpy(g, iq2s_grid, sizeof(iq2s_grid), cudaMemcpyHostToDevice));
    grid_dev[dev] = g;
  }
  launch_k(get_rows_iq2_s_kernel, dim3(n / 256, T), 32, 0, s, table, row_bytes, ids, grid_dev[dev], y, n, img);
}

void gemv_bf16_pair(const __nv_bfloat16* Wa, const __nv_bfloat16* Wb, const float* x, float* ya, float* yb, int N, int K,
                    int T, cudaStream_t s) {
  launch_k(gemv_bf16_pair_kernel, dim3(2 * N, T), 256, 0, s, Wa, Wb, x, ya, yb, N, K);
}

void gdn_gates(const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g, float* beta, int H, int T,
               cudaStream_t s) {
  launch_k(gdn_gates_kernel, (H * T + 127) / 128, 128, 0, s, a, b, ssm_a, dt_bias, g, beta, H, H * T);
}
void gdn_conv(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int T, bool snapshots,
              cudaStream_t s) {
  launch_k(gdn_conv_kernel, (C + 255) / 256, 256, 0, s, x, w, planes, in_plane, y, C, T, snapshots);
}
void l2norm_heads(float* x, int n, int heads, int T, int stride, float eps, cudaStream_t s) {
  launch_k(l2norm_heads_kernel, (heads * T * 32 + 127) / 128, 128, 0, s, x, n, heads, T, stride, eps);
}
void gdn_step(const float* q, const float* k, const float* v, int stride, const float* g, const float* beta, float* planes,
              const int* in_plane, float* o, int H, int HK, int T, float scale, bool snapshots, cudaStream_t s, bool flip) {
  launch_k(gdn_step_kernel, dim3(H, 128 / 4), dim3(32, 4), 0, s, q, k, v, stride, g, beta, planes, in_plane, o, H, HK, T, scale,
                                                           snapshots, flip);
}
void gated_rmsnorm(const float* o, const float* w, const float* z, float* y, int n, int heads, float eps, cudaStream_t s) {
  launch_k(gated_rmsnorm_kernel, heads, 128, 0, s, o, w, z, y, n, eps);
}

void attn_prep(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, float* qn,
               void* kcache, void* vcache, const int* pos0, int T, float eps, float theta_scale, int nqh, int nkvh,
               int n_ctx, bool q8, cudaStream_t s, const void* pf, size_t pf_bytes, const int* rope3, const int* rdelta) {
  launch_k(attn_prep_kernel, dim3(nqh + nkvh, T), 256, 0, s, qg, k, v, q_norm, k_norm, qn, kcache, vcache, pos0, eps, theta_scale,
                                                       nqh, nkvh, n_ctx, q8, pf, pf_bytes, rope3, rdelta);
}
}  // namespace q27
