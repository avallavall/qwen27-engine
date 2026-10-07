// Kernels for batched prompt reading (prefill): M tokens per pass (M up to a few thousand).
// The quantized GEMMs are in qgemm.cu; the rest of the layer is here.
// Numerics follow the decode kernels (and llama.cpp), see ops.cu / attn.cu.
#include "ops.h"
#include "prefill.h"
#include "common.cuh"

#include <algorithm>
#include <cfloat>

namespace q27 {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// ---------------------------------------------------------------- BF16 GEMM for ssm_alpha / ssm_beta
// y[t][n] for two matrices of N rows (Wa rows 0..N-1, Wb rows N..2N-1), x f32 [M][K]. Block: 32 tokens x 2N rows,
// K in pieces of 128 through shared memory. Each thread keeps a few (token, row) sums.
constexpr int BF_T = 32, BF_K = 128;
__global__ void __launch_bounds__(256) bf16_pair_gemm_kernel(const __nv_bfloat16* __restrict__ Wa, const __nv_bfloat16* __restrict__ Wb,
                                                             const float* __restrict__ x, float* __restrict__ ya, float* __restrict__ yb,
                                                             int N, int K, int M) {
  extern __shared__ float sm[];
  float* sx = sm;                      // [BF_T][BF_K + 1]
  float* sw = sx + BF_T * (BF_K + 1);  // [2N][BF_K + 1]
  const int t0 = blockIdx.x * BF_T, R = 2 * N;
  const int outs = BF_T * R;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
  for (int k0 = 0; k0 < K; k0 += BF_K) {
    __syncthreads();
    for (int i = threadIdx.x; i < BF_T * BF_K; i += blockDim.x) {
      const int t = i / BF_K, k = i % BF_K;
      sx[t * (BF_K + 1) + k] = (t0 + t < M) ? x[(size_t)(t0 + t) * K + k0 + k] : 0.f;
    }
    for (int i = threadIdx.x; i < R * BF_K; i += blockDim.x) {
      const int r = i / BF_K, k = i % BF_K;
      const __nv_bfloat16 w = r < N ? Wa[(size_t)r * K + k0 + k] : Wb[(size_t)(r - N) * K + k0 + k];
      sw[r * (BF_K + 1) + k] = __bfloat162float(w);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int o = threadIdx.x + j * blockDim.x;
      if (o >= outs) break;
      const int t = o / R, r = o % R;
      const float* xr = sx + t * (BF_K + 1);
      const float* wr = sw + r * (BF_K + 1);
      float a = acc[j];
      for (int k = 0; k < BF_K; k++) a += wr[k] * xr[k];
      acc[j] = a;
    }
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int o = threadIdx.x + j * blockDim.x;
    if (o >= outs) break;
    const int t = o / R, r = o % R;
    if (t0 + t >= M) continue;
    if (r < N) ya[(size_t)(t0 + t) * N + r] = acc[j];
    else yb[(size_t)(t0 + t) * N + r - N] = acc[j];
  }
}

// Same product, split over K for parallelism: grid (M / 64, K / BF2_KS); block = 64 tokens x 48 rows of one K slice,
// thread = 4 tokens x 3 rows. Partials [K / BF2_KS][M][2N] are added in slice order by bf16_pair_reduce_kernel.
constexpr int BF2_T = 64, BF2_R = 48, BF2_KC = 32, BF2_KS = 512;
__global__ void __launch_bounds__(256) bf16_pair_splitk_kernel(const __nv_bfloat16* __restrict__ Wa, const __nv_bfloat16* __restrict__ Wb,
                                                               const float* __restrict__ x, float* __restrict__ part, int N, int K,
                                                               int M) {
  pdl_wait();
  pdl_trigger();
  __shared__ __align__(16) float sx[BF2_KC][BF2_T];
  __shared__ float sw[BF2_KC][BF2_R];
  const int t0 = blockIdx.x * BF2_T, kbeg = blockIdx.y * BF2_KS;
  const int tid = threadIdx.x, tg = tid & 15, rg = tid >> 4;  // tokens 4tg..4tg+3, rows 3rg..3rg+2
  float acc[4][3];
#pragma unroll
  for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = acc[i][2] = 0.f;
  for (int k0 = kbeg; k0 < kbeg + BF2_KS; k0 += BF2_KC) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < BF2_T * BF2_KC / 256; j++) {
      const int i = tid + j * 256, t = i / BF2_KC, k = i % BF2_KC;
      sx[k][t] = t0 + t < M ? x[(size_t)(t0 + t) * K + k0 + k] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < BF2_R * BF2_KC / 256; j++) {
      const int i = tid + j * 256, r = i / BF2_KC, k = i % BF2_KC;
      sw[k][r] = r < 2 * N ? __bfloat162float(r < N ? Wa[(size_t)r * K + k0 + k] : Wb[(size_t)(r - N) * K + k0 + k]) : 0.f;
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < BF2_KC; k++) {
      const float4 xv = *(const float4*)&sx[k][4 * tg];
      const float w0 = sw[k][3 * rg], w1 = sw[k][3 * rg + 1], w2 = sw[k][3 * rg + 2];
      const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int i = 0; i < 4; i++) { acc[i][0] += w0 * xs[i]; acc[i][1] += w1 * xs[i]; acc[i][2] += w2 * xs[i]; }
    }
  }
  float* p = part + (size_t)blockIdx.y * M * (2 * N);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int t = t0 + 4 * tg + i;
    if (t >= M) continue;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      const int r = 3 * rg + j;
      if (r < 2 * N) p[(size_t)t * (2 * N) + r] = acc[i][j];
    }
  }
}
__global__ void bf16_pair_reduce_kernel(const float* __restrict__ part, int nks, float* __restrict__ ya, float* __restrict__ yb,
                                        int N, int M) {
  pdl_wait();
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * 2 * N) return;
  float s = 0.f;
  for (int k = 0; k < nks; k++) s += part[(size_t)k * M * (2 * N) + i];
  const int t = i / (2 * N), r = i % (2 * N);
  if (r < N) ya[(size_t)t * N + r] = s;
  else yb[(size_t)t * N + r - N] = s;
}

// ---------------------------------------------------------------- GDN input stage over M tokens
// Grid (heads = C/128, M), 128 threads (one per channel of the head). Causal conv (4 taps, the 3 inputs before
// the batch come from plane *in_plane) + SiLU, then the L2 norm for the q and k heads, the gates for the v heads.
// The block of the last token also writes the conv state (last 3 inputs) to the other plane (0 or 1).
// Same float order as gdn_conv_l2_kernel (ops.cu).
__global__ void gdn_conv_prefill_kernel(const float* __restrict__ x, const float* __restrict__ cw, float* __restrict__ planes,
                                        const int* __restrict__ in_plane, float* __restrict__ y, int C, int M,
                                        int n_norm_heads, float eps, const float* __restrict__ a, const float* __restrict__ b,
                                        const float* __restrict__ ssm_a, const float* __restrict__ dt_bias, float* __restrict__ g,
                                        float* __restrict__ beta, int H) {
  pdl_wait();
  pdl_trigger();
  __shared__ float sy[128];
  __shared__ float sscale;
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int c = head * 128 + tid;
  const float* st = planes + ((size_t)(*in_plane) * C + c) * 3;
  auto in = [&](int tt) { return tt >= 0 ? x[(size_t)tt * C + c] : st[3 + tt]; };  // tt = -3..-1 -> state
  const float s0 = in(t - 3), s1 = in(t - 2), s2 = in(t - 1), xn = x[(size_t)t * C + c];
  float sum = 0.f;
  sum += s0 * cw[4 * c];
  sum += s1 * cw[4 * c + 1];
  sum += s2 * cw[4 * c + 2];
  sum += xn * cw[4 * c + 3];
  float out = silu(sum);
  if (head < n_norm_heads) {
    sy[tid] = out;
    __syncthreads();
    if (tid < 32) {
      float acc = 0.f;
      for (int i = tid; i < 128; i += 32) acc += sy[i] * sy[i];
      acc = warp_sum(acc);
      if (tid == 0) sscale = rsqrtf(acc / 128 + eps / 128);
    }
    __syncthreads();
    out = (out * sscale) * (1.0f / sqrtf(128.f));
  } else if (tid == 0) {
    const int hv = head - n_norm_heads;
    const int i = t * H + hv;
    beta[i] = sigmoid(b[i]);
    g[i] = softplus(a[i] + dt_bias[hv]) * ssm_a[hv];
  }
  y[(size_t)t * C + c] = out;
  if (t == M - 1) {
    const int out_plane = *in_plane == 0 ? 1 : 0;
    float* o = planes + ((size_t)out_plane * C + c) * 3;
    o[0] = in(M - 3); o[1] = in(M - 2); o[2] = in(M - 1);
  }
}


__global__ void flip_plane_kernel(int* p) {
  pdl_wait();
  pdl_trigger();
  *p = *p == 0 ? 1 : 0;
}

// x += partial (exchange over both cards), RMSNorm + q8_1 per row. Persistent: block b handles rows b, b + G, ...
// Phase 1 sends all of this block's rows (each row's flag after its data), phase 2 waits for each peer row,
// adds and normalizes. Same per-row arithmetic as sum_norm_q8_kernel (ops.cu).
constexpr int SNR_NPT = 5;
__global__ void __launch_bounds__(1024) sum_norm_rows_kernel(float* __restrict__ x, const float* __restrict__ partial, int M,
                                                             PfExchange ex, bool exchange, const float* __restrict__ w, float eps,
                                                             float* __restrict__ h, int8_t* __restrict__ xq, float* __restrict__ xd,
                                                             float* __restrict__ xs, int n) {
  pdl_wait();
  pdl_trigger();
  __shared__ float red[32];
  if (exchange) {
    for (int r = blockIdx.x; r < M; r += gridDim.x) {
#pragma unroll
      for (int k = 0; k < SNR_NPT; k++) {
        const int i = threadIdx.x + k * 1024;
        ex.mine[(size_t)r * n + i] = __float2bfloat16(partial[(size_t)r * n + i]);
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) *(volatile int*)(ex.flag_mine + (size_t)r * 32) = ex.token;
    }
    __threadfence_system();
  }
  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    float* xr = x + (size_t)r * n;
    const float* pr = partial + (size_t)r * n;
    float v[SNR_NPT];
    if (exchange) {
      if (threadIdx.x == 0) {
        volatile const int* fo = ex.flag_other + (size_t)r * 32;
        const long long t0 = clock64();
        while (*fo != ex.token) {
          if (clock64() - t0 > 6000000000LL) { atomicAdd(ex.err, 1); break; }  // about 2 s: the peer is gone
        }
      }
      __syncthreads();
      __threadfence_system();
#pragma unroll
      for (int k = 0; k < SNR_NPT; k++) {
        const int i = threadIdx.x + k * 1024;
        const float own = __bfloat162float(__float2bfloat16(pr[i]));
        const float oth = __bfloat162float(((volatile const __nv_bfloat16*)ex.other)[(size_t)r * n + i]);
        v[k] = xr[i] + (own + oth);
        xr[i] = v[k];
      }
    } else {
#pragma unroll
      for (int k = 0; k < SNR_NPT; k++) {
        const int i = threadIdx.x + k * 1024;
        v[k] = xr[i] + pr[i];
        xr[i] = v[k];
      }
    }
    // RMSNorm (same order as rmsnorm_kernel with 1024 threads)
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < SNR_NPT; k++) t += v[k] * v[k];
    t = warp_sum(t);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = t;
    __syncthreads();
    t = (threadIdx.x & 31) < 32 ? red[threadIdx.x & 31] : 0.f;
    t = warp_sum(t);
    const float scale = rsqrtf(t / n + eps);
#pragma unroll
    for (int k = 0; k < SNR_NPT; k++) {
      const int i = threadIdx.x + k * 1024;
      const float y = scale * v[k] * w[i];
      if (h) h[(size_t)r * n + i] = y;
      q8_store(y, xq, xd, (size_t)r * n + i, xs);
    }
  }
}

// Cross-card exchange with both PCIe directions busy at once: in every block, warps 0-15 send this card's rows
// (BF16 to host memory, then the row flag) while warps 16-31 wait for the peer's rows, add them and normalize.
// Persistent: block b handles rows b, b + G, ... in both roles.
constexpr int SND = 512, RCV_NPT = 10;  // 512 receiver threads x 10 values = 5120
__device__ __forceinline__ void named_sync(int id) { asm volatile("bar.sync %0, 512;" ::"r"(id) : "memory"); }

__global__ void __launch_bounds__(1024) sum_norm_duplex_kernel(float* __restrict__ x, const float* __restrict__ partial, int M,
                                                               PfExchange ex, const float* __restrict__ w, float eps,
                                                               float* __restrict__ h, int8_t* __restrict__ xq, float* __restrict__ xd,
                                                               float* __restrict__ xs, int n) {
  pdl_wait();
  pdl_trigger();
  __shared__ float red[16];
  if (threadIdx.x < SND) {  // sender
    const int t = threadIdx.x;
    for (int r = blockIdx.x; r < M; r += gridDim.x) {
      const float* pr = partial + (size_t)r * n;
      __nv_bfloat16* dst = ex.mine + (size_t)r * n;
      for (int i = 2 * t; i < n; i += 2 * SND) {
        const float2 p2 = *(const float2*)(pr + i);
        *(__nv_bfloat162*)(dst + i) = __floats2bfloat162_rn(p2.x, p2.y);
      }
      __threadfence_system();
      named_sync(1);
      if (t == 0) *(volatile int*)(ex.flag_mine + (size_t)r * 32) = ex.token;
    }
    return;
  }
  const int t = threadIdx.x - SND, lane = t & 31, wr = t >> 5;  // receiver
  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    if (t == 0) {
      volatile const int* fo = ex.flag_other + (size_t)r * 32;
      const long long t0 = clock64();
      while (*fo != ex.token) {
        if (clock64() - t0 > 6000000000LL) { atomicAdd(ex.err, 1); break; }  // about 2 s: the peer is gone
      }
    }
    named_sync(2);
    __threadfence_system();
    float* xr = x + (size_t)r * n;
    const float* pr = partial + (size_t)r * n;
    const __nv_bfloat16* other = ex.other + (size_t)r * n;
    float v[RCV_NPT];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < RCV_NPT; k++) {
      const int i = t + k * SND;
      const float own = __bfloat162float(__float2bfloat16(pr[i]));
      const float oth = __bfloat162float(((volatile const __nv_bfloat16*)other)[i]);
      v[k] = xr[i] + (own + oth);
      xr[i] = v[k];
      ss += v[k] * v[k];
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wr] = ss;
    named_sync(2);
    ss = lane < 16 ? red[lane] : 0.f;
    ss = warp_sum(ss);
    named_sync(2);  // red is reused for the next row
    const float scale = rsqrtf(ss / n + eps);
#pragma unroll
    for (int k = 0; k < RCV_NPT; k++) {
      const int i = t + k * SND;
      const float y = scale * v[k] * w[i];
      if (h) h[(size_t)r * n + i] = y;
      q8_store(y, xq, xd, (size_t)r * n + i, xs);
    }
  }
}

__global__ void to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
  pdl_wait();
  pdl_trigger();
  for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 2; i < n; i += (size_t)gridDim.x * blockDim.x * 2) {
    const float2 v = *(const float2*)(x + i);
    *(__nv_bfloat162*)(y + i) = __floats2bfloat162_rn(v.x, v.y);
  }
}

// q8 wire (prefill): y = partial (+ ef unless first); int8 per value and fp16 scale per 32 -> wire row r at
// wire + r * (n + n / 16): n int8 values, then n / 32 scales; ef = y - dequant (error feedback, may be null).
// Block per row, 1024 threads, same rounding as the decode q8 wire (sum_norm_q8_kernel).
__global__ void __launch_bounds__(1024) to_q8_wire_kernel(const float* __restrict__ partial, float* __restrict__ ef, int first,
                                                          uint8_t* __restrict__ wire, int n, int qb) {
  pdl_wait();
  pdl_trigger();
  const size_t r = blockIdx.x;
  int8_t* q8 = (int8_t*)wire + r * (n + 2 * n / qb);
  __half* sc = (__half*)(wire + r * (n + 2 * n / qb) + n);
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    const float y = ef && !first ? partial[r * n + i] + ef[r * n + i] : partial[r * n + i];
    float amax = fabsf(y);
    for (int o = qb / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half dh = __float2half(amax / 127.0f);
    const int q = amax == 0.0f ? 0 : (int)roundf(y / (amax / 127.0f));
    q8[i] = (int8_t)q;
    if ((threadIdx.x & (qb - 1)) == 0) sc[i / qb] = dh;
    if (ef) ef[r * n + i] = y - __half2float(dh) * (float)q;
  }
}

// x += dequant(own) + dequant(recv) (both q8 wire rows), then RMSNorm + q8_1 per row (as add_norm_rows_kernel).
__global__ void __launch_bounds__(1024) add_norm_rows_q8_kernel(float* __restrict__ x, const uint8_t* __restrict__ own,
                                                                const uint8_t* __restrict__ recv, const float* __restrict__ w,
                                                                float eps, float* __restrict__ h, int8_t* __restrict__ xq,
                                                                float* __restrict__ xd, float* __restrict__ xs, int n, int qb) {
  pdl_wait();
  pdl_trigger();
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  float* xr = x + r * n;
  const size_t rb = (size_t)n + 2 * n / qb;
  const int8_t* oq = (const int8_t*)own + r * rb;
  const __half* os = (const __half*)(own + r * rb + n);
  const int8_t* rq = (const int8_t*)recv + r * rb;
  const __half* rs = (const __half*)(recv + r * rb + n);
  float v[SNR_NPT];
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    v[k] = xr[i] + (__half2float(os[i / qb]) * (float)oq[i] + __half2float(rs[i / qb]) * (float)rq[i]);
    xr[i] = v[k];
  }
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) t += v[k] * v[k];
  t = warp_sum(t);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = t;
  __syncthreads();
  t = red[threadIdx.x & 31];
  t = warp_sum(t);
  const float scale = rsqrtf(t / n + eps);
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    const float y = scale * v[k] * w[i];
    if (h) h[r * n + i] = y;
    q8_store(y, xq, xd, r * n + i, xs);
  }
}

__global__ void flip_plane_to_kernel(int* dst, const int* src) {
  pdl_wait();
  pdl_trigger();
  *dst = *src == 0 ? 1 : 0;
}

// x += bf16(partial) + recv (the other card's BF16 partial; null: x += partial), then RMSNorm + q8_1 per row.
// Block per row, 1024 threads; same arithmetic as sum_norm_q8_kernel (ops.cu).
__global__ void __launch_bounds__(1024) add_norm_rows_kernel(float* __restrict__ x, const float* __restrict__ partial,
                                                             const __nv_bfloat16* __restrict__ recv, const float* __restrict__ w,
                                                             float eps, float* __restrict__ h, int8_t* __restrict__ xq,
                                                             float* __restrict__ xd, float* __restrict__ xs, int n) {
  pdl_wait();
  pdl_trigger();
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  float* xr = x + r * n;
  const float* pr = partial + r * n;
  float v[SNR_NPT];
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    if (recv) v[k] = xr[i] + (__bfloat162float(__float2bfloat16(pr[i])) + __bfloat162float(recv[r * n + i]));
    else v[k] = xr[i] + pr[i];
    xr[i] = v[k];
  }
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) t += v[k] * v[k];
  t = warp_sum(t);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = t;
  __syncthreads();
  t = red[threadIdx.x & 31];
  t = warp_sum(t);
  const float scale = rsqrtf(t / n + eps);
#pragma unroll
  for (int k = 0; k < SNR_NPT; k++) {
    const int i = threadIdx.x + k * 1024;
    const float y = scale * v[k] * w[i];
    if (h) h[r * n + i] = y;
    q8_store(y, xq, xd, r * n + i, xs);
  }
}

}  // namespace

void bf16_pair_gemm(const __nv_bfloat16* Wa, const __nv_bfloat16* Wb, const float* x, float* ya, float* yb, int N, int K, int M,
                    cudaStream_t s) {
  static const bool old = [] { const char* e = getenv("Q27_BF16_OLD"); return e && e[0] == '1'; }();
  if (!old && 2 * N <= BF2_R && K % BF2_KS == 0) {
    static float* part[16] = {};
    static size_t part_n[16] = {};
    int dev; CK(cudaGetDevice(&dev));
    const int nks = K / BF2_KS;
    const size_t need = (size_t)nks * std::max(M, 2048) * 2 * N;  // sized once for batches up to 2048 rows
    if (need > part_n[dev]) { if (part[dev]) CK(cudaFree(part[dev])); CK(cudaMalloc(&part[dev], need * 4)); part_n[dev] = need; }
    launch_k(bf16_pair_splitk_kernel, dim3((M + BF2_T - 1) / BF2_T, nks), 256, 0, s, Wa, Wb, x, part[dev], N, K, M);
    launch_k(bf16_pair_reduce_kernel, (M * 2 * N + 255) / 256, 256, 0, s, (const float*)part[dev], nks, ya, yb, N, M);
    return;
  }
  if (K % BF_K || 2 * N * BF_T > 8 * 256) throw std::runtime_error("bf16_pair_gemm: bad shape");
  const size_t smem = sizeof(float) * (BF_T + 2 * N) * (BF_K + 1);
  static bool init[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!init[dev]) { CK(cudaFuncSetAttribute(bf16_pair_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024)); init[dev] = true; }
  launch_k(bf16_pair_gemm_kernel, (M + BF_T - 1) / BF_T, 256, smem, s, Wa, Wb, x, ya, yb, N, K, M);
}

void gdn_conv_prefill(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int M,
                      int n_norm_heads, float eps, const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g,
                      float* beta, int H, cudaStream_t s) {
  if (C % 128 || C / 128 - n_norm_heads != H) throw std::runtime_error("gdn_conv_prefill: bad shape");
  launch_k(gdn_conv_prefill_kernel, dim3(C / 128, M), 128, 0, s, x, w, planes, in_plane, y, C, M, n_norm_heads, eps, a, b,
           ssm_a, dt_bias, g, beta, H);
}

void gdn_flip_plane(int* plane, cudaStream_t s) { launch_k(flip_plane_kernel, 1, 1, 0, s, plane); }

void sum_norm_rows(float* x, const float* partial, int n, int M, const PfExchange* ex, const float* w, float eps, float* h,
                   int8_t* xq, float* xd, float* xs, cudaStream_t s) {
  if (n != SNR_NPT * 1024) throw std::runtime_error("sum_norm_rows: n must be 5120");
  int dev, nsm;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  const int G = M < nsm ? M : nsm;  // all blocks resident (one 1024-thread block per SM): no deadlock while waiting
  PfExchange e = ex ? *ex : PfExchange{};
  if (ex && !getenv("Q27_PF_SIMPLEX"))
    launch_k(sum_norm_duplex_kernel, G, 1024, 0, s, x, partial, M, e, w, eps, h, xq, xd, xs, n);
  else
    launch_k(sum_norm_rows_kernel, G, 1024, 0, s, x, partial, M, e, ex != nullptr, w, eps, h, xq, xd, xs, n);
}

void to_bf16(const float* x, __nv_bfloat16* y, size_t n, cudaStream_t s) {
  launch_k(to_bf16_kernel, 64, 256, 0, s, x, y, n);
}
void to_q8_wire(const float* partial, float* ef, bool first, uint8_t* wire, int n, int M, cudaStream_t s) {
  if (n != SNR_NPT * 1024) throw std::runtime_error("to_q8_wire: n must be 5120");
  launch_k(to_q8_wire_kernel, M, 1024, 0, s, partial, ef, first ? 1 : 0, wire, n, wire_qb());
}
void add_norm_rows_q8(float* x, const uint8_t* own, const uint8_t* recv, int n, int M, const float* w, float eps, float* h,
                      int8_t* xq, float* xd, float* xs, cudaStream_t s) {
  if (n != SNR_NPT * 1024) throw std::runtime_error("add_norm_rows_q8: n must be 5120");
  launch_k(add_norm_rows_q8_kernel, M, 1024, 0, s, x, own, recv, w, eps, h, xq, xd, xs, n, wire_qb());
}
void flip_plane_to(int* dst, const int* src, cudaStream_t s) { launch_k(flip_plane_to_kernel, 1, 1, 0, s, dst, src); }
void add_norm_rows(float* x, const float* partial, const __nv_bfloat16* recv, int n, int M, const float* w, float eps, float* h,
                   int8_t* xq, float* xd, float* xs, cudaStream_t s) {
  if (n != SNR_NPT * 1024) throw std::runtime_error("add_norm_rows: n must be 5120");
  launch_k(add_norm_rows_kernel, M, 1024, 0, s, x, partial, recv, w, eps, h, xq, xd, xs, n);
}

}  // namespace q27
