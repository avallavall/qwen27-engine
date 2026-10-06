// Decode attention for the full-attention layers: T query tokens (1..4) at positions pos0 .. pos0+T-1,
// 6 query heads per KV head, head dim 256, f16 KV cache in head-major layout [kvh][n_ctx][256].
//
// attn_decode: one CTA per (chunk of positions, KV head). All 6*T query rows of the KV head are in the
// CTA, so every K and V byte is read from DRAM once. Tiles of 32 positions are loaded with cp.async
// (double buffered), S = Q K^T and O += P V use mma.sync m16n8k16 (f16 in, f32 accumulate). Warps
// are (m-tile of 16 query rows) x (half of the tile's positions); the two halves are merged in shared
// memory at the end, then each CTA writes (max, sum, O) for its chunk. attn_combine merges the chunks
// and applies the sigmoid gate.
// Numerics as llama.cpp FlashAttention: Q scaled by kq_scale and rounded to f16, scores f32, P rounded
// to f16 for P V, row sums from the f32 P, exp of score differences below -20 flushed to zero.
#include "ops.h"
#include "common.cuh"

#include <algorithm>
#include <cfloat>

namespace q27 {

namespace {

constexpr int AD = 256;           // head dim
constexpr int GQA = 6;            // query heads per KV head
constexpr int TP = 32;            // positions per tile
constexpr int ROWB = AD * 2 + 16; // shared-memory row stride in bytes (16 B padding: no bank conflicts)
constexpr int PROW = AD + 2;      // floats per partial row: m, l, O[256]
constexpr int MAXR = GQA * 4;     // max query rows per KV head
constexpr float M_INIT = -1e30f;  // finite start value for the running max (no inf - inf)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
  const __half2 h = __floats2half2_rn(x, y);
  return *(const uint32_t*)&h;
}

// 16 int8 values x one f16 scale -> 16 f16 values (32 bytes). q + 128 is put into the low byte of the f16 number
// 1024 + (q + 128), then 1152 is subtracted: exact; then the product with the scale is rounded once to f16, the
// same as llama.cpp's dequantize_q8_0 followed by the f16 conversion.
__device__ __forceinline__ void cvt_q8_16(const uint8_t* src, __half d, uint8_t* dst) {
  const uint4 q = *(const uint4*)src;
  const uint32_t qw[4] = {q.x ^ 0x80808080u, q.y ^ 0x80808080u, q.z ^ 0x80808080u, q.w ^ 0x80808080u};
  const __half2 dd = __half2half2(d), off = __float2half2_rn(1152.f);
  uint32_t o[8];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    uint32_t lo = __byte_perm(qw[k], 0x64646464u, 0x4140), hi = __byte_perm(qw[k], 0x64646464u, 0x4342);
    __half2 a = __hmul2(__hsub2(*(__half2*)&lo, off), dd), b = __hmul2(__hsub2(*(__half2*)&hi, off), dd);
    o[2 * k] = *(uint32_t*)&a;
    o[2 * k + 1] = *(uint32_t*)&b;
  }
  *(uint4*)dst = make_uint4(o[0], o[1], o[2], o[3]);
  *(uint4*)(dst + 16) = make_uint4(o[4], o[5], o[6], o[7]);
}

// Chunk of positions [c0, c1) of CTA `ch` among `nch`, for n_max positions. Chunks are multiples of TP.
__device__ __forceinline__ void chunk_range(int n_max, int nch, int ch, int& c0, int& c1) {
  int chunk = (n_max + nch - 1) / nch;
  chunk = max(64, (chunk + TP - 1) / TP * TP);
  c0 = min(n_max, ch * chunk);
  c1 = min(n_max, c0 + chunk);
}

// Shared memory of attn_mma_kernel: Q tile, then the K/V area. f16 cache: two stages of f16 K|V tiles.
// q8_0 cache: one f16 K|V tile (converted) and two stages of int8 values + f16 scales.
constexpr int F16_TILE = 2 * TP * ROWB;              // K|V rows of one tile, f16, padded rows
constexpr int Q8_STAGE = 2 * TP * AD + 2 * TP * 16;  // int8 K|V values, then K|V scales (8 halves per row)
constexpr int kv_smem(bool q8) { return q8 ? F16_TILE + 2 * Q8_STAGE : 2 * F16_TILE; }
constexpr int attn_smem(int mt, bool q8) { return mt * 16 * ROWB + kv_smem(q8); }

// MT = m-tiles of 16 query rows (1: T = 1, 2: T <= 4). Warps: MT x 2 position halves.
// Q8: the cache is q8_0: per layer, int8 values [nkvh][n_ctx][256] then f16 scales [nkvh][n_ctx][8].
template <int MT, bool Q8>
__global__ void __launch_bounds__(MT * 64) attn_mma_kernel(const float* __restrict__ qn, const void* __restrict__ kc,
                                                           const void* __restrict__ vc, float* __restrict__ part,
                                                           const int* __restrict__ dpos0, int T, int nqh, int n_ctx,
                                                           int nch, float kq_scale) {
  pdl_wait();
  pdl_trigger();
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sQ = smem;                      // [MT*16][ROWB]
  uint8_t* sKV = smem + MT * 16 * ROWB;    // see kv_smem
  constexpr int NT = MT * 64;

  const int ch = blockIdx.x, kh = blockIdx.y, nkvh = gridDim.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mt = warp % MT, pg = warp / MT;
  const int R = GQA * T;
  const int pos0 = *dpos0;
  const int n_max = pos0 + T;
  int c0, c1;
  chunk_range(n_max, nch, ch, c0, c1);
  float* out = part + ((size_t)kh * nch + ch) * MAXR * PROW;

  if (c0 >= c1) {  // nothing to do: empty partial
    for (int i = tid; i < R * PROW; i += NT) out[i] = (i % PROW == 0) ? M_INIT : 0.f;
    return;
  }

  const size_t row_bytes = Q8 ? AD : AD * 2;  // bytes per cache row (values)
  const uint8_t* kbase = (const uint8_t*)kc + (size_t)kh * n_ctx * row_bytes;
  const uint8_t* vbase = (const uint8_t*)vc + (size_t)kh * n_ctx * row_bytes;
  const uint8_t* ksc = (const uint8_t*)kc + (size_t)nkvh * n_ctx * AD + (size_t)kh * n_ctx * 16;
  const uint8_t* vsc = (const uint8_t*)vc + (size_t)nkvh * n_ctx * AD + (size_t)kh * n_ctx * 16;
  auto load_tile = [&](int stage, int t0) {
    if constexpr (!Q8) {
      uint8_t* dst = sKV + stage * F16_TILE;
#pragma unroll 4
      for (int i = tid; i < 2 * TP * (AD / 8); i += NT) {
        const int row = i / (AD / 8), c = i % (AD / 8);
        const int p = t0 + (row % TP);
        const bool ok = p < c1;
        const uint8_t* src = (row < TP ? kbase : vbase) + (size_t)(ok ? p : c0) * AD * 2 + c * 16;
        cp_async16(dst + row * ROWB + c * 16, src, ok);
      }
    } else {
      uint8_t* dst = sKV + F16_TILE + stage * Q8_STAGE;
#pragma unroll 4
      for (int i = tid; i < 2 * TP * (AD / 16 + 1); i += NT) {
        const int row = i / (AD / 16 + 1), c = i % (AD / 16 + 1);
        const int p = t0 + (row % TP);
        const bool ok = p < c1;
        const size_t pr = (size_t)(ok ? p : c0);
        if (c < AD / 16) cp_async16(dst + row * AD + c * 16, (row < TP ? kbase : vbase) + pr * AD + c * 16, ok);
        else cp_async16(dst + 2 * TP * AD + row * 16, (row < TP ? ksc : vsc) + pr * 16, ok);
      }
    }
  };
  load_tile(0, c0);
  cp_async_commit();

  // Q tile: row r = t*6 + j (token t, query head kh*6+j), scaled, f16; padding rows are zero.
  for (int i = tid; i < MT * 16 * (AD / 2); i += NT) {
    const int r = i / (AD / 2), d2 = i % (AD / 2);
    uint32_t v = 0;
    if (r < R) {
      const float2 q = *(const float2*)(qn + ((size_t)(r / GQA) * nqh + kh * GQA + r % GQA) * AD + 2 * d2);
      v = pack_h2(q.x * kq_scale, q.y * kq_scale);
    }
    *(uint32_t*)(sQ + r * ROWB + 4 * d2) = v;
  }

  // This thread's two query rows and the end of the positions each may see.
  const int g = lane >> 2, tg = lane & 3;
  const int r0 = mt * 16 + g, r1 = r0 + 8;
  const int e0 = min(c1, pos0 + min(r0 / GQA, T - 1) + 1);
  const int e1 = min(c1, pos0 + min(r1 / GQA, T - 1) + 1);

  float o[AD / 8][4];
#pragma unroll
  for (int i = 0; i < AD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = M_INIT, m1 = M_INIT, l0 = 0.f, l1 = 0.f;

  const int ntiles = (c1 - c0 + TP - 1) / TP;
  for (int it = 0; it < ntiles; it++) {
    if (it + 1 < ntiles) load_tile((it + 1) & 1, c0 + (it + 1) * TP);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    if constexpr (Q8) {  // int8 values x f16 scale -> f16 tile (llama.cpp dequantize_q8_0, then to half)
      const uint8_t* src = sKV + F16_TILE + (it & 1) * Q8_STAGE;
      for (int i = tid; i < 2 * TP * (AD / 16); i += NT) {
        const int row = i / (AD / 16), c = i % (AD / 16);
        cvt_q8_16(src + row * AD + c * 16, *(const __half*)(src + 2 * TP * AD + row * 16 + (c / 2) * 2), sKV + row * ROWB + c * 32);
      }
      __syncthreads();
    }

    const uint8_t* sK = sKV + (Q8 ? 0 : (it & 1) * F16_TILE) + pg * 16 * ROWB;
    const uint8_t* sV = sK + TP * ROWB;
    const int pbase = c0 + it * TP + pg * 16;

    // S = Q K^T for 16 rows x 16 positions (two n-tiles of 8 positions)
    float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < AD / 16; ks++) {
      uint32_t a[4], b[4];
      ldsm_x4(a, sQ + (mt * 16 + (lane & 15)) * ROWB + (ks * 16 + (lane >> 4) * 8) * 2);
      ldsm_x4(b, sK + ((lane & 7) + (lane >> 4) * 8) * ROWB + (ks * 16 + ((lane >> 3) & 1) * 8) * 2);
      mma16816(s[0], a, b[0], b[1]);
      mma16816(s[1], a, b[2], b[3]);
    }
    // mask and online softmax
    float mx0 = M_INIT, mx1 = M_INIT;
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int p = pbase + nt * 8 + 2 * tg + e;
        if (p >= e0) s[nt][e] = -INFINITY;
        if (p >= e1) s[nt][2 + e] = -INFINITY;
        mx0 = fmaxf(mx0, s[nt][e]);
        mx1 = fmaxf(mx1, s[nt][2 + e]);
      }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float d0 = m0 - mn0, d1 = m1 - mn1;
    const float sc0 = d0 >= -20.f ? expf(d0) : 0.f;
    const float sc1 = d1 >= -20.f ? expf(d1) : 0.f;
    m0 = mn0; m1 = mn1;
    float add0 = 0.f, add1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        s[nt][e] = expf(s[nt][e] - mn0);
        s[nt][2 + e] = expf(s[nt][2 + e] - mn1);
        add0 += s[nt][e];
        add1 += s[nt][2 + e];
      }
    l0 = l0 * sc0 + add0;
    l1 = l1 * sc1 + add1;
#pragma unroll
    for (int i = 0; i < AD / 8; i++) { o[i][0] *= sc0; o[i][1] *= sc0; o[i][2] *= sc1; o[i][3] *= sc1; }
    uint32_t pa[4];
    pa[0] = pack_h2(s[0][0], s[0][1]);
    pa[1] = pack_h2(s[0][2], s[0][3]);
    pa[2] = pack_h2(s[1][0], s[1][1]);
    pa[3] = pack_h2(s[1][2], s[1][3]);
    // O += P V
#pragma unroll
    for (int dn = 0; dn < AD / 16; dn++) {
      uint32_t b[4];
      ldsm_x4_t(b, sV + ((lane & 7) + ((lane >> 3) & 1) * 8) * ROWB + (dn * 16 + (lane >> 4) * 8) * 2);
      mma16816(o[2 * dn], pa, b[0], b[1]);
      mma16816(o[2 * dn + 1], pa, b[2], b[3]);
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  // Row sums over the quad, then merge the two position halves through shared memory.
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  float* so = (float*)sKV;                      // [warps][16][AD]
  float* sm = so + MT * 2 * 16 * AD;            // [warps][16]
  float* sl = sm + MT * 2 * 16;                 // [warps][16]
  {
    float* w = so + (size_t)warp * 16 * AD;
#pragma unroll
    for (int i = 0; i < AD / 8; i++) {
      *(float2*)(w + g * AD + i * 8 + 2 * tg) = make_float2(o[i][0], o[i][1]);
      *(float2*)(w + (g + 8) * AD + i * 8 + 2 * tg) = make_float2(o[i][2], o[i][3]);
    }
    if (tg == 0) {
      sm[warp * 16 + g] = m0; sm[warp * 16 + g + 8] = m1;
      sl[warp * 16 + g] = l0; sl[warp * 16 + g + 8] = l1;
    }
  }
  __syncthreads();
  for (int i = tid; i < R * AD; i += NT) {
    const int r = i / AD, d = i % AD;
    const int w0 = r / 16, w1 = w0 + MT, rr = r % 16;
    const float ma = sm[w0 * 16 + rr], mb = sm[w1 * 16 + rr];
    const float M = fmaxf(ma, mb);
    const float fa = expf(ma - M), fb = expf(mb - M);
    out[r * PROW + 2 + d] = so[((size_t)w0 * 16 + rr) * AD + d] * fa + so[((size_t)w1 * 16 + rr) * AD + d] * fb;
    if (d == 0) {
      out[r * PROW] = M;
      out[r * PROW + 1] = sl[w0 * 16 + rr] * fa + sl[w1 * 16 + rr] * fb;
    }
  }
}

// Grid (nqh, T), 256 threads: merge the chunks of row (t, h), divide by the sum, apply the sigmoid gate.
__global__ void attn_combine_kernel(const float* __restrict__ part, const float* __restrict__ qg, float* __restrict__ o,
                                    int8_t* __restrict__ xq, float* __restrict__ xd, int nqh, int nch) {
  pdl_wait();
  pdl_trigger();
  const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  const int kh = h / GQA, r = t * GQA + h % GQA;
  const float* p = part + ((size_t)kh * nch * MAXR + r) * PROW;
  float M = M_INIT;
  for (int c = 0; c < nch; c++) M = fmaxf(M, p[(size_t)c * MAXR * PROW]);
  float L = 0.f, A = 0.f;
  for (int c = 0; c < nch; c++) {
    const float* q = p + (size_t)c * MAXR * PROW;
    const float f = expf(q[0] - M);
    L += q[1] * f;
    A += q[2 + d] * f;
  }
  const float gate = qg[((size_t)t * nqh + h) * 512 + 256 + d];
  const float y = (A / L) * (1.0f / (1.0f + expf(-gate)));
  const size_t i = ((size_t)t * nqh + h) * AD + d;
  if (xq) q8_store(y, xq, xd, i);
  else o[i] = y;
}

// ---------------------------------------------------------------- prefill attention (causal, M query tokens)
// Grid (ceil(M/16), nkvh), 6 warps: warp j = query head kh*6 + j, 16 query tokens per block (rows of the warp's
// m16 tile). K/V tiles of PT positions go through shared memory and serve all 6 heads. S = Q K^T in f32,
// P V with f16 accumulators and the llama.cpp max offset (P <= 1/8), as llama.cpp's prefill FlashAttention.
// Output: o [M][nqh][256] f32 = softmax(...) V * sigmoid(gate). Blocks with more positions start first.
constexpr float FA_MAX_OFFSET = 3.0f * 0.6931f;  // llama.cpp FATTN_KQ_MAX_OFFSET

__device__ __forceinline__ void mma16816_h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Shared memory: the Q tile first (96 rows, read once into registers), then reused for the K/V stages:
// f16: two f16 K|V tiles; q8_0: one f16 tile (converted) + two int8 stages.
template <int PT, bool Q8>
constexpr int pf_kv_bytes() { return Q8 ? 2 * PT * ROWB + 2 * (2 * PT * AD + 2 * PT * 16) : 2 * (2 * PT * ROWB); }
template <int PT, bool Q8>
constexpr int pf_smem() { return pf_kv_bytes<PT, Q8>() > 96 * ROWB ? pf_kv_bytes<PT, Q8>() : 96 * ROWB; }

template <int PT, bool Q8>
__global__ void __launch_bounds__(192) attn_prefill_kernel(const float* __restrict__ qn, const float* __restrict__ qg,
                                                           const void* __restrict__ kc, const void* __restrict__ vc,
                                                           float* __restrict__ o, const int* __restrict__ dpos0, int M, int nqh,
                                                           int n_ctx, float kq_scale) {
  pdl_wait();
  pdl_trigger();
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int F16T = 2 * PT * ROWB;               // f16 K|V tile
  constexpr int Q8S = 2 * PT * AD + 2 * PT * 16;    // int8 K|V values + scales
  const int kh = blockIdx.y, nkvh = gridDim.y;
  const int t0 = (gridDim.x - 1 - blockIdx.x) * 16;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int pos0 = *dpos0;
  const int n_end = pos0 + min(t0 + 16, M);

  // Q: scaled, f16, through shared memory into this warp's A fragments (16 rows x 256 dims)
  for (int i = tid; i < 96 * (AD / 2); i += 192) {
    const int r = i / (AD / 2), d2 = i % (AD / 2);
    const int j = r / 16, t = t0 + r % 16;
    uint32_t v = 0;
    if (t < M) {
      const float2 q = *(const float2*)(qn + ((size_t)t * nqh + kh * GQA + j) * AD + 2 * d2);
      v = pack_h2(q.x * kq_scale, q.y * kq_scale);
    }
    *(uint32_t*)(smem + r * ROWB + 4 * d2) = v;
  }
  __syncthreads();
  uint32_t qf[AD / 16][4];
#pragma unroll
  for (int ks = 0; ks < AD / 16; ks++) ldsm_x4(qf[ks], smem + (warp * 16 + (lane & 15)) * ROWB + (ks * 16 + (lane >> 4) * 8) * 2);
  __syncthreads();  // the Q area is reused for K/V

  const size_t row_bytes = Q8 ? AD : AD * 2;
  const uint8_t* kbase = (const uint8_t*)kc + (size_t)kh * n_ctx * row_bytes;
  const uint8_t* vbase = (const uint8_t*)vc + (size_t)kh * n_ctx * row_bytes;
  const uint8_t* ksc = (const uint8_t*)kc + (size_t)nkvh * n_ctx * AD + (size_t)kh * n_ctx * 16;
  const uint8_t* vsc = (const uint8_t*)vc + (size_t)nkvh * n_ctx * AD + (size_t)kh * n_ctx * 16;
  auto load_tile = [&](int stage, int p0) {
    if constexpr (!Q8) {
      uint8_t* dst = smem + stage * F16T;
      for (int i = tid; i < 2 * PT * (AD / 8); i += 192) {
        const int row = i / (AD / 8), c = i % (AD / 8);
        const int p = p0 + row % PT;
        const bool ok = p < n_end;
        cp_async16(dst + row * ROWB + c * 16, (row < PT ? kbase : vbase) + (size_t)(ok ? p : 0) * AD * 2 + c * 16, ok);
      }
    } else {
      uint8_t* dst = smem + F16T + stage * Q8S;
      for (int i = tid; i < 2 * PT * (AD / 16 + 1); i += 192) {
        const int row = i / (AD / 16 + 1), c = i % (AD / 16 + 1);
        const int p = p0 + row % PT;
        const bool ok = p < n_end;
        const size_t pr = (size_t)(ok ? p : 0);
        if (c < AD / 16) cp_async16(dst + row * AD + c * 16, (row < PT ? kbase : vbase) + pr * AD + c * 16, ok);
        else cp_async16(dst + 2 * PT * AD + row * 16, (row < PT ? ksc : vsc) + pr * 16, ok);
      }
    }
  };

  const int g = lane >> 2, tg = lane & 3;
  const int lim0 = pos0 + t0 + g + 1, lim1 = lim0 + 8;  // positions < lim are visible to rows g, g+8
  uint32_t oacc[AD / 8][2];
#pragma unroll
  for (int i = 0; i < AD / 8; i++) oacc[i][0] = oacc[i][1] = 0u;
  float m0 = M_INIT, m1 = M_INIT, l0 = 0.f, l1 = 0.f;

  const int ntiles = (n_end + PT - 1) / PT;
  load_tile(0, 0);
  cp_async_commit();
  for (int it = 0; it < ntiles; it++) {
    const int p0 = it * PT;
    if (it + 1 < ntiles) load_tile((it + 1) & 1, p0 + PT);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint8_t* tile;
    if constexpr (Q8) {
      const uint8_t* src = smem + F16T + (it & 1) * Q8S;
      for (int i = tid; i < 2 * PT * (AD / 16); i += 192) {
        const int row = i / (AD / 16), c = i % (AD / 16);
        cvt_q8_16(src + row * AD + c * 16, *(const __half*)(src + 2 * PT * AD + row * 16 + (c / 2) * 2), smem + row * ROWB + c * 32);
      }
      __syncthreads();
      tile = smem;
    } else {
      tile = smem + (it & 1) * F16T;
    }

    const uint8_t* sK = tile;
    const uint8_t* sV = tile + PT * ROWB;
    float s[PT / 8][4];
#pragma unroll
    for (int i = 0; i < PT / 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < AD / 16; ks++) {
#pragma unroll
      for (int np = 0; np < PT / 16; np++) {
        uint32_t b[4];
        ldsm_x4(b, sK + (np * 16 + (lane & 7) + (lane >> 4) * 8) * ROWB + (ks * 16 + ((lane >> 3) & 1) * 8) * 2);
        mma16816(s[2 * np], qf[ks], b[0], b[1]);
        mma16816(s[2 * np + 1], qf[ks], b[2], b[3]);
      }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < PT / 8; nt++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int p = p0 + nt * 8 + 2 * tg + e;
        if (p >= lim0) s[nt][e] = -INFINITY;
        else mx0 = fmaxf(mx0, s[nt][e] + FA_MAX_OFFSET);
        if (p >= lim1) s[nt][2 + e] = -INFINITY;
        else mx1 = fmaxf(mx1, s[nt][2 + e] + FA_MAX_OFFSET);
      }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float d0 = m0 - mx0, d1 = m1 - mx1;
    const float sc0 = d0 >= -20.f ? expf(d0) : 0.f;
    const float sc1 = d1 >= -20.f ? expf(d1) : 0.f;
    m0 = mx0; m1 = mx1;
    float add0 = 0.f, add1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < PT / 8; nt++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        s[nt][e] = expf(s[nt][e] - mx0);
        s[nt][2 + e] = expf(s[nt][2 + e] - mx1);
        add0 += s[nt][e];
        add1 += s[nt][2 + e];
      }
    l0 = l0 * sc0 + add0;
    l1 = l1 * sc1 + add1;
    {
      const __half2 h0 = __float2half2_rn(sc0), h1 = __float2half2_rn(sc1);
#pragma unroll
      for (int i = 0; i < AD / 8; i++) {
        __half2 v0 = *(__half2*)&oacc[i][0], v1 = *(__half2*)&oacc[i][1];
        v0 = __hmul2(v0, h0); v1 = __hmul2(v1, h1);
        oacc[i][0] = *(uint32_t*)&v0; oacc[i][1] = *(uint32_t*)&v1;
      }
    }
#pragma unroll
    for (int kk = 0; kk < PT / 16; kk++) {
      uint32_t pa[4];
      pa[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dn = 0; dn < AD / 16; dn++) {
        uint32_t b[4];
        ldsm_x4_t(b, sV + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * ROWB + (dn * 16 + (lane >> 4) * 8) * 2);
        mma16816_h(oacc[2 * dn], pa, b[0], b[1]);
        mma16816_h(oacc[2 * dn + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();  // this stage is overwritten by the load issued in the next iteration
  }
  cp_async_wait<0>();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const int h = kh * GQA + warp;
  const int ta = t0 + g, tb = t0 + g + 8;
#pragma unroll
  for (int i = 0; i < AD / 8; i++) {
    const int d = i * 8 + 2 * tg;
    const float2 v0 = __half22float2(*(__half2*)&oacc[i][0]);
    const float2 v1 = __half22float2(*(__half2*)&oacc[i][1]);
    if (ta < M) {
      const float* gt = qg + ((size_t)ta * nqh + h) * 512 + 256 + d;
      float2 r;
      r.x = (v0.x / l0) * (1.0f / (1.0f + expf(-gt[0])));
      r.y = (v0.y / l0) * (1.0f / (1.0f + expf(-gt[1])));
      *(float2*)(o + ((size_t)ta * nqh + h) * AD + d) = r;
    }
    if (tb < M) {
      const float* gt = qg + ((size_t)tb * nqh + h) * 512 + 256 + d;
      float2 r;
      r.x = (v1.x / l1) * (1.0f / (1.0f + expf(-gt[0])));
      r.y = (v1.y / l1) * (1.0f / (1.0f + expf(-gt[1])));
      *(float2*)(o + ((size_t)tb * nqh + h) * AD + d) = r;
    }
  }
}

// ---------------------------------------------------------------- reference (simple, slow)
// One block of 256 threads per (query head, token): plain softmax over all visible positions in f32,
// with the same f16 roundings (Q, P). Used by tools/bench_attn to check the fast kernel.
__device__ __forceinline__ float cache_val(const void* c, bool q8, int nkvh, int n_ctx, int kh, int p, int i) {
  if (!q8) return __half2float(((const __half*)c)[((size_t)kh * n_ctx + p) * AD + i]);
  const int8_t q = ((const int8_t*)c)[((size_t)kh * n_ctx + p) * AD + i];
  const __half d = ((const __half*)((const uint8_t*)c + (size_t)nkvh * n_ctx * AD))[((size_t)kh * n_ctx + p) * 8 + i / 32];
  return __half2float(__float2half((float)q * __half2float(d)));
}
__global__ void attn_ref_kernel(const float* __restrict__ qn, const float* __restrict__ qg, const void* __restrict__ kc,
                                const void* __restrict__ vc, float* __restrict__ o, float* __restrict__ scores,
                                const int* __restrict__ dpos0, int nqh, int nkvh, int n_ctx, float kq_scale, bool q8) {
  pdl_wait();
  pdl_trigger();
  const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  const int kh = h / GQA;
  const int n = *dpos0 + t + 1;
  __shared__ float q[AD];
  __shared__ float red[32];
  q[d] = __half2float(__float2half(qn[((size_t)t * nqh + h) * AD + d] * kq_scale));
  __syncthreads();
  float* sc = scores + ((size_t)t * nqh + h) * n_ctx;
  float mx = -INFINITY;
  for (int p = d; p < n; p += AD) {
    float a = 0.f;
    for (int i = 0; i < AD; i++) a += q[i] * cache_val(kc, q8, nkvh, n_ctx, kh, p, i);
    sc[p] = a;
    mx = fmaxf(mx, a);
  }
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  if ((d & 31) == 0) red[d >> 5] = mx;
  __syncthreads();
  mx = red[0];
  for (int i = 1; i < AD / 32; i++) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float L = 0.f, A = 0.f;
  for (int p = 0; p < n; p++) {
    const float e = expf(sc[p] - mx);
    L += e;
    A += __half2float(__float2half(e)) * cache_val(vc, q8, nkvh, n_ctx, kh, p, d);
  }
  const float gate = qg[((size_t)t * nqh + h) * 512 + 256 + d];
  o[((size_t)t * nqh + h) * AD + d] = (A / L) * (1.0f / (1.0f + expf(-gate)));
}

struct AttnWs {
  float* part = nullptr;
  int nch = 0;
  float* scores = nullptr;
  size_t scores_n = 0;
};
AttnWs& attn_ws() {
  static AttnWs ws[16];
  int dev; CK(cudaGetDevice(&dev));
  return ws[dev];
}

}  // namespace

int attn_chunks(int nkvh) {
  int dev, nsm;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  return std::max(1, nsm / nkvh);
}

size_t kv_cache_bytes(int nkvh, int n_ctx, bool q8) {
  return q8 ? (size_t)nkvh * n_ctx * (AD + 16) : (size_t)nkvh * n_ctx * AD * 2;
}

void attn_decode(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, int8_t* xq, float* xd,
                 const int* pos0, int T, float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s) {
  AttnWs& ws = attn_ws();
  const int nch = attn_chunks(nkvh);
  if (!ws.part) {
    CK(cudaMalloc(&ws.part, sizeof(float) * (size_t)nkvh * nch * MAXR * PROW));
    ws.nch = nch;
    CK(cudaFuncSetAttribute(attn_mma_kernel<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem(1, false)));
    CK(cudaFuncSetAttribute(attn_mma_kernel<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem(2, false)));
    CK(cudaFuncSetAttribute(attn_mma_kernel<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem(1, true)));
    CK(cudaFuncSetAttribute(attn_mma_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem(2, true)));
  }
  if (T < 1 || T > 4 || nqh != nkvh * GQA) throw std::runtime_error("attn_decode: bad shape");
  const dim3 grid(nch, nkvh);
  auto launch = [&](auto kern, int mt, int threads) {
    launch_k(kern, grid, threads, attn_smem(mt, q8), s, qn, kcache, vcache, ws.part, pos0, T, nqh, n_ctx, nch, kq_scale);
  };
  if (T == 1) {
    if (q8) launch(attn_mma_kernel<1, true>, 1, 64); else launch(attn_mma_kernel<1, false>, 1, 64);
  } else {
    if (q8) launch(attn_mma_kernel<2, true>, 2, 128); else launch(attn_mma_kernel<2, false>, 2, 128);
  }
  launch_k(attn_combine_kernel, dim3(nqh, T), AD, 0, s, ws.part, qg, o, xq, xd, nqh, nch);
  CK(cudaGetLastError());
}

void attn_prefill(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, const int* pos0, int M,
                  float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s) {
  static bool init[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!init[dev]) {
    CK(cudaFuncSetAttribute(attn_prefill_kernel<32, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf_smem<32, false>()));
    CK(cudaFuncSetAttribute(attn_prefill_kernel<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf_smem<32, true>()));
    init[dev] = true;
  }
  if (nqh != nkvh * GQA || M < 1) throw std::runtime_error("attn_prefill: bad shape");
  const dim3 grid((M + 15) / 16, nkvh);
  if (q8) launch_k(attn_prefill_kernel<32, true>, grid, 192, pf_smem<32, true>(), s, qn, qg, kcache, vcache, o, pos0, M, nqh, n_ctx, kq_scale);
  else launch_k(attn_prefill_kernel<32, false>, grid, 192, pf_smem<32, false>(), s, qn, qg, kcache, vcache, o, pos0, M, nqh, n_ctx, kq_scale);
  CK(cudaGetLastError());
}

void attn_decode_ref(const float* qn, const float* qg, const void* kcache, const void* vcache, float* o, const int* pos0,
                     int T, float kq_scale, int nqh, int nkvh, int n_ctx, bool q8, cudaStream_t s) {
  AttnWs& ws = attn_ws();
  const size_t need = (size_t)T * nqh * n_ctx;
  if (need > ws.scores_n) {
    if (ws.scores) cudaFree(ws.scores);
    CK(cudaMalloc(&ws.scores, need * sizeof(float)));
    ws.scores_n = need;
  }
  launch_k(attn_ref_kernel, dim3(nqh, T), AD, 0, s, qn, qg, kcache, vcache, o, ws.scores, pos0, nqh, nkvh, n_ctx, kq_scale, q8);
  CK(cudaGetLastError());
}

}  // namespace q27
