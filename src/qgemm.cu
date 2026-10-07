// Prefill GEMM for quantized weights: y[t][n] = sum_k W[n][k] x[t][k] for M tokens (M up to a few thousand).
// Numerics as llama.cpp MMQ (mmq-load-tiles.cuh, mmq-vec-dot.cuh; MIT, see THIRD_PARTY_NOTICES.md), except
// that the K-quant scale products are f32:
// activations are q8_1 (int8 per value, f32 scale per 32, f32 sum of the unquantized values per 16 for the
// types with a min term); weights are unpacked to exact int8 values with a float scale per 32 or per 16
// weights (and a float min for Q4_K and Q2_K); int8 tensor-core sums (mma.sync s8) are scaled in f32.
// Differences to llama.cpp are only the f32 summation order (and IQ1_M, which llama.cpp runs through
// dequantize + cuBLAS).
//
// Tiling: a CTA computes 128 weight rows x 64 tokens. For each 256-wide block of K, the 256 threads unpack
// the 128 weight rows (16 tiles of 8 rows in the QMat tile layout) and load the 64 token rows into shared
// memory; then 8 warps (4 row groups x 2 token groups, 32 x 32 outputs each) run the MMAs per 32-wide
// sub-block and apply the scales.
#include "qmat.h"
#include "qtypes.cuh"
#include "quant_tables.h"

namespace q27 {

namespace {

constexpr int RB = 256 + 16;      // shared row stride in bytes (padding: no ldmatrix bank conflicts)
constexpr int NTHR = 256;

// Per-type GEMM traits: k16 = scales per 16 weights (two k16 MMAs per sub-block), has_min = min term.
template <GType T> struct Gm { static constexpr bool k16 = false, has_min = false; };
template <> struct Gm<GType::Q4_K>   { static constexpr bool k16 = false, has_min = true; };
template <> struct Gm<GType::Q2_K>   { static constexpr bool k16 = false, has_min = true; };
template <> struct Gm<GType::Q6_K>   { static constexpr bool k16 = true,  has_min = false; };
template <> struct Gm<GType::IQ2_XS> { static constexpr bool k16 = true,  has_min = false; };
template <> struct Gm<GType::IQ2_S>  { static constexpr bool k16 = true,  has_min = false; };
template <> struct Gm<GType::IQ1_M>  { static constexpr bool k16 = true,  has_min = false; };

struct WPtr {
  const uint8_t *f0, *f1, *f2, *f3, *d;
  int N, K, nb;
};

__device__ __forceinline__ uint32_t u8(const uint8_t* p) { return *p; }
__device__ __forceinline__ uint32_t u16(const uint8_t* p) { return *(const uint16_t*)p; }
__device__ __forceinline__ uint32_t u32(const uint8_t* p) { return *(const uint32_t*)p; }
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(*(const __half*)p); }

// Pipelined form of unpack_row (same values): load_raw reads the bytes of row r of chunk (tb, s) into 8 words (the
// GEMM issues these loads one K block ahead), decode_raw turns them into the int8 weights and scales.
__device__ __forceinline__ float hbits2f(uint32_t b) { return __half2float(__ushort_as_half((unsigned short)b)); }
template <GType T>
__device__ __forceinline__ void load_raw(const WPtr& m, size_t tb, int s, int r, uint32_t (&w)[8]) {
  const size_t c = tb * 8 + s;
  if constexpr (T == GType::IQ3_S) {
    const uint2 qa = *(const uint2*)(m.f0 + c * 64 + r * 8);
    w[0] = qa.x; w[1] = qa.y; w[2] = u32(m.f1 + c * 32 + r * 4); w[3] = u8(m.f2 + c * 8 + r); w[4] = u32(m.f3 + c * 4);
    w[5] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ3_XXS) {
    const uint2 qa = *(const uint2*)(m.f0 + c * 64 + r * 8);
    w[0] = qa.x; w[1] = qa.y; w[2] = u32(m.f1 + c * 32 + r * 4); w[3] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ4_XS) {
    const uint4 q = *(const uint4*)(m.f0 + c * 128 + r * 16);
    w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w; w[4] = u16(m.d + tb * 16 + r * 2); w[5] = u8(m.f1 + c * 8 + r);
  } else if constexpr (T == GType::Q4_K) {
    const uint4 q = *(const uint4*)(m.f0 + c * 128 + r * 16);
    w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w; w[4] = u16(m.f1 + c * 16 + r * 2); w[5] = u32(m.d + tb * 32 + r * 4);
  } else if constexpr (T == GType::Q2_K) {
    const uint2 q = *(const uint2*)(m.f0 + c * 64 + r * 8);
    w[0] = q.x; w[1] = q.y; w[2] = u16(m.f1 + c * 16 + r * 2); w[3] = u32(m.d + tb * 32 + r * 4);
  } else if constexpr (T == GType::Q6_K) {
    const uint4 lo = *(const uint4*)(m.f0 + c * 128 + r * 16);
    const uint2 hi = *(const uint2*)(m.f1 + c * 64 + r * 8);
    w[0] = lo.x; w[1] = lo.y; w[2] = lo.z; w[3] = lo.w; w[4] = hi.x; w[5] = hi.y; w[6] = u16(m.f2 + c * 16 + r * 2);
    w[7] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ2_XXS) {
    const uint2 q = *(const uint2*)(m.f0 + c * 64 + r * 8);
    w[0] = q.x; w[1] = q.y; w[2] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ2_XS) {
    const uint2 q = *(const uint2*)(m.f0 + c * 64 + r * 8);
    w[0] = q.x; w[1] = q.y; w[2] = u8(m.f1 + c * 8 + r); w[3] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ2_S) {
    w[0] = u32(m.f0 + c * 32 + r * 4); w[1] = u32(m.f1 + c * 32 + r * 4); w[2] = u8(m.f2 + c * 8 + r); w[3] = u8(m.f3 + c * 8 + r);
    w[4] = u16(m.d + tb * 16 + r * 2);
  } else if constexpr (T == GType::IQ1_M) {
    w[0] = u32(m.f0 + c * 32 + r * 4); w[1] = u16(m.f1 + c * 16 + r * 2); w[2] = u8(m.f2 + c * 8 + r); w[3] = u16(m.d + tb * 16 + r * 2);
  }
}
template <GType T>
__device__ __forceinline__ void decode_raw(const uint32_t (&w)[8], int r, const uint32_t* lut, const uint32_t (&kv)[4],
                                           uint32_t (&q)[8], float (&scl)[2], float (&mn)[2]) {
  mn[0] = mn[1] = 0.f;
  if constexpr (T == GType::IQ3_S) {
    const uint32_t sg = w[2], h = w[3], nib = (w[4] >> (4 * r)) & 0xF;
#pragma unroll
    for (int l = 0; l < 8; l++) {
      const uint32_t src = l < 4 ? w[0] : w[1];
      const uint32_t idx = ((src >> (8 * (l & 3))) & 0xFF) | ((h << (8 - l)) & 0x100);
      q[l] = (uint32_t)apply_signs(lut[idx], (sg >> (4 * l)) & 0xF);
    }
    const float d = hbits2f(w[5]);
    scl[0] = scl[1] = (float)(1 + 2 * (int)nib) * d;
  } else if constexpr (T == GType::IQ3_XXS) {
    const uint32_t aux = w[2];
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t sb = ksigns((aux >> (7 * p)) & 0x7F);
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int l = 2 * p + e;
        const uint32_t src = l < 4 ? w[0] : w[1];
        q[l] = (uint32_t)apply_signs(lut[(src >> (8 * (l & 3))) & 0xFF], (sb >> (4 * e)) & 0xF);
      }
    }
    const float d = hbits2f(w[3]);
    const int ls = (int)(aux >> 28);
    scl[0] = scl[1] = (ls * d + d / 2) / 2;
  } else if constexpr (T == GType::IQ4_XS) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int2 v = table16(w[j], kv);
      q[j] = (uint32_t)v.x;
      q[j + 4] = (uint32_t)v.y;
    }
    const float d = hbits2f(w[4]);
    scl[0] = scl[1] = d * (float)(int8_t)w[5];
  } else if constexpr (T == GType::Q4_K) {
#pragma unroll
    for (int j = 0; j < 4; j++) { q[j] = w[j] & 0x0F0F0F0F; q[j + 4] = (w[j] >> 4) & 0x0F0F0F0F; }
    const uint32_t sm = w[4];
    const float2 dmf = __half22float2(*(const __half2*)&w[5]);
    scl[0] = scl[1] = dmf.x * (float)(sm & 0xFF);
    mn[0] = mn[1] = -dmf.y * (float)(sm >> 8);
  } else if constexpr (T == GType::Q2_K) {
    const uint32_t g = w[2];
    const uint32_t sc0 = g & 0xF, sc1 = (g >> 8) & 0xF;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      q[i] = ((w[0] >> (2 * i)) & 0x03030303) * sc0;
      q[4 + i] = ((w[1] >> (2 * i)) & 0x03030303) * sc1;
    }
    const float2 dmf = __half22float2(*(const __half2*)&w[3]);
    scl[0] = scl[1] = dmf.x;
    mn[0] = -dmf.y * (float)((g >> 4) & 0xF);
    mn[1] = -dmf.y * (float)((g >> 12) & 0xF);
  } else if constexpr (T == GType::Q6_K) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t qa = (w[i] & 0x0F0F0F0F) | (((w[4] >> (2 * i)) & 0x03030303) << 4);
      const uint32_t qb = ((w[i] >> 4) & 0x0F0F0F0F) | (((w[5] >> (2 * i)) & 0x03030303) << 4);
      q[i] = __vsub4(qa, 0x20202020);
      q[4 + i] = __vsub4(qb, 0x20202020);
    }
    const uint32_t g = w[6];
    const float d = hbits2f(w[7]);
    scl[0] = d * (float)(int8_t)(g & 0xFF);
    scl[1] = d * (float)(int8_t)(g >> 8);
  } else if constexpr (T == GType::IQ2_XXS) {
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t gi = (w[0] >> (8 * p)) & 0xFF;
      const uint32_t sb = ksigns((w[1] >> (7 * p)) & 0x7F);
      q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
      q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
    }
    const float d = hbits2f(w[2]);
    const int ls = (int)((w[1] >> 27) | 1);
    scl[0] = scl[1] = d * ls / 8;
  } else if constexpr (T == GType::IQ2_XS || T == GType::IQ2_S) {
    uint32_t lsb;
    float d;
    if constexpr (T == GType::IQ2_XS) {
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t q16 = ((p < 2 ? w[0] : w[1]) >> (16 * (p & 1))) & 0xFFFF;
        const uint32_t gi = q16 & 0x1FF, sb = ksigns(q16 >> 9);
        q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
        q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
      }
      lsb = w[2];
      d = hbits2f(w[3]);
    } else {
      const uint32_t qs = w[0], sg = w[1], qh = w[2];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t gi = ((qs >> (8 * p)) & 0xFF) | ((qh << (8 - 2 * p)) & 0x300);
        const uint32_t sb = (sg >> (8 * p)) & 0xFF;
        q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
        q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
      }
      lsb = w[3];
      d = hbits2f(w[4]);
    }
    scl[0] = ((lsb & 0xF) * d + d / 2) / 4;
    scl[1] = ((lsb >> 4) * d + d / 2) / 4;
  } else if constexpr (T == GType::IQ1_M) {
    const uint32_t qs = w[0], qh16 = w[1];
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t qhl = (qh16 >> (4 * p)) & 0xF;
      const uint32_t grid = lut[((qs >> (8 * p)) & 0xFF) | ((qhl & 7) << 8)];
      const uint32_t off = (qhl & 8) ? 0x09090909u : 0x07070707u;
      q[2 * p] = __vsub4((grid & 0x0F0F0F0F) << 3, off);
      q[2 * p + 1] = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, off);
    }
    const uint32_t sc6 = w[2];
    const float d = hbits2f(w[3]);
    scl[0] = d * (float)(2 * (int)(sc6 & 7) + 1) / 8;
    scl[1] = d * (float)(2 * (int)((sc6 >> 3) & 7) + 1) / 8;
  }
}

// Unpack row r (0..7) of chunk (tile-block tb, sub-block s): q = 32 int8 weights in k order, scl[h] = scale of
// the weights 16h..16h+15 (both equal for per-32 types), mn[h] = min (types with a min term).
template <GType T>
__device__ __forceinline__ void unpack_row(const WPtr& m, size_t tb, int s, int r, const uint32_t* lut,
                                           const uint32_t (&kv)[4], uint32_t (&q)[8], float (&scl)[2], float (&mn)[2]) {
  const size_t c = tb * 8 + s;
  mn[0] = mn[1] = 0.f;
  if constexpr (T == GType::IQ3_S) {
    const uint2 qa = *(const uint2*)(m.f0 + c * 64 + r * 8);
    const uint32_t sg = u32(m.f1 + c * 32 + r * 4);
    const uint32_t h = u8(m.f2 + c * 8 + r);
    const uint32_t nib = (u32(m.f3 + c * 4) >> (4 * r)) & 0xF;
#pragma unroll
    for (int l = 0; l < 8; l++) {
      const uint32_t src = l < 4 ? qa.x : qa.y;
      const uint32_t idx = ((src >> (8 * (l & 3))) & 0xFF) | ((h << (8 - l)) & 0x100);
      q[l] = (uint32_t)apply_signs(lut[idx], (sg >> (4 * l)) & 0xF);
    }
    const float d = h2f(m.d + tb * 16 + r * 2);
    scl[0] = scl[1] = (float)(1 + 2 * (int)nib) * d;
  } else if constexpr (T == GType::IQ3_XXS) {
    const uint2 qa = *(const uint2*)(m.f0 + c * 64 + r * 8);
    const uint32_t aux = u32(m.f1 + c * 32 + r * 4);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t sb = ksigns((aux >> (7 * p)) & 0x7F);
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int l = 2 * p + e;
        const uint32_t src = l < 4 ? qa.x : qa.y;
        q[l] = (uint32_t)apply_signs(lut[(src >> (8 * (l & 3))) & 0xFF], (sb >> (4 * e)) & 0xF);
      }
    }
    const float d = h2f(m.d + tb * 16 + r * 2);
    const int ls = (int)(aux >> 28);
    scl[0] = scl[1] = (ls * d + d / 2) / 2;
  } else if constexpr (T == GType::IQ4_XS) {
    const uint4 w = *(const uint4*)(m.f0 + c * 128 + r * 16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int2 v = table16(ww[j], kv);
      q[j] = (uint32_t)v.x;
      q[j + 4] = (uint32_t)v.y;
    }
    const float d = h2f(m.d + tb * 16 + r * 2);
    scl[0] = scl[1] = d * (float)(int8_t)u8(m.f1 + c * 8 + r);
  } else if constexpr (T == GType::Q4_K) {
    const uint4 w = *(const uint4*)(m.f0 + c * 128 + r * 16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) { q[j] = ww[j] & 0x0F0F0F0F; q[j + 4] = (ww[j] >> 4) & 0x0F0F0F0F; }
    const uint32_t sm = u16(m.f1 + c * 16 + r * 2);
    // d * sc and dmin * m in f32 (llama.cpp MMQ multiplies them in half precision; f32 is closer to the exact
    // product and to the decode GEMV)
    const float2 dmf = __half22float2(*(const __half2*)(m.d + tb * 32 + r * 4));
    scl[0] = scl[1] = dmf.x * (float)(sm & 0xFF);
    mn[0] = mn[1] = -dmf.y * (float)(sm >> 8);
  } else if constexpr (T == GType::Q2_K) {
    // weights pre-multiplied by the 4-bit scale of their 16-group (q*sc <= 45 fits int8); scale = d
    const uint2 w = *(const uint2*)(m.f0 + c * 64 + r * 8);
    const uint32_t g = u16(m.f1 + c * 16 + r * 2);
    const uint32_t sc0 = g & 0xF, sc1 = (g >> 8) & 0xF;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      q[i] = ((w.x >> (2 * i)) & 0x03030303) * sc0;
      q[4 + i] = ((w.y >> (2 * i)) & 0x03030303) * sc1;
    }
    const float2 dmf = __half22float2(*(const __half2*)(m.d + tb * 32 + r * 4));
    scl[0] = scl[1] = dmf.x;
    mn[0] = -dmf.y * (float)((g >> 4) & 0xF);
    mn[1] = -dmf.y * (float)((g >> 12) & 0xF);
  } else if constexpr (T == GType::Q6_K) {
    const uint4 lo = *(const uint4*)(m.f0 + c * 128 + r * 16);
    const uint2 hi = *(const uint2*)(m.f1 + c * 64 + r * 8);
    const uint32_t ll[4] = {lo.x, lo.y, lo.z, lo.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t qa = (ll[i] & 0x0F0F0F0F) | (((hi.x >> (2 * i)) & 0x03030303) << 4);
      const uint32_t qb = ((ll[i] >> 4) & 0x0F0F0F0F) | (((hi.y >> (2 * i)) & 0x03030303) << 4);
      q[i] = __vsub4(qa, 0x20202020);
      q[4 + i] = __vsub4(qb, 0x20202020);
    }
    const uint32_t g = u16(m.f2 + c * 16 + r * 2);
    const float d = h2f(m.d + tb * 16 + r * 2);
    scl[0] = d * (float)(int8_t)(g & 0xFF);
    scl[1] = d * (float)(int8_t)(g >> 8);
  } else if constexpr (T == GType::IQ2_XXS) {
    const uint2 w = *(const uint2*)(m.f0 + c * 64 + r * 8);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t gi = (w.x >> (8 * p)) & 0xFF;
      const uint32_t sb = ksigns((w.y >> (7 * p)) & 0x7F);
      q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
      q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
    }
    const float d = h2f(m.d + tb * 16 + r * 2);
    const int ls = (int)((w.y >> 27) | 1);
    scl[0] = scl[1] = d * ls / 8;
  } else if constexpr (T == GType::IQ2_XS || T == GType::IQ2_S) {
    uint32_t lsb;
    if constexpr (T == GType::IQ2_XS) {
      const uint2 w = *(const uint2*)(m.f0 + c * 64 + r * 8);
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t q16 = ((p < 2 ? w.x : w.y) >> (16 * (p & 1))) & 0xFFFF;
        const uint32_t gi = q16 & 0x1FF, sb = ksigns(q16 >> 9);
        q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
        q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
      }
      lsb = u8(m.f1 + c * 8 + r);
    } else {
      const uint32_t qs = u32(m.f0 + c * 32 + r * 4), sg = u32(m.f1 + c * 32 + r * 4);
      const uint32_t qh = u8(m.f2 + c * 8 + r);
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t gi = ((qs >> (8 * p)) & 0xFF) | ((qh << (8 - 2 * p)) & 0x300);
        const uint32_t sb = (sg >> (8 * p)) & 0xFF;
        q[2 * p] = (uint32_t)apply_signs(lut[2 * gi], sb & 0xF);
        q[2 * p + 1] = (uint32_t)apply_signs(lut[2 * gi + 1], sb >> 4);
      }
      lsb = u8(m.f3 + c * 8 + r);
    }
    const float d = h2f(m.d + tb * 16 + r * 2);
    scl[0] = ((lsb & 0xF) * d + d / 2) / 4;
    scl[1] = ((lsb >> 4) * d + d / 2) / 4;
  } else if constexpr (T == GType::IQ1_M) {
    // value = g + delta with delta = -7/8 or -9/8 per 8 weights: stored as 8g - 7 or 8g - 9, scale / 8
    const uint32_t qs = u32(m.f0 + c * 32 + r * 4);
    const uint32_t qh16 = u16(m.f1 + c * 16 + r * 2);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t qhl = (qh16 >> (4 * p)) & 0xF;
      const uint32_t grid = lut[((qs >> (8 * p)) & 0xFF) | ((qhl & 7) << 8)];
      const uint32_t off = (qhl & 8) ? 0x09090909u : 0x07070707u;
      q[2 * p] = __vsub4((grid & 0x0F0F0F0F) << 3, off);
      q[2 * p + 1] = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, off);
    }
    const uint32_t sc6 = u8(m.f2 + c * 8 + r);
    const float d = h2f(m.d + tb * 16 + r * 2);
    scl[0] = d * (float)(2 * (int)(sc6 & 7) + 1) / 8;
    scl[1] = d * (float)(2 * (int)((sc6 >> 3) & 7) + 1) / 8;
  }
}

// CTA barrier as volatile asm with a memory clobber: nvcc 13.4 removed a plain __syncthreads() between the
// shared-memory tile stores and the ldmatrix reads in this kernel (seen in the PTX), which made the GEMM racy.
__device__ __forceinline__ void cta_sync() { asm volatile("bar.sync 0;" ::: "memory"); }
__device__ __forceinline__ void cta_sync1() { asm volatile("bar.sync 1, 256;" ::: "memory"); }  // second barrier id

// int32 -> float without I2F (conversions run at 16 per clock per SM on this GPU family): for |v| < 2^22,
// float(v) = bits(v + 0x4B400000) - 12582912.0f exactly. MMA sums here are below 2^22 (32 x 127 x 127 < 2^19).
__device__ __forceinline__ float i2f_fast(int v) { return __int_as_float(v + 0x4B400000) - 12582912.0f; }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_k32(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
}

// Tile configuration: CBW weight rows x CBT tokens per CTA, WR x WT warps (each warp (CBW / WR) rows x (CBT / WT) tokens),
// MINB CTAs per SM (register bound for __launch_bounds__).
template <int CBW_, int CBT_, int WR_, int WT_, int MINB_>
struct GCfg {
  static constexpr int CBW = CBW_, CBT = CBT_, WR = WR_, WT = WT_, MINB = MINB_, NTHR = WR * WT * 32;
  static constexpr int MT = CBW / WR / 16, NT = CBT / WT / 8;  // m16 tiles and n8 tiles per warp
  static_assert(NT % 2 == 0, "B fragments come in pairs of n8 tiles");
};

template <GType T, class C>
constexpr int gemm_smem() {
  constexpr bool MIN = Gm<T>::has_min;
  constexpr int lut = Tr<T>::lut_n > 0 ? Tr<T>::lut_n * 4 : 16;
  return C::CBW * RB + C::CBW * 16 * 4 * (MIN ? 2 : 1) + C::CBT * RB + C::CBT * 8 * 4 + (MIN ? C::CBT * 16 * 4 : 0) + lut;
}

template <GType T, class C>
__global__ void __launch_bounds__(C::NTHR, C::MINB) qgemm_kernel(WPtr m, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                              const float* __restrict__ xs, float* __restrict__ y, int M,
                                                              const uint32_t* __restrict__ tables, int dbg) {
  constexpr bool K16 = Gm<T>::k16, MIN = Gm<T>::has_min;
  constexpr int NTHR = C::NTHR;
  constexpr int CBW = C::CBW, CBT = C::CBT, MT = C::MT, NT = C::NT;
  constexpr int WROWS = CBW / C::WR, WTOK = CBT / C::WT;
  extern __shared__ __align__(16) uint8_t sm[];
  uint8_t* sW = sm;                                        // [CBW][RB] int8 weights
  float* sWs = (float*)(sW + CBW * RB);                    // [CBW][16] scale per 16 weights
  float* sWm = sWs + CBW * 16;                             // [CBW][16] min per 16 weights (MIN)
  uint8_t* sX = (uint8_t*)(sWm + (MIN ? CBW * 16 : 0));    // [CBT][RB] int8 activations
  float* sXd = (float*)(sX + CBT * RB);                     // [CBT][8] activation scale per 32
  float* sXs = sXd + CBT * 8;                              // [CBT][16] activation sum per 16 (MIN)
  uint32_t* lut = (uint32_t*)(sXs + (MIN ? CBT * 16 : 0));

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // Braces after every `if constexpr`: nvcc 13.4 dropped the statement that follows a discarded
  // `if constexpr (false) for (...) {...}` (a missing barrier made this kernel racy).
  if constexpr (Tr<T>::lut_n > 0) {
    for (int i = tid; i < Tr<T>::lut_n; i += NTHR) lut[i] = tables[Tr<T>::lut_off + i];
  }
  uint32_t kv[4] = {0, 0, 0, 0};
  if constexpr (T == GType::IQ4_XS) {
#pragma unroll
    for (int i = 0; i < 4; i++) kv[i] = tables[TB_IQ4NL + i];
  }
  pdl_wait();
  pdl_trigger();

  const int K = m.K, nb = m.nb;
  // blockIdx.x = token tile (fastest), blockIdx.y = weight tile: the CTAs that share a weight tile run at the same time,
  // so the weights come from DRAM once and from L2 for the other token tiles.
  const int wtile = blockIdx.y;
  const int tile0 = wtile * (CBW / 8), tok0 = blockIdx.x * CBT;
  const int wr = warp % C::WR, wt = warp / C::WR;
  const int g = lane >> 2, tg = lane & 3;

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; a++)
#pragma unroll
    for (int b = 0; b < NT; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  // Loads run one K block ahead: the raw weight words and the activations of block kb + 1 are read into registers
  // before the MMAs of block kb, so their memory latency hides behind the tensor-core work.
  constexpr int UPT = CBW * 8 / NTHR, XPT = CBT * 16 / NTHR, DPT = (CBT * 8 + NTHR - 1) / NTHR, SPT = MIN ? CBT * 16 / NTHR : 0;
  static_assert(CBW * 8 % NTHR == 0 && CBT * 16 % NTHR == 0, "tile sizes");
  uint32_t raw[UPT][8];
  uint4 xv[XPT];
  float dv[DPT];
  [[maybe_unused]] float sv[SPT > 0 ? SPT : 1];
  auto load_block = [&](int kb) {
#pragma unroll
    for (int j = 0; j < UPT; j++) {
      const int u = tid + j * NTHR, r = u & 7, s = (u >> 3) & 7, tl = u >> 6;
      load_raw<T>(m, (size_t)(tile0 + tl) * nb + kb, s, r, raw[j]);
    }
#pragma unroll
    for (int j = 0; j < XPT; j++) {
      const int i = tid + j * NTHR, t = i >> 4, c = i & 15, tok = tok0 + t;
      xv[j] = tok < M ? *(const uint4*)(xq + (size_t)tok * K + kb * 256 + c * 16) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < DPT; j++) {
      const int i = tid + j * NTHR, t = i >> 3, tok = tok0 + t;
      dv[j] = (i < CBT * 8 && tok < M) ? __ldcg(xd + (size_t)tok * (K / 32) + kb * 8 + (i & 7)) : 0.f;
    }
    if constexpr (MIN) {
#pragma unroll
      for (int j = 0; j < SPT; j++) {
        const int i = tid + j * NTHR, t = i >> 4, tok = tok0 + t;
        sv[j] = tok < M ? xs[(size_t)tok * (K / 16) + kb * 16 + (i & 15)] : 0.f;
      }
    }
  };
  load_block(0);

  for (int kb = 0; kb < nb; kb++) {
    cta_sync();
    // weights: CBW * 8 (row, sub-block) units; 8 consecutive lanes = the 8 rows of one chunk
#pragma unroll
    for (int j = 0; j < UPT; j++) {
      if ((dbg & 2) && kb > 0) break;
      const int u = tid + j * NTHR, r = u & 7, s = (u >> 3) & 7, tl = u >> 6;
      uint32_t q[8];
      float scl[2], mn[2];
      decode_raw<T>(raw[j], r, lut, kv, q, scl, mn);
      uint8_t* dst = sW + (tl * 8 + r) * RB + s * 32;
      *(uint4*)dst = make_uint4(q[0], q[1], q[2], q[3]);
      *(uint4*)(dst + 16) = make_uint4(q[4], q[5], q[6], q[7]);
      *(float2*)(sWs + (tl * 8 + r) * 16 + 2 * s) = make_float2(scl[0], scl[1]);
      if constexpr (MIN) { *(float2*)(sWm + (tl * 8 + r) * 16 + 2 * s) = make_float2(mn[0], mn[1]); }
    }
#pragma unroll
    for (int j = 0; j < XPT; j++) {
      const int i = tid + j * NTHR, t = i >> 4, c = i & 15;
      *(uint4*)(sX + t * RB + c * 16) = xv[j];
    }
#pragma unroll
    for (int j = 0; j < DPT; j++) {
      const int i = tid + j * NTHR;
      if (i < CBT * 8) sXd[i] = dv[j];
    }
    if constexpr (MIN) {
#pragma unroll
      for (int j = 0; j < SPT; j++) sXs[tid + j * NTHR] = sv[j];
    }
    asm volatile("bar.sync 1, %0;" ::"n"(NTHR) : "memory");  // second barrier id
    if (kb + 1 < nb && !(dbg & 4)) load_block(kb + 1);

#pragma unroll 2
    for (int sb = 0; sb < 8; sb++) {
      if (dbg & 1) break;
      uint32_t a[MT][4], b[NT / 2][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
        ldsm_x4(a[mt], sW + (wr * WROWS + mt * 16 + (lane & 15)) * RB + sb * 32 + (lane >> 4) * 16);
#pragma unroll
      for (int np = 0; np < NT / 2; np++)
        ldsm_x4(b[np], sX + (wt * WTOK + np * 16 + (lane & 7) + (lane >> 4) * 8) * RB + sb * 32 + ((lane >> 3) & 1) * 16);
      // scales of this thread's rows (g, g+8 per m-tile) and tokens (2tg, 2tg+1 per n-tile)
      float ws[MT][2][2], dy[NT][2];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int row = wr * WROWS + mt * 16 + g + 8 * h;
          const float2 v = *(const float2*)(sWs + row * 16 + 2 * sb);
          ws[mt][h][0] = v.x; ws[mt][h][1] = v.y;
        }
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) dy[nt][e] = sXd[(wt * WTOK + nt * 8 + 2 * tg + e) * 8 + sb];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          const uint32_t* bb = b[nt >> 1];
          const int o = (nt & 1) * 2;
          if constexpr (!K16) {
            int ci[4] = {0, 0, 0, 0};
            mma_k32(ci, a[mt], bb[o], bb[o + 1]);
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] += ws[mt][e >> 1][0] * (dy[nt][e & 1] * i2f_fast(ci[e]));
          } else {
            int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
            mma_k16(lo, a[mt][0], a[mt][1], bb[o]);
            mma_k16(hi, a[mt][2], a[mt][3], bb[o + 1]);
#pragma unroll
            for (int e = 0; e < 4; e++)
              acc[mt][nt][e] += ws[mt][e >> 1][0] * (dy[nt][e & 1] * i2f_fast(lo[e])) + ws[mt][e >> 1][1] * (dy[nt][e & 1] * i2f_fast(hi[e]));
          }
          if constexpr (MIN) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
              const int row = wr * WROWS + mt * 16 + g + 8 * (e >> 1), t = wt * WTOK + nt * 8 + 2 * tg + (e & 1);
              const float2 mv = *(const float2*)(sWm + row * 16 + 2 * sb);
              const float2 sv = *(const float2*)(sXs + t * 16 + 2 * sb);
              acc[mt][nt][e] += mv.x * sv.x + mv.y * sv.y;
            }
          }
        }
    }
    cta_sync();
  }
  // write y[t][n]
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int row = wtile * CBW + wr * WROWS + mt * 16 + g + 8 * (e >> 1);
        const int t = tok0 + wt * WTOK + nt * 8 + 2 * tg + (e & 1);
        if (t < M) y[(size_t)t * m.N + row] = acc[mt][nt][e];
      }
}

// Q27_GEMM_CFG selects the tile configuration (tests): 0 = 128 x 64 (1 CTA per SM), 1 = 64 x 64 (2 per SM),
// 2 = 64 x 128, 3 = 128 x 128, 4 = 128 x 128 with 16 warps, 5 = 128 x 64 with 16 warps; unset = per type and M.
static int gemm_dbg() {
  static const int v = [] { const char* e = getenv("Q27_GEMM_DBG2"); return e ? atoi(e) : 0; }();
  return v;
}
static int gemm_cfg() {
  static const int v = [] { const char* e = getenv("Q27_GEMM_CFG"); return e ? atoi(e) : -1; }();
  return v;
}

template <GType T, class C>
void gemm_launch_c(const QMat& w, const int8_t* xq, const float* xd, const float* xs, float* y, int M, cudaStream_t s) {
  static bool init[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!init[dev]) {
    CK(cudaFuncSetAttribute(qgemm_kernel<T, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_smem<T, C>()));
    init[dev] = true;
  }
  if (w.N % C::CBW) throw std::runtime_error("qgemm: N must be a multiple of the tile rows");
  if (Gm<T>::has_min && !xs) throw std::runtime_error("qgemm: this type needs activation sums");
  const WPtr p{w.f[0], w.f[1], w.f[2], w.f[3], w.d, w.N, w.K, w.nb};
  launch_k(qgemm_kernel<T, C>, dim3((M + C::CBT - 1) / C::CBT, w.N / C::CBW), C::NTHR, gemm_smem<T, C>(), s, p, xq, xd, xs, y, M,
           device_tables(), gemm_dbg());
}

template <GType T>
void gemm_launch(const QMat& w, const int8_t* xq, const float* xd, const float* xs, float* y, int M, cudaStream_t s) {
  // Default (2026-10-07, bench_gemm at 1024 rows): 128 x 128 with 16 warps from 256 rows (+5-11% on the IQ types),
  // except Q6_K and Q4_K (128 x 64 is as fast or faster there).
  int cfg = gemm_cfg();
  if (cfg < 0) cfg = (M >= 256 && T != GType::Q6_K && T != GType::Q4_K) ? 4 : 0;
  switch (cfg) {
    case 1: gemm_launch_c<T, GCfg<64, 64, 2, 4, 2>>(w, xq, xd, xs, y, M, s); break;
    case 2: gemm_launch_c<T, GCfg<64, 128, 2, 4, 1>>(w, xq, xd, xs, y, M, s); break;
    case 3: gemm_launch_c<T, GCfg<128, 128, 4, 2, 1>>(w, xq, xd, xs, y, M, s); break;
    case 4: gemm_launch_c<T, GCfg<128, 128, 4, 4, 1>>(w, xq, xd, xs, y, M, s); break;  // 16 warps
    case 5: gemm_launch_c<T, GCfg<128, 64, 4, 4, 1>>(w, xq, xd, xs, y, M, s); break;   // 16 warps
    default: gemm_launch_c<T, GCfg<128, 64, 4, 2, 1>>(w, xq, xd, xs, y, M, s); break;
  }
}

}  // namespace

bool qgemm_needs_sums(GType t) { return t == GType::Q4_K || t == GType::Q2_K; }

void qgemm(const QMat& w, const int8_t* xq, const float* xd, const float* xs, float* y, int M, cudaStream_t s) {
  switch (w.type) {
    case GType::IQ3_S: gemm_launch<GType::IQ3_S>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ3_XXS: gemm_launch<GType::IQ3_XXS>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ4_XS: gemm_launch<GType::IQ4_XS>(w, xq, xd, xs, y, M, s); break;
    case GType::Q4_K: gemm_launch<GType::Q4_K>(w, xq, xd, xs, y, M, s); break;
    case GType::Q2_K: gemm_launch<GType::Q2_K>(w, xq, xd, xs, y, M, s); break;
    case GType::Q6_K: gemm_launch<GType::Q6_K>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ2_XXS: gemm_launch<GType::IQ2_XXS>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ2_XS: gemm_launch<GType::IQ2_XS>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ2_S: gemm_launch<GType::IQ2_S>(w, xq, xd, xs, y, M, s); break;
    case GType::IQ1_M: gemm_launch<GType::IQ1_M>(w, xq, xd, xs, y, M, s); break;
    default: throw std::runtime_error(std::string("qgemm: type not supported: ") + gtype_name(w.type));
  }
  CK(cudaGetLastError());
}

}  // namespace q27
