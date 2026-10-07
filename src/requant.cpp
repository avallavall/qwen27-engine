#include "requant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace q27 {
namespace {

constexpr int QK = 256;

// IEEE half <-> float, round to nearest even (what F16C and ggml's GGML_FP32_TO_FP16 do).
uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  uint32_t mant = x & 0x7fffff;
  const int exp = (int)((x >> 23) & 0xff);
  if (exp == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
  const int e = exp - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  if (e <= 0) {  // half subnormal
    if (e < -10) return (uint16_t)sign;
    mant |= 0x800000;
    const int shift = 14 - e;
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;  // a carry into the exponent is the right result
  return (uint16_t)(sign | h);
}

float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  int exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) x = sign;
    else {  // subnormal: normalize
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3ff;
      x = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) x = sign | 0x7f800000 | (mant << 13);
  else x = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
  float f; memcpy(&f, &x, 4);
  return f;
}

uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
void wr16(uint8_t* p, uint16_t v) { memcpy(p, &v, 2); }

int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i; memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// ---- dequantize one 256-weight block

// Q6_K block (210 bytes): ql[128], qh[64], scales[16] (int8), d (half).
void deq_q6_k(const uint8_t* b, float* y) {
  const float d = h2f(rd16(b + 208));
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = (const int8_t*)(b + 192);
  for (int n = 0; n < QK; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128; ql += 64; qh += 32; sc += 8;
  }
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
  else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Q4_K block (144 bytes): d (half), dmin (half), scales[12], qs[128].
void deq_q4_k(const uint8_t* b, float* y) {
  const float d = h2f(rd16(b)), mn = h2f(rd16(b + 2));
  const uint8_t* scales = b + 4;
  const uint8_t* q = b + 16;
  int is = 0;
  for (int j = 0; j < QK; j += 64) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc, m1 = mn * m;
    get_scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32; is += 2;
  }
}

// ---- quantize one 256-weight block

float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min, uint8_t* Laux,
                       float rmin, float rdelta, int nstep) {
  float min = x[0], max = x[0];
  float sum_w = weights[0], sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    const float w = weights[i];
    sum_w += w; sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min), scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < n; ++i) {
    const int l = nearest_int(iscale * (x[i] - min));
    L[i] = (uint8_t)std::max(0, std::min(nmax, l));
    const float diff = scale * L[i] + min - x[i];
    best_error += weights[i] * diff * diff;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = std::max(0, std::min(nmax, l));
      Laux[i] = (uint8_t)l;
      const float w = weights[i];
      sum_l += w * l; sum_l2 += w * l * l; sum_xl += w * l * x[i];
    }
    const float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
      float cur_error = 0;
      for (int i = 0; i < n; ++i) {
        const float diff = this_scale * Laux[i] + this_min - x[i];
        cur_error += weights[i] * diff * diff;
      }
      if (cur_error < best_error) {
        for (int i = 0; i < n; ++i) L[i] = Laux[i];
        best_error = cur_error; scale = this_scale; min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

void q_q4_k(const float* x, uint8_t* b) {
  uint8_t L[QK], Laux[32];
  float weights[32], mins[QK / 32], scales[QK / 32];
  float max_scale = 0, max_min = 0;
  for (int j = 0; j < QK / 32; ++j) {
    float sum_x2 = 0;
    for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
    const float av_x = sqrtf(sum_x2 / 32);
    for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
    scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20);
    max_scale = std::max(max_scale, scales[j]);
    max_min = std::max(max_min, mins[j]);
  }
  uint8_t* sc12 = b + 4;
  memset(sc12, 0, 12);
  const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
  const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
  for (int j = 0; j < QK / 32; ++j) {
    const uint8_t ls = (uint8_t)std::min(63, nearest_int(inv_scale * scales[j]));
    const uint8_t lm = (uint8_t)std::min(63, nearest_int(inv_min * mins[j]));
    if (j < 4) { sc12[j] = ls; sc12[j + 4] = lm; }
    else {
      sc12[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
      sc12[j - 4] |= ((ls >> 4) << 6);
      sc12[j - 0] |= ((lm >> 4) << 6);
    }
  }
  wr16(b, f2h(max_scale / 63.f));
  wr16(b + 2, f2h(max_min / 63.f));
  const float dd = h2f(rd16(b)), dmn = h2f(rd16(b + 2));
  for (int j = 0; j < QK / 32; ++j) {
    uint8_t sc, m;
    get_scale_min_k4(j, sc12, &sc, &m);
    const float d = dd * sc;
    if (!d) continue;
    const float dm = dmn * m;
    for (int ii = 0; ii < 32; ++ii) L[32 * j + ii] = (uint8_t)std::max(0, std::min(15, nearest_int((x[32 * j + ii] + dm) / d)));
  }
  uint8_t* q = b + 16;
  for (int j = 0; j < QK; j += 64) {
    for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
    q += 32;
  }
}

const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

int best_index_int8(int n, const int8_t* val, float x) {
  if (x <= val[0]) return 0;
  if (x >= val[n - 1]) return n - 1;
  int ml = 0, mu = n - 1;
  while (mu - ml > 1) {
    const int mav = (ml + mu) / 2;
    if (x < val[mav]) mu = mav; else ml = mav;
  }
  return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

// IQ4_XS block (136 bytes): d (half), scales_h (u16), scales_l[4], qs[128]. quantize_row_iq4_nl_impl(QK_K, 32, ...)
// with no importance weights and ntry = 7, as quantize_iq4_xs does.
void q_iq4_xs(const float* x, uint8_t* b) {
  const int8_t* values = kvalues_iq4nl;
  const int ntry = 7;
  uint8_t L[QK];
  float scales[QK / 32], weight[32];
  uint8_t* q4 = b + 8;
  uint8_t* scales_l = b + 4;
  memset(q4, 0, QK / 2);
  wr16(b, f2h(0.f));
  float max_scale = 0, amax_scale = 0;
  for (int ib = 0; ib < QK / 32; ++ib) {
    const float* xb = x + ib * 32;
    uint8_t* Lb = L + ib * 32;
    for (int j = 0; j < 32; ++j) weight[j] = xb[j] * xb[j];
    float amax = 0, max = 0;
    for (int j = 0; j < 32; ++j) {
      const float ax = fabsf(xb[j]);
      if (ax > amax) { amax = ax; max = xb[j]; }
    }
    if (amax < 1e-15f) { scales[ib] = 0; continue; }
    float d = -max / values[0];
    float id = 1 / d;
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < 32; ++j) {
      const int l = best_index_int8(16, values, id * xb[j]);
      Lb[j] = (uint8_t)l;
      const float q = values[l], w = weight[j];
      sumqx += w * q * xb[j]; sumq2 += w * q * q;
    }
    d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
    float best = d * sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry) {
      id = (itry + values[0]) / max;
      sumqx = sumq2 = 0;
      for (int j = 0; j < 32; ++j) {
        const int l = best_index_int8(16, values, id * xb[j]);
        const float q = values[l], w = weight[j];
        sumqx += w * q * xb[j]; sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) { d = sumqx / sumq2; best = d * sumqx; }
    }
    scales[ib] = d;
    const float abs_d = fabsf(d);
    if (abs_d > amax_scale) { amax_scale = abs_d; max_scale = d; }
  }
  uint16_t scales_h = 0;
  memset(scales_l, 0, QK / 64);
  const float d = -max_scale / 32;
  wr16(b, f2h(d));
  const float id = d ? 1 / d : 0.f;
  for (int ib = 0; ib < QK / 32; ++ib) {
    int l = nearest_int(id * scales[ib]);
    l = std::max(-32, std::min(31, l));
    const float dl = d * l;
    const float idl = dl ? 1 / dl : 0.f;
    uint8_t* Lb = L + ib * 32;
    const float* xb = x + ib * 32;
    for (int j = 0; j < 32; ++j) Lb[j] = (uint8_t)best_index_int8(16, values, idl * xb[j]);
    l += 32;
    const uint8_t l_l = l & 0xf, l_h = (uint8_t)(l >> 4);
    if (ib % 2 == 0) scales_l[ib / 2] = l_l;
    else scales_l[ib / 2] |= (l_l << 4);
    scales_h |= (uint16_t)(l_h << 2 * (ib % 8));
  }
  wr16(b + 2, scales_h);
  for (int i = 0; i < QK / 32; ++i)
    for (int j = 0; j < 16; ++j) q4[16 * i + j] = L[32 * i + j] | (L[32 * i + 16 + j] << 4);
}

}  // namespace

GType parse_gtype(const std::string& s) {
  std::string l;
  for (char c : s) l += (char)tolower((unsigned char)c);
  if (l == "q4_k") return GType::Q4_K;
  if (l == "iq4_xs") return GType::IQ4_XS;
  if (l == "q6_k") return GType::Q6_K;
  throw std::runtime_error("unknown quant type " + s + " (q4_k, iq4_xs or q6_k)");
}

std::vector<uint8_t> requant(const GTensor& t, GType to, int threads) {
  if (t.type != GType::Q6_K && t.type != GType::Q4_K) throw std::runtime_error("requant: source " + t.name + " must be Q6_K or Q4_K");
  if (to != GType::Q4_K && to != GType::IQ4_XS) throw std::runtime_error("requant: target must be Q4_K or IQ4_XS");
  const int64_t K = t.ne[0], nb = K / QK, rows = t.rows();
  const int src_bb = gtype_block_bytes(t.type), dst_bb = gtype_block_bytes(to);
  std::vector<uint8_t> out((size_t)(rows * nb * dst_bb));
  auto work = [&](int64_t r0, int64_t r1) {
    float x[QK];
    for (int64_t r = r0; r < r1; r++)
      for (int64_t b = 0; b < nb; b++) {
        const uint8_t* src = t.data + (size_t)(r * nb + b) * src_bb;
        uint8_t* dst = out.data() + (size_t)(r * nb + b) * dst_bb;
        if (t.type == GType::Q6_K) deq_q6_k(src, x); else deq_q4_k(src, x);
        if (to == GType::Q4_K) q_q4_k(x, dst); else q_iq4_xs(x, dst);
      }
  };
  const int nt = threads > 0 ? threads : (int)std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (int i = 0; i < nt; i++) th.emplace_back(work, rows * i / nt, rows * (i + 1) / nt);
  for (auto& x : th) x.join();
  return out;
}

}  // namespace q27
