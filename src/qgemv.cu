// Decode GEMV for quantized weights (1, 2 or 4 columns) in the tile layout of qmat.h.
// Integer math per 32-weight sub-block follows llama.cpp vecdotq.cuh (MIT, see THIRD_PARTY_NOTICES.md),
// so results match llama.cpp's MMVQ up to float summation order.
#include "qmat.h"
#include "common.cuh"
#include "quant_tables.h"
#include "qtypes.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace q27 {

const uint32_t* device_tables() {
  static const uint32_t* per_dev[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!per_dev[dev]) {
    std::vector<uint32_t> h(TB_WORDS);
    std::memcpy(&h[TB_IQ3S], iq3s_grid, sizeof(iq3s_grid));
    std::memcpy(&h[TB_IQ3XXS], iq3xxs_grid, sizeof(iq3xxs_grid));
    std::memcpy(&h[TB_IQ4NL], kvalues_iq4nl, sizeof(kvalues_iq4nl));
    std::memcpy(&h[TB_IQ2XXS], iq2xxs_grid, sizeof(iq2xxs_grid));
    std::memcpy(&h[TB_IQ2XS], iq2xs_grid, sizeof(iq2xs_grid));
    std::memcpy(&h[TB_IQ2S], iq2s_grid, sizeof(iq2s_grid));
    std::memcpy(&h[TB_IQ1M], iq1s_grid_gpu, sizeof(iq1s_grid_gpu));
    uint32_t* d; CK(cudaMalloc(&d, TB_WORDS * 4));
    CK(cudaMemcpy(d, h.data(), TB_WORDS * 4, cudaMemcpyHostToDevice));
    per_dev[dev] = d;
  }
  return per_dev[dev];
}


namespace {

struct Layout { int nf; int w[4]; int dw; };
template <GType T> Layout layout_t() { return {Tr<T>::nf, {Tr<T>::w[0], Tr<T>::w[1], Tr<T>::w[2], Tr<T>::w[3]}, Tr<T>::dw}; }

Layout layout_of(GType t) {
  switch (t) {
    case GType::IQ3_S: return layout_t<GType::IQ3_S>();
    case GType::IQ3_XXS: return layout_t<GType::IQ3_XXS>();
    case GType::IQ4_XS: return layout_t<GType::IQ4_XS>();
    case GType::Q4_K: return layout_t<GType::Q4_K>();
    case GType::Q2_K: return layout_t<GType::Q2_K>();
    case GType::Q6_K: return layout_t<GType::Q6_K>();
    case GType::IQ2_XXS: return layout_t<GType::IQ2_XXS>();
    case GType::IQ2_XS: return layout_t<GType::IQ2_XS>();
    case GType::IQ2_S: return layout_t<GType::IQ2_S>();
    case GType::IQ1_M: return layout_t<GType::IQ1_M>();
    default: throw std::runtime_error(std::string("qmat: type not supported: ") + gtype_name(t));
  }
}

}  // namespace

bool qmat_supported(GType t) {
  switch (t) {
    case GType::IQ3_S: case GType::IQ3_XXS: case GType::IQ4_XS: case GType::Q4_K: case GType::Q2_K:
    case GType::Q6_K: case GType::IQ2_XXS: case GType::IQ2_XS: case GType::IQ2_S: case GType::IQ1_M:
      return true;
    default:
      return false;
  }
}

size_t qmat_bytes(GType t, int N, int K) {
  Layout L = layout_of(t);
  const size_t ntb = (size_t)(N / 8) * (K / 256);
  size_t total = 0;
  for (int i = 0; i < L.nf; i++) total += align256(ntb * 8 * L.w[i]);
  total += align256(ntb * L.dw);
  return total;
}

// ---------------------------------------------------------------- repack kernels
// One thread per chunk (tile, block, sub-block); it copies the bytes of that sub-block for the
// 8 rows of the tile. raw = original blocks, [N][nb][block_bytes]. Offsets below are the block
// structs of ggml-common.h.
struct RepackArgs {
  const uint8_t* raw;
  uint8_t* f[4];
  uint8_t* d;
  int nb, ntiles, bb;  // blocks per row, tiles, block bytes
};

template <GType T>
__global__ void repack_kernel(RepackArgs a) {
  const size_t chunk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= (size_t)a.ntiles * a.nb * 8) return;
  const int s = (int)(chunk % 8);
  const size_t tb = chunk / 8;
  const int b = (int)(tb % a.nb);
  const size_t tile = tb / a.nb;
  constexpr int w0 = Tr<T>::w[0], w1 = Tr<T>::w[1], w2 = Tr<T>::w[2], w3 = Tr<T>::w[3];
  constexpr int DW = Tr<T>::dw;
  uint8_t* f0 = a.f[0] + chunk * w0;
  uint8_t* f1 = a.f[1] + chunk * w1;
  uint8_t* f2 = a.f[2] + chunk * w2;
  uint8_t* f3 = a.f[3] + chunk * w3;
  uint32_t nib = 0;  // IQ3_S: 8 scale nibbles
  for (int r = 0; r < 8; r++) {
    const uint8_t* src = a.raw + ((tile * 8 + r) * a.nb + b) * a.bb;
    uint8_t* d = a.d + tb * DW;
    if constexpr (T == GType::IQ3_S) {  // d qs[64] qh[8] signs[32] scales[4]
      for (int j = 0; j < 8; j++) f0[r * 8 + j] = src[2 + 8 * s + j];
      for (int j = 0; j < 4; j++) f1[r * 4 + j] = src[74 + 4 * s + j];
      f2[r] = src[66 + s];
      nib |= (uint32_t)((src[106 + s / 2] >> (4 * (s & 1))) & 0xF) << (4 * r);
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::IQ3_XXS) {  // d qs[64] aux[32]
      for (int j = 0; j < 8; j++) f0[r * 8 + j] = src[2 + 8 * s + j];
      for (int j = 0; j < 4; j++) f1[r * 4 + j] = src[66 + 4 * s + j];
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::IQ4_XS) {  // d scales_h(2) scales_l[4] qs[128]
      for (int j = 0; j < 16; j++) f0[r * 16 + j] = src[8 + 16 * s + j];
      const int sh = src[2] | (src[3] << 8);
      const int l = ((src[4 + s / 2] >> (4 * (s & 1))) & 0xF) | (((sh >> (2 * s)) & 3) << 4);
      f1[r] = (uint8_t)(int8_t)(l - 32);
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::Q4_K) {  // d dmin scales[12] qs[128]
      // 32 nibbles of sub-block s: q_i = (qs[32*(s/2) + i] >> 4*(s&1)) & 15. New byte b = q_b | q_{b+16} << 4.
      const uint8_t* qs = src + 16 + 32 * (s / 2);
      const int sh = 4 * (s & 1);
      for (int bb = 0; bb < 16; bb++)
        f0[r * 16 + bb] = (uint8_t)(((qs[bb] >> sh) & 0xF) | (((qs[bb + 16] >> sh) & 0xF) << 4));
      const uint8_t* sc = src + 4;
      int scv, mv;
      if (s < 4) { scv = sc[s] & 63; mv = sc[s + 4] & 63; }
      else { scv = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4); mv = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4); }
      f1[r * 2] = (uint8_t)scv; f1[r * 2 + 1] = (uint8_t)mv;
      if (s == 0) for (int j = 0; j < 4; j++) d[4 * r + j] = src[j];
    } else if constexpr (T == GType::Q2_K) {  // scales[16] qs[64] d dmin
      // weights of sub-block s = 4n + j: q_l = (qs[32n + l] >> 2j) & 3.
      // New word w (16 weights each): byte k = sum_i q_{16w + k + 4i} << 2i.
      const int n = s / 4, j = s % 4;
      const uint8_t* qs = src + 16 + 32 * n;
      for (int w = 0; w < 2; w++)
        for (int k = 0; k < 4; k++) {
          uint8_t v = 0;
          for (int i = 0; i < 4; i++) v |= (uint8_t)(((qs[16 * w + k + 4 * i] >> (2 * j)) & 3) << (2 * i));
          f0[r * 8 + 4 * w + k] = v;
        }
      f1[r * 2] = src[8 * n + 2 * j]; f1[r * 2 + 1] = src[8 * n + 2 * j + 1];
      if (s == 0) for (int jj = 0; jj < 4; jj++) d[4 * r + jj] = src[80 + jj];
    } else if constexpr (T == GType::Q6_K) {  // ql[128] qh[64] scales[16] d
      const int n = s / 4, j = s % 4;
      uint8_t q[32];
      for (int l = 0; l < 32; l++) {
        const uint8_t lo = src[64 * n + l + 32 * (j & 1)];
        const uint8_t hi = src[128 + 32 * n + l];
        q[l] = (uint8_t)(((j < 2 ? lo : lo >> 4) & 0xF) | (((hi >> (2 * j)) & 3) << 4));
      }
      for (int bb = 0; bb < 16; bb++) f0[r * 16 + bb] = (uint8_t)((q[bb] & 0xF) | ((q[bb + 16] & 0xF) << 4));
      for (int w = 0; w < 2; w++)
        for (int k = 0; k < 4; k++) {
          uint8_t v = 0;
          for (int i = 0; i < 4; i++) v |= (uint8_t)(((q[16 * w + k + 4 * i] >> 4) & 3) << (2 * i));
          f1[r * 8 + 4 * w + k] = v;
        }
      f2[r * 2] = src[192 + 8 * n + 2 * j]; f2[r * 2 + 1] = src[192 + 8 * n + 2 * j + 1];
      if (s == 0) { d[2 * r] = src[208]; d[2 * r + 1] = src[209]; }
    } else if constexpr (T == GType::IQ2_XXS) {  // d qs[64]
      for (int jj = 0; jj < 8; jj++) f0[r * 8 + jj] = src[2 + 8 * s + jj];
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::IQ2_XS) {  // d qs[64] scales[8]
      for (int jj = 0; jj < 8; jj++) f0[r * 8 + jj] = src[2 + 8 * s + jj];
      f1[r] = src[66 + s];
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::IQ2_S) {  // d qs[32] signs[32] qh[8] scales[8]
      for (int jj = 0; jj < 4; jj++) f0[r * 4 + jj] = src[2 + 4 * s + jj];
      for (int jj = 0; jj < 4; jj++) f1[r * 4 + jj] = src[34 + 4 * s + jj];
      f2[r] = src[66 + s];
      f3[r] = src[74 + s];
      if (s == 0) { d[2 * r] = src[0]; d[2 * r + 1] = src[1]; }
    } else if constexpr (T == GType::IQ1_M) {  // qs[32] qh[16] scales[8] (super-scale in top nibbles)
      for (int jj = 0; jj < 4; jj++) f0[r * 4 + jj] = src[4 * s + jj];
      f1[r * 2] = src[32 + 2 * s]; f1[r * 2 + 1] = src[33 + 2 * s];
      uint16_t s16[4];
      for (int jj = 0; jj < 4; jj++) s16[jj] = (uint16_t)(src[48 + 2 * jj] | (src[49 + 2 * jj] << 8));
      f2[r] = (uint8_t)((s16[s / 2] >> (6 * (s % 2))) & 63);
      if (s == 0) {
        const uint16_t dh = (uint16_t)((s16[0] >> 12) | ((s16[1] >> 8) & 0x00F0) | ((s16[2] >> 4) & 0x0F00) | (s16[3] & 0xF000));
        d[2 * r] = (uint8_t)(dh & 0xFF); d[2 * r + 1] = (uint8_t)(dh >> 8);
      }
    }
  }
  if constexpr (T == GType::IQ3_S) { *(uint32_t*)f3 = nib; }
}

template <GType T>
void repack_launch(const RepackArgs& a, size_t nchunks, cudaStream_t s) {
  const int thr = 256;
  repack_kernel<T><<<(unsigned)((nchunks + thr - 1) / thr), thr, 0, s>>>(a);
}

// Repack raw rows already in `scratch` ([N][nb][block bytes]) into a new QMat.
static QMat repack_from_scratch(GType type, int N, int K, const void* scratch, cudaStream_t s) {
  QMat m;
  m.type = type;
  m.N = N;
  m.K = K;
  if (K % 256) throw std::runtime_error("qmat: K not a multiple of 256");
  if (N % 8) throw std::runtime_error("qmat: N not a multiple of 8");
  m.nb = K / 256;
  m.ntiles = N / 8;
  const Layout L = layout_of(type);
  m.bytes = qmat_bytes(type, N, K);
  CK(cudaMalloc(&m.buf, m.bytes));
  const size_t ntb = (size_t)m.ntiles * m.nb;
  size_t off = 0;
  for (int i = 0; i < L.nf; i++) { m.f[i] = m.buf + off; off += align256(ntb * 8 * L.w[i]); }
  m.d = m.buf + off;
  m.dw = L.dw;
  RepackArgs a{(const uint8_t*)scratch, {m.f[0], m.f[1], m.f[2], m.f[3]}, m.buf + off, m.nb, m.ntiles, gtype_block_bytes(type)};
  const size_t nchunks = ntb * 8;
  switch (type) {
    case GType::IQ3_S: repack_launch<GType::IQ3_S>(a, nchunks, s); break;
    case GType::IQ3_XXS: repack_launch<GType::IQ3_XXS>(a, nchunks, s); break;
    case GType::IQ4_XS: repack_launch<GType::IQ4_XS>(a, nchunks, s); break;
    case GType::Q4_K: repack_launch<GType::Q4_K>(a, nchunks, s); break;
    case GType::Q2_K: repack_launch<GType::Q2_K>(a, nchunks, s); break;
    case GType::Q6_K: repack_launch<GType::Q6_K>(a, nchunks, s); break;
    case GType::IQ2_XXS: repack_launch<GType::IQ2_XXS>(a, nchunks, s); break;
    case GType::IQ2_XS: repack_launch<GType::IQ2_XS>(a, nchunks, s); break;
    case GType::IQ2_S: repack_launch<GType::IQ2_S>(a, nchunks, s); break;
    case GType::IQ1_M: repack_launch<GType::IQ1_M>(a, nchunks, s); break;
    default: throw std::runtime_error("qmat: no repack for " + std::string(gtype_name(type)));
  }
  CK(cudaGetLastError());
  return m;
}

QMat qmat_upload(const GTensor& t, int row0, int N, void* scratch, size_t scratch_bytes, cudaStream_t s) {
  const size_t raw_bytes = (size_t)N * t.row_bytes();
  if (raw_bytes > scratch_bytes) throw std::runtime_error("qmat: scratch too small for " + t.name);
  CK(cudaMemcpyAsync(scratch, t.data + (size_t)row0 * t.row_bytes(), raw_bytes, cudaMemcpyHostToDevice, s));
  return repack_from_scratch(t.type, N, (int)t.ne[0], scratch, s);
}

QMat qmat_upload_shard(const GTensor& t, const std::vector<int>& rows, const std::vector<int>& blocks, void* scratch,
                       size_t scratch_bytes, cudaStream_t s) {
  const int bb = gtype_block_bytes(t.type);
  const int src_nb = (int)(t.ne[0] / 256);
  const int nb = blocks.empty() ? src_nb : (int)blocks.size();
  const size_t row_bytes = (size_t)nb * bb;
  const size_t raw_bytes = rows.size() * row_bytes;
  if (raw_bytes > scratch_bytes) throw std::runtime_error("qmat: scratch too small for " + t.name);
  std::vector<uint8_t> h(raw_bytes);
  for (size_t r = 0; r < rows.size(); r++) {
    const uint8_t* src = t.data + (size_t)rows[r] * t.row_bytes();
    uint8_t* dst = h.data() + r * row_bytes;
    if (blocks.empty()) memcpy(dst, src, row_bytes);
    else for (int b = 0; b < nb; b++) memcpy(dst + (size_t)b * bb, src + (size_t)blocks[b] * bb, bb);
  }
  CK(cudaMemcpy(scratch, h.data(), raw_bytes, cudaMemcpyHostToDevice));
  QMat m = repack_from_scratch(t.type, (int)rows.size(), nb * 256, scratch, s);
  CK(cudaStreamSynchronize(s));
  return m;
}

void qmat_free(QMat& m) {
  if (m.buf) cudaFree(m.buf);
  m.buf = nullptr;
}

// ---------------------------------------------------------------- GEMV
// Persistent CTAs of 4 warps. Work unit = (tile of 8 rows, segment of the row). A warp handles one
// unit at a time: lane = (block offset 0..3, sub-block 0..7), 8 rows per lane. Rows are split in
// segments only when the tiles alone cannot fill the GPU; then each unit writes a partial sum and
// the last unit of the tile adds the partials in segment order (deterministic).
namespace {

constexpr int NWARPS = 4;

struct MatArgs {
  const uint8_t* f0; const uint8_t* f1; const uint8_t* f2; const uint8_t* f3;
  const uint8_t* d;
  int N, K, nb, ntiles;
  int ks;       // segments per row
  int seg_nb;   // blocks per segment (multiple of 4)
  float* partial;       // [ks][N][NC] when ks > 1
  unsigned* counters;   // [ntiles], zero between calls
};

// Sum a[V] across the warp. Afterwards lane l holds the total of index (l >> (5 - log2 V)).
template <int V>
__device__ __forceinline__ void warp_transpose_reduce(float (&a)[V], int lane) {
  int cur = V;
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    if (cur > 1) {
      const int h = cur / 2;
      const bool up = (lane & o) != 0;
#pragma unroll
      for (int i = 0; i < V / 2; i++) {
        if (i < h) {
          const float send = up ? a[i] : a[i + h];
          const float keep = up ? a[i + h] : a[i];
          a[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
      }
      cur = h;
    } else {
      a[0] += __shfl_xor_sync(0xffffffffu, a[0], o);
    }
  }
}

template <int OFF, int N16, int NW>
__device__ __forceinline__ void load_words(uint32_t (&w)[NW], const uint8_t* p) {
#pragma unroll
  for (int i = 0; i < N16; i++) {
    const uint4 v = ld_stream(p + 16 * i);
    w[OFF + 4 * i] = v.x; w[OFF + 4 * i + 1] = v.y; w[OFF + 4 * i + 2] = v.z; w[OFF + 4 * i + 3] = v.w;
  }
}
template <int OFF, int NW>
__device__ __forceinline__ void load_w2(uint32_t (&w)[NW], const uint8_t* p) {
  const uint2 v = ld_stream2(p); w[OFF] = v.x; w[OFF + 1] = v.y;
}

// Load one lane's chunk: fields in order, then the d words.
template <GType T>
__device__ __forceinline__ void load_chunk(const MatArgs& m, size_t tb, int sub, uint32_t (&w)[Tr<T>::cw]) {
  const size_t c = tb * 8 + sub;
  constexpr int DOFF = Tr<T>::cw - Tr<T>::dw / 4, DW = Tr<T>::dw;
  if constexpr (T == GType::IQ3_S) {
    load_words<0, 4>(w, m.f0 + c * 64); load_words<16, 2>(w, m.f1 + c * 32);
    load_w2<24>(w, m.f2 + c * 8); w[26] = ld_stream1(m.f3 + c * 4);
  } else if constexpr (T == GType::IQ3_XXS) {
    load_words<0, 4>(w, m.f0 + c * 64); load_words<16, 2>(w, m.f1 + c * 32);
  } else if constexpr (T == GType::IQ4_XS) {
    load_words<0, 8>(w, m.f0 + c * 128); load_w2<32>(w, m.f1 + c * 8);
  } else if constexpr (T == GType::Q4_K) {
    load_words<0, 8>(w, m.f0 + c * 128); load_words<32, 1>(w, m.f1 + c * 16);
  } else if constexpr (T == GType::Q2_K) {
    load_words<0, 4>(w, m.f0 + c * 64); load_words<16, 1>(w, m.f1 + c * 16);
  } else if constexpr (T == GType::Q6_K) {
    load_words<0, 8>(w, m.f0 + c * 128); load_words<32, 4>(w, m.f1 + c * 64); load_words<48, 1>(w, m.f2 + c * 16);
  } else if constexpr (T == GType::IQ2_XXS) {
    load_words<0, 4>(w, m.f0 + c * 64);
  } else if constexpr (T == GType::IQ2_XS) {
    load_words<0, 4>(w, m.f0 + c * 64); load_w2<16>(w, m.f1 + c * 8);
  } else if constexpr (T == GType::IQ2_S) {
    load_words<0, 2>(w, m.f0 + c * 32); load_words<8, 2>(w, m.f1 + c * 32);
    load_w2<16>(w, m.f2 + c * 8); load_w2<18>(w, m.f3 + c * 8);
  } else if constexpr (T == GType::IQ1_M) {
    load_words<0, 2>(w, m.f0 + c * 32); load_words<8, 1>(w, m.f1 + c * 16); load_w2<12>(w, m.f2 + c * 8);
  }
  load_words<DOFF, DW / 16>(w, m.d + tb * DW);
}

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
// L2 prefetch of the fields of one lane's chunk (see load_chunk).
template <GType T>
__device__ __forceinline__ void prefetch_chunk(const MatArgs& m, size_t tb, int sub) {
  const size_t c = tb * 8 + sub;
  constexpr int w0 = Tr<T>::w[0], w1 = Tr<T>::w[1], w2 = Tr<T>::w[2], w3 = Tr<T>::w[3];
  prefetch_l2(m.f0 + c * w0);
  if constexpr (w1 > 0) { prefetch_l2(m.f1 + c * w1); }
  if constexpr (w2 > 0) { prefetch_l2(m.f2 + c * w2); }
  if constexpr (w3 > 0) { prefetch_l2(m.f3 + c * w3); }
  if (sub == 0) prefetch_l2(m.d + tb * Tr<T>::dw);
}

template <int NW>
__device__ __forceinline__ uint32_t byte_of(const uint32_t (&w)[NW], int word0, int i) {
  return (w[word0 + (i >> 2)] >> (8 * (i & 3))) & 0xFF;
}

// Decode one chunk (8 rows x 32 weights) and add to acc[r*NC + c].
template <GType T, int NC>
__device__ __forceinline__ void dot_chunk(const uint32_t (&w)[Tr<T>::cw], const int (&u)[NC][8], const float (&dy)[NC],
                                          const uint32_t* lut, const uint32_t (&kv)[4], float (&acc)[8 * NC]) {
  constexpr int DOFF = Tr<T>::cw - Tr<T>::dw / 4;
  if constexpr (T == GType::IQ3_S) {
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t h = (w[24 + r / 4] >> (8 * (r & 3))) & 0xFF;
      const uint32_t sg = w[16 + r];
      int sumi[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) sumi[c] = 0;
#pragma unroll
      for (int l = 0; l < 8; l++) {
        const uint32_t src = w[2 * r + l / 4];
        const uint32_t idx = ((src >> (8 * (l & 3))) & 0xFF) | ((h << (8 - l)) & 0x100);
        const int g = apply_signs(lut[idx], (sg >> (4 * l)) & 0xF);
#pragma unroll
        for (int c = 0; c < NC; c++) sumi[c] = __dp4a(g, u[c][l], sumi[c]);
      }
      const int mul = 1 + 2 * (int)((w[26] >> (4 * r)) & 0xF);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) acc[r * NC + c] += (dw * dy[c]) * (float)(sumi[c] * mul);
    }
  } else if constexpr (T == GType::IQ3_XXS) {
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t aux = w[16 + r];
      int sumi[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) sumi[c] = 0;
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t sb = ksigns((aux >> (7 * p)) & 0x7F);
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int l = 2 * p + e;
          const uint32_t src = w[2 * r + l / 4];
          const int g = apply_signs(lut[(src >> (8 * (l & 3))) & 0xFF], (sb >> (4 * e)) & 0xF);
#pragma unroll
          for (int c = 0; c < NC; c++) sumi[c] = __dp4a(g, u[c][l], sumi[c]);
        }
      }
      const int ls = (int)(aux >> 28);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) {
        const int si = (ls * sumi[c] + sumi[c] / 2) / 2;
        acc[r * NC + c] += (dw * dy[c]) * (float)si;
      }
    }
  } else if constexpr (T == GType::IQ4_XS) {
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int sumi[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) sumi[c] = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int2 v = table16(w[4 * r + j], kv);
#pragma unroll
        for (int c = 0; c < NC; c++) {
          sumi[c] = __dp4a(v.x, u[c][j], sumi[c]);
          sumi[c] = __dp4a(v.y, u[c][j + 4], sumi[c]);
        }
      }
      const int ls = (int)(int8_t)((w[32 + r / 4] >> (8 * (r & 3))) & 0xFF);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) acc[r * NC + c] += (dw * dy[c]) * (float)(sumi[c] * ls);
    }
  } else if constexpr (T == GType::Q4_K) {
    // value = d*sc*q - dmin*m. Sum of the q8 values gives the min term (llama.cpp uses dp4a with 1s).
    int s8[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
      int t = 0;
#pragma unroll
      for (int k = 0; k < 8; k++) t = __dp4a(u[c][k], 0x01010101, t);
      s8[c] = t;
    }
    const __half2* dm = (const __half2*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int sumi[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) sumi[c] = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int lo = (int)(w[4 * r + j] & 0x0F0F0F0F), hi = (int)((w[4 * r + j] >> 4) & 0x0F0F0F0F);
#pragma unroll
        for (int c = 0; c < NC; c++) { sumi[c] = __dp4a(lo, u[c][j], sumi[c]); sumi[c] = __dp4a(hi, u[c][j + 4], sumi[c]); }
      }
      const uint32_t sm = w[32 + r / 2] >> (16 * (r & 1));
      const int sc = (int)(sm & 0xFF), mn = (int)((sm >> 8) & 0xFF);
      const float2 dmf = __half22float2(dm[r]);
#pragma unroll
      for (int c = 0; c < NC; c++)
        acc[r * NC + c] += dmf.x * (dy[c] * (float)(sumi[c] * sc)) - dmf.y * (dy[c] * (float)(s8[c] * mn));
    }
  } else if constexpr (T == GType::Q2_K) {
    int s8a[NC], s8b[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
      int ta = 0, tb2 = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) { ta = __dp4a(u[c][k], 0x01010101, ta); tb2 = __dp4a(u[c][k + 4], 0x01010101, tb2); }
      s8a[c] = ta; s8b[c] = tb2;
    }
    const __half2* dm = (const __half2*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int sa[NC], sb[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) { sa[c] = 0; sb[c] = 0; }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int va = (int)((w[2 * r] >> (2 * i)) & 0x03030303), vb = (int)((w[2 * r + 1] >> (2 * i)) & 0x03030303);
#pragma unroll
        for (int c = 0; c < NC; c++) { sa[c] = __dp4a(va, u[c][i], sa[c]); sb[c] = __dp4a(vb, u[c][4 + i], sb[c]); }
      }
      const uint32_t g = w[16 + r / 2] >> (16 * (r & 1));
      const int g0 = (int)(g & 0xFF), g1 = (int)((g >> 8) & 0xFF);
      const float2 dmf = __half22float2(dm[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) {
        const int sd = sa[c] * (g0 & 0xF) + sb[c] * (g1 & 0xF);
        const int smn = s8a[c] * (g0 >> 4) + s8b[c] * (g1 >> 4);
        acc[r * NC + c] += dmf.x * (dy[c] * (float)sd) - dmf.y * (dy[c] * (float)smn);
      }
    }
  } else if constexpr (T == GType::Q6_K) {
    // q in 0..63, value d*sc*(q-32): dot with q, then subtract 32 * sum of q8 per 16-weight half.
    int s8a[NC], s8b[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
      int ta = 0, tb2 = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) { ta = __dp4a(u[c][k], 0x01010101, ta); tb2 = __dp4a(u[c][k + 4], 0x01010101, tb2); }
      s8a[c] = ta; s8b[c] = tb2;
    }
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int sa[NC], sb[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) { sa[c] = 0; sb[c] = 0; }
      const uint32_t h0 = w[32 + 2 * r], h1 = w[32 + 2 * r + 1];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint32_t l = w[4 * r + i];
        const int qa = (int)((l & 0x0F0F0F0F) | (((h0 >> (2 * i)) & 0x03030303) << 4));
        const int qb = (int)(((l >> 4) & 0x0F0F0F0F) | (((h1 >> (2 * i)) & 0x03030303) << 4));
#pragma unroll
        for (int c = 0; c < NC; c++) { sa[c] = __dp4a(qa, u[c][i], sa[c]); sb[c] = __dp4a(qb, u[c][4 + i], sb[c]); }
      }
      const uint32_t g = w[48 + r / 2] >> (16 * (r & 1));
      const int sc0 = (int)(int8_t)(g & 0xFF), sc1 = (int)(int8_t)((g >> 8) & 0xFF);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) {
        const int si = (sa[c] - 32 * s8a[c]) * sc0 + (sb[c] - 32 * s8b[c]) * sc1;
        acc[r * NC + c] += dw * (dy[c] * (float)si);
      }
    }
  } else if constexpr (T == GType::IQ2_XXS) {
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t idx4 = w[2 * r], aux = w[2 * r + 1];
      int sumi[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) sumi[c] = 0;
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t gi = (idx4 >> (8 * p)) & 0xFF;
        const uint32_t sb = ksigns((aux >> (7 * p)) & 0x7F);
        const int gx = apply_signs(lut[2 * gi], sb & 0xF), gy = apply_signs(lut[2 * gi + 1], sb >> 4);
#pragma unroll
        for (int c = 0; c < NC; c++) { sumi[c] = __dp4a(gx, u[c][2 * p], sumi[c]); sumi[c] = __dp4a(gy, u[c][2 * p + 1], sumi[c]); }
      }
      const int ls = (int)((aux >> 27) | 1);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) acc[r * NC + c] += (dw * dy[c]) * (float)(sumi[c] * ls / 8);
    }
  } else if constexpr (T == GType::IQ2_XS || T == GType::IQ2_S) {
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int s0[NC], s1[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) { s0[c] = 0; s1[c] = 0; }
      uint32_t lsb;
      if constexpr (T == GType::IQ2_XS) { lsb = byte_of(w, 16, r); } else { lsb = byte_of(w, 18, r); }
#pragma unroll
      for (int p = 0; p < 4; p++) {
        uint32_t gi, sb;
        if constexpr (T == GType::IQ2_XS) {
          const uint32_t q16 = (w[2 * r + p / 2] >> (16 * (p & 1))) & 0xFFFF;
          gi = q16 & 0x1FF;
          sb = ksigns(q16 >> 9);
        } else {
          const uint32_t qh = byte_of(w, 16, r);
          gi = byte_of(w, r, p) | ((qh << (8 - 2 * p)) & 0x300);
          sb = byte_of(w, 8 + r, p);
        }
        const int gx = apply_signs(lut[2 * gi], sb & 0xF), gy = apply_signs(lut[2 * gi + 1], sb >> 4);
#pragma unroll
        for (int c = 0; c < NC; c++) {
          if (p < 2) { s0[c] = __dp4a(gx, u[c][2 * p], s0[c]); s0[c] = __dp4a(gy, u[c][2 * p + 1], s0[c]); }
          else       { s1[c] = __dp4a(gx, u[c][2 * p], s1[c]); s1[c] = __dp4a(gy, u[c][2 * p + 1], s1[c]); }
        }
      }
      const int ls0 = (int)(lsb & 0xF), ls1 = (int)(lsb >> 4);
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++) {
        const int si = (s0[c] * ls0 + s1[c] * ls1 + (s0[c] + s1[c]) / 2) / 4;
        acc[r * NC + c] += (dw * dy[c]) * (float)si;
      }
    }
  } else if constexpr (T == GType::IQ1_M) {
    int sy[NC][4];
#pragma unroll
    for (int c = 0; c < NC; c++)
#pragma unroll
      for (int p = 0; p < 4; p++) sy[c][p] = __dp4a(u[c][2 * p + 1], 0x01010101, __dp4a(u[c][2 * p], 0x01010101, 0));
    const __half* dh = (const __half*)&w[DOFF];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t qh16 = (w[8 + r / 2] >> (16 * (r & 1))) & 0xFFFF;
      int si[NC][2];
      float sf[NC][2];
#pragma unroll
      for (int c = 0; c < NC; c++) { si[c][0] = si[c][1] = 0; sf[c][0] = sf[c][1] = 0.f; }
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t qhl = (qh16 >> (4 * p)) & 0xF;
        const uint32_t grid = lut[byte_of(w, r, p) | ((qhl & 7) << 8)];
        const int g0 = (int)(grid & 0x0F0F0F0F), g1 = (int)((grid >> 4) & 0x0F0F0F0F);
        const float delta = -1.0f + 0.125f - (qhl & 0x08) * (2.0f * 0.125f / 0x08);
#pragma unroll
        for (int c = 0; c < NC; c++) {
          si[c][p / 2] = __dp4a(g1, u[c][2 * p + 1], __dp4a(g0, u[c][2 * p], si[c][p / 2]));
          sf[c][p / 2] += delta * (float)sy[c][p];
        }
      }
      const uint32_t sc6 = byte_of(w, 12, r);
      const int sc0 = 2 * (int)(sc6 & 7) + 1, sc1 = 2 * (int)((sc6 >> 3) & 7) + 1;
      const float dw = __half2float(dh[r]);
#pragma unroll
      for (int c = 0; c < NC; c++)
        acc[r * NC + c] += (dw * dy[c]) * (((float)si[c][0] + sf[c][0]) * sc0 + ((float)si[c][1] + sf[c][1]) * sc1);
    }
  }
}

template <GType T, int NC>
__global__ void __launch_bounds__(NWARPS * 32) gemv_kernel(MatArgs m, const int8_t* __restrict__ xq,
                                                          const float* __restrict__ xd, float* __restrict__ y,
                                                          const uint32_t* __restrict__ tables) {
  constexpr int LN = Tr<T>::lut_n > 0 ? Tr<T>::lut_n : 1;
  __shared__ uint32_t lut[LN];
  if constexpr (Tr<T>::lut_n > 0) {
    for (int i = threadIdx.x; i < Tr<T>::lut_n; i += blockDim.x) lut[i] = tables[Tr<T>::lut_off + i];
  }
  uint32_t kv[4] = {0, 0, 0, 0};
  if constexpr (T == GType::IQ4_XS) {
#pragma unroll
    for (int i = 0; i < 4; i++) kv[i] = tables[TB_IQ4NL + i];
  }
  __syncthreads();

  const int K = m.K, nb = m.nb;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int sub = lane & 7;
  const int lb = lane >> 3;
  const int units = m.ntiles * m.ks;

  // Weights do not depend on the previous kernel: bring the first chunks of this warp's first unit into L2
  // while the previous kernel finishes, then wait for the activations.
  {
    const int unit = blockIdx.x * NWARPS + warp;
    if (unit < units) {
      const int tile = unit / m.ks, seg = unit - (unit / m.ks) * m.ks;
      const int b_end = min(nb, seg * m.seg_nb + m.seg_nb);
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int b = seg * m.seg_nb + 4 * i + lb;
        if (b < b_end) prefetch_chunk<T>(m, (size_t)tile * nb + b, sub);
      }
    }
  }
  pdl_wait();
  pdl_trigger();

  for (int unit = blockIdx.x * NWARPS + warp; unit < units; unit += gridDim.x * NWARPS) {
    const int tile = unit / m.ks;
    const int seg = unit - tile * m.ks;
    const int b_beg = seg * m.seg_nb;
    const int b_end = min(nb, b_beg + m.seg_nb);

    float acc[8 * NC];
#pragma unroll
    for (int i = 0; i < 8 * NC; i++) acc[i] = 0.f;

#pragma unroll 2
    for (int b0 = b_beg; b0 < b_end; b0 += 4) {
      const int b = b0 + lb;
      if (b >= b_end) continue;
      uint32_t w[Tr<T>::cw];
      load_chunk<T>(m, (size_t)tile * nb + b, sub, w);
      int u[NC][8]; float dy[NC];
#pragma unroll
      for (int c = 0; c < NC; c++) {
        const int4* xp = (const int4*)(xq + (size_t)c * K + b * 256 + sub * 32);
        const int4 a0 = __ldg(xp), a1 = __ldg(xp + 1);
        u[c][0] = a0.x; u[c][1] = a0.y; u[c][2] = a0.z; u[c][3] = a0.w;
        u[c][4] = a1.x; u[c][5] = a1.y; u[c][6] = a1.z; u[c][7] = a1.w;
        dy[c] = __ldg(xd + (size_t)c * (K / 32) + b * 8 + sub);
      }
      dot_chunk<T, NC>(w, u, dy, lut, kv, acc);
    }

    warp_transpose_reduce<8 * NC>(acc, lane);
    constexpr int V = 8 * NC;  // 8, 16 or 32
    constexpr int shift = V == 8 ? 2 : V == 16 ? 1 : 0;
    const int idx = lane >> shift;
    const bool writer = (lane & ((1 << shift) - 1)) == 0;
    const int r = idx / NC, c = idx % NC;
    const int row = tile * 8 + r;
    if (m.ks == 1) {
      if (writer) y[(size_t)c * m.N + row] = acc[0];
      continue;
    }
    if (writer) m.partial[((size_t)seg * m.N + row) * NC + c] = acc[0];
    __threadfence();
    unsigned prev = 0;
    if (lane == 0) prev = atomicAdd(&m.counters[tile], 1u);
    prev = __shfl_sync(0xffffffffu, prev, 0);
    if (prev == (unsigned)m.ks - 1) {
      __threadfence();
      if (writer) {
        float s = 0.f;
        for (int k = 0; k < m.ks; k++) s += __ldcg(&m.partial[((size_t)k * m.N + row) * NC + c]);
        y[(size_t)c * m.N + row] = s;
      }
      if (lane == 0) m.counters[tile] = 0;
    }
  }
}

// Per-device scratch for split rows.
struct Workspace { float* partial = nullptr; size_t partial_n = 0; unsigned* counters = nullptr; size_t counters_n = 0; };
// One per (device, stream): GEMVs on parallel graph branches must not share the partial sums and counters.
Workspace& workspace(cudaStream_t s) {
  static std::vector<std::pair<cudaStream_t, Workspace*>> ws[16];
  int dev; CK(cudaGetDevice(&dev));
  for (auto& e : ws[dev]) if (e.first == s) return *e.second;
  ws[dev].push_back({s, new Workspace()});
  return *ws[dev].back().second;
}

// Split rows only when the tiles alone give fewer than `want` warps of work.
void choose_split(int ntiles, int nb, int want, int& ks, int& seg_nb) {
  const int iters = (nb + 3) / 4;
  ks = 1; seg_nb = iters * 4;
  if (ntiles >= want) return;
  int k = (want + ntiles - 1) / ntiles;
  if (k > iters) k = iters;
  const int ipu = (iters + k - 1) / k;
  ks = (iters + ipu - 1) / ipu;
  seg_nb = ipu * 4;
}

template <GType T, int NC>
void launch(const QMat& m, const int8_t* xq, const float* xd, float* y, cudaStream_t s) {
  static int nsm_dev[16] = {}, occ_dev[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!nsm_dev[dev]) {
    CK(cudaDeviceGetAttribute(&nsm_dev[dev], cudaDevAttrMultiProcessorCount, dev));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_dev[dev], gemv_kernel<T, NC>, NWARPS * 32, 0));
  }
  const int resident = nsm_dev[dev] * occ_dev[dev];  // CTAs that fit at once
  MatArgs a{m.f[0], m.f[1], m.f[2], m.f[3], m.d, m.N, m.K, m.nb, m.ntiles, 1, 0, nullptr, nullptr};
  choose_split(m.ntiles, m.nb, 2 * resident * NWARPS, a.ks, a.seg_nb);
  if (a.ks > 1) {
    Workspace& ws = workspace(s);
    const size_t need = (size_t)a.ks * m.N * NC;
    if (need > ws.partial_n) { if (ws.partial) cudaFree(ws.partial); CK(cudaMalloc(&ws.partial, need * 4)); ws.partial_n = need; }
    if ((size_t)m.ntiles > ws.counters_n) {
      if (ws.counters) cudaFree(ws.counters);
      CK(cudaMalloc(&ws.counters, m.ntiles * sizeof(unsigned)));
      CK(cudaMemset(ws.counters, 0, m.ntiles * sizeof(unsigned)));
      ws.counters_n = m.ntiles;
    }
    a.partial = ws.partial; a.counters = ws.counters;
  }
  const int units = m.ntiles * a.ks;
  int blocks = (units + NWARPS - 1) / NWARPS;
  if (blocks > resident) blocks = resident;
  launch_k(gemv_kernel<T, NC>, blocks, NWARPS * 32, 0, s, a, xq, xd, y, device_tables());
}

template <GType T>
void launch_nc(const QMat& m, const int8_t* xq, const float* xd, float* y, int nc, cudaStream_t s) {
  switch (nc) {
    case 1: launch<T, 1>(m, xq, xd, y, s); break;
    case 2: launch<T, 2>(m, xq, xd, y, s); break;
    case 4: launch<T, 4>(m, xq, xd, y, s); break;
    default: throw std::runtime_error("qgemv: ncols must be 1, 2 or 4");
  }
}

}  // namespace

void qgemv(const QMat& m, const int8_t* xq, const float* xd, float* y, int ncols, cudaStream_t s) {
  switch (m.type) {
    case GType::IQ3_S: launch_nc<GType::IQ3_S>(m, xq, xd, y, ncols, s); break;
    case GType::IQ3_XXS: launch_nc<GType::IQ3_XXS>(m, xq, xd, y, ncols, s); break;
    case GType::IQ4_XS: launch_nc<GType::IQ4_XS>(m, xq, xd, y, ncols, s); break;
    case GType::Q4_K: launch_nc<GType::Q4_K>(m, xq, xd, y, ncols, s); break;
    case GType::Q2_K: launch_nc<GType::Q2_K>(m, xq, xd, y, ncols, s); break;
    case GType::Q6_K: launch_nc<GType::Q6_K>(m, xq, xd, y, ncols, s); break;
    case GType::IQ2_XXS: launch_nc<GType::IQ2_XXS>(m, xq, xd, y, ncols, s); break;
    case GType::IQ2_XS: launch_nc<GType::IQ2_XS>(m, xq, xd, y, ncols, s); break;
    case GType::IQ2_S: launch_nc<GType::IQ2_S>(m, xq, xd, y, ncols, s); break;
    case GType::IQ1_M: launch_nc<GType::IQ1_M>(m, xq, xd, y, ncols, s); break;
    default: throw std::runtime_error(std::string("qgemv: type not supported: ") + gtype_name(m.type));
  }
  CK(cudaGetLastError());
}

// ---------------------------------------------------------------- q8_1 activation quantization
// Same as llama.cpp quantize_q8_1: one thread per value, a warp per 32-value block,
// d = amax / 127, q = roundf(x / d), d stored as fp16.
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd, int K,
                                     float* __restrict__ xs) {
  pdl_wait();
  pdl_trigger();
  const int c = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= K) return;
  const float xi = x[(size_t)c * K + i];
  float amax = fabsf(xi);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float d = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
  xq[(size_t)c * K + i] = q;
  if ((threadIdx.x & 31) == 0) xd[(size_t)c * (K / 32) + i / 32] = __half2float(__float2half(d));
  if (xs) {
    float sum = xi;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 15) == 0) xs[(size_t)c * (K / 16) + i / 16] = sum;
  }
}

void quantize_q8_1(const float* x, int8_t* xq, float* xd, int K, int ncols, cudaStream_t s, float* xs) {
  dim3 grid((K + 255) / 256, ncols);
  launch_k(quantize_q8_1_kernel, grid, 256, 0, s, x, xq, xd, K, xs);
  CK(cudaGetLastError());
}

}  // namespace q27
