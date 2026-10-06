// Per-type traits and decode helpers shared by the decode GEMV (qgemv.cu) and the prefill GEMM (qgemm.cu).
// Integer decode follows llama.cpp vecdotq.cuh / mmq (MIT, see THIRD_PARTY_NOTICES.md).
#pragma once
#include "qmat.h"
#include "common.cuh"

namespace q27 {

// ---------------------------------------------------------------- lookup tables on each device
// Word offsets in the table buffer. 64-bit grids are stored as two words per entry.
enum {
  TB_IQ3S = 0,                    // 512 words
  TB_IQ3XXS = TB_IQ3S + 512,      // 256 words
  TB_IQ4NL = TB_IQ3XXS + 256,     // 4 words (16 int8)
  TB_IQ2XXS = TB_IQ4NL + 4,       // 256 x 2 words
  TB_IQ2XS = TB_IQ2XXS + 512,     // 512 x 2 words
  TB_IQ2S = TB_IQ2XS + 1024,      // 1024 x 2 words
  TB_IQ1M = TB_IQ2S + 2048,       // 2048 words (iq1s_grid_gpu)
  TB_WORDS = TB_IQ1M + 2048,
};

// ---------------------------------------------------------------- per-type traits

constexpr size_t align256(size_t x) { return (x + 255) / 256 * 256; }

// Field widths in bytes per chunk (8 rows x 32 weights), d bytes per (tile, block),
// codebook location in the table buffer, and words one lane loads per chunk (fields + d).
template <GType T> struct Tr;
template <> struct Tr<GType::IQ3_S>   { static constexpr int nf = 4, w[4] = {64, 32, 8, 4},   dw = 16, lut_off = TB_IQ3S,   lut_n = 512,  cw = 31; };
template <> struct Tr<GType::IQ3_XXS> { static constexpr int nf = 2, w[4] = {64, 32, 0, 0},   dw = 16, lut_off = TB_IQ3XXS, lut_n = 256,  cw = 28; };
template <> struct Tr<GType::IQ4_XS>  { static constexpr int nf = 2, w[4] = {128, 8, 0, 0},   dw = 16, lut_off = 0,         lut_n = 0,    cw = 38; };
template <> struct Tr<GType::Q4_K>    { static constexpr int nf = 2, w[4] = {128, 16, 0, 0},  dw = 32, lut_off = 0,         lut_n = 0,    cw = 44; };
template <> struct Tr<GType::Q2_K>    { static constexpr int nf = 2, w[4] = {64, 16, 0, 0},   dw = 32, lut_off = 0,         lut_n = 0,    cw = 28; };
template <> struct Tr<GType::Q6_K>    { static constexpr int nf = 3, w[4] = {128, 64, 16, 0}, dw = 16, lut_off = 0,         lut_n = 0,    cw = 56; };
template <> struct Tr<GType::IQ2_XXS> { static constexpr int nf = 1, w[4] = {64, 0, 0, 0},    dw = 16, lut_off = TB_IQ2XXS, lut_n = 512,  cw = 20; };
template <> struct Tr<GType::IQ2_XS>  { static constexpr int nf = 2, w[4] = {64, 8, 0, 0},    dw = 16, lut_off = TB_IQ2XS,  lut_n = 1024, cw = 22; };
template <> struct Tr<GType::IQ2_S>   { static constexpr int nf = 4, w[4] = {32, 32, 8, 8},   dw = 16, lut_off = TB_IQ2S,   lut_n = 2048, cw = 24; };
template <> struct Tr<GType::IQ1_M>   { static constexpr int nf = 3, w[4] = {32, 16, 8, 0},   dw = 16, lut_off = TB_IQ1M,   lut_n = 2048, cw = 18; };


// Codebooks of all types on the current device (built on first use).
const uint32_t* device_tables();

// Negate the bytes of grid word g whose bit in the 4-bit mask nib is set. Grid bytes are >= 1,
// so (g ^ 0xFF) + 1 = 256 - g never carries into the next byte.
__device__ __forceinline__ int apply_signs(uint32_t g, uint32_t nib) {
  const uint32_t c = (nib * 0x00204081u) & 0x01010101u;
  return (int)((g ^ (c * 0xFFu)) + c);
}

// 7 sign bits plus the parity bit (llama.cpp unpack_ksigns without the broadcast).
__device__ __forceinline__ uint32_t ksigns(uint32_t v7) {
  return v7 | ((__popc(v7) & 1) << 7);
}

// llama.cpp get_int_from_table_16 (CUDA path): 8 nibbles -> int2 of table bytes (even, odd nibbles).
__device__ __forceinline__ int2 table16(uint32_t q4, const uint32_t (&t)[4]) {
  uint32_t tmp[2];
  const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t lo = __byte_perm(t[0], t[1], q4 >> shift);
    const uint32_t hi = __byte_perm(t[2], t[3], q4 >> shift);
    tmp[i] = __byte_perm(lo, hi, sel >> shift);
  }
  return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}


}  // namespace q27
