// A quantized weight matrix on the GPU, repacked for the decode GEMV.
//
// Layout ("tile layout"). Rows are grouped in tiles of 8. Each 256-weight block is split in
// 8 sub-blocks of 32 weights. A "chunk" is one sub-block of one block for the 8 rows of a tile.
// chunk index = (tile * nb + block) * 8 + sub. Each field of the original block struct goes to
// its own array, ordered by chunk, then by row inside the tile. The super-block scale d goes to
// an array ordered by (tile * nb + block), then row. The repack only moves bytes (and for IQ4_XS
// re-encodes the 6-bit scale as one int8), so every weight decodes to the same value.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "gguf.h"

namespace q27 {

struct QMat {
  GType type = GType::F32;
  int N = 0;   // rows (outputs)
  int K = 0;   // row length (inputs)
  int nb = 0;  // K / 256
  int ntiles = 0;
  uint8_t* buf = nullptr;     // one device allocation for all fields
  size_t bytes = 0;
  uint8_t* f[4] = {nullptr, nullptr, nullptr, nullptr};  // per-type field arrays
  const uint8_t* d = nullptr;  // super-block scales, [tile][block][8 rows] (half, or half2 d+dmin)
  int dw = 0;                  // bytes of d per (tile, block): 16 or 32
};

// Bytes of the repacked matrix (same as the GGUF size except IQ4_XS: +1.5%).
size_t qmat_bytes(GType t, int N, int K);
// Upload one GGUF tensor (rows [row0, row0+N) of it) to the current device and repack it.
// `scratch` must hold the raw rows (N * row_bytes). Returns a QMat that owns its buffer.
QMat qmat_upload(const GTensor& t, int row0, int N, void* scratch, size_t scratch_bytes, cudaStream_t s);
// Same, for a shard: `rows` lists the source rows (in order), `blocks` the 256-weight column blocks
// kept from each row (empty = all). Used for the tensor-parallel split.
QMat qmat_upload_shard(const GTensor& t, const std::vector<int>& rows, const std::vector<int>& blocks, void* scratch,
                       size_t scratch_bytes, cudaStream_t s);
void qmat_free(QMat& m);
bool qmat_supported(GType t);

// y[c][n] = sum_k W[n][k] * x[c][k] for c < ncols (1..4).
// x is given as q8_1 activations: xq[c][K] int8 and xd[c][K/32] float (value of the fp16 scale).
void qgemv(const QMat& m, const int8_t* xq, const float* xd, float* y, int ncols, cudaStream_t s);

// q8_1 quantization of activations, same numerics as llama.cpp quantize_q8_1 (fast math).
// x[c][K] f32 -> xq[c][K], xd[c][K/32]; xs (optional) [c][K/16] = sum of the unquantized x per 16 values.
void quantize_q8_1(const float* x, int8_t* xq, float* xd, int K, int ncols, cudaStream_t s, float* xs = nullptr);

// Prefill GEMM (qgemm.cu): y[t][n] = sum_k W[n][k] x[t][k] for M tokens, x as q8_1 (xq [M][K], xd [M][K/32]) plus
// xs [M][K/16] (sums per 16, needed when qgemm_needs_sums(type)). N must be a multiple of 128.
void qgemm(const QMat& m, const int8_t* xq, const float* xd, const float* xs, float* y, int M, cudaStream_t s);
bool qgemm_needs_sums(GType t);

}  // namespace q27
