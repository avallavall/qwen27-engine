// Kernels for batched prompt reading (prefill.cu). M tokens per pass.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace q27 {

// ya[t] = Wa x[t], yb[t] = Wb x[t] for M tokens (ssm_alpha / ssm_beta, BF16, N rows each), x f32 [M][K].
void bf16_pair_gemm(const __nv_bfloat16* Wa, const __nv_bfloat16* Wb, const float* x, float* ya, float* yb, int N, int K, int M,
                    cudaStream_t s);
// GDN conv + SiLU + L2 norm (q, k heads) + gates over M tokens. Reads the conv state of plane *in_plane and
// writes the state after the batch to plane (*in_plane == 0 ? 1 : 0) (see gdn_flip_plane).
void gdn_conv_prefill(const float* x, const float* w, float* planes, const int* in_plane, float* y, int C, int M,
                      int n_norm_heads, float eps, const float* a, const float* b, const float* ssm_a, const float* dt_bias, float* g,
                      float* beta, int H, cudaStream_t s);
// *plane = (*plane == 0 ? 1 : 0): after a prefill pass the GDN states are in the other plane.
void gdn_flip_plane(int* plane, cudaStream_t s);

// Cross-card exchange over M rows for prefill (pinned, mapped host memory; one flag per row, stride 32 ints).
struct PfExchange {
  __nv_bfloat16* mine = nullptr;        // [M][n] this card's partial (BF16)
  const __nv_bfloat16* other = nullptr; // [M][n] the other card's partial
  int* flag_mine = nullptr;             // [M][32]
  const int* flag_other = nullptr;
  int token = 0;                        // same value on both cards, new for every exchange
  int* err = nullptr;
};
// x += partial (summed over both cards when ex != null), then RMSNorm(x) * w -> h (may be null), q8_1 (xq, xd)
// and the per-16 sums xs (may be null), for M rows of n = 5120.
void sum_norm_rows(float* x, const float* partial, int n, int M, const PfExchange* ex, const float* w, float eps, float* h,
                   int8_t* xq, float* xd, float* xs, cudaStream_t s);

// y = bf16(x), n values (n even).
void to_bf16(const float* x, __nv_bfloat16* y, size_t n, cudaStream_t s);
// q8 wire rows (n int8 values + n / 32 fp16 scales per row) of partial (+ ef unless first); ef keeps the rounding
// error (error feedback; may be null). M rows of n = 5120.
void to_q8_wire(const float* partial, float* ef, bool first, uint8_t* wire, int n, int M, cudaStream_t s);
// x += dequant(own) + dequant(recv) (q8 wire rows), then RMSNorm(x) * w -> h (may be null), q8_1 and per-16 sums.
void add_norm_rows_q8(float* x, const uint8_t* own, const uint8_t* recv, int n, int M, const float* w, float eps, float* h,
                      int8_t* xq, float* xd, float* xs, cudaStream_t s);
// *dst = (*src == 0 ? 1 : 0)
void flip_plane_to(int* dst, const int* src, cudaStream_t s);
// x += bf16(partial) + recv (recv = the other card's BF16 partial; null: x += partial), then RMSNorm(x) * w ->
// h (may be null), q8_1 (xq, xd) and per-16 sums xs, for M rows of n = 5120.
void add_norm_rows(float* x, const float* partial, const __nv_bfloat16* recv, int n, int M, const float* w, float eps, float* h,
                   int8_t* xq, float* xd, float* xs, cudaStream_t s);

}  // namespace q27
