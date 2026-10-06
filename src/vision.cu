// Vision encoder kernels and VisionEncoder (see vision.h and research/vision.md).
//
// Layout of the activations for N patches (merge order: each 4 consecutive patches form one 2x2 block,
// (dy,dx) = (0,0),(0,1),(1,0),(1,1), blocks row-major; llama.cpp qwen3vl.cpp:18-31):
//   x        f32  [N][1152]          residual stream
//   h        bf16 [N][1152]          LayerNorm output, attention output, post-LN output
//   scratch  QKV: f16 [3][16][N][72] (Q, K with RoPE applied, V)  |  FFN hidden: bf16 [rows][4352] (row chunks)
//            | merger hidden: bf16 [N/4][4608]
//   out      f32  [N/4][5120]
//
// Numerics follow llama.cpp's CPU/CUDA path: BF16 weights times activations rounded to BF16 with f32 accumulate
// (ggml converts src1 to the weight's vec_dot type), f32 bias / residual / LayerNorm, GELU tanh, K/V (and Q) in
// f16 for flash attention with f32 softmax and f32 accumulate. The patch embedding runs in f32 (llama.cpp: F32
// conv). Q and K columns are stored in "pair order" (dim i at 2i, dim i+36 at 2i+1; the weight rows are permuted at
// load) so one thread holds both values of a RoPE pair in the GEMM epilogue; Q.K is unchanged by the common
// permutation. V keeps the natural order.
#include "vision.h"
#include "common.cuh"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace q27 {

namespace {

using bf16 = __nv_bfloat16;
using namespace vis;

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// Barrier as volatile asm with a memory clobber (nvcc 13.4 dropped a plain __syncthreads in qgemm.cu).
__device__ __forceinline__ void cta_sync() { asm volatile("bar.sync 0;" ::: "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t (&r)[2], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16_k8(float (&c)[4], const uint32_t (&a)[2], uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
  const __half2 h = __floats2half2_rn(x, y);
  return *(const uint32_t*)&h;
}

// llama.cpp ggml_gelu_f32 (tanh form), ggml-cpu/vec.h
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
  return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// Patch index in merge order -> (row, column) of the patch grid. nx2 = blocks per row (px / 2).
__device__ __forceinline__ void patch_hw(int p, int nx2, int& ph, int& pw) {
  const int b = p >> 2, sub = p & 3;
  const int by = b / nx2, bx = b - by * nx2;
  ph = 2 * by + (sub >> 1);
  pw = 2 * bx + (sub & 1);
}

// ---------------------------------------------------------------- patch embedding (f32)
// x[p][o] = sum_k A[p][k] Wp[o][k] + bias[o] + pos(p, o), A[p][k] = normalized pixel (c, y = ph*16+ky, x = pw*16+kx),
// k = c*256 + ky*16 + kx (ggml im2col order for the [16,16,3,1152] kernel). pos(p, o): the 48x48 table resized with
// bilinear + align corners (ggml upscale, ggml-cpu/ops.cpp and ggml-cuda/upscale.cu), coefficients precomputed on
// the host per patch column / row: int4 (i0, i1, bits of the fraction, 0).
// Tile: 64 patches x 64 outputs, k step 16, 256 threads with 4x4 outputs each.
__global__ void __launch_bounds__(256) patch_embed_kernel(const uint8_t* __restrict__ img, int W, const float* __restrict__ lut,
                                                          const float* __restrict__ wp, const float* __restrict__ bias,
                                                          const float* __restrict__ ptab, const int4* __restrict__ pcol,
                                                          const int4* __restrict__ prow, float* __restrict__ x, int n, int nx2) {
  __shared__ __align__(16) float sA[16][64 + 4];  // [k][patch]
  __shared__ __align__(16) float sB[16][64 + 4];  // [k][output]
  __shared__ float slut[256];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int p0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
  slut[tid] = lut[tid];
  int base[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int p = p0 + (tid >> 4) + 16 * e;
    base[e] = -1;
    if (p < n) {
      int ph, pw;
      patch_hw(p, nx2, ph, pw);
      base[e] = ((ph * 16) * W + pw * 16 + (tid & 15)) * 3;
    }
  }
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  for (int kt = 0; kt < 48; kt++) {
    const int c = kt >> 4, ky = kt & 15;
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 4; e++) sA[tid & 15][(tid >> 4) + 16 * e] = base[e] >= 0 ? slut[img[base[e] + ky * W * 3 + c]] : 0.f;
    {
      const float4 w4 = *(const float4*)(wp + (size_t)(o0 + (tid >> 2)) * 768 + kt * 16 + (tid & 3) * 4);
      sB[(tid & 3) * 4 + 0][tid >> 2] = w4.x;
      sB[(tid & 3) * 4 + 1][tid >> 2] = w4.y;
      sB[(tid & 3) * 4 + 2][tid >> 2] = w4.z;
      sB[(tid & 3) * 4 + 3][tid >> 2] = w4.w;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      const float4 a = *(const float4*)&sA[kk][ty * 4];
      const float4 b = *(const float4*)&sB[kk][tx * 4];
      const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
  }
  const int o = o0 + tx * 4;
  const float4 bb = *(const float4*)(bias + o);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int p = p0 + ty * 4 + i;
    if (p >= n) continue;
    int ph, pw;
    patch_hw(p, nx2, ph, pw);
    const int4 cx = pcol[pw], cy = prow[ph];
    const float dx = __int_as_float(cx.z), dy = __int_as_float(cy.z);
    const float4 ta = *(const float4*)(ptab + (size_t)(cy.x * 48 + cx.x) * kHid + o);
    const float4 tb = *(const float4*)(ptab + (size_t)(cy.x * 48 + cx.y) * kHid + o);
    const float4 tc = *(const float4*)(ptab + (size_t)(cy.y * 48 + cx.x) * kHid + o);
    const float4 td = *(const float4*)(ptab + (size_t)(cy.y * 48 + cx.y) * kHid + o);
    auto bil = [&](float a, float b, float c, float d) {
      return a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;
    };
    float4 r;
    r.x = (acc[i][0] + bb.x) + bil(ta.x, tb.x, tc.x, td.x);
    r.y = (acc[i][1] + bb.y) + bil(ta.y, tb.y, tc.y, td.y);
    r.z = (acc[i][2] + bb.z) + bil(ta.z, tb.z, tc.z, td.z);
    r.w = (acc[i][3] + bb.w) + bil(ta.w, tb.w, tc.w, td.w);
    *(float4*)(x + (size_t)p * kHid + o) = r;
  }
}

// ---------------------------------------------------------------- LayerNorm 1152, f32 -> bf16 (ggml_norm, mul, add)
__global__ void __launch_bounds__(256) layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                        const float* __restrict__ b, bf16* __restrict__ y, int rows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float4* xr = (const float4*)(x + (size_t)row * kHid);
  float4 v[9];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 9; j++) {
    v[j] = xr[j * 32 + lane];
    s += (v[j].x + v[j].y) + (v[j].z + v[j].w);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / (float)kHid;
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 9; j++) {
    const float a = v[j].x - mean, bq = v[j].y - mean, c = v[j].z - mean, d = v[j].w - mean;
    q += (a * a + bq * bq) + (c * c + d * d);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rstd = 1.0f / sqrtf(q / (float)kHid + 1e-6f);
#pragma unroll
  for (int j = 0; j < 9; j++) {
    const int idx = (j * 32 + lane) * 4;
    const float4 ww = *(const float4*)(w + idx), bb = *(const float4*)(b + idx);
    const __nv_bfloat162 lo = __floats2bfloat162_rn((v[j].x - mean) * rstd * ww.x + bb.x, (v[j].y - mean) * rstd * ww.y + bb.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn((v[j].z - mean) * rstd * ww.z + bb.z, (v[j].w - mean) * rstd * ww.w + bb.w);
    uint2 pk;
    pk.x = *(const uint32_t*)&lo;
    pk.y = *(const uint32_t*)&hi;
    *(uint2*)(y + (size_t)row * kHid + idx) = pk;
  }
}

// ---------------------------------------------------------------- BF16 GEMM with epilogues
// out = epi(A[M][K] W[N][K]^T + bias). CTA 256 x 128 x 32, 8 warps (4 along M x 2 along N, warp tile 64 x 64),
// 3-stage cp.async pipeline, ldmatrix + mma.sync m16n8k16 bf16 -> f32. Smem rows padded to 80 B (5 x 16 B, odd:
// no ldmatrix bank conflicts). CTAs walk the tiles in groups of 8 M-tiles (A and W tiles of a group stay in L2).
// Epilogues: QKV straight from the registers (RoPE pairs sit in one thread); the others go through shared memory
// in two halves of 128 rows, then each thread handles float4 column groups (coalesced; the residual loads are
// all issued before the stores).
constexpr int GBM = 256, GBN = 128, GBK = 32, GSTAGES = 3, GROW = GBK * 2 + 16;
constexpr int G_STAGE = (GBM + GBN) * GROW;
constexpr int G_CS = GBN + 8;                                  // f32 row stride of the staged C tile
constexpr int G_SMEM = GSTAGES * G_STAGE > 128 * G_CS * 4 ? GSTAGES * G_STAGE : 128 * G_CS * 4;  // 92160 B

struct GemmArgs {
  const bf16* A;
  const bf16* W;
  const float* bias;
  void* out;
  const float2* rope;
  int lda, ldw, ldo, M, N, K, nx2, n_rows;
};

template <int EPI>
__global__ void __launch_bounds__(256) gemm_bf16_kernel(GemmArgs a) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  // grouped raster (8 M-tiles per group, M fastest inside the group)
  const int nbn = a.N / GBN, nbm = (a.M + GBM - 1) / GBM;
  const int pid = blockIdx.x, width = 8 * nbn;
  const int first_m = (pid / width) * 8;
  const int gsize = min(nbm - first_m, 8);
  const int bm = first_m + (pid % width) % gsize, bn = (pid % width) / gsize;
  const int m0 = bm * GBM, n0 = bn * GBN;
  const int nk = a.K / GBK;

  auto load = [&](int st, int kt) {
    uint8_t* sA = sm + st * G_STAGE;
    uint8_t* sB = sA + GBM * GROW;
    const int k0 = kt * GBK;
#pragma unroll
    for (int i = tid; i < GBM * 4; i += 256) {
      const int r = i >> 2, c = i & 3;
      const int gr = m0 + r;
      const bool ok = gr < a.M;
      cp_async16(sA + r * GROW + c * 16, a.A + (size_t)(ok ? gr : 0) * a.lda + k0 + c * 8, ok);
    }
#pragma unroll
    for (int i = tid; i < GBN * 4; i += 256) {
      const int r = i >> 2, c = i & 3;
      cp_async16(sB + r * GROW + c * 16, a.W + (size_t)(n0 + r) * a.ldw + k0 + c * 8, true);
    }
  };

  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int st = 0; st < GSTAGES - 1; st++) {
    if (st < nk) { load(st, st); }
    cp_async_commit();
  }
  for (int kt = 0; kt < nk; kt++) {
    cp_async_wait<GSTAGES - 2>();
    cta_sync();
    {
      const int nt = kt + GSTAGES - 1;
      if (nt < nk) { load(nt % GSTAGES, nt); }
      cp_async_commit();
    }
    const uint8_t* sA = sm + (kt % GSTAGES) * G_STAGE;
    const uint8_t* sB = sA + GBM * GROW;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      uint32_t af[4][4], bfr[4][4];
#pragma unroll
      for (int i = 0; i < 4; i++) ldsm_x4(af[i], sA + (wm * 64 + i * 16 + (lane & 15)) * GROW + (ks * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
      for (int j = 0; j < 4; j++)
        ldsm_x4(bfr[j], sB + (wn * 64 + j * 16 + (lane & 7) + (lane >> 4) * 8) * GROW + (ks * 16 + ((lane >> 3) & 1) * 8) * 2);
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) mma_bf16(acc[i][j], af[i], bfr[j >> 1][(j & 1) * 2], bfr[j >> 1][(j & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2, tg = lane & 3;
  if constexpr (EPI == EPI_QKV) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int row = m0 + wm * 64 + i * 16 + g + 8 * hh;
        if (row >= a.M) continue;
        int ph, pw;
        patch_hw(row, a.nx2, ph, pw);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const int col = n0 + wn * 64 + j * 8 + 2 * tg;
          const float2 bb = *(const float2*)(a.bias + col);
          float v0 = acc[i][j][2 * hh] + bb.x, v1 = acc[i][j][2 * hh + 1] + bb.y;
          const int part = col / kHid, rem = col - part * kHid;
          const int head = rem / kHeadDim, d = rem - head * kHeadDim;
          if (part < 2) {
            const int pi = d >> 1;  // pair i: dims (i, i+36) of the head; pairs 0-17 use the row, 18-35 the column
            const float2 cs = pi < 18 ? a.rope[ph * 18 + pi] : a.rope[pw * 18 + pi - 18];
            const float r0 = v0 * cs.x - v1 * cs.y, r1 = v0 * cs.y + v1 * cs.x;
            v0 = r0;
            v1 = r1;
          }
          *(__half2*)((__half*)a.out + ((size_t)(part * kHeads + head) * a.n_rows + row) * kHeadDim + d) = __floats2half2_rn(v0, v1);
        }
      }
    }
  } else {
    float* sC = (float*)sm;  // [128][G_CS]
    const int c4 = tid & 31, col = n0 + c4 * 4;
    const float4 bb = *(const float4*)(a.bias + col);
#pragma unroll
    for (int half = 0; half < 2; half++) {
      cta_sync();  // main loop done / previous half consumed
      if ((wm >> 1) == half) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int rl = (wm & 1) * 64 + i * 16 + g + 8 * hh;
#pragma unroll
            for (int j = 0; j < 8; j++)
              *(float2*)(sC + rl * G_CS + wn * 64 + j * 8 + 2 * tg) = make_float2(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
          }
      }
      cta_sync();
      const int rbase = m0 + half * 128 + (tid >> 5);
      if constexpr (EPI == EPI_RES) {
        float4 xr[16];
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const int row = rbase + 8 * k;
          if (row < a.M) xr[k] = *(const float4*)((const float*)a.out + (size_t)row * a.ldo + col);
        }
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const int row = rbase + 8 * k;
          if (row < a.M) {
            const float4 c = *(const float4*)(sC + ((tid >> 5) + 8 * k) * G_CS + c4 * 4);
            float4 r;
            r.x = (c.x + bb.x) + xr[k].x;
            r.y = (c.y + bb.y) + xr[k].y;
            r.z = (c.z + bb.z) + xr[k].z;
            r.w = (c.w + bb.w) + xr[k].w;
            *(float4*)((float*)a.out + (size_t)row * a.ldo + col) = r;
          }
        }
      } else {
#pragma unroll 4
        for (int k = 0; k < 16; k++) {
          const int row = rbase + 8 * k;
          if (row < a.M) {
            const float4 c = *(const float4*)(sC + ((tid >> 5) + 8 * k) * G_CS + c4 * 4);
            const float v0 = c.x + bb.x, v1 = c.y + bb.y, v2 = c.z + bb.z, v3 = c.w + bb.w;
            if constexpr (EPI == EPI_GELU) {
              const __nv_bfloat162 lo = __floats2bfloat162_rn(gelu_tanh(v0), gelu_tanh(v1));
              const __nv_bfloat162 hi = __floats2bfloat162_rn(gelu_tanh(v2), gelu_tanh(v3));
              uint2 pk;
              pk.x = *(const uint32_t*)&lo;
              pk.y = *(const uint32_t*)&hi;
              *(uint2*)((bf16*)a.out + (size_t)row * a.ldo + col) = pk;
            } else {
              *(float4*)((float*)a.out + (size_t)row * a.ldo + col) = make_float4(v0, v1, v2, v3);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------- attention (non-causal, head dim 72)
// Grid (ceil(n/128), 16 heads), 8 warps x 16 query rows. K/V tiles of 64 positions, double-buffered with cp.async.
// Rows are 144 B in shared memory (9 x 16 B, odd: no ldmatrix bank conflicts), so head dim 72 needs no padding:
// S = Q K^T uses 4 k16 MMAs + 1 m16n8k8 MMA, O += P V uses 9 n-tiles of 8 dims. f16 inputs, f32 scores, online
// softmax in f32 (exp2 with the 1/sqrt(72) scale folded in), P rounded to f16, f32 accumulators.
constexpr int AROW = kHeadDim * 2;         // 144 B
constexpr int ABQ = 128, ABK = 64;
constexpr int A_STAGE = 2 * ABK * AROW;   // K|V tile, 18432 B (= the Q tile)
constexpr int A_SMEM = 2 * A_STAGE;       // 36864 B

template <bool PVH>
__global__ void __launch_bounds__(256, 2) vattn_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                                                       const __half* __restrict__ v, bf16* __restrict__ out, int n,
                                                       float scale_log2) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int head = blockIdx.y, q0 = blockIdx.x * ABQ;
  const size_t hoff = (size_t)head * n * kHeadDim;
  const __half* Q = q + hoff;
  const __half* K = k + hoff;
  const __half* V = v + hoff;
  const int ntiles = (n + ABK - 1) / ABK;

  auto load_kv = [&](int st, int t) {
    uint8_t* dst = sm + st * A_STAGE;
    const int p0 = t * ABK;
    for (int i = tid; i < 2 * ABK * 9; i += 256) {
      const int row = i / 9, c = i - (i / 9) * 9;
      const int p = p0 + (row & (ABK - 1));
      const bool ok = p < n;
      cp_async16(dst + row * AROW + c * 16, (row < ABK ? K : V) + (size_t)(ok ? p : 0) * kHeadDim + c * 8, ok);
    }
  };
  {  // Q tile into stage 1 (free until the loop loads tile 1)
    uint8_t* dst = sm + A_STAGE;
    for (int i = tid; i < ABQ * 9; i += 256) {
      const int row = i / 9, c = i - (i / 9) * 9;
      const int p = q0 + row;
      const bool ok = p < n;
      cp_async16(dst + row * AROW + c * 16, Q + (size_t)(ok ? p : 0) * kHeadDim + c * 8, ok);
    }
  }
  load_kv(0, 0);
  cp_async_commit();
  cp_async_wait<0>();
  cta_sync();
  uint32_t qf[4][4], qt[2];
  {
    const uint8_t* sQ = sm + A_STAGE + warp * 16 * AROW;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) ldsm_x4(qf[ks], sQ + (lane & 15) * AROW + (ks * 16 + (lane >> 4) * 8) * 2);
    ldsm_x2(qt, sQ + (lane & 15) * AROW + 128);
  }
  cta_sync();

  const int tg = lane & 3;
  float o[9][4];
#pragma unroll
  for (int i = 0; i < 9; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  for (int it = 0; it < ntiles; it++) {
    if (it + 1 < ntiles) { load_kv((it + 1) & 1, it + 1); }
    cp_async_commit();
    cp_async_wait<1>();
    cta_sync();
    const uint8_t* sK = sm + (it & 1) * A_STAGE;
    const uint8_t* sV = sK + ABK * AROW;

    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
#pragma unroll
      for (int np = 0; np < 4; np++) {
        uint32_t b[4];
        ldsm_x4(b, sK + (np * 16 + (lane & 7) + (lane >> 4) * 8) * AROW + (ks * 16 + ((lane >> 3) & 1) * 8) * 2);
        mma_f16(s[2 * np], qf[ks], b[0], b[1]);
        mma_f16(s[2 * np + 1], qf[ks], b[2], b[3]);
      }
    }
#pragma unroll
    for (int nq = 0; nq < 2; nq++) {  // dims 64..71
      uint32_t b[4];
      ldsm_x4(b, sK + (nq * 32 + lane) * AROW + 128);
#pragma unroll
      for (int j = 0; j < 4; j++) mma_f16_k8(s[nq * 4 + j], qt, b[j]);
    }
    if ((it + 1) * ABK > n) {  // last, partial tile: mask positions >= n
      const int p0 = it * ABK;
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          if (p0 + nt * 8 + 2 * tg + e >= n) { s[nt][e] = -INFINITY; s[nt][2 + e] = -INFINITY; }
        }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float c0 = exp2f((m0 - mx0) * scale_log2), c1 = exp2f((m1 - mx1) * scale_log2);
    m0 = mx0;
    m1 = mx1;
    const float b0 = mx0 * scale_log2, b1 = mx1 * scale_log2;
    float add0 = 0.f, add1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = exp2f(fmaf(s[nt][0], scale_log2, -b0));
      s[nt][1] = exp2f(fmaf(s[nt][1], scale_log2, -b0));
      s[nt][2] = exp2f(fmaf(s[nt][2], scale_log2, -b1));
      s[nt][3] = exp2f(fmaf(s[nt][3], scale_log2, -b1));
      add0 += s[nt][0] + s[nt][1];
      add1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * c0 + add0;
    l1 = l1 * c1 + add1;
#pragma unroll
    for (int i = 0; i < 9; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
    // PVH: the 64 positions of this tile are summed with f16 accumulators (twice the f32-accumulate MMA rate on
    // GeForce cards; P <= 1 here), then added to the f32 accumulators once per tile.
    uint32_t oh[9][2];
    if constexpr (PVH) {
#pragma unroll
      for (int i = 0; i < 9; i++) oh[i][0] = oh[i][1] = 0u;
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t pa[4];
      pa[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dn = 0; dn < 4; dn++) {
        uint32_t b[4];
        ldsm_x4_t(b, sV + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * AROW + (dn * 16 + (lane >> 4) * 8) * 2);
        if constexpr (PVH) {
          mma_f16h(oh[2 * dn], pa, b[0], b[1]);
          mma_f16h(oh[2 * dn + 1], pa, b[2], b[3]);
        } else {
          mma_f16(o[2 * dn], pa, b[0], b[1]);
          mma_f16(o[2 * dn + 1], pa, b[2], b[3]);
        }
      }
      uint32_t b2[2];
      ldsm_x2_t(b2, sV + (kk * 16 + (lane & 15)) * AROW + 128);
      if constexpr (PVH) {
        mma_f16h(oh[8], pa, b2[0], b2[1]);
      } else {
        mma_f16(o[8], pa, b2[0], b2[1]);
      }
    }
    if constexpr (PVH) {
#pragma unroll
      for (int i = 0; i < 9; i++) {
        const float2 lo = __half22float2(*(const __half2*)&oh[i][0]), hi = __half22float2(*(const __half2*)&oh[i][1]);
        o[i][0] += lo.x;
        o[i][1] += lo.y;
        o[i][2] += hi.x;
        o[i][3] += hi.y;
      }
    }
    cta_sync();  // this stage is overwritten by the load issued in the next iteration
  }
  cp_async_wait<0>();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const float i0 = 1.0f / l0, i1 = 1.0f / l1;
  const int r0 = q0 + warp * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
  for (int i = 0; i < 9; i++) {
    const int d = head * kHeadDim + i * 8 + 2 * tg;
    if (r0 < n) *(__nv_bfloat162*)(out + (size_t)r0 * kHid + d) = __floats2bfloat162_rn(o[i][0] * i0, o[i][1] * i0);
    if (r1 < n) *(__nv_bfloat162*)(out + (size_t)r1 * kHid + d) = __floats2bfloat162_rn(o[i][2] * i1, o[i][3] * i1);
  }
}

void init_kernels_once() {
  static bool done[64] = {};
  int dev;
  CK(cudaGetDevice(&dev));
  if (done[dev]) return;
  CK(cudaFuncSetAttribute(gemm_bf16_kernel<EPI_QKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_bf16_kernel<EPI_RES>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_bf16_kernel<EPI_GELU>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_bf16_kernel<EPI_F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(vattn_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM));
  CK(cudaFuncSetAttribute(vattn_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM));
  done[dev] = true;
}

// Restores the caller's current device on scope exit.
struct DevGuard {
  int prev = -1;
  explicit DevGuard(int dev) {
    CK(cudaGetDevice(&prev));
    if (prev != dev) CK(cudaSetDevice(dev));
  }
  ~DevGuard() {
    int cur;
    if (cudaGetDevice(&cur) == cudaSuccess && cur != prev) cudaSetDevice(prev);
  }
};

}  // namespace

// ---------------------------------------------------------------- kernel entry points
namespace vis {

void gemm(int epi, const bf16* A, int lda, const bf16* W, int ldw, const float* bias, int M, int N, int K, void* out, int ldo,
          cudaStream_t s, const float2* rope, int nx2, int n_rows) {
  init_kernels_once();
  if (K % GBK || N % GBN || M < 1) throw std::runtime_error("vis::gemm: bad shape");
  if (epi == EPI_QKV && (N != 3 * kHid || !rope || nx2 < 1 || n_rows < M)) throw std::runtime_error("vis::gemm: bad QKV args");
  GemmArgs a{A, W, bias, out, rope, lda, ldw, ldo, M, N, K, nx2, n_rows};
  const int nblk = (N / GBN) * ((M + GBM - 1) / GBM);
  switch (epi) {
    case EPI_QKV: gemm_bf16_kernel<EPI_QKV><<<nblk, 256, G_SMEM, s>>>(a); break;
    case EPI_RES: gemm_bf16_kernel<EPI_RES><<<nblk, 256, G_SMEM, s>>>(a); break;
    case EPI_GELU: gemm_bf16_kernel<EPI_GELU><<<nblk, 256, G_SMEM, s>>>(a); break;
    case EPI_F32: gemm_bf16_kernel<EPI_F32><<<nblk, 256, G_SMEM, s>>>(a); break;
    default: throw std::runtime_error("vis::gemm: bad epilogue");
  }
  CK(cudaGetLastError());
}

void attention(const __half* q, const __half* k, const __half* v, bf16* out, int n, cudaStream_t s) {
  init_kernels_once();
  if (n < 1) throw std::runtime_error("vis::attention: empty");
  const float scale_log2 = (float)(1.0 / std::sqrt((double)kHeadDim) * 1.4426950408889634);
  static const bool pvh = [] { const char* e = getenv("Q27_VIS_PV16"); return e && e[0] == '1'; }();
  if (pvh) vattn_kernel<true><<<dim3((n + ABQ - 1) / ABQ, kHeads), 256, A_SMEM, s>>>(q, k, v, out, n, scale_log2);
  else vattn_kernel<false><<<dim3((n + ABQ - 1) / ABQ, kHeads), 256, A_SMEM, s>>>(q, k, v, out, n, scale_log2);
  CK(cudaGetLastError());
}

void layernorm(const float* x, const float* w, const float* b, bf16* y, int rows, cudaStream_t s) {
  layernorm_kernel<<<(rows + 7) / 8, 256, 0, s>>>(x, w, b, y, rows);
  CK(cudaGetLastError());
}

}  // namespace vis

// ---------------------------------------------------------------- VisionEncoder
namespace {
struct LayerW {
  const float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *qkv_b, *out_b, *up_b, *down_b;
  const bf16 *qkv_w, *out_w, *up_w, *down_w;
};
enum Stage { ST_UPLOAD, ST_PATCH, ST_LN, ST_QKV, ST_ATTN, ST_OUT, ST_UP, ST_DOWN, ST_MERGER, ST_N };
}  // namespace

struct VisionEncoder::Impl {
  int dev = 0;
  uint8_t* wbuf = nullptr;
  size_t wbytes = 0;
  std::vector<LayerW> L;
  const float *patch_w = nullptr, *patch_b = nullptr, *pos_tab = nullptr, *post_w = nullptr, *post_b = nullptr,
              *mm0_b = nullptr, *mm2_b = nullptr, *lut = nullptr;
  const bf16 *mm0_w = nullptr, *mm2_w = nullptr;

  // activations (sized for cap_n patches)
  int cap_n = 0;
  uint8_t* abuf = nullptr;
  size_t abytes = 0;
  float* x = nullptr;
  bf16* h = nullptr;
  uint8_t* scratch = nullptr;
  size_t scratch_bytes = 0;
  float* out = nullptr;
  uint8_t* dimg = nullptr;
  // per-image tables: rope float2 [npos][18], pcol int4 [px], prow int4 [py] (device + pinned host)
  uint8_t* dtab = nullptr;
  uint8_t* htab = nullptr;
  size_t tab_cap = 0;
  uint8_t* himg = nullptr;
  size_t himg_cap = 0;
  cudaEvent_t upload_done = nullptr;

  bool profile = false;
  int debug_stop = -1;
  VisionTimes t;
  std::vector<cudaEvent_t> ev;
  std::vector<int> ev_stage;
  int nev = 0;

  void mark(int stage, cudaStream_t s) {
    if (!profile) return;
    if (nev >= (int)ev.size()) {
      cudaEvent_t e;
      CK(cudaEventCreate(&e));
      ev.push_back(e);
      ev_stage.push_back(0);
    }
    CK(cudaEventRecord(ev[nev], s));
    ev_stage[nev] = stage;
    nev++;
  }

  void ensure(const VisionPlan& p) {
    if (p.n_patches > cap_n) {
      if (abuf) CK(cudaFree(abuf));
      abuf = nullptr;
      const size_t n = (size_t)p.n_patches;
      const size_t T = n / 4;
      auto al = [](size_t b) { return (b + 255) & ~(size_t)255; };
      const size_t bx = al(n * kHid * 4), bh = al(n * kHid * 2);
      // QKV, merger hidden, and at least one FFN chunk of min(n, GBM) rows
      scratch_bytes = al(std::max({n * 3 * kHid * 2, T * kMerge * 2, std::min<size_t>(n, GBM) * kFfnPad * 2}));
      const size_t bo = al(T * VisionEncoder::kEmbd * 4), bi = al(n * 256 * 3);
      abytes = bx + bh + scratch_bytes + bo + bi;
      CK(cudaMalloc(&abuf, abytes));
      x = (float*)abuf;
      h = (bf16*)(abuf + bx);
      scratch = abuf + bx + bh;
      out = (float*)(scratch + scratch_bytes);
      dimg = (uint8_t*)out + bo;
      cap_n = p.n_patches;
    }
    const size_t img_bytes = (size_t)p.width * p.height * 3;
    if (img_bytes > himg_cap) {
      if (himg) CK(cudaFreeHost(himg));
      CK(cudaHostAlloc(&himg, img_bytes, cudaHostAllocDefault));
      himg_cap = img_bytes;
    }
    const size_t npos = (size_t)std::max(p.px, p.py);
    const size_t tb = npos * 18 * sizeof(float2) + (size_t)(p.px + p.py) * sizeof(int4);
    if (tb > tab_cap) {
      if (dtab) CK(cudaFree(dtab));
      if (htab) CK(cudaFreeHost(htab));
      CK(cudaMalloc(&dtab, tb));
      CK(cudaHostAlloc(&htab, tb, cudaHostAllocDefault));
      tab_cap = tb;
    }
  }
};

namespace {

void check_hp(const GGUF& g, const char* key, int64_t want) {
  const int64_t v = g.get_int(key);
  if (v != want) throw std::runtime_error(std::string("mmproj: ") + key + " = " + std::to_string(v) + ", expected " + std::to_string(want));
}

}  // namespace

VisionEncoder::VisionEncoder(const std::string& path, int device) : d_(new Impl) {
  Impl& m = *d_;
  m.dev = device;
  DevGuard guard(device);
  init_kernels_once();
  GGUF g(path);
  if (g.get_str("clip.projector_type") != "qwen3vl_merger") throw std::runtime_error("mmproj: projector is not qwen3vl_merger");
  check_hp(g, "clip.vision.block_count", 27);
  check_hp(g, "clip.vision.embedding_length", kHid);
  check_hp(g, "clip.vision.feed_forward_length", kFfn);
  check_hp(g, "clip.vision.attention.head_count", kHeads);
  check_hp(g, "clip.vision.patch_size", 16);
  check_hp(g, "clip.vision.projection_dim", kEmbd);
  check_hp(g, "clip.vision.spatial_merge_size", 2);
  const int nl = 27;

  auto ten = [&](const std::string& name, GType type, int64_t ne0, int64_t ne1) -> const GTensor& {
    const GTensor& t = g.tensor(name);
    if (t.type != type || t.ne[0] != ne0 || t.ne[1] * t.ne[2] * t.ne[3] != ne1)
      throw std::runtime_error("mmproj: unexpected type or shape of " + name);
    return t;
  };

  // Two passes: the first only counts bytes, the second allocates once and uploads.
  for (int pass = 0; pass < 2; pass++) {
    const bool dry = pass == 0;
    size_t off = 0;
    auto take = [&](size_t bytes) -> uint8_t* {
      uint8_t* p = dry ? nullptr : m.wbuf + off;
      off += (bytes + 255) & ~(size_t)255;
      return p;
    };
    auto up_raw = [&](const void* src, size_t bytes) -> uint8_t* {
      uint8_t* p = take(bytes);
      if (!dry) CK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
      return p;
    };
    auto f32 = [&](const std::string& name, int64_t n) {
      return (const float*)up_raw(ten(name, GType::F32, n, 1).data, n * 4);
    };
    auto bf = [&](const std::string& name, int64_t K, int64_t N) {
      return (const bf16*)up_raw(ten(name, GType::BF16, K, N).data, (size_t)K * N * 2);
    };
    std::vector<uint8_t> tmp;
    if (!dry) m.L.resize(nl);
    for (int l = 0; l < nl; l++) {
      const std::string b = "v.blk." + std::to_string(l) + ".";
      LayerW w;
      w.ln1_w = f32(b + "ln1.weight", kHid);
      w.ln1_b = f32(b + "ln1.bias", kHid);
      w.ln2_w = f32(b + "ln2.weight", kHid);
      w.ln2_b = f32(b + "ln2.bias", kHid);
      {  // QKV: Q and K rows in pair order (new row 2i <- dim i, 2i+1 <- dim i+36, per head)
        const GTensor& tw = ten(b + "attn_qkv.weight", GType::BF16, kHid, 3 * kHid);
        const GTensor& tb = ten(b + "attn_qkv.bias", GType::F32, 3 * kHid, 1);
        auto src_row = [](int r) {
          const int part = r / kHid;
          if (part == 2) return r;
          const int rem = r % kHid, head = rem / kHeadDim, j = rem % kHeadDim;
          const int orig = (j & 1) ? j / 2 + kHeadDim / 2 : j / 2;
          return part * kHid + head * kHeadDim + orig;
        };
        uint8_t* pw = take((size_t)3 * kHid * kHid * 2);
        uint8_t* pb = take((size_t)3 * kHid * 4);
        if (!dry) {
          tmp.resize((size_t)3 * kHid * kHid * 2);
          for (int r = 0; r < 3 * kHid; r++) memcpy(tmp.data() + (size_t)r * kHid * 2, tw.data + (size_t)src_row(r) * kHid * 2, kHid * 2);
          CK(cudaMemcpy(pw, tmp.data(), tmp.size(), cudaMemcpyHostToDevice));
          std::vector<float> bb(3 * kHid);
          for (int r = 0; r < 3 * kHid; r++) bb[r] = ((const float*)tb.data)[src_row(r)];
          CK(cudaMemcpy(pb, bb.data(), bb.size() * 4, cudaMemcpyHostToDevice));
        }
        w.qkv_w = (const bf16*)pw;
        w.qkv_b = (const float*)pb;
      }
      w.out_w = bf(b + "attn_out.weight", kHid, kHid);
      w.out_b = f32(b + "attn_out.bias", kHid);
      {  // FFN up padded to 4352 rows, down padded to 4352 columns (zeros)
        const GTensor& tu = ten(b + "ffn_up.weight", GType::BF16, kHid, kFfn);
        const GTensor& tub = ten(b + "ffn_up.bias", GType::F32, kFfn, 1);
        const GTensor& td = ten(b + "ffn_down.weight", GType::BF16, kFfn, kHid);
        uint8_t* pu = take((size_t)kFfnPad * kHid * 2);
        uint8_t* pub = take((size_t)kFfnPad * 4);
        uint8_t* pd = take((size_t)kHid * kFfnPad * 2);
        if (!dry) {
          tmp.assign((size_t)kFfnPad * kHid * 2, 0);
          memcpy(tmp.data(), tu.data, (size_t)kFfn * kHid * 2);
          CK(cudaMemcpy(pu, tmp.data(), tmp.size(), cudaMemcpyHostToDevice));
          std::vector<float> bb(kFfnPad, 0.f);
          memcpy(bb.data(), tub.data, kFfn * 4);
          CK(cudaMemcpy(pub, bb.data(), bb.size() * 4, cudaMemcpyHostToDevice));
          tmp.assign((size_t)kHid * kFfnPad * 2, 0);
          for (int r = 0; r < kHid; r++) memcpy(tmp.data() + (size_t)r * kFfnPad * 2, td.data + (size_t)r * kFfn * 2, kFfn * 2);
          CK(cudaMemcpy(pd, tmp.data(), tmp.size(), cudaMemcpyHostToDevice));
        }
        w.up_w = (const bf16*)pu;
        w.up_b = (const float*)pub;
        w.down_w = (const bf16*)pd;
      }
      w.down_b = f32(b + "ffn_down.bias", kHid);
      if (!dry) m.L[l] = w;
    }
    {  // patch embedding: the two temporal kernels summed (still image = frame repeated), f32 [1152][768].
       // llama.cpp's ggml_conv_2d builds its im2col in F16 (ggml.c:4772) and the matmul rounds the F32 kernel to
       // F16 too, then the two convolutions are added: so each kernel is rounded to f16 before the sum here, and
       // the pixel table below is rounded to f16 as well.
      const GTensor& t0 = g.tensor("v.patch_embd.weight");
      const GTensor& t1 = g.tensor("v.patch_embd.weight.1");
      if (t0.type != GType::F32 || t1.type != GType::F32 || t0.ne[0] != 16 || t0.ne[1] != 16 || t0.ne[2] != 3 || t0.ne[3] != kHid ||
          memcmp(t0.ne, t1.ne, sizeof(t0.ne)) != 0)
        throw std::runtime_error("mmproj: unexpected patch embedding shape");
      uint8_t* p = take((size_t)kHid * 768 * 4);
      if (!dry) {
        std::vector<float> s((size_t)kHid * 768);
        const float* a0 = (const float*)t0.data;
        const float* a1 = (const float*)t1.data;
        for (size_t i = 0; i < s.size(); i++) s[i] = __half2float(__float2half(a0[i])) + __half2float(__float2half(a1[i]));
        CK(cudaMemcpy(p, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
      }
      m.patch_w = (const float*)p;
    }
    m.patch_b = f32("v.patch_embd.bias", kHid);
    m.pos_tab = (const float*)up_raw(ten("v.position_embd.weight", GType::F32, kHid, 48 * 48).data, (size_t)kHid * 48 * 48 * 4);
    m.post_w = f32("v.post_ln.weight", kHid);
    m.post_b = f32("v.post_ln.bias", kHid);
    m.mm0_w = bf("mm.0.weight", kMerge, kMerge);
    m.mm0_b = f32("mm.0.bias", kMerge);
    m.mm2_w = bf("mm.2.weight", kMerge, kEmbd);
    m.mm2_b = f32("mm.2.bias", kEmbd);
    {  // pixel normalization as llama.cpp (clip-impl.h from_u8 + normalize): (v / 255 - 0.5) / 0.5, then f16 (im2col)
      float lut[256];
      for (int i = 0; i < 256; i++) {
        const float f = (float)i / 255.0f;
        lut[i] = __half2float(__float2half((f - 0.5f) / 0.5f));
      }
      m.lut = (const float*)up_raw(lut, sizeof(lut));
    }
    if (dry) {
      m.wbytes = off;
      CK(cudaMalloc(&m.wbuf, m.wbytes));
    }
  }
  CK(cudaEventCreateWithFlags(&m.upload_done, cudaEventDisableTiming));
  CK(cudaEventRecord(m.upload_done, 0));
  CK(cudaDeviceSynchronize());
}

VisionEncoder::~VisionEncoder() {
  Impl& m = *d_;
  int prev = 0;
  cudaGetDevice(&prev);
  cudaSetDevice(m.dev);
  cudaDeviceSynchronize();
  for (auto e : m.ev) cudaEventDestroy(e);
  if (m.upload_done) cudaEventDestroy(m.upload_done);
  if (m.abuf) cudaFree(m.abuf);
  if (m.dtab) cudaFree(m.dtab);
  if (m.htab) cudaFreeHost(m.htab);
  if (m.himg) cudaFreeHost(m.himg);
  if (m.wbuf) cudaFree(m.wbuf);
  cudaSetDevice(prev);
}

int VisionEncoder::device() const { return d_->dev; }
size_t VisionEncoder::weight_bytes() const { return d_->wbytes; }
size_t VisionEncoder::scratch_bytes() const { return d_->abytes + d_->tab_cap; }
void VisionEncoder::set_profile(bool on) { d_->profile = on; }
void VisionEncoder::set_debug_stop(int n_layers) { d_->debug_stop = n_layers; }
const float* VisionEncoder::debug_residual() const { return d_->x; }
const VisionTimes& VisionEncoder::times() const { return d_->t; }

const float* VisionEncoder::encode(const uint8_t* rgb, int w, int h, const VisionPlan& p, cudaStream_t s) {
  using clk = std::chrono::steady_clock;
  Impl& m = *d_;
  DevGuard guard(m.dev);
  if (p.n_patches < 4 || p.px % 2 || p.py % 2) throw std::runtime_error("vision encode: bad plan");
  // wait until the previous upload has left the pinned buffers (and before any buffer is reallocated)
  CK(cudaEventSynchronize(m.upload_done));
  if (p.n_patches > m.cap_n) CK(cudaDeviceSynchronize());  // the old activations may still be in use
  m.ensure(p);
  const int N = p.n_patches, T = p.n_tokens;

  // host side: resize and build the tables
  const auto t0 = clk::now();
  resize_into(rgb, w, h, p, m.himg);
  const int npos = std::max(p.px, p.py);
  float2* hrope = (float2*)m.htab;
  int4* hcol = (int4*)(hrope + (size_t)npos * 18);
  int4* hrow = hcol + p.px;
  for (int pos = 0; pos < npos; pos++)  // 2D RoPE, vision mode: theta = pos * 10000^(-2i/36), i = 0..17
    for (int i = 0; i < 18; i++) {
      const double th = pos * std::pow(10000.0, -2.0 * i / 36.0);
      hrope[pos * 18 + i] = make_float2((float)std::cos(th), (float)std::sin(th));
    }
  // position table resize 48x48 -> px x py, bilinear with align corners (ggml upscale; float math as ggml)
  auto coeffs = [](int dst, int src, int4* outc) {
    float sf = (float)dst / src;
    if (dst > 1 && src > 1) sf = (float)(dst - 1) / (src - 1);
    for (int i = 0; i < dst; i++) {
      const float f = ((float)i + 0.0f) / sf - 0.0f;
      int i0 = (int)floorf(f), i1 = i0 + 1;
      i0 = std::max(0, std::min(i0, src - 1));
      i1 = std::max(0, std::min(i1, src - 1));
      float d = f - (float)i0;
      d = std::max(0.0f, std::min(d, 1.0f));
      int db;
      memcpy(&db, &d, 4);
      outc[i] = make_int4(i0, i1, db, 0);
    }
  };
  coeffs(p.px, 48, hcol);
  coeffs(p.py, 48, hrow);
  m.t = VisionTimes();
  m.t.resize_cpu = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  m.nev = 0;

  m.mark(-1, s);
  const size_t tab_bytes = (size_t)npos * 18 * sizeof(float2) + (size_t)(p.px + p.py) * sizeof(int4);
  CK(cudaMemcpyAsync(m.dimg, m.himg, (size_t)p.width * p.height * 3, cudaMemcpyHostToDevice, s));
  CK(cudaMemcpyAsync(m.dtab, m.htab, tab_bytes, cudaMemcpyHostToDevice, s));
  CK(cudaEventRecord(m.upload_done, s));
  const float2* drope = (const float2*)m.dtab;
  const int4* dcol = (const int4*)(drope + (size_t)npos * 18);
  const int4* drow = dcol + p.px;
  m.mark(ST_UPLOAD, s);

  const int nx2 = p.px / 2;
  patch_embed_kernel<<<dim3(kHid / 64, (N + 63) / 64), 256, 0, s>>>(m.dimg, p.width, m.lut, m.patch_w, m.patch_b, m.pos_tab, dcol,
                                                                     drow, m.x, N, nx2);
  CK(cudaGetLastError());
  m.mark(ST_PATCH, s);

  __half* q = (__half*)m.scratch;
  __half* k = q + (size_t)kHeads * N * kHeadDim;
  __half* v = k + (size_t)kHeads * N * kHeadDim;
  bf16* hid = (bf16*)m.scratch;
  int chunk = (int)std::min<size_t>(N, m.scratch_bytes / ((size_t)kFfnPad * 2));
  if (chunk < N) chunk = std::max(GBM, chunk / GBM * GBM);
  const int nlayers = m.debug_stop >= 0 ? std::min(m.debug_stop, 27) : 27;
  for (int l = 0; l < nlayers; l++) {
    const LayerW& w = m.L[l];
    vis::layernorm(m.x, w.ln1_w, w.ln1_b, m.h, N, s);
    m.mark(ST_LN, s);
    vis::gemm(EPI_QKV, m.h, kHid, w.qkv_w, kHid, w.qkv_b, N, 3 * kHid, kHid, q, 0, s, drope, nx2, N);
    m.mark(ST_QKV, s);
    vis::attention(q, k, v, m.h, N, s);
    m.mark(ST_ATTN, s);
    vis::gemm(EPI_RES, m.h, kHid, w.out_w, kHid, w.out_b, N, kHid, kHid, m.x, kHid, s);
    m.mark(ST_OUT, s);
    vis::layernorm(m.x, w.ln2_w, w.ln2_b, m.h, N, s);
    m.mark(ST_LN, s);
    for (int r0 = 0; r0 < N; r0 += chunk) {
      const int rows = std::min(chunk, N - r0);
      vis::gemm(EPI_GELU, m.h + (size_t)r0 * kHid, kHid, w.up_w, kHid, w.up_b, rows, kFfnPad, kHid, hid, kFfnPad, s);
      m.mark(ST_UP, s);
      vis::gemm(EPI_RES, hid, kFfnPad, w.down_w, kFfnPad, w.down_b, rows, kHid, kFfnPad, m.x + (size_t)r0 * kHid, kHid, s);
      m.mark(ST_DOWN, s);
    }
  }
  // post-LN, then the merger on [T][4608] (4 consecutive patches = one token)
  if (m.debug_stop < 0) {
    vis::layernorm(m.x, m.post_w, m.post_b, m.h, N, s);
    vis::gemm(EPI_GELU, m.h, kMerge, m.mm0_w, kMerge, m.mm0_b, T, kMerge, kMerge, hid, kMerge, s);
    vis::gemm(EPI_F32, hid, kMerge, m.mm2_w, kMerge, m.mm2_b, T, kEmbd, kMerge, m.out, kEmbd, s);
    m.mark(ST_MERGER, s);
  }

  if (m.profile) {
    CK(cudaStreamSynchronize(s));
    double st[ST_N] = {};
    for (int i = 1; i < m.nev; i++) {
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, m.ev[i - 1], m.ev[i]));
      st[m.ev_stage[i]] += ms;
    }
    float tot = 0.f;
    CK(cudaEventElapsedTime(&tot, m.ev[0], m.ev[m.nev - 1]));
    m.t.upload = st[ST_UPLOAD];
    m.t.patch_embed = st[ST_PATCH];
    m.t.layernorm = st[ST_LN];
    m.t.qkv = st[ST_QKV];
    m.t.attention = st[ST_ATTN];
    m.t.out_proj = st[ST_OUT];
    m.t.ffn_up = st[ST_UP];
    m.t.ffn_down = st[ST_DOWN];
    m.t.merger = st[ST_MERGER];
    m.t.gpu_total = tot;
  }
  return m.out;
}

}  // namespace q27
