// Sampling on the GPU (see sampling.h).
#include "sampling.h"
#include "common.cuh"

#include <cfloat>

namespace q27 {

namespace {

constexpr int NB = 48;       // blocks per row in the first top-k stage
constexpr int THREADS = 256;

// Larger value wins; equal values: smaller id wins (deterministic on both cards).
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// ---- warp-level top-32 lists: lane l holds the l-th best (value, id) pair, sorted by better().
__device__ __forceinline__ void cswap(float& v, int& id, int j, bool keep_better) {
  const float ov = __shfl_xor_sync(0xffffffffu, v, j);
  const int oi = __shfl_xor_sync(0xffffffffu, id, j);
  if (keep_better == better(ov, oi, v, id)) { v = ov; id = oi; }
}
// Bitonic sort of 32 pairs across the warp, best first.
__device__ __forceinline__ void warp_sort(float& v, int& id, int lane) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) cswap(v, id, j, ((lane & k) == 0) == ((lane & j) == 0));
}
// (tv, ti) := best 32 of the two sorted lists (tv, ti) and (bv, bi).
__device__ __forceinline__ void warp_merge(float& tv, int& ti, float bv, int bi, int lane) {
  const float rv = __shfl_sync(0xffffffffu, bv, 31 - lane);
  const int ri = __shfl_sync(0xffffffffu, bi, 31 - lane);
  if (better(rv, ri, tv, ti)) { tv = rv; ti = ri; }  // bitonic now
#pragma unroll
  for (int j = 16; j > 0; j >>= 1) cswap(tv, ti, j, (lane & j) == 0);
}
// Merge the 8 warps' lists of a 256-thread block into warp 0.
__device__ __forceinline__ void block_merge(float& tv, int& ti, int lane, int warp) {
  __shared__ float sv[8][32];
  __shared__ int si[8][32];
  __syncthreads();
  sv[warp][lane] = tv;
  si[warp][lane] = ti;
  __syncthreads();
  if (warp == 0)
    for (int w = 1; w < 8; w++) warp_merge(tv, ti, sv[w][lane], si[w][lane], lane);
}

// Stage 1: grid (NB, rows), 256 threads. Each warp keeps the top 32 of the values it scans (batches of 32
// that cannot enter the list are skipped); the block merges its 8 lists and writes the top k (k <= 32).
__global__ void topk_blocks_kernel(const float* __restrict__ logits, int n, int k, int vocab0, float* __restrict__ vals,
                                   int* __restrict__ ids, const int* __restrict__ idmap) {
  pdl_wait();
  pdl_trigger();
  const int row = blockIdx.y, b = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int per = (n + NB - 1) / NB;
  const int beg = b * per, cnt = max(0, min(n, beg + per) - beg);
  const float* src = logits + (size_t)row * n + beg;
  float tv = -FLT_MAX;
  int ti = 0x7fffffff;
  for (int base = warp * 32; base < cnt; base += THREADS) {
    const int i = base + lane;
    float v = i < cnt ? src[i] : -FLT_MAX;
    int id = i < cnt ? (idmap ? idmap[beg + i] : vocab0 + beg + i) : 0x7fffffff;
    const float wv = __shfl_sync(0xffffffffu, tv, 31);
    const int wi = __shfl_sync(0xffffffffu, ti, 31);
    if (!__any_sync(0xffffffffu, better(v, id, wv, wi))) continue;
    warp_sort(v, id, lane);
    warp_merge(tv, ti, v, id, lane);
  }
  block_merge(tv, ti, lane, warp);
  if (warp == 0 && lane < k) {
    vals[((size_t)row * NB + b) * k + lane] = tv;
    ids[((size_t)row * NB + b) * k + lane] = ti;
  }
}

__device__ __forceinline__ float rng_uniform(uint64_t seed, int counter, int row, int salt) {
  uint64_t z = seed ^ ((uint64_t)(uint32_t)counter << 24) ^ ((uint64_t)row << 8) ^ (uint64_t)salt;
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return (float)((z >> 40) * (1.0 / 16777216.0));  // 24 bits -> [0, 1)
}

// Stage 2: one block for all rows. Merge the NB*k block candidates, exchange the card's top-k with the
// peer (two cards), merge again, then apply the sampling chain.
__global__ void topk_merge_kernel(const float* __restrict__ vals, const int* __restrict__ ids, int rows, int k, bool draft,
                                  SampleParams sp, CandRow* __restrict__ out, CandMailbox mb, bool exchange, int rank,
                                  const int* __restrict__ dstep, int n_ex, int index) {
  pdl_wait();
  pdl_trigger();
  __shared__ float top_v[4][kCand];
  __shared__ int top_i[4][kCand];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int row = 0; row < rows; row++) {
    float tv = -FLT_MAX;
    int ti = 0x7fffffff;
    for (int b = warp; b < NB; b += THREADS / 32) {
      const float v = lane < k ? vals[((size_t)row * NB + b) * k + lane] : -FLT_MAX;
      const int id = lane < k ? ids[((size_t)row * NB + b) * k + lane] : 0x7fffffff;
      warp_merge(tv, ti, v, id, lane);
    }
    block_merge(tv, ti, lane, warp);
    if (warp == 0 && lane < k) { top_v[row][lane] = tv; top_i[row][lane] = ti; }
  }
  __syncthreads();
  if (exchange) {
    const int token = (*dstep) * n_ex + index + 1;
    const int slot = index & 1;
    float* mv = mb.vals + (((size_t)slot * 2 + rank) * 4) * kCand;
    int* mi = mb.ids + (((size_t)slot * 2 + rank) * 4) * kCand;
    const float* ov = mb.vals + (((size_t)slot * 2 + (1 - rank)) * 4) * kCand;
    const int* oi = mb.ids + (((size_t)slot * 2 + (1 - rank)) * 4) * kCand;
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x) { mv[(i / k) * kCand + i % k] = top_v[i / k][i % k]; mi[(i / k) * kCand + i % k] = top_i[i / k][i % k]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      volatile int* fm = mb.flags + ((size_t)slot * 2 + rank) * 32;
      volatile const int* fo = mb.flags + ((size_t)slot * 2 + (1 - rank)) * 32;
      *fm = token;
      __threadfence_system();
      const long long t0 = clock64();
      while (*fo != token) {
        if (clock64() - t0 > 3000000000LL) { atomicAdd(mb.err, 1); break; }
      }
    }
    __syncthreads();
    __threadfence_system();
    // merge own k and peer k (both sorted): thread 0 per row, 2k elements
    if (threadIdx.x < rows) {
      const int row = threadIdx.x;
      float av[kCand], bv[kCand]; int ai[kCand], bi[kCand];
      for (int i = 0; i < k; i++) {
        av[i] = top_v[row][i]; ai[i] = top_i[row][i];
        bv[i] = ((volatile const float*)ov)[row * kCand + i]; bi[i] = ((volatile const int*)oi)[row * kCand + i];
      }
      int x = 0, y = 0;
      for (int r = 0; r < k; r++) {
        if (y >= k || (x < k && better(av[x], ai[x], bv[y], bi[y]))) { top_v[row][r] = av[x]; top_i[row][r] = ai[x]; x++; }
        else { top_v[row][r] = bv[y]; top_i[row][r] = bi[y]; y++; }
      }
    }
    __syncthreads();
  }
  // chain
  if (threadIdx.x < rows) {
    const int row = threadIdx.x;
    CandRow& o = out[row];
    if (sp.temp <= 0.f) {
      o.n = 1; o.id[0] = top_i[row][0]; o.p[0] = 1.f;
      return;
    }
    int n = k;
    if (!draft) {
      // top-p on softmax of the raw logits, keep the shortest prefix with cumulative sum >= top_p
      if (sp.top_p < 1.f) {
        const float mx = top_v[row][0];
        float sum = 0.f, e[kCand];
        for (int i = 0; i < n; i++) { e[i] = expf(top_v[row][i] - mx); sum += e[i]; }
        float cum = 0.f;
        for (int i = 0; i < n; i++) {
          cum += e[i] / sum;
          if (cum >= sp.top_p && i + 1 >= 1) { n = i + 1; break; }
        }
      }
      if (sp.min_p > 0.f) {
        const float mx = top_v[row][0];
        int m = 1;
        for (int i = 1; i < n; i++) if (expf(top_v[row][i] - mx) >= sp.min_p) m = i + 1;
        n = m;
      }
    }
    // temperature, then softmax over the kept candidates
    const float mx = top_v[row][0] / sp.temp;
    float sum = 0.f;
    for (int i = 0; i < n; i++) { o.p[i] = expf(top_v[row][i] / sp.temp - mx); sum += o.p[i]; }
    for (int i = 0; i < n; i++) { o.p[i] /= sum; o.id[i] = top_i[row][i]; }
    o.n = n;
  }
}

__global__ void sample_tokens_kernel(const CandRow* rows, int nrows, int* out, uint64_t seed, const int* counter, int salt) {
  pdl_wait();
  pdl_trigger();
  const int r = threadIdx.x;
  if (r >= nrows) return;
  const CandRow& c = rows[r];
  const float u = rng_uniform(seed, *counter, r, salt);
  float cum = 0.f;
  int pick = c.n > 0 ? c.id[c.n - 1] : 0;  // n == 0 only with NaN logits; never read id[-1]
  for (int i = 0; i < c.n; i++) { cum += c.p[i]; if (u < cum) { pick = c.id[i]; break; } }
  out[r] = pick;
}

}  // namespace

void sample_candidates(const float* logits, int rows, int n, int vocab0, int k, bool draft, const SampleParams& sp,
                       CandRow* out, const CandMailbox* mb, int rank, const int* dstep, int n_ex, int index,
                       cudaStream_t s, const int* idmap) {
  static float* vals_dev[16] = {};
  static int* ids_dev[16] = {};
  int dev; CK(cudaGetDevice(&dev));
  if (!vals_dev[dev]) {
    CK(cudaMalloc(&vals_dev[dev], sizeof(float) * 4 * NB * kCand));
    CK(cudaMalloc(&ids_dev[dev], sizeof(int) * 4 * NB * kCand));
  }
  if (k > kCand || rows > 4) throw std::runtime_error("sample_candidates: k or rows too large");
  launch_k(topk_blocks_kernel, dim3(NB, rows), THREADS, 0, s, logits, n, k, vocab0, vals_dev[dev], ids_dev[dev], idmap);
  CandMailbox m = mb ? *mb : CandMailbox{nullptr, nullptr, nullptr, nullptr};
  launch_k(topk_merge_kernel, 1, THREADS, 0, s, vals_dev[dev], ids_dev[dev], rows, k, draft, sp, out, m, mb != nullptr, rank, dstep,
                                          n_ex, index);
  CK(cudaGetLastError());
}

void sample_tokens(const CandRow* rows, int nrows, int* out_tokens, uint64_t seed, const int* counter, int salt, cudaStream_t s) {
  launch_k(sample_tokens_kernel, 1, 32, 0, s, rows, nrows, out_tokens, seed, counter, salt);
}

}  // namespace q27
