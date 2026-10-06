// Sampling on the GPU for the vocab-split output layer, and the MTP acceptance rule.
// Semantics follow llama.cpp (common/sampling.cpp, src/llama-sampler.cpp; MIT):
//   target chain: top-k -> top-p (on softmax of the raw logits) -> min-p -> temperature -> draw
//   draft chain (MTP, probabilistic): top-k 10 -> temperature -> draw
//   accept draft x if q(x) > 0 and (p(x) >= q(x) or U < p(x)/q(x)); else draw from max(0, p - q)
// temp <= 0 means greedy (top-1). Random numbers come from a counter-based hash, so both cards draw
// the same values without talking to each other.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace q27 {

constexpr int kCand = 20;  // max candidates kept per row

struct SampleParams {
  float temp = 1.0f;
  int top_k = 20;
  float top_p = 0.95f;
  float min_p = 0.0f;
  int draft_top_k = 10;
  uint64_t seed = 42;
};

// Candidate list of one row after the chain: ids sorted by logit (descending), probabilities, count.
struct CandRow {
  int n;
  int id[kCand];
  float p[kCand];
};

// Mailbox for the exchange of per-card candidates (pinned, mapped host memory; may be null for one card).
struct CandMailbox {
  float* vals;       // [2 slots][2 ranks][4 rows][kCand]
  int* ids;          // same shape
  int* flags;        // [2 slots][2 ranks][32]
  int* err;
};

// Top-k of `rows` rows of local logits [rows][n] (global ids = vocab0 + local index); exchange with the
// other card (when mb != null) and merge; then apply the chain (draft=false: target, true: draft) and
// write the candidate rows. The exchange token is (*dstep) * n_ex + index + 1; slot = index & 1.
// idmap (optional, device [n]): token id of each logit column (a vocabulary subset) instead of vocab0 + column.
void sample_candidates(const float* logits, int rows, int n, int vocab0, int k, bool draft, const SampleParams& sp,
                       CandRow* out, const CandMailbox* mb, int rank, const int* dstep, int n_ex, int index,
                       cudaStream_t s, const int* idmap = nullptr);

// Draw one token per row from the candidate rows. RNG key = (seed, *counter, row, salt).
void sample_tokens(const CandRow* rows, int nrows, int* out_tokens, uint64_t seed, const int* counter, int salt,
                   cudaStream_t s);

}  // namespace q27
