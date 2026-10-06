// Readers for the two logit reference formats used by the tools:
//  "_logits_" : llama-perplexity --kl-divergence-base (uint32 n_ctx, int32 n_vocab, n_chunk, tokens, rows)
//  "q27ref01" : bench/llama_ref (int32 n_tok, n_vocab, first, n_scored, ubatch, tokens, rows)
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct RefFile {
  bool q27 = false;        // q27ref01 format
  int n_ctx = 0;           // _logits_: chunk length; q27ref: number of tokens
  int n_vocab = 0;
  int n_chunk = 1;
  int first = 0, n_scored = 0, ubatch = 0;  // q27ref only
  std::vector<int32_t> tokens;
  std::streamoff rows_off = 0;               // file offset of the first log-prob row
};

inline RefFile read_ref(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  in.read(magic, 8);
  if (!in) throw std::runtime_error("cannot read " + path);
  RefFile r;
  if (memcmp(magic, "_logits_", 8) == 0) {
    uint32_t n_ctx; int32_t n_vocab, n_chunk;
    in.read((char*)&n_ctx, 4); in.read((char*)&n_vocab, 4); in.read((char*)&n_chunk, 4);
    r.n_ctx = (int)n_ctx; r.n_vocab = n_vocab; r.n_chunk = n_chunk;
    r.tokens.resize((size_t)n_ctx * n_chunk);
  } else if (memcmp(magic, "q27ref01", 8) == 0) {
    int32_t h[5];
    in.read((char*)h, sizeof(h));
    r.q27 = true; r.n_ctx = h[0]; r.n_vocab = h[1]; r.first = h[2]; r.n_scored = h[3]; r.ubatch = h[4];
    r.tokens.resize(r.n_ctx);
  } else {
    throw std::runtime_error(path + ": unknown format");
  }
  in.read((char*)r.tokens.data(), r.tokens.size() * 4);
  if (!in) throw std::runtime_error(path + ": truncated");
  r.rows_off = in.tellg();
  return r;
}
