// GGUF file reader: header, metadata, tensor table, memory-mapped data.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace q27 {

// ggml type ids used by this model (ggml.h enum ggml_type).
enum class GType : uint32_t {
  F32 = 0, F16 = 1, Q2_K = 10, Q4_K = 12, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
  IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, BF16 = 30,
};
const char* gtype_name(GType t);
// Bytes per block and weights per block. Throws for types this engine does not know.
int gtype_block_bytes(GType t);
int gtype_block_size(GType t);

struct GTensor {
  std::string name;
  GType type;
  int n_dims = 0;
  int64_t ne[4] = {1, 1, 1, 1};  // ne[0] = row length (K), ne[1] = rows (N)
  uint64_t offset = 0;           // from the start of the data section
  uint64_t nbytes = 0;
  const uint8_t* data = nullptr; // points into the mapped file
  int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
  int64_t row_bytes() const { return ne[0] / gtype_block_size(type) * gtype_block_bytes(type); }
};

struct GValue {
  enum Kind { INT, FLOAT, BOOL, STRING, ARR_INT, ARR_FLOAT, ARR_STRING } kind = INT;
  int64_t i = 0;
  double f = 0;
  std::string s;
  std::vector<int64_t> ai;
  std::vector<double> af;
  std::vector<std::string> as;
};

class GGUF {
 public:
  explicit GGUF(const std::string& path);
  ~GGUF();
  GGUF(const GGUF&) = delete;
  GGUF& operator=(const GGUF&) = delete;

  const GTensor& tensor(const std::string& name) const;  // throws if missing
  const GTensor* find(const std::string& name) const;     // nullptr if missing
  const std::vector<GTensor>& tensors() const { return tensors_; }

  bool has(const std::string& key) const { return kv_.count(key) != 0; }
  const GValue& kv(const std::string& key) const;  // throws if missing
  int64_t get_int(const std::string& key) const;
  double get_float(const std::string& key) const;
  const std::string& get_str(const std::string& key) const;

 private:
  std::string path_;
  const uint8_t* base_ = nullptr;
  uint64_t size_ = 0;
  void* hfile_ = nullptr;
  void* hmap_ = nullptr;
  int fd_ = -1;
  std::map<std::string, GValue> kv_;
  std::vector<GTensor> tensors_;
  std::map<std::string, size_t> index_;
};

}  // namespace q27
