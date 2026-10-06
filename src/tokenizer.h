// Tokenizer of the model GGUF: byte-level BPE (tokenizer.ggml.model = gpt2) with the qwen35 pre-tokenizer,
// ported from llama.cpp (src/llama-vocab.cpp, src/unicode.cpp; MIT, see THIRD_PARTY_NOTICES.md).
// Target: the same ids as llama.cpp's llama_tokenize on this GGUF.
#pragma once
#include <string>
#include <vector>

namespace q27 {

class GGUF;

class Tokenizer {
 public:
  explicit Tokenizer(const GGUF& g);  // reads tokenizer.ggml.{tokens,token_type,merges,...} from the GGUF
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // llama.cpp rules: user-defined tokens (<think>, <tool_call>, ...) are always matched in the text; control
  // tokens (<|im_start|>, <|im_end|>, ...) only with parse_special. No BOS is added (add_bos_token = false).
  // Invalid UTF-8 bytes are read as U+FFFD. A 4-byte sequence above U+10FFFF throws std::invalid_argument, as in
  // llama.cpp. Thread-safe.
  std::vector<int> encode(const std::string& text, bool parse_special) const;
  // Bytes of one token (byte-level BPE decoded). Special tokens give their text (as llama.cpp's token_to_piece
  // with special = true).
  std::string piece(int id) const;
  std::string decode(const std::vector<int>& ids) const;

  int n_vocab() const;
  int eos() const;                 // tokenizer.ggml.eos_token_id (248046, <|im_end|>)
  bool is_eog(int id) const;       // end-of-generation set as llama.cpp: <|im_end|>, <|endoftext|>, ...
  bool is_special(int id) const;   // control or user-defined token
  int find(const std::string& token_text) const;  // id of an exact token string, or -1

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace q27
