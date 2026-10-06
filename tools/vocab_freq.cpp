// Token counts of text files, for choosing the draft vocabulary.
// Usage: vocab_freq <model.gguf> <out.bin> <file> [file ...]
// out.bin: uint32 count[n_vocab] (tokens of all files, parse_special = true). A file ending in .json is read as a
// server log (bench\out\reqlog\res-*.json): only its "raw" field (the generated text) is counted.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "nlohmann/json.hpp"
#include "tokenizer.h"

int main(int argc, char** argv) try {
  if (argc < 4) { fprintf(stderr, "usage: vocab_freq <model.gguf> <out.bin> <file> [file ...]\n"); return 1; }
  q27::GGUF g(argv[1]);
  q27::Tokenizer tok(g);
  std::vector<uint32_t> cnt(tok.n_vocab(), 0);
  size_t total = 0;
  for (int i = 3; i < argc; i++) {
    std::ifstream f(argv[i], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    const std::string name = argv[i];
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0) {
      auto j = nlohmann::json::parse(text, nullptr, false);
      text = j.is_object() && j.contains("raw") && j["raw"].is_string() ? j["raw"].get<std::string>() : "";
    }
    try {
      for (int id : tok.encode(text, true)) { cnt[id]++; total++; }
    } catch (const std::exception&) {
      fprintf(stderr, "skipped %s (invalid UTF-8)\n", argv[i]);
    }
  }
  std::ofstream o(argv[2], std::ios::binary);
  o.write((const char*)cnt.data(), cnt.size() * sizeof(uint32_t));
  fprintf(stderr, "%zu tokens from %d files\n", total, argc - 3);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
