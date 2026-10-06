// Tokenizer oracle: llama.cpp's own tokenizer (production llama.dll in qwen38_27\bin-parches, used read-only)
// run on a fixed set of test strings. Writes the golden file that tools/test_tokenizer.cpp checks against.
//
// Usage: llama_tok <model.gguf> <corpus.txt> <long-text.txt> <out.bin>
// Build: bench\build-llama-tok.bat. Run with bin-parches on PATH. Only the vocab is loaded (no GPU work).
//
// Test strings: the corpus whole, in 64 KB and in 4 KB pieces; long-text whole and in 4 KB pieces; hand-written
// edge cases; every special token inside text; 10,000+ random strings (code points, raw bytes, mutated corpus
// slices). Each string is tokenized with add_special = false (as llama-server /tokenize and the chat path) and
// parse_special = false and true. llama_tokenize throws on some invalid UTF-8 (a 4-byte sequence above
// U+10FFFF); that is recorded as "threw".
//
// Output "q27tok01" (little endian):
//   int32 n_vocab, eos
//   per token id: int32 attr, uint8 is_eog, uint32 n, n bytes of llama_token_to_piece(special = true)
//   int32 n_texts; per text: uint64 n, n bytes
//   int32 n_cases; per case: uint32 n, name; int32 text index; uint64 offset, length;
//     two results (parse_special false, true): int32 n then n int32 ids; n = -1: threw; n = -2 (second only):
//     same ids as the first;
//     int32 n then n bytes: llama_detokenize(ids of parse_special true, remove_special false, unparse_special
//     true), n = -1: not stored (big cases)
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Out {
  std::ofstream f;
  void i32(int32_t v) { f.write((const char*)&v, 4); }
  void u32(uint32_t v) { f.write((const char*)&v, 4); }
  void u64(uint64_t v) { f.write((const char*)&v, 8); }
  void u8(uint8_t v) { f.write((const char*)&v, 1); }
  void bytes(const std::string& s) { f.write(s.data(), (std::streamsize)s.size()); }
};

struct Case {
  std::string name;
  int text;
  uint64_t off, len;
};

uint64_t rng_state = 0x243F6A8885A308D3ull;
uint64_t rnd() {  // splitmix64
  uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
uint32_t rnd(uint32_t n) { return (uint32_t)(rnd() % n); }

void put_utf8(std::string& s, uint32_t c) {
  if (c < 0x80) { s += (char)c; }
  else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
  else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

// One random code point from a mix of the classes the pre-tokenizer cares about.
uint32_t rnd_cpt() {
  static const uint32_t ws[] = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2005, 0x200A,
                                0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0x200B, 0xFEFF};
  static const char ascii_hot[] = "'sStTmMdDrReEvVlL \n\r\t0123456789.,;:!?()[]{}<>/\\|_-+=*&^%$#@~`\"";
  switch (rnd(16)) {
    case 0: case 1: case 2: return 0x20 + rnd(0x5F);                       // printable ASCII
    case 3: return (uint32_t)(unsigned char)ascii_hot[rnd(sizeof(ascii_hot) - 1)];
    case 4: return ws[rnd(sizeof(ws) / sizeof(ws[0]))];
    case 5: return rnd(0x20);                                              // C0 controls, NUL included
    case 6: return 0x80 + rnd(0x250 - 0x80);                               // Latin-1, Latin Extended
    case 7: return 0x300 + rnd(0x70);                                      // combining diacritics
    case 8: return 0x900 + rnd(0x80);                                      // Devanagari (marks, digits)
    case 9: return 0x600 + rnd(0x100);                                     // Arabic
    case 10: return 0x4E00 + rnd(0x5200);                                  // CJK
    case 11: return 0x3040 + rnd(0x100);                                   // kana
    case 12: return 0x1F300 + rnd(0x300);                                  // emoji
    case 13: { static const uint32_t z[] = {0x200D, 0xFE0F, 0x20E3, 0x1F3FB, 0x1F1EA, 0x1F1F8}; return z[rnd(6)]; }
    case 14: { static const uint32_t d[] = {0x660, 0x6F0, 0x966, 0xFF10, 0x2160, 0xB2, 0xBD, 0x1D7CE}; return d[rnd(8)] + rnd(10); }
    default: { uint32_t c; do { c = rnd(0x110000); } while (c >= 0xD800 && c < 0xE000); return c; }
  }
}

std::vector<std::string> edge_cases() {
  std::vector<std::string> v = {
    "", "Hello world", "Hello, world!", " Hello", "  Hello", "Hello ", "Hello  ", "hello\nworld", "a",
    // contractions
    "I'm", "you're", "they'll", "it's", "IT'S", "We'RE", "she'D", "I'Ve", "don't", "DON'T", "'s", "'S", "'ll",
    "'LL", "'lL", "'re", "'ve", "'m", "'d", "'t", "'", "''", "'''", "a'b", "''s", " 's", "x'S y'Ll z'rE",
    "'s'", "'sa", "'ll'll", "'\xC3\xA9", "'\xCC\x81s", "rock'n'roll", "'1", "' s", "'\n", "\xE2\x80\x99s",
    // CJK, Arabic, Hebrew, Devanagari, Thai, other scripts
    "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81",
    "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE3\x83\x86\xE3\x82\xAD\xE3\x82\xB9\xE3\x83\x88",
    "\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4 \xED\x85\x8D\xEC\x8A\xA4\xED\x8A\xB8",
    "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7 \xD8\xA8\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85",
    "\xD8\xA7\xD9\x84\xD8\xB9\xD9\x8E\xD8\xB1\xD9\x8E\xD8\xA8\xD9\x90\xD9\x8A\xD9\x8E\xD9\x91\xD8\xA9",
    "\xD7\xA9\xD6\xB8\xD7\x81\xD7\x9C\xD7\x95\xD6\xB9\xD7\x9D",
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87 \xE0\xA4\xA6\xE0\xA5\x81\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE",
    "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xAF",
    "\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5\xE0\xB8\x84\xE0\xB8\xA3\xE0\xB8\xB1\xE0\xB8\x9A",
    "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 \xD0\xBC\xD0\xB8\xD1\x80", "\xCE\x95\xCE\xBB\xCE\xBB\xCE\xB7\xCE\xBD\xCE\xB9\xCE\xBA\xCE\xAC",
    "Z\xC3\xBCrich na\xC3\xAFve caf\xC3\xA9", "\xEF\xAC\x81 \xC7\x85 \xC3\x9F \xC4\xB0stanbul \xE2\x84\xAA",
    // combining marks
    "e\xCC\x81", "a\xCC\x80\xCC\x81\xCC\x82", "\xCC\x81" "abc", " \xCC\x81", "\xCC\x81\xCC\x82", "1\xCC\x81",
    "\n\xCC\x81", ".\xCC\x81x", "x \xCC\x81 y", "\xE0\xA4\xBE\xE0\xA4\xBF", " \xE0\xA4\xBE" "a",
    // emoji
    "\xF0\x9F\x98\x80", "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6",
    "\xF0\x9F\x8F\xB3\xEF\xB8\x8F\xE2\x80\x8D\xF0\x9F\x8C\x88", "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",
    "\xF0\x9F\x87\xAA\xF0\x9F\x87\xB8\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", "\xC2\xA9\xEF\xB8\x8F", "1\xEF\xB8\x8F\xE2\x83\xA3",
    "Hi \xF0\x9F\x98\x80!", " \xF0\x9F\x98\x80\xF0\x9F\x98\x80 ",
    // whitespace
    " ", "  ", "   ", "\t", "\t\t", "\n", "\n\n\n", " \n ", "a  b", "a   \n  b", "\r\n", "\r\n\r\n",
    "line1\r\nline2", "\r", " \r\n", "\r\r\n\n", "\xC2\xA0", "\xE3\x80\x80", "\xE2\x80\xA8", "\xC2\x85", "\v\f",
    "a\xC2\xA0\xC2\xA0" "b", "abc   ", "   abc", "\t \t x", "x \t\n\t y", "  \n\n  \n", "a\n\n\nb",
    "    def f():\n        return 1\n", "}\n\n\n", "};\r\n", " .\n", "...\n\n", "\xE2\x80\x8B", "\xEF\xBB\xBF" "abc",
    // digits
    "1234567890", "3.14159", "1,000,000", "\xD9\xA3\xD9\xA4\xD9\xA5", "\xC2\xB2\xC2\xB3", "\xE2\x85\xAB",
    "1st 2nd 3rd", "0x1F 0b1010", " 123", "a1b2c3", "\xEF\xBC\x91\xEF\xBC\x92",
    // invalid UTF-8 and odd bytes
    "\xFF", "\xFE\xFF", "\xC0\xAF", "\xE0\x80\xAF", "\xC3", "abc\xE2\x82", "\x80", "\xED\xA0\x80",
    "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xF7\xBF\xBF\xBF", "\xF8\x88\x80\x80\x80", "\xC3\x28", "\xE2\x28\xA1",
    "ok\xFFok", "\xC0\x80", "\xF0\x80\x80\x80", "\xE0\x9F\xBF", "\xC1\xBF", "\xF0\x9F\x98", "\xF0\x9F",
    "a\xCC", "\xEF\xBF\xBD", "x\xED\xBF\xBFy", "\xF4\x8F\xBF\xBF", " \x80 \x81 ", "\xC2\xC2\xA0",
    std::string("a\0b", 3), std::string("\0", 1), std::string(" \0 ", 3),
    // code
    "def foo(x, y):\n    return x + y  # sum\n",
    "#include <stdio.h>\nint main(void) {\n\tprintf(\"%d\\n\", 42);\n\treturn 0;\n}\n",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Barcelona\", \"days\": 3}}",
    "<html><body><p class=\"x\">Hi &amp; bye</p></body></html>", "x = a->b; y = c >> 2; z <<= 1; w != v;",
    "SELECT * FROM t WHERE id = 1;", "    // comment\r\n", "if (a && b || !c) { return; }",
    "https://example.com/path?q=1&r=2#frag", "C:\\Users\\name\\file.txt", "$HOME/.config", "@decorator",
    "__init__", "camelCaseWord snake_case_word SCREAMING_CASE", "a.b.c", "---", "===", "***", "~~~",
    "\xE2\x80\x94 \xE2\x80\x9Cquoted\xE2\x80\x9D \xE2\x80\xA6", "\xC2\xAB\xC2\xBB", "\xE2\x86\x92 \xE2\x89\xA4 \xE2\x88\x9E",
    // special-token look-alikes
    "<|im_start", "<|im_start|", "<<|im_end|>>", "<|IM_START|>", "< think>", "<think >", "</think", "<think><think>",
    "[PAD248077]", "<|im_start|><|im_start|>", "<|endoftext|><|endoftext|>",
    "Hello<|im_start|>user\nHi<|im_end|>\n", "<think>\nreasoning\n</think>\n\nanswer",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
    "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>", "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>",
    "\xC3\xA9<think>\xC3\xA9", " <think> ", "\n<|im_end|>\n", "x<|im_end|>y</think>z<tool_response>w",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  };
  // long runs
  v.push_back(std::string(10000, 'a'));
  v.push_back(std::string(5000, ' '));
  v.push_back(std::string(1000, '\n'));
  v.push_back(std::string(3000, '-'));
  v.push_back(std::string(1000, '='));
  v.push_back(std::string(2000, '\t'));
  v.push_back(std::string(4000, '9'));
  v.push_back(std::string(4000, '\xFF'));
  { std::string s; for (int i = 0; i < 3000; i++) s += "ab"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 500; i++) s += "\xF0\x9F\x98\x80"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 2000; i++) s += "\xE7\x9A\x84"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 1000; i++) s += " \n"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 1000; i++) s += "\r\n"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 300; i++) s += "e\xCC\x81"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 200; i++) s += (char)('0' + i % 10); v.push_back(s); }
  { std::string s; for (int i = 0; i < 500; i++) s += "'s"; v.push_back(s); }
  { std::string s; for (int i = 0; i < 300; i++) s += "<think>"; v.push_back(s); }
  return v;
}

std::string read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

void quiet_log(ggml_log_level, const char*, void*) {}

// llama_tokenize with add_special = false. Returns false if it threw.
bool tokenize(const llama_vocab* v, const std::string& s, size_t off, size_t len, bool parse_special,
              std::vector<llama_token>& out) {
  out.resize(len + 16);
  try {
    int n = llama_tokenize(v, s.data() + off, (int32_t)len, out.data(), (int32_t)out.size(), false, parse_special);
    if (n < 0) {
      out.resize((size_t)-n);
      n = llama_tokenize(v, s.data() + off, (int32_t)len, out.data(), (int32_t)out.size(), false, parse_special);
    }
    out.resize((size_t)n);
    return true;
  } catch (...) {
    out.clear();
    return false;
  }
}

std::string detokenize(const llama_vocab* v, const std::vector<llama_token>& ids) {
  std::string s(ids.size() * 8 + 64, '\0');
  int n = llama_detokenize(v, ids.data(), (int32_t)ids.size(), &s[0], (int32_t)s.size(), false, true);
  if (n < 0) {
    s.resize((size_t)-n);
    n = llama_detokenize(v, ids.data(), (int32_t)ids.size(), &s[0], (int32_t)s.size(), false, true);
  }
  s.resize((size_t)n);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: llama_tok <model.gguf> <corpus.txt> <long-text.txt> <out.bin>\n");
    return 1;
  }
  llama_log_set(quiet_log, nullptr);
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  std::vector<std::string> texts;
  std::vector<Case> cases;
  texts.push_back(read_file(argv[2]));
  texts.push_back(read_file(argv[3]));
  auto chunks = [&](int t, const char* name, uint64_t size) {
    const uint64_t n = texts[t].size();
    for (uint64_t off = 0, k = 0; off < n; off += size, k++)
      cases.push_back({std::string(name) + "#" + std::to_string(k), t, off, std::min(size, n - off)});
  };
  cases.push_back({"corpus", 0, 0, texts[0].size()});
  chunks(0, "corpus/64k", 65536);
  chunks(0, "corpus/4k", 4096);
  cases.push_back({"long-text", 1, 0, texts[1].size()});
  chunks(1, "long-text/4k", 4096);
  auto add_text = [&](const std::string& name, const std::string& s) {
    texts.push_back(s);
    cases.push_back({name, (int)texts.size() - 1, 0, s.size()});
  };
  {
    const auto e = edge_cases();
    for (size_t i = 0; i < e.size(); i++) add_text("edge#" + std::to_string(i), e[i]);
  }
  // every special token alone and inside text
  std::vector<std::string> specials;
  for (int id = 0; id < n_vocab; id++) {
    const int a = llama_vocab_get_attr(vocab, id);
    if (a & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
      const std::string t = llama_vocab_get_text(vocab, id);
      specials.push_back(t);
      add_text("special/" + t, t);
      add_text("special-in/" + t, "x" + t + "y");
      add_text("special-sp/" + t, " " + t + " ");
      add_text("special-nl/" + t, "Hello\n" + t + "\nworld " + t + t + "\xC3\xA9");
    }
  }
  // fuzz: random code points, random bytes, mutated corpus slices
  for (int i = 0; i < 5000; i++) {
    std::string s;
    const uint32_t n = 1 + rnd(i < 4000 ? 24 : 200);
    for (uint32_t k = 0; k < n; k++) {
      if (rnd(40) == 0) s += specials[rnd((uint32_t)specials.size())];
      else put_utf8(s, rnd_cpt());
    }
    add_text("fuzz-cpt#" + std::to_string(i), s);
  }
  for (int i = 0; i < 3000; i++) {
    std::string s;
    const uint32_t n = 1 + rnd(32);
    const int kind = i % 3;  // 0: any byte, 1: mostly ASCII, 2: mostly UTF-8 lead/continuation bytes
    for (uint32_t k = 0; k < n; k++) {
      if (kind == 0) s += (char)rnd(256);
      else if (kind == 1) s += rnd(8) ? (char)(0x20 + rnd(0x60)) : (char)rnd(256);
      else s += (char)(0x80 + rnd(0x80));
    }
    add_text("fuzz-byte#" + std::to_string(i), s);
  }
  for (int i = 0; i < 2500; i++) {
    const std::string& c = texts[i % 2];
    const uint64_t len = 1 + rnd(300);
    std::string s = c.substr(rnd((uint32_t)(c.size() - len)), len);
    const uint32_t edits = rnd(4);
    for (uint32_t k = 0; k < edits && !s.empty(); k++) {
      const uint32_t p = rnd((uint32_t)s.size());
      switch (rnd(4)) {
        case 0: s[p] = (char)rnd(256); break;
        case 1: s.insert(p, specials[rnd((uint32_t)specials.size())]); break;
        case 2: { std::string u; put_utf8(u, rnd_cpt()); s.insert(p, u); } break;
        default: s.erase(p, 1 + rnd(4)); break;
      }
    }
    add_text("fuzz-mut#" + std::to_string(i), s);
  }

  Out o;
  o.f.open(argv[4], std::ios::binary);
  if (!o.f) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
  o.f.write("q27tok01", 8);
  o.i32(n_vocab);
  o.i32(llama_vocab_eos(vocab));
  std::vector<char> buf(1 << 16);
  int n_eog = 0;
  for (int id = 0; id < n_vocab; id++) {
    const int n = llama_token_to_piece(vocab, id, buf.data(), (int32_t)buf.size(), 0, true);
    if (n < 0) { fprintf(stderr, "piece too long: %d\n", id); return 1; }
    o.i32((int32_t)llama_vocab_get_attr(vocab, id));
    const bool eog = llama_vocab_is_eog(vocab, id);
    n_eog += eog;
    o.u8(eog ? 1 : 0);
    o.u32((uint32_t)n);
    o.f.write(buf.data(), n);
  }
  o.i32((int32_t)texts.size());
  for (const auto& t : texts) { o.u64(t.size()); o.bytes(t); }

  o.i32((int32_t)cases.size());
  std::vector<llama_token> ids0, ids1;
  size_t n_tok = 0, n_threw = 0;
  double t_corpus = 0;
  for (const Case& c : cases) {
    const std::string& s = texts[c.text];
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok0 = tokenize(vocab, s, c.off, c.len, false, ids0);
    if (c.name == "corpus") t_corpus = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok1 = tokenize(vocab, s, c.off, c.len, true, ids1);
    o.u32((uint32_t)c.name.size()); o.bytes(c.name);
    o.i32(c.text); o.u64(c.off); o.u64(c.len);
    if (!ok0) { o.i32(-1); n_threw++; }
    else { o.i32((int32_t)ids0.size()); o.f.write((const char*)ids0.data(), ids0.size() * 4); n_tok += ids0.size(); }
    if (!ok1) { o.i32(-1); n_threw++; }
    else if (ok0 && ids1 == ids0) { o.i32(-2); n_tok += ids1.size(); }
    else { o.i32((int32_t)ids1.size()); o.f.write((const char*)ids1.data(), ids1.size() * 4); n_tok += ids1.size(); }
    if (ok1 && c.len <= 65536 && c.text >= 2) {
      const std::string d = detokenize(vocab, ids1);
      o.i32((int32_t)d.size()); o.bytes(d);
    } else {
      o.i32(-1);
    }
  }
  o.f.close();
  printf("wrote %s: %d tokens in vocab (%d EOG), %zu texts, %zu cases, %zu ids, %zu threw\n", argv[4], n_vocab, n_eog,
         texts.size(), cases.size(), n_tok, n_threw);
  printf("llama.cpp tokenize of the whole corpus (%.1f MB): %.3f s\n", texts[0].size() / 1e6, t_corpus);
  llama_model_free(model);
  return 0;
}
