// Tokenizer regression test: q27::Tokenizer against llama.cpp's own tokenizer, through a golden file written by
// bench/llama_tok.cpp (production llama.dll).
//
// Usage: test_tokenizer <model.gguf> [golden = bench/out/tok_golden.bin]
//
// Checks:
// - encode(s, parse_special) for parse_special false and true gives llama_tokenize's ids (add_special = false), or
//   throws where llama_tokenize throws. Strings: the bench corpus (17.9 MB of code) whole and in 64 KB and 4 KB
//   pieces, long-text whole and in 4 KB pieces, hand-written edge cases, every special token inside text,
//   10,500 random strings.
// - decode() of llama.cpp's ids gives llama_detokenize's text (small strings).
// - decode(encode(s)) == s for every string that is valid UTF-8.
// - piece(), is_eog() and is_special() for every token id; eos(), n_vocab().
// - encode speed on the corpus and on a 180k-token prompt.
// Prints PASS/FAIL counts. Exit code 1 on any mismatch.
//
// Regenerate the golden (needs the production llama.dll in qwen38_27\bin-parches; vocab only, no GPU work):
//   bench\build-llama-tok.bat
//   (Git Bash) export PATH="$Q27_LLAMA_BIN:$PATH"   (the llama.cpp DLL folder)
//   bench/build/llama_tok.exe <model.gguf> bench/out/tok-corpus.txt bench/out/long-text.txt bench/out/tok_golden.bin
// tok-corpus.txt is a copy of the 17.9 MB code corpus of the llama.cpp benchmarks (qwen38_27 session scratchpad,
// corpus.txt). The golden keeps a copy of every string, so the test needs only the model and the golden.
// Build: tools\build_target.bat build-tok test_tokenizer
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"

using namespace q27;

namespace {

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Reader {
  const char* p;
  const char* end;
  template <class T> T get() {
    if (p + sizeof(T) > end) throw std::runtime_error("golden: truncated");
    T v;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string_view bytes(size_t n) {
    if (p + n > end) throw std::runtime_error("golden: truncated");
    std::string_view s(p, n);
    p += n;
    return s;
  }
  std::vector<int> ids(int32_t n) {
    std::vector<int> v((size_t)n);
    if (n > 0) memcpy(v.data(), bytes((size_t)n * 4).data(), (size_t)n * 4);
    return v;
  }
};

struct Case {
  std::string name;
  std::string_view text;
  bool threw[2];
  std::vector<int> ids[2];
  bool has_detok;
  std::string_view detok;
};

// Strict UTF-8 (no overlong forms, no surrogates, nothing above U+10FFFF).
bool valid_utf8(std::string_view s) {
  const uint8_t* p = (const uint8_t*)s.data();
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    const uint8_t c = p[i];
    int len;
    uint32_t cp;
    if (c < 0x80) { i++; continue; }
    if (c >= 0xC2 && c <= 0xDF) { len = 2; cp = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; cp = c & 0x0F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; cp = c & 0x07; }
    else return false;
    if (i + len > n) return false;
    for (int k = 1; k < len; k++) {
      if ((p[i + k] & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (p[i + k] & 0x3F);
    }
    if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF))
      return false;
    i += len;
  }
  return true;
}

std::string show(std::string_view s, size_t max = 60) {
  std::string o;
  for (size_t i = 0; i < s.size() && i < max; i++) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20 || c == 0x7F) { char b[8]; snprintf(b, sizeof(b), "\\x%02X", c); o += b; }
    else o += (char)c;
  }
  if (s.size() > max) o += "...";
  return o;
}

std::string show_ids(const std::vector<int>& v, size_t at) {
  std::string o;
  const size_t a = at > 3 ? at - 3 : 0, b = std::min(v.size(), at + 4);
  for (size_t i = a; i < b; i++) o += (i == at ? "[" : "") + std::to_string(v[i]) + (i == at ? "] " : " ");
  return o + "(n=" + std::to_string(v.size()) + ")";
}

std::string group_of(const std::string& name) {
  const size_t h = name.find('#');
  if (h != std::string::npos) return name.substr(0, h);
  const size_t sl = name.find('/');
  if (sl != std::string::npos) return name.substr(0, sl);
  return name;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_tokenizer <model.gguf> [golden = bench/out/tok_golden.bin]\n");
    return 1;
  }
  const char* golden_path = argc > 2 ? argv[2] : "bench/out/tok_golden.bin";
  int pass = 0, fail = 0;
  int shown = 0;
  auto check = [&](bool ok, const std::string& what) {
    if (ok) { pass++; return; }
    fail++;
    if (shown++ < 20) printf("FAIL %s\n", what.c_str());
  };

  double t0 = now_s();
  GGUF gguf(argv[1]);
  Tokenizer tok(gguf);
  printf("tokenizer load: %.3f s (%d tokens)\n", now_s() - t0, tok.n_vocab());

  std::string gold;
  {
    std::ifstream f(golden_path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s (see the header of tools/test_tokenizer.cpp)\n", golden_path); return 1; }
    std::stringstream ss;
    ss << f.rdbuf();
    gold = ss.str();
  }
  Reader r{gold.data(), gold.data() + gold.size()};
  if (r.bytes(8) != "q27tok01") { fprintf(stderr, "bad golden magic\n"); return 1; }

  // ---- vocab
  const int n_vocab = r.get<int32_t>();
  const int eos = r.get<int32_t>();
  check(n_vocab == tok.n_vocab(), "n_vocab " + std::to_string(tok.n_vocab()) + " != " + std::to_string(n_vocab));
  check(eos == tok.eos(), "eos " + std::to_string(tok.eos()) + " != " + std::to_string(eos));
  int bad_piece = 0, bad_eog = 0, bad_special = 0, n_eog = 0, n_special = 0;
  for (int id = 0; id < n_vocab; id++) {
    const int32_t attr = r.get<int32_t>();
    const bool eog = r.get<uint8_t>() != 0;
    const std::string_view piece = r.bytes(r.get<uint32_t>());
    const bool special = (attr & (8 | 16)) != 0;  // LLAMA_TOKEN_ATTR_CONTROL | USER_DEFINED
    n_eog += eog;
    n_special += special;
    if (id >= tok.n_vocab()) { bad_piece++; continue; }
    if (tok.piece(id) != piece) { if (bad_piece++ < 5) printf("piece %d: '%s' != '%s'\n", id, show(tok.piece(id)).c_str(), show(piece).c_str()); }
    if (tok.is_eog(id) != eog) { if (bad_eog++ < 5) printf("is_eog %d: %d != %d\n", id, tok.is_eog(id), eog); }
    if (tok.is_special(id) != special) { if (bad_special++ < 5) printf("is_special %d: %d != %d\n", id, tok.is_special(id), special); }
  }
  check(bad_piece == 0, std::to_string(bad_piece) + " pieces differ");
  check(bad_eog == 0, std::to_string(bad_eog) + " is_eog differ");
  check(bad_special == 0, std::to_string(bad_special) + " is_special differ");
  printf("vocab: %d pieces, %d EOG, %d special compared: %s\n", n_vocab, n_eog, n_special,
         bad_piece + bad_eog + bad_special ? "FAIL" : "PASS");

  // ---- golden strings
  std::vector<std::string_view> texts(r.get<int32_t>());
  for (auto& t : texts) t = r.bytes(r.get<uint64_t>());
  std::vector<Case> cases(r.get<int32_t>());
  for (Case& c : cases) {
    c.name = std::string(r.bytes(r.get<uint32_t>()));
    const int ti = r.get<int32_t>();
    const uint64_t off = r.get<uint64_t>(), len = r.get<uint64_t>();
    c.text = texts.at(ti).substr(off, len);
    for (int m = 0; m < 2; m++) {
      const int32_t n = r.get<int32_t>();
      c.threw[m] = n == -1;
      if (n == -2) c.ids[m] = c.ids[0];
      else if (n >= 0) c.ids[m] = r.ids(n);
    }
    const int32_t nd = r.get<int32_t>();
    c.has_detok = nd >= 0;
    if (nd >= 0) c.detok = r.bytes((size_t)nd);
  }

  struct Stat {
    int cases = 0, threw = 0, fail = 0;
    size_t ids = 0;
  };
  std::map<std::string, Stat> stats;
  size_t n_ids = 0;
  int n_rt = 0, n_detok = 0;
  double t_corpus = 0;
  size_t n_corpus_tok = 0, corpus_bytes = 0;
  for (const Case& c : cases) {
    Stat& st = stats[group_of(c.name)];
    st.cases++;
    std::vector<int> got[2];
    for (int m = 0; m < 2; m++) {
      bool threw = false;
      const double ts = now_s();
      try {
        got[m] = tok.encode(std::string(c.text), m == 1);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (c.name == "corpus" && m == 0) { t_corpus = now_s() - ts; n_corpus_tok = got[m].size(); corpus_bytes = c.text.size(); }
      const std::string what = c.name + " parse_special=" + std::to_string(m) + " '" + show(c.text) + "'";
      if (c.threw[m] || threw) {
        st.threw += c.threw[m];
        if (threw != c.threw[m]) st.fail++;
        check(threw == c.threw[m], what + (threw ? ": threw, llama.cpp did not" : ": llama.cpp threw, encode did not"));
        continue;
      }
      st.ids += c.ids[m].size();
      n_ids += c.ids[m].size();
      size_t at = 0;
      while (at < got[m].size() && at < c.ids[m].size() && got[m][at] == c.ids[m][at]) at++;
      const bool ok = got[m] == c.ids[m];
      if (!ok) st.fail++;
      check(ok, what + ": first difference at " + std::to_string(at) + ": got " + show_ids(got[m], at) + ", llama.cpp " +
                    show_ids(c.ids[m], at));
    }
    if (c.has_detok) {
      n_detok++;
      const std::string d = tok.decode(c.ids[1]);
      check(d == c.detok, c.name + ": decode '" + show(d) + "' != llama_detokenize '" + show(c.detok) + "'");
    }
    if (valid_utf8(c.text)) {
      for (int m = 0; m < 2; m++) {
        if (c.threw[m]) continue;
        n_rt++;
        check(tok.decode(got[m]) == c.text, c.name + ": decode(encode(s)) != s, parse_special=" + std::to_string(m));
      }
    }
  }
  printf("%-14s %7s %11s %7s %6s\n", "group", "cases", "ids", "threw", "FAIL");
  for (const auto& kv : stats)
    printf("%-14s %7d %11zu %7d %6d\n", kv.first.c_str(), kv.second.cases, kv.second.ids, kv.second.threw, kv.second.fail);
  printf("encode: %zu cases x 2 (parse_special false/true), %zu ids compared\n", cases.size(), n_ids);
  printf("decode vs llama_detokenize: %d strings; decode(encode(s)) == s: %d (valid UTF-8 strings x modes)\n", n_detok, n_rt);

  // ---- speed
  if (t_corpus > 0) {
    printf("speed: corpus %.1f MB -> %zu tokens in %.3f s = %.2f M tokens/s\n", corpus_bytes / 1e6, n_corpus_tok, t_corpus,
           n_corpus_tok / t_corpus / 1e6);
    // a ~180k-token prompt: corpus prefix made of whole 64 KB pieces
    size_t bytes = 0, ntok = 0;
    for (const Case& c : cases) {
      if (c.name.rfind("corpus/64k#", 0) != 0 || ntok >= 180000) continue;
      bytes += c.text.size();
      ntok += c.ids[0].size();
    }
    const std::string prompt(cases[0].text.substr(0, bytes));
    double best = 1e9;
    size_t n = 0;
    for (int rep = 0; rep < 5; rep++) {
      const double ts = now_s();
      n = tok.encode(prompt, true).size();
      best = std::min(best, now_s() - ts);
    }
    printf("speed: %zu-token prompt (%.0f KB) in %.1f ms (best of 5)\n", n, bytes / 1024.0, best * 1e3);
  }

  printf("PASS: %d  FAIL: %d\n", pass, fail);
  return fail ? 1 : 0;
}
