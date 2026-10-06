// Tokenizer: byte-level BPE with the qwen35 pre-tokenizer, ported from llama.cpp (MIT, see THIRD_PARTY_NOTICES.md):
//   src/llama-vocab.cpp: load() (token attributes, EOG set), tokenizer_st_partition, llm_tokenizer_bpe_session,
//                        token_to_piece
//   src/unicode.cpp:     unicode_cpt_from_utf8, unicode_regex_split_custom_qwen35, the GPT-2 byte map
// Same results as llama.cpp. The data structures differ for speed: merges are keyed by token-id pairs in an
// open-addressing table, words are byte ranges of the input (no per-word strings), and each encode() call keeps a
// cache of word -> ids.
#include "tokenizer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "gguf.h"
#include "unicode_tables.h"

namespace q27 {
namespace {

// llama_token_attr bits (llama.h).
enum : uint32_t {
  ATTR_UNKNOWN = 1 << 0, ATTR_UNUSED = 1 << 1, ATTR_NORMAL = 1 << 2, ATTR_CONTROL = 1 << 3,
  ATTR_USER_DEFINED = 1 << 4, ATTR_BYTE = 1 << 5,
};

// tokenizer.ggml.token_type (llama_token_type) -> attribute, as llama.cpp load().
uint32_t attr_of_type(int64_t t) {
  switch (t) {
    case 1: return ATTR_NORMAL;
    case 2: return ATTR_UNKNOWN;
    case 3: return ATTR_CONTROL;
    case 4: return ATTR_USER_DEFINED;
    case 5: return ATTR_UNUSED;
    case 6: return ATTR_BYTE;
    default: return 0;  // undefined
  }
}

// ---- Unicode

// Code point classes used by the pre-tokenizer (\p{L}, \p{M}, \p{N}, \s).
enum : uint8_t { C_LETTER = 1, C_MARK = 2, C_NUMBER = 4, C_SPACE = 8 };

const uint8_t* class_table() {
  static const std::vector<uint8_t> t = [] {
    std::vector<uint8_t> v(0x110000, 0);
    for (int i = 0; i + 1 < kNumUnicodeRanges; i++) {
      const uint16_t f = kUnicodeRanges[i].flags;
      const uint8_t c = (f & UCAT_LETTER ? C_LETTER : 0) | (f & UCAT_MARK ? C_MARK : 0) | (f & UCAT_NUMBER ? C_NUMBER : 0);
      std::fill(v.begin() + kUnicodeRanges[i].first, v.begin() + kUnicodeRanges[i + 1].first, c);
    }
    for (int i = 0; i < kNumUnicodeWhitespace; i++) v[kUnicodeWhitespace[i]] |= C_SPACE;
    return v;
  }();
  return t.data();
}

// GPT-2 byte-level map (unicode.cpp unicode_byte_to_utf8_map): each byte is shown as one printable code point.
struct ByteMap {
  uint32_t to_cpt[256];
  int16_t to_byte[0x144];  // code point -> byte, -1 if the code point is not a byte symbol
  ByteMap() {
    std::fill(std::begin(to_byte), std::end(to_byte), (int16_t)-1);
    int n = 0;
    for (int b = 0; b < 256; b++) {
      const bool direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      to_cpt[b] = direct ? (uint32_t)b : (uint32_t)(256 + n++);
      to_byte[to_cpt[b]] = (int16_t)b;
    }
  }
};
const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

// Bytes of a code point in UTF-8 (unicode_cpt_to_utf8). Throws above U+10FFFF, as llama.cpp.
int utf8_len(uint32_t c) {
  if (c <= 0x7F) return 1;
  if (c <= 0x7FF) return 2;
  if (c <= 0xFFFF) return 3;
  if (c <= 0x10FFFF) return 4;
  throw std::invalid_argument("invalid codepoint");
}

void put_utf8(std::string& s, uint32_t c) {
  switch (utf8_len(c)) {
    case 1: s += (char)c; break;
    case 2: s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); break;
    case 3: s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); break;
    default:
      s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F));
      s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
  }
}

// One code point at s[i] (unicode_cpt_from_utf8 and the catch in unicode_cpts_from_utf8). An invalid byte gives
// U+FFFD and is consumed alone. Overlong forms and surrogates are decoded, not rejected (as llama.cpp).
inline uint32_t next_cpt(const uint8_t* s, size_t n, size_t i, size_t& len) {
  const uint8_t c = s[i];
  auto cont = [&](size_t k) { return (s[i + k] & 0xC0) == 0x80; };
  if (!(c & 0x80)) { len = 1; return c; }
  if (!(c & 0x40)) { len = 1; return 0xFFFD; }
  if (!(c & 0x20)) {
    if (i + 1 >= n || !cont(1)) { len = 1; return 0xFFFD; }
    len = 2;
    return ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
  }
  if (!(c & 0x10)) {
    if (i + 2 >= n || !cont(1) || !cont(2)) { len = 1; return 0xFFFD; }
    len = 3;
    return ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
  }
  if (!(c & 0x08)) {
    if (i + 3 >= n || !cont(1) || !cont(2) || !cont(3)) { len = 1; return 0xFFFD; }
    len = 4;
    return ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu);
  }
  len = 1;
  return 0xFFFD;
}

// ---- BPE merges: (left id, right id) -> (rank, merged id), open addressing.

struct MergeTable {
  struct Slot {
    uint64_t key;
    int32_t rank;
    int32_t id;
  };
  std::vector<Slot> slots;
  int shift = 60;
  static uint64_t key(int a, int b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }
  size_t home(uint64_t k) const { return (size_t)((k * 0x9E3779B97F4A7C15ull) >> shift); }
  void init(size_t n) {
    int bits = 4;
    while (((size_t)1 << bits) < 2 * n) bits++;
    slots.assign((size_t)1 << bits, Slot{~0ull, -1, -1});
    shift = 64 - bits;
  }
  // Keeps the first entry of a pair, as llama.cpp's bpe_ranks.emplace.
  void insert(int a, int b, int rank, int id) {
    const uint64_t k = key(a, b);
    const size_t mask = slots.size() - 1;
    for (size_t i = home(k);; i = (i + 1) & mask) {
      if (slots[i].key == k) return;
      if (slots[i].key == ~0ull) { slots[i] = Slot{k, rank, id}; return; }
    }
  }
  const Slot* find(int a, int b) const {
    const uint64_t k = key(a, b);
    const size_t mask = slots.size() - 1;
    for (size_t i = home(k);; i = (i + 1) & mask) {
      if (slots[i].key == k) return &slots[i];
      if (slots[i].key == ~0ull) return nullptr;
    }
  }
};

// Pending merge of two adjacent symbols (llm_bigram_bpe). The queue pops the lowest rank, then the leftmost.
struct Bigram {
  int rank, left, right, id;
  uint32_t size;  // bytes of left + right when queued; a different sum later means the entry is stale
};
struct BigramOrder {
  bool operator()(const Bigram& l, const Bigram& r) const {
    return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
  }
};

// One code point of a text fragment, as the pre-tokenizer needs it.
struct Cp {
  uint8_t ch;   // the code point if ASCII, else 0x80
  uint8_t cls;  // C_* classes in the low 4 bits, UTF-8 length in the high 4 bits
};

struct Vocab {
  std::string text_buf;            // all token texts
  std::vector<uint32_t> text_off;  // n_vocab + 1 offsets into text_buf
  std::vector<uint32_t> attr;
  std::vector<uint8_t> eog;
  std::string piece_buf;
  std::vector<uint32_t> piece_off;
  std::unordered_map<std::string_view, int> to_id;
  std::vector<int> specials;  // control, user-defined and unknown tokens, longest text first
  int byte_tok[256];
  MergeTable merges;
  int eos = -1;

  std::string_view text(int id) const { return std::string_view(text_buf).substr(text_off[id], text_off[id + 1] - text_off[id]); }
  int id_of(std::string_view s) const {
    auto it = to_id.find(s);
    return it == to_id.end() ? -1 : it->second;
  }
};

// llama_decode_text: token text (GPT-2 byte symbols) -> bytes.
std::string decode_text(std::string_view text) {
  const ByteMap& bm = byte_map();
  std::string out;
  const uint8_t* s = (const uint8_t*)text.data();
  for (size_t i = 0; i < text.size();) {
    size_t len;
    const uint32_t c = next_cpt(s, text.size(), i, len);
    i += len;
    if (c < 0x144 && bm.to_byte[c] >= 0) {
      out += (char)bm.to_byte[c];
    } else {
      std::string u;
      put_utf8(u, c);
      out += "[UNK_BYTE_0x";
      static const char* hex = "0123456789abcdef";
      for (unsigned char b : u) { out += hex[b >> 4]; out += hex[b & 15]; }
      out.append(text.data(), text.size());
      out += "]";
    }
  }
  return out;
}

// Per-call buffers.
struct Work {
  std::vector<Cp> cps;
  std::string norm;
  std::vector<int> sym_id, sym_prev, sym_next;
  std::vector<uint32_t> sym_len;
  std::vector<Bigram> heap;
  std::unordered_map<std::string_view, std::pair<uint32_t, uint32_t>> cache;  // word -> (offset, count) in cache_ids
  std::vector<int> cache_ids;
};

// llm_tokenizer_bpe_session::tokenize for one word (bytes of the text; the GPT-2 byte symbols are implied).
void bpe_word(const Vocab& m, const uint8_t* w, size_t n, std::vector<int>& out, Work& k) {
  if (n == 1) { out.push_back(m.byte_tok[w[0]]); return; }
  if (k.sym_id.size() < n) {
    k.sym_id.resize(n); k.sym_prev.resize(n); k.sym_next.resize(n); k.sym_len.resize(n);
  }
  int* id = k.sym_id.data();
  int* prev = k.sym_prev.data();
  int* next = k.sym_next.data();
  uint32_t* len = k.sym_len.data();
  for (size_t i = 0; i < n; i++) {
    id[i] = m.byte_tok[w[i]];
    len[i] = 1;
    prev[i] = (int)i - 1;
    next[i] = i + 1 == n ? -1 : (int)i + 1;
  }
  std::vector<Bigram>& heap = k.heap;
  heap.clear();
  auto push = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    const MergeTable::Slot* s = m.merges.find(id[l], id[r]);
    if (!s) return;
    heap.push_back(Bigram{s->rank, l, r, s->id, len[l] + len[r]});
    std::push_heap(heap.begin(), heap.end(), BigramOrder());
  };
  for (int i = 1; i < (int)n; i++) push(i - 1, i);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), BigramOrder());
    const Bigram b = heap.back();
    heap.pop_back();
    if (len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size) continue;
    id[b.left] = b.id;
    len[b.left] += len[b.right];
    len[b.right] = 0;
    next[b.left] = next[b.right];
    if (next[b.right] >= 0) prev[next[b.right]] = b.left;
    push(prev[b.left], b.left);
    push(b.left, next[b.left]);
  }
  for (int i = 0; i != -1; i = next[i]) out.push_back(id[i]);
}

// Words longer than this skip the cache.
constexpr size_t kCacheMaxWord = 64;

void encode_word(const Vocab& m, const uint8_t* w, size_t n, bool cacheable, std::vector<int>& out, Work& k) {
  if (n == 1) { out.push_back(m.byte_tok[w[0]]); return; }
  if (!cacheable || n > kCacheMaxWord) { bpe_word(m, w, n, out, k); return; }
  const std::string_view key((const char*)w, n);
  auto it = k.cache.find(key);
  if (it != k.cache.end()) {
    out.insert(out.end(), k.cache_ids.begin() + it->second.first, k.cache_ids.begin() + it->second.first + it->second.second);
    return;
  }
  const size_t first = out.size();
  bpe_word(m, w, n, out, k);
  const uint32_t off = (uint32_t)k.cache_ids.size();
  k.cache_ids.insert(k.cache_ids.end(), out.begin() + first, out.end());
  k.cache.emplace(key, std::make_pair(off, (uint32_t)(out.size() - first)));
}

// One text fragment (no special tokens in it): UTF-8 decode, qwen35 pre-tokenizer, BPE per word.
void encode_fragment(const Vocab& m, const uint8_t* s, size_t n, std::vector<int>& out, Work& k) {
  const uint8_t* cls_of = class_table();
  // Decode as unicode_cpts_from_utf8: an invalid byte becomes U+FFFD, an overlong form becomes its short form.
  // llama.cpp rebuilds the words from the code points, so then the words come from a normalized copy.
  k.cps.resize(n);
  Cp* cp = k.cps.data();
  size_t nc = 0;
  bool normalized = false;
  for (size_t i = 0; i < n;) {
    size_t len;
    const uint32_t c = next_cpt(s, n, i, len);
    const int el = utf8_len(c);  // throws above U+10FFFF
    if (!normalized && (size_t)el != len) {
      normalized = true;
      k.norm.assign((const char*)s, i);
    }
    if (normalized) put_utf8(k.norm, c);
    cp[nc].ch = c < 0x80 ? (uint8_t)c : (uint8_t)0x80;
    cp[nc].cls = (uint8_t)(cls_of[c] | (el << 4));
    nc++;
    i += len;
  }
  const uint8_t* base = normalized ? (const uint8_t*)k.norm.data() : s;
  const bool cacheable = !normalized;  // cache keys point into the input text, which outlives the call

  // unicode_regex_split_custom_qwen35:
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  const uint8_t LM = C_LETTER | C_MARK, WLMN = C_SPACE | C_LETTER | C_MARK | C_NUMBER;
  auto K = [&](size_t p) -> uint8_t { return p < nc ? (uint8_t)(cp[p].cls & 15) : (uint8_t)0; };
  // unicode_tolower for the contraction letters: only ASCII A-Z map to them in llama.cpp's table
  auto lower = [](uint8_t c) -> uint8_t { return c >= 'A' && c <= 'Z' ? (uint8_t)(c + 32) : c; };
  size_t prev_end = 0, byte_pos = 0;
  auto add = [&](size_t end) -> size_t {
    const size_t len = end - prev_end;
    if (len > 0) {
      size_t nb = 0;
      for (size_t p = prev_end; p < end; p++) nb += cp[p].cls >> 4;
      encode_word(m, base + byte_pos, nb, cacheable, out, k);
      byte_pos += nb;
    }
    prev_end = end;
    return len;
  };
  for (size_t pos = 0; pos < nc;) {
    const uint8_t c = cp[pos].ch;
    const uint8_t f = cp[pos].cls & 15;
    if (c == '\'' && pos + 1 < nc) {
      const uint8_t c1 = lower(cp[pos + 1].ch);
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { pos += add(pos + 2); continue; }
      if (pos + 2 < nc) {
        const uint8_t c2 = lower(cp[pos + 2].ch);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { pos += add(pos + 3); continue; }
      }
    }
    if (!(c == '\r' || c == '\n' || (f & C_NUMBER))) {
      if ((f & LM) || (K(pos + 1) & LM)) {
        pos++;
        while (K(pos) & LM) pos++;
        add(pos);
        continue;
      }
    }
    if (f & C_NUMBER) { add(++pos); continue; }
    // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*' (a space at the end of the fragment also lands here, as in llama.cpp)
    const uint8_t f2 = c == ' ' ? K(pos + 1) : f;
    if (!(f2 & WLMN)) {
      pos += (c == ' ');
      while (pos < nc && !(cp[pos].cls & WLMN)) pos++;
      while (pos < nc && (cp[pos].ch == '\r' || cp[pos].ch == '\n')) pos++;
      add(pos);
      continue;
    }
    size_t nws = 0, last_rn = 0;
    while (K(pos + nws) & C_SPACE) {
      const uint8_t c2 = cp[pos + nws].ch;
      if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
      nws++;
    }
    if (last_rn > 0) { pos = last_rn; add(pos); continue; }               // \s*[\r\n]+
    if (nws > 1 && pos + nws < nc) { pos += nws - 1; add(pos); continue; }  // \s+(?!\S)
    if (nws > 0) { pos += nws; add(pos); continue; }                      // \s+
    add(++pos);
  }
}

// Leftmost occurrence of `pat` inside [from, to) of `s`, or npos.
size_t find_in(std::string_view s, size_t from, size_t to, std::string_view pat) {
  if (pat.empty() || to - from < pat.size()) return std::string_view::npos;
  const char first = pat[0];
  const char* p = s.data() + from;
  const char* last = s.data() + to - pat.size();
  while (p <= last) {
    p = (const char*)memchr(p, first, (size_t)(last - p) + 1);
    if (!p) return std::string_view::npos;
    if (memcmp(p, pat.data(), pat.size()) == 0) return (size_t)(p - s.data());
    p++;
  }
  return std::string_view::npos;
}

}  // namespace

struct Tokenizer::Impl : Vocab {};

Tokenizer::Tokenizer(const GGUF& g) : impl_(new Impl) {
  Impl& m = *impl_;
  try {
    if (g.get_str("tokenizer.ggml.model") != "gpt2") throw std::runtime_error("tokenizer: model is not gpt2 (byte-level BPE)");
    if (!g.has("tokenizer.ggml.pre") || g.get_str("tokenizer.ggml.pre") != "qwen35")
      throw std::runtime_error("tokenizer: pre-tokenizer is not qwen35");
    const std::vector<std::string>& toks = g.kv("tokenizer.ggml.tokens").as;
    const int n = (int)toks.size();
    m.text_off.reserve(n + 1);
    for (int i = 0; i < n; i++) {
      m.text_off.push_back((uint32_t)m.text_buf.size());
      m.text_buf += toks[i].empty() ? "[EMPTY_" + std::to_string(i) + "]" : toks[i];
    }
    m.text_off.push_back((uint32_t)m.text_buf.size());
    m.to_id.reserve(n);
    for (int i = 0; i < n; i++)
      if (!m.to_id.emplace(m.text(i), i).second) throw std::runtime_error("tokenizer: duplicate token text");

    m.attr.assign(n, ATTR_NORMAL);
    if (g.has("tokenizer.ggml.token_type")) {
      const std::vector<int64_t>& types = g.kv("tokenizer.ggml.token_type").ai;
      if ((int)types.size() < n) throw std::runtime_error("tokenizer: short token_type array");
      for (int i = 0; i < n; i++) m.attr[i] = attr_of_type(types[i]);
    }

    // Special ids (llama.cpp load(): GGUF keys, else detected by text; detected tokens are forced to CONTROL).
    auto key_id = [&](const char* key) -> int {
      if (!g.has(key)) return -1;
      const int64_t v = g.get_int(key);
      return v >= 0 && v < n ? (int)v : -1;
    };
    auto detect = [&](int id, std::initializer_list<const char*> texts) -> int {
      if (id < 0) {
        // llama.cpp takes the first match of a hash map walk; this vocab has at most one match per list
        for (const char* t : texts) {
          const int j = m.id_of(t);
          if (j >= 0 && (id < 0 || j < id)) id = j;
        }
        if (id >= 0) m.attr[id] |= ATTR_CONTROL;
      }
      return id;
    };
    m.eos = key_id("tokenizer.ggml.eos_token_id");
    if (m.eos < 0) m.eos = 11 < n ? 11 : -1;  // llama.cpp default for gpt2 vocabs
    const int eot = detect(key_id("tokenizer.ggml.eot_token_id"),
                           {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>",
                            "_<EOT>", "[EOT]", "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>", "<end_of_utterance>"});
    const int eom = detect(key_id("tokenizer.ggml.eom_token_id"), {"<|eom_id|>"});
    detect(key_id("tokenizer.ggml.fim_pre_token_id"),
           {"<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<\xEF\xBD\x9C" "fim\xE2\x96\x81" "begin\xEF\xBD\x9C>", "<PRE>",
            "\xE2\x96\x81<PRE>", "<|code_prefix|>", "<|prefix|>"});
    detect(key_id("tokenizer.ggml.fim_suf_token_id"),
           {"<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<\xEF\xBD\x9C" "fim\xE2\x96\x81hole\xEF\xBD\x9C>", "<SUF>",
            "\xE2\x96\x81<SUF>", "<|code_suffix|>", "<|suffix|>"});
    detect(key_id("tokenizer.ggml.fim_mid_token_id"),
           {"<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<\xEF\xBD\x9C" "fim\xE2\x96\x81" "end\xEF\xBD\x9C>", "<MID>",
            "\xE2\x96\x81<MID>", "<|code_middle|>", "<|middle|>"});
    const int fim_pad = detect(key_id("tokenizer.ggml.fim_pad_token_id"), {"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"});
    const int fim_rep = detect(key_id("tokenizer.ggml.fim_rep_token_id"),
                               {"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"});
    const int fim_sep = detect(key_id("tokenizer.ggml.fim_sep_token_id"), {"<|file_sep|>"});

    // control tokens named "unused" are unused
    for (int i = 0; i < n; i++)
      if ((m.attr[i] & ATTR_CONTROL) && m.text(i).find("unused") != std::string_view::npos) m.attr[i] |= ATTR_UNUSED;

    // End-of-generation set. llama.cpp's fix-ups for gpt-oss (<|end|>) and gemma4 (</s>) do not apply here.
    m.eog.assign(n, 0);
    for (int id : {fim_pad, fim_rep, fim_sep, m.eos, eot, eom})
      if (id >= 0) m.eog[id] = 1;
    for (const char* t : {"<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
                          "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
                          "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
                          "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>", "[e~["}) {
      const int id = m.id_of(t);
      if (id >= 0) { m.eog[id] = 1; m.attr[id] |= ATTR_CONTROL; }
    }

    // Special tokens matched in the text before BPE, longest first.
    for (int i = 0; i < n; i++)
      if (m.attr[i] & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN)) m.specials.push_back(i);
    std::stable_sort(m.specials.begin(), m.specials.end(),
                     [&](int a, int b) { return m.text(a).size() > m.text(b).size(); });

    // Byte symbols must all be tokens: BPE starts from them.
    const ByteMap& bm = byte_map();
    for (int b = 0; b < 256; b++) {
      std::string u;
      put_utf8(u, bm.to_cpt[b]);
      m.byte_tok[b] = m.id_of(u);
      if (m.byte_tok[b] < 0) throw std::runtime_error("tokenizer: byte symbol missing from the vocab");
    }

    // Merges "left right", rank = index. Every word symbol is a token (bytes, then merge results), so a merge is
    // only reachable if both parts are tokens; its result must be a token too (true for this GGUF; checked).
    const std::vector<std::string>& merges = g.kv("tokenizer.ggml.merges").as;
    m.merges.init(merges.size());
    for (size_t r = 0; r < merges.size(); r++) {
      const std::string_view w = merges[r];
      const size_t sp = w.find(' ', 1);
      if (sp == std::string_view::npos) continue;
      const int a = m.id_of(w.substr(0, sp)), b = m.id_of(w.substr(sp + 1));
      if (a < 0 || b < 0) continue;
      std::string joined(w.substr(0, sp));
      joined += w.substr(sp + 1);
      const int id = m.id_of(joined);
      if (id < 0) throw std::runtime_error("tokenizer: merge result is not a token: " + std::string(w));
      m.merges.insert(a, b, (int)r, id);
    }

    // Pieces (token_to_piece with special = true).
    m.piece_off.reserve(n + 1);
    for (int i = 0; i < n; i++) {
      m.piece_off.push_back((uint32_t)m.piece_buf.size());
      const uint32_t a = m.attr[i];
      const std::string_view t = m.text(i);
      if (a & (ATTR_UNKNOWN | ATTR_CONTROL | ATTR_USER_DEFINED)) m.piece_buf.append(t.data(), t.size());
      else if (a & ATTR_NORMAL) m.piece_buf += decode_text(t);
      else if (a & ATTR_BYTE) m.piece_buf += (char)strtol(std::string(t.substr(3, 2)).c_str(), nullptr, 16);
      // unused and undefined tokens have no piece
    }
    m.piece_off.push_back((uint32_t)m.piece_buf.size());
    class_table();
  } catch (...) {
    delete impl_;
    throw;
  }
}

Tokenizer::~Tokenizer() { delete impl_; }

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
  const Impl& m = *impl_;
  std::vector<int> out;
  if (text.empty()) return out;
  out.reserve(text.size() / 3 + 16);

  // tokenizer_st_partition: split the text at special tokens, one token at a time, longest first. Control (and
  // unknown) tokens only with parse_special; user-defined tokens always. No LSTRIP/RSTRIP tokens in this vocab.
  struct Frag {
    size_t off, len;
    int tok;  // -1: text
  };
  std::vector<Frag> frags{Frag{0, text.size(), -1}}, next;
  const std::string_view sv(text);
  for (int sid : m.specials) {
    if (!parse_special && (m.attr[sid] & (ATTR_CONTROL | ATTR_UNKNOWN))) continue;
    const std::string_view tok = m.text(sid);
    if (find_in(sv, 0, sv.size(), tok) == std::string_view::npos) continue;
    next.clear();
    for (const Frag& f : frags) {
      if (f.tok >= 0) { next.push_back(f); continue; }
      size_t p = f.off;
      const size_t end = f.off + f.len;
      for (;;) {
        const size_t at = find_in(sv, p, end, tok);
        if (at == std::string_view::npos) break;
        if (at > p) next.push_back(Frag{p, at - p, -1});
        next.push_back(Frag{at, 0, sid});
        p = at + tok.size();
      }
      if (p < end) next.push_back(Frag{p, end - p, -1});
    }
    frags.swap(next);
  }

  Work k;
  for (const Frag& f : frags) {
    if (f.tok >= 0) out.push_back(f.tok);
    else encode_fragment(m, (const uint8_t*)text.data() + f.off, f.len, out, k);
  }
  return out;
}

std::string Tokenizer::piece(int id) const {
  const Impl& m = *impl_;
  if (id < 0 || id >= (int)m.attr.size()) throw std::out_of_range("tokenizer: token id out of range");
  return m.piece_buf.substr(m.piece_off[id], m.piece_off[id + 1] - m.piece_off[id]);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  const Impl& m = *impl_;
  std::string s;
  for (int id : ids) {
    if (id < 0 || id >= (int)m.attr.size()) throw std::out_of_range("tokenizer: token id out of range");
    s.append(m.piece_buf, m.piece_off[id], m.piece_off[id + 1] - m.piece_off[id]);
  }
  return s;
}

int Tokenizer::n_vocab() const { return (int)impl_->attr.size(); }
int Tokenizer::eos() const { return impl_->eos; }
bool Tokenizer::is_eog(int id) const { return id >= 0 && id < n_vocab() && impl_->eog[id]; }
bool Tokenizer::is_special(int id) const {
  return id >= 0 && id < n_vocab() && (impl_->attr[id] & (ATTR_CONTROL | ATTR_USER_DEFINED));
}
int Tokenizer::find(const std::string& token_text) const { return impl_->id_of(token_text); }

}  // namespace q27
