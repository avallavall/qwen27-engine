// Streaming parser for reasoning, content and XML tool calls. See chat_parser.h for the llama.cpp behaviour it
// reproduces and the few places where it differs.
//
// Design: one byte-driven state machine. Input bytes go to `in`; each state consumes what it can decide and
// leaves the rest (a possible tag start, an incomplete UTF-8 sequence) for the next push. Every byte is looked
// at a bounded number of times, except a JSON-typed parameter value, which is held until its close tag and read
// a second time if it turns out not to be JSON. Cost per token: O(new bytes).
#include "chat_parser.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_map>

#include "tokenizer.h"

namespace q27 {

namespace {

bool is_ws(unsigned char c) {  // std::isspace in the C locale, as llama.cpp's p.space()
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

// ---- UTF-8, as llama.cpp common_parse_utf8_codepoint (common/unicode.cpp:17): no overlong or surrogate check.
enum class U8 { OK, INCOMPLETE, INVALID };

// Sequence at s[p]. n = its length (OK), the bytes present (INCOMPLETE) or the run to replace (INVALID).
U8 utf8_at(const char* s, size_t size, size_t p, size_t& n) {
  unsigned char c = static_cast<unsigned char>(s[p]);
  if (c < 0x80) { n = 1; return U8::OK; }
  if (!(c & 0x40)) { n = 1; return U8::INVALID; }
  size_t len = !(c & 0x20) ? 2 : !(c & 0x10) ? 3 : !(c & 0x08) ? 4 : 0;
  if (len == 0) { n = 1; return U8::INVALID; }
  for (size_t i = 1; i < len; i++) {
    if (p + i >= size) { n = i; return U8::INCOMPLETE; }
    if ((static_cast<unsigned char>(s[p + i]) & 0xC0) != 0x80) { n = i; return U8::INVALID; }
  }
  n = len;
  return U8::OK;
}

const char kReplacement[] = "\xEF\xBF\xBD";  // U+FFFD

// Whether a sequence the lax decoder accepted is valid UTF-8 (RFC 3629: no overlong form, no surrogate, nothing
// above U+10FFFF).
bool strict_utf8(const unsigned char* s, size_t n) {
  if (n == 1) return true;
  unsigned char a = s[0], b = s[1];
  if (n == 2) return a >= 0xC2;
  if (n == 3) return !(a == 0xE0 && b < 0xA0) && !(a == 0xED && b >= 0xA0);
  return a <= 0xF4 && !(a == 0xF0 && b < 0x90) && !(a == 0xF4 && b >= 0x90);
}

// Appends text as the inside of a JSON string, escaped as nlohmann dump does with ensure_ascii off: \" \\ \b \f
// \n \r \t, other bytes below 0x20 as \u00xx, the rest raw. Bytes that are not valid UTF-8 become U+FFFD, so
// the result is always valid JSON (nlohmann, and so llama.cpp, throws there).
void append_json_text(std::string& out, const char* s, size_t n) {
  size_t run = 0, i = 0;
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x20 && c < 0x80 && c != '"' && c != '\\') {
      i++;
      continue;
    }
    if (c >= 0x80) {
      size_t len;
      U8 u = utf8_at(s, n, i, len);
      if (u == U8::OK && strict_utf8(reinterpret_cast<const unsigned char*>(s) + i, len)) {
        i += len;
        continue;
      }
      out.append(s + run, i - run);
      out += kReplacement;
      i += len;
      run = i;
      continue;
    }
    out.append(s + run, i - run);
    run = ++i;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: {
        char buf[8];
        snprintf(buf, sizeof buf, "\\u%04x", c);
        out += buf;
      }
    }
  }
  out.append(s + run, n - run);
}

std::string json_string(std::string_view s) {
  std::string out = "\"";
  append_json_text(out, s.data(), s.size());
  out += '"';
  return out;
}

// llama.cpp chat-peg-parser.cpp:36: at most one leading and all trailing whitespace characters removed.
std::string_view trim_name(std::string_view s) {
  if (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_ws(s.back())) s.remove_suffix(1);
  return s;
}

// ---- Parameter typing from the tool's JSON schema (common/json-schema.cpp:227 build_node, :368 value_types).
enum : unsigned { T_NULL = 1, T_BOOL = 2, T_NUM = 4, T_INT = 8, T_STR = 16, T_ARR = 32, T_OBJ = 64, T_ALL = 127 };

unsigned json_type(const ojson& v) {
  if (v.is_null()) return T_NULL;
  if (v.is_boolean()) return T_BOOL;
  if (v.is_number_integer()) return T_INT;
  if (v.is_number()) return T_NUM;
  if (v.is_string()) return T_STR;
  if (v.is_array()) return T_ARR;
  return T_OBJ;
}

bool has_properties(const ojson& s) {
  return s.contains("properties") || (s.contains("additionalProperties") && s["additionalProperties"] != true);
}

bool known_string_format(const ojson& s) {
  if (!s.contains("format") || !s["format"].is_string()) return false;
  std::string f = s["format"].get<std::string>();
  return f == "date" || f == "time" || f == "date-time" || f == "uuid" ||
         (f.size() == 5 && f.compare(0, 4, "uuid") == 0 && f[4] >= '1' && f[4] <= '5');
}

// Value types of a schema node. Schemas llama.cpp rejects (it fails the request) count as "any".
unsigned schema_types(const ojson& s, const ojson& root, std::vector<std::string>& refs, int depth) {
  if (depth > 64 || !s.is_object()) return T_ALL;
  auto all_of = [&](const ojson& alts) {
    if (!alts.is_array() || alts.empty()) return unsigned(T_ALL);
    unsigned t = T_ALL;
    for (const auto& a : alts) t &= schema_types(a, root, refs, depth + 1);
    return t;
  };
  if (s.contains("$ref")) {
    const ojson& r = s["$ref"];
    if (!r.is_string()) return T_ALL;
    std::string ref = r.get<std::string>();
    if (ref.compare(0, 2, "#/") != 0) return T_ALL;
    for (const auto& seen : refs)
      if (seen == ref) return 0;  // a cycle contributes no type
    const ojson* target = &root;
    size_t pos = 2;
    while (pos <= ref.size()) {
      size_t end = ref.find('/', pos);
      if (end == std::string::npos) end = ref.size();
      std::string sel = ref.substr(pos, end - pos);
      if (target->is_object() && target->contains(sel)) {
        target = &(*target)[sel];
      } else if (target->is_array()) {
        size_t idx = 0;
        bool ok = !sel.empty() && sel.size() < 10;
        for (char c : sel) {
          if (c < '0' || c > '9') ok = false;
          idx = idx * 10 + static_cast<size_t>(c - '0');
        }
        if (!ok || idx >= target->size()) return T_ALL;
        target = &(*target)[idx];
      } else {
        return T_ALL;
      }
      pos = end + 1;
    }
    refs.push_back(ref);
    unsigned t = schema_types(*target, root, refs, depth + 1);
    refs.pop_back();
    return t;
  }
  if (s.contains("oneOf") || s.contains("anyOf")) {
    const ojson& alts = s.contains("oneOf") ? s["oneOf"] : s["anyOf"];
    if (!alts.is_array() || alts.empty()) return T_ALL;
    unsigned t = 0;
    for (const auto& a : alts) t |= schema_types(a, root, refs, depth + 1);
    return t;
  }
  ojson type = s.contains("type") ? s["type"] : ojson();
  if (type.is_array()) {
    if (type.empty()) return T_ALL;
    unsigned t = 0;
    for (const auto& one : type) {
      ojson alt = s;
      alt["type"] = one;
      t |= schema_types(alt, root, refs, depth + 1);
    }
    return t;
  }
  if (s.contains("const")) return json_type(s["const"]);
  if (s.contains("enum")) {
    const ojson& e = s["enum"];
    if (!e.is_array() || e.empty()) return T_ALL;
    unsigned t = 0;
    for (const auto& v : e) t |= json_type(v);
    return t;
  }
  if (!type.is_null() && !type.is_string()) return T_ALL;
  std::string name = type.is_string() ? type.get<std::string>() : "";
  if (name.empty()) {
    if (has_properties(s)) return T_OBJ;
    if (s.contains("allOf")) return all_of(s["allOf"]);
    if (s.contains("items") || s.contains("prefixItems")) return T_ARR;
    if (s.contains("pattern") || s.contains("minLength") || s.contains("maxLength") || known_string_format(s))
      return T_STR;
    return T_ALL;
  }
  if (name == "object") return !has_properties(s) && s.contains("allOf") ? all_of(s["allOf"]) : T_OBJ;
  if (name == "string") return s.contains("allOf") ? all_of(s["allOf"]) : T_STR;
  if (name == "array") return T_ARR;
  if (name == "integer") return T_INT;
  if (name == "number") return T_NUM | T_INT;
  if (name == "boolean") return T_BOOL;
  if (name == "null") return T_NULL;
  return T_ALL;
}

// Whether the parameters schema builds to an object node (only then has the tool named parameters,
// common/parsers/parsers.cpp:15).
bool schema_is_object(const ojson& s) {
  if (!s.is_object() || s.contains("$ref") || s.contains("oneOf") || s.contains("anyOf")) return false;
  if (s.contains("type") && s["type"].is_array()) return false;
  if (s.contains("const") || s.contains("enum")) return false;
  std::string name = s.contains("type") && s["type"].is_string() ? s["type"].get<std::string>() : "";
  if (name.empty()) return has_properties(s);
  if (name == "object") return has_properties(s) || !s.contains("allOf");
  return false;
}

// JSON kinds a value may start with.
enum : unsigned { J_OBJ = 1, J_ARR = 2, J_STR = 4, J_NUM = 8, J_BOOL = 16, J_NULL = 32, J_ALL = 63 };

enum class Mode { STRING, JSON, UNION };

struct Param {
  std::string name;
  Mode mode = Mode::UNION;
  unsigned kinds = J_ALL & ~J_STR;  // JSON kinds tried first (JSON, UNION)
};

void type_param(Param& p, unsigned t) {
  if (!(t & T_STR)) {
    p.mode = Mode::JSON;  // qwen3-coder.cpp:111: any JSON value, strings included
    p.kinds = J_ALL;
  } else if (t == T_STR) {
    p.mode = Mode::STRING;  // qwen3-coder.cpp:113
  } else {
    p.mode = Mode::UNION;  // qwen3-coder.cpp:114-135: the other types' JSON kinds, else the raw string
    p.kinds = 0;
    if (t & T_OBJ) p.kinds |= J_OBJ;
    if (t & T_ARR) p.kinds |= J_ARR;
    if (t & (T_NUM | T_INT)) p.kinds |= J_NUM;
    if (t & T_BOOL) p.kinds |= J_BOOL;
    if (t & T_NULL) p.kinds |= J_NULL;
  }
}

struct Tool {
  std::string name;
  std::string tag;  // name + ">\n", the end of "<function=NAME>\n"
  std::vector<Param> params;
};

bool is_json_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// ---- Incremental JSON recognizer. Same grammar as llama.cpp's PEG json() (common/peg-parser.cpp:1220-1295):
// escapes \" \\ \/ \b \f \n \r \t \uXXXX; numbers -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)? not followed by
// [0-9.eE+-]. It is stricter in three points, so that an accepted value is always valid JSON: whitespace is only
// space, \t, \n, \r (PEG: std::isspace); no raw control character in strings; strict UTF-8 in strings.
// llama.cpp's grammar never lets the model write those; where they appear, the value becomes a string.
class JsonScan {
 public:
  enum R { CONT, DONE_WITH, DONE_BEFORE, FAIL };  // DONE_BEFORE: the value ended before this byte

  void reset(unsigned top_kinds) {
    top_ = top_kinds;
    st_ = V;
    stack_.clear();
  }

  // at = index of the byte in the value text (to know what to leave out at EOS)
  R feed(unsigned char c, size_t at) {
    for (;;) {
      switch (st_) {
        case V: return start(c, at);
        case O_FIRST:
          if (is_json_ws(c)) return CONT;
          if (c == '}') return pop();
          if (c == '"') return string(true);
          return FAIL;
        case O_KEY:
          if (is_json_ws(c)) return CONT;
          if (c == '"') return string(true);
          return FAIL;
        case O_COLON:
          if (is_json_ws(c)) return CONT;
          if (c == ':') { st_ = O_VAL; return CONT; }
          return FAIL;
        case O_VAL:
        case A_VAL:
          if (is_json_ws(c)) return CONT;
          st_ = V;
          continue;
        case O_NEXT:
          if (is_json_ws(c)) return CONT;
          if (c == ',') { st_ = O_KEY; return CONT; }
          if (c == '}') return pop();
          return FAIL;
        case A_FIRST:
          if (is_json_ws(c)) return CONT;
          if (c == ']') return pop();
          st_ = V;
          continue;
        case A_NEXT:
          if (is_json_ws(c)) return CONT;
          if (c == ',') { st_ = A_VAL; return CONT; }
          if (c == ']') return pop();
          return FAIL;
        case S:
          if (c == '"') {
            if (key_) { st_ = O_COLON; return CONT; }
            return done();
          }
          if (c == '\\') { st_ = S_ESC; esc_at_ = at; return CONT; }
          if (c < 0x20) return FAIL;  // raw control character
          if (c < 0x80) return CONT;
          if (c < 0xC2 || c > 0xF4) return FAIL;  // UTF-8 lead bytes of RFC 3629
          left_ = c < 0xE0 ? 1 : c < 0xF0 ? 2 : 3;
          lo_ = c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80;  // no overlong form
          hi_ = c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;  // no surrogate, nothing above U+10FFFF
          st_ = S_U8;
          u8_at_ = at;
          return CONT;
        case S_U8:
          if (c < lo_ || c > hi_) return FAIL;
          lo_ = 0x80;
          hi_ = 0xBF;
          if (--left_ == 0) st_ = S;
          return CONT;
        case S_ESC:
          if (c == 'u') { st_ = S_U; left_ = 4; return CONT; }
          if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
            st_ = S;
            return CONT;
          }
          return FAIL;
        case S_U:
          if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return FAIL;
          if (--left_ == 0) st_ = S;
          return CONT;
        case N_SIGN:
          if (c == '0') { st_ = N_ZERO; return CONT; }
          if (c >= '1' && c <= '9') { st_ = N_INT; return CONT; }
          return FAIL;
        case N_ZERO:
        case N_INT:
        case N_FRAC:
        case N_EXP: {
          bool digit = c >= '0' && c <= '9';
          if (digit && st_ != N_ZERO) return CONT;
          if (c == '.' && (st_ == N_ZERO || st_ == N_INT)) { st_ = N_DOT; return CONT; }
          if ((c == 'e' || c == 'E') && st_ != N_EXP) { st_ = N_E; return CONT; }
          if (digit || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') return FAIL;
          // the number ends before c
          if (stack_.empty()) return DONE_BEFORE;
          st_ = stack_.back() == 'o' ? O_NEXT : A_NEXT;
          continue;
        }
        case N_DOT:
          if (c >= '0' && c <= '9') { st_ = N_FRAC; return CONT; }
          return FAIL;
        case N_E:
          if (c == '+' || c == '-') { st_ = N_ESIGN; return CONT; }
          if (c >= '0' && c <= '9') { st_ = N_EXP; return CONT; }
          return FAIL;
        case N_ESIGN:
          if (c >= '0' && c <= '9') { st_ = N_EXP; return CONT; }
          return FAIL;
        case LIT:
          if (c != static_cast<unsigned char>(lit_[lit_pos_])) return FAIL;
          if (lit_[++lit_pos_] == 0) return done();
          return CONT;
      }
      return FAIL;
    }
  }

  // Bytes of the value text that llama.cpp keeps when the output ends here (peg-parser.cpp:603,617): a pending
  // escape or an incomplete UTF-8 sequence in a string is left out.
  size_t kept(size_t len) const {
    if (st_ == S_ESC || st_ == S_U) return esc_at_;
    if (st_ == S_U8) return u8_at_;
    return len;
  }

 private:
  enum St { V, O_FIRST, O_KEY, O_COLON, O_VAL, O_NEXT, A_FIRST, A_VAL, A_NEXT, S, S_U8, S_ESC, S_U,
            N_SIGN, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP, LIT };

  R start(unsigned char c, size_t) {
    unsigned allowed = stack_.empty() ? top_ : J_ALL;
    unsigned kind = c == '{' ? J_OBJ : c == '[' ? J_ARR : c == '"' ? J_STR
                  : (c == '-' || (c >= '0' && c <= '9')) ? J_NUM : (c == 't' || c == 'f') ? J_BOOL
                  : c == 'n' ? J_NULL : 0;
    if (!(kind & allowed)) return FAIL;
    switch (c) {
      case '{': stack_.push_back('o'); st_ = O_FIRST; return CONT;
      case '[': stack_.push_back('a'); st_ = A_FIRST; return CONT;
      case '"': return string(false);
      case '-': st_ = N_SIGN; return CONT;
      case '0': st_ = N_ZERO; return CONT;
      case 't': lit_ = "true"; break;
      case 'f': lit_ = "false"; break;
      case 'n': lit_ = "null"; break;
      default: st_ = N_INT; return CONT;
    }
    lit_pos_ = 1;
    st_ = LIT;
    return CONT;
  }
  R string(bool key) {
    key_ = key;
    st_ = S;
    return CONT;
  }
  R pop() {
    stack_.pop_back();
    return done();
  }
  R done() {  // a value ended with the current byte
    if (stack_.empty()) return DONE_WITH;
    st_ = stack_.back() == 'o' ? O_NEXT : A_NEXT;
    return CONT;
  }

  unsigned top_ = J_ALL;
  St st_ = V;
  std::vector<char> stack_;  // 'o' object, 'a' array
  bool key_ = false;
  int left_ = 0;
  unsigned char lo_ = 0x80, hi_ = 0xBF;
  size_t esc_at_ = 0, u8_at_ = 0;
  const char* lit_ = "";
  int lit_pos_ = 0;
};

const char kParamClose[] = "\n</parameter>\n";
constexpr size_t kParamCloseLen = sizeof(kParamClose) - 1;

}  // namespace

struct StreamParser::Impl {
  enum class St {
    R_WS,          // leading whitespace of the reasoning (dropped)
    R_TEXT,        // reasoning, until "</think>" or "<tool_call>"
    C_WS,          // leading whitespace of the content (dropped)
    C_TEXT,        // content, until "<tool_call>" (with tools) or the end
    T_OPEN,        // after "<tool_call>": "\n<function="
    T_NAME,        // function name and ">\n"
    A_NEXT,        // "<parameter=" or "</function>\n"
    A_NAME,        // parameter name, ">", "\n"
    A_STR,         // string value, until "\n</parameter>\n"
    A_JSON,        // JSON value (held back until the close tag)
    A_JSON_CLOSE,  // "\n</parameter>\n" after a JSON value
    T_CLOSE,       // "</tool_call>"
    T_BETWEEN,     // whitespace, then "<tool_call>" for the next call
    DEAD,          // the rest of the output is ignored
  };
  enum Target { TO_REASONING, TO_CONTENT, TO_ARGS };
  struct Call {
    std::string id, name, args;
    bool complete = false;
  };

  const Tokenizer& tok;
  bool has_tools = false;
  std::vector<Tool> tools;
  std::unordered_map<int, bool> special_shown;  // special token id -> its text is part of the output

  St st;
  std::string in;  // input not consumed yet
  size_t ip = 0;
  std::string lit;  // tag bytes being matched
  bool name_closed = false;  // A_NAME: ">" seen
  const Tool* tool = nullptr;
  int arg_count = 0;
  bool union_mode = false;
  JsonScan js;
  std::string raw;  // A_JSON / A_JSON_CLOSE: value bytes consumed so far
  size_t json_len = 0, close_pos = 0;
  bool finished = false;

  std::string reasoning, content;
  std::vector<Call> calls;
  ParseDelta* d = nullptr;
  std::mt19937_64 rng;

  Impl(const Tokenizer& t, bool thinking, const ojson& tools_json)
      : tok(t), st(thinking ? St::R_WS : St::C_WS), rng(std::random_device{}()) {
    if (!tools_json.is_array() || tools_json.empty()) return;
    has_tools = true;  // even if no entry is usable: then every call is unknown (qwen3-coder.cpp:40)
    for (const auto& entry : tools_json) {
      if (!entry.is_object() || !entry.contains("type") || entry["type"] != "function" || !entry.contains("function"))
        continue;
      const ojson& f = entry["function"];
      if (!f.is_object() || !f.contains("name") || !f["name"].is_string()) continue;
      Tool tool;
      tool.name = f["name"].get<std::string>();
      tool.tag = tool.name + ">\n";
      ojson params = f.contains("parameters") ? f["parameters"] : ojson();
      if (params.is_null() || (params.is_object() && params.empty()))  // chat.cpp:577
        params = ojson{{"type", "object"}, {"properties", ojson::object()}};
      if (schema_is_object(params) && params.contains("properties") && params["properties"].is_object()) {
        for (const auto& [name, schema] : params["properties"].items()) {
          Param p;
          p.name = name;
          std::vector<std::string> refs;
          type_param(p, schema_types(schema, params, refs, 0));
          tool.params.push_back(std::move(p));
        }
      }
      tools.push_back(std::move(tool));
    }
  }

  // ---- output
  void emit(Target t, const char* s, size_t n) {
    if (n == 0) return;
    if (t == TO_REASONING) {
      reasoning.append(s, n);
      d->reasoning.append(s, n);
    } else if (t == TO_CONTENT) {
      content.append(s, n);
      d->content.append(s, n);
    } else {
      std::string esc;
      append_json_text(esc, s, n);
      emit_args(esc);
    }
  }
  void emit_args(std::string_view s) {
    if (s.empty()) return;
    Call& c = calls.back();
    c.args.append(s);
    int index = static_cast<int>(calls.size()) - 1;
    if (d->tool_calls.empty() || d->tool_calls.back().index != index) d->tool_calls.push_back({index, "", "", ""});
    d->tool_calls.back().arguments.append(s);
  }
  void open_call(const Tool& t) {
    static const char kAlnum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    Call c;
    c.id.resize(32);
    for (auto& ch : c.id) ch = kAlnum[rng() % 62];
    c.name = t.name;
    c.args = "{";
    d->tool_calls.push_back({static_cast<int>(calls.size()), c.id, c.name, "{"});
    calls.push_back(std::move(c));
    tool = &t;
    arg_count = 0;
  }
  void close_call() {
    emit_args("}");
    calls.back().complete = true;
  }

  // ---- text runs
  // llama.cpp common_trie::check_at (common/trie.cpp:7) for ASCII delimiters at s[p]: 2 = one starts here,
  // 1 = the text from p is a proper prefix of one (the input ends, or an incomplete UTF-8 sequence follows),
  // 0 = none.
  static int check_at(const char* s, size_t n, size_t p, const char* const* dl, int nd, int& which) {
    int best = 0;
    for (int k = 0; k < nd; k++) {
      const char* dk = dl[k];
      size_t j = 0, q = p;
      for (;;) {
        if (dk[j] == 0) {
          which = k;
          return 2;
        }
        if (q >= n) {
          if (j > 0) best = 1;
          break;
        }
        unsigned char c = static_cast<unsigned char>(s[q]);
        if (c >= 0x80) {
          size_t len;
          if (j > 0 && utf8_at(s, n, q, len) == U8::INCOMPLETE) best = 1;
          break;
        }
        if (c != static_cast<unsigned char>(dk[j])) break;
        j++;
        q++;
      }
    }
    return best;
  }

  // Emits text from `ip` until one of the delimiters (as llama.cpp's until(), peg-parser.cpp:685). Invalid UTF-8
  // runs become U+FFFD. Returns the delimiter index with `ip` at its first byte, or -1 with `ip` at the first
  // byte held back (a possible delimiter start or an incomplete UTF-8 sequence) or at the end.
  int scan(const char* const* dl, int nd, Target t) {
    const char* s = in.data();
    size_t n = in.size(), p = ip, run = ip;
    int found = -1;
    while (p < n) {
      unsigned char c = static_cast<unsigned char>(s[p]);
      if (c < 0x80) {
        bool cand = false;
        for (int k = 0; k < nd; k++) cand = cand || dl[k][0] == static_cast<char>(c);
        if (cand) {
          int which = -1;
          int m = check_at(s, n, p, dl, nd, which);
          if (m == 2) { found = which; break; }
          if (m == 1) break;
        }
        p++;
        continue;
      }
      size_t len;
      U8 u = utf8_at(s, n, p, len);
      if (u == U8::OK) { p += len; continue; }
      if (u == U8::INCOMPLETE) break;
      emit(t, s + run, p - run);
      emit(t, kReplacement, 3);
      p += len;
      run = p;
    }
    emit(t, s + run, p - run);
    ip = p;
    return found;
  }

  // Matches `in` from `ip` against tags, one byte at a time into `lit`. Returns the index of the completed tag,
  // -1 when the input ran out, -2 on a byte that fits none (not consumed).
  int match(const char* const* tags, int nt) {
    while (ip < in.size()) {
      lit.push_back(in[ip]);
      bool prefix = false;
      for (int k = 0; k < nt; k++) {
        size_t len = strlen(tags[k]);
        if (lit.size() <= len && memcmp(tags[k], lit.data(), lit.size()) == 0) {
          if (lit.size() == len) {
            ip++;
            lit.clear();
            return k;
          }
          prefix = true;
        }
      }
      if (!prefix) {
        lit.pop_back();
        return -2;
      }
      ip++;
    }
    return -1;
  }

  // The text inside an open call broke the format (see chat_parser.h): close the call, ignore the rest.
  void violation() {
    close_call();
    st = St::DEAD;
  }

  void param_open() {
    std::string_view name(lit);
    Param p;  // a name the schema does not list: typed like a parameter without schema (mixed types)
    p.name = lit;
    for (const auto& q : tool->params)
      if (q.name == name) { p = q; break; }
    std::string key = arg_count > 0 ? "," : "";
    key += json_string(trim_name(name));
    key += ':';
    arg_count++;
    lit.clear();
    if (p.mode == Mode::STRING) {
      key += '"';
      emit_args(key);
      st = St::A_STR;
      return;
    }
    emit_args(key);
    js.reset(p.kinds);
    raw.clear();
    union_mode = p.mode == Mode::UNION;
    st = St::A_JSON;
  }

  // The JSON reading of a value failed: read the same bytes again as a raw string (qwen3-coder.cpp:135 for mixed
  // types; for JSON-only types llama.cpp drops the call instead, see chat_parser.h).
  void json_to_string() {
    in = raw + in.substr(ip);
    ip = 0;
    raw.clear();
    emit_args("\"");
    st = St::A_STR;
  }

  void run(bool eof) {
    static const char* const kReasonEnd[] = {"</think>", "<tool_call>"};
    static const char* const kCallStart[] = {"<tool_call>"};
    static const char* const kFuncOpen[] = {"\n<function="};
    static const char* const kArgNext[] = {"<parameter=", "</function>\n"};
    static const char* const kCallEnd[] = {"</tool_call>"};
    static const char* const kParamEnd[] = {kParamClose};
    bool more = true;
    while (more) {
      const size_t n = in.size();
      switch (st) {
        case St::R_WS:
        case St::C_WS:
          while (ip < n && is_ws(static_cast<unsigned char>(in[ip]))) ip++;
          if (ip == n) { more = false; break; }
          st = st == St::R_WS ? St::R_TEXT : St::C_TEXT;
          break;
        case St::R_TEXT: {
          int k = scan(kReasonEnd, 2, TO_REASONING);
          if (k < 0) { more = false; break; }
          if (k == 0) ip += strlen(kReasonEnd[0]);  // "<tool_call>" is only peeked at (qwen3-coder.cpp:82)
          st = St::C_WS;
          break;
        }
        case St::C_TEXT: {
          int k = scan(kCallStart, has_tools ? 1 : 0, TO_CONTENT);
          if (k < 0) { more = false; break; }
          ip += strlen(kCallStart[0]);
          lit.clear();
          st = St::T_OPEN;
          break;
        }
        case St::T_OPEN: {
          int k = match(kFuncOpen, 1);
          if (k == -1) { more = false; break; }
          st = k == 0 ? St::T_NAME : St::DEAD;
          break;
        }
        case St::T_NAME: {
          // "<function=NAME>\n" for a listed NAME (qwen3-coder.cpp:149); anything else drops the rest.
          const Tool* hit = nullptr;
          bool alive = true;
          while (ip < n && !hit && alive) {
            lit.push_back(in[ip++]);
            alive = false;
            for (const auto& t : tools) {
              if (lit.size() > t.tag.size() || t.tag.compare(0, lit.size(), lit) != 0) continue;
              if (lit.size() == t.tag.size()) { hit = &t; break; }
              alive = true;
            }
          }
          if (hit) {
            lit.clear();
            open_call(*hit);
            st = St::A_NEXT;
          } else if (!alive) {
            st = St::DEAD;
          } else {
            more = false;
          }
          break;
        }
        case St::A_NEXT: {
          int k = match(kArgNext, 2);
          if (k == -1) { more = false; break; }
          if (k == -2) { violation(); break; }
          if (k == 0) {
            name_closed = false;
            st = St::A_NAME;
          } else {
            close_call();  // "}" at "</function>\n" (chat-peg-parser.cpp:445)
            st = St::T_CLOSE;
          }
          break;
        }
        case St::A_NAME: {
          bool bad = false, opened = false;
          while (ip < n && !bad && !opened) {
            char c = in[ip];
            if (name_closed) {
              if (c != '\n') { bad = true; break; }
              ip++;
              opened = true;
            } else if (c == '>') {
              name_closed = true;
              ip++;
            } else if (c == '\n' || c == '<' || lit.size() >= 256) {
              bad = true;
            } else {
              lit.push_back(c);
              ip++;
            }
          }
          if (bad) violation();
          else if (opened) param_open();
          else more = false;
          break;
        }
        case St::A_STR: {
          int k = scan(kParamEnd, 1, TO_ARGS);
          if (k < 0) { more = false; break; }
          ip += kParamCloseLen;
          emit_args("\"");
          lit.clear();
          st = St::A_NEXT;
          break;
        }
        case St::A_JSON: {
          bool next = false;
          while (ip < n && !next) {
            unsigned char c = static_cast<unsigned char>(in[ip]);
            JsonScan::R r = js.feed(c, raw.size());
            if (r == JsonScan::FAIL) {
              json_to_string();
              next = true;
            } else if (r == JsonScan::DONE_BEFORE) {
              json_len = raw.size();
              close_pos = 0;
              st = St::A_JSON_CLOSE;
              next = true;
            } else {
              raw.push_back(static_cast<char>(c));
              ip++;
              if (r == JsonScan::DONE_WITH) {
                json_len = raw.size();
                close_pos = 0;
                st = St::A_JSON_CLOSE;
                next = true;
              }
            }
          }
          if (!next) more = false;
          break;
        }
        case St::A_JSON_CLOSE: {
          bool next = false;
          while (ip < n && !next) {
            if (in[ip] != kParamClose[close_pos]) {
              json_to_string();
              next = true;
              break;
            }
            raw.push_back(in[ip++]);
            if (++close_pos == kParamCloseLen) {
              emit_args(std::string_view(raw.data(), json_len));
              raw.clear();
              lit.clear();
              st = St::A_NEXT;
              next = true;
            }
          }
          if (!next) more = false;
          break;
        }
        case St::T_CLOSE: {
          int k = match(kCallEnd, 1);
          if (k == -1) { more = false; break; }
          st = k == 0 ? St::T_BETWEEN : St::DEAD;
          break;
        }
        case St::T_BETWEEN: {
          if (lit.empty())
            while (ip < n && is_ws(static_cast<unsigned char>(in[ip]))) ip++;
          int k = match(kCallStart, 1);
          if (k == -1) { more = false; break; }
          st = k == 0 ? St::T_OPEN : St::DEAD;
          break;
        }
        case St::DEAD:
          ip = n;
          more = false;
          break;
      }
    }
    if (eof) {
      // What llama.cpp's LENIENT final parse keeps of an unfinished JSON value: the valid prefix for JSON-only
      // types, nothing for mixed types (the atomic alternative is still pending, qwen3-coder.cpp:135).
      if (st == St::A_JSON && !union_mode) emit_args(std::string_view(raw.data(), js.kept(raw.size())));
      if (st == St::A_JSON_CLOSE && !union_mode) emit_args(std::string_view(raw.data(), json_len));
      st = St::DEAD;
      in.clear();
      ip = 0;
      return;
    }
    in.erase(0, ip);
    ip = 0;
  }

  bool shown(int token) {
    if (!tok.is_special(token)) return true;
    auto it = special_shown.find(token);
    if (it != special_shown.end()) return it->second;
    // user-defined tokens are matched in raw text, control tokens are not (tokenizer.h)
    std::vector<int> ids = tok.encode(tok.piece(token), false);
    bool user_defined = ids.size() == 1 && ids[0] == token;
    special_shown[token] = user_defined;
    return user_defined;
  }
};

StreamParser::StreamParser(const Tokenizer& tok, bool thinking, const ojson& tools)
    : p_(std::make_unique<Impl>(tok, thinking, tools)) {}

StreamParser::~StreamParser() = default;

ParseDelta StreamParser::push(int token) {
  if (p_->finished || !p_->shown(token)) return {};
  return push_text(p_->tok.piece(token));
}

ParseDelta StreamParser::push_text(std::string_view s) {
  ParseDelta out;
  if (p_->finished || s.empty()) return out;
  p_->d = &out;
  p_->in.append(s.data(), s.size());
  p_->run(false);
  p_->d = nullptr;
  return out;
}

ParseDelta StreamParser::finish() {
  ParseDelta out;
  if (p_->finished) return out;
  p_->d = &out;
  p_->run(true);
  p_->d = nullptr;
  p_->finished = true;
  return out;
}

const std::string& StreamParser::reasoning() const { return p_->reasoning; }
const std::string& StreamParser::content() const { return p_->content; }

ojson StreamParser::tool_calls() const {
  ojson out = ojson::array();
  for (const auto& c : p_->calls)
    out.push_back({{"id", c.id}, {"type", "function"}, {"function", {{"name", c.name}, {"arguments", c.args}}}});
  return out;
}

int StreamParser::complete_tool_calls() const {
  int n = 0;
  for (const auto& c : p_->calls) n += c.complete ? 1 : 0;
  return n;
}

}  // namespace q27
