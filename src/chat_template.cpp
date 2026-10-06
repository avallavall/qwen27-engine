// The model's chat template (research/_chat_template.jinja) written out in C++. Each block below follows the
// template; comments give the template line numbers. The Python / jinja2 behaviour that the output depends on
// is copied here too: str.strip() (Unicode whitespace), json.dumps(ensure_ascii=False), str() of values in
// {{ }}, Python truthiness, and the `in` test on content parts.
// Tested byte by byte against jinja2 renders: tools/gen_template_fixtures.py + tools/test_template.cpp.
#include "chat_template.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace q27 {

namespace {

// ---- template text

const char* kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const char* kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";
const char* kToolsHead = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const char* kToolsTail =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> "
    "XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function "
    "call in natural language BEFORE the function call, but NOT after\n- If there is no function call "
    "available, answer the question like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

// ---- UTF-8

// Decode the code point at s[i] and move i past it. A byte that does not start a valid sequence decodes as
// 0xFFFFFFFF (never whitespace) and moves i by one.
uint32_t utf8_next(const std::string& s, size_t& i) {
  const unsigned char c = (unsigned char)s[i];
  int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
  if (len == 0 || i + len > s.size()) { ++i; return 0xFFFFFFFFu; }
  uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
  for (int k = 1; k < len; ++k) {
    const unsigned char d = (unsigned char)s[i + k];
    if ((d >> 6) != 2) { ++i; return 0xFFFFFFFFu; }
    cp = (cp << 6) | (d & 0x3F);
  }
  i += len;
  return cp;
}

// Python str.isspace() (Py_UNICODE_ISSPACE), the set str.strip() removes.
bool py_isspace(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}

// jinja2 |trim = Python str.strip().
std::string py_strip(const std::string& s) {
  size_t b = 0;
  while (b < s.size()) {
    size_t j = b;
    if (!py_isspace(utf8_next(s, j))) break;
    b = j;
  }
  size_t e = s.size();
  while (e > b) {
    size_t start = e - 1;  // step back over continuation bytes to the start of the last code point
    while (start > b && ((unsigned char)s[start] >> 6) == 2 && e - start < 4) --start;
    size_t j = start;
    const uint32_t cp = utf8_next(s, j);
    if (j != e || !py_isspace(cp)) break;
    e = start;
  }
  return s.substr(b, e - b);
}

bool starts_with(const std::string& s, const char* p) { return s.compare(0, strlen(p), p) == 0; }
bool ends_with(const std::string& s, const char* p) {
  const size_t n = strlen(p);
  return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

// ---- Python formatting of JSON values

// Python repr(float): shortest round-trip digits; exponent form when the decimal point position is
// <= -4 or > 16 (pystrtod.c, mode 'r'). json.dumps writes NaN / Infinity, str() writes nan / inf.
std::string py_float_repr(double d, bool for_json) {
  if (std::isnan(d)) return for_json ? "NaN" : "nan";
  if (std::isinf(d)) return d > 0 ? (for_json ? "Infinity" : "inf") : (for_json ? "-Infinity" : "-inf");
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  const std::string s(buf, res.ptr);  // [-]D[.DDD]e(+|-)XX
  std::string out;
  size_t p = 0;
  if (s[0] == '-') { out += '-'; p = 1; }
  const size_t epos = s.find('e');
  std::string digits;
  for (size_t k = p; k < epos; ++k)
    if (s[k] != '.') digits += s[k];
  const int exp10 = std::atoi(s.c_str() + epos + 1);
  const int decpt = exp10 + 1;  // value = 0.DIGITS * 10^decpt
  const int nd = (int)digits.size();
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (nd > 1) { out += '.'; out.append(digits, 1, std::string::npos); }
    char eb[16];
    snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    out += eb;
  } else if (decpt <= 0) {
    out += "0.";
    out.append((size_t)-decpt, '0');
    out += digits;
  } else if (decpt >= nd) {
    out += digits;
    out.append((size_t)(decpt - nd), '0');
    out += ".0";
  } else {
    out.append(digits, 0, (size_t)decpt);
    out += '.';
    out.append(digits, (size_t)decpt, std::string::npos);
  }
  return out;
}

// json.dumps string escaping with ensure_ascii=False: \" \\ \b \f \n \r \t, other controls as \u00xx,
// everything else (including non-ASCII, DEL and '/') raw.
void json_dump_string(const std::string& s, std::string& out) {
  out += '"';
  for (const char ch : s) {
    const unsigned char c = (unsigned char)ch;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

// json.dumps(v, ensure_ascii=False): separators ", " and ": ", keys in the given order.
void json_dumps(const ojson& v, std::string& out) {
  switch (v.type()) {
    case ojson::value_t::null: out += "null"; break;
    case ojson::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
    case ojson::value_t::number_integer: out += std::to_string(v.get<int64_t>()); break;
    case ojson::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
    case ojson::value_t::number_float: out += py_float_repr(v.get<double>(), true); break;
    case ojson::value_t::string: json_dump_string(v.get_ref<const std::string&>(), out); break;
    case ojson::value_t::array: {
      out += '[';
      bool first = true;
      for (const auto& e : v) {
        if (!first) out += ", ";
        first = false;
        json_dumps(e, out);
      }
      out += ']';
      break;
    }
    case ojson::value_t::object: {
      out += '{';
      bool first = true;
      for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) out += ", ";
        first = false;
        json_dump_string(it.key(), out);
        out += ": ";
        json_dumps(it.value(), out);
      }
      out += '}';
      break;
    }
    default: throw TemplateError("tojson: value type not supported");
  }
}

// Python str.isprintable() for one code point, approximate outside ASCII and Latin-1: it knows the
// whitespace, format and private-use blocks but not unassigned code points. Only used by py_repr.
bool py_isprintable(uint32_t cp) {
  if (cp < 0x20 || cp == 0x7F) return false;
  if (cp < 0x7F) return true;
  if (cp <= 0xA0 || cp == 0xAD) return false;
  if (py_isspace(cp)) return false;
  if (cp == 0x061C || cp == 0x180E || (cp >= 0x200B && cp <= 0x200F) || (cp >= 0x202A && cp <= 0x202E) ||
      (cp >= 0x2060 && cp <= 0x206F) || cp == 0xFEFF || (cp >= 0xFFF9 && cp <= 0xFFFB) || cp == 0xFFFE ||
      cp == 0xFFFF || (cp >= 0xD800 && cp <= 0xF8FF) || (cp >= 0xE0000 && cp <= 0xE007F) || cp >= 0xF0000)
    return false;
  return true;
}

// Python repr(str).
std::string py_repr_str(const std::string& s) {
  const bool has_sq = s.find('\'') != std::string::npos, has_dq = s.find('"') != std::string::npos;
  const char quote = (has_sq && !has_dq) ? '"' : '\'';
  std::string out(1, quote);
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t cp = utf8_next(s, i);
    char b[16];
    if (cp == (uint32_t)quote || cp == '\\') { out += '\\'; out += (char)cp; }
    else if (cp == '\t') out += "\\t";
    else if (cp == '\n') out += "\\n";
    else if (cp == '\r') out += "\\r";
    else if (cp == 0xFFFFFFFFu) { snprintf(b, sizeof b, "\\x%02x", (unsigned char)s[start]); out += b; }
    else if (py_isprintable(cp)) out.append(s, start, i - start);
    else {
      if (cp <= 0xFF) snprintf(b, sizeof b, "\\x%02x", cp);
      else if (cp <= 0xFFFF) snprintf(b, sizeof b, "\\u%04x", cp);
      else snprintf(b, sizeof b, "\\U%08x", cp);
      out += b;
    }
  }
  out += quote;
  return out;
}

std::string py_str(const ojson& v);

// Python repr() of a JSON value (list and dict items are shown with repr).
std::string py_repr(const ojson& v) {
  if (v.is_string()) return py_repr_str(v.get_ref<const std::string&>());
  if (v.is_array()) {
    std::string out = "[";
    bool first = true;
    for (const auto& e : v) {
      if (!first) out += ", ";
      first = false;
      out += py_repr(e);
    }
    return out + "]";
  }
  if (v.is_object()) {
    std::string out = "{";
    bool first = true;
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      out += py_repr_str(it.key()) + ": " + py_repr(it.value());
    }
    return out + "}";
  }
  return py_str(v);
}

// Python str() of a JSON value, which is what {{ x }} prints.
std::string py_str(const ojson& v) {
  switch (v.type()) {
    case ojson::value_t::null: return "None";
    case ojson::value_t::boolean: return v.get<bool>() ? "True" : "False";
    case ojson::value_t::number_integer: return std::to_string(v.get<int64_t>());
    case ojson::value_t::number_unsigned: return std::to_string(v.get<uint64_t>());
    case ojson::value_t::number_float: return py_float_repr(v.get<double>(), false);
    case ojson::value_t::string: return v.get<std::string>();
    default: return py_repr(v);
  }
}

// ---- jinja2 value access

// obj.key: the value if obj is a mapping with that key, else nullptr (undefined).
const ojson* field(const ojson& obj, const char* key) {
  if (!obj.is_object()) return nullptr;
  const auto it = obj.find(key);
  return it == obj.end() ? nullptr : &*it;
}

bool is_str(const ojson* v, const char* s) { return v && v->is_string() && v->get_ref<const std::string&>() == s; }

// Python truthiness.
bool truthy(const ojson& v) {
  switch (v.type()) {
    case ojson::value_t::null: return false;
    case ojson::value_t::boolean: return v.get<bool>();
    case ojson::value_t::number_integer: return v.get<int64_t>() != 0;
    case ojson::value_t::number_unsigned: return v.get<uint64_t>() != 0;
    case ojson::value_t::number_float: return v.get<double>() != 0.0;
    case ojson::value_t::string: return !v.get_ref<const std::string&>().empty();
    case ojson::value_t::array:
    case ojson::value_t::object: return !v.empty();
    default: return true;
  }
}

// Python `needle in item` for a content part: key test on a mapping, substring test on a string, element
// test on a list. Numbers, bools and null raise TypeError in jinja2.
bool py_in(const char* needle, const ojson& item) {
  if (item.is_object()) return item.contains(needle);
  if (item.is_string()) return item.get_ref<const std::string&>().find(needle) != std::string::npos;
  if (item.is_array()) {
    for (const auto& e : item)
      if (e.is_string() && e.get_ref<const std::string&>() == needle) return true;
    return false;
  }
  throw TemplateError(std::string("content part of type ") + item.type_name() + " is not iterable");
}

struct Renderer {
  const TemplateOptions& opt;
  int image_count = 0, video_count = 0;  // namespace(value=0), lines 1-2

  // Macro render_content, lines 3-41.
  std::string render_content(const ojson* content, bool do_vision_count, bool is_system_content) {
    if (!content || content->is_null()) return "";
    if (content->is_string()) return content->get<std::string>();
    if (!content->is_array()) throw TemplateError("Unexpected content type.");
    std::string out;
    for (const auto& item : *content) {
      if (py_in("image", item) || py_in("image_url", item) || is_str(field(item, "type"), "image")) {
        if (is_system_content) throw TemplateError("System message cannot contain images.");
        if (do_vision_count) ++image_count;
        if (opt.add_vision_id) out += "Picture " + std::to_string(image_count) + ": ";
        out += "<|vision_start|><|image_pad|><|vision_end|>";
      } else if (py_in("video", item) || is_str(field(item, "type"), "video")) {
        if (is_system_content) throw TemplateError("System message cannot contain videos.");
        if (do_vision_count) ++video_count;
        if (opt.add_vision_id) out += "Video " + std::to_string(video_count) + ": ";
        out += "<|vision_start|><|video_pad|><|vision_end|>";
      } else if (py_in("text", item)) {
        if (const ojson* t = field(item, "text")) out += py_str(*t);  // a list or string part prints ''
      } else {
        throw TemplateError("Unexpected item type in content.");
      }
    }
    return out;
  }
};

// Append the tool calls of one assistant message, lines 121-145.
void render_tool_calls(const ojson& calls, bool has_content, std::string& out) {
  if (calls.is_string()) throw TemplateError("tool call has no name");  // jinja2: iterates characters
  bool first = true;
  for (const auto& call : calls) {
    const ojson* tc = &call;
    if (const ojson* f = field(call, "function")) tc = f;
    const ojson* name = field(*tc, "name");
    if (!name || !name->is_string()) throw TemplateError("tool call name is missing or not a string");
    if (first) out += has_content ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=";
    else out += "\n<tool_call>\n<function=";
    first = false;
    out += name->get_ref<const std::string&>();
    out += ">\n";
    const ojson* args = field(*tc, "arguments");
    if (args && !(args->is_string() && args->get_ref<const std::string&>().empty())) {
      if (!args->is_object()) throw TemplateError("Can only get item pairs from a mapping.");  // |items
      for (auto it = args->begin(); it != args->end(); ++it) {
        out += "<parameter=";
        out += it.key();
        out += ">\n";
        if (it.value().is_string()) out += it.value().get_ref<const std::string&>();
        else json_dumps(it.value(), out);
        out += "\n</parameter>\n";
      }
    }
    out += "</function>\n</tool_call>";
  }
}

}  // namespace

std::string render_chat(const ojson& messages, const ojson& tools, const TemplateOptions& opt) {
  // lines 42-44
  if (!truthy(messages)) throw TemplateError("No messages provided.");
  if (!messages.is_array()) throw TemplateError("messages must be an array");
  Renderer r{opt};
  std::string out;

  // lines 45-56: reasoning effort, only when thinking is on
  std::string reasoning;
  if (!opt.enable_thinking || *opt.enable_thinking) {
    const std::string effort = opt.reasoning_effort.value_or("xhigh");
    if (effort != "xhigh" && effort != "medium" && effort != "low")
      throw TemplateError("Unexpected reasoning effort " + effort +
                          ". Supported types are xhigh (default), medium, and low.");
    if (effort == "xhigh") reasoning = kEffortXhigh;
    else if (effort == "low") reasoning = kEffortLow;
  }

  // lines 57-87: system turn
  const ojson& first = messages[0];
  const bool first_is_system = is_str(field(first, "role"), "system");
  if (truthy(tools) && (tools.is_array() || tools.is_string())) {
    out += "<|im_start|>system\n";
    if (!reasoning.empty()) out += reasoning + "\n\n";
    out += kToolsHead;
    if (tools.is_array()) {
      for (const auto& t : tools) {
        out += '\n';
        json_dumps(t, out);
      }
    } else {  // a string is iterable too: one character per "tool"
      const std::string& s = tools.get_ref<const std::string&>();
      for (size_t i = 0; i < s.size();) {
        const size_t start = i;
        utf8_next(s, i);
        out += '\n';
        json_dump_string(s.substr(start, i - start), out);
      }
    }
    out += "\n</tools>";
    out += kToolsTail;
    if (first_is_system) {
      const std::string c = py_strip(r.render_content(field(first, "content"), false, true));
      if (!c.empty()) out += "\n\n" + c;
    }
    out += "<|im_end|>\n";
  } else if (first_is_system) {
    const std::string c = py_strip(r.render_content(field(first, "content"), false, true));
    if (!c.empty()) out += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n") + c + "<|im_end|>\n";
    else if (!reasoning.empty()) out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
  } else if (!reasoning.empty()) {
    out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
  }

  // lines 88-101: last real user query (a user turn that is not only <tool_response>...</tool_response>)
  const size_t n = messages.size();
  size_t last_query = n - 1;
  bool multi_step_tool = true;
  for (size_t k = n; k-- > 0;) {
    const ojson& m = messages[k];
    if (!is_str(field(m, "role"), "user")) continue;
    const std::string c = py_strip(r.render_content(field(m, "content"), false, false));
    if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
      multi_step_tool = false;
      last_query = k;
      break;
    }
  }
  if (multi_step_tool) throw TemplateError("No user query found in messages.");

  // lines 102-162
  for (size_t i = 0; i < n; ++i) {
    const ojson& m = messages[i];
    const std::string content = py_strip(r.render_content(field(m, "content"), true, false));
    const ojson* role = field(m, "role");
    if (is_str(role, "system")) {
      if (i != 0) throw TemplateError("System message must be at the beginning.");
    } else if (is_str(role, "user")) {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (is_str(role, "assistant")) {
      const ojson* rc = field(m, "reasoning_content");
      const std::string reasoning_content = rc && rc->is_string() ? py_strip(rc->get<std::string>()) : "";
      if (!opt.preserve_thinking || *opt.preserve_thinking || i > last_query)
        out += "<|im_start|>assistant\n<think>\n" + reasoning_content + "\n</think>\n\n" + content;
      else
        out += "<|im_start|>assistant\n" + content;
      const ojson* calls = field(m, "tool_calls");
      if (calls && truthy(*calls) && (calls->is_array() || calls->is_string()))
        render_tool_calls(*calls, !content.empty(), out);
      out += "<|im_end|>\n";
    } else if (is_str(role, "tool")) {
      if (i > 0 && truthy(messages[i - 1]) && !is_str(field(messages[i - 1], "role"), "tool"))
        out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == n || !is_str(field(messages[i + 1], "role"), "tool")) out += "<|im_end|>\n";
    } else {
      throw TemplateError("Unexpected message role.");
    }
  }

  // lines 163-170
  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += opt.enable_thinking && !*opt.enable_thinking ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return out;
}

ojson normalize_messages(const ojson& messages) {
  ojson out = messages;
  if (!out.is_array()) return out;
  for (auto& m : out) {
    if (!m.is_object()) continue;
    const auto role = m.find("role");
    if (role != m.end() && *role == "developer") *role = "system";
    const auto content = m.find("content");
    if (content != m.end() && content->is_null()) *content = "";
    const auto calls = m.find("tool_calls");
    if (calls == m.end() || !calls->is_array()) continue;
    for (auto& tc : *calls) {
      if (!tc.is_object()) continue;
      const auto fn = tc.find("function");
      if (fn == tc.end() || !fn->is_object()) continue;
      const auto args = fn->find("arguments");
      if (args == fn->end() || !args->is_string()) continue;
      // Any valid JSON replaces the string (as llama.cpp); a string that does not parse stays a string,
      // and render_chat then fails on it unless it is "".
      ojson parsed = ojson::parse(args->get_ref<const std::string&>(), nullptr, false);
      if (!parsed.is_discarded()) *args = std::move(parsed);
    }
  }
  return out;
}

// SHA-256 of the 8,952 bytes of tokenizer.chat_template in Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf (LF line ends),
// computed by tools/gen_template_fixtures.py.
const char* chat_template_sha256() { return "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"; }

// ---- SHA-256 (FIPS 180-4)

std::string sha256_hex(const std::string& data) {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](uint32_t x, int c) { return (x >> c) | (x << (32 - c)); };

  // message + 0x80 + zero padding + 64-bit big-endian bit length, in 64-byte blocks
  std::string msg = data;
  const uint64_t bits = (uint64_t)data.size() * 8;
  msg += '\x80';
  while (msg.size() % 64 != 56) msg += (char)0;
  for (int k = 7; k >= 0; --k) msg += (char)((bits >> (8 * k)) & 0xFF);

  for (size_t blk = 0; blk < msg.size(); blk += 64) {
    uint32_t w[64];
    for (int t = 0; t < 16; ++t) {
      const unsigned char* p = (const unsigned char*)msg.data() + blk + 4 * t;
      w[t] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
    }
    for (int t = 16; t < 64; ++t) {
      const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int t = 0; t < 64; ++t) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[t] + w[t];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char hex[65];
  for (int k = 0; k < 8; ++k) snprintf(hex + 8 * k, 9, "%08x", h[k]);
  return std::string(hex, 64);
}

}  // namespace q27
