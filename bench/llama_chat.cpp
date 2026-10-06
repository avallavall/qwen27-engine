// Chat oracle: checks logged requests/results of our server (tools/q27_server.cpp --log-dir DIR) against
// llama.cpp's own chat code (production llama-common.dll in qwen38_27\bin-parches, used read-only).
//
// Usage: llama_chat <log dir> [--gguf PATH] [--show N]
// Build: bench\build-llama-chat.bat. Run with bin-parches on PATH. Only GGUF metadata is read (no model, no GPU).
//
// For each DIR/req-NNNNN.json ({"request": body, "prompt": our rendered prompt}) the tool builds the template
// inputs from the body as llama-server does (tools/server/server-common.cpp, oaicompat_chat_params_parse,
// lines 1224-1386, with the server's defaults: jinja on, prefill_assistant on, reasoning_format deepseek,
// chat_template_kwargs {"reasoning_effort":"medium"} as in qwen38_27\arranca.ps1), calls
// common_chat_templates_apply and compares the prompt byte for byte.
// With DIR/res-NNNNN.json ({"raw", "reasoning_content", "content", "tool_calls", "finish_reason", ...}) it builds
// the parser params as the server does (server-schema.cpp:295-335 and 539-558) from the apply result, calls
// common_chat_parse(raw, false, params) and compares reasoning_content, content, each tool call's name and
// arguments (as JSON values and as exact text; ids are not compared), and finish_reason (stop / tool_calls).
// A null or missing reasoning_content / content counts as "" (llama.cpp sends "" and leaves out an empty
// reasoning_content).
//
// --show N prints llama.cpp's prompt and parse for pair N.
// Exit code: 0 if everything passed, 1 on any FAIL, 2 on a usage or setup error.
#include "chat.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);

using json = common_json;

namespace {

const char* DEFAULT_GGUF = getenv("Q27_MODEL") ? getenv("Q27_MODEL")
    : "..\\qwen38_27\\models\\Qwen3.8-27B\\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf";

// The production server runs with --chat-template-kwargs {"reasoning_effort":"medium"}
// and the defaults of common/common.h:642-663: use_jinja = true, reasoning_format = deepseek,
// enable_reasoning = -1 (auto), prefill_assistant = true, force_pure_content_parser = false.
const char* MEDIA_MARKER = "<__media__>";                                   // server: random per process
const char* MEDIA_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>";  // what mtmd puts there (mtmd.cpp:700)

std::string read_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.u8string());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// server-common.h:43 json_value(): missing or null -> default, wrong type -> default.
template <typename T>
T jv(const json& body, const std::string& key, const T& def) {
  if (body.is_object() && body.contains(key) && !body.at(key).is_null()) {
    try {
      return body.at(key).get<T>();
    } catch (const std::exception&) {
      return def;
    }
  }
  return def;
}

std::string jstr(const std::string& s) { return json::make(common_json_value(s)).dump_safe(); }

// Short printable form of a byte range: control bytes escaped, UTF-8 kept.
std::string esc(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\\': o += "\\\\"; break;
      case '"': o += "\\\""; break;
      default:
        if (c < 0x20 || c == 0x7F) {
          char b[8];
          snprintf(b, sizeof(b), "\\x%02X", c);
          o += b;
        } else {
          o += (char)c;
        }
    }
  }
  return o;
}

// An exception text on one line (jinja errors span several lines).
std::string one_line(const std::string& s) {
  std::string o;
  auto ends_sep = [&] { return o.size() >= 3 && o.compare(o.size() - 3, 3, " | ") == 0; };
  for (char c : s) {
    if (c == '\n' || c == '\r') {
      if (!o.empty() && !ends_sep()) o += " | ";
    } else {
      o += c;
    }
  }
  if (ends_sep()) o.resize(o.size() - 3);
  return o;
}

size_t first_diff(const std::string& a, const std::string& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i]) return i;
  return a.size() == b.size() ? std::string::npos : n;
}

// "...before<<HERE>>after..." around byte off, cut on UTF-8 boundaries.
std::string context(const std::string& s, size_t off, size_t before = 50, size_t after = 50) {
  size_t a = off > before ? off - before : 0;
  while (a > 0 && ((unsigned char)s[a] & 0xC0) == 0x80) a--;
  size_t e = std::min(s.size(), off + after);
  while (e < s.size() && ((unsigned char)s[e] & 0xC0) == 0x80) e++;
  const size_t o = std::min(off, s.size());
  return std::string(a > 0 ? "..." : "") + "\"" + esc(s.substr(a, o - a)) + "<<HERE>>" + esc(s.substr(o, e - o)) +
         "\"" + (e < s.size() ? "..." : "");
}

// JSON value equality: objects ignore key order, numbers compare by value.
bool json_equal(const json& a, const json& b) {
  if (a.is_null() || b.is_null()) return a.is_null() && b.is_null();
  if (a.is_boolean() || b.is_boolean()) return a.is_boolean() && b.is_boolean() && a.get<bool>() == b.get<bool>();
  if (a.is_number() || b.is_number()) {
    if (!a.is_number() || !b.is_number()) return false;
    if (a.is_number_integer() && b.is_number_integer()) return a.dump() == b.dump();
    return a.get<double>() == b.get<double>();
  }
  if (a.is_string() || b.is_string())
    return a.is_string() && b.is_string() && a.get<std::string>() == b.get<std::string>();
  if (a.is_array() || b.is_array()) {
    if (!a.is_array() || !b.is_array() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
      if (!json_equal(a.at(i), b.at(i))) return false;
    return true;
  }
  if (a.is_object() && b.is_object()) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a.items()) {
      if (!b.contains(kv.key()) || !json_equal(kv.value(), b.at(kv.key()))) return false;
    }
    return true;
  }
  return false;
}

// Everything the oracle derives from one request.
struct Built {
  common_chat_templates_inputs inputs;
  bool stream = false;
  bool has_media = false;
  bool parse_tool_calls = true;  // common_chat_parser_params default
  common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
  bool is_continuation = false;
  bool echo = false;
};

// Copy of oaicompat_chat_params_parse (tools/server/server-common.cpp:1224-1386), the parts that reach the
// template inputs and the parser params. Throws where llama-server rejects the request.
Built build_inputs(json body, const common_chat_templates* tmpls, bool server_enable_thinking,
                   const std::map<std::string, std::string>& default_kwargs) {
  Built r;
  auto tools = jv(body, "tools", json());
  r.stream = jv(body, "stream", false);
  auto tool_choice = jv(body, "tool_choice", std::string("auto"));

  auto json_schema = jv(body, "json_schema", json());
  auto grammar = jv(body, "grammar", std::string());
  if (!json_schema.is_null() && !grammar.empty()) throw std::runtime_error("Cannot use both json_schema and grammar");

  if (body.contains("response_format")) {  // server-common.cpp:1259
    json response_format = jv(body, "response_format", json::object());
    std::string response_type = jv(response_format, "type", std::string());
    if (response_type == "json_object") {
      if (response_format.contains("schema") || json_schema.empty())
        json_schema = jv(response_format, "schema", json::object());
    } else if (response_type == "json_schema") {
      auto schema_wrapper = jv(response_format, "json_schema", json::object());
      json_schema = jv(schema_wrapper, "schema", json::object());
    } else if (!response_type.empty() && response_type != "text") {
      throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " +
                                  response_type);
    }
  }
  if (json_schema.is_object() && json_schema.empty()) json_schema["type"] = "object";

  if (!body.contains("messages")) throw std::invalid_argument("'messages' is required");
  json& messages = body.at("messages");
  if (!messages.is_array()) throw std::invalid_argument("Expected 'messages' to be an array");
  for (size_t i = 0; i < messages.size(); i++) {  // server-common.cpp:1287
    json& msg = messages.at(i);
    std::string role = jv(msg, "role", std::string());
    if (role != "assistant" && !msg.contains("content"))
      throw std::invalid_argument("All non-assistant messages must contain 'content'");
    if (role == "assistant") {
      if (!msg.contains("content") && !msg.contains("tool_calls"))
        throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
      if (!msg.contains("content")) continue;
    }
    json& content = msg.at("content");
    if (content.is_string() || content.is_null()) continue;
    if (!content.is_array()) throw std::invalid_argument("Expected 'content' to be a string or an array");
    // oaicompat_content_load_media (server-common.cpp:1151): media parts become media_marker text parts.
    for (size_t k = 0; k < content.size(); k++) {
      json& p = content.at(k);
      std::string type = jv(p, "type", std::string());
      if (type == "image_url") {
        p["type"] = "media_marker";
        p["text"] = MEDIA_MARKER;
        p.erase("image_url");
        r.has_media = true;
      } else if (type == "input_audio" || type == "input_video" || type == "video_url") {
        throw std::runtime_error("oracle: " + type + " parts are not supported");
      } else if (type != "text") {
        throw std::invalid_argument("unsupported content[].type");
      }
    }
  }

  auto caps = common_chat_templates_get_caps(tmpls);
  common_chat_templates_inputs& in = r.inputs;
  in.messages = common_chat_msgs_parse_oaicompat(messages);
  in.tools = common_chat_tools_parse_oaicompat(tools);
  in.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
  in.json_schema = json_schema.is_null() ? "" : json_schema.dump();
  in.grammar = grammar;
  in.use_jinja = true;
  in.parallel_tool_calls = jv(body, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
  in.add_generation_prompt = jv(body, "add_generation_prompt", true);
  in.continue_final_message = body.contains("continue_final_message")
                                  ? common_chat_continuation_parse(body.at("continue_final_message"))
                                  : COMMON_CHAT_CONTINUATION_NONE;
  const bool prefill_assistant = true;
  if (in.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && prefill_assistant && !in.messages.empty() &&
      in.messages.back().role == "assistant") {
    if (in.messages.size() >= 2 && in.messages[in.messages.size() - 2].role == "assistant")
      throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
    in.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
    in.add_generation_prompt = false;
  }
  if (in.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && in.add_generation_prompt)
    throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
  if (in.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !in.messages.empty() &&
      in.messages.back().role == "assistant" && !in.messages.back().tool_calls.empty())
    throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
  in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
  if (body.contains("reasoning_format"))
    in.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
  in.enable_thinking = server_enable_thinking;
  if (!in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
    if (body.contains("grammar")) throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
    r.parse_tool_calls = true;  // llama_params["parse_tool_calls"] = true
  } else {
    r.parse_tool_calls = jv(body, "parse_tool_calls", true);  // body fields are copied into llama_params
  }

  // chat_template_kwargs: server default, then the request's keys (server-common.cpp:1356)
  auto kw = jv(body, "chat_template_kwargs", json::object());
  in.chat_template_kwargs = default_kwargs;
  if (kw.is_object())
    for (const auto& item : kw.items()) in.chat_template_kwargs[item.key()] = item.value().dump();

  auto it = in.chat_template_kwargs.find("enable_thinking");  // server-common.cpp:1363
  const std::string et = it == in.chat_template_kwargs.end() ? "" : it->second;
  if (et == "true") in.enable_thinking = true;
  else if (et == "false") in.enable_thinking = false;
  else if (!et.empty() && et[0] == '"')
    throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");

  if (body.contains("reasoning_effort")) {  // server-common.cpp:1373
    auto reasoning_effort = jv(body, "reasoning_effort", std::string(""));
    if (reasoning_effort == "none") {
      in.enable_thinking = false;
      in.chat_template_kwargs.erase("reasoning_effort");
    } else if (!reasoning_effort.empty()) {
      in.chat_template_kwargs["reasoning_effort"] = json::make(common_json_value(reasoning_effort)).dump();
    }
  }
  in.force_pure_content = false;

  // parser params that come from body fields (server-schema.cpp:301-335)
  r.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
  if (body.contains("reasoning_format"))
    r.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
  r.is_continuation = body.contains("continue_final_message") &&
                      common_chat_continuation_parse(body.at("continue_final_message")) != COMMON_CHAT_CONTINUATION_NONE;
  r.echo = jv(body, "echo", false);
  return r;
}

common_chat_parser_params parser_params(const Built& b, const common_chat_params& cp) {
  common_chat_parser_params pp(cp);  // format, generation_prompt
  pp.reasoning_format = b.reasoning_format;
  pp.reasoning_in_content = b.stream && b.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
  pp.parse_tool_calls = b.parse_tool_calls;
  if (!cp.parser.empty()) pp.parser.load(cp.parser);
  pp.is_continuation = b.is_continuation;
  pp.echo = b.echo;
  return pp;
}

const char* choice_name(common_chat_tool_choice c) {
  switch (c) {
    case COMMON_CHAT_TOOL_CHOICE_AUTO: return "auto";
    case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
    case COMMON_CHAT_TOOL_CHOICE_NONE: return "none";
  }
  return "?";
}

std::string str_or_empty(const json& j, const std::string& key) {
  if (!j.contains(key) || j.at(key).is_null()) return "";
  if (j.at(key).is_string()) return j.at(key).get<std::string>();
  return j.at(key).dump();
}

void compare_text(const char* field, const std::string& ours, const std::string& ref,
                  std::vector<std::string>& out) {
  const size_t d = first_diff(ours, ref);
  if (d == std::string::npos) return;
  out.push_back(std::string(field) + ": differs at byte " + std::to_string(d) + " (ours " +
                std::to_string(ours.size()) + " B, llama.cpp " + std::to_string(ref.size()) + " B)");
  out.push_back("    ours:      " + context(ours, d));
  out.push_back("    llama.cpp: " + context(ref, d));
}

struct Totals {
  int pairs = 0, prompt_pass = 0, prompt_fail = 0, parse_pass = 0, parse_fail = 0, no_res = 0, media = 0;
};

}  // namespace

int main(int argc, char** argv) {
  SetConsoleOutputCP(65001);
  std::string dir, gguf_path = DEFAULT_GGUF;
  long show = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
    else if (a == "--show" && i + 1 < argc) show = strtol(argv[++i], nullptr, 10);
    else if (!a.empty() && a[0] != '-' && dir.empty()) dir = a;
    else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      dir.clear();
      break;
    }
  }
  if (dir.empty()) {
    fprintf(stderr,
            "usage: llama_chat <log dir> [--gguf PATH] [--show N]\n"
            "  checks DIR/req-NNNNN.json (prompt) and DIR/res-NNNNN.json (parse) against llama.cpp\n"
            "  needs the llama.cpp DLL folder (Q27_LLAMA_BIN) on PATH\n");
    return 2;
  }

  // chat template from the GGUF metadata (no tensors are read)
  std::string tmpl_src;
  {
    gguf_init_params gp = {/* no_alloc */ true, /* ctx */ nullptr};
    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) {
      fprintf(stderr, "cannot read GGUF %s\n", gguf_path.c_str());
      return 2;
    }
    const int64_t k = gguf_find_key(g, "tokenizer.chat_template");
    if (k < 0) {
      fprintf(stderr, "no tokenizer.chat_template in %s\n", gguf_path.c_str());
      return 2;
    }
    tmpl_src = gguf_get_val_str(g, k);
    if (gguf_find_key(g, "tokenizer.chat_template.tool_use") >= 0)
      fprintf(stderr, "warning: the GGUF also has tokenizer.chat_template.tool_use; the server would use it with "
                      "tools, this oracle does not\n");
    gguf_free(g);
  }

  common_chat_templates_ptr tmpls;
  bool server_enable_thinking = false;
  try {
    tmpls = common_chat_templates_init(nullptr, tmpl_src);
    // server-context.cpp:1518: enable_reasoning = -1 (auto) -> thinking on if the template supports it
    server_enable_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
  } catch (const std::exception& e) {
    fprintf(stderr, "llama.cpp cannot load the chat template: %s\n", e.what());
    return 2;
  }
  const std::map<std::string, std::string> default_kwargs = {{"reasoning_effort", "\"medium\""}};

  // pairs, by number
  std::map<long, std::string> reqs;
  try {
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      const std::string n = e.path().filename().u8string();
      if (n.size() > 9 && n.compare(0, 4, "req-") == 0 && n.compare(n.size() - 5, 5, ".json") == 0) {
        const std::string num = n.substr(4, n.size() - 9);
        if (num.find_first_not_of("0123456789") == std::string::npos) reqs[strtol(num.c_str(), nullptr, 10)] = num;
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "cannot list %s: %s\n", dir.c_str(), e.what());
    return 2;
  }
  if (reqs.empty()) {
    fprintf(stderr, "no req-NNNNN.json files in %s\n", dir.c_str());
    return 2;
  }
  if (show >= 0 && !reqs.count(show)) {
    fprintf(stderr, "no pair %ld in %s\n", show, dir.c_str());
    return 2;
  }

  Totals t;
  for (const auto& [n, num] : reqs) {
    if (show >= 0 && n != show) continue;
    t.pairs++;
    const std::filesystem::path dp = std::filesystem::u8path(dir);
    const auto req_path = dp / ("req-" + num + ".json");
    const auto res_path = dp / ("res-" + num + ".json");
    std::vector<std::string> notes;
    std::string prompt_status, parse_status;
    bool prompt_fail = false, parse_fail = false;

    json req, res;
    bool have_res = std::filesystem::exists(res_path);
    try {
      req = json::parse(read_file(req_path));
      if (have_res) res = json::parse(read_file(res_path));
    } catch (const std::exception& e) {
      printf("FAIL %s  cannot read the log files: %s\n", num.c_str(), e.what());
      t.prompt_fail++;
      continue;
    }

    Built b;
    common_chat_params cp;
    bool applied = false;
    try {
      b = build_inputs(req.contains("request") ? req.at("request") : json::object(), tmpls.get(),
                       server_enable_thinking, default_kwargs);
      cp = common_chat_templates_apply(tmpls.get(), b.inputs);
      applied = true;
    } catch (const std::exception& e) {
      prompt_status = "prompt FAIL";
      prompt_fail = true;
      notes.push_back("llama.cpp rejects the request: " + one_line(e.what()));
    }

    std::string ref_prompt;
    if (applied) {
      ref_prompt = cp.prompt;
      if (b.has_media) {  // compare with the image tokens mtmd puts where the marker is
        t.media++;
        for (size_t p = 0; (p = ref_prompt.find(MEDIA_MARKER, p)) != std::string::npos;) {
          ref_prompt.replace(p, strlen(MEDIA_MARKER), MEDIA_TOKENS);
          p += strlen(MEDIA_TOKENS);
        }
      }
      const std::string ours = str_or_empty(req, "prompt");
      std::vector<std::string> d;
      compare_text("prompt", ours, ref_prompt, d);
      if (d.empty()) {
        prompt_status = "prompt ok (" + std::to_string(ours.size()) + " B" + (b.has_media ? ", media" : "") + ")";
      } else {
        prompt_status = "prompt FAIL";
        prompt_fail = true;
        notes.insert(notes.end(), d.begin(), d.end());
      }
    }

    common_chat_msg msg;
    bool parsed = false, empty_parse = false;
    if (!have_res) {
      parse_status = "no res file";
      t.no_res++;
    } else if (!applied) {
      parse_status = "parse not checked";
      parse_fail = true;
    } else {
      const std::string raw = str_or_empty(res, "raw");
      try {
        const common_chat_parser_params pp = parser_params(b, cp);
        msg = common_chat_parse(raw, false, pp);
        if (msg.empty()) {  // server-task.cpp:417: an empty parse sends the generated text as content
          msg.role = "assistant";
          msg.content = b.stream ? "" : raw;
          empty_parse = true;
        }
        parsed = true;
      } catch (const std::exception& e) {
        parse_status = "parse FAIL";
        parse_fail = true;
        notes.push_back("llama.cpp parse throws: " + one_line(e.what()));
      }
      if (parsed) {
        std::vector<std::string> d;
        compare_text("reasoning_content", str_or_empty(res, "reasoning_content"), msg.reasoning_content, d);
        compare_text("content", str_or_empty(res, "content"), msg.content, d);
        json calls = res.contains("tool_calls") && res.at("tool_calls").is_array() ? res.at("tool_calls")
                                                                                     : json::array();
        if (calls.size() != msg.tool_calls.size())
          d.push_back("tool_calls: ours " + std::to_string(calls.size()) + ", llama.cpp " +
                      std::to_string(msg.tool_calls.size()));
        for (size_t i = 0; i < std::min(calls.size(), msg.tool_calls.size()); i++) {
          const auto& ref = msg.tool_calls[i];
          const json& c = calls.at(i);
          const json fn = c.contains("function") ? c.at("function") : json::object();
          const std::string tag = "tool_calls[" + std::to_string(i) + "]";
          const std::string name = str_or_empty(fn, "name");
          if (name != ref.name) d.push_back(tag + ".name: ours " + jstr(name) + ", llama.cpp " + jstr(ref.name));
          std::string args;
          if (fn.contains("arguments") && fn.at("arguments").is_string()) {
            args = fn.at("arguments").get<std::string>();
          } else {
            args = fn.contains("arguments") ? fn.at("arguments").dump() : "";
            d.push_back(tag + ".arguments: ours is not a JSON string");
          }
          const json va = json::parse_no_throw(args), vb = json::parse_no_throw(ref.arguments);
          const bool same_value = !va.is_discarded() && !vb.is_discarded() && json_equal(va, vb);
          if (!same_value) {
            d.push_back(tag + ".arguments: different JSON values" +
                        (va.is_discarded() ? " (ours is not valid JSON)" : "") +
                        (vb.is_discarded() ? " (llama.cpp's is not valid JSON)" : ""));
            compare_text((tag + ".arguments text").c_str(), args, ref.arguments, d);
          } else if (args != ref.arguments) {
            std::vector<std::string> d2;
            compare_text((tag + ".arguments text").c_str(), args, ref.arguments, d2);
            d2[0] = tag + ".arguments: same JSON value, " + d2[0].substr(d2[0].find(':') + 2);
            d.insert(d.end(), d2.begin(), d2.end());
          }
        }
        const std::string fr = str_or_empty(res, "finish_reason");
        if (fr == "stop" || fr == "tool_calls") {
          const std::string want = msg.tool_calls.empty() ? "stop" : "tool_calls";
          if (fr != want) d.push_back("finish_reason: ours \"" + fr + "\", llama.cpp \"" + want + "\"");
        }
        if (d.empty()) {
          parse_status = "parse ok (" + std::to_string(msg.tool_calls.size()) + " tool calls" +
                         (empty_parse ? ", empty parse: raw text as content" : "") + ")";
        } else {
          parse_status = "parse FAIL";
          parse_fail = true;
          notes.insert(notes.end(), d.begin(), d.end());
        }
      }
    }

    if (prompt_fail) t.prompt_fail++; else t.prompt_pass++;
    if (have_res) { if (parse_fail) t.parse_fail++; else t.parse_pass++; }
    printf("%s %s  %s  %s\n", prompt_fail || parse_fail ? "FAIL" : "PASS", num.c_str(), prompt_status.c_str(),
           parse_status.c_str());
    for (const auto& s : notes) printf("       %s\n", s.c_str());

    if (show >= 0 && applied) {
      const auto& in = b.inputs;
      printf("\n=== pair %s: llama.cpp ===\n", num.c_str());
      printf("format: %s   generation_prompt: %s\n", common_chat_format_name(cp.format),
             jstr(cp.generation_prompt).c_str());
      std::string kws;
      for (const auto& [k, v] : in.chat_template_kwargs) kws += (kws.empty() ? "" : ", ") + k + "=" + v;
      printf("enable_thinking: %d   add_generation_prompt: %d   continue_final_message: %d   tool_choice: %s   "
             "parallel_tool_calls: %d   tools: %zu   kwargs: {%s}\n",
             (int)in.enable_thinking, (int)in.add_generation_prompt, (int)in.continue_final_message,
             choice_name(in.tool_choice), (int)in.parallel_tool_calls, in.tools.size(), kws.c_str());
      printf("reasoning_format: %s   stream: %d   parse_tool_calls: %d\n",
             common_reasoning_format_name(b.reasoning_format), (int)b.stream, (int)b.parse_tool_calls);
      printf("----- prompt (%zu bytes) -----\n", cp.prompt.size());
      fwrite(cp.prompt.data(), 1, cp.prompt.size(), stdout);
      printf("\n----- end of prompt -----\n");
      if (parsed) {
        printf("----- parse -----\n");
        if (empty_parse)
          printf("(the parse is empty: llama-server sends the generated text as content%s)\n",
                 b.stream ? ", or nothing when streaming" : "");
        printf("reasoning_content: %s\n", jstr(msg.reasoning_content).c_str());
        printf("content: %s\n", jstr(msg.content).c_str());
        for (size_t i = 0; i < msg.tool_calls.size(); i++)
          printf("tool_calls[%zu]: name %s\n  arguments: %s\n", i, jstr(msg.tool_calls[i].name).c_str(),
                 msg.tool_calls[i].arguments.c_str());
        printf("----- end of parse -----\n");
      }
    }
  }

  printf("\npairs: %d   prompt: %d pass, %d fail   parse: %d pass, %d fail, %d without res file", t.pairs,
         t.prompt_pass, t.prompt_fail, t.parse_pass, t.parse_fail, t.no_res);
  if (t.media) printf("   (%d with images)", t.media);
  printf("\n");
  return t.prompt_fail || t.parse_fail ? 1 : 0;
}
