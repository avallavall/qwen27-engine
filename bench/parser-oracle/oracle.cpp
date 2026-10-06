// Oracle: llama.cpp's chat output parser (production llama-common.dll) on a list of cases.
// Usage: oracle <template.jinja> <cases.json> <out.json>
// cases.json: [{"name", "tools": [...] | null, "thinking": bool, "text": str, "parallel"?: bool, "partial"?: bool}]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "chat.h"
#include "nlohmann/json.hpp"

using njson = nlohmann::ordered_json;

static std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: oracle <template> <cases.json> <out.json>\n");
    return 2;
  }
  std::string tmpl_src = read_file(argv[1]);
  njson cases = njson::parse(read_file(argv[2]));
  auto tmpls = common_chat_templates_init(nullptr, tmpl_src, "", "<|im_end|>");
  auto caps = common_chat_templates_get_caps(tmpls.get());
  njson out = njson::array();
  for (auto& c : cases) {
    njson r;
    r["name"] = c["name"];
    try {
      common_chat_templates_inputs in;
      common_chat_msg user;
      user.role = "user";
      user.content = "Hello";
      in.messages = {user};
      if (c.contains("tools") && c["tools"].is_array()) {
        for (auto& t : c["tools"]) {
          common_chat_tool ct;
          ct.name = t["function"]["name"].get<std::string>();
          ct.description = t["function"].value("description", "");
          ct.parameters = t["function"].contains("parameters") ? t["function"]["parameters"].dump() : "{}";
          in.tools.push_back(ct);
        }
      }
      in.enable_thinking = c.value("thinking", true);
      in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;  // server default (common.h)
      in.parallel_tool_calls = c.contains("parallel") ? c["parallel"].get<bool>() : caps["supports_parallel_tool_calls"];
      in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
      auto params = common_chat_templates_apply(tmpls.get(), in);
      common_chat_parser_params pp(params);
      pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
      pp.parser.load(params.parser);
      r["gen_prompt"] = params.generation_prompt;
      r["parallel"] = in.parallel_tool_calls;
      bool partial = c.value("partial", false);
      if (c.contains("text_hex")) {
        std::string hex = c["text_hex"].get<std::string>(), raw;
        for (size_t i = 0; i + 1 < hex.size(); i += 2) raw += (char)std::stoi(hex.substr(i, 2), nullptr, 16);
        auto msg = common_chat_parse(raw, partial, pp);
        auto hexs = [](const std::string& s) {
          static const char* d = "0123456789abcdef";
          std::string o;
          for (unsigned char ch : s) { o += d[ch >> 4]; o += d[ch & 15]; }
          return o;
        };
        r["reasoning_hex"] = hexs(msg.reasoning_content);
        r["content_hex"] = hexs(msg.content);
        njson tcs = njson::array();
        for (auto& tc : msg.tool_calls) tcs.push_back({{"name", tc.name}, {"arguments_hex", hexs(tc.arguments)}});
        r["tool_calls"] = tcs;
        r["ok"] = true;
        out.push_back(r);
        continue;
      }
      auto msg = common_chat_parse(c["text"].get<std::string>(), partial, pp);
      r["reasoning"] = msg.reasoning_content;
      r["content"] = msg.content;
      njson tcs = njson::array();
      for (auto& tc : msg.tool_calls) tcs.push_back({{"name", tc.name}, {"arguments", tc.arguments}, {"id", tc.id}});
      r["tool_calls"] = tcs;
      // Streaming as the server does it: parse every byte prefix (is_partial) and check monotonic growth.
      if (c.value("stream", false)) {
        std::string text = c["text"].get<std::string>();
        common_chat_msg prev;
        std::string acc_r, acc_c;
        std::vector<std::string> acc_args;
        for (size_t i = 1; i <= text.size(); ++i) {
          auto m = common_chat_parse(text.substr(0, i), i < text.size(), pp);
          auto diffs = common_chat_msg_diff::compute_diffs(prev, m);
          for (auto& d : diffs) {
            acc_r += d.reasoning_content_delta;
            acc_c += d.content_delta;
            if (d.tool_call_index != std::string::npos) {
              while (acc_args.size() <= d.tool_call_index) acc_args.push_back("");
              acc_args[d.tool_call_index] += d.tool_call_delta.arguments;
            }
          }
          prev = m;
        }
        r["stream_ok"] = acc_r == msg.reasoning_content && acc_c == msg.content;
      }
      r["ok"] = true;
    } catch (const std::exception& e) {
      r["ok"] = false;
      r["error"] = e.what();
    }
    out.push_back(r);
  }
  std::ofstream f(argv[3], std::ios::binary);
  f << out.dump(1, ' ', false) << "\n";
  printf("caps.supports_parallel_tool_calls=%d, %zu cases\n", (int)caps["supports_parallel_tool_calls"], out.size());
  return 0;
}
