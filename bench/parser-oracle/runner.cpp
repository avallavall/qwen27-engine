// Runs q27::StreamParser on the oracle's case file (text_hex), fed in random chunks; writes hex results.
// Usage: runner <gguf> <cases.json> <out.json>
#include <fstream>
#include <random>
#include <sstream>

#include "chat_parser.h"
#include "gguf.h"
#include "tokenizer.h"

using namespace q27;

static std::string hexs(const std::string& s) {
  static const char* d = "0123456789abcdef";
  std::string o;
  for (unsigned char ch : s) { o += d[ch >> 4]; o += d[ch & 15]; }
  return o;
}

int main(int argc, char** argv) {
  GGUF g(argv[1]);
  Tokenizer tok(g);
  std::ifstream f(argv[2], std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  ojson cases = ojson::parse(ss.str());
  ojson out = ojson::array();
  unsigned idx = 0;
  for (auto& c : cases) {
    std::string hex = c["text_hex"], text;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) text += (char)std::stoi(hex.substr(i, 2), nullptr, 16);
    StreamParser p(tok, c.value("thinking", true), c["tools"]);
    std::mt19937 rng(++idx);
    std::string r, ct;
    std::vector<std::string> args;
    auto apply = [&](const ParseDelta& d) {
      r += d.reasoning;
      ct += d.content;
      for (auto& t : d.tool_calls) {
        while ((int)args.size() <= t.index) args.push_back("");
        args[t.index] += t.arguments;
      }
    };
    for (size_t i = 0; i < text.size();) {
      size_t n = std::min<size_t>(text.size() - i, 1 + rng() % 9);
      apply(p.push_text(std::string_view(text).substr(i, n)));
      i += n;
    }
    apply(p.finish());
    ojson res;
    res["name"] = c["name"];
    res["ok"] = true;
    res["reasoning_hex"] = hexs(p.reasoning());
    res["content_hex"] = hexs(p.content());
    ojson tcs = ojson::array();
    ojson calls = p.tool_calls();
    bool stream_ok = r == p.reasoning() && ct == p.content() && args.size() == calls.size();
    for (size_t i = 0; i < calls.size(); i++) {
      std::string a = calls[i]["function"]["arguments"];
      if (i < args.size() && args[i] != a) stream_ok = false;
      tcs.push_back({{"name", calls[i]["function"]["name"]}, {"arguments_hex", hexs(a)}});
    }
    res["tool_calls"] = tcs;
    res["stream_ok"] = stream_ok;
    res["complete"] = p.complete_tool_calls();
    out.push_back(res);
  }
  std::ofstream o(argv[3], std::ios::binary);
  o << out.dump(1) << "\n";
  return 0;
}
