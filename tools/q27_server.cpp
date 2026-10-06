// OpenAI-compatible HTTP server for the engine (one compute slot, FIFO queue, prompt cache).
// Endpoints: POST /v1/chat/completions (and /chat/completions), GET /health, /v1/health, /props, /v1/models,
// /models, POST /tokenize, /detokenize, /apply-template. Response shapes follow llama.cpp's server
// (tools/server/server-task.cpp, server-common.cpp; MIT).
//
// Usage: q27_server -m <model.gguf> [--host 127.0.0.1] [--port 8081] [--api-key KEY] [--ctx N] [--devices 0,1]
//        [--temp 1.0] [--top-p 0.95] [--top-k 20] [--min-p 0.0] [--chat-template-kwargs JSON]
//        [--cache-ram MB] [--alias NAME] [--log-dir DIR]
// The API key can also come from the environment variable Q27_API_KEY. KV cache type: Q27_KV (q8_0 default).
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "chat_parser.h"
#include "chat_template.h"
#include "engine.h"
#include "httplib/httplib.h"
#include "model.h"
#include "tokenizer.h"
#include "vision.h"

using namespace q27;
using clk = std::chrono::steady_clock;

namespace {

constexpr int kDraftVocabDefault = 32768;  // tokens the MTP drafts score (chosen with q27_gen ... accept, see PLAN.md)

struct Options {
  std::string model, host = "127.0.0.1", api_key, alias = "qwen3.8-27b", log_dir;
  int port = 8081, ctx = 0;
  std::vector<int> devices = {0, 1};
  SampleParams sp;
  ojson kwargs = ojson{{"reasoning_effort", "medium"}};
  size_t cache_ram_mb = 8192;
  std::string mmproj;
  int img_min = 1024, img_max = 4096;  // image token limits (llama-server --image-min-tokens / max)
  // MTP draft vocabulary: "N" = first N ids of data\draft_vocab.bin, "file[:N]", or "0" = full vocabulary
  std::string draft_vocab = std::to_string(kDraftVocabDefault);
};

struct HttpError : std::runtime_error {
  int code;
  std::string type;
  HttpError(int c, const std::string& t, const std::string& m) : std::runtime_error(m), code(c), type(t) {}
};
HttpError bad_request(const std::string& m) { return HttpError(400, "invalid_request_error", m); }

ojson error_json(int code, const std::string& type, const std::string& msg) {
  return ojson{{"error", ojson{{"code", code}, {"message", msg}, {"type", type}}}};
}

std::string dump(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

void send_json(httplib::Response& res, const ojson& j, int status = 200) {
  res.status = status;
  res.set_content(dump(j), "application/json; charset=utf-8");
}

std::string random_string(int n = 32) {
  static const char cs[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  static thread_local std::mt19937_64 g{std::random_device{}()};
  std::string s(n, ' ');
  for (auto& c : s) c = cs[g() % 62];
  return s;
}

std::vector<int> parse_devices(const std::string& d) {
  std::vector<int> v;
  size_t a = 0;
  while (a <= d.size()) {
    size_t b = d.find(',', a);
    if (b == std::string::npos) b = d.size();
    v.push_back(std::stoi(d.substr(a, b - a)));
    a = b + 1;
  }
  return v;
}

struct ImgIn {
  std::shared_ptr<std::vector<uint8_t>> rgb;
  VisionPlan plan;
  uint64_t hash = 0;
};

struct Rendered {
  std::string text;  // image places hold the media marker
  bool thinking = true;
  ojson tools;
  std::vector<ImgIn> images;  // in marker order
};

std::string base64_decode(const std::string& in) {
  static const std::vector<int> T = [] {
    std::vector<int> t(256, -1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) t[(unsigned char)a[i]] = i;
    t['-'] = 62;
    t['_'] = 63;
    return t;
  }();
  std::string out;
  out.reserve(in.size() * 3 / 4);
  int val = 0, bits = -8;
  for (unsigned char c : in) {
    if (c == '=') break;
    const int d = T[c];
    if (d < 0) continue;
    val = (val << 6) + d;
    bits += 6;
    if (bits >= 0) {
      out.push_back(char((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

uint64_t fnv1a64(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
  return h;
}

// Bytes of an image URL, with llama.cpp's checks (tools/server/server-common.cpp handle_media): only data: URIs.
std::string load_media(const std::string& url) {
  if (url.rfind("data:", 0) == 0) {
    const size_t comma = url.find(',');
    if (comma == std::string::npos || url.find(',', comma + 1) != std::string::npos)
      throw HttpError(400, "invalid_request_error", "Invalid uri-encoded base64 value");
    const std::string head = url.substr(0, comma);
    if (head.rfind("data:image/", 0) != 0) throw HttpError(400, "invalid_request_error", "Invalid uri format: " + head);
    if (head.size() < 6 || head.compare(head.size() - 6, 6, "base64") != 0)
      throw HttpError(400, "invalid_request_error", "uri must be base64 encoded");
    return base64_decode(url.substr(comma + 1));
  }
  if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0)
    throw HttpError(400, "invalid_request_error", "remote image URLs are not supported; send the image as a data: URI");
  std::string b = base64_decode(url);  // llama.cpp also takes a raw base64 string
  if (b.empty()) throw HttpError(400, "invalid_request_error", "Invalid base64 value");
  return b;
}

class Server {
 public:
  Server(const Options& o, const Model& m, Decoder& dec, const Tokenizer& tok, Engine& eng, VisionEncoder* venc)
      : o_(o), m_(m), dec_(dec), tok_(tok), eng_(eng), venc_(venc), t_start_(std::time(nullptr)) {
    tpl_text_ = m_.g_->get_str("tokenizer.chat_template");
    fp_ = "q27-" + std::string(chat_template_sha256()).substr(0, 8);
    marker_ = "<__media_" + random_string() + "__>";
    vision_start_ = tok_.find("<|vision_start|>");
    vision_end_ = tok_.find("<|vision_end|>");
    if (!o_.log_dir.empty()) {  // continue the numbering of an existing log
      int last = 0;
      for (const auto& e : std::filesystem::directory_iterator(o_.log_dir)) {
        const std::string f = e.path().filename().string();
        if (f.size() == 14 && f.rfind("req-", 0) == 0) last = std::max(last, atoi(f.c_str() + 4));
      }
      next_id_ = last + 1;
    }
  }
  void routes(httplib::Server& s);

 private:
  Rendered render(const ojson& body) const;
  std::string display(const std::string& text) const;
  void tokenize_prompt(const Rendered& r, GenRequest& g) const;
  void chat(const httplib::Request& req, httplib::Response& res);
  void log_request(int id, const ojson& body, const Rendered& r);
  void log_result(int id, const GenEvent& e);

  const Options& o_;
  const Model& m_;
  Decoder& dec_;
  const Tokenizer& tok_;
  Engine& eng_;
  VisionEncoder* venc_;
  std::time_t t_start_;
  std::string tpl_text_, fp_, marker_;
  int vision_start_ = -1, vision_end_ = -1;
  std::atomic<int> next_id_{1};
};

Rendered Server::render(const ojson& body) const {
  if (!body.is_object()) throw bad_request("request body must be a JSON object");
  if (!body.contains("messages") || !body["messages"].is_array()) throw bad_request("Expected 'messages' to be an array");
  ojson messages = body["messages"];
  Rendered r;
  // Media parts, as llama.cpp's server (server-common.cpp oaicompat_content_load_media): an image becomes a marker
  // text; other part types than text are refused. A message with media then has its parts joined into one string
  // (common/chat.cpp common_chat_msg::to_json_oaicompat): "\n" between text parts, nothing next to a marker.
  for (auto& m : messages) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    bool media = false;
    for (auto& p : m["content"]) {
      const std::string type = p.is_object() && p.contains("type") && p["type"].is_string() ? p["type"].get<std::string>() : "";
      if (type == "image_url") {
        if (!venc_)
          throw bad_request("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
        const std::string url = p.contains("image_url") && p["image_url"].is_object() && p["image_url"].contains("url") &&
                                        p["image_url"]["url"].is_string()
                                    ? p["image_url"]["url"].get<std::string>()
                                    : "";
        const std::string bytes = load_media(url);
        ImgIn im;
        int w = 0, h = 0;
        try {
          im.rgb = std::make_shared<std::vector<uint8_t>>(VisionEncoder::decode(bytes.data(), bytes.size(), w, h));
        } catch (const std::exception& e) {
          throw bad_request(std::string("failed to load image: ") + e.what());
        }
        im.plan = VisionEncoder::plan(w, h, o_.img_min, o_.img_max);
        im.hash = fnv1a64(bytes);
        r.images.push_back(std::move(im));
        p = ojson{{"type", "media_marker"}, {"text", marker_}};
        media = true;
      } else if (type == "input_audio") {
        throw bad_request("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
      } else if (type == "input_video" || type == "video_url") {
        throw bad_request("video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
      } else if (type != "text") {
        throw bad_request("unsupported content[].type");
      }
    }
    if (media) {
      std::string text;
      bool last_media = false;
      for (const auto& p : m["content"]) {
        const std::string type = p.value("type", "");
        bool nl;
        if (type == "text") {
          nl = !last_media && !text.empty();
          last_media = false;
        } else {
          nl = false;
          last_media = true;
        }
        if (nl) text += '\n';
        if (p.contains("text") && p["text"].is_string()) text += p["text"].get<std::string>();
      }
      m["content"] = text;
    }
  }
  // Tools are rebuilt as llama.cpp does (common/chat.cpp common_chat_tools_parse_oaicompat + _to_json_oaicompat):
  // {"type":"function","function":{"name","description" ("" if missing),"parameters" ({} if missing)}}; other keys
  // (e.g. "strict") are dropped. An empty list means no tools.
  ojson tools;
  if (body.contains("tools") && !body["tools"].is_null()) {
    if (!body["tools"].is_array()) throw bad_request("Failed to parse tools: Expected 'tools' to be an array");
    for (const auto& t : body["tools"]) {
      if (!t.is_object() || !t.contains("type") || t["type"] != "function" || !t.contains("function") ||
          !t["function"].is_object() || !t["function"].contains("name") || !t["function"]["name"].is_string())
        throw bad_request("Failed to parse tools: " + dump(t));
      const ojson& f = t["function"];
      ojson nf{{"name", f["name"]},
               {"description", f.contains("description") && f["description"].is_string() ? f["description"] : ojson("")},
               {"parameters", f.contains("parameters") ? f["parameters"] : ojson::object()}};
      if (!tools.is_array()) tools = ojson::array();
      tools.push_back(ojson{{"type", "function"}, {"function", nf}});
    }
  }
  r.tools = tools;  // the parser gets the tools unless tool_choice is "none" (then calls stay in the content)
  if (body.contains("tool_choice") && body["tool_choice"].is_string() && body["tool_choice"] == "none") r.tools = ojson();
  ojson kw = o_.kwargs;
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
    for (auto it = body["chat_template_kwargs"].begin(); it != body["chat_template_kwargs"].end(); ++it) kw[it.key()] = it.value();
  if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) kw["reasoning_effort"] = body["reasoning_effort"];
  else if (body.contains("reasoning") && body["reasoning"].is_object() && body["reasoning"].contains("effort") &&
           body["reasoning"]["effort"].is_string())
    kw["reasoning_effort"] = body["reasoning"]["effort"];  // {"reasoning": {"effort": ...}} (Qwen Code without samplingParams)
  // The template knows three levels: xhigh (default), medium, low. Other clients' names are mapped to them; llama.cpp
  // passes the value through, so there "high" fails. "none" / "off" turn thinking off (llama.cpp does this for "none").
  if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string()) {
    std::string e = kw["reasoning_effort"].get<std::string>();
    for (auto& c : e) c = (char)tolower((unsigned char)c);
    if (e == "none" || e == "off" || e == "disabled") {
      kw.erase("reasoning_effort");
      kw["enable_thinking"] = false;
    } else {
      if (e == "high" || e == "max" || e == "maximum") e = "xhigh";
      else if (e == "minimal" || e == "min") e = "low";
      kw["reasoning_effort"] = e;
    }
  }
  TemplateOptions to;
  to.add_generation_prompt = true;
  if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean()) to.enable_thinking = kw["enable_thinking"].get<bool>();
  if (kw.contains("preserve_thinking") && kw["preserve_thinking"].is_boolean())
    to.preserve_thinking = kw["preserve_thinking"].get<bool>();
  if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string())
    to.reasoning_effort = kw["reasoning_effort"].get<std::string>();
  if (kw.contains("add_vision_id") && kw["add_vision_id"].is_boolean()) to.add_vision_id = kw["add_vision_id"].get<bool>();
  try {
    r.text = render_chat(normalize_messages(messages), tools, to);
  } catch (const TemplateError& e) {
    throw bad_request(e.what());
  }
  r.thinking = to.enable_thinking.value_or(true);
  size_t markers = 0;
  for (size_t f = r.text.find(marker_); f != std::string::npos; f = r.text.find(marker_, f + marker_.size())) markers++;
  if (markers != r.images.size()) throw bad_request("an image was not placed in the prompt (images in this message role?)");
  return r;
}

// The prompt as llama.cpp's oracle shows it: each image as <|vision_start|><|image_pad|><|vision_end|>.
std::string Server::display(const std::string& text) const {
  std::string out;
  size_t pos = 0;
  for (size_t f = text.find(marker_); f != std::string::npos; f = text.find(marker_, pos)) {
    out.append(text, pos, f - pos);
    out += "<|vision_start|><|image_pad|><|vision_end|>";
    pos = f + marker_.size();
  }
  out.append(text, pos, std::string::npos);
  return out;
}

// Token ids, image spans and (with images) IMRoPE positions, as mtmd_tokenize lays them out: the text around each
// marker is tokenized on its own, <|vision_start|> + image rows + <|vision_end|> in between. Image rows get
// t = P, h = P + row, w = P + column (P = the position after <|vision_start|>), and the position then moves by
// max(nx, ny) (llama.cpp tools/mtmd/mtmd.cpp, mtmd-helper).
void Server::tokenize_prompt(const Rendered& r, GenRequest& g) const {
  g.prompt.clear();
  g.images.clear();
  size_t pos = 0;
  for (size_t k = 0;; k++) {
    const size_t f = r.text.find(marker_, pos);
    const std::vector<int> ids = tok_.encode(r.text.substr(pos, f == std::string::npos ? std::string::npos : f - pos), true);
    g.prompt.insert(g.prompt.end(), ids.begin(), ids.end());
    if (f == std::string::npos) break;
    const ImgIn& im = r.images[k];
    g.prompt.push_back(vision_start_);
    ImageSpan s;
    s.start = (int)g.prompt.size();
    s.plan = im.plan;
    s.rgb = im.rgb;
    s.hash = im.hash;
    for (int i = 0; i < im.plan.n_tokens; i++) g.prompt.push_back(image_row_id(im.hash, i));
    g.prompt.push_back(vision_end_);
    g.images.push_back(s);
    pos = f + marker_.size();
  }
  if (g.images.empty()) return;
  g.rope.assign(3 * g.prompt.size(), 0);
  int p = 0;
  size_t k = 0;
  for (size_t i = 0; i < g.prompt.size();) {
    if (k < g.images.size() && (int)i == g.images[k].start) {
      const VisionPlan& pl = g.images[k].plan;
      for (int t = 0; t < pl.n_tokens; t++, i++) {
        g.rope[3 * i] = p;
        g.rope[3 * i + 1] = p + t / pl.nx;
        g.rope[3 * i + 2] = p + t % pl.nx;
      }
      p += std::max(pl.nx, pl.ny);
      k++;
      continue;
    }
    g.rope[3 * i] = g.rope[3 * i + 1] = g.rope[3 * i + 2] = p++;
    i++;
  }
}

void Server::log_request(int id, const ojson& body, const Rendered& r) {
  if (o_.log_dir.empty()) return;
  char name[64];
  snprintf(name, sizeof(name), "/req-%05d.json", id);
  std::ofstream f(o_.log_dir + name, std::ios::binary);
  f << dump(ojson{{"request", body}, {"prompt", display(r.text)}});
}

void Server::log_result(int id, const GenEvent& e) {
  if (o_.log_dir.empty()) return;
  char name[64];
  snprintf(name, sizeof(name), "/res-%05d.json", id);
  std::ofstream f(o_.log_dir + name, std::ios::binary);
  f << dump(ojson{{"raw", e.raw},
                  {"reasoning_content", e.reasoning},
                  {"content", e.content},
                  {"tool_calls", e.tool_calls},
                  {"finish_reason", e.finish_reason},
                  {"timings", e.timings}});
}

// One chunk object of a chat stream.
ojson chunk(const std::string& id, const std::string& model, const std::string& fp, std::time_t t, const ojson& delta,
            const ojson& finish) {
  return ojson{{"choices", ojson::array({ojson{{"finish_reason", finish}, {"index", 0}, {"delta", delta}}})},
               {"created", t},
               {"id", id},
               {"model", model},
               {"system_fingerprint", fp},
               {"object", "chat.completion.chunk"}};
}

// The deltas of one parser step as llama.cpp sends them: reasoning/content in one chunk, one chunk per tool call delta.
std::vector<ojson> delta_json(const ParseDelta& d) {
  std::vector<ojson> v;
  if (!d.reasoning.empty() || !d.content.empty()) {
    ojson j = ojson::object();
    if (!d.reasoning.empty()) j["reasoning_content"] = d.reasoning;
    if (!d.content.empty()) j["content"] = d.content;
    v.push_back(j);
  }
  for (const ToolCallDelta& tc : d.tool_calls) {
    ojson c;
    c["index"] = tc.index;
    if (!tc.id.empty()) {
      c["id"] = tc.id;
      c["type"] = "function";
    }
    if (!tc.name.empty() || !tc.arguments.empty()) {
      ojson f = ojson::object();
      if (!tc.name.empty()) f["name"] = tc.name;
      if (!tc.arguments.empty()) f["arguments"] = tc.arguments;
      c["function"] = f;
    }
    v.push_back(ojson{{"tool_calls", ojson::array({c})}});
  }
  return v;
}

void Server::chat(const httplib::Request& req, httplib::Response& res) {
  ojson body;
  try {
    body = ojson::parse(req.body);
  } catch (const std::exception& e) {
    throw bad_request(std::string("invalid JSON: ") + e.what());
  }
  const int id_num = next_id_++;
  const Rendered r = render(body);
  log_request(id_num, body, r);
  auto job = std::make_shared<Job>();
  GenRequest& g = job->req;
  try {
    tokenize_prompt(r, g);
  } catch (const std::invalid_argument& e) {
    throw bad_request(std::string("invalid UTF-8 in the request: ") + e.what());
  }
  const int n = (int)g.prompt.size();
  // Overflow text: llama.cpp's sentence (pi-ai / dsh compact on "exceeds the available context size") plus OpenAI's
  // (Qwen Code compacts on "maximum context length").
  if (n >= eng_.capacity())
    throw HttpError(400, "exceed_context_size_error",
                    "request (" + std::to_string(n) + " tokens) exceeds the available context size (" +
                        std::to_string(dec_.n_ctx()) + " tokens), try increasing it. This model's maximum context length is " +
                        std::to_string(dec_.n_ctx()) + " tokens. However, your messages resulted in " + std::to_string(n) +
                        " tokens.");
  g.thinking = r.thinking;
  g.tools = r.tools;
  g.sp = o_.sp;
  auto num = [&](const char* k, float& dst) {
    if (body.contains(k) && body[k].is_number()) dst = body[k].get<float>();
  };
  num("temperature", g.sp.temp);
  num("top_p", g.sp.top_p);
  num("min_p", g.sp.min_p);
  if (body.contains("top_k") && body["top_k"].is_number()) {
    const int k = body["top_k"].get<int>();
    g.sp.top_k = (k <= 0 || k > kCand) ? kCand : k;  // the sampler keeps at most kCand candidates
  }
  if (body.contains("seed") && body["seed"].is_number_integer() && body["seed"].get<int64_t>() >= 0) {
    uint64_t x = (uint64_t)body["seed"].get<int64_t>() * 0x9E3779B97F4A7C15ull;
    x ^= x >> 31;
    g.rng_start = (int)(x & 0x3fffffff);
  } else {
    static thread_local std::mt19937 rg{std::random_device{}()};
    g.rng_start = (int)(rg() & 0x3fffffff);
  }
  for (const char* k : {"max_tokens", "max_completion_tokens", "n_predict"})
    if (body.contains(k) && body[k].is_number_integer()) g.max_tokens = body[k].get<int>();
  if (g.max_tokens < -1) g.max_tokens = -1;
  if (body.contains("stop")) {
    if (body["stop"].is_string()) g.stop.push_back(body["stop"]);
    else if (body["stop"].is_array())
      for (const auto& s : body["stop"])
        if (s.is_string()) g.stop.push_back(s);
  }
  char tag[32];
  snprintf(tag, sizeof(tag), "#%d", id_num);
  g.tag = tag;
  const bool stream = body.value("stream", false);
  const bool include_usage =
      body.contains("stream_options") && body["stream_options"].is_object() && body["stream_options"].value("include_usage", false);
  const std::string model = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : o_.alias;
  const std::string cid = "chatcmpl-" + random_string();
  eng_.submit(job);

  if (!stream) {
    for (;;) {
      std::vector<GenEvent> evs;
      job->pop(evs, std::chrono::milliseconds(1000));
      if (req.is_connection_closed()) { job->cancel = true; return; }
      for (GenEvent& e : evs) {
        if (e.kind == GenEvent::kError) {
          send_json(res, error_json(e.code, e.err_type, e.message), e.code);
          return;
        }
        if (e.kind != GenEvent::kDone) continue;
        log_result(id_num, e);
        ojson msg{{"role", "assistant"}, {"content", e.content}};
        if (!e.reasoning.empty()) msg["reasoning_content"] = e.reasoning;
        if (!e.tool_calls.empty()) msg["tool_calls"] = e.tool_calls;
        ojson out{{"choices", ojson::array({ojson{{"finish_reason", e.finish_reason}, {"index", 0}, {"message", msg}}})},
                  {"created", std::time(nullptr)},
                  {"model", model},
                  {"system_fingerprint", fp_},
                  {"object", "chat.completion"},
                  {"usage", e.usage},
                  {"id", cid},
                  {"timings", e.timings}};
        send_json(res, out);
        return;
      }
    }
  }

  // Streaming: headers and the role chunk now, then one chunk per delta, ":" pings while waiting.
  res.status = 200;
  res.set_header("Cache-Control", "no-cache");
  struct St {
    bool first = true, done = false;
    clk::time_point last_write = clk::now();
  };
  auto st = std::make_shared<St>();
  const httplib::Request* rq = &req;  // alive while the provider runs
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, job, st, cid, model, include_usage, id_num, rq](size_t, httplib::DataSink& sink) -> bool {
        auto w = [&](const std::string& s) { return sink.write(s.data(), s.size()); };
        auto data = [&](const ojson& j) { return w("data: " + dump(j) + "\n\n"); };
        const std::time_t t = std::time(nullptr);
        if (st->done) return false;
        if (st->first) {
          st->first = false;
          if (!data(chunk(cid, model, fp_, t, ojson{{"role", "assistant"}, {"content", nullptr}}, nullptr))) {
            job->cancel = true;
            return false;
          }
          return true;
        }
        std::vector<GenEvent> evs;
        if (!job->pop(evs, std::chrono::milliseconds(1000))) {
          // nothing new: check the connection every second, ping every 10 s (llama.cpp sends ":" pings too)
          if (rq->is_connection_closed()) { job->cancel = true; return false; }
          if (clk::now() - st->last_write > std::chrono::seconds(10)) {
            st->last_write = clk::now();
            if (!w(":\n\n")) { job->cancel = true; return false; }
          }
          return true;
        }
        st->last_write = clk::now();
        std::string out;
        for (GenEvent& e : evs) {
          if (e.kind == GenEvent::kDelta) {
            for (const ojson& d : delta_json(e.delta)) out += "data: " + dump(chunk(cid, model, fp_, t, d, nullptr)) + "\n\n";
          } else if (e.kind == GenEvent::kError) {
            out += "data: " + dump(error_json(e.code, e.err_type, e.message)) + "\n\n";
            st->done = true;
          } else {
            log_result(id_num, e);
            ojson fin = chunk(cid, model, fp_, t, ojson::object(), e.finish_reason);
            if (include_usage) {
              out += "data: " + dump(fin) + "\n\n";
              fin = ojson{{"choices", ojson::array()},
                          {"created", t},
                          {"id", cid},
                          {"model", model},
                          {"system_fingerprint", fp_},
                          {"object", "chat.completion.chunk"},
                          {"usage", e.usage}};
            }
            fin["timings"] = e.timings;
            out += "data: " + dump(fin) + "\n\n";
            out += "data: [DONE]\n\n";
            st->done = true;
          }
        }
        if (!out.empty() && !w(out)) { job->cancel = true; return false; }
        if (st->done) sink.done();
        return true;
      },
      [job](bool ok) {
        if (!ok) job->cancel = true;
      });
}

void Server::routes(httplib::Server& s) {
  s.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
    if (o_.api_key.empty() || req.path == "/health" || req.path == "/v1/health") return httplib::Server::HandlerResponse::Unhandled;
    std::string key = req.get_header_value("Authorization");
    if (key.rfind("Bearer ", 0) == 0) key = key.substr(7);
    if (key.empty()) key = req.get_header_value("X-Api-Key");
    if (key == o_.api_key) return httplib::Server::HandlerResponse::Unhandled;
    send_json(res, error_json(401, "authentication_error", "Invalid API Key"), 401);
    return httplib::Server::HandlerResponse::Handled;
  });
  s.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    try {
      std::rethrow_exception(ep);
    } catch (const HttpError& e) {
      send_json(res, error_json(e.code, e.type, e.what()), e.code);
    } catch (const std::exception& e) {
      send_json(res, error_json(500, "server_error", e.what()), 500);
    } catch (...) {
      send_json(res, error_json(500, "server_error", "unknown error"), 500);
    }
  });
  auto health = [](const httplib::Request&, httplib::Response& res) { send_json(res, ojson{{"status", "ok"}}); };
  s.Get("/health", health);
  s.Get("/v1/health", health);
  s.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
    ojson params{{"temperature", o_.sp.temp}, {"top_k", o_.sp.top_k}, {"top_p", o_.sp.top_p}, {"min_p", o_.sp.min_p},
                 {"n_predict", -1}, {"max_tokens", -1}};
    send_json(res, ojson{{"default_generation_settings", ojson{{"n_ctx", dec_.n_ctx()}, {"params", params}}},
                         {"total_slots", 1},
                         {"model_alias", o_.alias},
                         {"model_path", o_.model},
                         {"modalities", ojson{{"vision", venc_ != nullptr}, {"audio", false}}},
                         {"chat_template", tpl_text_},
                         {"chat_template_kwargs", o_.kwargs},
                         {"kv_cache_type", dec_.kv_q8() ? "q8_0" : "f16"},
                         {"queue", eng_.queued()},
                         {"build_info", fp_}});
  });
  auto models = [this](const httplib::Request&, httplib::Response& res) {
    ojson meta{{"n_ctx", dec_.n_ctx()}, {"n_ctx_train", m_.hp().ctx_train}, {"n_vocab", m_.hp().vocab}};
    ojson mdl{{"id", o_.alias}, {"object", "model"}, {"created", (int64_t)t_start_}, {"owned_by", "q27"}, {"meta", meta}};
    send_json(res, ojson{{"object", "list"}, {"data", ojson::array({mdl})}});
  };
  s.Get("/v1/models", models);
  s.Get("/models", models);
  s.Post("/tokenize", [this](const httplib::Request& req, httplib::Response& res) {
    const ojson b = ojson::parse(req.body);
    const std::string text = b.value("content", "");
    std::vector<int> ids;
    try {
      ids = tok_.encode(text, b.value("parse_special", true));
    } catch (const std::invalid_argument& e) {
      throw bad_request(e.what());
    }
    ojson toks = ojson::array();
    if (b.value("with_pieces", false))
      for (int id : ids) toks.push_back(ojson{{"id", id}, {"piece", tok_.piece(id)}});
    else
      for (int id : ids) toks.push_back(id);
    send_json(res, ojson{{"tokens", toks}});
  });
  s.Post("/detokenize", [this](const httplib::Request& req, httplib::Response& res) {
    const ojson b = ojson::parse(req.body);
    std::vector<int> ids;
    if (b.contains("tokens") && b["tokens"].is_array())
      for (const auto& t : b["tokens"])
        if (t.is_number_integer() && t.get<int>() >= 0 && t.get<int>() < tok_.n_vocab()) ids.push_back(t.get<int>());
    send_json(res, ojson{{"content", tok_.decode(ids)}});
  });
  s.Post("/apply-template", [this](const httplib::Request& req, httplib::Response& res) {
    send_json(res, ojson{{"prompt", display(render(ojson::parse(req.body)).text)}});
  });
  auto chat = [this](const httplib::Request& req, httplib::Response& res) { this->chat(req, res); };
  s.Post("/v1/chat/completions", chat);
  s.Post("/chat/completions", chat);
}

void usage() {
  fprintf(stderr,
          "usage: q27_server -m <model.gguf> [--host 127.0.0.1] [--port 8081] [--api-key KEY] [--ctx N] [--devices 0,1]\n"
          "       [--temp 1.0] [--top-p 0.95] [--top-k 20] [--min-p 0.0] [--chat-template-kwargs JSON]\n"
          "       [--reasoning-effort xhigh|medium|low] [--cache-ram MB] [--alias NAME] [--log-dir DIR]\n"
          "       [--mmproj mmproj.gguf] [--image-min-tokens 1024] [--image-max-tokens 4096]\n"
          "       [--draft-vocab N | file[:N] | 0]   (MTP draft vocabulary; default the first %d ids of data/draft_vocab.bin)\n",
          kDraftVocabDefault);
}

}  // namespace

int main(int argc, char** argv) try {
  Options o;
  if (const char* k = getenv("Q27_API_KEY")) o.api_key = k;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "-m" || a == "--model") o.model = next();
    else if (a == "--host") o.host = next();
    else if (a == "--port") o.port = std::stoi(next());
    else if (a == "--api-key") o.api_key = next();
    else if (a == "--ctx" || a == "-c") o.ctx = std::stoi(next());
    else if (a == "--devices") o.devices = parse_devices(next());
    else if (a == "--temp") o.sp.temp = std::stof(next());
    else if (a == "--top-p") o.sp.top_p = std::stof(next());
    else if (a == "--top-k") o.sp.top_k = std::min(kCand, std::max(1, std::stoi(next())));
    else if (a == "--min-p") o.sp.min_p = std::stof(next());
    else if (a == "--chat-template-kwargs") o.kwargs = ojson::parse(next());
    else if (a == "--reasoning-effort") o.kwargs["reasoning_effort"] = next();
    else if (a == "--cache-ram") o.cache_ram_mb = (size_t)std::stoll(next());
    else if (a == "--alias") o.alias = next();
    else if (a == "--log-dir") o.log_dir = next();
    else if (a == "--mmproj") o.mmproj = next();
    else if (a == "--image-min-tokens") o.img_min = std::stoi(next());
    else if (a == "--image-max-tokens") o.img_max = std::stoi(next());
    else if (a == "--draft-vocab") o.draft_vocab = next();
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else throw std::runtime_error("unknown option " + a);
  }
  if (o.model.empty()) { usage(); return 1; }
  if (!o.log_dir.empty()) std::filesystem::create_directories(o.log_dir);

  const auto t0 = clk::now();
  Model model(o.model, o.devices);
  // draft vocabulary: before the Decoder, whose graphs capture it
  if (o.draft_vocab != "0") {
    std::string path = "data/draft_vocab.bin";
    int n = 0;
    if (!o.draft_vocab.empty() && std::all_of(o.draft_vocab.begin(), o.draft_vocab.end(), ::isdigit)) {
      n = std::stoi(o.draft_vocab);
    } else {
      path = o.draft_vocab;
      const size_t c = path.rfind(':');
      if (c != std::string::npos && c > 1) { n = std::stoi(path.substr(c + 1)); path = path.substr(0, c); }
    }
    if (std::filesystem::exists(path)) {
      model.set_draft_vocab_file(path, n);
      fprintf(stderr, "draft vocabulary: %d tokens from %s\n", n, path.c_str());
    } else {
      fprintf(stderr, "draft vocabulary file %s not found: drafts use the full vocabulary\n", path.c_str());
    }
  } else {
    model.set_draft_vocab({});
  }
  const std::string& tpl = model.g_->get_str("tokenizer.chat_template");
  if (sha256_hex(tpl) != chat_template_sha256()) {
    fprintf(stderr, "error: the model's chat template differs from the one this server implements (sha256 %s)\n",
            sha256_hex(tpl).c_str());
    return 1;
  }
  Tokenizer tok(*model.g_);
  const bool q8 = Decoder::kv_q8_from_env();
  int n_ctx = o.ctx;
  if (n_ctx <= 0) {
    constexpr size_t MiB = 1 << 20;
    // card 0: 600 MB desktop growth + 300 MB checkpoint staging + 512 MB prefill; card 1: 1536 MB vision + 300 + 512
    const std::vector<size_t> reserve = {(512 + 300 + 600) * MiB, (512 + 300 + 1536) * MiB};
    n_ctx = Decoder::fit_ctx(model, reserve, q8);
  }
  n_ctx = std::min(n_ctx, model.hp().ctx_train);
  Decoder dec(model, n_ctx, q8);
  // warm up: capture the graphs with the default sampling settings
  {
    const std::vector<int> p = tok.encode("<|im_start|>user\nSay OK.<|im_end|>\n<|im_start|>assistant\n<think>\n", true);
    dec.begin(p, o.sp);
    int out[4];
    for (int i = 0; i < 3; i++) dec.spec_step(out);
    dec.reset();
  }
  EngineOptions eo;
  eo.cache_ram_mb = o.cache_ram_mb;
  // the image encoder goes on the last card (card 1: card 0 drives the desktop), after the decoder took its context
  std::unique_ptr<VisionEncoder> venc;
  if (!o.mmproj.empty()) venc = std::make_unique<VisionEncoder>(o.mmproj, o.devices.back());
  Engine eng(model, dec, tok, eo, venc.get());
  Server srv(o, model, dec, tok, eng, venc.get());
  httplib::Server http;
  http.set_read_timeout(600, 0);
  http.set_write_timeout(600, 0);
  http.set_keep_alive_timeout(30);
  srv.routes(http);
  fprintf(stderr, "q27_server: model loaded in %.1f s; n_ctx %d, KV %s, %d card(s); listening on http://%s:%d%s\n",
          std::chrono::duration<double>(clk::now() - t0).count(), n_ctx, q8 ? "q8_0" : "f16", (int)o.devices.size(),
          o.host.c_str(), o.port, o.api_key.empty() ? "" : " (API key required)");
  if (!http.listen(o.host, o.port)) {
    fprintf(stderr, "error: cannot listen on %s:%d\n", o.host.c_str(), o.port);
    return 1;
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
