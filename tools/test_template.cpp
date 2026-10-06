// Chat template test. Renders every fixture written by tools/gen_template_fixtures.py with the C++ renderer and
// compares the result byte by byte with the jinja2 output, or the exception with the jinja2 exception.
// Also checks sha256_hex (known answers) and chat_template_sha256() against the template text in the fixture
// file and, with --gguf, against the model GGUF.
// Usage: test_template [bench/out/template_fixtures.json] [--gguf model.gguf] [-v]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "chat_template.h"
#include "gguf.h"

using namespace q27;

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Bytes [from, to) of s, with control characters shown as escapes.
static std::string show(const std::string& s, size_t from, size_t to) {
  std::string out;
  for (size_t i = from; i < to && i < s.size(); ++i) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else if (c == '\\') out += "\\\\";
    else if (c < 0x20 || c == 0x7F) {
      char b[8];
      snprintf(b, sizeof b, "\\x%02x", c);
      out += b;
    } else {
      out += (char)c;
    }
  }
  return out;
}

static void print_diff(const std::string& got, const std::string& want) {
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  printf("  first difference at byte %zu (got %zu bytes, want %zu bytes)\n", i, got.size(), want.size());
  const size_t a = i > 80 ? i - 80 : 0;
  printf("  want: [%s]\n", show(want, a, i + 80).c_str());
  printf("  got : [%s]\n", show(got, a, i + 80).c_str());
}

static TemplateOptions parse_options(const ojson& o) {
  TemplateOptions opt;
  if (o.contains("add_generation_prompt")) opt.add_generation_prompt = o["add_generation_prompt"].get<bool>();
  if (o.contains("add_vision_id")) opt.add_vision_id = o["add_vision_id"].get<bool>();
  if (o.contains("enable_thinking")) opt.enable_thinking = o["enable_thinking"].get<bool>();
  if (o.contains("preserve_thinking")) opt.preserve_thinking = o["preserve_thinking"].get<bool>();
  if (o.contains("reasoning_effort")) opt.reasoning_effort = o["reasoning_effort"].get<std::string>();
  return opt;
}

int main(int argc, char** argv) {
  std::string path = "bench/out/template_fixtures.json", gguf_path;
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
    else if (a == "-v") verbose = true;
    else path = a;
  }
  int pass = 0, fail = 0;
  auto check = [&](bool ok, const std::string& what) {
    if (ok) ++pass;
    else { ++fail; printf("FAIL %s\n", what.c_str()); }
  };

  // SHA-256 known answers (FIPS 180-4 examples).
  check(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
  check(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
  check(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "sha256 448 bits");
  check(sha256_hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "sha256 million a");

  ojson doc;
  try {
    doc = ojson::parse(read_file(path));
  } catch (const std::exception& e) {
    printf("cannot load fixtures: %s\nRun: .venv\\Scripts\\python.exe tools\\gen_template_fixtures.py\n", e.what());
    return 1;
  }
  const std::string tmpl = doc["template"].get<std::string>();
  const std::string fx_sha = doc["template_sha256"].get<std::string>();
  check(sha256_hex(tmpl) == fx_sha, "sha256 of the fixture template text = template_sha256 in the file");
  check(fx_sha == chat_template_sha256(), "fixture template_sha256 = chat_template_sha256() (" + fx_sha + ")");
  if (!gguf_path.empty()) {
    GGUF g(gguf_path);
    const std::string h = sha256_hex(g.get_str("tokenizer.chat_template"));
    check(h == chat_template_sha256(), "GGUF tokenizer.chat_template sha256 = chat_template_sha256() (" + h + ")");
  }

  int n_out = 0, n_raise = 0, n_err = 0;
  double ms_total = 0;
  for (const auto& fx : doc["fixtures"]) {
    const std::string name = fx["name"].get<std::string>();
    const ojson& expect = fx["expect"];
    const TemplateOptions opt = parse_options(fx["options"]);
    std::string got, got_err;
    bool threw = false;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      const ojson msgs = fx["normalize"].get<bool>() ? normalize_messages(fx["messages"]) : fx["messages"];
      got = render_chat(msgs, fx["tools"], opt);
    } catch (const TemplateError& e) {
      threw = true;
      got_err = e.what();
    } catch (const std::exception& e) {
      threw = true;
      got_err = std::string("(not a TemplateError) ") + e.what();
    }
    ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    bool ok = false;
    if (expect.contains("output")) {
      ++n_out;
      const std::string want = expect["output"].get<std::string>();
      ok = !threw && got == want;
      if (!ok) {
        printf("FAIL %s\n", name.c_str());
        if (threw) printf("  threw: %s\n", got_err.c_str());
        else print_diff(got, want);
      }
    } else if (expect.contains("raise")) {
      ++n_raise;
      const std::string want = expect["raise"].get<std::string>();
      ok = threw && got_err == want;
      if (!ok) {
        printf("FAIL %s\n  want exception: %s\n", name.c_str(), want.c_str());
        if (threw) printf("  got exception : %s\n", got_err.c_str());
        else printf("  got output (%zu bytes)\n", got.size());
      }
    } else {
      // Another jinja2 error: any TemplateError is accepted, the message is not compared.
      ++n_err;
      ok = threw && got_err.rfind("(not a TemplateError)", 0) != 0;
      if (!ok) {
        printf("FAIL %s\n  want an error like: %s\n", name.c_str(), expect["error"].get<std::string>().c_str());
        if (threw) printf("  got exception: %s\n", got_err.c_str());
        else printf("  got output (%zu bytes)\n", got.size());
      }
    }
    if (ok) {
      ++pass;
      if (verbose) printf("PASS %s%s%s\n", name.c_str(), threw ? "  -> " : "", got_err.c_str());
    } else {
      ++fail;
    }
  }
  printf("fixtures: %d outputs, %d raise_exception, %d other errors; render time %.1f ms total\n", n_out, n_raise,
         n_err, ms_total);
  printf("PASS %d  FAIL %d\n", pass, fail);
  return fail == 0 ? 0 : 1;
}
