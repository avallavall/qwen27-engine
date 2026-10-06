// The model's chat template (GGUF tokenizer.chat_template, copy in research/_chat_template.jinja) hard-coded in
// C++. Output must be byte-identical to jinja2 rendering the template with trim_blocks + lstrip_blocks and
// tojson = json.dumps(ensure_ascii=False) (HF / llama.cpp behaviour), including the template's exceptions.
#pragma once
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace q27 {

using ojson = nlohmann::ordered_json;

// An exception raised by the template itself (text = the template's message, as jinja2 raise_exception).
// Also thrown, with its own message, where jinja2 fails with another error (TypeError, UndefinedError), e.g.
// tool-call arguments that are not a mapping or a tool call without a string name.
struct TemplateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TemplateOptions {
  bool add_generation_prompt = true;
  bool add_vision_id = false;
  // chat_template_kwargs (unset = not defined for the template)
  std::optional<bool> enable_thinking;
  std::optional<bool> preserve_thinking;
  std::optional<std::string> reasoning_effort;  // valid: xhigh (default), medium, low
};

// Render messages (OpenAI chat format, already normalized: see normalize_messages) and tools (array of
// {type:"function", function:{name, description, parameters}}; null or empty = no tools).
std::string render_chat(const ojson& messages, const ojson& tools, const TemplateOptions& opt);

// What llama.cpp does before rendering: role "developer" -> "system"; tool-call "arguments" given as a JSON
// string -> the parsed JSON value (left as a string if it does not parse; render_chat then throws unless it
// is ""); content null -> "".
ojson normalize_messages(const ojson& messages);

// SHA-256 (hex) of the template text this renderer implements; the server compares it with the GGUF's
// tokenizer.chat_template and refuses to start on a mismatch.
const char* chat_template_sha256();

// SHA-256 of a byte string, as 64 lowercase hex digits.
std::string sha256_hex(const std::string& data);

}  // namespace q27
