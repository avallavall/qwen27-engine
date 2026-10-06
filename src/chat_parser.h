// Streaming parser for the model's output: reasoning, content and XML tool calls (the Qwen3-Coder format of
// research/_chat_template.jinja). The target is llama.cpp's non-streaming parse of the same text.
//
// llama.cpp behaviour (llama-rig2, branch rig/full; checked against the production llama-common.dll of
// qwen38_27\bin-parches on 141 fixed inputs and 34,000 random well-formed outputs):
// - Handler: the template has <tool_call>, <function=, <parameter=, so the Qwen3-Coder PEG parser is used
//   (common/chat.cpp:1218-1226). The template has <think>, so reasoning is parsed, and a call must start with
//   <tool_call>: the fallback for a bare "<function=" is only for templates without <think>
//   (common/parsers/qwen3-coder.cpp:21,62-71,162). The parse runs on generation_prompt + output
//   (common/chat.cpp:1465) and is always LENIENT, also for the final parse (common/chat.cpp:1471).
// - Grammar: "<think>" space reasoning (until "</think>" or "<tool_call>") then "</think>" or a peek at
//   "<tool_call>" (qwen3-coder.cpp:79-82); then space, content, space, tool calls (qwen3-coder.cpp:169-170;
//   operator<< inserts p.space(), common/peg-parser.cpp:1020). space() is std::isspace (peg-parser.cpp:517).
// - Reasoning: leading whitespace is dropped. Trailing whitespace is kept: "a\n</think>" gives "a\n".
//   A "<tool_call>" also ends reasoning, even with no tools; it then starts the content (or the calls).
//   Whitespace-only reasoning is cleared (common/chat-peg-parser.cpp:300); it cannot occur after the space().
// - Content: leading whitespace is dropped, trailing whitespace is kept (also the "\n" before <tool_call>).
//   With tools, content ends at the first "<tool_call>" (qwen3-coder.cpp:170). Without tools (null or []),
//   content is the rest of the text (qwen3-coder.cpp:174).
// - Thinking off: the prompt ends "<think>\n\n</think>\n\n", so reasoning is "" and the output is content.
// - Calls: "<tool_call>\n<function=NAME>\n" (params) "</function>\n</tool_call>", then whitespace, then the next
//   call (qwen3-coder.cpp:149-166). parallel_tool_calls defaults to true for this template
//   (tools/server/server-common.cpp:1321). Text after the last call is dropped, and so is text between calls
//   together with every call after it. Content never resumes after the first "<tool_call>".
// - A call that does not match the grammar (unknown tool, unknown or misplaced parameter, missing required
//   parameter, non-JSON value for a non-string type, missing "\n" after a tag) makes the call and all later
//   text disappear (the repetition stops; there is no p.end()). In production a lazy grammar prevents this.
// - Parameter value: string type (schema value types only string) = raw text until "\n</parameter>\n"
//   (qwen3-coder.cpp:91-93,113); a trailing "\n" before that tag stays in the value. No string in the types =
//   one JSON value, no surrounding whitespace, then "\n</parameter>\n" (qwen3-coder.cpp:111-112); a JSON
//   string is allowed there. Mixed types (string + others, or no type at all) = the JSON kinds of the other
//   types (object, array, number, bool, null; never a JSON string) followed by the close tag, else the raw
//   text as a string (qwen3-coder.cpp:114-135). Types come from common/json-schema.cpp:227 and :368
//   (type arrays, anyOf/oneOf, allOf, enum, const, $ref; number also admits integer).
// - Required parameters may come in any order (permute, up to 6), but all of them before any optional one;
//   optional ones may repeat (qwen3-coder.cpp:144-146).
// - Arguments JSON text (chat-peg-parser.cpp:358-447): "{" when the name is known, then per parameter
//   "," (not first) + json(trim(name)) + ":" (no spaces), then the value: strings as a JSON string escaped by
//   nlohmann dump (ensure_ascii off, control chars as \u00xx, "/" not escaped), JSON values copied verbatim
//   (their own spacing kept), then "}" at "</function>\n". Example: {"city":"Paris","days":3}.
// - EOS inside a call (final parse, LENIENT): the call is kept with the text parsed so far and no closing:
//   "{" after the name line, {"city":" after the parameter line, {"city":"Par inside the value, no "}" until
//   "</function>\n" is complete. A union-typed value whose JSON prefix is still valid is left out entirely.
//   A trailing partial tag, a trailing backslash escape in a JSON string and a trailing incomplete UTF-8
//   sequence are left out (peg-parser.cpp:603,617,696,718).
// - UTF-8: invalid runs in reasoning/content become U+FFFD (peg-parser.cpp:169); the decoder is lax (no
//   overlong or surrogate check, common/unicode.cpp:17). Invalid UTF-8 in a string argument makes llama.cpp
//   throw (nlohmann type_error 316).
// - Tool call id: 32 random chars [0-9A-Za-z] (tools/server/server-common.cpp:111-132), set when the call
//   first appears (tools/server/server-task.cpp:175). The first stream delta of a call has id, name and "{".
//   finish_reason is "tool_calls" when the final message has calls (server-task.cpp:424,466).
// - Token text: control tokens give "" and user-defined tokens (<think>, <tool_call>, ...) give their text
//   (src/llama-vocab.cpp:3714; preserved tokens, tools/server/server-context.cpp:3974).
//
// Where this parser differs from llama.cpp's final parse:
// - Text that llama.cpp's grammar never lets the model write (this engine samples without a grammar). Once a
//   call's id and name are sent, the call is never removed. Parameters may come in any order; unknown parameter
//   names are accepted (typed like a parameter with no schema); missing required parameters are accepted; a
//   value that is not valid JSON for a non-string type becomes a JSON string. A structural break inside a call
//   (e.g. no "\n" after a tag) closes the call with "}" and drops the rest of the text. llama.cpp drops such a
//   call and all text after it.
// - JSON values are read as strict JSON: whitespace only space, \t, \n, \r; no raw control character and no
//   invalid UTF-8 inside strings. llama.cpp's PEG accepts those and copies them, which gives invalid JSON. Here
//   the value becomes a string. For a mixed-type value llama.cpp's grammar allows such text.
// - Calls cut by EOS keep llama.cpp's text, which is not valid JSON. This is on purpose: clients must not run a
//   truncated call. Every call that reached "</function>\n" has valid JSON arguments.
// - Output that ends with the start of a tag ("<", "</thi", "\n" in a value) and then an incomplete UTF-8
//   character: llama.cpp's trie reports a partial tag, the next literal fails, and the reasoning moves into the
//   content with "<think>\n" in front, or the open call disappears. Here only that tail is dropped. This was
//   the only difference in 34,000 random well-formed outputs cut at random bytes (12 cases).
// - Invalid UTF-8 in a string argument becomes U+FFFD (llama.cpp throws). Invalid UTF-8 right after a "<" that
//   starts a tag does not stop the text (llama.cpp's trie stops there, with the effects above).
// - parallel_tool_calls=false and tool_choice are not supported (pass tools=null for "none").
#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "chat_template.h"  // ojson

namespace q27 {

class Tokenizer;

struct ToolCallDelta {
  int index = 0;           // 0-based index of the call in this message
  std::string id;          // non-empty only in the first delta of a call
  std::string name;        // non-empty only in the first delta of a call
  std::string arguments;   // text to append to the call's arguments string
};

struct ParseDelta {
  std::string reasoning;   // text to append to reasoning_content
  std::string content;     // text to append to content
  std::vector<ToolCallDelta> tool_calls;
  bool empty() const { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

class StreamParser {
 public:
  // thinking: the prompt ended inside <think> (start in reasoning). tools: the request's "tools" array (null or
  // empty = no tool parsing: tool-call text stays content, as llama.cpp does without tools).
  StreamParser(const Tokenizer& tok, bool thinking, const ojson& tools);
  ~StreamParser();
  StreamParser(const StreamParser&) = delete;
  StreamParser& operator=(const StreamParser&) = delete;

  ParseDelta push(int token);               // one generated token (never the EOG token)
  ParseDelta push_text(std::string_view s); // raw output bytes (same path as push; for tests and replays)
  ParseDelta finish();                      // end of generation: flush held-back text, close what is open

  // Final message, equal to llama.cpp's non-streaming parse of the same text:
  const std::string& reasoning() const;
  const std::string& content() const;
  ojson tool_calls() const;  // [{"id","type":"function","function":{"name","arguments": JSON string}}]
  // Calls that reached "</function>\n" (the others were cut by EOS and have unterminated arguments).
  int complete_tool_calls() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace q27
