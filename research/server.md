# Server: the smallest OpenAI-compatible server for OpenCode, dsh and pi

Research only. Date: 2026-10-05. No server was started and no GPU work was done.
One change was made outside reports: `jinja2 3.1.6` and `markupsafe 3.0.4` were
installed into the project venv with `uv`, to render the chat template.

## Summary

1. **One endpoint does the work:** `POST /v1/chat/completions` with SSE streaming. Also needed: `GET /health` (public, no key), `GET /props` (`default_generation_settings.n_ctx`, `modalities.vision`), `POST /tokenize` (bench), `GET /v1/models` (dsh "fetch models" button only). Request fields really sent: `messages` (system or **developer**, user with `image_url` data URIs, assistant with `reasoning_content` and `tool_calls`, tool), `tools`, `stream`, `stream_options.include_usage`, `max_tokens`, `chat_template_kwargs {enable_thinking, preserve_thinking}` (dsh/pi), rarely `temperature`. **No harness sends `reasoning_effort`, `tool_choice`, `top_p`, `top_k`, `parallel_tool_calls` or `response_format` in normal use.**
2. **Template:** XML-style tool calls (`<tool_call><function=…><parameter=…>`), not JSON. The generation prompt opens `<think>\n`. `preserve_thinking` defaults to true, so reasoning stays in every past turn. `reasoning_effort` accepts only `xhigh` (default), `medium`, `low`. **`high` raises a template exception** (verified with jinja2). `LEEME.md:69` and `arranca.ps1:29` list `high` as valid; that is wrong. Hard-coding this one template in C++ (~300 lines) and testing it byte-for-byte against jinja2 renders is the cheapest safe path.
3. **Tool-call parsing:** llama.cpp re-parses the whole output on every token with a PEG parser and adds a lazy grammar. For one format, a small state machine is enough. `<think>`, `</think>`, `<tool_call>`, `</tool_call>` are single special tokens, so most state changes happen on token ids.
4. **Tokenizer:** port llama.cpp's BPE path (qwen35 splitter + byte-level BPE + special-token split, ~1k lines plus ~2.3k lines of Unicode tables). Test against llama.cpp and against HF `tokenizers`.
5. **Prompt cache:** the production log shows **99 of 114 slot reuses with `f_keep = 1.000`**. The harnesses keep the prefix stable, including the re-rendered previous assistant turn. Prefix breaks only at compaction, at system-prompt or tool-list changes, after aborted turns, and once a day in OpenCode (date in the system prompt). Proposed design: one compute slot, several resident sequences in one KV pool, 150 MiB GDN checkpoints at 4 kinds of positions, whole-sequence swap to RAM.
6. **Reuse:** cpp-httplib + nlohmann `ordered_json` + stb_image. **Write:** renderer, stream parser, sequence/checkpoint manager, OpenAI glue. Own code estimate 4-6k lines, against ~41k lines in llama.cpp's server + chat + jinja layers.

---

## 1. API surface each harness uses

### 1.1 Who talks to the server

| Client | Version on this PC | HTTP layer | Config |
|---|---|---|---|
| OpenCode | 1.18.31 (`npm/node_modules/opencode-ai/package.json`) | `@ai-sdk/openai-compatible` 2.0.41, bundled in `opencode.exe` (version from OpenCode's `packages/opencode/package.json:71` at tag v1.18.31) | `OPENCODE_CONFIG_CONTENT` in `qwen38_27\opencode-qwen.ps1:63-99` |
| dsh | 0.1.5-rc.2 | provider `llm-pi-ai` → `@earendil-works/pi-ai` 0.85.1, api `openai-completions` | `~\.dsh\settings.yaml:13-35` |
| pi | **not installed on this PC** | `@earendil-works/pi-coding-agent` 1.0.3 (bin `pi`) uses `@earendil-works/pi-ai` ^1.0.3, the same code path as dsh | `~/.pi/agent/models.json` (pi docs `models.md:45-64`) |
| Scripts | — | PowerShell / Python urllib | `arranca.ps1`, `mide-tps.py`, `prueba-imagen.py` |

How "pi" was checked: `npm ls -g` on Windows and in WSL Ubuntu 24.04 list only dsh, opencode-ai, pnpm, temu. No `pi` on PATH. No `~/.pi`. It may live on the laptop. Unknown.

pi has two ways to reach a llama.cpp server:
- **Built-in `llama.cpp` provider.** It requires llama.cpp **router mode**: `GET /models` must return objects with `status.value` (`extensions/llama/client.js:144-150`, error "Server is not running in llama.cpp router mode"). The current `arranca.ps1` runs single-model mode (`-m`), so this path fails today. Its model compat is `supportsDeveloperRole: false`, `supportsStore: false`, `supportsStrictMode: false`, `thinkingFormat: "qwen-chat-template"` (`extensions/llama/provider.js:72-99`).
- **`models.json` custom provider** with `api: "openai-completions"`. Same requests as dsh.

Either way, chat requests come from pi-ai's `openai-completions.js`. Supporting dsh covers pi. A fake router `/models` listing (one model, `status.value: "loaded"`, `meta.n_ctx`) costs ~40 lines and makes pi's zero-config provider work too.

### 1.2 Endpoints

| Endpoint | Used by | Fields read | Source |
|---|---|---|---|
| `GET /health` | `arranca.ps1:127`, `opencode-qwen.ps1:28,38`, `arranca-dsh.ps1:29,39` | `.status == "ok"`. Called **without** the API key. | llama.cpp keeps it public: `server-http.cpp:251-253` |
| `GET /props` | `opencode-qwen.ps1:50-52` (with key) | `default_generation_settings.n_ctx`, `modalities.vision` | llama.cpp `server-context.cpp:4797-4823` |
| `GET /props` | pi built-in provider | `chat_template` (checks it contains `enable_thinking`), `models_autoload` | `client.js:155-162`, `provider.js:74` |
| `POST /v1/chat/completions` | everyone | see 1.3 / 1.4 | |
| `POST /tokenize` | `mide-tps.py:21` | body `{"content": text}`, reads `tokens` | llama.cpp defaults: `add_special=false`, `parse_special=true` (`server-context.cpp:5282-5287`) |
| `GET /v1/models` | dsh, only for the "fetch available models" action in settings | standard `data` array | `dsh-llm-pi-ai/lib/index.js:2109-2164` |
| `GET /models`, `POST /models/load` | pi built-in provider (router mode) | `data[].id`, `data[].status.value`, `meta.n_ctx`, `architecture.input_modalities` | `client.js:144-180`, `provider.js:46-56,85` |
| `POST /apply-template`, `POST /completion` (`n_predict:1`, `n_probs`) | pi "classifier" models only (codemode scripts, extensions) | `top_logprobs` | `pi-ai 1.0.3 dist/api/llama-cpp-classify.js:13-15,223-298`. Optional. |

The `model` field must be ignored. OpenCode sends `qwen3.8-27b`, dsh sends `qwen38-27b-local` (`settings.yaml`), the scripts send none.

### 1.3 Request fields actually sent

| Field | OpenCode (AI SDK 2.0.41) | dsh (pi-ai 0.85.1 + `settings.yaml`) | pi built-in (pi-ai 1.0.3) | Scripts |
|---|---|---|---|---|
| `stream` | true (`openai-compatible-chat-language-model.ts:365-373`) | true (`openai-completions.js:587`) | true | false |
| `stream_options` | `{include_usage:true}`; OpenCode forces `includeUsage` (`provider.ts:1755-1756`) | `{include_usage:true}` (`:594-596`) | same | — |
| `max_tokens` | 32000 = min(OUTPUT_TOKEN_MAX 32000, limit.output 32768) (`transform.ts:18,1737`) | min(65536, ctx − estimate − 4096) (`simple-options.js:4-9,17`; `settings.yaml:23,32`) | min(ctx, …) | 8 / 120 / `--gen` |
| `temperature`, `top_p`, `top_k` | not sent for Qwen ids (`transform.ts:527-570` return undefined) | `temperature` only if the agent sets it (`dsh-llm-pi-ai/index.js:1869`) | same | `mide-tps.py:40-42`: temp 1.0, top_p 0.95, top_k 20, `cache_prompt` |
| system role | `system`, one message, parts joined with `\n` (`session/llm/request.ts:57-77,103-111`) | **`developer`**: detected `supportsDeveloperRole` is true for a local URL (`openai-completions.js:910-912,1279`) | `system` | — |
| user images | `{type:"image_url", image_url:{url:"data:<mime>;base64,…"}}` or a URL (`convert-to-openai-compatible-chat-messages.ts:60-77`) | data URI (`:933-948`) | same | data URI (`prueba-imagen.py:6-8`) |
| assistant reasoning | `reasoning_content` when non-empty (`convert…:203-209`); OpenCode keeps reasoning parts for the same model (`message-v2.ts:362-375`) | `reasoning_content` (field name taken from the stream, `:1001-1007`) | same | — |
| assistant tool calls | `tool_calls[{id,type:"function",function:{name,arguments: JSON string}}]` (`convert…:175-197`) | same (`:1019-1040`) | same | — |
| tool results | `role:"tool"`, `tool_call_id`, string content (`convert…:214-245`) | same; images from tool results go in a following user message "Attached image(s) from tool result:" (`:1079-1125`) | same | — |
| `tools` | `{type:"function", function:{name, description, parameters}}`, sorted by name (`request.ts:183`) | same plus `strict:false` (`:1146-1176`) | no `strict` | — |
| `tool_choice` | only `"required"` when the user asks for JSON-schema output (`session/prompt.ts:1285`) | never (dsh passes no toolChoice, `index.js:1867-1874`) | never | — |
| `chat_template_kwargs` | not sent | `{enable_thinking: <effort != off>, preserve_thinking: true}` (`:654-659`; `settings.yaml:31`) | same | — |
| `reasoning_effort` | not sent: Qwen ids get no effort variants (`transform.ts:825-842`) | not sent (`supportsReasoningEffort:false`, `settings.yaml:33`) | not sent | — |
| `parallel_tool_calls` | not sent | not sent | not sent | — |
| `response_format` | `{type:"json_object"}` only in `opencode agent create` (`agent/agent.ts:7,416-435`) | not sent | not sent | — |
| other | headers `x-session-affinity`, `X-Session-Id`, `x-parent-session-id` (`request.ts:187-203`) | `store:false` (`:597-599`) | — | — |

Consequences:
- **Effective reasoning effort today is always the server default (`medium`)** from `--chat-template-kwargs` (`arranca.ps1:114`). In dsh, choosing low/medium/high changes nothing; only "off" changes the prompt (`enable_thinking:false`).
- The server must **map `developer` to `system`**. The template raises "Unexpected message role." otherwise (verified with jinja2). llama.cpp does this mapping (`common/chat.cpp:1280-1283`).
- Request `chat_template_kwargs` merge key by key over the server default (llama.cpp `server-common.cpp:1355-1361`). Top-level `reasoning_effort` goes into the kwargs; `"none"` disables thinking (`server-common.cpp:1372-1383`).
- OpenCode may send 2 system messages if a plugin adds one (`request.ts:73-77`). The template raises "System message must be at the beginning." on a second system message. Merging leading system/developer messages is a cheap fix (a deviation from llama.cpp).

### 1.4 Response fields the clients read

| Field | OpenCode (AI SDK) | dsh / pi (pi-ai) | Scripts |
|---|---|---|---|
| `delta.content` | yes (`…language-model.ts:500-520`) | yes (`openai-completions.js:399-410`) | — |
| `delta.reasoning_content` | yes, or `delta.reasoning` (`:483-498`) | first non-empty of `reasoning_content`, `reasoning`, `reasoning_text` (`:415-440`) | — |
| `delta.tool_calls[]` | **first delta of each call must carry `id` and `function.name`**, else it throws `InvalidResponseDataError` (`:535-548`). The call counts as finished once the accumulated `arguments` parse as JSON (`:586,637`). | tracked by `index` or `id`; args accumulated, parsed at the end (`:441-469`) | — |
| `finish_reason` | `stop`, `length`, `tool_calls` | **required**: missing → "Stream ended without finish_reason"; unknown value → error (`:494-502,1208-1230`) | — |
| `usage` | `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens` (`convert-openai-compatible-chat-usage.ts:35-53`) | same, plus `prompt_cache_hit_tokens` / `cached_tokens` fallbacks (`:1178-1207`) | — |
| `timings` | ignored | ignored | `mide-tps.py:44-54`: `prompt_n`, `prompt_per_second`, `predicted_n`, `predicted_per_second`, `predicted_ms`, `draft_n`, `draft_n_accepted`. `arranca.ps1:151`: `predicted_per_second`. `prueba-imagen.py`: `prompt_n`, `predicted_n`. |
| non-stream `choices[0].message` | — | — | `content` (`prueba-imagen.py`) |

llama.cpp's shapes, to copy: final chunk with `finish_reason`, then (if `include_usage`) a chunk with `choices: []` and `usage`, then `data: [DONE]` (`server-task.cpp:465-525`). `usage.prompt_tokens` is the whole prompt; `cached_tokens` is the reused part (`server-task.cpp:365-371`). `finish_reason` is `tool_calls` when any tool call was parsed (`server-task.cpp:417-420`). `timings` object fields: `server-common.cpp:83-104`.

### 1.5 Constraints found in client code

1. **dsh idle timeout: 300 s between stream events** (`dsh-llm-pi-ai/index.js:877,1061`; watchdog `dsh-timeout/lib/index.js:85-121`). The `start` event yields nothing (`toStreamChunks`, `index.js:1455+`). SSE comment pings do not reset it, because the watchdog counts pi-ai events, not bytes. So queue wait + prefill must stay under 300 s. Today a full cache miss at 150k tokens takes 150,000 / 439 t/s ≈ 342 s (estimate from the brief's prefill rate), which already exceeds it. The fix on the client side is `streamIdleTimeoutMs` in the provider profile (`index.js:1010`).
2. **OpenCode: 300 s header timeout and 300 s between SSE byte reads** (`provider.ts:35,1799-1800`, `wrapSSE` `:36-78`). Here SSE comment pings do help. llama.cpp sends headers only after the first result (`server-context.cpp:4577-4605`), so a request queued behind a long one can hit the header timeout. The new server should send headers right after validation and send `:\n\n` pings (llama.cpp does pings every 30 s, `server-task.h:58`, `server-context.cpp:4659-4673`).
3. **Context-overflow wording must match llama.cpp.** pi-ai detects `/exceeds the available context size/i` (`pi-ai/dist/utils/overflow.js:48`) and dsh then compacts and retries. llama.cpp text: "request (N tokens) exceeds the available context size (M tokens), try increasing it", HTTP 400, type `exceed_context_size_error` (`server-context.cpp:3332-3335`, `server-common.cpp:67-70`). OpenCode also detects overflow (`provider/error.ts:175`).
4. Client disconnect must cancel generation (dsh abort, OpenCode Esc). pi-ai then drops the aborted assistant message from history (`transform-messages.js:159-161`).
5. Generation stops on `<|im_end|>` (248046, the GGUF `eos_token_id`) and `<|endoftext|>` (248044), as llama.cpp's EOG set does (`src/llama-vocab.cpp:2920-2940`).

---

## 2. The chat template (`research/_chat_template.jinja`)

### 2.1 What it does

| Lines | Behaviour |
|---|---|
| 3-41 | `render_content`: a string, or a list of parts. An image part becomes `<|vision_start|><|image_pad|><|vision_end|>` (optional `Picture N: ` prefix if `add_vision_id`). Video likewise. Images in a system message raise an exception. |
| 45-56 | Reasoning effort, only when thinking is on (`enable_thinking` undefined or true). Default `xhigh`. Allowed: `xhigh`, `medium`, `low`. Anything else raises "Unexpected reasoning effort …". `xhigh` and `low` add one instruction sentence. **`medium` adds nothing.** |
| 57-75 | With tools: one system turn = reasoning sentence + `# Tools` + each tool as `tojson` on its own line inside `<tools>` + the XML call format instructions + the system message text (trimmed) at the end. **Tools come before the system prompt text.** |
| 76-87 | Without tools: system turn = reasoning sentence (if any) + system text. |
| 88-101 | Finds the last real user query (a user message that is not only `<tool_response>…</tool_response>`). Raises "No user query found" if none. |
| 104-107 | A system message that is not first raises an exception. |
| 108-109 | User: `<|im_start|>user\n` + content **trimmed** + `<|im_end|>\n`. |
| 110-146 | Assistant: if `preserve_thinking` is undefined or true, or the turn is after the last query: `<think>\n` + `reasoning_content|trim` + `\n</think>\n\n` + content (trimmed). Otherwise content only. Tool calls follow: `\n\n` before the first call only if content is non-empty; `\n` between calls. Each call: `<tool_call>\n<function=NAME>\n` then per argument `<parameter=K>\n` + value + `\n</parameter>\n`, then `</function>\n</tool_call>`. String values are inserted raw; other values with `tojson`. `arguments` must be a mapping (`|items`). |
| 147-158 | Tool results: consecutive tool messages share one `<|im_start|>user` turn, each wrapped as `\n<tool_response>\n…\n</tool_response>`. |
| 163-170 | Generation prompt `<|im_start|>assistant\n<think>\n`. With `enable_thinking=false`: `<|im_start|>assistant\n<think>\n\n</think>\n\n`. |

**Tool-call format: XML-style, not JSON.** The name and each parameter are tags; values are raw text (strings) or JSON text (other types).

### 2.2 Verified with jinja2 3.1.6 (project venv)

Rendered the GGUF template with HF-style settings (`trim_blocks`, `lstrip_blocks`, `tojson` = `json.dumps(ensure_ascii=False)`):

| Input | Result |
|---|---|
| no `reasoning_effort` | system turn starts "Reasoning effort is set to xhigh." |
| `xhigh` | same |
| **`high`** | **exception: "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."** |
| `medium` | no reasoning sentence; system turn starts `# Tools` |
| `low` | "Reasoning effort is set to low." |
| `enable_thinking=false` | prompt ends `<think>\n\n</think>\n\n` |
| `preserve_thinking=false` | assistant turns before the last user query lose the whole `<think>` block |
| role `developer` | exception "Unexpected message role." |

llama.cpp does not remap `high` either. Its jinja only copies the value into `reasoning_effort` and `reasoning_strength` (`common/jinja/caps.cpp:29-33`). So **`$Thinking = "high"` in `arranca.ps1` would make every request fail.** `LEEME.md:69` and `arranca.ps1:29` should not list `high`. For the new server, a mapping table is a design choice to make: for example `high`/`max` → `xhigh`, `minimal` → `low`, `none`/`off` → `enable_thinking=false`, other values → HTTP 400.

### 2.3 Hard-code it, or embed a Jinja engine?

| Option | Size | Pros | Cons |
|---|---|---|---|
| **Hard-coded C++ renderer** | ~250-350 lines (estimate) | Fast (no AST walk over 180k tokens of history). Can emit segments (special vs text) directly. Easy to read. | Must replicate `trim`, `tojson` formatting, and the exceptions. Breaks silently if the GGUF template changes. |
| llama.cpp `common/jinja` | 6,374 lines (`common/jinja/*.cpp,h`) + `common_json` wrapper | Runs this template in production today. | Generic engine, depends on llama.cpp `common`. |
| minja (google/minja, header-only) | ~3k lines (estimate; llama.cpp replaced it with `common/jinja`, `common/jinja/README.md:1-3`) | Header-only. | Not maintained by llama.cpp anymore. |

**Recommendation: hard-code**, with three guards:
1. At load, hash `tokenizer.chat_template` from the GGUF (8,952 chars, `_gguf-model-dump.txt:50`). Refuse to start, or fall back, if the hash differs.
2. A fixture test: a Python script in the venv renders N JSON conversations with jinja2 and the GGUF template; the C++ renderer must produce byte-identical strings. Fixtures: no system; system with and without tools; each effort; `enable_thinking` false; reasoning with leading/trailing spaces; content `null`/`""`; content before a tool call; 1, 2, 3 parallel tool calls; string, integer, float, bool, null, array and object arguments; unicode in tool schemas (`ensure_ascii=False`); floats in schemas (Python `repr` vs C++ shortest round-trip); consecutive tool messages; images in user messages; the pi-ai "Attached image(s) from tool result" message; developer role (mapped before render); and every exception case.
3. A second oracle: llama.cpp's `POST /apply-template` on the same fixtures, run when the production server is idle and the user allows it.

`tojson` details to match: separators `", "` and `": "`, key order as received (llama.cpp uses `nlohmann::ordered_json`, `common/json.h:18-25`, `common/json.cpp:14`), non-ASCII left raw, control characters as `\n`, `\t`, `\u00XX`.

Tool-call `arguments` arrive as a JSON string from all clients. They must be parsed to an object before rendering (llama.cpp workaround `func_args_not_string`, `common/chat.cpp:1292-1294`). If they do not parse, a policy is needed (HTTP 400, or render as-is). Unknown what llama.cpp's template run does there; it probably fails on `|items`.

---

## 3. Tool-call parsing while streaming

### 3.1 How llama.cpp does it

- **Detection:** a template containing `<tool_call>`, `<function=`, `<parameter=` selects the "Qwen3-Coder" handler (`common/chat.cpp:1218-1226`).
- **Handler:** `common/parsers/qwen3-coder.cpp` (194 lines) builds a PEG grammar:
  - reasoning = optional `<think>` + text until `</think>` or `<tool_call>` (`:77-82`). A tool call without `</think>` also ends reasoning (`thinking_end_tags`, `:28`).
  - content = text until `<tool_call>`; then 0..n tool calls (`:166-170`). `parallel_tool_calls` defaults to the template capability (`server-common.cpp:1321`), which is true for this template (`jinja/caps.cpp:399-492` checks that a second call renders).
  - per function: `<function=NAME>\n`, parameters in any order for required ones (`p.permute`, `:142-146`), `</function>\n`, `</tool_call>` (`:149-159`).
  - string-typed parameters: value = text until `\n</parameter>\n` (`:91-93`). Non-string types: JSON value; mixed types try JSON first, then string (`:108-135`).
- **Grammar:** with tools and `tool_choice != none`, a **lazy grammar** is built and triggered by `<tool_call>` (`:179-191`). From that word on, sampling is constrained to valid tool names, parameter names and JSON values. This is on today for every OpenCode and dsh request with tools.
- **Streaming:** on every new piece of text, the server re-parses the **whole** generated text with `is_partial=true` (`server-task.cpp:163-237`, `common/chat.cpp:1453-1531`), computes a diff against the previous parse (`common/chat.cpp:267-330`, comment "TODO: these can become expensive for long messages") and sends deltas. A tool call's name and id are held back until its arguments start, so the first tool delta carries `id` + `name` (`server-task.cpp:181-235`). This satisfies the AI SDK rule in 1.4.
- **Arguments as JSON text:** the mapper writes compact JSON: `{"path":"a.py","limit":50}`. String values are streamed with JSON escaping as they arrive; the closing quote and `}` come at the close tags (`common/chat-peg-parser.cpp:385-446`).
- **Size of this machinery:** `common/chat*`, `peg-parser.*`, `parsers/*` = 11,813 lines, plus `json-schema-to-grammar` for the grammar.

### 3.2 Proposed parser for this one format

Special tokens in the vocab (read from the GGUF with `gguf`): `<think>` 248068, `</think>` 248069, `<tool_call>` 248058, `</tool_call>` 248059 are single **user-defined** tokens. `<function=`, `<parameter=`, `</parameter>`, `</function>` are plain text.

States:

| State | Leaves on | Emits |
|---|---|---|
| REASONING (start state when thinking is on) | token `</think>` → CONTENT; token `<tool_call>` → CALL_OPEN | `reasoning_content` deltas; drop the leading `\n` and trailing `\n` around the tags |
| CONTENT (start state when thinking is off) | token `<tool_call>` → CALL_OPEN | `content` deltas |
| CALL_OPEN | text `\n<function=NAME>\n` | first tool delta: `index`, `id` (`call_` + random), `type`, `function.name`, `arguments: "{"` |
| PARAMS | `<parameter=K>\n` → VALUE; `</function>\n` → CALL_CLOSE | `"K":` (with `,` after the first) |
| VALUE | text `\n</parameter>\n` | string params: escaped chunks inside quotes as they arrive; other types: the whole value at the close tag, typed with the tool's JSON schema (try JSON, else string, like llama.cpp `:108-135`) |
| CALL_CLOSE | token `</tool_call>` → AFTER_CALL | `}` |
| AFTER_CALL | whitespace, then `<tool_call>` → CALL_OPEN; EOS → done | — |

Holdback rules:
- Detokenized bytes can end inside a UTF-8 character. Hold incomplete sequences.
- In VALUE, hold back any tail that is a prefix of `\n</parameter>` (at most 13 bytes).
- In CALL_OPEN and PARAMS, hold text until the tag is complete. Names are short.
- No holdback is needed for `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, because they are single tokens. This removes most of llama.cpp's partial-parse work.

Failure handling: if the text inside a tool call does not match (unknown structure, EOS before `</tool_call>`), close the JSON as llama.cpp does (`chat-peg-parser.cpp:435-447`) or turn the raw text into content. Text after the last `</tool_call>` is outside the format; treat it as content. `finish_reason`: `tool_calls` if any call was emitted, `length` at `max_tokens` or full context, else `stop`.

Estimate: 400-600 lines of C++ with tests. Cost per token: O(new bytes), against llama.cpp's O(message length) re-parse.

**Grammar decision (open):** dropping the lazy grammar changes behaviour against the baseline. The model is trained on this format, but nobody has measured the malformed-call rate without the grammar. Plan: log every parse failure and every call to an unknown tool or with a missing required parameter, on real sessions, before deciding. `tool_choice:"required"` (OpenCode JSON-schema mode) and `response_format:json_object` (`opencode agent create`) also rely on grammars in llama.cpp. Both are rare. Without a grammar, they become best effort.

---

## 4. Tokenizer

### 4.1 Facts (read from the model GGUF in this session)

| Item | Value |
|---|---|
| `tokenizer.ggml.model` / `pre` | `gpt2` / `qwen35` (`_gguf-model-dump.txt:41-42`) |
| tokens / merges | 248,320 / 247,587 |
| token types | 248,044 normal, 27 control, 6 user-defined, 243 unused |
| user-defined | `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>` |
| control (selection) | `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056, `<|video_pad|>` 248057 |
| eos / bos / pad | 248046 / 248044 / 248044; `add_bos_token` false |

Pre-tokenizer regex (from tokenizer.json, quoted in `src/llama-vocab.cpp:392-397`):
`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
It differs from Qwen2 by `\p{M}` (combining marks) in letter runs. llama.cpp implements it by hand, without std::regex: `unicode_regex_split_custom_qwen35` (`src/unicode.cpp:608-670`), selected at `src/unicode.cpp:1063-1064`.

Special-token split rule in llama.cpp: user-defined tokens are always matched in raw text; control tokens only with `parse_special` (`src/llama-vocab.cpp:3289-3300`). The chat prompt is tokenized with `parse_special=true`, so a literal `<|im_start|>` or `<think>` inside user text or a tool result becomes the special token. llama.cpp's jinja can mark input strings (`common/jinja/README.md:24-80`, `mark_input` default true at `common/chat-auto-parser.h:73`), but the server flattens the result to one string (`common/chat.cpp:953-956`), so the marks are not used for tokenization. HF `tokenizers` also matches added tokens in raw text. Keep this behaviour for parity.

### 4.2 Options

| Option | Size | Notes |
|---|---|---|
| **Port from llama.cpp** | BPE session (`src/llama-vocab.cpp:280-700`, part of it), qwen35 splitter (~60 lines), special-token partition (~150 lines), Unicode flag table and whitespace set (`src/unicode-data.cpp:10-2314`, ~2.3k generated lines). NFD and case tables are not needed. | Identical output to the baseline by construction. MIT. |
| Small library | HF `tokenizers` needs Rust (tokenizers-cpp); PCRE2 with UCP could run the regex | Extra toolchain or dependency. Unicode tables may differ from llama.cpp's on rare code points. |
| Write from scratch | ~600-900 lines + a generated category table | Same work as porting, without the parity guarantee. |

**Recommendation:** port llama.cpp's BPE path into one file, keeping its Unicode tables. Use a rank hash map keyed by token-id pairs.

### 4.3 Correctness test plan

1. **Splitter goldens:** llama.cpp has `models/ggml-vocab-qwen35.gguf` with `.inp` (50 strings) and `.out`. That test vocab has **151,936 tokens, not 248,320** (checked with `gguf`). Its ids do not apply to our model. Its strings and the pre-tokenizer still do: compare our split against llama.cpp's split on them.
2. **Id goldens vs llama.cpp:** tokenize a corpus with llama.cpp on this exact GGUF (`/tokenize` on an idle server, or a small CPU-only tool linked to `llama-vocab`, later). Corpus: the real code corpus used by `mide-tps.py`, rendered template fixtures from 2.3, CJK, Arabic and Devanagari with combining marks, emoji ZWJ sequences, runs of spaces, tabs, `\r\n`, contractions (`'S`, `'ll`), long digit runs, invalid UTF-8 bytes, special-token strings inside text with `parse_special` on and off.
3. **Independent oracle:** build an HF `tokenizer.json` from the GGUF (byte-level BPE, the merges, the regex above, ByteLevel decoder, the 33 added tokens) and tokenize with Python `tokenizers` in the venv. Disagreements between llama.cpp and HF are worth a note; parity target stays llama.cpp.
4. **Round trip:** `detokenize(tokenize(s)) == s` for random byte strings.
5. **Fuzz:** 10^5 random Unicode strings, engine vs llama.cpp.
6. **Speed:** time tokenization of a 180k-token prompt on the 9600X. Unknown today. If it is more than a few tens of ms, use incremental tokenization (see 5.4).

---

## 5. Prompt cache across requests

### 5.1 llama.cpp today

| Mechanism | What it does | Source |
|---|---|---|
| Slot reuse | New prompt is matched to the slot by longest common prefix (LCP) of token ids. If `f_keep = LCP / slot tokens < 0.5`, the slot state is first saved to the RAM cache. | `server-context.cpp:1638-1708` |
| RAM prompt cache (`-cram`) | Saves the **full** sequence state (all KV + GDN state + MTP draft KV) plus its checkpoints. Loads the cached prompt with the best `f_sim` that keeps ≥ 25% of itself. FIFO eviction by size. Entries larger than the limit are skipped. | `server-task.cpp:1722-1907`; trigger `server-context.cpp:1691-1708` |
| Rollback for recurrent state | GDN state cannot be cut back. On a partial match, the newest checkpoint at or before the match is loaded (`PARTIAL_ONLY`), else the prompt is processed from 0 ("forcing full prompt re-processing"). | `server-context.cpp:3438-3528` |
| Checkpoint creation | Only for recurrent/SWA models. Batches stop at user-message starts (spans from `message_delimiters`), always at the last user message, and at `4 + n_ubatch` and `4` tokens before the prompt end. Min spacing `checkpoint_min_step`. Oldest evicted beyond `-ctxcp`. | `server-context.cpp:3620-3634, 3719-3753, 3796-3807, 2425-2490`; delimiters `parsers/qwen3-coder.cpp:32-38`, `server-context.cpp:4498-4522` |
| Defaults | 32 checkpoints, min step 8192 tokens, `-cram` 8192 MiB. Production: `-ctxcp 4`, `-cram 8192`. | `common/common.h:633-636`, `arranca.ps1:83,88` |
| Checkpoint content | GDN state **plus the whole MTP draft KV** (LEEME: the draft cache ignores `PARTIAL_ONLY`). Log: 221.3 MiB at 18,254 tokens; 487.6 MiB at 86,026; 661.5 MiB at 130,287. | `arranque.log.err:91,3497-3498,3807-3809`; `LEEME.md:384-387` |

Check of that size (arithmetic): GDN state 149.6 MiB (see 5.4) + 18,254 × 4 KiB = 71.3 MiB → 220.9 MiB. Matches 221.3 MiB.

### 5.2 Evidence from the production log (`qwen38_27\arranque.log.err`, 3 Oct 2026)

- 114 slot selections by LCP. **99 have `f_keep = 1.000`** (≥ 0.9995). 8 have 0.95-0.999; 6 are lower. 56 selections were by LRU (new prompts, mostly bench runs). Counts from `grep` in this session.
- Example of an agent loop (lines 60-90): the slot ends at 26,142 tokens (prompt + generated). The next request has `f_sim 0.883`, `f_keep 1.000`, and processes only 3,460 new tokens in 5.8 s. The next: slot 30,010, only 916 tokens processed. The next: 400 tokens.
- So the previous assistant turn, re-rendered from `reasoning_content` + `tool_calls` JSON and re-tokenized, matched the **generated token ids exactly**. With `preserve_thinking` on and these harnesses, no checkpoint is needed in a normal tool loop.
- Lines 133-160: a different short conversation (334 tokens) took the slot by LRU, then the main 31k-token state came back from the RAM cache (only 768 tokens processed).
- `-cram 8192` overflowed 10 times ("prompt state size 8529 MiB exceeds cache size limit", line 711, ~100k tokens). Above ~90-100k, a side request costs a full re-read of the main conversation. LEEME says the same (`LEEME.md:170-174`).
- The log does not record which harness sent each request. Some of the 0.95-0.999 cases are bench runs (`mide-tps.py` prompts that differ only at the end).

### 5.3 Where each harness breaks the prefix

| Event | OpenCode | dsh | pi | Where the prefix breaks |
|---|---|---|---|---|
| Normal tool loop | append-only | append-only by design ("KV Cache effect" sections in each plugin README, e.g. `dsh-agent-loop/README.md:158-160`, `dsh-time-context/README.md:126-128`) | append-only | nowhere (5.2) |
| Reasoning in history | kept, sent as `reasoning_content` (`message-v2.ts:362-375`, `convert…:206`); `preserve_thinking` undefined → true | kept (`preserve_thinking:true`, replay of `thinkingSignature` `dsh-llm-pi-ai/index.js:198-202`) | kept | nowhere |
| Tool output truncation | `prune` exists but is off unless `compaction.prune` is set (`compaction.ts:273-317`, check at `:275`); not set in `opencode-qwen.ps1` | at compaction time only: tool results > 8,192 chars cut to head 4,096 + tail 1,024 (`dsh-base/cordis.patch.yml:394-399`; README `:58-60`) | — | at the first trimmed result, which can be early in the history |
| Compaction | when tokens ≥ 180,224 − 32,000 = 148,224 (`overflow.ts:8-34`); summary + kept tail | at 0.8 × 180,224 = 144,179 tokens; keeps the newest 0.16 × 180,224 = 28,836 tokens (`dsh-compaction-basic/README.md:62-67`); the summary request replays system + old span byte for byte, so it reuses the warm prefix (`:107,118`) | at 180,224 − 16,384 = 163,840; keeps 20k (`docs/compaction.md:32-45`) | right after the system turn |
| System prompt / tool list change | tools sorted by name (`request.ts:183`); MCP tools appearing later change the list; **`Today's date` line changes once a day** (`session/system.ts:83`) | in-place system replacement breaks at node 0; tool schema change breaks at the first changed schema (`dsh-system-prompt/README.md:149,163`) | pi-ai collapses later system messages into the head (`pi-ai 1.0.3 utils/transcript.js:94-100`) | inside the system turn. The template puts tools **before** the system text, so any tool change breaks almost everything. |
| Plan mode | the plan reminder is appended to the last user message per request and not saved (`session/reminders.ts:26-34`) | — | — | at the previous last user message |
| Aborted turn | — | dsh records it; pi-ai drops aborted assistant messages (`transform-messages.js:159-161`) | same | at the end of the aborted request's prompt |
| Side requests in the one slot | subagents (task tool); titles disabled (`opencode-qwen.ps1:59-61,88-90`) | subagents in-process; compaction summaries | branch summaries | different conversation; needs a second resident sequence or a RAM swap |

`reasoning_content|trim` and the content trim (template lines 103, 115) do not break the match in practice (5.2). A case that can: the model ends reasoning with `<tool_call>` and no `</think>`. The re-render inserts `\n</think>\n\n`, so the match ends inside the last assistant turn.

### 5.4 Proposed design for one compute slot

**Sizes (arithmetic, f32 state as in llama.cpp; to confirm in the engine):**
- GDN state per sequence: SSM 48 layers × 48 heads × 128 × 128 × 4 B = 144.0 MiB; conv 48 × 3 × 10,240 × 4 B = 5.6 MiB. **Total 149.6 MiB.** With the tensor split it is ~75 MiB per card.
- Attention KV is truncatable: no checkpoint needed. The MTP draft KV (4 KiB/token) is an ordinary KV cache too: truncate it, never copy it. **A checkpoint is 150 MiB at any depth**, against 221-662 MiB in llama.cpp today.
- Copy cost (estimate): D2D snapshot 75 MiB per card at ~390 GB/s ≈ 0.2 ms. Restore from pinned RAM at ~3.5 GB/s per card, both cards in parallel ≈ 22 ms.
- Whole-sequence swap: 68 KiB/token (`LEEME.md:351`) × 100k = 6.5 GiB, ~3.3 GiB per card at ~3.5 GB/s ≈ 1 s (estimate). Re-reading 100k tokens at 540 t/s ≈ 185 s.

**Data structures:**
- One KV pool per attention layer (main 16 layers + MTP 1 layer) sized for the full context (≥ 180,224 tokens). Each **resident sequence** owns one contiguous range. Attention kernels take `(base, len)` and read only their own range. This avoids the `-np 2 --kv-unified` slowdown described in `LEEME.md:152-168`, where every request scanned the whole unified cache. Defragment by D2D move when needed (estimate: ~17 ms per 100k tokens per card, read + write).
- Per sequence: token list (ids, plus image placeholders carrying an image hash and their M-RoPE position span, because positions ≠ token index after images), live GDN state (VRAM), list of checkpoints, last-used time, optional client session id.
- Up to ~4 resident sequences (main agent, subagent, compaction/title side request). The sum of their tokens must fit the pool. Swap the LRU sequence to pinned RAM when the pool is short. Keep the main agent resident when possible.

**Checkpoint positions:**

| # | Position | Covers | Lifetime |
|---|---|---|---|
| P1 | end of the system turn (`<|im_end|>\n` after tools + system text) | every compaction in all three harnesses; forks for side requests that share system + tools | one per distinct system turn |
| P2 | start of the last user message | OpenCode plan reminder; "edit last message / regenerate" | newest 1-2 |
| P3 | end of each request's prompt (after `<think>\n`) | aborted turns, `</think>`-less tool calls, rare re-render mismatches | newest 2 |
| P4 | every 16,384 prompt tokens, at the next message boundary | dsh compaction summary request (shares system + oldest span), pi branch summaries, early tool-result trimming | at most 4 per sequence |

Keep the newest 2 checkpoints in VRAM (~150 MiB per card) and the rest in pinned RAM (~8 × 150 MiB = 1.2 GiB per sequence at most). Creating one is a 0.2 ms snapshot plus an async D2H copy that overlaps prefill.

**Request flow:**
1. Validate, render, tokenize. On error, HTTP 400 before any streaming. Prompt ≥ context → the llama.cpp overflow text (1.5).
2. Send HTTP 200 + `text/event-stream` headers at once. Send `:\n\n` every 10-15 s while queued or reading the prompt.
3. Pick the sequence with the longest LCP among resident and swapped ones. Optional hint: OpenCode's `X-Session-Id` / `x-parent-session-id` headers.
4. If LCP ≥ processed length: continue from the live state. Else load the newest checkpoint at or before the LCP, truncate KV and token list, drop newer checkpoints. If no sequence matches beyond a shared P1, **fork**: copy KV range `[0, P1)` into a new range (D2D) and load P1's state. Else start from 0.
5. Prefill the rest, taking P1-P4 snapshots on the way. Snapshot P3 at the prompt end.
6. Decode with MTP. At the end, the live state must be "after the last accepted token" (the decode loop's rollback of rejected drafts is a separate topic).
7. On client disconnect: stop at an accepted-token boundary; keep the token list and live state consistent.
8. FIFO queue for other requests.

**Incremental tokenization (optional):** keep the previous rendered text and its ids per sequence. On a new request, find the common text prefix, back up to the last special token boundary (`<|im_start|>`), reuse the ids before it, and tokenize only the rest. Special tokens split the text before BPE, so a boundary there is safe. This keeps CPU time per request proportional to new text. Needed only if 4.3 step 6 shows tokenization is slow.

**Token splice (optional safeguard):** if a re-rendered past assistant turn equals its generated text byte for byte but re-tokenizes to different ids, reuse the generated ids. 5.2 suggests this is rare; measure first.

**What this changes against today, per case (estimates):**

| Case | llama.cpp today | Proposed |
|---|---|---|
| tool loop, 100k context | only new tokens | same |
| subagent of 20k while main is 100k | main saved to RAM only if ≤ ~90k; else main re-read later: ~185 s | both resident, nothing re-read |
| aborted turn | restore checkpoint ~4 tokens before old prompt end | restore P3 (22 ms) + re-read the partial turn |
| dsh compaction at 144k | summary request: needs a checkpoint inside the history (exists only if one of 4 survives there); then re-read ~29k tail | P4 near the cut; P1 for the new history; tail re-read ~29k tokens ≈ 48 s at 600 t/s |
| OpenCode date change | full re-read | full re-read from the date line (inside the system turn) |

---

## 6. Reuse vs write

| Part | Reuse | Write (estimate, own lines) |
|---|---|---|
| HTTP server, auth, routing, SSE, keepalive, cancel on disconnect, FIFO queue | **cpp-httplib** 0.58.0 (vendored in llama.cpp: `httplib.h` 4,668 + `httplib.cpp` 18,289 lines). Windows and Linux. Chunked content provider for SSE (llama.cpp: `server-http.cpp:666`). Auth as llama.cpp: `Authorization: Bearer` or `X-Api-Key`, `/health` public (`server-http.cpp:251-290`). | 400-600 |
| JSON | **nlohmann::json 3.12.0** (`json.hpp` 25,526 lines), use `ordered_json` so tool-schema key order survives into the prompt | — |
| OpenAI request parsing and response building (stream and non-stream, usage, timings, errors) | — | 600-800 |
| Chat template renderer + fixture tests | jinja2 in the venv as the oracle | 300 + tests |
| Stream parser (reasoning, content, tool calls, schema typing) | — | 400-600 |
| Tokenizer | port llama.cpp BPE + qwen35 splitter + Unicode tables (~2.3k table lines) | 600-900 |
| Sequences, KV ranges, checkpoints, RAM swap | — | 600-900 |
| Image intake (data URI, base64, decode, hash) | **stb_image** (7,998 lines, vendored); base64 from `common/base64.hpp` | 150-250 |
| `/props`, `/health`, `/v1/models` (+ router-style `/models` for pi), `/tokenize`, optional `/apply-template`, `/detokenize` | — | 150-200 |
| **Total own code** | | **~3.8-5.9k lines (estimate)** |

For scale: llama.cpp's `tools/server/*.cpp,h` = 23,168 lines; chat layer (`common/chat*`, `peg-parser.*`, `parsers/*`) = 11,813; jinja = 6,374.

Not needed (no harness uses them): `/v1/responses`, `/v1/messages`, `/infill`, embeddings, rerank, LoRA, `/slots`, `/metrics`, n>1 completions, logprobs (except pi's classifier), grammar and JSON schema (see 3.2 for the open question), web UI, CORS (all clients call from Node/Bun processes).

---

## 7. Open questions and what to measure

1. Malformed tool-call rate without llama.cpp's lazy grammar, on real sessions.
2. Tokenizer time for a 180k-token prompt on the Ryzen 9600X.
3. Pinned host memory limits on Windows WDDM for 8-16 GiB of swap space. Unknown.
4. Whether pi is used (laptop?) and with which config (`models.json` or built-in router provider).
5. Which harness produced the 8 log cases with `f_keep` 0.95-0.999. Turn on per-request LCP logging in the new server from day one.
6. dsh 300 s idle timeout vs cold prefill time: decide whether to raise `streamIdleTimeoutMs` in `settings.yaml` or rely on faster prefill.
7. Policy for unknown `reasoning_effort` values (`high`) and for unparseable tool-call `arguments`.

## Sources

Local, production (read only):
- `qwen38_27\opencode-qwen.ps1:19-110`
- `qwen38_27\arranca-dsh.ps1:22-25,102-126`
- `qwen38_27\arranca.ps1:29,83,88,114-116,147-151`
- `qwen38_27\mide-tps.py:20-54`
- `qwen38_27\prueba-imagen.py:6-16`
- `qwen38_27\LEEME.md:64-69,152-174,351,384-387`
- `qwen38_27\arranque.log.err` (lines 60-160, 711, 3497-3498, 3807-3809; counts by grep)
- `%USERPROFILE%\\.dsh\settings.yaml:13-40`

Local, dsh 0.1.5-rc.2 (`%APPDATA%\\npm\node_modules\@deepseek-ai\dsh\node_modules\`):
- `@earendil-works/pi-ai/dist/api/openai-completions.js:369-502,584-659,881-1176,1178-1318`
- `@earendil-works/pi-ai/dist/api/simple-options.js:4-34`, `transform-messages.js:44-186`, `utils/overflow.js:20,48`
- `@deepseek-ai/dsh-llm-pi-ai/lib/index.js:60-251,379-406,877,1010,1061,1191-1274,1455,1827-1874,2109-2164`
- `@deepseek-ai/dsh-timeout/lib/index.js:85-121`
- `@deepseek-ai/dsh-base/cordis.patch.yml:320-330,394-399`
- READMEs: `dsh-compaction`, `dsh-compaction-basic` (`:62-67,107,118`), `dsh-compaction-tool-result-pruner` (`:52-60,123-125`), `dsh-system-prompt` (`:94,135,147-163`), `dsh-agent-loop` (`:158-160`), `dsh-time-context`, `dsh-repeat-tool-reminder`, `dsh-tmux-context`, `dsh-subagent`

Downloaded to the scratchpad with `npm pack` / `curl` (read only):
- `@ai-sdk/openai-compatible@2.0.41` `src/chat/openai-compatible-chat-language-model.ts:115-255,360-731,795-837`, `convert-to-openai-compatible-chat-messages.ts:30-257`, `openai-compatible-prepare-tools.ts:7-98`, `convert-openai-compatible-chat-usage.ts:3-55`
- OpenCode v1.18.31, `https://raw.githubusercontent.com/sst/opencode/v1.18.31/packages/opencode/`: `package.json:71`; `src/provider/provider.ts:35-78,1536-1547,1755-1756,1796-1830`; `src/provider/transform.ts:18,321-351,527-570,825-842,1408-1465,1737`; `src/session/llm/request.ts:55-125,180-205`; `src/session/overflow.ts:8-34`; `src/session/compaction.ts:28-31,271-317`; `src/session/system.ts:83`; `src/session/reminders.ts:15-92`; `src/session/message-v2.ts:240-385`; `src/session/prompt.ts:195-240,1270-1292`; `src/agent/agent.ts:7,416-435`; `src/provider/error.ts:175`
- `@earendil-works/pi-coding-agent@1.0.3`: `docs/llama-cpp.md`, `docs/models.md:45-64`, `docs/compaction.md:29-45`, `dist/extensions/llama/client.js:1-200`, `dist/extensions/llama/provider.js:1-130`
- `@earendil-works/pi-ai@1.0.3`: `dist/api/llama-cpp-classify.js:13-15,223-298`, `dist/utils/transcript.js:55-104`, `dist/api/openai-completions.js:641-646`

Local, llama.cpp `llama-rig2` (branch `rig/full`, HEAD `e2377cc96`):
- `common/chat.cpp:267-330,940-956,1218-1226,1229-1310,1447-1531`
- `common/parsers/qwen3-coder.cpp:1-194`
- `common/chat-peg-parser.cpp:330-447`
- `common/chat-auto-parser.h:73`, `common/json.h:18-25`, `common/json.cpp:14`
- `common/jinja/README.md`, `common/jinja/caps.cpp:22-33,399-492,542-566`
- `common/common.h:633-636`
- `tools/server/server-common.cpp:40-104,1230-1420`
- `tools/server/server-context.cpp:280-330,1625-1708,2425-2490,3322-3335,3420-3560,3600-3810,4498-4522,4560-4700,4795-4830,5282-5289`
- `tools/server/server-task.cpp:150-237,357-525,1700-1907`; `server-task.h:58,601-679`
- `tools/server/server-http.cpp:229-290,666`; `tools/server/server.cpp:249-297`
- `src/llama-vocab.cpp:280-300,380-397,584-640,2727-2734,2920-2940,3289-3320`; `src/unicode.cpp:608-670,1050-1064`; `src/unicode-data.cpp:1-12,2286-2314`
- `models/ggml-vocab-qwen35.gguf{,.inp,.out}`; `vendor/cpp-httplib`, `vendor/nlohmann`, `vendor/stb`

This project:
- `research/_brief.md`, `research/_chat_template.jinja:1-170`, `research/_gguf-model-dump.txt:41-50`
- Model GGUF tokenizer fields read with `gguf` 0.19.0 (project venv): `qwen38_27\models\Qwen3.8-27B\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- Template renders with `jinja2` 3.1.6 (project venv), script in the session scratchpad
