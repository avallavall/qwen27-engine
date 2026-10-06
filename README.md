# qwen27-engine

An inference engine written from scratch for one model, **Qwen3.8-27B**, on one kind of machine, **two
NVIDIA RTX 5060 Ti 16 GB cards** (Blackwell, sm_120) under Windows. It serves an OpenAI-compatible API with
streaming, tool calls, reasoning, image input and speculative decoding, so coding agents such as Qwen Code can use
it in place of `llama-server`.

A general engine supports many models and many GPUs. This one supports one model and one rig, so every kernel is
written for this case: quantized GEMV/GEMM kernels for the model's exact quant types, tensor parallelism over
the two cards with a cross-card sum through pinned host memory, Gated DeltaNet and attention kernels, MTP
speculative decoding with sampling on the GPU, and its own vision encoder.

## Speed against llama.cpp

Same PC, same model file, both servers with 180,224 tokens of context, vision and MTP on, the same requests
(`bench\compare.py`, `bench\compare_report.py`). llama.cpp: a tuned production build with tensor split
(`-sm tensor`), MTP drafting and flash attention.

| Test | llama.cpp | qwen27-engine | Gain |
|---|---|---|---|
| Generation, tok/s at 1k / 30k / 100k / 150k context | 76 / 70 / 57 / 53 | 117 / 106 / 90 / 83 | 1.51-1.58x |
| Prompt reading, tok/s (first read at 1k / 30k / 100k / 150k) | 517 / 664 / 534 / 427 | 655 / 789 / 645 / 495 | 1.16-1.27x |
| Resent prompt, time to the first token (1k → 150k) | 181 → 682 ms | 65 → 82 ms | 2.8-8.4x |
| Image 800x600 (1036 tokens), total | 3.75 s | 2.77 s | 1.36x |
| Image 3840x2160 (4099 tokens), total | 12.5 s | 7.3 s | 1.72x |
| Replay of 11 requests of a real Qwen Code session | 93.4 s | 73.9 s | 1.26x |
| VRAM per card | 14.5 + 15.4 GB | 10.9 + 11.3 GB | |
| Load time | 8.3 s | 10.1 s | |

Details and the full measurement log: [PLAN.md](PLAN.md).

## Requirements

- Windows 10/11, x64.
- Two NVIDIA GeForce RTX 50-series cards with 16 GB each (tested: 2x RTX 5060 Ti 16 GB). The code is built for
  sm_120 (`120a`) and splits the model over exactly the two cards.
- An NVIDIA driver for CUDA 13.4, and the **CUDA Toolkit 13.4**. CUDA 13.2 miscompiles IQ3_S on sm_120 (llama.cpp
  PR #27902); the build refuses anything older than 13.4.
- Visual Studio 2022 (the free Build Tools are enough) with "Desktop development with C++" and "C++ CMake tools
  for Windows" (CMake 3.28 or newer, Ninja).
- 32 GB of system RAM (the prompt cache keeps up to 8 GB of conversation state in RAM by default).
- Python 3.12 only for the tests and benchmarks.

## Model files

The engine reads one specific GGUF and its vision projector, both from
[ISTA-DASLab on Hugging Face](https://huggingface.co/ISTA-DASLab) (base model:
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0):

- `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB; mixed 1.75-6.5 bit quantization, IQ1_M up to Q6_K, with the
  MTP head)
- `mmproj-Qwen3.8-27B-BF16.gguf` (vision encoder, optional)

Put them in a `models\` folder in this repository, or point to them with `Q27_MODEL` and `Q27_MMPROJ`. The server
checks the model's architecture, its sizes and its chat template, and refuses another file.

## Build

```
git clone https://github.com/avallavall/qwen27-engine
cd qwen27-engine
build.bat
```

`build.bat` finds Visual Studio 2022 and CUDA 13.4 (in `%USERPROFILE%\cuda\v13.4` or the standard install folder).
Set `Q27_CUDA` to the CUDA 13.4 folder, or `Q27_VCVARS` to `vcvarsall.bat`, if they are elsewhere. The output goes
to `build\`: the server `q27_server.exe`, the tests and the benchmark tools. `build.bat --target q27_server`
builds only the server.

## Run

```
powershell -File start-server.ps1 -ApiKey <key>
```

This starts the server on `http://127.0.0.1:8081` with vision on (if the mmproj is found). Options:

| Option | Default | Meaning |
|---|---|---|
| `-Port` | 8081 | port |
| `-Bind` | 127.0.0.1 | `0.0.0.0` makes it reachable from the LAN |
| `-ApiKey` | `$env:Q27_API_KEY` | API key (`Authorization: Bearer` or `X-Api-Key`); none = open server |
| `-Ctx` | 0 | context in tokens; 0 = the largest that fits (262,144, the model's maximum, with q8_0 KV) |
| `-Effort` | medium | default reasoning effort: low, medium, xhigh |
| `-Kv` | q8_0 | KV cache type: q8_0 or f16 |
| `-Model`, `-Mmproj` | see "Model files" | model paths; `-NoVision` for text only |
| `-LogDir` | none | write every request and result as JSON files (for debugging and the checks below) |

`arranca-q27.ps1` is a drop-in replacement for a `llama-server` on port 8080: it uses port 8080 on all
interfaces, reads the API key that Qwen Code sends from `~\.qwen\.env` (`QWEN_LOCAL_API_KEY`), asks before
stopping a running `llama-server`, and checks that the server answers.

The server itself (`build\q27_server.exe --help`) also takes `--temp`, `--top-p`, `--top-k`, `--min-p`,
`--chat-template-kwargs JSON`, `--cache-ram MB`, `--image-min-tokens` (default 1024) / `--image-max-tokens`
(4096), `--draft-vocab N|file[:N]|0` and `--alias`.

## API

OpenAI-compatible, with llama.cpp's response shapes (`reasoning_content`, `timings`, usage chunk):

- `POST /v1/chat/completions` (also `/chat/completions`), streaming or not
- `GET /health`, `/v1/health` (no key needed), `/props`, `/v1/models`, `/models`
- `POST /tokenize`, `/detokenize`, `/apply-template`

```
curl http://127.0.0.1:8081/v1/chat/completions -H "Authorization: Bearer <key>" -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"stream\":true}"
```

What a request can use:

- `messages` with `system`/`developer`, `user`, `assistant` (with `reasoning_content` and `tool_calls`) and `tool`
  roles; text parts and `image_url` parts (`data:image/...;base64,...` URIs; PNG, JPEG, BMP, GIF, ...).
- `tools` (function tools): the model's XML tool calls are parsed while streaming into OpenAI `tool_calls`
  deltas, with the same results as llama.cpp's parser.
- `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k` (at most 20), `min_p`, `seed`, `stop`.
  Defaults: temperature 1.0, top-p 0.95, top-k 20, min-p 0 (the model's recommended settings).
- Reasoning: `reasoning_effort` (top level, in `chat_template_kwargs`, or `reasoning: {effort}`): `low`, `medium`,
  `xhigh`; `high`/`max` are taken as `xhigh`, `minimal` as `low`, `none`/`off` turn thinking off.
  `chat_template_kwargs.enable_thinking: false` also turns it off.

A context overflow returns HTTP 400 with llama.cpp's and OpenAI's wording, so clients that compact on either
message keep working.

### Qwen Code

Point a provider of `~\.qwen\settings.json` at the server (`"baseUrl": "http://127.0.0.1:8081/v1"`, with the key
in the variable named by `envKey`). To let Qwen Code choose the reasoning effort, add to that provider:

```json
"capabilities": { "reasoning": { "profile": "openai-effort", "efforts": ["low", "medium", "high", "xhigh"],
                                 "defaultEffort": "medium" } }
```

## How it works, in short

- **Decode:** both cards compute every token (tensor parallel): each holds half of the attention heads, GDN heads,
  FFN rows and vocabulary. The partial sums go through pinned host memory (the cards have no peer-to-peer link
  on Windows), with the next weights prefetched into L2 during the wait. One CUDA graph per step.
- **MTP speculative decoding:** the model's MTP head drafts 3 tokens per step; the target checks them in one
  4-token pass; acceptance and sampling run on the GPU. The drafts score only the 32k most useful tokens
  (`data\draft_vocab.bin`); the target still scores all of them, so the output distribution does not change.
- **Prompt reading:** batches of 512 tokens with own int8 tensor-core GEMMs (llama.cpp's MMQ numerics), a
  causal flash-attention kernel, and the cross-card sums of one half-batch hidden behind the other half's compute.
- **KV cache:** q8_0 by default (f16 optional), up to 262,144 tokens.
- **Prompt cache:** the longest common prefix with the live conversation is reused; checkpoints of the
  recurrent (GDN) state let a conversation restart from earlier points; a conversation that a side request pushes
  out is saved to RAM and comes back in a fraction of a second.
- **Vision:** own encoder kernels (BF16 tensor-core GEMMs, flash attention for head size 72); a 4K image is
  encoded in about 1.1 s. Image rows get IMRoPE positions like llama.cpp.
- **Server:** cpp-httplib, a hard-coded copy of the model's chat template (byte-identical to jinja2), a port of
  llama.cpp's tokenizer (identical ids), a streaming tool-call parser, one compute slot with a FIFO queue.

## Tests and benchmarks

| What | Command |
|---|---|
| Tokenizer vs llama.cpp | `build\test_tokenizer.exe <model.gguf>` (needs a golden file, see the header of `tools\test_tokenizer.cpp`) |
| Chat template vs jinja2 | `python tools\gen_template_fixtures.py`, then `build\test_template.exe bench\out\template_fixtures.json` |
| Tool-call parser | `build\test_parser.exe <model.gguf>` |
| Prompt cache (GPU) | `build\test_cache.exe <model.gguf> <token file>` |
| Vision encoder vs llama.cpp | `build\test_vision.exe <mmproj.gguf> compare bench\out\vision` (reference: `bench\build-llama-vision.bat`) |
| Server, end to end | start the server with a key, then `python bench\test_server.py` (`BENCH_KEY`, `BENCH_URL`) |
| Image answers | `python bench\vision_answers.py` |
| Prompts and tool calls vs llama.cpp | `bench\build\llama_chat.exe <server log dir>` (`bench\build-llama-chat.bat`) |
| Real Qwen Code session | `python bench\qwen_session.py` |
| Speed | `build\q27_gen.exe <model> <token file> 0,1 1000,30000,100000,150000 400 depth 2`, `python bench\mide-tps.py`, `python bench\compare.py` |

The Python scripts use a virtual environment with `jinja2`, `numpy`, `pillow` and `gguf`. The comparisons with
llama.cpp link a llama.cpp build (`Q27_LLAMA_BIN`) and its source headers (`Q27_LLAMA_SRC`); see `tools\env.bat`.

## Repository layout

| Path | Contents |
|---|---|
| `src\` | the engine: GGUF reader, kernels (`qgemv`, `qgemm`, `attn`, `ops`, `prefill`, `vision`), model and decoder, sampling, tokenizer, chat template, stream parser, server engine |
| `tools\` | `q27_server.cpp`, tests, benchmark tools, `env.bat` |
| `bench\` | benchmark and check scripts, llama.cpp reference tools |
| `data\` | `draft_vocab.bin` (token ranking for the MTP drafts, made by `bench\make_draft_vocab.py`) |
| `research\`, `PLAN.md` | design research, plan, milestone status and measurement log |
| `third_party\` | cpp-httplib, nlohmann/json, stb_image |

## Limits

- One model file and Windows only. A Linux build is planned but not done.
- One request computes at a time (others wait in a queue).
- No logprobs, grammars, JSON schema output or repetition penalties. `top_k` is at most 20.
- A trailing assistant message is answered with a new turn, not continued.
- Images: `data:` URIs only (no remote URLs), no WebP.

## License

MIT, see [LICENSE](LICENSE). Parts are derived from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT);
the details and the vendored libraries are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model
weights are not part of this repository and have their own license.
