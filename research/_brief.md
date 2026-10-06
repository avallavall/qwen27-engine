# Brief for research agents (read this first)

Project: a from-scratch inference engine for ONE model (Qwen3.8-27B) on ONE rig
(this PC). Goal: faster than the current llama.cpp setup, with every feature the
user uses today. This session only researches and plans. Nobody writes the engine yet.

Project folder: `qwen27-engine`.
The full task text was the initial session prompt (not part of the repository).

## Rules for you

- **Every fact needs a source.** Use `file:line` in a local clone, or a URL.
  Mark estimates with the word "estimate" and show the arithmetic.
- **Read only** in these folders. Change nothing there:
  - `qwen38_27` (production. Never copy the API
    key from the scripts into any file.)
  - `llama.cpp` and the worktree
    `llama-rig2` (branch `rig/full`, the build in use).
- **No GPU work.** Do not run benchmarks, servers, or GPU programs. Another
  session measures the GPU and needs it idle. Do not start or stop `llama-server`.
- **No big builds.** Reading and grepping only. Small Python in the project venv
  is OK: `qwen27-engine\.venv\Scripts\python.exe`
  (has `gguf` and `numpy`). Never `pip install` outside that venv.
- Reference clones (shallow, read only) are in
  `qwen27-engine\refs\`:
  `ik_llama.cpp`, `calm` (zeux), `flash-linear-attention`, `flashinfer`,
  `Megakernels` (HazyResearch), `mirage`, `vllm`, `sglang`.
  If you need another repo, clone it there with `git clone --depth 1` into a
  folder named after the repo. Check first that it is not already there.
  For very large repos (TensorRT-LLM), read single files on GitHub with WebFetch.
- Web: use WebSearch and WebFetch (load them with ToolSearch first).
- **Writing style:** English. Short sentences. One idea per sentence. No
  metaphors. Tables are welcome. Say plainly when something is unknown.
- Say when a finding is Windows-only (for example WDDM effects).

## Scope set by the user (2026-10-05)

- The engine is used only from an agent harness (OpenCode, dsh, pi). No chat web UI.
- Keep: fastest possible decode and prefill for this model on these 2 cards,
  OpenAI-compatible API (chat completions, streaming, tool calls, reasoning effort,
  API key, `/v1/models`, `/health`, `/props`), vision, MTP, prompt cache across
  requests (with DeltaNet state checkpoints).
- Drop: web UI, more than one request at a time (one slot, queue the rest), other
  models, quant types not in this file, samplers not used (keep temp, top_p,
  top_k, min_p), grammar / JSON schema unless a harness needs it.
- Final numbers count on Ubuntu Server 26 (headless: card 0 has no desktop there).

## The rig (verified earlier)

- 2x RTX 5060 Ti 16 GB (Blackwell GB206, sm_120). Driver 616.64. WDDM on Windows.
- Both cards on OcuLink x4, CPU root ports, **PCIe Gen 3 x4 fixed** (Gen 4 caused
  a card to drop off the bus). No NVLink. Practical PCIe Gen3 x4 ~3.5 GB/s.
- P2P: unknown on this rig. A driver P2P patch gave no gain in llama.cpp,
  because llama.cpp's 2-GPU all-reduce uses host staging.
- VRAM copy bandwidth measured: 388 / 395 GB/s stock, 403 / 404 GB/s with memory OC.
- Card 0 drives the Windows desktop (1.3-1.9 GB VRAM used).
- CPU Ryzen 5 9600X (6 cores, Zen 5), 32 GB RAM.
- Windows 11 is the test bench. **Deployment target: Ubuntu Server 26.** Engine must
  build and run on both.
- Toolchain: CUDA 13.4 (`%USERPROFILE%\\cuda\v13.4`), MSVC 14.44, LLVM 20.1.8 clang-cl.
  **Never CUDA 13.2** (miscompiles IQ3_S on sm_120, llama.cpp PR #27902).
  Nsight Systems 2025.6.3 and Nsight Compute 2026.1.0 are installed.

## The model (read from the GGUF header in this session)

Full dumps: `research/_gguf-model-dump.txt`, `research/_gguf-model-roles.txt`,
`research/_gguf-model-tensors.tsv` (every tensor: name, type, shape, bytes),
`research/_gguf-mmproj-dump.txt`, `research/_chat_template.jinja`.

- GGUF arch `qwen35`. `block_count` 65 = 64 blocks + 1 MTP block (`blk.64.*`,
  `nextn_predict_layers` 1).
- `embedding_length` 5120. `feed_forward_length` 17408. vocab 248320.
- Full attention every 4th block (`full_attention_interval` 4): blocks 3, 7, 11, ..., 63.
  16 attention blocks, 48 Gated DeltaNet (GDN) blocks.
- Attention: 24 query heads, 4 KV heads, key/value length 256. `attn_q` is
  [5120 -> 12288] = 24 x 256 x 2 (query plus an output gate). q_norm / k_norm.
  RoPE: `rope.dimension_count` 64 (partial, of 256), `dimension_sections` [11,11,10,0],
  `freq_base` 1e7. RMS eps 1e-6. `context_length` 262144.
- GDN: `ssm.group_count` 16 (K heads), `ssm.time_step_rank` 48 (V heads),
  `ssm.state_size` 128 (head dim), `ssm.inner_size` 6144 = 48 x 128,
  `ssm.conv_kernel` 4. `attn_qkv` [5120 -> 10240] = 16x128 (q) + 16x128 (k) + 48x128 (v).
  `attn_gate` [5120 -> 6144]. `ssm_out` [6144 -> 5120]. `ssm_alpha`, `ssm_beta`
  BF16 [5120 -> 48]. `ssm_conv1d` F32 [4, 10240].
- MTP block (`blk.64`): `nextn.eh_proj` [10240 -> 5120], one full-attention layer
  plus FFN, all Q6_K. `enorm`, `hnorm`, `shared_head_norm`. It reuses `token_embd`
  and `output.weight`.
- **The file is mixed precision, not pure IQ3_S.** Bytes by type: IQ3_S 3474 MiB,
  IQ4_XS 2882, IQ3_XXS 1979, Q4_K 1509, IQ2_S 763, Q6_K 332, Q2_K 230, IQ2_XS 211,
  IQ2_XXS 94, BF16 45, IQ1_M 19, F32 10. Total 11548 MiB, 866 tensors.
  `output.weight` Q4_K 682 MiB. `token_embd` IQ2_S 388 MiB.
- Bytes read per decode step (arithmetic): main model without `token_embd` and
  without MTP = 10828 MiB. MTP block = 332 MiB. MTP draft pass = MTP block +
  `output.weight` = 1014 MiB.
- Vision: `mmproj` GGUF arch `clip`, projector `qwen3vl_merger`, 27 blocks,
  hidden 1152, FFN 4304, 16 heads, patch 16, image_size 768, spatial merge 2,
  projection to 5120. BF16, 888 MiB.
- Chat template: in the GGUF (`research/_chat_template.jinja`). It has a
  `reasoning_effort` variable (xhigh default, medium, low).
- Measured sizes: target KV 64 KiB/token (16 layers, f16), MTP draft KV 4 KiB/token,
  recurrent state ~150 MiB per sequence.

## The baseline (llama.cpp, `qwen38_27\arranca.ps1`, reasons in `qwen38_27\LEEME.md`)

`-sm tensor` (both cards compute every token), `-c 180224`, one slot, f16 KV,
MTP `--spec-draft-n-max 3`, `--spec-draft-sampling probabilistic`, `--temp 1.0`,
`--top-p 0.95 --top-k 20 --min-p 0`, vision encoder on card 1, prompt cache in RAM
(`-cram 8192`), 4 checkpoints (`-ctxcp 4`), `-ub 1024 -b 2048`, `LLAMA_SCHED_POOL=8`.
Sampling falls back to CPU with `-sm tensor`.

- Decode ms per step: 37.7 (1k), 41.0 (30k), 48.3 (100k), 53.9 (150k).
  tok/s: 76.5 / 69.0 / 56.8 / 50.7.
- Prompt reading: ~667 t/s at 30k, 540 at 100k, 439 at 150k.
- 4K image: 12.7 s.
- First estimate (to confirm or reject): weight-read floor ~15 ms per step plus
  ~4 ms for 3 MTP drafts. So ~20 ms of each step may be overhead.

## Output

Write your report to the exact path given in your task, with the Write tool.
Structure: a short summary first (5-10 lines: the answers), then details, then a
"Sources" list. Then reply with only the word "written".
