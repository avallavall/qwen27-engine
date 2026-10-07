# PLAN: a Qwen3.8-27B engine for 2x RTX 5060 Ti

> This file is the development record: the scope, the plan made from the research, the status of each milestone
> and the raw measurement log. The current design is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
> and the current numbers in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
>
> Paths like `qwen38_27\` and `llama-rig2\` are folders next to this repository on the development PC (the
> production llama.cpp setup and the llama.cpp source used as a reference). Mentions of "HANDOFF" refer to the
> local work checklist, which is not part of the repository.

One model (Qwen3.8-27B, `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`). One rig (2x RTX 5060 Ti 16 GB,
PCIe Gen 3 x4 each, no P2P on Windows). One client type (agent harness: OpenCode, dsh, pi).
Goal: faster than the current llama.cpp setup, with every feature in use today.

Scope set by the user on 2026-10-05:

- Keep: decode and prefill speed, OpenAI-compatible API (chat completions, SSE streaming,
  tool calls, reasoning, API key, `/v1/models`, `/health`, `/props`, `/tokenize`), vision,
  MTP with the model's own head, prompt cache across requests.
- Drop: web UI, more than one request at a time (one slot, a queue), other models, quant
  types that are not in this file, unused samplers (keep temp, top_p, top_k, min_p),
  grammar and JSON schema.
- **For now, all work and all numbers are on Windows 11** (user decision, 2026-10-06). The code
  stays portable so a Linux build is possible later (M8), but nothing waits for it.
  Windows rules for the design: every pass runs as a CUDA graph (7 µs per plain launch); no
  kernel runs long enough to reach the 2 s TDR watchdog; one copy engine per card, so
  cross-card transfers that must use both link directions are done by kernels; card 0 keeps
  1.3-1.9 GB for the desktop.

Research reports with sources are in `research/`. Measurement tools are in `bench/`.

---

## Status

**2026-10-06: first complete version.** M1-M7 done. The engine (`start-server.ps1`, or `arranca-q27.ps1` in place
of production on port 8080) serves text, tools and images to Qwen Code, with every milestone check passed. Numbers:
measurement log, "Engine, final" and "Head-to-head at the same context". After those, the MTP drafts score a
32k-token subset of the vocabulary (+7% on code, +5% on Spanish). GDN rollback by replay was measured and does not
pay.

**2026-10-07: second speed round.** The engine was profiled with GPU time stamps inside its CUDA graphs
(`Q27_PROF=1`), and each bottleneck was changed and measured alone. Main changes: cross-card sums on an int8 wire
with error feedback, independent GEMVs on parallel graph branches, a pipelined prompt GEMM, prompt attention with
int8 Q K^T, and prompt batches of 2048 tokens. Head-to-head against llama.cpp at the same context: decode steps
1.65-1.67x faster at every depth (was 1.53-1.58x), prompt reading 2.27-2.41x (was 1.16-1.27x), a recorded Qwen
Code session 1.92x (was 1.26x). The logit tests stay inside the limits. The list of changes and their measured
effects is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md#optimization-history).

**2026-10-07: third speed round and Linux build.** The engine builds and passes every check under Linux (WSL2,
Ubuntu 24.04), with the same logits as on Windows. Speed changes: the token embedding in each card's VRAM, the MTP
drafter re-quantized to Q4_K at load, block verification of the drafts (exact), the L2 prefetch of each cross-card
sum moved to separate blocks (found with new timers inside the sum kernel), and the last checkpoint restored from
VRAM. Measured and not done: shorter or longer draft chains, a device-side step loop, 4096-token prompt batches,
and two GEMV changes suggested by Nsight Compute. Head-to-head against llama.cpp measured again the same day with a
quiet desktop: decode steps 1.72-1.78x faster (20.9 / 22.9 / 27.3 / 30.5 ms at 1k / 30k / 100k / 150k), prompt
reading 2.27-2.40x, resent prompts 4.6-12.2x sooner to the first token, the Qwen Code replay 1.94x. Details in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md#third-round-2026-10-07).

## Phase 0 result: where the time goes, and the floor

Measured 2026-10-05 on this rig with the production build (`qwen38_27\bin-parches`) and
production flags (`bench\prof-llama.cmd`). Profiler: Nsight Systems 2025.6.3. Details and
caveats are in the measurement log at the end.

### Decode step at 1k context (one step = verify 4 tokens + MTP catch-up + 3 MTP drafts)

"Now" is per card, from the profile (40.2 ms per step under the profiler, 39.5 ms without it
today, 37.7 ms on 2026-10-03). "Floor" is the time if each part ran at the hardware limit
measured today. Both cards work in parallel, so the per-card numbers are the step numbers.

| Part of one step | Now (ms) | Floor (ms) | How the floor is computed |
|---|---|---|---|
| Weight GEMV, 5 passes (7.45 GB per card) | 24.0 | 17.1 | 7.45 GB / 435 GB/s (measured read bandwidth) |
| Activation quantize (q8_1), separate kernel per matmul | 1.0 | ~0 | fused into the GEMV |
| Norms, RoPE, element-wise, copies (incl. GDN state copies) | 2.7 | 0.2 | fused; these bytes are small |
| Gated DeltaNet kernels (recurrent, conv, alpha/beta) | 0.75 | 0.4-1.0 | state read + write once (0.4); with 4 rollback snapshots (1.0) |
| Attention (KV read) | 0.3 | 0.1 | 38 KiB per token per card / 435 GB/s |
| Kernels that overlap (credit) | -2.2 | 0 | |
| **GPU busy** | **26.5** | **18.2** | |
| Cross-card sums (about 136 per step, 36-40 µs each) | 4.4-5.0 | 1.9 (0.4 if overlapped) | 128 x 15 µs (40 KiB at 3.55 GB/s + latency) |
| Pass borders: 5 host syncs, logits to CPU, CPU sampling over 248,320 tokens, graph launches | 9.1 | 0.2 | sampling and the draft loop on the GPU |
| Short gaps | 0.2 | 0 | |
| **Total** | **40.2** | **~20** | |

### Decode step at depth

Only attention grows with depth. The profile at 30k shows the attention kernels at 2.95 ms per
step (0.092 ms per 1000 tokens). The f16 KV read floor is 0.090 ms per 1000 tokens per card.
**llama.cpp attention is already at the floor.** The 100k and 150k rows use the 2026-10-03
step times and the same per-1000-token floor.

| Context | Now ms/step (2026-10-03) | Floor f16 KV | Floor q8_0 KV | Realistic target f16 | Target tok/s (today's MTP acceptance) | Now tok/s |
|---|---|---|---|---|---|---|
| 1k | 37.7 | 20.0 | 20.0 | 23.5 | ~123 | 76.5 |
| 30k | 41.0 | 22.6 | 21.3 | 26.2 | ~108 | 69.0 |
| 100k | 48.3 | 28.9 | 24.7 | 32.9 | ~83 | 56.8 |
| 150k | 53.9 | 33.4 | 27.0 | 37.6 | ~73 | 50.7 |

Realistic target assumptions: GEMV at 88% of the measured read bandwidth (`research/gemv-quant.md`
estimates 88-92%), GDN with rollback snapshots (1.0 ms), fused small kernels (0.5 ms), sums not
overlapped yet (2.0 ms), pass borders 0.5 ms, attention at 95% of floor. tok/s uses today's tokens
per step: 2.88 / 2.83 / 2.74 / 2.73.

### Prefill (prompt reading) at 30k

From the profile: 1546 ms per 1024-token batch (662 t/s). The cards compute for 649 ms and wait
for 897 ms (58%). The wait is the cross-card sum: 130 sums of 10 MiB per batch go through host
memory with copy engines, one direction at a time. The compute part: quantized GEMM 406 ms,
DeltaNet 79 ms, attention 76 ms, the rest 88 ms.

| Context | Now t/s | Target t/s (`research/prefill.md`) | Compute floor t/s (INT8, 2 cards) |
|---|---|---|---|
| 30k | 667 | 1,300 | ~5,400 |
| 100k | 540 | 880 | ~2,900 |
| 150k | 439 | 640 | ~1,800 |

The link sets the real limit: 1.35 GB per card per direction per 1024 tokens. At 3.55 GB/s this is
0.38 s per batch with both directions at once, 0.76 s one at a time. Windows reports one copy
engine per card, and a duplex copy test gave 3.56 GB/s total, so copy engines cannot use both
directions on Windows. Kernel-driven transfers or Linux are needed for that.

### Go / no-go

- **llama.cpp is not close to the floor.** Decode has about 1.9x headroom at 1k and 1.6x at 150k.
  A realistic target is **1.4x to 1.6x faster decode** and **1.5x to 1.9x faster prefill**.
- The headroom is in four places: GEMV efficiency (about 7 ms), pass borders with CPU sampling
  (about 9 ms), cross-card sums (about 3 ms), small kernels (about 2.5 ms).
- Attention at long context is already at the floor with f16 KV. The only big lever there is
  q8_0 KV (saves about 6.4 ms per step at 150k). It changes numerics against the f16 baseline.
- Cost: about 16-23 weeks of work in total (milestones below). The first version usable from a
  harness comes after milestone 6.
- A cheaper alternative exists for part of the gain: patch llama.cpp to sample on the GPU under
  `-sm tensor` and keep the draft loop on the GPU. That removes part of the 9 ms of pass borders,
  so maybe 1.2x, for about 2 weeks of work (estimate). It does not touch the other 12 ms.

---

## Design decisions (from research)

The decisions as taken before the code was written. Later changes are recorded in the milestone status lines,
the measurement log and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Topic | Decision | Why | Report |
|---|---|---|---|
| Split over 2 cards | Tensor parallel, 2 sums per layer, as llama.cpp | Pipeline split halves decode bandwidth. Duplicated weights do not fit at 180k. | `multigpu.md` |
| Cross-card sum, decode | One-shot sum through mapped pinned host memory, per-chunk flags, inside a CUDA graph; fused with residual add + RMSNorm; later overlapped with the next GEMV | Measured 9 µs (10 KiB) and 27 µs (40 KiB) per sum in a graph, against 36-40 µs in llama.cpp | `multigpu.md`, `bench/hw.cu` |
| Cross-card sum, prefill | Kernel-driven or two-stream transfers; two half-batches so transfer and compute overlap | Today 58% of prefill time is waiting on the link | `multigpu.md`, `prefill.md` |
| Kernel launches | One CUDA graph per pass at first; PDL inside graphs; a persistent kernel later only if the profile asks for it | Windows launch cost measured 7 µs per kernel; 1.2-1.6 µs per kernel inside a graph | `launch-overhead.md` |
| Sampling | On the GPU: running top-20 inside the output GEMV, then top-p, min-p, temp and the MTP rejection test on at most 20 candidates per card | Removes the logits copy and CPU sampling (most of the 9 ms) | `gemv-quant.md` |
| Decode GEMV | Own kernels for the 11 types in the file, 1-4 columns, 128-bit loads after a lossless repack; keep llama.cpp's q8_1 activations and integer truncations for logit match | MMVQ runs at ~310 GB/s here; repack verified bit-exact on all 10 quantized types | `gemv-quant.md` |
| Prefill GEMM | Port llama.cpp MMQ (INT8 `mma.sync`, q8_1 activations) for the 11 types | Only option that keeps llama.cpp numerics and the exact GGUF weights; CUTLASS sm_120 needs FP8/FP4 weights | `prefill.md` |
| Gated DeltaNet | f32 state. Decode: 4-token kernel with state on chip, 4 snapshots for rollback (as llama.cpp). Later: replay of accepted tokens (0.39 ms instead of 0.97 ms) | Keeps the match; snapshots are simple | `deltanet.md` |
| GDN prefill | Port llama.cpp chunked kernel or FLA chunk-64 with `mma.sync` | Small gain (1-3%), so low priority | `deltanet.md` |
| Attention decode | `mma.sync` split-KV kernel, all 24 query rows (6 heads x 4 tokens) in one tile, 2 KV heads per card, fused combine + sigmoid gate | llama.cpp pads 6 heads to 8 (25% waste); already at floor in bytes | `attention.md` |
| KV cache | f16 first (needed for the logit match). q8_0 as a switch after milestone 4. Never q4. | q8_0 saves 6.4 ms/step at 150k and doubles the context limit. The llama.cpp problem (q8_0 KV with `-sm tensor` leaves card 1 idle, `LEEME.md`) is a llama.cpp bug, not a hardware limit; the engine has its own attention kernel. No upstream fix found on 2026-10-06 (similar open issue: #26409). | `attention.md` |
| Context | 180224 minimum. f16: about 237k on Windows, 249k on headless Linux (estimate). q8_0: 262144 (model limit) | | `attention.md` |
| MTP | Same algorithm as llama.cpp (`common/sampling.cpp:732-836` rejection rule, draft top-k 10). Whole draft loop on the GPU | Keeps the output distribution | `model-mtp.md` |
| Vision | Own encoder kernels (BF16 GEMM, flash attention for head dim 72 padded to 80, LayerNorm, GELU, 2D RoPE) on card 1 | No ggml dependency; estimate 1.6-2.0 s per 4K image vs 3.5-5 s now | `vision.md` |
| Server | cpp-httplib + nlohmann `ordered_json` + stb_image. Own: template renderer (hard-coded, tested against jinja2), stream parser for XML tool calls, sequence and checkpoint manager | Harnesses send only a small part of the API | `server.md` |
| Tokenizer | Port llama.cpp BPE (qwen35 pre-tokenizer); test against llama.cpp and HF `tokenizers` | | `server.md` |
| Prompt cache | One compute slot; several resident sequences in one KV pool; GDN checkpoints (75 MiB per card) at prompt end and other fixed positions; swap whole sequences to RAM. No MTP KV in checkpoints. | Logs show 99 of 114 reuses with the full prefix kept | `server.md`, `deltanet.md` |
| Build | CMake + Ninja + nvcc 13.4, `CMAKE_CUDA_ARCHITECTURES=120a-real`, cl.exe host compiler on Windows, gcc 15.2 on Linux. Fail at configure time on CUDA 13.2. No NCCL. cuBLAS only for vision, optional. | CUDA 13.2 is first on PATH on this PC | `build-linux.md` |

---

## Milestones

Every milestone uses the same two checks.

- **Correctness.** (a) Logits: `llama-perplexity --kl-divergence-base` (from `qwen38_27\bin-parches`)
  writes the llama.cpp reference with its tokens; `build\q27_ppl.exe` reads the tokens, runs the engine
  and writes its own base file; `llama-perplexity --kl-divergence --kl-divergence-base <engine file>`
  reports KLD and top-1 agreement. Text: `bench\out\ppl-text.txt` (first 60 KB of the bench corpus),
  `-c 512 --chunks 8 -ub 1`. Target (set 2026-10-06 from measurements): the engine must be at least as
  close to llama.cpp as llama.cpp batch mode is to llama.cpp one-token mode: mean KLD <= 0.001, same top
  token >= 98.8%. (b) Perplexity of the same tokens within 0.1% of llama.cpp (2.3085 one-token mode).
- **Speed.** `bench/mide-tps.py` (copy of the production bench; port and key from env), 4 depths
  x 4 runs x 400 tokens. Report ms per step = `predicted_ms / (gen_tok - draft_n_accepted)`.
  The engine must return the same `timings` fields.

Work estimates are calendar weeks of focused work. They are estimates.

### M1. Load the GGUF. Decode on one card. Match llama.cpp logits. (3-4 weeks)

- GGUF loader (mmap or read, no ggml). Lossless repack of the 10 quantized types at load time.
- Kernels: GEMV for the 11 types (1-4 columns), q8_1 activation quantize (fused), RMSNorm,
  partial IMRoPE, attention decode (f16 KV), GDN decode (f32 state), conv1d, SiLU, sigmoid gate,
  output layer, GPU sampling.
- Everything on card 1 (no desktop there), short context (the 12.1 GB file fits one card).
- One CUDA graph per forward pass.
- Check: logit match and perplexity. Speed: GB/s of each GEMV kernel against the read bandwidth
  from `bench/hw.cu`. **Stop point:** if the GEMV kernels stay below 80% of 435 GB/s, re-plan
  before M2.
- **Status 2026-10-06: done, stop point passed.** Code: `src/` (gguf, qgemv, ops, model),
  tools `bench_gemv`, `bench_pass`, `q27_ppl`. Whole-model GEMV pass in one CUDA graph (card 1):
  401 / 390 GB/s at 1 / 4 tokens with full-size matrices, 380 / 362 GB/s with half-size (per-card TP)
  matrices. Logits vs llama.cpp one-token mode: KLD 0.00055, same top 99.27%, PPL 2.3070 vs 2.3085.
  (llama.cpp batch vs one-token: KLD 0.00099, same top 98.82%.) One token on one card: 33.65 ms vs
  31.5 ms for llama.cpp; about 1,170 kernels per token are not fused yet. GPU sampling moved to M3.

### M2. Two cards, decode. (2-3 weeks)

- Tensor-parallel split as llama.cpp (heads and FFN columns; row split for `attn_output`,
  `ssm_out`, `ffn_down`; vocab split for the output layer).
- One-shot sum through mapped host memory, fused with residual add and RMSNorm.
- Sampling across the vocab halves (exchange top-20 lists, not logits).
- Check: logit match again. Speed: ms per step without MTP against llama.cpp without MTP, at 1k.
  **Stop point:** if a sum costs more than 25 µs at 40 KiB inside the graph, re-plan the split.

- **Status 2026-10-06: core done.** `Model`/`Decoder` take 1 or 2 devices (`q27_ppl ... 0,1`). Split as
  llama.cpp; embedding table in mapped pinned host memory (read by both cards). Cross-card sum: one-shot
  through mapped host memory, BF16 wire like llama.cpp, inside each card's CUDA graph: 12 us per sum at
  one token. One token, no MTP, 2 cards: 19.6 ms (llama.cpp `-sm tensor -ub 1`: 19.9 ms). Logits vs
  llama.cpp 2-card: KLD 0.00057, same top 99.17%. Profile: GEMV 15.4 ms (353 GB/s per card), sums
  1.5 ms, gaps between ~1,240 kernels 3.5 ms. Fusion is postponed until the 4-token pass of M3 exists.

### M3. MTP speculation. (2 weeks)

- MTP head (`blk.64`, Q6_K), its own KV cache, catch-up pass, 3 drafts, verify of 4 tokens.
- Rejection sampling on the GPU (llama.cpp rule). GDN rollback with 4 snapshots.
- The whole step (verify + drafts) runs without returning to the CPU except to stream tokens.
- Check: acceptance rate within noise of llama.cpp (0.60-0.70 on the bench), output distribution
  test (histograms of sampled tokens on fixed prompts at temp 1.0). Speed: ms per step at 1k.

- **Status 2026-10-06: core done.** One CUDA graph per step per card: verify 4 tokens (GDN state snapshots
  in 4 planes), GPU sampling (top-k per vocab half, candidate exchange between cards, top-p, temperature),
  llama.cpp acceptance rule, MTP catch-up (4 rows), 3 draft passes. Tool `build\q27_gen.exe`.
  The 4-token pass alone vs llama.cpp `-ub 4`: KLD 0.00058, same top 99.31%; 23.0 ms vs 28.0 ms.
  Greedy check: speculative = plain decoding for 177 of 200 tokens; the split happens at the smallest
  top-1/top-2 logit gap of the run (0.027), which points to rounding between the 1-token and 4-token
  paths, not to a rollback error. Sampled (temp 1.0, top-k 20, top-p 0.95) at 1k context:
  **31.3-31.7 ms per step, 3.0-3.1 tokens per step, 95-99 tok/s** (llama.cpp: 37.7 ms, 76.5 tok/s).
  Known gaps: attention reads the KV cache once per query token (4x per verify), so long context is
  slow until M4; prompt reading uses decode passes (~156 t/s) until M5; no kernel fusion yet.

### M4. Long context. (1-2 weeks)

- KV pool sized from free VRAM (180224 minimum, more if it fits), split-KV attention to 262144.
- q8_0 KV as an option, measured against f16 for speed and KLD.
- Check: logit match at 30k / 100k / 150k. Speed: the full bench at 4 depths.
- **Status 2026-10-06: done.** New attention kernel (`src/attn.cu`): one CTA per (chunk of positions, KV head)
  with all 6T query rows, `mma.sync` f16/f32, split-KV over 18 chunks per head, gate fused in the combine;
  427-431 GB/s at depth (98% of the read limit). Long-context logit tests with `bench\llama_ref.exe` references:
  KLD 0.000507 / 99.12% at 32k, 0.000144 / 99.51% at 131k. Context sized from free VRAM: 188,928 tokens on this
  PC (f16, reserves for desktop, vision, prefill, checkpoints). Decode ms per step 26.8 / 30.0 / 36.9 / 41.8 at
  1k / 30k / 100k / 150k (llama.cpp 37.7 / 41.0 / 48.3 / 53.9). q8_0 KV measured as an option (same KLD,
  36.3 ms at 150k, prompt reading 12% slower at depth). **User decision 2026-10-06: q8_0 KV is the default.**

### M5. Prefill. (3-4 weeks)

- MMQ port for the 11 types, flash-attention prefill (`mma.sync`, f32 accumulate), chunked GDN.
- Prefill sums with overlap (two half-batches, kernel-driven or two-stream transfers).
- No host-built attention mask (causal mask in the kernel).
- Check: perplexity and logit match on long prompts. Speed: prompt t/s at 30k / 100k / 150k.
- **Status 2026-10-06: done.** Prompt reading in batches of 512: own int8 tensor-core GEMM with MMQ numerics
  (`src/qgemm.cu`), causal prefill FlashAttention (`attn_prefill`), GDN recurrence over the batch, MTP rows,
  cross-card sums of one half-batch overlapped with the other half's compute through the copy engine and stream
  memory operations (the link carries about 3.3 GB/s in total per card; kernel reads of host memory only
  1.66 GB/s). KLD vs llama.cpp batch mode 0.00086 / 98.93% at 32k. Prompt t/s 774 / 615 / 491 for the ranges
  0-30k / 30k-100k / 100k-150k (llama.cpp 667 / 540 / 439).

### M6. Server. (2-3 weeks)

- HTTP + SSE, API key, the endpoints in the scope list.
- Template renderer (byte-exact against jinja2 renders of the GGUF template), XML tool-call
  stream parser, `<think>` handling, `chat_template_kwargs` (`enable_thinking`, `preserve_thinking`,
  `reasoning_effort`: only `xhigh`, `medium`, `low` are valid).
- Tokenizer port. Prompt cache with GDN checkpoints and RAM swap.
- Check (harness tests use Qwen Code only, user rule 2026-10-06): replay real harness sessions and compare the rendered prompts and
  parsed tool calls with llama.cpp. Speed: time to first token on a resent 100k prompt.
- **After M6 the engine can replace llama.cpp for text sessions.**
- **Status 2026-10-06: done.** `tools\q27_server.cpp` + `src\engine.*`: OpenAI chat completions with SSE in
  llama.cpp's shapes, API key, FIFO queue, own tokenizer (0 differences on 41 M ids), C++ chat template
  (byte-equal to jinja2 on 576 fixtures), streaming XML tool-call parser (matches llama.cpp's parser on 33,988 of
  34,000 random outputs), prompt cache with GDN checkpoints in RAM and RAM swap of whole sequences. Real Qwen Code
  sessions work, and their prompts and parsed tool calls are identical to llama.cpp's. `mide-tps.py`
  generation tok/s 102-116 / 96-100 / 82-85 / 73-80 at 1k / 30k / 100k / 150k (llama.cpp 68-74 / 61-65 / 51-55 /
  46-48). Start: `start-server.ps1`.

### M7. Vision. (2-3 weeks)

- Image decode and resize (stb_image), patching, ViT encoder with own kernels on card 1,
  merger, image tokens with IMRoPE positions.
- Check: embeddings against llama.cpp `mtmd` on fixed images (cosine and max error), answers
  on test images. Speed: `qwen38_27\prueba-imagen.py` on a 4K image (now 12.7 s).
- **Status 2026-10-06: done.** Own encoder kernels on card 1 (`src/vision.cu`): 4K image in 1.12 s, 1.2 GB VRAM,
  embeddings vs llama.cpp mtmd cosine >= 0.99998 mean on 6 images; image rows with IMRoPE positions in the
  decoder, through the MTP layer as llama.cpp does; server image input with llama.cpp's marker layout and the
  image hash in the prompt cache. 4K test 7.7 s end to end (llama.cpp 12.7 s), 6/6 test questions right.

### M8. Later, optional: Linux build on Ubuntu Server 26.04. (1-2 weeks)

**Status 2026-10-07: builds and passes every check under WSL2 (Ubuntu 24.04).** `build.sh`, `CMakePresets.json`
(`win-release`, `linux-release`) and `start-server.sh`. gcc 13.3 needed one change (`setenv` in `test_vision`).
CUDA 13.4.59 (the same nvcc build as Windows) from NVIDIA's `ubuntu2404` repo, with an apt pin that blocks driver
packages; the `wsl-ubuntu` repo stops at 13.3. Results in WSL: tokenizer, template and parser tests pass;
`test_ar`, `test_cache` pass (mapped pinned memory and `cuStreamWaitValue32` work under WSL); logits 4 tokens KLD
0.00052 / 99.27%, 1 token 0.00067 / 98.92%, 32k prompt 0.00096 / 98.44%, 131k 0.00066 / 99.32% (the same as
Windows); greedy split at the smallest gap; `bench/test_server.py` ALL PASS; `bench/vision_answers.py` 6/6. Speed in
WSL (Windows driver underneath, does not count): 23-25 ms per step at 1k. Native Ubuntu (dual boot) is the user's
decision and is not tested.


- Dual boot, driver R615 (open modules), CUDA 13.4 from NVIDIA's `ubuntu2604` repo, persistence mode,
  memory overclock through NVML.
- Re-measure llama.cpp on Linux first (its 2-GPU sum uses NCCL there), then the engine.
- Use the extra VRAM on card 0 for more context.

Total without M8: about 15-21 weeks.

---

## Open decisions

| Decision | Option A | Option B | Recommendation |
|---|---|---|---|
| How to build it | New engine for the hot path; copy llama.cpp code for the rest (quant tables, MMQ prefill kernels, tokenizer, `mtmd` for vision at first). Ceiling 1.4-1.6x decode. | Fork llama.cpp and rewrite the hot spots inside it (GPU sampling under `-sm tensor`, draft loop, GEMV kernels, sums). First gains in 2-4 weeks. Ceiling about 1.2-1.3x (estimate): the graph split at every sum and the CPU-side MTP loop are deep in its design, and every upstream update needs a rebase. | A. M1 is the test: if the new GEMV kernels do not reach 80% of read bandwidth, move them into a llama.cpp fork (B) and stop. |
| KV cache type | f16 (matches llama.cpp, 1x context cost) | q8_0 (saves 6.4 ms/step at 150k, context up to 262144) | f16 for M1-M4 checks, then measure q8_0 and decide with numbers |
| ~~When to install Ubuntu~~ | | | Decided 2026-10-06: Windows only for now. |

Note: removing unused parts from llama.cpp (other models, other backends, the web UI) gives no
speed by itself. The speed comes only from changing the hot path.

---

## Risks

1. **GEMV kernels may be ALU-bound at 4 columns.** llama.cpp's inner loop uses 5.5-12
   instructions per weight for IQ2/IQ3/Q2_K at 4 columns (`gemv-quant.md`). If the new kernels
   cannot pass ~80% of the read bandwidth, the decode gain drops to about 1.25x.
2. **Cross-card sum.** The best sum measured today (27 µs at 40 KiB) is 2.4x the link time. If it
   cannot be improved or hidden, it costs about 3.5 ms per step.
3. **Windows (WDDM).** Launch cost is 7 µs per kernel outside graphs. Long-running kernels hit the
   2 s TDR watchdog. Only one copy engine is reported. All are Windows-only.
4. **Logit match.** llama.cpp numerics (q8_1 activations, fp16 scales, integer truncations in IQ
   dot products) must be copied. Without that, logits differ by about 0.6% of RMS.
5. **Size of the work.** 16-23 weeks. llama.cpp keeps changing; GPU sampling under `-sm tensor`
   could land upstream and shrink the gain.
6. **MTP acceptance is a model property.** The engine cannot raise it (0.60-0.70 today).
7. **P2P on Linux** needs a patched driver, and no AM5 + Blackwell success is reported. The plan
   does not depend on it.
8. **Profiling tools.** Nsight Systems 2025.6.3 does not fully trace driver 616.64 / CUDA 13.4: it
   misses kernels launched outside graphs and all memory copies, and it stops collecting after
   about 50 s. Install a newer Nsight Systems before M1.
9. **The PC freeze** with MTP + `-sm tensor` + long prompts was a race on the KV cache in llama.cpp
   (PR #26827). The new engine must never run two passes on the same KV at once.

What would stop the project: GEMV below 80% of read bandwidth after M1, or cross-card sums above
25 µs at 40 KiB after M2. Then the remaining gain is about 1.2x, which the llama.cpp patch route
reaches with much less work.

---

## Measurement log

All on 2026-10-05, Windows 11, driver 616.64 (WDDM), CUDA 13.4 runtime, both cards PCIe Gen 3 x4.

Machine state before GPU tests: card 0 1827 MiB used (desktop, browser), 2% busy; card 1 20 MiB,
0% busy. Free RAM 17.1 GB (rule asks for 19 GB; browser ~4 GB, WSL ~2.3 GB). Memory clocks
under load: **card 0 14001 MHz (stock), card 1 14201 MHz**. `LEEME.md` expects 14262 MHz on both,
so the Afterburner memory overclock was not fully applied today.

### Hardware (`bench/hw.cu`, `bench/build/hw.exe`)

| Item | Card 0 | Card 1 |
|---|---|---|
| SMs, L2, smem/SM, persisting L2 max | 36, 32 MiB, 100 KiB, 20 MiB | same |
| P2P (`cudaDeviceCanAccessPeer`) | 0 | 0 |
| Read bandwidth, 4 GiB streaming | 430.8 GB/s | 440.0 GB/s |
| Read bandwidth, GEMV-like rows (2208 B / 7488 B) | 427.5 / 426.7 GB/s | 436.0 / 436.3 GB/s |
| Read in 1 kernel per chunk, stream / graph: 1 MiB | 106 / 311 GB/s | 120 / 332 GB/s |
| same, 4 MiB | 276 / 381 GB/s | 292 / 403 GB/s |
| same, 16 MiB | 375 / 381 GB/s | 391 / 428 GB/s |
| same, 32 MiB | 405 / 398 GB/s | 419 / 434 GB/s |
| Kernel launch, host submit (no graph) | 7.05 µs | 6.95 µs |
| Empty kernel in a graph | 0.40 µs | 0.39 µs |
| Small dependent kernel: stream / graph / graph + PDL | 7.83 / 1.61 / 1.25 µs | 7.66 / 1.56 / 1.21 µs |
| Launch + sync round trip | 8.7 µs | 8.8 µs |
| PCIe D2H / H2D 64 MiB | 3.55 / 3.58 GB/s | 3.56 / 3.60 GB/s |
| PCIe both directions at once (64 MiB each) | 3.56 GB/s total | 3.59 GB/s total |
| `asyncEngineCount` | 1 | 1 |
| Both cards D2H at once | 7.09 GB/s total | |
| PCIe D2H 40 KiB (memcpy + sync) | 18.0 µs | 17.2 µs |

Card 0 showed ~40 µs for 4-10 KiB copies (card 1: 7-9 µs). Not explained; card 0 drives the desktop.

Cross-card sum emulation, 128 sums per step, per sum:

| Payload | Host staging + events (like llama.cpp prefill) | Host staging + host sync | Mapped one-shot, plain launches | Mapped one-shot in a CUDA graph (4 blocks) |
|---|---|---|---|---|
| 10 KiB | 64-163 µs | 103-114 µs | 25-36 µs | 8.9 µs |
| 20 KiB | 73-91 µs | 111-117 µs | 31-35 µs | 16.1 µs |
| 40 KiB | 81-86 µs | 120-126 µs | 38-40 µs | 27.6 µs |
| 80 KiB | 86-96 µs | 148-151 µs | 63-64 µs | 50.3 µs |

More blocks (16, 36, 64) made the mapped sum slower, not faster.

### llama.cpp decode (bench copy on port 8081, production flags)

| Run | Context | ms per step | tok/s | Acceptance |
|---|---|---|---|---|
| No profiler | 1k | 39.4 / 39.7 / 39.6 | 72.0 / 71.9 / 72.7 | 0.62-0.64 |
| Nsight Systems | 1k | 40.5 / 40.3 | 75.9 / 75.2 | 0.70 |
| Nsight Systems | 30k | 46.6 / 50.4 | 62.2 / 61.1 | 0.65-0.71 |
| Nsight Systems | 100k | 55.9 / 55.6 | 54.2 / 55.4 | 0.70-0.71 |
| Nsight Systems | 150k | 63.0 / 63.2 | 42.9 / 44.5 | 0.58-0.62 |

The 2026-10-03 baseline (no profiler): 37.7 / 41.0 / 48.3 / 53.9 ms per step. The profiler adds
little at 1k but several ms at depth; the depth rows of the Phase 0 table use the 2026-10-03 numbers.

Profile at 1k (66 steps, `bench/out/q27-a.nsys-rep`): about 2,000 kernels per card per step. 413
GEMV kernels per step (avg 58 µs). About 118 gaps of 20-50 µs per step, after the GEMV and before
the residual add (the cross-card sums; the profiler does not show these kernels in this run). About
11 gaps of 0.1-2 ms per step at pass borders (9.1 ms in total).

Profile at 30k (`bench/out/q27-b.nsys-rep`): prompt 29,893 tokens in 45.1 s (662 t/s), 1546 ms per
1024-token batch, 649 ms busy, 897 ms idle. First decode after the prompt: attention 2.95 ms per
step, cross-card sum kernel 36-40 µs average.

### llama.cpp prompt reading (same runs)

| Context | t/s |
|---|---|
| 0 to 30k | 645-659 |
| 30k to 100k | 504-509 |
| 100k to 150k | 407 |

### Engine, final (2026-10-06, q8_0 KV, server with vision loaded)

Build of 2026-10-06 evening. Server: `build\q27_server.exe` started like `start-server.ps1` (context 262,144).
Tools: `bench\final_bench.py` (`bench\out\final-bench.log`), `q27_gen ... depth 2` (`bench\out\depth_final.log`).

| | 1k | 30k | 100k | 150k |
|---|---|---|---|---|
| Decode ms per step, engine (`q27_gen`, 2 runs x 400 tokens) | 27.3 | 29.4 | 33.7 | 36.9 |
| Decode ms per step, llama.cpp 2026-10-03 | 37.7 | 41.0 | 48.3 | 53.9 |
| Server tok/s, `mide-tps.py` 4 runs x 400 tokens | 101-110 | 93-97 | 79-85 | 75-79 |
| llama.cpp tok/s, same script (`C-rig-base-prob`) | 68-74 | 61-65 | 51-55 | 46-48 |

| Prompt reading | 0-30k | 30k-100k | 100k-150k |
|---|---|---|---|
| Engine t/s (`q27_gen` / server) | 769 / 770 | 612 / 629 | 489 / 490 |
| llama.cpp t/s | 667 | 540 | 439 |

- A resent prompt (prompt cache hit) starts generating after 60-80 ms at any depth.
- 4K image (3840x2160, 4099 prompt tokens, 120-token answer): 7.7-7.8 s end to end (llama.cpp 12.7 s). Encoder
  1.14 s on card 1. A new question on the same image: 1.1-1.5 s.
- Context 262,144 tokens (the trained maximum) with q8_0 KV and vision. VRAM after a 150k prompt and three 4K
  images: 12,087 MiB on card 0 and 12,732 MiB on card 1 (of 16,311).
- MTP acceptance 0.56-0.67 (text dependent), the same range as llama.cpp.

### Head-to-head at the same context (2026-10-06, user request)

Both servers with 180,224 tokens of context (the production setting), vision on, MTP on, the same requests
(`bench\compare.py <label>`, table by `bench\compare_report.py`; results `bench\out\cmp_llama.json`,
`bench\out\cmp_q27.json`). llama.cpp: the production build and flags (`bench\run-llama.ps1 -Ctx 180224`, port 8081).
q27: `q27_server --ctx 180224 --mmproj ...`, q8_0 KV, draft vocabulary 32768. The image rows were run again on both
from a fresh server (`--only images`): in the full run q27's first image request also saved the 150k conversation
to RAM (5.5 GB, inside the 8 GB budget; llama.cpp skips that save because its copy does not fit its 8 GB limit).
The agent loop replays the 11 requests of a real Qwen Code session (`bench\out\qwen\run2\logs`).

| Test | llama | q27 | q27 vs llama |
|---|---|---|---|
| text 1k: generation tok/s (mean of 4) | 76.4 | 116.7 | 1.53x |
| text 1k: prompt t/s (first run, 905 / 905 tokens read) | 517 | 655 | 1.27x |
| text 1k: time to first token, resent prompt (ms) | 181 | 65 | 2.80x |
| text 1k: MTP draft acceptance | 0.618 | 0.609 |  |
| text 30k: generation tok/s (mean of 4) | 69.9 | 105.7 | 1.51x |
| text 30k: prompt t/s (first run, 29893 / 29893 tokens read) | 664 | 789 | 1.19x |
| text 30k: time to first token, resent prompt (ms) | 281 | 68 | 4.14x |
| text 30k: MTP draft acceptance | 0.615 | 0.596 |  |
| text 100k: generation tok/s (mean of 4) | 57.0 | 89.7 | 1.57x |
| text 100k: prompt t/s (first run, 70986 / 83467 tokens read) | 534 | 645 | 1.21x |
| text 100k: time to first token, resent prompt (ms) | 505 | 76 | 6.64x |
| text 100k: MTP draft acceptance | 0.581 | 0.582 |  |
| text 150k: generation tok/s (mean of 4) | 52.5 | 82.9 | 1.58x |
| text 150k: prompt t/s (first run, 51033 / 51552 tokens read) | 427 | 495 | 1.16x |
| text 150k: time to first token, resent prompt (ms) | 682 | 82 | 8.36x |
| text 150k: MTP draft acceptance | 0.608 | 0.613 |  |
| image 800x600 (1036 tokens): total s (mean of 3) | 3.75 | 2.77 | 1.36x |
| image 800x600 (1036 tokens): prompt ms (encoder + reading) | 2124 | 1774 | 1.20x |
| image 4K (4099 tokens): total s (mean of 3) | 12.53 | 7.27 | 1.72x |
| image 4K (4099 tokens): prompt ms (encoder + reading) | 10832 | 6311 | 1.72x |
| agent loop (11 Qwen Code requests): total s | 93.4 | 73.9 | 1.26x |
| agent loop: prompt ms, sum | 55150 | 47269 | 1.17x |
| agent loop: tokens read / prompt tokens | 34617 / 212591 | 34626 / 212591 |  |
| agent loop: generation tok/s (mean) | 85.5 | 126.3 | 1.48x |
| load time s (start to /health) | 8.3 | 10.1 | 0.82x |
| VRAM MiB per card at the end | 14518 + 15390 | 10914 + 11316 |  |

At 100k llama.cpp read fewer prompt tokens (it reused 29k cached tokens, q27 16k: q27's periodic checkpoints are
every 16384 tokens); t/s is per token read.

### Second speed round (2026-10-07)

Server with vision, context 262,144, q8_0 KV (`bench\final_bench.py`, log `bench\out\final-bench-2026-10-07.log`):

| | 1k | 30k | 100k | 150k |
|---|---|---|---|---|
| Server tok/s, `mide-tps.py` 4 runs x 400 tokens | 127-133 | 113-121 | 89-105 | 83-93 |
| Before (Engine, final) | 101-110 | 93-97 | 79-85 | 75-79 |
| llama.cpp | 68-74 | 61-65 | 51-55 | 46-48 |

| Prompt reading (server, first request) | 0-30k | 30k-100k | 100k-150k |
|---|---|---|---|
| Now t/s | 1449 | 1215 | 954 |
| Before (Engine, final) | 769 | 612-629 | 489-490 |
| llama.cpp | 667 | 540 | 439 |

- 4K image end to end: 4.9-6.0 s (before 7.7-7.8 s, llama.cpp 12.7 s).
- VRAM after a 150k prompt and three 4K images: 12,902 / 13,364 MiB.
- These runs came before the 16-warp prompt GEMM tile, which added about 5% prompt speed.

Head-to-head again at 180,224 context (`bench\compare.py q27v2`, results `bench\out\cmp_q27v2.json`, report
`bench\out\compare_report_v2.md`), with the final build. The tables with all three runs (llama.cpp, first version,
second round), the accuracy results, the time split and the link measurements are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

### Third speed round (2026-10-07)

Head-to-head at 180,224 context with a quiet desktop, llama.cpp measured again (`bench\compare.py llama_r3` and
`q27_r3`, log `bench\out\final_r3.log`, results `bench\out\cmp_llama_r3.json`, `cmp_q27_r3.json`):

| | 1k | 30k | 100k | 150k |
|---|---|---|---|---|
| Decode ms per step, engine | 20.9 | 22.9 | 27.3 | 30.5 |
| Decode ms per step, llama.cpp | 37.1 | 39.9 | 47.8 | 52.4 |
| Generation tok/s, engine / llama.cpp | 132.8 / 76.3 | 123.7 / 70.6 | 104.2 / 56.3 | 92.6 / 52.4 |
| Prompt t/s (first run), engine / llama.cpp | 1255 / 523 | 1522 / 664 | 1214 / 535 | 967 / 426 |
| Resent prompt, time to first token, ms | 38 / 175 | 41 / 276 | 49 / 510 | 55 / 670 |

- Images: 800 x 600 2.41 s (1.9-2.0 s without a one-time move of the 150k conversation to RAM; llama.cpp 3.78 s),
  4K 4.93 s (llama.cpp 12.68 s). Qwen Code replay 46.6 s (llama.cpp 90.3 s).
- VRAM at the end 12,530 + 12,268 MiB (llama.cpp 15,214 + 15,424). Load 9.6 s by the engine's own count after the
  load-order fix (13.1 s to /health in the run, before the fix).
- Changes, kernel timers, Nsight Compute findings and what was measured and not done: HANDOFF section H and
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
