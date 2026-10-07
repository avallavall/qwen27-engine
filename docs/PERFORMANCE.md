# Performance

This document gives the measured speed and accuracy of qwen27-engine, how they were measured, and where the time
goes. The design behind the numbers is in [ARCHITECTURE.md](ARCHITECTURE.md). The raw measurement log of the whole
project is in [PLAN.md](../PLAN.md).

## Contents

- [Test system](#test-system)
- [Method](#method)
- [Results against llama.cpp](#results-against-llamacpp)
- [Accuracy](#accuracy)
- [Where the time goes](#where-the-time-goes)
- [Hardware limits](#hardware-limits)
- [Optimization history](#optimization-history)
- [What a faster PCIe link would give](#what-a-faster-pcie-link-would-give)
- [Reproduce](#reproduce)

## Test system

| Part | Value |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 5060 Ti 16 GB (36 SMs, 32 MB L2, GDDR7), driver 616.64 (WDDM) |
| Link | each card on PCIe 3.0 x4; no peer-to-peer access (GeForce on Windows) |
| CPU, RAM | AMD Ryzen 5 9600X, 32 GB |
| OS, toolchain | Windows 11, CUDA 13.4, MSVC 2022 |
| Card 0 | also drives the desktop (about 0.8-1.6 GB in use, a few % of its time) |
| Model | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB) with `mmproj-Qwen3.8-27B-BF16.gguf` |
| llama.cpp | the production build of this PC: `-sm tensor`, MTP drafting (3 drafts, probabilistic), flash attention, f16 KV cache (`bench\run-llama.ps1`) |

## Method

- **Head-to-head** (`bench\compare.py`): both servers start with 180,224 tokens of context, vision and MTP on, and
  answer the same requests. Text runs use prompts of 1k, 30k, 100k and 150k tokens; the first run of each depth
  reads the prompt, the next three resend it (prompt cache). Each run generates 400 tokens at the model's default
  sampling settings. The image runs change one pixel per run, so no cache can reuse the image. The agent run
  replays the 11 requests of a recorded Qwen Code session in order.
- **Decode step time** is generation time divided by the number of steps (tokens minus accepted drafts). Unlike
  tok/s it does not depend on how many drafts happen to be accepted.
- **Accuracy** compares the full output distributions of both engines on the same tokens (mean KL divergence and
  same top token), with llama.cpp's `llama-perplexity` or with own reference files from `bench\llama_ref.exe`.
- **Kernel times** come from GPU time stamps inside the CUDA graphs (`Q27_PROF=1`). Nsight Systems cannot trace the
  engine's graphs with this driver.
- Card 0 also drives the desktop, which adds noise of about ±2%. Settings were compared with `bench\ab.py`, which
  runs them in turns.

## Results against llama.cpp

Three runs of the same head-to-head script. "2026-10-06" is the engine's first complete version, "2026-10-07" the
current one. Both engine runs use the engine's default q8_0 KV cache; llama.cpp uses an f16 cache. The next
section repeats the run with an f16 cache in the engine.

| Test | llama.cpp | Engine 2026-10-06 | Engine 2026-10-07 | Now vs llama.cpp |
|---|---|---|---|---|
| Decode step at 1k context, ms | 37.1 | 24.2 | **22.5** | 1.65x |
| Decode step at 30k, ms | 40.4 | 26.4 | **24.3** | 1.66x |
| Decode step at 100k, ms | 47.6 | 30.6 | **28.6** | 1.66x |
| Decode step at 150k, ms | 53.3 | 34.3 | **31.9** | 1.67x |
| Generation at 1k, tok/s | 76 | 117 | **122** | 1.60x |
| Generation at 30k, tok/s | 70 | 106 | **115** | 1.65x |
| Generation at 100k, tok/s | 57 | 90 | **99** | 1.73x |
| Generation at 150k, tok/s | 52 | 83 | **84** | 1.61x |
| Prompt reading to 1k, tok/s | 517 | 655 | **1,245** | 2.41x |
| Prompt reading to 30k, tok/s | 664 | 789 | **1,507** | 2.27x |
| Prompt reading 30k → 100k, tok/s | 534 | 645 | **1,249** | 2.34x |
| Prompt reading 100k → 150k, tok/s | 427 | 495 | **974** | 2.28x |
| Resent 1k prompt, time to first token, ms | 181 | 65 | **61** | 2.96x |
| Resent 150k prompt, time to first token, ms | 682 | 82 | **78** | 8.76x |
| Image 800 x 600 (1,036 tokens), total s | 3.75 | 2.77 | **2.03** | 1.85x |
| Image 3840 x 2160 (4,099 tokens), total s | 12.53 | 7.27 | **5.24** | 2.39x |
| Qwen Code session replay, total s | 93.4 | 73.9 | **48.6** | 1.92x |
| ... of which prompt reading, s | 55.2 | 47.3 | **25.9** | 2.13x |
| VRAM per card at the end, GB | 14.5 + 15.4 | 10.9 + 11.3 | 11.5 + 11.9 | |
| Load time, s | 8.3 | 10.1 | 10.3 | |

Notes:

- Draft acceptance was 0.57-0.62 in all runs. At 150k the current run had the lowest acceptance (0.565 against
  0.613 before), so its tok/s gain is smaller than its step-time gain.
- At 100k llama.cpp reused 29k tokens from its cache and read 70,986 tokens; the engine read 83,467 (its periodic
  checkpoints are every 16,384 tokens). The rate is per token read.
- In the agent replay both engines read the same 34.6k of 212.6k prompt tokens; the rest came from the prompt
  cache.
- The current engine uses 0.6 GB more VRAM per card than the first version because prompt batches grew from 512
  to 2048 tokens.

### Like for like: both engines with an f16 KV cache

The same head-to-head with `Q27_KV=f16` in the engine (`bench\out\cmp_q27v2_f16.json`). With an f16 cache, prompt
attention also uses the f16 Q K^T path, as llama.cpp does.

| Test | llama.cpp, f16 KV | Engine, f16 KV | Engine, q8_0 KV (default) | f16 vs llama.cpp |
|---|---|---|---|---|
| Decode step at 1k / 30k / 100k / 150k, ms | 37.1 / 40.4 / 47.6 / 53.3 | 22.7 / 25.6 / 32.7 / 37.4 | 22.5 / 24.3 / 28.6 / 31.9 | 1.63 / 1.58 / 1.46 / 1.43x |
| Generation at the same depths, tok/s | 76 / 70 / 57 / 52 | 124 / 114 / 85 / 73 | 122 / 115 / 99 / 84 | 1.39-1.64x |
| Prompt reading to 1k / 30k, 30k → 100k, 100k → 150k, tok/s | 517 / 664 / 534 / 427 | 1,212 / 1,390 / 1,031 / 741 | 1,245 / 1,507 / 1,249 / 974 | 1.73-2.34x |
| Image 3840 x 2160, total s | 12.53 | 4.99 | 5.24 | 2.51x |
| Qwen Code session replay, total s | 93.4 | 49.5 | 48.6 | 1.89x |
| VRAM per card at the end, GB | 14.5 + 15.4 | 14.3 + 14.7 | 11.5 + 11.9 | |

The kernels and the speculative decoding give the engine its lead at every depth. The q8_0 cache adds speed only
where attention reads a lot of cache: 1% of the step time at 1k, 15% at 150k. It also halves the cache memory, so
the full 262,144-token context fits with vision. Its effect on accuracy is in [PRECISION.md](PRECISION.md#kv-cache).

## Accuracy

All results above use the defaults: a q8_0 KV cache, the int8 wire between the cards and int8 prompt attention.
On the same tokens, the engine's output distribution differs from llama.cpp's (f16 KV cache) by a mean KL
divergence of 0.0005-0.001, with the same top token at 98.4-99.3% of positions. llama.cpp's own batch and
one-token modes differ by 0.001 and 98.8%. [PRECISION.md](PRECISION.md) lists every approximation with its
measured cost, the long-context retrieval test, and the comparison with an f16 KV cache.

## Where the time goes

### One decode step at 1k context (22.6 ms, card 1)

| Part | ms | Share | Floor |
|---|---|---|---|
| Weight GEMVs of the 4-token verify pass (5.7 GB) | 15.2 | 67% | 13.2 ms at 430 GB/s |
| Cross-card sums (128 per pass, about 27 µs each) | 3.5 | 15% | about 15 µs each on this link |
| Other kernels of the pass (delta rule, attention, norms) | 1.1 | 5% | |
| MTP catch-up and 3 drafts | 2.8 | 12% | |
| Sampling and acceptance | 0.1 | <1% | |

Two limits remain. The GEMVs reach about 385 GB/s on average against 430-440 GB/s possible. The cross-card sums
cost 3.5 ms, of which roughly half is link time and half is latency and waiting for the slower card. Attention
grows with the context: at 150k a step takes 9.4 ms more than at 1k.

### One prompt batch at 30k context (2048 tokens, card 1, 1.36 s)

| Part | Share |
|---|---|
| Quantized GEMMs (int8 tensor cores) | about 70% |
| Attention | about 8% (grows with depth) |
| Waiting for the cross-card exchange | about 5% |
| Gated DeltaNet recurrence | about 5% |
| Norms, SwiGLU, gates, other | about 10% |

With the exchange switched off (test switch `Q27_PF_NOEX=1`, wrong results) a batch takes 1.17 s, 14% less. Part
of that cost is the wait above; the rest is copy-engine traffic that slows the GEMMs a little.

The GEMMs reach 66-72 TOPS for IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS, and 40-45 TOPS for Q2_K, Q4_K, Q6_K, IQ2_S,
IQ2_XS and IQ1_M (`bench_gemm` at 1024 rows). The specified int8 tensor-core peak of the card is about 190 TOPS.
A likely limit is the f32 work that applies the per-32-weight scales after every tensor-core instruction.
Confirming it needs Nsight Compute, which requires the GPU performance-counter permission on this PC.

## Hardware limits

Measured on this rig with `bench\hw.cu` and `tools\bench_link2.cu`:

| Quantity | Card 0 | Card 1 |
|---|---|---|
| Memory read bandwidth, large stream | 431 GB/s | 440 GB/s |
| Kernel launch, outside / inside a CUDA graph | 7.0 / 1.6 µs | 7.0 / 1.6 µs |
| Copy engine, device to host / host to device | 3.55 / 3.58 GB/s | 3.56 / 3.60 GB/s |

Link from a kernel to mapped pinned host memory (40 KB per kernel, the size of one cross-card sum; the same on
both cards, alone or together):

| Access | Rate |
|---|---|
| Writes, 16-byte stores | 2.96 GB/s |
| Reads, 16-byte volatile loads | 3.0 GB/s |
| Reads, 2-byte volatile loads | 2.7 GB/s |
| Reads with `ld.relaxed.sys` or `ld.global.cv` | 1.6 GB/s |
| Reads and writes at the same time, total | 2.4-2.7 GB/s |

The last line decides the design of the cross-card sum. The link does not give extra bandwidth when it reads and
writes at the same time, so only fewer bytes make a sum shorter.

## Optimization history

### First version (2026-10-06)

Built in milestones from a one-card decode pass to the full server. Each milestone passed the logit tests and is
described with its numbers in [PLAN.md](../PLAN.md).

### Second round (2026-10-07)

Each change was measured alone with the time stamps and `bench\ab.py`. Effects at 1k context for decode and at 30k
for prompt reading:

| Change | Effect |
|---|---|
| Cross-card sums as int8 with error feedback (was bf16), 4 MB L2 prefetch | decode -1.0 ms per step, prompt +5% |
| L2 prefetch only in kernels that wait for the other card | decode -0.15 ms |
| Draft head rows split evenly over the cards (was 204 µs per draft on card 0, 23 µs on card 1) | decode -0.6 ms |
| Independent GEMVs on parallel graph branches | decode -0.65 ms |
| Prompt GEMM: loads issued one block ahead, weight tiles shared in L2, 16-warp tiles | IQ3_S 33-41 → 66-69 TOPS |
| Prompt alpha/beta GEMM split over K | 506 → 90 µs per call |
| Prompt delta rule with shared-memory staging | 445 → 164 µs per call |
| Prompt attention with int8 Q K^T, 16-position tiles, split-KV | 32.5 → 66 TFLOPS |
| Prompt batch 2048 tokens (was 512) | prompt +18% |
| Prompt-cache checkpoint 512 tokens before the end of a long last message | a new question about the same 32k document: 10.8 s → 0.8 s (llama.cpp 2.1 s) |

Tried and not kept:

| Idea | Result |
|---|---|
| Stream-K split of the decode GEMV (equal work per warp) | 0.2 ms slower per step |
| Programmatic dependent launch for the GEMV after each sum | same gain as a smaller L2 prefetch |
| int8 wire with one scale per 32 values | one test position reached KLD 0.43 |
| Warp-per-head gated RMSNorm in prompt reading | one sensitive test position reached KLD 1.0 |
| 16 warps per CTA for small prompt GEMMs | slower below 256 rows and for Q6_K |

## What a faster PCIe link would give

The test rig runs each card at PCIe 3.0 x4. Estimates from the time split above:

| Link per card | Decode | Prompt reading |
|---|---|---|
| PCIe 4.0 x4 (2x bandwidth) | about +4% | up to about +8% |
| PCIe 5.0 x8 or 3.0 x16 (4-8x) | about +7% | up to about +15% |

Decode gains less than the bandwidth ratio suggests, because half of each sum's time is latency. Prompt reading
already hides most of the link time behind compute.

## Reproduce

| Measurement | Command |
|---|---|
| Head-to-head | start a server with `--ctx 180224`, then `python bench\compare.py <label>`; `python bench\compare_report.py llama <label>` |
| Server benchmark at 4 depths | `python bench\mide-tps.py <label> 1000 1000 30000 30000 ... --gen 400` or `python bench\final_bench.py` |
| Decode and prompt speed without the server | `build\q27_gen.exe <model> <tokens> 0,1 1000,30000,100000,150000 400 depth 2` |
| Time per kernel group | `set Q27_PROF=1`, then `build\q27_gen.exe <model> <ref> 0,1 1000 400 sample` |
| A/B of settings | `python bench\ab.py 2 "a=Q27_WIRE=bf16" "b=Q27_WIRE=q8b16"` (`BENCH_MODE=depth:30000` for prompt speed) |
| Kernel benchmarks | `build\bench_gemv.exe`, `build\bench_gemm.exe <model> 1 1024`, `build\bench_attn.exe 1 131072 q8 prefill`, `build\bench_link2.exe` |
| Accuracy | see [Tests and benchmarks](../README.md#tests-and-benchmarks) |
