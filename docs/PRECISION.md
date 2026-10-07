# Precision and approximations

This document lists every place where qwen27-engine computes with less precision than the model's original BF16
weights would allow, what each one changes, how much accuracy it costs, and how to turn it off. It also states
where the benchmark against llama.cpp is not like for like.

## Contents

- [Summary](#summary)
- [How accuracy is measured](#how-accuracy-is-measured)
- [Number formats in short](#number-formats-in-short)
- [Weights](#weights)
- [Activations](#activations)
- [KV cache](#kv-cache)
- [Cross-card sums](#cross-card-sums)
- [Attention](#attention)
- [Speculative decoding and sampling](#speculative-decoding-and-sampling)
- [Vision](#vision)
- [Benchmark fairness](#benchmark-fairness)
- [What is not measured](#what-is-not-measured)

## Summary

| Item | qwen27-engine | llama.cpp in the benchmark | Measured effect | Switch |
|---|---|---|---|---|
| Weights | 3.55 bits per weight on average (IQ1_M to Q6_K), the GGUF as is | same file | the largest loss of both engines; not measured against BF16 | none |
| Matrix inputs (activations) | int8 with a scale per 32 values (q8_1) | same | same in both engines | none |
| KV cache | **int8 with a scale per 32 values (q8_0)** | **f16** | no measurable change: same KLD within the test noise, all 56 retrieval facts found up to 170k tokens | `Q27_KV=f16` |
| Sums between the two cards | **int8 with a scale per 16 values, with error feedback** | bf16 | same KLD as a bf16 wire | `Q27_WIRE=bf16` |
| Prompt attention Q K^T | **Q rounded to int8 per 32 values**, K exact | Q and K rounded to f16 | lower KLD than the f16 path | `Q27_ATTN_I8=0` |
| Attention P V | f16 with llama.cpp's offset | same | none | none |
| Gated DeltaNet state | f32 | same | none | none |
| Speculative decoding | **block verification**, exact: the output distribution is unchanged | token-by-token rejection rule | none on the distribution | `Q27_ACCEPT=token` |
| MTP block (drafter only) | **re-quantized from Q6_K to Q4_K** at load | Q6_K | none on the output; 0.3-1.1% fewer accepted drafts | `Q27_MTP_TYPE=q6_k` |
| Draft vocabulary | drafts propose only the 32k most frequent tokens | all tokens | none on the output; only on speed | `--draft-vocab 0` |
| Sampling options | `top_k` at most 20; no repetition, presence or frequency penalties | all samplers | a request outside these limits behaves differently | none |

Bold entries depart from what llama.cpp did in the benchmark. The rest is the same in both engines.

## How accuracy is measured

Both engines read the same tokens with the same weights. For every position, each engine produces a probability
distribution over the 248,320 tokens of the vocabulary. Two numbers compare them:

- **Mean KL divergence (KLD)** between the two distributions, in nats. 0 means identical. A mean of 0.001 means
  the distributions are almost the same at nearly every position; a value of 0.1 at one position means a clear
  change of the likely next tokens there.
- **Same top token**: the share of positions where both engines rank the same token first.

The yardstick is llama.cpp itself. llama.cpp's batch mode and its one-token mode differ by **KLD 0.00099** and
**98.82% same top token**, only because the floating-point operations run in a different order. The project's
limit is therefore KLD 0.001 and 98.8% same top token: the engine may not differ from llama.cpp more than
llama.cpp differs from itself.

| Test | Positions | KLD | Same top token |
|---|---|---|---|
| Decode path, 1 token per pass | 2,040 | 0.00055 | 98.92% |
| Verify path, 4 tokens per pass | 2,040 | 0.00055 | 99.31% |
| Prompt path, last 1,024 positions of a 32k prompt | 1,024 | 0.00079-0.00096 | 98.4-99.1% |
| Prompt path, last 1,024 positions of a 131k prompt | 1,024 | 0.00066 | 99.32% |

These tests run with all defaults (q8_0 KV cache, int8 wire, int8 prompt attention) against llama.cpp with an f16
KV cache, so they include every approximation in the table above. The range at 32k comes from the prompt batch
size, which only changes the order of float additions in attention.

## Number formats in short

| Format | Bits | Precision of one value |
|---|---|---|
| f32 | 32 | relative error at most 0.000006% |
| f16 | 16 | relative error at most 0.05% |
| bf16 | 16 | relative error at most 0.4% |
| int8 with a scale per block (q8_0, q8_1) | 8 + a shared scale | absolute error at most 1/254 of the largest value in the block |
| fp8 (e4m3) | 8 | relative error at most 6.25% |

An int8 block format and fp8 use the same 8 bits per value. They spend them differently. For values close to the
largest value of their block, int8 is 16 times more precise than fp8. Only for values below 1/16 of the block's
largest value is fp8 more precise, and such values add little to a dot product. For the dot products of attention
the int8 block format has the smaller error. The engine does not use fp8 anywhere.

## Weights

The GGUF holds 27.3 billion weights in 12.1 GB:

| Type | Share of weights | Bits per weight |
|---|---|---|
| IQ3_S | 31.0% | 3.44 |
| IQ4_XS | 20.8% | 4.25 |
| IQ3_XXS | 19.8% | 3.06 |
| Q4_K | 10.3% | 4.50 |
| IQ2_S (includes the token embedding table) | 9.1% | 2.56 |
| IQ2_XS, Q2_K, IQ2_XXS, IQ1_M | 7.2% | 1.75-2.62 |
| Q6_K (MTP block; the engine re-quantizes it to Q4_K for the drafts) | 1.6% | 6.56 |
| BF16, F32 (small gates and norms) | 0.1% | 16-32 |

This quantization is the largest approximation of both engines: 3.55 bits per weight on average, against 16 bits
in the original model. It is the same file in both engines, so it does not affect the comparison. The engine
decodes every weight to the exact value stored in the GGUF; the repacking at load time only moves bytes.

## Activations

Every matrix multiply takes its input as int8 with one fp16 scale per 32 values (the q8_1 format), as llama.cpp
does in its MMQ and MMVQ kernels. The engine keeps llama.cpp's rounding rules for this format, so this step adds
no difference between the two engines.

## KV cache

The KV cache stores the keys and values of every token for the 16 attention layers and the MTP layer. By default
it uses q8_0: int8 values with one fp16 scale per 32 values, the same format and rounding as llama.cpp's q8_0
cache. The f16 cache is available with `Q27_KV=f16` (server: `-Kv f16` in `start-server.ps1`).

| | q8_0 (default) | f16 |
|---|---|---|
| Memory per token per card | 18 KiB | 34 KiB |
| Largest context with vision on this rig | 262,144 (the model's maximum) | 171,264 |
| Decode step at 150k context | 31.9 ms | 37.4 ms |
| Prompt reading 100k → 150k | 974 tok/s | 741 tok/s |
| KLD and same top token against llama.cpp (f16 cache), 32k prompt | 0.00096, 98.44% | 0.00089, 98.44% |
| KLD and same top token against llama.cpp (f16 cache), 131k prompt | 0.00066, 99.32% | 0.00069, 99.32% |

A KLD average can hide a problem with long-range recall: one wrong fact deep in the context changes few
positions. The retrieval test (`bench\needle.py`) checks this directly. It hides 8 facts (a 6-digit code per
warehouse name) at depths from 5% to 95% of a long prompt of source code, then asks for each fact with greedy
decoding and thinking off.

| Retrieval test | llama.cpp, f16 KV | Engine, f16 KV | Engine, q8_0 KV |
|---|---|---|---|
| 8 facts in a 32k-token prompt | 8 of 8 | 8 of 8 | 8 of 8 |
| 8 facts in a 100k-token prompt | 8 of 8 | 8 of 8 | 8 of 8 |
| 8 facts in a 170k-token prompt | 8 of 8 | 8 of 8 | 8 of 8 |
| 32 facts and 32 look-alike distractors (keys with two digits swapped) in a 170k-token prompt | 32 of 32 | 32 of 32 | 32 of 32 |

Results: `bench\out\needle_*.json`. The test runs with `bench\needle.py <label> [sizes] [--hard]` against any
OpenAI-compatible server.

Both measurements agree: on these tests the q8_0 cache changes nothing that the tests can detect. The KLD
difference between the two cache types (0.00007 at 32k, 0.00003 at 131k, in opposite directions) is smaller than
the variation that a different prompt batch size causes. Both caches found every hidden fact at every depth.

The tests have limits. A pass/fail retrieval test cannot show a small loss of confidence. The text is source code;
other content, such as long tables of numbers, was not tested. Contexts above 170k tokens were not tested. When
in doubt, use the f16 cache (`Q27_KV=f16`). It costs decode speed at long context (15% at 150k) and context length
(171,264 instead of 262,144 tokens with vision on this rig).

## Cross-card sums

The two cards add their partial results 128 times per pass. llama.cpp sends these partials as bf16. The engine
sends int8 with one fp16 scale per 16 values, which needs 44% fewer bytes on the PCIe link. On its own this
would add a rounding error at every sum. The engine therefore keeps each card's rounding error and adds it to that
card's next partial (error feedback). The rounding errors then cancel over the layers instead of adding up, and
the error in the residual stream stays at the size of a single rounding.

| Wire | 1-token KLD | 4-token KLD | 32k prompt KLD | 131k prompt KLD |
|---|---|---|---|---|
| bf16 (as llama.cpp) | 0.00066 | 0.00059 | 0.00086 | 0.00063 |
| int8, scale per 16, error feedback (default) | 0.00055 | 0.00055 | 0.00090 | 0.00063 |
| int8, scale per 32, no error feedback (rejected) | | 0.00075 | 0.00142 | |

Each column compares the wires on the same build; the 32k and 131k columns were measured with 512-token prompt
batches. The version with one scale per 32 values failed at one position of the 32k test (KLD 0.43 with error
feedback) and was dropped.

## Attention

- **Decode** reads the KV cache with f16 tensor cores. Q is rounded to f16 and P to f16, as in llama.cpp.
- **Prompt reading** computes Q K^T on the int8 tensor cores. K is used as the exact int8 data of the q8_0 cache.
  Q is rounded to int8 with an f32 scale per 32 dimensions. The f16 path (llama.cpp's method) instead rounds the
  dequantized K values and Q to f16. On the 131k test with 512-token batches, the int8 path gave KLD 0.00034
  against 0.00066 for the f16 path. P V stays in f16 with llama.cpp's offset in both paths.
- With an f16 KV cache, prompt attention uses the f16 path.

## Speculative decoding and sampling

- The MTP head drafts 3 tokens and the full model verifies them with block verification (Sun et al., ICLR 2025).
  Like llama.cpp's token-by-token rule (keep a draft with probability min(1, p/q), resample a rejected position from
  max(0, p - q)), it keeps the output distribution equal to plain sampling from the full model; it accepts slightly
  more drafts (+0.1-0.6% tokens per step here). `build\test_accept.exe` checks both rules: over millions of
  simulated steps on synthetic distributions, the frequency of every emitted 3- or 4-token sequence matches the
  target probability (chi-square below the limit), and a deliberately wrong rule fails by a factor of 2,500 or more.
  Speculative decoding changes only the speed.
- The engine re-quantizes the MTP block from Q6_K to Q4_K (7.5% weight error relative to the Q6_K values, checked
  with `tools\check_requant.py`). Only the drafts use it, so it changes which tokens are proposed, not the output.
- With greedy decoding, speculative and plain decoding give the same tokens until the first position where the
  top two logits are almost equal (in the test: identical for 140 tokens, then a split at a gap of 0.004). The
  4-token verify pass adds floats in another order than a 1-token pass, which decides such near ties.
- The draft vocabulary only limits which tokens the drafts can propose. The verify pass scores all 248,320 tokens,
  so the output distribution does not change.
- Random numbers come from a counter-based hash instead of llama.cpp's generator. The distribution is the same;
  the sampled sequences differ from llama.cpp's for the same seed.
- `top_k` is limited to 20 (the default of the model). Repetition, presence and frequency penalties are ignored.

## Vision

The image is decoded and resized exactly as llama.cpp does (the input tensor is bit-identical). The encoder uses
BF16 tensor cores with f32 accumulation. Against llama.cpp's encoder (`mtmd`) the image embeddings have a mean
cosine similarity of at least 0.99998 on six test images. The engine answers all six test questions about five
test images correctly (`benchision_answers.py`).

## Benchmark fairness

The head-to-head table in the README compares the engine with its defaults against llama.cpp with the production
settings of this PC. Two differences matter:

1. **KV cache: q8_0 against f16.** llama.cpp ran with an f16 cache because its q8_0 cache with `-sm tensor` leaves
   the second card idle on this setup and is slower. The q8_0 cache makes the engine's decode faster at long
   context, because attention reads half the bytes. The list below gives the engine's numbers with an f16 cache.
2. **Cross-card wire: int8 against bf16.** It has no measurable cost in accuracy (see above).

With an f16 cache in the engine as well (the like-for-like table in
[PERFORMANCE.md](PERFORMANCE.md#like-for-like-both-engines-with-an-f16-kv-cache)):

- decode steps are 1.63x faster than llama.cpp at 1k context, 1.46x at 100k and 1.43x at 150k (with q8_0:
  1.65-1.67x at every depth);
- prompt reading is 2.34x faster at 1k and 1.73x from 100k to 150k (with q8_0: 2.27-2.41x);
- the replay of a real Qwen Code session takes 49.5 s against 93.4 s for llama.cpp (with q8_0: 48.6 s).

So the kernels, the overlapped prompt reading and the GPU-side speculative decoding account for most of the gain.
The q8_0 cache adds the rest at long context. Both configurations give the same accuracy on the tests above.

## What is not measured

- **Quality against the original BF16 model.** Both engines run the same 3.55-bit file. How much the quantized
  file loses against the original weights was not measured here.
- **Task benchmarks.** No coding or reasoning test suites were run. The accuracy evidence is the distribution
  comparison with llama.cpp, the retrieval test and real Qwen Code sessions.
- **Recall beyond 170k tokens.** The retrieval test stops at 170k because the benchmark servers ran with a
  180,224-token context.
