# Attention: the 16 full-attention layers and the MTP attention

Scope: decode and prefill attention for the 16 full-attention blocks (3, 7, ..., 63)
and the MTP block `blk.64`, at 180224 tokens and beyond. Research only. No GPU work
was done. Every number is either cited or marked "estimate" with the arithmetic.

## Summary

1. **KV size.** One layer stores 4 KiB per token in f16 and 2176 B in q8_0. The 16
   layers store 64 KiB (f16) or 34 KiB (q8_0) per token. The MTP layer adds 4 KiB or
   2.125 KiB. With 2 KV heads per card, each card reads **38 KiB per token per decode
   step in f16** (verify once, MTP draft KV 3 times) and **20.2 KiB in q8_0**.
2. **llama.cpp is already near the f16 floor at depth.** At 400 GB/s the f16 floor
   grows 0.097 ms per 1000 tokens. The measured step grows 0.109 ms per 1000 tokens.
   So the depth-dependent part of the step is only 10-17% above the KV-read floor.
   Part of the gap is a host-built attention mask that llama.cpp uploads every pass.
3. **q8_0 KV is the only large lever for depth cost.** It saves 6.8 ms per step at
   150k and 8.2 ms at 180k (floor arithmetic). It also halves KV VRAM. Published KL
   numbers for q8_0 KV are small (99% KLD 0.0064 on Qwen3.8-27B at 65k context,
   close to the 0.0059 of a bf16 K cache).
4. **llama.cpp today:** for head dim 256 on sm_120 it runs the `mma.sync` FlashAttention
   kernel (`fattn-mma-f16.cuh`) with stream-K split, for 1 and for 4 query tokens. It
   pads the 6 query heads of each KV head to 8 slots, so 25% of the tensor-core work is
   wasted. With q8_0 KV and 4 tokens it converts the whole KV cache to f16 on every
   call, which makes q8_0 slower than f16 there. A new kernel fixes both.
5. **Decode kernel design:** one CTA works on one KV head and one chunk of positions,
   with all 24 query rows (6 heads x 4 tokens) in the tile. Use `mma.sync` m16n8k16
   (f16 in, f32 accumulate). Split heads 2+2 across the cards. Fuse the combine step
   with the sigmoid gate.
6. **Prefill attention** costs 393,216 x d FLOP per token at depth d (16 layers). At
   the estimated 94.8 TFLOPS of 2 cards (f16 in, f32 accumulate), that is 0.12 ms per
   token at 30k, 0.42 ms at 100k, 0.62 ms at 150k. From the measured prompt speeds,
   llama.cpp runs attention at about 54% of that peak. Beyond about 124k depth the
   attention FLOPs per token exceed all weight-matmul FLOPs.
7. **Long context:** 262144 with q8_0 fits easily on both OSes (4624 MiB KV per card).
   With f16, the estimate is about 237k tokens on Windows and 249k on headless Linux.
   The GGUF has no RoPE scaling keys, so 262144 is the model limit.

---

## 1. KV bytes and the decode bandwidth floor

### 1.1 Bytes per token

Model facts (from `research/_gguf-model-dump.txt:27-33`): 24 query heads, 4 KV heads,
key and value length 256. `blk.3.attn_k.weight` is [5120, 1024] = 4 x 256
(`research/_gguf-model-tensors.tsv`). The MTP block has the same shape
(`blk.64.attn_k.weight` [5120, 1024]).

q8_0 stores 32 values in 34 bytes (2-byte f16 scale + 32 int8), so 1.0625 B per value.

| Item | f16 | q8_0 |
|---|---|---|
| K per token per layer (4 heads x 256) | 2048 B | 1088 B |
| K + V per token per layer | **4096 B (4 KiB)** | **2176 B** |
| 16 target layers, per token | **64 KiB** | **34 KiB** |
| MTP layer, per token | **4 KiB** | **2.125 KiB** |
| Total per token (both cards) | 68 KiB | 36.125 KiB |
| Per card with 2 KV heads per card | 34 KiB | 18.06 KiB |

The f16 numbers match the measured 64 KiB + 4 KiB from the brief. At 180224 tokens the
f16 cache is 11,968 MiB in total (5,984 MiB per card). LEEME says "unos 12.250 MiB"
(`qwen38_27\LEEME.md:355`). That figure is about 12.55e9 bytes, which is 11,968 MiB, so
the two agree once the units match.

### 1.2 What one decode step reads

A step in the baseline is one verify pass of 4 tokens (1 sampled + 3 drafts) plus 3 MTP
draft passes. llama.cpp's MTP driver merges the catch-up rows into the first draft
decode (`llama-rig2\common\speculative.cpp:1416-1420`), so there are 3 MTP passes per
step. Each pass reads the full visible KV of its layer once.

In `-sm tensor` mode llama.cpp splits the KV cache by head (axis 2) between the cards
(`llama-rig2\ggml\src\ggml-backend-meta.cpp:911-929`). Each card holds and reads 2 of the
4 KV heads. A new engine should do the same (section 2.4).

Bytes per card per step at depth d:

- f16: (16 x 2048 + 3 x 2048) x d = **38,912 B x d** (38 KiB per token)
- q8_0: (16 x 1088 + 3 x 1088) x d = **20,672 B x d** (20.19 KiB per token)

### 1.3 Floor per step (per card, cards run in parallel)

Bandwidth: 400 GB/s (measured copy 388-404 GB/s, brief). 448 GB/s is the datasheet
number (28 Gbps x 128 bit, [TechPowerUp review](https://www.techpowerup.com/review/palit-geforce-rtx-5060-ti-infinity-3-16-gb/)).

| Depth | f16 GB/card | f16 ms @400 | of which MTP | q8_0 ms @400 | q8_0 saves |
|---|---|---|---|---|---|
| 1k | 0.039 | 0.10 | 0.02 | 0.05 | 0.05 |
| 30k | 1.17 | 2.92 | 0.46 | 1.55 | 1.37 |
| 100k | 3.89 | 9.73 | 1.54 | 5.17 | 4.56 |
| 150k | 5.84 | 14.59 | 2.30 | 7.75 | 6.84 |
| 180224 | 7.01 | 17.53 | 2.77 | 9.31 | 8.22 |
| 262144 | 10.20 | 25.50 | 4.03 | 13.55 | 11.95 |

At 448 GB/s every number is 11% lower (150k f16: 13.03 ms).

### 1.4 Comparison with the measured step growth

Measured: 37.7 ms (1k), 41.0 (30k), 48.3 (100k), 53.9 (150k) (brief). Growth from 1k:

| Depth | Measured growth | f16 floor growth @400 | Ratio |
|---|---|---|---|
| 30k | 3.3 ms | 2.82 ms | 1.17 |
| 100k | 10.6 ms | 9.63 ms | 1.10 |
| 150k | 16.2 ms | 14.49 ms | 1.12 |

Slope: measured 0.109 ms per 1000 tokens. Floor 0.097 ms per 1000 tokens (f16, MTP
included, 400 GB/s). The first estimate in the brief (0.08) left out the 3 MTP reads.

Conclusion: in f16, llama.cpp's attention already streams the KV at close to full
bandwidth. A new f16 engine can win at most about 1-2 ms at 150k on attention itself.

A likely part of the remaining 10-17%: llama.cpp builds the KQ mask on the host
(`llama-rig2\src\llama-kv-cache.cpp:1759`, asserts a host buffer) with shape
n_kv x n_tokens in f16 (`llama-rig2\src\llama-graph.cpp:29-46`, no padding). The mask is
mirrored to both cards (`ggml-backend-meta.cpp:912`). Per step at 150k, about 8-10 rows
(4 verify + draft rows) x 150k x 2 B = 2.4-3 MB per card goes over PCIe Gen3 x4
(estimate: about 0.7-0.9 ms at 3.5 GB/s, plus the CPU time to fill it). A new engine
needs no mask: the causal rule comes from positions inside the kernel.

---

## 2. Decode kernel design

### 2.1 What llama.cpp runs today (sm_120, head dim 256, f16 KV)

Selection is in `llama-rig2\ggml\src\ggml-cuda\fattn.cu:541-720`.

- `gqa_ratio` = 12 / 2 = 6 per card. The vector kernel is excluded for f16 when
  `gqa_ratio > 4 && Q->ne[0] >= 256` (`fattn.cu:645-647`). So the result is
  `BEST_FATTN_KERNEL_MMA_F16` (`fattn.cu:664`) for 1 token and for 4 tokens.
- `gqa_ratio > 4` gives `ncols2 = 8` (`fattn.cu:266-268`). The 6 query heads of one KV
  head go into 8 column slots. **2 of 8 slots are empty: 25% of the MMA work is
  padding.** This also applies to prefill.
- 1 token (MTP draft pass): `ncols1 = 1`, tile of 8 columns (`fattn.cu:170-174`).
- 4 tokens (verify): `ncols1 = 4`, tile of 32 columns, 24 used (`fattn.cu:184-187`).
- sm_120 uses the Ampere config table (`fattn-mma-f16.cuh:234-237`). For D=256:
  8 columns: 128 threads, 64 KV rows per softmax step, 2 pipeline stages; 32 columns:
  128 threads, 32 KV rows per step (`fattn-mma-f16.cuh:69-72`).
- Split over the KV length: stream-K is always used on Ada and newer
  (`fattn-common.cuh:1144-1179`), with a separate fixup kernel that merges partial
  results (`fattn-common.cuh:1261-1296`). The sparse gather path is off for this model,
  because `n_kv_max` is 0 for normal attention (`llama-graph.cpp:2825`, `fattn.cu:142-150`).

With q8_0 KV (what would run if `-sm tensor` worked with it):

- 1 or 2 query tokens: the vector kernel, which reads q8_0 directly
  (`fattn.cu:650-653`). It runs one CUDA block per **query** head
  (`fattn-vec.cuh:106-111`), so each KV head is read by 6 blocks. Reuse then depends on
  L2 hits.
- 4 query tokens (MTP verify): the MMA kernel. It needs f16 K and V
  (`fattn.cu:737-742`), so `launch_fattn` converts the **whole visible K and V** of the
  layer to f16 into a scratch buffer on every call (`fattn-common.cuh:1031-1093`, buffer
  size in `fattn-common.cuh:53-85`). Traffic per value: read 1.06 B + write 2 B + read
  2 B = 5.06 B, against 2 B for f16. So in llama.cpp, q8_0 KV makes long-context MTP
  verify slower, not faster. It also costs a scratch buffer of 2 KiB x n_kv per card
  (352 MiB at 180k).

### 2.2 Prior art

| Kernel | Where | Tensor cores | GQA handling | Split-KV | Quantized KV |
|---|---|---|---|---|---|
| llama.cpp MMA | `fattn-mma-f16.cuh` | mma.sync | heads in N, power-of-2 slots (8 for ratio 6) | stream-K + fixup kernel | converts to f16 first |
| llama.cpp VEC | `fattn-vec.cuh` | no | one block per query head | `parallel_blocks` + combine kernel | reads q8_0 directly (Q as q8_1) |
| FlashInfer decode | `refs/flashinfer/include/flashinfer/attention/decode.cuh` | no | `threadIdx.y` = query head in the group (`decode.cuh:239`), K/V tiles in shared memory shared by the group | chunks of `max(ceil(len/max_chunks), 256)` (`decode.cuh:724`) + `MergeStates` (`cascade.cuh:220`) | f16/bf16/fp8 |
| FlashInfer decode with `use_tensor_cores=True` | `flashinfer/decode.py:1063-1065` | mma.sync (prefill kernel) | packed rows = tokens x group size (`scheduler.cuh:557-562`) | yes | f16/bf16/fp8 |
| TensorRT-LLM XQA (copy in FlashInfer) | `refs/flashinfer/csrc/xqa/` | `mma.sync` m16n8k16 (`mma.cuh:36`) in `mha.cu`, with an sm_86/89/120 config (`mha.cu:114-117`); the Hopper version is `mha_sm90.cu` | group rows in the tile (`HEAD_GRP_SIZE`, `defines.h:31-33`) | multi-block mode (`defines.h:100-103`) | int8 or fp8 with one scale (`defines.h:86-88`); no per-block q8_0 |

XQA also has a speculative-decoding mode (`SPEC_DEC`) with a "SWAP AB" option for a small
fixed number of query tokens (`defines.h:68-83`). Head size up to 256 (`defines.h:26-28`).
Its comment says sm_86/89/120 have 99 KB of shared memory per block (`mha.cu:106-108`).

### 2.3 Proposed decode kernel (one design for verify and for MTP drafts)

Shapes per card per layer: 2 KV heads, 12 query heads. Query rows per KV head:
6 heads x T tokens, with T = 4 for verify and T = 1 for a draft pass (up to 4 rows in
the first draft pass with catch-up rows). So 6 to 24 rows.

Work split:

- One CTA = (one KV head, one chunk of positions). All 6T query rows of that KV head
  are in the same CTA, so each K/V byte is read from DRAM once.
- Number of chunks per KV head: about (number of SMs x CTAs per SM) / 2. The card has
  36 SMs ([TechPowerUp review](https://www.techpowerup.com/review/palit-geforce-rtx-5060-ti-infinity-3-16-gb/)).
  Minimum chunk 256 positions, as FlashInfer does (`decode.cuh:724`).
  At 180k with 1 CTA per SM: 18 chunks per head, about 10k positions (10 MB f16) each.
- At 1k depth the kernel is latency-bound (2 MB of KV per layer per card is 5 us at
  full bandwidth). 19 attention calls per step (16 + 3 MTP) cost about 0.1-0.2 ms
  (estimate). CUDA graphs or a persistent kernel remove most launch cost.

Inner loop per tile of 32-64 positions:

1. Load the K and V tiles with `cp.async` (or TMA if the sm_120 features report
   confirms it) into shared memory, 2-3 stages. Tile of 32 positions x 256 x 2 B =
   16 KB for K and 16 KB for V. 3 stages = 96 KB, inside the 99 KB limit.
   Bytes in flight needed: 400 GB/s x about 1 us = 400 KB per card, about 11 KB per SM
   (estimate). One stage already covers it.
2. S = Q K^T with `mma.sync.m16n8k16` f16 x f16 -> f32. Two layouts work:
   - Q as the A operand (M = rows, padded to 16 or 32), K as B (N = 8 positions).
   - "Swap AB" as in XQA: K as A (M = 16 positions), Q as B (N = 8 rows). 24 rows
     fill exactly 3 N tiles. 6 rows use 1 N tile (25% padding, compute only).
   Either way there is no padding of heads to a power of 2.
3. Online softmax per row in f32 (running max m, running sum l). Scale 1/sqrt(256) =
   1/16 (`llama-rig2\src\models\qwen35.cpp:331`). Causal rule for the verify tokens:
   query token i at position p+i sees positions <= p+i. Only the last tile needs it.
4. O += P V with `mma.sync`, P converted to f16, f32 accumulators. O for 24 rows x 256
   in f32 = 24 KB per CTA, spread over the warps.

Combine and epilogue:

- Each CTA writes partial O (f32), m and l. At 180k: 18 chunks x 24 rows x 256 x 4 B =
  442 KB per head. Reading it back is about 1 us.
- The last CTA of each KV head (atomic counter) merges the partials, divides by l, and
  multiplies by sigmoid(gate). The gate is per query head and lives on the same card
  (`qwen35.cpp:304-308` and `338-341`). This removes a separate combine kernel and a
  separate sigmoid-multiply kernel. The output goes straight into the input format of
  the `attn_output` GEMV.

Before the kernel (in the QKV projection epilogue): q_norm and k_norm (RMS over 256,
eps 1e-6), partial RoPE on 64 of 256 dims (IMROPE, `llama-rig2\src\llama-model.cpp:3208-3212`;
sections [11,11,10,0], `freq_base` 1e7), then write K and V (quantized if q8_0) into the
cache. K is stored after RoPE, so the attention kernel does no RoPE.

### 2.4 Splitting heads across 2 cards

Recommended: **2 KV heads (12 query heads) per card**, as llama.cpp does.

- `attn_q`, `attn_k`, `attn_v` are split by output column (by head). Each card computes
  Q, gate, K, V only for its heads.
- Attention runs locally. No data crosses PCIe inside attention.
- `attn_output` [6144 -> 5120] is split by input row. Each card produces a partial sum
  of the 5120 outputs. One all-reduce per attention block, the same as for an FFN.
- KV bytes per card are equal (2 heads each).

Rejected alternative: split the sequence (each card holds half of the positions for
all 4 heads). It needs the full Q on both cards and an exchange of partial (O, m, l)
per layer: 4 tokens x 24 heads x 256 x 4 B = 98 KB plus the gate. That is more than
the 40 KB all-reduce payload (5120 x 4 tokens x 2 B) and adds a second exchange.

### 2.5 Tensor cores or CUDA cores?

Compute per card per step at 150k (verify + 3 drafts): 48 rows x 256 x 4 FLOP x 150k x 16
+ 12 rows x 256 x 4 x 150k x 3 = 1.24e11 FLOP (estimate).

- Tensor cores, f16 in, f32 accumulate: about 47.4 TFLOPS per card (section 4.2).
  2.6 ms at peak, about 4-5 ms realistic. Below the 14.6 ms (f16) and 7.75 ms (q8_0)
  memory floors.
- CUDA cores, f32 FMA: 4608 x 2 x 2.572 GHz = 23.7 TFLOPS peak. 5.2 ms at peak, about
  10 ms realistic. This is above the q8_0 floor.

So with q8_0 KV, the verify pass must use tensor cores to stay memory-bound. With f16
it is borderline on CUDA cores. Tensor cores for both.

### 2.6 MTP attention

Same kernel with T = 1 (6 rows per KV head). Its KV is 4 KiB per token (f16).

- If the MTP block is head-split like the target layers: 2 KiB per token per card,
  read 3 times per step. 2.30 ms per card at 150k (f16), 1.22 ms (q8_0).
- If the whole MTP block runs on one card (to avoid all-reduces in the draft passes,
  a decision for the split report): 4 KiB per token on that card. 4.6 ms at 150k (f16)
  on that card, and 704 MiB of KV at 180k instead of 352 per card.

The draft KV does not affect the output distribution. With probabilistic acceptance the
target distribution is kept for any draft distribution
([Leviathan et al. 2023](https://arxiv.org/abs/2211.17192),
[Chen et al. 2023](https://arxiv.org/abs/2302.01318)); LEEME says the same for
`--spec-draft-sampling probabilistic` (`qwen38_27\LEEME.md:60`). A cheaper draft KV
(q8_0, or a limited window of recent positions) only changes the acceptance rate. The
user's rule "never q4" was written for quality. Applying a cheaper format only to the
draft KV is a user decision. Expected gain: q8_0 draft KV saves 1.1 ms per step at 150k.

---

## 3. q8_0 KV

### 3.1 Speed

From the table in 1.3, per step and per card at 400 GB/s:

| Depth | f16 floor | q8_0 floor | Saved | Share of today's step |
|---|---|---|---|---|
| 30k | 2.92 ms | 1.55 ms | 1.37 ms | 3% of 41.0 |
| 100k | 9.73 ms | 5.17 ms | 4.56 ms | 9% of 48.3 |
| 150k | 14.59 ms | 7.75 ms | 6.84 ms | 13% of 53.9 |
| 180224 | 17.53 ms | 9.31 ms | 8.22 ms | n/a |

If the new engine removes most of the fixed overhead, these savings become a larger
share of the step. Example (estimate): a 25 ms step at 1k would be about 40 ms at 150k
in f16 and about 33 ms in q8_0.

Dequant cost: K and V must be converted to f16 before `mma.sync` (scale folded in: one
HFMA2 per 2 values). Per card per step at 150k: 2.46e9 values x about 1-1.5 instructions
/ about 12e12 simple ops per second = about 0.3 ms, overlapped with the loads (estimate).

Older evidence on this rig: with `-sm layer`, going from q8_0 to q4_0 KV gave only 6% at
100k (`qwen38_27\LEEME.md:695`). That matches a few ms saved on a step of about 100 ms.

### 3.2 VRAM

| Context | f16 per card | q8_0 per card |
|---|---|---|
| 180224 | 5,984 MiB | 3,179 MiB |
| 196608 | 6,528 MiB | 3,468 MiB |
| 229376 | 7,616 MiB | 4,046 MiB |
| 262144 | 8,704 MiB | 4,624 MiB |

(Target + MTP KV, 2 heads per card.) q8_0 saves 2.8 GiB per card at 180224.

### 3.3 Accuracy evidence (published)

| Source | Model, setting | Metric | f16 KV | q8_0 K + q8_0 V |
|---|---|---|---|---|
| [llama.cpp PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412) | model not stated in the fetched text; rows are weights/K/V | PPL / mean KLD | f16/f16/f16: 6.232196 / 0.000189 | f16/q8_0/q8_0: 6.234369 / 0.000980 |
| [Discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470) | Qwen2.5-7B, 512 ctx, wikitext-2 | mean KLD, same top token | n/a | 0.001782, 98.015% |
| same | Qwen3.6-27B (jkrauss82, May 2026) | mean KLD vs reference | bf16: 0.000375 | 0.002328 |
| same | **Qwen3.8-27B UD-Q6_K, 65K ctx** (sanmai, Sep 2026) | 99% KLD vs f16/f16 | f16/f16: 0.000037 | 0.006425 |

In the Qwen3.8-27B matrix, K in bf16 (V f16) gives 0.005940 and K in q8_0 (V f16) gives
0.006311. So q8_0 K adds about as much error as storing K in bf16. The PR author wrote:
"There seems to be no significant quality loss from using q8_0 instead of FP16 for the
KV cache", and noted that K is more sensitive than V. Both pages also show that q4 V
roughly doubles the KLD, which supports the user's "never q4" rule.

Not measured anywhere: q8_0 KV on this exact IQ3_S file at 150k+ depth. Phase 0 or the
correctness milestone should run llama.cpp perplexity / KLD with `-ctk q8_0 -ctv q8_0`
in `-sm layer` mode (which works) against f16 on the same text.

### 3.4 Why llama.cpp `-sm tensor` fails with q8_0

LEEME documents only the symptom: the server loads, answers `/health`, then card 1
stays at 7 W and 0% use, and the request never ends. Tested at 262144 and 212992
(`qwen38_27\LEEME.md:145-148` and `522-527`). An old fix (issue #21788, commit
`c11c362`) targets code that no longer exists (`LEEME.md:699`). The root cause is not
documented. A new engine owns its KV layout and kernels, so it does not inherit it.

### 3.5 Layout for the new engine

- Keep the q8_0 values bit-exact, but store them GPU-friendly (lossless repack): per
  (layer, KV head), positions contiguous; int8 values in one array (256 B per head per
  token, 16-B aligned for `cp.async`/`ldmatrix`) and f16 scales in another (8 per head
  per token). The q8_0 block of 34 B is not 16-B aligned.
- Quantize K after k_norm and RoPE, in the QKV epilogue, with the same rounding as
  ggml's `quantize_row_q8_0` so logits can match llama.cpp in q8_0 mode.
- Logit-match tests: compare f16 against llama.cpp f16 (`-sm tensor`), and q8_0 against
  llama.cpp q8_0 (`-sm layer`).

---

## 4. Prefill attention

### 4.1 FLOPs

Per token at depth d (core attention only; projections are counted with the weights):

    FLOP(d) = layers x 4 x n_q_heads x head_dim x d
            = 16 x 4 x 24 x 256 x d = 393,216 x d

(2 for Q K^T and 2 for P V, multiply-add = 2 FLOP. Causality is already in "d": a
token only sees the d earlier positions.) If the MTP layer also fills its KV during
prefill, as llama.cpp does, use 17 layers: 417,792 x d.

Whole prompt from 0 to N: 393,216 x N^2 / 2.

Weight matmuls per prompt token: 2 x 24.35e9 = 48.7 GFLOP (24.35e9 = all 2D weights
except `token_embd`, `output` and MTP, counted from `research/_gguf-model-tensors.tsv`).
Attention FLOPs per token equal this at d = 48.7e9 / 393,216 = **124k**. Deeper than
that, attention is the larger compute cost per token.

### 4.2 Tensor-core rate of this card (estimate)

- NVIDIA's RTX Blackwell whitepaper tables, as quoted by
  [Guru3D](https://www.guru3d.com/story/nvidia-discloses-blackwell-architecture-whitepaper-detailed-look-at-geforce-rtx-5070-ti-and-5070/):
  RTX 5070, 48 SMs at 2512 MHz: 61.7 dense TFLOPS FP16 with FP32 accumulate. That is
  512 FLOP per clock per SM.
- RTX 5090: 209.5 dense with FP32 accumulate and 419 with FP16 accumulate
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)).
  Same 512 FLOP per clock per SM with FP32 accumulate.
- RTX 5060 Ti: 36 SMs, boost 2572 MHz
  ([NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/),
  [TechPowerUp](https://www.techpowerup.com/review/palit-geforce-rtx-5060-ti-infinity-3-16-gb/)).
  NVIDIA does not publish its FP16 tensor rate. Estimate: 36 x 512 x 2.572e9 =
  **47.4 TFLOPS per card**, 94.8 for both. At the measured 2713-2782 MHz under load
  (`LEEME.md:434`) it is about 50.
- Cross-check: the published 189.6 dense INT8 TOPS for this card is 36 x 2048 x 2.572e9,
  consistent with the same per-SM rates.
- Whether FP16 accumulate runs at 2x on GB206 is unclear (sources disagree for the
  5070). llama.cpp uses FP32 accumulation for attention (`llama-graph.cpp:2675`). Use
  FP32 accumulation.

### 4.3 Times

Per token at depth d, both cards (12 query heads each), at 94.8 TFLOPS:

| Depth | GFLOP per token | ms per token at peak | at 55% | ceiling from attention alone |
|---|---|---|---|---|
| 30k | 11.8 | 0.124 | 0.23 | 8,000 t/s |
| 100k | 39.3 | 0.415 | 0.75 | 2,400 t/s |
| 150k | 59.0 | 0.622 | 1.13 | 1,600 t/s |
| 180224 | 70.9 | 0.747 | 1.36 | 1,340 t/s |

Whole prompt from 0: 30k: 1.9 s at peak (3.4 s at 55%). 100k: 20.7 s (37.7 s).
150k: 46.7 s (84.8 s). 180224: 67.4 s (122.5 s).

The bench (`qwen38_27\mide-tps.py:32-43`) sends prompts with `cache_prompt`, so each depth
only processes the new part. From `resultados-tps.csv` rows `V-M-clang` and
`Z-M-clang-again` (approximate ranges):

| Bench row | New tokens | Approx. range | Measured time | Attention at peak | Share |
|---|---|---|---|---|---|
| 30k | 29,893 | 0-29.9k | 44.9 s | 1.9 s | 4% |
| 100k | 70,986 | 28.5k-99.4k | 132.3 s | 18.8 s | 14% |
| 150k | 51,033 | 99k-150k | 119.2 s | 26.3 s | 22% |

A line fit through these three rows gives 1.385 ms per token + 0.00762 ms per token per
1000 depth. If all of the slope is attention, llama.cpp reaches about 51.6 TFLOPS on 2
cards, **about 54% of the estimated peak**. Part of the slope is the host mask upload
(1024 x n_kv x 2 B per ubatch, about 7% of the slope by estimate). With the 25% head
padding (section 2.1), the kernel runs at about 73% of peak on the work it actually
issues. A kernel without the padding and without the mask could be about 1.3x faster on
prefill attention (estimate).

### 4.4 Kernel options on sm_120

- **FlashAttention-2 style with `mma.sync`** is the only tensor-core path. FA3 is
  "optimized for Hopper GPUs (e.g. H100)" and FA4 targets Hopper and datacenter
  Blackwell ([flash-attention README](https://github.com/Dao-AILab/flash-attention)).
  FA3 uses `wgmma`, which sm_120 does not have. vLLM's FlashInfer backend says
  "Architectures with only fa2 (e.g. SM89, SM120)" and "SM12x prefill is fa2-only"
  (`refs/vllm/vllm/v1/attention/backends/flashinfer.py:946-966`).
- **FlashInfer** picks FA3 only on sm_90a, otherwise FA2
  (`refs/flashinfer/flashinfer/utils.py:574-629`). On sm_120 it can use TRT-LLM's
  `fmha_v2` HMMA kernels ("SM12x supports Ampere-compatible HMMA tensor core
  instructions", `utils.py:552-571`), but only for MHA (`num_qo_heads == num_kv_heads`,
  `flashinfer/prefill.py:4770-4806`). This model is GQA, so FlashInfer would use FA2.
  FA2 packs query rows as tokens x group size (`scheduler.cuh:557-562`) and uses a Q
  tile of at most 64 rows for head dim 256 (`include/flashinfer/utils.cuh:411-441`).
  The MMA instruction is `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
  (`include/flashinfer/mma.cuh:318-333`).
- **llama.cpp** uses its own `mma.sync` kernel on sm_120 for prefill too
  (`fattn.cu:638-664`), 64 columns per tile (8 tokens x 8 head slots), 128 threads
  (`fattn-mma-f16.cuh:72`). ik_llama.cpp also converts quantized KV to f16 first
  (`refs/ik_llama.cpp/ggml/src/ggml-cuda/fattn-new-mma.cu:1903`).

Design point for head dim 256: per KV position, a tile of M query rows does 1024 x M FLOP
on 1024 B of f16 K+V. Intensity = M FLOP per byte. The card's balance point is
47.4e12 / 400e9 = 118 FLOP/B. A 64-row tile (64 FLOP/B) relies on L2 hits from other
CTAs that read the same KV at the same time (32 MB L2). Better: put all 6 query heads of
one KV head in the same CTA, e.g. 6 heads x 16 tokens = 96 rows (6 warps of 16 rows; the
O accumulator is 16 x 256 f32 = 128 registers per thread). With q8_0 K/V the bytes drop
to 544 per position, so the same tile reaches 181 FLOP/B. Recommendation: own FA2 kernel
(or adapt llama.cpp's MIT code) with GQA packing by 6, no power-of-2 padding, q8_0 tile
dequant in shared memory, causal tile skipping, and no host mask. FP8 attention is not
proposed: it changes numerics beyond the f16/q8_0 rule.

---

## 5. Long context beyond 180224

### 5.1 VRAM budget per card (estimate)

Card total: 16,311 MiB (32,622 for both, `LEEME.md:82`).

| Item | MiB per card | Basis |
|---|---|---|
| Weights | 5,580 | (11,548 - 388 `token_embd` kept on host) / 2 |
| CUDA context, runtime, workspaces | 300-600 | estimate |
| GDN state + rollback copies for MTP | 75-300 | estimate (state ~144 MiB per sequence in f32, half per card) |
| Activations, prefill ubatch 1024 | 300-600 | estimate (no mask buffers, no f16 KV copies) |
| Vision encoder (one card) | ~1,518 | 888 MiB weights + ~630 MiB for a 4K image (free memory fell from 1,185 to 555 MiB, `LEEME.md:139, 224-229`) |
| Windows desktop (card 0) | 1,300-1,900 | `LEEME.md:244-247`. **Windows only.** |

For reference, llama.cpp uses 8,509 MiB (card 0) and 9,258 MiB (card 1) for everything
except KV at `-c 180224`, including weights, desktop, vision and its buffers
(14,493 / 15,242 measured, `LEEME.md:82`, minus 5,984 MiB KV).

### 5.2 Largest context

KV per card per token: 34 KiB (f16) or 18.06 KiB (q8_0). Vision on one card, desktop on
the other (Windows). The fuller card sets the limit.

| Overhead case | OS | Free for KV (fuller card) | f16 max context | q8_0 max context |
|---|---|---|---|---|
| lean (675 MiB) | Linux | 8,538 MiB | 257k | 484k -> 262,144 |
| middle (950 MiB) | Linux | 8,263 MiB | 249k | 468k -> 262,144 |
| heavy (1,500 MiB) | Linux | 7,713 MiB | 232k | 437k -> 262,144 |
| lean | Windows | 8,156 MiB | 246k | 462k -> 262,144 |
| middle | Windows | 7,881 MiB | 237k | 447k -> 262,144 |
| heavy | Windows | 7,331 MiB | 221k | 416k -> 262,144 |

- **q8_0 reaches 262,144 on both OSes** with 3-4 GiB per card left for prompt-cache
  checkpoints in VRAM or larger ubatches.
- **f16 at 262,144 needs 8,704 MiB per card.** On Linux it fits only on the card
  without vision. One way to balance: put the whole MTP block and its KV on the card
  without vision (that card: 36 KiB per token; the vision card: 32 KiB per token). Middle
  case on Linux: 273k and 270k tokens, so 262,144 fits. On Windows f16 stays below
  262,144 in all cases.
- Each 100 MiB of overhead costs about 3,000 tokens in f16 and 5,700 in q8_0.

### 5.3 Cost of the extra depth

- Decode at 262,144: KV floor 25.5 ms per step in f16, 13.6 ms in q8_0 (table 1.3).
- Prefill of a full 262,144-token prompt: 393,216 x 262,144^2 / 2 = 1.35e16 FLOP for
  attention, 142 s at peak, about 200 s at 70% (estimate). Agent sessions depend on the
  prompt cache at this size.
- The GGUF has no RoPE scaling keys (`research/_gguf-model-dump.txt:15-57`), and
  `context_length` is 262,144. Beyond that the model runs outside its trained range.
  Not recommended without a quality test.

---

## Open points for Phase 0 (measurements, not done here)

1. Nsight Systems on the current build at 1k and 150k: time of the FA kernels (verify
   and draft) and of the mask H2D copies per step. Compare with tables 1.3 and 1.4.
2. A pure read kernel over 7 GB per card for the real read bandwidth (the 388-404 GB/s
   figures are copy bandwidth).
3. Perplexity and KLD of q8_0 vs f16 KV on this IQ3_S file at long context, with
   llama.cpp `-sm layer`.
4. Whether TMA and 2x-rate FP16 accumulation exist on sm_120 (for the kernel-features
   report).

---

## Sources

Local (read only):

- `qwen27-engine\research\_brief.md`, `research\_gguf-model-dump.txt:15-57`, `research\_gguf-model-tensors.tsv`
- `qwen38_27\LEEME.md:56, 60, 82, 130-148, 224-247, 351-357, 434, 522-527, 695, 699`
- `qwen38_27\mide-tps.py:32-56`, `resultados-tps.csv` (rows `V-M-clang`, `Z-M-clang-again`)
- `llama-rig2\ggml\src\ggml-cuda\fattn.cu:142-150, 154-191, 193-286, 541-720, 722-756`
- `llama-rig2\ggml\src\ggml-cuda\fattn-mma-f16.cuh:38-88, 234-262, 1800-1806, 1887-2002, 2020-2117`
- `llama-rig2\ggml\src\ggml-cuda\fattn-common.cuh:9, 53-85, 981-1306`
- `llama-rig2\ggml\src\ggml-cuda\fattn-vec.cuh:104-111, 531-572`
- `llama-rig2\ggml\src\ggml-backend-meta.cpp:911-929`
- `llama-rig2\src\llama-graph.cpp:29-46, 2624-2675, 2825`
- `llama-rig2\src\llama-kv-cache.cpp:1756-1759`
- `llama-rig2\src\models\qwen35.cpp:269-348, 501-620`
- `llama-rig2\src\llama-model.cpp:3208-3212`
- `llama-rig2\common\speculative.cpp:1390-1443`
- `refs\flashinfer\include\flashinfer\attention\decode.cuh:216-330, 690-745`
- `refs\flashinfer\include\flashinfer\attention\cascade.cuh:220-260`
- `refs\flashinfer\include\flashinfer\attention\scheduler.cuh:557-623`
- `refs\flashinfer\include\flashinfer\utils.cuh:411-441`, `refs\flashinfer\include\flashinfer\mma.cuh:318-353`
- `refs\flashinfer\flashinfer\utils.py:552-629`, `refs\flashinfer\flashinfer\prefill.py:4770-4806`, `refs\flashinfer\flashinfer\decode.py:1063-1065`
- `refs\flashinfer\csrc\xqa\defines.h:20-103`, `refs\flashinfer\csrc\xqa\mha.cu:82-145`, `refs\flashinfer\csrc\xqa\mma.cuh:36-52`
- `refs\vllm\vllm\v1\attention\backends\flashinfer.py:946-966`
- `refs\ik_llama.cpp\ggml\src\ggml-cuda\fattn-new-mma.cu:1853-1903`

Web:

- [llama.cpp PR #7412, quantized KV cache PPL/KLD](https://github.com/ggml-org/llama.cpp/pull/7412)
- [llama.cpp discussion #23470, KV cache KLD on Qwen models](https://github.com/ggml-org/llama.cpp/discussions/23470)
- [NVIDIA RTX 5060 family specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)
- [TechPowerUp RTX 5060 Ti review: 36 SM, 32 MB L2, 448 GB/s](https://www.techpowerup.com/review/palit-geforce-rtx-5060-ti-infinity-3-16-gb/)
- [Guru3D: Blackwell whitepaper figures for RTX 5070 / 5070 Ti](https://www.guru3d.com/story/nvidia-discloses-blackwell-architecture-whitepaper-detailed-look-at-geforce-rtx-5070-ti-and-5070/)
- [NVIDIA forum: RTX 5090 peak BF16/FP16 tensor TFLOPS](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)
- [igor'sLAB: RTX 5060 Ti, 144 tensor cores](https://www.igorslab.de/en/nvidia-geforce-rtx-5060-ti-officially-unveiled-blackwell-architecture-for-the-mid-range/)
- [Dao-AILab flash-attention README (FA2/FA3/FA4 GPU support)](https://github.com/Dao-AILab/flash-attention)
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al., Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
