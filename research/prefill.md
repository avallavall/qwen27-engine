# Prefill (prompt reading) on 2x RTX 5060 Ti, Qwen3.8-27B GGUF

Research only. No GPU was used. Every number marked "estimate" is arithmetic from the sources below, with the steps shown.

## Summary

1. **FLOPs per prompt token.** The linear layers cost 49.6 GFLOP (48.7 for the 64 blocks, 0.85 for the MTP block). The chunked GDN scan costs 0.42 GFLOP. Attention costs 0.418 GFLOP for every 1,000 tokens of depth (16 layers plus the MTP layer). Over the three benchmark ranges this gives 56 / 77 / 102 GFLOP per token.
2. **Peak per card** (36 SMs, 2,572 MHz boost, dense): FP16/BF16 with FP32 accumulate 47.4 TFLOPS. FP16 with FP16 accumulate 94.8. FP8 94.8 (FP32 accumulate) or 189.6 (FP16 accumulate). INT8 189.6 TOPS. The compute floor for 2 cards is about **5,400 / 2,900 / 1,800 t/s** at 30k / 100k / 150k with INT8 linear layers.
3. **Link.** Each prompt token needs 130 all-reduces of 5,120 BF16 values. That is 1.35 MB per card in each direction. llama.cpp sends the upload and the download one after the other, and never overlaps them with compute. That costs **0.77 ms per token**, so the link alone caps prefill at **~1,300 t/s**. Sending both directions at once halves this (2,600 t/s cap). Two micro-batches in flight can hide the rest behind compute.
4. **GEMM.** A port of llama.cpp MMQ (INT8 MMA, q8_1 activations) is the only option that keeps llama.cpp numerics. It is also the fastest option with exact GGUF weights: it reaches about 40-55 effective TOPS per card. That is already above the 47.4 TFLOPS peak of any FP16/BF16 GEMM with FP32 accumulate. CUTLASS sm_120 builders only accept FP8/FP6/FP4, so CUTLASS would need re-quantized weights and would change the model.
5. **Where the time goes today (estimate).** Link 51% / 41% / 33%. Attention 6% / 22% / 35%. KQ-mask upload 1% / 4% / 6%. Linear layers, GDN and small kernels 41% / 33% / 26%.
6. **Target for the new engine: 1,300 / 880 / 640 t/s** (1.9x / 1.6x / 1.5x). It assumes llama.cpp-speed kernels, a hidden link, and no KQ-mask upload. With better MMQ and attention kernels the stretch target is 1,650 / 1,120 / 810 t/s. Without compute/link overlap it drops to 910 / 705 / 555 t/s.

---

## 0. What the baseline numbers measure

The benchmark (`qwen38_27\mide-tps.py`) sends prompts that are prefixes of the same corpus, with `cache_prompt: true`. The server reuses the cached prefix. So "100k" and "150k" measure only the new part of the prompt.

| Label | `prompt_tok` processed | Range of positions (estimate) | Mean position | `prompt_ts` (run `V-M-clang`) | ms per token |
|---|---|---|---|---|---|
| 30k | 29,893 | 0 to 29.9k | 15.0k | 665.3 | 1.503 |
| 100k | 70,986 | 28.9k to 99.9k | 64.4k | 536.4 | 1.864 |
| 150k | 51,033 | 98.9k to 149.9k | 124.4k | 428.0 | 2.336 |

The brief quotes 667 / 540 / 439. The CSV rows of the current build give 665 / 536 / 428. This report uses the CSV rows. The range start is total prompt length minus `prompt_tok` (estimate: total = requested size minus ~100 tokens).

## 1. FLOPs per prompt token

### Linear layers

FLOPs per token = 2 x (number of weights in all 2D matmul tensors that every token passes through). Summed from `research/_gguf-model-tensors.tsv` with a small script (venv Python):

| Part | Weights | GFLOP per token |
|---|---|---|
| 64 blocks (all `blk.0`-`blk.63` matrices, `ssm_conv1d` excluded) | 24.35 B | 48.70 |
| MTP block `blk.64` (`eh_proj`, attention, FFN) | 0.425 B | 0.85 |
| `output.weight` | 1.27 B | 2.54, **only for tokens that need logits** (the last prompt token) |
| **Total per prompt token** | | **49.55** |

By role (64 blocks): `ffn_up`, `ffn_gate`, `ffn_down` 5.70 B each. GDN `attn_qkv` 2.52 B, `attn_gate` 1.51 B, `ssm_out` 1.51 B. Attention `attn_q` 1.01 B, `attn_output` 0.50 B, `attn_k` and `attn_v` 0.084 B each. `ssm_alpha` and `ssm_beta` (BF16) 0.012 B each.

The MTP block runs over every prompt token. The MTP hook decodes each target batch again in the draft context, with the target hidden states shifted by one (`llama-rig2\common\speculative.cpp:1625-1728`). The target copies the hidden state of every token to the host for this (`llama-rig2\src\llama-context.cpp:2105-2122`). The draft batch asks for no logits (`speculative.cpp:1722`, last argument `false`). So the output head runs only for the last token.

### GDN chunked scan

Per V head, head dim d = 128 (dk = dv), chunk size C = 64, as in FLA `chunk_gated_delta_rule_fwd` (`refs\flash-linear-attention\fla\ops\gated_delta_rule\chunk.py:33-123`). Matmuls per chunk:

| Step | FLOPs per chunk |
|---|---|
| K Kᵀ, Q Kᵀ, W = A·K, U = A·V, (masked QKᵀ)·V_new | 5 x 2·C²·d |
| V_new = U - W·S, state update Kᵀ·V_new, output Q·S | 3 x 2·C·d² |

Per token per head: 10·C·d + 6·d² = 81,920 + 98,304 = 180k FLOP. Times 48 V heads times 48 GDN layers: **0.415 GFLOP per token**. The triangular solve and the gates add a few percent. For comparison, the token-by-token recurrence costs about 7·d² per token per head, so 0.26 GFLOP per token.

### Attention as a function of depth

Per attention layer, one token at position p: QKᵀ = 2·24·256·p, PV = 2·24·256·p. So 24,576·p FLOP per layer. 16 layers plus the MTP layer: **417,792·p FLOP per token** (0.418 GFLOP per 1k of depth).

| | 30k | 100k | 150k |
|---|---|---|---|
| Attention, one token at depth D (marginal) | 12.5 | 41.8 | 62.7 |
| Attention, mean over a prompt 0 to D | 6.3 | 20.9 | 31.3 |
| Attention, mean over the benchmark range (section 0) | 6.2 | 26.9 | 52.0 |
| **Total per token, benchmark range** (linear 49.55 + GDN 0.42 + attention) | **56.2** | **76.9** | **101.9** |
| Total per token, prompt 0 to D | 56.2 | 70.9 | 81.3 |
| Total per token, marginal at D | 62.5 | 91.7 | 112.6 |

All in GFLOP. Each card does half of everything except the mirrored MTP `eh_proj` (both cards compute it whole, +0.1 GFLOP per card).

Today's achieved rate (benchmark FLOPs x t/s / 2 cards): 18.7 / 20.6 / 21.8 TFLOPS per card. That rate includes the time the cards spend waiting on the link.

## 2. Tensor-core peak and compute floor

NVIDIA does not publish tensor rows for the RTX 5060 Ti (GB206). The RTX Blackwell whitepaper gives them for the RTX 5090 and RTX 5070. The per-SM rate is the same in both:

| Card | SMs | Boost MHz | FP16 FP32-acc | INT8 dense | Per SM per clock |
|---|---|---|---|---|---|
| RTX 5090 (whitepaper p. 46-47) | 170 | 2,407 | 209.5 | 838 | 512 FP16 / 2,048 INT8 |
| RTX 5070 (whitepaper p. 54-55, Table 6) | 48 | 2,512 | 61.7 | 246.9 | 512 / 2,048 |

RTX 5060 Ti: 36 SMs, 2.57 GHz boost, "759 AI TOPS" (NVIDIA product page). Scaling the per-SM rates by 36 SMs x 2.572 GHz:

| Precision (dense, per card) | Peak | Ratio from whitepaper |
|---|---|---|
| FP16 / BF16, FP32 accumulate | **47.4 TFLOPS** | 512 / SM / clk |
| FP16, FP16 accumulate | 94.8 TFLOPS | 2x the FP32-accumulate rate on GeForce |
| FP8, FP32 accumulate | **94.8 TFLOPS** | same as FP16 with FP16 accumulate |
| FP8, FP16 accumulate | 189.6 TFLOPS | |
| INT8 | **189.6 TOPS** | 2,048 / SM / clk |
| FP4 | 379.3 dense, 758.5 sparse | matches NVIDIA's "759 AI TOPS". This confirms the scaling. |

Sparse rates are 2x and do not apply here. Under load the cards ran at 2,713 and 2,782 MHz (`LEEME.md:430-438`), so real peaks are 5-8% higher.

llama.cpp FlashAttention accumulates QKᵀ in FP32 and PV in FP16 (`fattn-mma-f16.cuh:1090-1127`: `T_C_KQ` is `float`, `T_C_VKQ` is `half2`). Half the attention FLOPs run at 47.4 and half at 94.8, so the effective attention peak is 2/(1/47.4 + 1/94.8) = **63.2 TFLOPS per card**.

**Compute floor, 2 cards, 100% of peak, benchmark ranges:**

| | 30k | 100k | 150k |
|---|---|---|---|
| Linear in INT8 (49.55 / 379.2 TOPS) | 0.131 ms | 0.131 | 0.131 |
| GDN at FP16/FP32-acc (0.415 / 94.8) | 0.004 | 0.004 | 0.004 |
| Attention at the llama.cpp mixed rate (/126.4) | 0.049 | 0.213 | 0.411 |
| **Floor with INT8 linear** | **5,420 t/s** | **2,875 t/s** | **1,830 t/s** |
| Floor if linear runs FP16/BF16 with FP32 accumulate (/94.8) | 1,735 t/s | 1,350 t/s | 1,065 t/s |

## 3. Cross-card traffic with tensor parallel

### What llama.cpp does today

- **Split.** Column split for `attn_q`, `attn_k`, `attn_v`, `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ffn_up`, `ffn_gate`. Row split for `attn_output`, `ssm_out`, `ffn_down`. Everything else is mirrored (`llama-rig2\src\llama-model.cpp:521-581, 611`). This is two all-reduces per block: after `attn_output`/`ssm_out` and after `ffn_down`.
- **Count per prompt token:** 64 blocks x 2 = 128, plus 2 in the MTP pass = **130**.
- **Payload:** BF16 on the wire by default (`allreduce.cu:452-455` and `788-797`, env `GGML_CUDA_AR_BF16_THRESHOLD`, default 1 byte). 5,120 x 2 B = 10,240 B per token per all-reduce.
- **Path for prefill:** above 1 MB the copy-engine path is used (`allreduce.cu:245-247, 810-812`). With `-ub 1024` one all-reduce is 10 MiB, in 2 MiB chunks (`allreduce.cu:248-254, 401-407`).
- **Order:** stage 1 (D2H of my partial) and stage 2 (H2D of the peer's partial) are queued on the same stream, `p->streams[i]` (`allreduce.cu:643-673` and `683-703`). So on each card the download starts only after the whole upload ends. The compute stream then waits for the H2D event before the add kernel (`allreduce.cu:711-712`). The AR stream waits for the compute stream before the D2H (`allreduce.cu:409-414`). The meta backend runs subgraph, all-reduce, subgraph, in strict order (`ggml-backend-meta.cpp:2735-2764`). **Nothing overlaps with compute.**

### Bytes and time (estimate)

| Item | Per token, per card | Per 1,024-token ubatch, per card |
|---|---|---|
| 130 all-reduces, BF16 | 1.331 MB up + 1.331 MB down | 1.363 GB up + 1.363 GB down |
| MTP hidden state, FP32 (one card up, each card down) | 20 KB + 20 KB | 21 MB + 21 MB |
| KQ masks, F16, target + MTP context (see section 5) | 4·p bytes down (p = depth) | 264 MB down at the 100k range |

Time at ~3.5 GB/s per direction on Gen 3 x4:

- Today: (1.352 + 1.352) MB / 3.5 GB/s = **0.77 ms per token**, all of it exposed. This is ~0.79 s per 1,024-token ubatch.
- Link-only ceiling today: **~1,300 t/s**.

**Cross-check with the Gen 4 measurement.** On 17 Sep, Gen 4 moved prefill at 100k from 417 to 503 t/s (`LEEME.md:422-425`). That is 2.398 to 1.988 ms per token, so 0.41 ms saved. Halving a 0.77 ms link term predicts 0.39 ms saved. The model fits. Caveat: those two numbers came from an older build and an older benchmark setup. The same build (b11026) gave 534 t/s at 100k on the 3 Oct bench, so the 17 Sep runs measured something slightly different.

### Options to reduce it

| Option | Wire per token per card per direction | Exposed link, ms per token | Link-only ceiling | Effect on numerics |
|---|---|---|---|---|
| Today: BF16, up then down on one stream, no overlap | 1.35 MB | 0.77 | ~1,300 t/s | reference |
| FP32 wire (`GGML_CUDA_AR_BF16_THRESHOLD=0`) | 2.70 MB | 1.54 | ~650 t/s | changes results vs today's llama.cpp |
| BF16, up and down at the same time (separate streams) | 1.35 MB | 0.39 | ~2,600 t/s | same |
| Same, plus two micro-batches in flight (overlap with compute) | 1.35 MB | ~0 while compute per token ≥ 0.39 ms | ~2,600 t/s | same |
| FP8 or INT8 wire with a per-row scale | 0.68 MB | 0.19 (both directions at once) | ~5,200 t/s | changes results; needs a KL-divergence test |
| Gen 4 x4 in BIOS | same bytes | half | 2x | same. A card dropped off the bus with Gen 4 on this rig (`LEEME.md:400-404`). |
| P2P instead of host staging | same bytes on each link | same | same | same. Host staging already crosses each x4 link once per direction. P2P only saves latency. |
| All-gather the 3,072-wide head output and run `attn_output`/`ssm_out` whole on both cards | -19% bytes | -19% | | +8% linear FLOPs per card and ~0.5 GB more VRAM per card. Only worth it if still link-bound after overlap. |
| Reduce-scatter + all-gather (sequence parallel) | same bytes on 2 GPUs | same | same | same |
| Layer (pipeline) split for prefill only | ~10 KB once per token | ~0.003 | | Needs every layer whole on one card, while decode needs half of every layer on each card. It also needs all 4 KV heads per layer on one card (TP keeps 2 per card). At 100k that is ~1.6 GB of KV to move or duplicate. There is no VRAM for this. |

**How the overlap works.** Split each ubatch into two halves, A and B. Run A one sub-layer ahead of B. While A's all-reduce is on the link, the card computes B's sub-layer, and the other way round. B's attention at layer L needs A's K/V at layer L, and B's GDN needs A's final GDN state at layer L. A is ahead, so both are ready. The copy engines are separate from the SMs, so the transfer and the compute run at the same time.

**To verify before relying on full duplex:** `cudaDeviceProp::asyncEngineCount` on both OSes. Concurrent D2H and H2D need 2 copy engines. If only 1 is exposed, the transfers can still go both ways at once with SM-driven stores and loads to mapped host memory (as in the small-tensor path, `allreduce.cu:79-108`), at the cost of a few SMs. WDDM scheduling of copy engines next to compute is a Windows-only unknown. On Ubuntu, NCCL without P2P also stages through host memory, so the link math is the same on both OSes.

## 4. GEMM kernels at 512-2,048 tokens

### Shapes per card

| Matrix | K | N per card | Types in this file |
|---|---|---|---|
| GDN `attn_qkv` | 5,120 | 5,120 | IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XS, IQ2_XXS, Q4_K, Q2_K |
| GDN `attn_gate` | 5,120 | 3,072 | mixed |
| GDN `ssm_out` (row split) | 3,072 | 5,120 | IQ4_XS, IQ3_S, IQ3_XXS, Q4_K |
| `attn_q` | 5,120 | 6,144 | mixed |
| `attn_k`, `attn_v` | 5,120 | 512 | mixed |
| `attn_output` (row split) | 3,072 | 5,120 | mixed |
| `ffn_gate`, `ffn_up` | 5,120 | 8,704 | mixed, one IQ1_M (`blk.13.ffn_gate`) |
| `ffn_down` (row split) | 8,704 | 5,120 | mixed |
| `ssm_alpha`, `ssm_beta` | 5,120 | 24 | BF16 |

With M = 512 to 2,048 tokens all of these are compute-bound. One weight byte is reused M times, and the cards read only 5.4 GB of weights per ubatch (13 ms at 400 GB/s, against ~1.5 s of compute).

### Options

| | llama.cpp MMQ (`mmq.cu`, `mmq-config-blackwell.cuh`) | Dequant to FP16/BF16 + cuBLAS | CUTLASS sm_120 |
|---|---|---|---|
| Math | Weights unpacked to exact INT8 values. Activations quantized to q8_1 per 32 values (`quantize.cu:458-520`, `d = amax/127`, `roundf`). `mma.sync.m16n8k32.s32.s8.s8.s32` (`mma.cuh:1026`). Scales applied in FP32 per 32-block. | llama.cpp's own fallback dequantizes weights to FP16 and calls cuBLAS with `CUBLAS_COMPUTE_16F`, so FP16 accumulate (`ggml-cuda.cu:1618-1625` and `1393-1396`). | The 3.x sm_120 builder accepts only F8F6F4 element types (`refs\cutlass\include\cutlass\gemm\collective\builders\sm120_mma_builder.inl:80, 115`). Block-scaled NVFP4/MXFP8 kernels exist (`examples\79_blackwell_geforce_gemm`). The mixed-input GEMM (INT4 x BF16) is sm_100 only (`examples\86_blackwell_mixed_dtype_gemm\86_blackwell_mixed_dtype.cu:67, 106`). No GGUF i-quant or k-quant support. |
| Blackwell tuning in llama-rig2 | The Blackwell file only adds MXFP4/NVFP4 cases. All types in this model fall back to the Ampere config (`mmq-config-blackwell.cuh:36`): 256 threads, tile 128 x up to 128, stream-k (`mmq-config-ampere.cuh:295-324`). | | |
| Peak it can use | INT8 189.6 TOPS | 47.4 (FP32 accumulate) or 94.8 (FP16 accumulate) | FP8 94.8 / FP4 379 |
| Measured or estimated | ~29% of INT8 peak on Q4_0: 4,196 t/s pp512 on Llama 2 7B, an RTX 5060 Ti in discussion #15013, so 13.0 GFLOP x 4,196 = 54.5 TOPS. Estimate for this model's i-quant mix: 41-50 TOPS per card (section 5). | On RTX 4090, Llama 8B at batch 512: Q8_0 MMQ 11,190 t/s vs cuBLAS 7,388. Q2_K 8,108 vs 7,772 (PR #8062). | Not measured. Needs weights converted to FP8/NVFP4. |
| Matches llama.cpp | **Yes**, if the same q8_1 rounding and per-block scaling are kept. Differences come only from the FP32 summation order (tile shape, stream-k). | No. It removes the 8-bit activation rounding, and FP16 accumulate over K up to 17,408 adds its own error. | No. It changes the weights. |

`IQ1_M` is not in llama.cpp's MMQ list (`mmq.cu:325-356`). So `blk.13.ffn_gate` (89 M weights, 0.4%) already goes through dequant + cuBLAS FP16 in llama.cpp.

### Which is exact enough

Bit-identical output to llama.cpp cannot be reached in practice. llama.cpp itself changes the summation order with the tile shape and with stream-k. It uses different kernels for 1-8 tokens (MMVQ) and for bigger batches (MMQ). The `-sm tensor` path also rounds the all-reduce partial sums to BF16. The usable gate is KL divergence. The new engine's KLD against a llama.cpp run should be no larger than the KLD between two llama.cpp runs that differ only in `-ub` or `-sm` (use `llama-perplexity --kl-divergence`).

Under that gate, an MMQ port is the safe choice. It keeps the same activation rounding, so its error profile is the same as llama.cpp's. A dequant to FP16 with FP32 accumulate is likely closer to the true dequantized model. Its KLD against llama.cpp would still be dominated by llama.cpp's own q8_1 rounding. FP16 activations can overflow (> 65,504) in `ffn_down` inputs. BF16 weights lose ~3 bits on IQ grid x scale products. Both need checking.

### Which reaches more of peak

INT8 MMQ reaches a low share of its own peak (about 20-30%). It still delivers more absolute throughput than any FP16/BF16 GEMM with FP32 accumulate can, even at 100% of that peak (47.4). The INT8 peak is 4x higher, so there is headroom. A tuned INT8 kernel at 32-35% of peak gives 60-66 TOPS.

Possible limiter in MMQ, to check with Nsight Compute: each m16n8k32 INT8 MMA produces 128 INT32 results that must be converted to FP32 and scaled before the next 32-block. The conversion and scale FMAs run on the regular cores and may cap the tensor-core use. This is a hypothesis, not a measurement. `ik_llama.cpp` has its own CUDA MMQ for all i-quants (`refs\ik_llama.cpp\README.md:154`). It is worth reading before writing a new kernel. It has not been measured here.

## 5. Where the prefill time goes today (estimate)

Method: fit ms per token = c0 + c1 x (attention GFLOP per token) over the three benchmark points, after removing an estimated KQ-mask upload. Set the link term from the bytes in section 3. Three points and two parameters leave one degree of freedom, so this is a rough split. The profiling session should replace it.

- **Link:** 0.772 ms per token (section 3).
- **KQ mask:** llama.cpp builds the F16 mask (n_kv x n_tokens) on the CPU in a host buffer, single-threaded (`llama-kv-cache.cpp:1756-1793`). The scheduler then uploads it. The target and the MTP context each have one. If both are uploaded to both cards (likely, not verified), that is 4·p bytes per token per card: 0.017 / 0.074 / 0.142 ms per token at the mean depths.
- **Attention:** fitted slope 0.0155 ms per GFLOP, so **32 TFLOPS per card**. That is ~51% of the 63.2 TFLOPS mixed peak. Without the mask correction the slope gives 27 TFLOPS per card.
- **Rest:** intercept 1.384 ms minus link 0.772 = **0.61 ms per token**. It holds the MMQ matmuls, the GDN recurrence, norms, q8_1 quantization, all-reduce add kernels and the MTP pass. If all of it were MMQ, MMQ would run at 24.8 GFLOP / 0.61 ms = 41 TOPS per card (21% of INT8 peak).

| ms per token | 30k | 100k | 150k |
|---|---|---|---|
| Link (all-reduces, serialized, BF16) | 0.77 (51%) | 0.77 (41%) | 0.77 (33%) |
| KQ-mask upload | 0.02 (1%) | 0.07 (4%) | 0.14 (6%) |
| Attention compute | 0.10 (6%) | 0.42 (22%) | 0.81 (35%) |
| Linear + GDN + small kernels | 0.61 (41%) | 0.61 (33%) | 0.61 (26%) |
| Model sum / measured | 1.50 / 1.50 | 1.87 / 1.86 | 2.33 / 2.34 |

Inside "linear + GDN + small kernels" (estimate):

- **GDN.** With `-sm tensor` each card has 8 of 16 K heads. So the chunked GDN kernel from PR #29353 does not run (`LEEME.md:580, 693`). The token-by-token kernel runs instead. One warp owns one state column and loops over the 1,024 tokens of the ubatch (`gated_delta_net.cu:69`, grid at `:189-190`). That is 3,072 warps per card, about 2 waves on 36 SMs. At an estimated ~700 cycles per token step (L2 load, two warp reductions), this costs ~0.5 ms per layer per ubatch. 48 layers give ~25 ms per ubatch, so **~0.03 ms per token** (range 0.02-0.06).
- **Small kernels.** q8_1 quantization, norms, BF16 conversion and add for the all-reduces move about 15 MB per token per card, so ~0.04 ms at 400 GB/s.
- **MMQ.** That leaves ~0.5-0.55 ms for MMQ, so **45-50 TOPS per card**. This matches the Q4_0 reference point above (54.5 TOPS).

Cheap checks for the profiling session, using llama.cpp as it is:

- `GGML_CUDA_AR_BF16_THRESHOLD=0` doubles the wire bytes. If this model is right, prefill at 30k should drop from ~665 to ~440 t/s (+0.77 ms per token).
- Nsight Systems should show, per card, D2H then H2D on one stream with idle SMs during both.

## 6. Target for the new engine

Assumptions, all on the same benchmark ranges as the baseline:

- Tensor parallel stays, with the same split as decode. There is no VRAM for a second weight layout.
- All-reduce in BF16, up and down at the same time, two micro-batches in flight. The link costs 0.386 ms per token and hides behind compute when compute per token is longer. 10% is added for pipeline bubbles.
- The causal mask is generated inside the attention kernel. No mask upload.
- A chunked GDN kernel per head. Heads are independent, so the TP split does not block it (~0.01 ms per token).

| | 30k | 100k | 150k |
|---|---|---|---|
| Baseline today | 665 | 536 | 428 |
| **Target**: llama.cpp kernel speed (MMQ ~41 TOPS, attention 32 TFLOPS per card), link hidden, no mask | **1,280** | **880** | **640** |
| Stretch: MMQ 60 TOPS per card (32% of INT8 peak), attention 40 TFLOPS per card (63% of mixed peak), 0.06 ms of small kernels | 1,650 | 1,120 | 810 |
| Minimum: up and down at the same time, no overlap, no mask | 910 | 705 | 555 |
| Compute floor (section 2) | 5,420 | 2,875 | 1,830 |

The arithmetic for the target row at 30k: compute = 24.8 GFLOP / 40.5 TOPS + 6.25 GFLOP / 64.5 TFLOPS = 0.611 + 0.097 = 0.708 ms. That is longer than the 0.386 ms link, so the link hides. 0.708 x 1.10 = 0.78 ms per token, so 1,280 t/s. At 100k and 150k only the attention term changes (0.417 and 0.806 ms).

At 30k the link (0.386 ms) is 70% of the stretch compute (0.55 ms), so the overlap has to work well there. At 100k and 150k attention dominates. Then the attention kernel matters most, and the link barely does.

**Outside reference.** The same two cards with vLLM, tensor parallel 2, an NVFP4 W4A4 checkpoint of Qwen3.8-27B and FP8 KV cache, read a 131k prompt from empty at 1,432 t/s (club-5060ti issue #7). The mean depth (65k) matches this report's 100k range. That setup uses FP4 GEMMs (379 TOPS peak) and different weights, so its numerics do not match llama.cpp. Its PCIe link width and generation are not stated.

**Unknowns that move the target:**

- Number of copy engines per card (full duplex).
- WDDM effect on copy/compute overlap (Windows only).
- Real Gen 3 x4 bandwidth with pinned memory on this board (3.5 GB/s assumed).
- Real MMQ and FA efficiency per kernel (profiling session).

---

## Sources

Local files (read only):

- `qwen27-engine\research\_brief.md` (rig, model, baseline)
- `qwen27-engine\research\_gguf-model-tensors.tsv` (shapes and types; sums by script)
- `qwen38_27\resultados-tps.csv` rows `V-M-clang`, `Z-M-clang-again`, `A-official-b11026` (`prompt_tok`, `prompt_ts`)
- `qwen38_27\mide-tps.py:23-42` (prefix prompts, `cache_prompt`)
- `qwen38_27\LEEME.md:68, 84, 106-125, 398-438, 574-593, 693`
- `llama-rig2\ggml\src\ggml-backend-meta.cpp:2616-2764` (subgraph / all-reduce order)
- `llama-rig2\ggml\src\ggml-cuda\allreduce.cu:13-38, 79-108, 237-257, 401-414, 452-455, 622-735, 771-985`
- `llama-rig2\src\llama-model.cpp:377-612` (tensor split axes)
- `llama-rig2\common\speculative.cpp:1625-1728` (MTP pass over prompt batches)
- `llama-rig2\src\llama-context.cpp:2016-2122` (logits only for outputs; nextn rows for every token)
- `llama-rig2\src\llama-kv-cache.cpp:1566-1798`, `llama-rig2\src\llama-graph.cpp:482` (KQ mask on host)
- `llama-rig2\ggml\src\ggml-cuda\gated_delta_net.cu:10-160, 176-190` (token-by-token GDN kernel)
- `llama-rig2\ggml\src\ggml-cuda\fattn-mma-f16.cuh:1090-1127` (FA accumulator types)
- `llama-rig2\ggml\src\ggml-cuda\mmq.cu:318-373`, `mmq.cuh:165-218`, `mmq-config-blackwell.cuh:1-37`, `mmq-config-ampere.cuh:295-324`, `mma.cuh:1004-1040`, `quantize.cu:458-520`
- `llama-rig2\ggml\src\ggml-cuda\ggml-cuda.cu:1360-1405, 1618-1660` (cuBLAS fallback types)
- `refs\flash-linear-attention\fla\ops\gated_delta_rule\chunk.py:33-123`
- `refs\cutlass\include\cutlass\gemm\collective\builders\sm120_mma_builder.inl:80-115`; `refs\cutlass\examples\79_blackwell_geforce_gemm\79a_blackwell_geforce_nvfp4_bf16_gemm.cu:28-50`; `refs\cutlass\examples\86_blackwell_mixed_dtype_gemm\86_blackwell_mixed_dtype.cu:67, 106` (CUTLASS clone dated 2026-09-23)
- `refs\ik_llama.cpp\README.md:154`

Web:

- NVIDIA RTX Blackwell GPU Architecture whitepaper V1.1, pages 46-47 (RTX 5090) and 54-55 (RTX 5070, Table 6): https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- NVIDIA RTX 5060 family page (4,608 CUDA cores, 2.57 GHz boost, 759 AI TOPS): https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/
- llama.cpp discussion #15013, RTX 5060 Ti 16 GB, Llama 2 7B Q4_0 pp512 = 3,737 t/s (FA off) and 4,196 t/s (FA on): https://github.com/ggml-org/llama.cpp/discussions/15013
- llama.cpp PR #8062 (MMQ vs cuBLAS on RTX 4090) and PR #8075 (MMQ made default): https://github.com/ggml-org/llama.cpp/pull/8062, https://github.com/ggml-org/llama.cpp/pull/8075
- club-5060ti issue #7 (vLLM TP 2 on 2x RTX 5060 Ti, NVFP4, prefill 1,432 t/s at 131k): https://github.com/5p00kyy/club-5060ti/issues/7
