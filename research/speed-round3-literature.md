# Speed round 3: literature and web search

Date: 2026-10-07. About 25 searches and fetches. Abstracts, READMEs and PR text only.
"Reported" means a number from a source. "Estimate" means my own guess for this rig.
Baseline used for estimates: 22.6 ms per step, about 2.75 tokens per step, about 122 tok/s at 1k context.
Some pages came back as unreadable PDF; those items say so. The fetch tool summarizes pages with a small model, so treat details as unconfirmed until read in the source.

## Summary: ideas ranked by expected gain on this rig

1. Native Linux, and maybe P2P (E). Estimate +5-9% tok/s (saves 1-2 ms of the 3.5 ms cross-card time). Confidence: low. Only a hint exists that WDDM slows host-GPU transfers (reported 2x on an RTX 5090). Nobody measured small P2P latency.
2. Tree drafting for the first draft position, with GDN state factors (A). Estimate +5-10% tokens per step, minus extra verify cost. Confidence: low-medium. Large engineering cost.
3. Block verification (A). Estimate +2-5% tokens per step (reported +5-8% wall clock in the paper). Confidence: medium. Small cost, exact.
4. Confidence stop for drafting, with a higher max depth (A). Estimate +2-6%. Confidence: medium. Reported +2.7% tok/s on a similar setup. Small cost, exact.
5. Fine-tune the MTP head for multi-step drafting (A, FastMTP). Estimate +4-8% tokens per step. Confidence: low. It changes weights of the drafter, so it may break your "no retraining" rule.
6. Programmatic dependent launch (PDL) between kernels in the graph, or a partial megakernel (D). Estimate +2-4%. Confidence: low-medium. Your own measurement of kernel gaps inside the graph decides this.
7. int4 all-reduce payload (B). Estimate under +2%, KL risk. Confidence: low. Not recommended.
8. Prefill scale overhead (C). No literature found. FP8/FP4 block-scaled MMA cannot hold IQ3_S/K-quant weights exactly (my analysis). Expected gain about 0 from literature.

Expected combined for the cheap exact set (3 + 4): about +5-9% tok/s. Tree drafting is the only item in the literature with a large reported upside, and the reported numbers come from other drafters and hardware.

---

## A. Lossless speculative decoding upgrades

### A1. Tree drafting and tree verification for GDN hybrids
What it is. Draft several candidate tokens per position and verify the whole tree in one pass. The problem for GDN layers is that each tree node needs the recurrent state of its own path. Storing one full 128x128 state per node per head is too large. The 2026 papers solve this in three ways: tree-masked parallel kernels, "factors" instead of state snapshots, and path-parallel scans.

Reported work:
- STree (arXiv 2505.14969, 2025). First tree decoding for SSMs and SSM/Transformer hybrids. It lets the scan follow a tree structure instead of replaying unrolled paths. Sources: https://arxiv.org/pdf/2505.14969 and https://openreview.net/forum?id=a95Vd41o1u
- Bole, "Efficient Tree Speculation for Hybrid-Attention Language Models" (arXiv 2608.01651). Keeps one committed state per layer and writes each candidate branch as small token-size factors (P, K, U). After sampling, rebuilds only the accepted path's state with a matmul. Reported: 82-99x less transient memory than full snapshots; linear-attention tree verification 3.4-7.7x faster than baselines. Qwen3.5 4B/9B/27B/122B-A10B, drafter = native MTP heads, top-k 4, max depth 8. Mean accepted tokens 6.29-7.07. End-to-end up to 4.72x over autoregressive on GB10 and 3.62x on A100, up to 2.03x / 1.39x over the best tree baseline. Reported TPOT -49.9% on agent workloads. Caveat: MAT of 6-7 is much higher than your 2.75, probably from agent and code workloads and a deep tree. Do not expect this on chat text. Source: https://arxiv.org/html/2608.01651v1
- SpecLA (arXiv 2607.16673). Three verify kernels: state-resident serial (chains), tree-masked parallel, chain-decomposed hybrid. Factor buffering cuts state recovery latency 2.74-4.28x versus token replay. Reported end-to-end 1.42x (mixed), 1.70x (GSM8K), 1.06x (HumanEval) on an H100 with a 1.3B GDN model and an EAGLE-style drafter. Source: https://arxiv.org/html/2607.16673v1
- TreeWY (arXiv 2608.20961). Tree-structured WY transform. Stores a small pseudo-value matrix and rebuilds the accepted state with one triangular solve. Qwen3.5 35B and 397B. Mainly saves memory and TTFT, costs "a few percent" when not memory-bound. Abstract only; PDF unreadable. Source: https://arxiv.org/abs/2608.20961
- LumoTree / "GDN Tree-Scan" (arXiv 2609.23900). Path-parallel verification. Reuses recurrent state tiles, gathers conv history per path, remaps attention caches, fuses selection, device-resident acceptance and graph replay. Measured on DGX Spark. PDF unreadable, no numbers retrieved. Source: https://arxiv.org/abs/2609.23900
- llama.cpp PR 22400 keeps per-token GDN intermediates so a rejected draft needs only a partial rollback. Reported by a blog on Qwen 27B with MTP: 2.24x at gamma=2 (acceptance 0.83), 2.40x at gamma=3 (0.72), on DGX Spark. Source: https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/

Exact? Yes, if the tree uses a correct multi-draft verification rule (see A2). Plain "accept the longest argmax-matching path" is not exact at temperature 1.
Cost here. High. You need a tree-masked GDN verify kernel (or factors), conv state handling per path, attention mask per node for the 16 full-attention layers, and tree-aware KV commit. Your cross-card sums scale with token count, so a 6-7 node tree makes each all-reduce payload larger. The link is about half of the 27 us, so a 1.5x payload costs roughly +3-4 us per sum, about +0.4-0.5 ms per step.
Estimate (mine): a small tree (top-2 only at draft position 1, then chains) raises tokens per step by maybe 0.15-0.3 (+5-10%). The verify pass costs +1-2 ms because of extra tokens in GDN/attention and sums. Net gain +3-7%. Low confidence. A 4-token chain verify is cheap because GEMV reads weights once; going to 6-7 nodes is still mostly memory-bound, so extra GEMV cost is small.

### A2. Block verification and multi-draft verification
- Block Verification (Sun et al., ICLR 2025). Verifies the draft block jointly instead of token by token. Proven optimal in expected tokens per iteration and never worse than token-level. Reported +5-8% wall-clock over standard verification across tasks. Exact distribution. Source: https://arxiv.org/html/2403.10444v3
  Cost: small. You must keep the draft probabilities (after your temperature/top-k/top-p processing) for all 3 drafts, and the target probabilities for 4 positions, then run a backward-style acceptance pass. Works for chains only. Estimate: +0.05-0.15 tokens per step. Since your acceptance is 0.6 per draft, there is room; I have no number for this regime. Medium confidence.
- Greedy multi-path block verification (arXiv 2602.16961). Block verification extended to multiple paths. Not read in detail. Source: https://arxiv.org/html/2602.16961
- Traversal Verification (arXiv 2505.12398). For trees. Goes leaf to root, uses sequence-level probabilities, and keeps a parent usable after its children fail. Proven exact. Abstract gives no numbers. Source: https://arxiv.org/abs/2505.12398
- SpecTr (optimal transport, NeurIPS 2023), SpecHub (EMNLP 2024), recursive rejection sampling. SpecHub reported up to +0.29 tokens per step on Llama and +0.19 on Vicuna over recursive rejection sampling. UniVer (2605.04543) reported +4.2% to +8.5% accepted length over recursive rejection sampling without replacement, exact. Sources: https://arxiv.org/pdf/2411.05289 , https://pith.science/paper/2605.04543
Use with trees: traversal verification or a multi-draft rule is required for exactness. Plan A1 and A2 together.

### A3. Dynamic draft length and confidence stop
- SpecDec++ (arXiv 2405.19715). Trains an acceptance-prediction head; stops drafting when the predicted chance of a rejection passes a threshold. Reported +7.2% to +11.1% over fixed-length speculative decoding (2.04x-2.26x total) with a separate draft model. Source: https://arxiv.org/abs/2405.19715
- vLLM PR 60068: `draft_confidence_threshold`; chain ends before the first draft whose draft top-1 probability is under the threshold, the first draft is always kept. Reported on Qwen3.8-Flash-Next NVFP4, DGX Spark, one request, max depth 6, threshold 0.6, fallback depth 3: throughput 49.1 -> 50.5 tok/s (+2.7%), mean accepted length 2.80 -> 3.07 (+9.6%). By category: coding +16.7%, RAG +9.0%, math +4.2%, summarization -7.7%. Batch 4: -1.2% (noise). Source: https://github.com/vllm-project/vllm/pull/60068
- A related DGX Spark project (issue 94) uses threshold 0.70, base depth 4, up to 6, expecting +12-16% on predictable text. Design only, no results. Source: https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/issues/94
- SpecLA also prunes low-probability paths before verification (see A1).
Exact? Yes, if the stop rule depends only on the context and previous draft tokens and the draft distribution (not on the target's outputs and not on the token being decided). Stopping on the draft top-1 probability fits this.
Cost here: small. Problem: your step is one CUDA graph with a fixed 3-draft chain. A variable length needs either several graphs (verify with 2, 3, 4, 5 tokens, picked on the host after drafts) or a device-side early exit. Drafting 3 tokens already costs about 0.9 ms each in your 2.8 ms MTP budget, so every skipped draft saves time too. Estimate: +2-6%. The reported +9.6% accepted length comes from a deeper maximum (6) plus the stop; with your acceptance of 0.57-0.62, deeper drafts are cheap only if the stop avoids most of them.

### A4. MTP heads drafting more than one token
- Qwen3-Next and similar: one MTP module applied autoregressively (feeds its own token and hidden state back). Most providers ship one module. Source: https://sebastianraschka.com/llm-architecture-gallery/mtp/
- FastMTP (Red Hat, Sept 2026). Fine-tunes the single head for recursive drafting, because the shipped head only trained on ground-truth inputs. Reported on Qwen3-Next-80B (GSM8K data): acceptance per position 0.897 -> 0.912, 0.719 -> 0.776, 0.476 -> 0.616. Up to 1.25x lower median inter-token latency in vLLM. The page says "lossy", but I read it as only the drafter changing while verification stays exact (my reading, unconfirmed). It needs training on your side. Source: https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads
- llama.cpp MTP3 on Qwen3.6-35B-A3B Q4_K_M, RTX 5090: acceptance about 0.706, decode 279 -> 291 tok/s after a CUDA graph cache fix (+4-5%). Not relevant to your graph, but shows 3 drafts with acceptance about 0.7 on a comparable model. Source: https://github.com/ggml-org/llama.cpp/pull/28549
- Qwen 27B dense with MTP, llama.cpp on DGX Spark: 0.83 acceptance at gamma=2 and 0.72 at gamma=3 (reported by the zolotukhin.ai blog above). Your 0.57-0.62 per draft is lower than these numbers. Sampling at temperature 1.0 lowers acceptance compared with greedy tests; check whether those numbers were greedy.
- DeepSeek-V3 reports about 85-90% acceptance for the second token with MTP-1 (from memory, not re-checked today).
Exact? Yes (drafter-only change). Cost: training run and data. Estimate +4-8% tokens per step if acceptance at positions 2-3 rises about 0.05-0.1.

---

## B. Tensor parallel decode over a slow link

- Communication Compression for Tensor Parallel LLM Inference (arXiv 2411.09510). Fine-grained quantization of selected activations, 3.5-4.5x less data, up to 2x lower TTFT, small quality loss. Source: https://arxiv.org/abs/2411.09510
- Flash Communication (arXiv 2412.04964). Low-bit all-reduce, more than 3x faster intra-node communication, 2x lower TTFT, nearly no accuracy loss. Source: https://arxiv.org/abs/2412.04964
Both change model outputs slightly (lossy compression of activations) and both aim at prefill/TTFT with large messages. Your decode messages are tiny, so latency (about 13 us) dominates over link time (about 13 us). Your int8 + scale + error feedback is already in line with this literature. Going to int4: link time halves, saving about 6 us x 128 = 0.8 ms at best (3%). It adds quantization noise that may cost KL. Estimate under +2% net. I recommend against it.
- Low-latency all-reduce kernels (one-shot / "direct data access" where each rank reads the other's buffer) cut latency from O(N) steps to one hop. You already do one hop with 2 ranks. Sources found: https://arxiv.org/pdf/2607.16100 ("Every Microsecond Matters"). Not read.
- Overlapping all-reduce with GEMV: no useful method found. In decode each sum feeds the next norm and GEMV, so there is nothing independent to overlap, except splitting work across layers (for example starting the next layer's independent GEMVs, like the MTP head or the GDN gate and conv projections, while the sum is in flight). Estimate: 0.3-0.8 ms if the GDN layers have projections that do not depend on the sum. Needs a dependency analysis of your layer graph; I did not look at the code.
Cheaper latency ideas (my own, not from papers): pin the host copy threads, use `cudaHostRegister` with write-combined flags, and poll host flags from the kernel instead of using events. Only if not already done.

---

## C. Low-bit codebook kernels

- FLUTE (arXiv 2407.10960): LUT-quantized GEMM, copies the lookup table across shared-memory banks to avoid conflicts. Aimed at 3-4 bit on A100/RTX 4090, batch sizes 1-32. No bandwidth percentage found. Source: https://arxiv.org/pdf/2407.10960
- Marlin: FP16 x INT4, near 4x over FP16 on Ampere, fused dequant and async pipeline. Not codebook. No sm_120 data.
- QTIP: reported 37.6% slower than SPHQuant at batch 1 on RTX 4090 (arXiv 2609.24875). Trellis decoding costs ALU time, so codebook kernels are often compute-bound at batch 1 on fast-memory cards. On your 440 GB/s card there is more ALU headroom per byte.
- ik_llama.cpp has newer IQ kernels (IQ2_KS, IQ3_KT, IQ4_KT); one note reports about +2% decode on IQ2_KS. Source: https://github.com/ikawrakow/ik_llama.cpp
- I found no source that reports above 90% of DRAM bandwidth for 3-bit codebook GEMV. Your 385 of about 430 GB/s (about 90%) already matches the best known numbers I could find (Hazy megakernel reports 78% for Llama-1B on H100, see D). Remaining gain: at most 3-5% of the 15.2 ms GEMV time (about 0.5-0.7 ms, 2-3% per step). Estimate.

Prefill with block-scaled FP8/FP6/FP4 MMA on sm_120:
- Reported: on consumer Blackwell, `mma.sync kind::mxf8f6f4` and `kind::mxf4nvf4` exist. One source reports INT8 about 246.9 TFLOPS, FP4 about 474 TFLOPS, MXFP8 block-scaled about 1014 TFLOPS in cuBLASLt (note that conflicts with another note of "~202 TFLOP/s" for MXFP8 and with your 190 int8 peak; I could not confirm which is right). FP32-accumulate throughput is said to be halved for legacy warp MMA, while block-scaled instructions are not throttled. Sources: https://github.com/Theodore-Liu/sm120-fp4 , https://github.com/pantheongpu/pantheonsim/pull/174 (the PR has no throughput data), https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/
- Exactness (my analysis, not from a source). MX scales are powers of two (E8M0) per 32 values. IQ3_S has an fp16 super-scale and a 4-bit sub-scale giving an odd integer multiplier (1+2s, up to 31) per 32 values. That multiplier is not a power of two, so it cannot go into the MX scale. Folding it into the element needs values up to 15 x 31 = 465, which FP8 e4m3 (3 mantissa bits) cannot store exactly. Activations would also change from llama.cpp's q8_1 (int8 with fp scale per 32) to FP8, which changes numerics. IQ2_XXS and IQ2_S have the same problem; Q4_K and Q6_K have integer sub-scales up to 63 with the same issue. Verdict: not exact, so excluded by your KL rule, unless you accept a measured KL under 0.001 (unlikely to be safe on all layers).
- Cutting the f32 scale cost without changing numerics: I found no paper. Options (estimates, no source): keep the integer sub-scale multiply in int32 (if not already), batch two 16-value blocks that share a sub-scale before the f32 FMA, and convert the accumulator with packed ops. Gain 3-8% of GEMM time if scale work is 20-30% of it. Low confidence.

---

## D. Removing host round trips

- Mirage Persistent Kernel (MPK, arXiv 2512.22219): compiles the forward pass into one megakernel with an in-kernel task scheduler. Reported 1.0-1.7x throughput versus SGLang/vLLM, 1.2-6.7x lower latency (mostly small models and datacenter GPUs, multi-GPU too). One 2026 measurement: kernel-per-operator launch cost is about 14.6% of decode time on Qwen2.5-1.5B. Sources: https://arxiv.org/html/2512.22219v2 , https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Hazy Research megakernel ("Look Ma, No Bubbles"): Llama-1B, 2.5x faster than vLLM and 1.5x faster than SGLang on H100, 3.5x on B200. Uses 78% of H100 bandwidth versus about 50% for vLLM/SGLang. Mechanism: weights for the next stage are loaded while the previous stage finishes. Source: https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main (and the earlier blog; page not read in full)
- CUDA graph conditional WHILE nodes: one captured decode step in a WHILE node, patched with `cudaGraphExecUpdate`, used by a small project (kekzl/imp PR 1895). Only a design note; no numbers retrieved. Source: https://github.com/kekzl/imp/pull/1895 . TensorRT-LLM PR 19816 removes host work between speculative steps. Source: https://github.com/NVIDIA/TensorRT-LLM/pull/19816
- LumoTree (see A1) puts acceptance on the device and replays graphs, but gives no number I could read.
Fit for your rig. The headline gains come from tiny models where launch gaps dominate. Your step is already one graph per card, and GEMVs reach about 90% of bandwidth. What remains: gaps between roughly 500-700 kernel nodes in the graph (about 1-2 us each, so about 0.5-1.2 ms, estimate) and the 1.1 ms of "other kernels". Cheap route (my knowledge, no source retrieved today): programmatic dependent launch (PDL, `cudaLaunchAttributeProgrammaticStreamSerialization` + `griddepcontrolwait`), supported in graphs on recent CUDA, lets the next kernel prefetch weights before the previous one ends. Estimate +2-4%. A full megakernel is a rewrite and, with the host-staged all-reduce, hard to build. Host round trip between steps (two-card sync, read accepted count): estimate under 1% unless measured otherwise.

---

## E. Native Linux versus Windows WDDM, and P2P

- Reported: NVIDIA/cuda-python issue 1207. RTX 5090 and 3090 Ti, Windows 11, driver 581.57. GPU<->RAM copy is about 2x faster on Linux than in WDDM. TCC mode on Windows matches Linux but is blocked on GeForce. No GB/s numbers in the issue. This hits your pinned-host all-reduce path directly. Source: https://github.com/NVIDIA/cuda-python/issues/1207
- Reported: Microsoft's MCDM driver model is meant to close the gap but is not available on consumer cards. Hardware-accelerated GPU scheduling reduces launch overhead (stated for WSL2). Source: https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/
- Reported (WSL2, single GPU llama.cpp): penalty about 1.7% on token generation; clearing other GPU consumers on the host gave +12% (67.3 -> 75.6 tok/s) with much less variance. Source: https://ianlpaterson.com/blog/llama-cpp-build-from-source-cuda-benchmark/ . Hint: close every other GPU-using program (browser, overlays) on Windows, and check HAGS on/off. Free to test.
- No measured native Windows versus native Linux tok/s on the same card with a decode-sized workload found.
- P2P on RTX 4090/5090 with aikitoria/open-gpu-kernel-modules (driver 590.48.01): forces BAR1 P2P, the GPU writes directly to the other GPU's physical addresses. Needs IOMMU disabled or in passthrough, and a large BAR; the notes say 5090s do not come up with large BAR by default (unknown for your 5060 Ti). Linux only. Source: https://github.com/aikitoria/open-gpu-kernel-modules
- A user report on 2x RTX 3090 enabling P2P (driver patch + vLLM): vLLM Qwen 35B 90-197 tok/s with P2P versus about 35-65 without; gains 10-30% overall, much bigger for MoE. No latency numbers. Source: https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
- I found no measured small-transfer latency between two cards on separate x4 links.
Estimate (mine): with P2P over two x4 links, a 4-token int8 sum (about 5 KB) should take about 5-10 us per direction including a flag write, compared with 27 us now. That would save 128 x (27 - ~10) = about 2 ms (about 9%). Without P2P but on Linux with native pinned memory, maybe 27 -> 18-22 us, saving about 0.6-1.1 ms. The path between the two root ports also matters: on many consumer boards P2P between CPU root ports is slow or unsupported, and the cards share the CPU link for both. Cost: a Linux install (dual boot), CUDA and driver setup, and porting the Windows-specific code paths. The test is cheap before porting: boot Linux, run a pinned-memory ping-pong between the two cards and a P2P `cudaMemcpyPeer`/store ping-pong, and compare with 27 us.

---

## Not applicable here

- Lossy or relaxed acceptance (Margins Not Windows 2609.02897, Nucleus Speculative Decoding 2610.07822, fastmtp-style lossy acceptance): break the exact-distribution rule.
- Sparse attention, KV eviction, 4-bit KV: excluded by your rules.
- Flash Communication, int4 communication compression papers (2411.09510, 2412.04964): lossy and aimed at large prefill messages; your int8 + error feedback is already the right idea for decode.
- In-network or in-switch all-reduce (SiFAR, switch-centric designs): needs special switches. (The SiFAR PDF was unreadable; this line is from its title only.)
- NVLink-based methods (FlexLink and others): no NVLink on 5060 Ti.
- Mirage MPK as a drop-in compiler: it targets standard Transformer graphs and datacenter GPUs; GDN hybrids with codebook GGUF weights would need a full backend.
- TCC mode on Windows: blocked on GeForce; needs a driver patch of unknown safety.
- MXFP8/MXFP4 block-scaled MMA for exact IQ/K-quant weights: scale format cannot hold the non-power-of-two sub-scales (see C).
- QTIP, AQLM, QuIP# kernels: need those formats; your weights are GGUF IQ/K quants, and the reported kernels do not beat 90% bandwidth.
- Marlin: INT4 uniform quant; your weights are codebook-based.
- SpecDec++ acceptance-prediction head: needs training; a plain draft top-1 threshold gives most of the benefit.
- Batch-size-dependent dynamic speculation (vLLM issue 49548): concerns concurrency; you run batch 1.
- llama.cpp PR 28549 (CUDA graph cache for MTP): you already capture one graph per step.
