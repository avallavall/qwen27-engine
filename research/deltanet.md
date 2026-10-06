# Gated DeltaNet (GDN) for Qwen3.8-27B: math, llama.cpp kernels, decode, MTP rollback, prefill, checkpoints

Scope: the 48 GDN layers (16 K heads, 48 V heads, head dim 128, conv kernel 4).
Rig numbers come from `_brief.md`. Every estimate is marked "estimate" and shows its arithmetic.
No GPU work was done. One small CPU numpy check was run in the project venv (section 3.4).

## Summary

1. **Math.** Per token and layer: depthwise causal conv (4 taps, no bias) + SiLU on 10240 channels, L2 norm of q and k
   (eps 1e-6), q times 1/sqrt(128), `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`, then per V head
   `S = e^g S; d = beta (v - S^T k); S = S + k d^T; o = S^T q`, then `RMSNorm(o) * w * SiLU(z)`. The state is f32,
   48 heads x 128 x 128 per layer = 3 MiB, 144 MiB for 48 layers. Conv state is 5.625 MiB. **Total 149.625 MiB.**
   A fit of 19 llama.cpp checkpoint sizes from the production log gives an intercept of 149.63 MiB, which confirms it.
2. **llama.cpp kernels.** The decode/AR kernel keeps one V column per warp in registers (f32) and loops over tokens.
   The chunked MMA kernel (PR #29353) uses 16-token chunks and a bf16 "hi + lo" split with 3 MMAs per product.
   **The MMA kernel never runs in production**, for two independent reasons: its gate needs `H_k == 16` and
   `H in {16,32,48,64}` (with `-sm tensor` each card sees 8 and 24), and it needs `K == 1` state snapshot
   (MTP sets `K = n_rs_seq + 1 = 4`).
3. **Decode floor.** Read + write the state once per step: 151 MB per card, **0.389 ms per card** at 388 GB/s, both
   cards in parallel. From code reading, llama.cpp moves about 3.5x that per verify step (an extra `get_rows` copy of
   the state plus 4 snapshots): **1.36 ms per card (estimate)**. A 4-token kernel can keep the state in registers
   (96 CTAs per card, 16 KiB of state each). f32 state is possible and recommended. It costs 0.19 ms per step more than bf16.
4. **MTP rollback.** llama.cpp and vLLM write one state per draft position (1 read + 4 writes, 0.97 ms per card).
   SGLang's default adds a copy (1.36 ms). The cheapest exact option: write no snapshots, keep per-token `k, v, g, beta`
   (3.2 MB per card), and replay the accepted tokens at the start of the next step's kernel (1 read + 1 write, 0.39 ms).
   ReplaySSM (SGLang, FlashInfer, vLLM fork) goes further with a history ring and rare flushes, at much higher complexity.
5. **Chunked prefill.** FLA uses chunk 64 and the WY/UT form. About 0.18 MFLOP per head-token, 0.42 GFLOP per token for
   all 48 layers (about 0.8% of the dense weight FLOPs). It maps to `mma.sync m16n8k16` bf16 on sm_120 (no wgmma).
   Cheapest path: port llama.cpp's `gdn_single` (1-2 weeks, estimate). Expected prefill gain on this rig: small
   (1-3% estimate). It must be measured.
6. **Checkpoints.** 149.6 MiB each (74.8 MiB per card). llama.cpp checkpoints also carry the full MTP draft KV
   (4 KiB per token, 661 MiB at 130k tokens) because the plain KV cache ignores `PARTIAL_ONLY`. The new engine can drop
   that part. Save to VRAM: 0.4 ms. Save to or restore from host: about 22 ms per card, both cards in parallel.

---

## 1. The math

### 1.1 Per-token computation (one GDN layer)

Sources: HF `modeling_qwen3_5.py` (transformers main), paper arXiv 2412.06464 eq. 10, llama.cpp `qwen35.cpp`.

Input `x` [5120] is the output of `attn_norm`.

| Step | Formula | Shape | Source |
|---|---|---|---|
| Projections | `qkv = W_qkv x`, `z = W_gate x`, `a = W_alpha x`, `b = W_beta x` | 10240, 6144, 48, 48 | HF 555-562; `qwen35.cpp:369-380` |
| Channel order | `qkv = [q: 16x128, k: 16x128, v: 48x128]` | 2048+2048+6144 | HF 593-605; `qwen35.cpp:419-435` |
| Conv1d | `u_t[c] = sum_{j=0..3} w[c][j] * qkv_{t-3+j}[c]`, depthwise, causal, no bias | 10240 | HF 260-279, 512-519 |
| Activation | `u = SiLU(u)` on all 10240 channels | | HF 580-586; `qwen35.cpp:406-409` |
| L2 norm | `q = q / sqrt(sum q^2 + 1e-6)`, same for k, per K head | 16 x 128 | HF 284-287, 624; `models.h:14-18` |
| Query scale | `q = q / sqrt(128)` | | HF 458; `gated_delta_net.cu:116` |
| Decay (log) | `g = -exp(A_log) * softplus(a + dt_bias)` (g <= 0) | 48 | HF 609; `qwen35.cpp:384-388` |
| Beta | `beta = sigmoid(b)` | 48 | HF 607; `qwen35.cpp:377` |
| Recurrence | `S = e^g S`; `d = beta (v - S^T k)`; `S = S + k d^T`; `o = S^T q` | S: 128 (K) x 128 (V) per V head | HF 470-481 |
| Gated norm | `y = (o / sqrt(mean(o^2) + 1e-6)) * w_norm * SiLU(z)` per V head | 48 x 128 | HF 208-224, 649 |
| Output | `out = W_out y` | 6144 -> 5120 | HF 652 |

Notes:
- The recurrence is the paper's eq. 10, `S_t = S_{t-1}(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T`
  (paper orientation V x K). HF applies the decay first, then the delta step on the decayed state. llama.cpp does the
  same: `delta = (v - g*kv) * beta` with `kv` from the undecayed state, then `S = g*S + k*delta`
  (`gated_delta_net.cu:91-110`).
- GGUF stores `ssm_a = -exp(A_log)` and `ssm_dt.bias = dt_bias` (`conversion/qwen.py:396-399`). llama.cpp's softplus
  uses threshold 20 like torch (`unary.cuh:113-115`).
- `ssm_norm.weight` is plain `w`. All other RMSNorm weights got `+1` at conversion (`conversion/qwen.py:402-403`;
  HF `Qwen3_5RMSNorm` uses `1 + w`, line 844; `Qwen3_5RMSNormGated` uses `w`, line 221).
- Head mapping. HF repeats q, k with `repeat_interleave` (HF 610-612), so V head j reads K head `j // 3`. The llama.cpp
  converter reorders V heads to tiled order (`conversion/qwen.py:454-466`), so in the GGUF **V head j reads K head
  `j % 16`**. Both llama.cpp kernels index q/k with `h % H_k` (`gated_delta_net.cu:43`, `gated-delta-net-mma.cu:144`).
  FLA uses the grouped mapping `i_h // (HV // H)` (`chunk_fwd.py:89`). Any code ported from FLA must switch to `% 16`.
- Output gate type: the Qwen3.8-27B `config.json` says `"output_gate_type": "swish"` (= SiLU) and
  `"mamba_ssm_dtype": "float32"`.

### 1.2 State shape and dtype per layer

| Item | Shape | dtype | Bytes per layer | x 48 layers |
|---|---|---|---|---|
| Recurrent state S | [48 V heads][128 V][128 K], K contiguous | f32 | 48 x 128 x 128 x 4 = 3,145,728 (3 MiB) | 150,994,944 (144 MiB) |
| Conv state | [10240 channels][3 past inputs] | f32 | 3 x 10240 x 4 = 122,880 (120 KiB) | 5,898,240 (5.625 MiB) |
| **Total** | | | | **156,893,184 B = 149.625 MiB** |

- Layout `[head][V][K]` with K contiguous is used by llama.cpp ("state is stored transposed: M[col][i] = S[i][col]",
  `gated_delta_net.cu:60`), by vLLM (`mamba_utils.py:296-300`, shape `(HV, V, K)`) and by FlashInfer pools
  (`[pool, HV, V, K]`). The HF reference uses `[B, HV, K, V]`.
- Sizes in llama.cpp: `n_embd_r = 3 x (6144 + 2 x 16 x 128) = 30720` floats (`llama-hparams.cpp:208-233`),
  `n_embd_s = 128 x 6144 = 786432` floats (`llama-hparams.cpp:236-257`), both F32 (`llama-model.cpp:2741-2742`).
- Per card with the head split used by `-sm tensor` (8 K heads, 24 V heads per card): S = 72 MiB, conv = 2.8125 MiB,
  total **74.8 MiB per card**.

### 1.3 Confirmation of the ~150 MiB number

- Arithmetic: 144 MiB + 5.625 MiB = 149.625 MiB.
- Production log `qwen38_27\arranque.log.err`: 19 "erasing old context checkpoint" lines where `pos_max + 1 == n_tokens`.
  Least-squares fit: **size = 149.626 MiB + 4.0234 KiB x n_tokens**, max residual 0.0005 MiB. The intercept is the
  GDN state. The per-token part is the MTP draft KV (section 6.2).
- With MTP, llama.cpp allocates `1 + n_rs_seq = 4` planes (`llama-memory-recurrent.cpp:101-103`), so the recurrent
  buffer is 4 x 149.625 = 598.5 MiB in total, about 299 MiB per card.

---

## 2. llama.cpp kernels in `llama-rig2`

### 2.1 Decode / autoregressive kernel: `ggml/src/ggml-cuda/gated_delta_net.cu`

- Kernel `gated_delta_net_cuda<S_v=128, KDA=false, keep_rs_t>` (lines 10-173). Launch: grid `(H, n_seqs, 128/4)`,
  block `(32, 4)` (lines 186-190), `__launch_bounds__(128, 2)`.
- **One warp owns one V column.** Each lane holds 4 rows (K indices `r*32 + lane`) of that column in registers
  (lines 39-67). A warp reads 512 contiguous bytes, so loads are coalesced.
- Token loop with the state in registers (lines 69-164). Per token: load q, k (4 values per lane), compute
  `kv = warp_reduce_sum(s . k)`, `delta = (v[col] - e^g kv) * beta`, update `s = e^g s + k delta`, compute
  `o = warp_reduce_sum(s . q)`, lane 0 writes `o * scale` (lines 90-117). Two warp reductions per token.
- Snapshots (`keep_rs_t`, lines 151-163): after token t it writes the state into slot `n_tokens - 1 - t` if that slot
  is `< K`. Slot 0 is the newest state. Without snapshots it writes only the final state (lines 166-172).
- Inputs are f32 and already prepared by other ops: conv + SiLU (fused `ssm_conv`), L2 norm (`rms_norm` + scale),
  and the alpha/beta projection kernel `k_gdn_ab_mul_mat` (lines 408-445, fused for up to 8 tokens,
  `gated_delta_net.cuh:16`) which outputs `softplus(a + dt) * ssm_a` and `sigmoid(b)`.
- With the fusion `ggml_cuda_try_gdn_cache_fusion` (`ggml-cuda.cu:2811-2873`) the kernel writes the snapshots directly
  into the recurrent cache planes and the separate `cpy` is skipped.

### 2.2 Chunked kernel: `gated-delta-net-mma.cu` (PR #29353)

Status: PR #29353 is open upstream (author am17an). The rig carries it with an off switch `RIG_OFF_29353`
(`gated_delta_net.cu:343-353`).

How `gdn_single<C=16, V, BLOCKS>` works (lines 123-358):
- One CTA of 256 threads (8 warps) per (sequence, V head, V slice). V slice is 32 or 128 columns. Chunk C = 16 tokens.
- The state lives in **f32 accumulator tiles in registers** for the whole sequence (lines 148-159). For the whole-head
  path each thread holds 64 floats.
- Operands go through shared memory as **two bf16 numbers per value, `hi = bf16(x)`, `lo = bf16(x - hi)`**
  (lines 11-42). Every product runs 3 `mma.sync` calls: `lo*hi + hi*lo + hi*hi` (lines 94-107). This keeps about
  16 mantissa bits. Tiles: `acc = tile<16,8,float>`, A = 16x16 bf16, B = 8x16 bf16 (`m16n8k16`), loaded with
  `ldmatrix` from XOR-swizzled shared memory (lines 59-92).
- Per chunk: store q (pre-scaled) and k; warp 0 computes the inclusive prefix sum of g in f32 (lines 197-218);
  Gram `K K^T` and `Q K^T` with decay `exp(G_r - G_c)` and causal masks (lines 254-275); forward substitution per row
  gives `(I + L)^-1` with `L = beta * strictLower(decayed KK^T)` (lines 280-303); projections `K S`, `Q S`;
  `delta = beta (v - e^G K S)`; `solved = inverse * delta` (lines 308-313); output `e^G Q S + P solved`
  (lines 315-327); state update `S = e^{G_last} S + K^T (solved * e^{G_last - G_t})` (lines 328-343).
- Shared memory for the whole-head variant (estimate from the struct, lines 44-57): q, k 16 KiB + p, inverse 2 KiB +
  delta 8 KiB + state scratch 64 KiB + small = about 91 KiB, under the 99 KB per-block limit of sm_120.
- Path selection (lines 380-412): returns the AR path unless `eligible`, `H_k == 16`, `H in {16,32,48,64}`,
  `n_tokens >= 64`, and the device is one of the benchmarked archs (GA10x, Ada, Blackwell 1200, DGX Spark).
  If `H x n_seqs x 2 <= n_SM` it slices V into 32-column CTAs, else one CTA per head.
- `eligible` (`gated_delta_net.cu:263`) requires scalar g per head, **op param K == 1**, and S_v = 128.
- Upstream reports +10% to +19% prompt speed. For Qwen3.8-27B Q8_0 on an RTX 5090: 3852 -> 4302 t/s at 2048 tokens.
  That is 2048/3852 - 2048/4302 = 56 ms saved per 2048 tokens, about 27 ms per 1024 tokens.

### 2.3 Why the chunked kernel does not run in production

| Condition in `select_gdn_mma_path` / `eligible` | Production value | Result |
|---|---|---|
| `a.H_k == 16` (`gated-delta-net-mma.cu:382`) | 8 per card | fails |
| `a.H in {16,32,48,64}` (line 382) | 24 per card | fails |
| op param `K == 1` (`gated_delta_net.cu:263`) | 4 (MTP: `n_rs_seq = 3`, `delta-net-base.cpp:572`) | fails |
| `n_tokens >= 64` (line 382) | prefill ubatches yes, verify (4 tokens) no | |

Why each card has 8 K heads and 24 V heads with `-sm tensor`:
- The meta backend splits the fused `attn_qkv` and `ssm_conv1d` in segments of `key_dim` with `2 + head_ratio = 5`
  segments (`llama-model.cpp:651`), the state cache in segments of `16 x 128 x 128` (`llama-model.cpp:663-665`), and
  the alpha/beta/dt/A tensors by K-head groups. Each segment is split between the cards. So card 0 gets K heads 0-7 and
  V heads {0-7, 16-23, 32-39}. Local V head j then uses local K head `j % 8`, which keeps the tiled mapping valid.
- `handle_gated_delta_net` asserts that q, k, v, g, beta are split on axis 1 (heads) (`ggml-backend-meta.cpp:951-966`).

The `H_k == 16` gate is a whitelist of benchmarked shapes. The kernel math only uses `h % H_k` (line 144), so
8 K heads / 24 V heads should work, but nobody has tested it. The `K == 1` condition is a separate problem: the kernel
writes only the final state, and with MTP llama.cpp asks every GDN op for 4 snapshots, even during prefill. The PR text
says the same ("only activates when K=1"). Prefill does not need snapshots, because rollback only happens after a verify
step, and a verify step writes its own snapshots.

### 2.4 Extra state traffic in llama.cpp (from code reading, to confirm with nsys)

- `build_rs` gathers the current state with `ggml_get_rows` (`llama-graph.cpp:3520-3554`, called at `qwen35.cpp:402`).
  That is a real copy: 3 MiB per layer read and 3 MiB written (1.5 + 1.5 MiB per card). The GDN kernel then reads the copy.
- With `n_rs_seq = 3` the conv state is saved by 4 separate `ggml_cpy` nodes per layer (`delta-net-base.cpp:497-530`).
- Per verify step (T = 4), per card, estimate: gather 75.5 MB read + 75.5 MB write, kernel read 75.5 MB, 4 snapshot
  writes 302 MB. Total 528.5 MB / 388 GB/s = **1.36 ms**. Plain decode (T = 1): 4 x 75.5 = 302 MB = 0.78 ms.
- Kernel count per GDN layer per card in decode (estimate from the graph): 2 GEMV, alpha/beta, conv gather, concat,
  4 conv cpy, conv+SiLU, 2 L2 norms, state gather, GDN, norm, SiLU, mul, out GEMV, plus the all-reduce. About 15-18.
  A fused design needs 3-4 (section 3.3).

---

## 3. Decode kernel design for the new engine

### 3.1 Bandwidth floor per decode step

Assumption: the state is read once and written once per step. Copy bandwidth from the brief counts read + write bytes
(`vram-bw.cu:66` uses `2 x bytes`), so it applies directly.

| Case | Bytes moved | Time at 388 GB/s | Time at 403 GB/s (OC) |
|---|---|---|---|
| One card holds all 48 heads | 2 x 150,994,944 = 302.0 MB | 0.778 ms | 0.749 ms |
| **Two cards, 24 heads each (per card, in parallel)** | 2 x 75,497,472 = 151.0 MB | **0.389 ms** | 0.375 ms |
| Conv state, per card | 2 x 2,949,120 = 5.9 MB | 0.015 ms | 0.015 ms |
| Same as row 2 with bf16 state | 75.5 MB | 0.195 ms | 0.187 ms |
| llama.cpp verify step today (2.4, estimate) | 528.5 MB | 1.36 ms | 1.31 ms |

- Per layer per card the floor is 3.15 MB, so 8.1 us at 388 GB/s. 48 layers x 8.1 us = 0.39 ms.
- For scale: the weight-read floor per step per card is 10828 MiB / 2 = 5677 MB, so 14.6 ms at 388 GB/s.
  The GDN state is 2.7% of that. The llama.cpp state traffic is 9.3% of that (1.36 / 14.6).
- Savings vs llama.cpp per step: 1.36 - 0.39 = 0.97 ms (estimate). At 1k context that is 2.6% of the 37.7 ms step.
- The state does not fit in L2. GB206 has 32 MB of L2 and 36 SMs. The state is 72 MiB per card and the weights stream
  5.4 GB per card between two uses. L2 persistence (`cudaAccessPolicyWindow`) could pin about a third of it. Blackwell
  supports L2 persistence (Blackwell tuning guide). Whether it works under WDDM, and what it costs the weight stream,
  is unknown. It must be tested.

### 3.2 One kernel for T = 1..4 tokens (MTP verify) with the state on chip

Recommended layout (per card, per layer):
- **Grid: 24 V heads x 4 V slices = 96 CTAs, 256 threads each.** A slice is 32 V rows x 128 K = 4096 floats = 16 KiB.
  With `[head][V][K]` layout the slice is one contiguous 16 KiB block. 96 CTAs on 36 SMs is 2.7 CTAs per SM, all
  resident at once, so the whole 1.5 MiB per layer is in flight together.
- Thread mapping as in vLLM's CUDA kernel: each warp owns 4 V rows, each lane holds 4 K values of each row, so
  16 floats per thread. Reductions over K use warp shuffles, two per token (`S^T k` and `S^T q`).
- Per token (sequential, state stays in registers): decay, `S^T k`, delta, rank-1 update, `S^T q`.
  4 tokens cost 8 reduction rounds, about 8 x 5 x 25 = 1000 cycles = 0.4 us (estimate). That is far below the
  8 us memory time per layer.
- Fuse the prologue: conv (from the raw `in_proj` output for the T tokens plus the conv state), SiLU, L2 norm of q and
  k, `g` and `beta`. The 12 CTAs that share a K head each recompute that head's q/k conv and norm (256 channels x 4 taps
  x T, negligible). One designated CTA updates each conv-state channel.
- Leave the gated RMSNorm to the next kernel. It needs all 128 V values of a head, which span 4 CTAs. The `ssm_out` GEMV
  can compute the 24 per-head RMS values in its prologue (3072 values per card per token). Thread block clusters exist
  on Blackwell (portable size 8) and could do it in-kernel through DSMEM. This was not checked on sm_120.
- Launch with programmatic dependent launch. The state load does not depend on the current token, so the kernel can
  start loading the state while the `in_proj` GEMV is still running, then wait (`griddepcontrol.wait`) before reading
  q/k/v. This hides latency. It does not reduce bytes.
- Resulting kernels per GDN layer: `in_proj` GEMV(s) (qkv, z, alpha, beta), GDN fused kernel, `ssm_out` GEMV with
  norm-and-gate prologue, all-reduce. 3-4 kernels instead of 15-18.

References with the same design: vLLM `fused_gdn_decode_kernel.cu` (256 threads, 32-row V chunks double-buffered with
`cp.async`, up to 8 MTP tokens, fused L2 norm, gates and gated RMSNorm; lines 85-95, 150-385), and FlashInfer
`gdn_decode_mtp.py` (f32 state, V-major, T > 1). vLLM uses one CTA per (request, V head). At batch 1 that is only 24 CTAs
per card on 36 SMs, which is why the V split above is proposed.

### 3.3 Can the state stay f32?

Yes. Reasons and costs:
- The model config declares `"mamba_ssm_dtype": "float32"` (Qwen3.8-27B `config.json`). vLLM follows that field for
  Qwen3.5-family models (`vllm/model_executor/models/config.py:806-831`). HF computes the recurrence in f32
  (HF 447-466). llama.cpp stores f32 (`llama-model.cpp:2741-2742`).
- VRAM: 72 MiB per card (one plane) plus conv. Small.
- Time: f32 costs 0.195 ms per step more than bf16 (151.0 vs 75.5 MB per card). That is about 0.5% of a 38 ms step.
- Registers: 16 floats per thread for the state slice. No pressure.
- Bitwise equality with llama.cpp is not realistic. The reduction order differs (llama.cpp sums 4 rows per lane, then a
  butterfly over 32 lanes). Equality within about 1e-6 relative is realistic with f32 FMA.

### 3.4 Synthetic numerics check (CPU, numpy, project venv)

Script in the session scratchpad (`gdn_check.py`). One head, random inputs with L2-normed q and k, `beta = sigmoid(N(0,1))`,
`g` uniform in [-0.1, 0]. These are not real activations.

| Test | Result |
|---|---|
| Chunked (UT/WY, C = 64) vs recurrent, float64, 1024 tokens | max output diff 5.6e-17, state diff 4.4e-16 |
| f32 state vs float64, 4096 tokens | max relative output error 1.6e-7 |
| State rounded to bf16 after every token, 4096 tokens | max relative output error 4.3e-3, final state 9.5e-3 |

The bf16 state error is 4 orders of magnitude larger than f32 on this synthetic data. Real-data error is unknown.

---

## 4. MTP rollback

### 4.1 What llama.cpp does now

- `n_rs_seq = draft.n_max = 3` when MTP is on (`common/common.h:398-404`, applied at `common/common.cpp:1667`).
- The recurrent tensors get `1 + n_rs_seq = 4` planes (`llama-memory-recurrent.cpp:101-103`).
- The GDN op is built with `K = n_rs_seq + 1 = 4` (`delta-net-base.cpp:571-575`). During a 4-token verify the kernel
  writes the state after each token into a plane, slot 0 = newest (`gated_delta_net.cu:151-163`). The conv state gets
  4 `cpy` nodes per layer (`delta-net-base.cpp:497-530`).
- `split_equal` keeps the last `n_rs_seq + 1` tokens of a sequence in one ubatch (`llama-memory-recurrent.cpp:455-457`).
- Rollback: `seq_rm` sets `rs_idx = rollback count` if it is 1..n_rs_seq (`llama-memory-recurrent.cpp:200-215`). The
  next graph reads the state from plane `rs_idx` through `s_copy` (`llama-memory-recurrent.cpp:1364-1383`). No copy.
- If a draft were longer than `n_rs_seq`, the server would fall back to a full speculative checkpoint
  (`server-context.cpp:3193-3216`). With 3 drafts this path is not used.

### 4.2 vLLM

- Each request in spec decode owns `num_spec + 1` state slots (`gdn_attn.py:366-368`).
- The verify kernel reads the initial state from slot `num_accepted_tokens - 1` and writes the state after token t into
  slot t (Triton: `third_party/flash_linear_attention/ops/fused_recurrent.py:106, 122-166`; CUDA:
  `fused_gdn_decode_kernel.cu:174-180, 323-347`). After acceptance nothing is copied. The next step picks the slot.
- Conv state has `kernel - 1 + num_spec` columns (`mamba_utils.py:290-294`). The update reads from offset
  `num_accepted_tokens - 1` and rolls the window (`causal_conv1d.py:860-876`).
- Default SSM dtype is "auto" (model dtype), except for Qwen3.5-family configs that set `mamba_ssm_dtype`
  (`config.py:806-831`). For this model that means f32.

### 4.3 SGLang

- Default: an `intermediate_ssm` buffer `[layers, slots, draft_tokens, HV, V, K]` (`memory_pool.py:851-880`) is written
  during verify. After acceptance, `update_mamba_state_after_mtp_verify` scatters the accepted step into the main slot
  (`hybrid_linear_attn_backend.py:1421-1520`). That adds one state read + write per step.
- ReplaySSM mode (opt-in): the verify kernel runs with `disable_state_update=True` and appends raw `(v, k, g, beta)` to a
  per-slot ring (`gdn_backend.py:1355-1392`). The commit kernel replays the accepted prefix into the f32 checkpoint
  (`gdn_replayssm_spec_fold.py`, documented as a bitwise clone of the recurrent kernel's GDN branch). A circular variant
  folds only every L tokens (`gdn_replayssm_spec_decode.py:1-40`).
- FlashInfer has CUDA (CuTe DSL) versions: `gdn_decode_bf16_wy_ucache.py` (verify only, ring of 16, T in {4, 8},
  SM90+) and `..._ucache_flush.py` (fold the ring into the state when it fills).
- The Dao AI Lab ReplaySSM post says rollback is "a pointer move with no state write-back" and that the method "halves
  the dominant state traffic from 8dn to 4dn".

### 4.4 Options and cost for this engine

Per card, per verify step with T = 4. S = 75.5 MB (one f32 state). Times at 388 GB/s.

| Option | State traffic | Time | Extra VRAM per card | Extra compute | Complexity |
|---|---|---|---|---|---|
| A. Snapshot after each position, pick slot on rollback (llama.cpp, vLLM) | 1R + 4W = 377.5 MB | 0.97 ms | 3 more planes = 216 MiB | none | low |
| A + llama.cpp's gather copy (today) | 7 S = 528.5 MB | 1.36 ms | as A | none | |
| B. Intermediate buffer + scatter after accept (SGLang default) | 1R + 4W + 1R + 1W = 528.5 MB | 1.36 ms | 4 planes | none | low |
| C. Verify without writing, separate commit kernel replays accepted tokens | 1R + (1R + 1W) = 226.5 MB | 0.58 ms | ring 3.2 MB | 1-4 extra token steps | medium |
| **D. No snapshots; next step's kernel first replays the accepted tokens, then runs the new T tokens** | 1R + 1W = 151.0 MB | **0.39 ms** | ring 3.2 MB | 1-4 extra token steps | medium |
| E. ReplaySSM ring with periodic flush | about 1R + 1/3 x (1R + 1W) = 126 MB | 0.32 ms (estimate) | ring | T x T solves, history GEMMs | high |
| Re-run the full forward for accepted tokens | + a full weight pass | >= 14.6 ms | | | not viable |

Arithmetic and notes:
- Ring for C and D: per layer per token per card `k` 8 x 128 + `v` 24 x 128 + `g` 24 + `beta` 24 = 4144 floats =
  16.6 KB. x 4 tokens x 48 layers = 3.2 MB.
- D compute: at most 4 replayed + 4 new token steps. Per card 8 x 24 heads x 48 layers x 115k FLOP = 1.06 GFLOP,
  so 45 us at the 23.7 TFLOPS f32 peak (4608 cores x 2 x 2.57 GHz). Spread over 48 layers and overlapped with the
  state stream (estimate).
- E: assumes a ring of 16 that flushes every about 3 steps (flush rule `h + 2T > L` from the ReplaySSM post). The
  history-ring terms change the summation order, so outputs will not match llama.cpp as closely as A-D.
- Conv state for C and D: keep the last 3 + T raw inputs per channel (6.9 MB per card) and pick the window by the
  accepted count, as vLLM does. No compute.
- D needs one extra rule: before saving a prompt-cache entry or a checkpoint at the end of a request, fold the pending
  accepted tokens into the state (one small kernel, about 0.39 ms).
- Recommendation: **D**, with C as a simpler first version. Saving vs today's llama.cpp: about 0.97 ms per step (estimate).

---

## 5. Chunked prefill

### 5.1 FLA `chunk_gated_delta_rule` (refs/flash-linear-attention)

- Driver `chunk_gated_delta_rule_fwd` (`fla/ops/gated_delta_rule/chunk.py:33-120`), default **chunk size 64**.
  Allowed sizes 16, 32, 64 (`chunk_fwd.py:369`). g is cumsummed per chunk and scaled by 1/ln 2 so kernels use `exp2`
  (`chunk.py:57-66`).
- Step 1, intra-chunk (`chunk_fwd.py:39-345`): one fused kernel computes the 10 lower-triangular 16x16 blocks of
  `beta * K K^T * exp(G_i - G_j)` (BC = 16 sub-blocks of the 64 chunk, line 383), inverts the 4 diagonal blocks by
  forward substitution, merges the blocks to get `T = (I + A)^-1`, and stores T. Block merges use TF32 dots when
  available (lines 20-23).
- Step 2, WY representation (`wy_fast.py:88-106`): `u = T (beta * v)`, `w = T (beta * e^G * k)`.
- Step 3, state pass over chunks (`fla/ops/common/chunk_delta_h.py:59-350`): for each chunk `v_new = u - w S`,
  `S = e^{G_last} S + (k * e^{G_last - G})^T v_new`. The state is held in f32 registers, but it is cast to bf16 before
  the `w S` product (line 216). The state after every chunk is stored in `k.dtype` (bf16) (line 722). The final state is
  f32 (line 723). On Blackwell this kernel is pinned to `num_warps = 2` because of a Triton race (lines 27-33).
- Step 4, output (`chunk_o.py`): `o = (e^G q) S + ((q k^T) * mask * e^{G_i - G_j}) v_new`.
- In the paper's notation: `T = [I + strictLower(diag(beta) K K^T)]^-1 diag(beta)`, `W = T K`, `U = T V`, state update
  with gated `U~` and `W` (arXiv 2412.06464, eqs. 6-7 and the gated chunk update).

### 5.2 FLOPs per token

Per V head per token, d = 128, C = 64 (FLA form, full squares counted):

| Term | FLOPs |
|---|---|
| `K K^T` (Gram; can be shared by the 3 V heads of a K head) | 2 C d = 16,384 |
| Triangular inverse | about C^2 / 3 = 1,365 |
| `W = T (beta e^G K)` | 2 C d = 16,384 |
| `U = T (beta V)` | 16,384 |
| `W S` | 2 d^2 = 32,768 |
| `K^T v_new` (state update) | 32,768 |
| `Q S` | 32,768 |
| `Q K^T` | 16,384 |
| `(Q K^T) v_new` | 16,384 |
| **Total** | **about 181,600 = 0.18 MFLOP** |

- All layers: 0.1816 M x 48 V heads x 48 layers = **418 MFLOP per token** (209 MFLOP per card).
- Recurrent form for comparison: decay + `S^T k` + rank-1 update + `S^T q` = 16,384 + 3 x 32,768 = 114,688 FLOP per
  head-token, 264 MFLOP per token, on f32 CUDA cores.
- Dense weight matmuls: about 2 x 27e9 = 54 GFLOP per token (estimate). The GDN core is about 0.8% of prefill FLOPs.
- llama.cpp's C = 16 form: about 115k useful FLOP per head-token, run 3 times for the hi/lo split, so about 344k tensor
  FLOP. Per card: 24 x 48 x 344k = 396 MFLOP per token.

### 5.3 Mapping to `mma.sync` on sm_120

- sm_120 has no `wgmma` and no tensor memory (`tcgen05`). Tensor cores are used through warp-level `mma.sync`.
  TMA bulk copies are available (FlashInfer's sm120 kernel uses a TMA store, `delta_rule_sm120.py:11`).
  Max 99 KB shared memory per block, 128 KB per SM (Blackwell tuning guide).
- Instruction: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `ldmatrix` (`.trans` for the transposed
  operand) from swizzled shared memory. llama.cpp's kernel and FlashInfer's sm120 kernel both use it
  (`gated-delta-net-mma.cu:66-68`; FlashInfer `delta_rule_sm120.py:787`, `MmaF16BF16Op (16, 8, 16)`).
- GEMM shapes per chunk (C = 64, d = 128): Gram and `Q K^T`: M = N = 64, K = 128. `W S`, `Q S`: M = 64, N = V slice,
  K = 128. `K^T v_new`: M = 128, N = V slice, K = 64. `T x`, `P v_new`: M = 64, N = V slice, K = 64.
- The state (64 KiB f32 per head) stays in accumulator registers: 64 floats per thread with 256 threads per head, or
  16 per thread with a 32-column V slice.
- Precision choices: plain bf16 operands (FLA, vLLM, FlashInfer; FlashInfer even keeps the inverse in fp16,
  `delta_rule_sm120.py:155`), TF32 `m16n8k8`, or llama.cpp's bf16 hi/lo split. Matching the production f32 AR path
  favors the hi/lo split.
- FlashInfer's sm120 design: one CTA per (sequence, head), 384 threads in 3 warp groups (load/store, KK math, QK math),
  BLK_Q = BLK_KV = 64, D = 128, f32 state, optional state checkpoints every N tokens (N a multiple of 64)
  (`delta_rule_sm120.py:56-60, 128-162, 687-718`; `gdn_prefill.py:583, 807-811`).

### 5.4 Porting effort and expected gain

| Route | Language today | Effort (estimate) | Notes |
|---|---|---|---|
| Port llama.cpp `gdn_single` | CUDA C++, MIT, 436 lines + parts of `mma.cuh` | 1-2 weeks with tests | Already runs on cc 1200. Needs the gate relaxed for 8/24 heads, the 32-column slice path (96 CTAs per card), checkpoint output, no K > 1. |
| Port FLA chunk pipeline | Triton, MIT, 4-5 kernels | 3-5 weeks | Hand-written `mma.sync` fragments; bf16 state casts would need changing to match f32. |
| Port FlashInfer sm120 kernel | CuTe DSL (Python), Apache-2.0 | 4-8 weeks | Highest ceiling. Warp-specialized, TMA. Not C++ today. |

Expected gain (estimate):
- The RTX 5090 data point in 2.2 implies the AR kernel costs at least 27 ms per 1024 tokens for the whole 27B model.
  The AR kernel is latency bound: each token step waits for global loads of q, k, v, g, beta and two shuffle reductions.
- On a 5060 Ti card with 24 heads: 768 CTAs x 4 warps = 3072 warps, 36 SMs x 48 warps = 1728 resident, so 2 waves.
  At about 1000 cycles per token step: 2 x 1024 x 1000 / 2.57 GHz = 0.8 ms per layer, 38 ms per 1024 tokens for 48 layers.
- A chunked kernel at 30-50% of tensor peak would take about 5-17 ms per 1024 tokens. The bf16 peak with f32
  accumulate on GB206 is not known to me (between 47 and 95 TFLOPS dense).
- Saving about 20-30 ms per 1024 tokens against about 1535 ms per 1024 tokens at 30k context (667 t/s). That is 1-2%.
  The share is larger at short context. This must be measured with nsys before spending weeks on it.

---

## 6. Prompt-cache checkpoints

### 6.1 Bytes per checkpoint

| Part | Total | Per card (TP split) |
|---|---|---|
| Recurrent state, 48 layers, f32 | 144 MiB | 72 MiB |
| Conv state, 48 layers, f32 | 5.625 MiB | 2.8125 MiB |
| **Checkpoint** | **149.625 MiB (156.9 MB)** | **74.8 MiB (78.4 MB)** |

Storing checkpoints in bf16 would halve this. The restored state would then differ from the state of a continuous run,
so it is not recommended.

### 6.2 What llama.cpp stores today

- Context checkpoints save the target with `PARTIAL_ONLY` (only the recurrent part, `llama-memory-hybrid.cpp:190-195`)
  and the MTP draft context with `PARTIAL_ONLY` too (`server-context.cpp:2479-2483`).
- The draft context uses a plain `llama_kv_cache`, whose `state_write` ignores the flags (`llama-kv-cache.cpp:2065-2071`).
  So **every checkpoint also stores the whole MTP draft KV**. This matches the log fit: 4.0234 KiB per token on top of
  149.63 MiB. The MTP KV is 4 KiB per token; the extra 24 B per token is probably cell metadata (not verified).
- Examples from the log: 221.3 MiB at 18,254 tokens, 661.5 MiB at 130,287 tokens. In the new engine the MTP KV can be
  truncated by position like the target KV, so a checkpoint is a fixed 149.6 MiB.

Where llama.cpp places checkpoints (`server-context.cpp:3620-3807`, defaults `common/common.h:633-635`):
- Only for completion tasks, and only before decoding the batch (so the checkpoint is the state after the previous batch).
- At the start of a user message if it is the last user message, or the first checkpoint, or more than
  `checkpoint_min_step = 8192` tokens after the previous one (lines 3722-3730).
- At `prompt_end - (4 + n_ubatch)` and `prompt_end - 4` (lines 3737-3750). The batch is broken at these positions.
- Not after image chunks. At most `n_ctx_checkpoints` (production: 4) per slot. The oldest is erased first.

Other engines:
- vLLM "align" mode caches the state "of the last token of each scheduler step and when the token is at position
  i * block_size" (`vllm/config/cache.py:188-195`).
- FlashInfer's sm120 prefill kernel can write state checkpoints every N tokens (N a multiple of 64) while it runs
  (`gdn_prefill.py:807-811`).

### 6.3 Where to take them in the new engine

The engine serves agent harnesses. The next request is usually the previous prompt + the previous reply (+ tool output).
- **Keep the live state at the end of generation.** It covers the reply. It helps only if the harness re-renders the
  reply with exactly the generated tokens. The template trims reasoning content and rebuilds
  `<think>\n...\n</think>\n\n` (`_chat_template.jinja:111-117`), so whitespace differences are possible.
- **Checkpoint at the end of the prompt, before the generation prompt** (before `<|im_start|>assistant\n<think>\n`).
  This is the point where the next request diverges when the re-rendered reply differs. It is llama.cpp's
  `prompt_end - 4` checkpoint.
- **Checkpoint at message starts in long prompts, at least 8k tokens apart.** This covers harness edits and history
  compaction.
- Place each checkpoint on a ubatch boundary (split the ubatch there), or have the chunked kernel write the state at
  that chunk boundary. With FLA-style chunks the state exists only at chunk boundaries.
- Note: the template puts the reasoning-effort sentence at the start of the system message (`_chat_template.jinja:46-60,
  80-85`). A request with a different `reasoning_effort` shares almost no prefix. No checkpoint can help there.

### 6.4 Save and restore cost

| Operation | Bytes per card | Time (estimate) |
|---|---|---|
| Save to a VRAM slot (D2D) | 78.4 MB read + 78.4 MB write | 156.9 MB / 388 GB/s = 0.40 ms |
| Save to host RAM (D2H, PCIe Gen3 x4 at about 3.5 GB/s) | 78.4 MB | 22.4 ms, both cards in parallel on their own links |
| Restore from host (H2D) | 78.4 MB | 22.4 ms |
| Restore from VRAM | 156.9 MB | 0.40 ms |
| Extra state write from a chunked prefill kernel at a chunk boundary | 78.4 MB | 0.20 ms |

- Break-even against recomputation: 22.4 ms of prefill at 667 t/s is about 15 tokens. Restoring a checkpoint pays off
  for any suffix longer than that.
- To hide the D2H time: copy the state to a VRAM staging slot (0.4 ms stall), then copy to pinned host memory on a
  second stream while prefill continues. With `-sm tensor`-style splitting the PCIe links also carry the all-reduce
  traffic (1.4-2.15 GB/s per card per direction during decode, `qwen38_27\LEEME.md:410-421`). The slowdown from sharing
  the link during prefill is unknown.
- Windows only: pinned host memory and async copies under WDDM can be slower or batched by the driver. Large pinned
  allocations may be limited. Measure on both OSes.
- Host RAM: 4 checkpoints x 149.6 MiB = 598 MiB. Today llama.cpp needs up to 4 x 661 MiB = 2.6 GiB at 130k tokens
  for the same 4 checkpoints. The KV cache dominates prompt-cache entries (64 KiB per token for the target).
- VRAM checkpoints (0.4 ms restore) fit only if VRAM is free. On Ubuntu headless there may be room for a few
  (75 MiB per card each). On Windows card 0 loses 1.3-1.9 GB to the desktop.

---

## 7. To measure (GPU session)

1. nsys of one llama.cpp verify step: time of the state `get_rows` copies, the GDN kernel, and the 4 conv `cpy` per layer.
   This confirms or rejects the 1.36 ms estimate in 2.4.
2. AR GDN kernel time per 1024-token ubatch on one 5060 Ti with 24 heads (the 38 ms estimate in 5.4).
3. Achieved bandwidth of a 96-CTA state-streaming kernel per card (read 1.5 MiB + write 1.5 MiB per layer).
4. Whether `cudaAccessPolicyWindow` L2 persistence works under WDDM and on Linux for this card.
5. bf16 `mma.sync` throughput with f32 accumulate on GB206.
6. Divergence between an f32 GDN implementation and llama.cpp on real prompts (hidden states after each GDN layer).

---

## Sources

Local clones (read only):
- `llama-rig2/ggml/src/ggml-cuda/gated_delta_net.cu:10-173, 186-190, 229-265, 331-353, 408-445`
- `llama-rig2/ggml/src/ggml-cuda/gated_delta_net.cuh:16`
- `llama-rig2/ggml/src/ggml-cuda/gated-delta-net-mma.cu:8-57, 59-121, 123-358, 380-412, 415-436`
- `llama-rig2/ggml/src/ggml-cuda/ggml-cuda.cu:2811-2873, 3725-3737`
- `llama-rig2/ggml/src/ggml-cuda/unary.cuh:113-115`
- `llama-rig2/ggml/src/ggml-backend-meta.cpp:951-966`
- `llama-rig2/src/models/qwen35.cpp:258-267, 350-483`
- `llama-rig2/src/models/models.h:14-18`
- `llama-rig2/src/models/delta-net-base.cpp:373-423, 449-533, 535-614`
- `llama-rig2/src/llama-graph.cpp:3520-3554`
- `llama-rig2/src/llama-memory-recurrent.cpp:101-103, 200-215, 455-457, 1364-1383`
- `llama-rig2/src/llama-memory-hybrid.cpp:190-195`
- `llama-rig2/src/llama-kv-cache.cpp:2065-2071`
- `llama-rig2/src/llama-hparams.cpp:208-257`
- `llama-rig2/src/llama-model.cpp:545-560, 614-668, 2741-2743`
- `llama-rig2/common/common.h:398-404, 633-635`; `common/common.cpp:1667`
- `llama-rig2/tools/server/server-context.cpp:2425-2490, 3193-3216, 3620-3807`
- `llama-rig2/conversion/qwen.py:396-403, 454-466`
- `qwen38_27/arranque.log.err` (checkpoint sizes, fit done in this session); `qwen38_27/LEEME.md:399-421`;
  `qwen38_27/vram-bw/vram-bw.cu:66`
- `qwen27-engine/research/_chat_template.jinja:46-60, 80-85, 111-117`
- `refs/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py:33-120`, `chunk_fwd.py:20-23, 39-345, 348-428`,
  `wy_fast.py:88-106`, `naive.py:13-61`; `fla/ops/common/chunk_delta_h.py:27-33, 212-316, 688-748`
- `refs/vllm/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:27-176`
- `refs/vllm/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu:85-95, 150-385`
- `refs/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:120-129, 280-301`
- `refs/vllm/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:860-876`
- `refs/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:90, 372-560`
- `refs/vllm/vllm/v1/attention/backends/gdn_attn.py:366-368`
- `refs/vllm/vllm/model_executor/models/config.py:806-831`; `vllm/config/cache.py:180-195`
- `refs/sglang/python/sglang/srt/mem_cache/memory_pool.py:851-880`
- `refs/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:1421-1520`
- `refs/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py:1355-1392`
- `refs/sglang/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py:1-40`,
  `gdn_replayssm_spec_fold.py:1-12`
- `refs/flashinfer/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py:56-60, 122-162, 687-718, 787-788`
- `refs/flashinfer/flashinfer/gdn_prefill.py:583, 807-811`
- `refs/flashinfer/flashinfer/gdn_kernels/gdn_decode_mtp.py` (module doc),
  `gdn_decode_bf16_wy_ucache.py` and `gdn_decode_bf16_wy_ucache_flush.py` (module docs)

Web:
- HF transformers `modeling_qwen3_5.py` (main, fetched 2026-10-05), lines 208-224, 240-287, 291-487, 494-653, 831-845:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Qwen3.8-27B config (`mamba_ssm_dtype: float32`, `output_gate_type: swish`): https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- Qwen3.5-27B config (`mamba_ssm_dtype: float32`): https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json
- Gated Delta Networks paper, eq. 10 and chunkwise WY form: https://arxiv.org/abs/2412.06464 (HTML: https://arxiv.org/html/2412.06464)
- llama.cpp PR #29353 (GDN chunked kernel, benchmarks, K=1 limit): https://github.com/ggml-org/llama.cpp/pull/29353
- ReplaySSM post (Dao AI Lab): https://dao-lab.ai/blog/2026/replayssm
- Blackwell tuning guide (cc 12.0: 128 KB shared memory per SM, 99 KB per block; L2 persistence):
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- sm_120 has no wgmma and no tensor memory: https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html and
  https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/
- RTX 5060 Ti / GB206: 36 SMs, 4608 cores, 32 MB L2, 2.57 GHz boost:
  https://videocardz.com/newz/nvidia-geforce-rtx-5060-ti-final-specs-confirmed-gb206-gpu-16-8gb-gddr7-and-2-57-ghz-boost
