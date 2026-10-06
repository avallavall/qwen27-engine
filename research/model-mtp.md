# Forward pass and MTP loop of Qwen3.8-27B (GGUF arch `qwen35`) as llama.cpp runs it

Reference build: worktree `llama-rig2`, branch `rig/full`
(HEAD `e2377cc96`). All `file:line` references below are in that worktree unless a
different root is given. "Production" means the flags in `qwen38_27\arranca.ps1`
(`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-sampling probabilistic`,
`-sm tensor`, f16 KV, `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0`). The env var
`LLAMA_SPEC_CHAIN` is not set in production, so the chained-draft graph is off.

## Summary

1. **Norms.** Every RMSNorm weight in the GGUF is stored as `1 + w` (the converter adds 1).
   The only exception is the GDN gated norm `ssm_norm`, which is stored as plain `w`.
   llama.cpp computes `x / sqrt(mean(x^2) + 1e-6) * w_gguf` everywhere. Verified on the
   real file: norm weights have mean 0.54 to 2.25, `ssm_norm` mean 0.87 to 0.96.
2. **RoPE** is `IMROPE` (interleaved M-RoPE), 64 of 256 dims, NeoX pairing (i, i+32),
   theta_j = pos * 1e7^(-j/32). For text tokens all three position components are equal,
   so it is plain NeoX partial RoPE. Attention: q/k RMSNorm per head, then RoPE, softmax
   scale 1/16, output multiplied by `sigmoid(gate)`. GQA: q head h reads kv head h/6.
3. **GDN**: conv1d (4 taps, no bias) + SiLU, L2 norm of q and k (eps 1e-6 inside the sqrt),
   `g = ssm_a * softplus(alpha + dt_bias)` with `ssm_a = -exp(A_log)` already in the file,
   `beta = sigmoid(b)`, delta rule with q scaled by 1/sqrt(128). **V head h uses K head
   h mod 16** (the converter reordered V heads to "tiled" order). Gated norm:
   `RMSNorm(o) * ssm_norm * silu(z)` per 128-dim head.
4. **MTP input**: token embedding of the token at position p, and the target's hidden state
   at position p-1 **after** `output_norm` (the same vector the LM head reads).
   Order: `eh_proj @ concat(enorm(emb), hnorm(h))`, embedding first. Next draft step uses the
   MTP's own `shared_head_norm` output as h. LM head is the shared `output.weight`.
5. **One decode step in production = 5 GPU passes**: target verify (4 tokens: last sampled +
   3 drafts), MTP "catch-up" over the same 4 tokens, then 3 sequential single-token MTP
   draft passes. Every pass ends in a host sync and CPU sampling over the full 248,320 vocab
   (backend sampling is refused under `-sm tensor`).
6. **Rejection rule** (`common/sampling.cpp:732-836`): accept draft x if `p(x) >= q(x)` or
   `U < p(x)/q(x)`. On reject, sample from `max(0, p - q)` renormalized and stop. p is the
   target distribution after top-k 20, top-p 0.95, temp; q is the draft top-k 10 at the same temp.
7. **Rollback**: no checkpoints and no re-run. Each GDN layer keeps 4 state planes
   (n_rs_seq = n_max = 3). The kernel writes the state after each of the 4 verify tokens.
   A rejection only sets an index that picks the right plane on the next pass.
8. **Counts and bytes (estimates)**: about 4,400 graph nodes and about 2,100 kernel launches
   per GPU per step. Bytes read per step: 14,203 MiB of weights + about 1,060 MiB of
   recurrent state traffic + 80 KiB per context token of KV. A pure bandwidth model at
   390 GB/s with a perfect 2-way split gives 20.6 / 23.7 / 31.3 / 36.7 ms at 1k / 30k / 100k
   / 150k. Measured: 37.7 / 41.0 / 48.3 / 53.9 ms. The gap is a constant ~17 ms per step.

---

## 1. Target forward pass, op by op

### 1.0 Constants

| Item | Value | Source |
|---|---|---|
| hidden `n_embd` | 5120 | GGUF |
| layers | 64 (+1 MTP block `blk.64`) | GGUF |
| attention layers | il where (il+1) % 4 == 0: 3, 7, ..., 63 | `src/models/qwen35.cpp:18-24` |
| RMS eps | 1e-6 (stored 9.99999997e-07) | GGUF, `qwen35.cpp:6` |
| FFN | 17408, SwiGLU, no bias | GGUF |
| vocab | 248320 | GGUF |
| attention | 24 q heads, 4 kv heads, head dim 256 | GGUF |
| GDN | 16 K heads x 128, 48 V heads x 128, conv kernel 4 | GGUF |
| RoPE | `IMROPE`, n_rot 64, base 1e7, sections [11,11,10,0], no YaRN keys | `src/llama-model.cpp:3208-3212`, GGUF |
| Norm stored as | `1 + w` for all `*norm.weight` except `linear_attn.norm.weight` (= `ssm_norm`) | `conversion/qwen.py:402-403` |
| `ssm_a` stored as | `-exp(A_log)` | `conversion/qwen.py:396-397` |
| `ssm_dt.bias` | HF `dt_bias` | `conversion/qwen.py:398-399` |

Numeric check on the production GGUF (script in the session scratchpad, read with `gguf`):

| Tensor | mean | min | max | Reading |
|---|---|---|---|---|
| `blk.0.attn_norm` | 0.967 | 0.868 | 1.198 | 1+w |
| `blk.0.post_attention_norm` | 0.783 | 0.004 | 1.003 | 1+w |
| `blk.3.attn_q_norm` | 1.230 | 0.824 | 1.481 | 1+w |
| `output_norm` | 1.944 | 0.715 | 2.711 | 1+w |
| `blk.64.nextn.enorm` | 0.539 | 0.250 | 0.815 | 1+w |
| `blk.64.nextn.shared_head_norm` | 2.252 | 0.775 | 2.930 | 1+w |
| `blk.0.ssm_norm` | 0.869 | 0.785 | 0.930 | plain w |
| `blk.0.ssm_a` | -0.048 | -0.338 | -0.004 | -exp(A_log), all negative |
| `blk.0.ssm_dt.bias` | 2.79 | -5.72 | 19.25 | raw dt_bias |

HF reference: `Qwen3_5RMSNorm.forward` does `output * (1.0 + self.weight)`; the gated norm
does `self.weight * x_normed * silu(gate)` with plain weight
(https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
So `x * w_gguf` in llama.cpp equals `x * (1 + w_hf)` in HF.

### 1.1 Layer skeleton (both layer types)

`src/models/qwen35.cpp:160-208`. For one token, x is [5120] f32:

```
h   = RMSNorm(x) * attn_norm.w                  # eps 1e-6
m   = GDN(h)  or  Attention(h)                  # [5120]
x1  = x + m
h2  = RMSNorm(x1) * post_attention_norm.w
x2  = x1 + W_down( silu(W_gate h2) * (W_up h2) )
```

This matches the HF decoder layer residual order (input_layernorm, mixer, add,
post_attention_layernorm, mlp, add). FFN: `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)`
gives `swiglu_split(gate, up) = silu(gate) * up` (`src/llama-graph.cpp:1770-1969`).
No biases. `build_cvec` (control vector) is a no-op without `--control-vector`.

Embedding: `get_rows(token_embd)` (IQ2_S). No embedding scale (`src/llama-graph.cpp:2384-2470`).
llama.cpp keeps `token_embd` in host RAM (LEEME: "solo la tabla de tokens (388 MiB) se queda
en RAM"), so this lookup runs on the CPU.

### 1.2 GDN block (48 layers), one token, shapes per token

`src/models/qwen35.cpp:350-483`, `src/models/delta-net-base.cpp`, CUDA kernel
`ggml/src/ggml-cuda/gated_delta_net.cu:10-173`.

| # | Op | Shape / formula | Source |
|---|---|---|---|
| 1 | `qkv = W_qkv h` (`attn_qkv`) | [5120] -> [10240] = q 16x128 \| k 16x128 \| v 48x128 | `qwen35.cpp:248` |
| 2 | `z = W_gate h` (`attn_gate`) | [5120] -> [6144] = 48x128 | `qwen35.cpp:252` |
| 3 | `b = W_beta h` (BF16) | [5120] -> [48]; `beta = sigmoid(b)` | `qwen35.cpp:373-378` |
| 4 | `a = W_alpha h` (BF16) | [5120] -> [48]; `g = ssm_a * softplus(a + dt_bias)` | `qwen35.cpp:380-388` |
| 5 | conv window | X[c] = [state[c][0], state[c][1], state[c][2], qkv[c]] for c in 0..10239 | `delta-net-base.cpp:449-473` |
| 6 | conv + SiLU | `y[c] = silu( sum_{j=0..3} w[c][j] * X[c][j] )`, no bias; new state = X[c][1..3] | `qwen35.cpp:406-410`, `ssm-conv.cu:39-55` |
| 7 | split | q = y[0:2048] (16x128), k = y[2048:4096] (16x128), v = y[4096:10240] (48x128) | `qwen35.cpp:419-435` |
| 8 | L2 norm q, k | per 128-dim head: `x / sqrt(sum x^2 + 1e-6)` | `models.h:14-18` (`rms_norm(x, eps/n) * 1/sqrt(n)`) |
| 9 | delta rule | per V head h (0..47), K head `kh = h mod 16`, see below | `gated_delta_net.cu:43, 90-117` |
| 10 | gated norm | per V head: `RMSNorm_128(o_h) * ssm_norm.w * silu(z_h)` (eps 1e-6) | `qwen35.cpp:258-267, 466-469` |
| 11 | `out = W_ssm_out y` | [6144] -> [5120] | `qwen35.cpp:476` |

Delta rule for one token, per V head h, state S_h is 128x128 f32, S[i][j] with i = key dim,
j = value dim:

```
d     = exp(g_h)                       # g_h <= 0
kv    = S^T k_kh                       # [128], computed on the old S
delta = beta_h * (v_h - d * kv)
S     = d * S + k_kh delta^T
o_h   = (S^T q_kh) / sqrt(128)
```

This equals the HF `torch_recurrent_gated_delta_rule` (decay first, then
`kv_mem = (S*k).sum`, `delta = (v - kv_mem)*beta`, `S += k delta^T`, `o = (S*q).sum`, q scaled
by 1/sqrt(d_k)). The CUDA kernel applies the 1/sqrt(128) to the output instead of q
(`gated_delta_net.cu:116`, scale set at `:327`). Same math.

GQA mapping in GDN. HF uses `repeat_interleave`, so HF V head j reads K head j // 3. The
converter permutes all V-head-indexed tensors (`in_proj_qkv` V rows, `in_proj_z`,
`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, conv1d V channels, `out_proj` columns) into
"tiled" order (`conversion/qwen.py:454-635`). In GGUF order, V head h reads K head
`h mod 16`. The kernel computes `iq1 = h_idx % neqk1` with neqk1 = 16
(`gated_delta_net.cu:43`, `:292`). The fused op never materializes the repeat
(`qwen35.cpp:453`, `cparams.fused_gdn_ar/ch` default true, `src/llama-context.cpp:234-235`).

Memory layouts the new engine may need if it wants to read llama.cpp state files:
- conv state per layer: [3, 10240] f32, the 3 taps of one channel are contiguous, oldest first
  (`delta-net-base.cpp:466`). `ssm_conv1d` weight is [4, 10240], taps contiguous per channel.
- SSM state per layer: [128, 128, 48] f32, stored transposed: row = value index j, contiguous
  over key index i (`gated_delta_net.cu:60`).
- Size: conv 120 KiB + SSM 3 MiB per layer, x48 = 149.6 MiB per plane. With n_rs_seq = 3
  there are 4 planes (`src/llama-memory-recurrent.cpp:101-104`) = 598 MiB of VRAM.

### 1.3 Attention block (16 layers), one token

`src/models/qwen35.cpp:269-348`. This GGUF has separate `attn_q`, `attn_k`, `attn_v`
(no fused qkv), so `build_qkv` takes the separate path (`src/llama-graph.cpp:1710-1755`).

| # | Op | Shape / formula |
|---|---|---|
| 1 | `qg = W_q h` | [5120] -> [12288]. Per head h: `qg[h*512 : h*512+256]` = q_h, `qg[h*512+256 : h*512+512]` = gate_h |
| 2 | `k = W_k h`, `v = W_v h` | [5120] -> [1024] each = 4 x 256 |
| 3 | q norm | per q head: `RMSNorm_256(q_h) * q_norm.w` (eps 1e-6, w stored 1+w) |
| 4 | k norm | per kv head: `RMSNorm_256(k_h) * k_norm.w` |
| 5 | RoPE on q and k | dims 0..63 only, see 1.4. Dims 64..255 unchanged. V has no norm and no RoPE |
| 6 | KV store | K, V written to the cache as f16 at the token's cell |
| 7 | attention | `o_h = softmax( (q_h . K_{h/6}) / 16 + causal mask ) V_{h/6}` over all cached positions <= current |
| 8 | output gate | `o = o * sigmoid(gate)` elementwise on [24 x 256] = [6144] |
| 9 | `out = W_o o` | [6144] -> [5120] |

`kq_scale = 1/sqrt(256)` because `f_attention_scale` is 0 (`qwen35.cpp:331`). HF does the same
split: `q_proj(...).view(..., -1, head_dim*2)` then `chunk(2)` gives per-head [q | gate].
FlashAttention is used (`-fa auto` resolves on CUDA). The kernel converts Q to f16 and
pre-scales it (`ggml/src/ggml-cuda/fattn-vec.cuh:221-226`), K/V are f16, accumulation is f32
(`src/llama-graph.cpp:2675`). No Hadamard KV rotation with f16 KV
(`src/llama-kv-cache.cpp:321-336`: only for quantized K/V).

### 1.4 RoPE details

- Mode `LLAMA_ROPE_TYPE_IMROPE` for `LLM_ARCH_QWEN35` (`src/llama-model.cpp:3208-3212`).
- CUDA kernel `rope_multi` (`ggml/src/ggml-cuda/rope.cu:199-287`): pair j (0..31) rotates
  elements (j, j+32), NeoX style. `theta_j = p_s * base^(-2j/64)`, base 1e7.
- Which position component p_s feeds pair j (`rope.cu:256-265`):
  j % 3 == 0 -> t (11 pairs), j % 3 == 1 -> h (11 pairs), j % 3 == 2 -> w (10 pairs).
  The 4th component is never used because 11+11+10 = 32 covers all pairs.
  vLLM uses the same channel rule (`refs/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:236-246`).
- Text tokens: llama.cpp sets t = h = w = pos and the 4th = 0 (`src/llama-graph.cpp:131-141`).
  So for text-only decode the op is plain partial NeoX RoPE with 64 rotary dims.
  Image tokens carry real (t, h, w) positions from mtmd.
- freq_scale 1, ext_factor 0, attn_factor 1 (no rope-scaling keys in the GGUF dump).
- Consequence: the CUDA fusions `rms_norm+mul+rope(+set_rows)` and `rope+set_rows` refuse
  IMROPE (`ggml/src/ggml-cuda/ggml-cuda.cu:2789-2793`, `:2748-2752`). llama.cpp runs q-norm,
  rope and the cache write as separate kernels. A text-only engine can fuse them.

### 1.5 Final norm and LM head

`qwen35.cpp:209-237`: `h_out = RMSNorm(x_64) * output_norm.w`, then
`logits = output.weight @ h_out` (Q4_K, [5120 -> 248320]). `h_out` is also exported as
`t_h_nextn`, the hidden fed to the MTP (`qwen35.cpp:213-214`). The target context has
`embeddings_nextn_masked = false` (`common/speculative.cpp:1495`), so `h_out` is computed
and copied to host for every token of the batch.

### 1.6 Numeric points that affect a logits match

- Quantized matmuls in decode use MMVQ. The f32 activation vector is first quantized to
  Q8_1 (32-value blocks), once per matmul call (`ggml/src/ggml-cuda/mmvq.cu:1531-1536`).
  Exact match with llama.cpp needs the same Q8_1 rounding. Without it, expect small drift.
- `ssm_alpha` / `ssm_beta` are BF16 and use the f32-activation path (no Q8_1).
- KV cache is f16. FA rounds Q to f16 after scaling.
- softplus: `x > 20 ? x : log(1 + exp(x))` (`ggml/src/ggml-cuda/unary.cuh:113-115`), same
  threshold as torch. SiLU `x/(1+exp(-x))`, sigmoid `1/(1+exp(-x))` (`unary.cuh:98-111`).
- All norms, GDN state and conv run in f32.

---

## 2. MTP draft path and the speculative loop

### 2.1 MTP block op by op (one draft row)

`src/models/qwen35.cpp:760-864` (non-chain graph, the one production runs).

| # | Op | Shape |
|---|---|---|
| 1 | `e = token_embd[x]` (shared table; `nextn.embed_tokens` absent in this GGUF) | [5120] |
| 2 | `e_n = RMSNorm(e) * enorm.w` | [5120] |
| 3 | `h_n = RMSNorm(h) * hnorm.w` | [5120] |
| 4 | `c = concat(e_n, h_n)` (embedding first) | [10240] |
| 5 | `x0 = eh_proj @ c` (Q6_K, [10240 -> 5120]) | [5120] |
| 6 | full-attention decoder layer, identical to 1.1 + 1.3 (gated q, q/k norm, IMROPE, sigmoid gate, SwiGLU FFN), weights `blk.64.*` (Q6_K), own KV cache | [5120] |
| 7 | `h_next = RMSNorm(x_out) * shared_head_norm.w` | [5120] |
| 8 | `logits = output.weight @ h_next` (shared Q4_K head; `shared_head_head` absent) | [248320] |

The order of the concat matches vLLM `Qwen3_5MTP.forward`
(`refs/vllm/vllm/model_executor/models/qwen3_5_mtp.py:161-165`: `cat([inputs_embeds, hidden_states])`).
vLLM also returns the post-final-norm hidden from the target
(`refs/vllm/vllm/model_executor/models/qwen3_5.py:730`) and the MTP output after `self.norm`
(`qwen3_5_mtp.py:188`). So both engines feed normalized hidden states into `hnorm`.

### 2.2 Which hidden and which token

MTP row at position p pairs token x_p with the hidden h_{p-1}
(`common/speculative.cpp:1412-1415`, `:1711-1729`).

- h_{p-1} for committed positions = the **target's** `output_norm` output at p-1.
- h for draft rows = the **MTP's** own `shared_head_norm` output of the previous draft row
  (`common/speculative.cpp:1986`, `:2042-2044`).
- After a verify that accepted n drafts, `pending_h = verify_h[n]`, the target hidden at the
  position of the last accepted token (`common/speculative.cpp:2071-2084`).

### 2.3 The MTP KV cache

- The draft context (`ctx_type = LLAMA_CONTEXT_TYPE_MTP`) has a plain KV cache that holds
  only layer 64 (`src/llama-model.cpp:2808-2819`). f16 (`-ctkd f16 -ctvd f16`), 4 KiB per
  token, n_ctx 180224 -> 704 MiB. Its own `n_rs_seq` is 0 (`common/common.cpp:1233`).
- Positions are the same absolute positions as the target. RoPE is the same IMROPE.
- During prefill, `process()` runs the MTP over every prompt ubatch, so the MTP KV covers
  the whole prompt (`tools/server/server-context.cpp:3913-3926`). Image chunks are skipped
  (`common/speculative.cpp:1630-1633`), so image positions are empty cells in the MTP KV.
- After drafting, the server removes MTP KV cells at positions >= pos0
  (`tools/server/server-context.cpp:3188`). The next catch-up pass writes those positions
  again, paired with target hiddens. After verify, `slot.mem.seq_rm(pos_next, -1)` trims both
  caches to the accepted prefix (`server-context.cpp:4192`).

### 2.4 One decode step, in order (production config)

Notation: target KV holds positions < P. `s` = last sampled token (position P, not yet in
any cache). d1..d3 = drafts from the previous step.

1. **Verify batch** (`server-context.cpp:513-550`): tokens `[s@P, d1@P+1, d2@P+2, d3@P+3]`,
   all with output flag. One ubatch, T = 4. Target forward pass as in section 1.
   Outputs: 4 logit rows and 4 `h_nextn` rows, copied to host.
2. **MTP catch-up** (`server-context.cpp:3919` -> `common/speculative.cpp:1704-1761`):
   the MTP runs on the same 4 tokens, no output rows, T = 4. Row k uses h_tgt[k-1]; row 0
   uses `pending_h`. This writes MTP KV at P..P+3. Then `verify_h` = the 4 target rows.
3. **Accept** (`server-context.cpp:4078-4192`): rejection sampling (2.6) on CPU. Result:
   n accepted drafts plus one new token t. `pending_h = h_tgt[P+n]`. Both caches are trimmed
   to positions <= P+n. The GDN rollback is an index change (2.7).
4. **Drafts** (`common/speculative.cpp:1784-2069`): pos0 = P+n+1, `id_last` = t.
   - pass 1: MTP row (t@pos0, h = pending_h), T = 1, full LM head, CPU sample -> d1'.
   - pass 2: row (d1'@pos0+1, h = MTP h of pass 1) -> d2'.
   - pass 3: row (d2'@pos0+2, h = MTP h of pass 2) -> d3'. The loop stops at n_max = 3.
   So yes: 3 sequential single-token passes, each with a host round trip.
5. Server removes MTP KV >= pos0 and goes back to step 1 with s = t, drafts d1'..d3'.

`--spec-draft-p-min` is 0 and `n_min` is 0 (`common/common.h:328-332`), so drafting never
stops early unless the context or `n_predict` budget is nearly full
(`server-context.cpp:491-510`).

Observed redundancy: draft pass 1 computes row (t@pos0, h[P+n]) and its KV. Step 5 deletes
it. Step 2 of the next round computes the same row again (the code marks this
`[TAG_SPEC_AVOID_DRAFT_REEVAL]`, `server-context.cpp:3915-3917`). The chain mode
(`LLAMA_SPEC_CHAIN`) merges the catch-up rows into the first draft decode, but it drafts
greedily and lost in measurements (`qwen38_27\LEEME.md`, "Lo que se probo y no sirve").

### 2.5 Draft sampling (q)

- With `probabilistic` and request temp > 0, the draft sampler is rebuilt per request as
  `top_k(10) -> temperature(T_req) -> dist` with seed `seed ^ 0x85ebca6b`
  (`common/speculative.cpp:35-70`, `:1928-1931`). The draft token is **sampled** from this
  distribution (`:1997`). q = the 10 candidate probabilities, renormalized over those 10
  (softmax of logit/T over the top 10, `src/llama-sampler.cpp:1150-1210`). The full q
  array is stored per draft position (`:2014-2016`).
- Backend sampling is refused under `-sm tensor`; the startup log shows
  `spec common_specu: backend offload failed for seq_id=0; using CPU sampler`
  (`qwen38_27\arranque.log.err`). So each draft pass copies 248,320 logits to the host.
- With temp 0 the server uses greedy match instead (`server-context.cpp:477-480`).

### 2.6 Verify: exact acceptance and resampling rule (PR #27694)

PR: https://github.com/ggml-org/llama.cpp/pull/27694 (merged 2 Oct 2026). Code:
`common/sampling.cpp:732-836`. For each draft position i = 0..2:

```
id_tgt = sample target chain at row i      # chain: top_k 20 -> top_p 0.95 -> min_p 0 (no-op) -> temp -> dist
P      = target candidates after the chain  # <= 20 tokens, p renormalized over them
q_x    = q_i(d_i)          (0 if d_i not in the draft's 10)
p_x    = P(d_i)            (0 if d_i not in P)
if q_x > 0 and (p_x >= q_x or U(0,1) < p_x / q_x):
    accept d_i; continue
r_k = max(0, P(k) - q_i(k)) for k in P      # tokens outside q keep all of P(k)
if sum r > 0: draw k with prob r_k / sum r  (u = U*sum, walk P in its sorted order)
else:         k = id_tgt
emit k; stop
if all 3 accepted: emit a 4th token sampled normally from row 3
```

Details that matter:
- p is the **truncated** target distribution (top-k 20, top-p 0.95). This is the same
  distribution plain sampling uses, so the output distribution is preserved.
- No RNG draw happens when `p_x >= q_x`. The uniform draws use a separate mt19937 seeded
  from the chain seed `^ 0x9e3779b9` (`common/sampling.cpp:125`, init near the end of
  `common_sampler_init`). The target chain's own `dist` RNG still advances at every row.
- With a lazy grammar active (tool calls), p is masked and renormalized before the test
  (`sampling.cpp:754-779`).
- n_rollback = 3 + 1 - (emitted tokens) (`server-context.cpp:4118`).

### 2.7 GDN state and conv state on rejection

- `n_rs_seq = draft.n_max = 3` for MTP (`common/common.h:398-404`, applied at
  `common/common.cpp:1667`). The recurrent memory allocates `1 + n_rs_seq = 4` planes per
  layer for both conv and SSM state.
- Verify pass (T = 4): the GDN kernel writes the state after token t into plane
  `T - 1 - t` (`gated_delta_net.cu:151-163`), so plane 0 = after d3, plane 3 = after s.
  The conv state is copied into 4 planes the same way (`delta-net-base.cpp:497-529`).
- `seq_rm(p0)` with rollback r = 1..3 sets `rs_idx = r` and moves the cell position back
  (`src/llama-memory-recurrent.cpp:200-214`). The next graph reads plane r through the
  `s_copy` index `r * size + src` (`llama-memory-recurrent.cpp:1368-1381`).
- The server uses checkpoints only when `draft.size() > n_rs_seq`
  (`server-context.cpp:3194-3199`, `:4118-4122`). With n_max 3 that never happens.
  So: no checkpoint, no re-run, per-position snapshots.
- The `-ctxcp 4` context checkpoints are for prompt reuse across requests. They are not
  part of the per-step loop.

---

## 3. Graph nodes and kernels per decode step (estimate)

Assumptions: production config, single sequence, T = 4 verify, fused GDN on, CUDA op fusion
on, FA on, IMROPE (so no rope fusions). Counts are per GPU; with `-sm tensor` each GPU runs
the same op list on its slice. Kernel counts include the Q8_1 quantize kernel that each
MMVQ matmul launches (`mmvq.cu:1531-1536`). Counted from the code, not from a trace.

### 3.1 Per layer, target verify (T = 4)

GDN layer, kernels after fusion:

| Op group | Kernels | Fusion used |
|---|---|---|
| attn_norm (rms_norm + mul) | 1 | `ggml-cuda.cu:4423` |
| `attn_qkv`, `attn_gate` matmuls | 2 + 2 | none (each quantizes its input again) |
| alpha + beta matmuls with add, softplus, mul, sigmoid | 1 | PR #29187 `ggml_cuda_try_gdn_ab_fusion` (`ggml-cuda.cu:3158-3245`), needs CUDA graphs |
| conv state `get_rows` copy | 1 | |
| concat (state + new) | 1 | |
| conv snapshot copies | 4 | (T=1 would be 2) |
| SSM state `get_rows` copy (3 MiB) | 1 | |
| ssm_conv + silu | 1 | `ggml-cuda.cu:4438` |
| L2 norm q, k (rms_norm + scale) | 2 | `ggml-cuda.cu:4428` |
| gated_delta_net + snapshot cpy | 1 | `ggml-cuda.cu:2811-2866` |
| gated norm (rms+mul, silu+mul) | 2 | `ggml-cuda.cu:4423`, `:4443` |
| `ssm_out` | 2 | |
| residual add, post norm | 2 | |
| FFN up, gate, swiglu, down | 2 + 2 + 1 + 2 | gate+up+GLU fusion only when ncols = 1 (`ggml-cuda.cu:1816`) |
| residual add | 1 | |
| **Total** | **~31** | |

Attention layer: norm 1, q/k/v matmuls 6, q norm 1, q rope 1, k norm 1, k rope 1,
k set_rows 1, v set_rows 1, gate `cont` 1, FA 1-2 (stream-k fixup or combine,
`fattn-common.cuh:731-924`), sigmoid*mul 1, `attn_output` 2, add 1, post norm 1, FFN 7,
add 1 = **~28-29 kernels**.

Graph nodes (including views, reshapes, permutes and zero-size nodes): about 74 per GDN
layer and about 42 per attention layer.

### 3.2 Per step

| Pass | Graph nodes | Kernels per GPU |
|---|---|---|
| Target verify, T=4 (48 GDN + 16 attn + head) | ~4,240 | ~1,940 (48x31 + 16x28 + 4) |
| MTP catch-up, T=4 (no LM head) | ~50 | ~35 |
| MTP draft x3, T=1 (gate+up fused, LM head) | 3 x ~50 | 3 x ~35 |
| **Step total** | **~4,440** | **~2,080** |

Extra work not in the table:
- `-sm tensor` all-reduce after every row-split output (`attn_output`/`ssm_out` and
  `ffn_down`, split rules `src/llama-model.cpp:519-575`): 2 per layer, so 128 per target
  pass and 2 per MTP pass, 136 per step. The meta backend falls back to a host-staged
  all-reduce when no backend all-reduce exists (`ggml/src/ggml-backend-meta.cpp:2616-2760`).
- 5 graph evaluations per step, each with input uploads (tokens, positions, KQ mask,
  `s_copy`, MTP h rows) and 4 of them with a blocking logits download plus CPU sampling.
- Whether CUDA graphs are active under the meta backend was not checked here. The GDN
  alpha/beta fusion only runs when they are (`ggml-cuda.cu:3168-3171`).

---

## 4. Bytes per op per decode step

Source: `research/_gguf-model-tensors.tsv`, summed by role (script in the session
scratchpad). One step = verify (T=4) + catch-up (T=4) + 3 drafts (T=1). Weights are read
once per pass (MMVQ handles up to 8 columns per weight read). N = tokens in context.

### 4.1 Weights

| Op | Per layer (MiB) | Layers x passes | Per step (MiB) | Types |
|---|---|---|---|---|
| GDN `attn_qkv` | 21.51 | 48 x 1 | 1,032.4 | IQ3_S, IQ3_XXS, IQ4_XS, ... |
| GDN `attn_gate` (z) | 13.25 | 48 x 1 | 635.9 | |
| GDN `ssm_alpha` + `ssm_beta` | 0.94 | 48 x 1 | 45.0 | BF16 |
| GDN `ssm_conv1d` | 0.16 | 48 x 1 | 7.5 | F32 |
| GDN `ssm_out` | 14.29 | 48 x 1 | 685.8 | |
| GDN FFN gate / up / down | 35.31 / 35.67 / 38.81 | 48 x 1 | 5,269.4 | |
| GDN norms, a, dt | 0.04 | 48 x 1 | 1.9 | F32 |
| Attn `attn_q` (q + gate) | 23.58 | 16 x 1 | 377.3 | |
| Attn `attn_k` / `attn_v` | 2.56 / 2.50 | 16 x 1 | 81.0 | |
| Attn `attn_output` | 13.93 | 16 x 1 | 222.9 | |
| Attn FFN gate / up / down | 36.57 / 36.65 / 38.39 | 16 x 1 | 1,785.7 | |
| Attn norms | 0.04 | 16 x 1 | 0.7 | |
| `output.weight` (target) | 682.03 | 1 x 1 | 682.0 | Q4_K |
| **Target subtotal** | | | **10,827.4** | |
| MTP block (`eh_proj` 41.0, q 49.2, k 4.1, v 4.1, o 24.6, FFN 3 x 69.7) | 332.3 | 1 x 4 | 1,329.2 | Q6_K |
| `output.weight` in drafts | 682.03 | 1 x 3 | 2,046.1 | Q4_K |
| **MTP subtotal** | | | **3,375.3** | |
| `token_embd` rows (8 lookups) | ~0 | | ~0 | IQ2_S, on CPU |
| **All weights per step** | | | **14,202.7** | |

### 4.2 State, KV cache and transfers

| Item | Per step | Arithmetic |
|---|---|---|
| SSM state traffic (GDN) | ~1,008 MiB | per layer: `get_rows` copy read 3 + write 3 MiB, kernel read 3 + write 4 snapshots x 3 MiB = 21 MiB; x48 |
| conv state traffic | ~50 MiB (estimate) | per layer about 1 MiB: copy, concat, 4 snapshot writes of 120 KiB |
| target KV read | 64 KiB x N | 16 layers x (K 2 KiB + V 2 KiB) per token, f16 |
| MTP KV read | 16 KiB x N | 1 layer x 4 KiB x 4 passes |
| KV writes | ~0.3 MiB | 4 tokens x 4 KiB x 16 layers + MTP rows |
| logits to host | 6.6 MiB total | verify 4 x 248320 x 4 B = 3.79 MiB, drafts 3 x 0.95 MiB; split between the 2 links |
| KQ masks to host->GPU | 22 B x N | f16 mask [n_kv, T] per pass, no row padding (`src/llama-graph.cpp:29-45`): (4 + 4 + 1 + 1 + 1) x 2 B x N; 2.1 MiB at 100k |

### 4.3 Bandwidth model against the measured step time (estimate)

Assumptions: both GPUs read exactly half of every item, 390 GB/s effective read bandwidth
(the measured copy figure), compute and launch gaps ignored.

| Depth | Bytes per step | Per GPU | Model ms | Measured ms | Gap ms |
|---|---|---|---|---|---|
| 1k | 15,344 MiB | 8.04 GB | 20.6 | 37.7 | 17.1 |
| 30k | 17,664 MiB | 9.26 GB | 23.7 | 41.0 | 17.3 |
| 100k | 23,264 MiB | 12.20 GB | 31.3 | 48.3 | 17.0 |
| 150k | 27,264 MiB | 14.29 GB | 36.7 | 53.9 | 17.2 |

The depth slope of the model (10.4 ms from 1k to 100k) matches the measured slope
(10.6 ms). So the KV reads explain the slowdown with depth, and the remaining ~17 ms per
step is constant overhead (launches, all-reduces, 5 host syncs, CPU sampling, PCIe
transfers). The MTP share of the weight bytes is 3,375 MiB, about 4.5 ms per step in this
model (the brief estimated ~4 ms).

---

## 5. What the new engine can drop or do differently

Graph features that do nothing for this model and config:

| Item | Where | Why it can go |
|---|---|---|
| LoRA adapters, control vectors, `cls_out`, `*_s` weight scales (NVFP4), biases | `llama-graph.cpp:1520-1557`, `qwen35.cpp:203, 223-231` | none in this GGUF |
| `f_clamp_kqv`, softcap, alibi, sinks, attention temperature, swiglu clamp | `llama-graph.cpp` | all zero/off |
| Hadamard K/V rotation | `llama-kv-cache.cpp:321-336` | only for quantized KV; this config is f16 |
| `output == NULL` fallback to `token_embd`, `nextn.embed_tokens`, `nextn.shared_head_head` | `qwen35.cpp:51-54, 118-120` | tensors present / absent as fixed facts |
| KDA path, non-fused GDN (autoregressive and chunked graph versions), q/k `repeat_4d` | `delta-net-base.cpp:16-371`, `qwen35.cpp:453-457` | one fused kernel with the `h mod 16` mapping covers decode |
| multi-sequence support (`n_seqs > 1`, streams, `s_copy_extra`, zero-state `scale_inplace`) | `llama-graph.cpp:3520-3555` | one slot |
| SSM/conv `get_rows` copy before the kernel | `llama-graph.cpp:3543` | read the right plane directly; saves 6 MiB per layer per step (288 MiB) |
| 4-plane snapshot writes of the 3 MiB state | `gated_delta_net.cu:151-163` | alternative: keep s_old, store the 4 tokens' (k, v, g, beta), and on the next pass replay the accepted ones before the new tokens. Write 1 state instead of 4. Saves ~430 MiB per step (estimate: 9 MiB x 48) |
| 4th M-RoPE component and section logic for text | `rope.cu:256-265` | for text, IMROPE = NeoX on 64 dims; enables norm+rope+KV-write fusion that llama.cpp refuses for IMROPE. Keep the (t,h,w) path for image tokens |
| `ggml_cont` of the attention gate | `qwen35.cpp:308` | read the gate strided inside the sigmoid-mul |
| separate Q8_1 quantize per matmul | `mmvq.cu:1531-1536` | quantize each activation once and share it (wqkv/z, q/k/v, up/gate) if the engine keeps Q8_1 activations |
| MTP catch-up as its own pass | `common/speculative.cpp:1704-1761` | merge the n+1 committed rows into draft pass 1 (T = n+2 rows). Saves one 332 MiB weight pass and one sync per step (~0.45 ms, estimate) |
| re-evaluation of draft row 0 | `server-context.cpp:3915-3917` | keep its KV entry; it is identical to catch-up row 0 |
| full-vocab logits download and CPU top-k | `common/sampling.cpp`, `src/llama-sampler.cpp` | do top-k 20 / top-p / temp and the rejection test on the GPU; download only candidates. Removes 6.6 MiB of PCIe per step and 4 CPU sampling passes |
| host-built KQ mask | `llama-graph.cpp:29-45` | causal mask from lengths on the GPU; removes 22 B x N of uploads per step |
| target `h_nextn` round trip through host | `common/speculative.cpp:1688-1702, 1772-1778` | keep the 4 hidden rows on the GPU; also removes 20 KiB per prompt token during prefill |
| MTP chain mode, chain-heads mode (step35), shared-memory mode (gemma4), sub-head `LLAMA_SPEC_CHAIN_SUB` | `qwen35.cpp:560-758`, `common/speculative.cpp:1404-1410` | not used by this model in production |
| draft `p_min` / `n_min` checks | `common/speculative.cpp:2000, 2065` | both 0 |
| penalties, DRY, XTC, typical, top-n-sigma, mirostat, min_p 0 | default sampler chain `common/common.h:263-273` | all neutral at the production values; keep temp, top_k, top_p, min_p per the brief |
| graph rebuild / scheduler / meta backend splitting | whole of `llama-graph.cpp`, `ggml-backend-meta.cpp` | fixed shapes: T in {1, 4} for decode, so the step can be a fixed set of captured graphs |

Things to keep even though they look optional:
- `1 + w` is already folded into the GGUF norm weights. Do not add 1 again.
- The tiled V-head order (`h mod 16`) is a property of the GGUF tensors.
- `ssm_a` already holds `-exp(A_log)`.
- Lazy-grammar masking inside rejection sampling, if a harness uses forced tool calls.
- Image-token positions (real t/h/w) for vision prompts.

---

## Sources

Local (worktree `llama-rig2`, branch `rig/full`):
- `src/models/qwen35.cpp:5-32, 34-129, 138-240, 242-267, 269-348, 350-483, 485-498, 501-864`
- `src/models/delta-net-base.cpp:289-371, 373-447, 449-533, 535-614`
- `src/models/models.h:14-18`
- `src/llama-graph.cpp:29-45, 127-147, 1520-1557, 1605-1636, 1655-1767, 1770-1969, 2384-2470, 2624-2763, 2876-2955, 3520-3555`
- `src/llama-model.cpp:519-575, 600-640, 2660-2730, 2808-2819, 3208-3212`
- `src/llama-memory-recurrent.cpp:101-104, 168-216, 1368-1381`
- `src/llama-context.cpp:105-110, 234-235, 562-567`
- `src/llama-kv-cache.cpp:321-336`
- `src/llama-sampler.cpp:265-317, 1150-1210, 1549-1600, 1749-1754, 2950-2955`
- `common/speculative.cpp:35-70, 1390-2085, 3117-3129, 3221-3256`
- `common/sampling.cpp:125, 191-420, 600-680, 686-726, 732-836`
- `common/common.h:263-273, 327-347, 398-404, 1028-1049`
- `common/common.cpp:1220-1240, 1522-1532, 1667`
- `tools/server/server-context.cpp:477-480, 491-550, 3100-3215, 3913-3926, 4078-4192`
- `conversion/qwen.py:278-350, 373-437, 454-635, 637-660`
- `ggml/src/ggml.c:6373-6425`
- `ggml/src/ggml-cuda/gated_delta_net.cu:10-173, 280-373`
- `ggml/src/ggml-cuda/ssm-conv.cu:5-56`
- `ggml/src/ggml-cuda/rope.cu:199-287`
- `ggml/src/ggml-cuda/unary.cuh:98-115`
- `ggml/src/ggml-cuda/mmvq.cu:1434-1565`, `mmvq.cuh:3`
- `ggml/src/ggml-cuda/fattn-vec.cuh:221-226`, `fattn-common.cuh:731-924`
- `ggml/src/ggml-cuda/ggml-cuda.cu:1798-1825, 2723-2752, 2775-2805, 2811-2866, 3158-3245, 3666-3735, 4120-4240, 4408-4447`

Other local files:
- `qwen38_27\arranca.ps1` (flags; API key lines not read into this report)
- `qwen38_27\LEEME.md` (build patches, CPU-sampling warning, chain-mode result)
- `qwen38_27\arranque.log.err` (startup warnings, acceptance stats)
- `qwen27-engine\research\_gguf-model-tensors.tsv`, `_gguf-model-dump.txt`, `_gguf-model-roles.txt`
- Production GGUF `qwen38_27\models\Qwen3.8-27B\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (norm, `ssm_a`, `dt_bias` values read with `gguf`)
- `refs/vllm/vllm/model_executor/models/qwen3_5_mtp.py:147-192`, `qwen3_5.py:730`
- `refs/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:236-246`

Web:
- HF transformers Qwen3.5 modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- llama.cpp PR #27694 (probabilistic drafting, rejection verify): https://github.com/ggml-org/llama.cpp/pull/27694
