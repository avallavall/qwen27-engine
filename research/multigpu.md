# Splitting Qwen3.8-27B over 2x RTX 5060 Ti on PCIe Gen3 x4, and cheap all-reduce

Research date: 2026-10-05. Read-only. No GPU work was done. All "estimate" values are derived numbers with the arithmetic shown.

## Summary

1. **llama.cpp `-sm tensor` today.** Every projection that writes to the residual stream is split on its input dimension (`attn_output`, `ssm_out`, `ffn_down`). Each of these leaves a partial sum on both cards, and each one costs one all-reduce. That is 2 per layer, 128 per forward pass. One decode step (verify 4 tokens, MTP catch-up of 4 rows, 3 MTP drafts) does **136 all-reduces**, 5.14 MiB per card per direction in BF16. One 1024-token prefill ubatch does **130 all-reduces of 10 MiB each**.
2. **How data moves.** Nothing goes card to card. Everything goes through pinned, mapped host memory. Decode uses one kernel per all-reduce: write own partial to host, set a flag, spin, read the peer's partial. Since PR #27173 the host does not wait. Prefill uses copy-engine D2H, then H2D, **on the same stream**, so the two link directions never work at the same time. Estimate: about 6 ms per prefill all-reduce, about 0.78 s per 1024-token ubatch. That is about half of today's prefill time at 30k. The Gen3 to Gen4 measurement in `LEEME.md` agrees with this number.
3. **Best layout: keep tensor parallel (TP=2).** Pipeline parallel halves decode bandwidth (measured: TP is +25% at 1k and +88% at 100k). Duplicating weights does not fit (arithmetic in section 2.1). A dual layout (TP for decode, PP for prefill) needs +2.8 GB per card and fits only at about 128k context.
4. **How to make the all-reduce cheap.** Decode: one CUDA graph per forward; a pipelined one-shot all-reduce through mapped host memory (per-chunk flags or Lamport sentinels); fused with residual add + RMSNorm; overlapped with the GEMV that produces it. Prefill: D2H and H2D on separate streams, plus two half-ubatches that alternate compute and transfer. Also: sample on the GPU (top-k per vocab half), so the logits never cross PCIe, and build masks on the GPU.
5. **Other implementations.** TensorRT-LLM, vLLM, SGLang and FlashInfer PCIe-IPC all need P2P. Without P2P they all fall back to NCCL, and NCCL then uses SHM, which is itself a GPU kernel polling mapped host memory (ring algorithm). A one-shot all-reduce through mapped host memory is feasible. Estimate on Gen3 x4: 15-24 us for 40 KiB, 50-74 us for 160 KiB, 6.0-8.4 ms for 20 MiB.
6. **P2P.** Windows WDDM: no (GeForce, no TCC, no known hack). Linux: the aikitoria patched open modules support RTX 50 and work on 4x 5060 Ti (Intel). They need ReBAR with a 16 GB BAR1, `iommu=pt` and ACS off. No AM5 + Blackwell success is reported. One AM5 + 5090 failure is reported. On this topology P2P moves the same bytes per link as host staging. It saves about 1-2 us per all-reduce, about 0.2-0.3 ms per decode step (estimate). Low priority.
7. **Cost (estimates, section 5).** Exposed communication per decode step: today 5.1-7.2 ms (1k to 150k); better all-reduce 2-3 ms; with overlap 0.4-1 ms. Per 1024-token prefill ubatch: today about 0.82 s; separate streams about 0.43 s; with overlap about 0.05 s.

---

## 1. How llama.cpp `-sm tensor` works today (`llama-rig2`, branch `rig/full`)

All paths below are relative to `llama-rig2`.

### 1.1 Components

| Layer | What it does | Where |
|---|---|---|
| Split rules | One function decides the split axis of every weight and cache tensor by name | `src/llama-model.cpp:377-889` |
| Meta backend | One "backend" that wraps the two CUDA backends. It propagates split states through the graph, cuts the graph after every PARTIAL node, and runs an all-reduce between the pieces | `ggml/src/ggml-backend-meta.cpp` |
| All-reduce provider | Linux default: NCCL. Other platforms: the internal pipeline. Env `GGML_CUDA_ALLREDUCE=nccl/internal/none` | `ggml/src/ggml-cuda/ggml-cuda.cu:1208-1245` |
| Internal all-reduce | 2 GPUs only. Kernel path for small tensors, copy-engine path for large | `ggml/src/ggml-cuda/allreduce.cu` |
| Scheduler | Puts graph inputs on the CPU backend and copies them into the meta backend. Pipeline parallelism is only for `-sm layer` | `ggml/src/ggml-backend.cpp:1084-1089`, `1833-1840`; `src/llama-context.cpp:431-436` |

### 1.2 Which tensors are split on which dimension

ggml weight shape is `[in, out]`. "Axis 1" splits the output rows (column-parallel). "Axis 0" splits the input (row-parallel).

| Tensor (shape in -> out) | Split | Per card | Where |
|---|---|---|---|
| GDN `attn_qkv` 5120 -> 10240 | axis 1, in 5 segments of 2048 (q, k, v, v, v) | q 1024 (8 K heads), k 1024, v 3072 (24 V heads) | `llama-model.cpp:527-529`, `649-652` |
| GDN `attn_gate` (z) 5120 -> 6144 | axis 1, 3 segments of 2048 | 3072 | `:546-548`, `653-654` |
| GDN `ssm_alpha`, `ssm_beta` 5120 -> 48 | axis 1, segments {16 x 3} | 24 | `:552-555`, `656-658` |
| GDN `ssm_dt`, `ssm_a` [48] | axis 0 | 24 | `:549-551` |
| GDN `ssm_conv1d` [4, 10240] | axis 1 | 5120 channels | `:563-565` |
| GDN conv state and SSM state | axis 0, by head | half | `:556-562`, `660-665` |
| GDN `ssm_out` 6144 -> 5120 | **axis 0 (input)** | 3072 -> full 5120, partial sum | `:566-568` |
| Attn `attn_q` 5120 -> 12288 (query + gate) | axis 1 | 6144 = 12 heads with gates | `:521-523`, `768-776` |
| Attn `attn_k`, `attn_v` 5120 -> 1024 | axis 1 | 512 = 2 of 4 KV heads | `:521-523`, `795-801` |
| `attn_q_norm`, `attn_k_norm` [256] | mirrored | full | `:533-535` |
| KV cache | axis 0, by KV head | 2 KV heads, 32 KiB/token | `:536-538` |
| Attn `attn_output` 6144 -> 5120 | **axis 0 (input)** | 3072 -> full 5120, partial sum | `:539-541` |
| `ffn_gate`, `ffn_up` 5120 -> 17408 | axis 1 | 8704 | `:571-573` |
| `ffn_down` 17408 -> 5120 | **axis 0 (input)** | 8704 -> full 5120, partial sum | `:580-582` |
| Norms, `nextn.eh_proj`, everything not listed | mirrored | full | `:610-611` |
| `output.weight` 5120 -> 248320 | axis 1 (vocab), or mirrored with `LLAMA_META_MIRROR_OUTPUT=1` | 124160 vocab rows | `:590-603` |
| `token_embd` | on the CPU (input layer) | none | `llama-model.cpp:1612` |

The meta backend turns these into all-reduces with two rules:
- matmul(weight axis 1, activation mirrored) gives an output split on the feature axis. No communication (`ggml-backend-meta.cpp:719-725`).
- matmul(weight axis 0, activation split on axis 0) gives a PARTIAL result (`ggml-backend-meta.cpp:729-731`).

The graph is cut after every PARTIAL node (`ggml-backend-meta.cpp:2426-2467`). The cut pieces run in order, with one all-reduce between each pair (`ggml-backend-meta.cpp:2735-2763`). There is a "delayed all-reduce" merge for independent partial branches (`ggml-backend-meta.cpp:2370-2424`). It does not apply to this dense model, because the FFN input depends on the reduced attention output.

In `src/models/qwen35.cpp` the PARTIAL nodes are `ssm_out` (line 476), `wo` (line 344) and the FFN down projection (lines 485-495). The MTP block uses the same tensor names, so it also has 2 all-reduces (`wo` at line 645, FFN after it). `nextn.eh_proj` is mirrored, so it has none.

### 1.3 Where and how many all-reduces

Production runs MTP with `--spec-draft-n-max 3` and without `LLAMA_SPEC_CHAIN`. In that mode the MTP catch-up is not deferred (`common/speculative.cpp:1420-1422`, `1503-1507`). So each decode step runs 5 graphs:

| Graph | Rows | All-reduces | Wire bytes per all-reduce, per card, per direction (BF16) | Where |
|---|---:|---:|---:|---|
| Target verify | 4 | 128 (2 x 64 layers) | 4 x 5120 x 2 = 40,960 | server decode |
| MTP catch-up (`process()`) | 4 | 2 | 40,960 | `speculative.cpp:1704-1761` (decode at 1746) |
| MTP draft 1, 2, 3 | 1 each | 2 each = 6 | 10,240 | `speculative.cpp:1950-2053` (decode at 1967) |
| **Total per step** | | **136** | 130 x 40,960 + 6 x 10,240 = **5,386,240 B (5.14 MiB)** | |

Cross-check: an independent count for vLLM with the same geometry (hidden 5120, 64 layers) gives 139 per verify step: 128 + 10 for a 5-layer drafter + 1 for a vocab-parallel embedding (HyperQwen issue #254).

Per 1024-token prefill ubatch: 128 (target) + 2 (MTP catch-up of the same tokens) = **130 all-reduces**. Each is 1024 x 5120 x 2 B = **10,485,760 B (10 MiB)** per card per direction. Total 1.36 GB per card per direction per ubatch.

Wire format: the tensors are F32, but `GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1, so every F32 all-reduce travels as BF16 (`allreduce.cu:452-455`, `788-800`). Both cards also round their own partial to BF16 before adding, so both get bit-identical results (`allreduce.cu:181-201`, `204-222`).

Path choice: wire size below 1 MiB uses the kernel path. 1 MiB or more uses the copy-engine path (`allreduce.cu:245-247`, `807-812`). 1 MiB is 102 tokens in BF16. So decode uses the kernel path and prefill uses the copy-engine path.

### 1.4 Decode path: the chunked kernel (`allreduce.cu:109-202`, launch at `951-972`)

- Buffers: per card, a 2-slot ring of 1 MiB in pinned host memory allocated with `cudaHostAllocPortable | cudaHostAllocMapped` and used through the device pointer (`allreduce.cu:275-302`, `521-533`). Arrival flags are ints in mapped host memory, one 64-byte line per block (`allreduce.cu:69-72`, `503-519`).
- Launch: 8 blocks x 256 threads on each card's compute stream, the same stream as the model kernels.
- Phase 1: each thread converts F32 to BF16 and stores 16-byte vectors into its own host buffer. Then `__threadfence_system()` (`allreduce.cu:134-152`).
- Phase 2: thread 0 of each block writes the call number into its arrival slot, fences, then spins with `__nanosleep(100)` until the peer's slot shows the same number (`allreduce.cu:154-176`).
- Phase 3: each thread reads the peer's 16-byte vectors from host memory, adds, and writes the result in place (`allreduce.cu:178-201`).
- Synchronization: no CUDA events and no host waits since PR #27173. Slot reuse is safe because of the in-kernel handshake (`allreduce.cu:381-397`). With `RIG_AR_SYNC=1` the old behaviour returns: `acquire_slot` calls `cudaEventSynchronize` on both cards for the all-reduce two calls back, so the host can run at most 2 all-reduces ahead (`allreduce.cu:366-379`, `928-932`, `977-979`).
- Cost structure: per block, all writes finish before the flag, and all reads start after the flag. The payload crosses each card's link twice in sequence (up, then down). Estimate for 40 KiB: 3 + 2 x 11.7 = **26.4 us** (L = 3 us, 3.5 GB/s; agent model in section 3.6).
- Launch structure: each piece between two all-reduces is its own CUDA graph, keyed by its first node (`ggml-cuda.cu:2647-2649`, `4700-4755`). The all-reduce kernel is a plain launch between two `cudaGraphLaunch` calls. Per verify, each card gets 129 graph launches and 128 kernel launches, all from one host thread.

### 1.5 Prefill path: the copy engine (`allreduce.cu:615-735`, `829-910`)

1. A conversion kernel writes F32 to a BF16 temp buffer (`allreduce.cu:835-851`).
2. `acquire_slot`: host `cudaEventSynchronize` on both cards for the all-reduce two calls back (`allreduce.cu:366-379`, called at `637`). This is a host wait on every large all-reduce.
3. Stage 1, per card, on its own all-reduce stream `p->streams[i]`: wait for compute (event), wait for the peer's "done reading my buffer" event (cross-device), then D2H in chunks of 2 MiB into its pinned buffer, one event per chunk (`allreduce.cu:643-674`; chunk size `clamp(n/4, 512 KiB, 2 MiB)`, `253-254`, `399-407`).
4. Stage 2, per card, **on the same stream** `p->streams[i]`: for each chunk, wait for the peer's D2H event (cross-device), then H2D into device scratch. Then an event hands over to the compute stream, which runs the add kernel (`allreduce.cu:683-730`).

Because stage 1 and stage 2 share one stream per card, each card does all its D2H first and all its H2D after. Each link is used in one direction at a time. Estimate: 2 x 10,485,760 B / 3.5 GB/s = **6.0 ms per all-reduce**. 130 x 6.0 ms = **0.78 s per 1024-token ubatch**.

Check against measurements (`qwen38_27\LEEME.md:423-425`, older build): Gen4 raised prompt speed at 100k from 417 to 503 t/s. That is 2.456 s -> 2.036 s per 1024-token ubatch, 0.42 s less. Halving 0.78 s predicts 0.39 s less. The two agree.

Share of today's prefill (estimate): at 30k, 1024 / 667 t/s = 1.535 s per ubatch, so 0.78 s is 51%. At 150k, 1024 / 439 = 2.33 s, so 33%.

### 1.6 Other PCIe traffic and host waits per step

| Item | Bytes | Sync | Where |
|---|---|---|---|
| Graph inputs (positions, KQ mask, state indices, embeddings) | KQ mask = n_kv x rows x 2 B, F16, to **both** cards. At 150k: 1.2 MB (verify) + 1.2 MB (catch-up) + 3 x 0.3 MB (drafts) = 3.3 MB per card per step. At 30k: 0.66 MB | Before the first user input the scheduler calls `ggml_backend_synchronize` on the meta backend, which syncs both cards (the meta backend has no events). Then each input is copied synchronously, card 0 then card 1 | `ggml-backend.cpp:1833-1840`; `ggml-backend-meta.cpp:1668-1672`, `2782-2783`; `ggml-cuda.cu:779-785`; `src/llama-graph.cpp:29-46` |
| Logits | Vocab split: each card sends its half of every output row. Verify 4 x 124,160 x 4 B = 1.99 MB per card; each draft 0.50 MB per card. About 3.5 MB per card per step | Host waits for them before CPU sampling | `ggml-backend-meta.cpp:2112-2134` |
| `h_nextn` (hidden state for MTP) | Mirrored, read from card 0 only. 20 KiB per row | same | `ggml-backend-meta.cpp:2136-2140` |
| Host waits | At least 4 per step: after verify and after each draft (CPU sampling) | Each one drains both cards | `speculative.cpp:1985`; log line "backend sampling not supported with SPLIT_MODE_TENSOR" (`LEEME.md`) |
| Prefill extras | Token embeddings come from CPU `get_rows`: 20 MiB per ubatch to both cards, for the target graph and again for the MTP graph. MTP hidden rows: 20 MiB to both cards. `h_nextn`: 20 MiB from card 0 | Synchronous copies | `llama-model.cpp:1612` |

Estimates: the mask copies cost about 0.4 ms per step at 30k and about 1.9 ms at 150k (6.6 MB of sequential synchronous copies at 3.5 GB/s). They grow with depth. This may explain part of why Gen4 helped decode at 100k and 150k but not at 1k and 30k (`LEEME.md:106-111`). It is a hypothesis. The logits cost about 1.0 ms per step (3.5 MB at 3.5 GB/s). The prefill extras cost about 40 ms per ubatch (60 MiB H2D per card plus 20 MiB D2H), about 2.6% at 30k.

**Open question.** `LEEME.md:416-421` reports 1.4-2.15 GB/s per direction per card during decode at 30k (`nvidia-smi dmon -s t`). My byte count gives about 9 MB D2H and 6 MB H2D per card per 41 ms step, about 0.15-0.22 GB/s. That is 7-10x less. Possible reasons: the NVML counter includes protocol overhead and the polling reads of the spin loop, or there is traffic I did not find. Measure it with Nsight Systems PCIe metrics before using "the link is half full" as a design input.

### 1.7 PR #27173 and `LLAMA_SCHED_POOL`

| Change | What it does | Measured here | Where |
|---|---|---|---|
| All-reduce without host sync | The kernel path stops calling `acquire_slot` (`cudaEventSynchronize` on both cards per all-reduce). The host can now queue a full token ahead | -0.6% step time | `allreduce.cu:381-397`, `928-932`; `LEEME.md:579`, `602` |
| `LLAMA_SCHED_POOL=N` | One scheduler, allocation and graph per recurring batch shape (up to 32 tokens, up to 16 slots). Verify and draft shapes stop evicting each other, so the graph, the meta plan and the CUDA graphs are reused | -1.7% step time, about +190 MiB on card 1 | `src/llama-context.cpp:731-746`, `1445-1496`; `src/llama-context.h:360-375`; `LEEME.md:61`, `603` |
| Meta plan cache | Per-graph-uid cache of the split decomposition (16 plans), checked with per-node fingerprints | included above | `ggml-backend-meta.cpp:2160-2209`, `2503-2524` |
| `LLAMA_META_MIRROR_OUTPUT=1` | Whole output head on both cards; logits mirrored; GPU sampling possible | +3 ms per step here (each card reads 682 MiB instead of 341 MiB per graph; 4 graphs with logits per step: 4 x 341 MiB / 390 GB/s = 3.7 ms, estimate) | `llama-model.cpp:590-603`; `LEEME.md:610` |
| `LLAMA_SPEC_CHAIN=1` | All drafts in one graph with in-graph argmax | Lower acceptance; slower overall here | `LEEME.md:611` |

Upstream numbers on 2x RTX 5090 (PR #27173 description): no-sync about +3 t/s, sched pool about +5 t/s, mirrored output +5-8 t/s, MTP chain +12.8%.

---

## 2. Options for this link

### 2.1 Memory check: duplicated weights do not fit

Inputs: GPU weights 11,548 - 388 (`token_embd`, on CPU) = 11,160 MiB. KV f16 at 180,224 tokens: 64 KiB x 180,224 = 11,264 MiB, so 5,632 MiB per card when split by head. MTP KV: 4 KiB x 180,224 = 704 MiB, 352 per card. Measured today: card 1 uses 15,242 of 16,311 MiB after a 150k prompt and a 4K image, 1,069 MiB free (`LEEME.md:82`).

| Layout | Per card | Fits at 180k? |
|---|---|---|
| Full replication, KV split | 11,160 + 5,632 = 16,792 MiB before any buffer | No |
| Full replication, KV replicated (data parallel) | 11,160 + 11,264 = 22,424 MiB | No |
| Dual layout: TP weights + the missing half of own PP layers | today + 2,790 MiB -> card 1 about 18,032 MiB | No. Fits at about 128k: remove 2,790 - 1,069 = 1,721 MiB of KV; 1,721 MiB / 34 KiB per token (32 target + 2 MTP) = 51.8k tokens less (estimate) |

The dual layout also needs a KV re-layout after prefill (PP stores 8 layers x 4 KV heads per card, TP needs 16 layers x 2 KV heads). That is 16 KiB per prompt token per direction, about 0.47 s per 100k tokens at 3.5 GB/s (estimate). It can overlap with prefill.

### 2.2 The options

| Option | Idea | Verdict |
|---|---|---|
| A. TP as now | llama.cpp today | Baseline |
| B. TP, better all-reduce | Section 2.3 and 2.4, no compute overlap | Do it |
| C. TP, all-reduce overlapped with compute | Decode: GEMV epilogue pushes finished output tiles. Prefill: two half-ubatches alternate | Do it after B |
| D. TP + P2P | Linux only, patched driver | Optional; small gain (section 4) |
| E. Fewer all-reduces per layer | Exact TP of a sequential block needs one sum before each RMSNorm: 2 per layer is the minimum. Parallel attention/FFN or "ladder residual" change the model and need retraining. Replicating the mixer weights (about 48 x 62 + 16 x 34 MB = 3.5 GB) to drop one all-reduce per layer adds about 1.76 GB of weight reads per card per step (+4.5 ms at 390 GB/s) to save about 2 ms | No |
| E2. Narrower wire format | 8-bit with block scales halves the bytes of BF16. ik_llama.cpp already offers `--graph-reduce-type` with `q8_0` (`refs/ik_llama.cpp/common/common.cpp:2495-2497`, `3342`; `ggml/src/ggml-cuda/reduce.cu:29`) | Only if quality checks pass. Helps prefill (bandwidth-bound) more than decode (latency-bound) |
| F. Pipeline parallel (layer split) | 32 layers per card | No for decode: each card reads its half in turn, so the step reads 10,828 MiB at one card's speed: 28.7 ms vs 14.4 ms for TP (estimate). Measured: `-sm layer` 43.4 vs 54.4 tok/s at 1k, 24.2 vs 45.6 at 100k (`LEEME.md:106-111`). Good for prefill (1 hand-off per ubatch) |
| G. Mix: TP decode + PP prefill | Two weight layouts | Does not fit at 180k (2.1) |
| H. Replication | | Does not fit (2.1) |

All 136 all-reduces per step are barriers. The slower card sets the pace at each one. On Windows card 0 drives the desktop and ran at 2,713 MHz average against 2,782 MHz on card 1 (`LEEME.md`, clocks table). This goes away on headless Ubuntu. (Windows-only.)

### 2.3 Decode design (options B and C)

1. **One CUDA graph per forward per card.** The all-reduce kernels go inside the graph. Cross-card waiting happens inside the kernels, so the graph does not need cross-device events. This removes 256 launches per card per verify.
2. **Pipelined one-shot all-reduce through mapped host memory.** Split the 40 KiB into many chunks. Each chunk gets its own flag after its data (`st.release.sys` / `ld.acquire.sys`), or uses Lamport sentinels. The reader starts on chunk i while the writer still writes chunk i+k. The write (upstream) and the read (downstream) then overlap on each link. Estimate: 14.7-24.4 us for 40 KiB instead of about 26 us (section 3.6).
3. **Fuse with residual add + RMSNorm** (as TensorRT-LLM `kARResidualRMSNorm`). The reduced vector is used at once by the next layer's input norm.
4. **Overlap with the producing GEMV (option C).** Per-card weight read time of the row-split GEMVs (estimate, 390 GB/s): `ffn_down` IQ3_S 38.3 MB / 2 = 49 us; `ffn_down` IQ2_S 28.5 MB / 2 = 37 us; `ssm_out` 16.7 MB / 2 = 21 us; `attn_output` 13.5 MB / 2 = 17 us (sizes from `research/_gguf-model-tensors.tsv`). The 40 KiB transfer takes 11.7 us, less than any of them. If each CTA pushes its finished output rows as soon as they are done, most of the transfer hides behind the GEMV. Exposed part: the last tile plus one latency, about 3-8 us (estimate). I found no published kernel that does this over host-staged PCIe.
5. **GPU-side sampling over the vocab split.** Each card computes its local top-20 (temp, top_k 20, then top_p / min_p) plus its local max and sum of exp. Merging two top-20 lists gives the exact global top-20. Only about 20 x 8 B + 8 B per row cross PCIe instead of 0.5 MB per row per card. This also allows exact probabilistic draft acceptance, because with top_k 20 both distributions live on at most 20 tokens. It removes about 1.0 ms of logits transfer per step and the host waits between drafts.
6. **Build inputs on the GPU.** For one causal sequence no mask tensor is needed. Positions and state indices are a few bytes.

### 2.4 Prefill design (options B and C)

1. **Separate D2H and H2D streams** (or one copy stream per direction), chunks of about 1 MiB. Each link then works in both directions at once. Estimate: 10,485,760 / 3.5 GB/s = 3.0 ms + about 0.3 ms chunk fill = **3.3 ms per all-reduce**, 130 x 3.3 = 0.43 s per ubatch.
2. **Two half-ubatches (2 x 512) that alternate (option C).** While the copy engines move the all-reduce of half A, the SMs compute half B. Compute per half-layer at 30k, estimate: (1.535 - 0.82) s / 128 / 2 = 2.8 ms. Transfer per half: 5 MiB, about 1.6-1.8 ms. So the transfer can hide completely (estimate). Cost: the weights are read twice per ubatch, about +14 ms per ubatch per card (5.4 GiB / 390 GB/s), about 1%.
3. Unknown: the number of async copy engines on GB206. Two are needed for D2H and H2D at the same time. Check `asyncEngineCount` with `deviceQuery`.
4. **Bandwidth limit of TP prefill on this link.** BF16 all-reduce bytes per token per card per direction: 128 x 5120 x 2 = 1.31 MB. With perfect overlap, the link allows 3.5 GB/s / 1.31 MB = **about 2,670 t/s** (estimate). vLLM on the same two cards reached 2,613 t/s at 32k with NVFP4 (`club-5060ti` issue #7, link speed not stated). So if prefill compute becomes much faster, the link becomes the limit for TP prefill on Gen3 x4. An 8-bit wire format would move this limit to about 5,300 t/s. Pipeline parallel has no such limit, but it does not fit together with TP decode (2.1).

---

## 3. All-reduce implementations to learn from

Details, file:line and URLs are from a sub-report (`scratchpad/allreduce-impls.md`), checked against the files it names.

### 3.1 Overview

| Implementation | Algorithm | Signalling | Needs P2P? | Where |
|---|---|---|---|---|
| TensorRT-LLM classic | one-shot (pull, or push with ping/pong buffers), two-shot (reduce-scatter + all-gather) | `st.release.sys` / `ld.acquire.sys` block barriers | Yes. `TLLM_CHECK(p2p_supported)`. Falls back to NCCL if P2P **or NVLink** is missing | `cpp/tensorrt_llm/kernels/customAllReduceKernels.cu:1347-1464`, `1466-1660`; `thop/allreduceOp.cpp:1442-1451` |
| TensorRT-LLM fused | one-shot Lamport (tokens <= 128), two-shot; fused residual + RMSNorm (+FP8/FP4 quant); FP32 accumulation | Lamport: buffer pre-filled with -0.0; producer rewrites real -0.0 to +0.0; consumer polls the data with `ld.volatile.v4`; triple buffer, the buffer from two calls back is reset locally | Yes (`allReduceWorkspace.cu:43-44`). Cluster launch is off on SM120 | `communicationKernels/allReduceFusionKernels.cu:447-525`, `527-593`, `655-666` |
| vLLM `CustomAllreduce` | 1-stage for world size 2 (each GPU reads the peer buffer and sums in rank order); 2-stage for larger | per-block counters in the peer's `Signal` struct, release/acquire | Yes. For exactly 2 GPUs plain PCIe is allowed, but only with working P2P. Max 8 MiB, chunked above | `refs/vllm/vllm/distributed/device_communicators/custom_all_reduce.py:121`, `277-291`, `137-141`; `csrc/custom_all_reduce.cuh:7-23`, `297-299` |
| vLLM without P2P | custom AR off; symm-mem off on sm_120; FlashInfer fusion off on sm_120 | | | falls back to PyNCCL (`cuda_communicator.py:355-420`) |
| SGLang legacy / V2 | same gates as vLLM; V2 (`1shot_push` Lamport with +0.0 sentinel) needs full NVLink | | Yes | `custom_all_reduce.py:260-274`; `custom_all_reduce_v2.py:482-497`; Lamport push kernel `kernels/jit/csrc/distributed/custom_all_reduce.cuh:136-209` |
| FlashInfer / SGLang `pcie_ipc` | staged pushes so each rank has one inbound and one outbound stream through the root complex; copy-engine variants | +0.0 sentinels | Yes (CUDA IPC device memory). Measured on 8x SM120 with P2P: 12 KiB 5.3 us vs NCCL 205 us | `sglang/.../pcie_ipc_ar.py:15-25`; FlashInfer `pcie_ipc_all_reduce.cuh:19-34` |
| NCCL, P2P off | ring over the SHM transport. The SHM transport is GPU kernels writing and polling mapped, pinned host memory. Small messages use the LL protocol: 16-byte line = data, flag, data, flag (50% payload) | flags in data | No | NCCL `src/transport.cc:15-18`, `src/transport/shm.cc:153-200`, `src/device/prims_ll.h:108-120` |
| exllamav3 TP backend | partials go to pinned shared host memory; a CPU helper sums (AVX2/AVX-512); a GPU-only kernel exists but is disabled | `st.release.sys` + `ld.acquire.sys` / `ld.cv` polling, `__nanosleep`, 2 s timeout | No. Runs on Windows too | exllamav3 `model/model_tp_backend.py:234-246`, `458-470`; `exllamav3_ext/parallel/ll.cuh:3-37` |
| llama.cpp internal | one-shot through mapped host memory, one flag per block after all data (kernel path); copy-engine path for large | volatile int tokens + `__threadfence_system` | No | section 1.4 |

### 3.2 What needs P2P

Every fast custom kernel in TensorRT-LLM, vLLM, SGLang and FlashInfer shares **device** memory over CUDA IPC (`cudaIpcOpenMemHandle`). That needs peer access. CUDA IPC is also Linux-only. Without P2P all of them use NCCL, and NCCL uses SHM. Only NCCL SHM, exllamav3 and llama.cpp work without P2P, and all three go through mapped host memory.

### 3.3 NCCL without P2P

- Transport order: P2P, SHM, NET. P2P is rejected by topology level (default PXB, raised to SYS on AMD x86 hosts with at most 2 GPUs), by NVML P2P status, or when `cudaDeviceCanAccessPeer` is 0 (NCCL `src/graph/paths.cc:303-313`, `382-395`, `400-436`; `src/transport/p2p.cc:173-184`).
- So on this box: stock driver -> SHM. Patched driver with working P2P -> NCCL should pick P2P with no env var (reading of the code).
- Measured NCCL SHM all-reduce on 4x RTX 5060 Ti, TP4 ring, Gen3 x8 (HyperQwen #254): 10 KiB 44 us; 80 KiB 64 us; 160 KiB 98 us; 20 MiB 7.35 ms. With P2P: 29 / 56 / 85 us / 7.14 ms. TP2 will be faster (2 ring steps instead of 6). By how much is unknown.
- llama.cpp on Linux defaults to NCCL (`ggml-cuda.cu:1224-1226`). `GGML_CUDA_ALLREDUCE=internal` forces the host-memory kernel. Which one is faster for decode on this box is unknown. Measure both.
- NCCL on Windows: not supported (install guide is Linux-only; PyTorch says Windows supports all backends but NCCL). NCCL master has an experimental Windows layer. Do not rely on it.

### 3.4 Can a one-shot all-reduce work through pinned, mapped host memory?

Yes. llama.cpp already does it (section 1.4), and NCCL SHM does the same as a ring. Rules that make it correct:

| Topic | Rule | Source |
|---|---|---|
| Atomicity | Aligned 1, 2, 4, 8 and 16-byte device loads and stores to host memory are single accesses for the host and other devices | CUDA Programming Guide, Mapped Memory |
| Atomics | Atomics on mapped memory are not atomic for the host or other devices. Use plain stores plus fences | same |
| Ordering | `__threadfence_system()` orders earlier writes before later ones for peer devices. Use release store for the flag, acquire load on the reader | CUDA PG, Memory Fence Functions; PTX ISA |
| Polling | Use `ld.volatile`, `ld.relaxed.sys`, `ld.acquire.sys` or `ld.cv`. A plain load can hit a stale L2 line | PTX ISA, cache operators |
| Portable | `cudaHostAllocPortable` so both cards can use the buffer | CUDA runtime API |

Choices specific to host memory:
- **Lamport sentinel reset.** In device memory the reset is a cheap local write. In host memory the reset by the reading GPU would double the upstream bytes. Alternatives: a CPU thread resets used buffers (triple buffering); or an epoch tag in each 16-byte unit (12 B data + 4 B tag, 1.33x bytes); or per-chunk flags after data (no reset, one extra round trip per chunk, overlapped).
- **Write-combined memory** (`cudaHostAllocWriteCombined`): not snooped, may be faster; the CPU never reads it in this design. Effect on AM5 is unknown. Measure cached and WC.
- **Several blocks are needed.** exllamav3 measured that one block reaches only 17 of 26 GB/s of pinned reads on PCIe 4 (`exllamav3_ext/parallel/context.cuh:10-14`).

Windows (WDDM) caveats:
- Mapped pinned memory works (llama.cpp uses it by default on Windows; NCCL's Windows layer and exllamav3 also do).
- Launch latency 5-20 us under WDDM. Event wait between streams about 2 us with HAGS off, 20-30 us with HAGS on (NVIDIA forum threads).
- TDR: a kernel that spins longer than 2 s (default `TdrDelay`) is killed. Spin loops need a timeout and an abort flag.
- WDDM batches launches. One card's spinning kernel may wait for a launch on the other card that is still queued. Impact unknown. Measure with Nsight Systems.

### 3.5 Published latency numbers

| Quantity | Number | Source |
|---|---|---|
| PCIe DMA read 64 B, device to host DRAM | median 547 ns (Xeon E5), median 1,213 ns (Xeon E3) | Neugebauer et al., SIGCOMM 2018 |
| GPU zero-copy read round trip | 1.0-1.6 us | EMOGI, arXiv 2006.06890, sec. 3.3 |
| PCIe round trip from GPU | 1-5 us, variable | Min et al., arXiv 2103.03330 |
| Max outstanding reads | 256 (PCIe 3.0) | same two papers |
| Posted write latency GPU -> host | no published number | unknown |
| Kernel launch latency | about 5 us minimum; 5-20 us on WDDM | NVIDIA forum (njuffa) |
| `cudaMemcpyPeerAsync`, P2P off (driver staging) | GPU latency 10.6-14.4 us | p2pBandwidthLatencyTest results (aikitoria README; tinygrad issue #14) |
| Same, P2P on | 0.38-0.92 us | same |
| NCCL SHM step | about 7.3 us per ring step (44 us / 6 steps), P2P about 4.8 us | derived from HyperQwen #254 |

### 3.6 Estimate for a one-shot all-reduce through mapped host memory on Gen3 x4

Model (estimate): `T = L + P / BW`, when write and read are pipelined per chunk. L = 3 us (optimistic: 0.75 us write landing + 1.5 us poll round trip + 0.75 us average poll slack) to 8 us (pessimistic, close to NCCL's SHM step). BW = 3.5 GB/s (practical) to 2.5 GB/s (NCCL SHM reached about 54% of the raw link).

| Payload | Low (3 us, 3.5 GB/s) | High (8 us, 2.5 GB/s) | NCCL SHM TP4 Gen3 x8 (scale only) |
|---|---:|---:|---:|
| 40 KiB (4 tokens BF16) | 3 + 11.7 = **14.7 us** | 8 + 16.4 = **24.4 us** | 64 us at 80 KiB |
| 160 KiB (8 tokens F32) | 3 + 46.8 = **49.8 us** | 8 + 65.5 = **73.5 us** | 98 us |
| 20 MiB (1024 tokens F32) | 3 + 5,992 us = **6.0 ms** | 8 + 8,389 us = **8.4 ms** | 7.35 ms |

Variants at 3.5 GB/s, L = 3 us (estimate):

| Variant | 40 KiB | 20 MiB |
|---|---:|---:|
| Pipelined (per-chunk flag or Lamport, reset by CPU) | 14.7 us | 6.0 ms |
| Epoch tag 12 B + 4 B | 18.6 us | 8.0 ms |
| One flag after all data (llama.cpp kernel path today) | 26.4 us | 12.0 ms |

For 2 ranks, one-shot, two-shot and ring move the same bytes per link direction. One-shot has the fewest sync rounds, so it is the right choice for TP2. On this topology host staging also moves the same bytes per link as P2P: each card has its own x4 link to the CPU, and data goes up one link and down the other in both cases.

---

## 4. P2P on GeForce Blackwell

Details, sources and caveats are in a sub-report (`scratchpad/p2p-blackwell.md`).

### 4.1 Windows (WDDM) [Windows-only]

- The only documented P2P under WDDM is NVLink + SLI (CUDA 10 release notes; GTC 2019 talk S9957). The 5060 Ti has no NVLink.
- TCC: "NVIDIA GeForce GPUs (excluding GeForce GTX Titan GPUs) do not support TCC mode" (CUDA Windows installation guide).
- MCDM: from R595 some TCC-capable GPUs start in MCDM. Whether a GeForce RTX 50 accepts `nvidia-smi -dm 2` is unknown. Whether P2P works under MCDM is unknown.
- A report with driver 616.64 (this rig's driver) and 2x RTX 4090 under WSL2 shows `GNS` (P2P not supported) in `nvidia-smi topo -p2p r` (HyperQwen issue #190).
- No Windows P2P hack was found. All known unlocks are Linux kernel-module options.
- Conclusion: on the Windows test bench, plan without P2P.

### 4.2 Linux [Linux-only]

- The tinygrad patch maps all of VRAM into BAR1 and replaces the peer aperture with a non-coherent system aperture pointing at the other card's BAR1. Development stopped at 570.x. Its README claims 4090/5090 support.
- The aikitoria fork continues to 615.71.09-p2p. It lists RTX 30, 40 and 50 support, "including models below" the top cards. Requirements: Above 4G decoding and Resizable BAR in BIOS; BAR1 large enough for all usable VRAM (16 GB for a 5060 Ti); `amd_iommu=on iommu=pt`; ACS off on the root ports. NVIDIA's own guide says IOMMU must be off for bare-metal P2P on Linux.
- Driver-free variant: `rtx-p2p` with `NVreg_RegistryDwords="RMDisableFeatureDisablement=1;EnableResizableBar=1"` on stock 590+. Tested on RTX 5090 only.
- RTX 50 reports: works on 8x 5090 + RTX PRO 6000 (EPYC), on 4x 5090 (tinybox), and on **4x 5060 Ti** (Xeon, Gen3 x8, aikitoria patch ported to 595.91.07; NCCL needed `NCCL_P2P_LEVEL=PHB`; decode +2-6% at 1 stream, +10-16% at 4 streams). Fails on 2x 5090 D (570.133), 8x 5090 (tinygrad 570.148), and **2x 5090 on Ryzen 7 7800X3D / X670E (AM5)** with `cudaErrorMapBufferObjectFailed` (tinygrad issue #44, unresolved).

### 4.3 Through CPU root ports on AM5

- AMD Zen root complexes forward peer traffic between root ports (Christian König, 2019; Linux allows P2PDMA on Zen and newer since 5.9).
- Reports of working P2P across two CPU root ports on AM5 exist with RTX 3090-class cards: Ryzen 7 7700 (13.19 GB/s peer vs 6.65 host-staged, Gen4 x8) and Ryzen 9 9950X (13.16 vs 6.63). The 7700 machine **hard-reset** under sustained decode with vLLM's custom all-reduce over P2P.
- An idle destination card drops its link to Gen1 and peer copies fell to 3.22 GB/s (9950X report). Watch `LnkSta`.
- No report of RTX 50 P2P working on AM5 was found.
- OcuLink adapters do not change the PCIe tree. They can affect link training (a Fedora report shows an OcuLink eGPU training at 2.5 GT/s on Linux).

### 4.4 What P2P would give on this rig (estimates)

| Metric | P2P on | P2P off (driver staging) | Note |
|---|---|---|---|
| Unidirectional copy | 3.2-3.5 GB/s | 1.6-2.7 GB/s | 84-88% vs 42-69% of 3.94 GB/s, scaled from published runs |
| Small copy latency | 1-2 us | 10-15 us | `cudaMemcpyPeerAsync` |
| One-shot all-reduce, 40 KiB, own kernel | about 13 us (push into peer VRAM, local poll) | about 15 us (pipelined through host) | P2P saves one PCIe read round trip |
| Per decode step (130 all-reduces) | | | about 0.2-0.3 ms saved (estimate) |

The large gap between P2P and "P2P off" in published tests comes from the driver's staging in `cudaMemcpyPeerAsync`. A kernel that writes and reads mapped host memory directly does not pay that cost. That is why the earlier P2P patch gave nothing in llama.cpp: its internal all-reduce never uses P2P. On this rig P2P is a small gain with a known risk (hard resets, failures on AM5 + 5090). Keep it as a later Linux-only experiment.

---

## 5. Cost table

All values are **estimates** unless marked "measured". "Exposed" means time the GPUs wait for communication. It does not include compute. Link: 3.5 GB/s per direction. Decode step = verify 4 + catch-up 4 + 3 drafts. Prefill = one 1024-token ubatch.

### 5.1 Decode, per step

| Option | Cross-card syncs per step | Bytes per card per direction | Exposed comm per step | Arithmetic |
|---|---|---|---|---|
| A. llama.cpp today | 136 in-kernel barriers + at least 4 host waits (each drains both cards) + sync input copies per graph | AR 5.14 MiB; logits 3.5 MB D2H; inputs 0.66 MB (30k) to 3.3 MB (150k) H2D | **5.1 ms at 1k to 7.2 ms at 150k** (step measured: 37.7 ms at 1k, 53.9 ms at 150k) | AR: 130 x 26.4 us + 6 x 8.9 us = 3.5 ms, plus about 136 x 2 x 2-3 us of extra launch boundaries = 0.5-0.8 ms; logits 3.5 MB / 3.5 GB/s = 1.0 ms; inputs 0.1 ms (1k) to 1.9 ms (150k) |
| B. TP, pipelined one-shot AR, one CUDA graph, GPU sampling, GPU-built inputs | 136 in-kernel barriers, 1 host wait (or 0) | AR 5.14 MiB; logits under 4 KiB; inputs a few bytes | **2.0-3.2 ms** | 130 x 14.7-24.4 us + 6 x 6-12 us |
| C. B + overlap with GEMV | same as B | same as B | **0.4-1.0 ms** | 130 x 3-8 us (latency only). Needs a custom fused kernel |
| D. B or C with P2P (Linux) | same | same | **1.7-1.8 ms** (no overlap), **0.2-0.5 ms** (overlap) | 130 x (1-2 + 11.7) us |
| F. Pipeline parallel | about 5 hand-offs of 40 KiB | about 0.2 MiB | about 0.1 ms comm, **but +14 ms** weight read per step | 10,828 MiB / 395 GB/s = 28.7 ms vs 14.4 ms for TP. Measured: TP +25% at 1k, +88% at 100k |
| G. Dual layout | like B/C | like B/C | like B/C | does not fit at 180k (2.1) |
| H. Replication | none | none | 0 | does not fit (2.1) |

Step time if only the communication part changes (estimate, 1k, from 37.7 ms measured): B about 34-36 ms, C about 32.5-33.5 ms. Other overheads (launches, sampling, kernels) are outside this report.

### 5.2 Prefill, per 1024-token ubatch

| Option | Cross-card syncs | Bytes per card per direction | Exposed comm | Ubatch time at 30k (estimate) |
|---|---|---|---|---|
| A. llama.cpp today | 130 all-reduces, each with a host `cudaEventSynchronize` and 5 cross-device chunk events; synchronous input copies | 1.36 GB AR + about 60 MiB inputs H2D / 20 MiB D2H | **about 0.82 s** (130 x 6.0 ms + 0.04 s) | 1.535 s (**measured**, 667 t/s) |
| B. Separate D2H/H2D streams, inputs on GPU | 130 | 1.36 GB | **about 0.43 s** (130 x 3.3 ms) | 0.72 + 0.43 = 1.15 s, about 890 t/s |
| C. B + two half-ubatches | 260 (half-size) | 1.36 GB | **about 0.02-0.06 s** | about 0.75 s, about 1,370 t/s |
| D. P2P copies | 130 | 1.36 GB | about 0.39 s without overlap; hidden with overlap | like B / C |
| E2. 8-bit wire (with B) | 130 | 0.69 GB | about 0.22 s | about 0.94 s; quality unknown |
| F. Pipeline parallel | 1 hand-off per ubatch (10 MiB BF16) | 10 MiB | about 3 ms, plus one stage of fill per prompt | about 0.72-0.75 s with 2+ ubatches in flight; conflicts with TP decode |

The 0.72 s non-communication part comes from 1.535 - 0.82 s. Faster kernels would lower it. Then the TP link limit of about 2,670 t/s (section 2.4) starts to matter.

### 5.3 What to measure first (no code changes needed except small probes)

1. `ggml_cuda_ar_kernel` duration in decode with Nsight Systems (take the minimum over both cards, because the faster card's kernel includes waiting). This checks the 26 us estimate.
2. Copy-engine all-reduce timeline in prefill: confirm that D2H and H2D do not overlap.
3. PCIe throughput during decode (Nsight Systems PCIe metrics), to resolve the 7-10x gap in section 1.6.
4. `deviceQuery`: `asyncEngineCount` on GB206.
5. Pointer-chase kernel on mapped host memory: read round trip on AM5 through OcuLink; sustained zero-copy read and write rate per link, with both links active; cached vs write-combined.
6. On Ubuntu: NCCL (SHM) vs `GGML_CUDA_ALLREDUCE=internal` for decode and prefill.
7. Windows only: WDDM launch batching with spin kernels; HAGS on/off.

---

## Sources

Local (read-only):
- `llama-rig2\ggml\src\ggml-cuda\allreduce.cu` (lines cited above)
- `llama-rig2\ggml\src\ggml-backend-meta.cpp`
- `llama-rig2\ggml\src\ggml-cuda\ggml-cuda.cu`
- `llama-rig2\ggml\src\ggml-backend.cpp`
- `llama-rig2\src\llama-model.cpp`, `src\llama-context.cpp`, `src\llama-context.h`, `src\llama-graph.cpp`, `src\models\qwen35.cpp`
- `llama-rig2\common\speculative.cpp`, `common\common.h:398-404`, `tools\server\server-context.cpp:3193-3217`
- Git history: merge `27a09c551` (PR #27173 port), `4ca109571` (`RIG_*` switches), `f3c3e0e9a` (PR #22299 internal all-reduce)
- `qwen38_27\LEEME.md` (measurements: lines 61, 82, 95-135, 400-425, 579-611) and `arranca.ps1`
- `qwen27-engine\research\_gguf-model-tensors.tsv`
- `qwen27-engine\refs\ik_llama.cpp\common\common.cpp:2495-2497`, `ggml\src\ggml-cuda\reduce.cu`
- `qwen27-engine\refs\vllm`, `refs\sglang` (paths in section 3.1)

Web:
- llama.cpp PR #22299 (internal AllReduce): https://github.com/ggml-org/llama.cpp/pull/22299
- llama.cpp PR #27173: https://github.com/ggml-org/llama.cpp/pull/27173
- llama.cpp release b9095: https://github.com/ggml-org/llama.cpp/releases/tag/b9095
- llama.cpp multi-GPU docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- vLLM on 2x 5060 Ti (club-5060ti issue #7): https://github.com/5p00kyy/club-5060ti/issues/7
- HyperQwen issue #254 (4x 5060 Ti, NCCL SHM vs P2P): https://github.com/syv-ai/HyperQwen/issues/254
- HyperQwen issue #190 (driver 616.64, GNS): https://github.com/syv-ai/HyperQwen/issues/190
- TensorRT-LLM (commit dc6f88de): https://github.com/NVIDIA/TensorRT-LLM/blob/dc6f88de69ce869dc0ce095c888a70705a8b663c/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu , `.../customAllReduceKernels.cu`, `.../allReduceWorkspace.cu`, `cpp/tensorrt_llm/thop/allreduceOp.cpp`, `cpp/tensorrt_llm/runtime/ipcUtils.cpp`
- NCCL (commit 12df1a11): https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport/shm.cc , `src/graph/paths.cc`, `src/device/prims_ll.h`, `src/os/windows.cc`
- NCCL env docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL install guide: https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html ; Windows PR: https://github.com/NVIDIA/nccl/pull/1922 ; release v2.32.3-1: https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1
- PyTorch distributed docs: https://docs.pytorch.org/docs/2.14/distributed.html
- FlashInfer (commit 58171ea8): https://github.com/flashinfer-ai/flashinfer/blob/58171ea83f32185a7990cfc15c969ecf79bb406c/include/flashinfer/comm/pcie_ipc_all_reduce.cuh
- exllamav3 (commit 16a49792): https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/model/model_tp_backend.py , `exllamav3_ext/parallel/context.cuh`, `ll.cuh`
- CUDA C++ Programming Guide 12.9.1 (mapped memory, memory fences, IPC): https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html
- CUDA Programming Guide, multi-GPU systems: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Neugebauer et al., "Understanding PCIe performance", SIGCOMM 2018: https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf
- EMOGI: https://arxiv.org/pdf/2006.06890 ; Min et al. (PyTorch-Direct): https://ar5iv.labs.arxiv.org/html/2103.03330
- NVIDIA forum, kernel launch latency: https://forums.developer.nvidia.com/t/kernel-launch-latency/62455
- NVIDIA forum, HAGS event latency: https://forums.developer.nvidia.com/t/increased-time-to-synchronize-streams-via-event-record-wait-when-hags-is-enabled/282181
- TDR registry keys: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- CUDA 10.0 release notes (P2P on WDDM 2.0): https://docs.nvidia.com/cuda/archive/10.0/cuda-toolkit-release-notes/index.html
- GTC 2019 S9957 "Using CUDA on Windows": https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9957-using-cuda-on-windows.pdf
- CUDA Windows installation guide (TCC): https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- nvidia-smi docs (driver models): https://docs.nvidia.com/deploy/nvidia-smi/index.html ; CUDA 13.2 blog (MCDM default): https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/ ; MCDM forum thread: https://forums.developer.nvidia.com/t/will-microsoft-windows-mcdm-improve-the-wddm-vs-tcc-situation/310058
- tinygrad P2P modules: https://raw.githubusercontent.com/tinygrad/open-gpu-kernel-modules/550.54.15-p2p/README.md , https://raw.githubusercontent.com/tinygrad/open-gpu-kernel-modules/570.148.08-p2p/README.md ; issues #14, #35, #42, #44: https://github.com/tinygrad/open-gpu-kernel-modules/issues/44
- aikitoria P2P modules: https://raw.githubusercontent.com/aikitoria/open-gpu-kernel-modules/615.71.09-p2p/README.md
- rtx-p2p: https://github.com/kacper-daftcode/rtx-p2p
- NVIDIA `nv-reg.h` (EnableResizableBar): https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/main/kernel-open/nvidia/nv-reg.h
- NVIDIA staff on RTX 50 P2P: https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8
- AMD Zen P2P DMA: https://www.phoronix.com/news/Linux-5.2-AMD-Zen-P2P-DMA ; https://lkml.iu.edu/hypermail/linux/kernel/2007.3/08010.html
- AM5 P2P reports: https://github.com/noonghunna/club-3090/issues/1332 ; https://github.com/noonghunna/club-3090/issues/1507 ; https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
- OcuLink link training on Linux: https://discussion.fedoraproject.org/t/oculink-egpu-link-speed-limits-to-2-5-gt-s/183531
- AM5 lane layout: https://www.pugetsystems.com/labs/articles/amd-x870e-vs-x870-vs-x670e-vs-x670-vs-b650e-vs-b650/
- p2pBandwidthLatencyTest source: https://raw.githubusercontent.com/NVIDIA/cuda-samples/v13.1/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu
