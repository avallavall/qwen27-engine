# Launch overhead, syncs and host work in a decode step

Topic: the part of a decode step that is not weight or KV reading. Kernel launches,
gaps between kernels, host syncs, CPU work between MTP drafts, and the ways to remove
them (CUDA graphs, PDL, persistent kernels, megakernels).

## Summary

1. **About 19 ms of every step is not weight or KV reading.** At 1k context the step is
   37.7 ms and the weight floor is 19.1 ms. At 150k the step is 53.9 ms and weights plus
   KV are 34.9 ms. The remainder is ~19 ms at both depths (arithmetic in section 1).
2. **llama.cpp uses CUDA graphs, but in small pieces.** With `-sm tensor` the meta backend
   cuts each forward pass at every all-reduce. One verify pass becomes ~129 small CUDA
   graphs per GPU. The 128 all-reduce kernels run between the graphs, outside them.
   Graphs are never fully off for this model. They are bypassed for two calls after any
   graph rebuild, which happens every 256 tokens of context (n_kv padding).
3. **One step launches about 2,000 kernels per GPU (estimate).** ~1,850 in the verify pass,
   ~150 in the four MTP passes. The step has 4 host syncs. Each sync copies logits to the
   CPU and runs CPU sampling over 248,320 tokens.
4. **Estimated split of the ~19 ms:** all-reduce transfers ~3.5 ms, host round trips
   2.3-4.6 ms, launch gaps 1.6-3.7 ms, fixed cost of ~1,400 tiny kernels 2.2-4.2 ms,
   GDN state snapshots ~1 ms, rest unknown (matvec below copy bandwidth). Nobody has
   profiled this on the rig. It needs one Nsight Systems run.
5. **Launch cost numbers:** 2-4 µs per plain launch on Linux, 0.5-1.3 µs per node inside a
   graph. Windows WDDM: 5-20 µs per launch, fluctuating, from old (2015, 2018) forum
   measurements. No published CUDA numbers for HAGS on vs off were found.
6. **Megakernels:** Hazy (Llama-1B, H100/B200, 78% of bandwidth), MPK (1.0-1.7x,
   A100/H100/B200), calm (RTX 4090, ~90% of bandwidth, one cooperative kernel). None runs
   on sm_120 out of the box except calm-style code. All multi-GPU megakernels found use
   NVLink or NVSHMEM. None works over PCIe host staging.
7. **A persistent kernel can do the 2-GPU all-reduce itself.** llama.cpp already does it
   in a short kernel: mapped pinned host memory plus a polled flag
   (`ggml/src/ggml-cuda/allreduce.cu:110-202`). Main risks: WDDM TDR (2 s) and preemption
   on card 0, missing co-residency guarantees, and a hung peer. All are manageable if the
   kernel runs once per pass and every spin loop has a timeout.
8. **PDL works on sm_120 and llama.cpp already uses it on this rig.** It only helps
   inside one graph. It does not cross the graph and all-reduce boundaries.
9. **Floor per step (estimate):** plain launches 12-20 ms, CUDA graphs as today 10-16 ms,
   PDL + one graph per pass with GPU sampling 4-5 ms, megakernel 1-3 ms. After graphs,
   the two biggest items are host round trips and the exposed all-reduce. Moving sampling
   and the draft loop to the GPU, and hiding the all-reduce, matter more than the last
   step from "one graph per pass" to "one megakernel".

---

## 1. What one decode step contains today

Setup: `qwen38_27\arranca.ps1` (`-sm tensor`, MTP `--spec-draft-n-max 3`,
`--spec-draft-sampling probabilistic`, `LLAMA_SCHED_POOL=8`, build `llama-rig2`).

### 1.1 Passes per step

`p_min` defaults to 0 (`common/common.h:332`), so the draft loop always makes 3 drafts.
The verify batch is always 4 tokens.

| Pass | Context | Rows | Outputs | Code |
|---|---|---|---|---|
| Verify | target | 4 | 4 logits rows | server decode |
| MTP catch-up | MTP draft ctx | 4 | none | `common/speculative.cpp:1709-1753` (`llama_process` at 1746) |
| Draft 1, 2, 3 | MTP draft ctx | 1 each | 1 logits row each | `common/speculative.cpp:1950-2053` (`llama_process` at 1967) |

The catch-up pass runs because `defer_enabled` is only on with `LLAMA_SPEC_CHAIN`
(`common/speculative.cpp:1503-1507`, 1673).

### 1.2 Weight floor and the remainder

Bandwidth used: 390 GB/s (measured copy, 388 / 395 GB/s stock). Each GPU reads half.

| Item | Bytes per GPU | Time per GPU |
|---|---|---|
| Verify, weights | 10,828 MiB / 2 = 5,414 MiB | 14.56 ms |
| MTP catch-up, MTP block only | 332 / 2 = 166 MiB | 0.45 ms |
| 3 drafts, MTP block + `output.weight` | 3 x 1,014 / 2 = 3 x 507 MiB | 4.09 ms |
| **Weight floor per step** | | **19.10 ms** |
| Target KV at 150k | 64 KiB x 150,000 / 2 = 4.92 GB | 12.6 ms |
| MTP KV at 150k, 4 passes | 4 x (4 KiB x 150,000 / 2) = 1.23 GB | 3.15 ms |

The brief estimated ~15 + ~4 ms. The catch-up pass adds another 0.45 ms.

| Depth | Measured step | Weights + KV (estimate) | Remainder |
|---|---|---|---|
| 1k | 37.7 ms | 19.1 + ~0.1 = 19.2 ms | **~18.5 ms** |
| 150k | 53.9 ms | 19.1 + 15.8 = 34.9 ms | **~19.0 ms** |

The remainder does not grow with context. KV reads explain the growth from 1k to 150k
(16.2 ms measured, 15.8 ms estimated). So the remainder is a fixed per-step cost:
launches, syncs, host work, all-reduce, small kernels, and matvec inefficiency.

---

## 2. llama.cpp today: CUDA graphs, kernel count, host work

### 2.1 How CUDA graphs work in `ggml-cuda.cu`

- Built when `GGML_CUDA_USE_GRAPHS` is set (`common.cuh:1259-1261`).
- Each CUDA backend keeps a map from the first node pointer to a graph
  (`common.cuh:1460-1488`, key at `ggml-cuda.cu:2647-2649`). A sweep every 5 s evicts
  graphs unused for 10 s (`common.cuh:1470-1480`).
- `ggml_backend_cuda_graph_compute` (`ggml-cuda.cu:4698-4755`):
  1. Checks the env var and the arch (`ggml-cuda.cu:4682-4696`, `common.cuh:1290-1293`).
  2. Checks compatibility. The only blocker is `MUL_MAT_ID` with a stream sync, which is
     MoE only (`ggml-cuda.cu:2614-2645`). This model is dense, so it never triggers.
  3. Checks whether properties changed. If the cgraph `uid` matches, it skips the check
     (`ggml-cuda.cu:2657-2662`). Otherwise it compares every node with `memcmp`
     (`ggml-cuda.cu:2672-2689`).
  4. Warmup rule: a graph is captured only after two calls in a row with no change
     (`ggml-cuda.cu:4718-4737`). Until then the nodes run as plain launches.
  5. Capture with `cudaStreamBeginCapture(..., Relaxed)` (`ggml-cuda.cu:4749`), run the
     nodes with fusion, end capture (`ggml-cuda.cu:4652`), instantiate (`4667`) or
     `cudaGraphExecUpdate` with a re-instantiate fallback (`ggml-cuda.cu:2694-2720`),
     then `cudaGraphLaunch` (`ggml-cuda.cu:4673`).
- `GGML_CUDA_GRAPH_OPT=1` adds multi-stream fork/join inside a graph. It is off by
  default (`ggml-cuda.cu:4904-4918`).

### 2.2 What `-sm tensor` does to the graphs

- `-sm tensor` creates one "meta" device over both GPUs (`src/llama.cpp:158-220`).
- The meta backend splits the forward pass into subgraphs. A new subgraph starts after
  every node whose output is a partial sum (`ggml-backend-meta.cpp:2426-2466`). For this
  model that is the attention or GDN output projection and the FFN down projection, so
  2 per layer.
- It runs subgraph i on GPU 0, then on GPU 1, then the all-reduce, then subgraph i+1
  (`ggml-backend-meta.cpp:2735-2764`). Each subgraph goes through
  `ggml_backend_cuda_graph_compute`, so **each subgraph is its own CUDA graph**.
- The all-reduce is not captured. On Windows the default is the internal pipeline
  (`ggml-cuda.cu:1222-1229`). For tensors under 1 MB it launches one kernel per GPU on the
  compute stream with `<<<>>>` (`allreduce.cu:917-981`, launch at 951-960). On Linux the
  default is NCCL if compiled in (`ggml-cuda.cu:1225-1226`), also between subgraphs.
- Result for one verify pass: ~129 CUDA graph launches and 128 all-reduce kernel launches
  per GPU. Each CUDA graph holds only the ~14 kernels between two all-reduces.
- Subgraph `uid`s are set when the meta plan is built (`ggml-backend-meta.cpp:2563`).
  The meta backend caches 16 plans by `uid` (`ggml-backend-meta.cpp:1958-1981`,
  2160-2209). On a cached plan the CUDA backend skips the property comparison.

### 2.3 When graphs are off or bypassed

| Condition | Effect here | Source |
|---|---|---|
| `GGML_CUDA_DISABLE_GRAPHS` set | Off | `common.cuh:1291` |
| GPU older than Volta | Off | `ggml-cuda.cu:4686-4691` |
| `MUL_MAT_ID` that needs a stream sync | Off for that graph. MoE only, never here | `ggml-cuda.cu:2626-2636` |
| Multi-GPU or split mode tensor | **Not off.** Each GPU captures its own subgraphs. Older versions disabled graphs for multi-GPU and batch > 1 (PR #6766); that check is gone | `ggml-cuda.cu:2614-2645` |
| Batch size > 1 (verify batch of 4) | **Not off** | same |
| First call of a new graph key, or any property change | 2+ calls as plain launches, then capture and instantiate | `ggml-cuda.cu:4718-4737` |
| llama graph rebuild | New `uid`, new meta plan, new subgraph pointers, so new keys and full warmup for ~130 subgraphs per GPU | `llama-context.cpp:1496-1533`, `ggml-backend-meta.cpp:2226-2247` |
| Context grows past a multiple of 256 | Triggers the rebuild above, because `n_kv` is padded to 256 | `llama-kv-cache.cpp:1260-1274` |
| Graph unused for 10 s (idle between requests) | Evicted, recaptured on next use | `common.cuh:1470-1480` |
| Speculative decoding shape changes | Handled. #28549 keeps two graph results per context (outputs vs no outputs). `LLAMA_SCHED_POOL=8` keeps one scheduler and graph per (n_tokens, n_outputs) shape | `llama-context.cpp:2562-2568`, 734-745, 1445-1478 |

The rebuild every 256 tokens is ~1 rebuild every ~65-90 steps (2.9-3.9 tokens per
step). Each rebuild costs a graph build, a scheduler allocation, a meta plan, and ~260
captures plus instantiations. Estimate: a few ms per rebuild, so ~0.1-0.4 ms per step
on average. Not measured.

`LLAMA_SCHED_POOL=8` measured on this rig: -1.7% step time (`qwen38_27\LEEME.md:61`,
579, 603). The all-reduce without the CPU wait (#27173): -0.6% (`LEEME.md:579`).

### 2.4 Kernels per step (estimate from the graph)

Counted from `src/models/qwen35.cpp` with the fusions in `ggml_cuda_try_fuse`
(`ggml-cuda.cu:3667-4461`). Facts used:
- MMVQ quantizes the activation to q8_1 in a separate kernel on every call
  (`mmvq.cu:1531-1536`).
- With MTP, `n_rs_seq` > 0, so each GDN layer writes K = 4 conv-state copies
  (`delta-net-base.cpp:512-529`). The 4 GDN state snapshots are fused into the GDN kernel
  (`ggml-cuda.cu:3725-3737`).
- Alpha/beta projections are fused (#29187, `ggml-cuda.cu:3695-3711`).

GDN layer, per GPU:

| Kernels | Count |
|---|---|
| residual add, rms_norm+mul | 2 |
| wqkv matvec + quantize | 2 |
| z (gate) matvec + quantize | 2 |
| alpha/beta fused | 1 |
| conv state get_rows, concat, 4 conv-state copies | 6 |
| ssm state get_rows | 1 |
| ssm_conv+silu, 2 x l2_norm | 3 |
| gated_delta_net (with fused snapshots) | 1 |
| gated norm (rms_norm+mul, silu+mul) | 2 |
| ssm_out matvec + quantize | 2 |
| all-reduce | 1 |
| residual add, rms_norm+mul | 2 |
| up/gate+swiglu fused matvec + quantize | 2 |
| down matvec + quantize | 2 |
| all-reduce | 1 |
| **Total** | **~30** |

Full-attention layer, per GPU: ~25 (3 separate Q/K/V matvecs with quantize, fused
norm+rope and norm+rope+set_rows, gate copy, V set_rows, flash-attention plus combine,
sigmoid+mul, wo, 2 all-reduces, FFN as above).

| Pass | Kernels per GPU (estimate) |
|---|---|
| Verify: 48 x 30 + 16 x 25 + head ~5 | ~1,850 |
| MTP catch-up (no head) | ~33 |
| 3 drafts x ~37 | ~111 |
| **Per step** | **~2,000 per GPU, ~4,000 for both** |

Other per-step counts (per GPU): ~140 all-reduces (128 in verify, ~3 per MTP pass),
~145 CUDA graph launches, 4 host syncs. Of the ~1,850 verify kernels, ~385 are matvecs
and ~1,340 are tiny (norms, quantize, copies, rope).

Uncertainty: ±30%. To count exactly: one `nsys profile --trace=cuda` run of a few
steps and `nsys stats -r cuda_gpu_kern_sum`.

### 2.5 CPU work between MTP drafts

Per draft step (`common/speculative.cpp:1950-2053`):

1. `llama_process(ctx_dft, ...)` (line 1967):
   - graph reuse check and `set_inputs`. This includes the CPU `get_rows` of the token
     embedding (the 388 MiB `token_embd` stays in host RAM), positions, and the KQ mask.
     The mask is `n_kv x n_tokens` F16, built on the CPU each pass
     (`llama-graph.cpp:982`, `llama-kv-cache.cpp:1567`), and copied to both GPUs.
     At 150k that is ~0.3 MB per draft pass and ~1.2 MB per 4-row pass.
   - meta backend loop: ~4 subgraph graph launches and ~3 all-reduces per GPU.
2. Sync and copy logits to the host. The output head is split by vocab, so each GPU
   sends half: 124,160 x 4 B = 0.5 MB per draft row, 2 MB for the 4 verify rows.
3. CPU sampling. Backend sampling is refused with `-sm tensor` unless the output head is
   mirrored (`llama-context.cpp:1318-1340`; the log confirms it, `qwen38_27\arranque.log.err`).
   The mirror costs +3 ms per step (`LEEME.md:610`).
   `common_sampler_sample` fills a 248,320-entry candidate array
   (`common/sampling.cpp:159-161`) and runs top-k 10 (`common/speculative.cpp:1473-1474`).
4. Copy the h row (`llama_get_embeddings_nextn_ith`) and rebuild the batch.

After the verify pass the server does the same for 4 rows, with temp / top-k 20 /
top-p 0.95 and the probabilistic acceptance.

Graph rebuild: not per draft. `LLAMA_SCHED_POOL` keeps the 1-row and 4-row shapes warm.

### 2.6 Where the ~19 ms probably goes (all estimates)

| Item | Arithmetic | Estimate per step |
|---|---|---|
| All-reduce transfers | Batch 4, BF16 wire: 4 x 5120 x 2 B = 40 KiB each way. 40 KiB / 3.5 GB/s = 11.7 µs write + ~12 µs read + 1-3 µs flag = ~25-27 µs. x 128. MTP ARs add ~0.15 ms | **~3.5 ms** |
| Host round trips (4 syncs) | Verify: logits 2 MB at 3.5 GB/s = 0.57 ms, CPU sampling 4 rows x 0.15-0.4 ms, restart 0.05-0.2 ms. Each draft: 0.14 + 0.15-0.4 + 0.05-0.2 ms | **2.3-4.6 ms** |
| Launch gaps inside and between graphs | 2,000 x 0.5-1.3 µs + ~285 graph/AR boundaries x 2-4 µs | **1.6-3.7 ms** |
| Fixed runtime of tiny kernels | ~1,450 tiny kernels x 1.5-3 µs | **2.2-4.2 ms** |
| GDN state snapshots (not launch related) | 4 snapshots x 1.5 MiB + 1.5 MiB read, x 48 layers = 360 MiB per GPU at 390 GB/s | **~1 ms** |
| Matvec below copy bandwidth, imbalance between GPUs, other | not known | **rest, ~2-8 ms** |

Supporting data from the rig (`LEEME.md:606-614`, "Lo que se probo y no sirve"):
`LLAMA_SPEC_CHAIN=1` drafts all 3 tokens in one decode with in-graph argmax. The step
was 10% faster (~4.5 ms), even though it needed the mirrored head (+3 ms). Part of the
gain is a smaller draft head (32,768 rows instead of 248,320). My arithmetic for that
part: 3 x (341 - 90) MiB per GPU = 753 MiB, ~2 ms. The rest (~5.5 ms) came from fewer
passes, fewer host round trips and fewer launches. That fits the estimates above.

---

## 3. Launch cost numbers

| Measurement | Value | Platform | Source |
|---|---|---|---|
| Plain launch, async stream | 3.8 µs per 2.9 µs kernel | V100, CUDA 10.1, OS not stated | NVIDIA blog "Getting Started with CUDA Graphs" |
| Same with CUDA graph | 3.4 µs per kernel | same | same |
| Launch + sync after each kernel | 9.6 µs per kernel | same | same |
| Launch on a stream | ~2.1 µs | H100 | Hazy "No Bubbles" |
| Same with CUDA graph | ~1.3 µs | H100 | same |
| Eager launch | 3.8 µs (1.1 ms per token for Qwen3-8B, 293 launches) | B200 | MPK paper |
| CUDA graph | 0.8 µs per launch (0.2 ms per token) | B200 | MPK paper |
| Graph CPU launch, straight line | 2.5 µs + ~1 ns per node (CUDA 12.6) | RTX 3060, Ubuntu 22.04 | NVIDIA blog "Constant Time Launch..." |
| Graph device runtime | 53 µs / 100 nodes, 567 µs / 1025 nodes (~0.55 µs per node) | same | same |
| Graph instantiation | 127 µs / 100 nodes, 1.5 ms / 1025 nodes | same | same |
| Null kernel launch, Linux | ~5 µs (2015, 2018); ~2.5-3 µs with PCIe 5 hardware (2025) | various | njuffa, NVIDIA forums |
| **Windows WDDM** null kernel launch | 10-80 µs, average ~20 µs (2015) | **Windows only** | njuffa, NVIDIA forums |
| **Windows WDDM** launch latency | "fluctuating between 5us and 20us" (2018) | **Windows only** | njuffa, NVIDIA forums |
| **Windows WDDM** batching | Driver holds commands until a size limit or a sync | **Windows only** | NVIDIA forums (2015) |
| WDDM "somewhat longer, but not 10x" | statement, no number | **Windows only** | Robert Crovella, NVIDIA forums (2025) |
| `grid.sync()` per barrier | ~3 µs | RTX 5090 (sm_120) | alpindale blog |
| Custom flag barrier per barrier | ~2.2 µs (8.8 µs for 4 barriers per layer) | RTX 5090 (sm_120) | alpindale blog |

**HAGS (Windows only).** NVIDIA's WSL2 post says that with hardware scheduling the
user-mode driver submits straight to hardware queues. Without it (packet scheduling),
"all work of one submission must finish before any work of the next submission can
start". I found no published CUDA launch-latency numbers for native Windows with HAGS
on versus off. Gaming articles report 1-3 ms input-lag changes, which says nothing about
CUDA. **Unknown.** On this PC the registry value `HwSchMode` under
`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` is not set, so Windows uses its
default. The actual state was not checked (Settings > Display > Graphics).

The old Windows numbers are 8-11 years old. They predate HAGS and current drivers. They
must be re-measured on this rig before any decision rests on them (micro-benchmark list
in section 8).

**Linux (deployment target):** no WDDM, no batching, no watchdog on a headless card.
Launch costs should be in the 2-4 µs range above.

---

## 4. Megakernels and persistent kernels

### 4.1 Hazy Research, "Look Ma, No Bubbles" (May 2025), low-latency Llama-1B

- What: one kernel per forward pass for Llama-3.2-1B, BF16, batch 1. Each SM runs an
  interpreter over a list of instructions. 7 instruction types (fused RMS norm + QKV +
  RoPE + append, partial attention, attention reduction, O-proj + residual, RMS +
  up/gate + SiLU, down + residual, RMS + LM head) (`refs/Megakernels/demos/low-latency-llama/llama.cuh:5-11`).
  Shared memory is split into 13 pages of 16 KiB (213 KB on H100). Dependencies use
  counters in global memory.
- Why: the blog names three costs at each kernel boundary: straggler blocks of the old
  kernel, launch cost (2.1 µs stream, 1.3 µs graph on H100), and the wait to load weights
  at the start of the next kernel. It says PDL's `cudaGridDependencySynchronize` is too
  coarse.
- Measured: H100 under 1 ms per forward, 78% of memory bandwidth, ~2.5x vLLM, >1.5x
  SGLang. B200 under 680 µs, >3.5x vLLM.
- GPUs: H100 (sm_90a), B200 (sm_100a) (`refs/Megakernels/demos/low-latency-llama/Makefile:20-27`).
- sm_120: **not supported as is.**
  - `static_assert(NUM_PAGES == 13)` needs ~213 KB of shared memory
    (`include/config.cuh:43-44`). sm_120 allows 99 KB per block (Blackwell tuning guide).
    About 5 pages would fit.
  - Uses TMA load, store and store-add (`demos/low-latency-llama/*.cu`). sm_120 has TMA
    (CUTLASS SM120 kernels default to `KernelTmaWarpSpecializedCooperative`).
  - Uses warp-level `mma` (available on sm_120) and register reallocation
    (`setmaxnreg`), which CUTLASS enables for sm_120a (`refs/cutlass/include/cutlass/arch/reg_reconfig.h:46-55`).
  - Cluster size is 1 (`include/megakernel.cuh:167-168`). GeForce Blackwell has no TMA
    multicast and fixes clusters to 1x1x1 (CUTLASS Blackwell docs), so nothing is lost.
  - Built on ThunderKittens with `KITTENS_HOPPER` or `KITTENS_BLACKWELL`. Whether
    ThunderKittens compiles for sm_120 was not checked.
  - Weights are BF16. This model is IQ3_S/IQ4_XS/IQ2 etc., so every matvec instruction
    would need new dequant code.
- Multi-GPU: none in the low-latency kernel. The local clone (commit 7309cec, June 2025)
  has tensor parallelism only in the PyTorch reference model (`megakernels/llama.py:50-82`).

### 4.2 Hazy Research, tensor-parallel megakernel (Sept 2025)

- Posts: "We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU" and
  "One Kernel for All Your GPUs" (ParallelKittens / PGL).
- What: throughput megakernel for Llama-70B, 8-way tensor parallel, batch 1,024-8,192.
  Compute, memory and communication are overlapped. Dedicated "storer" threads do
  asynchronous stores straight into other GPUs' global memory, so loader and compute
  threads move on.
- Measured: 23,468 tok/s total vs SGLang 19,170 (>22%), 8x H100.
- Communication: NVLink peer loads/stores, TMA to peer addresses, NVSwitch multimem
  reductions. The PGL post says: "all cross-GPU communication happens through
  NVLink/NVSwitch. The PCIe path is only used for CPU-GPU communication."
- On this rig: **not applicable.** No NVLink. P2P over PCIe is unknown here (brief).
  The design target is throughput at large batch, not batch-1 latency.
- The code is not in `refs/Megakernels` (that clone predates it).

### 4.3 Mirage Persistent Kernel (MPK, OSDI 2026, arXiv 2512.22219)

- What: a compiler that turns a model into an SM-level task graph and one persistent
  kernel. Workers and schedulers run on SMs (`num_workers`, `num_local_schedulers` in
  `refs/mirage/README.md`). Tasks wait on events.
- Measured: 1.0-1.7x over SGLang/vLLM. Qwen3-8B on A100: 14.5 ms to 12.5 ms per token,
  with a hardware limit of ~10 ms. 8x H100: 1.1-1.4x. The paper measured 3.8 µs per eager
  launch and 0.8 µs per graph launch on B200. Its in-kernel scheduler uses 0.28% of runtime.
- GPUs: A100, H100, B200. No consumer GPUs, no PCIe-only systems.
- sm_120: the build script has TMA paths only for compute capability 90 and 100. Any
  other target builds with `-arch=native` and no TMA
  (`refs/mirage/python/mirage/mpk/persistent_kernel.py:362-380`). Untested on sm_120.
- Multi-GPU: NVSHMEM. All-reduce becomes data-transfer tasks with
  `nvshmem_signal_wait_until` plus local reduction tasks
  (`refs/mirage/include/mirage/persistent_kernel/persistent_kernel.cuh:23-39`, 1095-1100).
  NVSHMEM needs 64-bit Linux and GPUs connected by P2P (NVLink/PCIe) or GPUDirect RDMA
  (NVSHMEM install guide). **Not usable on Windows. On Linux it needs P2P, which is
  unknown on this rig.**

### 4.4 calm (zeux)

- What: single-GPU, batch-1 engine. All layers run in **one cooperative kernel**
  (`refs/calm/src/infer.cu:405-626`), launched with `cudaLaunchCooperativeKernel`, one
  1024-thread block per SM (`infer.cu:722`). A hand-written grid barrier
  (`atom.add.release.gpu` + `ld.acquire.gpu` spin, `infer.cu:332-351`) separates 6-7
  stages per layer. The LM head is a second kernel (`infer.cu:734`), then one
  `cudaStreamSynchronize` per token (`infer.cu:737`).
- Measured (README): RTX 4090, Llama3 8B fp16 61 tok/s (923 GB/s), fp8 120 tok/s
  (903 GB/s), gf4 225 tok/s (846 GB/s). Peak ~1,008 GB/s; the author saw ~955 GB/s max
  from any kernel. That is ~90% of peak. zeux's "LLM inference speed of light" post
  calls it "around 90% of the theoretically possible performance".
- GPUs: consumer (sm_89), no TMA, no clusters. **This style fits sm_120 most easily.**
- Requires `cooperativeLaunch` (`infer.cu:80`). On Windows this used to need TCC mode
  (forum, CUDA 9.2). A 2024 forum thread reports it now works on GeForce RTX 30/40 under
  WDDM. Not checked on the 5060 Ti.
- Multi-GPU: none.

### 4.5 Other single-GPU megakernels relevant to this rig

| Project | Model | GPU | Result | Notes |
|---|---|---|---|---|
| Lucebox megakernel | Qwen3.5-0.8B (18 DeltaNet + 6 attention layers, same hybrid family) | RTX 3090 | 413 tok/s vs llama.cpp 267 (1.55x) | 82 blocks x 512 threads, cooperative grid sync. Shows the GDN hybrid can be one kernel |
| alpindale | Qwen3-0.6B BF16 | RTX 5090 (sm_120) | 1,000 tok/s (0.97 ms per token) | 128 blocks x 512 threads, plain launch, custom flag barriers. 494 -> 813 -> 890 -> 905 -> 1000 tok/s. CUDA graphs "didn't make a difference"; the per-step `cudaStreamSynchronize` was the limit |

Both are small models where launch overhead is a large share. For a 27B model the share
is smaller, but here it is still ~19 ms of a 38 ms step.

---

## 5. A persistent kernel that does the 2-GPU all-reduce itself

### 5.1 What exists already

llama.cpp's chunked all-reduce kernel (`allreduce.cu:110-202`) already does the
cross-GPU part inside a kernel:

1. Each GPU writes its partial (cast to BF16) into **mapped pinned host memory**
   (`cudaHostAllocPortable | cudaHostAllocMapped`, `allreduce.cu:275-302`).
2. `__threadfence_system()`, then thread 0 of each of 8 blocks writes a token to its own
   flag (cache-line padded, `allreduce.cu:69-77`, 157-162).
3. It spins on the peer's flag with `__nanosleep(100)` (`allreduce.cu:164-173`).
4. `__threadfence_system()`, read the peer partial from host memory, add, write back.
- Plain volatile stores are used because `atomicAdd_system` needs
  `hostNativeAtomicSupported`, which PCIe consumer GPUs lack (`allreduce.cu:57-59`).
- Slot reuse is safe without host sync because the two GPUs run in lockstep
  (`allreduce.cu:381-397`).

This runs in production on this rig today, on Windows, with the same OcuLink links.

### 5.2 Extending it to one launch per pass

Each GPU runs one kernel for the whole pass. At each of the 128 all-reduce points:

- The blocks that produce the projection output write their slice to host memory as
  they finish (no grid barrier needed before the write).
- One flag per producing block, or one counter per all-reduce, tells the peer.
- The consumers of the next layer wait on the peer flags, read, add, continue.
- Final step: the vocab-split LM head gives each GPU half the logits. Each GPU computes
  its top-k candidates (for example 20 x (id, logit) = 160 B per row) and exchanges them
  the same way. Sampling and the probabilistic acceptance then run on the GPU.
- With the MTP draft loop also inside, one step becomes one launch per GPU, plus one
  small readback of accepted tokens. That readback can also go through mapped memory.

Transfer cost does not change: ~25-27 µs per batch-4 all-reduce, ~8 µs at batch 1
(estimate, section 2.6). What changes:

- No launch gap around each all-reduce (2 x ~2-4 µs saved per all-reduce).
- The kernel can stream the **next layer's weights** while it waits for the peer.
  These weights do not depend on the all-reduce result. At 390 GB/s, 27 µs of waiting
  equals ~10 MB of weight streaming. Prefetching into L2 or shared memory could hide
  most of the 3.5 ms. **Estimate, unproven on this hardware.** L2 size of the 5060 Ti
  was not verified (sources disagree); read it with `cudaDevAttrL2CacheSize`.
- A cheaper variant without a megakernel: keep normal kernels, but launch an L2
  prefetch kernel for the next layer on a second stream during the all-reduce kernel.
  The all-reduce kernel uses only 8 blocks (`allreduce.cu:77`), so most SMs are idle.

### 5.3 Risks

| Risk | Detail | Mitigation | OS |
|---|---|---|---|
| TDR watchdog | Windows resets a GPU that does not finish or preempt work within 2 s (`TdrDelay`, NVIDIA Nsight TDR docs). A kernel per pass runs ~5-25 ms, far below that. A kernel that never exits (waits for the next request) would depend on preemption | One launch per pass or per step. Never an endless kernel. Every spin loop has a timeout well below 2 s (`%globaltimer`) and an error flag | **Windows only** |
| Preemption on card 0 | Card 0 drives the desktop. The compositor's work can preempt a long kernel. Size of the stalls is unknown | Measure. On Ubuntu Server card 0 has no desktop | **Windows only** |
| WDDM command batching | The driver can hold commands until a buffer fills or a sync (section 3). A spinning kernel on GPU 0 then waits for a GPU 1 kernel that is not submitted yet. Production does not deadlock, but delays are possible | Flush both streams after enqueueing (for example `cudaStreamQuery`), measure with HAGS on and off | **Windows only** |
| Co-residency inside one GPU | CUDA does not guarantee forward progress between blocks that are not resident at the same time. PDL docs warn that relying on concurrent execution "can lead to deadlock" | `cudaLaunchCooperativeKernel` (guarantees all blocks launch), or grid = SMs x occupancy and nothing else on the GPU. Nothing else runs during decode in a one-request engine | both |
| Peer hang or crash | One GPU spins forever if the other faults | Timeouts in all spin loops, an abort flag in host memory, then a host-side error path | both |
| Memory ordering | Data must be visible before the flag. CUDA's `__threadfence_system()` orders a thread's writes as seen by host and peer devices. In a persistent kernel the same host buffer is read many times, so stale L1/L2 lines are possible | Writer: data, `fence.sc.sys` / `__threadfence_system`, then flag. Reader: acquire load of the flag (`ld.acquire.sys`), then `ld.volatile` / `ld.global.cv` for data, or alternate buffers with tokens as llama.cpp does | both |
| PCIe ordering | PCIe keeps posted writes in order unless relaxed ordering is set. llama.cpp relies on this and works on this rig. Not proven for every root complex | Keep the fence. Add a checksum or token in each data chunk if in doubt | both |
| No system atomics | `atomicAdd_system` is not available over PCIe here (`allreduce.cu:57-59`) | One writer per flag, monotonic tokens (as llama.cpp) | both |
| Debugging and profiling | Nsight Systems sees one long kernel | In-kernel `%globaltimer` stamps per stage (as calm `coopstage`, `infer.cu:390-402`, and Hazy timings) | both |
| Resource limits | One kernel must fit every stage: 99 KB shared memory per block, 64K registers per SM, 48 warps per SM on sm_120 | Size per stage, or use "one kernel per layer type" instead of one for everything | both |

---

## 6. Programmatic dependent launch (PDL) on sm_120

- PDL needs compute capability 9.0 or newer (CUDA programming guide). sm_120 qualifies.
  The secondary kernel can start early. `cudaGridDependencySynchronize()` blocks until
  the primary grids complete and flush. `cudaTriggerProgrammaticLaunchCompletion()` lets
  the primary release the secondary early. Inside graphs it works through stream capture
  or programmatic edges.
- **llama.cpp already uses it on this rig.**
  - Enabled when CUDART >= 12.3 for MSVC builds (`common.cuh:119-126`). The rig uses
    CUDA 13.4.
  - On by default; `GGML_CUDA_PDL=0` turns it off (`common.cuh:1700-1712`).
  - Only for kernels whose PTX version is >= 90 (`common.cuh:1678-1686`). The rig builds
    `120a-real`, so it should be on. Not verified at runtime.
  - Most kernels use `ggml_cuda_kernel_launch` with the PDL attribute (binbcast, cpy,
    getrows, mmvf, mmvq, norm, quantize, rope, set-rows, ssm-conv, gated_delta_net, unary,
    flash attention, and more).
- Measured (PR #22522, OS not stated in what I read): RTX PRO 6000 (sm_120) token
  generation 1.12-1.19x (gpt-oss 20B, Nemotron 31B-A3.5B, Qwen3 4B). DGX Spark (sm_121)
  1.04-1.06x. Gains for batch <= 16. "PDL + CG > CG > PDL".
- Limits here:
  - PDL acts between kernels in the same stream. A graph launch boundary is a full
    dependency. With ~129 subgraphs per pass, the first kernel of each subgraph gets no
    overlap.
  - The all-reduce kernel is launched with `<<<>>>` (`allreduce.cu:951-960`), so it
    neither waits early nor releases early.
  - The dependency is coarse: the whole primary grid must finish before any secondary
    block passes `cudaGridDependencySynchronize` (Hazy's objection).
- Windows: no documentation says PDL behaves differently under WDDM. No measurement
  found. **Unknown.** A/B on the rig: `GGML_CUDA_PDL=0` vs default.

---

## 7. Overhead floor per step for each approach (estimates)

Assumptions: step = verify (4 tokens) + MTP catch-up + 3 drafts. "llama.cpp-like" means
~2,000 kernels per GPU per step. "Fused" means a custom engine with ~10 kernels per layer,
~700 per GPU per step (for example: norm fused into matvec prologue; one in-projection
matvec for wqkv + z + alpha/beta; one kernel for conv + l2norm + GDN + gated norm; one
for up/gate + SwiGLU; one for down; one all-reduce). Linux numbers unless marked.

| Approach | Launches per GPU per step | GPU gaps | Tiny-kernel fixed time | Host round trips | All-reduce exposed | **Floor** |
|---|---|---|---|---|---|---|
| A. Plain launches, llama.cpp-like | ~2,000 | 2,000 x 2-4 µs = 4-8 ms | 2.2-4.2 ms | 2.3-4.6 ms (CPU sampling) | 3.5 ms | **12-20 ms** |
| A on **Windows WDDM** | same | CPU issue: 2 GPUs x 2,000 x 5-20 µs = 20-80 ms per step. Likely CPU-bound | | | | **worse than A, unknown** |
| B. CUDA graphs as today (~145 graphs + ~140 AR kernels) | ~285 | 2,000 x 0.5-1.3 µs + 285 x 2-4 µs = 1.6-3.7 ms | 2.2-4.2 ms | 2.3-4.6 ms | 3.5 ms | **10-16 ms** |
| B'. Graphs, fused engine, still cut at each all-reduce | ~285 | 700 x 0.5-1.3 µs + 285 x 2-4 µs = 0.9-2.0 ms | ~0.3 ms | 2.3-4.6 ms | 3.5 ms | **7-10 ms** |
| C. PDL + one graph per pass per GPU, all-reduce kernel captured (device-side token counter), GPU sampling, draft loop in the graph (conditional WHILE node, CUDA 12.4+) or 4 graph launches | 1-4 | 700 x 0.3-0.6 µs = 0.2-0.4 ms (PDL gap is a guess) | ~0.3 ms | 0.1-0.3 ms (one small readback) | 3.5 ms; 1-2 ms with L2 prefetch on a side stream | **4-5 ms** (2-3.5 with prefetch) |
| D. Megakernel, one launch per pass or per step, in-kernel all-reduce and sampling | 1-4 | ~650 dependency waits x 0.3-2 µs = 0.2-1.3 ms | ~0 | 0.05-0.1 ms | 0.7-1.8 ms if weight prefetch hides most of it (unproven) | **1-3 ms** |

Notes on the table:

- B matches today: 10-16 ms of launch-related cost inside the measured ~19 ms
  remainder. The rest would be matvec inefficiency and GDN snapshots.
- In C and D, Windows launch latency matters little, because there are only 1-4 launches
  per GPU per step. This also makes C and D behave the same on Windows and Linux.
- Rough step time if the matvecs reach the copy bandwidth minus ~3 ms (estimate):
  B ~37 ms (today), C ~26 ms, D ~24 ms at 1k context. At 2.88 tokens per step
  (76.5 tok/s x 37.7 ms) that is ~110 tok/s for C and ~120 tok/s for D, if acceptance
  stays the same. **Estimates with wide error bars.**
- The largest items left after B are host round trips (2.3-4.6 ms) and the all-reduce
  (3.5 ms). GPU sampling plus an in-graph draft loop removes the first. Overlap with
  weight prefetch is the only way found to reduce the second without P2P.

---

## 8. Consequences for the plan

1. Keep the whole step on the GPU: sampling (temp, top-k, top-p, min-p, probabilistic
   acceptance) and the MTP draft loop. The 2-GPU vocab split needs only a small top-k
   exchange per row.
2. Make the all-reduce a kernel that can live inside a CUDA graph or a persistent kernel.
   Pass the token from a device counter instead of a kernel argument. Reuse the llama.cpp
   host-staging protocol.
3. Start with approach C (fused kernels + PDL + one graph per pass). It gets most of the
   gain with normal kernels that Nsight can profile. Move to D only if measurements show
   the remaining gaps and all-reduce waits are worth it.
4. Try hiding the all-reduce behind next-layer weight prefetch in C (side-stream L2
   prefetch) before building D.
5. The causal decode mask can be computed in the attention kernel from positions. That
   removes the CPU mask fill and the H2D copy (up to ~1.2 MB per pass at 150k).

### Micro-benchmarks to run when the GPU is free (not run in this session)

| Bench | What it measures | Decision it feeds |
|---|---|---|
| `launch_lat` | Empty and 2 µs kernels: plain stream, graph, graph + PDL, chains of 100 and 1,000. Windows with HAGS on, HAGS off, and Ubuntu | Gap numbers for every row of section 7 |
| `ar_pingpong` | 2 GPUs, mapped pinned host memory: one-way flag latency, 10 KiB and 40 KiB exchange time, with BF16 | The 3.5 ms all-reduce estimate |
| `ar_overlap` | Same exchange while a second stream prefetches 10 MB into L2 | Whether the all-reduce can be hidden |
| `persist_wddm` | A 20 ms persistent kernel on card 0 with the desktop active: step-time jitter | Windows-only preemption risk |
| `tiny_kernels` | Runtime of rms_norm / quantize / rope at 5,120 elements on a 36-SM 5060 Ti | The 1.5-3 µs per tiny kernel estimate |
| deviceQuery | `cooperativeLaunch`, L2 size, `kernelExecTimeoutEnabled` on both cards | Cooperative launch on WDDM, L2 prefetch budget |

Plus one `nsys profile` of `llama-server` for ~20 decode steps. It would replace the
estimates in section 2.6 with measured numbers.

---

## 9. Open questions

- The real split of the ~19 ms remainder. Only estimates exist.
- HAGS state on this PC, and its effect on CUDA launch latency. No published data found.
- Whether PDL is active and helps on this rig under WDDM.
- Whether cooperative launch works on the 5060 Ti under WDDM (driver 616.64).
- P2P between the two cards on Linux. With P2P, each all-reduce could skip host memory
  and halve its PCIe crossings. Without it, NVSHMEM and the Hazy TP design do not apply.
- How much of the all-reduce time weight prefetch can hide.

---

## Sources

Local code (read only):

- `llama-rig2\ggml\src\ggml-cuda\ggml-cuda.cu`: 959-1255 (comm context, NCCL/internal choice at 1222-1229), 2613-2721 (graph compatibility, key, update), 3667-4461 (fusions), 4463-4679 (evaluate and capture), 4682-4755 (graph compute), 4904-4918 (`GGML_CUDA_GRAPH_OPT`).
- `...\llama-rig2\ggml\src\ggml-cuda\common.cuh`: 119-144 (PDL defines), 1259-1295 (graph struct, env var), 1460-1510 (graph map, eviction), 1609-1717 (PDL launch).
- `...\llama-rig2\ggml\src\ggml-cuda\allreduce.cu`: 13-77, 110-202, 235-260, 275-302, 381-397, 771-985.
- `...\llama-rig2\ggml\src\ggml-cuda\mmvq.cu`: 1531-1536.
- `...\llama-rig2\ggml\src\ggml-backend-meta.cpp`: 1940-2043, 2155-2766.
- `...\llama-rig2\src\llama.cpp`: 158-220.
- `...\llama-rig2\src\llama-context.cpp`: 734-745, 1318-1340, 1438-1555, 2562-2568.
- `...\llama-rig2\src\llama-kv-cache.cpp`: 1260-1274, 1567.
- `...\llama-rig2\src\llama-graph.cpp`: 982.
- `...\llama-rig2\src\models\qwen35.cpp`: 138-240, 269-483, 501-864.
- `...\llama-rig2\src\models\delta-net-base.cpp`: 449-614.
- `...\llama-rig2\common\speculative.cpp`: 1390-2085. `common\sampling.cpp`: 134-161. `common\common.h`: 329-332.
- `llama.cpp` git log: `2f3fd0252` (#28549), `e94722822` (#22522).
- `qwen38_27\arranca.ps1`, `LEEME.md` (lines 55-61, 564-620, 690-720), `arranque.log.err` (lines 1-25).
- `qwen27-engine\refs\Megakernels`: `include/config.cuh`, `include/megakernel.cuh`, `demos/low-latency-llama/*`, `megakernels/llama.py`.
- `...\refs\mirage`: `README.md`, `python/mirage/mpk/persistent_kernel.py:290-380`, `include/mirage/persistent_kernel/persistent_kernel.cuh`.
- `...\refs\calm`: `README.md`, `src/infer.cu:80, 332-402, 405-626, 722-738`.
- `...\refs\cutlass\include\cutlass\arch\reg_reconfig.h:45-92`.
- `...\refs\vllm\vllm\distributed\device_communicators\custom_all_reduce.py:467-500`.

Web:

- Hazy Research, "Look Ma, No Bubbles!": https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Hazy Research, "We Bought the Whole GPU...": https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
- Hazy Research, "How Many Llamas Can Dance in the Span of a Kernel?": https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro
- Hazy Research, "One Kernel for All Your GPUs": https://hazyresearch.stanford.edu/blog/2025-09-22-pgl
- MPK paper: https://arxiv.org/abs/2512.22219 and https://arxiv.org/html/2512.22219
- Mirage repo: https://github.com/mirage-project/mirage
- zeux, "LLM inference speed of light": https://zeux.io/2024/03/15/llm-inference-sol/
- alpindale, "Hitting 1,000 tokens per second on a single RTX 5090": https://blog.alpindale.net/posts/5090_decode_optimization/
- Lucebox, "Megakernel: matching Apple Silicon efficiency at 2x throughput on a RTX 3090": https://www.lucebox.com/blog/megakernel
- NVIDIA, "Getting Started with CUDA Graphs": https://developer.nvidia.com/blog/cuda-graphs/
- NVIDIA, "Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements": https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements
- NVIDIA, "Optimizing llama.cpp AI Inference with CUDA Graphs": https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/
- NVIDIA, "Leveling up CUDA Performance on WSL2 with New Enhancements": https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements
- NVIDIA, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes": https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes
- CUDA Programming Guide, PDL: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- CUDA Programming Guide, Cooperative Groups: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- Blackwell Tuning Guide (cc 12.0: 99 KB shared per block, 48 warps per SM): https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUTLASS Blackwell functionality (GeForce: no multicast, cluster 1x1x1): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVSHMEM install guide (Linux, P2P required): https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/
- Nsight VSE, TDR (2 s default): https://docs.nvidia.com/nsight-visual-studio-edition/5.2/Content/Timeout_Detection_Recovery.htm
- llama.cpp PR #22522 (PDL): https://github.com/ggml-org/llama.cpp/pull/22522
- llama.cpp PR #6766 (first CUDA graphs): https://github.com/ggml-org/llama.cpp/pull/6766
- NVIDIA forums, launch overhead (2025): https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194
- NVIDIA forums, kernel launch latency (2018): https://forums.developer.nvidia.com/t/kernel-launch-latency/62455
- NVIDIA forums, very slow kernel launches, WDDM numbers (2015): https://forums.developer.nvidia.com/t/very-slow-kernel-launches/37345
- NVIDIA forums, cooperative launch on WDDM (2024): https://forums.developer.nvidia.com/t/since-when-was-cooperative-launch-now-supported-in-windows-non-tcc-mode/313144
- RTX 5060 Ti specs (36 SMs, 448 GB/s): https://videocardz.com/newz/nvidia-geforce-rtx-5060-ti-final-specs-confirmed-gb206-gpu-16-8gb-gddr7-and-2-57-ghz-boost
