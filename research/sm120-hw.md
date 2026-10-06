# RTX 5060 Ti (GB206, CC 12.0, sm_120 / sm_120f / sm_120a): what the hardware supports

Date: 2026-10-05. Toolchain checked: CUDA 13.4 (`nvcc` V13.4.59, PTX ISA 9.4) in `%USERPROFILE%\\cuda\v13.4`.
No GPU work was done. The "ptxas check" column below means: a tiny PTX file was assembled
with CUDA 13.4 `ptxas` for each target, and the SASS was read with `nvdisasm`. Appendix A has the method.

## Summary

1. The card has 36 SMs, a 2.57 GHz boost clock, 128-bit GDDR7 at 28 Gbps (448 GB/s theoretical, 388-395 GB/s measured on this rig), and 32 MB L2 (L2 size is from third-party sources; NVIDIA does not publish it). Each SM has 64K registers, a 128 KB unified L1/shared array, at most 100 KB of shared memory, 99 KB (101376 B) per block, and 48 warps (1536 threads).
2. Tensor cores are used only through warp-level `mma.sync`. `wgmma` is Hopper only (`sm_90a`). `tcgen05` and Tensor Memory are datacenter Blackwell only. `ptxas` 13.4 rejects all three for `sm_120`, `sm_120f` and `sm_120a`.
3. Dense tensor rates per SM per clock (from NVIDIA whitepaper numbers): FP16 with FP16 accumulate 1024 FLOP, FP16 or BF16 with FP32 accumulate 512 FLOP (half rate, confirmed), TF32 256, INT8 2048 OPS, FP8 2048 (FP16 accumulate), FP4 block-scaled (`kind::mxf4` / `mxf4nvf4`) 4096. On this card that is 94.8 / 47.4 / 23.7 / 189.6 / 189.6 / 379.3 T(FL)OPS. INT4 `mma` has no native unit. `ptxas` splits it into two INT8 `IMMA` instructions.
4. FP4 and FP6 `mma` (`kind::f8f6f4`) and all block-scaled `mma` (`mxf8f6f4`, `mxf4`, `mxf4nvf4`) need `sm_120a` or `sm_120f`. Plain `sm_120` rejects them. FP16, BF16, TF32, INT8, INT4 and legacy FP8 (`.e4m3` / `.e5m2` without `.kind`) work on plain `sm_120`.
5. TMA (`cp.async.bulk.tensor`), bulk copies, `mbarrier` with transaction counts, clusters, distributed shared memory, PDL (`griddepcontrol`), `elect.sync` and `st.bulk` all assemble on plain `sm_120`. TMA multicast assembles, but `ptxas` prints an advisory, and CUTLASS says GeForce has no multicast feature. `setmaxnreg` needs `sm_120a`/`sm_120f`.
6. Non-tensor ALU per SM per clock (CUDA 13.4 Best Practices Guide): FP32 FMA 128, FP16x2 FMA 64 instructions (the same FLOPs as FP32), INT32 add 128, IMAD 64, FP64 2, warp shuffle 32 lanes. Float to int32 conversion is only 2 per clock per SM on CC > 10.0. NVIDIA does not publish a `dp4a` rate. My estimate is at most 64 per clock per SM, which gives about 47 TOPS on this card, a quarter of the INT8 tensor rate.
7. CUDA 13.x: the `sm_120f` family target exists since CUDA 12.9 (PTX 8.8) and covers CC 12.0 and 12.1. CUDA 13.3 fixed a miscompile present since CUDA 12.8 (lost thread reconvergence in nested divergence). The llama.cpp IQ3_S failure under CUDA 13.2 (PR #27902) is real. Its exact root cause is not proven. **CUDA 13.2 is first on PATH on this PC**, so builds must point at 13.4 explicitly.

## 1. RTX 5060 Ti 16 GB specs

| Item | Value | Source / arithmetic |
|---|---|---|
| Chip | GB206, CC 12.0 | NVIDIA CUDA GPUs page lists "GeForce RTX 5060 Ti" under 12.0 [S8]. GB206 die name: Wikipedia table [S17] |
| SMs | 36 | 4608 CUDA cores [S7] / 128 cores per SM [S6 p.10] = 36 |
| Tensor cores | 144 (4 per SM, 5th gen) | 4 per SM [S6 p.10] x 36 |
| Boost / base clock | 2.57 / 2.41 GHz (TechPowerUp metadata: 2572 MHz) | [S7], [S18]. The real clock under load may be higher. Not measured here. |
| L2 cache | 32 MB | Wikipedia [S17]. **NVIDIA does not publish this.** Query `cudaDeviceProp::l2CacheSize` when the GPU is free. |
| L1 / shared array per SM | 128 KB unified | Whitepaper "128 KB of L1/Shared Memory" [S6 p.10]; Programming Guide 13.4 Table 32 "12.x: Unified Data Cache 128 KB" [S1] |
| Shared memory per SM (max carveout) | 100 KB. Carveouts: 0, 8, 16, 32, 64, 100 KB | Programming Guide 13.4 Tables 31, 32 [S1] |
| Shared memory per block | 99 KB with opt-in (= 101376 B). 48 KB without opt-in or for static arrays. 1 KB per block is reserved for the system. | [S1] Table 31 and footnote 3; [S2] section 20.10.3; FlashInfer observed the 101376 B cap on sm_120 (`refs/flashinfer/include/flashinfer/attention/sparse_mla_sm120/kernels/fp8_decode/resources.cuh:17`); PTX static shared max for `sm_120a` is 100 KB (PTX 5.1.7 table) [S5] |
| Registers | 64K x 32-bit per SM (256 KB). Max 255 per thread. Max 64K per block. | [S1] Table 31; whitepaper "256 KB Register File" per SM [S6 p.10] |
| Threads / warps per SM | 1536 threads, 48 warps | [S1] Table 30; Blackwell Tuning Guide "48 for compute capability 12.0" [S4] |
| Blocks per SM | **24 or 32: NVIDIA documents disagree.** Guide 13.4 Table 30 gives 24 for 12.x. Tuning Guide and Guide 12.9 Table 28 give 32. | [S1], [S4], [S2]. Query `maxBlocksPerMultiProcessor`. |
| Memory | 16 GB GDDR7, 128-bit | [S7] |
| Memory data rate | 28 Gbps (TechPowerUp: 1750 MHz x 16; Afterburner on this rig shows 14001 MHz stock) | [S18], [S17]; `qwen38_27/LEEME.md:659-660` |
| Theoretical bandwidth | 448 GB/s | 28 Gbps x 128 bit / 8 = 448 GB/s |
| Measured bandwidth (this rig) | 388 / 395 GB/s copy stock (86.6 % / 88.2 % of 448), 403 / 404 GB/s with memory OC | brief; `qwen38_27/LEEME.md:659-660` |
| PCIe | Card is PCIe 5.0 x8. The rig runs it at Gen3 x4. | Wikipedia [S17]; brief |
| FP32 : FP64 ratio | 64 : 1 | [S1] Table 30 |
| Green contexts | minimum SM partition 8 SMs, alignment 8 (useFlags 0) on 12.x | [S1] Table 30 |

## 2. Tensor cores on sm_120

### 2.1 Instructions, targets, SASS and rates

"Per SM per clock" comes from the whitepaper RTX 5070 table (48 SMs, 2512 MHz). Example: FP16 with FP32 accumulate 61.7 TFLOPS / (48 x 2.512e9) = 512 FLOP/clk/SM [S6 Table 6, p.55]. The RTX 5090 table gives the same per-SM values (419 / (170 x 2.407e9) = 1024) [S6 Table 3].
"5060 Ti dense" = per-SM rate x 36 SMs x 2.572e9 Hz (= 92.59e9 SM-cycles/s). These are **estimates** at the rated boost clock.
Cross-check: FP4 379.3 TFLOPS dense x 2 (sparsity) = 758.6. NVIDIA lists "759 AI TOPS" for this card [S7].

| PTX form | Shape(s) | Lowest target (PTX ISA 9.4) | ptxas check (13.4) | SASS on sm_120a | Dense rate / SM / clk | 5060 Ti dense (est.) |
|---|---|---|---|---|---|---|
| `.f16` inputs, `.f16` accumulate | m16n8k8, m16n8k16 | sm_75 / sm_80 | OK on `sm_120` | (not disassembled) | 1024 FLOP | 94.8 TFLOPS |
| `.f16` inputs, `.f32` accumulate | m16n8k8, m16n8k16 | sm_75 / sm_80 | OK on `sm_120` | `HMMA.16816.F32` | 512 FLOP (half rate) | 47.4 TFLOPS |
| `.bf16`, `.f32` accumulate | m16n8k8, m16n8k16 | sm_80 | OK on `sm_120` | `HMMA.16816.F32.BF16` | 512 FLOP | 47.4 TFLOPS |
| `.tf32`, `.f32` accumulate | m16n8k4, m16n8k8 | sm_80 | (not tested) | | 256 FLOP | 23.7 TFLOPS |
| `.s8`/`.u8`, `.s32` accumulate | m8n8k16, m16n8k16, m16n8k32 | sm_75 / sm_80 | OK on `sm_120` | `IMMA.16832.S8.S8` | 2048 OPS | 189.6 TOPS |
| `.s4`/`.u4`, `.s32` accumulate | m8n8k32, m16n8k32, m16n8k64 | sm_75 / sm_80 | OK on `sm_120` | **2x `IMMA.16832.S8.S8`** (emulated) | no gain over INT8 | same MAC rate as INT8 |
| `.e4m3`/`.e5m2` (no `.kind`), `.f32` accumulate | m16n8k32 | sm_89 | OK on `sm_120` | `QMMA.16832.F32.E4M3.E4M3` | 1024 FLOP (whitepaper) | 94.8 TFLOPS |
| `.e4m3`/`.e5m2` (no `.kind`), `.f16` accumulate | m16n8k16, m16n8k32 | m16n8k16 + `.f16`: PTX 8.7; `mma.sp` notes say sm_120 | OK on `sm_120` | `QMMA.16816.F16.E4M3.E4M3` | 2048 FLOP | 189.6 TFLOPS |
| `.kind::f8f6f4` (`e4m3, e5m2, e3m2, e2m3, e2m1`), `.f16`/`.f32` acc. | m16n8k32 | **sm_120a, or sm_120f from PTX 8.8** | **Rejected on `sm_120`.** OK on `120f`, `120a` | `QMMA.16832.F32.E2M1.E2M1` | FP8 rate. CUTLASS: "1x Ada Fp8 Tensor Core (2x for FP32 accumulator)" | 189.6 TFLOPS (FP4/FP6 here run at the FP8 rate) |
| `.kind::mxf8f6f4.block_scale.scale_vec::1X`, `.ue8m0` scales, `.f32` | m16n8k32 | sm_120a / sm_120f | Rejected on `sm_120`. OK on `120f`, `120a` | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | FP8 rate | 189.6 TFLOPS |
| `.kind::mxf4.block_scale.scale_vec::2X`, `.ue8m0`, `.f32` | m16n8k64 | sm_120a / sm_120f (dense) | Rejected on `sm_120`. OK on `120f`, `120a` | `OMMA.SF.16864.F32.E2M1.E2M1.E8` | 4096 FLOP | 379.3 TFLOPS |
| `.kind::mxf4nvf4.block_scale.scale_vec::{2X,4X}`, `.ue8m0`/`.ue4m3` | m16n8k64 | dense: sm_120a / sm_120f. Sparse (`mma.sp`) with mxf4/mxf4nvf4: **sm_120a / sm_121a only** | Dense rejected on `sm_120`, OK on `120f`, `120a` | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 4096 FLOP | 379.3 TFLOPS |
| `.f64` | m8n8k4, m16n8k{4,8,16} | sm_80 / sm_90 | (not tested) | | "very minimal number of FP64 Tensor Cores ... for program correctness" | negligible |
| `wgmma.*` | m64nNk* | **sm_90a only** | Rejected on `sm_120`, `120f`, `120a` | none | none | none |
| `tcgen05.*` (mma, ld, st, cp, alloc, shift) and Tensor Memory | | **sm_100a/f, sm_103a/f, sm_110a/f, sm_107a/f only** | `tcgen05.alloc` rejected on `sm_120`, `120f`, `120a` | none | none | none |

Sources for the table: PTX ISA 9.4 sections 9.7.16.1 (shapes), 9.7.16.2 (types), 9.7.16.3 (block scaling), 9.7.16.5.14 (`mma` target notes), `mma.sp` target notes, 9.7.17.5.2 (`wgmma` "Requires sm_90a"), 9.7.18 and Table 72 (`tcgen05` targets) [S5]. Whitepaper Tables 3 and 6 [S6]. CUTLASS `blackwell_functionality.md:654-676` [S13]. Programming Guide 13.4 Table 33: CC 12.x tensor input types TF32, BF16, FP16, FP8, FP6, FP4, INT8. FP64 and INT4 are not marked [S1].

### 2.2 Answers to the specific questions

- **FP16 with FP32 accumulate is half rate on GeForce: yes.** Whitepaper RTX 5070: FP16 tensor 123.5 TFLOPS with FP16 accumulate and 61.7 with FP32 accumulate. BF16 is 61.7 (it has only FP32 accumulate). The RTX 5090 table shows the same 2:1 ratio (419 vs 209.5) [S6].
- **FP8 with FP32 accumulate: the sources disagree.** The whitepaper gives half rate (RTX 5070: 246.9 vs 123.5 TFLOPS) [S6]. CUTLASS says `kind::f8f6f4` runs at "1x Ada Fp8 Tensor Core (2x for FP32 accumulator)" [S13 line 663]. That suggests FP32 accumulate at full rate. A microbenchmark is needed.
- **Fast FP4 needs block scaling.** FP4 through `kind::f8f6f4` compiles to `QMMA` and runs at the FP8 rate. Only `kind::mxf4` and `kind::mxf4nvf4` compile to `OMMA` and reach 4096 FLOP/clk/SM. This matches CUTLASS ("2x Ada Fp8 ... 4x for FP32 accumulator") [S13 lines 665-666].
- **Layouts:** the FP8/FP6/FP4 `mma` forms are `.row.col` only (A row-major, B column-major). CUTLASS SM120 GEMMs support only the TN layout [S13 line 676].
- **ldmatrix/stmatrix:** `ldmatrix .m8n8 .b16` and `stmatrix .m8n8` work on plain `sm_120`. `ldmatrix .m16n16 .b8`, `.m8n16` with sub-byte sources, and `stmatrix .m16n8 .b8` need `sm_120a` or `sm_120f` (ptxas check). PTX 9.4 adds `ldmatrix .m8n16 .s8.s4`, which sign-extends 4-bit values to 8-bit on load. It needs `sm_120f` [S5 Table 72].
- **llama.cpp uses `kind::mxf4` and `kind::mxf4nvf4`** for MXFP4/NVFP4 (`llama.cpp/ggml/src/ggml-cuda/mma.cuh:1131-1150`). This is why it builds `120a-real` (`llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:40-55`). `BLACKWELL_MMA_AVAILABLE` is set for `__CUDA_ARCH__` 1200 to 1299 (`common.cuh:296-298`). Our model file has no MXFP4/NVFP4 tensors, so this path is unused for it.

## 3. Async copy, clusters, PDL, graphs, L2 control

"Plain sm_120" means it assembled with `.target sm_120` in the ptxas check. Docs give the minimum target.

| Feature | PTX / API | Documented minimum | Plain sm_120? | Notes |
|---|---|---|---|---|
| TMA tensor copy, tile mode | `cp.async.bulk.tensor.Nd.shared::cluster.global.tile.mbarrier::complete_tx::bytes` | sm_90 [S5 9.7.10.28.5.3] | **Yes** (ptxas OK) | Table 29: TMA "Yes" for 12.x [S1] |
| TMA modes `.tile::gather4`, `.tile::scatter4`, `.im2col::w`, `.im2col::w::128`, `.cta_group` | same | sm_100a / sm_100f / sm_110f only [S5] | **No** (not on any sm_120 target) | |
| TMA multicast `.multicast::cluster` | same | sm_90 (advised only for sm_90a/100a/100f/103/107/110) [S5] | Assembles, with advisory: "should be used on .target sm_90a/sm_100a/... instead of .target 'sm_120a' as this feature is expected to have substantially reduced performance" | CUTLASS: "On Geforce series graphics card, there is no multicast feature therefore the cluster shape is fixed to 1x1x1" [S13 line 672]. Treat as unavailable. |
| TMA sub-byte types into `.shared::cluster`, `.swizzle_atomicity` | `cp.async.bulk.tensor` | | Not supported on sm_120a | PTX 9.7.10.28.5.1 "restrictions for sm_120a" [S5] |
| Bulk (non-tensor) copy, bulk reduce, bulk prefetch | `cp.async.bulk`, `cp.reduce.async.bulk`, `cp.async.bulk.prefetch(.tensor)` | sm_90 | **Yes** | `.weak/.relaxed/.scope/.type` qualifiers on `cp.async.bulk` need sm_90a or sm_100f. They are not available on sm_120 [S5]. |
| `cp.async` (LDGSTS) | `cp.async`, `cp.async.mbarrier.arrive` | sm_80 | Yes | |
| mbarrier with transaction count | `mbarrier.init`, `mbarrier.arrive.expect_tx`, `try_wait` | init sm_80, `expect_tx` sm_90 | **Yes** | Table 29: "Hardware-accelerated Split Arrive/Wait Barrier: Yes" [S1] |
| Thread block clusters | `barrier.cluster`, `%cluster_*`, launch attribute | sm_90 | **Yes** | Table 29: "Thread Block Cluster: Yes" for 12.x [S1] |
| Distributed shared memory | `mapa`, `ld/st.shared::cluster` | sm_90 | **Yes** | Table 29 "Distributed Shared Memory: Yes" [S1] |
| Max cluster size | `cudaOccupancyMaxPotentialClusterSize` | Portable max 8. Non-portable 16 is stated only for B200 [S4]. | **Unknown for GB206.** | A forum user says "consumer blackwells support clusters of 8 blocks" [S20]. A non-NVIDIA wiki claims size 1 only [S21]. That claim contradicts Table 29. SGLang disables clusters on SM120 ("SM120 has poor cluster performance", `refs/sglang/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh:32`). FlashInfer and vLLM SM120 GEMMs use 1x1x1 only (`refs/flashinfer/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:198`, `refs/vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh:102`). **Query and measure before use.** |
| Programmatic dependent launch | `griddepcontrol.wait`, `griddepcontrol.launch_dependents`; `cudaLaunchAttributeProgrammaticStreamSerialization` | sm_90 / CC 9.0 [S5 9.7.15.14], [S11] | **Yes** | The llama.cpp rig branch already uses it (`llama-rig2/ggml/src/ggml-cuda/common.cuh:123-144, 1609-1625`). |
| `elect.sync`, `st.async`, `red.async` | | sm_90 | Yes | `.mmio`, `.release`, `.global` forms need sm_100+, which includes sm_120 [S5] |
| `st.bulk` | | sm_100 | Yes (baseline) | [S5] |
| Cluster launch control (work stealing) | `clusterlaunchcontrol.try_cancel` | sm_100 | Yes | `.multicast::cluster::all` form needs sm_120a/sm_120f [S5] |
| `setmaxnreg` (warp-specialized register rebalancing) | `setmaxnreg.inc/dec` | sm_90a, sm_100a/f, sm_120a/f | **No on plain sm_120**. OK on `120a`/`120f`. | ptxas check |
| `tensormap.replace` (edit TMA descriptors on device) | | sm_90a, sm_100a/f, sm_120a/f | No on plain sm_120 | [S5 Table 72] |
| CUDA graphs: conditional nodes (IF / WHILE / SWITCH) | `cudaGraphConditionalHandle` | The guide states no compute-capability limit | Yes (by docs) | Body graphs may contain only kernel, empty, memcpy, memset, child-graph and conditional nodes [S12 4.2.4] |
| CUDA graphs: device graph launch | `cudaGraphInstantiateFlagDeviceLaunch` | No compute-capability limit stated | Yes (by docs) | Device graphs may contain only kernel, memcpy, memset and child-graph nodes. CUDA Dynamic Parallelism is not allowed inside [S12 4.2.6] |
| L2 persistence | `cudaLimitPersistingL2CacheSize`, `cudaAccessPolicyWindow` (stream or graph kernel node attribute) | CC 8.0+ [S10] | Yes (by docs). No GeForce exclusion is stated. | The set-aside size is capped by `persistingL2CacheMaxSize`. **Not queried yet on this card.** It is disabled under MIG. Under MPS it is set only at server start [S10]. |

## 4. What datacenter Blackwell (sm_100, B200) has that sm_120 lacks

1. `tcgen05.*` instructions and Tensor Memory. On sm_100a/f, TMEM is 512 columns x 128 lanes x 32 bit per CTA (256 KB) [S5 9.7.18.1]. ptxas rejects `tcgen05.alloc` on all sm_120 targets.
2. 2-SM (`cta_group::2`) MMA and the matching TMA `.cta_group` qualifier [S5 Table 72].
3. Shared memory: 228 KB per SM and 227 KB per block on 10.0, against 100 KB and 99 KB on 12.x [S1 Table 31], [S4].
4. Occupancy: 64 warps (2048 threads) per SM on 10.0, against 48 warps (1536 threads) [S4].
5. Non-portable cluster size 16 ("NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16") [S4]. TMA multicast performance (advised only for sm_90a/sm_100*). GeForce has no multicast feature [S13].
6. TMA modes `.tile::gather4`, `.tile::scatter4`, `.im2col::w`, `.im2col::w::128` [S5].
7. Packed FP32 math: `fma/add/mul.f32x2` compile to one `FFMA2` on sm_100 and to two `FFMA` on sm_120 (ptxas + nvdisasm check).
8. `redux.sync` on `.f32` with `.abs` / `.NaN` (sm_100a/f only; ptxas rejects it on sm_120a) [S5 9.7.15.13].
9. FP64: 64 FP64 FMA per clock per SM on 10.0 against 2 on 12.0. FP32:FP64 ratio 2:1 against 64:1 [S3], [S1 Table 30]. FP64 tensor cores (Table 33 marks FP64 for 10.0, not for 12.x) [S1].
10. FP32 min/max at 128 per clock per SM on 10.0/10.3, against 64 on 12.0 [S3].
11. Native DPX instructions (Table 29: "Native" for 9.0 and 10.x, "Multiple Instr." for 12.x) [S1].
12. `cp.async.bulk` `.weak/.relaxed/.scope/.type` qualifiers and `cvt` `.rs` rounding [S5].
13. System level (not relevant to this rig): HBM, NVLink 5, `multimem` on NVLink multicast objects.

Neither sm_100 nor sm_120 has `wgmma`. It requires `sm_90a` [S5 9.7.17.5.2].

## 5. Integer and non-tensor throughput

NVIDIA source: CUDA 13.4 Best Practices Guide, 12.1.1 "Throughput of Native Arithmetic Instructions", column "12.0 / 12.1" [S3]. Units are operations (thread-lanes) per clock per SM. A 2-way SIMD instruction counts once per lane. 5060 Ti numbers multiply by 92.59e9 SM-cycles/s (36 SMs x 2.572 GHz), so they are **estimates** at rated boost.

| Operation | Ops/clk/SM on 12.0 | 5060 Ti (est.) | Note |
|---|---|---|---|
| FP32 add/mul/FMA (`add.f32`) | 128 | 23.7 TFLOPS (FMA = 2 FLOP) | Whitepaper: unified FP32/INT32 cores. Each core does FP32 or INT32 in a given clock [S6 p.12] |
| FP16x2 add/mul/FMA (`add.f16x2`) | 64 | 23.7 TFLOPS | Same FLOPs as FP32. Whitepaper RTX 5070 also gives FP16 non-tensor = FP32 (30.9 = 30.9) [S6]. The older 12.9 Guide Table 7 said 256 results; the 13.4 BPG and the whitepaper agree on the lower value. |
| BF16x2 | not listed separately for 12.0 | | Whitepaper: BF16 non-tensor = FP32 rate [S6] |
| FP64 FMA | 2 | 0.37 TFLOPS | |
| INT32 add/sub | 128 | 11.9 T ops/s | |
| INT32 mul / IMAD | 64 | 5.9 T ops/s | |
| INT32 min/max | 128 | | |
| Shift, AND/OR/XOR | 64 | | |
| Warp shuffle | 32 (one warp instruction per clock per SM) | 92.6e9 warp shuffles/s | matters for warp reductions in GEMV kernels |
| Warp vote | 128 | | |
| popc, clz | 16 | | |
| f32 to f16 / bf16 conversion | 16 | | footnote 10 [S3] |
| f16 to f32 conversion | 64 | | |
| **f32 to int32 conversion** | **2** | | footnote 11: "2 for float->32-bit integer type (SM > 10.0)" [S3]. Use the add-magic-number trick (`x + 12582912.0f`, then reinterpret bits) when quantizing activations to int8. |
| `dp4a` (`IDP.4A.S8.S8` in SASS) | **not documented** | estimate: at most 64 instr/clk/SM, about 47 TOPS | Nsight Compute: integer dot products run on the "FMA Heavy" pipe only, not "FMA Lite" [S14]. IMAD on the same pipe is 64/clk/SM [S3]. Estimate: 64 x 4 MAC x 2 ops x 92.59e9 = 47.4 TOPS. That is 1/4 of INT8 `mma` (189.6 TOPS). Measure. |

What this means for decode (**estimate**). Main model without `token_embd` and MTP has 25.6e9 parameters (summed from `research/_gguf-model-tensors.tsv`). One MTP verify step with 4 tokens is 4 x 25.6e9 = 102e9 MAC. With `-sm tensor` each card does half, 51e9 MAC. With `dp4a` at the 23.7e12 MAC/s estimate that is about 2.2 ms. The weight read per card is 5414 MiB at 390 GB/s, about 14.6 ms. So the `dp4a` path costs about 15 % of the read time at batch 4, before dequantization instructions. INT8 `mma` cuts the MAC part to about 0.5-1.1 ms (1.1 ms if 4 tokens are padded to n=8).

PTX 9.2 added SIMD byte ops (`add/sub/min/max/neg .s8x4/.u8x4`, `add.sat .u16x2/.s16x2`). They need `sm_120f` (ptxas: `add.s8x4` rejected on plain `sm_120`, OK on `120f`/`120a`) [S5 Table 72]. They may help sign handling in i-quant dequantization. Their rate is not documented.

## 6. CUDA 13.x items that matter for sm_120 kernels

1. **Target choice.**
   - `sm_120`: baseline features only.
   - `sm_120f` (since CUDA 12.9 / PTX 8.8): adds family features. Runs on CC 12.0 and 12.1 [S1 Table 28], [S9].
   - `sm_120a` (since PTX 8.7): the full arch-specific set. It is a superset of `f` and runs only on CC 12.0 [S1 5.1.2.3].
   - The only `a`-only features relevant to us are sparse `mma.sp` with `mxf4`/`mxf4nvf4`, `cvt` `.s2f6x2`, and `multimem` FP8 types.
   - For one rig, `-gencode arch=compute_120a,code=sm_120a` is the simplest choice. llama.cpp on this rig uses `-DCMAKE_CUDA_ARCHITECTURES=120a-real` (`qwen38_27/LEEME.md:645`).
   - CMake accepts the string `120f` only from 3.31.8 (below 4.0) or from 4.0.2. `120a-real` works with any CMake (`llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:41-51`).
   - Device code can test `__CUDA_ARCH_SPECIFIC__`, `__CUDA_ARCH_FAMILY_SPECIFIC__` and `__CUDA_ARCH_FEAT_SM120_ALL` (`cuda/v13.4/include/crt/host_defines.h:296-308`, `cuda/v13.4/include/cccl/nv/detail/__target_macros:527`).
2. **New sm_120f PTX in 13.1-13.4.**
   - PTX 9.1: `cvt` from `.f16x2` and `.bf16x2` to `e2m1x2/e2m3x2/e3m2x2` (and `.e4m3x2/.e5m2x2` from `.bf16x2`).
   - PTX 9.2: `cvt` from those types to `.bf16x2`, plus the SIMD byte ops above.
   - PTX 9.4: `ldmatrix .m8n16 .s8.s4` [S5 Table 72].
   - CUDA 13.4 ships PTX ISA 9.4. Its compiler "Known Issues" list is empty [S9].
3. **Compiler bug fixed in CUDA 13.3**, "present since CUDA 12.8". "Compiler-inserted thread reconvergence" could fail and "leave stale or corrupted values in registers". It affects kernels with "two or more nested levels of thread divergence" [6156910] [S15]. CUDA 12.8 through 13.2 have this bug. Use 13.3 or newer.
4. **llama.cpp PR #27902** (open, not merged) [S16].
   - Reported on an RTX PRO 6000 Blackwell (CC 12.0) with CUDA 13.2. IQ3_S `MUL_MAT` passed 0/11 tests. IQ1_S, IQ2_S and IQ3_S MMQ/MMVQ results were wrong.
   - The author blames "the 2-byte-aligned packed-load path in `get_int_b2`" and also removes a signed-shift undefined behaviour. The current code is `x16[2*i32+1] << 16` on a promoted `int` (`llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:18-25`).
   - On this rig, CUDA 13.4 gives correct output without the PR (perplexity matches the official build, `qwen38_27/LEEME.md:620-623`).
   - The exact root cause is **not proven**. It could be the 13.3-fixed reconvergence bug, the undefined behaviour, or both.
   - For our engine: use CUDA 13.4. Write packed loads with unsigned arithmetic or `__byte_perm`. Keep a bit-exact unit test per quant type against a CPU reference.
5. **CUDA 13.2 is first on PATH on this PC.** `which nvcc` gives `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc` (V13.2.51). The old build dir `llama.cpp/build-cuda/CMakeCache.txt` uses that compiler with `CMAKE_CUDA_ARCHITECTURES=120`. The engine build must set `CMAKE_CUDA_COMPILER` (or `CUDAToolkit_ROOT`) to `%USERPROFILE%\\cuda\v13.4` explicitly, and should fail if the version is below 13.3. On Ubuntu, pin the toolkit path the same way.
6. **cuBLAS issues on CC 12.x** (only if we call cuBLAS, for example for the BF16 vision encoder) [S9]:
   - Strided-batched GEMM with broadcast operands could read out of bounds on CC 12.0/12.1.
   - GEMM kernels on CC 10.x/12.x could read device alpha/beta before `cudaGridDependencySynchronize()`, which is a WAR hazard with PDL.
   - GEMV-like ops gave wrong results on CC 12.x when transposed, the non-accumulation dimension was above 3,145,680, and beta was not 0.
   - These were listed as known issues from 13.0 Update 2 to 13.3 Update 1. They are not in the 13.4 known-issues list. The 13.4 notes point to the separate cuBLAS 13.6.1 patch release.
   - In 13.0 Update 2, "FP8 matmuls may fail to launch on multi-device Blackwell GeForce systems". It was fixed in 13.1 [CUB-9487].
7. **CUDA Tile (13.3) known issue:** "Declaring an f16 constant, converting it to an FP8 type, and then printing it on sm_120 can cause a compiler crash" [S15]. This affects CUDA Tile only.
8. The `wgmma` copy-propagation miscompile (13.2 known issue, fixed in 13.3) affects only `sm_90a` [S15]. It is not relevant to sm_120.

## 7. Consequences for the engine design (short)

- Kernels will use `mma.sync` with `ldmatrix`, fed by `cp.async` or TMA, and synchronized with `mbarrier`. The register file holds the accumulators. There is no TMEM, `tcgen05` or `wgmma`.
- INT8 `mma` (189.6 TOPS est.) is the fastest exact path for i-quant weights after dequantization to int8. FP16 `mma` with FP16 accumulate (94.8) is twice FP32 accumulate (47.4), but FP16 accumulate has precision risk.
- 99 KB of shared memory per block and 48 warps per SM limit tile size and pipeline depth. Tiles sized for Hopper or B200 (227 KB) will not fit.
- Decode is memory bound. Bandwidth is 448 GB/s theoretical and about 390-404 GB/s measured. TMA bulk loads and PDL are available to overlap kernel tails. Multicast is not available. Cluster use is unproven on this chip.

## 8. Open items to measure when the GPU is free (no load, a few seconds each)

1. `cudaGetDeviceProperties`: `l2CacheSize`, `persistingL2CacheMaxSize`, `accessPolicyMaxWindowSize`, `maxBlocksPerMultiProcessor` (24 or 32), `sharedMemPerBlockOptin` (expect 101376), `clockRate`.
2. `cudaOccupancyMaxPotentialClusterSize` for a trivial kernel, with and without `cudaFuncAttributeNonPortableClusterSizeAllowed`.
3. Microbenchmarks: FP8 `mma` with FP32 accumulate (whitepaper half rate against CUTLASS full rate), `dp4a` rate, `HFMA2` rate, the real sustained SM clock under load.

## Appendix A. ptxas check method

- For each feature, a PTX file with `.version 9.4`, `.target {sm_120 | sm_120f | sm_120a}` and one instruction was assembled with `%USERPROFILE%\\cuda\v13.4\bin\ptxas.exe -arch=<same target>`.
- For SASS, inputs were loaded from global memory and results stored, so the instruction is not removed. The cubin was disassembled with `nvdisasm` from the CUDA 13.2 install (disassembly only, no code generation).
- Results:

| Test | sm_120 | sm_120f | sm_120a |
|---|---|---|---|
| mma f16 (f32 acc), bf16, s8, s4, e4m3 k32 f32, e4m3 k16 f16 | OK | OK | OK |
| mma `kind::f8f6f4` e2m1; `mxf8f6f4`; `mxf4`; `mxf4nvf4` (dense) | error | OK | OK |
| `cvt.rn.satfinite.e2m1x2.f32`; `add.s8x4`; `ldmatrix .m16n16 .b8`; `ldmatrix .m8n16 .s8.s4` | error | OK | OK |
| `setmaxnreg.inc` | error | OK (info C7508 in the toy kernel) | OK (same) |
| TMA 2D tile load; `cp.async.bulk`; `mbarrier.expect_tx`; `barrier.cluster` + `mapa` + `ld.shared::cluster`; `griddepcontrol`; `elect.sync`; `stmatrix .m8n8`; `dp4a` | OK | OK | OK |
| TMA 2D with `.multicast::cluster` | OK + advisory | OK + advisory | OK + advisory |
| `wgmma.fence`; `tcgen05.alloc`; `redux.sync.min.f32` | error | error | error |
| `fma.rn.f32x2` | OK, 2x `FFMA` | | (sm_100: 1x `FFMA2`) |

## Sources

- [S1] CUDA Programming Guide 13.4, 5.1 Compute Capabilities (Tables 28-33, 5.1.2.x feature sets): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- [S2] CUDA C++ Programming Guide 12.9.1 (archive), 20.10 "Compute Capability 12.0", Table 7, Table 28: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html
- [S3] CUDA C++ Best Practices Guide 13.4, 12.1.1 "Throughput of Native Arithmetic Instructions": https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- [S4] NVIDIA Blackwell Tuning Guide (occupancy, shared memory, clusters): https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- [S5] PTX ISA 9.4: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html. Sections used: 5.1.7 (static shared memory per target), 9.7.1.24 `dp4a`, 9.7.10.24 `cvt`, 9.7.10.28.4-5 bulk/tensor copy and 9.7.10.28.5.1 restrictions, 9.7.15.13 `redux.sync`, 9.7.15.14 `griddepcontrol`, 9.7.15.16 `mbarrier`, 9.7.15.18 `clusterlaunchcontrol`, 9.7.16.1-9.7.16.5.16 `mma` / `ldmatrix` / `stmatrix`, 9.7.17 `wgmma`, 9.7.18 `tcgen05` and Tensor Memory, 9.7.21.5 `setmaxnreg`, Table 72 (release history of arch- and family-specific instructions), release-notes table of PTX versions.
- [S6] NVIDIA RTX Blackwell GPU Architecture whitepaper: p.10 (SM: 128 cores, 4 tensor cores, 256 KB RF, 128 KB L1/shared), p.12 (unified INT32/FP32), Table 3 (RTX 5090), Table 6 p.54-55 (RTX 5070, GB205): https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- [S7] NVIDIA GeForce RTX 5060 family product page (4608 cores, 2.57/2.41 GHz, 128-bit GDDR7, 759 AI TOPS, 180 W): https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/
- [S8] NVIDIA CUDA GPUs compute capability list: https://developer.nvidia.com/cuda-gpus
- [S9] CUDA Toolkit 13.4 Update 1 release notes (compiler, cuBLAS 13.0-13.4): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- [S10] CUDA Programming Guide 13.4, 4.14 L2 Cache Control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- [S11] CUDA Programming Guide 13.4, 4.5 Programmatic Dependent Launch: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- [S12] CUDA Programming Guide 13.4, 4.2 CUDA Graphs (4.2.4 conditional nodes, 4.2.6 device graph launch): https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- [S13] CUTLASS `media/docs/cpp/blackwell_functionality.md`, section "Blackwell SM120 GEMMs", lines 654-676 (commit 1b5b7d3, 2026-09-03): https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
- [S14] Nsight Compute Profiling Guide, pipeline descriptions (`fma`, `fmaheavy`, `fmalite`): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- [S15] CUDA 13.3.0 release notes (2.4.1 reconvergence fix [6156910], CUDA Tile known issue, WGMMA fix): https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html ; CUDA 13.2.1 notes (WGMMA known issue): https://docs.nvidia.com/cuda/archive/13.2.1/cuda-toolkit-release-notes/index.html
- [S16] llama.cpp PR #27902 "Blackwell IQ Failures issue Fixes" (open): https://github.com/ggml-org/llama.cpp/pull/27902
- [S17] Wikipedia, GeForce RTX 50 series, desktop table (third party; L2 32 MB, 28 Gbps, 448 GB/s, PCIe 5.0 x8): https://en.wikipedia.org/wiki/GeForce_RTX_50_series
- [S18] TechPowerUp GPU database page metadata (third party; "GB206, 2572 MHz, 4608 Cores, ... 1750 MHz, 128 bit"): https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292
- [S19] NVIDIA blog, family-specific architecture features (CUDA 12.9): https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- [S20] NVIDIA Developer Forums, "Thread block clustering in Blackwell GPUs" (user statement, not NVIDIA staff): https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471
- [S21] Third-party wiki claiming no clusters on SM 12.0 (unverified, contradicts [S1]): https://0xsero.github.io/blackwell-gpu-wiki/blackwell/thread-block-clusters/
- Local code:
  - `llama.cpp\ggml\src\ggml-cuda\common.cuh:58-62, 296-298, 376-379`
  - `...\ggml-cuda\mma.cuh:1126-1153`
  - `...\ggml-cuda\CMakeLists.txt:40-55`
  - `...\ggml-cuda\vecdotq.cuh:18-25`
  - `...\ggml-cuda\mmvq.cu:334-345`
  - `llama-rig2\ggml\src\ggml-cuda\common.cuh:123-144, 1609-1625`
  - `qwen38_27\LEEME.md:620-623, 645, 659-660`
  - refs: `flashinfer/.../fp8_decode/resources.cuh:17`, `flashinfer/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:198`, `sglang/.../deepseek_v4/topk_impl.cuh:32`, `vllm/.../scaled_mm_sm120_fp8_dispatch.cuh:102`
