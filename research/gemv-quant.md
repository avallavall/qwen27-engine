# Decode GEMV for the 11 weight types, weight layout, output layer, GPU sampling

Scope: matrix-vector kernels for 1 to 4 tokens (MTP drafts run 1 token, the verify pass runs 1 + drafts, usually 4).
All per-card numbers assume the current `-sm tensor` split (half the rows or half of K per card).
"R" below means the achievable read bandwidth of one card. The only measured value is copy: 388-395 GB/s stock, 403-404 GB/s with memory OC (brief). I use R = 390 GB/s. The read-only rate is not measured yet.

## Summary

1. Three types carry 82% of the main GEMV bytes: IQ3_S 34%, IQ4_XS 28%, IQ3_XXS 20%. All IQ types decode through a codebook (1-8 KB table), sign bits and a 4-bit or 6-bit scale. Q4_K (incl. output head), Q6_K (MTP block) and Q2_K are affine. BF16 is only `ssm_alpha`/`ssm_beta` (48 rows each).
2. llama.cpp `mul_mat_vec_q` (MMVQ) quantizes the input to q8_1 in a separate kernel for every matmul, then uses `dp4a`. Weight loads are 8/16/32-bit, never 128-bit. I compiled its inner loop for sm_120 and counted SASS: 4.0-5.8 instructions per weight at 1 column and 5.5-12.0 at 4 columns. At 4 columns the IQ2, IQ3 and Q2_K loops are near or below an ALU ceiling of roughly 205-390 GB/s (estimate). So the verify pass is probably ALU/LSU-limited for those types.
3. llama.cpp fuses gate+up+SiLU only for 1 column and only when gate and up have the same type (25 of 65 layers here). The verify pass (4 columns) gets no MMVQ fusion at all.
4. To match llama.cpp logits closely the new engine must copy: q8_1 quantization of every GEMV input (changing it moves outputs by ~0.6% of RMS), the fp16 storage of the q8_1 scale (~1.7e-4), and the integer truncations in the IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS dot products (~3e-5). Summation order does not need copying.
5. Lossless repacks that only move bits (split fields into separate arrays, 16-byte alignment, row interleave) cost 0 bytes and allow coalesced 128-bit loads. I verified bit-identical dequantized values on real tensors of all 10 quantized types. Unpacking scales costs +1.5% to +2.8% bytes. One common value format for all types is not worth it (+24% to +340% bytes).
6. Output layer (Q4_K, 682 MiB): ~1.0 ms per pass per card with the vocab split, 4 passes per step (3 drafts + verify). llama.cpp copies all logits to the CPU and samples there (~1 ms of PCIe per step plus CPU sort time). On the GPU, top-20 can be kept as a running list inside the output GEMV; the rest of sampling and the MTP rejection test work on at most 20 candidates. Estimate: 15-50 µs per pass, most of it the cross-card exchange.
7. A good kernel should reach about 88-92% of R for most types (estimate). The verify pass then reads its 5.68 GB per card in ~16.1-16.5 ms, against 14.6 ms at 100% of R. Three drafts add ~4.4 ms. GEMV quality alone is worth a few ms per step. The rest of the 37.7 ms is elsewhere (launch count, all-reduce, CPU sampling).

---

## 1. The weight types in this file

### 1.1 Formats

All IQ and K types use super-blocks of 256 weights along K. Sizes are from `ggml-common.h` (static_asserts next to each struct). MiB are from `research/_gguf-model-tensors.tsv` ("main" = blk.0-63).

| Type | Bytes / 256 w | bpw | MiB in file | Fields per 256 weights | Value | Codebook | Signs | Scale granularity |
|---|---|---|---|---|---|---|---|---|
| IQ1_M | 56 | 1.75 | 18.6 main (1 tensor) | `qs[32]` low 8 bits of index, `qh[16]` 3 high bits + 1 shift bit per 8 w, `scales[8]` = four u16, each with four 3-bit scales + 4 bits of the fp16 super-scale | d·(2s+1)·(g+δ), g∈{-1,0,1}, δ=±0.125 per 8 w | 2048 × 8 values (`iq1s_grid_gpu`, 8 KB) | none (grid has ±1) | 3 bits per 16 w |
| IQ2_XXS | 66 | 2.0625 | 94.1 | `d`, 32 × u16: per 32 w, 4 bytes of 8-bit indices + one u32 with four 7-bit sign fields and a 4-bit scale | d·(0.5+s)·0.25·g·sign | 256 × 8 from {8,25,43} (2 KB) | 7 bits + parity per 8 w | 4 bits per 32 w |
| IQ2_XS | 74 | 2.3125 | 211.0 | `d`, 32 × u16 (9-bit index + 7-bit signs), `scales[8]` | same | 512 × 8 (4 KB) | 7 bits + parity | 4 bits per 16 w |
| IQ2_S | 82 | 2.5625 | 374.8 main + 388.4 `token_embd` | `d`, `qs[64]` (32 B low index bits, 32 B explicit signs), `qh[8]` (2 high bits per index), `scales[8]` | same | 1024 × 8 (8 KB) | 8 explicit bits | 4 bits per 16 w |
| IQ3_XXS | 98 | 3.0625 | 1979.1 | `d`, `qs[96]`: 64 B of 8-bit indices + 8 u32 (four 7-bit signs + 4-bit scale) | d·(0.5+s)·0.5·g·sign | 256 × 4 from {4,…,62} (1 KB) | 7 bits + parity | 4 bits per 32 w |
| IQ3_S | 110 | 3.4375 | 3474.0 | `d`, `qs[64]` low 8 bits, `qh[8]` 1 high bit per index, `signs[32]`, `scales[4]` | d·(1+2s)·g·sign | 512 × 4 from {1,3,…,15} (2 KB) | 8 explicit bits | 4 bits per 32 w |
| IQ4_XS | 136 | 4.25 | 2882.0 | `d`, `scales_h` (u16, 2 bits × 8), `scales_l[4]` (4 bits × 8), `qs[128]` nibbles | d·(ls−32)·kv[q] | 16 int8 (`kvalues_iq4nl`) | in table | 6 bits per 32 w |
| Q2_K | 84 | 2.625 | 229.7 | `scales[16]` (4-bit scale + 4-bit min), `qs[64]`, `d`, `dmin` | d·sc·q − dmin·m | linear | — | per 16 w |
| Q4_K | 144 | 4.5 | 826.9 main + 682.0 `output.weight` | `d`, `dmin`, `scales[12]` (6-bit scale + 6-bit min × 8), `qs[128]` | d·sc·q − dmin·m | linear | — | per 32 w |
| Q6_K | 210 | 6.5625 | 332.2 (MTP block) | `ql[128]`, `qh[64]`, `scales[16]` int8, `d` | d·sc·(q−32) | linear | — | per 16 w |
| BF16 | 2 per w | 16 | 45.0 (96 tensors, 48 rows each) | — | float | — | — | — |

Sources: structs `ggml-common.h:298-460`; tables `ggml-common.h:509-1650`; dequant formulas `dequantize.cuh:128-266` (K types) and `dequantize.cuh:274-436` (IQ types); `IQ1M_DELTA` `ggml-common.h:1133`.

Points that matter for kernels:

- **Alignment.** IQ2_XXS (66 B), IQ2_XS (74), IQ2_S (82), IQ3_XXS (98), IQ3_S (110) and Q6_K (210) blocks are only 2-byte aligned. llama.cpp therefore reads them with `get_int_b2`, two 16-bit loads per 32-bit word (`vecdotq.cuh:18-25`). Q4_K (144 B), IQ4_XS (136 B), Q2_K (84 B) and IQ1_M (56 B) allow 4-byte loads.
- **Codebooks live in global memory.** On CUDA the tables are `static const __device__` arrays (`ggml-common.h:493`). Lookups are LDG gathers through L1. All codebooks together are ~26 KB, so they fit in shared memory.
- **Signs.** IQ2_XXS, IQ2_XS and IQ3_XXS store 7 sign bits per 8 weights. The 8th bit makes the count of minus signs even. llama.cpp recovers it with `popc` (`vecdotq.cuh:97-104`).
- **Grid values are never 0** for IQ2_*/IQ3_* (smallest magnitudes 8, 4, 1). This allows a cheap sign trick (section 7.2).
- **Float exactness.** For all IQ types the dequantized value has at most 24 significant bits (for example IQ4_XS: 11-bit fp16 mantissa × 6-bit scale × 7-bit table value), so fp32 products are exact in any order. For Q2_K/Q4_K (min subtraction) and Q6_K (d·sc·(q−32) can need 25 bits) the fp32 result depends on operation order.
- **Scale bit packing** costs ALU: Q4_K has 6-bit scales/mins spread over 12 bytes; IQ4_XS splits 6-bit scales over two fields; IQ1_M hides its fp16 super-scale in the top 4 bits of four u16 words.

### 1.2 ALU cost per weight (llama.cpp kernels, measured from SASS)

Method: I wrote a probe kernel with the same inner loop as `mul_mat_vec_q` (`mmvq.cu:721-761`, 4 warps, 1 row per CTA for 1 column, 2 rows for 4 columns), calling the unchanged `vec_dot_*_q8_1` functions from `llama-rig2/ggml/src/ggml-cuda/vecdotq.cuh`. I compiled it with nvcc 13.4, `-O3 -arch=sm_120` (CPU only, no GPU used), disassembled with `cuobjdump -sass`, and counted the instructions of the main loop. The compiler hoists most of the weight decode out of the column loop. Per weight, 4 columns cost +29% to +67% more instructions than 1 column for the IQ types, and about ×2.1-2.3 for the K types.

| Type | instr / weight, 1 col | instr / weight, 4 col | instr / byte, 4 col | load instr / byte, 4 col | widest weight load |
|---|---|---|---|---|---|
| IQ1_M | 4.00 | 6.69 | 30.6 | 4.3 | 32-bit |
| IQ2_XXS | 4.62 | 6.62 | 25.7 | 3.3 | 16-bit (+64-bit grid gather) |
| IQ2_XS | 4.75 | 7.12 | 24.6 | 3.0 | 16-bit (+64-bit grid gather) |
| IQ2_S | 5.12 | 7.52 | 23.5 | 2.8 | 16-bit (+64-bit grid gather) |
| IQ3_XXS | 5.28 | 7.31 | 19.1 | 2.7 | 16-bit |
| IQ3_S | 5.75 | 7.39 | 17.2 | 2.6 | 16-bit |
| IQ4_XS | 3.97 | 5.50 | 10.4 | 1.5 | 32-bit |
| Q2_K | 5.31 | 12.03 | 36.7 | 4.2 | 32-bit |
| Q4_K | 4.19 | 8.88 | 15.8 | 2.0 | 32-bit |
| Q6_K | 5.25 | 11.44 | 13.9 | 2.3 | 16-bit |

The instruction mix is dominated by LOP3, IMAD, IADD, SHF and IDP (dp4a). In the IQ types the sign handling (`__vcmpne4`, `__vsub4`, which have no single SASS instruction) takes a large share. BF16 is not in MMVQ; it goes through `mmvf` with fp32 activations and costs ~1 conversion plus 1 FFMA per weight per column (trivial against 2 bytes per weight).

---

## 2. How llama.cpp MMVQ works (llama-rig2, branch rig/full)

### 2.1 Dispatch

- `ggml_cuda_mul_mat` (`ggml-cuda.cu:1876-1929`) uses MMVQ when `ggml_cuda_should_use_mmvq` says so. On Blackwell (`mmvq.cu:334-347`, "tuned on RTX 5090") Q2_K/Q4_K use MMVQ up to 5 columns, Q6_K up to 7, all IQ types up to `MMVQ_MAX_BATCH_SIZE` = 8 (`mmvq.cuh:3`). Above that, MMQ.
- So all decode matmuls of this model (1-4 columns) use MMVQ. BF16 uses MMVF.

### 2.2 Activation quantization (the kernel before every MMVQ)

- `ggml_cuda_mul_mat_vec_q` allocates a q8_1 buffer and calls `quantize_row_q8_1_cuda` on every call (`mmvq.cu:1530-1537`). Two matmuls with the same input (for example `attn_qkv` and `attn_gate`) quantize it twice.
- Kernel `quantize_q8_1` (`quantize.cu:53-101`): 256 threads per CTA, one element per thread, one warp per 32-element block. `amax` and `sum` by warp shuffle. `d = amax/127`, `q = roundf(x/d)` (a division, not a reciprocal multiply), `ds = half2(d, sum)`. Grid: (K/256, columns, …). Row padding to 512 elements (`common.cuh:187`).
- `block_q8_1` = half2 `ds` + 32 int8 = 36 bytes (`ggml-common.h:258-269`).

### 2.3 Thread mapping (`mmvq.cu:599-851`)

- CTA = 32 × `nwarps` threads. On sm_120 the GENERIC table applies: `nwarps` = 4 for 1-4 columns, 2 for 5-8 (`mmvq.cu:452-467`). Rows per CTA = 1 for 1 column, 2 for 2-8 columns (`mmvq.cu:579-597`).
- Each thread handles one (super-block, sub-position) pair: `kbx = tid / (qi/vdr)`, `kqs = vdr·(tid % (qi/vdr))`, stepping `kbx += blocks_per_iter`.
- The same thread computes all columns and all rows of the CTA for that slice, so the weight decode is shared across columns (confirmed by the SASS counts).
- End: warps 1..n−1 write partial sums to shared memory, `__syncthreads`, warp 0 adds them and does a 5-step xor-shuffle reduction per (row, column) (`mmvq.cu:763-843`).

| Type | threads per super-block | weights per thread per step | super-blocks per CTA step (4 warps) | lane use at K=5120 (20 sb) | K=8704 (34 sb, `ffn_down` per card) | K=3072 (12 sb, `ssm_out`/`attn_output` per card) |
|---|---|---|---|---|---|---|
| IQ1_M, IQ2_*, IQ3_*, IQ4_XS | 8 | 32 | 16 | 62.5% (2 steps) | 71% (3 steps) | 75% (1 step) |
| Q2_K, Q4_K | 16 | 16 | 8 | 83% (3 steps) | 85% (5 steps) | 75% (2 steps) |
| Q6_K | 32 | 8 | 4 | 100% (5 steps) | 94% (9 steps) | 100% (3 steps) |

A 1-column IQ3_S CTA reads one row of 2.2 KB and lives for 2 loop steps. The per-row reduction and CTA start-up are paid 5120 times per tensor.
There is a `small_k` variant for short rows (`mmvq.cu:1119-1154`); on NVIDIA it is disabled for IQ3_XXS and IQ3_S at 1 column.

### 2.4 Memory access width

- Weights: 8, 16 and 32-bit loads (`LDG.E.U8/U16/E`), plus 64-bit grid gathers for IQ2_*. No 128-bit loads. Loads use the read-only path (`.CONSTANT`). Counts per loop step are in the SASS (for IQ3_S at 1 column: 2 × U8, 8 × U16, 16 × 32-bit per 32 weights).
- Activations: 32-bit loads of q8_1 data. In a warp, each thread reads a different 36-byte q8_1 block, so one warp instruction touches ~9 cache lines. The same activation bytes are re-read from L1 for every row.
- Grid lookups: random gathers into 1-8 KB global tables through L1.
- Estimate: 1.5-4.3 load instructions per weight byte at 4 columns. With scattered narrow loads, the L1/LSU path is a likely limiter in addition to ALU. Not measured; see section 8.

### 2.5 Fusions already present

- In MMVQ: gate+up+GLU (SwiGLU, GeGLU, SwiGLU-OAI, clamp), bias, NVFP4 scale, MoE shared expert (`mmvq.cu:661-676, 808-840`).
- Limits: only for 1 column (`mmvq.cu:1035-1046`, `ggml-cuda.cu:1815-1818`) and only if gate and up have the same type, shape and stride (`ggml-cuda.cu:1744-1747`). Here gate and up have the same type in 25 of 65 layers (23 distinct pairs). The verify pass has 4 columns, so the main model gets no MMVQ fusion during normal MTP decode.
- Graph level (`ggml-cuda.cu:4408-4455`): RMS_NORM+MUL, RMS_NORM+MUL+ADD, RMS_NORM+MUL+ROPE(+SET_ROWS), SSM_CONV(+ADD)+SILU, UNARY+MUL. None of them joins the norm with the q8_1 quantization or the GEMV.
- PDL: MMVQ calls `cudaGridDependencySynchronize` before any load (`mmvq.cu:640`, `common.cuh:134-145`). It does not prefetch weights before that point.
- Matmul kernels per verify pass (count from the graph structure, estimate): 48 GDN layers × 6 MMVQ + 16 attention layers × 7 MMVQ = 400 MMVQ, each with its own quantize kernel, plus 48 MMVF and the output head. About 850 matmul-related kernels.

### 2.6 Published numbers on bandwidth

- No published per-kernel bandwidth numbers for the IQ types on RTX 40/50 were found.
- F16 matrix-vector kernel: "RTX 4090 is already at 94% of peak" (issue #9817).
- CUDA scoreboard (discussion #15013): Llama 2 7B Q4_0 (3.56 GiB) tg128 on RTX 5060 Ti = 93.46 t/s. That is ~3.75 GB read per token (estimate, excluding the embedding table) × 93.46 = ~350 GB/s, 78% of the 448 GB/s spec, ~90% of the measured copy rate. This is a whole model with simple 4-bit weights.
- PR #26705: making the Q4_K/Q5_K scale unpack branchless gave up to +22.7% at 6 columns on RTX 5090. This shows K-quant MMVQ is instruction-bound at several columns. The same PR found L2 prefetch hurts on high-bandwidth GPUs (comment at `mmvq.cu:9-11`).
- Code comment `mmvq.cu:322`: "k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner."
- PR #8215 (2024): refactoring IQ MMVQ gave IQ2_XXS ×1.89 at batch 8 on RTX 4090, so those kernels were ALU-bound then.
- QuIP# (E8P 2-bit codebook, the origin of the IQ2 grids): ">50% of peak memory bandwidth" on RTX 4090.

### 2.7 ALU ceilings of the llama.cpp loops on this card (estimate)

Arithmetic: 390 GB/s ÷ (36 SMs × 2.6 GHz) = 4.17 bytes per SM-clock must be consumed. Issue limit = 128 thread-instructions per SM-clock. LOP3, SHF, IMAD and conversions run at 64 per SM-clock, POPC at 16 (CUDA Best Practices Guide, throughput table, cc 12.0). dp4a assumed 64 (secondary source). "Half-rate bound" counts LOP3/SHF/IMAD/PRMT/IDP/LEA/I2FP/IADD/ISETP at 64 and POPC at 16; "issue bound" counts everything at 128. Ceiling = 36 × 2.6e9 ÷ (clocks per byte).

| Type | 1 col: half-rate / issue bound (GB/s) | 4 col: half-rate / issue bound (GB/s) |
|---|---|---|
| IQ1_M | 401 / 655 | 243 / 392 |
| IQ2_XXS | 354 / 668 | 253 / 466 |
| IQ2_XS | 389 / 729 | 263 / 486 |
| IQ2_S | 405 / 749 | 280 / 511 |
| IQ3_XXS | 472 / 868 | 345 / 627 |
| IQ3_S | 492 / 895 | 387 / 697 |
| IQ4_XS | 975 / 1604 | 688 / 1157 |
| Q2_K | 466 / 740 | 205 / 327 |
| Q4_K | 989 / 1609 | 467 / 759 |
| Q6_K | 1182 / 1872 | 528 / 859 |

Reading: a value under ~430 GB/s (R / 0.9) means the loop cannot stream at full bandwidth even with perfect latency hiding. At 4 columns this holds for all IQ2, IQ3 and Q2_K variants on the conservative bound. Real kernels reach maybe 70-85% of an ALU ceiling. If I apply 85% of the ceiling, capped at 90% of R, the main-model GEMVs of the verify pass take ~16.2-17.9 ms per card (estimate) against 14.6 ms at 100% of R. The probe uses the real `vec_dot` code but not the exact kernel, and the clock (2.6 GHz) is assumed.

### 2.8 What the new engine must copy to match llama.cpp logits

Measured with numpy on real weights from the production GGUF and synthetic activations (5120 normal values with 1% outliers ×5). Scripts were run in the project venv; see section 4.3 for the method.

| Item | Where in llama.cpp | Effect if not copied (difference / RMS of the GEMV output) |
|---|---|---|
| q8_1 quantization of every quantized-GEMV input: blocks of 32 along K, `d = amax/127`, `q = roundf(x/d)` | `quantize.cu:84-100` | mean 5.7e-3, max 3.1e-2 (blk.1.attn_qkv, IQ3_S, 1024 rows) |
| q8_1 scale stored as fp16 and the fp16 value used in the dot (the quants were computed with the fp32 `d`) | `quantize.cu:100`, `vecdotq.cuh:1076, 1246` | mean 1.7e-4, max 6.9e-4 |
| Integer truncation in the IQ sub-block scale: IQ2_XXS `sumi*ls/8`, IQ2_XS/IQ2_S `(Σ sumi_k·ls_k + (sumi0+sumi1)/2)/4`, IQ3_XXS `(ls*sumi + sumi/2)/2` | `vecdotq.cuh:1074-1075, 1116, 1163, 1201-1202` | IQ2_XXS mean 3.1e-5 (max 1.3e-4); IQ3_XXS mean 7.8e-6 (max 3.1e-5) |
| Float order per sub-block: `(float(d_w)·float(d_y))·sumi`; Q4_K/Q2_K mins via integer `dp4a` sums of the q8 values, not `ds.y` | `vecdotq.cuh:521-529, 380-391` | small (fp32 rounding) |
| BF16 tensors use fp32 activations (MMVF), not q8_1 | `ggml-cuda.cu:1896-1900` | same size as row 1 |
| Thread mapping and reduction order | `mmvq.cu:763-843` | ~1e-7 relative; not worth copying |

Also note: with `-sm tensor` the K-split tensors (`ffn_down`, `ssm_out`, `attn_output`) give two fp32 partial sums that are added after the all-reduce. Per-card K (8704, 3072) are multiples of 32, so the q8_1 blocks are the same as unsplit.
An exact bit match with llama.cpp is not realistic (different reduction trees). Copying the first four rows keeps per-GEMV differences near fp32 rounding level.

---

## 3. Prior art

### 3.1 ik_llama.cpp (`refs/ik_llama.cpp`)

- Generic MMVQ (`mmvq-templates.cuh:68-150`) is the older mainline design: same thread mapping, `nwarps` 4 for ≤4 columns, rows per CTA 1 or 2.
- Difference 1: fused gate+up+activation for any column count 1-8 (`mmvq-templates.cuh:152-280`), still requiring the same type for gate and up.
- Difference 2: extra quant types (IQ2_K…IQ6_K, *_KS, *_KT trellis) with their own CUDA MMVQ (`iqk_mmvq.cu:10-94`). These are new formats; converting our tensors to them would change values.
- Row-interleaved types on CUDA: IQ2_K_R4…IQ5_KS_R4, IQ1_S_R4, IQ1_M_R4, IQ4_KS_R16 (`iqk_mmvq.cu:63-89`). The kernel uses 1 warp, `rows_per_cuda_block = n_interleaved` (4), and each thread accumulates 4 rows per activation load (`iqk_mmvq_templates.cuh:45-51, 80-83`). IQ1_M_R4 is a different format (4-bit scales, per-row fp16), not a repack of IQ1_M.
- Lossless repacks exist for exactly our types, but only on the CPU: `block_iq2_xxs_r4`, `iq2_xs_r4`, `iq2_s_r4`, `iq3_xxs_r4`, `iq3_s_r4`, `iq4_xs_r8`, `q2_k_r4`, `q4_k_r4`, `q6_k_r4` (ik `ggml-common.h:315-673`; each struct is exactly R × the original size). Example `repack_iq3_s` (`iqk_quantize.cpp:8205-8240`) interleaves 4 rows and reshuffles even the sign bits; `repack_q4_k` (`iqk_quantize.cpp:6265-6300`) re-splits the 6-bit scales. Enabled with `-rtr` (`common/common.cpp:2443`). No CUDA kernels for these.

### 3.2 calm (zeux, `refs/calm`)

- RTX 4090: 846-923 GB/s of 1008 (84-92%) for fp16/fp8/gf4 weights, whole model (README table). The author notes ~955 GB/s is the practical maximum on his card, also for cuBLAS.
- How:
  - One persistent cooperative kernel per forward pass: 1 CTA of 1024 threads per SM, software grid barrier between stages (`infer.cu:332-351, 404-626, 722`). No launches inside a layer.
  - Warp per output row; lane j reads bytes at `lane*8…`, so consecutive lanes read consecutive addresses with 64-128-bit loads (`helpers.cuh:127-278`).
  - Groups of 4 warps interleaved across SMs for work balance (`infer.cu:422-424`).
  - RMSNorm recomputed by every CTA into shared memory; the norm scale is applied after the dot (`infer.cu:442, 453`).
  - Fused epilogues: RoPE + KV-cache write in the QKV matmul, SiLU(gate)·up in one loop over both matrices, residual add in the output projections (`atomicAdd`, non-deterministic order).
  - Simple decode (gf4: eight 3-bit values + fp8 scale in 32 bits).

### 3.3 Marlin, Machete, others

- Marlin (IST-DASLab, int4 × fp16, sm_80+): "close to ideal (4x) speedups up to batchsizes of 16-32". Techniques: weights and scales reshuffled offline into the access order; all loads at maximum vector width; asynchronous global loads; L2 evict-first policy for weights; striped partitioning over SMs; tensor cores (`mma.sync`). Format: uniform int4. Not usable for codebook formats, but the layout lessons apply. It runs on sm_120 through `mma.sync`.
- Machete (vLLM): CUTLASS-based successor "optimized for Hopper" (`refs/vllm/csrc/libtorch_stable/quantization/machete/Readme.md`). Uses wgmma, which sm_120 lacks. Not applicable.
- vLLM GGUF path: a plugin that pins llama.cpp's own MMVQ/MMQ kernels (vllm-gguf-plugin PR #141). No better kernels there.
- QuIP# E8P: >50% of peak on RTX 4090 for 2-bit codebook GEMV (arXiv 2402.04396). This is a realistic warning: 2-bit codebooks are hard to stream at full bandwidth.
- Correctness note: on sm_120 with CUDA 13.2.0-13.2.1, byte indexing like `((const uint8_t*)&x)[i]` in IQ1_S/IQ2_S/IQ3_S kernels miscompiled (PR #28784). Use explicit `__byte_perm`/BFE in new kernels and keep CUDA 13.4.

### 3.4 Lessons for the new engine

1. Load with 128-bit, coalesced, in a layout prepared at load time.
2. Put codebooks in shared memory, not in global tables.
3. Read activations once per CTA (registers or shared memory), not once per row.
4. Decode once, use for all 1-4 columns.
5. Avoid one launch per small matrix: group matrices that share an input, or use a persistent kernel.
6. Issue the first weight loads before waiting on the previous kernel (PDL), with evict-first hints for weights.

---

## 4. Lossless repack

### 4.1 Options

"Bytes" is the change in bytes read per decode step for the affected tensors.

| Option | Bytes | What it gives | Values identical? |
|---|---|---|---|
| A. Pad each block to a multiple of 16 B | IQ3_S +1.8%, IQ4_XS +5.9%, Q6_K +6.7%, IQ2_XS +8.1%, IQ1_M/IQ3_XXS/Q2_K +14%, IQ2_S +17%, IQ2_XXS +21%, Q4_K 0 | 128-bit loads of whole blocks | yes (bytes copied, zeros added) |
| B. Split each block's fields into separate arrays (SoA: all `qs`, all `signs`, all `qh`, all scales, all `d`), each array 16-B aligned | 0 (padding only at tile ends) | coalesced 128-bit loads per field; scales and `d` separate from quants | yes (pure byte permutation) |
| C. Interleave R rows (R = 2-8) at sub-block or super-block level, as ik_llama does on the CPU | 0 | one activation read serves R rows; wider per-lane loads | yes (pure permutation) |
| D. Order data inside a tile in the exact order lanes consume it (Marlin idea) | 0 | each lane's bytes contiguous; one LDG.128 per lane per field | yes (pure permutation) |
| E. Unpack packed scales: Q4_K 12 B → 8 + 8 B; IQ4_XS 6 B → 8 × int8 (ls−32) | Q4_K +2.8%, IQ4_XS +1.5% | fewer shift/mask ops | yes (verified, 4.3) |
| F. Store the 8th (parity) sign bit explicitly: IQ2_XXS, IQ2_XS, IQ3_XXS | +6.1%, +5.4%, +4.1% | no popc | yes, but costs bandwidth |
| G. One common value format for all types | IQ3_S → linear 4-bit: +24%; any type → int8 + fp32 scale: up to 9 bpw (+340% for IQ2_XXS) | one decoder | yes for int8, but far slower |

Recommendation:
- Use B + C + D for every quantized type. They cost no bytes. All per-card row counts (5120, 3072, 8704, 6144, 512, 124160) are multiples of 8, so R = 8 tiles fit. Per-card K values (5120, 8704, 3072) are whole super-blocks.
- Use E only for a type that profiling shows as ALU-bound.
- Do not use A, F or G. Bandwidth is the scarce resource; ALU is not, once the kernel is written well (section 7).

"One kernel per group of same-type tensors": the types change from layer to layer (only 25/65 layers have gate type = up type; 23 distinct gate/up pairs). Layers run in sequence, so tensors of one type cannot share a launch across layers. The useful grouping is by shared input, inside one layer (qkv + gate + alpha + beta; q + k + v; gate + up). Two ways:
- a runtime type switch per CTA (uniform branch, no divergence; one kernel; register use = worst type), or
- templated kernels per (type A, type B) pair (23 pairs for gate/up).

A common container layout (B + C + D with type-specific field contents) gives one kernel skeleton (loads, pipelining, activation staging, reduction) with small per-type decode functions. That is worth doing. A common value format is not.

### 4.2 Why permutations keep values identical

A pure permutation sends every byte (or bit) of the original block to exactly one place in the new layout. The decoder for the new layout reads, for each weight, the same index bits, sign bits, scale bits and fp16 `d` as the original decoder. Dequantization is a deterministic function of those bits. So each weight decodes to the same integer and the same float. Transforms that re-encode a field (option E) are a function of one field only, so equality can be tested exhaustively or on the real tensors.

### 4.3 Verification on the production GGUF

Run in the project venv with the reference `dequantize` of the `gguf` package (CPU, read-only memmap of `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`). Floats compared bit for bit (`view(np.uint32)`).

| Test | Tensor (first rows) | Result |
|---|---|---|
| B + C round trip (fields split, 8-row interleave, then back) | one tensor per type: `output.weight` Q4_K, `token_embd` IQ2_S, `blk.0.attn_gate` IQ4_XS, `blk.0.ffn_gate` IQ2_XS, `blk.0.ffn_up` IQ2_XXS, `blk.1.attn_gate` IQ3_XXS, `blk.1.attn_qkv` IQ3_S, `blk.7.attn_q` Q2_K, `blk.13.ffn_gate` IQ1_M, `blk.64.nextn.eh_proj` Q6_K (64 rows each) | bytes equal and values bit-equal for all 10 types |
| Decoder that reads only the SoA tile arrays (no inverse permutation) | `blk.1.attn_qkv` IQ3_S, 128 rows | bit-equal |
| Option E, IQ4_XS scales → int8 | `blk.0.attn_gate`, 128 rows | bit-equal |
| Option E, Q4_K scales → 8 + 8 bytes | `output.weight`, 128 rows | bit-equal |

The core of the round-trip test:

```python
FIELDS = {T.IQ3_S: [('d',2),('qs',64),('qh',8),('signs',32),('scales',4)], ...}  # struct order, ggml-common.h
def repack(blocks):                       # [rows, nb, bs] -> per field [rows/8, nb, 8, w]
    out, off = {}, 0
    for name, w in FIELDS[t]:
        out[name] = blocks[:, :, off:off+w].reshape(rows//8, 8, nb, w).transpose(0, 2, 1, 3).copy(); off += w
    return out
# unpack = inverse transpose + concatenate; compare dequantize(raw) and dequantize(unpack(repack(raw))) bitwise
```

Repack cost at load time: one permutation per tensor on the GPU (a copy kernel, ~5.7 GB per card, well under a second) or on the CPU. It needs a temporary buffer the size of the largest tensor (output head shard, 358 MB).

---

## 5. Output layer and GPU sampling

### 5.1 Output layer cost

- `output.weight` Q4_K, 248320 × 5120, 715,161,600 B. Split by vocab across the 2 cards (`llama-model.cpp:598-603`): 357.6 MB per card.
- One pass at 92% of R: 357.6 MB ÷ 359 GB/s = 1.0 ms per card (estimate). With 3 drafts + 1 verify, the head is read 4 times per step: ~4.0 ms per card per step.
- Mirroring the head on both cards was measured at +3 ms per step (`qwen38_27/LEEME.md:610, 723-724`). Keep the vocab split.
- Q4_K at 4 columns is not ALU-bound (ceiling 467-759 GB/s in llama.cpp's loop), so the verify pass reads the head at nearly the same cost as a draft pass.

### 5.2 Sampling semantics to reproduce

- Sampler order (`common/common.h:263-273`): penalties, DRY, top-n-sigma, **top-k**, typical, **top-p**, **min-p**, XTC, **temperature**. With the user's settings, penalties/DRY/typical/XTC are neutral, min-p = 0 is off and temp = 1.0 is identity.
- top-k = 20 runs first on raw logits (`llama-sampler.cpp:321-338`, partial sort). Everything after it sees ≤ 20 candidates.
- top-p: softmax over the 20, keep the shortest sorted prefix whose cumulative sum is ≥ 0.95 (`llama-sampler.cpp:1574-1601`, test `cum_sum >= p` at line 1582). The final sampler renormalizes.
- Full-vocabulary work is only: find the top-20 logits of 248320.
- MTP probabilistic drafting (`common/speculative.cpp:369-380`): the draft samples with the target's temp and seed through the same chain and keeps its candidate list q (≤ 20 entries).
- Verify (`common/sampling.cpp:729-836`): accept draft x if `p_x ≥ q_x` or `u < p_x/q_x`. Else sample from `max(0, p − q)` over the target candidates; tokens outside q keep all of p. After all drafts are accepted, sample the bonus token normally.
- RNG: CPU `std::mt19937`. A GPU sampler cannot reproduce the same random draws. It can reproduce the same distribution exactly.

### 5.3 What llama.cpp costs today (estimate)

- With `-sm tensor`, backend (GPU) sampling is refused and falls back to the CPU, also for the MTP draft (`LEEME.md:705-717`). The PR that added backend sampling measured +25% on other setups (same file).
- Each sampled row moves 248320 × 4 B = 0.99 MB of logits to the host, half from each card. Per step (4 verify rows + 3 draft rows): 3.5 MB per card over PCIe Gen3 x4 at ~3.5 GB/s = ~1.0 ms (the two links in parallel).
- Then a CPU partial sort over 248320 entries per row (not measured; estimate 0.2-0.5 ms per row, so 1.4-3.5 ms per step).

### 5.4 GPU design

1. **Top-20 inside the output GEMV.** Each persistent CTA keeps a running top-20 per column in a warp's registers. A new logit is compared with the current 20th value (one compare, rare insertions). At the end each CTA writes 20 (value, id) pairs. With 72 CTAs per card that is 1440 candidates per column. The last CTA to finish (atomic counter) merges them into the card's top-20. No logits leave L2.
2. **Cross-card merge.** Each card sends 20 pairs per row (160 B + ids) to the other card or to one card. Transport is P2P if it works on this rig, otherwise mapped pinned host memory. This is the slowest part. Estimate 10-40 µs on Windows WDDM, less on Linux. Unmeasured; Windows-specific.
3. **Final step in one warp.** Softmax over 20 (temp), top-p cut with the same `>=` rule in fp32 in sorted order, Philox counter-based RNG (as FlashInfer: `curand_init(seed, row, offset)`), draw.
4. **MTP rejection.** Same warp. Inputs: target candidates p for each verify position, stored draft candidates q for each draft position, draft ids. Loop over drafts with the llama.cpp rule; on reject, build the residual over ≤ 20 entries and draw. Output: accepted count + tokens. Cost ~µs.
5. Keep everything on the device. The host needs only the accepted count and tokens (a few bytes) to stream text and to plan the next step.

FlashInfer reference points:
- `TopKTopPSamplingFromProbKernel` (`sampling.cuh:1211-1342`) does rejection-style top-k/top-p over the full vocabulary with repeated passes. That is needed when k is large; here k = 20, so explicit selection is simpler.
- `ChainSpeculativeSampling` (`sampling.cuh:1885-2023`) is the same accept rule as llama.cpp, but it reads full-vocab probability rows (two 248320-float rows per position). With top-k = 20 this is unnecessary.
- `topk.cuh` has a multi-CTA radix top-k with inter-CTA barriers. Its comments note a past hang on SM120/SM121 caused by a barrier reset race (`topk.cuh:195-205`). Any inter-CTA barrier in our sampler needs the same care.

### 5.5 Cost per call (estimate)

| Part | Cost per pass | Notes |
|---|---|---|
| Output GEMV (per card, vocab half) | ~1.0 ms | 357.6 MB at ~92% of R |
| Running top-20 in the GEMV | ~0 | compares only |
| Per-card merge (last CTA) | 2-5 µs | 1440 candidates |
| Cross-card exchange | 10-40 µs | WDDM/host path; unknown P2P |
| Softmax/top-p/sample or rejection (1 warp) | 2-5 µs | ≤ 20 entries |
| **Total sampling overhead** | **~15-50 µs per pass, ~0.06-0.2 ms per step** | against ~2.4-4.5 ms per step today (5.3) |

### 5.6 Option that changes draft cost, not the output distribution

The draft head can use a subset of the vocabulary, for example the 32k most frequent tokens (FR-Spec, arXiv 2502.14856, ACL 2025: up to 75% less LM-head compute, 1.12× over EAGLE-2). The rejection test stays exact with respect to the target, because q is the real draft distribution and tokens outside q keep all of p in the residual.
- Draft head bytes per card: 357.6 MB × 32768/248320 = 47 MB, so ~0.13 ms instead of ~1.0 ms per draft. That saves ~2.6 ms per step with 3 drafts (estimate).
- Cost: acceptance drops when the target picks a token outside the subset (unknown for this model; must be measured).
- The subset rows are copied from `output.weight` at load time (lossless rows, +47 MB VRAM per card).

---

## 6. Fusion options for decode

Time saved per removed kernel boundary: launch gap ~1.3 µs with CUDA graphs and ~2.1 µs without on H100 (Hazy Research), plus ramp-up and tail. Estimate 2-5 µs on this card; probably more on Windows WDDM (unmeasured).

| Fusion | Kernels removed (per layer, verify pass) | Requirements | Numerics vs llama.cpp |
|---|---|---|---|
| RMSNorm (×weight) + q8_1 quantize in one kernel, done once per input vector | 1 norm + 1-2 extra quantize kernels | none | same formulas; sum-of-squares order differs (~1e-7) |
| GDN input group: `attn_qkv` + `attn_gate` + `ssm_alpha` + `ssm_beta` in one launch (mixed types, per-CTA type switch) | 3 launches + 1 quantize | BF16 rows read the fp32 normed input | same |
| … plus epilogues: causal conv1d (kernel 4) + SiLU on qkv rows using the 3-token conv state; `sigmoid(beta)`; `-exp(A)·softplus(alpha + dt_bias)` | 2-3 kernels | CTA owns whole channels; all 4 tokens are in the CTA | same ops |
| Attention input group: `attn_q` (with gate) + `attn_k` + `attn_v`, epilogue q_norm/k_norm + partial RoPE (64 of 256 dims) + KV-cache write | 3 launches + ~4 small kernels | row tile = whole heads (256 rows); `attn_k`/`attn_v` are only 512 rows per card, too small for their own launch | same |
| FFN: `ffn_gate` + `ffn_up` + SiLU(gate)·up + q8_1 quantize of the product (input of `ffn_down`) | ~3-4 kernels | gate row i and up row i in the same CTA; 23 type pairs; output row tiles of 32-multiple rows so each q8_1 block is made in one CTA | same if the product is quantized with the same formula |
| `ffn_down` / `ssm_out` / `attn_output` + residual add (+ next RMSNorm partial sums) | 1-2 kernels | the TP all-reduce sits between partial sum and residual | add order of 2 partials |
| Output: final norm + quantize + GEMV + running top-20 | 2-3 kernels and the logits copy | section 5.4 | same candidates |
| Weight prefetch before `griddepcontrol.wait` (PDL) | hides ramp-up | weights do not depend on the previous kernel | — |
| Persistent per-layer kernel or whole-pass megakernel | almost all launches | grid barrier; calm and HazyResearch Megakernels show it works (78% of H100 bandwidth vs ≤ 50% for vLLM/SGLang on Llama-1B) | — |

Count (estimate): llama.cpp issues ~850 matmul-related kernels per verify pass (2.5). The grouped design needs ~5 per layer (~320 per pass). A megakernel needs ~1. At 2-5 µs per boundary, removing ~500 boundaries saves ~1-2.5 ms per verify pass.

---

## 7. Fraction of peak bandwidth a good kernel can reach

### 7.1 Basis

1. **Ceiling.** Good simple-format GEMVs reach 84-94% of spec on RTX 4090 (calm, F16 MMV). On this card copy is 87-88% of the 448 GB/s spec. So a long read stream should reach ~95-100% of R.
2. **Short kernels.** Per-card GEMVs are small: 1.4 MB (`attn_k`) to 19 MB (`ffn_up`), 4-50 µs. A 2-4 µs ramp/tail per launch costs 5-15% unless hidden by PDL prefetch, grouping or a persistent kernel.
3. **ALU.** A redesigned decode keeps ALU under ~50% of the full issue budget (7.2). Then enough loads stay in flight. The 2-bit codebook types and Q2_K at 4 columns come closest to the limit, so they get lower estimates.

### 7.2 Target decode cost (design estimates, not measured)

Ideas that cut the llama.cpp counts:
- 128-bit loads from the repacked layout: ~4-6 load instructions per 64 weights instead of 26 per 32.
- Codebook in shared memory (LDS gather instead of LDG).
- Signs with 2-3 instructions per 4 weights. Let c be a word with 0x01 in each byte whose weight is negative: `c = ((s & 0xF) * 0x00204081) & 0x01010101` for 4 sign bits s; let M = c·0xFF. Then `(G ^ M) + c` negates exactly the negative bytes of the 4-byte grid word G. Because every grid byte is ≥ 1, `(g ^ 0xFF) + 1 = 256 − g` never carries into the next byte, so one 32-bit IADD is correct. I checked that the multiply puts bit i at bit 8i with no overlaps.
- Integer scale applied once per 32 weights per column; decode shared across columns; activations from registers or shared memory with broadcast.

| Type | target instr/weight 1 col / 4 col (estimate) | instr/byte 1 / 4 col |
|---|---|---|
| IQ1_M | 2.5 / 3.5 | 11 / 16 |
| IQ2_XXS | 1.9 / 2.6 | 7.4 / 10.1 |
| IQ2_XS, IQ2_S | 2.0 / 2.7 | 6.2-6.9 / 8.4-9.3 |
| IQ3_XXS | 2.0 / 2.8 | 5.2 / 7.3 |
| IQ3_S | 2.2 / 3.0 | 5.1 / 7.0 |
| IQ4_XS | 1.5 / 2.3 | 2.8 / 4.3 |
| Q2_K | 1.5 / 2.5 | 4.6 / 7.6 |
| Q4_K | 1.2 / 2.0 | 2.1 / 3.6 |
| Q6_K | 1.5 / 2.3 | 1.8 / 2.8 |

Budget at R: ~15 instructions per byte if everything runs at the half rate, ~31 at full issue rate (2.7). At 4 columns, IQ3_*, IQ4_XS and the K types stay under 50% of the half-rate budget. IQ2_* use 55-66% of it. IQ1_M (0.2% of bytes) is the only type above it.

Another option for 4 columns: int8 `mma.sync` m16n8k32 with decoded weights as the A operand (Marlin-style fragment layout). With k = 32 equal to the IQ sub-block, the per-sub-block integer sums are the same as with `dp4a`, so the llama.cpp truncation rules can still be applied. It saves ~0.5-1 instruction per weight at 4 columns. Worth trying only for the 2-bit types if they stay ALU-bound.

### 7.3 Estimates per type

Share = share of main-model GEMV bytes (10136 MiB, blk.0-63, without `token_embd`). Percent = fraction of R.

| Type | Share | llama.cpp loop ALU ceiling 1 col / 4 col (GB/s, half-rate–issue) | Good kernel, 1 token | Good kernel, 4 tokens | Reason |
|---|---|---|---|---|---|
| IQ3_S | 34.3% | 492-895 / 387-697 | 90% | 88% | ALU ~35-50% of budget; field split removes 16-bit loads |
| IQ4_XS | 28.4% | 975-1604 / 688-1157 | 92% | 91% | cheap 16-entry table decode |
| IQ3_XXS | 19.5% | 472-868 / 345-627 | 90% | 88% | like IQ3_S, plus popc for parity |
| Q4_K (+ output head) | 8.2% (+682 MiB) | 989-1609 / 467-759 | 92% | 90% | affine; scale unpack is the main cost |
| IQ2_S | 3.7% | 405-749 / 280-511 | 88% | 83% | 8 KB codebook, 2.56 bpw: more decode per byte |
| Q2_K | 2.3% | 466-740 / 205-327 | 88% | 80% | 16-weight sub-blocks with min terms |
| IQ2_XS | 2.1% | 389-729 / 263-486 | 88% | 82% | 2.31 bpw codebook + parity |
| IQ2_XXS | 0.9% | 354-668 / 253-466 | 88% | 82% | 2.06 bpw: most decode work per byte |
| IQ1_M | 0.2% | 401-655 / 243-392 | 85% | 75% | 11-bit index, delta terms; irrelevant by size |
| Q6_K (MTP block, 1 token) | — | 1182-1872 / 528-859 | 92% | 88% | simple decode, 6.56 bpw |
| BF16 | 0.4% | — | ~50% | ~50% | 24 rows per card: latency-bound unless fused into the qkv group |

Step-level arithmetic (estimate, per card, both cards in parallel):
- Verify pass: 5414 MiB = 5.68 GB per card. At 100% of R: 14.6 ms. With the 1-token percentages: 16.1 ms (352 GB/s effective). With the 4-token percentages: 16.5 ms (344 GB/s).
- Draft pass: (332 + 682) MiB ÷ 2 = 0.53 GB per card, at 92% of R: 1.48 ms. Three drafts: 4.4 ms.
- GEMV total ≈ 20.5-21 ms per step with 3 drafts. With the memory OC (403 GB/s) about 3% less.
- The measured llama.cpp step is 37.7 ms at 1k context. The ALU ceilings in 2.7 suggest llama.cpp's verify-pass GEMVs take ~16-18 ms per card, before L1/LSU effects and launch gaps. So better GEMV kernels save roughly 1-4 ms per step. Fusion (section 6), GPU sampling (section 5) and the draft head (5.6) are of similar size each. The rest of the gap is outside this topic.

---

## 8. Open points and what the GPU session should measure

1. Read-only bandwidth per card: the `lect` columns of `qwen38_27\vram-bw\vram-bw.exe`. This sets R.
2. Nsight Systems on one decode step: total time of `mul_mat_vec_q*` and `quantize_q8_1` kernels per step, kernel count per step, gaps between kernels (Windows and later Ubuntu).
3. Nsight Compute on `mul_mat_vec_q<IQ3_S,4>`, `<IQ3_XXS,4>`, `<IQ4_XS,4>`, `<IQ2_S,4>`, `<Q4_K,1>` (output head): `dram__throughput` % of peak, `smsp__issue_active` %, L1TEX wavefronts per request, top stall reasons. This decides between ALU-bound and LSU-bound and validates 2.7.
4. CPU time of sampling per step in llama-server (top-k partial sort + rejection).
5. Unknown: dp4a throughput on sm_120 (assumed 64 per SM per clock from a secondary source), real boost clock under this load (2.6 GHz assumed), P2P between the cards.

---

## Sources

Local code (read only):
- `llama-rig2/ggml/src/ggml-common.h:258-269` (q8_1), `:298-460` (block structs), `:493` (`__device__` tables), `:509-1650` (codebooks, sign tables), `:1131-1133` (IQ1 grid size, deltas)
- `llama-rig2/ggml/src/ggml-cuda/vecdotq.cuh:18-104` (loads, table lookup, `unpack_ksigns`), `:363-392, 504-530, 623-647` (Q2_K/Q4_K/Q6_K dot cores), `:868-1043` (K vec_dots), `:1045-1380` (IQ vec_dots; truncations at 1074-1075, 1116, 1163, 1201-1202)
- `llama-rig2/ggml/src/ggml-cuda/dequantize.cuh:128-436`
- `llama-rig2/ggml/src/ggml-cuda/mmvq.cu:9-11` (prefetch comment), `:318-428` (dispatch, Blackwell 334-347, comment 322), `:452-597` (nwarps, rows per block), `:599-851` (kernel), `:1023-1053` (fusion only for 1 column), `:1119-1154` (small_k), `:1434-1566` (host: quantize per call at 1530-1537)
- `llama-rig2/ggml/src/ggml-cuda/mmvq.cuh:3`; `quantize.cuh:8`; `quantize.cu:53-101, 558-573`; `common.cuh:123-145` (PDL), `:187` (row padding), `:712` (dp4a)
- `llama-rig2/ggml/src/ggml-cuda/ggml-cuda.cu:1676-1825` (fusion checks), `:1876-1929` (matmul dispatch), `:4408-4455` (graph fusions)
- `llama-rig2/src/llama-model.cpp:520-611` (tensor-split axes; output 594-603)
- `llama-rig2/common/common.h:263-273` (sampler order); `common/sampling.cpp:729-836` (rejection); `common/speculative.cpp:369-380`; `src/llama-sampler.cpp:321-338` (top-k), `:1556-1601` (top-p)
- `qwen38_27/LEEME.md:566-616` (patches, mirror-output +3 ms at 610), `:705-724` (backend sampling refused with `-sm tensor`, +25% claim), `:621` (CUDA 13.2); `qwen38_27/vram-bw/vram-bw.cu` (copy and read-only kernels)
- `research/_gguf-model-tensors.tsv`, `research/_gguf-model-roles.txt`
- `refs/ik_llama.cpp/ggml/src/ggml-cuda/mmvq-templates.cuh:68-459`; `iqk_mmvq.cu:10-94`; `iqk_mmvq_templates.cuh:36-138`; `template-instances/mmvq-instance-iq1_m_r4.cu`, `mmvq-instance-iq4_ks_r4.cu`; `ggml/src/ggml-common.h:315-673` (ik repacked structs); `ggml/src/iqk/iqk_quantize.cpp:5792, 6265-6300, 7848, 8205-8240`; `common/common.cpp:2443`
- `refs/calm/README.md` (performance table, footnote 3); `refs/calm/src/helpers.cuh:127-278`; `refs/calm/src/infer.cu:332-351, 404-649, 722`
- `refs/flashinfer/include/flashinfer/sampling.cuh:1211-1342, 1885-2064`; `topk.cuh:61-250`
- `refs/vllm/csrc/libtorch_stable/quantization/machete/Readme.md`; `.../marlin/marlin.cuh`

Measurements made for this report (CPU only, no GPU):
- SASS probe: `vec_dot_*` from llama-rig2 in an MMVQ-shaped loop, nvcc 13.4 `-O3 -arch=sm_120 -cubin`, MSVC 14.44 host, disassembled with `cuobjdump` from the CUDA 13.2 install (disassembler only).
- numpy checks in `.venv` with the `dequantize` of the `gguf` package on the production GGUF (sections 2.8 and 4.3).

Web:
- CUDA C++ Best Practices Guide, throughput table (cc 12.0): https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- RTX 5060 Ti specs (36 SMs, 32 MB L2, 448 GB/s, 2.57 GHz boost): https://www.guru3d.com/review/nvidia-geforce-rtx-5060-ti-16gb-review/ and https://videocardz.net/nvidia-geforce-rtx-5060-ti-16gb
- dp4a 64/SM/clock (secondary source): https://zolotukhin.ai/zinc/docs/nvidia-gpu-reference/
- F16 MMV at 94% of peak on RTX 4090: https://github.com/ggml-org/llama.cpp/issues/9817
- CUDA scoreboard, RTX 5060 Ti 93.46 t/s Q4_0: https://github.com/ggml-org/llama.cpp/discussions/15013
- MMVQ branchless scale unpack and prefetch data: https://github.com/ggml-org/llama.cpp/pull/26705
- IQ MMVQ refactor numbers: https://github.com/ggml-org/llama.cpp/pull/8215
- sm_120 byte-indexing miscompile with CUDA 13.2.0-13.2.1: https://github.com/ggml-org/llama.cpp/pull/28784
- vLLM GGUF plugin uses llama.cpp kernels: https://github.com/vllm-project/vllm-gguf-plugin/pull/141
- Marlin: https://github.com/IST-DASLab/marlin
- QuIP# (E8P GEMV >50% of peak on 4090): https://arxiv.org/abs/2402.04396
- HazyResearch, "Look Ma, No Bubbles" (launch costs, 78% bandwidth): https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- FR-Spec (reduced-vocabulary draft head): https://arxiv.org/abs/2502.14856 and https://github.com/thunlp/FR-Spec
