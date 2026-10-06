# Vision: the smallest correct way to run the mmproj encoder

Research date: 2026-10-05. Source tree: `llama-rig2` (branch `rig/full`, commit `e2377cc96`).
All paths below without a prefix are relative to `llama-rig2`.
Nothing was run on the GPU. All times are estimates unless a log or LEEME line is cited.

## Summary

1. **Encoder.** It is a ViT with 27 pre-LayerNorm blocks. Hidden size is 1152, with 16 heads of 72 dims and an FFN of 4304 (GELU tanh). Every block uses **full attention over all patches** (no windows). Positions use a learned 48x48 table (bilinear resize) plus 2D RoPE on h and w. A 2x2 merge follows, then `mm.0` (4608->4608), GELU and `mm.2` (4608->5120). There are no deepstack layers. The model has 461M params (888 MiB).
2. **Token count.** A 3840x2160 image is resized to 2720x1536, with the aspect ratio kept and 3 black rows added at top and bottom. That gives 16320 patches and **4080 LLM tokens** (85 x 48). The image sits between `<|vision_start|>` and `<|vision_end|>`. It fills 4080 KV cells but moves the RoPE position by only 85. `--image-min-tokens 1024` only scales up images below about 1 Mpx. A 4K image hits the 4096-token cap instead.
3. **Encoder cost.** One 4K image costs about 47 TFLOP, and 33 TFLOP of that (71%) is attention. llama.cpp runs this attention on a kernel that does not use tensor cores. The reason is that CUDA flash attention leaves head dim 72 out of its MMA path (`fattn.cu:638`).
4. **Where the 12.7 s goes.** The logs have **no image timing lines**. The server prints library INFO lines only at `-lv 4`. Estimate: encoder 3.5-5 s, LLM prefill of the 4080 image tokens about 5.7 s, answer up to 1.6 s, CPU decode and resize 0.3-0.7 s.
5. **mtmd as a library.** The `clip` layer needs only ggml, ggml-cpu and ggml-cuda. The public `mtmd` API also needs libllama, for the tokenizer and the rope type. The license is MIT. It can run on card 1 only. It always returns the embeddings in host memory.
6. **Own kernels.** Estimate: 1.6-2.0 s per 4K image on one card, about 1-1.5k lines of code, and 1-3 weeks of work. The pieces are BF16 GEMM (cuBLASLt), a tensor-core flash attention for d=72 (padded to 80), LayerNorm, GELU and 2D RoPE.
7. **Recommendation.** Write the encoder with own kernels in the engine (milestone 7). VRAM is about 0.9 GiB with resident weights, or about 0.15 GiB with weights streamed from pinned RAM. The activations can share the LLM prefill scratch buffer. A separate quick win for today's llama.cpp: pad the vision head dim from 72 to 80, so the MMA flash-attention kernel is used. It needs a measurement.

---

## 1. The encoder as llama.cpp runs it

### 1.1 Hyperparameters and tensors

From `research/_gguf-mmproj-dump.txt:24-39` and a full tensor listing done in this session with `gguf` in the project venv:

| Item | Value | Source |
|---|---|---|
| Projector | `qwen3vl_merger` | dump:34 |
| Blocks | 27 | dump:30 |
| Hidden | 1152, 16 heads, head dim 72 | dump:28,31; `clip.cpp:260` (d_head = n_embd/n_head) |
| FFN | 4304, `use_gelu` = true -> `FFN_GELU` (tanh form) | dump:29,35; `clip.cpp:1376-1385` |
| Patch | 16 px, temporal patch 2 (two conv weights) | dump:27; tensors `v.patch_embd.weight`, `v.patch_embd.weight.1` (F32 [16,16,3,1152] each) |
| Learned positions | `v.position_embd.weight` F32 [1152, 2304] = 48x48 grid | tensor listing |
| LayerNorm eps | 1e-6, with bias | dump:37 |
| Spatial merge | 2 | dump:36 |
| Output | 5120 (= LLM `embedding_length`) | dump:25 |
| Deepstack | none (all 27 flags false) | dump:38 |
| Normalization | mean 0.5, std 0.5 (pixel/127.5 - 1) | dump:32-33 |
| Tensors | 334 = 27 x 12 + 10 (`mm.0`, `mm.2`, `v.post_ln`, `v.patch_embd` x2 + bias, `v.position_embd`) | tensor listing |
| There is no `v.pre_ln` | the pre-layernorm branch is skipped | `qwen3vl.cpp:61-63` |

Per block: `ln1`, `attn_qkv` [1152->3456] + bias, `attn_out` [1152->1152] + bias, `ln2`, `ffn_up` [1152->4304] + bias, `ffn_down` [4304->1152] + bias. There is no FFN gate.

### 1.2 Graph, step by step (`tools/mtmd/models/qwen3vl.cpp`)

1. **Patch embedding** (`models/qwen2vl.cpp:3-16`). For a still image it runs `conv2d(W0, img) + conv2d(W1, img)`, stride 16. This equals one conv with `W0 + W1` (the HF model repeats the frame for temporal patch 2). Both weights are F32. The conv runs as im2col plus a GEMM.
2. **Reorder into merge order** (`qwen3vl.cpp:18-31`). The patch sequence is permuted so that each 4 consecutive patches form one 2x2 block: (dy,dx) = (0,0),(0,1),(1,0),(1,1), with blocks in row-major order.
3. **Patch bias** (`qwen3vl.cpp:34-37`).
4. **Learned position embedding** (`qwen3vl.cpp:40-52`, `clip.cpp:312-332`). The 48x48 table is resized to (W/16) x (H/16) with bilinear interpolation and align-corners (`GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS`). It is reordered the same way as the patches, then added. HF does the same (`fast_pos_embed_interpolate`, bilinear, align_corners true).
5. **27 blocks** (`qwen3vl.cpp:70-161`). Each block is pre-norm: LN1 -> fused QKV + bias -> 2D RoPE on Q and K -> attention -> out-proj + bias -> residual -> LN2 -> up + bias -> GELU -> down + bias -> residual.
   - **Attention is full in every block.** It is built with no mask (`qwen3vl.cpp:114-115`, mask `nullptr`). Qwen3-VL has no window pattern; only Qwen2.5-VL reads one (`clip.cpp:1675`). Scale is 1/sqrt(72) (`clip.cpp:264`).
   - **2D RoPE** (`qwen3vl.cpp:14, 104-109`; CUDA kernel `ggml/src/ggml-cuda/rope.cu:294-352`). It uses `ggml_rope_multi` in `GGML_ROPE_TYPE_VISION` mode with n_dims = 36 (d_head/2), sections {18,18,..}, base 10000, and no YaRN. Pair i is (x[i], x[i+36]) for i = 0..35. Pairs 0-17 use the row position h with freq 10000^(-2p/36), p = i. Pairs 18-35 use the column position w with p = i - 18. This matches HF `VisionRotaryEmbedding(head_dim/2)` with `cat(freq_h, freq_w)` and `rotate_half`.
   - The positions are written in merge order: h = y+dy, w = x+dx (`clip.cpp:4846-4870`).
6. **Post-LN** with `v.post_ln` (`qwen3vl.cpp:164-166`). In HF this is the merger norm (`Qwen3VLVisionPatchMerger.norm`, applied before the shuffle).
7. **Merger** (`qwen3vl.cpp:169-176`). A reshape to [4608, N/4] concatenates the 4 patches of each block, then `mm.0` (4608->4608) + bias -> GELU -> `mm.2` (4608->5120) + bias. llama.cpp uses `ggml_gelu` (tanh approximation) here. HF uses `nn.GELU()` (exact erf) in the merger. The difference is small, but a bit-parity test against llama.cpp must use tanh.

### 1.3 How llama.cpp executes it on CUDA

- **GEMMs.** BF16 weights with more than 16 columns go to cuBLAS (`ggml-cuda.cu:1876-1929`, `mmf.cu:176-177`). The F32 activations are converted to BF16, then the GEMM runs on tensor cores with FP32 compute and F32 output (`ggml-cuda.cu:1508-1520`, `1621-1667`).
- **Attention.** Flash attention is enabled in AUTO mode if the backend supports it (`clip.cpp:3740-3775`). The server passes `-fa auto` (`server-context.cpp:1081`). K and V are cast to F16 and accumulation is F32 (`clip.cpp:775-791`). For head dim 72 the CUDA dispatcher **skips the tensor-core MMA kernel**: `if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72)` (`fattn.cu:638`). It falls back to the **tile kernel** (`fattn-tile.cu:16-19`). That kernel does SIMT `half2` multiply-adds without tensor cores (`fattn-tile.cuh:505-550`). Its own source says: "TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112" (`fattn-tile.cuh:8`).
- **Without flash attention** the KQ matrix for one layer would be 16320^2 x 16 heads x 4 B = 17 GB. That cannot fit, so flash attention must be on in production.
- All ops are supported on CUDA. The "unsupported operators" warning is printed at WARN level, which is visible at `-lv 3` (`clip.cpp:3779-3797`), and it is absent from `arranque.log.err`.

### 1.4 Image preprocessing

- **Decode:** `stb_image` from the raw bytes (`mtmd-helper.cpp:404`).
- **Target size** (`mtmd-image.cpp:122-157`, called from `mtmd-image.cpp:772-791`). This is "smart_resize" with factor 32 (patch 16 x merge 2):
  1. Round each side to the nearest multiple of 32.
  2. If the area is above `max_pixels`: beta = sqrt(H*W/max_pixels), then floor each side/beta to a multiple of 32.
  3. Else, if the area is below `min_pixels`: beta = sqrt(min_pixels/(H*W)), then ceil each side*beta to a multiple of 32.
- **Limits** (`clip.cpp:1668-1686`, `clip-model.h:192-197`). The default is `set_limit_image_tokens(8, 4096)`. One token is 16*16*2*2 = 1024 px, so min_pixels = 8 x 1024 and max_pixels = 4096 x 1024 = 4,194,304.
- **`--image-min-tokens 1024`** goes server -> `mtmd_context_params` -> `clip_ctx` (`server-context.cpp:1083`, `clip.cpp:210-212`). It replaces the 8 with 1024, so min_pixels = 1,048,576. It only scales up images smaller than about 1 Mpx (for example 1024x768 -> 1184x896 -> 1036 tokens). It has no effect on 4K images. The server caps max tokens to `n_ubatch` only for non-causal models (`server-context.cpp:1207-1216`). Qwen is causal (`mtmd.cpp:2178-2198`), so no cap applies.
- **Resize:** bicubic, Pillow-style, single-threaded CPU (`mtmd-image.cpp:204`). The pad mode is the default `PAD_CEIL` (`clip-model.h:67`), because the Qwen3-VL case does not override it. The image is scaled by min(scale_w, scale_h) and padded with black (`mtmd-image.cpp:63-90`). HF stretches the image to the exact target instead. The difference is a few pixel rows.
- **Normalize** to f32 on the CPU (`mtmd-image.cpp:7-13`). The image then goes to the GPU as an f32 [W, H, 3] tensor.
- **HF reference limits** for Qwen3-VL: `longest_edge` 16777216 px (16384 tokens) and `shortest_edge` 65536 px (64 tokens) (HF `preprocessor_config.json`). llama.cpp's cap of 4096 tokens is 4x lower. Raising it to the HF value would make a 4K image 8160 tokens and 132 TFLOP of attention (computed below). The cap is a speed and quality setting that the user controls.

### 1.5 Token counts

Computed with the same rules (Python, this session):

| Input | Resized | Patches | LLM tokens (nx x ny) | Encoder TFLOP (linear + attention) | LEEME check |
|---|---|---|---|---|---|
| **3840x2160** | **2720x1536** (content 2720x1530) | **16320** | **4080** (85x48) | 13.4 + 33.1 = **47.0** | 4099 prompt tokens = 4080 + 19 (`LEEME.md:86`) |
| 2560x1440 | 2560x1440 | 14400 | 3600 (80x45) | 11.8 + 25.8 = 38.0 | 3631 = 3600 + 31 (`LEEME.md:241`) |
| 1920x1080 | 1920x1088 | 8160 | 2040 (60x34) | 6.7 + 8.3 = 15.2 | |
| 1432x950 | 1440x960 | 5400 | 1350 (45x30) | 4.4 + 3.6 = 8.2 | 1369 = 1350 + 19 (`LEEME.md:228`) |
| 1024x768 | 1184x896 | 4144 | 1036 (37x28) | 3.4 + 2.1 = 5.6 | |

For 4K, beta = sqrt(8294400/4194304) = 1.40625 exactly. So h = 2160/1.40625 = 1536 and w = floor(2730.7/32)*32 = 2720.

---

## 2. How image tokens enter the LLM

### 2.1 Token layout

- The server turns an `image_url` content part into a text part that holds the media marker (`tools/server/server-common.cpp:1163`). So the Jinja branch `<|vision_start|><|image_pad|><|vision_end|>` (`research/_chat_template.jinja:18`) is not used by llama-server.
- `mtmd_tokenize` splits the text at the marker. It adds `<|vision_start|>` before the image chunk and `<|vision_end|>` after it (`mtmd.cpp:694-704`, `1360-1361`, `1555-1556`).
- Token ids in this GGUF (read this session): `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056, `<|im_start|>` 248045, `<|im_end|>` 248046. `<|image_pad|>` never reaches the model in llama.cpp. In HF it is repeated N times and then replaced by the embeddings, so the result is the same.
- The image chunk is 4080 rows of 5120 floats. They go in as `inp_embd` with no `token_embd` lookup and no scaling (`src/llama-graph.cpp:2384-2420`). The 48 GDN layers process them like any token, causally (`mtmd_decode_use_non_causal` returns false for Qwen, `mtmd.cpp:2178-2198`).

### 2.2 Positions (imrope, sections [11,11,10,0])

- The LLM rope type is `LLAMA_ROPE_TYPE_IMROPE` for `qwen35` (`src/llama-model.cpp:3207-3212`). With it, mtmd uses `MTMD_POS_TYPE_MROPE` (`mtmd.cpp:549-565`).
- **Image token i** (row-major, nx = 85 columns, ny = 48 rows), with P = the position after `<|vision_start|>` (`mtmd.cpp:2478-2486`, `mtmd-helper-common.h:100-112`):
  - section 0 (t) = P
  - section 1 (h) = P + i / nx (row)
  - section 2 (w) = P + i % nx (column)
  - section 3 = 0
- **After the image** the position moves by max(nx, ny) = 85 (`mtmd.cpp:2537-2541`, `mtmd-helper.cpp:196`). So `<|vision_end|>` gets position P + 85. This matches HF `get_rope_index` (next text = max position + 1).
- **Text tokens:** t = h = w = p, section 3 = 0 (`src/llama-graph.cpp:131-140`).
- **How the sections map to rotary pairs** (`ggml/src/ggml-cuda/rope.cu:251-263`). The model rotates 64 of the 256 head dims, NeoX pairing (k, k+32), with theta = pos x 1e7^(-2k/64) for k = 0..31. Interleaved assignment:
  - k % 3 == 0 -> t: 11 pairs
  - k % 3 == 1 -> h: 11 pairs
  - k % 3 == 2 -> w: 10 pairs
  - section 3 is never used.
- **Consequence for the engine.** An image uses 4080 KV cells and only 85 positions. The engine must track cell count (for context limits and the KV cache) separately from the RoPE position.

### 2.3 Other places that see image tokens

- **MTP.** llama-server feeds the same image embeddings and positions to the MTP draft context through a callback (`server-context.cpp:770-799`). The engine must do the same, or the MTP KV cache has gaps.
- **Prompt cache.** Image chunks carry an id (`mtmd_input_chunk_get_id`, `mtmd.h:237`). The server uses it to match cached prompts. The engine needs an image hash in its token history, so that a cached prefix with an image can be reused.
- **Transfer.** In llama.cpp the embeddings go GPU (card 1) -> host -> LLM. That is 4080 x 5120 x 4 B = 83.6 MB in f32. Estimate: about 24 ms per copy at 3.5 GB/s, under 0.1 s in total.

---

## 3. Can `mtmd` be used as a library by an engine that does not use llama.cpp for the LLM?

### 3.1 Dependencies

| Layer | Files | Depends on | Source |
|---|---|---|---|
| `clip` (encoder + preprocessing) | `clip.cpp` (6142 lines), `mtmd-image.cpp`, `models/*.cpp` | ggml, ggml-base (gguf), a GPU backend (ggml-cuda) **and** the CPU backend (it throws if the CPU backend fails to init, `clip.cpp:184-187`) | includes `clip.cpp:1-11`. `clip.cpp`, `mtmd-image.cpp`, `qwen2vl.cpp` and `qwen3vl.cpp` make **no** `llama_*` calls (grep, this session) |
| `mtmd` public API | `mtmd.cpp`, `mtmd-helper.cpp` | **libllama**: `llama_tokenize`, `llama_token_to_piece`, `llama_vocab_*`, `llama_model_n_embd_inp`, `llama_model_rope_type` | `mtmd.cpp:9`, `531-565`, `1061-1078`, `1317-1318`, `1719-1735`; `CMakeLists.txt:87` (`target_link_libraries(mtmd PUBLIC ggml llama)`) |

- `mtmd_init_from_file(mmproj, const llama_model * text_model, params)` accepts `text_model = nullptr` (`mtmd.cpp:537-538`). In that case `mtmd_tokenize` throws "llama_vocab is not provided", because it must tokenize `<|vision_start|>` (`mtmd.cpp:1317-1318`). The position type also stays NORMAL, which is wrong for Qwen.
- **Workaround A1.** Load the same GGUF through libllama in vocab-only mode, to get a `llama_model *` with the tokenizer and hparams but no weights. This works in principle but is not tested. It keeps two tokenizers in the process.
- **Workaround A2.** Vendor only the `clip` layer: `clip.cpp`, `clip*.h`, `mtmd-image.*`, `models/qwen2vl.cpp`, `models/qwen3vl.cpp`. Patch the graph-builder switch (`clip.cpp:977`) so the other ~40 model files are not needed. `clip.h` still includes `mtmd.h` -> `llama.h`, but only for types. Then call `clip_init`, `mtmd_image_preprocessor_dyn_size::preprocess` and `clip_image_batch_encode` (`clip.h:69, 90-91`). These are internal headers ("Internal header, to be used by mtmd only", `clip.h:11`), so they can change with any upstream commit.

### 3.2 C API (public, `tools/mtmd/mtmd.h`)

- Init: `mtmd_context_params_default`, `mtmd_init_from_file` (`mtmd.h:97-135`). Params include `device` (a `ggml_backend_dev_t`), `flash_attn_type`, `image_min_tokens`/`image_max_tokens`, `cb_eval`, `warmup`.
- Input: `mtmd_bitmap_init(nx, ny, rgb)`, `mtmd_tokenize(ctx, chunks, text, bitmaps, n)` (`mtmd.h:175, 302`).
- Encode: `mtmd_encode_chunk` then `mtmd_get_output_embd` -> `float *` (`mtmd.h:326-332`), or the batch API `mtmd_batch_init/add_chunk/encode/get_output_embd` (`mtmd.h:338-351`). The server uses the batch API (`server-context.cpp:820-853`).
- Positions: `mtmd_image_tokens_get_decoder_pos`, `mtmd_input_chunk_get_n_pos` (`mtmd.h:239, 284`).
- Debug: `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the final embeddings as raw `[int32 n_tokens][int32 n_embd][f32 ...]` (`clip.cpp:228, 5922-5940`). This is the reference for validating any own implementation.

### 3.3 License and size

- llama.cpp / ggml / mtmd: **MIT** (`LICENSE:1-3`). Vendored `stb_image`: public domain or MIT (`vendor/stb/stb_image.h:1` and its license footer). `miniaudio` is compiled into mtmd for audio: public domain or MIT-0 (`vendor/miniaudio/miniaudio.h:2`).
- DLL sizes in the production build `qwen38_27\bin-parches` (Windows, measured with `ls`): `mtmd.dll` 1.7 MiB, `llama.dll` 3.0 MiB, `ggml-cuda.dll` 34.3 MiB, `ggml-cpu.dll` 1.5 MiB, `ggml-base.dll` 0.8 MiB, `ggml.dll` 0.1 MiB. ggml-cuda also loads `cublas64_13.dll` 52.4 MiB and `cublasLt64_13.dll` 469.9 MiB. Linux sizes were not measured.

### 3.4 Card 1 only, and embeddings on the GPU

- **Card 1 only: yes.** `params.device` selects the backend device (`clip.cpp:188-193`). Production uses `-mmdev CUDA1` (`qwen38_27\arranca.ps1:68`). A CPU backend is also created, but only for unsupported ops, and there are none for this graph (see 1.3).
- **Embeddings on the GPU: no, not through the API.** `clip_encode` copies the last graph node to a host `std::vector<float>` with `ggml_backend_tensor_get` (`clip.cpp:5823-5845`). Getting a device pointer would need either a patch to clip, or the `cb_eval` callback to grab the tensor. The callback forces the scheduler to split compute at the observed node. The host round trip costs under 0.1 s (2.3), so this is not worth fighting for.
- **Speed: the same as today**, because it is the same code (3.5-5 s estimate, see 5).
- **Side effects in a non-ggml engine.** ggml-cuda keeps its own memory pool, streams and cuBLAS handle in the same process. The vision compute buffer is reserved at the first image and kept (`LEEME.md:220-222`).

---

## 4. Option B: own kernels

### 4.1 FLOPs for one 4K image (N = 16320 patches, T = 4080 tokens)

| Part | Formula | TFLOP |
|---|---|---|
| Linear, per block | 2·N·1152·(3456 + 1152 + 2·4304) = 0.497 | |
| Linear, 27 blocks | x 27 | **13.42** |
| Attention, per block | QK^T + PV = 2 x 2·N²·1152 = 1.227 | |
| Attention, 27 blocks | x 27 | **33.14** |
| Merger | 2·T·4608·(4608 + 5120) | 0.37 |
| Patch embedding | 2·N·768·1152 (one GEMM with W0+W1 pre-summed) | 0.03 |
| **Total** | | **46.95** |

Attention scales with N². With the HF cap (8160 tokens, 32640 patches) attention alone would be 132.5 TFLOP.

### 4.2 Peak rates for one RTX 5060 Ti (estimate, scaled from the whitepaper)

NVIDIA's RTX Blackwell whitepaper lists the RTX 5070 (48 SM, 2512 MHz) at:
- 30.9 TFLOPS FP32
- 30.9 TFLOPS FP16 non-tensor
- 123.5 dense TFLOPS FP16 tensor with FP16 accumulate
- **61.7** dense TFLOPS FP16/BF16 tensor with FP32 accumulate

(guru3d copy of the table, see Sources.) The 5060 Ti has the same SM type, with 36 SM at 2572 MHz. Scale factor = 36·2572 / (48·2512) = 0.768 (estimate). That gives:
- FP32 or FP16 non-tensor: **23.7 TFLOPS**
- BF16 tensor with FP32 accumulate: **47.4 TFLOPS**
- FP16 tensor with FP16 accumulate: 94.8 TFLOPS

Real clocks under load may differ.

### 4.3 Time estimates (all estimates)

| Part | llama.cpp today | Own kernels (B) | Arithmetic |
|---|---|---|---|
| GEMMs (13.8 TFLOP) | 0.4-0.5 s | 0.4-0.5 s | 13.8 / (28-36 TFLOPS = 60-75% of 47.4) |
| Attention (33.1 TFLOP) | **2.8-4.1 s** | **1.1-1.5 s** | today: SIMT tile kernel at 8-12 TFLOPS (35-50% of 23.7). B: tensor cores, d padded 72->80 makes 35.0 TFLOP, at 24-31 TFLOPS (50-65% of 47.4) |
| Elementwise (LN, bias, RoPE, GELU, casts, F32->BF16 conversions) | 0.2-0.3 s | 0.05-0.1 s | today: about 3.5 GB of memory traffic per block x 27 at ~350 GB/s. B: fused into GEMM epilogues and the FA prologue |
| **Encoder total** | **3.4-4.9 s** | **1.6-2.0 s** | |

A middle option: keep llama.cpp's clip (vendored or in production), but pad Q/K/V from 72 to 80 with zeros so that `fattn.cu` picks the MMA kernel. Kernel configs for d=80 exist in `fattn-mma-f16.cuh:44-47`, and the MMA kernel handles a KV length that is not a multiple of 256 (`oob_check`, `fattn-mma-f16.cuh:773-892`). The scale stays 1/sqrt(72). The zero V columns are dropped. Estimate: 36.8 TFLOP at 19-28 TFLOPS -> 1.3-1.9 s of attention, so **about 1.9-2.7 s** for the encoder. This is untested and must be measured.

Splitting the encoder across both cards does not pay. Each block needs two all-reduces of N x 1152 BF16 = 37.6 MB, which is 2 GB per image over PCIe Gen3 x4. Estimate: about 0.6 s of transfer to save about 0.8 s of compute.

### 4.4 Work list for B

| Piece | What | Reuse | Effort (estimate) |
|---|---|---|---|
| Decode | PNG/JPEG/BMP/GIF | `stb_image` (as llama.cpp) | < 1 day |
| Resize | smart_resize + Pillow bicubic + `PAD_CEIL`. Copy from `mtmd-image.cpp:122-157, 204+` (MIT) for parity. Could move to GPU later | copy | 1 day |
| Patchify + embed | GPU kernel: gather 3x16x16 patches in 2x2-merge order, element order c·256 + y·16 + x (matches the ggml weight [16,16,3,1152]). One GEMM with W0+W1 summed at load, + bias + position embedding | own | 1 day |
| Position table | bilinear with align-corners 48x48 -> (W/16)x(H/16), reordered to merge order, cached per grid size | own | 0.5 day |
| LayerNorm | F32 in, BF16 out, with weight and bias, eps 1e-6 | own | 0.5 day |
| GEMMs | qkv, out, up, down, mm.0, mm.2: BF16 x BF16 -> FP32 accumulate, epilogues bias / bias+GELU / +residual | cuBLASLt (already shipped), or own `mma.sync` GEMM for these fixed shapes | 1-2 days (cuBLASLt) |
| 2D RoPE | fused with the QKV split and the pad to 80 | own | 1 day |
| Flash attention | non-causal, 16 heads MHA, d=72 padded to 80, up to 16320 x 16320, FP32 softmax and accumulate | own `mma.sync` kernel, or a template variant of the engine's prefill FA (d=256, causal, GQA) | 3-7 days (1-2 if prefill FA exists) |
| Merger | post-LN, a free reshape, GEMM+bias+GELU, GEMM+bias -> write straight into the LLM input buffer | cuBLASLt | 0.5 day |
| LLM side | token layout, (t,h,w) positions, +max(nx,ny), cells versus positions, MTP feed, image hash in the prompt cache, copy to the other card | own | 2-3 days |
| Validation | compare with the `MTMD_DEBUG_EMBEDDINGS` dump on fixed images (target: per-token cosine >= 0.999, an estimate), then LLM logits after an image prompt | | 2-3 days |
| **Total** | about 1,000-1,500 lines C++/CUDA | | **1-1.5 weeks with existing GEMM+FA. 2-3 weeks without.** |

An unfused attention (GEMM + softmax + GEMM) is not an option. The score matrix is 16320² x 16 heads x 2 B = 8.5 GB per layer. Even when chunked, about 3 passes over it cost about 690 GB of traffic per image, which is about 2 s (estimate). The cuDNN fused SDPA would work for d=72, but cuDNN is a large extra dependency.

---

## 5. How much of the 12.7 s is the encoder

### 5.1 What the logs say

- `qwen38_27\arranque.log` is empty (0 bytes).
- `arranque.log.err` (3887 lines, about 742 minutes of server time) and `prueba.log.err` contain no image lines. The only vision line is `loaded multimodal model` (`arranque.log.err:7`).
- **Why.** The server runs at verbosity 3 (`arranque.log.err:2`). mtmd and clip log through `common_log_default_callback`, which maps `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` = 4 (`common/log.cpp:529-546`, `common/log.h:25`). So "decoding image batch" and "image decoded (batch i/n) in X ms" (`mtmd-helper.cpp:173, 191`) are hidden. The server also sets `print_timings = false` (`server-context.cpp:1079`) and does not time the encode itself (`server-context.cpp:846`).
- Prompt-eval lines with low t/s (13 of 165) all come from long contexts (90k-158k tokens at release). So they do not point to images.
- **How to measure (for the session that owns the GPU):**
  - `llama-mtmd-cli -lv 4` with the same `--mmproj`, `-mmdev CUDA1`, `--image-min-tokens 1024` prints "image slice encoded in N ms" (`mtmd-helper.cpp:247-255`).
  - `llama-server -lv 4` prints the LLM part per image batch. Encoder time is then about: slot prompt-eval time - Σ "image decoded" - text tokens.
  - `nsys` shows the `flash_attn_tile` kernels directly.

### 5.2 Estimated split of the 12.7 s (all estimates)

| Part | Time | Basis |
|---|---|---|
| HTTP + base64 + PNG decode + bicubic resize + normalize (CPU, one thread) | 0.3-0.7 s | not measured |
| Image upload (50 MB f32) and embeddings round trip (83.6 MB) | < 0.1 s | (50 + 3 x 83.6) MB / 3.5 GB/s ≈ 0.09 s |
| **Encoder on card 1** | **3.5-5 s** | FLOP model in 4.3 (3.4-4.9 s). By subtraction 12.7 - the other rows ≈ 4-6 s |
| LLM prefill of 4082 image/marker tokens + ~17 text tokens | ~5.6-5.9 s | 4099 tokens / 700-730 t/s. 726 t/s for the first 4096 tokens of a fresh prompt (`arranque.log.err:33`) |
| Answer | ≤ 1.6 s | `max_tokens` 120 (`prueba-imagen.py:10`) at ~75 tok/s |

**The largest part is probably the LLM prefill of the 4080 image tokens, not the encoder.** The image token count drives both. Faster vision in the new engine depends as much on prefill speed (another research topic) as on the encoder.

---

## 6. Recommendation

### 6.1 VRAM per option (on the card that runs the encoder)

| Option | Weights | Compute / activations | Total resident | Speed (encoder, 4K) | Dependencies |
|---|---|---|---|---|---|
| Today (llama.cpp, `-mmdev CUDA1`) | 888 MiB | about 0.55-0.7 GiB compute buffer, reserved at the first image and kept, plus ggml pool and cuBLAS workspace (unknown) | **about 1.5-1.8 GiB** (estimate) | 3.5-5 s | n/a |
| A1: public mtmd API + vocab-only libllama | 888 MiB | same | about 1.5-1.8 GiB | same as today | mtmd, llama, ggml, ggml-cpu, ggml-cuda (~41 MiB DLLs + cuBLAS) |
| A2: vendored clip layer + pad 72->80 | 888 MiB | same | about 1.5-1.8 GiB | 1.9-2.7 s (untested) | ggml, ggml-cpu, ggml-cuda |
| **B1: own kernels, resident weights** | ~875 MiB (W0+W1 pre-summed, pos table to BF16) | ~0.3 GB peak, **shared with the LLM prefill scratch** (they never run at the same time with one slot) | **about 0.9 GiB** | 1.6-2.0 s | cuBLASLt (or own GEMM) |
| **B2: own kernels, weights streamed per block from pinned RAM** | double buffer 2 x 31.5 MiB + merger 86 MiB | shared scratch | **about 0.15 GiB** | 1.6-2.0 s + ≤ 0.25 s if the upload does not overlap | as B1, plus 888 MiB pinned host RAM |

How the compute-buffer estimate was made:
- `LEEME.md:226-229` (17 Sep, encoder then on card 0): free VRAM after a 1432x950 image (5400 patches) was 924 MiB. After a 4K image (16320 patches) it was 555 MiB.
- The difference is 369 MiB for 10920 more patches, which is about 34 KB per patch.
- 34 KB x 16320 ≈ 0.55 GiB.
- This matches the FFN tensors of ggml (up output 17.2 KB/patch + GELU output 17.2 KB/patch, f32).

Context gained (estimate): with `-sm tensor` each card holds half the KV, 32 KiB per token per card (64 KiB/token from the brief, split 2/2 heads). If card 1 is the card that limits context, freeing 1.4 GiB there is worth about 1.4 x 1024² / 32 ≈ 46k tokens. On Ubuntu headless the two cards are equal, so the engine can put the encoder on either one.

Activation estimate for B (estimate):
- residual f32: 75 MB
- LN output BF16: 38 MB
- QKV padded to 80 in BF16: 125 MB
- FFN hidden BF16: 140 MB
- Peak ≈ 75 + 38 + 140 ≈ 0.25-0.3 GB.
- Processing the FFN in row chunks lowers it further.

### 6.2 What to do

1. **Engine, milestone 7: Option B.** Own kernels, with B2 (streamed weights) as the default if VRAM for context matters more than 0.25 s per image. Reasons:
   - No ggml or libllama in the process.
   - About 2x faster encoder (estimate).
   - The smallest VRAM.
   - The engine controls positions and cells itself, and it must write that logic anyway.
   - The parts it needs (BF16 GEMM, tensor-core flash attention) overlap with the prefill work.
2. **Correctness gate.** Compare embeddings with llama.cpp's `MTMD_DEBUG_EMBEDDINGS` dump on fixed images, using the same resize code (`PAD_CEIL`, Pillow bicubic) and tanh GELU in the merger. Then compare LLM logits after an image prompt.
3. **Bridge, if vision is needed before the kernels exist: A2.** Vendor the clip layer, MIT. It costs the ggml dependency and today's VRAM. Avoid A1 unless libllama is already linked for another reason.
4. **Production llama.cpp (optional, outside this project).** Pad the vision head dim from 72 to 80 in `clip_graph::build_attn` (`clip.cpp:750-791`) so CUDA uses the MMA flash-attention kernel. Estimated saving: 1.5-2.5 s per 4K image. It is unmeasured, and correctness must be checked with the embedding dump.
5. **Measure first (Phase 0).** Run `llama-mtmd-cli -lv 4` on the 4K test image to get the real encoder ms. Run `nsys` to get the attention share. Every encoder number in this file is an estimate until then.
6. **Keep the 4096-token cap.** Raising it toward the HF limit makes attention grow with N² (132 TFLOP for 8160 tokens). Lowering it (for example to 2048) cuts encoder and prefill time a lot, but it lowers the detail the model sees in screenshots. That trade-off is the user's choice.

Windows-only notes:
- The reason for `-mmdev CUDA1` (a 4K image hung the server when the encoder was on card 0, `LEEME.md:52`) involves the Windows desktop VRAM on card 0. It probably does not apply on Ubuntu headless. Not tested.
- Pinned host memory for B2 works on both systems. Its behaviour under WDDM memory pressure is not known.

---

## Sources

Local (read only), paths under `llama-rig2\` unless noted:
- `tools/mtmd/models/qwen3vl.cpp:3-186` (whole graph); `tools/mtmd/models/qwen2vl.cpp:3-16` (temporal conv); `tools/mtmd/models/models.h:42-51`
- `tools/mtmd/clip.cpp`: 181-229 (backend and device, min/max tokens, debug env), 248-274 (d_head, kq_scale), 312-332 (position resize), 591-704 (norm, ffn, gelu), 750-821 (attention, FA branch), 977 (builder switch), 1376-1385 (gelu), 1668-1686 (Qwen limits, bicubic, warmup), 2505-2511 (mm tensors), 3736-3797 (FA auto, unsupported-op warning), 4087-4218 (token counts), 4846-4870 (vision positions), 5816-5845 (output copied to host), 5922-5940 (embedding dump)
- `tools/mtmd/clip-model.h:67, 192-218`; `tools/mtmd/clip.h:11, 49-91`
- `tools/mtmd/mtmd-image.cpp:7-13, 38-157, 204, 772-791`
- `tools/mtmd/mtmd.cpp:202-231, 531-565, 694-704, 1061-1078, 1317-1361, 1501-1556, 1719-1735, 2178-2201, 2363-2368, 2478-2541`
- `tools/mtmd/mtmd-helper.cpp:118-199, 247-258, 404`; `tools/mtmd/mtmd-helper-common.h:85-150`
- `tools/mtmd/mtmd.h:97-135, 175, 237-239, 284, 302-351`; `tools/mtmd/CMakeLists.txt:15-79, 87-88`
- `tools/server/server-context.cpp:758-853, 1077-1085, 1203-1216`; `tools/server/server-common.cpp:1163`
- `common/log.cpp:529-546`; `common/log.h:25-32`
- `src/llama-graph.cpp:131-140, 2384-2420`; `src/llama-model.cpp:3207-3212`
- `ggml/src/ggml-cuda/fattn.cu:574-583, 638-660`; `fattn-tile.cu:16-19`; `fattn-tile.cuh:8, 34-38, 505-550`; `fattn-mma-f16.cuh:44-47, 773-892`
- `ggml/src/ggml-cuda/ggml-cuda.cu:1508-1520, 1621-1667, 1876-1929`; `ggml/src/ggml-cuda/mmf.cu:133-190`; `ggml/src/ggml-cuda/rope.cu:251-263, 294-352`
- `LICENSE:1-3`; `vendor/stb/stb_image.h:1`; `vendor/miniaudio/miniaudio.h:2`
- `qwen27-engine\research\_gguf-mmproj-dump.txt:24-59`; `research\_chat_template.jinja:18`; tensor list and token ids read with `gguf` (project venv) from `qwen38_27\models\Qwen3.8-27B\mmproj-Qwen3.8-27B-BF16.gguf` and `...\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- `qwen38_27\LEEME.md:50-52, 82, 86, 211-245`; `arranca.ps1:62-68, 80`; `prueba-imagen.py:10`; `arranque.log.err:2, 7, 33`; `arranque.log` (0 bytes); `bin-parches\*.dll` sizes

Web:
- HF Transformers Qwen3-VL model code (merger `nn.GELU()`, full attention with cu_seqlens, axial RoPE, bilinear align-corners position interpolation, deepstack): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- Qwen3-VL vision config (`gelu_pytorch_tanh`, depth 27, hidden 1152, 16 heads, 2304 position embeddings): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
- Qwen3-VL preprocessor limits (`longest_edge` 16777216, `shortest_edge` 65536, mean/std 0.5): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json
- RTX Blackwell whitepaper table for RTX 5070 (FP32 30.9, FP16 non-tensor 30.9, FP16/BF16 tensor with FP32 accumulate 61.7 dense, FP16 accumulate 123.5 dense, 48 SM, 2512 MHz): https://www.guru3d.com/story/nvidia-discloses-blackwell-architecture-whitepaper-detailed-look-at-geforce-rtx-5070-ti-and-5070/
- RTX 5060 Ti: 36 SM, 2572 MHz boost: https://www.storagereview.com/review/pny-geforce-rtx-5060-ti-review-blackwells-entry-point-for-creators-and-developers
