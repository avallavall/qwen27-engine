# Third-party notices

Parts of this engine copy code, lookup tables or numeric formulas from llama.cpp / ggml
(https://github.com/ggml-org/llama.cpp):

- `src/quant_tables.h` (IQ codebooks, generated from `ggml/src/ggml-common.h`)
- per-type dot-product math in `src/qgemv.cu` (from `ggml/src/ggml-cuda/vecdotq.cuh`)
- kernel numerics in `src/ops.cu` and `src/sampling.cu`, log-prob compression in `tools/q27_ppl.cu`
- the tokenizer in `src/tokenizer.cpp` (BPE merge loop, `qwen35` pre-tokenizer, special-token split, token
  attributes, EOG set and token pieces, ported from `src/llama-vocab.cpp` and `src/unicode.cpp`)
- `src/unicode_tables.cpp` (Unicode category and whitespace tables, copied from `src/unicode-data.cpp`)
- `src/vision_img.cpp` (image size plan `calc_size_preserved_ratio`, PAD_CEIL resize and the Pillow-style bicubic
  resampler, ported from `tools/mtmd/mtmd-image.cpp`; the resampler is itself adapted from Pillow's `Resample.c`)
  and the vision encoder graph and numerics in `src/vision.cu` (from `tools/mtmd/models/qwen3vl.cpp`, `clip.cpp`)

License of llama.cpp:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Vendored libraries (`third_party/`)

- cpp-httplib 0.58.0 (`third_party/httplib/`): HTTP server of `tools/q27_server.cpp`. MIT license, in
  `third_party/httplib/LICENSE`.
- nlohmann/json 3.12.0 (`third_party/nlohmann/json.hpp`): JSON. MIT license, at the top of the header.
- stb_image 2.30 (`third_party/stb/stb_image.h`): image decoding in `src/vision_img.cpp`. Public domain or MIT,
  at the end of the header.
- Pillow: the bicubic resampler in `src/vision_img.cpp` comes from Pillow's `Resample.c` through llama.cpp.
  Pillow uses the HPND license (MIT-like): https://github.com/python-pillow/Pillow/blob/main/LICENSE
