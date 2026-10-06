// Vision encoder for the Qwen3.8-27B mmproj (projector type qwen3vl_merger) with own CUDA kernels.
// See research/vision.md. The graph is llama.cpp's tools/mtmd/models/qwen3vl.cpp:
//   patch embedding (two 16x16 convs, summed) in 2x2-merge order + bias + learned 48x48 position table
//   (bilinear, align corners), 27 pre-LN ViT blocks (full attention, 16 heads of 72, 2D RoPE, GELU-tanh FFN),
//   post-LN, merger (4 patches -> 4608, mm.0 + GELU-tanh, mm.2 -> 5120).
// Output: f32 [n_tokens][5120], tokens row-major over the merged grid (nx columns, ny rows), the order of
// llama.cpp's mtmd output.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q27 {

// Size decisions for one image (llama.cpp mtmd_image_preprocessor_dyn_size + img_tool::resize, PAD_CEIL).
struct VisionPlan {
  int src_w = 0, src_h = 0;          // decoded image
  int width = 0, height = 0;         // encoder canvas (multiples of 32)
  int content_w = 0, content_h = 0;  // the resized image inside the canvas
  int off_x = 0, off_y = 0;          // its top-left corner; the rest of the canvas is black
  int px = 0, py = 0;                // patches per row / per column (width / 16, height / 16)
  int nx = 0, ny = 0;                // token grid after the 2x2 merge
  int n_patches = 0, n_tokens = 0;   // px * py, nx * ny
};

// Milliseconds per stage of the last encode() with profiling on (GPU stages from CUDA events).
struct VisionTimes {
  double resize_cpu = 0, upload = 0, patch_embed = 0, layernorm = 0, qkv = 0, attention = 0, out_proj = 0,
         ffn_up = 0, ffn_down = 0, merger = 0, gpu_total = 0;
};

class VisionEncoder {
 public:
  static constexpr int kEmbd = 5120;  // output width (= LLM embedding length)

  // Loads the mmproj GGUF (BF16 + F32) to `device`. Weights stay resident.
  VisionEncoder(const std::string& mmproj_path, int device);
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  // Image file bytes (PNG, JPEG, BMP, GIF first frame, TGA, PSD, HDR, PNM: what stb_image reads) -> RGB8.
  // Throws std::runtime_error with stb's reason on failure.
  static std::vector<uint8_t> decode(const void* bytes, size_t n, int& w, int& h);

  // llama.cpp's smart_resize for Qwen3-VL (factor 32). min/max_tokens as llama-server's
  // --image-min-tokens / --image-max-tokens (production: 1024 and the model default 4096).
  static VisionPlan plan(int w, int h, int min_tokens = 1024, int max_tokens = 4096);

  // CPU resize to the plan's canvas exactly as llama.cpp (Pillow bicubic, PAD_CEIL with black).
  // dst: width * height * 3 bytes. Threads split rows; the result does not depend on the thread count.
  static void resize_into(const uint8_t* rgb, int w, int h, const VisionPlan& p, uint8_t* dst, int threads = 8);
  static std::vector<uint8_t> resize(const uint8_t* rgb, int w, int h, const VisionPlan& p, int threads = 8);

  // Resize (CPU), upload, run the encoder on `stream` (must belong to device()). Returns a device pointer to
  // f32 [p.n_tokens][kEmbd], owned by the encoder and valid until the next encode(). The work is queued on
  // `stream`; synchronize it before reading. rgb is the decoded image (w x h, RGB8). One encoder runs one image at
  // a time: before the next encode() on another stream, the previous one must be finished (same stream is fine).
  // Q27_VIS_PV16=1 in the environment: attention sums P V per 64-position tile in f16 (about 0.2 s faster on a
  // 4K image, slightly less accurate; default off).
  const float* encode(const uint8_t* rgb, int w, int h, const VisionPlan& p, cudaStream_t stream);

  int device() const;
  size_t weight_bytes() const;   // device memory held by the weights
  size_t scratch_bytes() const;  // device memory held by activations and the output (grows with the image)

  // Profiling: when on, encode() records events per stage and synchronizes the stream at the end.
  void set_profile(bool on);
  const VisionTimes& times() const;

  // Debug (tests): run only the first n blocks (0 = stop after the patch embedding + position table) and skip the
  // merger; -1 = full encoder. debug_residual() = device f32 [n_patches][1152] residual stream, merge order.
  void set_debug_stop(int n_layers);
  const float* debug_residual() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

// ---- kernel entry points, exposed for tools/test_vision (all on the current device)
namespace vis {
constexpr int kHid = 1152, kHeads = 16, kHeadDim = 72, kFfn = 4304, kFfnPad = 4352, kMerge = 4608;
enum Epi : int { EPI_QKV = 0, EPI_RES = 1, EPI_GELU = 2, EPI_F32 = 3 };
// out = epi(A[M][K] . W[N][K]^T + bias). bf16 inputs, f32 accumulate (mma.sync m16n8k16).
//   EPI_RES:  float out[M][ldo] += result     EPI_GELU: bf16 out = gelu_tanh(result)     EPI_F32: float out = result
//   EPI_QKV:  N = 3*1152; Q,K columns are in pair order (see vision.cu), get 2D RoPE; Q,K,V written as f16
//             [3][16][n_rows][72] into out. rope: float2 [n_pos][18] (cos, sin); nx2 = merged columns (px / 2).
// K must be a multiple of 32 and N of 128.
void gemm(int epi, const __nv_bfloat16* A, int lda, const __nv_bfloat16* W, int ldw, const float* bias, int M, int N,
          int K, void* out, int ldo, cudaStream_t s, const float2* rope = nullptr, int nx2 = 0, int n_rows = 0);
// Non-causal attention, 16 heads, head dim 72: q, k, v f16 [16][n][72] -> out bf16 [n][16*72].
void attention(const __half* q, const __half* k, const __half* v, __nv_bfloat16* out, int n, cudaStream_t s);
// LayerNorm over 1152 (eps 1e-6, weight and bias): f32 rows -> bf16 rows.
void layernorm(const float* x, const float* w, const float* b, __nv_bfloat16* y, int rows, cudaStream_t s);
}  // namespace vis

}  // namespace q27
