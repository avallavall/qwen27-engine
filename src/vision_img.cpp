// Host side of the vision encoder: decode (stb_image), size plan (smart_resize) and resize (Pillow bicubic +
// PAD_CEIL). The plan and the resize are ported from llama.cpp tools/mtmd/mtmd-image.cpp (MIT, see
// THIRD_PARTY_NOTICES.md), llama-rig2 commit e2377cc96:
//   img_tool::calc_size_preserved_ratio   mtmd-image.cpp:122-157
//   img_tool::resize (PAD_CEIL branch)    mtmd-image.cpp:39-93
//   img_tool::resize_pillow               mtmd-image.cpp:204-486 (itself adapted from Pillow Resample.c)
//   dyn_size preprocessor (factor 32)     mtmd-image.cpp:772-791; limits clip.cpp:1668-1686, clip-model.h:192-197
// The float expressions are kept as in llama.cpp, so the canvas and content sizes match bit for bit
// (compile without /fp:fast).
#include "vision.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

namespace q27 {

std::vector<uint8_t> VisionEncoder::decode(const void* bytes, size_t n, int& w, int& h) {
  int nc = 0;
  w = h = 0;
  // as llama.cpp mtmd-helper.cpp:404: stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3)
  stbi_uc* data = stbi_load_from_memory((const stbi_uc*)bytes, (int)n, &w, &h, &nc, 3);
  if (!data) throw std::runtime_error(std::string("image decode failed: ") + stbi_failure_reason());
  std::vector<uint8_t> out(data, data + (size_t)w * h * 3);
  stbi_image_free(data);
  return out;
}

VisionPlan VisionEncoder::plan(int width, int height, int min_tokens, int max_tokens) {
  if (width <= 0 || height <= 0) throw std::runtime_error("vision plan: empty image");
  const int patch = 16, merge = 2, align = patch * merge;  // factor 32
  const int patch_area = patch * patch * merge * merge;     // 1024 pixels per token
  const int min_pixels = min_tokens * patch_area, max_pixels = max_tokens * patch_area;

  // mtmd-image.cpp:122-157 (longest_edge = 0)
  auto round_by_factor = [f = align](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
  auto ceil_by_factor = [f = align](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
  auto floor_by_factor = [f = align](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };
  int w_bar = std::max(align, round_by_factor(width));
  int h_bar = std::max(align, round_by_factor(height));
  if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
    const auto beta = std::sqrt(static_cast<float>(height) * width / max_pixels);
    h_bar = std::max(align, floor_by_factor(height / beta));
    w_bar = std::max(align, floor_by_factor(width / beta));
  } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
    const auto beta = std::sqrt(static_cast<float>(min_pixels) / (static_cast<float>(height) * width));
    h_bar = ceil_by_factor(height * beta);
    w_bar = ceil_by_factor(width * beta);
  }

  VisionPlan p;
  p.src_w = width;
  p.src_h = height;
  p.width = w_bar;
  p.height = h_bar;
  // mtmd-image.cpp:63-91, PAD_CEIL: scale by min(scale_w, scale_h), ceil, center with integer division
  if (w_bar == width && h_bar == height) {
    p.content_w = width;
    p.content_h = height;
  } else {
    float scale_w = static_cast<float>(w_bar) / width;
    float scale_h = static_cast<float>(h_bar) / height;
    float scale = std::min(scale_w, scale_h);
    p.content_w = std::min(static_cast<int>(std::ceil(width * scale)), w_bar);
    p.content_h = std::min(static_cast<int>(std::ceil(height * scale)), h_bar);
  }
  p.off_x = (w_bar - p.content_w) / 2;
  p.off_y = (h_bar - p.content_h) / 2;
  p.px = w_bar / patch;
  p.py = h_bar / patch;
  p.nx = p.px / merge;
  p.ny = p.py / merge;
  p.n_patches = p.px * p.py;
  p.n_tokens = p.nx * p.ny;
  return p;
}

namespace {

// ---- Pillow-compatible bicubic resampling, from mtmd-image.cpp:204-486 (bicubic only, a = -0.5)
constexpr int PRECISION_BITS = 32 - 8 - 2;

double bicubic_filter(double x) {
  if (x < 0.0) x = -x;
  constexpr double a = -0.5;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

inline uint8_t clip8(int val) { return val < 0 ? 0 : val > 255 ? 255 : (uint8_t)val; }

// mtmd-image.cpp:278-359 (precompute_weights)
int precompute_weights(int inSize, int outSize, std::vector<int>& bounds, std::vector<int32_t>& weights) {
  const double filter_support = 2.0;
  double support, scale, filterscale;
  double center, ww, ss;
  int xx, x, ksize, xmin, xmax;
  filterscale = scale = static_cast<double>(inSize) / outSize;
  if (filterscale < 1.0) filterscale = 1.0;
  support = filter_support * filterscale;
  ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  std::vector<double> pre_weights((size_t)outSize * ksize);
  bounds.resize((size_t)outSize * 2);
  for (xx = 0; xx < outSize; xx++) {
    center = (xx + 0.5) * scale;
    ww = 0.0;
    ss = 1.0 / filterscale;
    xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    xmax = static_cast<int>(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    for (x = 0; x < xmax; x++) {
      double w = bicubic_filter((x + xmin - center + 0.5) * ss);
      pre_weights[(size_t)xx * ksize + x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++) {
      if (ww != 0.0) pre_weights[(size_t)xx * ksize + x] /= ww;
    }
    for (; x < ksize; x++) pre_weights[(size_t)xx * ksize + x] = 0;
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  weights.resize((size_t)outSize * ksize);
  const double fxp_scale = std::ldexp(1.0, PRECISION_BITS);
  for (size_t i = 0; i < (size_t)outSize * ksize; i++) {
    const double rounded = pre_weights[i] * fxp_scale + (pre_weights[i] < 0 ? -0.5 : 0.5);
    weights[i] = static_cast<int32_t>(rounded);
  }
  return ksize;
}

template <typename F>
void parallel_rows(int n, int threads, F&& f) {
  threads = std::max(1, std::min(threads, n / 64 + 1));
  if (threads == 1) { f(0, n); return; }
  std::vector<std::thread> th;
  const int per = (n + threads - 1) / threads;
  for (int t = 0; t < threads; t++) {
    const int a = t * per, b = std::min(n, a + per);
    if (a < b) th.emplace_back([&f, a, b] { f(a, b); });
  }
  for (auto& t : th) t.join();
}

// mtmd-image.cpp:363-404 (resample_horizontal), rows [y0, y1)
void resample_horizontal(const uint8_t* src, int in_nx, uint8_t* out, int out_nx, int ksize, const std::vector<int>& bounds,
                         const std::vector<int32_t>& weights, int y0, int y1) {
  for (int yy = y0; yy < y1; yy++) {
    const uint8_t* src_row = src + (size_t)yy * in_nx * 3;
    uint8_t* dst_row = out + (size_t)yy * out_nx * 3;
    for (int xx = 0; xx < out_nx; xx++) {
      const int xmin = bounds[xx * 2 + 0];
      const int xcnt = bounds[xx * 2 + 1];
      const int32_t* k = &weights[(size_t)xx * ksize];
      const uint8_t* p = src_row + (size_t)xmin * 3;
      int32_t ss0 = 1 << (PRECISION_BITS - 1);
      int32_t ss1 = 1 << (PRECISION_BITS - 1);
      int32_t ss2 = 1 << (PRECISION_BITS - 1);
      for (int x = 0; x < xcnt; x++) {
        ss0 += p[0] * k[x];
        ss1 += p[1] * k[x];
        ss2 += p[2] * k[x];
        p += 3;
      }
      dst_row[xx * 3 + 0] = clip8(ss0 >> PRECISION_BITS);
      dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
      dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
    }
  }
}

// mtmd-image.cpp:406-439 (resample_vertical), output rows [y0, y1); out has row stride out_stride bytes
void resample_vertical(const uint8_t* src, int in_nx, uint8_t* out, size_t out_stride, int ksize, const std::vector<int>& bounds,
                       const std::vector<int32_t>& weight, int y0, int y1) {
  const size_t row_elems = (size_t)in_nx * 3;
  std::vector<int32_t> acc(row_elems);
  for (int yy = y0; yy < y1; yy++) {
    const int ymin = bounds[yy * 2 + 0];
    const int ycnt = bounds[yy * 2 + 1];
    const int32_t* k = &weight[(size_t)yy * ksize];
    std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
    for (int y = 0; y < ycnt; y++) {
      const uint8_t* src_row = src + (size_t)(ymin + y) * row_elems;
      const int32_t w = k[y];
      for (size_t i = 0; i < row_elems; i++) acc[i] += src_row[i] * w;
    }
    uint8_t* dst_row = out + (size_t)yy * out_stride;
    for (size_t i = 0; i < row_elems; i++) dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
  }
}

// resize_pillow main part (mtmd-image.cpp:441-485), writing the tw x th result into dst with row stride dst_stride bytes
void resize_pillow(const uint8_t* src, int sw, int sh, uint8_t* dst, size_t dst_stride, int tw, int th, int threads) {
  if (tw <= 0 || tw > 65536 || th <= 0 || th > 65536) throw std::runtime_error("vision resize: target out of range");
  const bool need_h = tw != sw, need_v = th != sh;
  std::vector<int> bh, bv;
  std::vector<int32_t> wh, wv;
  int kh = 0, kv = 0;
  if (need_h) kh = precompute_weights(sw, tw, bh, wh);
  if (need_v) kv = precompute_weights(sh, th, bv, wv);
  if (need_h && need_v) {
    std::vector<uint8_t> tmp((size_t)tw * sh * 3);
    parallel_rows(sh, threads, [&](int a, int b) { resample_horizontal(src, sw, tmp.data(), tw, kh, bh, wh, a, b); });
    parallel_rows(th, threads, [&](int a, int b) { resample_vertical(tmp.data(), tw, dst, dst_stride, kv, bv, wv, a, b); });
  } else if (need_h) {
    std::vector<uint8_t> tmp((size_t)tw * sh * 3);
    parallel_rows(sh, threads, [&](int a, int b) { resample_horizontal(src, sw, tmp.data(), tw, kh, bh, wh, a, b); });
    for (int y = 0; y < th; y++) std::copy_n(tmp.data() + (size_t)y * tw * 3, (size_t)tw * 3, dst + (size_t)y * dst_stride);
  } else if (need_v) {
    parallel_rows(th, threads, [&](int a, int b) { resample_vertical(src, sw, dst, dst_stride, kv, bv, wv, a, b); });
  } else {
    for (int y = 0; y < th; y++) std::copy_n(src + (size_t)y * sw * 3, (size_t)sw * 3, dst + (size_t)y * dst_stride);
  }
}

}  // namespace

void VisionEncoder::resize_into(const uint8_t* rgb, int w, int h, const VisionPlan& p, uint8_t* dst, int threads) {
  if (w != p.src_w || h != p.src_h) throw std::runtime_error("vision resize: image size differs from the plan");
  const size_t stride = (size_t)p.width * 3;
  if (p.width == w && p.height == h) {  // mtmd-image.cpp:53-57: same size, plain copy
    std::copy_n(rgb, (size_t)w * h * 3, dst);
    return;
  }
  // PAD_CEIL: black canvas, resized content composited at (off_x, off_y)
  std::fill_n(dst, stride * p.height, (uint8_t)0);
  resize_pillow(rgb, w, h, dst + (size_t)p.off_y * stride + (size_t)p.off_x * 3, stride, p.content_w, p.content_h, threads);
}

std::vector<uint8_t> VisionEncoder::resize(const uint8_t* rgb, int w, int h, const VisionPlan& p, int threads) {
  std::vector<uint8_t> out((size_t)p.width * p.height * 3);
  resize_into(rgb, w, h, p, out.data(), threads);
  return out;
}

}  // namespace q27
