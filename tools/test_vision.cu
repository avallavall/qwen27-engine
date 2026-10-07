// Tests of the vision encoder (src/vision.*). Card 1 by default (CUDA_DEVICE_ORDER=PCI_BUS_ID is set here).
//
//   test_vision <mmproj.gguf> unit [--dev N]
//       kernel checks against simple reference kernels: BF16 GEMM (every epilogue, odd M), the QKV + RoPE epilogue,
//       attention (partial tiles), LayerNorm.
//   test_vision <mmproj.gguf> compare <dir> [name ...] [--dev N] [--min-tokens N] [--max-tokens N]
//       for each <dir>/<name>.png with <name>.emb.bin from bench/llama_vision (llama.cpp mtmd on the CPU):
//       grid and token count, input image vs <name>.img.bin (if present), per-token cosine and relative RMS.
//       Without names: every <name>.emb.bin in <dir>. Pass: same grid, cosine mean >= 0.999 and min >= 0.99.
//   test_vision <mmproj.gguf> bench [W H | image] [--dev N] [--runs N]
//       synthetic W x H image (default 3840 x 2160) or an image file (decode timed): time per stage, device memory.
//   test_vision <mmproj.gguf> layers <dir> <name> [--min-tokens N]
//       residual stream after the position table and after each block vs llama_vision --dump files
//       (<name>.inp_pos_emb.bin, <name>.layer_out-<i>.bin), e.g. "layers bench\out\vision sq448_min8 --min-tokens 8".
//   test_vision <mmproj.gguf> dump <image> <out.bin> [--dev N]
//       our embeddings in the llama.cpp dump format [int32 n_tokens][int32 n_embd][f32 ...].
#include "common.cuh"
#include "vision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace q27;
using bf16 = __nv_bfloat16;
using clk = std::chrono::steady_clock;
namespace fs = std::filesystem;

static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static float bf2f(bf16 v) { return __bfloat162float(v); }
static float h2f(__half v) { return __half2float(v); }

template <typename T>
struct DBuf {
  T* p = nullptr;
  size_t n = 0;
  explicit DBuf(size_t n_) : n(n_) { CK(cudaMalloc(&p, n * sizeof(T))); }
  ~DBuf() { cudaFree(p); }
  void up(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> down() const {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// ---------------------------------------------------------------- reference kernels
__global__ void ref_gemm_kernel(const bf16* A, int lda, const bf16* W, int ldw, const float* bias, int M, int N, int K, float* out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col >= N || row >= M) return;
  double s = 0;
  for (int k = 0; k < K; k++) s += (double)__bfloat162float(A[(size_t)row * lda + k]) * __bfloat162float(W[(size_t)col * ldw + k]);
  out[(size_t)row * N + col] = (float)s + bias[col];
}

// one block per (query row index in `rows`, head); scores in dynamic shared memory
__global__ void ref_attn_kernel(const __half* q, const __half* k, const __half* v, const int* rows, int n, float scale, float* out) {
  extern __shared__ float sc[];
  __shared__ float red[128];
  const int r = rows[blockIdx.x], h = blockIdx.y, t = threadIdx.x;
  const __half* Q = q + ((size_t)h * n + r) * 72;
  float mx = -INFINITY;
  for (int j = t; j < n; j += 128) {
    const __half* Kj = k + ((size_t)h * n + j) * 72;
    float s = 0;
    for (int d = 0; d < 72; d++) s += __half2float(Q[d]) * __half2float(Kj[d]);
    s *= scale;
    sc[j] = s;
    mx = fmaxf(mx, s);
  }
  red[t] = mx;
  __syncthreads();
  for (int o = 64; o > 0; o >>= 1) {
    if (t < o) red[t] = fmaxf(red[t], red[t + o]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float sum = 0;
  for (int j = t; j < n; j += 128) {
    sc[j] = expf(sc[j] - mx);
    sum += sc[j];
  }
  red[t] = sum;
  __syncthreads();
  for (int o = 64; o > 0; o >>= 1) {
    if (t < o) red[t] += red[t + o];
    __syncthreads();
  }
  sum = red[0];
  if (t < 72) {
    double a = 0;
    for (int j = 0; j < n; j++) a += (double)sc[j] * __half2float(v[((size_t)h * n + j) * 72 + t]);
    out[((size_t)blockIdx.x * 16 + h) * 72 + t] = (float)(a / sum);
  }
}

// ---------------------------------------------------------------- unit tests
// rel = relative RMS error; max_el = max over elements of |d| / (|ref| + floor * rms(ref)) (element-wise relative
// error with a floor for values near zero). BF16 outputs have rel ~1.7e-3 and max_el <= 2^-9 from the rounding alone.
struct Err {
  double max_abs = 0, ref_rms = 0, rel = 0, max_el = 0;
};
static Err compare(const std::vector<float>& got, const std::vector<float>& ref, double floor = 0.02) {
  Err e;
  double s = 0, d2 = 0;
  for (size_t i = 0; i < ref.size(); i++) s += (double)ref[i] * ref[i];
  e.ref_rms = std::sqrt(s / ref.size());
  for (size_t i = 0; i < ref.size(); i++) {
    const double d = std::fabs((double)got[i] - ref[i]);
    e.max_abs = std::max(e.max_abs, d);
    e.max_el = std::max(e.max_el, d / (std::fabs((double)ref[i]) + floor * e.ref_rms));
    d2 += d * d;
  }
  e.rel = std::sqrt(d2 / std::max(s, 1e-30));
  return e;
}

static int g_fail = 0;
static void report(const char* name, const Err& e, double tol_rel, double tol_el) {
  const bool ok = e.rel <= tol_rel && e.max_el <= tol_el;
  printf("  %-34s %s  rel rms %.2e  max el. rel %.2e  max|d| %.2e (ref rms %.2e)\n", name, ok ? "PASS" : "FAIL", e.rel, e.max_el,
         e.max_abs, e.ref_rms);
  if (!ok) g_fail++;
}

static std::vector<bf16> rand_bf16(size_t n, std::mt19937& rng, float scale) {
  std::normal_distribution<float> nd(0.f, scale);
  std::vector<bf16> v(n);
  for (auto& x : v) x = __float2bfloat16(nd(rng));
  return v;
}
static std::vector<float> rand_f32(size_t n, std::mt19937& rng, float scale) {
  std::normal_distribution<float> nd(0.f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}

static void test_gemm(int M, int N, int K, std::mt19937& rng) {
  printf("GEMM M=%d N=%d K=%d\n", M, N, K);
  DBuf<bf16> A((size_t)M * K), W((size_t)N * K);
  DBuf<float> bias(N), ref((size_t)M * N), out((size_t)M * N);
  A.up(rand_bf16(A.n, rng, 1.0f));
  W.up(rand_bf16(W.n, rng, 1.0f / std::sqrt((float)K)));
  bias.up(rand_f32(N, rng, 0.5f));
  ref_gemm_kernel<<<dim3((N + 127) / 128, M), 128>>>(A.p, K, W.p, K, bias.p, M, N, K, ref.p);
  CK(cudaGetLastError());
  const std::vector<float> r = ref.down();
  // F32
  vis::gemm(vis::EPI_F32, A.p, K, W.p, K, bias.p, M, N, K, out.p, N, 0);
  report("f32 epilogue", compare(out.down(), r), 1e-5, 1e-3);
  // RES: out = init + result
  std::vector<float> init = rand_f32(out.n, rng, 1.0f);
  out.up(init);
  vis::gemm(vis::EPI_RES, A.p, K, W.p, K, bias.p, M, N, K, out.p, N, 0);
  {
    std::vector<float> rr(r.size());
    for (size_t i = 0; i < r.size(); i++) rr[i] = r[i] + init[i];
    report("residual epilogue", compare(out.down(), rr), 1e-5, 1e-3);
  }
  // GELU -> bf16
  {
    DBuf<bf16> ob((size_t)M * N);
    vis::gemm(vis::EPI_GELU, A.p, K, W.p, K, bias.p, M, N, K, ob.p, N, 0);
    const std::vector<bf16> g = ob.down();
    std::vector<float> got(g.size()), rr(r.size());
    for (size_t i = 0; i < r.size(); i++) {
      const double x = r[i];
      rr[i] = (float)(0.5 * x * (1.0 + std::tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))));
      got[i] = bf2f(g[i]);
    }
    report("gelu epilogue (bf16 out)", compare(got, rr), 2.5e-3, 5e-3);
  }
}

// QKV epilogue: compare against the f32 GEMM result processed on the host (pair order, RoPE, f16 layout)
static void test_qkv(int px, int py, std::mt19937& rng) {
  const int M = px * py, N = 3 * vis::kHid, K = vis::kHid, nx2 = px / 2;
  printf("QKV epilogue, grid %dx%d patches (M=%d)\n", px, py, M);
  DBuf<bf16> A((size_t)M * K), W((size_t)N * K);
  DBuf<float> bias(N), lin((size_t)M * N);
  A.up(rand_bf16(A.n, rng, 1.0f));
  W.up(rand_bf16(W.n, rng, 1.0f / std::sqrt((float)K)));
  bias.up(rand_f32(N, rng, 0.5f));
  const int npos = std::max(px, py);
  std::vector<float2> rope((size_t)npos * 18);
  for (int p = 0; p < npos; p++)
    for (int i = 0; i < 18; i++) {
      const double th = p * std::pow(10000.0, -2.0 * i / 36.0);
      rope[p * 18 + i] = make_float2((float)std::cos(th), (float)std::sin(th));
    }
  DBuf<float2> drope(rope.size());
  drope.up(rope);
  DBuf<__half> qkv((size_t)3 * 16 * M * 72);
  vis::gemm(vis::EPI_F32, A.p, K, W.p, K, bias.p, M, N, K, lin.p, N, 0);
  vis::gemm(vis::EPI_QKV, A.p, K, W.p, K, bias.p, M, N, K, qkv.p, 0, 0, drope.p, nx2, M);
  const std::vector<float> L = lin.down();
  const std::vector<__half> G = qkv.down();
  std::vector<float> got, ref;
  got.reserve(G.size());
  ref.reserve(G.size());
  for (int r = 0; r < M; r++) {
    const int b = r / 4, sub = r % 4, by = b / nx2, bx = b % nx2;
    const int ph = 2 * by + sub / 2, pw = 2 * bx + sub % 2;
    for (int part = 0; part < 3; part++)
      for (int h = 0; h < 16; h++)
        for (int d = 0; d < 72; d += 2) {
          const size_t base = (size_t)r * N + part * 1152 + h * 72;
          double v0 = L[base + d], v1 = L[base + d + 1];
          if (part < 2) {
            // weights were not permuted in this test, so the "pair" is columns (d, d+1) as stored
            const int pi = d / 2;
            const int pos = pi < 18 ? ph : pw;
            const double th = pos * std::pow(10000.0, -2.0 * (pi % 18) / 36.0);
            const double a = v0 * std::cos(th) - v1 * std::sin(th), c = v0 * std::sin(th) + v1 * std::cos(th);
            v0 = a;
            v1 = c;
          }
          const size_t o = (((size_t)(part * 16 + h)) * M + r) * 72 + d;
          got.push_back(h2f(G[o]));
          got.push_back(h2f(G[o + 1]));
          ref.push_back((float)v0);
          ref.push_back((float)v1);
        }
  }
  report("qkv + rope epilogue (f16 out)", compare(got, ref), 5e-4, 1e-3);
}

static void test_attention(int n, int nrows_check, std::mt19937& rng) {
  printf("attention n=%d (checking %d query rows x 16 heads)\n", n, nrows_check);
  const size_t sz = (size_t)16 * n * 72;
  std::vector<__half> hq(sz), hk(sz), hv(sz);
  std::normal_distribution<float> nd(0.f, 1.f);
  // scores with a realistic spread: q, k ~ N(0, 1.5) -> q.k / sqrt(72) ~ N(0, 2.25)
  for (size_t i = 0; i < sz; i++) {
    hq[i] = __float2half(1.5f * nd(rng));
    hk[i] = __float2half(1.5f * nd(rng));
    hv[i] = __float2half(nd(rng));
  }
  DBuf<__half> q(sz), k(sz), v(sz);
  q.up(hq);
  k.up(hk);
  v.up(hv);
  DBuf<bf16> out((size_t)n * 1152);
  vis::attention(q.p, k.p, v.p, out.p, n, 0);
  std::vector<int> rows;
  for (int i = 0; i < nrows_check; i++) rows.push_back((int)((long long)i * (n - 1) / std::max(1, nrows_check - 1)));
  DBuf<int> drows(rows.size());
  drows.up(rows);
  DBuf<float> ref(rows.size() * 16 * 72);
  ref_attn_kernel<<<dim3((unsigned)rows.size(), 16), 128, n * sizeof(float)>>>(q.p, k.p, v.p, drows.p, n, 1.0f / std::sqrt(72.0f), ref.p);
  CK(cudaGetLastError());
  const std::vector<float> r = ref.down();
  const std::vector<bf16> o = out.down();
  std::vector<float> got(r.size());
  for (size_t i = 0; i < rows.size(); i++)
    for (int h = 0; h < 16; h++)
      for (int d = 0; d < 72; d++) got[(i * 16 + h) * 72 + d] = bf2f(o[(size_t)rows[i] * 1152 + h * 72 + d]);
  report("attention vs naive f32", compare(got, r, 0.1), 2.5e-3, 1e-2);
}

static void test_layernorm(int rows, std::mt19937& rng) {
  printf("layernorm rows=%d\n", rows);
  std::vector<float> x = rand_f32((size_t)rows * 1152, rng, 3.0f), w = rand_f32(1152, rng, 1.0f), b = rand_f32(1152, rng, 0.3f);
  for (int r = 0; r < rows; r++)
    for (int i = 0; i < 1152; i++) x[(size_t)r * 1152 + i] += (float)(r % 7);  // nonzero means
  DBuf<float> dx(x.size()), dw(1152), db(1152);
  dx.up(x);
  dw.up(w);
  db.up(b);
  DBuf<bf16> y(x.size());
  vis::layernorm(dx.p, dw.p, db.p, y.p, rows, 0);
  const std::vector<bf16> g = y.down();
  std::vector<float> got(x.size()), ref(x.size());
  for (int r = 0; r < rows; r++) {
    double s = 0, q = 0;
    for (int i = 0; i < 1152; i++) s += x[(size_t)r * 1152 + i];
    const double mean = s / 1152;
    for (int i = 0; i < 1152; i++) q += (x[(size_t)r * 1152 + i] - mean) * (x[(size_t)r * 1152 + i] - mean);
    const double rstd = 1.0 / std::sqrt(q / 1152 + 1e-6);
    for (int i = 0; i < 1152; i++) {
      ref[(size_t)r * 1152 + i] = (float)((x[(size_t)r * 1152 + i] - mean) * rstd * w[i] + b[i]);
      got[(size_t)r * 1152 + i] = bf2f(g[(size_t)r * 1152 + i]);
    }
  }
  report("layernorm (bf16 out)", compare(got, ref), 2.5e-3, 5e-3);
}

static int run_unit() {
  std::mt19937 rng(42);
  test_gemm(1000, 1152, 1152, rng);
  test_gemm(333, 4352, 1152, rng);
  test_gemm(517, 1152, 4352, rng);
  test_gemm(270, 5120, 4608, rng);
  test_qkv(12, 10, rng);
  test_qkv(36, 22, rng);
  test_attention(1000, 1000, rng);
  test_attention(4100, 256, rng);
  test_attention(37, 37, rng);
  test_layernorm(1001, rng);
  CK(cudaDeviceSynchronize());
  printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------- comparison with llama.cpp
// Minimal JSON helpers: first integer value of "key" anywhere in the text.
static bool json_int(const std::string& js, const std::string& key, long long& v) {
  const std::string pat = "\"" + key + "\"";
  size_t p = js.find(pat);
  while (p != std::string::npos) {
    size_t q = p + pat.size();
    while (q < js.size() && (js[q] == ' ' || js[q] == '\t' || js[q] == '\n' || js[q] == '\r')) q++;
    if (q < js.size() && js[q] == ':') {
      q++;
      while (q < js.size() && (js[q] == ' ' || js[q] == '\t' || js[q] == '\n' || js[q] == '\r')) q++;
      char* end = nullptr;
      v = std::strtoll(js.c_str() + q, &end, 10);
      if (end != js.c_str() + q) return true;
    }
    p = js.find(pat, p + 1);
  }
  return false;
}

static bool compare_one(VisionEncoder& enc, const fs::path& dir, const std::string& name, int min_tok, int max_tok, cudaStream_t s) {
  printf("== %s\n", name.c_str());
  // the image: <name>.<ext>, else <name without its last "_suffix">.<ext> (sq448_min8 -> sq448.png)
  fs::path img;
  for (std::string base : {name, name.substr(0, name.rfind('_'))})
    for (const char* ext : {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"})
      if (img.empty() && fs::exists(dir / (base + ext))) img = dir / (base + ext);
  if (img.empty()) { printf("  no image file\n"); return false; }
  // token limits used by the reference run (json "min_tokens" / "max_tokens"; -1 = model default)
  std::string js;
  const fs::path jpath = dir / (name + ".json");
  if (fs::exists(jpath)) {
    const std::vector<uint8_t> jb = read_file(jpath.string());
    js.assign(jb.begin(), jb.end());
    long long v;
    if (json_int(js, "min_tokens", v) && v > 0) min_tok = (int)v;
    if (json_int(js, "max_tokens", v) && v > 0) max_tok = (int)v;
  }
  const std::vector<uint8_t> bytes = read_file(img.string());
  int w, h;
  const std::vector<uint8_t> rgb = VisionEncoder::decode(bytes.data(), bytes.size(), w, h);
  const VisionPlan p = VisionEncoder::plan(w, h, min_tok, max_tok);
  printf("  image %dx%d (min %d, max %d tokens) -> canvas %dx%d (content %dx%d at %d,%d), patches %dx%d, tokens %dx%d = %d\n", w, h,
         min_tok, max_tok, p.width, p.height, p.content_w, p.content_h, p.off_x, p.off_y, p.px, p.py, p.nx, p.ny, p.n_tokens);
  bool ok = true;
  if (!js.empty()) {  // grid from the json
    long long v;
    const char* keys[] = {"nx", "ny", "n_image_tokens"};
    const int mine[] = {p.nx, p.ny, p.n_tokens};
    for (int i = 0; i < 3; i++) {
      if (json_int(js, keys[i], v)) {
        const bool same = v == mine[i];
        printf("  json %-14s %lld (ours %d) %s\n", keys[i], v, mine[i], same ? "ok" : "MISMATCH");
        if (!same) ok = false;
      } else {
        printf("  json %-14s missing\n", keys[i]);
        ok = false;
      }
    }
  }
  // input image vs llama.cpp's inp_raw
  const std::vector<uint8_t> canvas = VisionEncoder::resize(rgb.data(), w, h, p);
  const fs::path ipath = dir / (name + ".img.bin");
  if (fs::exists(ipath)) {
    const std::vector<uint8_t> ib = read_file(ipath.string());
    int32_t hdr[3];
    memcpy(hdr, ib.data(), 12);
    if (hdr[0] != p.width || hdr[1] != p.height || hdr[2] != 3) {
      printf("  img.bin size %dx%dx%d differs from ours %dx%d\n", hdr[0], hdr[1], hdr[2], p.width, p.height);
      ok = false;
    } else {
      const float* f = (const float*)(ib.data() + 12);
      double maxd = 0;
      size_t ndiff = 0;
      for (int c = 0; c < 3; c++)
        for (int y = 0; y < p.height; y++)
          for (int x = 0; x < p.width; x++) {
            const float mine = ((float)canvas[((size_t)y * p.width + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
            const double d = std::fabs(mine - f[((size_t)c * p.height + y) * p.width + x]);
            maxd = std::max(maxd, d);
            if (d > 0) ndiff++;
          }
      printf("  input image vs llama.cpp: max |d| %.3g, %zu of %zu values differ %s\n", maxd, ndiff, (size_t)3 * p.width * p.height,
             maxd == 0 ? "ok" : "DIFFERENT");
      if (maxd > 0) ok = false;
    }
  }
  // embeddings
  const std::vector<uint8_t> eb = read_file((dir / (name + ".emb.bin")).string());
  int32_t eh[2];
  memcpy(eh, eb.data(), 8);
  const float* ref = (const float*)(eb.data() + 8);
  if (eh[0] != p.n_tokens || eh[1] != VisionEncoder::kEmbd) {
    printf("  reference has %d tokens x %d, ours %d x %d: MISMATCH\n", eh[0], eh[1], p.n_tokens, VisionEncoder::kEmbd);
    return false;
  }
  enc.set_profile(true);
  const auto t0 = clk::now();
  const float* dout = enc.encode(rgb.data(), w, h, p, s);
  CK(cudaStreamSynchronize(s));
  const double wall = ms_since(t0);
  std::vector<float> got((size_t)p.n_tokens * VisionEncoder::kEmbd);
  CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
  double cmin = 1e9, csum = 0, num = 0, den = 0;
  int imin = 0;
  for (int t = 0; t < p.n_tokens; t++) {
    const float* a = got.data() + (size_t)t * VisionEncoder::kEmbd;
    const float* b = ref + (size_t)t * VisionEncoder::kEmbd;
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < VisionEncoder::kEmbd; i++) {
      ab += (double)a[i] * b[i];
      aa += (double)a[i] * a[i];
      bb += (double)b[i] * b[i];
      num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    }
    den += bb;
    const double c = ab / std::sqrt(std::max(aa * bb, 1e-300));
    csum += c;
    if (c < cmin) { cmin = c; imin = t; }
  }
  const double cmean = csum / p.n_tokens, rrms = std::sqrt(num / den);
  const bool pass = cmean >= 0.999 && cmin >= 0.99;
  printf("  cosine mean %.6f min %.6f (token %d)  rel rms %.4e  -> %s\n", cmean, cmin, imin, rrms, pass ? "PASS" : "FAIL");
  printf("  encode %.1f ms wall (resize %.1f ms, gpu %.1f ms)\n", wall, enc.times().resize_cpu, enc.times().gpu_total);
  return ok && pass;
}

static int run_compare(VisionEncoder& enc, const std::string& dir, std::vector<std::string> names, int min_tok, int max_tok) {
  if (names.empty()) {
    for (const auto& e : fs::directory_iterator(dir)) {
      const std::string f = e.path().filename().string();
      const std::string suf = ".emb.bin";
      if (f.size() > suf.size() && f.compare(f.size() - suf.size(), suf.size(), suf) == 0) names.push_back(f.substr(0, f.size() - suf.size()));
    }
    std::sort(names.begin(), names.end());
  }
  if (names.empty()) { printf("no <name>.emb.bin in %s\n", dir.c_str()); return 1; }
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  int fails = 0;
  for (const auto& n : names) {
    try {
      if (!compare_one(enc, dir, n, min_tok, max_tok, s)) fails++;
    } catch (const std::exception& e) {
      printf("  ERROR %s\n", e.what());
      fails++;
    }
  }
  CK(cudaStreamDestroy(s));
  printf("%s (%d of %zu failed)\n", fails ? "FAIL" : "PASS", fails, names.size());
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------- intermediate tensors (llama_vision --dump)
// <dir>/<name>.inp_pos_emb.bin and <name>.layer_out-<i>.bin: [int32 4][int32 ne0..ne3][f32], ne0 = 1152, ne1 = patches
static int run_layers(VisionEncoder& enc, const fs::path& dir, const std::string& name, int min_tok, int max_tok) {
  fs::path img;
  std::string base = name;
  for (const char* suffix : {"_min8"})  // the image of "sq448_min8" is sq448.png
    if (base.size() > strlen(suffix) && base.compare(base.size() - strlen(suffix), strlen(suffix), suffix) == 0)
      base = base.substr(0, base.size() - strlen(suffix));
  img = dir / (base + ".png");
  const std::vector<uint8_t> bytes = read_file(img.string());
  int w, h;
  const std::vector<uint8_t> rgb = VisionEncoder::decode(bytes.data(), bytes.size(), w, h);
  const VisionPlan p = VisionEncoder::plan(w, h, min_tok, max_tok);
  printf("%s: %d patches\n", img.string().c_str(), p.n_patches);
  int fails = 0;
  for (int L = 0; L <= 27; L++) {
    const fs::path f = dir / (name + (L == 0 ? std::string(".inp_pos_emb.bin") : ".layer_out-" + std::to_string(L - 1) + ".bin"));
    if (!fs::exists(f)) continue;
    const std::vector<uint8_t> b = read_file(f.string());
    int32_t hd[5];
    memcpy(hd, b.data(), 20);
    if (hd[1] != 1152 || hd[2] != p.n_patches) { printf("  %s: shape %d x %d, expected 1152 x %d\n", f.filename().string().c_str(), hd[1], hd[2], p.n_patches); fails++; continue; }
    const float* ref = (const float*)(b.data() + 20);
    enc.set_debug_stop(L);
    enc.encode(rgb.data(), w, h, p, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)p.n_patches * 1152);
    CK(cudaMemcpy(got.data(), enc.debug_residual(), got.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, cmin = 1, csum = 0;
    for (int r = 0; r < p.n_patches; r++) {
      double ab = 0, aa = 0, bb = 0;
      for (int i = 0; i < 1152; i++) {
        const double x = got[(size_t)r * 1152 + i], y = ref[(size_t)r * 1152 + i];
        ab += x * y; aa += x * x; bb += y * y; num += (x - y) * (x - y);
      }
      den += bb;
      const double c = ab / std::sqrt(std::max(aa * bb, 1e-300));
      cmin = std::min(cmin, c);
      csum += c;
    }
    printf("  %-22s rel rms %.3e  cosine mean %.7f min %.7f\n", f.filename().string().c_str() + name.size() + 1, std::sqrt(num / den),
           csum / p.n_patches, cmin);
  }
  enc.set_debug_stop(-1);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------- bench
static void print_times(const VisionTimes& t, double wall) {
  printf("  cpu resize %7.1f ms\n", t.resize_cpu);
  printf("  upload     %7.1f ms\n  patch emb  %7.1f ms\n  layernorm  %7.1f ms\n  qkv+rope   %7.1f ms\n  attention  %7.1f ms\n",
         t.upload, t.patch_embed, t.layernorm, t.qkv, t.attention);
  printf("  out proj   %7.1f ms\n  ffn up     %7.1f ms\n  ffn down   %7.1f ms\n  merger     %7.1f ms\n", t.out_proj, t.ffn_up,
         t.ffn_down, t.merger);
  printf("  gpu total  %7.1f ms   wall (resize + gpu) %.1f ms\n", t.gpu_total, wall);
}

static size_t free_mem() {
  size_t f, t;
  CK(cudaMemGetInfo(&f, &t));
  return f;
}

// image: a file to decode (timed), or empty for a synthetic W x H image
static int run_bench(const std::string& mmproj, int dev, int W, int H, const std::string& image, int runs) {
  CK(cudaSetDevice(dev));
  CK(cudaFree(0));
  const size_t f0 = free_mem();
  {
    size_t f, t;
    CK(cudaMemGetInfo(&f, &t));
    printf("card: %.0f MiB total, %.0f MiB used after CUDA init (context and other processes)\n", t / 1048576.0, (t - f) / 1048576.0);
  }
  auto tl = clk::now();
  VisionEncoder enc(mmproj, dev);
  printf("load %.0f ms, weights %.1f MiB\n", ms_since(tl), enc.weight_bytes() / 1048576.0);
  const size_t f1 = free_mem();
  std::vector<uint8_t> rgb;
  if (!image.empty()) {
    const std::vector<uint8_t> bytes = read_file(image);
    const auto td = clk::now();
    rgb = VisionEncoder::decode(bytes.data(), bytes.size(), W, H);
    printf("decode %s (%zu bytes): %.1f ms\n", image.c_str(), bytes.size(), ms_since(td));
  } else {  // synthetic image: gradients, stripes and noise
    rgb.resize((size_t)W * H * 3);
    std::mt19937 rng(7);
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        uint8_t* p = &rgb[((size_t)y * W + x) * 3];
        const int n = (int)(rng() % 17) - 8;
        p[0] = (uint8_t)std::clamp(255 * x / W + n, 0, 255);
        p[1] = (uint8_t)std::clamp(255 * y / H + n, 0, 255);
        p[2] = (uint8_t)std::clamp(((x / 37 + y / 23) % 2) * 200 + 20 + n, 0, 255);
      }
  }
  const VisionPlan p = VisionEncoder::plan(W, H);
  printf("image %dx%d -> canvas %dx%d (content %dx%d), %d patches, tokens %dx%d = %d\n", W, H, p.width, p.height, p.content_w,
         p.content_h, p.n_patches, p.nx, p.ny, p.n_tokens);
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  enc.set_profile(false);
  enc.encode(rgb.data(), W, H, p, s);  // warmup (allocations)
  CK(cudaStreamSynchronize(s));
  const size_t f2 = free_mem();
  for (int r = 0; r < runs; r++) {
    enc.set_profile(r == runs - 1);
    const auto t0 = clk::now();
    enc.encode(rgb.data(), W, H, p, s);
    CK(cudaStreamSynchronize(s));
    const double wall = ms_since(t0);
    if (r < runs - 1) printf("run %d: %.1f ms wall (no profiling)\n", r, wall);
    else { printf("run %d (profiled):\n", r); print_times(enc.times(), wall); }
  }
  // attention TFLOPS (27 layers, 2 GEMMs of n x n x 72 per head)
  const double n = p.n_patches;
  const double att = 27.0 * 16 * 2 * 2 * n * n * 72;
  printf("attention %.1f TFLOP -> %.1f TFLOPS\n", att / 1e12, att / (enc.times().attention * 1e-3) / 1e12);
  const double lin = 27.0 * 2 * n * 1152 * (3456 + 1152 + 2 * 4352) + 2.0 * (n / 4) * 4608 * (4608 + 5120);
  const double lt = enc.times().qkv + enc.times().out_proj + enc.times().ffn_up + enc.times().ffn_down + enc.times().merger;
  printf("linear %.1f TFLOP -> %.1f TFLOPS (incl. post-LN)\n", lin / 1e12, lin / (lt * 1e-3) / 1e12);
  printf("device memory: weights %.1f MiB, activations+tables %.1f MiB; cudaMemGetInfo drop: load %.1f MiB, after encode %.1f MiB\n",
         enc.weight_bytes() / 1048576.0, enc.scratch_bytes() / 1048576.0, (f0 - f1) / 1048576.0, (f0 - f2) / 1048576.0);
  CK(cudaStreamDestroy(s));
  return 0;
}

static int run_dump(VisionEncoder& enc, const std::string& image, const std::string& out) {
  const std::vector<uint8_t> bytes = read_file(image);
  int w, h;
  const std::vector<uint8_t> rgb = VisionEncoder::decode(bytes.data(), bytes.size(), w, h);
  const VisionPlan p = VisionEncoder::plan(w, h);
  const float* d = enc.encode(rgb.data(), w, h, p, 0);
  CK(cudaDeviceSynchronize());
  std::vector<float> e((size_t)p.n_tokens * VisionEncoder::kEmbd);
  CK(cudaMemcpy(e.data(), d, e.size() * 4, cudaMemcpyDeviceToHost));
  FILE* f = fopen(out.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + out);
  const int32_t hdr[2] = {p.n_tokens, VisionEncoder::kEmbd};
  fwrite(hdr, sizeof(hdr), 1, f);
  fwrite(e.data(), 4, e.size(), f);
  fclose(f);
  printf("wrote %s: %d tokens (%dx%d)\n", out.c_str(), p.n_tokens, p.nx, p.ny);
  return 0;
}

int main(int argc, char** argv) try {
#ifdef _WIN32
  _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
#endif
  if (argc < 3) {
    fprintf(stderr, "usage: test_vision <mmproj.gguf> unit|compare|bench|dump ... [--dev N]\n");
    return 1;
  }
  int dev = 1, min_tok = 1024, max_tok = 4096, runs = 3;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--dev" && i + 1 < argc) dev = atoi(argv[++i]);
    else if (a == "--min-tokens" && i + 1 < argc) min_tok = atoi(argv[++i]);
    else if (a == "--max-tokens" && i + 1 < argc) max_tok = atoi(argv[++i]);
    else if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]);
    else pos.push_back(a);
  }
  const std::string mmproj = pos[0], mode = pos[1];
  if (dev == 0) fprintf(stderr, "warning: running on card 0\n");
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device %d: %s, bus %02x\n", dev, prop.name, prop.pciBusID);
  if (mode == "unit") return run_unit();
  if (mode == "bench") {
    if (pos.size() == 3) return run_bench(mmproj, dev, 0, 0, pos[2], runs);  // bench <image file>
    const int W = pos.size() > 3 ? atoi(pos[2].c_str()) : 3840, H = pos.size() > 3 ? atoi(pos[3].c_str()) : 2160;
    return run_bench(mmproj, dev, W, H, "", runs);
  }
  VisionEncoder enc(mmproj, dev);
  if (mode == "compare") {
    if (pos.size() < 3) throw std::runtime_error("compare needs a directory");
    return run_compare(enc, pos[2], std::vector<std::string>(pos.begin() + 3, pos.end()), min_tok, max_tok);
  }
  if (mode == "layers") {
    if (pos.size() < 4) throw std::runtime_error("layers needs <dir> <name>");
    return run_layers(enc, pos[2], pos[3], min_tok, max_tok);
  }
  if (mode == "dump") {
    if (pos.size() < 4) throw std::runtime_error("dump needs <image> <out.bin>");
    return run_dump(enc, pos[2], pos[3]);
  }
  throw std::runtime_error("unknown mode " + mode);
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
