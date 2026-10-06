// Vision oracle: llama.cpp's own image path (production mtmd.dll + llama.dll in qwen38_27\bin-parches, used read
// only) run on one image, on the CPU only. The engine's own encoder is checked against what this writes.
//
// Usage: llama_vision <image file> <out prefix> [--min-tokens N] [--max-tokens N] [--threads N]
//                     [--model text.gguf] [--mmproj mmproj.gguf] [--dump NAME]... [--list-nodes]
// Build: bench\build-llama-vision.bat. Run with bin-parches on PATH and CUDA_VISIBLE_DEVICES=-1 (the tool also
// sets it before any ggml call and stops if a GPU device is still visible). The text model is loaded with
// vocab_only = true: mtmd only needs its tokenizer and rope type (it sizes the output from the mmproj).
//
// Settings copied from production (arranca.ps1 + llama-server): --image-min-tokens 1024 (default here), max
// tokens left at the model default (4096), -fa auto, add_special = parse_special = true in mtmd_tokenize, batch
// encode API. Differences: CPU backend instead of CUDA1, warmup off (the first encode reserves the graph instead;
// the flash-attention AUTO probe then runs on the real image), default media marker.
//
// Outputs:
//   <prefix>.json     sizes, token grid, text chunks + token ids, M-RoPE positions of the image tokens, n_pos, times
//   <prefix>.emb.bin  [int32 n_tokens][int32 n_embd][f32 n_tokens*n_embd] (same layout as MTMD_DEBUG_EMBEDDINGS=path)
//   <prefix>.img.bin  [int32 W][int32 H][int32 C=3][f32 C*H*W, planar: c, then y, then x] = the "inp_raw" tensor that
//                     goes into the encoder (resized, padded, normalized). Captured with the public cb_eval hook.
//   <prefix>.<name>.bin  (--dump NAME) any contiguous f32 graph tensor: [int32 4][int32 ne0..ne3][f32 data]
//   <prefix>.nodes.txt   (--list-nodes) name, op, type and shape of every graph node, to pick --dump names
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#pragma warning(disable : 4996)  // mtmd_image_tokens_get_nx/ny are marked deprecated; they are still the grid

namespace {

const char* kModel = getenv("Q27_MODEL") ? getenv("Q27_MODEL") : "..\\qwen38_27\\models\\Qwen3.8-27B\\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf";
const char* kMmproj = getenv("Q27_MMPROJ") ? getenv("Q27_MMPROJ") : "..\\qwen38_27\\models\\Qwen3.8-27B\\mmproj-Qwen3.8-27B-BF16.gguf";

struct Capture {
  std::string prefix;
  std::set<std::string> want;     // --dump names
  std::set<std::string> dumped;
  FILE* nodes = nullptr;          // --list-nodes
  bool got_img = false;
  int64_t img_ne[4] = {0, 0, 0, 0};
  std::vector<float> img;
  int64_t last_ne[2] = {0, 0};    // shape of the last graph node = the output embeddings [n_embd, n_tokens]
};

const ggml_tensor* inp_raw_src(const ggml_tensor* t) {
  for (int i = 0; i < GGML_MAX_SRC; i++)
    if (t->src[i] && strcmp(t->src[i]->name, "inp_raw") == 0) return t->src[i];
  return nullptr;
}

void dump_tensor(Capture& c, const ggml_tensor* t) {
  const std::string name = t->name;
  if (c.dumped.count(name)) return;
  c.dumped.insert(name);
  if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
    fprintf(stderr, "--dump %s: skipped (type %s, contiguous %d; only contiguous f32 is written)\n", name.c_str(),
            ggml_type_name(t->type), (int)ggml_is_contiguous(t));
    return;
  }
  std::vector<float> buf((size_t)ggml_nelements(t));
  ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
  std::string fn = name;
  for (char& ch : fn)
    if (!(isalnum((unsigned char)ch) || ch == '-' || ch == '_')) ch = '_';
  fn = c.prefix + "." + fn + ".bin";
  FILE* f = fopen(fn.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", fn.c_str()); return; }
  const int32_t hdr[5] = {4, (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3]};
  fwrite(hdr, sizeof(hdr), 1, f);
  fwrite(buf.data(), sizeof(float), buf.size(), f);
  fclose(f);
  fprintf(stderr, "dumped %s [%lld %lld %lld %lld] -> %s\n", name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
          (long long)t->ne[2], (long long)t->ne[3], fn.c_str());
}

// ggml_backend_sched eval callback. ask = true: "do you want this node?"; ask = false: the node is computed.
// It only splits the graph where we observe; the kernels and the math are unchanged.
bool cb_eval(ggml_tensor* t, bool ask, void* ud) {
  Capture& c = *(Capture*)ud;
  if (ask) {
    c.last_ne[0] = t->ne[0];
    c.last_ne[1] = t->ne[1];
    if (c.nodes)
      fprintf(c.nodes, "%-28s %-14s %-5s [%lld %lld %lld %lld]\n", t->name, ggml_op_desc(t), ggml_type_name(t->type),
              (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    return (!c.got_img && inp_raw_src(t)) || c.want.count(t->name) > 0;
  }
  if (!c.got_img) {
    if (const ggml_tensor* s = inp_raw_src(t)) {
      if (s->type == GGML_TYPE_F32 && ggml_is_contiguous(s)) {
        c.img.resize((size_t)ggml_nelements(s));
        ggml_backend_tensor_get(s, c.img.data(), 0, ggml_nbytes(s));
        for (int i = 0; i < 4; i++) c.img_ne[i] = s->ne[i];
        c.got_img = true;
      }
    }
  }
  if (c.want.count(t->name)) dump_tensor(c, t);
  return true;
}

std::vector<std::string> g_log_keep;  // mtmd/clip INFO lines worth keeping in the JSON

void log_cb(ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_DEBUG) return;
  fputs(text, stderr);
  const std::string s = text;
  for (const char* key : {"flash attention is", "CLIP using", "image_min_pixels", "image_max_pixels", "projector:",
                          "n_layer", "model size"})
    if (s.find(key) != std::string::npos) {
      std::string t = s;
      while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
      g_log_keep.push_back(t);
      break;
    }
}

void quiet_log(ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += (char)ch;
    }
  }
  return o + "\"";
}

template <class T>
std::string jarr(const std::vector<T>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); i++) { if (i) o += ","; o += std::to_string(v[i]); }
  return o + "]";
}

// Same float math as mtmd-image.cpp img_tool::calc_size_preserved_ratio (no longest_edge).
void calc_target(int width, int height, int align, int min_px, int max_px, int& w_bar, int& h_bar) {
  auto rnd = [&](float x) { return (int)std::round(x / (float)align) * align; };
  auto cl = [&](float x) { return (int)std::ceil(x / (float)align) * align; };
  auto fl = [&](float x) { return (int)std::floor(x / (float)align) * align; };
  w_bar = std::max(align, rnd((float)width));
  h_bar = std::max(align, rnd((float)height));
  if (max_px > 0 && h_bar * w_bar > max_px) {
    const float beta = std::sqrt((float)height * width / max_px);
    h_bar = std::max(align, fl(height / beta));
    w_bar = std::max(align, fl(width / beta));
  } else if (min_px > 0 && h_bar * w_bar < min_px) {
    const float beta = std::sqrt((float)min_px / ((float)height * width));
    h_bar = cl(height * beta);
    w_bar = cl(width * beta);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: llama_vision <image> <out prefix> [--min-tokens N] [--max-tokens N] [--threads N]\n"
            "                    [--model text.gguf] [--mmproj mmproj.gguf] [--dump NAME]... [--list-nodes]\n");
    return 1;
  }
  // No GPU: hide the CUDA devices before ggml-cuda enumerates them (it does so on the first backend call).
  _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
  SetEnvironmentVariableA("CUDA_VISIBLE_DEVICES", "-1");

  const std::string image_path = argv[1];
  Capture cap;
  cap.prefix = argv[2];
  int min_tokens = 1024, max_tokens = -1, n_threads = 6;
  bool list_nodes = false;
  const char* model_path = kModel;
  const char* mmproj_path = kMmproj;
  for (int i = 3; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); }
      return argv[++i];
    };
    if (a == "--min-tokens") min_tokens = atoi(next());
    else if (a == "--max-tokens") max_tokens = atoi(next());
    else if (a == "--threads") n_threads = atoi(next());
    else if (a == "--model") model_path = next();
    else if (a == "--mmproj") mmproj_path = next();
    else if (a == "--dump") cap.want.insert(next());
    else if (a == "--list-nodes") list_nodes = true;
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
  }

  llama_log_set(quiet_log, nullptr);
  llama_backend_init();
  if (llama_supports_gpu_offload()) {
    fprintf(stderr, "a GPU device is visible; refusing to run (set CUDA_VISIBLE_DEVICES=-1)\n");
    return 1;
  }
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(model_path, mp);
  if (!model) { fprintf(stderr, "text model load failed: %s\n", model_path); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  // vocab_only skips the hparams, so llama_model_n_embd_inp() returns 0. Read the GGUF key instead (the metadata
  // strings are loaded before the vocab_only early return); it is checked against the graph's output shape below.
  int n_embd_text = 0;
  {
    char arch[128] = {0}, val[64] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch) > 0) {
      const std::string key = std::string(arch) + ".embedding_length";
      if (llama_model_meta_val_str(model, key.c_str(), val, sizeof val) > 0) n_embd_text = atoi(val);
    }
  }

  mtmd_helper_log_set(log_cb, nullptr);
  if (list_nodes) {
    const std::string fn = cap.prefix + ".nodes.txt";
    cap.nodes = fopen(fn.c_str(), "w");
  }
  mtmd_context_params cp = mtmd_context_params_default();
  cp.use_gpu = false;
  cp.device = nullptr;
  cp.print_timings = false;
  cp.n_threads = n_threads;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  cp.warmup = false;
  cp.image_min_tokens = min_tokens;
  cp.image_max_tokens = max_tokens;
  cp.cb_eval = cb_eval;
  cp.cb_eval_user_data = &cap;
  const auto t_load0 = std::chrono::steady_clock::now();
  mtmd_context* ctx = mtmd_init_from_file(mmproj_path, model, cp);
  if (!ctx) { fprintf(stderr, "mtmd_init_from_file failed: %s\n", mmproj_path); return 1; }
  const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image_path.c_str(), false,
                                                                     mtmd_helper_init_opt_default());
  mtmd_bitmap* bmp = bw.bitmap;
  if (!bmp || mtmd_bitmap_is_audio(bmp)) { fprintf(stderr, "cannot load image %s\n", image_path.c_str()); return 1; }
  const int in_w = (int)mtmd_bitmap_get_nx(bmp), in_h = (int)mtmd_bitmap_get_ny(bmp);

  const std::string marker = mtmd_default_marker();
  const std::string prompt = "<|im_start|>user\n" + marker + "Describe.<|im_end|>\n<|im_start|>assistant\n";
  mtmd_input_text txt = {prompt.c_str(), prompt.size(), /*add_special*/ true, /*parse_special*/ true};
  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  const mtmd_bitmap* bmps[1] = {bmp};
  const auto t_tok0 = std::chrono::steady_clock::now();
  const int32_t rt = mtmd_tokenize(ctx, chunks, &txt, bmps, 1);
  const double tok_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0).count();
  if (rt != 0) { fprintf(stderr, "mtmd_tokenize failed: %d\n", rt); return 1; }

  // walk the chunks: text tokens, image grid and positions
  std::string jchunks = "[";
  const mtmd_input_chunk* img_chunk = nullptr;
  llama_pos n_past = 0, img_pos0 = 0;
  size_t n_tok_total = 0;
  int nx = 0, ny = 0, n_img_tok = 0, img_n_pos = 0;
  std::vector<int> pt, ph, pw, pz;
  for (size_t ci = 0; ci < mtmd_input_chunks_size(chunks); ci++) {
    const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, ci);
    const auto type = mtmd_input_chunk_get_type(ch);
    const llama_pos np = mtmd_input_chunk_get_n_pos(ch);
    const size_t nt = mtmd_input_chunk_get_n_tokens(ch);
    if (ci) jchunks += ",";
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n = 0;
      const llama_token* ids = mtmd_input_chunk_get_tokens_text(ch, &n);
      std::vector<int> v(ids, ids + n);
      std::string pieces = "[";
      for (size_t k = 0; k < n; k++) {
        char buf[256];
        const int len = llama_token_to_piece(vocab, ids[k], buf, sizeof buf, 0, true);
        pieces += (k ? "," : "") + jstr(std::string(buf, len > 0 ? len : 0));
      }
      pieces += "]";
      jchunks += "\n    {\"type\": \"text\", \"pos_0\": " + std::to_string(n_past) + ", \"n_tokens\": " +
                 std::to_string(nt) + ", \"n_pos\": " + std::to_string(np) + ", \"tokens\": " + jarr(v) +
                 ", \"pieces\": " + pieces + "}";
    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      img_chunk = ch;
      const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
      nx = (int)mtmd_image_tokens_get_nx(it);
      ny = (int)mtmd_image_tokens_get_ny(it);
      n_img_tok = (int)nt;
      img_n_pos = (int)np;
      img_pos0 = n_past;
      std::vector<mtmd_decoder_pos> pos(nt);
      mtmd_helper_image_get_decoder_pos(it, img_pos0, pos.data());
      for (const auto& p : pos) { pt.push_back((int)p.t); ph.push_back((int)p.y); pw.push_back((int)p.x); pz.push_back((int)p.z); }
      const char* id = mtmd_input_chunk_get_id(ch);
      jchunks += "\n    {\"type\": \"image\", \"pos_0\": " + std::to_string(n_past) + ", \"n_tokens\": " +
                 std::to_string(nt) + ", \"n_pos\": " + std::to_string(np) + ", \"nx\": " + std::to_string(nx) +
                 ", \"ny\": " + std::to_string(ny) + ", \"id\": " + jstr(id ? id : "") + "}";
    } else {
      jchunks += "\n    {\"type\": \"other\"}";
    }
    n_past += np;
    n_tok_total += nt;
  }
  jchunks += "\n  ]";
  if (!img_chunk) { fprintf(stderr, "no image chunk\n"); return 1; }

  // encode, the way llama-server does it (batch API, one image)
  mtmd_batch* batch = mtmd_batch_init(ctx);
  if (mtmd_batch_add_chunk(batch, img_chunk) != 0) { fprintf(stderr, "mtmd_batch_add_chunk failed\n"); return 1; }
  const auto t_enc0 = std::chrono::steady_clock::now();
  if (mtmd_batch_encode(batch) != 0) { fprintf(stderr, "mtmd_batch_encode failed\n"); return 1; }
  const double enc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc0).count();
  const float* embd = mtmd_batch_get_output_embd(batch, img_chunk);
  if (!embd) { fprintf(stderr, "no output embeddings\n"); return 1; }
  if (cap.last_ne[1] != n_img_tok || (n_embd_text > 0 && cap.last_ne[0] != n_embd_text)) {
    fprintf(stderr, "output shape mismatch: graph [%lld, %lld], expected [%d, %d]\n", (long long)cap.last_ne[0],
            (long long)cap.last_ne[1], n_embd_text, n_img_tok);
    return 1;
  }
  if (n_embd_text <= 0) n_embd_text = (int)cap.last_ne[0];

  // embeddings
  const std::string emb_fn = cap.prefix + ".emb.bin";
  {
    FILE* f = fopen(emb_fn.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", emb_fn.c_str()); return 1; }
    const int32_t hdr[2] = {n_img_tok, n_embd_text};
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(embd, sizeof(float), (size_t)n_img_tok * n_embd_text, f);
    fclose(f);
  }
  double sum = 0, sum2 = 0;
  const size_t n_el = (size_t)n_img_tok * n_embd_text;
  size_t n_bad = 0;
  for (size_t i = 0; i < n_el; i++) {
    if (!std::isfinite(embd[i])) { n_bad++; continue; }
    sum += embd[i];
    sum2 += (double)embd[i] * embd[i];
  }
  const double mean = sum / n_el, stdv = std::sqrt(std::max(0.0, sum2 / n_el - mean * mean));

  // preprocessed image tensor
  const std::string img_fn = cap.prefix + ".img.bin";
  int pad_top = -1, pad_bottom = -1, pad_left = -1, pad_right = -1;
  float img_min = 0, img_max = 0;
  if (cap.got_img) {
    const int W = (int)cap.img_ne[0], H = (int)cap.img_ne[1], C = (int)cap.img_ne[2];
    FILE* f = fopen(img_fn.c_str(), "wb");
    if (f) {
      const int32_t hdr[3] = {W, H, C};
      fwrite(hdr, sizeof(hdr), 1, f);
      fwrite(cap.img.data(), sizeof(float), cap.img.size(), f);
      fclose(f);
    }
    auto px = [&](int c, int y, int x) { return cap.img[((size_t)c * H + y) * W + x]; };
    auto row_black = [&](int y) { for (int c = 0; c < C; c++) for (int x = 0; x < W; x++) if (px(c, y, x) != -1.0f) return false; return true; };
    auto col_black = [&](int x) { for (int c = 0; c < C; c++) for (int y = 0; y < H; y++) if (px(c, y, x) != -1.0f) return false; return true; };
    pad_top = 0; while (pad_top < H && row_black(pad_top)) pad_top++;
    pad_bottom = 0; while (pad_bottom < H && row_black(H - 1 - pad_bottom)) pad_bottom++;
    pad_left = 0; while (pad_left < W && col_black(pad_left)) pad_left++;
    pad_right = 0; while (pad_right < W && col_black(W - 1 - pad_right)) pad_right++;
    img_min = *std::min_element(cap.img.begin(), cap.img.end());
    img_max = *std::max_element(cap.img.begin(), cap.img.end());
  }

  // expected preprocessing (formula from mtmd-image.cpp, PAD_CEIL)
  const int align = 32, patch_area = 1024;
  const int min_px = (min_tokens > 0 ? min_tokens : 8) * patch_area;
  const int max_px = (max_tokens > 0 ? max_tokens : 4096) * patch_area;
  int tw = 0, th = 0;
  calc_target(in_w, in_h, align, min_px, max_px, tw, th);
  int cw = tw, chh = th, ox = 0, oy = 0;
  const bool copy = (tw == in_w && th == in_h);
  if (!copy) {
    const float sw = (float)tw / in_w, sh = (float)th / in_h, s = std::min(sw, sh);
    cw = std::min((int)std::ceil(in_w * s), tw);
    chh = std::min((int)std::ceil(in_h * s), th);
    ox = (tw - cw) / 2;
    oy = (th - chh) / 2;
  }

  std::string kept = "[";
  for (size_t i = 0; i < g_log_keep.size(); i++) kept += (i ? ", " : "") + jstr(g_log_keep[i]);
  kept += "]";
  std::string dumped = "[";
  {
    bool first = true;
    for (const auto& d : cap.dumped) { dumped += (first ? "" : ", ") + jstr(d); first = false; }
  }
  dumped += "]";

  const std::string json_fn = cap.prefix + ".json";
  FILE* jf = fopen(json_fn.c_str(), "w");
  if (!jf) { fprintf(stderr, "cannot write %s\n", json_fn.c_str()); return 1; }
  fprintf(jf, "{\n");
  fprintf(jf, "  \"image\": %s,\n", jstr(image_path).c_str());
  fprintf(jf, "  \"mmproj\": %s,\n  \"text_model\": %s,\n", jstr(mmproj_path).c_str(), jstr(model_path).c_str());
  fprintf(jf, "  \"backend\": \"CPU\", \"n_threads\": %d, \"min_tokens\": %d, \"max_tokens\": %d,\n", n_threads,
          min_tokens, max_tokens);
  fprintf(jf, "  \"min_pixels\": %d, \"max_pixels\": %d,\n", min_px, max_px);
  fprintf(jf, "  \"input_size\": [%d, %d],\n", in_w, in_h);
  fprintf(jf, "  \"resized_size\": [%d, %d],\n", nx * 32, ny * 32);
  fprintf(jf, "  \"expected\": {\"target_size\": [%d, %d], \"resize\": %s, \"content_size\": [%d, %d], "
              "\"content_offset\": [%d, %d]},\n",
          tw, th, copy ? "\"none (copy)\"" : "\"bicubic + PAD_CEIL\"", cw, chh, ox, oy);
  if (cap.got_img)
    fprintf(jf, "  \"inp_raw\": {\"file\": %s, \"W\": %lld, \"H\": %lld, \"C\": %lld, \"min\": %.6f, \"max\": %.6f, "
                "\"black_rows_top\": %d, \"black_rows_bottom\": %d, \"black_cols_left\": %d, \"black_cols_right\": %d},\n",
            jstr(img_fn).c_str(), (long long)cap.img_ne[0], (long long)cap.img_ne[1], (long long)cap.img_ne[2], img_min,
            img_max, pad_top, pad_bottom, pad_left, pad_right);
  else
    fprintf(jf, "  \"inp_raw\": null,\n");
  fprintf(jf, "  \"nx\": %d, \"ny\": %d, \"n_image_tokens\": %d, \"n_embd\": %d,\n", nx, ny, n_img_tok, n_embd_text);
  fprintf(jf, "  \"use_mrope\": %s, \"use_non_causal\": %s,\n", mtmd_decode_use_mrope(ctx) ? "true" : "false",
          mtmd_decode_use_non_causal(ctx, img_chunk) ? "true" : "false");
  fprintf(jf, "  \"marker\": %s,\n  \"prompt\": %s,\n", jstr(marker).c_str(), jstr(prompt).c_str());
  fprintf(jf, "  \"chunks\": %s,\n", jchunks.c_str());
  fprintf(jf, "  \"n_tokens_total\": %zu, \"n_pos_total\": %d,\n", n_tok_total, (int)n_past);
  fprintf(jf, "  \"image_n_pos\": %d, \"image_pos_0\": %d, \"pos_after_image\": %d,\n", img_n_pos, (int)img_pos0,
          (int)img_pos0 + img_n_pos);
  fprintf(jf, "  \"image_positions\": {\n    \"note\": \"absolute M-RoPE positions per image token (row-major); "
              "llama sections are [t, h, w, z]\",\n");
  fprintf(jf, "    \"t\": %s,\n    \"h\": %s,\n    \"w\": %s,\n    \"z\": %s\n  },\n", jarr(pt).c_str(),
          jarr(ph).c_str(), jarr(pw).c_str(), jarr(pz).c_str());
  fprintf(jf, "  \"emb_file\": %s, \"emb_mean\": %.8f, \"emb_std\": %.8f, \"emb_nonfinite\": %zu,\n",
          jstr(emb_fn).c_str(), mean, stdv, n_bad);
  fprintf(jf, "  \"dumped\": %s,\n", dumped.c_str());
  fprintf(jf, "  \"mmproj_load_ms\": %.1f, \"tokenize_ms\": %.1f, \"encode_ms\": %.1f,\n", load_ms, tok_ms, enc_ms);
  fprintf(jf, "  \"log\": %s\n", kept.c_str());
  fprintf(jf, "}\n");
  fclose(jf);

  printf("%s: %dx%d -> %dx%d, grid %dx%d = %d tokens, n_pos %d, emb mean %.5f std %.5f, tokenize %.0f ms, "
         "encode %.0f ms (CPU, %d threads), inp_raw %s\n",
         image_path.c_str(), in_w, in_h, nx * 32, ny * 32, nx, ny, n_img_tok, img_n_pos, mean, stdv, tok_ms, enc_ms,
         n_threads, cap.got_img ? "captured" : "NOT captured");

  if (cap.nodes) fclose(cap.nodes);
  mtmd_batch_free(batch);
  mtmd_input_chunks_free(chunks);
  mtmd_bitmap_free(bmp);
  mtmd_free(ctx);
  llama_model_free(model);
  return 0;
}
