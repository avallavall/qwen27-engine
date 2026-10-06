#include "gguf.h"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace q27 {

const char* gtype_name(GType t) {
  switch (t) {
    case GType::F32: return "F32";
    case GType::F16: return "F16";
    case GType::Q2_K: return "Q2_K";
    case GType::Q4_K: return "Q4_K";
    case GType::Q6_K: return "Q6_K";
    case GType::IQ2_XXS: return "IQ2_XXS";
    case GType::IQ2_XS: return "IQ2_XS";
    case GType::IQ3_XXS: return "IQ3_XXS";
    case GType::IQ3_S: return "IQ3_S";
    case GType::IQ2_S: return "IQ2_S";
    case GType::IQ4_XS: return "IQ4_XS";
    case GType::IQ1_M: return "IQ1_M";
    case GType::BF16: return "BF16";
  }
  return "?";
}

int gtype_block_bytes(GType t) {
  switch (t) {
    case GType::F32: return 4;
    case GType::F16: return 2;
    case GType::BF16: return 2;
    case GType::Q2_K: return 84;
    case GType::Q4_K: return 144;
    case GType::Q6_K: return 210;
    case GType::IQ2_XXS: return 66;
    case GType::IQ2_XS: return 74;
    case GType::IQ3_XXS: return 98;
    case GType::IQ3_S: return 110;
    case GType::IQ2_S: return 82;
    case GType::IQ4_XS: return 136;
    case GType::IQ1_M: return 56;
  }
  throw std::runtime_error("unknown ggml type " + std::to_string((uint32_t)t));
}

int gtype_block_size(GType t) {
  switch (t) {
    case GType::F32: case GType::F16: case GType::BF16: return 1;
    default: gtype_block_bytes(t); return 256;
  }
}

namespace {

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  template <class T> T get() {
    if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated file");
    T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
  }
  std::string str() {
    uint64_t n = get<uint64_t>();
    if (p + n > end) throw std::runtime_error("gguf: truncated string");
    std::string s((const char*)p, n); p += n; return s;
  }
};

// gguf value types
enum { T_U8 = 0, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

bool read_scalar(Reader& r, uint32_t t, int64_t& i, double& f) {
  switch (t) {
    case T_U8: i = r.get<uint8_t>(); return true;
    case T_I8: i = r.get<int8_t>(); return true;
    case T_U16: i = r.get<uint16_t>(); return true;
    case T_I16: i = r.get<int16_t>(); return true;
    case T_U32: i = r.get<uint32_t>(); return true;
    case T_I32: i = r.get<int32_t>(); return true;
    case T_U64: i = (int64_t)r.get<uint64_t>(); return true;
    case T_I64: i = r.get<int64_t>(); return true;
    case T_BOOL: i = r.get<uint8_t>(); return true;
    case T_F32: f = r.get<float>(); return false;
    case T_F64: f = r.get<double>(); return false;
  }
  throw std::runtime_error("gguf: bad value type " + std::to_string(t));
}

}  // namespace

GGUF::GGUF(const std::string& path) : path_(path) {
#ifdef _WIN32
  HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
  LARGE_INTEGER sz; GetFileSizeEx(f, &sz); size_ = (uint64_t)sz.QuadPart;
  HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!m) { CloseHandle(f); throw std::runtime_error("cannot map " + path); }
  base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  if (!base_) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("cannot map view " + path); }
  hfile_ = f; hmap_ = m;
#else
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  struct stat st; fstat(fd_, &st); size_ = (uint64_t)st.st_size;
  void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
  base_ = (const uint8_t*)p;
#endif

  Reader r{base_, base_ + size_};
  if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("not a GGUF file: " + path);
  uint32_t version = r.get<uint32_t>();
  if (version != 3) throw std::runtime_error("GGUF version " + std::to_string(version) + " not supported");
  uint64_t n_tensors = r.get<uint64_t>();
  uint64_t n_kv = r.get<uint64_t>();

  for (uint64_t k = 0; k < n_kv; k++) {
    std::string key = r.str();
    uint32_t t = r.get<uint32_t>();
    GValue v;
    if (t == T_STR) { v.kind = GValue::STRING; v.s = r.str(); }
    else if (t == T_ARR) {
      uint32_t et = r.get<uint32_t>();
      uint64_t n = r.get<uint64_t>();
      if (et == T_STR) { v.kind = GValue::ARR_STRING; v.as.reserve(n); for (uint64_t j = 0; j < n; j++) v.as.push_back(r.str()); }
      else {
        int64_t iv; double fv; bool isint = true;
        for (uint64_t j = 0; j < n; j++) {
          isint = read_scalar(r, et, iv, fv);
          if (isint) v.ai.push_back(iv); else v.af.push_back(fv);
        }
        v.kind = (et == T_F32 || et == T_F64) ? GValue::ARR_FLOAT : GValue::ARR_INT;
      }
    } else {
      bool isint = read_scalar(r, t, v.i, v.f);
      v.kind = isint ? (t == T_BOOL ? GValue::BOOL : GValue::INT) : GValue::FLOAT;
    }
    kv_[key] = std::move(v);
  }

  tensors_.resize(n_tensors);
  for (uint64_t k = 0; k < n_tensors; k++) {
    GTensor& t = tensors_[k];
    t.name = r.str();
    t.n_dims = (int)r.get<uint32_t>();
    if (t.n_dims < 1 || t.n_dims > 4) throw std::runtime_error("bad n_dims for " + t.name);
    for (int d = 0; d < t.n_dims; d++) t.ne[d] = (int64_t)r.get<uint64_t>();
    t.type = (GType)r.get<uint32_t>();
    t.offset = r.get<uint64_t>();
    int64_t n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    t.nbytes = (uint64_t)(n / gtype_block_size(t.type)) * gtype_block_bytes(t.type);
    index_[t.name] = k;
  }

  uint64_t align = has("general.alignment") ? (uint64_t)get_int("general.alignment") : 32;
  uint64_t data_start = (uint64_t)(r.p - base_);
  data_start = (data_start + align - 1) / align * align;
  for (auto& t : tensors_) {
    if (data_start + t.offset + t.nbytes > size_) throw std::runtime_error("tensor out of file: " + t.name);
    t.data = base_ + data_start + t.offset;
  }
}

GGUF::~GGUF() {
#ifdef _WIN32
  if (base_) UnmapViewOfFile(base_);
  if (hmap_) CloseHandle((HANDLE)hmap_);
  if (hfile_) CloseHandle((HANDLE)hfile_);
#else
  if (base_) munmap((void*)base_, size_);
  if (fd_ >= 0) close(fd_);
#endif
}

const GTensor* GGUF::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const GTensor& GGUF::tensor(const std::string& name) const {
  const GTensor* t = find(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  return *t;
}

const GValue& GGUF::kv(const std::string& key) const {
  auto it = kv_.find(key);
  if (it == kv_.end()) throw std::runtime_error("missing key " + key);
  return it->second;
}

int64_t GGUF::get_int(const std::string& key) const {
  const GValue& v = kv(key);
  if (v.kind == GValue::INT || v.kind == GValue::BOOL) return v.i;
  if (v.kind == GValue::ARR_INT && v.ai.size() == 1) return v.ai[0];
  throw std::runtime_error("key is not an int: " + key);
}

double GGUF::get_float(const std::string& key) const {
  const GValue& v = kv(key);
  if (v.kind == GValue::FLOAT) return v.f;
  if (v.kind == GValue::INT) return (double)v.i;
  throw std::runtime_error("key is not a float: " + key);
}

const std::string& GGUF::get_str(const std::string& key) const {
  const GValue& v = kv(key);
  if (v.kind != GValue::STRING) throw std::runtime_error("key is not a string: " + key);
  return v.s;
}

}  // namespace q27
