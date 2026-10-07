// Writes one GGUF tensor re-quantized by src/requant.cpp, for tools/check_requant.py (which dequantizes both with
// gguf-py and compares them). Usage: test_requant <model.gguf> <tensor name> <q4_k|iq4_xs> <out.bin>
#include <chrono>
#include <cstdio>
#include <exception>

#include "gguf.h"
#include "requant.h"

using namespace q27;

int main(int argc, char** argv) try {
  if (argc < 5) {
    fprintf(stderr, "usage: test_requant <model.gguf> <tensor name> <q4_k|iq4_xs> <out.bin>\n");
    return 1;
  }
  GGUF g(argv[1]);
  const GTensor& t = g.tensor(argv[2]);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<uint8_t> out = requant(t, parse_gtype(argv[3]));
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  FILE* f = fopen(argv[4], "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  printf("%s: %s %lld x %lld -> %s, %zu bytes in %.2f s\n", t.name.c_str(), gtype_name(t.type), (long long)t.ne[0],
         (long long)t.rows(), argv[3], out.size(), s);
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
