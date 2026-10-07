#include "prof.h"
#include "common.cuh"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

namespace q27::prof {

namespace {
constexpr int kSlots = 1 << 16;
struct Dev {
  unsigned long long* buf = nullptr;
  std::vector<std::string> labels;   // graph slots [0, kSlots / 2)
  std::vector<std::string> elabels;  // eager slots [kSlots / 2, kSlots)
  bool eager = false;
  const std::string& label(int slot) const { return slot < kSlots / 2 ? labels[slot] : elabels[slot - kSlots / 2]; }
  int samples = 0;
  std::map<std::string, std::pair<double, long>> tot;  // label -> (ns, gaps)
  std::vector<std::string> order;                       // labels in first-seen order
};
Dev& dev() {
  static Dev d[16];
  int i; CK(cudaGetDevice(&i));
  return d[i];
}
__global__ void stamp_kernel(unsigned long long* buf, int slot) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  buf[slot] = t;
}
}  // namespace

bool on() {
  static const bool v = [] { const char* e = getenv("Q27_PROF"); return e && e[0] == '1'; }();
  return v;
}

int mark(cudaStream_t s, const char* label) {
  if (!on()) return -1;
  Dev& d = dev();
  if (!d.buf) { CK(cudaMalloc(&d.buf, kSlots * sizeof(unsigned long long))); CK(cudaMemset(d.buf, 0, kSlots * 8)); }
  std::vector<std::string>& L = d.eager ? d.elabels : d.labels;
  const int slot = (int)L.size() + (d.eager ? kSlots / 2 : 0);
  if ((int)L.size() >= kSlots / 2) throw std::runtime_error("prof: out of slots");
  L.push_back(label);
  stamp_kernel<<<1, 1, 0, s>>>(d.buf, slot);
  return slot;
}

void eager(bool e) { if (on()) { dev().eager = e; if (e) dev().elabels.clear(); } }

int next_slot() {
  if (!on()) return 0;
  Dev& d = dev();
  return d.eager ? (int)d.elabels.size() + kSlots / 2 : (int)d.labels.size();
}
int eager_base() { return kSlots / 2; }

void collect(int a, int b) {
  if (!on() || b - a < 2) return;
  Dev& d = dev();
  std::vector<unsigned long long> t(b - a);
  CK(cudaMemcpy(t.data(), d.buf + a, (b - a) * 8, cudaMemcpyDeviceToHost));
  for (int i = 1; i < b - a; i++) {
    const std::string& l = d.label(a + i);
    auto it = d.tot.find(l);
    if (it == d.tot.end()) { d.order.push_back(l); it = d.tot.emplace(l, std::make_pair(0.0, 0L)).first; }
    it->second.first += (double)(t[i] - t[i - 1]);
    it->second.second += 1;
  }
  d.samples++;
}

void report(const char* title) {
  if (!on()) return;
  int cur, nd; CK(cudaGetDevice(&cur)); CK(cudaGetDeviceCount(&nd));
  for (int i = 0; i < nd; i++) {
    CK(cudaSetDevice(i));
    Dev& d = dev();
    if (!d.samples) continue;
    double total = 0;
    for (auto& kv : d.tot) total += kv.second.first;
    printf("== prof %s, device %d: %d samples, %.3f ms per sample (stamped part)\n", title, i, d.samples, total / d.samples / 1e6);
    std::vector<std::pair<std::string, std::pair<double, long>>> v(d.tot.begin(), d.tot.end());
    std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
    for (auto& kv : v)
      printf("  %8.3f ms %6.1f us x %5.1f  %s\n", kv.second.first / d.samples / 1e6, kv.second.first / kv.second.second / 1e3,
             (double)kv.second.second / d.samples, kv.first.c_str());
    d.tot.clear(); d.order.clear(); d.samples = 0;
  }
  CK(cudaSetDevice(cur));
}

}  // namespace q27::prof
