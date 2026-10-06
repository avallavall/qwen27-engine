# Check GEMV outputs written by bench_gemv against a numpy reference.
# Reference: y = W_deq @ (xq * xd), with W_deq from the gguf package's dequantize (float64 sums).
# Usage: python tools/check_gemv.py <model.gguf> <check files...>
import sys, struct, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

r = GGUFReader(sys.argv[1])
tensors = {t.name: t for t in r.tensors}
worst = 0.0
for fn in sys.argv[2:]:
    b = open(fn, "rb").read()
    nlen, N, K, NC = struct.unpack_from("<4i", b, 0)
    off = 16
    name = b[off:off + nlen].decode(); off += nlen
    xq = np.frombuffer(b, np.int8, NC * K, off).reshape(NC, K); off += NC * K
    xd = np.frombuffer(b, np.float32, NC * K // 32, off).reshape(NC, K // 32); off += NC * K // 32 * 4
    y = np.frombuffer(b, np.float32, NC * N, off).reshape(NC, N)
    t = tensors[name]
    W = dequantize(t.data, t.tensor_type).reshape(N, K).astype(np.float64)
    x = xq.astype(np.float64) * np.repeat(xd.astype(np.float64), 32, axis=1)
    yref = (W @ x.T).T
    err = np.abs(y - yref)
    rms = np.sqrt(np.mean(yref ** 2))
    rel = err.max() / rms
    worst = max(worst, rel)
    print(f"{fn.split('/')[-1].split(chr(92))[-1]:40s} {name:28s} max|err|/rms {rel:.2e}  mean|err|/rms {err.mean()/rms:.2e}")
print(f"worst max|err|/rms: {worst:.2e}")
sys.exit(0 if worst < 1e-3 else 1)
