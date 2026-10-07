# Check of src/requant.cpp: dequantizes the original tensor and the re-quantized one (written by test_requant) with
# gguf-py's reference dequantizers and prints the error relative to the weights' RMS.
# Usage: .venv\Scripts\python.exe tools\check_requant.py <model.gguf> <tensor name> <q4_k|iq4_xs> <requant.bin>
# A correct quantizer gives a small error (4-bit: a few percent); a layout bug gives an error near or above 1.
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

model, name, qt, path = sys.argv[1:5]
t = next(x for x in GGUFReader(model).tensors if x.name == name)
src = dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float64)
dst_type = {"q4_k": GGMLQuantizationType.Q4_K, "iq4_xs": GGMLQuantizationType.IQ4_XS}[qt.lower()]
raw = np.fromfile(path, dtype=np.uint8)
dst = dequantize(raw, dst_type).reshape(-1).astype(np.float64)
assert dst.size == src.size, (dst.size, src.size)
rms = np.sqrt(np.mean(src ** 2))
err = np.sqrt(np.mean((dst - src) ** 2))
# error of a dot product with random inputs, per row (what a GEMV sees)
K = int(t.shape[0])
rows = src.size // K
rng = np.random.default_rng(0)
x = rng.standard_normal(K)
pick = rng.choice(rows, size=min(rows, 2048), replace=False)
a = src.reshape(rows, K)[pick] @ x
b = dst.reshape(rows, K)[pick] @ x
print(f"{name} {t.tensor_type.name} -> {qt}: weight rel RMS error {err / rms:.4f}, "
      f"GEMV rel error {np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)):.4f}, max |d| {np.max(np.abs(dst - src)):.4g}")
