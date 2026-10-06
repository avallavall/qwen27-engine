"""Read and check the files written by bench/llama_vision.cpp (vision oracle).

Run with the project venv:
  .venv\\Scripts\\python.exe bench\\vision_ref_check.py <file.emb.bin> [other.emb.bin]
  .venv\\Scripts\\python.exe bench\\vision_ref_check.py <file.img.bin>

.emb.bin: [int32 n_tokens][int32 n_embd][f32 n_tokens*n_embd] (same as llama.cpp MTMD_DEBUG_EMBEDDINGS=path).
          Prints shape, mean, std, min, max and per-token L2 norms. With a second .emb.bin it also prints the
          max abs difference and the per-token cosine similarity between the two.
.img.bin: [int32 W][int32 H][int32 C][f32 C*H*W planar]. Prints shape, range, per-channel mean and black borders.
"""
import sys

import numpy as np


def read_emb(path):
    with open(path, "rb") as f:
        n_tokens, n_embd = np.fromfile(f, dtype="<i4", count=2)
        data = np.fromfile(f, dtype="<f4")
    if data.size != int(n_tokens) * int(n_embd):
        raise SystemExit(f"{path}: header says {n_tokens}x{n_embd} = {int(n_tokens) * int(n_embd)} floats, "
                         f"file has {data.size}")
    return data.reshape(int(n_tokens), int(n_embd))


def read_img(path):
    with open(path, "rb") as f:
        w, h, c = (int(v) for v in np.fromfile(f, dtype="<i4", count=3))
        data = np.fromfile(f, dtype="<f4")
    if data.size != w * h * c:
        raise SystemExit(f"{path}: header says {c}x{h}x{w}, file has {data.size} floats")
    return data.reshape(c, h, w)


def show_emb(path, e):
    norms = np.linalg.norm(e.astype(np.float64), axis=1)
    print(f"{path}")
    print(f"  shape [n_tokens={e.shape[0]}, n_embd={e.shape[1]}], finite {np.isfinite(e).all()}")
    print(f"  mean {e.mean(dtype=np.float64):.6f}  std {e.std(dtype=np.float64):.6f}  "
          f"min {e.min():.4f}  max {e.max():.4f}")
    print(f"  per-token L2 norm: min {norms.min():.3f}  mean {norms.mean():.3f}  max {norms.max():.3f}  "
          f"(argmax token {int(norms.argmax())})")
    k = min(8, len(norms))
    print(f"  first {k} norms: " + " ".join(f"{v:.3f}" for v in norms[:k]))
    print(f"  token 0, first 8 values: " + " ".join(f"{v:.5f}" for v in e[0, :8]))


def compare(a, b):
    if a.shape != b.shape:
        print(f"shapes differ: {a.shape} vs {b.shape}")
        return
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = (a.astype(np.float64) * b).sum(1) / np.maximum(na * nb, 1e-30)
    print(f"compare: max abs diff {d.max():.6g}  mean abs diff {d.mean():.6g}  bit-identical {bool((a == b).all())}")
    print(f"  per-token cosine: min {cos.min():.6f} (token {int(cos.argmin())})  mean {cos.mean():.6f}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    p = sys.argv[1]
    if p.endswith(".img.bin"):
        x = read_img(p)
        c, h, w = x.shape
        black = (x == -1.0).all(axis=0)
        rows = black.all(axis=1)
        cols = black.all(axis=0)
        top = int(np.argmin(rows)) if not rows.all() else h
        bottom = int(np.argmin(rows[::-1])) if not rows.all() else h
        left = int(np.argmin(cols)) if not cols.all() else w
        right = int(np.argmin(cols[::-1])) if not cols.all() else w
        print(f"{p}\n  shape [C={c}, H={h}, W={w}]  min {x.min():.4f}  max {x.max():.4f}")
        print("  channel mean: " + " ".join(f"{m:.5f}" for m in x.reshape(c, -1).mean(1)))
        print(f"  black (-1.0) rows top {top} bottom {bottom}, cols left {left} right {right}")
        return
    a = read_emb(p)
    show_emb(p, a)
    if len(sys.argv) > 2:
        b = read_emb(sys.argv[2])
        show_emb(sys.argv[2], b)
        compare(a, b)


if __name__ == "__main__":
    main()
