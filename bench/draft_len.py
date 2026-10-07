# Draft-length study (H5): reads Q27_DRAFTLOG files of `q27_gen ... accept` and estimates tok/s if the verify pass
# checked only the drafts before the first one whose draft top-1 probability is below a threshold (at least one).
# The drafts are still computed; only the verify rows shrink. Step time model (ms), from the Q27_SUMPROF timers and
# bench_gemv: T4 = base, each verify row less saves ROW_MS of cross-card link time, and 2 rows (nc=2 GEMV) save
# NC2_MS more. Exact: the stop rule reads only draft values.
# Usage: .venv\Scripts\python.exe bench\draft_len.py <log> [<log> ...]   (env BASE_MS=21.5 ROW_MS=0.5 NC2_MS=0.6)
import os, sys

BASE, ROW, NC2 = (float(os.environ.get(k, d)) for k, d in (("BASE_MS", 21.5), ("ROW_MS", 0.5), ("NC2_MS", 0.6)))
def step_ms(k):  # k = drafts verified (1..3)
    return BASE - ROW * (3 - k) - (NC2 if k == 1 else 0.0)

for path in sys.argv[1:]:
    rows = [list(map(float, l.split())) for l in open(path)]
    n_steps = len(rows)
    base_tok = sum(r[7] for r in rows) / n_steps
    print(f"{os.path.basename(path)}: {n_steps} steps, {base_tok:.3f} tok/step, base {base_tok / BASE * 1000:.1f} tok/s (model)")
    hist = [sum(1 for r in rows if r[7] == n) for n in (1, 2, 3, 4)]
    print("  emitted 1/2/3/4:", " ".join(f"{h / n_steps:.3f}" for h in hist))
    for key, col in (("top1", 1), ("pdraw", 4)):
        for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            tok = ms = 0.0
            for r in rows:
                k = 3
                for j in (1, 2):  # draft 0 is always verified
                    if r[col + j] < th:
                        k = j
                        break
                acc = int(r[7]) - 1  # drafts accepted in the full chain
                tok += min(acc, k) + 1
                ms += step_ms(k)
            print(f"  stop on {key} < {th:.1f}: {tok / n_steps:.3f} tok/step, {ms / n_steps:.2f} ms/step, "
                  f"{tok / ms * 1000:.1f} tok/s ({(tok / ms) / (base_tok / BASE) * 100 - 100:+.1f}%)")
