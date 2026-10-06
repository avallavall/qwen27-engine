# Gap analysis of one device in the last decode window of an nsys SQLite export.
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); dev = int(sys.argv[2]); steps = int(sys.argv[3])
S = {i: v for i, v in c.execute("select id, value from StringIds")}
K = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,)).fetchall()
# last segment (gap > 300 ms splits requests)
cut = 0
for i in range(1, len(K)):
    if K[i][0] - K[i-1][1] > 300e6: cut = i
K = K[cut:]
gaps = []
ce = K[0][1]
for i in range(1, len(K)):
    g = K[i][0] - ce
    if g > 0: gaps.append((g, i))
    ce = max(ce, K[i][1])
bins = [(0, 2e3), (2e3, 5e3), (5e3, 10e3), (10e3, 20e3), (20e3, 50e3), (50e3, 100e3), (100e3, 300e3), (300e3, 1e6), (1e6, 1e12)]
print(f"device {dev}: {len(K)} kernels, window {(K[-1][1]-K[0][0])/1e6:.1f} ms, steps {steps}")
for lo, hi in bins:
    sel = [g for g, _ in gaps if lo <= g < hi]
    print(f"  gaps {lo/1e3:7.0f}-{hi/1e3:7.0f} us: count/step {len(sel)/steps:7.1f}  ms/step {sum(sel)/1e6/steps:7.3f}")
# the kernel names right after the big gaps (what waits)
after = collections.Counter(); before = collections.Counter()
for g, i in gaps:
    if g >= 20e3:
        after[S[K[i][2]]] += 1; before[S[K[i-1][2]]] += 1
print("  kernels right AFTER gaps >= 20 us:", after.most_common(6))
print("  kernels right BEFORE gaps >= 20 us:", before.most_common(6))
# sequence of big gaps (>= 100 us) in the first 3 steps, with kernel counts between them
print("  big-gap sequence (>=100us): gap_us@kernel_index")
big = [(g, i) for g, i in gaps if g >= 100e3]
print("  ", " ".join(f"{g/1e3:.0f}@{i}" for g, i in big[:40]))
