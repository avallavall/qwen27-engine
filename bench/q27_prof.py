# Per-step profile of the engine's speculative step graph from an nsys SQLite export.
# Usage: .venv\Scripts\python.exe bench\q27_prof.py <report.sqlite> [device=0]
# Finds the graph launched most often on the device (the step graph), splits its kernels into launches,
# and prints: wall time per step, busy time, gaps, kernel count, and time per kernel name per step.
import collections
import sqlite3
import sys

db = sys.argv[1]
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
S = {i: v for i, v in c.execute("select id, value from StringIds")}
cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
gid = "graphId" if "graphId" in cols else None
rows = c.execute(f"select start, end, shortName, {gid or '0'}, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
                 f"where deviceId = ? order by start", (dev,)).fetchall()
by_graph = collections.Counter(r[3] for r in rows)
# step graph = the graph with the most distinct kernel nodes (verify + catch-up + 3 drafts)
nodes = collections.defaultdict(set)
for r in rows: nodes[r[3]].add(r[4])
cand = [k for k in by_graph if by_graph[k] >= 20 * len(nodes[k]) // 1 or True]
g = max(cand, key=lambda k: (len(nodes[k]) if by_graph[k] / max(1, len(nodes[k])) >= 20 else 0))
ks = [r for r in rows if r[3] == g]
# split into launches: a new launch starts when a node id repeats
launches, cur, seen = [], [], set()
for r in ks:
    if r[4] in seen:
        launches.append(cur); cur, seen = [], set()
    cur.append(r); seen.add(r[4])
if cur: launches.append(cur)
launches = launches[2:-1]  # drop the first (warm-up) and the last
n = len(launches)
wall = sum(l[-1][1] - l[0][0] for l in launches) / n
busy = 0.0
for l in launches:
    t_end = l[0][0]
    for s, e, *_ in l:
        busy += max(0, e - max(s, t_end)); t_end = max(t_end, e)
busy /= n
print(f"device {dev}: graph {g}, {n} step launches, {len(launches[0])} kernels per step")
print(f"wall {wall/1e6:.2f} ms per step, busy {busy/1e6:.2f} ms, idle {(wall-busy)/1e6:.2f} ms")
tot = collections.Counter(); cnt = collections.Counter()
for l in launches:
    for s, e, sn, *_ in l:
        tot[S.get(sn, str(sn))] += e - s; cnt[S.get(sn, str(sn))] += 1
print(f"{'ms/step':>8} {'count':>6} {'us avg':>8}  kernel")
for k, v in tot.most_common(40):
    print(f"{v/n/1e6:8.3f} {cnt[k]/n:6.0f} {v/cnt[k]/1e3:8.1f}  {k[:110]}")
# gaps: idle time between consecutive kernels, by the kernel that follows
gap = collections.Counter(); gcnt = collections.Counter()
for l in launches:
    for a, b in zip(l, l[1:]):
        d = b[0] - a[1]
        if d > 0:
            gap[S.get(b[2], str(b[2]))] += d; gcnt[S.get(b[2], str(b[2]))] += 1
print("gaps before kernel (ms per step):")
for k, v in gap.most_common(15):
    print(f"{v/n/1e6:8.3f} {gcnt[k]/n:6.0f}  {k[:110]}")
