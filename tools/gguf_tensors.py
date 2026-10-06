# Full tensor table + per-role byte totals for the qwen35 GGUF.
import sys, re, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
rows = []
for t in r.tensors:
    rows.append((t.name, t.tensor_type.name, list(map(int, t.shape)), int(t.n_bytes)))
with open(sys.argv[2], "w") as f:
    for n, ty, sh, b in rows: f.write(f"{n}\t{ty}\t{sh}\t{b}\n")
# layer kinds
kind = {}
for n, ty, sh, b in rows:
    m = re.match(r"blk\.(\d+)\.", n)
    if m:
        i = int(m.group(1))
        if "ssm_" in n: kind[i] = "gdn"
        elif "attn_q.weight" in n and i not in kind: kind.setdefault(i, "attn")
role = collections.Counter(); types_by_role = collections.defaultdict(collections.Counter)
for n, ty, sh, b in rows:
    m = re.match(r"blk\.(\d+)\.(.*)", n)
    if m:
        i = int(m.group(1)); k = "mtp" if i == 64 else kind.get(i, "?")
        key = f"{k}:{m.group(2)}"
    else:
        key = n
    role[key] += b; types_by_role[key][ty] += 1
print("layer kinds:", collections.Counter(v for i, v in kind.items() if i < 64))
print("attn layers:", sorted(i for i, v in kind.items() if v == "attn" and i < 64))
tot = 0
for k, v in sorted(role.items(), key=lambda x: -x[1]):
    tot += v
    print(f"{k:40s} {v/2**20:9.1f} MiB  {dict(types_by_role[k])}")
print(f"total {tot/2**20:.1f} MiB")
