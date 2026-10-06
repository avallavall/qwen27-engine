# Table from two bench\compare.py results. Usage: .venv\Scripts\python.exe bench\compare_report.py [llama] [q27]
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
la, lb = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("llama", "q27")
A = json.load(open(os.path.join(HERE, "out", f"cmp_{la}.json"), encoding="utf-8"))
B = json.load(open(os.path.join(HERE, "out", f"cmp_{lb}.json"), encoding="utf-8"))


def mean(v):
    return sum(v) / len(v) if v else 0.0


def ratio(x, y, higher_better=True):
    if not x or not y:
        return ""
    r = y / x if higher_better else x / y
    return f"{r:.2f}x"


rows = []
def add(name, x, y, fmt="{:.1f}", hb=True):
    rows.append((name, fmt.format(x), fmt.format(y), ratio(x, y, hb)))

for d in (1000, 30000, 100000, 150000):
    ta = [r for r in A["text"] if r["depth"] == d]
    tb = [r for r in B["text"] if r["depth"] == d]
    k = f"{d // 1000}k"
    add(f"text {k}: generation tok/s (mean of 4)", mean([r["gen_tps"] for r in ta]), mean([r["gen_tps"] for r in tb]))
    add(f"text {k}: prompt t/s (first run, {ta[0]['prompt_n']} / {tb[0]['prompt_n']} tokens read)", ta[0]["prompt_tps"],
        tb[0]["prompt_tps"], "{:.0f}")
    add(f"text {k}: time to first token, resent prompt (ms)", mean([r["prompt_ms"] for r in ta[1:]]),
        mean([r["prompt_ms"] for r in tb[1:]]), "{:.0f}", False)
    acc = lambda t: sum(r["draft_acc"] for r in t) / max(1, sum(r["draft_n"] for r in t))
    rows.append((f"text {k}: MTP draft acceptance", f"{acc(ta):.3f}", f"{acc(tb):.3f}", ""))
for name, label in (("chart800x600.png", "image 800x600 (1036 tokens)"), ("dash4k.png", "image 4K (4099 tokens)")):
    ia = [r for r in A["images"] if r["image"] == name]
    ib = [r for r in B["images"] if r["image"] == name]
    add(f"{label}: total s (mean of 3)", mean([r["total_s"] for r in ia]), mean([r["total_s"] for r in ib]), "{:.2f}", False)
    add(f"{label}: prompt ms (encoder + reading)", mean([r["prompt_ms"] for r in ia]), mean([r["prompt_ms"] for r in ib]), "{:.0f}", False)
ga, gb = A["agent"], B["agent"]
add(f"agent loop ({len(ga)} Qwen Code requests): total s", sum(r["total_s"] for r in ga), sum(r["total_s"] for r in gb), "{:.1f}", False)
add("agent loop: prompt ms, sum", sum(r["prompt_ms"] for r in ga), sum(r["prompt_ms"] for r in gb), "{:.0f}", False)
rows.append(("agent loop: tokens read / prompt tokens", f"{sum(r['prompt_n'] for r in ga)} / {sum(r['prompt_tokens'] for r in ga)}",
             f"{sum(r['prompt_n'] for r in gb)} / {sum(r['prompt_tokens'] for r in gb)}", ""))
add("agent loop: generation tok/s (mean)", mean([r["gen_tps"] for r in ga]), mean([r["gen_tps"] for r in gb]))
add("load time s (start to /health)", A.get("load_s", 0), B.get("load_s", 0), "{:.1f}", False)
rows.append(("VRAM MiB per card at the end", " + ".join(map(str, A["vram_mib"])), " + ".join(map(str, B["vram_mib"])), ""))

print(f"| Test | {la} | {lb} | {lb} vs {la} |")
print("|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
