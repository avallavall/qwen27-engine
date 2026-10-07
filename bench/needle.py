# Long-context retrieval check ("needle in a haystack") against a running server (q27 or llama.cpp).
# Builds haystacks of code from bench\out\tok-corpus.txt at the given sizes, hides 8 facts (a 6-digit code per
# warehouse name) at depths 5% .. 95%, then asks for each fact in its own request. The haystack is the same prefix in
# every request, so the server's prompt cache reads it once. Greedy decoding, thinking off, max 16 tokens.
# --hard: 32 facts and 32 look-alike distractors (the key with two digits swapped, another code) spread over the
# haystack; the 32 real keys are asked. Tests precise recall when similar keys compete.
# Usage: .venv\Scripts\python.exe bench\needle.py <label> [sizes=32000,100000,170000] [--hard] [--url ...] [--key ...]
# Output: bench\out\needle_<label>.json and a table on stdout.
import argparse, json, os, random, re, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("sizes", nargs="?", default="32000,100000,170000")
ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://127.0.0.1:8081"))
ap.add_argument("--key", default=os.environ.get("BENCH_KEY", "none"))
ap.add_argument("--hard", action="store_true")
a = ap.parse_args()

NAMES = ["Albacete", "Bergen", "Cordoba", "Dresden", "Esbjerg", "Faro", "Gdansk", "Hamar"]
DEPTHS = [0.05, 0.18, 0.31, 0.44, 0.57, 0.70, 0.83, 0.95]


def post(path, body, timeout=3600):
    r = urllib.request.Request(a.url + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def count(text):
    return len(post("/tokenize", {"content": text})["tokens"])


def cut(text, target):
    lo, hi = 0, len(text)
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        if count(text[:mid]) < target: lo = mid
        else: hi = mid
    return text[:lo]


corpus = open(os.path.join(HERE, "out", "tok-corpus.txt"), encoding="utf-8", errors="ignore").read()
rng = random.Random(1234)
results = []
def unit_names():
    # 32 keys like "unit 4817" and a distractor for each with two adjacent digits swapped
    keys, dist = [], []
    while len(keys) < 32:
        k = f"{rng.randint(1000, 9999)}"
        i = rng.randint(0, 2)
        d = k[:i] + k[i + 1] + k[i] + k[i + 2:]
        if d == k or k in keys or k in dist or d in keys or d in dist: continue
        keys.append(k); dist.append(d)
    return [f"unit {k}" for k in keys], [f"unit {d}" for d in dist]


for size in [int(x) for x in a.sizes.split(",")]:
    if a.hard:
        names, distractors = unit_names()
        n_ins = 64
    else:
        names, distractors = NAMES, []
        n_ins = 8
    hay = cut(corpus, size - n_ins * 30 - 100)
    codes = {n: str(rng.randint(100000, 999999)) for n in names + distractors}
    if a.hard:
        lines = names + distractors
        rng.shuffle(lines)
        depths = {n: 0.02 + 0.96 * i / (len(lines) - 1) for i, n in enumerate(lines)}
    else:
        depths = dict(zip(NAMES, DEPTHS))
    noun = "storage" if a.hard else "warehouse"
    # insert at line starts, from the deepest position backwards so earlier offsets stay valid
    for name, depth in sorted(depths.items(), key=lambda x: -x[1]):
        pos = hay.rfind("\n", 0, int(len(hay) * depth)) + 1
        hay = hay[:pos] + f"NOTE: the access code of the {name} {noun} is {codes[name]}.\n" + hay[pos:]
    for name in names:
        depth = depths[name]
        q = (f"\n\n===== QUESTION =====\nThe text above contains a note with the access code of the {name} {noun}. "
             f"What is that code? Answer with the 6-digit number only.")
        body = {"messages": [{"role": "user", "content": hay + q}], "max_tokens": 16, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
        t0 = time.time()
        r = post("/v1/chat/completions", body)
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        reply = (msg.get("content") or "").strip()
        ok = codes[name] in reply
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        results.append({"size": size, "prompt_tokens": pt, "name": name, "depth": depth, "code": codes[name], "reply": reply,
                        "ok": ok, "s": round(dt, 2)})
        print(f"{a.label} {pt:7d} tokens  depth {depth:4.2f}  {name:9s} {'OK  ' if ok else 'MISS'} {reply[:40]!r}  {dt:6.1f} s",
              flush=True)
out = os.path.join(HERE, "out", f"needle_{a.label}.json")
json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
print(f"wrote {out}")
for size in sorted({r["size"] for r in results}):
    rs = [r for r in results if r["size"] == size]
    print(f"{a.label}: ~{rs[0]['prompt_tokens']} tokens: {sum(r['ok'] for r in rs)} of {len(rs)} found")
