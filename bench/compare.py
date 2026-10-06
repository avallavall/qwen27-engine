# Head-to-head benchmark: the same requests against one server (llama.cpp or q27), results to bench\out\cmp_<label>.json.
# Run it once per system, both started with the same context (180224, production), then bench\compare_report.py.
#   1. text: 4 depths (1k, 30k, 100k, 150k) x 4 runs x 400 tokens, mide-tps.py prompts (first run reads the prompt,
#      runs 2-4 resend it: prompt cache); prompt t/s, generation tok/s, MTP acceptance, time to first token
#   2. images: chart 800x600 (1036 tokens) and 4K 3840x2160 (4099 tokens), 3 runs each, one pixel changed per run
#      so neither system can reuse a cached image; total time, prompt ms
#   3. agent loop: the 11 requests of a real Qwen Code session (bench\out\qwen\run2\logs), in order,
#      max_tokens 400, non-streaming; prompt tokens, cached tokens, prompt ms, total time
#   4. VRAM per card at the end (nvidia-smi)
# Usage: .venv\Scripts\python.exe bench\compare.py <label> [--url http://127.0.0.1:8081] [--key KEY]
import argparse, base64, glob, io, json, os, subprocess, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default="http://127.0.0.1:8081")
ap.add_argument("--key", default=os.environ.get("BENCH_KEY", "none"))
ap.add_argument("--load-s", type=float, default=0.0, help="load time measured by the caller, stored in the result")
ap.add_argument("--only", default="", help="text|images|agent: run one part and replace it in an existing result")
a = ap.parse_args()
URL, KEY = a.url, a.key


def post(path, body, timeout=3600):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def chat(body):
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    return r, time.time() - t0


res = {"label": a.label, "load_s": a.load_s, "text": [], "images": [], "agent": []}

# ---- 1. text (prompts as bench\mide-tps.py)
if a.only in ("", "text"):
    corpus = open(os.path.join(HERE, "out", "tok-corpus.txt"), encoding="utf-8", errors="ignore").read()


    def count(text):
        return len(post("/tokenize", {"content": text})["tokens"])


    def cut(text, target):
        lo, hi = 0, len(text)
        while hi - lo > 2000:
            mid = (lo + hi) // 2
            if count(text[:mid]) < target: lo = mid
            else: hi = mid
        return text[:lo]


    for depth in (1000, 30000, 100000, 150000):
        prompt = cut(corpus, depth - 120) + "\n\n===== TASK =====\nSummarize in one paragraph what these files do. Then list three risks."
        for run in range(4):
            r, dt = chat({"messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
                          "temperature": 1.0, "top_p": 0.95, "top_k": 20, "cache_prompt": True})
            t = r["timings"]
            row = {"depth": depth, "run": run, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
                   "prompt_tps": t.get("prompt_per_second", 0), "gen_n": t["predicted_n"], "gen_ms": t["predicted_ms"],
                   "gen_tps": t["predicted_per_second"], "draft_n": t.get("draft_n", 0),
                   "draft_acc": t.get("draft_n_accepted", 0), "total_s": dt}
            res["text"].append(row)
            print(a.label, "text", row, flush=True)

# ---- 2. images
if a.only in ("", "images"):
    for name, n in (("chart800x600.png", 3), ("dash4k.png", 3)):
        im0 = Image.open(os.path.join(HERE, "out", "vision", name)).convert("RGB")
        for i in range(n):
            im = im0.copy()
            im.putpixel((i + 1, 1), (int(time.time()) % 255, i * 50, 7))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "Describe this image in one sentence."}]}],
                "max_tokens": 120, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            r, dt = chat(body)
            t = r["timings"]
            row = {"image": name, "run": i, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "gen_n": t["predicted_n"],
                   "gen_tps": t["predicted_per_second"], "total_s": dt}
            res["images"].append(row)
            print(a.label, "image", row, flush=True)

# ---- 3. agent loop replay
if a.only in ("", "agent"):
    logs = sorted(glob.glob(os.path.join(HERE, "out", "qwen", "run2", "logs", "*.json")))
    for p in logs:
        req = json.load(open(p, encoding="utf-8"))["request"]
        req = dict(req, stream=False, max_tokens=400)
        req.pop("stream_options", None)
        r, dt = chat(req)
        t = r["timings"]
        row = {"file": os.path.basename(p), "prompt_tokens": r["usage"]["prompt_tokens"], "cached": t.get("cache_n", 0),
               "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "gen_n": t["predicted_n"], "gen_tps": t["predicted_per_second"],
               "total_s": dt}
        res["agent"].append(row)
        print(a.label, "agent", row, flush=True)

# ---- 4. VRAM
q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
res["vram_mib"] = [int(x) for x in q.stdout.split()]
out = os.path.join(HERE, "out", f"cmp_{a.label}.json")
if a.only:  # replace one part of an earlier full run
    old = json.load(open(out, encoding="utf-8"))
    old[a.only] = res[a.only]
    res = old
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
