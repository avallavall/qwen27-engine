# E2 check: questions on the test images (bench\out\vision, made by bench\make_vision_images.py and final_bench.py's
# dash4k.png); an answer passes when it contains every expected word (case-insensitive).
# Usage: $env:BENCH_KEY = "<key>"; .venv\Scripts\python.exe bench\vision_answers.py
import base64, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "none")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "vision")
CASES = [
    ("chart800x600.png", "What are the North and South values in Q3? Answer with the two numbers.", ["61", "39"]),
    ("chart800x600.png", "Which quarter has the highest North value?", ["Q4"]),
    ("doc1300x850.png", "What is the part number of the seal kit, and the bearing temperature after the repair?",
     ["44-1093", "48"]),
    ("tall360x640.png", "Which items on the list are already checked?", ["eggs", "coffee"]),
    ("photo640x480.png", "What color is the roof, and how many windows does the house have?", ["red", "two|2"]),
    ("dash4k.png", "In the low stock table, which item has the fewest left and how many?", ["pallet wrap", "5"]),
]
fails = 0
for name, q, want in CASES:
    b64 = base64.b64encode(open(os.path.join(D, name), "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": q}]}],
        "max_tokens": 3000}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    a = d["choices"][0]["message"]["content"].strip()
    for h in "‐‑‒–−":  # hyphen-like characters
        a = a.replace(h, "-")
    ok = all(any(alt in a.lower() for alt in w.lower().split("|")) for w in want)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {time.time() - t0:.1f} s  {a[:150]!r}", flush=True)
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
