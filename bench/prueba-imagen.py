# Same request as qwen38_27\prueba-imagen.py (written again here; port 8081, key from BENCH_KEY).
# Sends one image and measures the total time. Usage: python bench\prueba-imagen.py <image file> [question]
import base64, json, mimetypes, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "none")
path = sys.argv[1]
question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in one sentence."
mime = mimetypes.guess_type(path)[0] or "image/png"
b64 = base64.b64encode(open(path, "rb").read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + b64}},
    {"type": "text", "text": question}]}],
    "max_tokens": 120, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
t0 = time.time()
try:
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    t = d["timings"]
    print(f"OK  {time.time() - t0:.1f} s  prompt_tok={t['prompt_n']}  prompt_ms={t['prompt_ms']:.0f}  gen={t['predicted_n']}"
          f"  gen_ts={t['predicted_per_second']:.1f}")
    print("answer:", d["choices"][0]["message"]["content"].strip()[:300])
except urllib.error.HTTPError as e:
    print("FAIL:", e.code, e.read().decode()[:300])
    sys.exit(1)
except Exception as e:
    print("FAIL:", e)
    sys.exit(1)
