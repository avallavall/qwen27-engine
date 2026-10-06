# Final benchmark against a running q27 server with vision (port 8081):
#   1. mide-tps.py: 4 depths x 4 runs x 400 tokens (1k / 30k / 100k / 150k), as the 2026-10-03 llama.cpp baseline
#   2. the 4K image test (bench\prueba-imagen.py style), 3 times, on a fresh image each time
#   3. VRAM per card after a 150k prompt plus a 4K image
# Usage: $env:BENCH_KEY = "<key>"; .venv\Scripts\python.exe bench\final_bench.py
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
env = dict(os.environ)
out = []

def run(args):
    r = subprocess.run([PY] + args, capture_output=True, text=True, encoding="utf-8", env=env)
    print(r.stdout, end="", flush=True)
    if r.returncode:
        print(r.stderr[-2000:], flush=True)
    return r.stdout

run([os.path.join(HERE, "mide-tps.py"), "q27-final", "1000", "1000", "1000", "1000", "30000", "30000", "30000", "30000",
     "100000", "100000", "100000", "100000", "150000", "150000", "150000", "150000", "--gen", "400"])
img = os.path.join(HERE, "out", "vision", "dash4k.png")
for i in range(3):
    # a new image each time (one pixel changed), so neither the encoder nor the prompt cache is reused
    from PIL import Image
    im = Image.open(img).convert("RGB")
    im.putpixel((i, 0), (i * 40, 0, 0))
    p = os.path.join(HERE, "out", "vision", f"dash4k_{i}.png")
    im.save(p)
    run([os.path.join(HERE, "prueba-imagen.py"), p])
q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("VRAM after the runs (the server holds a 150k context and has encoded 4K images):")
print(q.stdout)
