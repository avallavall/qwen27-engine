# Draft acceptance A/B: runs `q27_gen ... accept 16` for each setting on code (tok_200k.bin) and Spanish
# (es_test.txt) prompts, settings in turn, repeated. The prompts and random draws are the same for every setting.
# Usage: .venv\Scripts\python.exe bench\accept_ab.py <reps> "<label>=<ENV=v,ENV=v>" ["<label>=..."] ...
#   env ACCEPT_PROMPT (default 1000 tokens), ACCEPT_GEN (300), ACCEPT_INPUTS (comma list, default both files).
import os, re, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
MODEL = os.environ.get("Q27_MODEL") or next(
    (p for p in (os.path.join(ROOT, "models", GGUF), os.path.join(ROOT, "..", "qwen38_27", "models", "Qwen3.8-27B", GGUF))
     if os.path.exists(p)), os.path.join(ROOT, "models", GGUF))
reps = int(sys.argv[1])
cfgs = []
for a in sys.argv[2:]:
    label, _, envs = a.partition("=")
    cfgs.append((label, dict(kv.partition("=")[::2] for kv in filter(None, envs.split(",")))))
inputs = os.environ.get("ACCEPT_INPUTS", "bench/out/tok_200k.bin,bench/out/es_test.txt").split(",")
np_, ngen = os.environ.get("ACCEPT_PROMPT", "1000"), os.environ.get("ACCEPT_GEN", "300")
res = {(l, i): [] for l, _ in cfgs for i in inputs}
for r in range(reps):
    for label, env in cfgs:
        for inp in inputs:
            e = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                     Q27_DRAFT_VOCAB=os.path.join(ROOT, "data", "draft_vocab.bin") + ":32768")
            e.update(env)
            cmd = [os.path.join(ROOT, "build", "q27_gen.exe"), MODEL, os.path.join(ROOT, inp), "0,1", np_, ngen, "accept", "16"]
            out = subprocess.run(cmd, env=e, capture_output=True, text=True, cwd=ROOT).stdout
            m = re.search(r"([\d.]+) tok/step, ([\d.]+) ms/step, ([\d.]+) tok/s", out)
            if not m:
                print(f"rep {r} {label} {inp}: no result\n{out[-500:]}", flush=True)
                continue
            res[(label, inp)].append(tuple(map(float, m.groups())))
            print(f"rep {r} {label:10s} {os.path.basename(inp):14s} {m.group(0)}", flush=True)
print("setting / input: median tok/step, ms/step, tok/s")
for label, _ in cfgs:
    for inp in inputs:
        v = res[(label, inp)]
        if v:
            print(f"  {label:10s} {os.path.basename(inp):14s} " + ", ".join(f"{statistics.median(x[k] for x in v):.3f}"
                                                                      for k in range(3)))
