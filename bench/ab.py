# A/B timing of engine settings with q27_gen, settings run in turn and repeated (the desktop on card 0 adds noise).
# Usage: .venv\Scripts\python.exe bench\ab.py <reps> "<label>=<ENV=v,ENV=v>" ["<label>=..."] ...
#   env BENCH_MODE=sample (default; 1k prompt, 400 tokens, 3 runs, the first run is dropped) or depth:<D> (prefill to
#   depth D, then 2 runs of 400 tokens). Prints the median and min ms/step per setting (and prompt t/s in depth mode).
import os, re, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
# model: $Q27_MODEL, else models\ in this repo, else qwen38_27\models\Qwen3.8-27B next to the repo (as start-server.ps1)
MODEL = os.environ.get("Q27_MODEL") or next(
    (p for p in (os.path.join(ROOT, "models", GGUF), os.path.join(ROOT, "..", "qwen38_27", "models", "Qwen3.8-27B", GGUF))
     if os.path.exists(p)), os.path.join(ROOT, "models", GGUF))
reps = int(sys.argv[1])
cfgs = []
for a in sys.argv[2:]:
    label, _, envs = a.partition("=")
    env = {}
    for kv in filter(None, envs.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    cfgs.append((label, env))
mode = os.environ.get("BENCH_MODE", "sample")
res = {l: [] for l, _ in cfgs}
pp = {l: [] for l, _ in cfgs}
for r in range(reps):
    for label, env in cfgs:
        e = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", Q27_DRAFT_VOCAB=os.path.join(ROOT, "data", "draft_vocab.bin") + ":32768")
        e.update(env)
        if mode == "sample":
            cmd = [os.path.join(ROOT, "build", "q27_gen.exe"), MODEL, os.path.join(ROOT, "bench", "out", "llama_tp2_ub4_base.bin"),
                   "0,1", "1000", "400", "sample"]
        else:
            d = mode.split(":")[1]
            cmd = [os.path.join(ROOT, "build", "q27_gen.exe"), MODEL, os.path.join(ROOT, "bench", "out", "tok_200k.bin"), "0,1", d,
                   "400", "depth", "2"]
        out = subprocess.run(cmd, env=e, capture_output=True, text=True, cwd=ROOT).stdout
        ms = [float(m) for m in re.findall(r"([\d.]+) ms/step", out)]
        if mode == "sample": ms = ms[1:]
        res[label] += ms
        tps = re.findall(r"\(([\d.]+) t/s\)", out)
        if tps: pp[label].append(float(tps[-1]))
        print(f"rep {r} {label}: {' '.join(f'{x:.2f}' for x in ms)}" + (f" | prompt {tps[-1]} t/s" if tps else ""), flush=True)
print("setting: median / min ms per step" + (" | prompt t/s median" if mode != "sample" else ""))
for label, _ in cfgs:
    v = res[label]
    extra = f" | {statistics.median(pp[label]):.0f}" if pp[label] else ""
    print(f"  {label:20s} {statistics.median(v):7.2f} / {min(v):7.2f}  (n={len(v)}){extra}")
