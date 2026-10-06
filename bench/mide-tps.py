# Copy of qwen38_27/mide-tps.py. Changed: port 8081, key from BENCH_KEY, CSV in bench/out.
# Mide tok/s de llama-server con prompts de un tamano dado.
# Uso:  python mide-tps.py <etiqueta> <tokens1> [tokens2 ...] [--slot2 N] [--gen N]
# Ej:   python mide-tps.py "base 256k" 1000 100000 --gen 300
# Escribe una linea por prueba en resultados-tps.csv y la imprime.
import json, sys, time, urllib.request, os, re

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "none")
AQUI = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(AQUI, "out", "tok-corpus.txt")  # same 17.9 MB corpus as the production script uses
CSV = os.path.join(AQUI, "out", "resultados-tps.csv")

def pide(ruta, cuerpo=None, metodo="POST", timeout=1800):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    r = urllib.request.Request(URL + ruta, data=datos, method=metodo,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

def cuenta(texto):
    return len(pide("/tokenize", {"content": texto})["tokens"])

def recorta(texto, objetivo):
    # busca por biseccion el trozo de texto que da ~objetivo tokens
    bajo, alto = 0, len(texto)
    while alto - bajo > 2000:
        medio = (bajo + alto) // 2
        if cuenta(texto[:medio]) < objetivo: bajo = medio
        else: alto = medio
    return texto[:bajo]

def prueba(etiqueta, ntok, gen, texto):
    if ntok <= 200:
        prompt = "Reply with exactly one short sentence about Python lists."
    else:
        prompt = recorta(texto, ntok - 120) + \
            "\n\n===== TASK =====\nSummarize in one paragraph what these files do. Then list three risks."
    t0 = time.time()
    r = pide("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "cache_prompt": True})
    seg = time.time() - t0
    t = r["timings"]
    fila = dict(etiqueta=etiqueta, ctx_pedido=ntok,
                prompt_tok=t["prompt_n"], prompt_ts=round(t["prompt_per_second"] or 0, 1),
                gen_tok=t["predicted_n"], gen_ts=round(t["predicted_per_second"], 2),
                ms_por_token=round(t["predicted_ms"] / max(t["predicted_n"], 1), 1),
                total_s=round(seg, 1))
    d = r.get("timings", {})
    for k in ("draft_n", "draft_n_accepted"):
        if k in d: fila[k] = d[k]
    if fila.get("draft_n"):
        fila["aceptacion"] = round(fila["draft_n_accepted"] / fila["draft_n"], 3)
    nuevo = not os.path.exists(CSV)
    with open(CSV, "a", encoding="utf-8") as f:
        if nuevo: f.write(",".join(fila.keys()) + "\n")
        f.write(",".join(str(v) for v in fila.values()) + "\n")
    print("  ".join(f"{k}={v}" for k, v in fila.items()), flush=True)
    return fila

if __name__ == "__main__":
    args = sys.argv[1:]
    gen = 300
    if "--gen" in args:
        i = args.index("--gen"); gen = int(args[i+1]); del args[i:i+2]
    etiqueta = args[0]
    tams = [int(x) for x in args[1:]]
    texto = open(CORPUS, encoding="utf-8", errors="ignore").read()
    print(f"--- {etiqueta} ---", flush=True)
    for n in tams:
        prueba(etiqueta, n, gen, texto)
