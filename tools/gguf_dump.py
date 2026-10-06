# Dump GGUF metadata (no big arrays) and a tensor table summary.
import sys, collections
from gguf import GGUFReader
path = sys.argv[1]
r = GGUFReader(path)
print("== metadata ==")
for k, f in r.fields.items():
    if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
        print(f"{k}: <array len {len(f.data)}>"); continue
    if k == "tokenizer.chat_template":
        v = bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        print(f"{k}: <{len(v)} chars>"); continue
    try:
        if len(f.data) == 1:
            v = f.parts[f.data[0]]
            v = bytes(v).decode("utf-8", "replace") if f.types and f.types[0].name == "STRING" else v.tolist()
        else:
            v = [f.parts[i].tolist() for i in f.data][:64]
    except Exception as e:
        v = f"<{e}>"
    print(f"{k}: {v}")
print("== tensors ==")
bytype = collections.Counter(); tot = 0
for t in r.tensors:
    bytype[t.tensor_type.name] += int(t.n_bytes); tot += int(t.n_bytes)
for k, v in bytype.most_common(): print(f"type {k}: {v/2**20:.1f} MiB")
print(f"total {tot/2**20:.1f} MiB, n_tensors {len(r.tensors)}")
want = sys.argv[2] if len(sys.argv) > 2 else "blk.0.|blk.3.|blk.64.|output|token_embd|output_norm"
import re
for t in r.tensors:
    if re.match(want, t.name) or re.match(r"^(output|token_embd)", t.name):
        print(f"{t.name:45s} {t.tensor_type.name:6s} {list(map(int,t.shape))} {int(t.n_bytes)/2**20:.2f} MiB")
