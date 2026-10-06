# Ranks the vocabulary for the MTP draft head: data\draft_vocab.bin = int32 token ids, most useful first.
# The server keeps the first N (--draft-vocab N). Ranking: special tokens first, then the mean of the token
# frequencies of three sources (each normalized to 1): code + English text, Spanish text, the model's own outputs;
# tokens never seen follow in id order (low ids are early BPE merges, frequent in the tokenizer's training data).
# Counts come from build\vocab_freq.exe:
#   vocab_freq <gguf> bench\out\freq_corpus.bin bench\out\tok-corpus.txt bench\out\long-text.txt bench\out\ppl-text.txt
#   vocab_freq <gguf> bench\out\freq_es.bin <Spanish text files>
#   vocab_freq <gguf> bench\out\freq_out.bin bench\out\reqlog\res-*.json          (server logs: generated text)
#   vocab_freq <gguf> bench\out\freq_out_a.bin <half of them>, freq_out_b.bin <other half>   (held-out check)
# Usage: .venv\Scripts\python.exe bench\make_draft_vocab.py
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
V = 248320
SPECIAL = range(248044, 248077)  # <|endoftext|> .. the user-defined and control tokens (not the [PAD] ones)


def load(name):
    p = os.path.join(OUT, name)
    return np.fromfile(p, dtype=np.uint32).astype(np.float64) if os.path.exists(p) else None


def rank(sources):
    score = np.zeros(V)
    for c in sources:
        if c is not None and c.sum() > 0:
            score += c / c.sum()
    score[list(SPECIAL)] = 10.0  # always first
    order = np.lexsort((np.arange(V), -score))  # by score, then by id
    return order


corpus, es, out = load("freq_corpus.bin"), load("freq_es.bin"), load("freq_out.bin")
out_a, out_b = load("freq_out_a.bin"), load("freq_out_b.bin")
es_test = load("freq_es_test.bin")  # Spanish model outputs not used for the ranking (bench\gen_es.py, odd files)
if out_a is not None and out_b is not None:
    # held-out check: rank without out_b, measure how much of out_b (and of es_test) each size covers
    order = rank([corpus, es, out_a])
    for K in (16384, 32768, 49152, 65536, 98304):
        sub = np.zeros(V, bool)
        sub[order[:K]] = True
        msg = f"K {K:6d}: held-out outputs covered {out_b[sub].sum() / out_b.sum():.4f}"
        if es_test is not None:
            msg += f", held-out Spanish outputs {es_test[sub].sum() / es_test.sum():.4f}"
        print(msg)
order = rank([corpus, es, out])
os.makedirs(os.path.join(HERE, "..", "data"), exist_ok=True)
dst = os.path.join(HERE, "..", "data", "draft_vocab.bin")
order.astype(np.int32).tofile(dst)
print("wrote", os.path.normpath(dst), len(order), "ids")
