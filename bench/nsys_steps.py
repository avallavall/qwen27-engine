# Split llama.cpp decode time into parts from an nsys SQLite export.
# Usage: python nsys_steps.py <report.sqlite> [--list]
# Prints: requests found (by gaps), per-device busy/idle time in the last pure-decode window,
# and time per kernel category.
import sqlite3, sys, re, collections

db = sys.argv[1]
c = sqlite3.connect(db)
S = {i: v for i, v in c.execute("select id, value from StringIds")}
rows = c.execute("select deviceId, start, end, shortName, demangledName, streamId, graphNodeId, "
                 "gridX, gridY, gridZ, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
K = [(d, s, e, S.get(sn, str(sn)), S.get(dn, str(dn)), st, gn, gx * gy * gz, bx) for d, s, e, sn, dn, st, gn, gx, gy, gz, bx in rows]

def cat(short, full):
    n = short.lower()
    f = full.lower()
    if "mul_mat_vec_q" in n or "mmvq" in n: return "gemv_quant"
    if "mul_mat_vec_f" in n or "mmvf" in n: return "gemv_bf16_f32"
    if "mul_mat_q" in n or "mmq" in n: return "gemm_quant"
    if "quantize" in n: return "act_quantize_q8_1"
    if "gated_delta" in n or "delta_net" in n or "gdn" in n: return "deltanet"
    if "ssm_conv" in n or "conv1d" in n: return "deltanet_conv"
    if "flash_attn" in n or "fattn" in n: return "attention"
    if "allreduce" in n or "all_reduce" in n or "reduce_add" in f: return "allreduce_kernel"
    if "rms_norm" in n or "l2_norm" in n or "norm" in n: return "norm"
    if "rope" in n: return "rope"
    if "argmax" in n or "softmax" in n or "top_k" in n or "argsort" in n or "sampl" in n: return "sampling_softmax"
    if "cpy" in n or "copy" in n or "get_rows" in n or "concat" in n or "set_rows" in n: return "copy_getrows"
    if "glu" in n or "silu" in n or "swiglu" in n or "sigmoid" in n or "softplus" in n or "unary" in n or "exp" in n: return "elementwise_act"
    if "bin_bcast" in n or "add" in n or "mul" in n or "scale" in n or "sub" in n: return "elementwise_bin"
    return "other:" + short[:40]

if "--list" in sys.argv:
    tot = collections.Counter(); cnt = collections.Counter()
    for d, s, e, sn, dn, *_ in K:
        tot[sn] += e - s; cnt[sn] += 1
    for k, v in tot.most_common(60):
        print(f"{v/1e6:10.1f} ms {cnt[k]:8d}  {cat(k, k):20s} {k[:100]}")
    sys.exit()

# Find request windows: gaps > 300 ms between kernels on any device separate requests.
segs = []
beg = K[0][1]; last = K[0][2]
for k in K[1:]:
    if k[1] - last > 300e6:
        segs.append((beg, last)); beg = k[1]
    last = max(last, k[2])
segs.append((beg, last))
for i, (a, b) in enumerate(segs):
    n = sum(1 for k in K if a <= k[1] <= b)
    print(f"segment {i}: {(a-K[0][1])/1e9:8.3f}s .. {(b-K[0][1])/1e9:8.3f}s  dur {(b-a)/1e6:9.1f} ms  kernels {n}")

seg = int(sys.argv[sys.argv.index("--seg") + 1]) if "--seg" in sys.argv else len(segs) - 1
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 0
a, b = segs[seg]
W = [k for k in K if a <= k[1] <= b]
dur = b - a
print(f"\nwindow = segment {seg}, {dur/1e6:.1f} ms, steps given = {steps}")
for dev in sorted(set(k[0] for k in W)):
    ks = [k for k in W if k[0] == dev]
    # union of kernel intervals = busy time
    busy = 0; cs, ce = ks[0][1], ks[0][2]
    for k in ks[1:]:
        if k[1] > ce: busy += ce - cs; cs, ce = k[1], k[2]
        else: ce = max(ce, k[2])
    busy += ce - cs
    ktime = sum(k[2] - k[1] for k in ks)
    per = collections.Counter(); pcnt = collections.Counter()
    for k in ks:
        cc = cat(k[3], k[4]); per[cc] += k[2] - k[1]; pcnt[cc] += 1
    div = steps if steps else 1
    unit = "ms/step" if steps else "ms total"
    print(f"\n== device {dev}: kernels {len(ks)} ({len(ks)/div:.0f} per step), kernel time {ktime/1e6/div:.2f} {unit}, "
          f"busy {busy/1e6/div:.2f}, idle {(dur-busy)/1e6/div:.2f} {unit}, window {dur/1e6/div:.2f}")
    for cc, v in per.most_common():
        print(f"   {cc:28s} {v/1e6/div:8.3f} {unit}  {pcnt[cc]/div:7.1f} kernels  avg {v/pcnt[cc]/1e3:7.2f} us")
