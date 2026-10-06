import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
cases = {c["name"]: c for c in json.load(open(os.path.join(here, "cases_rand.json"), encoding="utf-8"))}
a = {r["name"]: r for r in json.load(open(os.path.join(here, "out_rand.json"), encoding="utf-8"))}
b = {r["name"]: r for r in json.load(open(os.path.join(here, "mine_rand.json"), encoding="utf-8"))}
bad = 0
errors = 0
ncalls = 0
for name, ra in a.items():
    rb = b[name]
    if not ra["ok"]:
        errors += 1
        print("ORACLE ERROR", name, ra["error"][:100])
        continue
    ta = [(t["name"], t["arguments_hex"]) for t in ra["tool_calls"]]
    tb = [(t["name"], t["arguments_hex"]) for t in rb["tool_calls"]]
    ncalls += len(ta)
    same = ra["reasoning_hex"] == rb["reasoning_hex"] and ra["content_hex"] == rb["content_hex"] and ta == tb
    if not same or not rb["stream_ok"]:
        bad += 1
        if bad <= 8:
            print("MISMATCH", name, "stream_ok=%s" % rb["stream_ok"])
            print("  text :", bytes.fromhex(cases[name]["text_hex"]))
            for k in ("reasoning_hex", "content_hex"):
                if ra[k] != rb[k]:
                    print("  %s llama: %r" % (k, bytes.fromhex(ra[k])))
                    print("  %s mine : %r" % (k, bytes.fromhex(rb[k])))
            if ta != tb:
                print("  calls llama:", [(n, bytes.fromhex(h)) for n, h in ta])
                print("  calls mine :", [(n, bytes.fromhex(h)) for n, h in tb])
print("cases %d, calls %d, mismatches %d, oracle errors %d" % (len(a), ncalls, bad, errors))
