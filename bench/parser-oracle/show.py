import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(os.path.join(here, sys.argv[1] if len(sys.argv) > 1 else "out.json"), encoding="utf-8"))
cases = {c["name"]: c for c in json.load(open(os.path.join(here, "cases.json"), encoding="utf-8"))}
for r in out:
    c = cases[r["name"]]
    print("==", r["name"], "think" if c["thinking"] else "nothink", "par=%s" % r.get("parallel"))
    print("   in :", repr(c["text"]) if "text" in c else bytes.fromhex(c["text_hex"]))
    if not r["ok"]:
        print("   ERROR:", r["error"][:300])
        continue
    if "content_hex" in r:
        print("   R  :", bytes.fromhex(r["reasoning_hex"]))
        print("   C  :", bytes.fromhex(r["content_hex"]))
        for tc in r["tool_calls"]:
            print("   TC :", tc["name"], bytes.fromhex(tc["arguments_hex"]))
        continue
    print("   R  :", repr(r["reasoning"]))
    print("   C  :", repr(r["content"]))
    for tc in r["tool_calls"]:
        print("   TC :", tc["name"], tc["arguments"], "id=%r" % tc["id"])
    if "stream_ok" in r:
        print("   stream_ok:", r["stream_ok"])
