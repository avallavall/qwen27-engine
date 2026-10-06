# Random outputs that llama.cpp's grammar admits (well-formed calls), plus random EOS cut points.
import json, os, random, sys

here = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("cases_mod", os.path.join(here, "cases.py"))
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
TOOLS = cm.TOOLS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
DELIM = "\n</parameter>\n"

FRAG = ["a", "bc", " ", "  ", "\n", "\n\n", "\t", "x < y", "<", ">", "</", "</thi", "think>", "<tool_", "tool_call",
        "</parameter", "</function>", "<parameter=x>", "{", "}", "[1, 2]", "\"q\"", "\\", "\\n", "null", "true", "42",
        "é", "世界", "😀", " ", "\r\n", "'", "/", "&amp;", "<b>", "</tool_call>", "<tool_call>"]

def text(maxn=8, allow_call_open=False):
    s = "".join(rng.choice(FRAG) for _ in range(rng.randint(0, maxn)))
    if not allow_call_open:
        s = s.replace("<tool_call>", "<tool_call >")
    return s

def string_value():
    while True:
        v = text(10, allow_call_open=True)
        if (v + DELIM).find(DELIM) == len(v):
            return v

def json_value(depth=0):
    k = rng.randint(0, 6 if depth < 2 else 4)
    if k == 0: return None
    if k == 1: return rng.choice([True, False])
    if k == 2: return rng.randint(-10**6, 10**6)
    if k == 3: return rng.choice([0.5, -1.25, 1e-3, 12345.678, 3.0])
    if k == 4: return string_value()
    if k == 5: return [json_value(depth + 1) for _ in range(rng.randint(0, 3))]
    return {text(3): json_value(depth + 1) for _ in range(rng.randint(0, 3))}

def dump(v):
    seps = rng.choice([(",", ":"), (", ", ": ")])
    s = json.dumps(v, ensure_ascii=rng.random() < 0.3, separators=seps)
    return s

def num_text(integer):
    if integer:
        return str(rng.choice([0, 7, -3, 123456789, -0]))
    return rng.choice(["0.5", "-1.25", "1e-3", "2E+10", "12345.678", "3", "-0.0"])

def kinds_of(schema):
    t = schema.get("type")
    if "enum" in schema: return "enum"
    if "$ref" in schema: return "integer"
    if "anyOf" in schema: return "anyof"
    if isinstance(t, list): return "union:" + ",".join(t)
    return t or "any"

def value_for(schema):
    k = kinds_of(schema)
    if k == "string": return string_value()
    if k == "integer": return num_text(True)
    if k == "number": return num_text(False)
    if k == "boolean": return rng.choice(["true", "false"])
    if k == "null": return "null"
    if k == "object": return dump({text(3): json_value(1) for _ in range(rng.randint(0, 3))})
    if k == "array": return dump([json_value(1) for _ in range(rng.randint(0, 3))])
    if k == "anyof": return rng.choice([num_text(True), "true", "false"])
    if k == "enum": return rng.choice(["1", "2", "x", "y z"])
    if k.startswith("union:") and "string" not in k:  # JSON only
        return rng.choice([num_text(True), "null"])
    # string unions and no type: JSON of the other kinds or any text
    if rng.random() < 0.5:
        return string_value()
    if k == "any": return dump(json_value())
    if "object" in k: return dump({text(3): json_value(1)})
    if "integer" in k: return num_text(True)
    if "null" in k: return "null"
    return string_value()

def call(tname):
    fn = TOOLS[tname]["function"]
    params = fn["parameters"].get("properties", {})
    req = list(fn["parameters"].get("required", []))
    opt = [p for p in params if p not in req]
    rng.shuffle(req)
    order = req[:] if len(req) <= 6 else list(req)
    for _ in range(rng.randint(0, len(opt))):
        order.append(rng.choice(opt))  # optional ones may repeat
    s = "<tool_call>\n<function=" + tname + ">\n"
    for p in order:
        s += "<parameter=" + p + ">\n" + value_for(params[p]) + DELIM
    return s + "</function>\n</tool_call>"

TSETS = [["get_weather", "read_file"], ["types"], ["set_union", "edit"], ["empty_args", "get_weather"],
         ["set_nullable_str", "set_nullable_int", "set_unit"], ["python", "html"]]

cases = []
for i in range(N):
    thinking = rng.random() < 0.6
    tools = rng.choice(TSETS) if rng.random() < 0.85 else []
    out = ""
    if thinking:
        out += text(10).replace("</think>", "</th ink>")
        r = rng.random()
        if r < 0.75: out += "\n</think>\n\n"
        elif r < 0.85: out += "</think>"
    if not thinking or "</think>" in out or rng.random() < 0.3:
        out += text(10)
    if tools and rng.random() < 0.7:
        for j in range(rng.randint(1, 3)):
            out += ("" if j == 0 else rng.choice(["\n", "\n\n", " \n"])) + call(rng.choice(tools))
        if rng.random() < 0.2:
            out += rng.choice(["\n", "\nDone.", "  "])
    b = out.encode("utf-8")
    if rng.random() < 0.5 and len(b) > 0:
        b = b[:rng.randint(0, len(b))]
    cases.append({"name": "rand_%05d" % i, "tools": [TOOLS[t] for t in tools] if tools else None,
                  "thinking": thinking, "text_hex": b.hex()})
json.dump(cases, open(os.path.join(here, "cases_rand.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(cases), "random cases")
