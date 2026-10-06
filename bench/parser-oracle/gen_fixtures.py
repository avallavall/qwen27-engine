# Fills the fixture block of tools/test_parser.cpp from cases.json + out.json (oracle = production llama-common.dll).
import json, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(here, "..", ".."))
sys.path.insert(0, here)
import importlib.util
spec = importlib.util.spec_from_file_location("cases_mod", os.path.join(here, "cases.py"))
cases_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases_mod)
TOOLS = cases_mod.TOOLS

cases = json.load(open(os.path.join(here, "cases.json"), encoding="utf-8"))
out = {r["name"]: r for r in json.load(open(os.path.join(here, "out.json"), encoding="utf-8"))}

SKIP = {"own_nothink_parallel_false"}  # parallel_tool_calls=false is not supported

# Cases where llama.cpp drops a call whose id/name a stream has already sent (it never sees such text in
# production, its grammar forbids it). Value: this parser's result (reasoning, content, calls).
POLICY = {
    "own_types_any_json": ("", "", [("types", '{"any":7,"e":"x","anyof":"maybe"}')]),
    "own_int_not_json": ("", "", [("types", '{"i":"abc"}')]),
    "own_int_leading_space": ("", "", [("types", '{"i":" 42"}')]),
    "own_int_trailing_space": ("", "", [("types", '{"i":"42 "}')]),
    "own_unknown_param": ("", "", [("get_weather", '{"city":"Paris","country":"FR"}')]),
    "own_missing_required": ("", "", [("get_weather", '{"days":2}')]),
    "own_dup_required": ("", "", [("get_weather", '{"city":"Paris","city":"Rome"}')]),
    "own_call_no_trailing_nl_func": ("", "", [("get_weather", '{"city":"Paris"}')]),
    "own_param_no_nl": ("", "", [("get_weather", '{}')]),
    "own_python_literals": ("", "", [("types", '{"o":"{\'a\': True, \'b\': None, \'c\': \'it\'\'s\'}"}')]),
    "own_optional_before_required": ("", "", [("get_weather", '{"days":2,"city":"Paris"}')]),
    "own_json_bad_number": ("", "", [("types", '{"f":"01"}')]),
    "own_json_bad_number2": ("", "", [("types", '{"f":"1."}')]),
    "own_json_bad_escape": ("", "", [("types", '{"o":"{\\"q\\": \\"a\\\\x\\"}"}')]),
    "own_param_name_space": ("", "", [("get_weather", '{"city":"Paris"}')]),
    "raw_param_invalid": (b"", b"", [("read_file", b'{"path":"a\xef\xbf\xbdb"}')]),
    # EOS after a tag start + incomplete UTF-8: llama.cpp moves the reasoning to content / drops the open call
    "raw_quirk_reasoning_tail": (b"abc", b"", []),
    "raw_quirk_param_tail": (b"", b"", [("read_file", b'{"path":"ab')]),
    # strict JSON: a raw newline inside a JSON string is not accepted (llama.cpp copies it: invalid JSON)
    "own_json_raw_newline_in_string": ("", "", [("types", '{"o":"{\\"a\\": \\"x"}')]),
}

def is_hex(ch):
    return ch in "0123456789abcdefABCDEF"

def cstr(b, raw_bytes=False):
    """C++ string literal(s) for bytes; long strings are split after newlines."""
    if isinstance(b, str):
        b = b.encode("utf-8")
    parts, cur = [], ""
    i = 0
    text = b
    # keep valid UTF-8 characters raw unless raw_bytes
    s = None
    if not raw_bytes:
        try:
            s = text.decode("utf-8")
        except UnicodeDecodeError:
            s = None
    items = list(s) if s is not None else [bytes([x]) for x in text]
    prev_hex_escape = False
    for it in items:
        if isinstance(it, bytes):
            c = it[0]
            ch = chr(c) if c < 0x80 else None
        else:
            ch = it
            c = ord(it)
        if ch == "\\":
            esc = "\\\\"
        elif ch == '"':
            esc = '\\"'
        elif ch == "\n":
            esc = "\\n"
        elif ch == "\r":
            esc = "\\r"
        elif ch == "\t":
            esc = "\\t"
        elif ch is None or c < 0x20 or c == 0x7f:
            esc = "\\x%02x" % c
        else:
            esc = ch
        if prev_hex_escape and is_hex(esc[0]):
            cur += '" "'
        cur += esc
        prev_hex_escape = esc.startswith("\\x")
        if ch == "\n" and len(text) > 60:
            parts.append(cur)
            cur = ""
            prev_hex_escape = False
    if cur or not parts:
        parts.append(cur)
    return " ".join('"%s"' % p for p in parts) if len(parts) == 1 else ("\n         ".join('"%s"' % p for p in parts))

lines = []
lines.append("// Tool definitions used by the cases (OpenAI format, as sent in \"tools\").")
lines.append("const std::vector<std::pair<const char*, const char*>> kToolDefs = {")
for name, t in TOOLS.items():
    lines.append("    {%s, %s}," % (cstr(name), cstr(json.dumps(t, ensure_ascii=False, separators=(",", ":")))))
lines.append("};")
lines.append("")
lines.append("// Expected values: llama.cpp = output of the production llama-common.dll (common_chat_parse, final parse) on")
lines.append("// the same text and tools; policy = this parser's rule where llama.cpp drops a call (see chat_parser.h).")
lines.append("const std::vector<Case> kCases = {")
n_llama = n_policy = 0
for c in cases:
    name = c["name"]
    if name in SKIP:
        continue
    r = out[name]
    raw = "text_hex" in c
    text = bytes.fromhex(c["text_hex"]) if raw else c["text"].encode("utf-8")
    tools = [t["function"]["name"] for t in c["tools"]] if c["tools"] else []
    if name in POLICY:
        rs, cs, calls = POLICY[name]
        src = "policy"
        n_policy += 1
        if not r["ok"]:
            note = "llama.cpp throws: " + r["error"][:60]
        elif "content_hex" in r:
            note = "llama.cpp: reasoning %r, content %r, %d call(s)" % (
                bytes.fromhex(r["reasoning_hex"]), bytes.fromhex(r["content_hex"]), len(r["tool_calls"]))
        else:
            note = "llama.cpp: content %r, %d call(s)" % (r["content"], len(r["tool_calls"]))
            if r["tool_calls"]:
                note += " " + " ".join(tc["arguments"] for tc in r["tool_calls"])
    else:
        if not r["ok"]:
            raise SystemExit("oracle error without policy: " + name)
        src = "llama.cpp"
        n_llama += 1
        note = None
        if "content_hex" in r:
            rs = bytes.fromhex(r["reasoning_hex"])
            cs = bytes.fromhex(r["content_hex"])
            calls = [(tc["name"], bytes.fromhex(tc["arguments_hex"])) for tc in r["tool_calls"]]
        else:
            rs, cs = r["reasoning"], r["content"]
            calls = [(tc["name"], tc["arguments"]) for tc in r["tool_calls"]]
    if note:
        lines.append("    // " + note.replace("\n", "\\n"))
    lines.append("    {%s, %s, %s, {%s}," % (cstr(name), '"%s"' % src, "true" if c["thinking"] else "false",
                                            ", ".join(cstr(t) for t in tools)))
    lines.append("     %s," % cstr(text, raw))
    lines.append("     %s, %s," % (cstr(rs, raw), cstr(cs, raw)))
    lines.append("     {%s}, %s}," % (", ".join("{%s, %s}" % (cstr(n), cstr(a, raw)) for n, a in calls),
                                    "false" if raw else "true"))
lines.append("};")

path = os.path.join(REPO, "tools", "test_parser.cpp")
src = open(path, encoding="utf-8").read()
begin = "// ---- fixtures (generated) begin\n"
end = "// ---- fixtures (generated) end\n"
a, b = src.index(begin) + len(begin), src.index(end)
src = src[:a] + "\n".join(lines) + "\n" + src[b:]
open(path, "w", encoding="utf-8", newline="\n").write(src)
print("cases: %d llama.cpp, %d policy" % (n_llama, n_policy))
