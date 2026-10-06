# Check of the q27 server. Start the server with an API key, then run:
#   $env:Q27_API_KEY = "test-key"; .\build\q27_server.exe -m <model.gguf> --port 8081
#   $env:BENCH_KEY = "test-key"; .venv\Scripts\python.exe bench\test_server.py
# Prints PASS / FAIL per check and exits with 1 on any failure. Needs the GPU server running; takes about 2 minutes.
import json, os, re, socket, sys, threading, time, urllib.error, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "tok-corpus.txt")
fails = 0


def check(name, ok, info=""):
    global fails
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({info})" if info else ""), flush=True)
    if not ok:
        fails += 1


def req(path, body=None, key=KEY, method=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(URL + path, data=data, headers=h, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, f.headers, f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def chat(body):
    st, _, b = req("/v1/chat/completions", body)
    return st, json.loads(b)


def stream(body):
    """Returns (status, list of chunk dicts, raw text)."""
    body = dict(body, stream=True)
    st, hd, b = req("/v1/chat/completions", body)
    text = b.decode("utf-8")
    chunks = []
    for ev in text.split("\n\n"):
        ev = ev.strip()
        if ev.startswith("data: ") and ev != "data: [DONE]":
            chunks.append(json.loads(ev[6:]))
    return st, hd, chunks, text


def collect(chunks):
    r, c, calls, fin, usage, timings = "", "", {}, None, None, None
    for ch in chunks:
        if ch.get("usage"):
            usage = ch["usage"]
        if ch.get("timings"):
            timings = ch["timings"]
        for choice in ch.get("choices", []):
            d = choice.get("delta", {})
            r += d.get("reasoning_content") or ""
            c += d.get("content") or ""
            for tc in d.get("tool_calls", []):
                e = calls.setdefault(tc["index"], {"id": None, "name": None, "arguments": "", "first": tc})
                if tc.get("id"):
                    e["id"] = tc["id"]
                f = tc.get("function", {})
                if f.get("name"):
                    e["name"] = f["name"]
                e["arguments"] += f.get("arguments", "")
            if choice.get("finish_reason"):
                fin = choice["finish_reason"]
    return r, c, calls, fin, usage, timings


WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]

# ---- endpoints and the API key
st, _, b = req("/health", key="")
check("GET /health without a key", st == 200 and json.loads(b)["status"] == "ok")
if KEY:
    st, _, b = req("/v1/models", key="")
    check("missing key -> 401", st == 401 and json.loads(b)["error"]["message"] == "Invalid API Key", st)
    st, _, b = req("/v1/models", key="wrong")
    check("wrong key -> 401", st == 401, st)
st, _, b = req("/v1/models")
models = json.loads(b)
check("GET /v1/models", st == 200 and models["data"][0]["object"] == "model", models["data"][0]["id"])
st, _, b = req("/props")
props = json.loads(b)
n_ctx = props["default_generation_settings"]["n_ctx"]
check("GET /props", st == 200 and n_ctx > 0 and "enable_thinking" in props["chat_template"], f"n_ctx {n_ctx}")
st, _, b = req("/tokenize", {"content": "Hello <think> world<|im_end|>"})
toks = json.loads(b)["tokens"]
st2, _, b2 = req("/detokenize", {"tokens": toks})
check("tokenize / detokenize round trip", json.loads(b2)["content"] == "Hello <think> world<|im_end|>" and 248068 in toks and 248046 in toks, toks)
st, _, b = req("/apply-template", {"messages": [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Hi"}]})
p = json.loads(b)["prompt"]
check("apply-template (developer role -> system)", p.startswith("<|im_start|>system\nBe brief.<|im_end|>") and p.endswith("<|im_start|>assistant\n<think>\n"))

# ---- plain chat
st, r = chat({"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 300})
m = r["choices"][0]["message"]
t = r.get("timings", {})
check("non-stream chat", st == 200 and r["choices"][0]["finish_reason"] == "stop" and m["content"].strip() != ""
      and not m["content"].startswith("\n") and "<think>" not in m["content"] and m.get("reasoning_content"), repr(m["content"][:40]))
check("usage and timings fields", r["usage"]["prompt_tokens"] > 0 and r["usage"]["completion_tokens"] > 0 and
      all(k in t for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                          "predicted_per_second", "draft_n", "draft_n_accepted")), t.get("predicted_per_second"))

st, hd, chunks, text = stream({"messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 400,
                               "stream_options": {"include_usage": True}})
rr, cc, calls, fin, usage, timings = collect(chunks)
check("stream: content type and role chunk", hd.get("Content-Type", "").startswith("text/event-stream") and
      chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None})
check("stream: reasoning, content, finish, usage chunk, [DONE]", rr and cc and fin == "stop" and usage and
      chunks[-1]["choices"] == [] and text.rstrip().endswith("data: [DONE]") and timings, repr(cc[:40]))

st, r = chat({"messages": [{"role": "user", "content": "Write a long poem."}], "max_tokens": 7})
check("max_tokens -> finish length", r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 7,
      r["usage"]["completion_tokens"])
st, r = chat({"messages": [{"role": "user", "content": "Count from 1 to 10, comma separated."}], "max_tokens": 2000,
              "chat_template_kwargs": {"enable_thinking": False}, "stop": ["7"]})
m = r["choices"][0]["message"]
check("thinking off: no reasoning", not m.get("reasoning_content"), repr(m["content"][:30]))
check("stop string -> finish stop", r["choices"][0]["finish_reason"] == "stop" and "8" not in m["content"], repr(m["content"]))
def tpl(body):
    st, _, b = req("/apply-template", body)
    return json.loads(b)["prompt"] if st == 200 else ""
eff = {e: tpl({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": e})
       for e in ("low", "medium", "high", "xhigh", "none")}
check("effort levels: low, medium, high -> xhigh, xhigh, none -> thinking off",
      "Reasoning effort is set to low." in eff["low"] and "Reasoning effort" not in eff["medium"] and
      "Reasoning effort is set to xhigh." in eff["high"] and eff["high"] == eff["xhigh"] and
      eff["none"].endswith("<think>\n\n</think>\n\n"))
check("effort given as {reasoning: {effort}} (Qwen Code form)",
      tpl({"messages": [{"role": "user", "content": "Hi"}], "reasoning": {"effort": "low"}}) == eff["low"])
st, r = chat({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "ultra"})
check("unknown effort -> 400 (template message)", st == 400 and "Unexpected reasoning effort" in r["error"]["message"])
st, r = chat({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none", "max_tokens": 200})
check("reasoning_effort none -> thinking off", st == 200 and not r["choices"][0]["message"].get("reasoning_content"))

# ---- tool calls
msgs = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
st, hd, chunks, text = stream({"messages": msgs, "tools": WEATHER, "max_tokens": 2000})
rr, cc, calls, fin, usage, timings = collect(chunks)
ok = fin == "tool_calls" and 0 in calls
args = {}
if ok:
    first = calls[0]["first"]
    try:
        args = json.loads(calls[0]["arguments"])
    except Exception:
        args = None
    ok = first.get("id") and first.get("function", {}).get("name") == "get_weather" and isinstance(args, dict) and "paris" in args.get("city", "").lower()
check("stream tool call (first delta has id + name, JSON arguments)", ok, f"{fin} {calls.get(0, {}).get('arguments')}")
if ok:
    msgs += [{"role": "assistant", "content": cc or None, "reasoning_content": rr,
              "tool_calls": [{"id": calls[0]["id"], "type": "function", "function": {"name": "get_weather", "arguments": calls[0]["arguments"]}}]},
             {"role": "tool", "tool_call_id": calls[0]["id"], "content": [{"type": "text", "text": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}]}]
    st, r = chat({"messages": msgs, "tools": WEATHER, "max_tokens": 2000})
    m = r["choices"][0]["message"]
    check("answer after the tool result (and the prompt cache reused the turn)", st == 200 and "18" in m["content"] and
          r["timings"]["cache_n"] > 0.8 * r["usage"]["prompt_tokens"], f"cache {r['timings']['cache_n']} of {r['usage']['prompt_tokens']}")
st, r = chat({"messages": [{"role": "user", "content": "Weather in Rome and in Oslo? Call the tool for both cities at once."}],
              "tools": WEATHER, "max_tokens": 3000})
tcs = r["choices"][0]["message"].get("tool_calls", [])
check("non-stream: several tool calls", r["choices"][0]["finish_reason"] == "tool_calls" and len(tcs) >= 2 and
      len({tc["id"] for tc in tcs}) == len(tcs), [tc["function"]["arguments"] for tc in tcs])

# ---- context overflow
text_all = open(CORPUS, encoding="utf-8", errors="ignore").read()
big = text_all[: int(n_ctx * 4.2)]
st, r = chat({"messages": [{"role": "user", "content": big}], "max_tokens": 10})
msg = r.get("error", {}).get("message", "")
check("overflow -> 400 with llama.cpp and OpenAI wording", st == 400 and r["error"]["type"] == "exceed_context_size_error" and
      re.search("exceeds the available context size", msg) and re.search("maximum context length", msg), msg[:120])

# ---- queue: three requests at once, all must finish
res = {}
def one(i):
    res[i] = chat({"messages": [{"role": "user", "content": f"Name {i + 2} colors."}], "max_tokens": 400})[0]
th = [threading.Thread(target=one, args=(i,)) for i in range(3)]
[x.start() for x in th]
[x.join() for x in th]
check("three requests at once (queue)", all(v == 200 for v in res.values()) and len(res) == 3, res)

# ---- client disconnect during the prompt: the server must stop and stay usable
data = json.dumps({"messages": [{"role": "user", "content": text_all[:300000] + "\nSummarize."}], "stream": True}).encode()
s = socket.create_connection(("127.0.0.1", int(URL.rsplit(":", 1)[1])))
hdr = f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
if KEY:
    hdr += f"Authorization: Bearer {KEY}\r\n"
s.sendall(hdr.encode() + b"\r\n" + data)
time.sleep(2)
s.close()
t0 = time.time()
st, r = chat({"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 200})
dt = time.time() - t0
check("disconnect during a long prompt stops it", st == 200 and dt < 15, f"next request took {dt:.1f} s")

# ---- prompt cache: a side request, then the long conversation again (comes back from RAM)
conv = [{"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": text_all[1000000:1060000] + "\n\nWhich language is this? One sentence."}]
st, r1 = chat({"messages": conv, "max_tokens": 600})
st, r2 = chat({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 100})
conv2 = conv + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"],
                 "reasoning_content": r1["choices"][0]["message"].get("reasoning_content", "")},
                {"role": "user", "content": "And which libraries does it use?"}]
st, r3 = chat({"messages": conv2, "max_tokens": 600})
pt = r1["usage"]["prompt_tokens"]
check("cache: long conversation back after a side request", st == 200 and r3["timings"]["cache_n"] >= pt - 8,
      f"first prompt {pt}, cached now {r3['timings']['cache_n']} of {r3['usage']['prompt_tokens']}, {r3['timings']['prompt_ms']:.0f} ms")
conv3 = conv[:1] + [{"role": "user", "content": conv[1]["content"].replace("One sentence.", "Two sentences.")}]
st, r4 = chat({"messages": conv3, "max_tokens": 600})
# the edit is near the end of the last message: the restart point is the newest checkpoint before it
check("cache: edited last message restarts from a checkpoint", st == 200 and r4["timings"]["cache_n"] > 0,
      f"cached {r4['timings']['cache_n']}, read {r4['timings']['prompt_n']}")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
