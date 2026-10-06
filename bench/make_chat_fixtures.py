# Synthetic req/res pairs for the chat oracle (bench/llama_chat.cpp), in the format of q27_server --log-dir:
#   req-NNNNN.json  {"request": <OpenAI chat body>, "prompt": <rendered prompt>}
#   res-NNNNN.json  {"raw", "reasoning_content", "content", "tool_calls", "finish_reason", "timings"}
#
# Usage (project venv):
#   .venv\Scripts\python.exe bench\make_chat_fixtures.py [--out bench\out\chatfix]
#
# The "prompt" is rendered with jinja2 from research/_chat_template.jinja (trim_blocks, lstrip_blocks,
# tojson = json.dumps(ensure_ascii=False)), after the message fixes our server applies (developer -> system,
# content null -> "", string tool-call arguments -> parsed JSON) and with our server's kwargs rules (default
# {"reasoning_effort": "medium"} merged key by key with chat_template_kwargs, top-level reasoning_effort goes
# into the kwargs, "none" means enable_thinking = false). Tools go to the template as the request gives them.
# When jinja2 raises, the prompt is "<<jinja2 raised: ...>>".
#
# reasoning_content / content / tool_calls hold what llama.cpp gives (checked with llama_chat.exe on
# 2026-10-06): the trailing whitespace before </think> and before <tool_call> is kept, tool-call arguments keep
# the model's text for non-string values. A few pairs are wrong on purpose, to show that the oracle catches
# them; index.txt and the script output mark them "expect FAIL".
import argparse
import copy
import json
import os

import jinja2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "research", "_chat_template.jinja")
DEFAULT_OUT = os.path.join(ROOT, "bench", "out", "chatfix")


class TemplateError(Exception):
    pass


def raise_exception(msg):
    raise TemplateError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_template():
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = tojson
    with open(TEMPLATE, encoding="utf-8") as f:
        return env.from_string(f.read())


def normalize(messages):
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "developer":
            m["role"] = "system"
        if "content" in m and m["content"] is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
    return out


def render(tmpl, body):
    kwargs = {"reasoning_effort": "medium"}
    kwargs.update(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    if effort == "none":
        kwargs["enable_thinking"] = False
        kwargs.pop("reasoning_effort", None)
    elif effort:
        kwargs["reasoning_effort"] = effort
    ctx = dict(kwargs)
    ctx["messages"] = normalize(body["messages"])
    ctx["tools"] = body.get("tools")
    ctx["add_generation_prompt"] = body.get("add_generation_prompt", True)
    try:
        return tmpl.render(**ctx)
    except TemplateError as e:
        return f"<<jinja2 raised: {e}>>"


# ---------------------------------------------------------------- tools

def fn_tool(name, desc, props, required, **extra):
    fn = {"name": name, "description": desc,
          "parameters": {"type": "object", "properties": props, "required": required}}
    fn.update(extra)
    return {"type": "function", "function": fn}


T_WEATHER = fn_tool("get_weather", "Get the weather forecast for a city.",
                    {"city": {"type": "string", "description": "City name"},
                     "days": {"type": "integer", "description": "Number of days", "minimum": 1, "maximum": 14},
                     "units": {"type": "string", "enum": ["metric", "imperial"]}},
                    ["city"])
T_WRITE = fn_tool("write_file", "Write a file to disk.",
                  {"path": {"type": "string"}, "content": {"type": "string"},
                   "overwrite": {"type": "boolean", "default": False}},
                  ["path", "content"])
T_SEARCH = fn_tool("search", "Search documents.",
                   {"query": {"type": "string"},
                    "limit": {"type": "number", "description": "Max results, may be fractional for paging"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object", "properties": {"year": {"type": "integer"},
                                                                 "lang": {"type": "string"}}},
                    "cursor": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                   ["query"])
T_BASH = fn_tool("bash", "Run a shell command.",
                 {"command": {"type": "string", "description": "The command"},
                  "timeout": {"type": "number", "description": "Timeout in ms"}},
                 ["command"])
TOOLS = [T_WEATHER, T_WRITE, T_SEARCH, T_BASH]


def call(name, args, cid="call_1", as_string=True):
    """A tool call in the request history."""
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False) if as_string else args}}


def expect_call(name, args):
    """An expected tool call in the result; args is a dict (compact JSON) or the exact arguments text."""
    text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": text}}


def tc_text(name, params):
    """Tool-call text in the model's format."""
    s = f"<tool_call>\n<function={name}>\n"
    for k, v in params:
        s += f"<parameter={k}>\n{v}\n</parameter>\n"
    return s + "</function>\n</tool_call>"


# ---------------------------------------------------------------- fixtures

SYS = {"role": "system", "content": "You are a helpful assistant."}
FIXTURES = []


def add(name, note, body, raw, reasoning, content, calls=None, finish=None, fail=None):
    """fail: None if the oracle should PASS, else why it should FAIL."""
    calls = calls or []
    finish = finish or ("tool_calls" if calls else "stop")
    FIXTURES.append((name, note, body, raw, reasoning, content, calls, finish, fail))


add("system-user", "plain chat, thinking on (medium)",
    {"messages": [SYS, {"role": "user", "content": "Hello!"}]},
    "The user greets me.\n</think>\n\nHello! How can I help you today?",
    "The user greets me.\n", "Hello! How can I help you today?")

add("whitespace-reasoning", "reasoning and content with surrounding whitespace: leading cut, trailing kept",
    {"messages": [SYS, {"role": "user", "content": "  Hi with spaces  \n"}]},
    "\n\n  Thinking with spaces.  \n\n</think>\n\n\n  Answer with spaces.  \n",
    "Thinking with spaces.  \n\n", "Answer with spaces.  \n")

add("thinking-off-effort-none", "reasoning_effort none -> enable_thinking false",
    {"messages": [{"role": "user", "content": "Say hi."}], "reasoning_effort": "none"},
    "Hi.", "", "Hi.")

add("thinking-off-kwarg", "chat_template_kwargs enable_thinking false, no system",
    {"messages": [{"role": "user", "content": "Say hi."}], "chat_template_kwargs": {"enable_thinking": False}},
    "Hi there.", "", "Hi there.")

add("effort-low", "top-level reasoning_effort low, adds the instruction system block",
    {"messages": [SYS, {"role": "user", "content": "2+2?"}], "reasoning_effort": "low"},
    "Easy.\n</think>\n\n4", "Easy.\n", "4")

add("effort-xhigh-kwarg", "chat_template_kwargs reasoning_effort xhigh",
    {"messages": [{"role": "user", "content": "2+2?"}], "chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    "Simple arithmetic.\n</think>\n\nThe answer is 4.", "Simple arithmetic.\n", "The answer is 4.")

add("developer-role", "developer message becomes system",
    {"messages": [{"role": "developer", "content": "Answer in French."}, {"role": "user", "content": "Hello"}]},
    "French greeting.\n</think>\n\nBonjour !", "French greeting.\n", "Bonjour !")

add("tool-1-call", "tools with several parameter types, one call, no content",
    {"messages": [SYS, {"role": "user", "content": "Weather in Barcelona for 3 days?"}], "tools": TOOLS},
    "I need the weather tool.\n</think>\n\n" + tc_text("get_weather", [("city", "Barcelona"), ("days", "3")]),
    "I need the weather tool.\n", "", [expect_call("get_weather", {"city": "Barcelona", "days": 3})])

add("tool-3-calls", "content before 3 parallel calls; multi-line string, number, bool, array, object, null",
    {"messages": [{"role": "user", "content": "Do three things."}], "tools": TOOLS},
    "Three calls needed.\n</think>\n\nI'll run these now.\n\n"
    + tc_text("write_file", [("path", "C:\\tmp\\a.py"), ("content", "def f():\n    return \"é\"\n"),
                             ("overwrite", "true")]) + "\n"
    + tc_text("search", [("query", "123"), ("limit", "2.50"), ("tags", "[\"a\", \"b\"]"),
                         ("filters", "{\"year\": 2024, \"lang\": \"ca\"}"), ("cursor", "null")]) + "\n"
    + tc_text("bash", [("command", "ls -la | grep \"x\""), ("timeout", "1000")]),
    "Three calls needed.\n", "I'll run these now.\n\n",
    [expect_call("write_file", {"path": "C:\\tmp\\a.py", "content": "def f():\n    return \"é\"\n",
                                "overwrite": True}),
     expect_call("search", '{"query":"123","limit":2.50,"tags":["a", "b"],"filters":{"year": 2024, "lang": "ca"},'
                           '"cursor":null}'),
     expect_call("bash", {"command": "ls -la | grep \"x\"", "timeout": 1000})])

add("tool-2-calls-unicode", "2 calls, unicode values, optional parameter after the required one",
    {"messages": [{"role": "user", "content": "Tiempo en Zürich y 北京"}], "tools": TOOLS},
    "Two cities.\n</think>\n\n"
    + tc_text("get_weather", [("city", "Zürich"), ("units", "metric")]) + "\n"
    + tc_text("get_weather", [("city", "北京 🌧"), ("days", "2")]),
    "Two cities.\n", "",
    [expect_call("get_weather", {"city": "Zürich", "units": "metric"}),
     expect_call("get_weather", {"city": "北京 🌧", "days": 2})])

add("tool-call-in-reasoning", "tool call starts before </think>",
    {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS},
    "I should list the files.\n" + tc_text("bash", [("command", "ls")]),
    "I should list the files.\n", "", [expect_call("bash", {"command": "ls"})])

add("history-tools", "assistant history with reasoning_content and tool_calls, tool results, new user turn",
    {"messages": [
        SYS,
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": "", "reasoning_content": "  Two lookups.  \n",
         "tool_calls": [call("get_weather", {"city": "Paris"}, "call_a"),
                        call("get_weather", {"city": "Rome", "days": 2}, "call_b")]},
        {"role": "tool", "tool_call_id": "call_a", "name": "get_weather", "content": "Paris: 18C, sunny"},
        {"role": "tool", "tool_call_id": "call_b", "name": "get_weather", "content": "Rome: 24C, clear"},
        {"role": "assistant", "content": "Paris is 18C and sunny; Rome is 24C and clear.",
         "reasoning_content": "Both results are in."},
        {"role": "user", "content": "Thanks! And Oslo?"}],
     "tools": TOOLS},
    "One more lookup.\n</think>\n\n" + tc_text("get_weather", [("city", "Oslo")]),
    "One more lookup.\n", "", [expect_call("get_weather", {"city": "Oslo"})])

add("history-after-tool", "last message is a tool result (agent loop), arguments given as objects",
    {"messages": [
        {"role": "user", "content": "Run ls"},
        {"role": "assistant", "content": "Running it.", "reasoning_content": "Use bash.",
         "tool_calls": [call("bash", {"command": "ls", "timeout": 5000}, "call_1", as_string=False)]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}],
     "tools": TOOLS},
    "Two files.\n</think>\n\nThere are two files: a.txt and b.txt.",
    "Two files.\n", "There are two files: a.txt and b.txt.")

add("typed-content", "user content as an array of text parts (joined without separator)",
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one."},
                                               {"type": "text", "text": "Part two."}]}]},
    "Two parts.\n</think>\n\nGot both parts.", "Two parts.\n", "Got both parts.")

add("tool-choice-none", "tools present but tool_choice none: tool-call text stays in content",
    {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "tool_choice": "none"},
    "No tools allowed.\n</think>\n\n" + tc_text("get_weather", [("city", "Paris")]),
    "No tools allowed.\n", tc_text("get_weather", [("city", "Paris")]))

add("tool-extra-keys", "tool with strict and a tool without description or parameters",
    {"messages": [{"role": "user", "content": "Ping"}],
     "tools": [{"type": "function", "function": {"name": "ping", "strict": True,
                                                 "description": "Ping the server.",
                                                 "parameters": {"type": "object", "properties": {},
                                                                "additionalProperties": False}}},
               {"type": "function", "function": {"name": "noop"}}]},
    "Call ping.\n</think>\n\n" + tc_text("ping", []),
    "Call ping.\n", "", [expect_call("ping", {})],
    fail="prompt: llama.cpp rebuilds each tool as {type, function: {name, description, parameters}}")

add("length-cut", "generation stopped inside the reasoning (finish_reason length)",
    {"messages": [{"role": "user", "content": "Write a long essay."}]},
    "Let me plan the essay carefully. First",
    "Let me plan the essay carefully. First", "", finish="length")

add("second-system", "a second system message in the middle",
    {"messages": [SYS, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                  {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Bye"}]},
    "Short.\n</think>\n\nBye.", "Short.\n", "Bye.",
    fail="llama.cpp rejects the request (the template raises), like jinja2")

add("assistant-prefill", "last message is assistant",
    {"messages": [{"role": "user", "content": "Count to 3."}, {"role": "assistant", "content": "1, 2,"}]},
    " 3.", "", "1, 2, 3.",
    fail="prompt: llama-server continues the last assistant message (no <|im_end|>, no new turn)")

add("optional-before-required", "optional parameter written before the required one",
    {"messages": [{"role": "user", "content": "Weather in Zurich, metric"}], "tools": TOOLS},
    "One call.\n</think>\n\n" + tc_text("get_weather", [("units", "metric"), ("city", "Zurich")]),
    "One call.\n", "")  # llama.cpp drops the call and the text after the reasoning

add("text-after-calls", "text after the last </tool_call>",
    {"messages": [{"role": "user", "content": "Weather in Oslo"}], "tools": TOOLS},
    "One call.\n</think>\n\n" + tc_text("get_weather", [("city", "Oslo")]) + "\nDone.",
    "One call.\n", "", [expect_call("get_weather", {"city": "Oslo"})])  # llama.cpp drops "Done."

add("thinking-off-tool", "thinking off with tools, content before the call",
    {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS, "reasoning_effort": "none"},
    "Calling.\n\n" + tc_text("bash", [("command", "ls")]),
    "", "Calling.\n\n", [expect_call("bash", {"command": "ls"})])

add("empty-parse", "raw is only a newline: the parse is empty and llama-server sends the raw text",
    {"messages": [{"role": "user", "content": "Say nothing."}]},
    "\n", "", "\n")

add("args-text-normalized", "expected arguments re-serialized (2.5) while the model wrote 2.50",
    {"messages": [{"role": "user", "content": "Search"}], "tools": TOOLS},
    "Search.\n</think>\n\n" + tc_text("search", [("query", "x"), ("limit", "2.50")]),
    "Search.\n", "", [expect_call("search", {"query": "x", "limit": 2.5})],
    fail="parse: same JSON value, different arguments text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if (f.startswith("req-") or f.startswith("res-")) and f.endswith(".json"):
            os.remove(os.path.join(args.out, f))
    tmpl = make_template()
    index = []
    for i, (name, note, body, raw, reasoning, content, calls, finish, fail) in enumerate(FIXTURES, 1):
        req = {"request": body, "prompt": render(tmpl, body)}
        res = {"raw": raw, "reasoning_content": reasoning or None, "content": content, "tool_calls": calls,
               "finish_reason": finish, "timings": {}}
        with open(os.path.join(args.out, f"req-{i:05d}.json"), "w", encoding="utf-8", newline="") as f:
            json.dump(req, f, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, f"res-{i:05d}.json"), "w", encoding="utf-8", newline="") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        index.append(f"{i:05d}  {'expect FAIL' if fail else 'expect PASS'}  {name}: {note}"
                     + (f" [{fail}]" if fail else ""))
    with open(os.path.join(args.out, "index.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index) + "\n")
    print(f"wrote {len(FIXTURES)} pairs to {args.out}")
    print("\n".join(index))


if __name__ == "__main__":
    main()
