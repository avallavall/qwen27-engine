# Oracle for src/chat_template.cpp: render many conversations with jinja2 and write inputs + expected outputs.
#
# Usage (project venv):
#   .venv\Scripts\python.exe tools\gen_template_fixtures.py [--gguf PATH] [--out bench\out\template_fixtures.json]
#
# The template text is read from the model GGUF (tokenizer.chat_template) and must equal
# research/_chat_template.jinja. jinja2 settings as HF transformers / llama.cpp: trim_blocks, lstrip_blocks,
# raise_exception() raises, tojson = json.dumps(x, ensure_ascii=False). Every fixture is also rendered with
# HF's ImmutableSandboxedEnvironment + loopcontrols, and the two results must agree.
#
# Fixture record: {name, normalize, messages, tools, options, expect}. expect is one of
#   {"output": text}   the rendered prompt
#   {"raise": msg}     the template called raise_exception(msg)
#   {"error": msg}     another jinja2 error (TypeError, UndefinedError): the C++ side must throw too.
# With normalize = true, both sides first apply the llama.cpp message fixes (developer -> system, string
# tool-call arguments -> parsed JSON, content null -> "").
import argparse
import copy
import hashlib
import json
import os
import random
import struct
import sys

import jinja2
import jinja2.sandbox
from gguf import GGUF_MAGIC, GGUFValueType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GGUF = os.environ.get("Q27_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qwen38_27",
                                                  "models", "Qwen3.8-27B", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
DEFAULT_OUT = os.path.join(ROOT, "bench", "out", "template_fixtures.json")


# ---------------------------------------------------------------- GGUF

def read_gguf_string(path, key):
    """One string value from the GGUF metadata. Arrays are skipped without decoding (GGUFReader takes minutes
    on this file because it decodes the 248k-entry vocab arrays)."""
    fixed = {GGUFValueType.UINT8: 1, GGUFValueType.INT8: 1, GGUFValueType.UINT16: 2, GGUFValueType.INT16: 2,
             GGUFValueType.UINT32: 4, GGUFValueType.INT32: 4, GGUFValueType.FLOAT32: 4, GGUFValueType.BOOL: 1,
             GGUFValueType.UINT64: 8, GGUFValueType.INT64: 8, GGUFValueType.FLOAT64: 8}
    with open(path, "rb") as f:
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def skip(t):
            if t == GGUFValueType.STRING:
                f.seek(u64(), 1)
            elif t == GGUFValueType.ARRAY:
                et, n = GGUFValueType(u32()), u64()
                if et in fixed:
                    f.seek(fixed[et] * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                f.seek(fixed[t], 1)

        if u32() != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version = u32()
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported")
        u64()  # tensor count
        n_kv = u64()
        for _ in range(n_kv):
            k = f.read(u64()).decode("utf-8")
            t = GGUFValueType(u32())
            if k == key:
                if t != GGUFValueType.STRING:
                    raise ValueError(f"{key} is not a string")
                return f.read(u64())
            skip(t)
    raise KeyError(key)


# ---------------------------------------------------------------- jinja2, HF style

class TemplateError(Exception):
    pass


def raise_exception(msg):
    raise TemplateError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_template(text, sandboxed):
    if sandboxed:
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])
    else:
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = tojson
    return env.from_string(text)


def normalize(messages):
    """Same as q27::normalize_messages."""
    out = copy.deepcopy(messages)
    if not isinstance(out, list):
        return out
    for m in out:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "developer":
            m["role"] = "system"
        if "content" in m and m["content"] is None:
            m["content"] = ""
        calls = m.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tc in calls:
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                continue
            fn = tc["function"]
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
    return out


def render(tmpl, fx):
    messages = normalize(fx["messages"]) if fx["normalize"] else fx["messages"]
    opts = fx["options"]
    ctx = {"messages": messages, "tools": fx["tools"],
           "add_generation_prompt": opts.get("add_generation_prompt", True)}
    for k in ("add_vision_id", "enable_thinking", "preserve_thinking", "reasoning_effort"):
        if k in opts:
            ctx[k] = opts[k]
    try:
        return {"output": tmpl.render(**ctx)}
    except TemplateError as e:
        return {"raise": str(e)}
    except Exception as e:  # noqa: BLE001 - any other jinja2 failure
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- tool schemas

SCHEMA = "http://json-schema.org/draft-07/schema#"


def tool(name, desc, props, required, strict=None, extra=None):
    params = {"type": "object", "properties": props, "required": required, "additionalProperties": False,
              "$schema": SCHEMA}
    if extra:
        params.update(extra)
    fn = {"name": name, "description": desc, "parameters": params}
    if strict is not None:
        fn["strict"] = strict
    return {"type": "function", "function": fn}


T_BASH = tool(
    "bash",
    "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper "
    "handling and security measures.\n\nBefore executing the command, please follow these steps:\n\n"
    "1. Directory Verification:\n   - If the command will create new directories or files, first use the List "
    "tool to verify the parent directory exists and is the correct location\n\nUsage notes:\n"
    "  - The command argument is required.\n  - You can specify an optional timeout in milliseconds (up to "
    "600000ms / 10 minutes).\n  - Quote paths with spaces: cd \"/Users/name/My Documents\" (correct) vs "
    "cd /Users/name/My Documents (incorrect)\n  - Use '&&' or ';' to chain commands; avoid `cd <dir> && <cmd>`.",
    {"command": {"type": "string", "description": "The command to execute"},
     "timeout": {"type": "number", "description": "Optional timeout in milliseconds"},
     "workdir": {"type": "string", "description": "The working directory. Defaults to C:\\Users\\dev\\proj."},
     "description": {"type": "string", "description": "Clear, concise description of what this command does in "
                     "5-10 words. Examples:\nInput: ls\nOutput: Lists files in current directory"}},
    ["command", "description"])
T_EDIT = tool(
    "edit",
    "Performs exact string replacements in files.\n\nUsage:\n- You must use your `Read` tool at least once "
    "before editing.\n- The edit will FAIL if `oldString` is not found in the file.",
    {"filePath": {"type": "string", "description": "The absolute path to the file to modify"},
     "oldString": {"type": "string", "description": "The text to replace"},
     "newString": {"type": "string", "description": "The text to replace it with (must be different from "
                   "oldString)"},
     "replaceAll": {"type": "boolean", "default": False, "description": "Replace all occurrences of oldString"}},
    ["filePath", "oldString", "newString"])
T_GLOB = tool(
    "glob", "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like "
    "\"**/*.js\" or \"src/**/*.ts\".",
    {"pattern": {"type": "string", "description": "The glob pattern to match files against"},
     "path": {"type": "string", "description": "The directory to search in. If not specified, the current "
              "working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT "
              "enter \"undefined\" or \"null\""}},
    ["pattern"])
T_GREP = tool(
    "grep", "Fast content search tool. Searches file contents using regular expressions (eg. \"log.*Error\", "
    "\"function\\s+\\w+\").",
    {"pattern": {"type": "string", "description": "The regex pattern to search for in file contents"},
     "path": {"type": "string", "description": "The directory to search in."},
     "include": {"type": "string", "description": "File pattern to include in the search (e.g. \"*.js\", "
                 "\"*.{ts,tsx}\")"}},
    ["pattern"])
T_LIST = tool(
    "list", "Lists files and directories in a given path.",
    {"path": {"type": "string", "description": "The absolute path to the directory to list"},
     "ignore": {"type": "array", "items": {"type": "string"}, "description": "List of glob patterns to ignore"}},
    [])
T_READ = tool(
    "read", "Reads a file from the local filesystem. Lines longer than 2000 characters will be truncated. "
    "Results are returned using cat -n format, with line numbers starting at 1.",
    {"filePath": {"type": "string", "description": "The path to the file to read"},
     "offset": {"type": "number", "description": "The line number to start reading from (0-based)"},
     "limit": {"type": "number", "description": "The number of lines to read (defaults to 2000)"}},
    ["filePath"])
T_TASK = tool(
    "task", "Launch a new agent to handle complex, multi-step tasks autonomously.",
    {"description": {"type": "string", "description": "A short (3-5 words) description of the task"},
     "prompt": {"type": "string", "description": "The task for the agent to perform"},
     "subagent_type": {"type": "string", "enum": ["general", "explore", "review"],
                       "description": "The type of specialized agent to use for this task"},
     "options": {"anyOf": [{"type": "object", "properties": {"maxSteps": {"type": "integer", "minimum": 1,
                                                                            "maximum": 50},
                                                               "temperature": {"type": "number", "minimum": 0.0,
                                                                               "maximum": 2.0, "default": 0.7}},
                            "additionalProperties": False},
                           {"type": "null"}]}},
    ["description", "prompt", "subagent_type"])
T_TODOWRITE = tool(
    "todowrite", "Use this tool to create and manage a structured task list for your current coding session.",
    {"todos": {"type": "array", "description": "The updated todo list",
               "items": {"type": "object",
                         "properties": {"content": {"type": "string", "minLength": 1,
                                                    "description": "Brief description of the task"},
                                        "status": {"type": "string",
                                                   "enum": ["pending", "in_progress", "completed", "cancelled"]},
                                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                        "id": {"type": "string"}},
                         "required": ["content", "status", "priority", "id"], "additionalProperties": False}}},
    ["todos"])
T_WEBFETCH = tool(
    "webfetch", "Fetches content from a specified URL (https://example.com/path?a=1&b=2) and returns it as "
    "<text>, <markdown> or <html>. Don't use it for 'localhost'.",
    {"url": {"type": "string", "format": "uri", "description": "The URL to fetch content from"},
     "format": {"type": "string", "enum": ["text", "markdown", "html"], "default": "markdown"},
     "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 120, "description": "Optional timeout in "
                 "seconds (max 120)"}},
    ["url", "format"])
T_WRITE = tool(
    "write", "Writes a file to the local filesystem. This tool will overwrite the existing file if there is one "
    "at the provided path.",
    {"content": {"type": "string", "description": "The content to write to the file"},
     "filePath": {"type": "string", "description": "The absolute path to the file to write (must be absolute, "
                  "not relative)"}},
    ["content", "filePath"])

OPENCODE_TOOLS = sorted([T_WRITE, T_READ, T_BASH, T_EDIT, T_GLOB, T_GREP, T_LIST, T_TASK, T_TODOWRITE, T_WEBFETCH],
                        key=lambda t: t["function"]["name"])
BASIC_TOOLS = [T_READ, T_BASH]

# pi-ai / dsh tools: strict: false, simpler schemas, one tool returns images.
PI_TOOLS = [
    tool("read", "Read the contents of a file. Supports text files and images (jpg, png, gif, webp).",
         {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
          "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
          "limit": {"type": "number", "description": "Maximum number of lines to read"}}, ["path"], strict=False),
    tool("bash", "Execute a bash command in the current working directory. Returns stdout and stderr.",
         {"command": {"type": "string", "description": "Bash command to execute"},
          "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}},
         ["command"], strict=False),
    tool("edit", "Edit a file by replacing exact text. The oldText must match exactly (including whitespace).",
         {"path": {"type": "string"}, "oldText": {"type": "string"}, "newText": {"type": "string"}},
         ["path", "oldText", "newText"], strict=False),
    tool("screenshot", "Capture the screen and return it as an image.", {}, [], strict=False),
]

# Unicode and odd characters in the schema: must stay raw (ensure_ascii=False), controls escaped.
T_UNICODE = tool(
    "cercar_fitxer", "Cerca un fitxer pel nom. Búsqueda «rápida» — 搜索文件 🔍 — Ελληνικά — עברית — "
    "café ñ ü ß. Tab:\there. Ctrl-A:\x01. DEL:\x7f. Bell:\x07. Esc:\x1b[0m. Line sep:\u2028. "
    "NBSP:\u00a0. ZWSP:\u200b. Backslash: C:\\tmp\\x. Quotes: \"q\" and 'q'. HTML: <a href=\"x\">&amp;</a>. "
    "Slash: a/b/c. Emoji with ZWJ: 👩‍💻. Astral: 𝄞.",
    {"nom": {"type": "string", "description": "Nom del fitxer (p. ex. «informe_2026.pdf»)"},
     "carpeta": {"type": "string", "description": "目录 / carpeta", "default": "~/Documents"},
     "\u00e0ccents": {"type": "boolean", "description": "Clau amb accent"}},
    ["nom"])
T_FLOATS = tool(
    "set_params", "Numbers of every kind in the schema.",
    {"temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 1.0, "multipleOf": 1e-05},
     "top_p": {"type": "number", "default": 0.95, "exclusiveMaximum": 1e20},
     "big": {"type": "number", "examples": [1e16, 1e15, 123456789012345678.0, 1.7976931348623157e308, 5e-324,
                                            0.0001, 0.00001, -0.0, 0.1 + 0.2, 100.0, 2.5e-10, 1.5e-07, 3.14159]},
     "ints": {"type": "integer", "examples": [0, -1, 42, 9007199254740993, -9223372036854775808,
                                              18446744073709551615]},
     "flags": {"type": "array", "items": {"type": ["boolean", "null"]}, "default": [True, False, None]},
     "empty": {"type": "object", "properties": {}, "default": {}},
     "empty_list": {"type": "array", "default": []}},
    ["temperature"])


# ---------------------------------------------------------------- message helpers

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
                                                 "FcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
IMG_URL = {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg", "detail": "high"}}
IMG_QWEN = {"type": "image", "image": "file:///tmp/a.png"}
VIDEO = {"type": "video", "video": "file:///tmp/v.mp4"}


def text(t):
    return {"type": "text", "text": t}


def sysm(c):
    return {"role": "system", "content": c}


def dev(c):
    return {"role": "developer", "content": c}


def user(c):
    return {"role": "user", "content": c}


def asst(content="", reasoning=None, calls=None, **extra):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = calls
    m.update(extra)
    return m


def toolmsg(c, cid="call_0"):
    return {"role": "tool", "tool_call_id": cid, "content": c}


def call(name, args, cid="call_0", as_string=False):
    """OpenAI tool call. as_string: arguments as compact JSON text (what OpenCode / pi-ai send)."""
    if as_string:
        args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


# ---------------------------------------------------------------- fixtures

FIXTURES = []


def fx(name, messages, tools=None, normalize=False, **options):
    FIXTURES.append({"name": name, "normalize": normalize, "messages": messages, "tools": tools,
                     "options": options})


OPENCODE_SYSTEM = (
    "You are opencode, an interactive CLI tool that helps users with software engineering tasks.\n\n"
    "# Tone and style\nYou should be concise, direct, and to the point.\n\n"
    "<env>\n  Working directory: C:\\Users\\dev\\projects\\demo\n  Is directory a git repo: yes\n"
    "  Platform: win32\n  Today's date: Tue Oct 06 2026\n</env>\n"
    "<project>\n  src/\n    parser.ts\n    parser.test.ts\n</project>\n")

HELLO = [user("Hello!")]

# --- basic shapes
fx("no_system", HELLO)
fx("no_system_no_genprompt", HELLO, add_generation_prompt=False)
fx("system_no_tools", [sysm("You are a helpful assistant."), user("Hi")])
fx("system_with_tools", [sysm("You are a helpful assistant."), user("List the files.")], BASIC_TOOLS)
fx("tools_no_system", [user("List the files.")], BASIC_TOOLS)
fx("tools_empty_list", [sysm("S"), user("Hi")], [])
fx("tools_mapping_ignored", [sysm("S"), user("Hi")], {"type": "function", "function": {"name": "x"}})
fx("tools_number_ignored", [user("Hi")], 5)
fx("tools_string_iterated", [user("Hi")], "a\u00e9\n")
fx("tools_opencode_sorted", [sysm(OPENCODE_SYSTEM), user("hi")], OPENCODE_TOOLS)
fx("tools_pi_strict_false", [dev("You are pi."), user("hi")], PI_TOOLS, normalize=True)
fx("tools_unicode_schema", [user("Cerca «informe»")], [T_UNICODE])
fx("tools_float_schema", [user("Set params")], [T_FLOATS, T_UNICODE])
fx("tools_non_object_items", [user("x")], [1, "two", None, True, 2.5, [1, {"a": "b"}]])
fx("system_content_parts", [sysm([text("Part one. "), text(" Part two.")]), user("Hi")])
fx("system_whitespace_only", [sysm("  \n\t "), user("Hi")])
fx("system_whitespace_only_tools", [sysm("  \n\t "), user("Hi")], BASIC_TOOLS)
fx("system_empty_medium", [sysm(""), user("Hi")], reasoning_effort="medium")
fx("system_null_medium", [sysm(None), user("Hi")], reasoning_effort="medium")
fx("system_null_normalized", [sysm(None), user("Hi")], normalize=True)
fx("system_missing_content", [{"role": "system"}, user("Hi")])
fx("system_padded", [sysm("\n\n  Be brief.  \n"), user("  Hi  ")])

# --- reasoning effort x thinking
for eff in (None, "xhigh", "medium", "low"):
    kw = {} if eff is None else {"reasoning_effort": eff}
    tag = eff or "unset"
    fx(f"effort_{tag}_nosys", HELLO, **kw)
    fx(f"effort_{tag}_sys", [sysm("Sys."), user("Hi")], **kw)
    fx(f"effort_{tag}_tools", [sysm("Sys."), user("Hi")], BASIC_TOOLS, **kw)
    fx(f"effort_{tag}_tools_nosys", HELLO, BASIC_TOOLS, **kw)
    for et in (False, True):
        fx(f"effort_{tag}_thinking_{str(et).lower()}", [sysm("Sys."), user("Hi")], enable_thinking=et, **kw)
for bad in ("high", "", "XHIGH", "none", "max", "minimal", " low"):
    fx(f"effort_bad_{bad.strip() or 'empty'}{'_space' if bad.startswith(' ') else ''}", HELLO, reasoning_effort=bad)
fx("effort_high_thinking_off", HELLO, reasoning_effort="high", enable_thinking=False)
fx("effort_high_thinking_on", HELLO, reasoning_effort="high", enable_thinking=True)
fx("thinking_off_tools", [sysm("Sys."), user("Hi")], BASIC_TOOLS, enable_thinking=False)
fx("thinking_off_nosys", HELLO, enable_thinking=False)
fx("thinking_off_no_genprompt", HELLO, enable_thinking=False, add_generation_prompt=False)

# --- assistant turns, reasoning, preserve_thinking
MULTI = [
    sysm("Sys."),
    user("Q1"),
    asst("A1", reasoning="  \n\nthinking one\n\n  "),
    user("Q2"),
    asst("A2", reasoning="thinking two"),
    user("Q3"),
    asst("\n  A3 padded  \n", reasoning="\nthinking three\n"),
]
for pt in (None, False, True):
    kw = {} if pt is None else {"preserve_thinking": pt}
    tag = "unset" if pt is None else str(pt).lower()
    fx(f"preserve_{tag}_multi", MULTI, **kw)
    fx(f"preserve_{tag}_multi_then_user", MULTI + [user("Q4")], **kw)
    fx(f"preserve_{tag}_multi_no_genprompt", MULTI, add_generation_prompt=False, **kw)
fx("reasoning_nonstring", [user("Q"), asst("A", reasoning=None), user("Q2"), asst("B", reasoning=42),
                           user("Q3"), asst("C", reasoning=["x"]), user("Q4")])
fx("reasoning_missing", [user("Q"), asst("A"), user("Q2")])
fx("reasoning_unicode_ws", [user("Q"), asst("\u3000A\u00a0", reasoning="\u2028\u0085thought\u2003\u200b"),
                            user("Q2")])
fx("content_unicode_ws", [user("\u00a0\u1680\u2000Q\u205f\u3000\x1c\x1f\x0b\x0c"), asst("\u180eA\ufeff")])
fx("content_null_assistant", [user("Q"), asst(None, reasoning="r"), user("Q2")])
fx("content_null_assistant_normalized", [user("Q"), asst(None, reasoning="r"), user("Q2")], normalize=True)
fx("content_empty_assistant", [user("Q"), asst("", reasoning=""), user("Q2")])
fx("content_missing_user", [{"role": "user"}, user("Q2")])
fx("assistant_last_prefill", [user("Q"), asst("Partial answer", reasoning="r")], add_generation_prompt=False)
fx("assistant_content_parts", [user("Q"), asst([text("Part A "), text("and B")], reasoning="r"), user("Q2")])

# --- tool calls
fx("toolcall_one", [user("Read a.py"), asst("", reasoning="need to read",
                                           calls=[call("read", {"filePath": "/w/a.py"})]),
                    toolmsg("print('hi')")], BASIC_TOOLS)
fx("toolcall_content_before", [user("Read a.py"), asst("Let me read the file.", reasoning="r",
                                                      calls=[call("read", {"filePath": "/w/a.py"})]),
                               toolmsg("x = 1")], BASIC_TOOLS)
fx("toolcall_content_ws_before", [user("Read a.py"), asst("  \n ", calls=[call("read", {"filePath": "a"})]),
                                  toolmsg("x")], BASIC_TOOLS)
fx("toolcall_two", [user("Read both"), asst("", calls=[call("read", {"filePath": "a"}, "c1"),
                                                     call("read", {"filePath": "b"}, "c2")]),
                    toolmsg("A", "c1"), toolmsg("B", "c2")], BASIC_TOOLS)
fx("toolcall_three_content", [user("Do three things"),
                              asst("Running three tools.", calls=[call("read", {"filePath": "a"}, "c1"),
                                                                  call("bash", {"command": "ls", "description": "List"}, "c2"),
                                                                  call("read", {"filePath": "b", "limit": 10}, "c3")]),
                              toolmsg("A", "c1"), toolmsg("ls output", "c2"), toolmsg("B", "c3"),
                              asst("Done.", reasoning="all good")], BASIC_TOOLS)
ARG_KINDS = {
    "s": "plain string", "s_empty": "", "s_multiline": "line1\n\tline2\n  line3 \\n literal \"q\" 'q'",
    "s_unicode": "héllo 世界 🎉 \u2028", "s_jsonish": "{\"a\": 1}", "s_ws": "  padded  ",
    "i": 42, "i_neg": -7, "i_zero": 0, "i_big": 9007199254740993, "i_u64": 18446744073709551615,
    "f": 3.14, "f_one": 1.0, "f_small": 1e-05, "f_tiny": 0.0001, "f_big": 1e20, "f_1e16": 1e16,
    "f_1e15": 1e15, "f_negzero": -0.0, "f_sum": 0.1 + 0.2, "f_neg": -2.5e-10, "f_max": 1.7976931348623157e308,
    "b_true": True, "b_false": False, "n": None,
    "arr": [1, "two", 3.0, None, True, [], {}], "arr_empty": [],
    "obj": {"nested": {"k": "v\n\"x\"", "list": [1.5, {"deep": "ñ"}]}, "z": None}, "obj_empty": {},
}
fx("toolcall_arg_types", [user("Call with everything"), asst("", calls=[call("everything", ARG_KINDS)])])
fx("toolcall_arg_types_string_normalized", [user("Call with everything"),
                                            asst("", calls=[call("everything", ARG_KINDS, as_string=True)])],
   normalize=True)
fx("toolcall_args_empty_object", [user("x"), asst("", calls=[call("screenshot", {})]), toolmsg("ok")])
fx("toolcall_args_empty_string", [user("x"), asst("", calls=[call("screenshot", "")]), toolmsg("ok")])
fx("toolcall_args_empty_string_normalized", [user("x"), asst("", calls=[call("screenshot", "")]), toolmsg("ok")],
   normalize=True)
fx("toolcall_args_missing", [user("x"), asst("", calls=[{"id": "c", "type": "function",
                                                          "function": {"name": "screenshot"}}]), toolmsg("ok")])
fx("toolcall_no_function_wrapper", [user("x"), asst("", calls=[{"name": "read", "arguments": {"filePath": "a"}}]),
                                    toolmsg("ok")])
fx("toolcall_args_string_braces_normalized", [user("x"), asst("", calls=[call("read", "{}")]), toolmsg("ok")],
   normalize=True)
fx("toolcall_args_ws_json_normalized", [user("x"), asst("", calls=[call("read", " \n{\"filePath\": \"a\"} \n")]),
                                        toolmsg("ok")], normalize=True)
fx("toolcall_args_json_empty_string_normalized", [user("x"), asst("", calls=[call("read", "\"\"")]),
                                                  toolmsg("ok")], normalize=True)
fx("toolcall_empty_list", [user("x"), asst("Nothing to call.", calls=[])])
fx("toolcall_null", [user("x"), asst("Nothing.", calls=None)])
fx("toolcall_mapping_ignored", [user("x"), asst("Mapping.", calls={"name": "read"})])
fx("toolcall_reasoning_dropped", [user("Q1"), asst("", reasoning="old", calls=[call("read", {"filePath": "a"})]),
                                  toolmsg("A"), asst("Answer 1", reasoning="old 2"), user("Q2")],
   BASIC_TOOLS, preserve_thinking=False)
fx("toolcall_unicode_names", [user("x"), asst("", calls=[call("cercar_fitxer", {"nom": "«informe».pdf",
                                                                                "\u00e0ccents": True})]),
                              toolmsg("Trobat: C:\\Users\\Àlex\\informe.pdf")], [T_UNICODE])

# --- tool messages
fx("tool_consecutive_three", [user("x"), asst("", calls=[call("a", {}, "1"), call("b", {}, "2"), call("c", {}, "3")]),
                              toolmsg("r1", "1"), toolmsg("r2", "2"), toolmsg("r3", "3"), asst("done")])
fx("tool_last_no_genprompt", [user("x"), asst("", calls=[call("a", {})]), toolmsg("r")], add_generation_prompt=False)
fx("tool_then_user", [user("x"), asst("", calls=[call("a", {})]), toolmsg("r"), user("and now?")])
fx("tool_first_message", [toolmsg("orphan result"), user("x")])
fx("tool_content_padded", [user("x"), asst("", calls=[call("a", {})]), toolmsg("\n\n  result with spaces \n")])
fx("tool_content_null", [user("x"), asst("", calls=[call("a", {})]), toolmsg(None)])
fx("tool_content_parts", [user("x"), asst("", calls=[call("a", {})]), toolmsg([text("part1 "), text("part2")])])
fx("tool_content_image", [user("x"), asst("", calls=[call("screenshot", {})]), toolmsg([text("Screen:"), IMG])])
fx("user_tool_response_only", [user("Real question"), asst("", calls=[call("a", {})]),
                               user("<tool_response>\nresult\n</tool_response>"), asst("final", reasoning="r2")],
   preserve_thinking=False)
fx("user_tool_response_only_padded", [user("Real question"), asst("A", reasoning="r1"),
                                      user("  <tool_response>x</tool_response>\n"), asst("B", reasoning="r2")],
   preserve_thinking=False)
fx("user_tool_response_parts", [user("Real"), asst("A", reasoning="r1"),
                                user([text("<tool_response>"), IMG, text("</tool_response>")]),
                                asst("B", reasoning="r2")], preserve_thinking=False)
fx("user_tool_response_prefix_only", [user("Q"), asst("A", reasoning="r1"),
                                      user("<tool_response>x</tool_response> and more"), asst("B", reasoning="r2")],
   preserve_thinking=False)

# --- images and video
fx("image_one", [user([IMG, text("What is in this image?")])])
fx("image_several", [user([text("Compare "), IMG, text(" and "), IMG_URL, text(" and "), IMG_QWEN])])
fx("image_several_vision_id", [user([IMG, IMG_URL]), asst("Two images."), user([text("And this:"), IMG_QWEN, VIDEO])],
   add_vision_id=True)
fx("image_vision_id_false", [user([IMG, VIDEO])], add_vision_id=False)
fx("image_in_assistant_and_tool", [user([IMG]), asst([text("see "), IMG]), user("x"),
                                   asst("", calls=[call("screenshot", {})]), toolmsg([IMG, text("shot")]),
                                   user([IMG])], add_vision_id=True)
fx("image_type_only", [user([{"type": "image"}, {"type": "video"}])])
fx("image_key_without_type", [user([{"image_url": "x"}, {"image": "y"}, {"video": "z"}, {"text": "t"}])])
fx("video_one", [user([VIDEO, text("Describe the video.")])])
fx("content_string_parts", [user(["some text", "an image here", "a video", "image_url"])])
fx("content_list_parts", [user([["image"], ["video"], ["text"], ["text", "image"]])])
fx("text_part_non_string", [user([{"type": "text", "text": 5}, {"type": "text", "text": None},
                                  {"type": "text", "text": True}, {"type": "text", "text": 2.5},
                                  {"type": "text", "text": 1e20}, {"type": "text", "text": ["a", 1, None, False]},
                                  {"type": "text", "text": {"k": "it's", "q": "say \"hi\"", "b": "both ' \""}},
                                  {"type": "text", "text": ["é", "\u00a0", "\u200b", "tab\there", "\x01"]}])])
fx("pi_attached_images", [
    dev("You are a coding agent."),
    user("Take a screenshot and tell me what you see."),
    asst("", reasoning="I'll take a screenshot.", calls=[call("screenshot", "{}", "call_s1")]),
    toolmsg("Screenshot captured (1 image).", "call_s1"),
    user([text("Attached image(s) from tool result:"), IMG]),
    asst("I see a white square.", reasoning="The image is a 1x1 pixel."),
    user("Now read two images."),
    asst("", calls=[call("read", "{\"path\":\"a.png\"}", "c1"), call("read", "{\"path\":\"b.png\"}", "c2")]),
    toolmsg("Read image file [image/png]", "c1"), toolmsg("Read image file [image/png]", "c2"),
    user([text("Attached image(s) from tool result:"), IMG, IMG]),
], PI_TOOLS, normalize=True, enable_thinking=True, preserve_thinking=True)
fx("pi_attached_images_preserve_false", FIXTURES[-1]["messages"], PI_TOOLS, normalize=True, preserve_thinking=False)

# --- developer role
fx("developer_normalized", [dev("Dev prompt."), user("Hi")], normalize=True)
fx("developer_normalized_tools", [dev("Dev prompt."), user("Hi")], BASIC_TOOLS, normalize=True)
fx("developer_raw_raises", [dev("Dev prompt."), user("Hi")])
fx("developer_second_normalized", [user("Hi"), dev("late")], normalize=True)


# --- OpenCode-like multi-turn session with tool loops
def opencode_session():
    s = True  # arguments as JSON strings, as OpenCode sends them
    return [
        sysm(OPENCODE_SYSTEM),
        user("Arregla el test que falla en `src/parser.ts` y explica qué pasaba. 🙏"),
        asst("", reasoning="The user wants me to fix a failing test. Let me find the tests first.",
             calls=[call("glob", {"pattern": "**/*.test.ts"}, "call_01", s),
                    call("grep", {"pattern": "parse\\(", "include": "*.ts"}, "call_02", s)]),
        toolmsg("C:\\Users\\dev\\projects\\demo\\src\\parser.test.ts\n"
                "C:\\Users\\dev\\projects\\demo\\src\\lexer.test.ts", "call_01"),
        toolmsg("Found 3 matches\nsrc/parser.ts:\n  Line 12: export function parse(input: string): Node[] {\n"
                "src/parser.test.ts:\n  Line 5:   expect(parse(\"a + b\")).toEqual([...]);\n"
                "  Line 9:   expect(parse(\"\")).toEqual([]);", "call_02"),
        asst("Voy a leer el parser.", reasoning="Read parser.ts around line 12.",
             calls=[call("read", {"filePath": "C:\\Users\\dev\\projects\\demo\\src\\parser.ts",
                                  "offset": 0, "limit": 200}, "call_03", s)]),
        toolmsg("<file>\n00001| import { lex, Token } from \"./lexer\";\n00002| \n"
                "00012| export function parse(input: string): Node[] {\n00013| \tconst tokens = lex(input);\n"
                "00014| \tif (tokens.length === 0) return null; // BUG\n00015| \treturn build(tokens, '\\n');\n"
                "00016| }\n\n(End of file - total 16 lines)\n</file>", "call_03"),
        asst("", reasoning="Line 14 returns null instead of []. Run the tests to confirm.",
             calls=[call("bash", {"command": "npx vitest run src/parser.test.ts",
                                  "description": "Run parser tests", "timeout": 120000}, "call_04", s)]),
        toolmsg("\x1b[31mFAIL\x1b[39m src/parser.test.ts > parse > empty input\n"
                "AssertionError: expected null to deeply equal []\n\n"
                " Test Files  1 failed (1)\n      Tests  1 failed | 11 passed (12)\n", "call_04"),
        asst("", reasoning="Confirmed. Fix: return [].",
             calls=[call("edit", {"filePath": "C:\\Users\\dev\\projects\\demo\\src\\parser.ts",
                                  "oldString": "\tif (tokens.length === 0) return null; // BUG",
                                  "newString": "\tif (tokens.length === 0) return [];",
                                  "replaceAll": False}, "call_05", s)]),
        toolmsg("Edit applied successfully.", "call_05"),
        asst("", reasoning="",
             calls=[call("bash", {"command": "npx vitest run src/parser.test.ts",
                                  "description": "Run parser tests again", "timeout": 120000}, "call_06", s)]),
        toolmsg("\u2713 src/parser.test.ts (12 tests) 34ms\n\n Test Files  1 passed (1)\n"
                "      Tests  12 passed (12)\n", "call_06"),
        asst("Arreglado. `parse(\"\")` devolvía `null` en lugar de `[]` (línea 14). Ahora devuelve una lista "
             "vacía y los 12 tests pasan.", reasoning="All tests pass. Summarize in Spanish."),
        user("Gracias. Ahora añade un test para entradas con solo espacios, y apunta las tareas."),
        asst("", reasoning="Plan with todowrite, then write the test.",
             calls=[call("todowrite", {"todos": [
                 {"content": "Add whitespace-only input test", "status": "in_progress", "priority": "high", "id": "1"},
                 {"content": "Run the test suite", "status": "pending", "priority": "medium", "id": "2"}]},
                 "call_07", s)]),
        toolmsg("[\n  {\n    \"content\": \"Add whitespace-only input test\",\n    \"status\": \"in_progress\"\n"
                "  }\n]", "call_07"),
        asst("", calls=[call("edit", {"filePath": "C:\\Users\\dev\\projects\\demo\\src\\parser.test.ts",
                                       "oldString": "  expect(parse(\"\")).toEqual([]);\n",
                                       "newString": "  expect(parse(\"\")).toEqual([]);\n"
                                                    "  expect(parse(\"  \\t\\n\")).toEqual([]);\n"}, "call_08", s),
                        call("bash", {"command": "npx vitest run", "description": "Run all tests"}, "call_09", s)]),
        toolmsg("Edit applied successfully.", "call_08"),
        toolmsg("\u2713 src/lexer.test.ts (8 tests)\n\u2713 src/parser.test.ts (13 tests)\n\n"
                " Test Files  2 passed (2)\n      Tests  21 passed (21)", "call_09"),
        asst("Test añadido; los 21 tests pasan.", reasoning="Done. Mark todos complete next time."),
        user("Perfecto. ¿Puedes revisar también el README?"),
    ]


fx("opencode_session", opencode_session(), OPENCODE_TOOLS, normalize=True)
fx("opencode_session_preserve_false", opencode_session(), OPENCODE_TOOLS, normalize=True, preserve_thinking=False)
fx("opencode_session_thinking_off", opencode_session(), OPENCODE_TOOLS, normalize=True, enable_thinking=False)
fx("opencode_session_low", opencode_session(), OPENCODE_TOOLS, normalize=True, reasoning_effort="low")
fx("opencode_session_mid_loop", opencode_session()[:9], OPENCODE_TOOLS, normalize=True)


# --- long agent conversation (dsh / pi style, ~80 messages)
def long_session():
    msgs = [dev("You are an expert coding assistant operating inside dsh. Today is 2026-10-06.\n"
                "Current working directory: /home/dev/engine\n\nGuidelines:\n- Be concise.\n- Prefer edits.")]
    msgs.append(user("Profile the tokenizer and make it faster. Report numbers before/after."))
    for k in range(12):
        cid = f"call_{k:03d}"
        reasoning = (f"Step {k}: inspect the next piece.\n\n" + ("Check hot loop.\n" * (k % 3)) +
                     ("  trailing spaces  \n" if k % 4 == 0 else ""))
        if k % 3 == 0:
            calls = [call("bash", {"command": f"perf stat -e cycles ./bench --iter {100 * (k + 1)}",
                                   "timeout": 60 + k}, cid, True)]
            result = (f"Performance counter stats for './bench --iter {100 * (k + 1)}':\n\n"
                      f"     {123456789 + k:,} cycles\n\n       {1.25 + k * 0.1:.3f} seconds time elapsed\n")
        elif k % 3 == 1:
            calls = [call("read", {"path": f"src/tok_{k}.cpp", "offset": 1, "limit": 80}, cid, True),
                     call("read", {"path": f"src/tok_{k}.h"}, cid + "b", True)]
            result = None
        else:
            calls = [call("edit", {"path": f"src/tok_{k}.cpp",
                                   "oldText": "for (size_t i = 0; i < n; ++i) {\n\tv.push_back(s[i]);\n}",
                                   "newText": "v.reserve(n);\nfor (size_t i = 0; i < n; ++i) v.push_back(s[i]);"},
                          cid, True)]
            result = "Successfully replaced text in src/tok_%d.cpp." % k
        content = "" if k % 2 == 0 else f"Paso {k}: sigo con el análisis — {'ñ' * k}"
        msgs.append(asst(content, reasoning=reasoning, calls=calls))
        if result is None:
            msgs.append(toolmsg(f"// tok_{k}.cpp\n#include \"tok.h\"\nstatic const char* kSep = \"\\t\\n\";\n"
                                f"int f{k}(int x) {{ return x * {k}; }}\n", cid))
            msgs.append(toolmsg(f"#pragma once\nint f{k}(int x);\n", cid + "b"))
        else:
            msgs.append(toolmsg(result, cid))
        if k == 5:
            msgs.append(asst("", reasoning="Need a screenshot of the flame graph.",
                             calls=[call("screenshot", "{}", "call_shot", False)]))
            msgs.append(toolmsg("Screenshot captured.", "call_shot"))
            msgs.append(user([text("Attached image(s) from tool result:"), IMG]))
        if k == 8:
            msgs.append(asst("Intermediate result: 1.8x faster so far.", reasoning="Report progress."))
            msgs.append(user("Good. Keep going, but don't touch the Unicode tables."))
    msgs.append(asst("Final: 2.3x faster (1.25 s -> 0.54 s). Changes in src/tok_2.cpp, src/tok_5.cpp, "
                     "src/tok_8.cpp, src/tok_11.cpp.", reasoning="Summarize the numbers."))
    msgs.append(user("Great, thanks!"))
    return msgs


fx("long_session", long_session(), PI_TOOLS, normalize=True, enable_thinking=True, preserve_thinking=True)
fx("long_session_preserve_false", long_session(), PI_TOOLS, normalize=True, preserve_thinking=False)
fx("long_session_vision_id", long_session(), PI_TOOLS, normalize=True, add_vision_id=True, reasoning_effort="medium")

# --- template exceptions (raise_exception)
fx("raise_no_messages", [])
fx("raise_no_messages_mapping", {})
fx("raise_unexpected_role", [user("x"), {"role": "function", "content": "y"}])
fx("raise_missing_role", [user("x"), {"content": "y"}])
fx("raise_role_null", [user("x"), {"role": None, "content": "y"}])
fx("raise_role_case", [{"role": "User", "content": "x"}, user("y")])
fx("raise_system_not_first", [user("x"), sysm("late system")])
fx("raise_system_twice", [sysm("a"), sysm("b"), user("x")])
fx("raise_no_user_only_system", [sysm("only system")])
fx("raise_no_user_assistant_only", [asst("hello")])
fx("raise_no_user_tool_responses", [sysm("s"), user("<tool_response>\nr\n</tool_response>"),
                                    user("<tool_response></tool_response>")])
fx("raise_image_in_system", [sysm([text("sys"), IMG]), user("x")])
fx("raise_image_in_system_tools", [sysm([IMG]), user("x")], BASIC_TOOLS)
fx("raise_video_in_system", [sysm([VIDEO]), user("x")])
fx("raise_image_in_late_system", [user("x"), sysm([IMG])])
fx("raise_unexpected_item", [user([text("a"), {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}])])
fx("raise_unexpected_item_video_url", [user([{"type": "video_url", "video_url": {"url": "x"}}])])
fx("raise_unexpected_item_string", [user(["hello"])])
fx("raise_unexpected_content_number", [user(5)])
fx("raise_unexpected_content_mapping", [user({"type": "text", "text": "x"})])
fx("raise_unexpected_content_bool", [user("x"), asst(True)])
fx("raise_unexpected_content_in_system", [sysm(3.5), user("x")])
fx("raise_order_effort_first", [user("x"), {"role": "bad"}], reasoning_effort="high")
fx("raise_order_system_image_before_no_user", [sysm([IMG])])
fx("raise_order_no_user_before_role", [{"role": "bad"}, asst("x")])
fx("raise_order_content_before_role", [user("x"), {"role": "bad", "content": 7}])

# --- other jinja2 errors (the C++ side must throw; message not compared)
fx("error_args_unparseable_normalized", [user("x"), asst("", calls=[call("read", "{bad json")])], normalize=True)
fx("error_args_unparseable_raw", [user("x"), asst("", calls=[call("read", "{\"filePath\": \"a\"}")])])
fx("error_args_list", [user("x"), asst("", calls=[call("read", [1, 2])])])
fx("error_args_null", [user("x"), asst("", calls=[call("read", None)])])
fx("error_args_number_normalized", [user("x"), asst("", calls=[call("read", "5")])], normalize=True)
fx("error_call_no_name", [user("x"), asst("", calls=[{"function": {"arguments": {}}}])])
fx("error_call_name_number", [user("x"), asst("", calls=[{"function": {"name": 5, "arguments": {}}}])])
fx("error_call_function_null", [user("x"), asst("", calls=[{"function": None}])])
fx("error_calls_string", [user("x"), asst("", calls="abc")])
fx("error_content_part_number", [user([text("a"), 5])])
fx("error_content_part_null", [user([None])])
fx("error_messages_mapping", {"role": "user", "content": "x"})

# --- line ends and odd whitespace in contents
fx("content_crlf", [sysm("Line 1\r\nLine 2\r\n"), user("\r\n  Q with CRLF\r\n"),
                    asst("A\r\n", reasoning="\r\nR\r\n", calls=[call("bash", {"command": "echo a\r\necho b"})]),
                    toolmsg("out\r\n")], BASIC_TOOLS)


# --- argument strings with number and escape spellings: json.loads and nlohmann must agree
fx("args_string_spellings_normalized", [user("x"), asst("", calls=[call("f", (
    '{"neg_zero": -0, "neg_zero_f": -0.0, "exp": 1E5, "exp2": 1.5E+300, "small": 1e-7, "frac": 0.000001, '
    '"sub": 2e-320, "u64": 12345678901234567890, "i64min": -9223372036854775808, "esc": "\\u00e9\\/\\ud83d\\ude00", '
    '"ctl": "\\u0001\\b\\f", "nested": {"x": [1, 2.50, "3"]}}'))])], normalize=True)
fx("args_string_duplicate_keys_normalized", [user("x"), asst("", calls=[call("f", '{"a": 1, "b": 2, "a": 3}')])],
   normalize=True)

# --- seeded random fixtures: tojson, strip, str(), structure
def fuzz_fixtures(seed=1234, n=400):
    rng = random.Random(seed)
    spaces = [chr(c) for c in range(0x110000) if chr(c).isspace()]
    # Characters whose Python repr the C++ side knows (py_isprintable is exact for these).
    pool = (list("abcXYZ019 _-.,:;!?/<>=&'\"\\{}[]()") + ["\n", "\t", "\r", "\x00", "\x01", "\x1b", "\x7f"] +
            spaces + ["é", "ñ", "中", "文", "😀", "👩‍💻", "𝄞", "​", "﻿", "᠎", "­",
                      "؜", "‮", "⁠", "", "\U000e0001", "￿"] +
            ["<tool_response>", "</tool_response>", "<|im_end|>", "<think>"])

    def rstr(maxlen=12):
        return "".join(rng.choice(pool) for _ in range(rng.randint(0, maxlen)))

    def rfloat():
        k = rng.random()
        if k < 0.3:
            return rng.choice([0.0, -0.0, 1.0, 0.5, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 0.1, 1e22, 1e21,
                               5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 123.456, -1e-7])
        if k < 0.6:
            return rng.uniform(-1, 1) * 10.0 ** rng.randint(-30, 30)
        while True:  # any finite double from random bits
            d = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            if d == d and abs(d) != float("inf"):
                return d

    def rvalue(depth=0):
        k = rng.randint(0, 9 if depth < 3 else 6)
        if k == 0: return None
        if k == 1: return rng.choice([True, False])
        if k == 2: return rng.randint(-2**63, 2**63 - 1) if rng.random() < 0.3 else rng.randint(-1000, 1000)
        if k == 3: return rfloat()
        if k in (4, 5, 6): return rstr()
        if k in (7, 8): return [rvalue(depth + 1) for _ in range(rng.randint(0, 4))]
        return {rstr(6): rvalue(depth + 1) for _ in range(rng.randint(0, 4))}

    def rpart():
        k = rng.randint(0, 9)
        if k < 5: return text(rstr(20))
        if k == 5: return IMG
        if k == 6: return rng.choice([IMG_URL, IMG_QWEN])
        if k == 7: return VIDEO
        if k == 8: return {"type": "text", "text": rng.choice([None, True, 7, rfloat(), ["a", None, 1.5, rstr(4)]])}
        return rng.choice(["has text", "an image", "video!"])

    def rcontent():
        k = rng.randint(0, 9)
        if k == 0: return None
        if k <= 5: return rstr(30)
        if k == 6: return "<tool_response>" + rstr(10) + "</tool_response>"
        return [rpart() for _ in range(rng.randint(0, 4))]

    def rcall(i):
        args = rng.choice([{rstr(6): rvalue() for _ in range(rng.randint(0, 4))}, {}, ""])
        as_string = rng.random() < 0.5 and args != ""
        return call("fn_" + rstr(4).replace("\x00", ""), args, f"c{i}", as_string)

    out = []
    for i in range(n):
        msgs = []
        first = rng.randint(0, 3)
        if first == 0: msgs.append(sysm(rcontent() if rng.random() < 0.7 else rstr(30)))
        if first == 1: msgs.append(dev(rstr(30)))
        for _ in range(rng.randint(1, 7)):
            role = rng.choice(["user", "user", "assistant", "assistant", "tool", "tool"])
            if role == "user":
                msgs.append(user(rcontent()))
            elif role == "tool":
                msgs.append(toolmsg(rcontent()))
            else:
                m = asst(rcontent() if rng.random() < 0.5 else rstr(30))
                r = rng.randint(0, 3)
                if r == 0: m["reasoning_content"] = rstr(30)
                if r == 1: m["reasoning_content"] = rng.choice([None, 3, ["x"]])
                if rng.random() < 0.6: m["tool_calls"] = [rcall(j) for j in range(rng.randint(0, 3))]
                msgs.append(m)
        if rng.random() < 0.8:  # most conversations get a real user query somewhere
            msgs.insert(rng.randint(1 if first < 2 else 0, len(msgs)), user("q " + rstr(20)))
        tools = None
        if rng.random() < 0.15:
            tools = [{"type": "function", "function": {"name": rstr(5), "description": rstr(20),
                                                       "parameters": rvalue()}} for _ in range(rng.randint(0, 2))]
        opts = {}
        if rng.random() < 0.3: opts["add_generation_prompt"] = rng.random() < 0.5
        if rng.random() < 0.3: opts["add_vision_id"] = rng.random() < 0.5
        for k in ("enable_thinking", "preserve_thinking"):
            if rng.random() < 0.5: opts[k] = rng.random() < 0.5
        if rng.random() < 0.3: opts["reasoning_effort"] = rng.choice(["xhigh", "medium", "low", "high"])
        out.append({"name": f"fuzz_{i:03d}", "normalize": rng.random() < 0.7, "messages": msgs, "tools": tools,
                    "options": opts})
    # One big schema with 3000 random doubles: Python float repr against the C++ formatter.
    floats = [rfloat() for _ in range(3000)]
    out.append({"name": "fuzz_floats", "normalize": False, "messages": [user("x")],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"examples": floats}}}],
                "options": {}})
    return out


FIXTURES.extend(fuzz_fixtures())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = read_gguf_string(args.gguf, "tokenizer.chat_template")
    # The research copy may have CRLF line ends (Windows checkout); the GGUF text has LF.
    with open(os.path.join(ROOT, "research", "_chat_template.jinja"), "rb") as f:
        if f.read().replace(b"\r\n", b"\n") != raw:
            sys.exit("research/_chat_template.jinja differs from the GGUF's tokenizer.chat_template")
    text_ = raw.decode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    plain, sandboxed = make_template(text_, False), make_template(text_, True)

    names = set()
    records = []
    counts = {"output": 0, "raise": 0, "error": 0}
    for fx_ in FIXTURES:
        assert fx_["name"] not in names, fx_["name"]
        names.add(fx_["name"])
        # Round-trip through JSON text: both sides see exactly the values stored in the file.
        rec = json.loads(json.dumps(fx_, ensure_ascii=False))
        exp = render(plain, rec)
        exp2 = render(sandboxed, json.loads(json.dumps(fx_, ensure_ascii=False)))
        if exp != exp2:
            sys.exit(f"{rec['name']}: Environment and ImmutableSandboxedEnvironment disagree:\n{exp}\n{exp2}")
        rec["expect"] = exp
        counts[next(iter(exp))] += 1
        records.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    doc = {"template_sha256": sha, "template": text_, "jinja2_version": jinja2.__version__, "fixtures": records}
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"template sha256 {sha} ({len(raw)} bytes)")
    print(f"{len(records)} fixtures: {counts['output']} outputs, {counts['raise']} raise_exception, "
          f"{counts['error']} other errors -> {args.out}")


if __name__ == "__main__":
    main()
