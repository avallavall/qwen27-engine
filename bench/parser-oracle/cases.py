# Writes cases.json for the oracle. Each case: name, tools (list of tool names), thinking, text.
import json, os

def fn(name, props, required=None, desc="d"):
    p = {"type": "object", "properties": props}
    if required is not None:
        p["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}

TOOLS = {
    "special_function": fn("special_function", {"arg1": {"type": "integer", "description": "The arg."}}, ["arg1"]),
    "special_function_with_opt": fn("special_function_with_opt", {"arg1": {"type": "integer"}, "arg2": {"type": "integer"}}, ["arg1"]),
    "python": fn("python", {"code": {"type": "string"}}, ["code"]),
    "html": fn("html", {"markup": {"type": "string"}}, ["markup"]),
    "todo_list": fn("todo_list", {"todos": {"type": "array"}}, ["todos"]),
    "edit": fn("edit", {"filename": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}},
               ["filename", "oldString", "newString"]),
    "manage_todo_list": fn("manage_todo_list", {"todos": {"type": "array"}}, ["todos"]),
    "run_in_terminal": fn("run_in_terminal", {"command": {"type": "string"}}, ["command"]),
    "empty_args": fn("empty_args", {}),
    "empty_args_no_props": {"type": "function", "function": {"name": "empty_args_no_props", "description": "d",
                                                             "parameters": {"type": "object"}}},
    "tool_2req_4opt": fn("tool_2req_4opt", {"req1": {"type": "string"}, "req2": {"type": "integer"}, "opt1": {"type": "string"},
                                            "opt2": {"type": "integer"}, "opt3": {"type": "string"}, "opt4": {"type": "integer"}},
                         ["req1", "req2"]),
    "tool_2req_5opt": fn("tool_2req_5opt", {"req1": {"type": "string"}, "req2": {"type": "integer"}, "opt1": {"type": "string"},
                                            "opt2": {"type": "integer"}, "opt3": {"type": "string"}, "opt4": {"type": "integer"},
                                            "opt5": {"type": "string"}}, ["req1", "req2"]),
    "set_nullable_str": fn("set_nullable_str", {"name": {"type": ["string", "null"]}}, ["name"]),
    "set_nullable_str_nf": fn("set_nullable_str_nf", {"name": {"type": ["null", "string"]}}, ["name"]),
    "set_nullable_int": fn("set_nullable_int", {"count": {"type": ["integer", "null"]}}, ["count"]),
    "set_union": fn("set_union", {"value": {"type": ["string", "object"]}, "amount": {"type": ["string", "integer"]}},
                    ["value", "amount"]),
    "set_unit": fn("set_unit", {"unit": {"enum": ["celsius", "fahrenheit"]}}, ["unit"]),
    # own tools
    "types": fn("types", {"s": {"type": "string"}, "i": {"type": "integer"}, "f": {"type": "number"}, "b": {"type": "boolean"},
                          "o": {"type": "object"}, "a": {"type": "array"}, "n": {"type": "null"}, "any": {"description": "no type"},
                          "e": {"enum": [1, 2, "x"]}, "ref": {"$ref": "#/properties/i"}, "anyof": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}}),
    "get_weather": fn("get_weather", {"city": {"type": "string"}, "days": {"type": "integer"}}, ["city"]),
    "read_file": fn("read_file", {"path": {"type": "string"}}, ["path"]),
}

def call(name, *args):
    s = "<tool_call>\n<function=" + name + ">\n"
    for k, v in args:
        s += "<parameter=" + k + ">\n" + v + "\n</parameter>\n"
    return s + "</function>\n</tool_call>"

SPECIAL1 = call("special_function", ("arg1", "1"))
RUN_PWD = call("run_in_terminal", ("command", "pwd"))

cases = []
def add(name, text, tools=(), thinking=True, **kw):
    c = {"name": name, "tools": list(tools), "thinking": thinking, "text": text}
    c.update(kw)
    cases.append(c)

# ---- llama.cpp fixtures: tests/test-chat.cpp, Qwen3.5-4B block (line 2147)
add("q35_thoughts", "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?")
add("q35_call_nothink", SPECIAL1, ["special_function"], thinking=False)
add("q35_call_thoughts", "I'm\nthinking\n</think>\n\n" + SPECIAL1, ["special_function"])
add("q35_parallel", SPECIAL1 + "\n" + call("special_function_with_opt", ("arg1", "1"), ("arg2", "2")),
    ["special_function", "special_function_with_opt"], thinking=False, parallel=True)
add("q35_python", call("python", ("code", "def hello():\n    print(\"Hello, world!\")\n\nhello()")), ["python"], thinking=False)
add("q35_edit", call("edit", ("filename", "foo.c"), ("oldString", "#iclunde"), ("newString", "#include")), ["edit"], thinking=False)
add("q35_edit_trailing_nl", call("edit", ("filename", "foo.c"), ("oldString", "#iclunde"), ("newString", "#include\n")), ["edit"], thinking=False)
add("q35_python_indent", call("python", ("code", "    print(\"Hello, world!\")")), ["python"], thinking=False)
add("q35_call_no_think_close", RUN_PWD, ["run_in_terminal"])
add("q35_reasoning_then_call", "Need to inspect the current directory.\n" + RUN_PWD, ["run_in_terminal"])
add("q35_empty_args", "<tool_call>\n<function=empty_args>\n</function>\n</tool_call>", ["empty_args"], thinking=False)
add("q35_empty_args_no_props", "<tool_call>\n<function=empty_args_no_props>\n</function>\n</tool_call>", ["empty_args_no_props"], thinking=False)
add("q35_think_tags_again", "<think>\n\n</think>\n\n" + SPECIAL1, ["special_function"])
add("q35_empty_reasoning_call", "</think>\n\n" + SPECIAL1, ["special_function"])
add("q35_empty_reasoning_call2", "</think>\n\n" + RUN_PWD, ["run_in_terminal"])
add("q35_content_then_call", "</think>\n\nLet me inspect the current directory.\n" + RUN_PWD, ["run_in_terminal"])
add("q35_all_three", "I should inspect the directory.\n</think>\n\nLet me inspect it now.\n" + RUN_PWD, ["run_in_terminal"])
add("q35_two_tools", "I need to run a terminal command.\n</think>\n\n" + RUN_PWD, ["manage_todo_list", "run_in_terminal"])
# ---- llama.cpp fixtures: Qwen3-Coder block (line 3551), run with our template (thinking off)
add("qc_content", "Hello, world!\nWhat's up?", thinking=False)
add("qc_call", SPECIAL1, ["special_function"], thinking=False)
add("qc_no_open_tag", "<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>", ["special_function"], thinking=False)
add("qc_no_open_tag_content", "Let me call it.\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>",
    ["special_function"], thinking=False)
add("qc_parallel", SPECIAL1 + "\n" + call("special_function_with_opt", ("arg1", "1"), ("arg2", "2")),
    ["special_function", "special_function_with_opt"], thinking=False)
add("qc_unicode", call("python", ("code", "格")), ["python"], thinking=False)
add("qc_html", call("html", ("markup", "<html>\n <head>\n  <title>Hello!</title>\n </head>\n</html>")), ["html"], thinking=False)
add("qc_todo", call("todo_list", ("todos", "[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]")),
    ["todo_list"], thinking=False)
add("qc_edit_order", call("edit", ("newString", "#include"), ("filename", "foo.c"), ("oldString", "#iclunde")), ["edit"], thinking=False)
add("qc_2req4opt_a", call("tool_2req_4opt", ("req2", "42"), ("req1", "hello"), ("opt2", "200")), ["tool_2req_4opt"], thinking=False)
add("qc_2req4opt_b", call("tool_2req_4opt", ("req1", "hello"), ("req2", "42"), ("opt4", "100"), ("opt2", "200")), ["tool_2req_4opt"], thinking=False)
add("qc_2req5opt_a", call("tool_2req_5opt", ("req1", "world"), ("req2", "7"), ("opt5", "last"), ("opt3", "middle"), ("opt1", "first")),
    ["tool_2req_5opt"], thinking=False)
add("qc_2req5opt_b", call("tool_2req_5opt", ("req1", "test"), ("req2", "99"), ("opt3", "c"), ("opt1", "a"), ("opt5", "e"), ("opt4", "4"), ("opt2", "2")),
    ["tool_2req_5opt"], thinking=False)
add("qc_nullable_str", call("set_nullable_str", ("name", "hello world")), ["set_nullable_str"], thinking=False)
add("qc_nullable_str_nf", call("set_nullable_str_nf", ("name", "hello world")), ["set_nullable_str_nf"], thinking=False)
add("qc_nullable_int", call("set_nullable_int", ("count", "42")), ["set_nullable_int"], thinking=False)
add("qc_nullable_str_null", call("set_nullable_str", ("name", "null")), ["set_nullable_str"], thinking=False)
add("qc_union", call("set_union", ("value", "{\"a\": 1}"), ("amount", "2 dollars")), ["set_union"], thinking=False)
add("qc_union_bad_json", call("set_union", ("value", "{not valid json"), ("amount", "42")), ["set_union"], thinking=False)
add("qc_enum", call("set_unit", ("unit", "celsius")), ["set_unit"], thinking=False)

# ---- own cases
add("own_plain_think", "Let me think.\nSecond line.\n</think>\n\nThe answer is 4.")
add("own_plain_nothink", "The answer is 4.", thinking=False)
add("own_nothink_ws", "\n\n  Hi there  \n\n", thinking=False)
add("own_think_ws_content", "  \n reasoning \n\n</think>\n\n\n  Answer  \n\n")
add("own_reasoning_only", "Still thinking about it")
add("own_reasoning_only_trailing_ws", "Still thinking\n\n")
add("own_empty_reasoning_nl", "\n</think>\n\nAnswer.")
add("own_empty_reasoning", "</think>\n\nAnswer.")
add("own_empty_output_think", "")
add("own_empty_output_nothink", "", thinking=False)
add("own_think_close_twice", "a\n</think>\n\nb</think>c")
add("own_toolcall_text_no_tools_think", "I could call <tool_call> here\n</think>\n\nok")
add("own_toolcall_text_no_tools_nothink", "x <tool_call>\n<function=f>\n</function>\n</tool_call> y", thinking=False)
add("own_types_all", call("types", ("s", "hello"), ("i", "42"), ("f", "3.14"), ("b", "true"), ("o", "{\"a\": [1, 2], \"b\": null}"),
                          ("a", "[1, \"x\", false]"), ("n", "null"), ("any", "free text"), ("e", "2"), ("ref", "7"), ("anyof", "false")), ["types"], thinking=False)
add("own_types_any_json", call("types", ("any", "7"), ("e", "x"), ("anyof", "maybe")), ["types"], thinking=False)
add("own_types_any_obj", call("types", ("any", "{\"k\": 'v', \"t\": True}")), ["types"], thinking=False)
add("own_types_float_exp", call("types", ("f", "-1.5e+3"), ("i", "0")), ["types"], thinking=False)
add("own_int_not_json", call("types", ("i", "abc")), ["types"], thinking=False)
add("own_int_leading_space", call("types", ("i", " 42")), ["types"], thinking=False)
add("own_int_trailing_space", call("types", ("i", "42 ")), ["types"], thinking=False)
add("own_int_as_jsonstring", call("types", ("i", "\"42\"")), ["types"], thinking=False)
add("own_str_lt", call("python", ("code", "if a < b and c </parameter> d:\n  x = '<parameter=code>'\n</parameter>x")), ["python"], thinking=False)
add("own_str_escapes", call("python", ("code", "q=\"x\"\\t\tb\\\\ \x01 \u00e9 \U0001F600 /")), ["python"], thinking=False)
add("own_str_empty", "<tool_call>\n<function=python>\n<parameter=code>\n\n</parameter>\n</function>\n</tool_call>", ["python"], thinking=False)
add("own_str_multiline_unicode", call("python", ("code", "h\u00e9llo\n\u4e16\u754c \U0001F30D\n\n  line3  ")), ["python"], thinking=False)
add("own_str_crlf", call("python", ("code", "a\r\nb\r")), ["python"], thinking=False)
add("own_three_calls", "Doing three.\n</think>\n\nOk.\n" + call("get_weather", ("city", "Paris")) + "\n" + call("get_weather", ("city", "Rome"), ("days", "3"))
    + "\n" + call("read_file", ("path", "/tmp/a b.txt")), ["get_weather", "read_file"])
add("own_content_between_calls", call("get_weather", ("city", "Paris")) + "\nBetween\n" + call("read_file", ("path", "x")), ["get_weather", "read_file"], thinking=False)
add("own_content_after_calls", call("get_weather", ("city", "Paris")) + "\nDone.", ["get_weather"], thinking=False)
add("own_ws_after_calls", call("get_weather", ("city", "Paris")) + "\n\n", ["get_weather"], thinking=False)
add("own_unknown_tool", "Text\n" + call("nope", ("x", "1")), ["get_weather"], thinking=False)
add("own_unknown_tool_then_known", call("nope", ("x", "1")) + "\n" + call("get_weather", ("city", "Paris")), ["get_weather"], thinking=False)
add("own_unknown_param", call("get_weather", ("city", "Paris"), ("country", "FR")), ["get_weather"], thinking=False)
add("own_missing_required", call("get_weather", ("days", "2")), ["get_weather"], thinking=False)
add("own_dup_optional", call("get_weather", ("city", "Paris"), ("days", "2"), ("days", "3")), ["get_weather"], thinking=False)
add("own_dup_required", call("get_weather", ("city", "Paris"), ("city", "Rome")), ["get_weather"], thinking=False)
add("own_call_in_reasoning", "I will check.\nLet me call <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    ["get_weather"])
add("own_call_no_trailing_nl_func", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function></tool_call>", ["get_weather"], thinking=False)
add("own_call_spaces_between", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>  \n \n" + call("get_weather", ("city", "Rome")), ["get_weather"], thinking=False)
add("own_call_no_nl_after_open", "<tool_call><function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", ["get_weather"], thinking=False)
add("own_param_no_nl", "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>", ["get_weather"], thinking=False)
add("own_nothink_parallel_false", SPECIAL1 + "\n" + SPECIAL1, ["special_function"], thinking=False, parallel=False)
# EOS inside a call (final parse of truncated text)
full = "Hm.\n</think>\n\nSure.\n" + call("get_weather", ("city", "Paris"), ("days", "3"))
cuts = ["<tool_call>", "<tool_call>\n", "<tool_call>\n<function=get_w", "<tool_call>\n<function=get_weather>", "<tool_call>\n<function=get_weather>\n",
        "<tool_call>\n<function=get_weather>\n<parameter=ci", "<tool_call>\n<function=get_weather>\n<parameter=city>\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</param",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"]
for i, cut in enumerate(cuts):
    add("own_eos_%02d" % i, "Hm.\n</think>\n\nSure.\n" + cut, ["get_weather"])
add("own_eos_types_obj", "<tool_call>\n<function=types>\n<parameter=o>\n{\"a\": [1, 2", ["types"], thinking=False)
add("own_eos_types_str", "<tool_call>\n<function=types>\n<parameter=s>\nabc\n", ["types"], thinking=False)
add("own_eos_types_float", "<tool_call>\n<function=types>\n<parameter=f>\n3.", ["types"], thinking=False)
add("own_eos_in_think_close", "abc\n</thi")
add("own_eos_in_content_tag", "x\n</think>\n\nHello <tool_", ["get_weather"])
add("own_partial_tag_in_content", "x\n</think>\n\nHello <tool_ end", ["get_weather"])
add("own_python_literals", call("types", ("o", "{'a': True, 'b': None, 'c': 'it''s'}")), ["types"], thinking=False)

add("own_optional_before_required", call("get_weather", ("days", "2"), ("city", "Paris")), ["get_weather"], thinking=False)
U = "<tool_call>\n<function=set_union>\n<parameter="
add("own_eos_union_obj", U + "value>\n{\"a\": 1", ["set_union"], thinking=False)
add("own_eos_union_str", U + "value>\nhello wor", ["set_union"], thinking=False)
add("own_eos_union_num", U + "value>\n{\"a\": 1}\n</parameter>\n<parameter=amount>\n4", ["set_union"], thinking=False)
add("own_eos_union_numstr", U + "value>\n{\"a\": 1}\n</parameter>\n<parameter=amount>\n2 d", ["set_union"], thinking=False)
add("own_eos_union_close", U + "value>\n{\"a\": 1}\n</parameter>\n<parameter=amount>\n42\n</par", ["set_union"], thinking=False)
add("own_union_obj_trailing", call("set_union", ("value", "{\"a\": 1} trailing"), ("amount", "-0.5e2")), ["set_union"], thinking=False)
add("own_union_jsonstring", call("set_union", ("value", "\"quoted\""), ("amount", "[1]")), ["set_union"], thinking=False)
T = "<tool_call>\n<function=types>\n<parameter="
add("own_eos_json_backslash", T + "o>\n{\"k\": \"x\\", ["types"], thinking=False)
add("own_eos_json_uescape", T + "o>\n{\"k\": \"x\\u00", ["types"], thinking=False)
add("own_eos_json_int_close", T + "i>\n42\n</parameter>", ["types"], thinking=False)
add("own_eos_json_null", T + "n>\nnu", ["types"], thinking=False)
add("own_eos_json_bool", T + "b>\ntr", ["types"], thinking=False)
add("own_eos_json_arr", T + "a>\n[1, ", ["types"], thinking=False)
add("own_eos_json_minus", T + "f>\n-", ["types"], thinking=False)
add("own_json_ws_inside", call("types", ("o", "{ \"a\" :\t[ 1 ,\n2 ] ,\"b\":{}}"), ("a", "[]")), ["types"], thinking=False)
add("own_json_raw_newline_in_string", call("types", ("o", "{\"a\": \"x\n</parameter>\ny\"}")), ["types"], thinking=False)
add("own_json_number_forms", call("types", ("f", "0"), ("i", "-0"), ("a", "[0.5, 1E3, -2e-2, 10]")), ["types"], thinking=False)
add("own_json_bad_number", call("types", ("f", "01")), ["types"], thinking=False)
add("own_json_bad_number2", call("types", ("f", "1.")), ["types"], thinking=False)
add("own_json_escapes_ok", call("types", ("o", "{\"q\": \"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\\u00e9\"}")), ["types"], thinking=False)
add("own_json_bad_escape", call("types", ("o", "{\"q\": \"a\\x\"}")), ["types"], thinking=False)
add("own_param_name_space", "<tool_call>\n<function=get_weather>\n<parameter= city>\nParis\n</parameter>\n</function>\n</tool_call>", ["get_weather"], thinking=False)
add("own_two_calls_eos_second", call("get_weather", ("city", "Paris")) + "\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRo", ["get_weather"], thinking=False)
add("own_call_then_bad_call", call("get_weather", ("city", "Paris")) + "\n<tool_call>\n<function=nope>\n</function>\n</tool_call>", ["get_weather"], thinking=False)
add("own_call_then_text_then_call", call("get_weather", ("city", "Paris")) + "\nx" + call("get_weather", ("city", "Rome")), ["get_weather"], thinking=False)
add("own_lt_in_content", "x\n</think>\n\na <b> c </tool> d <tool_call", ["get_weather"])
add("own_lt_in_reasoning", "a < b </th ink> <tool_ c\n</think>\n\nok")
add("own_tool_response_tag", "x\n</think>\n\n<tool_response>\nhi\n</tool_response>", ["get_weather"])
add("own_think_tag_in_content_nothink", "a</think>b<think>c", thinking=False)
add("own_name_prefix", "<tool_call>\n<function=get>\n</function>\n</tool_call>", ["get_weather"], thinking=False)

def addhex(name, b, tools=(), thinking=True):
    cases.append({"name": name, "tools": list(tools), "thinking": thinking, "text_hex": b.hex()})
addhex("raw_invalid_content", b"x\n</think>\n\nab\xffcd\x80e")
addhex("raw_invalid_reasoning", b"a\xc3(b\xe4\xb8a\n</think>\n\nc")
addhex("raw_incomplete_end", b"x\n</think>\n\nab\xe4\xb8")
addhex("raw_lt_incomplete_end", b"x\n</think>\n\nab<\xe4", ["get_weather"])
addhex("raw_lax_utf8", b"x\n</think>\n\n\xc0\x80 \xed\xa0\x80 \xf5\x80\x80\x80 \xf8 z")
addhex("raw_param_incomplete_eos", b"<tool_call>\n<function=read_file>\n<parameter=path>\nab\xe4\xb8", ["read_file"], thinking=False)
addhex("raw_json_incomplete_eos", b"<tool_call>\n<function=types>\n<parameter=o>\n{\"k\": \"ab\xe4\xb8", ["types"], thinking=False)
addhex("raw_quirk_reasoning_tail", b"abc</thi\xe4")
addhex("raw_quirk_param_tail", b"<tool_call>\n<function=read_file>\n<parameter=path>\nab\n\xe4", ["read_file"], thinking=False)
addhex("raw_param_invalid",b"<tool_call>\n<function=read_file>\n<parameter=path>\na\xffb\n</parameter>\n</function>\n</tool_call>", ["read_file"], thinking=False)

for c in cases:
    c["tools"] = [TOOLS[t] for t in c["tools"]] if c["tools"] else None
here = os.path.dirname(os.path.abspath(__file__))
json.dump(cases, open(os.path.join(here, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cases), "cases")
