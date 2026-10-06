// Test of the streaming output parser (src/chat_parser.h) against llama.cpp's parser.
//
// Usage: test_parser <model.gguf>
// Build: tools\build_target.bat build-parse test_parser
//
// For every case the text is fed three ways: token by token (Tokenizer::encode(text, true), as the model would
// produce it), byte by byte, and in random chunks. Each time:
// - the deltas, applied as a client does, give the parser's final values;
// - the final values equal the expected ones;
// - the first delta of a call carries its id (32 chars [0-9A-Za-z], unique) and name, later deltas do not;
// - calls that reached "</function>" have valid JSON arguments; a call cut by EOS (only the last) does not.
// Expected values: "llama.cpp" cases come from the production build (llama-common.dll in qwen38_27\bin-parches,
// built from llama-rig2 rig/full): common_chat_templates_apply with research/_chat_template.jinja, the case's
// tools and enable_thinking, reasoning_format deepseek (server default), then common_chat_parse(text, false).
// "policy" cases are inputs where llama.cpp drops a call that a stream has already announced (text its grammar
// never lets the model write); there the expected value is this parser's rule (chat_parser.h).
// Also checked: control tokens, finish() twice, tools = [], and the speed on a 20k-token message.
// Prints PASS/FAIL counts. Exit code 1 on any failure.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "chat_parser.h"
#include "gguf.h"
#include "tokenizer.h"

using namespace q27;

namespace {

struct Case {
  const char* name;
  const char* source;  // "llama.cpp" or "policy"
  bool thinking;
  std::vector<const char*> tools;
  std::string text;
  std::string reasoning, content;
  std::vector<std::pair<std::string, std::string>> calls;  // name, arguments
  bool tokens;  // false: bytes that tokenization would change (invalid UTF-8); fed as bytes only
};

// clang-format off
// ---- fixtures (generated) begin
// Tool definitions used by the cases (OpenAI format, as sent in "tools").
const std::vector<std::pair<const char*, const char*>> kToolDefs = {
    {"special_function", "{\"type\":\"function\",\"function\":{\"name\":\"special_function\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"arg1\":{\"type\":\"integer\",\"description\":\"The arg.\"}},\"required\":[\"arg1\"]}}}"},
    {"special_function_with_opt", "{\"type\":\"function\",\"function\":{\"name\":\"special_function_with_opt\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"arg1\":{\"type\":\"integer\"},\"arg2\":{\"type\":\"integer\"}},\"required\":[\"arg1\"]}}}"},
    {"python", "{\"type\":\"function\",\"function\":{\"name\":\"python\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}"},
    {"html", "{\"type\":\"function\",\"function\":{\"name\":\"html\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"markup\":{\"type\":\"string\"}},\"required\":[\"markup\"]}}}"},
    {"todo_list", "{\"type\":\"function\",\"function\":{\"name\":\"todo_list\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"todos\":{\"type\":\"array\"}},\"required\":[\"todos\"]}}}"},
    {"edit", "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"filename\":{\"type\":\"string\"},\"oldString\":{\"type\":\"string\"},\"newString\":{\"type\":\"string\"}},\"required\":[\"filename\",\"oldString\",\"newString\"]}}}"},
    {"manage_todo_list", "{\"type\":\"function\",\"function\":{\"name\":\"manage_todo_list\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"todos\":{\"type\":\"array\"}},\"required\":[\"todos\"]}}}"},
    {"run_in_terminal", "{\"type\":\"function\",\"function\":{\"name\":\"run_in_terminal\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}}}"},
    {"empty_args", "{\"type\":\"function\",\"function\":{\"name\":\"empty_args\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}"},
    {"empty_args_no_props", "{\"type\":\"function\",\"function\":{\"name\":\"empty_args_no_props\",\"description\":\"d\",\"parameters\":{\"type\":\"object\"}}}"},
    {"tool_2req_4opt", "{\"type\":\"function\",\"function\":{\"name\":\"tool_2req_4opt\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"req1\":{\"type\":\"string\"},\"req2\":{\"type\":\"integer\"},\"opt1\":{\"type\":\"string\"},\"opt2\":{\"type\":\"integer\"},\"opt3\":{\"type\":\"string\"},\"opt4\":{\"type\":\"integer\"}},\"required\":[\"req1\",\"req2\"]}}}"},
    {"tool_2req_5opt", "{\"type\":\"function\",\"function\":{\"name\":\"tool_2req_5opt\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"req1\":{\"type\":\"string\"},\"req2\":{\"type\":\"integer\"},\"opt1\":{\"type\":\"string\"},\"opt2\":{\"type\":\"integer\"},\"opt3\":{\"type\":\"string\"},\"opt4\":{\"type\":\"integer\"},\"opt5\":{\"type\":\"string\"}},\"required\":[\"req1\",\"req2\"]}}}"},
    {"set_nullable_str", "{\"type\":\"function\",\"function\":{\"name\":\"set_nullable_str\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":[\"string\",\"null\"]}},\"required\":[\"name\"]}}}"},
    {"set_nullable_str_nf", "{\"type\":\"function\",\"function\":{\"name\":\"set_nullable_str_nf\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":[\"null\",\"string\"]}},\"required\":[\"name\"]}}}"},
    {"set_nullable_int", "{\"type\":\"function\",\"function\":{\"name\":\"set_nullable_int\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"count\":{\"type\":[\"integer\",\"null\"]}},\"required\":[\"count\"]}}}"},
    {"set_union", "{\"type\":\"function\",\"function\":{\"name\":\"set_union\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"value\":{\"type\":[\"string\",\"object\"]},\"amount\":{\"type\":[\"string\",\"integer\"]}},\"required\":[\"value\",\"amount\"]}}}"},
    {"set_unit", "{\"type\":\"function\",\"function\":{\"name\":\"set_unit\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"unit\":{\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"unit\"]}}}"},
    {"types", "{\"type\":\"function\",\"function\":{\"name\":\"types\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"},\"i\":{\"type\":\"integer\"},\"f\":{\"type\":\"number\"},\"b\":{\"type\":\"boolean\"},\"o\":{\"type\":\"object\"},\"a\":{\"type\":\"array\"},\"n\":{\"type\":\"null\"},\"any\":{\"description\":\"no type\"},\"e\":{\"enum\":[1,2,\"x\"]},\"ref\":{\"$ref\":\"#/properties/i\"},\"anyof\":{\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"boolean\"}]}}}}}"},
    {"get_weather", "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}"},
    {"read_file", "{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}"},
};

// Expected values: llama.cpp = output of the production llama-common.dll (common_chat_parse, final parse) on
// the same text and tools; policy = this parser's rule where llama.cpp drops a call (see chat_parser.h).
const std::vector<Case> kCases = {
    {"q35_thoughts", "llama.cpp", true, {},
     "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?",
     "I'm\nthinking\n", "Hello, world!\nWhat's up?",
     {}, true},
    {"q35_call_nothink", "llama.cpp", false, {"special_function"},
     "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"special_function", "{\"arg1\":1}"}}, true},
    {"q35_call_thoughts", "llama.cpp", true, {"special_function"},
     "I'm\n"
         "thinking\n"
         "</think>\n"
         "\n"
         "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "I'm\nthinking\n", "",
     {{"special_function", "{\"arg1\":1}"}}, true},
    {"q35_parallel", "llama.cpp", false, {"special_function", "special_function_with_opt"},
     "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=special_function_with_opt>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "<parameter=arg2>\n"
         "2\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"special_function", "{\"arg1\":1}"}, {"special_function_with_opt", "{\"arg1\":1,\"arg2\":2}"}}, true},
    {"q35_python", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "def hello():\n"
         "    print(\"Hello, world!\")\n"
         "\n"
         "hello()\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}"}}, true},
    {"q35_edit", "llama.cpp", false, {"edit"},
     "<tool_call>\n"
         "<function=edit>\n"
         "<parameter=filename>\n"
         "foo.c\n"
         "</parameter>\n"
         "<parameter=oldString>\n"
         "#iclunde\n"
         "</parameter>\n"
         "<parameter=newString>\n"
         "#include\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"edit", "{\"filename\":\"foo.c\",\"oldString\":\"#iclunde\",\"newString\":\"#include\"}"}}, true},
    {"q35_edit_trailing_nl", "llama.cpp", false, {"edit"},
     "<tool_call>\n"
         "<function=edit>\n"
         "<parameter=filename>\n"
         "foo.c\n"
         "</parameter>\n"
         "<parameter=oldString>\n"
         "#iclunde\n"
         "</parameter>\n"
         "<parameter=newString>\n"
         "#include\n"
         "\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"edit", "{\"filename\":\"foo.c\",\"oldString\":\"#iclunde\",\"newString\":\"#include\\n\"}"}}, true},
    {"q35_python_indent", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "    print(\"Hello, world!\")\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"    print(\\\"Hello, world!\\\")\"}"}}, true},
    {"q35_call_no_think_close", "llama.cpp", true, {"run_in_terminal"},
     "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"q35_reasoning_then_call", "llama.cpp", true, {"run_in_terminal"},
     "Need to inspect the current directory.\n"
         "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "Need to inspect the current directory.\n", "",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"q35_empty_args", "llama.cpp", false, {"empty_args"},
     "<tool_call>\n<function=empty_args>\n</function>\n</tool_call>",
     "", "",
     {{"empty_args", "{}"}}, true},
    {"q35_empty_args_no_props", "llama.cpp", false, {"empty_args_no_props"},
     "<tool_call>\n"
         "<function=empty_args_no_props>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"empty_args_no_props", "{}"}}, true},
    {"q35_think_tags_again", "llama.cpp", true, {"special_function"},
     "<think>\n"
         "\n"
         "</think>\n"
         "\n"
         "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "<think>\n\n", "",
     {{"special_function", "{\"arg1\":1}"}}, true},
    {"q35_empty_reasoning_call", "llama.cpp", true, {"special_function"},
     "</think>\n"
         "\n"
         "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"special_function", "{\"arg1\":1}"}}, true},
    {"q35_empty_reasoning_call2", "llama.cpp", true, {"run_in_terminal"},
     "</think>\n"
         "\n"
         "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"q35_content_then_call", "llama.cpp", true, {"run_in_terminal"},
     "</think>\n"
         "\n"
         "Let me inspect the current directory.\n"
         "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "Let me inspect the current directory.\n",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"q35_all_three", "llama.cpp", true, {"run_in_terminal"},
     "I should inspect the directory.\n"
         "</think>\n"
         "\n"
         "Let me inspect it now.\n"
         "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "I should inspect the directory.\n", "Let me inspect it now.\n",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"q35_two_tools", "llama.cpp", true, {"manage_todo_list", "run_in_terminal"},
     "I need to run a terminal command.\n"
         "</think>\n"
         "\n"
         "<tool_call>\n"
         "<function=run_in_terminal>\n"
         "<parameter=command>\n"
         "pwd\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "I need to run a terminal command.\n", "",
     {{"run_in_terminal", "{\"command\":\"pwd\"}"}}, true},
    {"qc_content", "llama.cpp", false, {},
     "Hello, world!\nWhat's up?",
     "", "Hello, world!\nWhat's up?",
     {}, true},
    {"qc_call", "llama.cpp", false, {"special_function"},
     "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"special_function", "{\"arg1\":1}"}}, true},
    {"qc_no_open_tag", "llama.cpp", false, {"special_function"},
     "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     {}, true},
    {"qc_no_open_tag_content", "llama.cpp", false, {"special_function"},
     "Let me call it.\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "Let me call it.\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     {}, true},
    {"qc_parallel", "llama.cpp", false, {"special_function", "special_function_with_opt"},
     "<tool_call>\n"
         "<function=special_function>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=special_function_with_opt>\n"
         "<parameter=arg1>\n"
         "1\n"
         "</parameter>\n"
         "<parameter=arg2>\n"
         "2\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"special_function", "{\"arg1\":1}"}, {"special_function_with_opt", "{\"arg1\":1,\"arg2\":2}"}}, true},
    {"qc_unicode", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "格\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"格\"}"}}, true},
    {"qc_html", "llama.cpp", false, {"html"},
     "<tool_call>\n"
         "<function=html>\n"
         "<parameter=markup>\n"
         "<html>\n"
         " <head>\n"
         "  <title>Hello!</title>\n"
         " </head>\n"
         "</html>\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"html", "{\"markup\":\"<html>\\n <head>\\n  <title>Hello!</title>\\n </head>\\n</html>\"}"}}, true},
    {"qc_todo", "llama.cpp", false, {"todo_list"},
     "<tool_call>\n"
         "<function=todo_list>\n"
         "<parameter=todos>\n"
         "[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"todo_list", "{\"todos\":[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]}"}}, true},
    {"qc_edit_order", "llama.cpp", false, {"edit"},
     "<tool_call>\n"
         "<function=edit>\n"
         "<parameter=newString>\n"
         "#include\n"
         "</parameter>\n"
         "<parameter=filename>\n"
         "foo.c\n"
         "</parameter>\n"
         "<parameter=oldString>\n"
         "#iclunde\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"edit", "{\"newString\":\"#include\",\"filename\":\"foo.c\",\"oldString\":\"#iclunde\"}"}}, true},
    {"qc_2req4opt_a", "llama.cpp", false, {"tool_2req_4opt"},
     "<tool_call>\n"
         "<function=tool_2req_4opt>\n"
         "<parameter=req2>\n"
         "42\n"
         "</parameter>\n"
         "<parameter=req1>\n"
         "hello\n"
         "</parameter>\n"
         "<parameter=opt2>\n"
         "200\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"tool_2req_4opt", "{\"req2\":42,\"req1\":\"hello\",\"opt2\":200}"}}, true},
    {"qc_2req4opt_b", "llama.cpp", false, {"tool_2req_4opt"},
     "<tool_call>\n"
         "<function=tool_2req_4opt>\n"
         "<parameter=req1>\n"
         "hello\n"
         "</parameter>\n"
         "<parameter=req2>\n"
         "42\n"
         "</parameter>\n"
         "<parameter=opt4>\n"
         "100\n"
         "</parameter>\n"
         "<parameter=opt2>\n"
         "200\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"tool_2req_4opt", "{\"req1\":\"hello\",\"req2\":42,\"opt4\":100,\"opt2\":200}"}}, true},
    {"qc_2req5opt_a", "llama.cpp", false, {"tool_2req_5opt"},
     "<tool_call>\n"
         "<function=tool_2req_5opt>\n"
         "<parameter=req1>\n"
         "world\n"
         "</parameter>\n"
         "<parameter=req2>\n"
         "7\n"
         "</parameter>\n"
         "<parameter=opt5>\n"
         "last\n"
         "</parameter>\n"
         "<parameter=opt3>\n"
         "middle\n"
         "</parameter>\n"
         "<parameter=opt1>\n"
         "first\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"tool_2req_5opt", "{\"req1\":\"world\",\"req2\":7,\"opt5\":\"last\",\"opt3\":\"middle\",\"opt1\":\"first\"}"}}, true},
    {"qc_2req5opt_b", "llama.cpp", false, {"tool_2req_5opt"},
     "<tool_call>\n"
         "<function=tool_2req_5opt>\n"
         "<parameter=req1>\n"
         "test\n"
         "</parameter>\n"
         "<parameter=req2>\n"
         "99\n"
         "</parameter>\n"
         "<parameter=opt3>\n"
         "c\n"
         "</parameter>\n"
         "<parameter=opt1>\n"
         "a\n"
         "</parameter>\n"
         "<parameter=opt5>\n"
         "e\n"
         "</parameter>\n"
         "<parameter=opt4>\n"
         "4\n"
         "</parameter>\n"
         "<parameter=opt2>\n"
         "2\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"tool_2req_5opt", "{\"req1\":\"test\",\"req2\":99,\"opt3\":\"c\",\"opt1\":\"a\",\"opt5\":\"e\",\"opt4\":4,\"opt2\":2}"}}, true},
    {"qc_nullable_str", "llama.cpp", false, {"set_nullable_str"},
     "<tool_call>\n"
         "<function=set_nullable_str>\n"
         "<parameter=name>\n"
         "hello world\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_nullable_str", "{\"name\":\"hello world\"}"}}, true},
    {"qc_nullable_str_nf", "llama.cpp", false, {"set_nullable_str_nf"},
     "<tool_call>\n"
         "<function=set_nullable_str_nf>\n"
         "<parameter=name>\n"
         "hello world\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_nullable_str_nf", "{\"name\":\"hello world\"}"}}, true},
    {"qc_nullable_int", "llama.cpp", false, {"set_nullable_int"},
     "<tool_call>\n"
         "<function=set_nullable_int>\n"
         "<parameter=count>\n"
         "42\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_nullable_int", "{\"count\":42}"}}, true},
    {"qc_nullable_str_null", "llama.cpp", false, {"set_nullable_str"},
     "<tool_call>\n"
         "<function=set_nullable_str>\n"
         "<parameter=name>\n"
         "null\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_nullable_str", "{\"name\":null}"}}, true},
    {"qc_union", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{\"a\": 1}\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "2 dollars\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_union", "{\"value\":{\"a\": 1},\"amount\":\"2 dollars\"}"}}, true},
    {"qc_union_bad_json", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{not valid json\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "42\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_union", "{\"value\":\"{not valid json\",\"amount\":42}"}}, true},
    {"qc_enum", "llama.cpp", false, {"set_unit"},
     "<tool_call>\n"
         "<function=set_unit>\n"
         "<parameter=unit>\n"
         "celsius\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_unit", "{\"unit\":\"celsius\"}"}}, true},
    {"own_plain_think", "llama.cpp", true, {},
     "Let me think.\nSecond line.\n</think>\n\nThe answer is 4.",
     "Let me think.\nSecond line.\n", "The answer is 4.",
     {}, true},
    {"own_plain_nothink", "llama.cpp", false, {},
     "The answer is 4.",
     "", "The answer is 4.",
     {}, true},
    {"own_nothink_ws", "llama.cpp", false, {},
     "\n\n  Hi there  \n\n",
     "", "Hi there  \n\n",
     {}, true},
    {"own_think_ws_content", "llama.cpp", true, {},
     "  \n reasoning \n\n</think>\n\n\n  Answer  \n\n",
     "reasoning \n\n", "Answer  \n\n",
     {}, true},
    {"own_reasoning_only", "llama.cpp", true, {},
     "Still thinking about it",
     "Still thinking about it", "",
     {}, true},
    {"own_reasoning_only_trailing_ws", "llama.cpp", true, {},
     "Still thinking\n\n",
     "Still thinking\n\n", "",
     {}, true},
    {"own_empty_reasoning_nl", "llama.cpp", true, {},
     "\n</think>\n\nAnswer.",
     "", "Answer.",
     {}, true},
    {"own_empty_reasoning", "llama.cpp", true, {},
     "</think>\n\nAnswer.",
     "", "Answer.",
     {}, true},
    {"own_empty_output_think", "llama.cpp", true, {},
     "",
     "", "",
     {}, true},
    {"own_empty_output_nothink", "llama.cpp", false, {},
     "",
     "", "",
     {}, true},
    {"own_think_close_twice", "llama.cpp", true, {},
     "a\n</think>\n\nb</think>c",
     "a\n", "b</think>c",
     {}, true},
    {"own_toolcall_text_no_tools_think", "llama.cpp", true, {},
     "I could call <tool_call> here\n</think>\n\nok",
     "I could call ", "<tool_call> here\n</think>\n\nok",
     {}, true},
    {"own_toolcall_text_no_tools_nothink", "llama.cpp", false, {},
     "x <tool_call>\n<function=f>\n</function>\n</tool_call> y",
     "", "x <tool_call>\n<function=f>\n</function>\n</tool_call> y",
     {}, true},
    {"own_types_all", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=s>\n"
         "hello\n"
         "</parameter>\n"
         "<parameter=i>\n"
         "42\n"
         "</parameter>\n"
         "<parameter=f>\n"
         "3.14\n"
         "</parameter>\n"
         "<parameter=b>\n"
         "true\n"
         "</parameter>\n"
         "<parameter=o>\n"
         "{\"a\": [1, 2], \"b\": null}\n"
         "</parameter>\n"
         "<parameter=a>\n"
         "[1, \"x\", false]\n"
         "</parameter>\n"
         "<parameter=n>\n"
         "null\n"
         "</parameter>\n"
         "<parameter=any>\n"
         "free text\n"
         "</parameter>\n"
         "<parameter=e>\n"
         "2\n"
         "</parameter>\n"
         "<parameter=ref>\n"
         "7\n"
         "</parameter>\n"
         "<parameter=anyof>\n"
         "false\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"s\":\"hello\",\"i\":42,\"f\":3.14,\"b\":true,\"o\":{\"a\": [1, 2], \"b\": null},\"a\":[1, \"x\", false],\"n\":null,\"any\":\"free text\",\"e\":2,\"ref\":7,\"anyof\":false}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_types_any_json", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=any>\n"
         "7\n"
         "</parameter>\n"
         "<parameter=e>\n"
         "x\n"
         "</parameter>\n"
         "<parameter=anyof>\n"
         "maybe\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"any\":7,\"e\":\"x\",\"anyof\":\"maybe\"}"}}, true},
    {"own_types_any_obj", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=any>\n"
         "{\"k\": 'v', \"t\": True}\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"any\":\"{\\\"k\\\": 'v', \\\"t\\\": True}\"}"}}, true},
    {"own_types_float_exp", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=f>\n"
         "-1.5e+3\n"
         "</parameter>\n"
         "<parameter=i>\n"
         "0\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"f\":-1.5e+3,\"i\":0}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_int_not_json", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=i>\n"
         "abc\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"i\":\"abc\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_int_leading_space", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=i>\n"
         " 42\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"i\":\" 42\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_int_trailing_space", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=i>\n"
         "42 \n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"i\":\"42 \"}"}}, true},
    {"own_int_as_jsonstring", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=i>\n"
         "\"42\"\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"i\":\"42\"}"}}, true},
    {"own_str_lt", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "if a < b and c </parameter> d:\n"
         "  x = '<parameter=code>'\n"
         "</parameter>x\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"if a < b and c </parameter> d:\\n  x = '<parameter=code>'\\n</parameter>x\"}"}}, true},
    {"own_str_escapes", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "q=\"x\"\\t\tb\\\\ \x01 é 😀 /\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"q=\\\"x\\\"\\\\t\\tb\\\\\\\\ \\u0001 é 😀 /\"}"}}, true},
    {"own_str_empty", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"\"}"}}, true},
    {"own_str_multiline_unicode", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "héllo\n"
         "世界 🌍\n"
         "\n"
         "  line3  \n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"héllo\\n世界 🌍\\n\\n  line3  \"}"}}, true},
    {"own_str_crlf", "llama.cpp", false, {"python"},
     "<tool_call>\n"
         "<function=python>\n"
         "<parameter=code>\n"
         "a\r\n"
         "b\r\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"python", "{\"code\":\"a\\r\\nb\\r\"}"}}, true},
    {"own_three_calls", "llama.cpp", true, {"get_weather", "read_file"},
     "Doing three.\n"
         "</think>\n"
         "\n"
         "Ok.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Rome\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=read_file>\n"
         "<parameter=path>\n"
         "/tmp/a b.txt\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "Doing three.\n", "Ok.\n",
     {{"get_weather", "{\"city\":\"Paris\"}"}, {"get_weather", "{\"city\":\"Rome\",\"days\":3}"}, {"read_file", "{\"path\":\"/tmp/a b.txt\"}"}}, true},
    {"own_content_between_calls", "llama.cpp", false, {"get_weather", "read_file"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "Between\n"
         "<tool_call>\n"
         "<function=read_file>\n"
         "<parameter=path>\n"
         "x\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_content_after_calls", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "Done.",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_ws_after_calls", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "\n",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_unknown_tool", "llama.cpp", false, {"get_weather"},
     "Text\n"
         "<tool_call>\n"
         "<function=nope>\n"
         "<parameter=x>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "Text\n",
     {}, true},
    {"own_unknown_tool_then_known", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=nope>\n"
         "<parameter=x>\n"
         "1\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_unknown_param", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=country>\n"
         "FR\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\",\"country\":\"FR\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_missing_required", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=days>\n"
         "2\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"days\":2}"}}, true},
    {"own_dup_optional", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "2\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":2,\"days\":3}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_dup_required", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=city>\n"
         "Rome\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\",\"city\":\"Rome\"}"}}, true},
    {"own_call_in_reasoning", "llama.cpp", true, {"get_weather"},
     "I will check.\n"
         "Let me call <tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "I will check.\nLet me call ", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_call_no_trailing_nl_func", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function></tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_call_spaces_between", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>  \n"
         " \n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Rome\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}, {"get_weather", "{\"city\":\"Rome\"}"}}, true},
    {"own_call_no_nl_after_open", "llama.cpp", false, {"get_weather"},
     "<tool_call><function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_param_no_nl", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>Paris</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{}"}}, true},
    {"own_eos_00", "llama.cpp", true, {"get_weather"},
     "Hm.\n</think>\n\nSure.\n<tool_call>",
     "Hm.\n", "Sure.\n",
     {}, true},
    {"own_eos_01", "llama.cpp", true, {"get_weather"},
     "Hm.\n</think>\n\nSure.\n<tool_call>\n",
     "Hm.\n", "Sure.\n",
     {}, true},
    {"own_eos_02", "llama.cpp", true, {"get_weather"},
     "Hm.\n</think>\n\nSure.\n<tool_call>\n<function=get_w",
     "Hm.\n", "Sure.\n",
     {}, true},
    {"own_eos_03", "llama.cpp", true, {"get_weather"},
     "Hm.\n</think>\n\nSure.\n<tool_call>\n<function=get_weather>",
     "Hm.\n", "Sure.\n",
     {}, true},
    {"own_eos_04", "llama.cpp", true, {"get_weather"},
     "Hm.\n</think>\n\nSure.\n<tool_call>\n<function=get_weather>\n",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{"}}, true},
    {"own_eos_05", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=ci",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{"}}, true},
    {"own_eos_06", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\""}}, true},
    {"own_eos_07", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Par",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Par"}}, true},
    {"own_eos_08", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</param",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris"}}, true},
    {"own_eos_09", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\""}}, true},
    {"own_eos_10", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":3"}}, true},
    {"own_eos_11", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":3"}}, true},
    {"own_eos_12", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n"
         "</function>",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":3"}}, true},
    {"own_eos_13", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n"
         "</function>\n",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":3}"}}, true},
    {"own_eos_14", "llama.cpp", true, {"get_weather"},
     "Hm.\n"
         "</think>\n"
         "\n"
         "Sure.\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "<parameter=days>\n"
         "3\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "Hm.\n", "Sure.\n",
     {{"get_weather", "{\"city\":\"Paris\",\"days\":3}"}}, true},
    {"own_eos_types_obj", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=o>\n{\"a\": [1, 2",
     "", "",
     {{"types", "{\"o\":{\"a\": [1, 2"}}, true},
    {"own_eos_types_str", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=s>\nabc\n",
     "", "",
     {{"types", "{\"s\":\"abc"}}, true},
    {"own_eos_types_float", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=f>\n3.",
     "", "",
     {{"types", "{\"f\":3."}}, true},
    {"own_eos_in_think_close", "llama.cpp", true, {},
     "abc\n</thi",
     "abc\n", "",
     {}, true},
    {"own_eos_in_content_tag", "llama.cpp", true, {"get_weather"},
     "x\n</think>\n\nHello <tool_",
     "x\n", "Hello ",
     {}, true},
    {"own_partial_tag_in_content", "llama.cpp", true, {"get_weather"},
     "x\n</think>\n\nHello <tool_ end",
     "x\n", "Hello <tool_ end",
     {}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_python_literals", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=o>\n"
         "{'a': True, 'b': None, 'c': 'it''s'}\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"o\":\"{'a': True, 'b': None, 'c': 'it''s'}\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_optional_before_required", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=days>\n"
         "2\n"
         "</parameter>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"days\":2,\"city\":\"Paris\"}"}}, true},
    {"own_eos_union_obj", "llama.cpp", false, {"set_union"},
     "<tool_call>\n<function=set_union>\n<parameter=value>\n{\"a\": 1",
     "", "",
     {{"set_union", "{\"value\":"}}, true},
    {"own_eos_union_str", "llama.cpp", false, {"set_union"},
     "<tool_call>\n<function=set_union>\n<parameter=value>\nhello wor",
     "", "",
     {{"set_union", "{\"value\":\"hello wor"}}, true},
    {"own_eos_union_num", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{\"a\": 1}\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "4",
     "", "",
     {{"set_union", "{\"value\":{\"a\": 1},\"amount\":"}}, true},
    {"own_eos_union_numstr", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{\"a\": 1}\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "2 d",
     "", "",
     {{"set_union", "{\"value\":{\"a\": 1},\"amount\":\"2 d"}}, true},
    {"own_eos_union_close", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{\"a\": 1}\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "42\n"
         "</par",
     "", "",
     {{"set_union", "{\"value\":{\"a\": 1},\"amount\":"}}, true},
    {"own_union_obj_trailing", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "{\"a\": 1} trailing\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "-0.5e2\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_union", "{\"value\":\"{\\\"a\\\": 1} trailing\",\"amount\":-0.5e2}"}}, true},
    {"own_union_jsonstring", "llama.cpp", false, {"set_union"},
     "<tool_call>\n"
         "<function=set_union>\n"
         "<parameter=value>\n"
         "\"quoted\"\n"
         "</parameter>\n"
         "<parameter=amount>\n"
         "[1]\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"set_union", "{\"value\":\"\\\"quoted\\\"\",\"amount\":\"[1]\"}"}}, true},
    {"own_eos_json_backslash", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=o>\n{\"k\": \"x\\",
     "", "",
     {{"types", "{\"o\":{\"k\": \"x"}}, true},
    {"own_eos_json_uescape", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=o>\n{\"k\": \"x\\u00",
     "", "",
     {{"types", "{\"o\":{\"k\": \"x"}}, true},
    {"own_eos_json_int_close", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=i>\n42\n</parameter>",
     "", "",
     {{"types", "{\"i\":42"}}, true},
    {"own_eos_json_null", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=n>\nnu",
     "", "",
     {{"types", "{\"n\":nu"}}, true},
    {"own_eos_json_bool", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=b>\ntr",
     "", "",
     {{"types", "{\"b\":tr"}}, true},
    {"own_eos_json_arr", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=a>\n[1, ",
     "", "",
     {{"types", "{\"a\":[1, "}}, true},
    {"own_eos_json_minus", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=f>\n-",
     "", "",
     {{"types", "{\"f\":-"}}, true},
    {"own_json_ws_inside", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=o>\n"
         "{ \"a\" :\t[ 1 ,\n"
         "2 ] ,\"b\":{}}\n"
         "</parameter>\n"
         "<parameter=a>\n"
         "[]\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"o\":{ \"a\" :\t[ 1 ,\n2 ] ,\"b\":{}},\"a\":[]}"}}, true},
    // llama.cpp: content '', 1 call(s) {"o":{"a": "x\n</parameter>\ny"}}
    {"own_json_raw_newline_in_string", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=o>\n"
         "{\"a\": \"x\n"
         "</parameter>\n"
         "y\"}\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"o\":\"{\\\"a\\\": \\\"x\"}"}}, true},
    {"own_json_number_forms", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=f>\n"
         "0\n"
         "</parameter>\n"
         "<parameter=i>\n"
         "-0\n"
         "</parameter>\n"
         "<parameter=a>\n"
         "[0.5, 1E3, -2e-2, 10]\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"f\":0,\"i\":-0,\"a\":[0.5, 1E3, -2e-2, 10]}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_json_bad_number", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=f>\n"
         "01\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"f\":\"01\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_json_bad_number2", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=f>\n"
         "1.\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"f\":\"1.\"}"}}, true},
    {"own_json_escapes_ok", "llama.cpp", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=o>\n"
         "{\"q\": \"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\\u00e9\"}\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"o\":{\"q\": \"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\\u00e9\"}}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_json_bad_escape", "policy", false, {"types"},
     "<tool_call>\n"
         "<function=types>\n"
         "<parameter=o>\n"
         "{\"q\": \"a\\x\"}\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"types", "{\"o\":\"{\\\"q\\\": \\\"a\\\\x\\\"}\"}"}}, true},
    // llama.cpp: content '', 0 call(s)
    {"own_param_name_space", "policy", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter= city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_two_calls_eos_second", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Ro",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}, {"get_weather", "{\"city\":\"Ro"}}, true},
    {"own_call_then_bad_call", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "<tool_call>\n"
         "<function=nope>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_call_then_text_then_call", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Paris\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>\n"
         "x<tool_call>\n"
         "<function=get_weather>\n"
         "<parameter=city>\n"
         "Rome\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"get_weather", "{\"city\":\"Paris\"}"}}, true},
    {"own_lt_in_content", "llama.cpp", true, {"get_weather"},
     "x\n</think>\n\na <b> c </tool> d <tool_call",
     "x\n", "a <b> c </tool> d ",
     {}, true},
    {"own_lt_in_reasoning", "llama.cpp", true, {},
     "a < b </th ink> <tool_ c\n</think>\n\nok",
     "a < b </th ink> <tool_ c\n", "ok",
     {}, true},
    {"own_tool_response_tag", "llama.cpp", true, {"get_weather"},
     "x\n</think>\n\n<tool_response>\nhi\n</tool_response>",
     "x\n", "<tool_response>\nhi\n</tool_response>",
     {}, true},
    {"own_think_tag_in_content_nothink", "llama.cpp", false, {},
     "a</think>b<think>c",
     "", "a</think>b<think>c",
     {}, true},
    {"own_name_prefix", "llama.cpp", false, {"get_weather"},
     "<tool_call>\n<function=get>\n</function>\n</tool_call>",
     "", "",
     {}, true},
    {"raw_invalid_content", "llama.cpp", true, {},
     "x\n</think>\n\nab\xff" "cd\x80" "e",
     "x\n", "ab\xef\xbf\xbd" "cd\xef\xbf\xbd" "e",
     {}, false},
    {"raw_invalid_reasoning", "llama.cpp", true, {},
     "a\xc3(b\xe4\xb8" "a\n</think>\n\nc",
     "a\xef\xbf\xbd(b\xef\xbf\xbd" "a\n", "c",
     {}, false},
    {"raw_incomplete_end", "llama.cpp", true, {},
     "x\n</think>\n\nab\xe4\xb8",
     "x\n", "ab",
     {}, false},
    {"raw_lt_incomplete_end", "llama.cpp", true, {"get_weather"},
     "x\n</think>\n\nab<\xe4",
     "x\n", "ab",
     {}, false},
    {"raw_lax_utf8", "llama.cpp", true, {},
     "x\n</think>\n\n\xc0\x80 \xed\xa0\x80 \xf5\x80\x80\x80 \xf8 z",
     "x\n", "\xc0\x80 \xed\xa0\x80 \xf5\x80\x80\x80 \xef\xbf\xbd z",
     {}, false},
    {"raw_param_incomplete_eos", "llama.cpp", false, {"read_file"},
     "<tool_call>\n<function=read_file>\n<parameter=path>\nab\xe4\xb8",
     "", "",
     {{"read_file", "{\"path\":\"ab"}}, false},
    {"raw_json_incomplete_eos", "llama.cpp", false, {"types"},
     "<tool_call>\n<function=types>\n<parameter=o>\n{\"k\": \"ab\xe4\xb8",
     "", "",
     {{"types", "{\"o\":{\"k\": \"ab"}}, false},
    // llama.cpp: reasoning b'', content b'<think>\nabc</thi', 0 call(s)
    {"raw_quirk_reasoning_tail", "policy", true, {},
     "abc</thi\xe4",
     "abc", "",
     {}, false},
    // llama.cpp: reasoning b'', content b'', 0 call(s)
    {"raw_quirk_param_tail", "policy", false, {"read_file"},
     "<tool_call>\n<function=read_file>\n<parameter=path>\nab\n\xe4",
     "", "",
     {{"read_file", "{\"path\":\"ab"}}, false},
    // llama.cpp throws: [json.exception.type_error.316] invalid UTF-8 byte at index 
    {"raw_param_invalid", "policy", false, {"read_file"},
     "<tool_call>\n"
         "<function=read_file>\n"
         "<parameter=path>\n"
         "a\xff" "b\n"
         "</parameter>\n"
         "</function>\n"
         "</tool_call>",
     "", "",
     {{"read_file", "{\"path\":\"a\xef\xbf\xbd" "b\"}"}}, false},
};
// ---- fixtures (generated) end
// clang-format on

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& what) {
  if (ok) {
    g_pass++;
    return;
  }
  g_fail++;
  printf("FAIL %s\n", what.c_str());
}

std::string show(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20 || c == 0x7f) {
      char b[8];
      snprintf(b, sizeof b, "\\x%02x", c);
      o += b;
    } else o += static_cast<char>(c);
  }
  return o;
}

ojson tools_json(const std::vector<const char*>& names) {
  if (names.empty()) return ojson();
  ojson a = ojson::array();
  for (const char* n : names)
    for (const auto& [k, v] : kToolDefs)
      if (std::string(k) == n) a.push_back(ojson::parse(v));
  return a;
}

// What a client builds from the deltas.
struct Stream {
  std::string reasoning, content;
  std::vector<std::array<std::string, 3>> calls;  // id, name, arguments
  std::vector<std::string> errors;

  void apply(const ParseDelta& d) {
    reasoning += d.reasoning;
    content += d.content;
    for (const auto& t : d.tool_calls) {
      if (t.index == static_cast<int>(calls.size())) {
        if (t.id.empty() || t.name.empty())
          errors.push_back("first delta of call " + std::to_string(t.index) + " has no id or name");
        calls.push_back({t.id, t.name, t.arguments});
      } else if (t.index >= 0 && t.index < static_cast<int>(calls.size())) {
        if (!t.id.empty() || !t.name.empty())
          errors.push_back("a later delta of call " + std::to_string(t.index) + " repeats id or name");
        calls[t.index][2] += t.arguments;
      } else {
        errors.push_back("delta for call " + std::to_string(t.index) + " out of order");
      }
    }
  }
};

bool id_ok(const std::string& id) {
  if (id.size() != 32) return false;
  for (char c : id)
    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) return false;
  return true;
}

enum class Feed { TOKENS, BYTES, CHUNKS };

void run_case(const Tokenizer& tok, const Case& c, Feed feed, unsigned seed) {
  const char* how = feed == Feed::TOKENS ? "tokens" : feed == Feed::BYTES ? "bytes" : "chunks";
  std::string tag = std::string(c.name) + " [" + how + "]";
  StreamParser p(tok, c.thinking, tools_json(c.tools));
  Stream s;
  if (feed == Feed::TOKENS) {
    std::vector<int> ids = tok.encode(c.text, true);
    if (tok.decode(ids) != c.text) {
      check(false, tag + ": the tokens do not decode to the text");
      return;
    }
    for (int id : ids) s.apply(p.push(id));
  } else if (feed == Feed::BYTES) {
    for (char ch : c.text) s.apply(p.push_text(std::string_view(&ch, 1)));
  } else {
    std::mt19937 rng(seed);
    for (size_t i = 0; i < c.text.size();) {
      size_t n = std::min<size_t>(c.text.size() - i, 1 + rng() % 17);
      s.apply(p.push_text(std::string_view(c.text).substr(i, n)));
      i += n;
    }
  }
  s.apply(p.finish());

  std::string err;
  for (const auto& e : s.errors) err += " " + e + ";";
  // deltas == final values
  if (s.reasoning != p.reasoning()) err += " deltas give reasoning '" + show(s.reasoning) + "';";
  if (s.content != p.content()) err += " deltas give content '" + show(s.content) + "';";
  ojson calls = p.tool_calls();
  if (s.calls.size() != calls.size()) {
    err += " deltas give " + std::to_string(s.calls.size()) + " calls, final has " + std::to_string(calls.size()) + ";";
  } else {
    for (size_t i = 0; i < calls.size(); i++) {
      const ojson& fc = calls[i];
      if (fc["type"] != "function") err += " call type;";
      if (s.calls[i][0] != fc["id"].get<std::string>()) err += " id differs from the delta's;";
      if (s.calls[i][1] != fc["function"]["name"].get<std::string>()) err += " name differs from the delta's;";
      if (s.calls[i][2] != fc["function"]["arguments"].get<std::string>()) err += " deltas give other arguments;";
    }
  }
  // final values == expected
  if (p.reasoning() != c.reasoning)
    err += " reasoning '" + show(p.reasoning()) + "', expected '" + show(c.reasoning) + "';";
  if (p.content() != c.content) err += " content '" + show(p.content()) + "', expected '" + show(c.content) + "';";
  if (calls.size() != c.calls.size()) {
    err += " " + std::to_string(calls.size()) + " calls, expected " + std::to_string(c.calls.size()) + ";";
  } else {
    for (size_t i = 0; i < calls.size(); i++) {
      std::string name = calls[i]["function"]["name"], args = calls[i]["function"]["arguments"];
      if (name != c.calls[i].first) err += " call " + std::to_string(i) + " name '" + name + "';";
      if (args != c.calls[i].second)
        err += " call " + std::to_string(i) + " arguments '" + show(args) + "', expected '" + show(c.calls[i].second) + "';";
    }
  }
  // ids, JSON validity
  std::set<std::string> ids;
  int complete = p.complete_tool_calls();
  if (complete < static_cast<int>(calls.size()) - 1) err += " more than one unfinished call;";
  for (size_t i = 0; i < calls.size(); i++) {
    std::string id = calls[i]["id"], args = calls[i]["function"]["arguments"];
    if (!id_ok(id)) err += " bad id '" + id + "';";
    ids.insert(id);
    bool valid = ojson::accept(args);
    if (static_cast<int>(i) < complete && !valid) err += " call " + std::to_string(i) + " is complete but not valid JSON;";
    if (static_cast<int>(i) >= complete && valid) err += " call " + std::to_string(i) + " is cut but valid JSON;";
  }
  if (ids.size() != calls.size()) err += " ids not unique;";
  check(err.empty(), tag + ":" + err);
}

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void unit_checks(const Tokenizer& tok) {
  ojson weather = tools_json({"get_weather"});

  // Control tokens give no text (llama.cpp's token_to_piece with special = false); user-defined ones do.
  {
    StreamParser p(tok, false, ojson());
    Stream s;
    int im_start = tok.find("<|im_start|>"), think = tok.find("<think>");
    check(im_start >= 0 && think >= 0, "unit: special tokens found");
    for (int id : tok.encode("Hello", true)) s.apply(p.push(id));
    s.apply(p.push(im_start));
    s.apply(p.push(think));
    for (int id : tok.encode(" world", true)) s.apply(p.push(id));
    s.apply(p.finish());
    check(p.content() == "Hello<think> world" && s.content == p.content(),
          "unit: control token ignored, user-defined token kept: '" + show(p.content()) + "'");
  }
  // finish() twice, push after finish
  {
    StreamParser p(tok, true, weather);
    p.push_text("abc\n</think>\n\nx");
    ParseDelta a = p.finish(), b = p.finish(), c = p.push_text("more");
    check(b.empty() && c.empty() && p.content() == "x", "unit: finish twice / push after finish");
  }
  // tools = [] is the same as no tools
  {
    StreamParser p(tok, false, ojson::array());
    p.push_text("a<tool_call>\n<function=get_weather>\n</function>\n</tool_call>");
    p.finish();
    check(p.content() == "a<tool_call>\n<function=get_weather>\n</function>\n</tool_call>" && p.tool_calls().empty(),
          "unit: tools [] parses no calls");
  }
  // the first delta of a call comes as soon as the name line is complete, with id, name and "{"
  {
    StreamParser p(tok, false, weather);
    ParseDelta d1 = p.push_text("<tool_call>\n<function=get_weather>");
    ParseDelta d2 = p.push_text("\n");
    ParseDelta d3 = p.push_text("<parameter=city>\nPa");
    check(d1.tool_calls.empty() && d2.tool_calls.size() == 1 && d2.tool_calls[0].arguments == "{" &&
              d2.tool_calls[0].name == "get_weather" && id_ok(d2.tool_calls[0].id) && d3.tool_calls.size() == 1 &&
              d3.tool_calls[0].id.empty() && d3.tool_calls[0].arguments == "\"city\":\"Pa",
          "unit: header delta timing");
  }
  // a string value streams; a held-back "\n" is released when it is not the close tag
  {
    StreamParser p(tok, false, weather);
    p.push_text("<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n");
    ParseDelta d = p.push_text("B");
    check(d.tool_calls.size() == 1 && d.tool_calls[0].arguments == "\\nB", "unit: held newline released");
  }
}

void speed(const Tokenizer& tok) {
  // ~20k tokens: long reasoning, some content, one call with a long string value.
  std::string reasoning, code;
  for (int i = 0; reasoning.size() < 26000; i++)
    reasoning += "Step " + std::to_string(i) + ": the value of x < y holds, so we check the <next> case and "
                 "compare it with the previous one (\xC3\xA9t\xC3\xA9, \xE4\xB8\x96\xE7\x95\x8C).\n";
  for (int i = 0; code.size() < 27000; i++)
    code += "    if (a[" + std::to_string(i) + "] < b && c > d) { printf(\"%d\\n\", x); }  // </param\n";
  std::string text = reasoning + "\n</think>\n\nI will write the file now.\n<tool_call>\n<function=write_file>\n"
                     "<parameter=path>\nsrc/big.c\n</parameter>\n<parameter=content>\n" + code +
                     "\n</parameter>\n</function>\n</tool_call>";
  ojson tools = ojson::parse(
      R"([{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":)"
      R"({"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}])");
  std::vector<int> ids = tok.encode(text, true);
  double best = 1e9;
  std::string args;
  for (int rep = 0; rep < 5; rep++) {
    StreamParser p(tok, true, tools);
    double t0 = now_s();
    for (int id : ids) p.push(id);
    p.finish();
    best = std::min(best, now_s() - t0);
    args = p.tool_calls()[0]["function"]["arguments"];
  }
  ojson parsed = ojson::parse(args);
  check(parsed["content"].get<std::string>() == code && parsed["path"] == "src/big.c", "speed: arguments round trip");
  printf("speed: %zu tokens (%zu bytes) in %.2f ms = %.3f us/token\n", ids.size(), text.size(), best * 1e3,
         best * 1e6 / ids.size());
  check(ids.size() >= 20000, "speed: message has at least 20k tokens");
  check(best * 1e6 / ids.size() < 5.0, "speed: under 5 us per token");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_parser <model.gguf>\n");
    return 2;
  }
  GGUF gguf(argv[1]);
  Tokenizer tok(gguf);

  int n_llama = 0, n_policy = 0;
  for (const auto& c : kCases) {
    (std::string(c.source) == "policy" ? n_policy : n_llama)++;
    if (c.tokens) run_case(tok, c, Feed::TOKENS, 0);
    run_case(tok, c, Feed::BYTES, 0);
    for (unsigned seed = 1; seed <= 3; seed++) run_case(tok, c, Feed::CHUNKS, seed);
  }
  printf("cases: %d (llama.cpp %d, policy %d)\n", n_llama + n_policy, n_llama, n_policy);
  unit_checks(tok);
  speed(tok);
  printf("PASS %d FAIL %d\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
