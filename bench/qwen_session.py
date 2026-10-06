# Run one real Qwen Code agent session against the test server (port 8081), headless, in a scratch folder.
# The user's own Qwen Code config (~\.qwen) is not read or changed: QWEN_HOME points to a scratch home.
# Usage: .venv\Scripts\python.exe bench\qwen_session.py --yolo [--name run1] [--prompt "..."] [--key KEY]
# Output: bench\out\qwen\<name>\ with work\ (the agent's folder), logs\ (one JSON per API request, written by
# Qwen Code), stream.jsonl (Qwen Code's stream-json output) and a printed summary.
# WARNING: a headless session cannot answer approval prompts, so Qwen Code runs with --approval-mode yolo: the model
# can run any shell command and edit any file this user can reach, not only the scratch folder. The script refuses
# to start without --yolo. Use it only with prompts you trust, on a machine where that is acceptable.
import argparse, glob, json, os, shutil, subprocess, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Q = os.environ.get("QWEN_CODE_DIR", os.path.join(os.environ.get("LOCALAPPDATA", ""), "qwen-code", "qwen-code"))
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT = ("Create a Python module stats.py with functions mean, median and stdev (population standard "
                  "deviation), without imports. Then write test_stats.py with unittest tests for all three functions, "
                  "including an even-length list for median. Run the tests with python, fix any failure, and report "
                  "the final test output.")

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="run1")
ap.add_argument("--prompt", default=DEFAULT_PROMPT)
ap.add_argument("--key", default=os.environ.get("BENCH_KEY", "local-test"))
ap.add_argument("--url", default="http://127.0.0.1:8081/v1")
ap.add_argument("--minutes", type=int, default=45)
ap.add_argument("--effort", default="", help="low|medium|high|xhigh: turn on Qwen Code's openai-effort profile")
ap.add_argument("--yolo", action="store_true",
                help="required: accept that Qwen Code approves every tool call (any command, any file) by itself")
a = ap.parse_args()
if not a.yolo:
    sys.exit("qwen_session.py runs Qwen Code with --approval-mode yolo (the model can run any command and edit any "
             "file without asking). Add --yolo to accept that.")

T = os.path.join(HERE, "out", "qwen", a.name)
if os.path.exists(T):
    shutil.rmtree(T)
for d in ("home", "work", "logs"):
    os.makedirs(os.path.join(T, d))
subprocess.run(["git", "-C", os.path.join(T, "work"), "init", "-q"], check=True)
settings = {
    "$version": 4,
    "general": {"enableAutoUpdate": False},
    "security": {"auth": {"selectedType": "openai"}},
    "model": {"name": "qwen3.8-27b-local", "reasoningEffort": "xhigh"},
    "modelProviders": {"openai": [{
        "id": "qwen3.8-27b-local", "name": "qwen27-engine (test)", "envKey": "Q27_API_KEY", "baseUrl": a.url,
        "capabilities": {"vision": True, "agent": True},
        "generationConfig": {"timeout": 900000, "streamIdleTimeoutMs": 600000, "maxRetries": 1,
                             "contextWindowSize": 180224, "modalities": {"image": True},
                             "samplingParams": {"max_tokens": 32768}}}]},
}
if a.effort:
    # Qwen Code then sends "reasoning_effort": <model.reasoningEffort> on main requests and "none" on background ones
    settings["model"]["reasoningEffort"] = a.effort
    settings["modelProviders"]["openai"][0]["capabilities"]["reasoning"] = {
        "profile": "openai-effort", "efforts": ["low", "medium", "high", "xhigh"], "defaultEffort": "medium"}
with open(os.path.join(T, "home", "settings.json"), "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2)

env = dict(os.environ, QWEN_HOME=os.path.join(T, "home"), Q27_API_KEY=a.key, QWEN_STREAM_MAX_LIFETIME_MS="0",
           QWEN_CODE_SKIP_UPDATE_CHECK_ONCE="true", QWEN_DISABLE_AUTO_TITLE="1", QWEN_CODE_SUPPRESS_YOLO_WARNING="1",
           QWEN_CODE_LAUNCHER_PATH=os.path.join(Q, "bin", "qwen.cmd"))
cmd = [os.path.join(Q, "node", "node.exe"), os.path.join(Q, "lib", "cli-entry.js"), "--approval-mode", "yolo",
       "-o", "stream-json", "--openai-logging", "--openai-logging-dir", os.path.join(T, "logs"),
       "--max-session-turns", "60", "--max-wall-time", f"{a.minutes}m"]
t0 = time.time()
r = subprocess.run(cmd, input=a.prompt, text=True, encoding="utf-8", capture_output=True, cwd=os.path.join(T, "work"), env=env)
dt = time.time() - t0
with open(os.path.join(T, "stream.jsonl"), "w", encoding="utf-8") as f:
    f.write(r.stdout)
with open(os.path.join(T, "stderr.txt"), "w", encoding="utf-8") as f:
    f.write(r.stderr)

# summary
events = []
for line in r.stdout.splitlines():
    try:
        events.append(json.loads(line))
    except Exception:
        pass
result = next((e for e in reversed(events) if e.get("type") == "result"), {})
tools = []
for e in events:
    if e.get("type") == "assistant":
        for part in e.get("message", {}).get("content", []):
            if part.get("type") == "tool_use":
                tools.append(part.get("name"))
logs = sorted(glob.glob(os.path.join(T, "logs", "*.json")))
errors = 0
for p in logs:
    try:
        if json.load(open(p, encoding="utf-8")).get("error"):
            errors += 1
    except Exception:
        pass
print(f"exit code {r.returncode}, {dt:.0f} s, {len(logs)} API requests ({errors} with errors), tool calls: {len(tools)}")
print("tools used:", ", ".join(tools))
print("result:", result.get("subtype"), "|", str(result.get("result", ""))[:600])
print("files in work:", sorted(os.listdir(os.path.join(T, "work"))))
print("folder:", T)
sys.exit(0 if r.returncode == 0 and result.get("subtype") == "success" and errors == 0 else 1)
