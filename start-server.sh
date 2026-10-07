#!/usr/bin/env bash
# Start the q27 engine server on Linux: Qwen3.8-27B on the two cards, OpenAI-compatible API (like llama-server).
# Usage:  ./start-server.sh [--port 8081] [--bind 127.0.0.1] [--ctx 0] [--effort medium] [--kv q8_0|f16]
#                           [--model FILE] [--mmproj FILE|none] [--no-vision] [--log-dir DIR] [--api-key KEY]
#                           [-- more q27_server options, e.g. --cache-ram 4096]
#         --bind 0.0.0.0 makes it reachable over the LAN / VPN.
#         The API key comes from --api-key or the environment variable Q27_API_KEY. Without one the server is open.
# Same options as start-server.ps1. Stop it with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"

port=8081; bind=127.0.0.1; ctx=0; api_key=${Q27_API_KEY:-}; effort=medium; kv=q8_0
model=""; mmproj=""; no_vision=0; log_dir=""; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port=$2; shift 2 ;;
    --bind) bind=$2; shift 2 ;;
    --ctx) ctx=$2; shift 2 ;;
    --api-key) api_key=$2; shift 2 ;;
    --effort) effort=$2; shift 2 ;;      # reasoning effort default: xhigh, medium or low
    --kv) kv=$2; shift 2 ;;              # KV cache type: q8_0 (default) or f16
    --model) model=$2; shift 2 ;;
    --mmproj) mmproj=$2; shift 2 ;;      # image encoder of the same model; "none" = text only
    --no-vision) no_vision=1; shift ;;
    --log-dir) log_dir=$2; shift 2 ;;    # write every request and result as JSON files here
    --) shift; extra=("$@"); break ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done
case "$kv" in q8_0|f16) ;; *) echo "--kv must be q8_0 or f16" >&2; exit 1 ;; esac

# Model files: --model / --mmproj, else $Q27_MODEL / $Q27_MMPROJ, else models/ in this repo, else a
# qwen38_27/models/Qwen3.8-27B folder next to this repo.
find_model_file() {  # <given> <env var name> <file name>
  if [ -n "$1" ]; then echo "$1"; return; fi
  local e=${!2:-}
  if [ -n "$e" ]; then echo "$e"; return; fi
  local d
  for d in models ../qwen38_27/models/Qwen3.8-27B; do
    if [ -f "$d/$3" ]; then realpath "$d/$3"; return; fi
  done
  echo "models/$3"
}
model=$(find_model_file "$model" Q27_MODEL Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf)
if [ "$no_vision" = 1 ] || [ "$mmproj" = none ]; then mmproj=""
else mmproj=$(find_model_file "$mmproj" Q27_MMPROJ mmproj-Qwen3.8-27B-BF16.gguf); fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # same card order as nvidia-smi
if [ "$kv" = f16 ]; then export Q27_KV=f16; else unset Q27_KV; fi

exe=./build/q27_server
for f in "$exe" "$model"; do
  if [ ! -e "$f" ]; then echo "MISSING: $f" >&2; exit 1; fi
done

# Another server would hold the VRAM. This script does not stop it.
if pgrep -x 'llama-server|q27_server' >/dev/null; then
  pgrep -a -x 'llama-server|q27_server' | sed 's/^/already running: /' >&2
  echo "Stop it first, for example:  pkill -x llama-server" >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | sed 's/^/  /'
n_cards=$(nvidia-smi -L | grep -c '^GPU')
if [ "$n_cards" -lt 2 ]; then echo "Only $n_cards card found. The engine needs both." >&2; exit 1; fi

args=(-m "$model" --host "$bind" --port "$port" --reasoning-effort "$effort")
vis="text only"
if [ -n "$mmproj" ] && [ -f "$mmproj" ]; then
  # images: encoder on card 1; small images are scaled up to at least 1024 tokens, as production does
  args+=(--mmproj "$mmproj" --image-min-tokens 1024)
  vis="vision on"
fi
if [ "$ctx" -gt 0 ]; then args+=(--ctx "$ctx"); fi
if [ -n "$log_dir" ]; then args+=(--log-dir "$log_dir"); fi
if [ -n "$api_key" ]; then export Q27_API_KEY=$api_key; fi

echo "starting q27_server on http://$bind:$port (KV $kv, effort $effort, $vis)"
exec "$exe" "${args[@]}" "${extra[@]}"
