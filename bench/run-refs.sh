#!/bin/bash
# Long-context llama.cpp references (A2). Run from the repo root in Git Bash.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PROD="${Q27_PROD:-../qwen38_27}"  # llama.cpp build in $PROD/bin-parches, model in $PROD/models
export PATH="$PROD/bin-parches:$PATH"
m="${Q27_MODEL:-$PROD/models/Qwen3.8-27B/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
for n in "$@"; do
  ./bench/build/llama_ref.exe "$m" bench/out/long-text.txt bench/out/ref_${n}_ub4.bin $n 1024 4 2>&1 | grep -E "^done|tokens,|error|failed" | tail -3
done
