#!/usr/bin/env bash
# Build the engine on Linux: gcc + CUDA 13.4 + CMake + Ninja. Output in build/ (or Q27_BUILD_DIR).
#   Q27_CUDA       CUDA 13.4 folder (default: /usr/local/cuda-13.4, else /usr/local/cuda)
#   Q27_BUILD_DIR  build folder (default: build next to this script)
# Extra arguments go to "cmake --build", e.g. "./build.sh --target q27_server".
# CUDA 13.2 miscompiles IQ3_S on sm_120 (llama.cpp PR #27902); CMakeLists.txt refuses anything older than 13.4.
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${Q27_CUDA:-}
if [ -z "$CUDA" ]; then
  if [ -x /usr/local/cuda-13.4/bin/nvcc ]; then CUDA=/usr/local/cuda-13.4; else CUDA=/usr/local/cuda; fi
fi
if [ ! -x "$CUDA/bin/nvcc" ]; then
  echo "CUDA 13.4 not found in $CUDA. Set Q27_CUDA to the CUDA 13.4 folder." >&2
  exit 1
fi
B=${Q27_BUILD_DIR:-build}
if [ ! -f "$B/build.ninja" ]; then
  cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120a-real
fi
cmake --build "$B" "$@"
