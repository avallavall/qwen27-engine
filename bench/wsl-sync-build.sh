#!/usr/bin/env bash
# WSL test helper: copy this Windows working tree to ~/qwen27-engine on the WSL disk (ext4; builds from /mnt/c are
# slow) without build output or big data, then build it there with build.sh.
# Run from Windows:  wsl -d Ubuntu-24.04 -- bash /mnt/c/<path to repo>/bench/wsl-sync-build.sh [cmake --build args]
# Reference files for the tests go to ~/qwen27-engine/bench/out (copy them once; --delete leaves that folder alone).
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)/
DST=${Q27_WSL_DIR:-$HOME/qwen27-engine}/
rsync -a --delete \
  --exclude '/build/' --exclude '/build-*/' --exclude '/.venv/' --exclude '/refs/' --exclude '/.git/' \
  --exclude '/bench/out/' --exclude '/bench/build/' --exclude '/profiles/' --exclude '__pycache__/' \
  --exclude '/models/' --exclude '*.log' --exclude '*.log.err' \
  "$SRC" "$DST"
cd "$DST"
# a Windows checkout may have CRLF line endings; make the shell scripts runnable
for f in *.sh bench/*.sh; do sed -i 's/\r$//' "$f"; chmod +x "$f"; done
./build.sh "$@"
