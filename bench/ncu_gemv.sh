#!/usr/bin/env bash
# Nsight Compute summary of the decode GEMV per weight type (H8): duration, DRAM throughput, registers, occupancy and
# the share of stall samples spent waiting on global loads. Real clocks (--clock-control none), cold caches (ncu
# default), one launch per (type, shape, columns) after the warm-up launches of bench_gemv.
# Usage (Git Bash, repo root): bash bench/ncu_gemv.sh <model.gguf> [nc=4] [shape list "TYPE:K:N ..."]
set -u
M=$1
NC=${2:-4}
NCU=${NCU:-"/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.0/target/windows-desktop-win7-x64/ncu.exe"}
SHAPES=${3:-"IQ3_S:5120:17408 IQ3_XXS:5120:17408 IQ4_XS:5120:17408 IQ2_S:5120:17408 IQ2_XS:5120:17408 IQ2_XXS:5120:17408 Q4_K:5120:17408 Q2_K:5120:12288 Q6_K:5120:12288 IQ3_S:17408:5120 IQ4_XS:17408:5120 IQ3_XXS:17408:5120"}
printf "%-8s %6s %6s %3s %9s %7s %5s %6s %7s\n" type K N nc dur_us DRAM% regs occ% ldwait%
for s in $SHAPES; do
  IFS=: read -r T K N <<<"$s"
  out=$(BENCH_ONLY=$T:$K:$N:$NC CUDA_DEVICE_ORDER=PCI_BUS_ID "$NCU" --clock-control none --kernel-name regex:gemv_kernel \
    --launch-skip 12 --launch-count 1 --section SpeedOfLight --section Occupancy --section LaunchStats \
    --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_sample_count \
    ./build/bench_gemv.exe "$M" 1 bench/out 2>&1)
  num() { echo "$out" | grep -E "$1" | head -1 | awk '{print $NF}' | tr -d '.' | tr ',' '.'; }
  dur=$(num "^ +Duration"); dram=$(num "DRAM Throughput"); regs=$(num "Registers Per Thread"); occ=$(num "Achieved Occupancy")
  ls=$(num "stalled_long_scoreboard"); sc=$(num "pcsamp_sample_count")
  lw=$(awk -v a="$ls" -v b="$sc" 'BEGIN { if (b > 0) printf "%.0f", 100 * a / b; else print "-" }')
  printf "%-8s %6s %6s %3s %9s %7s %5s %6s %7s\n" "$T" "$K" "$N" "$NC" "$dur" "$dram" "$regs" "$occ" "$lw"
done
