#!/bin/bash
# usage: runjob.sh <label> <worktree> [python3-shim-target]
D=<SCRATCH>/q429/run0
LABEL=$1; WT=$2; SHIM=$3
OUT=$D/$LABEL; rm -rf "$OUT"; mkdir -p "$OUT/runner_temp" "$OUT/bin"
if [ -n "$SHIM" ]; then ln -s "$SHIM" "$OUT/bin/python3"; export PATH="$OUT/bin:$PATH"; fi
export RUNNER_TEMP="$OUT/runner_temp" GITHUB_STEP_SUMMARY="$OUT/step_summary.md" GITHUB_WORKSPACE="$WT" CI=true GITHUB_ACTIONS=true
: > "$GITHUB_STEP_SUMMARY"
cd "$WT"
echo "python3 -> $(command -v python3) $(python3 --version)"
failed=0
for s in 2 3 4 6; do
  if [ $failed = 1 ] && [ $s != 6 ]; then echo "STEP $s SKIPPED (prior failure)"; continue; fi
  start=$(date +%s.%N)
  bash --noprofile --norc -eo pipefail "$D/controls_step$s.sh" > "$OUT/step$s.log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  printf 'STEP %s rc=%s seconds=%.2f\n' $s $rc $(echo "$end - $start" | bc)
  [ $rc != 0 ] && failed=1
done
