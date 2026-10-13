#!/usr/bin/env bash
# usage: gatelean.sh <log> <status> <lean args...>
# Runs one Lean process the way replay.sh does (inside `lake env` of the pinned formal/ project, two threads), with
# LEAN_PATH prefixed by this lane's emulated companion side directory gate/side (never the worktree).
set -u
export PATH="$HOME/.elan/bin:$PATH"
LOG="$1"; STATUS="$2"; shift 2
SIDE_EMU=<SCRATCH>/lvs/c264_controls/gate/side
cd <SCRATCH>/repos/math_c263reg/formal || exit 99
START=$(date +%s)
code=0
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c 'export LEAN_PATH="'"$SIDE_EMU"':$LEAN_PATH"; exec lean "$@"' lean "$@" > "$LOG" 2>&1 || code=$?
printf '%s\n' "$code" > "$STATUS"
echo "exit=$code secs=$(( $(date +%s) - START )) $(basename "$LOG")"
