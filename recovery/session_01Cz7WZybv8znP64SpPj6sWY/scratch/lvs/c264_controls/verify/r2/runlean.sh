#!/usr/bin/env bash
# usage: runlean.sh <log> <status> <lean args...>
# One Lean process inside `lake env` of the pinned formal/ project, two threads, LEAN_PATH prefixed by verify/r2/side only.
set -u
export PATH="$HOME/.elan/bin:$PATH"
S=<SCRATCH>
SIDE=$S/lvs/c264_controls/verify/r2/side
LOG="$1"; STATUS="$2"; shift 2
n=$(pgrep -c -x lean || true)
if [ "${n:-0}" -ge 2 ]; then echo "BUSY: $n lean processes running"; exit 98; fi
cd $S/repos/math_c263reg/formal || exit 99
START=$(date +%s)
code=0
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c 'export LEAN_PATH="'"$SIDE"':$LEAN_PATH"; exec lean "$@"' lean "$@" > "$LOG" 2>&1 || code=$?
printf '%s\n' "$code" > "$STATUS"
echo "exit=$code secs=$(( $(date +%s) - START )) $(basename "$LOG")"
exit 0
