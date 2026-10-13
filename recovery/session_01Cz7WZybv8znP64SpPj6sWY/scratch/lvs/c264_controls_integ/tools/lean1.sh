#!/usr/bin/env bash
# usage: lean1.sh <label> <outdir> <lean args...>
# One Lean (or leanchecker) process inside `lake env` of the worktree's pinned formal/ project, LEAN_PATH prefixed
# by the companion directory exactly as replay.sh does; waits until no other lean/leanchecker runs; two threads.
# Records <outdir>/<label>.log, .status, .time (wall seconds and the peak VmHWM of any lean/leanchecker process,
# sampled every second from /proc; a sampled peak, not an exact one).
set -u
export PATH="$HOME/.elan/bin:$PATH"
R=<SCRATCH>/repos/math_c264ctl
SIDE=$R/companions/torus_filtration_20261011
LABEL="$1"; OUTD="$2"; shift 2
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
cd "$R/formal" || exit 99
export TORUS_SIDE="$SIDE"
export LEAN_NUM_THREADS=2
START=$(date +%s)
lake env bash -c 'export LEAN_PATH="$TORUS_SIDE:${LEAN_PATH:-}"; export LEAN_SRC_PATH="$TORUS_SIDE:${LEAN_SRC_PATH:-}"; exec "$@"' x "$@" \
  > "$OUTD/$LABEL.log" 2>&1 &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  for p in $(pgrep -x lean; pgrep -x leanchecker); do
    h=$(awk '/^VmHWM:/ {print $2}' /proc/$p/status 2>/dev/null)
    if [ -n "${h:-}" ] && [ "$h" -gt "$PEAK" ]; then PEAK=$h; fi
  done
  sleep 1
done
code=0
wait "$PID" || code=$?
printf '%s\n' "$code" > "$OUTD/$LABEL.status"
printf '%s s wall, sampled peak VmHWM %s KiB (%s GiB)\n' "$(( $(date +%s) - START ))" "$PEAK" \
  "$(awk -v k="$PEAK" 'BEGIN {printf "%.2f", k / 1048576}')" > "$OUTD/$LABEL.time"
echo "$LABEL exit=$code $(cat "$OUTD/$LABEL.time") log_bytes=$(wc -c < "$OUTD/$LABEL.log")"
