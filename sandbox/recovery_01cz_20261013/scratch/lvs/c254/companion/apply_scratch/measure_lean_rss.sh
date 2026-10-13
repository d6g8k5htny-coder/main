#!/usr/bin/env bash
# Scratch: peak RSS and wall time of one `lean` compile of ClosedSuperlevel at LEAN_NUM_THREADS=2, for comparison.
set -u
export PATH=$HOME/.elan/bin:$PATH
S=<SCRATCH>
SIDE=$S/repos/math_c254/companions/torus_filtration_20261011
OUT=$S/lvs/c254/companion/apply_scratch
cd "$S/repos/math_c254/formal" || exit 99
export TORUS_SIDE="$SIDE" TORUS_OUT="$OUT"
lake env bash <<'INNER'
set -u
export LEAN_PATH="$TORUS_SIDE:${LEAN_PATH:-}"
export LEAN_SRC_PATH="$TORUS_SIDE:${LEAN_SRC_PATH:-}"
export LEAN_NUM_THREADS=2
while pgrep -x lean >/dev/null 2>&1 || pgrep -x leanchecker >/dev/null 2>&1; do sleep 3; done
echo "START lean-ClosedSuperlevel $(date -u +%T)"
SECONDS=0
lean -DwarningAsError=true --root="$TORUS_SIDE" -o "$TORUS_OUT/ClosedSuperlevel.scratch.olean" "$TORUS_SIDE/ClosedSuperlevel.lean" > "$TORUS_OUT/lean-ClosedSuperlevel.log" 2>&1 &
pid=$!
maxrss=0
while kill -0 "$pid" 2>/dev/null; do
  for p in $(pgrep -x lean); do
    rss=$(awk '/VmRSS/ {print $2}' /proc/$p/status 2>/dev/null || echo 0)
    [ -n "$rss" ] && [ "$rss" -gt "$maxrss" ] && maxrss=$rss
  done
  sleep 1
done
wait "$pid"; code=$?
echo "END lean-ClosedSuperlevel exit=$code wall_seconds=$SECONDS max_rss_kb=$maxrss $(date -u +%T)"
echo "exit=$code wall_seconds=$SECONDS max_rss_kb=$maxrss threads=2" > "$TORUS_OUT/lean-ClosedSuperlevel.time"
rm -f "$TORUS_OUT/ClosedSuperlevel.scratch.olean"
INNER
