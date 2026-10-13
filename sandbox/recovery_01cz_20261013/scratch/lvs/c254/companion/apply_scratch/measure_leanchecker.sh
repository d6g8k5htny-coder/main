#!/usr/bin/env bash
# Scratch measurement: compile the root module, then time `leanchecker` WITHOUT --fresh on the
# companion modules at LEAN_NUM_THREADS=2 (the hosted setting). One Lean process at a time.
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
while pgrep -f 'toolchains/[^ ]*/bin/lean |bin/leanchecker' >/dev/null 2>&1; do sleep 3; done
mods="--verbose TorusTopology ClosedSuperlevel TorusFiltration"
echo "START leanchecker [$mods] $(date -u +%T)"
SECONDS=0
timeout 1500 leanchecker $mods > "$TORUS_OUT/leanchecker-verbose.log" 2>&1 &
pid=$!
maxrss=0
while kill -0 "$pid" 2>/dev/null; do
  for p in $(pgrep -x leanchecker); do
    rss=$(awk '/VmRSS/ {print $2}' /proc/$p/status 2>/dev/null || echo 0)
    [ -n "$rss" ] && [ "$rss" -gt "$maxrss" ] && maxrss=$rss
  done
  sleep 2
done
wait "$pid"; code=$?
echo "END leanchecker exit=$code wall_seconds=$SECONDS max_rss_kb=$maxrss $(date -u +%T)"
echo "exit=$code wall_seconds=$SECONDS max_rss_kb=$maxrss modules=$mods threads=2" > "$TORUS_OUT/leanchecker-verbose.time"
echo MEASURE_DONE
INNER
