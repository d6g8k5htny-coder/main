#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
SIDE=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
F=<SCRATCH>/repos/math_c263reg/formal
E=<SCRATCH>/lvs/c26x/probe/emit
cd "$F" || exit 1
for C in RejectEssentialDeathBound RejectReducedPointClass; do
  while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
  echo "== $C start $(date -u +%FT%TZ)"
  code=0
  lake env bash -c "export LEAN_PATH=\"$SIDE:\$LEAN_PATH\"; lean -DwarningAsError=true --root=$E $E/$C.lean" > "$E/$C.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$E/$C.status"
  echo "== $C exit $code $(date -u +%FT%TZ)"; cat "$E/$C.log"
done
for L in Audit Types Census Positive; do
  while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
  echo "== $L start $(date -u +%FT%TZ)"
  code=0
  lake env bash -c "export LEAN_PATH=\"$SIDE:\$LEAN_PATH\"; lean -DwarningAsError=true --root=$E $E/$L.lean" > "$E/$L.log" 2>&1 || code=$?
  echo "== $L exit $code $(date -u +%FT%TZ) log bytes $(wc -c < "$E/$L.log")"
done
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
echo "== leanchecker start $(date -u +%FT%TZ)"
code=0
lake env bash -c "export LEAN_PATH=\"$SIDE:\$LEAN_PATH\"; leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 TorusFiltration" > "$E/leanchecker.log" 2>&1 || code=$?
echo "== leanchecker exit $code $(date -u +%FT%TZ) log bytes $(wc -c < "$E/leanchecker.log")"
