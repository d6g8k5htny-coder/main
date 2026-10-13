#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
SIDE=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
F=<SCRATCH>/repos/math_c263reg/formal
OUT=<SCRATCH>/lvs/c26x
cd "$F" || exit 1
for M in TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 TorusFiltration; do
  while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
  echo "== $M start $(date -u +%FT%TZ)"
  lake env bash -c "export LEAN_PATH=\"$SIDE:\$LEAN_PATH\"; lean -DwarningAsError=true --root=$SIDE -o $SIDE/$M.olean $SIDE/$M.lean" > "$OUT/build-$M.log" 2>&1
  echo "== $M exit $? $(date -u +%FT%TZ) log bytes $(wc -c < "$OUT/build-$M.log")"
done
