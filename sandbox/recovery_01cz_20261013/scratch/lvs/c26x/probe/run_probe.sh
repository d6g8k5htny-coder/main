#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
SIDE=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
F=<SCRATCH>/repos/math_c263reg/formal
P=<SCRATCH>/lvs/c26x/probe
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
cd "$F" && lake env bash -c "export LEAN_PATH=\"$SIDE:\$LEAN_PATH\"; lean -DwarningAsError=true --root=$P $P/${1:-Probe}.lean" > "$P/${1:-Probe}.log" 2>&1
echo "exit $?"
