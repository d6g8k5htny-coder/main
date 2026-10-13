#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
L=<SCRATCH>/lvs
F=<SCRATCH>/repos/math_c254/formal
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
cd "$F" && lake env bash <<'INNER'
set -uo pipefail
L=<SCRATCH>/lvs
P=$L/c262/proto/side
export LEAN_PATH="$L/c263:$L/c264:$P:${LEAN_PATH:-}"
lean --root="$L/c263/census2" "$L/c263/census2/Census.lean" > "$L/c263/census2/census.log" 2>&1; echo "c263 census exit $?"
lean --root="$L/c264/census2" "$L/c264/census2/Census.lean" > "$L/c264/census2/census.log" 2>&1; echo "c264 census exit $?"
INNER
