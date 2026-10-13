#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
P=<SCRATCH>/lvs/c262/proto
F=<SCRATCH>/repos/math_c254/formal
# wait until no other lean/leanchecker process is running (memory: one Mathlib import at a time)
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
date -u +%FT%TZ
cd "$F" && lake env bash <<'INNER'
set -uo pipefail
P=<SCRATCH>/lvs/c262/proto
export LEAN_PATH="$P/side:${LEAN_PATH:-}"
for m in TorusTopology ClosedSuperlevel TorusFiltration; do
  lean -DwarningAsError=true --root="$P/side" -o "$P/side/$m.olean" "$P/side/$m.lean" > "$P/build-$m.log" 2>&1; echo "build $m exit $?"
done
lean --root="$P" "$P/Census.lean" > "$P/census.log" 2>&1; echo "census exit $?"
INNER
date -u +%FT%TZ
