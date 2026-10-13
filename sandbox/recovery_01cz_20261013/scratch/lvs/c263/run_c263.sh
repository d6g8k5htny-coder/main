#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
D=<SCRATCH>/lvs/c263
F=<SCRATCH>/repos/math_c254/formal
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
date -u +%FT%TZ
cd "$F" && lake env bash <<'INNER'
set -uo pipefail
D=<SCRATCH>/lvs/c263
export LEAN_PATH="$D:${LEAN_PATH:-}"
lean -DwarningAsError=true --root="$D" -o "$D/SuperlevelBar.olean" "$D/SuperlevelBar.lean" > "$D/compile.log" 2>&1; echo "compile exit $?"
lean --root="$D" "$D/Audit263.lean" > "$D/axioms.log" 2>&1; echo "axioms exit $?"
INNER
date -u +%FT%TZ
