#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
export LEAN_NUM_THREADS=2
D=<SCRATCH>/lvs/c264
P=<SCRATCH>/lvs/c262/proto/side
F=<SCRATCH>/repos/math_c254/formal
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
date -u +%FT%TZ
ls "$P"/*.olean
cd "$F" && lake env bash <<'INNER'
set -uo pipefail
D=<SCRATCH>/lvs/c264
P=<SCRATCH>/lvs/c262/proto/side
export LEAN_PATH="$D:$P:${LEAN_PATH:-}"
lean -DwarningAsError=true --root="$D" -o "$D/OrdinaryH0.olean" "$D/OrdinaryH0.lean" > "$D/compile.log" 2>&1; echo "compile exit $?"
if [ -f "$D/OrdinaryH0.olean" ]; then lean --root="$D" "$D/Audit264.lean" > "$D/axioms.log" 2>&1; echo "axioms exit $?"; fi
INNER
date -u +%FT%TZ
