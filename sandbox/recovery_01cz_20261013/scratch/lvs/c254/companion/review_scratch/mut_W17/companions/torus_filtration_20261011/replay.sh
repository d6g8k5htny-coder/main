#!/usr/bin/env bash
# Replay of the C254 torus-filtration companion. It runs neither formal/gate.py nor the formal
# tests: the companion consumes only the pinned toolchain, lakefile and dependency lock from
# formal/, and its evidence is kernel-checked declarations, not alignment or acceptance.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIDE="$ROOT/companions/torus_filtration_20261011"
OUT="$ROOT/.lake/torus-filtration-evidence"
cd "$ROOT"
test ! -e "$OUT"
mkdir -p "$OUT"
python3 -B -S "$SIDE/check.py" source > "$OUT/source.json"
python3 -B -S "$SIDE/test_torus_filtration.py" 2>&1 | tee "$OUT/tests.log"
python3 -B -O -S "$SIDE/test_torus_filtration.py" 2>&1 | tee "$OUT/tests-optimized.log"
python3 -B -S "$SIDE/check.py" emit "$OUT"
export TORUS_SIDE="$SIDE" TORUS_OUT="$OUT" TORUS_ROOT="$ROOT"
cd "$ROOT/formal"
lake env bash <<'INNER'
set -euo pipefail
export LEAN_PATH="$TORUS_SIDE:${LEAN_PATH:-}"
export LEAN_SRC_PATH="$TORUS_SIDE:${LEAN_SRC_PATH:-}"
lean --version > "$TORUS_OUT/toolchain.log"
cat "$TORUS_OUT/toolchain.log"
: > "$TORUS_OUT/packages.log"
for pkg in "$TORUS_ROOT"/formal/.lake/packages/*/; do
  printf '%s %s\n' "$(basename "$pkg")" "$(git -C "$pkg" rev-parse HEAD)" >> "$TORUS_OUT/packages.log"
done
cat "$TORUS_OUT/packages.log"
run_lean() {
  local label="$1"; shift
  local code=0
  "$@" >"$TORUS_OUT/$label.log" 2>&1 || code=$?
  printf '%s\n' "$code" >"$TORUS_OUT/$label.status"
  cat "$TORUS_OUT/$label.log"
  test "$code" -eq 0
}
run_lean build-TorusTopology lean -DwarningAsError=true --root="$TORUS_SIDE" -o "$TORUS_SIDE/TorusTopology.olean" "$TORUS_SIDE/TorusTopology.lean"
run_lean build-ClosedSuperlevel lean -DwarningAsError=true --root="$TORUS_SIDE" -o "$TORUS_SIDE/ClosedSuperlevel.olean" "$TORUS_SIDE/ClosedSuperlevel.lean"
run_lean build-TorusFiltration lean -DwarningAsError=true --root="$TORUS_SIDE" -o "$TORUS_SIDE/TorusFiltration.olean" "$TORUS_SIDE/TorusFiltration.lean"
run_lean positive lean -DwarningAsError=true --root="$TORUS_OUT" "$TORUS_OUT/Positive.lean"
run_lean axioms lean -DwarningAsError=true --root="$TORUS_OUT" "$TORUS_OUT/Audit.lean"
run_lean types lean -DwarningAsError=true --root="$TORUS_OUT" "$TORUS_OUT/Types.lean"
run_lean leanchecker leanchecker --fresh TorusFiltration
# False statements about the companion's own definitions: Lean itself must refuse each with exactly `⊢ False`.
for control in RejectSuperlevelSignFlip RejectSuperlevelBelowThreshold RejectConstantNotPeriodic; do
  code=0
  lean -DwarningAsError=true --root="$TORUS_OUT" "$TORUS_OUT/$control.lean" >"$TORUS_OUT/$control.log" 2>&1 || code=$?
  printf '%s\n' "$code" >"$TORUS_OUT/$control.status"
  python3 -B -S "$TORUS_SIDE/check.py" negative "$TORUS_OUT/$control.log" --control "$control" --status "$code"
done
# Admission injections run WITHOUT warnings-as-errors: Lean accepts the file, only the axiom audit refuses it.
for control in RejectSorry RejectCustomAxiom RejectNativeDecide; do
  code=0
  lean --root="$TORUS_OUT" "$TORUS_OUT/$control.lean" >"$TORUS_OUT/$control.log" 2>&1 || code=$?
  printf '%s\n' "$code" >"$TORUS_OUT/$control.status"
  python3 -B -S "$TORUS_SIDE/check.py" negative "$TORUS_OUT/$control.log" --control "$control" --status "$code"
done
INNER
cd "$ROOT"
python3 -B -S "$SIDE/check.py" finish "$OUT"
echo 'PASS: torus-filtration companion replay; kernel-checked declarations only. Alignment review, scientific effect and acceptance are not decided here.'
