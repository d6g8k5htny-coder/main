#!/usr/bin/env bash
# Replays every recorded check for this package. Needs Lean 4.34.1 (lean-toolchain) and the
# pinned Mathlib (lake-manifest.json). Exit status is nonzero if any expectation fails.
set -u
cd "$(dirname "$0")"
fail=0
expect() { local want=$1; shift; "$@" >/dev/null 2>&1; local got=$?
  if { [ "$want" = 0 ] && [ $got -eq 0 ]; } || { [ "$want" = 1 ] && [ $got -ne 0 ]; }; then
    echo "ok   ($want) $*"; else echo "FAIL ($want, got $got) $*"; fail=1; fi; }
expect 0 lake build
expect 0 lake env leanchecker WholeFieldRegression.Basic
expect 1 lake env leanchecker WholeFieldRegression.NoSuchModule
expect 1 lake env lean controls/NegSign.lean
expect 1 lake env lean controls/NegNoIndep.lean
expect 1 lake env lean controls/NegFinite.lean
out=$(lake env lean controls/Audit.lean 2>&1)
if echo "$out" | grep -q -E "sorryAx|axiom [A-Za-z]" || [ "$(echo "$out" | grep -c 'depends on axioms')" -ne 8 ]; then
  echo "FAIL axiom audit"; fail=1; else echo "ok   axiom audit: 8 theorems, standard axioms only"; fi
if lake env lean controls/SorryControl.lean 2>&1 | grep -q sorryAx; then echo "ok   sorry control detected";
  else echo "FAIL sorry control not detected"; fail=1; fi
echo "whole-field regression kernel checks: $([ $fail = 0 ] && echo OK || echo FAIL)"
exit $fail
