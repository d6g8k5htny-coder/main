#!/usr/bin/env bash
# Round-11 local checks (records of computations on an unverified worktree, not evidence about any commit).
# usage: run_checks11.sh stmt|proof
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
unset PYTHONOPTIMIZE
M=<SCRATCH>/repos/math_r5
O=<SCRATCH>/lvs/register_update11
cd "$M"
date -u +%FT%TZ
if [ "$1" = stmt ]; then
  python3 -B -S specifications/lean_statements_20261010/check.py --allow-dirty > "$O/stmt_check.log" 2>&1; echo "exit=$?" > "$O/stmt_check.exit"
  python3 -B -S specifications/lean_statements_20261010/test_check.py > "$O/stmt_tests.log" 2>&1; echo "exit=$?" > "$O/stmt_tests.exit"
  python3 -B -O -S specifications/lean_statements_20261010/test_check.py > "$O/stmt_tests_O.log" 2>&1; echo "exit=$?" > "$O/stmt_tests_O.exit"
else
  python3 -B -S specifications/lean_statement_proofs_20261011/check.py --allow-dirty > "$O/proof_check.log" 2>&1; echo "exit=$?" > "$O/proof_check.exit"
  python3 -B -S specifications/lean_statement_proofs_20261011/test_check.py > "$O/proof_tests.log" 2>&1; echo "exit=$?" > "$O/proof_tests.exit"
  python3 -B -O -S specifications/lean_statement_proofs_20261011/test_check.py > "$O/proof_tests_O.log" 2>&1; echo "exit=$?" > "$O/proof_tests_O.exit"
fi
rm -rf specifications/lean_statement_proofs_20261011/__pycache__ specifications/lean_statements_20261010/__pycache__
date -u +%FT%TZ
