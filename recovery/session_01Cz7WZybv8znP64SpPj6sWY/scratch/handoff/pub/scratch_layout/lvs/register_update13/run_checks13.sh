#!/usr/bin/env bash
# Round-13 local checks of the statement packet (records of computations on an unverified worktree, not evidence about any commit).
set -uo pipefail
unset PYTHONOPTIMIZE
M=${LVS_SCRATCH:?set LVS_SCRATCH to the scratch root}/repos/math_r5
O=${LVS_SCRATCH:?set LVS_SCRATCH to the scratch root}/lvs/register_update13
cd "$M"
date -u +%FT%TZ
python3 -B -S specifications/lean_statements_20261010/check.py --allow-dirty > "$O/stmt_check.log" 2>&1; echo "exit=$?" > "$O/stmt_check.exit"
python3 -B -S specifications/lean_statements_20261010/test_check.py > "$O/stmt_tests.log" 2>&1; echo "exit=$?" > "$O/stmt_tests.exit"
python3 -B -O -S specifications/lean_statements_20261010/test_check.py > "$O/stmt_tests_O.log" 2>&1; echo "exit=$?" > "$O/stmt_tests_O.exit"
rm -rf specifications/lean_statements_20261010/__pycache__
date -u +%FT%TZ
