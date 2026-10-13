#!/usr/bin/env bash
# Round-13 local checks of the proof packet after edit_proof_docs13.py (records of computations on an unverified worktree,
# not evidence about any commit). NO Lean process: check.py source mode never calls lake, and test_check.py's LeanHelpers
# class (skipUnless(check.lake_binary())) would call ~/.elan/bin/lake through lake_binary()'s fallback, so this runner hides
# lake by pointing HOME at an empty directory (git keeps its real global config through GIT_CONFIG_GLOBAL) and refuses to
# run anything if lake_binary() still finds a lake.
set -uo pipefail
unset PYTHONOPTIMIZE
M=<SCRATCH>/repos/math_r5
O=<SCRATCH>/lvs/register_update13
P=specifications/lean_statement_proofs_20261011
export GIT_CONFIG_GLOBAL=/root/.gitconfig
export HOME="$O/nolake_home"
mkdir -p "$HOME"
case ":$PATH:" in *elan*) echo "REFUSED: elan on PATH"; exit 3;; esac
cd "$M"
date -u +%FT%TZ
python3 -B -S -c "import sys; sys.path.insert(0, '$P'); import check; l = check.lake_binary(); print('lake_binary:', l); sys.exit(0 if l is None else 3)" \
  || { echo "REFUSED: lake is reachable; not running the tests"; exit 3; }
python3 -B -S $P/check.py --allow-dirty > "$O/proof_check.log" 2>&1; echo "exit=$?" > "$O/proof_check.exit"
python3 -B -S $P/test_check.py > "$O/proof_tests.log" 2>&1; echo "exit=$?" > "$O/proof_tests.exit"
python3 -B -O -S $P/test_check.py > "$O/proof_tests_O.log" 2>&1; echo "exit=$?" > "$O/proof_tests_O.exit"
find "$M/specifications" -name __pycache__ -type d -prune -exec rm -rf {} +
rmdir "$HOME" 2>/dev/null || true
date -u +%FT%TZ
