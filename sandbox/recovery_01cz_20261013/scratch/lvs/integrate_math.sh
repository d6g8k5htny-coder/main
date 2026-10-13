#!/usr/bin/env bash
# Integration of the Math- deliverable: run after all checks pass. Usage: bash integrate_math.sh
set -euo pipefail
S=<SCRATCH>
P=$S/repos/math/specifications/lean_statements_20261010
export PATH="$HOME/.elan/bin:$PATH"
cd $S/repos/math
git status --short
python3 -B -S $P/test_check.py 2>&1 | tail -2
python3 -B -O -S $P/test_check.py 2>&1 | tail -2
python3 -B -S $P/check.py --allow-dirty 2>&1 | tail -3
python3 -B -S -m unittest discover -s tests 2>&1 | tail -3
git add specifications/lean_statements_20261010 .github/workflows/lean-specifications.yml
git status --short
