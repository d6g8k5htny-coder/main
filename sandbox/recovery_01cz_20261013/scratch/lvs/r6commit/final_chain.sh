#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
unset PYTHONOPTIMIZE
S=<SCRATCH>
O=$S/lvs/register_update13/final2; mkdir -p $O
cd $S/repos/math_r5
(cd specifications/lean_statement_proofs_20261011 && lake env lean -DwarningAsError=true SpecificationProofs/Side24Images.lean > $O/lean_wae.log 2>&1); echo "lean -DwarningAsError exit=$?"
fail=0
for P in specifications/lean_statements_20261010 specifications/lean_statement_proofs_20261011; do n=$(basename $P)
  python3 -B -S $P/check.py --allow-dirty > $O/$n.check.log 2>&1; r=$?; echo "$n check exit=$r"; [ $r = 0 ] || fail=1
  python3 -B -S $P/test_check.py > $O/$n.tests.log 2>&1; r=$?; echo "$n tests exit=$r $(tail -3 $O/$n.tests.log | tr '\n' ' ')"; [ $r = 0 ] || fail=1
  python3 -B -O -S $P/test_check.py > $O/$n.testsO.log 2>&1; r=$?; echo "$n testsO exit=$r $(tail -3 $O/$n.testsO.log | tr '\n' ' ')"; [ $r = 0 ] || fail=1
done
find specifications -name __pycache__ -type d -prune -exec rm -rf {} +
[ $fail = 0 ] || { echo "CHECKS FAILED; not committing"; exit 1; }
git add specifications/lean_statement_proofs_20261011/SpecificationProofs/Side24Images.lean specifications/lean_statement_proofs_20261011/SpecificationProofs.lean specifications/lean_statement_proofs_20261011/README.md specifications/lean_statement_proofs_20261011/SCOPE.md specifications/lean_statements_20261010/README.md specifications/lean_statements_20261010/SCOPE.md specifications/lean_statements_20261010/REGISTER.json
git commit -q -F $S/lvs/r6commit/commit.msg && echo "committed $(git rev-parse HEAD)"
git status --short | head
$S/lvs/r6commit/run_execute.sh
