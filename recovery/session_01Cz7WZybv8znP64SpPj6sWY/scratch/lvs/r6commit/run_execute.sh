#!/usr/bin/env bash
# Round-6 local execute (a record of a computation on a local commit, not a hosted receipt and not evidence about any pushed commit).
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
unset PYTHONOPTIMIZE
S=<SCRATCH>
O=$S/lvs/r6commit
cd $S/repos/math_r5
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 15; done
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD)"
(cd specifications/lean_statement_proofs_20261011 && lake build > $O/lake_build.log 2>&1); echo "lake build exit=$? $(date -u +%FT%TZ)"
python3 -B -S specifications/lean_statement_proofs_20261011/check.py --execute > $O/execute.log 2>&1; echo "check.py --execute exit=$? $(date -u +%FT%TZ)"
git diff --exit-code > /dev/null; echo "git diff exit=$?"
git status --short | head
