#!/usr/bin/env bash
set -uo pipefail
S=<SCRATCH>
export PATH=$HOME/.elan/bin:$PATH
unset PYTHONOPTIMIZE LEAN_NUM_THREADS
cd $S/repos/math_c263reg
python3 -B -S companions/torus_filtration_20261011/check.py source > $S/lvs/c26x_fix/source3.json 2>&1; echo "source exit=$?"
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
rm -rf .lake/torus-filtration-evidence companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
date -u +%FT%TZ
bash companions/torus_filtration_20261011/replay.sh > $S/lvs/c26x_fix/replay_final2.out 2>&1; echo "replay exit=$?"
date -u +%FT%TZ
rm -f companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
