#!/usr/bin/env bash
set -uo pipefail
S=<SCRATCH>
export PATH=$HOME/.elan/bin:$PATH
unset PYTHONOPTIMIZE
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
echo "== round 4 $(date -u +%T)"
( export LEAN_NUM_THREADS=2; bash $S/lvs/r4amend/run_checks.sh ) 2>&1 | sed 's/^/  r4: /'
echo "== registration $(date -u +%T)"
unset LEAN_NUM_THREADS
cd $S/repos/math_c263reg
python3 -B -S companions/torus_filtration_20261011/check.py source > $S/lvs/c26x_fix/source2.json 2>&1; echo "source exit=$?"
rm -rf .lake/torus-filtration-evidence companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
rm -f $S/lvs/c26x_fix/rss_samples.log
( while true; do for p in $(pgrep -x leanchecker; pgrep -x lean); do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); n=$(cat /proc/$p/comm 2>/dev/null); [ -n "$r" ] && echo "$(date -u +%T) $n $p $r" >> $S/lvs/c26x_fix/rss_samples.log; done; sleep 2; done ) &
MON=$!
env | grep -c LEAN_NUM_THREADS || true
bash companions/torus_filtration_20261011/replay.sh > $S/lvs/c26x_fix/replay_final.out 2>&1; echo "replay exit=$?"
kill $MON 2>/dev/null
rm -f companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
awk '{if($4>m[$2]) m[$2]=$4} END{for(k in m) printf "peak %s %.2f GB\n", k, m[k]/1048576}' $S/lvs/c26x_fix/rss_samples.log
date -u +%T
