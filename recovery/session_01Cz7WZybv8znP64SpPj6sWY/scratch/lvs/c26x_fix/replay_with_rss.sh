#!/usr/bin/env bash
set -uo pipefail
S=<SCRATCH>
W=$S/repos/math_c263reg
O=$S/lvs/c26x_fix
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
unset PYTHONOPTIMIZE
cd "$W"
rm -rf .lake/torus-filtration-evidence companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
( peak=0; while true; do for p in $(pgrep -x leanchecker; pgrep -x lean); do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); n=$(cat /proc/$p/comm 2>/dev/null); [ -n "$r" ] && echo "$(date -u +%T) $n $p $r" >> "$O/rss_samples.log"; done; sleep 2; done ) &
MON=$!
date -u +%FT%TZ
bash companions/torus_filtration_20261011/replay.sh > "$O/replay_8a7f3735_c.out" 2>&1; code=$?
kill $MON 2>/dev/null
echo "replay exit=$code"
date -u +%FT%TZ
rm -f companions/torus_filtration_20261011/*.olean companions/torus_filtration_20261011/*.ilean
awk '{if($4>m[$2]) m[$2]=$4} END{for(k in m) printf "peak %s %.2f GB\n", k, m[k]/1048576}' "$O/rss_samples.log"
