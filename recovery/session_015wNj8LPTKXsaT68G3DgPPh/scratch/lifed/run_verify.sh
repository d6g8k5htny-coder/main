#!/bin/sh
cd /tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/lifed/pkt_run
start=$(date +%s)
python3 -B -O -S theorem_b.py --check-full --procs 4 > ../verify_full.log 2>&1; echo "check-full rc=$? $(( $(date +%s) - start ))s" >> ../verify_summary.log
for m in no-om-term no-bad-set no-det no-torus cap-half; do
  s=$(date +%s)
  python3 -B -S theorem_b.py --check --procs 4 --mutant $m > ../verify_mut_$m.log 2>&1; echo "mutant $m rc=$? $(( $(date +%s) - s ))s" >> ../verify_summary.log
done
s=$(date +%s)
python3 -B -S theorem_b.py --controls > ../controls.json 2> ../controls.err; echo "controls rc=$? $(( $(date +%s) - s ))s" >> ../verify_summary.log
echo DONE >> ../verify_summary.log
