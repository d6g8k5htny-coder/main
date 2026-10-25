#!/bin/bash
# Runs scenarios strictly one stack at a time.
R=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220/replicate
for SC in "$@"; do
  echo "$(date -u +%H:%M:%S) start $SC" >> $R/evidence/runs.log
  if [ "${SC%r[0-9]*}" = S2 ]; then timeout 600 $R/scripts/run_stack.sh "$SC" --disable-webgl --disable-3d-apis
  else timeout 600 $R/scripts/run_stack.sh "$SC"; fi
  echo "$(date -u +%H:%M:%S) end $SC rc=$?" >> $R/evidence/runs.log
  python3 -I $R/scripts/extract.py $R/evidence/$SC > $R/evidence/$SC/extract.out 2>&1
done
