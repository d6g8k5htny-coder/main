#!/bin/bash
# usage: runstep.sh <workdir> <logname> <bash script>
cd "$1" || exit 99
LOG=<SCRATCH>/q429/logs/$2.log
start=$(date +%s.%N)
bash --noprofile --norc -e -c "$3" > "$LOG" 2>&1
rc=$?
end=$(date +%s.%N)
echo "RC=$rc SECONDS=$(echo "$end - $start" | bc)" | tee -a "$LOG"
