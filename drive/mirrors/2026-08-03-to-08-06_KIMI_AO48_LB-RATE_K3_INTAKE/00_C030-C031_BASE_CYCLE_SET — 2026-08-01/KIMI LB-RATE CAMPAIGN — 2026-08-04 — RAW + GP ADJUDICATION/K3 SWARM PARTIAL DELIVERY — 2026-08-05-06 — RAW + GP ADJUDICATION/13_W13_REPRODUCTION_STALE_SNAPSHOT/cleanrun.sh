#!/bin/bash
# W13 clean-room runner: fresh environment (env -i), captures program stdout,
# exit code, and wall-clock seconds. Usage: cleanrun.sh <workdir> <outfile> [pyargs...]
set -u
WD="$1"; OUT="$2"; shift 2
cd "$WD" || exit 125
START=$(date +%s.%N)
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 \
  python3 "$@" > "$OUT" 2> "$OUT.err"
RC=$?
END=$(date +%s.%N)
ELAPSED=$(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.2f", b-a}')
echo "$ELAPSED" > "$OUT.time"
exit $RC
