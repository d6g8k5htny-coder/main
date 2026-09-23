#!/bin/bash
# W13 clean-room runner v2: writes <out>.rc and <out>.time always.
set -u
WD="$1"; OUT="$2"; shift 2
cd "$WD" || { echo 125 > "$OUT.rc"; exit 125; }
START=$(date +%s.%N)
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 \
  python3 "$@" > "$OUT" 2> "$OUT.err"
RC=$?
END=$(date +%s.%N)
awk -v a="$START" -v b="$END" 'BEGIN{printf "%.2f\n", b-a}' > "$OUT.time"
echo "$RC" > "$OUT.rc"
exit $RC
