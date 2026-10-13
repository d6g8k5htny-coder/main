#!/usr/bin/env bash
# usage: lrun.sh <log> <extra LEAN_PATH dir or -> <lean args...>
# One Lean process inside `lake env` of the pinned formal/ project, 2 threads, companion dir on LEAN_PATH,
# optional extra dir (this verifier's side directory) after it. Refuses to start if >=2 lean processes run.
set -u
export PATH="$HOME/.elan/bin:$PATH"
LOG="$1"; EXTRA="$2"; shift 2
n=$(pgrep -c -x lean || true); if [ "${n:-0}" -ge 2 ]; then echo "BUSY: $n lean processes"; exit 75; fi
C=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
LP="$C"; [ "$EXTRA" != "-" ] && LP="$C:$EXTRA"
cd <SCRATCH>/repos/math_c263reg/formal || exit 99
START=$(date +%s); code=0
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c 'export LEAN_PATH="'"$LP"':$LEAN_PATH"; exec lean "$@"' lean "$@" > "$LOG" 2>&1 || code=$?
echo "exit=$code secs=$(( $(date +%s) - START )) log_bytes=$(wc -c < "$LOG") $(basename "$LOG")"
exit $code
