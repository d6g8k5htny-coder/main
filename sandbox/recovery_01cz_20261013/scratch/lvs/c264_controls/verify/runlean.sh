#!/usr/bin/env bash
# usage: runlean.sh <root-dir> <file.lean> <log> [extra lean flags]   one Lean process, 2 threads
set -u
export PATH="$HOME/.elan/bin:$PATH"
S=<SCRATCH>
C=$S/repos/math_c263reg/companions/torus_filtration_20261011
V=$S/lvs/c264_controls/verify
ROOTD="$1"; F="$2"; LOG="$3"; shift 3; EXTRA="$*"
n=$(pgrep -c -x lean); if [ "$n" -ge 2 ]; then echo "BUSY lean=$n"; exit 98; fi
cd $S/repos/math_c263reg/formal || exit 99
START=$(date +%s)
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c "export LEAN_PATH=\"$C:$V/deps:$V/out:\$LEAN_PATH\"; lean $EXTRA --root=$ROOTD $F" > "$LOG" 2>&1
CODE=$?
echo "$CODE" > "${LOG%.log}.status"
echo "exit=$CODE secs=$(( $(date +%s) - START ))"
exit $CODE
