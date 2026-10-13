#!/usr/bin/env bash
# usage: compile.sh <file.lean> <log> [extra lean args...]   (exact task command; PATH gains ~/.elan/bin)
set -u
export PATH="$HOME/.elan/bin:$PATH"
F="$1"; LOG="$2"; shift 2
EXTRA="$*"
cd <SCRATCH>/repos/math_c263reg/formal || exit 99
START=$(date +%s)
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c 'export LEAN_PATH="<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011:<SCRATCH>/lvs/c264_controls:$LEAN_PATH"; lean -DwarningAsError=true --root=<SCRATCH>/lvs/c264_controls '"$EXTRA"' '"$F" > "$LOG" 2>&1
CODE=$?
echo "exit=$CODE secs=$(( $(date +%s) - START ))"
exit $CODE
