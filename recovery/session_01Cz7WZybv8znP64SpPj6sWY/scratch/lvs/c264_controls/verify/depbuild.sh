#!/usr/bin/env bash
# Build one tracked companion module (read unmodified) into verify/deps, replay.sh flags.
set -u
export PATH="$HOME/.elan/bin:$PATH"
S=<SCRATCH>
C=$S/repos/math_c263reg/companions/torus_filtration_20261011
V=$S/lvs/c264_controls/verify
M="$1"
cd $S/repos/math_c263reg/formal || exit 99
START=$(date +%s)
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c "export LEAN_PATH=\"$C:$V/deps:\$LEAN_PATH\"; lean -DwarningAsError=true --root=$C -o $V/deps/$M.olean $C/$M.lean" > $V/logs/dep-$M.log 2>&1
CODE=$?
echo "$M exit=$CODE secs=$(( $(date +%s) - START ))"
exit $CODE
