#!/usr/bin/env bash
# Build one companion module (source read unmodified from the worktree) into this lane's directory, replay.sh flags.
set -u
export PATH="$HOME/.elan/bin:$PATH"
M="$1"
cd <SCRATCH>/repos/math_c263reg/formal || exit 99
START=$(date +%s)
LEAN_NUM_THREADS=2 timeout 3000 lake env bash -c 'export LEAN_PATH="<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011:<SCRATCH>/lvs/c264_controls:$LEAN_PATH"; lean -DwarningAsError=true --root=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011 -o <SCRATCH>/lvs/c264_controls/'"$M"'.olean <SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011/'"$M"'.lean' > <SCRATCH>/lvs/c264_controls/dep-$M.log 2>&1
CODE=$?
echo "$M exit=$CODE secs=$(( $(date +%s) - START ))"
exit $CODE
