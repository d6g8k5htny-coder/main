#!/usr/bin/env bash
# Sequential builds into verify/r2/side; waits while 2+ Lean processes run elsewhere.
set -u
V=<SCRATCH>/lvs/c264_controls/verify/r2; C=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
wait_slot() { while [ "$(pgrep -c -x lean || true)" -ge 2 ]; do sleep 20; done; }
for M in TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0; do
  wait_slot
  $V/runlean.sh $V/logs/build-$M.log $V/logs/build-$M.status -DwarningAsError=true --root=$C -o $V/side/$M.olean $C/$M.lean
  [ "$(cat $V/logs/build-$M.status)" = 0 ] || { echo "STOP $M"; exit 1; }
done
wait_slot
$V/runlean.sh $V/logs/build-OrdinaryH0Controls.log $V/logs/build-OrdinaryH0Controls.status -DwarningAsError=true --root=$V -o $V/side/OrdinaryH0Controls.olean $V/OrdinaryH0Controls.lean
[ "$(cat $V/logs/build-OrdinaryH0Controls.status)" = 0 ] || { echo "STOP controls"; exit 1; }
wait_slot
$V/runlean.sh $V/logs/build-TorusFiltration.log $V/logs/build-TorusFiltration.status -DwarningAsError=true --root=$V/rootsrc -o $V/side/TorusFiltration.olean $V/rootsrc/TorusFiltration.lean
echo CHAIN_DONE
