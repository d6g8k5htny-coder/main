#!/usr/bin/env bash
set -u
V=<SCRATCH>/lvs/c264_controls/verify/r2
wait_slot() { while [ "$(pgrep -c -x lean || true)" -ge 2 ]; do sleep 20; done; }
wait_slot; $V/runlean.sh $V/logs/module-taskform.log $V/logs/module-taskform.status -DwarningAsError=true --root=$V $V/OrdinaryH0Controls.lean
for f in Statements Audit Census; do wait_slot; $V/runlean.sh $V/audit/$f.log $V/audit/$f.status -DwarningAsError=true --root=$V/audit $V/audit/$f.lean; done
for f in RejectMergerKillsPointClass RejectSeparationWithoutNonzeroCoeff RejectNonmergerKillsDifference; do wait_slot; $V/runlean.sh $V/out/$f.log $V/out/$f.status -DwarningAsError=true --root=$V/out $V/out/$f.lean; done
for f in $V/discrim/*.lean; do b=${f%.lean}; wait_slot; $V/runlean.sh $b.log $b.status -DwarningAsError=true --root=$V/discrim $f; done
echo RUN_DONE
