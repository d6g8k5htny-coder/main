#!/usr/bin/env bash
# Pre-commit Lean run on the worktree bytes (not a receipt): positive, axioms, types, census, the six-module leanchecker,
# the eleven gate controls, the three injections and the three discriminators, one process at a time, two threads.
set -u
I=<SCRATCH>/lvs/c264_controls_integ
T=$I/tools/lean1.sh
O=$I/precommit/out
D=$I/precommit/discrim
L=$I/precommit/lean
mkdir -p $L
$T positive $L lean -DwarningAsError=true --root=$O $O/Positive.lean
$T axioms $L lean -DwarningAsError=true --root=$O $O/Audit.lean
$T types $L lean -DwarningAsError=true --root=$O $O/Types.lean
$T census $L lean -DwarningAsError=true --root=$O $O/Census.lean
$T leanchecker $L leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 OrdinaryH0Controls TorusFiltration
for c in RejectSuperlevelSignFlip RejectSuperlevelBelowThreshold RejectConstantNotPeriodic RejectEssentialDeathBound RejectReducedPointClass RejectMergerKillsPointClass RejectSeparationWithoutNonzeroCoeff RejectNonmergerKillsDifference; do
  $T $c $L lean -DwarningAsError=true --root=$O $O/$c.lean
done
for c in RejectSorry RejectCustomAxiom RejectNativeDecide; do
  $T $c $L lean --root=$O $O/$c.lean
done
for c in RejectMergerKillsPointClass RejectSeparationWithoutNonzeroCoeff RejectNonmergerKillsDifference; do
  $T discrim-$c $L lean -DwarningAsError=true --root=$D $D/$c.lean
done
echo PRECOMMIT_LEAN_DONE
