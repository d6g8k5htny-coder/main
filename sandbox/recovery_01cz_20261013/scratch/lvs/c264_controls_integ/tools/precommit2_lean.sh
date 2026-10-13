#!/usr/bin/env bash
# Fix-round pre-commit Lean run on the worktree bytes (not a receipt): the six module builds exactly as replay.sh
# compiles them, then positive, axioms, types and census on the emitted files; one process at a time, two threads.
set -u
I=<SCRATCH>/lvs/c264_controls_integ
S=<SCRATCH>/repos/math_c264ctl/companions/torus_filtration_20261011
T=$I/tools/lean1.sh
O=$I/precommit2/out
L=$I/precommit2/lean
mkdir -p $L
for m in TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 OrdinaryH0Controls TorusFiltration; do
  $T build-$m $L lean -DwarningAsError=true --root=$S -o $S/$m.olean $S/$m.lean
done
$T positive $L lean -DwarningAsError=true --root=$O $O/Positive.lean
$T axioms $L lean -DwarningAsError=true --root=$O $O/Audit.lean
$T types $L lean -DwarningAsError=true --root=$O $O/Types.lean
$T census $L lean -DwarningAsError=true --root=$O $O/Census.lean
echo PRECOMMIT2_LEAN_DONE
