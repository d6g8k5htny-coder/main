#!/bin/bash
# Runs the axiom audit then the type check, strictly one Lean process at a time.
export PATH=$HOME/.elan/bin:$PATH
S=<SCRATCH>
cd $S/repos/math_c254/formal || exit 99
echo "AUDIT START $(date -u +%FT%TZ)"; SECONDS=0
lake env lean $S/lvs/c254/integrate/Audit.lean > $S/lvs/c254/integrate/axioms.out 2>&1
echo "AUDIT exit=$? elapsed=${SECONDS}s lines=$(wc -l < $S/lvs/c254/integrate/axioms.out)"
echo "TYPES START $(date -u +%FT%TZ)"; SECONDS=0
lake env lean $S/lvs/c254/integrate/Types.lean > $S/lvs/c254/integrate/types.out 2>&1
echo "TYPES exit=$? elapsed=${SECONDS}s lines=$(wc -l < $S/lvs/c254/integrate/types.out)"
