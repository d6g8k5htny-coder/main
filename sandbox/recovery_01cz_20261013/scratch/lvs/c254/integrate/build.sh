#!/bin/bash
export PATH=$HOME/.elan/bin:$PATH
cd <SCRATCH>/repos/math_c254/formal || exit 99
N=$(nproc); J=$(( N < 4 ? N : 4 ))
echo "START $(date -u +%FT%TZ) nproc=$N jobs=$J pwd=$(pwd)"
SECONDS=0
lake build 2>&1
rc=$?
echo "END $(date -u +%FT%TZ) elapsed=${SECONDS}s exit=$rc"
exit $rc
