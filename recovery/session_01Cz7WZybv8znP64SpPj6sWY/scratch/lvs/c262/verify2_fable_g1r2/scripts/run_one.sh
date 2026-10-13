#!/usr/bin/env bash
# usage: run_one.sh <mutant_dir_name> <mode: plain|O> [ambient PYTHONOPTIMIZE value or -]
D=<SCRATCH>/lvs/c262/verify2_fable_g1r2
M="$1"; MODE="$2"; AMB="${3:--}"
T="$D/$M/companions/torus_filtration_20261011/test_torus_filtration.py"
LOG="$D/logs/${M}_${MODE}$( [ "$AMB" != - ] && echo "_amb$AMB" ).log"
mkdir -p "$D/logs"
if [ "$MODE" = O ]; then FLAGS="-B -O -S"; else FLAGS="-B -S"; fi
if [ "$AMB" = - ]; then
  env -u PYTHONOPTIMIZE python3 $FLAGS "$T" > "$LOG" 2>&1; rc=$?
else
  env PYTHONOPTIMIZE="$AMB" python3 $FLAGS "$T" > "$LOG" 2>&1; rc=$?
fi
echo "EXIT=$rc" >> "$LOG"
