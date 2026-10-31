#!/bin/bash
# Batch-4 validation after characterization run A (and the --procs 2 control) have finished:
#   1. merge the measured verdicts into the contract (run A as both runs; the helper replay below is
#      the second independent run), moving the custody digest;
#   2. GREEN: the check-mode tests per new family and for all seven families, both modes;
#   3. RED: the same tests against the unchanged base tree (CHECK_MODE_SOURCE_ROOT);
#   4. the real checkers through the helper (both families, each in both modes) and, beside them at
#      lower priority, the full suite on the branch tree.
# Logs: C14/char/b4/val. Usage: validate_b4.sh [steps]   (steps: merge tests red replay suite; default all)
set -uo pipefail
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$C/wt4; BASE=$C/pristine4; V=$C/char/b4/val; S=$C/char/scripts
mkdir -p $V $C/tmp
export TMPDIR=$C/tmp PYTHONDONTWRITEBYTECODE=1
STEPS=${*:-merge tests red replay suite}
FAMS="c8-lifetime-difference-constant c8-theorem-a-constants-d3"
log() { echo "$(date -u +%T) $*" | tee -a $V/validate.log; }
for step in $STEPS; do
  case $step in
    merge)
      log "merge: restoring the contract file, then merging run A"
      (cd $WT && git checkout -- tools/check_mode_contracts.json) || exit 1
      python3 -I -B -S $S/merge_contracts_b4.py $WT $C/char/b4/runA $C/char/b4/runA > $V/merge.log 2>&1 || { log "merge FAILED"; cat $V/merge.log; exit 1; }
      head -2 $V/merge.log | tee -a $V/validate.log ;;
    tests)
      for fam in $FAMS all; do for mode in normal optimized; do
        bash $S/run_tests_b4.sh $WT $fam $mode > $V/green_${fam}_$mode.log 2>&1
        log "GREEN $(grep -E '^Ran ' $V/green_${fam}_$mode.log) $(grep -E '^(OK|FAILED)' $V/green_${fam}_$mode.log) :: $(tail -1 $V/green_${fam}_$mode.log)"
      done; done ;;
    red)
      for fam in $FAMS; do for mode in normal optimized; do
        bash $S/run_tests_b4.sh $WT $fam $mode $BASE > $V/red_${fam}_$mode.log 2>&1
        log "RED $(grep -E '^Ran ' $V/red_${fam}_$mode.log) $(grep -E '^(OK|FAILED)' $V/red_${fam}_$mode.log) :: $(tail -1 $V/red_${fam}_$mode.log)"
      done; done
      python3 -I -B -S $S/red_counts.py $V/red_*.log > $V/red_counts.txt; cat $V/red_counts.txt | tee -a $V/validate.log ;;
    replay)
      log "replay: starting the real checkers through the helper (lifetime, d3)"
      bash $S/real_replay_b4.sh $WT $C/char/b4/runA > $V/replay.log 2>&1
      log "replay exit=$? :: $(grep -E 'REAL_REPLAY_OK|exit=' $V/replay.log | tr '\n' ' ')" ;;
    suite)
      log "suite: full suite on the branch tree, normal mode"
      (cd $WT && nice -n 10 python3 -B -S -m unittest discover -s tests -v > $V/full_suite.log 2>&1; echo "exit=$?" >> $V/full_suite.log)
      log "suite $(grep -E '^Ran ' $V/full_suite.log) $(grep -E '^(OK|FAILED)' $V/full_suite.log) $(tail -1 $V/full_suite.log)" ;;
    *) log "unknown step $step"; exit 2 ;;
  esac
done
log "validate done: $STEPS"
