#!/bin/bash
# tests/test_check_mode_workflows.py for one family in one interpreter mode.
# Usage: run_tests_b4.sh WORKTREE FAMILY MODE [SOURCE_ROOT]
# GREEN when SOURCE_ROOT is omitted (the worktree's own wrappers); RED control when it names
# a tree whose wrappers are unchanged (the tests then run the old shells). FAMILY 'all' runs
# every family. Prints the unittest output, then one summary line.
set -uo pipefail
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$1; FAMILY=$2; MODE=$3; SRC=${4:-}
cd "$WT" || exit 2
export TMPDIR=$C/tmp PYTHONDONTWRITEBYTECODE=1
[ "$FAMILY" != all ] && export CHECK_MODE_TEST_FAMILY=$FAMILY
[ -n "$SRC" ] && export CHECK_MODE_SOURCE_ROOT=$SRC
case $MODE in normal) F="-B -S" ;; optimized) F="-B -O -S" ;; *) echo "mode?"; exit 2 ;; esac
start=$(date +%s)
python3 $F tests/test_check_mode_workflows.py -v 2>&1
rc=${PIPESTATUS[0]}
echo "SUMMARY family=$FAMILY mode=$MODE source=${SRC:-self} exit=$rc seconds=$(( $(date +%s) - start )) python=$(python3 -c 'import platform; print(platform.python_version())')"
