#!/usr/bin/env bash
# LOCAL runtime row: fresh detached worktree at the candidate head, shared clean packages, hosted step commands.
# Usage: row.sh <row> <head>
set -uo pipefail
S=<SCRATCH>
row=$1; head=$2
W=$S/lg2_rt/$row; L=$S/lg2_runs/$row.log
export PATH=$HOME/.elan/bin:$PATH LEAN_NUM_THREADS=2
rm -rf "$W"; git -C $S/lg2 worktree prune
git -C $S/lg2 worktree add --detach "$W" "$head" >/dev/null 2>&1 || { echo "worktree failed" > "$L"; exit 9; }
cd "$W"
link() { mkdir -p "$1/.lake"; ln -sfn $S/mathf/.lake/packages "$1/.lake/packages"; }
step() { local t0=$SECONDS; echo "### STEP $* @ $(date -u +%FT%TZ)"; "$@"; local c=$?; echo "### EXIT $c after $((SECONDS-t0))s: $*"; return $c; }
{
echo "### ROW LOCAL:claude-01Cz-container:$row head=$(git rev-parse HEAD) tree=$(git rev-parse HEAD^{tree}) start=$(date -u +%FT%TZ)"
echo "### HOST $(uname -srm) | $(python3 --version 2>&1) | $(git --version) at $(command -v git) | $(. /etc/os-release; echo $PRETTY_NAME) | nproc=$(nproc)"
rc=0
case ${row%%-*} in
  formal)
    link formal
    step python3 -B -S -m unittest discover -s formal/tests -v 2>&1 | tail -5
    step python3 formal/gate.py || rc=1
    step python3 formal/gate.py --execute || rc=1 ;;
  d2atom)   link formal; step bash companions/d2_atom_integral_20261006/replay.sh || rc=1 ;;
  d2moment) link formal; step bash companions/d2_moment_bridge_20261006/replay.sh || rc=1 ;;
  d2square) link formal; step bash companions/d2_square_schur_20261006/replay.sh || rc=1 ;;
  torus)    link formal; step bash companions/torus_filtration_20261011/replay.sh || rc=1 ;;
  cap)
    P=frontiers/cap_first_exit_lean_20261002; link $P
    step python3 -B -S -m unittest discover -s "$P" -p 'test_*.py' 2>&1 | tail -5
    step python3 -B -O -S -m unittest discover -s "$P" -p 'test_*.py' 2>&1 | tail -5
    step python3 -B -S "$P/gate.py" || rc=1
    step python3 -B -S "$P/gate.py" --execute || rc=1 ;;
  spec|specproof)
    if [ $row = spec ]; then P=specifications/lean_statements_20261010; else P=specifications/lean_statement_proofs_20261011; fi
    link $P
    step python3 -B -S "$P/test_check.py" 2>&1 | tail -5
    step python3 -B -O -S "$P/test_check.py" 2>&1 | tail -5
    step python3 -B -S "$P/check.py" || rc=1
    step python3 -B -S "$P/test_check.py" LeanHelpers 2>&1 | tail -5
    step python3 -B -S "$P/check.py" --execute || rc=1 ;;
esac
echo "### TRACKED DIFF:"; git diff --stat; git diff --quiet || rc=1
echo "### PACKAGES AFTER:"; for p in $S/mathf/.lake/packages/*/; do printf '%s %s %s\n' "$(basename $p)" "$(git -C $p rev-parse HEAD)" "$( [ -z "$(git -C $p status --porcelain --untracked-files=normal)" ] && echo clean || echo dirty)"; done
echo "### ROW RESULT rc=$rc end=$(date -u +%FT%TZ)"
} > "$L" 2>&1
exit $rc
