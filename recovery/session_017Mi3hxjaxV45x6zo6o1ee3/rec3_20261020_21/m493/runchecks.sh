#!/usr/bin/env bash
# Summoner-side replay for Math-#493: the new test at the head (both modes) and against the base wrapper (RED).
set -u
M=/tmp/claude-0/-home-user/159201a6-4a35-5470-bbb3-a5e6d3d36154/scratchpad/rec3/m493
W=$M/wt; WB=$M/wt_base
python3 --version
cd "$W"
for flags in "-B -S" "-B -O -S"; do
  echo "=== HEAD python3 $flags"; start=$(date +%s)
  python3 $flags -m unittest discover -s tests -p 'test_elder_lower_review_workflow.py' -v > "$M/head_${flags// /_}.log" 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-start ))"; tail -4 "$M/head_${flags// /_}.log"
done
git status --porcelain | head; echo "head worktree dirty lines: $(git status --porcelain | wc -l)"
cd /home/user/Math- && rm -rf "$WB" && git worktree prune && git worktree add -q --detach "$WB" f2350f47f18105ed73a004b67c2f699ca7f2ffe4
cd "$WB" && git show 5e20834d7daccb8d6bf8b89036e8eb9eb98a1452:.github/workflows/elder-lower-review.yml > .github/workflows/elder-lower-review.yml && git commit -q -am "probe: base wrapper with head test (local only)" --author="probe <probe@invalid>" -c user.name=probe -c user.email=probe@invalid 2>/dev/null || git -c user.name=probe -c user.email=probe@invalid commit -q -am "probe: base wrapper with head test (local only)"
echo "base-wrapper probe commit $(git rev-parse --short HEAD), workflow blob $(git rev-parse --short=8 HEAD:.github/workflows/elder-lower-review.yml)"
for flags in "-B -S" "-B -O -S"; do
  echo "=== BASE WRAPPER python3 $flags"; start=$(date +%s)
  python3 $flags -m unittest discover -s tests -p 'test_elder_lower_review_workflow.py' -v > "$M/base_${flags// /_}.log" 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-start ))"; tail -4 "$M/base_${flags// /_}.log"
done
echo DONE
