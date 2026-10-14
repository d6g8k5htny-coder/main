#!/usr/bin/env bash
# Activated manual-review run for main#403 (reviewer-private). usage: run403.sh EXPECTED_HEAD EXPECTED_BASE
set -u
EH=$1; EB=$2
S=${SCRATCH:?set SCRATCH to a scratch directory}; R=$S/rec2/r403; WT=$S/wt403; REPO=/home/user/main
echo "start $(date -u +%FT%TZ)"
cd $REPO || exit 2
git fetch -q origin main "+refs/pull/403/head:refs/remotes/pr403/head" "+refs/pull/403/merge:refs/remotes/pr403/merge" || exit 2
H=$(git rev-parse refs/remotes/pr403/head); B=$(git rev-parse origin/main); M=$(git rev-parse refs/remotes/pr403/merge)
echo "head $H (expected $EH) tree $(git rev-parse $H^{tree})"; echo "main $B (expected $EB)"
echo "parents of head: $(git rev-list --parents -n1 $H | cut -d' ' -f2-)"
echo "merge $M parents $(git rev-list --parents -n1 $M | cut -d' ' -f2-) tree $(git rev-parse $M^{tree})"
[ "$H" = "$EH" ] && [ "$B" = "$EB" ] || { echo "IDENTITY CHANGED - stop"; exit 3; }
echo "== changed paths vs base"; git diff --stat $B $H | tail -n 4
for p in formal/README.md formal/REVIEW_LANE.md formal/manifest.json; do echo "  $p base $(git rev-parse --short=8 $B:$p) head $(git rev-parse --short=8 $H:$p)"; done
echo "== dependency slice (head blob == base blob?)"
for p in tools/formal_gate_check.py tests/test_formal_gate.py tools/required_formal_check.py tests/test_required_formal_check.py tests/test_required_formal_binding_control.py tests/test_architecture_evidence_adapter.py tools/navigation_check.py tools/workspace_landing_check.py docs/FORMAL_VERIFICATION.md formal/SCOPE.md formal/GLOSSARY.md formal/ALIGNMENT.md docs/NAVIGATION.json; do
  if git cat-file -e $H:$p 2>/dev/null; then a=$(git rev-parse $H:$p); b=$(git rev-parse $B:$p); echo "  $p ${a:0:8} $([ "$a" = "$b" ] && echo same || echo DIFFERENT)"; else echo "  $p absent"; fi
done
rm -rf $WT; git worktree prune; git worktree add -q --detach $WT $H || exit 2
echo "== controls"; python3 -I $R/controls403.py $WT $B $H python3
cd $WT
for mode in "-B -S" "-B -O -S"; do
  echo "== PY [$mode] formal-related suites"
  python3 $mode -m unittest tests.test_formal_gate tests.test_required_formal_check tests.test_required_formal_binding_control tests.test_architecture_evidence_adapter tests.test_formal_execution_context 2>&1 | tail -n 3
done
echo "== navigation_check (offline)"; python3 -B -S tools/navigation_check.py 2>&1 | grep -E '"problems"|"pages"|"local_links"|"fragment_links"'
echo "== workspace_landing_check"; python3 -B -S tools/workspace_landing_check.py 2>&1 | head -n 1
echo "== git status"; git status --porcelain | head -n 5
echo "end $(date -u +%FT%TZ)"
