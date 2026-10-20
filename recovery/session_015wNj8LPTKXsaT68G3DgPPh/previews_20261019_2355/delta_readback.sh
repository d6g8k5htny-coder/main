#!/usr/bin/env bash
# Bounded delta readback for a base-only refresh of a reviewed main PR.
# Usage: [DEPS='path ...'] delta_readback.sh PR REVIEWED_HEAD PRIOR_HEAD TRIAL_TREE [SIMULATED_HEAD]
#   DEPS: reviewed dependencies, checked at the reviewed base, current main and the new head
#   REVIEWED_HEAD: head the C8 review covered (blobs compared against it)
#   PRIOR_HEAD:    head before the refresh (expected first parent)
#   TRIAL_TREE:    C8's preview tree for merge(main, PRIOR_HEAD)
set -u
PR=$1; REVIEWED=$2; PRIOR=$3; TRIAL=$4
R=/home/user/main
git -C "$R" fetch -q origin main "pull/$PR/head:refs/c8/pr$PR" 2>&1 | tail -2
MAIN=$(git -C "$R" rev-parse origin/main)
NEW=${5:-$(git -C "$R" rev-parse "refs/c8/pr$PR")}  # optional 5th arg: simulated head for a dry run
API=$(gh api "repos/d6g8k5htny-coder/main/pulls/$PR" --jq '"\(.head.sha) \(.base.sha) \(.mergeable_state) \(.merged)"')
echo "### $(date -u +%H:%M:%SZ) PR#$PR main=$MAIN"
echo "api head/base/state/merged: $API"
echo "fetched head: $NEW"
echo "parents: $(git -C "$R" rev-list --parents -n1 "$NEW" | cut -d' ' -f2-)"
NEWTREE=$(git -C "$R" rev-parse "$NEW^{tree}")
FRESH=$(git -C "$R" merge-tree --write-tree "$MAIN" "$PRIOR" | head -1)
echo "new head tree:   $NEWTREE"
echo "fresh merge-tree(main, prior head): $FRESH"
echo "C8 trial tree:   $TRIAL"
[ "$NEWTREE" = "$TRIAL" ] && echo "new tree == trial tree: yes" || echo "new tree == trial tree: NO"
[ "$FRESH" = "$TRIAL" ] && echo "fresh == trial: yes" || echo "fresh == trial: NO"
[ "$NEWTREE" = "$FRESH" ] && echo "new tree == fresh merge-tree: yes" || echo "new tree == fresh merge-tree: NO"
echo "--- delta paths main..new head vs reviewed blobs"
same=0; diff=0
while IFS= read -r p; do
  a=$(git -C "$R" rev-parse -q --verify "$NEW:$p" 2>/dev/null || echo missing)
  b=$(git -C "$R" rev-parse -q --verify "$REVIEWED:$p" 2>/dev/null || echo missing)
  if [ "$a" = "$b" ]; then same=$((same+1)); else diff=$((diff+1)); echo "DIFF $p new=$a reviewed=$b"; fi
done < <(git -C "$R" diff --name-only "$MAIN" "$NEW")
echo "delta paths: $((same+diff)); unchanged vs reviewed head: $same; changed: $diff"
echo "--- main..new head delta == merge-base..reviewed delta (paths)?"
MB=$(git -C "$R" merge-base "$MAIN" "$REVIEWED")
cmp <(git -C "$R" diff --name-only "$MAIN" "$NEW" | sort) <(git -C "$R" diff --name-only "$MB" "$REVIEWED" | sort) && echo yes || echo NO
echo "--- raw diff main..new head == raw diff merge-base..reviewed head (same paths, modes, before and after blobs, so identical patches)?"
cmp <(git -C "$R" diff --raw --no-abbrev "$MAIN" "$NEW") <(git -C "$R" diff --raw --no-abbrev "$MB" "$REVIEWED") && echo yes || echo NO
echo "--- delta paths changed on main between reviewed base $MB and $MAIN (expect none)"
comm -12 <(git -C "$R" diff --name-only "$MB" "$MAIN" | sort) <(git -C "$R" diff --name-only "$MAIN" "$NEW" | sort) | sed 's/^/OVERLAP /'
echo "(end of overlap list)"
echo "--- base delta: files changed on main since the reviewed base $MB"
git -C "$R" diff --name-status "$MB" "$MAIN"
if [ -n "${DEPS:-}" ]; then
  echo "--- reviewed dependencies (DEPS): object at reviewed base / current main / new head"
  for d in $DEPS; do
    a=$(git -C "$R" rev-parse -q --verify "$MB:$d" 2>/dev/null || echo missing)
    b=$(git -C "$R" rev-parse -q --verify "$MAIN:$d" 2>/dev/null || echo missing)
    c=$(git -C "$R" rev-parse -q --verify "$NEW:$d" 2>/dev/null || echo missing)
    [ "$a" = "$b" ] && [ "$b" = "$c" ] && st=unchanged || st=CHANGED
    echo "$st ${a:0:8} ${b:0:8} ${c:0:8} $d"
  done
fi
echo "--- diffstat main..new head"
git -C "$R" diff --numstat "$MAIN" "$NEW" | awk '{a+=$1; d+=$2; n++} END {print n" files, +"a"/-"d}'
[ "$(git -C "$R" diff --name-only "$MAIN" "$NEW" | wc -l)" -le 10 ] && git -C "$R" diff --numstat "$MAIN" "$NEW"
echo "--- check runs at new head"
gh api "repos/d6g8k5htny-coder/main/commits/$NEW/check-runs?per_page=100" --jq '.check_runs[] | "\(.name) \(.status) \(.conclusion) \(.completed_at)"' | sort
