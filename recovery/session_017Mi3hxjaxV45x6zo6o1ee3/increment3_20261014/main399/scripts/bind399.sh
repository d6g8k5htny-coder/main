#!/usr/bin/env bash
# Reader-binding controls for a refreshed main#399 head (reviewer-private).
# usage: bind399.sh NEW_HEAD NEW_BASE   (both full SHAs, already fetchable)
set -u
NEW_HEAD=$1; NEW_BASE=$2
S=${SCRATCH:?set SCRATCH to a scratch directory}
R=$S/rec2/r388/review; REPO=/home/user/main; WT=$S/bind399
cd $REPO || exit 2
git fetch -q origin "+refs/pull/399/head:refs/remotes/pr399/head" "+refs/pull/399/merge:refs/remotes/pr399/merge" main || exit 2
echo "head=$(git rev-parse refs/remotes/pr399/head) merge=$(git rev-parse refs/remotes/pr399/merge) main=$(git rev-parse origin/main)"
[ "$(git rev-parse refs/remotes/pr399/head)" = "$NEW_HEAD" ] || echo "WARN: fetched head differs from NEW_HEAD"
echo "== parents of head: $(git rev-list --parents -n1 $NEW_HEAD)"
echo "== tree head $(git rev-parse $NEW_HEAD^{tree}) merge $(git rev-parse refs/remotes/pr399/merge^{tree})"
echo "== semantic blobs (reviewed: 191f5a8f e8944c9a a687aa44 1b60543c; contract 84ac641d)"
for p in docs/site/atlas.mjs docs/site/current-model.mjs tests/test_current_site.mjs tools/current_site_browser_check.py tests/test_current_site_browser_contract.py; do
  echo "  $p $(git rev-parse $NEW_HEAD:$p | cut -c1-8)"
done
echo "== reviewed head -> new head paths: $(git diff --name-only e80f370e28a32eda18c7364cd92c36919db21e4a $NEW_HEAD | wc -l)"
git diff --stat e80f370e28a32eda18c7364cd92c36919db21e4a $NEW_HEAD | tail -3
echo "== relcheck vs new base"
python3 -I $R/../review/relcheck.py $REPO $NEW_BASE $NEW_HEAD docs/site/atlas.mjs docs/site/current-model.mjs tests/test_current_site.mjs tools/current_site_browser_check.py 2>&1 | tail -8
rm -rf $WT; git worktree prune; git worktree add -q --detach $WT $NEW_HEAD || exit 2
cd $WT
echo "== --check"; python3 -B tools/site_asset_release.py --check > $R/bind_check.out 2>&1; echo "rc=$?"; tail -2 $R/bind_check.out
echo "== node test_current_site.mjs"; node --test tests/test_current_site.mjs 2>&1 | grep -E '^# (tests|pass|fail|skipped)|^not ok'
for mode in "-B -S" "-B -O -S"; do
  echo "== PY [$mode] current-site suites"; python3 $mode -m unittest tests.test_current_site_data tests.test_current_site_build tests.test_current_site_history tests.test_current_site_publication tests.test_current_site_browser_contract 2>&1 | tail -3
done
echo "== git status"; git status --porcelain | head -3; echo "end $(date -u +%H:%M:%SZ)"
