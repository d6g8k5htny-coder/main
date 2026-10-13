#!/bin/sh
# P5: tests/test_museum_data.py::test_historical_snapshot_ignores_unrelated_working_tree_append runs
# `git init`, `git add`, `git commit` in a temp dir WITHOUT clearing inherited GIT_DIR/GIT_WORK_TREE.
# With GIT_DIR exported (git hooks export it; so did this review's first object-reading harness) the
# test re-initialises and COMMITS INTO THE REPOSITORY GIT_DIR NAMES, then still passes.
# Reproduced here only against a hand-made scratch host repository (no `git init` used to create it).
H=$(cd "$(dirname "$0")" && pwd); HOST="$H/hostrepo"; N="$H/wt_p5"
rm -rf "$HOST" "$N"; mkdir -p "$HOST/.git/objects" "$HOST/.git/refs/heads" "$HOST/.git/refs/tags"
printf 'ref: refs/heads/host-branch\n' > "$HOST/.git/HEAD"
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > "$HOST/.git/config"
cp -a "$H/../../cd32303" "$N"; rm -rf "$N/tools/__pycache__"; cd "$N"
echo "--- host repo commits before: $(git --git-dir="$HOST/.git" rev-list --all 2>/dev/null | wc -l)"
GIT_DIR="$HOST/.git" python3 -B -E -s -m unittest -q tests.test_museum_data.MuseumDataTests.test_historical_snapshot_ignores_unrelated_working_tree_append 2>&1 | tail -1
echo "--- host repo after:"; git --git-dir="$HOST/.git" log --all --format='    %h %an <%ae> | %s' ; git --git-dir="$HOST/.git" show --stat --format= HEAD | sed 's/^/    /'
