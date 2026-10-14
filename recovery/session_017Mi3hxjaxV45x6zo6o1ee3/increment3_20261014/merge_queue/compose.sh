#!/usr/bin/env bash
# Local-only composition of open main PRs onto origin/main (never pushed).
# usage: compose.sh WORKTREE ORDER...   (ORDER = PR numbers whose heads are fetched as refs/remotes/prN/head)
set -u
WT=$1; shift
cd /home/user/main || exit 2
rm -rf "$WT"; git worktree prune; git worktree add -q --detach "$WT" origin/main || exit 2
cd "$WT"
echo "base $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})"
for n in "$@"; do
  h=$(git rev-parse refs/remotes/pr$n/head)
  if git -c user.name=local-sim -c user.email=local-sim@invalid merge --no-ff --no-edit -q "$h" >/dev/null 2>&1; then
    echo "merge #$n ${h:0:8}: clean -> tree $(git rev-parse HEAD^{tree})"
  else
    echo "merge #$n ${h:0:8}: CONFLICT"; git diff --name-only --diff-filter=U; git merge --abort; exit 1
  fi
done
echo "final tree $(git rev-parse HEAD^{tree})"
