#!/usr/bin/env bash
# T1 resolution for main#282: resolve one thread per call through the proxy's native endpoint.
# Usage: bash t1_resolve.sh <first-comment-id> [more ids...]
# Prints one line per id with the HTTP outcome; stops at the first failure so the outcome can be reported.
set -u
for id in "$@"; do
  out=$(gh api -X POST "repos/d6g8k5htny-coder/main/pulls/282/ccr/comments/${id}/resolve" 2>&1)
  rc=$?
  echo "RESOLVE ${id} rc=${rc} :: $(echo "$out" | tr '\n' ' ' | cut -c1-240)"
  if [ $rc -ne 0 ]; then
    echo "STOP at ${id}"
    exit $rc
  fi
done
