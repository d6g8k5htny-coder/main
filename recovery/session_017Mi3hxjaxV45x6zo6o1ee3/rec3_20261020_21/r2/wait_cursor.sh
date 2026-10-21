#!/usr/bin/env bash
# Bounded watcher: report cursor[bot] (and any other new) comments on main#451 after the summon.
# Exits when a cursor[bot] comment other than a bare ack appears, or a decline/pickup, or after MAX seconds.
SINCE="${1:-2026-10-20T01:56:52Z}"; MAX="${2:-900}"; t0=$(date +%s); seen=""
while :; do
  out=$(gh api "repos/d6g8k5htny-coder/main/issues/451/comments?since=$SINCE&per_page=100" \
        --jq '.[] | select(.id != 6281523784) | "\(.id)|\(.created_at)|\(.updated_at)|\(.user.login)|\(.body[0:220] | gsub("\n";" "))"' 2>/dev/null)
  if [ "$out" != "$seen" ]; then echo "== $(date -u +%H:%M:%SZ)"; echo "$out"; seen="$out"; fi
  # stop when cursor posted something beyond the ack (pickup/decline/result) or ack was edited
  if echo "$out" | grep -q '|cursor\[bot\]|' && echo "$out" | grep '|cursor\[bot\]|' | grep -qvE '\|Taking a look!'; then echo "STOP: cursor substantive post"; exit 0; fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX" ]; then echo "STOP: timeout ${MAX}s"; exit 0; fi
  sleep 20
done
