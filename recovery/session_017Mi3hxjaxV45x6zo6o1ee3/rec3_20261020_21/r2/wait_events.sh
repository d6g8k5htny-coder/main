#!/usr/bin/env bash
# Bounded watcher: new activity relevant to 017's open offers (main#455 integration; #451 R2; #419 routing).
# Exits on: any new comment on #455 or #451 not authored by 017's own posts; a #419 comment by dot (codex connector),
# by the Grok app, or mentioning 017/455/R2/451; or timeout.
SINCE="${1:-2026-10-20T02:03:00Z}"; MAX="${2:-1500}"; t0=$(date +%s)
MINE="6281602703 6281606897 6281593007 6281523784"
while :; do
  hits=""
  for n in 455 451; do
    o=$(gh api "repos/d6g8k5htny-coder/main/issues/$n/comments?since=$SINCE&per_page=100" \
      --jq '.[] | "\(.id)|\(.created_at)|\(.updated_at)|\(.user.login)|\(.performed_via_github_app.slug // "-")|\(.body[0:240] | gsub("\n";" "))"' 2>/dev/null)
    while IFS= read -r l; do [ -z "$l" ] && continue; id=${l%%|*}; case " $MINE " in *" $id "*) continue;; esac; hits+="#$n $l"$'\n'; done <<< "$o"
  done
  o=$(gh api "repos/d6g8k5htny-coder/main/issues/419/comments?since=$SINCE&per_page=100" \
      --jq '.[] | "\(.id)|\(.created_at)|\(.updated_at)|\(.user.login)|\(.performed_via_github_app.slug // "-")|\(.body[0:240] | gsub("\n";" "))"' 2>/dev/null)
  while IFS= read -r l; do [ -z "$l" ] && continue; id=${l%%|*}; case " $MINE " in *" $id "*) continue;; esac
    if echo "$l" | grep -qiE '\|chatgpt-codex-connector\||\|grok-by-xai\||017|#455|main#455|R2|#451|GROK-H0'; then hits+="#419 $l"$'\n'; fi; done <<< "$o"
  if [ -n "$hits" ]; then echo "== $(date -u +%H:%M:%SZ) EVENTS"; printf '%s' "$hits"; exit 0; fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX" ]; then echo "== $(date -u +%H:%M:%SZ) TIMEOUT ${MAX}s, no relevant events since $SINCE"; exit 0; fi
  sleep 30
done
