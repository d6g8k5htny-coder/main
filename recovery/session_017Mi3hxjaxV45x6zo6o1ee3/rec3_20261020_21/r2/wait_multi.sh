#!/usr/bin/env bash
# Bounded watcher for: dot's #453 integrator naming, Cursor progress on #458, new #455 activity.
SINCE="${1:-2026-10-20T02:35:30Z}"; MAX="${2:-1500}"; t0=$(date +%s)
R=repos/d6g8k5htny-coder/main
MINE="6282008955 6282004051 6281995971 6281983583 6281941179"
VHEAD0=$(gh api $R/pulls/458 --jq .head.sha 2>/dev/null)
while :; do
  hits=""
  for n in 453 458 455; do
    o=$(gh api "$R/issues/$n/comments?since=$SINCE&per_page=100" --jq '.[] | "\(.id)|\(.created_at)|\(.updated_at)|\(.user.login)|\(.performed_via_github_app.slug // "-")|\(.body[0:260] | gsub("\n";" "))"' 2>/dev/null)
    while IFS= read -r l; do [ -z "$l" ] && continue; id=${l%%|*}; case " $MINE " in *" $id "*) continue;; esac
      case "$n" in
        458) if echo "$l" | grep -q '|cursor\[bot\]|' && echo "$l" | grep -q 'Taking a look!'; then continue; fi; hits+="#$n $l"$'\n';;
        453) if echo "$l" | grep -qiE '\|chatgpt-codex-connector\||017|Claude017|integrat'; then hits+="#$n $l"$'\n'; fi;;
        455) hits+="#$n $l"$'\n';;
      esac
    done <<< "$o"
  done
  VH=$(gh api $R/pulls/458 --jq .head.sha 2>/dev/null); [ -n "$VH" ] && [ "$VH" != "$VHEAD0" ] && hits+="#458 HEAD MOVED $VHEAD0 -> $VH"$'\n'
  M453=$(gh api $R/pulls/453 --jq .merged 2>/dev/null); [ "$M453" = "true" ] && hits+="#453 MERGED"$'\n'
  if [ -n "$hits" ]; then echo "== $(date -u +%H:%M:%SZ) EVENTS"; printf '%s' "$hits"; exit 0; fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX" ]; then echo "== $(date -u +%H:%M:%SZ) TIMEOUT ${MAX}s"; exit 0; fi
  sleep 30
done
