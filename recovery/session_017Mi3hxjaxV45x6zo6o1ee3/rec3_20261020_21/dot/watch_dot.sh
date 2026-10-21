#!/usr/bin/env bash
# Wait for dot's reply to 419/6321647892, anyone addressing Claude017, or a
# competing main#472 integration offer. Prints matches and exits; gives up
# after MAX_MIN minutes.
set -u
SINCE="${1:-2026-10-21T16:57:05Z}"
MAX_MIN="${2:-55}"
SELF_IDS="${3:-6321647892}"
OUT=/tmp/claude-0/-home-user/159201a6-4a35-5470-bbb3-a5e6d3d36154/scratchpad/rec3/dot/watch_dot.out
: > "$OUT"
for i in $(seq 1 "$MAX_MIN"); do
  hits=""
  for r in main Math- query-; do
    page=$(gh api "repos/d6g8k5htny-coder/$r/issues/comments?since=$SINCE&per_page=100" --paginate 2>/dev/null) || continue
    h=$(printf '%s' "$page" | jq -s -r --arg r "$r" --arg since "$SINCE" --arg self "$SELF_IDS" '
      add // [] | .[]
      | select(.created_at > $since)
      | select((.id|tostring) as $id | ($self | split(",") | index($id)) | not)
      | select(
          (.body | test("^\\s*(\\*\\*)?dot\\b"; "i"))
          or (.body | test("dot \\(OpenAI\\)"))
          or (.body | test("017Mi3|Claude017|6321647892"))
          or ((.body | test("#472|/pull/472|main#472")) and (.body | test("INTEGRATION|integrat"; "i")))
        )
      | [$r, (.id|tostring), .created_at, .html_url, (.body | gsub("\n"; " ") | .[0:400])] | @tsv')
    [ -n "$h" ] && hits="$hits$h"$'\n'
  done
  if [ -n "$hits" ]; then
    printf '%s' "$hits" | sort -u | tee "$OUT"
    exit 0
  fi
  sleep 60
done
echo "NO MATCH after $MAX_MIN min (since $SINCE)" | tee "$OUT"
