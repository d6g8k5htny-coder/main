#!/bin/bash
# C8 watch v5: board/queue/site threads for anything addressed to C8, a resumption
# decision or Math-#371; Math- comments naming C8 or on #371/#454; and the check
# runs of C8's two PR heads (#371 e7524e60, #454 e8b08e63). No scheduled-run
# discovery (coordinator 6151507231 names 01a11d37 the one run-list observer).
# Patterns are passed into jq string literals, so a regex \b is written \\b here.
st=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C13/watch_state
mkdir -p "$st"
last=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pat='(?<![0-9A-Za-z])C8\\b|015wNj8L|6150458427|6150630672|6150875061|6151730158|#366|#329|371|BQ50|BQ38|Math-?#?219|XA-219|#454|(?i:resum|priority|successor|board cap)'
mpat='(?<![0-9A-Za-z])C8\\b|015wNj8L|XA-219'
self='Anthropic Claude, session `session_015wNj8L'
heads="main:9844f785910d097b7b4591dcbd9e377bd7b38280 454:d517287c9f603d5dc59896261c3147ab15a06411"
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for i in 307 275 388 227; do
    gh api "repos/d6g8k5htny-coder/main/issues/$i/comments?since=$last&per_page=100" \
      --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | select(.body|test(\"$pat\")) | \"main#$i \(.id) \(.created_at[11:19]) \(.body[0:220]|gsub(\"\n\";\" \"))\"" 2>/dev/null || true
  done
  # main#387 (the site PR): every new comment, so the operator's genuine-run announcement is seen.
  # main#358 (v9): every new comment while C8's DOC358 pickup 6153049449 is open.
  # main#390 (v11): the merged Share fix, where the integrator reports its push publication.
  for i in 387 390 358; do
    gh api "repos/d6g8k5htny-coder/main/issues/$i/comments?since=$last&per_page=100" \
      --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | \"main#$i \(.id) \(.created_at[11:19]) \(.body[0:260]|gsub(\"\n\";\" \"))\"" 2>/dev/null || true
  done
  # Math-#454 state (v10): report a head move (integrator refresh) or the merge, once each.
  s454=$(gh api repos/d6g8k5htny-coder/Math-/pulls/454 --jq '"\(.head.sha) \(.state) \(.merged)"' 2>/dev/null || true)
  if [ -n "$s454" ] && [ "$s454" != "d517287c9f603d5dc59896261c3147ab15a06411 open false" ]; then
    k=$(printf '%s' "$s454" | tr ' ' '_')
    [ -e "$st/s454.$k" ] || { : > "$st/s454.$k"; echo "STATE Math-#454 now: $s454"; }
  fi
  gh api "repos/d6g8k5htny-coder/Math-/issues/comments?since=$last&per_page=100" \
    --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | (.issue_url|split(\"/\")|.[-1]) as \$n | select((\$n == \"371\") or (\$n == \"454\") or (.body|test(\"$mpat\"))) | \"Math-#\(\$n) \(.id) \(.created_at[11:19]) \(.body[0:200]|gsub(\"\n\";\" \"))\"" 2>/dev/null || true
  for pair in $heads; do
    pr=${pair%%:*}; sha=${pair#*:}
    gh api "repos/d6g8k5htny-coder/Math-/commits/$sha/check-runs?per_page=100" \
      --jq '.check_runs[] | "\(.id) \(.status) \(.conclusion) \(.name)"' 2>/dev/null | while read -r id status concl name; do
        if [ "$status" = "completed" ] && [ ! -e "$st/run.$id" ]; then
          : > "$st/run.$id"
          echo "CHECK Math-#$pr ${sha:0:8} $name: $status/$concl"
        fi
      done
  done
  last=$now
  sleep 60
done
