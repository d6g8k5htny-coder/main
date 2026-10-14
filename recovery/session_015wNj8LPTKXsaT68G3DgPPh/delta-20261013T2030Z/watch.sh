#!/bin/bash
# C8 watch v17 (read-only). Reports:
#  - board/queue threads (#307 #275 #388 #227): comments addressed to C8, BQ38, a resumption or priority decision;
#  - v17: any comment anywhere in main that names C8 (repo-wide scan), so an offer on any PR is seen;
#  - every new comment on the threads listed in $st/allposts (one number per line; edit the file to add
#    the reader-test repair PR without restarting);
#  - new PRs opened on main, once each, so the repair PR's number is seen;
#  - a state change (closed/merged) of any open PR listed in $st/allposts, once each;
#  - Math- comments on #371/#454/#380/#329/#336 (v16: #329 preflight 6157624313, #336 HOLD 6157525280) or naming C8.
# No scheduled-run discovery (coordinator 6151507231 names 01a11d37 the one run-list observer).
# Patterns are passed into jq string literals, so a regex \b is written \\b here.
# Usage: watch.sh [START_ISO]   (START_ISO defaults to now; overlapping windows are de-duplicated by comment id)
st=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C13/watch_state
mkdir -p "$st"
[ -s "$st/allposts" ] || printf '%s\n' 387 390 358 396 > "$st/allposts"
last=${1:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}
start=$last
pat='(?<![0-9A-Za-z])C8\\b|015wNj8L|6150458427|6150630672|6150875061|6151730158|#366|#329|371|BQ50|BQ38|Math-?#?219|XA-219|#454|(?i:resum|priority|successor|board cap)'
mpat='(?<![0-9A-Za-z])C8\\b|015wNj8L|XA-219'
self='Anthropic Claude, session `session_015wNj8L'
R=d6g8k5htny-coder/main
once() { [ -e "$st/$1" ] && return 1; : > "$st/$1"; return 0; }
emit() { while read -r id rest; do [ -n "$id" ] && once "c.$id" && echo "$rest"; done; }
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for i in 307 275 388 227; do
    gh api "repos/$R/issues/$i/comments?since=$last&per_page=100" \
      --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | select(.body|test(\"$pat\")) | \"\(.id) main#$i \(.id) \(.created_at[11:19]) \(.body[0:220]|gsub(\"\n\";\" \"))\"" 2>/dev/null | emit
  done
  gh api "repos/$R/issues/comments?since=$last&per_page=100" \
    --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | (.issue_url|split(\"/\")|.[-1]) as \$n | select(.body|test(\"$mpat\")) | \"\(.id) main#\(\$n) \(.id) \(.created_at[11:19]) \(.body[0:220]|gsub(\"\n\";\" \"))\"" 2>/dev/null | emit
  for i in $(cat "$st/allposts"); do
    gh api "repos/$R/issues/$i/comments?since=$last&per_page=100" \
      --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | \"\(.id) main#$i \(.id) \(.created_at[11:19]) \(.body[0:260]|gsub(\"\n\";\" \"))\"" 2>/dev/null | emit
    s=$(gh api "repos/$R/pulls/$i" --jq '"\(.state)_\(.merged)_\(.head.sha[0:8])"' 2>/dev/null || true)
    if [ -n "$s" ]; then
      if [ -e "$st/pr.$i" ]; then
        old=$(cat "$st/pr.$i")
        [ "$old" != "$s" ] && { echo "STATE main#$i: $old -> $s"; printf '%s' "$s" > "$st/pr.$i"; }
      else
        printf '%s' "$s" > "$st/pr.$i"
      fi
    fi
  done
  gh api "repos/$R/pulls?state=all&sort=created&direction=desc&per_page=5" \
    --jq ".[] | select(.created_at >= \"$start\") | \"\(.number) NEWPR main#\(.number) \(.created_at[11:19]) \(.head.ref) | \(.title[0:120])\"" 2>/dev/null \
    | while read -r n rest; do [ -n "$n" ] && once "newpr.$n" && echo "$rest"; done
  gh api "repos/d6g8k5htny-coder/Math-/issues/comments?since=$last&per_page=100" \
    --jq ".[] | select(.created_at >= \"$last\") | select((.body|contains(\"$self\"))|not) | (.issue_url|split(\"/\")|.[-1]) as \$n | select((\$n == \"371\") or (\$n == \"454\") or (\$n == \"380\") or (\$n == \"329\") or (\$n == \"336\") or (.body|test(\"$mpat\"))) | \"\(.id) Math-#\(\$n) \(.id) \(.created_at[11:19]) \(.body[0:200]|gsub(\"\n\";\" \"))\"" 2>/dev/null | emit
  last=$now
  sleep 60
done
