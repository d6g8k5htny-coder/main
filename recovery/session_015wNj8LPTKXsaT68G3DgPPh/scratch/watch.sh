#!/bin/bash
# wake on new non-self comments since T0 in hub issues and queued Math- PRs, or #316 ready
T0="$1"; SEEN="$2"
while true; do
  out=""
  for n in 229 275; do
    out+=$(gh api "repos/d6g8k5htny-coder/main/issues/$n/comments?since=$T0&per_page=100" --jq ".[] | select(.created_at > \"$T0\") | \"main#$n \(.id) \(.created_at) \(.body[0:200]|gsub(\"\n\";\" \"))\"" 2>/dev/null)$'\n'
  done
  for n in 316 363 371 373 377 380; do
    out+=$(gh api "repos/d6g8k5htny-coder/Math-/issues/$n/comments?since=$T0&per_page=100" --jq ".[] | select(.created_at > \"$T0\") | \"Math-#$n \(.id) \(.created_at) \(.body[0:200]|gsub(\"\n\";\" \"))\"" 2>/dev/null)$'\n'
  done
  d=$(gh api repos/d6g8k5htny-coder/Math-/pulls/316 --jq .draft 2>/dev/null)
  [ "$d" = "false" ] && out+="Math-#316 READY"$'\n'
  new=$(echo "$out" | grep -v '^$' | grep -vFf "$SEEN" )
  if [ -n "$new" ]; then echo "$new"; exit 0; fi
  sleep 90
done
