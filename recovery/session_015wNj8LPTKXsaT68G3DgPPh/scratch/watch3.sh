#!/bin/bash
# wake on new C8-relevant comments since T0 (all pages) in hub issues and queued Math- PRs, or #316 ready
T0="$1"; SEEN="$2"
fetch() { # $1=repo $2=issue $3=label
  local p=1
  while :; do
    local r; r=$(gh api "repos/d6g8k5htny-coder/$1/issues/$2/comments?since=$T0&per_page=100&page=$p" --jq ".[] | select(.created_at > \"$T0\") | \"$3 \(.id) \(.created_at) \(.body[0:900]|gsub(\"\n\";\" \"))\"" 2>/dev/null) || break
    [ -z "$r" ] && break; echo "$r"; p=$((p+1)); [ $p -gt 10 ] && break
  done
}
while true; do
  out=""
  for n in 229 275; do out+=$(fetch main $n "main#$n")$'\n'; done
  for n in 316 363 371 373 377 380; do out+=$(fetch Math- $n "Math-#$n")$'\n'; done
  d=$(gh api repos/d6g8k5htny-coder/Math-/pulls/316 --jq .draft 2>/dev/null)
  [ "$d" = "false" ] && out+="Math-#316 READY"$'\n'
  new=$(echo "$out" | grep -v "^$" | awk 'NR==FNR{s[$1];next} !($2 in s)' "$SEEN" - | grep -E "^Math-|C8|015wNj8|BQ-?38|BQ38|BQ-?44|BQ44|BQ-?35|BQ35|TIP LOCK|#3(63|71|73|77|80)\b")
  if [ -n "$new" ]; then echo "$new"; exit 0; fi
  sleep 90
done
