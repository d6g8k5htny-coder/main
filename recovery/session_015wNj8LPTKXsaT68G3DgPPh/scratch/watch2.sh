#!/bin/bash
# wake on new non-self comments since T0 in hub issues and queued Math- PRs, or #316 ready
T0="$1"; SEEN="$2"
while true; do
  out=""
  for n in 229 275; do
    out+=$(gh api "repos/d6g8k5htny-coder/main/issues/$n/comments?since=$T0&per_page=100" --jq ".[] | select(.created_at > \"$T0\") | \"main#$n \(.id) \(.created_at) \(.body[0:900]|gsub(\"\n\";\" \"))\"" 2>/dev/null)$'\n'
  done
  for n in 316 363 383 371 373 377 380; do
    out+=$(gh api "repos/d6g8k5htny-coder/Math-/issues/$n/comments?since=$T0&per_page=100" --jq ".[] | select(.created_at > \"$T0\") | \"Math-#$n \(.id) \(.created_at) \(.body[0:900]|gsub(\"\n\";\" \"))\"" 2>/dev/null)$'\n'
  done
  d=$(gh api repos/d6g8k5htny-coder/Math-/pulls/316 --jq .draft 2>/dev/null)
  [ "$d" = "false" ] && out+="Math-#316 READY"$'\n'
  new=$(echo "$out" | grep -v "^$" | awk "NR==FNR{s[\$1];next} !(\$2 in s)" "$SEEN" - | grep -E "^Math-|C8|015wNj8|6020039992|6020660006|6019578759|6021006690|6021396860|W v1\.1|U v1\.1|retrofit-W|retrofit-U|U-DELTA-r2|U-CORR-r3|U-READ-r4|BQ-10|BQ10|6022792894|6022625756|6022474553|TIP LOCK|#3(63|71|73|77|80)|Math-#3(63|71|73|77|80)|PR ?3(63|71|73|77|80))")
  if [ -n "$new" ]; then echo "$new"; exit 0; fi
  sleep 90
done
