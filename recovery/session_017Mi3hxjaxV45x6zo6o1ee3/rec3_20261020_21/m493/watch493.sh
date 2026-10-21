#!/usr/bin/env bash
# Wake on the Cursor verdict for Math-#493 (summon 6301631363, 21:14:30Z): any comment other than the summon and
# the author's 15:27Z release, any edit of a cursor "Taking a look!" comment, a PR review or review comment, or a head/base change.
R=repos/d6g8k5htny-coder/Math-
look=""
while true; do
  js=$(gh api "$R/issues/493/comments?per_page=100" 2>/dev/null) || { sleep 60; continue; }
  other=$(jq -r '[.[] | select(.id!=6294511354 and .id!=6301631363) | select((.user.login=="cursor[bot]" and (.body|test("^Taking a look";"i"))) | not)] | length' <<<"$js")
  newlook=$(jq -r '[.[] | select(.user.login=="cursor[bot]" and (.body|test("^Taking a look";"i"))) | .id] | join(",")' <<<"$js")
  if [ -n "$newlook" ] && [ "$newlook" != "$look" ]; then echo "$(date -u +%H:%M:%SZ) cursor acknowledged: $newlook"; look=$newlook; fi
  edited=$(jq -r '[.[] | select(.user.login=="cursor[bot]" and (.body|test("^Taking a look";"i")|not))] | length' <<<"$js")
  rv=$(gh api "$R/pulls/493/reviews?per_page=100" --jq length 2>/dev/null || echo 0)
  rc=$(gh api "$R/pulls/493/comments?per_page=100" --jq length 2>/dev/null || echo 0)
  hb=$(gh api "$R/pulls/493" --jq '.head.sha[0:8]+" "+.base.sha[0:8]+" "+.state' 2>/dev/null || echo err)
  if [ "$other" != "0" ] || [ "$edited" != "0" ] || [ "$rv" != "0" ] || [ "$rc" != "0" ] || { [ "$hb" != "f2350f47 5e20834d open" ] && [ "$hb" != "err" ]; }; then
    echo "$(date -u +%H:%M:%SZ) #493 activity: other_comments=$other cursor_nonlook=$edited reviews=$rv review_comments=$rc pr=[$hb]"
    exit 0
  fi
  sleep 60
done
