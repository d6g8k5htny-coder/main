#!/usr/bin/env bash
# Wake when anything new appears on main#467 after the Cursor "Taking a look!" (6301389521, 21:01:04Z):
# a new issue comment (any author but this session's summon), an edit of the look comment, a PR review or a review comment.
R=repos/d6g8k5htny-coder/main
while true; do
  c=$(gh api "$R/issues/467/comments?per_page=100&since=2026-10-20T21:01:05Z" --jq '[.[] | select(.id!=6301389521 and .id!=6301387598)] | length' 2>/dev/null || echo err)
  u=$(gh api "$R/issues/comments/6301389521" --jq .updated_at 2>/dev/null || echo err)
  rv=$(gh api "$R/pulls/467/reviews?per_page=100" --jq 'length' 2>/dev/null || echo err)
  rc=$(gh api "$R/pulls/467/comments?per_page=100&since=2026-10-20T21:00:00Z" --jq 'length' 2>/dev/null || echo err)
  hd=$(gh api "$R/pulls/467" --jq '.head.sha[0:8]+" "+.state+" draft="+(.draft|tostring)' 2>/dev/null || echo err)
  if { [ "$c" != "0" ] && [ "$c" != "err" ]; } || { [ "$u" != "2026-10-20T21:01:04Z" ] && [ "$u" != "err" ]; } || { [ "$rv" != "0" ] && [ "$rv" != "err" ]; } || { [ "$rc" != "0" ] && [ "$rc" != "err" ]; } || { [ "$hd" != "d19b619c open draft=true" ] && [ "$hd" != "err" ]; }; then
    echo "$(date -u +%H:%M:%SZ) #467 activity: new_comments=$c look_updated=$u reviews=$rv review_comments=$rc pr=[$hd]"
    exit 0
  fi
  sleep 60
done
