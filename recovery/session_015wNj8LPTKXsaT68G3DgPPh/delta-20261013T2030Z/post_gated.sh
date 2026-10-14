#!/bin/bash
# Usage: post_gated.sh REPO ISSUE MDFILE SINCE_ISO
# Posts MDFILE to REPO#ISSUE only if no comment newer than SINCE_ISO exists there; then reads back.
set -u
R="$1"; N="$2"; F="$3"; S="$4"
new=$(gh api "repos/d6g8k5htny-coder/$R/issues/$N/comments?since=$S&per_page=100" --jq "[.[] | select(.created_at > \"$S\")] | length")
if [ "$new" != "0" ]; then
  echo "ABORT: $new new comment(s) on $R#$N since $S"
  gh api "repos/d6g8k5htny-coder/$R/issues/$N/comments?since=$S&per_page=100" --jq ".[] | select(.created_at > \"$S\") | \"\(.id) \(.created_at[11:19]) \(.body[0:200]|gsub(\"\n\";\" \"))\""
  exit 3
fi
python3 -I -c 'import json,sys; print(json.dumps({"body": open(sys.argv[1]).read()}))' "$F" > "$F.json" || exit 1
id=$(gh api -X POST "repos/d6g8k5htny-coder/$R/issues/$N/comments" --input "$F.json" --jq '.id') || exit 1
echo "posted $id"
gh api "repos/d6g8k5htny-coder/$R/issues/comments/$id" --jq '"\(.created_at)", .body' > "rb_$id.txt"
python3 -I -c '
import sys
a=open(sys.argv[1]).read().rstrip("\n")
lines=open(sys.argv[2]).read().split("\n",1)
print("created", lines[0])
b=lines[1].rstrip("\n")
n=lambda s: s.replace("\n\n\n---\n","\n\n---\n")
print("MATCH" if n(a)==n(b) else "DIFF", len(a), len(b))
' "$F" "rb_$id.txt"
: > "watch_state/c.$id"
