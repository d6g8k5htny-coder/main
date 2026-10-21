#!/bin/bash
# Bounded watcher for the Ready-triggered automatic Codex review on main#455.
# Usage: wait_codex.sh <head_sha> <since_iso> [max_seconds]
H=$1; SINCE=$2; MAX=${3:-1200}
F=/tmp/claude-0/-home-user/159201a6-4a35-5470-bbb3-a5e6d3d36154/scratchpad/rec3/m455/fresh
R=repos/d6g8k5htny-coder/main; t0=$(date +%s)
while :; do
  gh api "$R/issues/455/comments?since=$SINCE&per_page=100" > "$F/codex_comments.json" 2>/dev/null
  gh api "$R/pulls/455/reviews?per_page=100" > "$F/codex_reviews.json" 2>/dev/null
  gh api "$R/pulls/455/comments?per_page=100" > "$F/codex_review_comments.json" 2>/dev/null
  gh api "$R/issues/455/reactions?per_page=100" > "$F/codex_reactions.json" 2>/dev/null
  out=$(python3 -I - "$F" "$H" "$SINCE" <<'PY'
import json, re, sys
F, H, SINCE = sys.argv[1:4]
bot = 'chatgpt-codex-connector[bot]'
cm = [c for c in json.load(open(f'{F}/codex_comments.json')) if c['user']['login'] == bot]
rv = [r for r in json.load(open(f'{F}/codex_reviews.json')) if r['user']['login'] == bot]
rc = [c for c in json.load(open(f'{F}/codex_review_comments.json')) if c['user']['login'] == bot]
rx = [x for x in json.load(open(f'{F}/codex_reactions.json')) if x['user']['login'] == bot and x['created_at'] >= SINCE]
notes = []
limit = [c for c in cm if 'usage limit' in c['body'].lower()]
state = None
for c in cm:
    m = re.search(r'codex-review-state:v1 (\{.*?\}) -->', c['body'])
    if m:
        st = json.loads(m.group(1))['code_review']
        state = (c['id'], c['updated_at'], st)
revH = [r for r in rv if r.get('commit_id') == H]
notes.append(f"comments={len(cm)} limit_notices={[c['id'] for c in limit]} state={state} reviews_at_head={[(r['id'], r['state']) for r in revH]} review_comments={len(rc)} reactions={[(x['content'], x['created_at']) for x in rx]}")
done = False
if limit: done = True
if state and state[2].get('commit_sha') == H and state[2].get('status') in ('completed', 'failed', 'success'): done = True
if revH: done = True
if any(x['content'] == '+1' for x in rx): done = True
print(('DONE ' if done else 'WAIT ') + ' '.join(notes))
PY
)
  echo "$(date -u +%TZ) $out" >> "$F/wait_codex.log"
  case "$out" in DONE*) echo "$out"; exit 0;; esac
  if [ $(( $(date +%s) - t0 )) -ge "$MAX" ]; then echo "TIMEOUT $out"; exit 0; fi
  sleep 30
done
