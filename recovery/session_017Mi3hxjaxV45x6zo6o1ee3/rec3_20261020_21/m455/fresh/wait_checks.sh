#!/bin/bash
# Wait until the check runs seen at the reviewed head are all completed on the refreshed head (or 25 min pass).
H=${1:-94ff9719be4450a278f4774602c764d779e60b84}
F=/tmp/claude-0/-home-user/159201a6-4a35-5470-bbb3-a5e6d3d36154/scratchpad/rec3/m455/fresh
end=$(( $(date +%s) + 1500 ))
while [ "$(date +%s)" -lt "$end" ]; do
  gh api "repos/d6g8k5htny-coder/main/commits/$H/check-runs?per_page=100" > "$F/checks_${H:0:8}.json" 2>/dev/null || { sleep 45; continue; }
  out=$(python3 -I - "$F/checks_${H:0:8}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))['check_runs']
need={'verify','public-intake','public-shop','landing-checks','architecture / research-architecture','formal / formal-evidence','browser-smoke','navigation'}
seen={}
for r in sorted(d, key=lambda r: r.get('started_at') or ''): seen.setdefault(r['name'],[]).append((r['status'],r['conclusion']))
done = need <= set(seen) and all(s=='completed' for v in seen.values() for (s,_) in v)
bad = any(s=='completed' and c not in ('success','neutral','skipped') for v in seen.values() for (s,c) in v)
print(('DONE' if done else ('FAILED' if bad else 'WAIT')), sorted((k, v[-1][0], v[-1][1]) for k,v in seen.items()))
PY
)
  echo "$(date -u +%TZ) $out" >> "$F/wait_checks.log"
  case "$out" in DONE*|FAILED*) echo "$out"; exit 0;; esac
  sleep 45
done
echo "TIMEOUT"; tail -1 "$F/wait_checks.log"; exit 1
