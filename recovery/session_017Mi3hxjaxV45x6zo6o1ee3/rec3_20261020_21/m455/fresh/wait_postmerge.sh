#!/bin/bash
# Wait until all check runs on the landed merge commit are completed (at least 3 present, stable for one poll), or 25 min.
M=d43a7c6387a9c02814412470447adc42a56763d8
F=/tmp/claude-0/-home-user/159201a6-4a35-5470-bbb3-a5e6d3d36154/scratchpad/rec3/m455/fresh
end=$(( $(date +%s) + 1500 )); prev=""
while [ "$(date +%s)" -lt "$end" ]; do
  gh api "repos/d6g8k5htny-coder/main/commits/$M/check-runs?per_page=100" > "$F/checks_post.json" 2>/dev/null || { sleep 60; continue; }
  out=$(python3 -I - "$F/checks_post.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))['check_runs']
allc = len(d)>=3 and all(r['status']=='completed' for r in d)
bad = [r['name'] for r in d if r['status']=='completed' and r['conclusion'] not in ('success','neutral','skipped')]
print(('DONE' if allc else 'WAIT'), len(d), 'bad=' + ','.join(bad), sorted((r['name'], r['status'], r['conclusion']) for r in d))
PY
)
  echo "$(date -u +%TZ) $out" >> "$F/wait_postmerge.log"
  case "$out" in DONE*) if [ "$out" = "$prev" ]; then echo "$out"; exit 0; fi;; esac
  prev="$out"; sleep 60
done
echo "TIMEOUT $prev"; exit 1
