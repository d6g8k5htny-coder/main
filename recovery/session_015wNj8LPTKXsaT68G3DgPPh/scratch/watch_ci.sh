#!/bin/bash
# wait until certified-replay on #215 f3642ea and #212 3d46520 are both completed; print conclusions
for i in $(seq 1 60); do
  a=$(gh api "repos/d6g8k5htny-coder/Math-/commits/f3642ea941d285eda6a591561a7444d61745d2db/check-runs" --jq '[.check_runs[] | select(.status!="completed")] | length' 2>/dev/null || echo 9)
  b=$(gh api "repos/d6g8k5htny-coder/Math-/commits/3d46520ae417425ea24ab32574b44837f868bac5/check-runs" --jq '[.check_runs[] | select(.status!="completed")] | length' 2>/dev/null || echo 9)
  if [ "$a" = "0" ] && [ "$b" = "0" ]; then break; fi
  sleep 60
done
for s in f3642ea941d285eda6a591561a7444d61745d2db 3d46520ae417425ea24ab32574b44837f868bac5; do
  echo "== ${s:0:7}"; gh api "repos/d6g8k5htny-coder/Math-/commits/$s/check-runs" --jq '.check_runs[] | "\(.name): \(.status) \(.conclusion)"'
done
