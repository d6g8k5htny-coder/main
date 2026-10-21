#!/usr/bin/env bash
# Report #466 check-run completions at 3dd0be24 and any new comment/review (e.g. the automatic review's response);
# exit when all check runs are complete AND the connector has responded, or on a failed check, or if the head moves.
R=repos/d6g8k5htny-coder/main; H=3dd0be245634f3b9f75bd90709993183c98b7f1c
seen=""; codex=0
while true; do
  head=$(gh api $R/pulls/466 --jq .head.sha 2>/dev/null || echo err)
  if [ "$head" != "$H" ] && [ "$head" != "err" ]; then echo "$(date -u +%H:%M:%SZ) HEAD MOVED to $head"; exit 0; fi
  runs=$(gh api "$R/commits/$H/check-runs?per_page=100" 2>/dev/null) || { sleep 45; continue; }
  done_now=$(jq -r '.check_runs[] | select(.status=="completed") | "\(.name)=\(.conclusion)"' <<<"$runs" | sort)
  new=$(comm -13 <(echo "$seen") <(echo "$done_now") | tr '\n' ' ')
  [ -n "${new// /}" ] && echo "$(date -u +%H:%M:%SZ) completed: $new"
  seen=$done_now
  total=$(jq '.total_count' <<<"$runs"); pending=$(jq '[.check_runs[] | select(.status!="completed")] | length' <<<"$runs")
  failed=$(jq -r '[.check_runs[] | select(.status=="completed" and (.conclusion!="success" and .conclusion!="skipped" and .conclusion!="neutral"))] | length' <<<"$runs")
  c=$(gh api "$R/issues/466/comments?per_page=100&since=2026-10-20T21:31:40Z" --jq '[.[] | select(.user.login|test("codex|bot";"i"))] | length' 2>/dev/null || echo 0)
  rv=$(gh api "$R/pulls/466/reviews?per_page=100" --jq '[.[] | select(.user.login|test("codex";"i"))] | length' 2>/dev/null || echo 0)
  if [ "$c" != "0" ] || [ "$rv" != "0" ]; then [ "$codex" = "0" ] && echo "$(date -u +%H:%M:%SZ) connector activity: comments=$c reviews=$rv"; codex=1; fi
  if [ "$failed" != "0" ]; then echo "$(date -u +%H:%M:%SZ) FAILED checks present ($failed)"; exit 0; fi
  if [ "$total" -ge 3 ] && [ "$pending" = "0" ] && [ "$codex" = "1" ]; then echo "$(date -u +%H:%M:%SZ) ALL $total checks complete; connector responded"; exit 0; fi
  sleep 45
done
