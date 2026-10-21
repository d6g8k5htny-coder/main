#!/usr/bin/env bash
# Exit when every check run on the #466 merge commit has completed (or after a failure), printing the final table.
R=repos/d6g8k5htny-coder/main; M=bf60f9d2ca4330a896fa0ee24a9e2242a2ef2db2
sleep 60
while true; do
  js=$(gh api "$R/commits/$M/check-runs?per_page=100" 2>/dev/null) || { sleep 45; continue; }
  total=$(jq .total_count <<<"$js"); pending=$(jq '[.check_runs[]|select(.status!="completed")]|length' <<<"$js")
  bad=$(jq '[.check_runs[]|select(.status=="completed" and .conclusion!="success" and .conclusion!="skipped" and .conclusion!="neutral")]|length' <<<"$js")
  if [ "$total" -gt 0 ] && { [ "$pending" = "0" ] || [ "$bad" != "0" ]; }; then
    date -u +%H:%M:%SZ; jq -r '.check_runs[] | "\(.name) \(.status)/\(.conclusion) run=\(.details_url|capture("runs/(?<r>[0-9]+)").r)"' <<<"$js"
    gh api "$R/actions/runs?head_sha=$M&per_page=50" --jq '.workflow_runs[] | "\(.id) \(.name) \(.event) \(.status)/\(.conclusion)"'
    exit 0
  fi
  sleep 45
done
