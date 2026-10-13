#!/bin/bash
cd /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/p343336/agents/343-tests-mutation
for d in $(python3.12 -c "import json;print(' '.join(r['mutant'] for r in json.load(open('mutation_results.json')) if r['verdict']=='SURVIVED'))"); do
  out=$(python3.12 -B -E -S proposed_tests.py mutants/$d 2>&1)
  failed=$(echo "$out" | grep -E '^(FAIL|ERROR): ' | sed -E 's/^(FAIL|ERROR): (test_[a-z_]+).*/\2/' | sort -u | tr '\n' ' ')
  if echo "$out" | grep -q '^OK'; then v=SURVIVES_PROPOSED; else v=KILLED_BY_PROPOSED; fi
  printf '%-20s %-42s %s\n' "$v" "$d" "$failed"
done
echo DONE
