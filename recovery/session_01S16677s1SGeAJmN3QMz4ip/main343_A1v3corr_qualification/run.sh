#!/bin/bash
set -u
W=/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/wt3way
Q=/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/qual3way
cd $W
echo "tree $(git write-tree) python $(python3.12 --version)" > $Q/summary.txt
step(){ name=$1; shift; "$@" > $Q/$name.log 2>&1; rc=$?; echo "$name rc=$rc $(grep -E '^Ran |^OK|^FAILED' $Q/$name.log | tr '\n' ' ')" >> $Q/summary.txt; }
step s1_all_normal python3.12 -B -E -S -m unittest discover -s experiments/universality -p 'test_*.py' -v
step s1_all_O python3.12 -B -E -O -S -m unittest discover -s experiments/universality -p 'test_*.py' -v
step s2_pop_normal python3.12 -B -E -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p 'test_population_certificate.py' -v
step s2_pop_O python3.12 -B -E -O -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p 'test_population_certificate.py' -v
rm -rf $Q/explore
step s3_pilot_run python3.12 -B -E -S experiments/universality/run_pilot.py --output $Q/explore
step s3_pilot_verify python3.12 -B -E -S experiments/universality/run_pilot.py --verify $Q/explore/observations.json
step s4_published python3.12 -B -E -S experiments/universality/verify_published.py experiments/universality/results/exploratory50
rm -rf $Q/campaign
step s5_campaign python3.12 -B -S $Q/runcamp.py "$W" $Q/campaign
echo "worktree status after: $(git status --short | wc -l) changed (expect 1 staged test file)" >> $Q/summary.txt
git status --short >> $Q/summary.txt
echo DONE >> $Q/summary.txt
