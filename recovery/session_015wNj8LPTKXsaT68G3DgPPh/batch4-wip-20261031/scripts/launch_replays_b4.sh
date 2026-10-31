#!/bin/bash
# Start, detached: the real lifetime and d3 checkers through the helper (one job each, both modes,
# compared with run A's digests) and the full suite on the branch tree at lower priority.
# Logs: val/replay_lifetime.log, val/replay_d3.log, val/full_suite.log; a 'done' marker per job.
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$C/wt4; V=$C/char/b4/val; S=$C/char/scripts
export TMPDIR=$C/tmp PYTHONDONTWRITEBYTECODE=1
mkdir -p $V
for job in lifetime d3; do
  nohup bash -c "bash $S/real_replay_b4.sh $WT $C/char/b4/runA $job > $V/replay_$job.log 2>&1; echo \"exit=\$? \$(date -u +%FT%TZ)\" > $V/replay_$job.done" > /dev/null 2>&1 &
done
nohup bash -c "cd $WT && nice -n 10 python3 -B -S -m unittest discover -s tests -v > $V/full_suite.log 2>&1; echo \"exit=\$? \$(date -u +%FT%TZ)\" >> $V/full_suite.log; echo done > $V/full_suite.done" > /dev/null 2>&1 &
sleep 2
echo "launched $(date -u +%FT%TZ): $(pgrep -fc real_replay_b4.sh) replay jobs, suite pid $(pgrep -f 'unittest discover -s tests' | head -1)"
