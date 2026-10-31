#!/bin/bash
# After characterization run A finishes, run the --procs 2 control. (Run B is replaced by the
# real replay through the helper, which compares every digest with run A.)
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
export TMPDIR=$C/tmp PYTHONDONTWRITEBYTECODE=1
LOG=$C/char/b4/chain.log
echo "chain2 start $(date -u +%FT%TZ) pid $$" >> $LOG
while pgrep -f "characterize_b4.py .*b4/runA" >/dev/null; do sleep 20; done
echo "runA finished $(date -u +%FT%TZ)" >> $LOG
if [ ! -f $C/char/b4/runA/c8-theorem-a-constants-d3.json ] || [ ! -f $C/char/b4/runA/c8-lifetime-difference-constant.json ]; then
  echo "runA incomplete; no control $(date -u +%FT%TZ)" >> $LOG; exit 1
fi
timeout 7200 python3 -I -B -S $C/char/control_b4.py $C/pristine4 $C/char/b4/control >> $C/char/b4/control.log 2>&1
echo "control exit=$? $(date -u +%FT%TZ)" >> $LOG
