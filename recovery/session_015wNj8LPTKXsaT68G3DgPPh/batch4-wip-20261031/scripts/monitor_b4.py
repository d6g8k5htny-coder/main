#!/usr/bin/env python3
"""Emit one line per newly recorded characterization child, per new chain.log line and per
traceback in the run logs; exit when the chain reports the control's exit or an incomplete run A.
Usage: monitor_b4.py B4DIR [POLL_SECONDS]"""
import json, pathlib, sys, time
b4 = pathlib.Path(sys.argv[1]); poll = float(sys.argv[2]) if len(sys.argv) > 2 else 20
seen = {}
def new_lines(path):
    if not path.exists():
        return []
    lines = path.read_text(errors='replace').splitlines()
    n = seen.get(path, 0); seen[path] = len(lines)
    return lines[n:]
# do not replay history that already happened before the monitor started, except the chain log
for p in list((b4 / 'runA').glob('*.jsonl')) + list((b4 / 'control').glob('*.jsonl')) + [b4 / 'runA.log', b4 / 'control.log']:
    new_lines(p)
while True:
    for d in ('runA', 'control'):
        for p in sorted((b4 / d).glob('*.jsonl')):
            for line in new_lines(p):
                try:
                    r = json.loads(line)
                    print('%s child: %s %s %s rc=%s %ss out=%s err=%s' % (d, r['family'].split('-')[1][:8], r['mode'], r['label'],
                          r['returncode'], r['seconds'], r['stdout_sha256'][:8], r['stderr_sha256'][:8]), flush=True)
                except Exception as e:
                    print(d, 'unparsable record:', e, flush=True)
    for name in ('runA.log', 'control.log'):
        for line in new_lines(b4 / name):
            if 'Traceback' in line or 'Error' in line or 'Killed' in line:
                print(name + ':', line[:200], flush=True)
    done = False
    for line in new_lines(b4 / 'chain.log'):
        print('chain:', line, flush=True)
        if 'control exit=' in line or 'runA incomplete' in line:
            done = True
    if done:
        break
    time.sleep(poll)
