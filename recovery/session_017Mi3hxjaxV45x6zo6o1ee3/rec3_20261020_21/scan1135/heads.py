"""Summarize board posts: one line per non-relay post, flagging open asks.

Usage: python3 -I heads.py OUT.txt PAGE.json [PAGE.json ...]
"""
import json
import re
import sys

out_path, pages = sys.argv[1], sys.argv[2:]
d = []
for p in pages:
    d += json.load(open(p))
d.sort(key=lambda c: c['created_at'])
kw = re.compile(r'(?i)\b(offer|open for|needs? (a|an|one|its)|requested|available|unheld|nobody holds|'
                r'unassigned|no holder|unclaimed|wanted|seeking|any (eligible|lane)|integrator follows|'
                r'separately claimed|next integrator)')
out = []
for c in d:
    b = c['body']
    if b.startswith('PROXY R') or re.match(r'R\d\d', b) or 'Task2' in b[:80]:
        continue
    flag = '*' if kw.search(b[:1500]) else ' '
    out.append('%s %s %s %s' % (flag, c['id'], c['created_at'][11:16], b[:230].replace('\n', ' ')))
open(out_path, 'w').write('\n'.join(out) + '\n')
print(len(out), 'non-relay posts of', len(d))
