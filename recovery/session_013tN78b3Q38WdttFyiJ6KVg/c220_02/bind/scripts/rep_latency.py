#!/usr/bin/env python3
"""DOM-change -> speechd 'played' latency for each WIDGET message in the replication.
usage: python3 -I rep_latency.py EVIDENCE_ROOT RUN..."""
import re, sys
from datetime import datetime
root = sys.argv[1]
def t(s): return datetime.strptime(s, '%H:%M:%S.%f')
for run in sys.argv[2:]:
    dom = []
    for l in open(f'{root}/{run}/driver.log', encoding='utf-8', errors='replace'):
        m = re.search(r'dom\((\d\d:\d\d:\d\d\.\d+)\): \[WIDGET status\] "(.*)"', l)
        if m: dom.append((t(m.group(1)), m.group(2)))
    out = []
    for l in open(f'{root}/{run}/extract.out', encoding='utf-8', errors='replace'):
        m = re.search(r'sent (\S+) played (\S+) \[WIDGET live region\] (.*?)  <', l)
        if not m: continue
        played, text = t(m.group(2)), m.group(3)
        prior = [d for d in dom if d[1] == text and d[0] <= played]
        lat = (played - prior[-1][0]).total_seconds() if prior else None
        out.append(f'{text[:28]}… {lat:.2f}s' if lat is not None else f'{text[:28]}… ?')
    print(run, '|', '; '.join(out))
