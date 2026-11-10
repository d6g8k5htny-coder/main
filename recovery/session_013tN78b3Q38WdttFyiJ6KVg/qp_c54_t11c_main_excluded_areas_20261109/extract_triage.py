#!/usr/bin/env python3
"""Collect triage verdicts from workflow journals into one JSON {"N": result}; a triage-retry result replaces a missing one.
Usage: python3 -I -B extract_triage.py OUT JOURNAL [JOURNAL...]"""
import json, sys
out, *journals = sys.argv[1:]
res, retry = {}, {}
for j in journals:
    rs = [json.loads(l) for l in open(j)]
    lab = {e['key']: e.get('label') for e in rs if e.get('type') == 'started'}
    for e in rs:
        if e.get('type') == 'result' and e.get('result') and lab.get(e['key'], '').startswith('triage'):
            kind, n = lab[e['key']].split(':')
            (retry if kind == 'triage-retry' else res)[n] = e['result']
for n, r in retry.items():
    res.setdefault(n, r)
json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
print(len(res), 'batches with results;', sorted(int(k) for k in res)[:5], '...')
