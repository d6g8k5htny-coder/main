#!/usr/bin/env python3
"""Tranche-8 objects: the Math- directories that no disposition locates or binds (from math_dirs.py), grouped and batched.
Usage: python3 -I -B t8_objects.py MATH_DIRS_JSON CANDIDATE OUT_DIR"""
import json, os, sys
dirs_p, cand_p, out_dir = sys.argv[1:4]
d = json.load(open(dirs_p))
cand = json.load(open(cand_p))
res = [r for r in d['dirs'] if not r['covered_by_location'] and r['n_unbound'] and not r['excluded_engineering']]
names = {r['dir'] for r in res}
MERGE_SUFFIX = ('/nodes', '/results', '/controls', '/controls/results', '/benjamin', '/lucas')
def owner(dname):
    x = dname
    while any(x.endswith(sfx) for sfx in MERGE_SUFFIX) and x.rsplit('/', 1)[0] in names:
        x = x.rsplit('/', 1)[0]
    return x
records = {}
for key in ('quarantines', 'proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments'):
    for r in cand.get(key, []):
        records[r['id']] = (key, r)
def where(obj, needle, path=''):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += where(v, needle, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += where(v, needle, f'{path}[{i}]')
    elif isinstance(obj, str) and needle in obj:
        out.append(path)
    return out
objs = {}
for r in res:
    o = objs.setdefault(owner(r['dir']), {'object': owner(r['dir']), 'repository': 'd6g8k5htny-coder/Math-', 'commit': d['ref'],
                                          'files': [], 'mentioned_by': {}})
    o['files'] += [f for f in r['files'] if not (f['bound'] or f['closed'])]
for o in objs.values():
    needles = [o['object'] + '/'] + [f['path'] for f in o['files']]
    for rid, (key, rec) in records.items():
        hits = sorted({p for n in needles for p in where(rec, n)})
        if hits:
            o['mentioned_by'][rid] = {'section': key, 'fields': hits[:12], 'title': rec.get('title', '')[:240]}
def batch(name):
    if name.startswith('frontiers/lifetime_dd_model_d_20261013'):
        return 1
    if name.startswith('frontiers'):
        return 2
    if name.startswith('reviews/'):
        return 3
    if name.startswith(('repairs/', 'incoming/', 'certificates', 'claims')):
        return 4
    return 5
os.makedirs(out_dir, exist_ok=True)
by = {}
for name in sorted(objs):
    by.setdefault(batch(name), []).append(objs[name])
for b, os_ in sorted(by.items()):
    with open(os.path.join(out_dir, f'mbatch{b}.json'), 'x') as f:
        json.dump({'batch': b, 'objects': os_}, f, indent=1, ensure_ascii=False)
    print(b, len(os_), 'objects', sum(len(o['files']) for o in os_), 'files', [o['object'] for o in os_][:20])
