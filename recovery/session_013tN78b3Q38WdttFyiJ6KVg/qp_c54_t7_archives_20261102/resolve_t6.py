#!/usr/bin/env python3
"""Second resolution step for tranche 7: mark archive members that are byte-identical, at the same path, to a file a tranche-6 row
binds in the archive's own repository at the same commit. Such a member is a copy whose disposition is that tranche-6 row's.

Usage: python3 -I -B resolve_t6.py FACTS_DIR RECORDS_T6 OUT_DIR

Reads abatch<N>.json from FACTS_DIR (as the workflow saw them) and writes the same objects to OUT_DIR, with those members moved
from "members" to "copies" (copy_of "tranche-6 <row>"). Nothing else changes."""
import glob, json, os, sys
facts_dir, t6_path, out_dir = sys.argv[1:4]
t6 = json.load(open(t6_path))
bound = {}
for r in t6['records'] + t6['other_items']:
    b = r['source_binding']
    for f in b.get('files') or []:
        bound[(b['repository'], b['commit'], f['path'], f['git_blob'])] = r.get('id') or f'{r["object"]}#{r["item_id"]}'
os.makedirs(out_dir, exist_ok=True)
moved = 0
for path in sorted(glob.glob(os.path.join(facts_dir, 'abatch*.json'))):
    d = json.load(open(path))
    for o in d['objects']:
        keep = []
        for m in o['members']:
            key = (o['repository'], o['commit'], m['member'], m['git_blob'])
            if key in bound:
                o['copies'].append(dict(m, copy_of=f'tranche-6 {bound[key]}'))
                moved += 1
            else:
                keep.append(m)
        o['members'] = keep
    with open(os.path.join(out_dir, os.path.basename(path)), 'x') as f:
        json.dump(d, f, indent=1)
print(moved, 'members resolved as copies of tranche-6 bindings')
