#!/usr/bin/env python3
"""Tranche-9 objects: the unmerged mathematical files of the stalled Math- PR stacks, from pr_inventory.py outputs.
Each file is bound at the latest PR head of its stack that carries it (stacks are listed bottom to top); files are grouped by
directory into objects; byte-identical files in other directories of the same stack are marked copy_of.
Usage: python3 -I -B t9_objects.py INVENTORY_JSON [INVENTORY_JSON...] OUT_DIR"""
import json, os, sys, collections
*invs, out_dir = sys.argv[1:]
prs = {}
for p in invs:
    for r in json.load(open(p))['prs']:
        prs[r['pr']] = r
STACKS = [  # (batch, stack name, PRs bottom to top)
    (1, 'compact_source_cutoff (Math-#414, #415, #424, #425, #427 closed; #426, #430 open)', [430]),
    (2, 'C299 critical ridge (Math-#491 -> #496 -> #497 -> #504)', [491, 496, 497, 504]),
    (3, 'determinant stack (Math-#531, #532, #536)', [531, 532, 536]),
    (3, 'downstream-gate SOURCE_ADMISSION graph (Math-#490)', [490]),
    (3, 'finite MomentTail CI binding (Math-#545)', [545]),
]
batches = collections.defaultdict(list)
for batch, stack, order in STACKS:
    files = {}
    for n in order:
        for f in prs[n]['files_not_on_main']:
            files[f['path']] = dict(f, pr=n, commit=prs[n]['head'])  # later PR wins
    carried = collections.defaultdict(list)
    for n in order:
        for f in prs[n]['files_not_on_main']:
            carried[f['path']].append(n)
    first_dir = {}
    for path in sorted(files):
        first_dir.setdefault(files[path]['git_blob'], path)
    objs = collections.defaultdict(list)
    for path, f in sorted(files.items()):
        d = path.rsplit('/', 1)[0] if '/' in path else '(root)'
        cp = first_dir[f['git_blob']]
        objs[(d, f['pr'], f['commit'])].append({'path': path, 'git_blob': f['git_blob'], 'carried_by_prs': carried[path],
                                                'copy_of': cp if cp != path else None})
    for (d, n, commit), fs in sorted(objs.items()):
        batches[batch].append({'object': d, 'stack': stack, 'pr': n, 'repository': 'd6g8k5htny-coder/Math-', 'commit': commit,
                               'files': fs})
os.makedirs(out_dir, exist_ok=True)
for b, objs in sorted(batches.items()):
    with open(os.path.join(out_dir, f'pbatch{b}.json'), 'x') as f:
        json.dump({'batch': b, 'objects': objs}, f, indent=1)
    print(b, len(objs), 'objects', sum(len(o['files']) for o in objs), 'files',
          sum(1 for o in objs for x in o['files'] if x['copy_of']), 'in-stack copies')
