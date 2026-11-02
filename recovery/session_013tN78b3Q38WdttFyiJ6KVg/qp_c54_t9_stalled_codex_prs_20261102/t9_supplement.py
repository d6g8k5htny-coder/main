#!/usr/bin/env python3
"""Tranche-9 supplement: Math-#531 is not an ancestor of #532 or #536, so t9_objects.py's "later PR wins" rule bound some of
#531's paths at #532's or #536's different bytes. This writes pbatch4.json: one object per directory holding #531's own version
of each such path, bound at #531's head. The object id is "<directory>@pr531"; "directory" is the repository directory.
Usage: python3 -I -B t9_supplement.py INVENTORY_JSON FACTS_DIR"""
import json, os, sys, collections
inv, facts_dir = sys.argv[1:]
prs = {r['pr']: r for r in json.load(open(inv))['prs']}
bound = {}
for b in (3,):
    for o in json.load(open(os.path.join(facts_dir, f'pbatch{b}.json')))['objects']:
        for f in o['files']:
            bound[f['path']] = (o['pr'], f['git_blob'])
r531 = prs[531]
objs = collections.defaultdict(list)
for f in r531['files_not_on_main']:
    pr, blob = bound[f['path']]
    if pr != 531 and not f['git_blob'].startswith(blob) and not blob.startswith(f['git_blob']):
        d = f['path'].rsplit('/', 1)[0] if '/' in f['path'] else '(root)'
        objs[d].append({'path': f['path'], 'git_blob': f['git_blob'], 'carried_by_prs': [531], 'copy_of': None,
                        'other_version': {'pr': pr, 'git_blob': blob}})
out = [{'object': f'{d}@pr531', 'directory': d, 'stack': 'determinant stack (Math-#531, #532, #536); #531 is not an ancestor of '
        '#532 or #536', 'pr': 531, 'repository': 'd6g8k5htny-coder/Math-', 'commit': r531['head'], 'files': fs}
       for d, fs in sorted(objs.items())]
with open(os.path.join(facts_dir, 'pbatch4.json'), 'x') as f:
    json.dump({'batch': 4, 'objects': out}, f, indent=1)
print(len(out), 'objects', sum(len(o['files']) for o in out), 'files')
