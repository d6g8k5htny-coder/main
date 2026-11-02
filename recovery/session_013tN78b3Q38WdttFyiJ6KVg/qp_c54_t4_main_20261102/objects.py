#!/usr/bin/env python3
"""Group uncovered main directories into objects. Usage: MAIN_DIRS_JSON CLOSED_JSON CANDIDATE OUT"""
import json, sys, collections
md, closed_p, cand_p, out = sys.argv[1:5]
d = json.load(open(md))
closed = {e['path'] for e in json.load(open(closed_p))['entries'] if e['repository'].endswith('main')}
cand = json.load(open(cand_p))
# main PROOF.md-bearing disposed packet directories (for packet-level coverage)
covered_dirs = {r['dir'] for r in d['dirs'] if r['covered_by_location']}
DEPTH = {'docs/reports': 3, 'docs/manuscript': 3, 'docs/cbz': 2, 'docs/superpowers': 3, 'docs/research-translation': 4,
         'experiments/periodic_h0': 3, 'experiments/universality': 4, 'formal/UniversalLaw': 3, 'formal/sources': 3,
         'audits/dropbox_import': 3, 'incoming/claude-bundle-v3-native-copies-20260928': 2}
def obj_of(dirpath):
    parts = dirpath.split('/')
    for pre, n in DEPTH.items():
        if dirpath == pre or dirpath.startswith(pre + '/'):
            return '/'.join(parts[:n])
    return '/'.join(parts[:2])
objs = collections.defaultdict(lambda: {'dirs': [], 'files': [], 'census_artifacts': set()})
for r in d['dirs']:
    if r['covered_by_location']:
        continue
    o = obj_of(r['dir'])
    objs[o]['dirs'].append(r['dir'])
    objs[o]['files'] += r['files']
    objs[o]['census_artifacts'] |= set(r['census_artifacts'])
rows = []
for o in sorted(objs):
    v = objs[o]
    lean = [f['path'] for f in v['files'] if f['path'].endswith('.lean')]
    pre = 'PRE_'
    if lean and all(p in closed for p in lean) and all(f['path'].endswith('.lean') for f in v['files']):
        pre_cls = 'CLOSED_SOURCE'
    elif all(f['copy_of'] or f['exact_location'] for f in v['files']):
        pre_cls = 'ALL_DISPOSED_OR_COPIES'
    elif o.startswith('formal/UniversalLaw/P15'):
        pre_cls = 'EXCLUDED_740_LANE'
    else:
        pre_cls = 'TO_CLASSIFY'
    rows.append({'object': o, 'pre': pre_cls, 'n_dirs': len(v['dirs']), 'n_files': len(v['files']),
                 'bytes': sum(f['bytes'] for f in v['files']), 'lean_files': lean,
                 'lean_closed': [p for p in lean if p in closed], 'copies': sum(bool(f['copy_of']) for f in v['files']),
                 'exact_locations': sum(f['exact_location'] for f in v['files']),
                 'census_artifacts': sorted(v['census_artifacts']), 'dirs': v['dirs'], 'files': v['files']})
json.dump({'ref': d['summary']['ref'], 'objects': rows}, open(out, 'x'), indent=1)
c = collections.Counter(r['pre'] for r in rows)
print(len(rows), dict(c))
for r in rows:
    print(r['pre'][:14], r['n_files'], r['bytes']//1024, 'K', r['object'], 'copies=%d' % r['copies'] if r['copies'] else '', 'exact=%d' % r['exact_locations'] if r['exact_locations'] else '', 'census=%d' % len(r['census_artifacts']) if r['census_artifacts'] else '', 'lean=%d/%d' % (len(r['lean_closed']), len(r['lean_files'])) if r['lean_files'] else '')
