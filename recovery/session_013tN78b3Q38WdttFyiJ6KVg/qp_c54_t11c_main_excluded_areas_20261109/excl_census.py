#!/usr/bin/env python3
"""T11c scoping, read-only. For main unmerged-ref census entries under recovery/, drive/ and incoming/ (math extensions), find
whether each blob's bytes occur anywhere else: main history at NEW_MAIN, any ref of any federation clone (Math- and the others,
--all), any non-excluded path in the main census, or a 40-hex id in BOUND_DIR JSON files / the catalog.
Usage: python3 -I -B excl_census.py CENSUS CLONES_JSON NEW_MAIN CATALOG BOUND_DIR OUT"""
import collections, json, os, re, subprocess, sys
census_p, clones_p, new_main, cat_p, bound_dir, out = sys.argv[1:]
AREAS = ('recovery/', 'drive/', 'incoming/')
EXCL_OTHER = ('cbz/', 'tests/', 'tools/', '.github/', 'docs/site/', 'docs/retro/', 'governance/', 'ctx/', 'history/', 'interface/',
              'records/', 'architecture/', 'data/')
OWN = {'refs/heads/claude/help-other-claude-models-829990', 'refs/pull/441/head'}
clones = json.load(open(clones_p))
def objects(path, *revs):
    r = subprocess.run(['git', '-C', path, 'rev-list', '--objects', *revs], capture_output=True)
    return {l.split(b' ', 1)[0].decode() for l in r.stdout.splitlines()}
where = {}
where['main@' + new_main[:8]] = objects(clones['main'], new_main)
for repo, path in clones.items():
    if repo == 'main':
        continue
    where[repo + ' --all'] = objects(path, '--all')
named = set()
for root, dirs, files in os.walk(bound_dir):
    for f in files:
        if f.endswith('.json'):
            named.update(re.findall(r'\b[0-9a-f]{40}\b', open(os.path.join(root, f), errors='replace').read()))
named.update(re.findall(r'\b[0-9a-f]{40}\b', open(cat_p).read()))
census = json.load(open(census_p))
occ = collections.defaultdict(list)
other_paths = collections.defaultdict(set)
for r in census['prs']:
    for f in r['files']:
        p = f['path']
        if p.startswith(AREAS):
            if r['ref'] not in OWN:
                occ[f['git_blob']].append((r['ref'], r['pr'], r['head'], p, r.get('orphan', False)))
        elif r['ref'] not in OWN:
            other_paths[f['git_blob']].add(p)
rows = {}
for b, os_ in occ.items():
    hits = [k for k, s in where.items() if b in s]
    rows[b] = {'areas': sorted({o[3].split('/')[0] for o in os_}), 'refs': sorted({o[0] for o in os_}),
               'paths': sorted({o[3] for o in os_})[:5], 'n_paths': len({o[3] for o in os_}),
               'orphan_only': all(o[4] for o in os_), 'elsewhere_git': hits,
               'other_census_paths': sorted(other_paths.get(b, ()))[:3], 'named': b in named}
summ = collections.Counter()
for b, r in rows.items():
    a = '+'.join(r['areas'])
    k = 'elsewhere in git' if r['elsewhere_git'] else ('at a non-excluded census path' if r['other_census_paths'] else
                                                      ('named by a record' if r['named'] else 'ONLY HERE'))
    summ[(a, k)] += 1
json.dump({'schema': 'claude54.t11c-excluded-census.v1', 'new_main': new_main, 'sources': {k: len(v) for k, v in where.items()},
           'summary': {f'{a} | {k}': n for (a, k), n in sorted(summ.items())}, 'blobs': rows}, open(out, 'x'), indent=1)
for (a, k), n in sorted(summ.items()):
    print(f'{a:28s} {k:32s} {n}')
