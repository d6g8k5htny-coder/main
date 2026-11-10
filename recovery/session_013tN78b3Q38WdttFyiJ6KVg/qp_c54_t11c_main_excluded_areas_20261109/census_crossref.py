#!/usr/bin/env python3
"""T11c cross-reference, read-only: which artifacts of Claude S's comment-math census (main 683fdfa2,
cbz/evidence/comment-math-census-20261031/CENSUS.json) and implementation sourcing (main#748 6af4b9ad SOURCING.json) cite a
T11c residual blob, in which role (custody carrier or implementation carrier), and, once the deep pass is assembled, which T11c
record or item holds that blob.
Usage: python3 -I -B census_crossref.py CENSUS SOURCING EXCL_CENSUS NORM_DEDUP OUT [RECORDS_JSON]"""
import collections, json, re, sys
cen_p, src_p, ex_p, nd_p, out, *rec = sys.argv[1:]
nd = json.load(open(nd_p))['rows']; ex = json.load(open(ex_p))['blobs']
res = {b for b, r in nd.items() if r['kind'] == 'residual'}
held = {}
if rec:
    d = json.load(open(rec[0]))
    for r in d['records']:
        for f in r['source_binding']['files']:
            held[f['git_blob']] = r['id']
    for o in d['other_items']:
        for f in o['source_binding']['files']:
            held.setdefault(f['git_blob'], f'{o["object"]}#{o["item_id"]} ({o["classification"]})')
rows = []
cen = json.load(open(cen_p))
for a in cen['artifacts']:
    roles = collections.defaultdict(set)
    for part in ('custody', 'implementation'):
        for c in (a.get(part) or {}).get('carriers', []) or []:
            if c.get('blob') in res:
                roles[c['blob']].add(part)
    for b in re.findall(r'\b[0-9a-f]{40}\b', json.dumps(a)):
        if b in res and b not in roles:
            roles[b].add('mentioned')
    if roles:
        rows.append({'census_key': a['key'], 'family': a.get('family'), 'custody_class': (a.get('custody') or {}).get('class'),
                     'implementation_class': (a.get('implementation') or {}).get('class'),
                     'blobs': [{'git_blob': b, 'roles': sorted(r), 'path': ex[b]['paths'][0], 'refs': ex[b]['refs'][:3],
                                't11c': held.get(b)} for b, r in sorted(roles.items())]})
src_rows = []
src_text = open(src_p).read()
for b in sorted(res):
    if b in src_text:
        src_rows.append({'git_blob': b, 'path': ex[b]['paths'][0], 't11c': held.get(b)})
json.dump({'schema': 'claude54.t11c-census-crossref.v1', 'census': '683fdfa2:cbz/evidence/comment-math-census-20261031/CENSUS.json',
           'sourcing': '6af4b9ad:cbz/evidence/comment-math-impl-sourcing-20261102/SOURCING.json',
           'census_artifacts': rows, 'sourcing_blobs': src_rows}, open(out, 'x'), indent=1, ensure_ascii=False)
impl = [r for r in rows if any('implementation' in b['roles'] for b in r['blobs'])]
print(len(rows), 'census artifacts cite residual blobs;', len(impl), 'as implementation carriers;', len(src_rows), 'sourcing blobs')
print(collections.Counter(r['family'] for r in impl))
