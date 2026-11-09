#!/usr/bin/env python3
"""For each tranche-10 ref, the engineering disposition that Claude01N's Math- cbz/evidence/math-pr-dispositions (at MAIN_REF)
recommends: open PRs from dispositions.json, PR-less branches from unmerged_branches.json, with whether the head observed then
equals the head bound now. Read-only. Usage: python3 -I -B claude01n_crossref.py MATH_CLONE MAIN_REF OBJECTS_JSON OUT_JSON"""
import collections, json, subprocess, sys
clone, ref, objs, out = sys.argv[1:]
def show(p):
    return json.loads(subprocess.run(['git', '-C', clone, 'show', f'{ref}:cbz/evidence/math-pr-dispositions/{p}'],
                                     capture_output=True, check=True).stdout)
d, u = show('dispositions.json'), show('unmerged_branches.json')
prd = {r['number']: r for r in d['pull_requests']}
brd = {r['branch']: r for r in u['unmerged_branches']}
b = json.load(open(objs))
rows, c = [], collections.Counter()
for r_, pr, head in sorted({(o['ref'], o['pr'], o['commit']) for o in b['math'] + b['eng']}):
    if pr is not None:
        x = prd.get(pr)
        row = {'ref': r_, 'pr': pr, 'head': head, 'claude01n': (x['recommended_disposition'] if x else
               'NOT_OBSERVED (not among the open PRs at the observation)'), 'head_then': x['head_sha'] if x else None}
    else:
        name = r_[len('refs/heads/'):]
        x = brd.get(name)
        row = {'ref': r_, 'branch': name, 'head': head, 'claude01n': (x['recommended_disposition'] if x else
               'NOT_OBSERVED (not among the unmerged PR-less branches at the observation)'), 'head_then': None}
    row['same_head'] = row['head_then'] == head if row['head_then'] else None
    rows.append(row); c[row['claude01n'].split(' ')[0]] += 1
res = {'schema': 'claude54.t10-claude01n-crossref.v1', 'source': f'Math-@{ref}:cbz/evidence/math-pr-dispositions',
       'observed_at_utc': d['meta'].get('observed_at_utc'), 'counts': dict(c), 'refs': rows}
with open(out, 'x') as f:
    json.dump(res, f, indent=1)
print(dict(c))
