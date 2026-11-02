#!/usr/bin/env python3
"""Mechanical checks behind the tranche-9 objects. Read-only; PR heads must be fetched as refs/remotes/pr/N.

1. The inventory as run (pr_inventory.py) lists abbreviated blob ids, because `git diff --raw` abbreviates them, so it never found
   a file on main. pr_inventory_v2.py uses `--no-abbrev`: its ids are all 40 characters, and the as-run list of each PR equals
   the corrected list plus the files whose bytes are on Math- main (same paths, same blob prefixes).
2. Ancestry of every pair of adjacent PRs in each stack, and of the closed compact_source_cutoff predecessors under #430.
3. #426's files are byte-identical at the same paths in #430.
4. For each stack, the paths carried by more than one PR and those whose bytes differ, split by whether the earlier PR is an
   ancestor of the PR whose bytes the objects bind (superseded history) or not (separate work, accounted in pbatch4.json).

Usage: python3 -I -B stack_checks.py MATH_CLONE INVENTORY_AS_RUN INVENTORY_V2 FACTS_DIR OUTPUT_JSON"""
import glob, json, os, re, subprocess, sys
clone, inv1, inv2, facts_dir, out = sys.argv[1:]


def anc(a, b):
    return subprocess.run(['git', '-C', clone, 'merge-base', '--is-ancestor', f'refs/remotes/pr/{a}', f'refs/remotes/pr/{b}']
                          ).returncode == 0


a = {r['pr']: r for r in json.load(open(inv1))['prs']}
MAIN = json.load(open(inv2))['main_ref']
main_blobs = set()
for rec in subprocess.run(['git', '-C', clone, 'ls-tree', '-r', '-z', MAIN], capture_output=True, check=True).stdout.split(b'\0'):
    if rec:
        main_blobs.add(rec.split(b'\t', 1)[0].split()[2].decode())


def git_diff_full(n):
    head = subprocess.run(['git', '-C', clone, 'rev-parse', f'refs/remotes/pr/{n}'], capture_output=True, text=True).stdout.strip()
    base = subprocess.run(['git', '-C', clone, 'merge-base', head, MAIN], capture_output=True, text=True).stdout.strip()
    raw = subprocess.run(['git', '-C', clone, 'diff', '--no-renames', '--no-abbrev', '-z', '--raw', base, head], capture_output=True,
                         check=True).stdout
    rows = []
    for rec in raw.split(b':')[1:]:
        meta, _, path = rec.partition(b'\0')
        parts = meta.split()
        path = path.rstrip(b'\0').decode('utf-8', 'replace')
        if parts[4].decode().startswith('D') or not re.search(r'\.(md|tex|py|json|lean|txt|sage|ipynb|m|jl)$', path):
            continue
        rows.append({'path': path, 'git_blob': parts[3].decode(), 'on_main_tip': parts[3].decode() in main_blobs})
    return rows

b = {r['pr']: r for r in json.load(open(inv2))['prs']}
res = {'schema': 'claude54.t9-stack-checks.v1'}
lengths = {len(f['git_blob']) for r in b.values() for f in r['files_not_on_main']}
agree = {}
for n in a:
    on_main = [f for f in git_diff_full(n) if f['on_main_tip']]
    full = sorted((f['path'], f['git_blob'][:8]) for f in b[n]['files_not_on_main'] + on_main)
    agree[n] = full == sorted((f['path'], f['git_blob'][:8]) for f in a[n]['files_not_on_main'])
res['corrected_blob_id_lengths'] = sorted(lengths)
res['as_run_equals_corrected_plus_on_main'] = all(agree.values())
res['files_on_main_tip'] = {f'#{n}': b[n]['already_on_main_tip'] for n in sorted(b)}
res['files_not_on_main_tip'] = {f'#{n}': b[n]['not_on_main_tip'] for n in sorted(b)}
pairs = [(426, 430), (414, 430), (415, 430), (424, 430), (425, 430), (427, 430), (491, 496), (496, 497), (497, 504),
         (531, 532), (532, 536), (531, 536)]
res['ancestry'] = {f'#{x} ancestor of #{y}': anc(x, y) for x, y in pairs
                   if subprocess.run(['git', '-C', clone, 'rev-parse', '-q', '--verify', f'refs/remotes/pr/{x}'],
                                     capture_output=True).returncode == 0}
b430 = {f['path']: f['git_blob'] for f in b[430]['files_not_on_main']}
res['pr426_files'] = len(b[426]['files_not_on_main'])
res['pr426_files_not_identical_in_pr430'] = [f['path'] for f in b[426]['files_not_on_main'] if b430.get(f['path']) != f['git_blob']]
bound = {}
for fp in sorted(glob.glob(os.path.join(facts_dir, 'pbatch[123].json'))):
    for o in json.load(open(fp))['objects']:
        for f in o['files']:
            bound[f['path']] = o['pr']
res['stacks'] = {}
for name, order in [('critical_ridge', [491, 496, 497, 504]), ('determinant', [531, 532, 536])]:
    vers = {}
    for n in order:
        for f in b[n]['files_not_on_main']:
            vers.setdefault(f['path'], {})[n] = f['git_blob']
    diff = {p: v for p, v in vers.items() if len(set(v.values())) > 1}
    superseded, separate = [], []
    for p, v in sorted(diff.items()):
        top = bound[p]
        for n, blob in v.items():
            if n != top and blob != v[top]:
                (superseded if anc(n, top) else separate).append({'path': p, 'pr': n, 'bound_at': top})
    res['stacks'][name] = {'paths': len(vers), 'carried_by_more_than_one_pr': sum(1 for v in vers.values() if len(v) > 1),
                           'with_differing_bytes': len(diff), 'superseded_versions_in_ancestor_prs': superseded,
                           'separate_versions_in_non_ancestor_prs': separate}
supp = json.load(open(os.path.join(facts_dir, 'pbatch4.json')))
supp_paths = sorted(f['path'] for o in supp['objects'] for f in o['files'])
sep = sorted(x['path'] for s in res['stacks'].values() for x in s['separate_versions_in_non_ancestor_prs'])
res['pbatch4_equals_separate_versions'] = supp_paths == sep
with open(out, 'x') as f:
    json.dump(res, f, indent=1)
    f.write('\n')
print(json.dumps({k: v for k, v in res.items() if k != 'stacks'}, indent=1))
for k, s in res['stacks'].items():
    print(k, s['paths'], s['carried_by_more_than_one_pr'], s['with_differing_bytes'], 'superseded',
          len(s['superseded_versions_in_ancestor_prs']), 'separate', len(s['separate_versions_in_non_ancestor_prs']))
