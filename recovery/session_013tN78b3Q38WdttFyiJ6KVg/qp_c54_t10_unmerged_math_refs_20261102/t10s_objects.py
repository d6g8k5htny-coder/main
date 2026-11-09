#!/usr/bin/env python3
"""Tranche-10 supplement objects (9 November 2026): the blobs tranche 10 set aside because a deferred ref carried them. Those refs
changed state after the 2 November cut: #482 and #487 (and their closed duplicates) and #556/#557 were closed dormant; #541 and
#554 merged. Same census and rules as t10_objects.py, with two differences: only blobs carried by a ref of DEFERRED_PRS are kept
(nothing is deferred now), and a blob is also dropped when it occurs in Math- main's history at NEW_MAIN (it landed after the cut).
Objects get the same "<directory>@<ref>" ids; batches are numbered from 23, after tranche 10's 22.

Usage: python3 -I -B t10s_objects.py MATH_CLONE CENSUS CLONES_JSON MERGE_REFS_TXT CATALOG T9_FACTS_DIR OUT_DIR NEW_MAIN"""
import collections, glob, json, os, re, subprocess, sys

clone, census_p, clones_p, merge_p, cat_p, t9_dir, out_dir, new_main = sys.argv[1:]
MAIN = 'c88c34b2e912df8b4885b04a19293685f0cb6c00'
TIPS = {'Math-': MAIN, 'main': 'd2b0216e15d545c47f25b6a9265dd3298971e9fe', 'trial': '3738c859'}
DEFERRED_PRS = {541, 554, 556, 557, 482, 487}
MAX = 60
MATH_AREAS = {'frontiers', 'reviews', 'companions', 'certificates', 'coefficients', 'repairs', 'imports', 'formalization',
              'specifications', 'closed-lemmas', 'open-lemmas', 'closed-theorems', 'claims', 'audits', 'formal', 'archive'}


def git(*a, check=True):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=check)


census = json.load(open(census_p))
rows = census['prs']
heads_of = {}
for r in rows:
    heads_of.setdefault(r['head'], set()).add(r['ref'])
deferred_heads = {r['head'] for r in rows if r['pr'] in DEFERRED_PRS}
deferred_refs = {ref for h in deferred_heads for ref in heads_of[h]}
open_prs = {int(m) for m in re.findall(r'refs/pull/(\d+)/merge', open(merge_p).read())}
# blobs already accounted
tip_blobs = set()
for repo, path in json.load(open(clones_p)).items():
    r = subprocess.run(['git', '-C', path, 'ls-tree', '-r', '-z', TIPS.get(repo, 'HEAD')], capture_output=True)
    for rec in r.stdout.split(b'\0'):
        if rec:
            tip_blobs.add(rec.split(b'\t', 1)[0].split()[2].decode())
cat_text = open(cat_p).read()
cat = json.loads(cat_text)
cat_blobs = set(re.findall(r'\b[0-9a-f]{40}\b', cat_text))
pr_rows = set()
for q in cat['quarantines']:
    if q.get('repository') == 'Math-' and q.get('pr'):
        for loc in (q['location'] if isinstance(q['location'], list) else [q['location']]):
            sha, p = loc.split('@', 1)[1].split(':', 1)
            for line in git('ls-tree', '-r', sha, '--', p).stdout.decode().splitlines():
                pr_rows.add(line.split()[2])
t9 = set()
for fp in glob.glob(os.path.join(t9_dir, 'pbatch*.json')):
    for o in json.load(open(fp))['objects']:
        for f in o['files'] + o.get('copies', []):
            t9.add(git('rev-parse', f'{o["commit"]}:{f["path"]}').stdout.decode().strip())
occ = collections.defaultdict(list)  # blob -> [(ref, pr, head, path)]
for r in rows:
    for f in r['files']:
        occ[f['git_blob']].append((r['ref'], r['pr'], r['head'], f['path']))
new_hist = set()
_objs = git('rev-list', '--objects', new_main).stdout
_ids = b'\n'.join(l.split(b' ', 1)[0] for l in _objs.splitlines()) + b'\n'
for _l in subprocess.run(['git', '-C', clone, 'cat-file', '--batch-check=%(objectname) %(objecttype)'], input=_ids,
                         capture_output=True, check=True).stdout.splitlines():
    _sha, _typ = _l.split()
    if _typ == b'blob':
        new_hist.add(_sha.decode())
drop = collections.Counter()
keep = {}
for blob, os_ in occ.items():
    if blob in tip_blobs:
        drop['at a federation tip'] += 1
    elif blob in pr_rows:
        drop['under a PR-located adopted row'] += 1
    elif blob in cat_blobs:
        drop['named in the catalog'] += 1
    elif blob in t9:
        drop['bound by tranche 9'] += 1
    elif not any(ref in deferred_refs for ref, _, _, _ in os_):
        drop['not deferred (tranche 10 proper)'] += 1
    elif blob in new_hist:
        drop['in Math- main history at NEW_MAIN'] += 1
    else:
        keep[blob] = os_


SNAP = re.compile(r'(^|/)(evidence|frozen_sources|source-snapshot)(/|$)')


def rank(o):
    ref, pr, head, path = o
    within = (bool(SNAP.search(path)), len(path), path)
    if pr is not None and pr in open_prs:
        return (0, -pr, '', within)
    if pr is not None:
        return (1, -pr, '', within)
    return (2, 0, ''.join(chr(0x10FFFF - ord(c)) for c in ref), within)


rep = {b: min(os_, key=rank) for b, os_ in keep.items()}
# supersession: rep ref is an ancestor of another in-scope ref carrying the same path with other bytes
by_path = collections.defaultdict(set)  # path -> {(head, blob)}
for r in rows:
    pass
    for f in r['files']:
        by_path[f['path']].add((r['head'], f['git_blob']))
anc_cache = {}


def is_anc(a, b):
    if (a, b) not in anc_cache:
        anc_cache[(a, b)] = git('merge-base', '--is-ancestor', a, b, check=False).returncode == 0
    return anc_cache[(a, b)]


superseded = []
live = {}
for b, (ref, pr, head, path) in rep.items():
    later = sorted({h for h, bb in by_path[path] if bb != b and h != head and is_anc(head, h)})
    if later:
        superseded.append({'git_blob': b, 'ref': ref, 'head': head, 'path': path, 'superseded_at': later})
    else:
        live[b] = (ref, pr, head, path)


def packet(path, depth_extra=0):
    parts = path.split('/')
    if len(parts) == 1:
        return '(root)'
    if parts[0] == 'cbz' and len(parts) > 2 and parts[1].startswith('CBZ-MATH-'):
        d = 4
    elif parts[0] in MATH_AREAS or parts[0] in ('cbz', 'evidence'):
        d = 2
    else:
        d = 1
    d = min(d + depth_extra, len(parts) - 1)
    return '/'.join(parts[:d])


groups = collections.defaultdict(list)
for b, (ref, pr, head, path) in live.items():
    groups[(ref, packet(path))].append((path, b))
objs = []
for (ref, pk), fs in sorted(groups.items()):
    stack = [(pk, fs, 0)]
    while stack:
        name, fl, extra = stack.pop()
        if len(fl) > MAX and extra < 4:
            sub = collections.defaultdict(list)
            for p, b in fl:
                sub[packet(p, extra + 1) if p.startswith(name + '/') or name == '(root)' else name].append((p, b))
            if len(sub) > 1:
                stack += [(n, x, extra + 1) for n, x in sub.items()]
                continue
        objs.append((ref, name, sorted(fl)))
ref_info = {r['ref']: r for r in rows}
batches = {'math': [], 'eng': []}
for ref, name, fl in sorted(objs):
    r = ref_info[ref]
    top = name.split('/')[0]
    kind = 'math' if (top in MATH_AREAS or name.startswith('cbz/CBZ-MATH-198-20261029/project/formal')) else 'eng'
    batches[kind].append({'object': f'{name}@{ref}', 'directory': name, 'ref': ref, 'pr': r['pr'],
                          'pr_open': r['pr'] in open_prs if r['pr'] else None, 'repository': 'd6g8k5htny-coder/Math-',
                          'commit': r['head'],
                          'files': [{'path': p, 'git_blob': b, 'also_at': sorted({o[0] for o in keep[b]} - {ref}),
                                     'same_bytes_at': sorted({o[3] for o in keep[b] if o[0] == ref} - {p})} for p, b in fl]})
os.makedirs(out_dir, exist_ok=True)
summary = {'schema': 'claude54.t10-supplement-objects.v1', 'main_ref': MAIN, 'new_main': new_main, 'deferred_prs': sorted(DEFERRED_PRS),
           'deferred_refs': sorted(deferred_refs), 'blobs_never_in_main_history': len(occ), 'dropped': dict(drop),
           'kept_blobs': len(keep), 'superseded': superseded, 'live_blobs': len(live),
           'objects': {k: len(v) for k, v in batches.items()}}
with open(os.path.join(out_dir, 'summary.json'), 'x') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(out_dir, 'objects.json'), 'x') as f:
    json.dump(batches, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k not in ('superseded', 'deferred_refs')}, indent=1))
print('superseded blobs', len(superseded))
for k, v in batches.items():
    print(k, len(v), 'objects', sum(len(o['files']) for o in v), 'files', sum(1 for o in v for f in o['files'] if f['path'].endswith('.lean')), 'lean')
    print('  by ref kind:', collections.Counter('open PR' if o['pr_open'] else ('closed PR' if o['pr'] else 'branch') for o in v))
