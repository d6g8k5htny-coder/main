#!/usr/bin/env python3
"""Tranche-11b objects: mathematical-type files on main's pull-request heads and branch tips whose bytes occur in no commit of
main's history and that no record binds. Read-only. The rules follow tranche 10 (t10_objects.py) with main's census rules from
tranche 4 (main_dirs.py).

Kept: a file outside root and outside the engineering directories (EXCL), with a mathematical extension. Dropped, per blob:
  - at a censused federation tip (any repository of CLONES_JSON, main at MAIN);
  - under a location of a PR-located catalog row (repository main or Math-, a "pr" field), at that row's commit;
  - named (40 hex digits) in the catalog or in any JSON file under BOUND_DIR (the QP-C54 retention, skipping this tranche's
    own qp_c54_t11b* directory);
  - carried only by OWN refs (this session's retention branch and its closed retention PR).
Representative occurrence: open PR head (by merge ref at the cut), then closed PR head (highest number), then branch tip; within a
ref, a path outside evidence snapshots, then the shortest. A blob whose representative ref is an ancestor of another in-scope ref
carrying the same path with different bytes is superseded (listed, not classified). Objects: (representative ref, packet
directory), split one level further above MAX files. Batches: "math" for MATH_AREAS (draft, fidelity, Layer A), else "eng"
(draft, screen).

Usage: python3 -I -B t11b_objects.py MAIN_CLONE CENSUS CLONES_JSON MERGE_REFS_TXT CATALOG BOUND_DIR OUT_DIR"""
import collections, json, os, re, subprocess, sys

clone, census_p, clones_p, merge_p, cat_p, bound_dir, out_dir = sys.argv[1:]
MAIN = 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'
TIPS = {'Math-': 'c88c34b2e912df8b4885b04a19293685f0cb6c00', 'main': MAIN, 'trial': '3738c859'}
EXCL = ('cbz/', 'tests/', 'tools/', '.github/', 'docs/site/', 'docs/retro/', 'governance/', 'ctx/', 'history/', 'interface/',
        'records/', 'architecture/', 'data/', 'recovery/', 'drive/', 'incoming/')
MATHY = re.compile(r'\.(md|tex|py|json|lean|txt)$')
OWN = {'refs/heads/claude/help-other-claude-models-829990', 'refs/pull/441/head'}
MATH_AREAS = {'research', 'experiments', 'reviews', 'formal', 'claims', 'audits'}
DEPTH2 = MATH_AREAS | {'docs', 'custody', 'registers', 'engine', 'evidence', 'sandbox', 'attestations', 'quarantine',
                       'v19_dependency_supplement', 'v20_planner_source_carrier', 'coordination', 'results', 'src'}
MAX = 60
SNAP = re.compile(r'(^|/)(evidence|frozen_sources|source-snapshot)(/|$)')


def git(*a, check=True, inp=None):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=check, input=inp)


census = json.load(open(census_p))
rows = [r for r in census['prs']]
open_prs = {int(m) for m in re.findall(r'refs/pull/(\d+)/merge', open(merge_p).read())}
tip_blobs = set()
for repo, path in json.load(open(clones_p)).items():
    r = subprocess.run(['git', '-C', path, 'ls-tree', '-r', '-z', TIPS.get(repo, 'HEAD')], capture_output=True)
    for rec in r.stdout.split(b'\0'):
        if rec:
            tip_blobs.add(rec.split(b'\t', 1)[0].split()[2].decode())
cat_text = open(cat_p).read()
cat = json.loads(cat_text)
named = set(re.findall(r'\b[0-9a-f]{40}\b', cat_text))
for root, dirs, files in os.walk(bound_dir):
    dirs[:] = [x for x in dirs if not x.startswith('qp_c54_t11b')]  # never read this tranche's own retained output
    for f in files:
        if f.endswith('.json'):
            named.update(re.findall(r'\b[0-9a-f]{40}\b', open(os.path.join(root, f), errors='replace').read()))
pr_rows = set()
clones = json.load(open(clones_p))
for q in cat['quarantines']:
    if q.get('pr') and q.get('repository') in ('main', 'Math-'):
        cl = clones['main' if q['repository'] == 'main' else 'Math-']
        for loc in (q['location'] if isinstance(q['location'], list) else [q['location']]):
            if '@' not in loc or ':' not in loc.split('@', 1)[1]:
                continue
            sha, p = loc.split('@', 1)[1].split(':', 1)
            out = subprocess.run(['git', '-C', cl, 'ls-tree', '-r', sha, '--', p], capture_output=True, text=True).stdout
            pr_rows.update(line.split()[2] for line in out.splitlines())
occ = collections.defaultdict(list)
for r in rows:
    for f in r['files']:
        p = f['path']
        if p.startswith(EXCL) or '/' not in p or not MATHY.search(p):
            continue
        occ[f['git_blob']].append((r['ref'], r['pr'], r['head'], p))
drop, keep = collections.Counter(), {}
for blob, os_ in occ.items():
    if blob in tip_blobs:
        drop['at a federation tip'] += 1
    elif blob in pr_rows:
        drop['under a PR-located catalog row'] += 1
    elif blob in named:
        drop['named in the catalog or a QP-C54 file'] += 1
    elif all(o[0] in OWN for o in os_):
        drop['only on own retention refs'] += 1
    else:
        keep[blob] = [o for o in os_ if o[0] not in OWN]


def rank(o):
    ref, pr, head, path = o
    within = (bool(SNAP.search(path)), len(path), path)
    if pr is not None and pr in open_prs:
        return (0, -pr, '', within)
    if pr is not None:
        return (1, -pr, '', within)
    return (2, 0, ''.join(chr(0x10FFFF - ord(c)) for c in ref), within)


rep = {b: min(os_, key=rank) for b, os_ in keep.items()}
by_path = collections.defaultdict(set)
for r in rows:
    if r['ref'] in OWN:
        continue
    for f in r['files']:
        by_path[f['path']].add((r['head'], f['git_blob']))
anc = {}


def is_anc(a, b):
    if (a, b) not in anc:
        anc[(a, b)] = git('merge-base', '--is-ancestor', a, b, check=False).returncode == 0
    return anc[(a, b)]


superseded, live = [], {}
for b, (ref, pr, head, path) in rep.items():
    later = sorted({h for h, bb in by_path[path] if bb != b and h != head and is_anc(head, h)})
    if later:
        superseded.append({'git_blob': b, 'ref': ref, 'head': head, 'path': path, 'superseded_at': later})
    else:
        live[b] = (ref, pr, head, path)


def packet(path, extra=0):
    parts = path.split('/')
    d = 2 if parts[0] in DEPTH2 else 1
    return '/'.join(parts[:min(d + extra, len(parts) - 1)])


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
                sub[packet(p, extra + 1)].append((p, b))
            if len(sub) > 1:
                stack += [(n, x, extra + 1) for n, x in sub.items()]
                continue
        objs.append((ref, name, sorted(fl)))
info = {r['ref']: r for r in rows}
batches = {'math': [], 'eng': []}
for ref, name, fl in sorted(objs):
    r = info[ref]
    kind = 'math' if name.split('/')[0] in MATH_AREAS else 'eng'
    batches[kind].append({'object': f'{name}@{ref}', 'directory': name, 'ref': ref, 'pr': r['pr'],
                          'pr_open': (r['pr'] in open_prs) if r['pr'] else None, 'orphan': r.get('orphan', False),
                          'repository': 'd6g8k5htny-coder/main', 'commit': r['head'],
                          'files': [{'path': p, 'git_blob': b, 'also_at': sorted({o[0] for o in keep[b]} - {ref}),
                                     'same_bytes_at': sorted({o[3] for o in keep[b] if o[0] == ref} - {p})} for p, b in fl]})
os.makedirs(out_dir, exist_ok=True)
summary = {'schema': 'claude54.t11b-objects.v1', 'main_ref': MAIN, 'excluded_dirs': list(EXCL), 'own_refs': sorted(OWN),
           'math_area_blobs': len(occ), 'dropped': dict(drop), 'kept_blobs': len(keep), 'superseded': superseded,
           'live_blobs': len(live), 'objects': {k: len(v) for k, v in batches.items()}}
with open(os.path.join(out_dir, 'summary.json'), 'x') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(out_dir, 'objects.json'), 'x') as f:
    json.dump(batches, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k not in ('superseded', 'excluded_dirs')}, indent=1))
print('superseded', len(superseded))
for k, v in batches.items():
    print(k, len(v), 'objects', sum(len(o['files']) for o in v), 'files', sum(f['path'].endswith('.lean') for o in v for f in o['files']),
          'lean; refs', len({o['ref'] for o in v}), '; orphan-ref objects', sum(o['orphan'] for o in v))
