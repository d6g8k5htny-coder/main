#!/usr/bin/env python3
"""Mechanical pinning amendments: source paths that adopted quarantine records name in their text but do not bind.

Usage: python3 -I -B pin_sweep.py CATALOG MATH_CLONE MAIN_CLONE OUT

For every adopted record whose text says "not pinned" and names a repository path (frontiers/, reviews/, experiments/, ...) that is
absent from its location and source_binding(s): if the text right before the path negates it ("NOT in", "not in"), it is skipped
and reported. Otherwise the path is pinned at the first commit named in the record's text (7 to 40 hex digits) at which the path
exists in the repository where it resolves, else at the catalog's source cut. Recorded: blob, bytes and sha256 at that commit, and
whether the bytes are identical at the current tip. Only Git facts; no content judgement.
"""
import hashlib, json, re, subprocess, sys
cat_p, math_c, main_c, out = sys.argv[1:5]
CUTS = {'Math-': '8151f592918a845bf19eb8eafaf8444540a32734', 'main': '81acc060341e8a31cc4cd4f9523ef71fb6c06175'}
TIPS = {'Math-': '3b11bff6479f3f40ac838fe1d8b87c7ed6814606', 'main': 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'}
CLONES = {'Math-': math_c, 'main': main_c}
ROOT = r'(?:frontiers|coefficients|certificates|imports|reviews|companions|specifications|incoming|formal|experiments|docs)'
PAT = re.compile(rf'({ROOT}/[A-Za-z0-9_./\-]+?\.(?:md|json|py|lean|tex|txt))')
def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def full(repo, short):
    o = git(CLONES[repo], 'rev-parse', '--verify', '-q', short + '^{commit}')
    return o.decode().strip() if o else None
cat = json.load(open(cat_p))
pins, skipped = [], []
for r in cat['quarantines']:
    s = json.dumps(r, ensure_ascii=False)
    if 'not pinned' not in s:
        continue
    bound = set()
    for l in (r['location'] if isinstance(r['location'], list) else [r['location']]):
        if isinstance(l, str):
            bound.add(l.partition(':')[2].split('#')[0] if '@' in l else l.split('/blob/')[-1].split('/', 1)[-1])
    for b in (r.get('source_bindings') or []) + ([r['source_binding']] if isinstance(r.get('source_binding'), dict) else []):
        if isinstance(b, dict) and b.get('path'):
            bound.add(b['path'])
    hexes = [h for h in re.findall(r'\b([0-9a-f]{7,40})\b', s) if not re.fullmatch(r'\d+', h)]
    for m in PAT.finditer(s):
        path = m.group(1)
        if path in bound or any(p['record'] == r['id'] and p['path'] == path for p in pins):
            continue
        before = s[max(0, m.start() - 12):m.start()]
        if re.search(r'not in\s*`?$', before, re.I):
            skipped.append({'record': r['id'], 'path': path, 'reason': f'negated mention: ...{before}{path}'})
            continue
        chosen = None
        for repo in ('Math-', 'main'):
            for h in hexes:
                c = full(repo, h)
                if c and git(CLONES[repo], 'cat-file', '-e', f'{c}:{path}') is not None:
                    chosen = (repo, c, 'commit named in the record'); break
            if chosen:
                break
            if git(CLONES[repo], 'cat-file', '-e', f'{CUTS[repo]}:{path}') is not None:
                chosen = (repo, CUTS[repo], 'catalog source cut'); break
        if not chosen:
            skipped.append({'record': r['id'], 'path': path, 'reason': 'path not found at any commit named in the record or at the cuts'})
            continue
        repo, commit, why = chosen
        data = git(CLONES[repo], 'cat-file', 'blob', f'{commit}:{path}')
        blob = git(CLONES[repo], 'rev-parse', f'{commit}:{path}').decode().strip()
        tip_blob = git(CLONES[repo], 'rev-parse', f'{TIPS[repo]}:{path}')
        pins.append({'record': r['id'], 'repository': f'd6g8k5htny-coder/{repo}', 'commit': commit, 'commit_basis': why,
                     'path': path, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                     'tip': TIPS[repo], 'identical_at_tip': bool(tip_blob) and tip_blob.decode().strip() == blob})
for i, p in enumerate(pins, 1):
    p['id'] = f'QP-C54-PIN-{i:03d}'
json.dump({'schema': 'claude54.pinning-sweep.v1', 'proposal_status': 'PROPOSED', 'pins': pins, 'skipped': skipped},
          open(out, 'x'), indent=1)
print(len(pins), 'pins across', len({p['record'] for p in pins}), 'records;', len(skipped), 'skipped')
for p in pins: print(p['id'], p['record'], p['repository'].split('/')[1], p['commit'][:8], p['commit_basis'][:12], p['path'], 'tip-identical' if p['identical_at_tip'] else 'CHANGED/ABSENT at tip')
for k in skipped: print('SKIP', k)
