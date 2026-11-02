#!/usr/bin/env python3
"""Directory-level object census of the non-main, non-Math- federation repositories. Usage: CONFIG_TIP CANDIDATE MAIN MATH OUT"""
import json, re, subprocess, sys, collections
cfg_p, cand_p, main_c, math_c, out = sys.argv[1:6]
cfg = json.load(open(cfg_p)); cand = json.load(open(cand_p))
def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ''
REPOS = {'d6g8k5htny-coder/main': main_c, 'd6g8k5htny-coder/Math-': math_c}
locs = []
for r in cand['quarantines'] + cand['proposed_quarantines'] + cand['proposed_pinning_amendments']:
    for l in (r['location'] if isinstance(r['location'], list) else [r['location']]):
        if isinstance(l, str):
            m = re.match(r'(d6g8k5htny-coder/(?:main|Math-))@([0-9a-f]{40}):([^#]+)', l) or \
                re.match(r'https://github.com/(d6g8k5htny-coder/(?:main|Math-))/blob/([0-9a-f]{40})/([^#]+)', l)
            if m: locs.append((m.group(1), m.group(2), m.group(3).rstrip('/'), r['id']))
for r in cand['main_census_independent_validation']:
    locs.append(('d6g8k5htny-coder/main', r['commit'], r['path'], r['id']))
disposed = {}
for repo, commit, path, rid in sorted(set(locs)):
    for line in git(REPOS[repo], 'ls-tree', '-r', commit, '--', path).splitlines():
        meta, p = line.split('\t', 1)
        disposed.setdefault(meta.split()[2], f'{rid} {repo.split("/")[1]}@{commit[:8]}:{p}')
mathy = re.compile(r'\.(md|tex|py|json|lean|txt)$')
EXCL = ('.github/',)
objs = []
for repo in ('meta-framework', 'google-drive', 'governance-', 'Universal-Law-Workspace', 'query-', 'trial', 'sandbox', 'd6g8k5htny-coder'):
    c = cfg[repo]
    groups = collections.defaultdict(list)
    for line in git(c['path'], 'ls-tree', '-r', '-l', c['ref']).splitlines():
        meta, p = line.split('\t', 1)
        if p.startswith(EXCL) or not mathy.search(p):
            continue
        parts = p.split('/')
        key = '/'.join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) == 2 else '.')
        if repo == 'trial' and parts[0] == 'portable' and len(parts) > 3:
            key = '/'.join(parts[:3])
        groups[key].append({'path': p, 'git_blob': meta.split()[2], 'bytes': int(meta.split()[3]), 'copy_of': disposed.get(meta.split()[2])})
    for k, fs in sorted(groups.items()):
        objs.append({'repository': repo, 'ref': c['ref'], 'clone': c['path'], 'object': k, 'n_files': len(fs),
                     'bytes': sum(f['bytes'] for f in fs), 'copies': sum(bool(f['copy_of']) for f in fs),
                     'all_copies': all(f['copy_of'] for f in fs), 'files': fs})
json.dump({'objects': objs}, open(out, 'x'), indent=1)
for o in objs:
    print(o['repository'], o['object'], o['n_files'], o['bytes'] // 1024, 'K', 'ALLCOPIES' if o['all_copies'] else (f"copies={o['copies']}" if o['copies'] else ''))
