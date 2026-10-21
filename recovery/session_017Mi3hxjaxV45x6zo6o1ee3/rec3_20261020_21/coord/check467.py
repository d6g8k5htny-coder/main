"""Summoner check of the Cursor #467 verdict: key-only migration and occurrence counts (stdlib; reads git objects)."""
import re, subprocess, sys
W = sys.argv[1]
BASE, PARENT, HEAD = '6e8587b94a81185c99447357ae146c247bd8bb22', 'cbaec300a02ceae1fd85d5fd1eb17f125fda339b', 'd19b619c54bdbda8d7c63089cdffd5447a5e59f7'
KEY = re.compile(r'site-release=[0-9a-f]{64}')
def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, check=True).stdout
def show(rev, path):
    r = subprocess.run(['git', '-C', W, 'show', f'{rev}:{path}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None
changed = git('diff', '--name-only', PARENT, HEAD).decode().split()
nonkey, pairs, readme = 0, 0, None
for p in changed:
    a, b = show(PARENT, p), show(HEAD, p)
    if p == 'docs/site/README.md':
        readme = (a, b); continue
    la, lb = a.decode().splitlines(), b.decode().splitlines()
    if len(la) != len(lb):
        print('LINECOUNT DIFFERS', p); nonkey += 1; continue
    for x, y in zip(la, lb):
        if x != y:
            pairs += 1
            if KEY.sub('site-release=K', x) != KEY.sub('site-release=K', y):
                nonkey += 1; print('NONKEY', p, repr(x[:120]), repr(y[:120]))
def occ(rev):
    files = git('ls-tree', '-r', '--name-only', rev).decode().split('\n')
    vals = []
    for f in files:
        if not f or not (f.startswith('docs/site/') or f.startswith('docs/public-math/')): continue
        d = show(rev, f)
        if d is None: continue
        vals += KEY.findall(d.decode('utf-8', 'replace'))
    return vals
vb, vh = occ(BASE), occ(HEAD)
print('changed files (d19b619c vs parent):', len(changed))
print('changed line pairs outside README:', pairs, '| non-key differences:', nonkey)
print('occurrences base/head:', len(vb), len(vh), '| distinct base/head:', len(set(vb)), len(set(vh)))
a, b = readme
import difflib
print('README changed lines:')
for l in difflib.unified_diff(a.decode().splitlines(), b.decode().splitlines(), lineterm='', n=0):
    if l.startswith(('+', '-')) and not l.startswith(('+++', '---')): print('  ', l[:300])
