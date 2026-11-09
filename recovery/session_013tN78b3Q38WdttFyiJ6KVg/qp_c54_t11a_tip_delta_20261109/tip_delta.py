#!/usr/bin/env python3
"""Tip-delta census: mathematical-type files that a repository's default branch added or changed between an old cut and a new tip,
outside engineering directories, whose blob no QP-C54 tranche record, catalog row or CLOSED index entry already binds (by git blob
id or SHA-256 named in those files). Read-only. Usage: python3 -I -B tip_delta.py CLONE OLD NEW BOUND_DIR [BOUND_DIR...] OUT_JSON"""
import collections, hashlib, json, os, re, subprocess, sys
clone, old, new, *bdirs, out = sys.argv[1:]
ENG = ('cbz/', 'tests/', 'tools/', '.github/', '.claude/', 'docs/site/', 'docs/retro/', 'docs/superpowers/', 'governance/', 'ctx/',
       'history/', 'interface/', 'records/', 'architecture/', 'data/', 'formal/tests/', 'recovery/', 'incoming/')
MATHY = re.compile(r'\.(md|tex|py|json|lean|txt|sage|ipynb|m|jl|csv)$')
bound = set()
for d in bdirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.json'):
                bound.update(re.findall(r'\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b', open(os.path.join(root, f), errors='replace').read()))
def git(*a):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=True).stdout
toks = git('diff', '--no-renames', '--no-abbrev', '-z', '--raw', old, new).split(b'\0')
rows = []
for meta, path in zip(toks[0::2], toks[1::2]):
    parts = meta[1:].split(); st, blob = parts[4].decode(), parts[3].decode(); p = path.decode('utf-8', 'replace')
    if st.startswith('D') or not MATHY.search(p) or p.startswith(ENG) or '/' not in p:
        continue
    sha = hashlib.sha256(git('cat-file', 'blob', blob)).hexdigest()
    rows.append({'path': p, 'status': st, 'git_blob': blob, 'sha256': sha, 'bound': blob in bound or sha in bound})
un = [r for r in rows if not r['bound']]
dirs = collections.Counter(r['path'].rsplit('/', 1)[0] for r in un)
json.dump({'schema': 'claude54.tip-delta.v1', 'old': old, 'new': new, 'rows': rows}, open(out, 'x'), indent=1)
print(f'{len(rows)} changed mathematical-type files outside engineering dirs; bound {len(rows) - len(un)}; unbound {len(un)} '
      f'(lean {sum(r["path"].endswith(".lean") for r in un)}) in {len(dirs)} dirs')
for d, n in dirs.most_common(25):
    print(f'  {n:4d} {d}')
