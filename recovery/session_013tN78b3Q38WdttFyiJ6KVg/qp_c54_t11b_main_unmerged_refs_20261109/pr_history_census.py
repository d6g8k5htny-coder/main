#!/usr/bin/env python3
"""Read-only census of every Math- pull-request head and branch tip: the mathematical-type files each changes relative to its merge base with
Math- main whose bytes (Git blob) occur nowhere in main's history (any commit reachable from MAIN_REF). Heads must be local. A ref with no
common ancestor with main is compared with the empty tree and marked orphan.
Usage: python3 -I -B pr_history_census.py MATH_CLONE MAIN_REF OUT_JSON REFS_TXT [REFS_TXT...]   (`git ls-remote` output for
refs/pull/*/head and refs/heads/*)"""
import json, re, subprocess, sys, collections
clone, main_ref, out, *heads_txts = sys.argv[1:]
def git(*a, inp=None):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=True, input=inp).stdout
main_ref = git('rev-parse', main_ref).decode().strip()
objs = git('rev-list', '--objects', main_ref)
ids = b'\n'.join(l.split(b' ', 1)[0] for l in objs.splitlines()) + b'\n'
hist_blobs = set()
for l in git('cat-file', '--batch-check=%(objectname) %(objecttype)', inp=ids).splitlines():
    sha, typ = l.split()
    if typ == b'blob':
        hist_blobs.add(sha.decode())
mathy = re.compile(r'\.(md|tex|py|json|lean|txt|sage|ipynb|m|jl)$')
rows = []
for line in (l for t in heads_txts for l in open(t)):
    sha, ref = line.split()
    n = int(ref.split('/')[2]) if ref.startswith('refs/pull/') else None
    in_main = subprocess.run(['git', '-C', clone, 'merge-base', '--is-ancestor', sha, main_ref]).returncode == 0
    mb = subprocess.run(['git', '-C', clone, 'merge-base', sha, main_ref], capture_output=True)
    orphan = mb.returncode != 0  # no common ancestor: compare with the empty tree
    base = '4b825dc642cb6eb9a060e54bf8d69288fbee4904' if orphan else mb.stdout.decode().strip()
    off = []
    toks = git('diff', '--no-renames', '--no-abbrev', '-z', '--raw', base, sha).split(b'\0')
    for meta, path in zip(toks[0::2], toks[1::2]):  # -z --raw: ":<meta>" NUL "<path>" NUL, one path each with --no-renames
        assert meta.startswith(b':'), meta[:40]
        parts = meta[1:].split()
        status, blob = parts[4].decode(), parts[3].decode()
        path = path.decode('utf-8', 'replace')
        if status.startswith('D') or not mathy.search(path) or blob in hist_blobs:
            continue
        off.append({'path': path, 'git_blob': blob})
    dirs = collections.Counter(p['path'].rsplit('/', 1)[0] if '/' in p['path'] else '(root)' for p in off)
    rows.append({'ref': ref, 'pr': n, 'head': sha, 'head_in_main_history': in_main, 'merge_base': base, 'orphan': orphan, 'files_never_in_main_history': len(off),
                 'lean': sum(1 for p in off if p['path'].endswith('.lean')), 'top_dirs': dirs.most_common(6), 'files': off})
rows.sort(key=lambda r: (r['pr'] is None, r['pr'] or 0, r['ref']))
json.dump({'schema': 'claude54.math-pr-history-census.v1', 'main_ref': main_ref, 'main_history_blobs': len(hist_blobs),
           'prs': rows}, open(out, 'x'), indent=1)
hit = [r for r in rows if r['files_never_in_main_history']]
print('heads', len(rows), 'in main history', sum(r['head_in_main_history'] for r in rows), 'with files never in main history',
      len(hit), 'files', sum(r['files_never_in_main_history'] for r in hit), 'lean', sum(r['lean'] for r in hit))
for r in hit:
    print(f"{('#' + str(r['pr'])) if r['pr'] else r['ref'][11:]} {r['head'][:8]} files {r['files_never_in_main_history']} lean {r['lean']} {r['top_dirs'][:2]}")
