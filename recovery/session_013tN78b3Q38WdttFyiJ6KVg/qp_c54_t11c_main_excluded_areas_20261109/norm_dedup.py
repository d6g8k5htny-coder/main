#!/usr/bin/env python3
"""T11c scoping, read-only. For the 'only here' blobs of excl_census.json, compute a normalized-text SHA-256 (BOM stripped, CRLF/CR
to LF, trailing whitespace per line stripped, trailing blank lines dropped) and compare with the normalized hashes of every blob
reachable from main NEW_MAIN, every ref of every federation clone, and the main census at non-excluded paths. Also supersession:
an only-here blob whose (path) also carries other bytes at a descendant ref head (ancestry in the main clone).
Usage: python3 -I -B norm_dedup.py EXCL_CENSUS CENSUS CLONES_JSON NEW_MAIN OUT"""
import collections, hashlib, json, re, subprocess, sys
ex_p, census_p, clones_p, new_main, out = sys.argv[1:]
clones = json.load(open(clones_p))
ex = json.load(open(ex_p))['blobs']
only = {b: r for b, r in ex.items() if not r['elsewhere_git'] and not r['other_census_paths'] and not r['named']}
def norm(data):
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    t = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    lines = [l.rstrip() for l in t.split(b'\n')]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256(b'\n'.join(lines)).hexdigest()
def blob_hashes(path, ids, limit=8 << 20):
    ids = list(ids)
    res = {}
    if not ids:
        return res
    p = subprocess.Popen(['git', '-C', path, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    import threading
    def feed():
        for i in ids:
            p.stdin.write(i.encode() + b'\n')
        p.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    for i in ids:
        hdr = p.stdout.readline().split()
        if len(hdr) < 3 or hdr[1] == b'missing':
            continue
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        if hdr[1] == b'blob' and n <= limit and b'\0' not in data[:8000]:
            res[i] = norm(data)
    p.wait()
    return res
def objs(path, *revs):
    r = subprocess.run(['git', '-C', path, 'rev-list', '--objects', *revs], capture_output=True)
    return {l.split(b' ', 1)[0].decode() for l in r.stdout.splitlines()}
ref_norm = {}
src = {}
mh = objs(clones['main'], new_main)
for b, h in blob_hashes(clones['main'], mh).items():
    ref_norm.setdefault(h, f'main@{new_main[:8]}')
for repo, path in clones.items():
    if repo == 'main':
        continue
    for b, h in blob_hashes(path, objs(path, '--all')).items():
        ref_norm.setdefault(h, f'{repo} --all')
census = json.load(open(census_p))
EXC = ('recovery/', 'drive/', 'incoming/')
OWN = {'refs/heads/claude/help-other-claude-models-829990', 'refs/pull/441/head'}
other = {f['git_blob'] for r in census['prs'] if r['ref'] not in OWN for f in r['files'] if not f['path'].startswith(EXC)}
for b, h in blob_hashes(clones['main'], other).items():
    ref_norm.setdefault(h, 'main census, non-excluded path')
own_norm = blob_hashes(clones['main'], only)
# supersession by path within the excluded areas
by_path = collections.defaultdict(set)
heads = {}
for r in census['prs']:
    if r['ref'] in OWN:
        continue
    heads[r['ref']] = r['head']
    for f in r['files']:
        if f['path'].startswith(EXC):
            by_path[f['path']].add((r['head'], f['git_blob']))
anc = {}
def is_anc(a, b):
    if (a, b) not in anc:
        anc[(a, b)] = subprocess.run(['git', '-C', clones['main'], 'merge-base', '--is-ancestor', a, b]).returncode == 0
    return anc[(a, b)]
rows = {}
cnt = collections.Counter()
groups = collections.defaultdict(list)
for b, r in only.items():
    h = own_norm.get(b)
    row = {'norm_sha256': h, 'binary_or_large': h is None}
    if h and h in ref_norm:
        row['normalized_copy_of'] = ref_norm[h]; k = 'normalized copy elsewhere'
    else:
        hs = {heads[x] for x in r['refs']}
        sup = []
        for p in r['paths'][:5]:
            for hh, bb in by_path[p]:
                if bb != b and hh not in hs and any(is_anc(x, hh) for x in hs):
                    sup.append(hh)
        if sup:
            row['superseded_at'] = sorted(set(sup))[:3]; k = 'superseded in lineage'
        else:
            k = 'residual'
    row['kind'] = k
    if h:
        groups[h].append(b)
    rows[b] = row
for b, row in rows.items():
    if row['kind'] == 'residual' and row['norm_sha256'] and len(groups[row['norm_sha256']]) > 1 and min(groups[row['norm_sha256']]) != b:
        row['kind'] = 'normalized duplicate within residual'; row['dup_of'] = min(groups[row['norm_sha256']])
    cnt[(only[b]['paths'][0].split('/')[0], row['kind'])] += 1
json.dump({'schema': 'claude54.t11c-norm-dedup.v1', 'reference_norm_hashes': len(ref_norm), 'rows': rows}, open(out, 'x'), indent=1)
for k, n in sorted(cnt.items()):
    print(k, n)
