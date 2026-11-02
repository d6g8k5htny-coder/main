#!/usr/bin/env python3
"""Inventory of committed archives at the federation tips, read in memory (nothing extracted to disk, nothing executed).

Usage: python3 -I -B archive_inventory.py CLONES_JSON CANDIDATE_CATALOG OUTPUT

For every .zip/.tgz/.tar.gz/.gz file at each repository's tip (nested zips are opened one level down), each member gets its
size, SHA-256 and Git blob id (sha1 of "blob <n>\\0" + bytes, as git hash-object computes it). A member is marked copy_of when
its blob equals a file in main's tree at MAIN_REF or Math-'s tree at MATH_REF, and disposed_location when that file lies at or
under a location of the candidate catalog (adopted or proposed rows)."""
import gzip, hashlib, io, json, re, subprocess, sys, tarfile, zipfile
REFS = {'main': 'd2b0216e15d545c47f25b6a9265dd3298971e9fe', 'Math-': 'c88c34b2e912df8b4885b04a19293685f0cb6c00'}
ARCH = re.compile(r'\.(zip|tgz|tar\.gz|tar|gz)$', re.I)
clones_p, cand_p, out = sys.argv[1:4]
clones = json.load(open(clones_p))
cand = json.load(open(cand_p))

def git(clone, *a, text=True):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, text=text, check=True).stdout

def blob_id(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()

tree_blobs = {}
for name in ('main', 'Math-'):
    ref = git(clones[name], 'rev-parse', REFS[name]).strip()
    REFS[name] = ref
    for line in git(clones[name], 'ls-tree', '-r', ref).splitlines():
        meta, path = line.split('\t', 1)
        tree_blobs.setdefault(meta.split()[2], f'{name}:{path}')
locs = set()
for key in ('quarantines', 'proposed_quarantines', 'proposed_pinning_amendments'):
    for r in cand.get(key, []):
        for l in (r['location'] if isinstance(r['location'], list) else [r['location']]):
            m = isinstance(l, str) and re.match(r'd6g8k5htny-coder/(main|Math-)@[0-9a-f]{40}:([^#]+)', l)
            if m:
                locs.add((m.group(1), m.group(2).rstrip('/')))

def disposed(where):
    repo, _, path = where.partition(':')
    parts = path.split('/')
    return next(('/'.join(parts[:i]) for i in range(len(parts), 0, -1) if (repo, '/'.join(parts[:i])) in locs), None)

def members(name, data, depth=0):
    rows = []
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for i in z.infolist():
                if not i.is_dir():
                    rows.append((i.filename, z.read(i)))
    elif re.search(r'\.(tgz|tar\.gz|tar)$', name, re.I):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                if m.isfile():
                    rows.append((m.name, t.extractfile(m).read()))
    elif name.lower().endswith('.gz'):
        rows.append((name.rsplit('/', 1)[-1][:-3], gzip.decompress(data)))
    out = []
    for mname, b in rows:
        rec = {'member': mname, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'git_blob': blob_id(b)}
        hit = tree_blobs.get(rec['git_blob'])
        rec['copy_of'] = hit
        rec['copy_disposed_location'] = disposed(hit) if hit else None
        if depth == 0 and ARCH.search(mname):
            try:
                rec['nested'] = members(mname, b, depth + 1)
            except Exception as e:
                rec['nested_error'] = f'{type(e).__name__}: {e}'
        out.append(rec)
    return out

archives = []
for name, clone in sorted(clones.items()):
    try:
        ref = git(clone, 'rev-parse', REFS.get(name, 'HEAD')).strip()
        listing = git(clone, 'ls-tree', '-r', '-l', ref)
    except Exception:
        continue
    for line in listing.splitlines():
        meta, path = line.split('\t', 1)
        if not ARCH.search(path):
            continue
        data = subprocess.run(['git', '-C', clone, 'cat-file', 'blob', f'{ref}:{path}'], capture_output=True, check=True).stdout
        rec = {'repository': f'd6g8k5htny-coder/{name}', 'commit': ref, 'path': path, 'git_blob': meta.split()[2], 'bytes': len(data),
               'sha256': hashlib.sha256(data).hexdigest()}
        try:
            rec['members'] = members(path, data)
        except Exception as e:
            rec['members'], rec['error'] = [], f'{type(e).__name__}: {e}'
        archives.append(rec)
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.archive-inventory.v1', 'refs': REFS, 'archives': archives}, f, indent=1)
    f.write('\n')
for a in archives:
    ms = a['members']
    flat = ms + [n for m in ms for n in m.get('nested', [])]
    print(a['repository'].split('/')[1], a['commit'][:8], a['path'], len(ms), 'members', sum(1 for m in flat if m['copy_of']), 'copies',
          sum(1 for m in ms if 'nested' in m), 'nested', a.get('error', ''))
