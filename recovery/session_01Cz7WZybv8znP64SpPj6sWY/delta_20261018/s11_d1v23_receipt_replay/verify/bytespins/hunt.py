"""Independent pin hunter for d1_falsify_v4.py (bytes-and-pins lens).

Re-implements the gate's hash rules from the gate source (not from pinscan.py) and
applies EVERY rule to EVERY blob reachable from the given refs (any path, any name),
plus members of zip/tar archives found in those trees. A pin is LANDED if some blob
satisfies it under the gate's own rule for that pin.
usage: hunt.py GATE OUTJSON repo:reflist [repo:reflist ...]
"""
import hashlib, io, json, re, subprocess, sys, tarfile, zipfile, gzip, ast

gate_src = open(sys.argv[1]).read()
OUT = sys.argv[2]
specs = [a.split(':', 1) for a in sys.argv[3:]]

# ---- parse pins from the gate source with ast (list literals) ----
tree = ast.parse(gate_src)
lists = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        nm = node.targets[0].id
        if nm in ('WHOLE', 'BODY', 'SEP', 'BEFORE_HASHLINE', 'HIST'):
            lists[nm] = ast.literal_eval(node.value)
pins = []  # (label, rule, hash)
for rel, h in lists['WHOLE']:
    pins.append((rel, 'whole', h))
for rel, h, mode in lists['BODY']:
    pins.append((rel, 'markers_' + mode, h))
for rel, h in lists['SEP']:
    pins.append((rel, 'sep', h))
for rel, h in lists['BEFORE_HASHLINE']:
    pins.append((rel, 'before_hashline', h))
for rel, h in lists['HIST']:
    pins.append(('D1_assembly/' + rel, 'd1_body', h))
# inline cks (strings in ck(...) calls): body_strip and excl_hashline
for m in re.finditer(r'ck\("A-body-strip ([^"]+)",\s*\n\s*hashlib\.sha256\(body_strip\(.*?\n\s*==\s*"([0-9a-f]{64})"\)', gate_src):
    pins.append((m.group(1), 'body_strip', m.group(2)))
for m in re.finditer(r'ck\("A-body-exclhash ([^"]+)",\s*\n\s*hashlib\.sha256\(body_excl_hashline\(.*?\n\s*==\s*"([0-9a-f]{64})"\)', gate_src):
    pins.append((m.group(1), 'excl_hashline', m.group(2)))
assert len(pins) == 47, len(pins)

BEG = b"BEGIN_FROZEN_BODY\n"

def r_whole(t):
    return t

def _markers(t):
    a = t.find(BEG) + len(BEG)
    b = t.find(b"END_FROZEN_BODY")
    if not (a >= len(BEG) and b > a):
        return None
    return t[a:b]

def r_markers_raw(t):
    return _markers(t)

def r_markers_rstrip(t):
    s = _markers(t)
    return None if s is None else s.rstrip(b"\n") + b"\n"

def r_body_strip(t):
    # gate body_strip has NO marker ck; replicate exactly (find may return -1)
    a = t.find(BEG) + len(BEG); b = t.find(b"END_FROZEN_BODY")
    if t.find(BEG) < 0 or b < 0:
        return None
    return t[a:b].strip(b"\n") + b"\n"

def r_sep(t):
    i = t.find(b"\n---\n\n")
    return t[:i] if i > 0 else None

def r_before_hashline(t):
    L = t.split(b"\n")
    k = [i for i, l in enumerate(L) if l.startswith(b"body-sha256:")]
    return b"\n".join(L[:k[0]]) if len(k) == 1 else None

def r_excl_hashline(t):
    L = t.split(b"\n")
    k = [i for i, l in enumerate(L) if b"Body hash of this document" in l]
    return b"\n".join(L[:k[0]] + L[k[0] + 1:]) if len(k) == 1 else None

def r_d1_body(t):
    try:
        L = t.decode().split("\n")
    except UnicodeDecodeError:
        return None
    idx = [i for i, l in enumerate(L) if l in ("BEGIN_FROZEN_BODY", "END_FROZEN_BODY")]
    if len(idx) != 2:
        return None
    seg = L[idx[0] + 1:idx[1]]
    while seg and seg[0] == "":
        seg = seg[1:]
    while seg and seg[-1] == "":
        seg = seg[:-1]
    return ("\n".join(seg) + "\n").encode()

RULES = {'whole': r_whole, 'markers_raw': r_markers_raw, 'markers_rstrip': r_markers_rstrip,
         'body_strip': r_body_strip, 'sep': r_sep, 'before_hashline': r_before_hashline,
         'excl_hashline': r_excl_hashline, 'd1_body': r_d1_body}
PINH = {}
for lab, rule, h in pins:
    PINH.setdefault(h, []).append((lab, rule))

hits = []  # (pin label, pin rule, rule that matched, repo, ref, path, blob)
nblobs = 0

def check_bytes(t, where):
    global nblobs
    nblobs += 1
    for rn, fn in RULES.items():
        try:
            o = fn(t)
        except Exception:
            o = None
        if o is None:
            continue
        h = hashlib.sha256(o).hexdigest()
        if h in PINH:
            for lab, prule in PINH[h]:
                hits.append(dict(pin=lab, pin_rule=prule, matched_rule=rn, own_rule=(rn == prule), **where))

def archive_members(name, t):
    out = []
    try:
        if name.lower().endswith('.zip') or t[:4] == b'PK\x03\x04':
            z = zipfile.ZipFile(io.BytesIO(t))
            for i in z.infolist():
                if not i.is_dir() and i.file_size < 50_000_000:
                    out.append((i.filename, z.read(i)))
        elif re.search(r'\.(tar|tgz|tar\.gz|tar\.xz|tar\.bz2)$', name.lower()):
            tf = tarfile.open(fileobj=io.BytesIO(t))
            for m in tf.getmembers():
                if m.isfile() and m.size < 50_000_000:
                    out.append((m.name, tf.extractfile(m).read()))
        elif name.lower().endswith('.gz'):
            out.append((name[:-3], gzip.decompress(t)))
    except Exception as e:
        out.append(('__ERR__' + repr(e)[:80], b''))
    return out

archives = []
for repo, reflist in specs:
    refs = [l.split() for l in open(reflist) if l.strip()]
    blob_paths = {}
    for lab, ref in refs:
        out = subprocess.run(['git', '-C', repo, 'ls-tree', '-r', '-z', ref], capture_output=True).stdout
        for ent in out.split(b'\0'):
            if not ent:
                continue
            meta, path = ent.split(b'\t', 1)
            mode, typ, sha = meta.split()
            if typ != b'blob':
                continue
            blob_paths.setdefault(sha.decode(), []).append((lab, path.decode('utf-8', 'surrogateescape')))
    shas = list(blob_paths)
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin'})
    inp = ('\n'.join(shas) + '\n').encode()
    import threading
    threading.Thread(target=lambda: (p.stdin.write(inp), p.stdin.close())).start()
    missing = 0
    for s in shas:
        hdr = p.stdout.readline().split()
        if len(hdr) < 3:
            missing += 1
            continue
        n = int(hdr[2]); t = p.stdout.read(n); p.stdout.read(1)
        lab0, path0 = blob_paths[s][0]
        where = dict(repo=repo.split('/')[-1], ref=lab0, path=path0, blob=s[:8], n_paths=len(blob_paths[s]),
                     refs=sorted({l for l, _ in blob_paths[s]}))
        check_bytes(t, where)
        if re.search(r'\.(zip|tar|tgz|gz|xz|bz2|7z|rar)$', path0.lower()) or t[:4] == b'PK\x03\x04':
            mem = archive_members(path0, t)
            archives.append(dict(repo=where['repo'], path=path0, blob=s[:8], members=len(mem)))
            for mn, mt in mem:
                check_bytes(mt, dict(where, path=path0 + '!' + mn))
    print(repo, 'refs', len(refs), 'blobs', len(shas), 'missing', missing, file=sys.stderr)

status = []
for lab, rule, h in pins:
    own = [x for x in hits if x['pin'] == lab and x['own_rule']]
    other = [x for x in hits if x['pin'] == lab and not x['own_rule']]
    status.append(dict(pin=lab, rule=rule, sha=h, status='LANDED' if own else 'ABSENT',
                       own_hits=own, cross_rule_hits=other))
json.dump(dict(npins=len(pins), nblobs=nblobs, archives=archives, status=status), open(OUT, 'w'), indent=1)
for s in status:
    locs = sorted({(x['repo'], x['path'][-70:]) for x in s['own_hits']})
    print(f"{s['status']:7s} {s['rule']:15s} {s['pin']:52s} {s['sha'][:8]} n={len(s['own_hits'])} x={len(s['cross_rule_hits'])} " + ' | '.join(f'{r}:{p}' for r, p in locs[:2]))
print('LANDED', sum(s['status'] == 'LANDED' for s in status), 'ABSENT', sum(s['status'] == 'ABSENT' for s in status), 'blobs', nblobs, 'archives', len(archives))
