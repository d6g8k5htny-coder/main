"""Recursive hunt: reconstruct every DG-MIGRATION chunked object (hardening) and every blob in the given
refs; recurse into zip/tar/gz members to depth 5; apply all gate rules; report pin hits.
Also records name/content candidates for the still-absent pins inside archives.
usage: GATE=... hunt2.py OUTJSON repo:reflist [...]
"""
import hashlib, io, json, re, subprocess, sys, tarfile, zipfile, gzip, threading, os, bz2, lzma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_lib as L

OUT = sys.argv[1]
specs = [a.split(':', 1) for a in sys.argv[2:]]
HARD = 'a01c72f1937875b94bfd78e183a6dbbdb2b04da4'
DG = 'drive/deltas/2026-09-19/DG-MIGRATION-20260919'
ABS_NAMES = ['C1_ALPHA', 'h5_totals', 'zb_normal', 'zb_O', 'B1_TAXONOMY', 'C2_B_CLASSES', 'D2_BRANCH_CONTROL',
             'H4_CLOSURE', 'D1_ASSEMBLY', 'ALPHA_INTENSITY']
hits, names, errs, archives = [], [], [], []
seen_hash = set()
stats = dict(units=0, archive_members=0)

def rules_check(t, where):
    for rn, fn in L.RULES.items():
        try:
            o = fn(t)
        except Exception:
            o = None
        if o is None:
            continue
        h = hashlib.sha256(o).hexdigest()
        if h in L.PINH:
            for lab, prule in L.PINH[h]:
                hits.append(dict(pin=lab, pin_rule=prule, matched_rule=rn, own_rule=(rn == prule), where=where))

def members(name, t):
    low = name.lower()
    if t[:4] == b'PK\x03\x04' or t[:4] == b'PK\x05\x06':
        z = zipfile.ZipFile(io.BytesIO(t))
        return [(i.filename, z.read(i)) for i in z.infolist() if not i.is_dir()]
    if re.search(r'\.(tar|tgz|tar\.gz|tar\.xz|tar\.bz2|txz)$', low) or (len(t) > 262 and t[257:262] == b'ustar'):
        tf = tarfile.open(fileobj=io.BytesIO(t))
        return [(m.name, tf.extractfile(m).read()) for m in tf.getmembers() if m.isfile()]
    if t[:2] == b'\x1f\x8b':
        return [(re.sub(r'\.gz$', '', name), gzip.decompress(t))]
    if t[:3] == b'BZh':
        return [(name + '.unbz2', bz2.decompress(t))]
    if t[:6] == b'\xfd7zXZ\x00':
        return [(name + '.unxz', lzma.decompress(t))]
    return None

def scan(name, t, where, depth):
    h = hashlib.sha256(t).hexdigest()
    stats['units'] += 1
    key = (h,)
    first = key not in seen_hash
    seen_hash.add(key)
    if not first:
        return  # identical bytes already scanned (same results)
    w = dict(where, path=name, sha=h[:12], size=len(t))
    rules_check(t, w)
    base = name.rsplit('/', 1)[-1].rsplit('!', 1)[-1]
    if any(k.lower() in base.lower() for k in ABS_NAMES):
        names.append(w)
    if depth < 6:
        try:
            m = members(name, t)
        except Exception as e:
            errs.append(dict(w, err=repr(e)[:120]))
            m = None
        if m is not None:
            archives.append(dict(w, n=len(m)))
            for mn, mt in m:
                stats['archive_members'] += 1
                scan(name + '!' + mn, mt, where, depth + 1)

def batch_blobs(repo, shas):
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin'})
    inp = ('\n'.join(shas) + '\n').encode()
    threading.Thread(target=lambda: (p.stdin.write(inp), p.stdin.close())).start()
    for s in shas:
        hdr = p.stdout.readline().split()
        if len(hdr) < 3:
            yield s, None
            continue
        n = int(hdr[2]); t = p.stdout.read(n); p.stdout.read(1)
        yield s, t

# ---- 1. DG-MIGRATION chunked objects on hardening ----
G = '/home/user/main'
objs = json.loads(subprocess.run(['git', '-C', G, 'show', HARD + ':' + DG + '/objects.json'], capture_output=True).stdout)
packs = {}
def pack(pth):
    if pth not in packs:
        b = subprocess.run(['git', '-C', G, 'show', HARD + ':' + DG + '/' + pth], capture_output=True).stdout
        packs[pth] = zipfile.ZipFile(io.BytesIO(b))
    return packs[pth]
cov = {}
for l in subprocess.run(['git', '-C', G, 'show', HARD + ':' + DG + '/coverage.jsonl'], capture_output=True).stdout.decode().splitlines():
    r = json.loads(l)
    if 'storage' in r and 'object' in r['storage']:
        cov.setdefault(r['storage']['object'], []).append(r['id'])
inv = {}
for l in subprocess.run(['git', '-C', G, 'show', HARD + ':drive/inventory.jsonl'], capture_output=True).stdout.decode().splitlines():
    r = json.loads(l); inv[r['id']] = r.get('path')
bad = 0
for oh, o in objs['objects'].items():
    data = b''.join(pack(objs['chunks'][c]['pack']).read(c) for c in o['chunks'])
    if hashlib.sha256(data).hexdigest() != oh:
        bad += 1
        continue
    ids = cov.get(oh, [])
    nm = next((inv[i] for i in ids if inv.get(i)), None) or ('DGobj/' + oh)
    scan('DG:' + nm, data, dict(repo='main', ref='hardening:DG-MIGRATION', blob=oh[:12]), 0)
print('DG objects', len(objs['objects']), 'bad', bad, file=sys.stderr)

# ---- 2. all blobs in all given refs ----
for repo, reflist in specs:
    blob_paths = {}
    for lab, ref in [l.split() for l in open(reflist) if l.strip()]:
        out = subprocess.run(['git', '-C', repo, 'ls-tree', '-r', '-z', ref], capture_output=True).stdout
        for ent in out.split(b'\0'):
            if ent:
                meta, path = ent.split(b'\t', 1)
                mode, typ, sha = meta.split()
                if typ == b'blob':
                    blob_paths.setdefault(sha.decode(), []).append((lab, path.decode('utf-8', 'surrogateescape')))
    for s, t in batch_blobs(repo, list(blob_paths)):
        if t is None:
            errs.append(dict(repo=repo, blob=s, err='missing'))
            continue
        refs = sorted({l for l, _ in blob_paths[s]})
        scan(blob_paths[s][0][1], t, dict(repo=repo.rstrip('/').split('/')[-1], ref=refs[0], nrefs=len(refs), refs6=refs[:6], blob=s[:12]), 0)
    print(repo, len(blob_paths), file=sys.stderr)

status = []
for lab, rule, h in L.pins:
    own = [x for x in hits if x['pin'] == lab and x['own_rule']]
    status.append(dict(pin=lab, rule=rule, sha=h, status='LANDED' if own else 'ABSENT', own=own,
                       cross=[x for x in hits if x['pin'] == lab and not x['own_rule']]))
json.dump(dict(stats=stats, status=status, names=names, errs=errs, archives=archives), open(OUT, 'w'), indent=1)
for s in status:
    locs = sorted({x['where']['path'][-90:] for x in s['own']})
    print(f"{s['status']:7s} {s['rule']:15s} {s['pin']:52s} {s['sha'][:8]} n={len(s['own'])} " + ' | '.join(locs[:2]))
print('LANDED', sum(s['status'] == 'LANDED' for s in status), 'ABSENT', sum(s['status'] == 'ABSENT' for s in status), stats,
      'archives', len(archives), 'errs', len(errs))
