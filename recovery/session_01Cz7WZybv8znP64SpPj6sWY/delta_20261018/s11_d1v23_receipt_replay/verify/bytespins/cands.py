"""Name/content candidate scan for the still-absent pins (recursing nested archives).
usage: cands.py OUTJSON repo:reflist [...]   (or repo:@objects for 'all historical blobs')
"""
import hashlib, io, json, re, subprocess, sys, tarfile, zipfile, gzip, threading
OUT = sys.argv[1]
specs = [a.split(':', 1) for a in sys.argv[2:]]
NAME_TOK = [b'C1_ALPHA', b'ALPHA_INTENSITY', b'h5_totals', b'zb_normal', b'zb_O', b'B1_TAXONOMY', b'C2_B_CLASSES',
            b'D2_BRANCH_CONTROL', b'H4_CLOSURE', b'D1_ASSEMBLY', b'zband', b'ZBAND']
# content signatures: pin hex prefixes (anyone citing them), titles, frozen markers
CONTENT_TOK = [b'ca2c02b8', b'8d7028e4', b'a570dade', b'f7697bcf', b'808d6901', b'7d7ddbec', b'2aba099d', b'9e95648a',
               b'a67d50b9', b'006b8a7d', b'634338b4', b'a7e1958c', b'86882dca', b'bcc177fe',
               b'B1 \xe2\x80\x94 ', b'C1 \xe2\x80\x94 ', b'C2 \xe2\x80\x94 ', b'D2 \xe2\x80\x94 ', b'H4 CLOSURE', b'H4 \xe2\x80\x94 CLOSURE',
               b'D1 ASSEMBLY', b'D1-ASM-', b'"C1_TOTAL"', b'"I_hi"']
rows = []

def scan(name, t, where, depth=0):
    base = name.rsplit('/', 1)[-1].encode('utf-8', 'surrogateescape')
    nt = [k.decode() for k in NAME_TOK if k.lower() in base.lower()]
    ct = []
    if len(t) < 20_000_000:
        ct = [k.decode('utf-8', 'replace') for k in CONTENT_TOK if k in t]
    if nt or (ct and b'BEGIN_FROZEN_BODY' in t) or any(c.startswith('"') for c in ct):
        rows.append(dict(where=where, name=name, size=len(t), sha=hashlib.sha256(t).hexdigest()[:8], name_tok=nt,
                         content_tok=ct, frozen=(b'BEGIN_FROZEN_BODY' in t), first=t[:120].decode('utf-8', 'replace')))
    if depth < 3:
        try:
            if t[:4] == b'PK\x03\x04':
                z = zipfile.ZipFile(io.BytesIO(t))
                for i in z.infolist():
                    if not i.is_dir() and i.file_size < 80_000_000:
                        scan(name + '!' + i.filename, z.read(i), where, depth + 1)
            elif re.search(r'\.(tar|tgz|tar\.gz|tar\.xz|tar\.bz2)$', name.lower()):
                tf = tarfile.open(fileobj=io.BytesIO(t))
                for m in tf.getmembers():
                    if m.isfile():
                        scan(name + '!' + m.name, tf.extractfile(m).read(), where, depth + 1)
            elif name.lower().endswith('.gz'):
                scan(name[:-3], gzip.decompress(t), where, depth + 1)
        except Exception as e:
            rows.append(dict(where=where, name=name, err=repr(e)[:100]))

for repo, reflist in specs:
    blob_paths = {}
    if reflist == '@objects':
        out = subprocess.run(['git', '-C', repo, 'rev-list', '--all', '--objects'], capture_output=True).stdout.decode('utf-8', 'surrogateescape')
        shas = []
        for l in out.splitlines():
            parts = l.split(' ', 1)
            if len(parts) == 2:
                blob_paths.setdefault(parts[0], []).append(('hist', parts[1]))
        typed = subprocess.run(['git', '-C', repo, 'cat-file', '--batch-check=%(objectname) %(objecttype)'],
                               input=('\n'.join(blob_paths) + '\n').encode(), capture_output=True).stdout.decode().split('\n')
        keep = {l.split()[0] for l in typed if l.endswith(' blob')}
        blob_paths = {k: v for k, v in blob_paths.items() if k in keep}
    else:
        for lab, ref in [l.split() for l in open(reflist) if l.strip()]:
            out = subprocess.run(['git', '-C', repo, 'ls-tree', '-r', '-z', ref], capture_output=True).stdout
            for ent in out.split(b'\0'):
                if not ent:
                    continue
                meta, path = ent.split(b'\t', 1)
                mode, typ, sha = meta.split()
                if typ == b'blob':
                    blob_paths.setdefault(sha.decode(), []).append((lab, path.decode('utf-8', 'surrogateescape')))
    shas = list(blob_paths)
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin'})
    inp = ('\n'.join(shas) + '\n').encode()
    threading.Thread(target=lambda: (p.stdin.write(inp), p.stdin.close())).start()
    for s in shas:
        hdr = p.stdout.readline().split()
        if len(hdr) < 3:
            continue
        n = int(hdr[2]); t = p.stdout.read(n); p.stdout.read(1)
        lab0, path0 = blob_paths[s][0]
        scan(path0, t, dict(repo=repo.rstrip('/').split('/')[-1], blob=s[:8], refs=sorted({l for l, _ in blob_paths[s]})[:6],
                            nrefs=len({l for l, _ in blob_paths[s]})))
    print(repo, reflist, len(shas), file=sys.stderr)
json.dump(rows, open(OUT, 'w'), indent=1)
print(len(rows))
