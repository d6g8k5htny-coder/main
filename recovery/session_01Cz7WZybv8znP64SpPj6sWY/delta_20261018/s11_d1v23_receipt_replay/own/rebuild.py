"""Rebuild one DG-MIGRATION logical object from its chunk packs (own implementation for the S11 replay)."""
import hashlib, io, json, subprocess, sys, zipfile
G, REF, D = '/home/user/main', 'a01c72f1937875b94bfd78e183a6dbbdb2b04da4', 'drive/deltas/2026-09-19/DG-MIGRATION-20260919/'
want = sys.argv[1]
idx = json.load(open('objects.json'))
key = [k for k in idx['objects'] if k.startswith(want)]
assert len(key) == 1, key
obj = idx['objects'][key[0]]
packs = {}
out = bytearray()
for ch in obj['chunks']:
    meta = idx['chunks'][ch]
    p = meta['pack']
    if p not in packs:
        packs[p] = zipfile.ZipFile(io.BytesIO(subprocess.run(['git', '-C', G, 'show', REF + ':' + D + p], capture_output=True, check=True).stdout))
    z = packs[p]
    names = [n for n in z.namelist() if ch in n]
    assert len(names) == 1, (ch, names[:3])
    data = z.read(names[0])
    assert len(data) == meta['bytes'] and hashlib.sha256(data).hexdigest() == ch, ch
    out += data
h = hashlib.sha256(out).hexdigest()
print('object', key[0], 'bytes', len(out), 'expected', obj['bytes'], 'sha256', h, 'MATCH' if h == key[0] and len(out) == obj['bytes'] else 'MISMATCH')
open(sys.argv[2], 'wb').write(out)
