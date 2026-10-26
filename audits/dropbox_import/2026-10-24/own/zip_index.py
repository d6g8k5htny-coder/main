#!/usr/bin/env python3
"""content_hash + sha256 index of zip members (read as data; nothing executed). Usage: zip_index.py <label> <zip> <out.json>"""
import hashlib, json, sys, zipfile
label, zp, out = sys.argv[1:4]
def dbx(data):
    h = hashlib.sha256()
    for i in range(0, len(data), 4 * 1024 * 1024):
        h.update(hashlib.sha256(data[i:i + 4 * 1024 * 1024]).digest())
    return h.hexdigest()
idx = {}
with zipfile.ZipFile(zp) as z:
    for info in z.infolist():
        if info.is_dir(): continue
        data = z.read(info)
        idx[f'{label}:{info.filename}'] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'content_hash': dbx(data)}
json.dump(idx, open(out, 'w'), separators=(',', ':'))
print(label, len(idx))
