import hashlib, json, pathlib
PK = pathlib.Path('/home/user/wt_c2/frontiers/c2_exact_20261001')
files = []
for p in sorted(PK.iterdir()):
    if p.name in ('SOURCE_FILES.json', '__pycache__'):
        continue
    data = p.read_bytes()
    files.append({'path': p.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                  'git_blob': hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()})
(PK / 'SOURCE_FILES.json').write_text(json.dumps({'object': 'CL-C2-EXACT-20261001-v1', 'files': files}, indent=1) + '\n')
print([f['path'] for f in files])
