"""Refresh bytes/sha256/git_blob of packet files in SOURCE_FILES.json, preserving indent. Std lib only."""
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
p = root / 'SOURCE_FILES.json'
raw = p.read_text(encoding='utf-8')
m = json.loads(raw)
for e in m['files']:
    d = (root / e['path']).read_bytes()
    e['bytes'] = len(d); e['sha256'] = hashlib.sha256(d).hexdigest()
    if 'git_blob' in e: e['git_blob'] = hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest()
indent = 2 if raw.startswith('{\n  "') else 1
p.write_text(json.dumps(m, indent=indent, ensure_ascii=False) + '\n', encoding='utf-8')
print('refreshed', p, 'indent', indent)
