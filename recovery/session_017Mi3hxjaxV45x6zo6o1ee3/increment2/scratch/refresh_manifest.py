"""Refresh bytes/sha256 of the packet files in MANIFEST.json (Lean unchanged). Standard library only."""
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
p = root / 'MANIFEST.json'
m = json.loads(p.read_text(encoding='utf-8'))
for e in m['files']:
    d = (root / e['path']).read_bytes()
    e['bytes'] = len(d); e['sha256'] = hashlib.sha256(d).hexdigest()
p.write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(hashlib.sha256(p.read_bytes()).hexdigest())
