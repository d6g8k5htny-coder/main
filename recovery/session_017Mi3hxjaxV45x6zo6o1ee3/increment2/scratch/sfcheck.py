"""Manifest identity + pins-on-main check (the cheap part of each packet workflow). Std lib only."""
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(); repo = pathlib.Path(sys.argv[2]).resolve()
m = json.loads((root / 'SOURCE_FILES.json').read_text())
present = sorted(p.name for p in root.iterdir() if p.name != '__pycache__')
assert present == sorted([e['path'] for e in m['files']] + ['SOURCE_FILES.json']), present
for e in m['files']:
    d = (root / e['path']).read_bytes()
    assert len(d) == e['bytes'] and hashlib.sha256(d).hexdigest() == e['sha256'], e['path']
    if 'git_blob' in e: assert hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest() == e['git_blob'], e['path']
for pin in m.get('pins_on_main', []):
    d = (repo / pin['path']).read_bytes()
    assert hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest() == pin['git_blob'], pin['path']
print('manifest and pins OK:', root.name)
