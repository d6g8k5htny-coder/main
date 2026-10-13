import hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path('reviews/iba1_periodic_jet_claude_20261005').resolve()
meta = json.loads((root / 'SOURCE_FILES.json').read_text())
present = sorted(p.name for p in root.iterdir())
if present != sorted([e['path'] for e in meta['files']] + ['SOURCE_FILES.json']):
    raise ValueError('packet tree differs from manifest: ' + repr(present))
def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
for entry in meta['files'] + meta['pins_on_main']:
    path = pathlib.Path(entry['path']) if 'pins' in entry.get('kind', '') else root / entry['path']
    if path.is_symlink():
        raise ValueError('regular file required: ' + entry['path'])
    data = path.read_bytes()
    if blob(data) != entry['git_blob']:
        raise ValueError('git blob mismatch: ' + entry['path'])
    if 'sha256' in entry and (len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']):
        raise ValueError('source identity mismatch: ' + entry['path'])
flags = os.environ['FLAGS'].split()
checker = str(root / 'periodic_jet_check.py')
run = subprocess.run([sys.executable] + flags + [checker], capture_output=True, cwd=root)
if run.returncode != 0 or run.stdout != (root / 'RESULTS.json').read_bytes():
    raise ValueError('replay differs from RESULTS.json')
for label, code in (('M1', 1), ('M2', 1), ('M3', 1), ('M9', 2)):
    m = subprocess.run([sys.executable] + flags + [checker, '--mutant', label], capture_output=True, cwd=root)
    if m.returncode != code:
        raise ValueError('mutant %s exited %d, expected %d' % (label, m.returncode, code))
print('periodic jet replay: manifest, pins, replay and mutants PASS')

