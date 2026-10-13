import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path('frontiers/c8_theorem_a_constants_planar_20260930').resolve()
manifest = json.loads((root / 'SOURCE_FILES.json').read_text())['files']
present = sorted(p.name for p in root.iterdir())
if present != sorted([e['path'] for e in manifest] + ['SOURCE_FILES.json']):
    raise ValueError('packet tree differs from manifest: ' + repr(present))
for entry in manifest:
    path = root / entry['path']
    if pathlib.PurePosixPath(entry['path']).name != entry['path'] or path.is_symlink():
        raise ValueError('flat regular source required: ' + entry['path'])
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('source identity mismatch: ' + entry['path'])
pins = json.loads((root / 'SOURCE_MAP.json').read_text())
for entry in pins['sources']:
    if not entry['current_required']:
        continue
    data = pathlib.Path(entry['path']).read_bytes()
    blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256'] or blob != entry['git_blob']:
        raise ValueError('consumed source identity mismatch on this tree: ' + entry['path'])
script = str(root / 'theorem_a.py')
# every one of the 153 bands in one mode, the sampled replay in the other
print('manifest+pins OK')
