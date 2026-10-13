import hashlib, json, pathlib
root = pathlib.Path('frontiers/cusp_coefficient_certified_20261001').resolve()
manifest = json.loads((root / 'SOURCE_FILES.json').read_text())['files']
present = sorted(p.name for p in root.iterdir())
if present != sorted([e['path'] for e in manifest] + ['SOURCE_FILES.json']):
    raise ValueError('packet tree differs from manifest: ' + repr(present))
for entry in manifest:
    path = root / entry['path']
    if pathlib.PurePosixPath(entry['path']).name != entry['path'] or path.is_symlink():
        raise ValueError('flat regular source required: ' + entry['path'])
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256'] or blob != entry['git_blob']:
        raise ValueError('source identity mismatch: ' + entry['path'])
pins = json.loads((root / 'SOURCE_MAP.json').read_text())
for entry in pins['sources']:
    if not entry['current_required']:
        continue
    data = pathlib.Path(entry['path']).read_bytes()
    blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256'] or blob != entry['git_blob']:
        raise ValueError('consumed source identity mismatch on this tree: ' + entry['path'])
print('manifest and pins verified')
