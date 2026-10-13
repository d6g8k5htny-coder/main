import hashlib, json, pathlib
root = pathlib.Path('frontiers/c2_torus_transfer_20261001').resolve()
manifest = json.loads((root / 'SOURCE_FILES.json').read_text())['files']
present = sorted(p.name for p in root.iterdir())
if present != sorted([e['path'] for e in manifest] + ['SOURCE_FILES.json']):
    raise ValueError('packet tree differs from manifest: ' + repr(present))
def ident(data):
    return len(data), hashlib.sha256(data).hexdigest(), hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
for entry in manifest:
    path = root / entry['path']
    if pathlib.PurePosixPath(entry['path']).name != entry['path'] or path.is_symlink():
        raise ValueError('flat regular source required: ' + entry['path'])
    if ident(path.read_bytes()) != (entry['bytes'], entry['sha256'], entry['git_blob']):
        raise ValueError('source identity mismatch: ' + entry['path'])
pins = json.loads((root / 'SOURCE_MAP.json').read_text())
for entry in pins['sources']:
    if not entry['current_required']:
        continue
    if ident(pathlib.Path(entry['path']).read_bytes()) != (entry['bytes'], entry['sha256'], entry['git_blob']):
        raise ValueError('consumed source identity mismatch on this tree: ' + entry['path'])
for copy, orig in pins['copies'].items():
    if (root / copy).read_bytes() != pathlib.Path(orig).read_bytes():
        raise ValueError('copy differs from its pinned original: ' + copy)
print('manifest, pins and copies verified')