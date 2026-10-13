import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path('frontiers/c8_cap_failure_coefficient_dims_20261001').resolve()
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
script = str(root / 'cap_coefficient_dims.py')
for flags in (['-B', '-S'], ['-B', '-O', '-S']):
    out = subprocess.run([sys.executable, *flags, script, '--check'], cwd=root, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError('check failed under ' + ' '.join(flags) + ': ' + out.stdout.decode()[-400:] + out.stderr.decode()[-400:])
    print(flags, out.stdout.decode().strip()[-300:])
for flags in (['-B', '-S'], ['-B', '-O', '-S']):
    for m in ('no-y-tail', 'r3-one', 'no-bp', 'frob-omega-off'):
        rc = subprocess.run([sys.executable, *flags, script, '--check', '--mutant', m], cwd=root, capture_output=True, timeout=600).returncode
        if rc != 1:
            raise RuntimeError('mutant ' + m + ' not rejected under ' + ' '.join(flags))
print(json.dumps({'scientific_effect': 'NONE', 'certified_values_replayed': True, 'mutants_rejected': 8}))
