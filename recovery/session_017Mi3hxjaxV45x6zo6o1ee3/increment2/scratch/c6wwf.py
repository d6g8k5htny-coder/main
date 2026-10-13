import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path('reviews/c6_witness_collision_reconciliation_20260929').resolve()
manifest = json.loads((root / 'SOURCE_FILES.json').read_text())['files']
present = sorted(p.name for p in root.iterdir())
expected_names = sorted([e['path'] for e in manifest] + ['SOURCE_FILES.json'])
if present != expected_names:
    raise ValueError('packet tree differs from manifest: ' + repr(present))
for entry in manifest:
    path = root / entry['path']
    if pathlib.PurePosixPath(entry['path']).name != entry['path'] or path.is_symlink():
        raise ValueError('flat regular source required: ' + entry['path'])
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('source identity mismatch: ' + entry['path'])
script = str(root / 'reconciliation_check.py')
expected = (root / 'RESULTS.json').read_bytes()
mutants = ['allow-symlink', 'no-hash', 'drop-edge', 'stale-fingerprint', 'executed-flag', 'close-residual', 'drop-lower-bound', 'regional-strict']
for flags in (['-B', '-S'], ['-B', '-O', '-S']):
    out = subprocess.run([sys.executable, *flags, script],
                         check=True, capture_output=True, timeout=300).stdout
    if out != expected:
        raise RuntimeError('output differs from RESULTS.json under ' + ' '.join(flags))
    for m in mutants:
        rc = subprocess.run([sys.executable, *flags, script, '--mutant', m],
                            capture_output=True, timeout=300).returncode
        if rc != 1:
            raise RuntimeError('mutant ' + m + ' not rejected')
print(json.dumps({'scientific_effect': 'NONE', 'mathematical_acceptance': False}))
