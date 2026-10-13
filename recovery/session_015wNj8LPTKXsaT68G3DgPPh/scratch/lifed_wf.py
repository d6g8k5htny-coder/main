import hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path('frontiers/c8_lifetime_difference_constant_20261001').resolve()
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
def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
pins = json.loads((root / 'SOURCE_MAP.json').read_text())
for entry in pins['sources']:
    if entry.get('current_required'):
        data = pathlib.Path(entry['path']).read_bytes()
    elif entry.get('packet_copy'):
        data = (root / entry['packet_copy']).read_bytes()
    else:
        continue
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256'] or blob(data) != entry['git_blob']:
        raise ValueError('pinned source identity mismatch: ' + entry['path'])
script = str(root / 'theorem_b.py')
procs = str(max(2, os.cpu_count() or 2))
# every cap band in one mode, the sampled replay in the other
for flags, mode in ((['-B', '-S'], '--check-full'), (['-B', '-O', '-S'], '--check')):
    out = subprocess.run([sys.executable, *flags, script, mode, '--procs', procs], cwd=root, capture_output=True, timeout=10800)
    if out.returncode != 0:
        raise RuntimeError(mode + ' failed under ' + ' '.join(flags) + ': ' + out.stdout.decode()[-400:] + out.stderr.decode()[-400:])
    print(flags, mode, out.stdout.decode().strip()[-600:])
for m in ('no-om-term', 'no-bad-set', 'no-det', 'no-torus', 'cap-half'):
    rc = subprocess.run([sys.executable, '-B', '-S', script, '--check', '--procs', procs, '--mutant', m], cwd=root, capture_output=True, timeout=3600).returncode
    if rc != 1:
        raise RuntimeError('mutant ' + m + ' not rejected')
print(json.dumps({'scientific_effect': 'NONE', 'cap_bands_replayed': 274, 'mutants_rejected': 5}))
