"""Exercise actual archive readback programs against specified hostile tar inputs."""
import copy
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent
LABEL = sys.argv[1]
OUTPUT = ROOT / ('tag-custody-check-' + LABEL)
OUTPUT.mkdir(exist_ok=False)
CASES = ('baseline', 'corrupt-source', 'missing-tree', 'duplicate-member',
         'unexpected-member', 'missing-binding', 'duplicate-binding', 'relabeled-binding')


def make_case(source, kind, case, destination):
    with tarfile.open(source, 'r:gz') as original:
        items = [(copy.copy(member), original.extractfile(member).read()) for member in original.getmembers()]
    if case != 'baseline':
        if case == 'corrupt-source':
            i = next(i for i, (member, _) in enumerate(items) if member.name.startswith('sources/'))
            member, data = items[i]
            if not data:
                raise ValueError('source chosen for corruption was empty')
            items[i] = (member, bytes([data[0] ^ 1]) + data[1:])
        elif case == 'missing-tree':
            i = next(i for i, (member, _) in enumerate(items) if member.name.startswith('git-objects/tree/'))
            items.pop(i)
        elif case == 'duplicate-member':
            member, data = next((copy.copy(m), d) for m, d in items if m.name.startswith('sources/'))
            items.append((member, data))
        elif case == 'unexpected-member':
            member = tarfile.TarInfo('unexpected-evidence.txt')
            data = b'unexpected\n'
            member.size = len(data)
            items.append((member, data))
        else:
            i = next(i for i, (member, _) in enumerate(items) if member.name == 'MANIFEST.json')
            member, data = items[i]
            manifest = json.loads(data)
            key = 'tree_bindings' if kind == 'math' else 'files'
            rows = manifest[key]
            if case == 'missing-binding':
                rows.pop()
            elif case == 'duplicate-binding':
                rows[-1] = copy.deepcopy(rows[0])
            elif case == 'relabeled-binding':
                tag = rows[0]['tag']
                other_tag = next(row['tag'] for row in rows if row['tag'] != tag)
                rows[0]['tag'] = other_tag
            else:
                raise ValueError('unrecognized case')
            data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
            member.size = len(data)
            items[i] = (member, data)
    with tarfile.open(destination, 'w:gz') as out:
        for member, data in items:
            member.size = len(data)
            out.addfile(member, io.BytesIO(data))


records = []
for kind in ('math', 'main'):
    source = ROOT / (kind + '-tag-native-custody-20261110.tar.gz')
    verifier = ROOT / ('verify_' + kind + '_tag_native_custody.py')
    for case in CASES:
        candidate = OUTPUT / (kind + '-' + case + '.tar.gz')
        make_case(source, kind, case, candidate)
        for mode, flags in (('normal', []), ('optimized', ['-O'])):
            result = subprocess.run([sys.executable, '-B', *flags, '-S', str(verifier), str(candidate)],
                                    capture_output=True, timeout=30)
            stem = kind + '-' + case + '-' + mode
            (OUTPUT / (stem + '.stdout')).write_bytes(result.stdout)
            (OUTPUT / (stem + '.stderr')).write_bytes(result.stderr)
            expected = result.returncode == 0 if case == 'baseline' else result.returncode != 0
            records.append({'kind': kind, 'case': case, 'mode': mode, 'exit': result.returncode,
                            'expected_observed': expected,
                            'archive_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                            'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
                            'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()})
report = {'cases': records, 'unexpected': sum(not row['expected_observed'] for row in records),
          'scope': 'exact listed inputs only; does not prove exhaustive archive or theorem coverage'}
(OUTPUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'label': LABEL, 'cases': len(records), 'unexpected': report['unexpected'],
                  'failed_cases': [f"{r['kind']}:{r['case']}:{r['mode']} exit={r['exit']}" for r in records if not r['expected_observed']]}, indent=2))
sys.exit(1 if report['unexpected'] else 0)
