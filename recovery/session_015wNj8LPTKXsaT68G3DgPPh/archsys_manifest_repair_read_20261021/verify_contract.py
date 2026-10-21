"""Exercise a pinned manifest utility on synthetic files only.
Usage: python3 -I -B -S verify_contract.py SOURCE EXPECTED_GIT_BLOB
Exit 0 means every exact protocol matches the proposed contract, not scientific acceptance.
"""
import copy
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

CASES = (
    'valid', 'total_plus_one', 'total_missing', 'total_string', 'total_float',
    'total_bool', 'total_negative', 'duplicate_identical', 'duplicate_bad_first',
    'duplicate_bad_last', 'wrong_hash', 'missing_row', 'extra_row',
    'payload_changed', 'two_simultaneous', 'raw_metadata_only',
)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1]).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    if blob != sys.argv[2]:
        raise SystemExit('Source identity mismatch: ' + blob)
    report = dict(source_blob=blob, source_bytes=len(source),
                  source_sha256=hashlib.sha256(source).hexdigest(),
                  python=sys.version, platform=platform.platform(), cases=[])
    for optimized in (False, True):
        with tempfile.TemporaryDirectory(prefix='manifest_contract_') as temporary:
            root = Path(temporary)
            script = root / 'make_manifest.py'
            script.write_bytes(source)
            payload = root / 'payload.txt'
            payload.write_bytes(b'data\n')
            command = [sys.executable, '-I', '-B', '-S']
            if optimized:
                command.append('-O')
            command.append(str(script))
            generated = subprocess.run(command, capture_output=True, timeout=10)
            if generated.returncode != 0 or generated.stderr:
                raise SystemExit('Generator failed: ' + repr(generated))
            manifest = root / 'MANIFEST.json'
            original = json.loads(manifest.read_text(encoding='utf-8'))
            if ([r['path'] for r in original['committed_files']]
                    != ['make_manifest.py', 'payload.txt']
                    or original['committed_total_bytes'] != len(source) + 5):
                raise SystemExit('Unexpected generated fixture')
            for case in CASES:
                payload.write_bytes(b'data\n')
                data = copy.deepcopy(original)
                good = copy.deepcopy(data['committed_files'][1])
                bad = dict(good, sha256='0' * 64)
                errors = []
                if case == 'total_plus_one':
                    data['committed_total_bytes'] += 1
                elif case == 'total_missing':
                    del data['committed_total_bytes']
                elif case == 'total_string':
                    data['committed_total_bytes'] = str(data['committed_total_bytes'])
                elif case == 'total_float':
                    data['committed_total_bytes'] = float(data['committed_total_bytes'])
                elif case == 'total_bool':
                    data['committed_total_bytes'] = True
                elif case == 'total_negative':
                    data['committed_total_bytes'] = -1
                elif case == 'duplicate_identical':
                    data['committed_files'].append(good)
                elif case == 'duplicate_bad_first':
                    data['committed_files'].insert(0, bad)
                elif case == 'duplicate_bad_last':
                    data['committed_files'].append(bad)
                    errors.append('payload.txt')
                elif case == 'wrong_hash':
                    data['committed_files'][1] = bad
                    errors.append('payload.txt')
                elif case == 'missing_row':
                    data['committed_files'].pop()
                    errors.append('payload.txt')
                elif case == 'extra_row':
                    data['committed_files'].append(dict(good, path='ghost.txt'))
                    errors.append('ghost.txt')
                elif case == 'payload_changed':
                    payload.write_bytes(b'xxxx\n')
                    errors.append('payload.txt')
                elif case == 'two_simultaneous':
                    data['committed_files'].append(good)
                    data['committed_total_bytes'] += 1
                elif case == 'raw_metadata_only':
                    data['raw_total_bytes'] = 'not-verified-by-this-contract'
                    data['raw_files'] = [dict(path='results/raw/example.npz',
                                              bytes=999, sha256='f' * 64)]
                if case.startswith('duplicate_') or case == 'two_simultaneous':
                    errors.append('duplicate committed path: payload.txt')
                if case.startswith('total_') or case == 'two_simultaneous':
                    errors.append('committed_total_bytes')
                wanted_stdout = ('ok\n' if not errors else ''.join(
                    'MISMATCH ' + e + '\n' for e in errors) +
                    str(len(errors)) + ' mismatches\n')
                manifest.write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')
                before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
                run = subprocess.run(command + ['--check'], capture_output=True, timeout=10)
                after = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
                entry = dict(optimized=optimized, case=case,
                             expected_exit=int(bool(errors)), expected_stdout=wanted_stdout,
                             actual_exit=run.returncode, stdout=run.stdout.decode(),
                             stderr=run.stderr.decode(), unchanged=before == after)
                entry['pass'] = (run.returncode == int(bool(errors)) and
                                 run.stdout == wanted_stdout.encode() and
                                 not run.stderr and before == after)
                report['cases'].append(entry)
    report['passed'] = sum(r['pass'] for r in report['cases'])
    report['failed'] = len(report['cases']) - report['passed']
    print(json.dumps(report, indent=2))
    return int(report['failed'] != 0)


if __name__ == '__main__':
    sys.exit(main())
