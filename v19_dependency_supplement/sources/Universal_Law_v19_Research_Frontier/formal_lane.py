"""Fixed, byte-pinned scoped Lean evidence. This module grants no UL certificate.

Only profile IDs are accepted. Callers cannot supply Lean code, declarations,
compiler switches, plugins, commands, imports, or alternative trust manifests.
See FORMAL_LANE.md for the trust boundary and portable installation procedure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SOURCE_ROOT = Path(__file__).resolve().parent / 'formal_examples'
PROFILE_ID = 'side24_arithmetic_31'
PROFILE_SHA256 = 'fecc3e7def5cf9eb8fc14de4db3dd944fe3999c10f54e2acf2d6bec398514f38'
PINS_SHA256 = '9c67dad424caffccd46e73403f77b085d92be510ceb31dbb346b93467395a8ff'
SCHEMA = 'ul-v18-scoped-lean-1'
COMMAND_TIMEOUT = 45
OUTPUT_LIMIT = 512 * 1024


class IntegrityError(ValueError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _checked_json(name, expected):
    path = SOURCE_ROOT / name
    if path.is_symlink() or _sha(path) != expected:
        raise IntegrityError(name + ' differs from reviewed bytes')
    return json.loads(path.read_text(encoding='utf8'))


def _profile():
    return _checked_json('profile.json', PROFILE_SHA256)


def _pins():
    return _checked_json('toolchain-pins.json', PINS_SHA256)


def profiles():
    """Return the fixed reviewed profile descriptors, including exact statements."""
    try:
        return [_profile()]
    except (OSError, ValueError) as exc:
        return [{'id': PROFILE_ID, 'state': 'blocked', 'reason': str(exc)}]


def _sources(profile):
    result = {}
    for name, expected in profile['sources'].items():
        path = SOURCE_ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE_ROOT.resolve()):
            raise IntegrityError('source path escaped reviewed source root: ' + name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise IntegrityError('source hash mismatch: ' + name)
        result[name] = data
    expected_audit = (SOURCE_ROOT / 'Audit.expected.txt').read_bytes()
    if hashlib.sha256(expected_audit).hexdigest() != profile['audit_stdout_sha256']:
        raise IntegrityError('reviewed statement output mismatch')
    return result, expected_audit


def _environment(toolchain, project):
    # In particular do not inherit LEAN_PATH, LEAN_SYSROOT, LD_PRELOAD,
    # LD_LIBRARY_PATH, HOME init files, ELAN_TOOLCHAIN, or arbitrary plugins.
    return {'PATH': str(toolchain / 'bin') + os.pathsep + os.defpath,
            'LEAN_PATH': str(project), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
            'HOME': str(project)}


def _child_limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (35, 36))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 ** 2, 16 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _bounded_run(label, args, cwd, env, timeout=COMMAND_TIMEOUT):
    """Run an argv vector with a wall bound, process-group kill and output cap."""
    started = time.monotonic()
    row = {'label': label, 'command': [str(x) for x in args],
           'exit_code': None, 'timed_out': False, 'output_truncated': False,
           'stdout': '', 'stderr': ''}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            child = subprocess.Popen(row['command'], cwd=cwd, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, preexec_fn=_child_limits)
            try:
                row['exit_code'] = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row['timed_out'] = True
                os.killpg(child.pid, signal.SIGKILL)
                row['exit_code'] = child.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            row['stderr'] = type(exc).__name__ + ': ' + str(exc)
        for field, stream in [('stdout', out), ('stderr', err)]:
            stream.seek(0)
            data = stream.read(OUTPUT_LIMIT + 1)
            row['output_truncated'] |= len(data) > OUTPUT_LIMIT
            row[field] += data[:OUTPUT_LIMIT].decode('utf8', errors='replace')
    row['elapsed_seconds'] = round(time.monotonic() - started, 6)
    return row


def _ok(row):
    return row['exit_code'] == 0 and not row['timed_out'] and not row['output_truncated']


def inventory(toolchain=None):
    """Inspect version and actual binary/shared-library/Init bytes; never download."""
    result = {'state': 'blocked', 'configured': False, 'reason_code': 'toolchain_missing',
              'reason': 'Lean toolchain is not configured; set UL_LEAN_TOOLCHAIN or pass toolchain',
              'full_theorem_certificates': []}
    selected = toolchain if toolchain is not None else os.environ.get('UL_LEAN_TOOLCHAIN')
    if not selected:
        return result
    root = Path(selected).expanduser().resolve()
    result['toolchain_root'] = str(root)
    if not (root / 'bin/lean').is_file() or not (root / 'bin/leanchecker').is_file():
        result['reason'] = 'Lean toolchain missing bin/lean or bin/leanchecker at ' + str(root)
        return result
    try:
        pins = _pins()
        # The import closure is Init. Reject extra artifacts in that closure,
        # including files which could override normally absent sidecar files.
        actual_init = {p.relative_to(root).as_posix() for p in (root / 'lib/lean/Init').rglob('*') if p.is_file() and p.name.endswith(('.olean', '.olean.private', '.olean.server', '.ir', '.ir.sig'))}
        actual_init.update(p.relative_to(root).as_posix() for p in (root / 'lib/lean').glob('Init.*') if p.is_file() and p.name.endswith(('.olean', '.olean.private', '.olean.server', '.ir', '.ir.sig')))
        expected_init = {name for name in pins['files'] if name.startswith('lib/lean/Init.') or name.startswith('lib/lean/Init/')}
        if actual_init != expected_init:
            raise IntegrityError('Init import library file inventory mismatch')
        # Reject native-library shadows in the loader's toolchain search roots.
        # The host loader and its system C/POSIX libraries remain host trust.
        allowed_native = {name for name in pins['files'] if name.endswith('.so')}
        # Derive every Lean SONAME from the reviewed pins, including Lake,
        # which is loaded by leanchecker but not by the version probe. The
        # remaining names are the audited DT_NEEDED union of both binaries
        # and all five pinned shared libraries (see native dependency audit).
        native_names = {Path(name).name for name in allowed_native} | {
            'libc.so.6', 'libpthread.so.0', 'libdl.so.2', 'libm.so.6',
            'librt.so.1', 'ld-linux-x86-64.so.2'}
        for directory in [root / 'bin', root / 'lib', root / 'lib/lean']:
            if directory.is_symlink():
                raise IntegrityError('loader search directory is a symlink: ' + directory.relative_to(root).as_posix())
            for path in directory.rglob('*'):
                # Path.rglob does not descend symlinked directories, whereas
                # the loader follows glibc-hwcaps directories. Reject them
                # before any probe can execute an uninspected native object.
                if path.is_symlink() and path.is_dir():
                    raise IntegrityError('loader search subtree is a symlink: ' + path.relative_to(root).as_posix())
                if path.name in native_names and (path.is_file() or path.is_symlink()):
                    relative = path.relative_to(root).as_posix()
                    if relative not in allowed_native:
                        raise IntegrityError('unexpected loader shadow: ' + relative)
        for name, expected in pins['files'].items():
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise IntegrityError('toolchain pinned file is a symlink or escapes root: ' + name)
            if path.stat().st_size != expected['bytes'] or _sha(path) != expected['sha256']:
                raise IntegrityError('toolchain pinned file differs: ' + name)
        with tempfile.TemporaryDirectory(prefix='ul-lean-version-') as temp:
            version = _bounded_run('pinned_version', [root / 'bin/lean', '--version'], Path(temp), _environment(root, Path(temp)))
        if not _ok(version) or version['stdout'].strip() != pins['lean_version']:
            raise IntegrityError('pinned Lean version mismatch or version process failed')
        result.update(state='configured', configured=True, reason_code=None, reason=None,
            toolchain_version=pins['lean_version'], archive_sha256=pins['archive_sha256'],
            pins_sha256=PINS_SHA256, pinned_files=len(pins['files']),
            pinned_bytes=sum(x['bytes'] for x in pins['files'].values()),
            pin_scope='compiler, standalone checker, five Lean shared libraries and complete Init import artifact closure',
            host_trust='Linux kernel, system dynamic loader and system C/POSIX libraries are outside these Lean byte pins')
    except (OSError, ValueError) as exc:
        result.update(reason_code='toolchain_integrity_mismatch', reason=str(exc))
    return result


def _blocked(profile_id, code, reason):
    return {'schema': SCHEMA, 'profile_id': profile_id if isinstance(profile_id, str) else None,
            'status': 'blocked', 'reason_code': code, 'reason': reason,
            'commands': [], 'full_theorem_certificates': []}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def _artifacts(project):
    return {path.relative_to(project).as_posix(): {'sha256': _sha(path), 'bytes': path.stat().st_size}
            for path in sorted(project.rglob('*')) if path.is_file()}


def run_profile(profile_id, output_dir, toolchain=None):
    """Execute a fresh fixed profile; blocked configuration creates no receipt.

    A successful return includes artifact_dir and receipt_path. Failures retain
    attempt.json and diagnostics, but never produce a successful receipt.json.
    """
    if not isinstance(profile_id, str) or profile_id != PROFILE_ID:
        return _blocked(profile_id, 'unknown_profile', 'Only a registered reviewed profile ID is accepted')
    try:
        profile = _profile()
        sources, expected_audit = _sources(profile)
    except (OSError, ValueError) as exc:
        return _blocked(profile_id, 'source_integrity_mismatch', str(exc))
    tool = inventory(toolchain)
    if tool['state'] != 'configured':
        return _blocked(profile_id, tool['reason_code'], tool['reason'])
    root = Path(tool['toolchain_root'])
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=PROFILE_ID + '-', dir=destination))
    project = run_dir / 'project'
    project.mkdir()
    for name, data in sources.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    audit = 'import UniversalLaw\n' + ''.join('#print axioms ' + row['declaration'] + '\n#check ' + row['declaration'] + '\n' for row in profile['statements'])
    (project / 'Audit.lean').write_text(audit, encoding='utf8')
    negative_source = sources['UniversalLaw/Side24/Endpoints.lean'].decode('utf8').replace(
        'd2Lower < d2Upper ∧ d2Upper = d2Lower + 1', 'd2Upper < d2Lower ∧ d2Upper = d2Lower + 1')
    (project / 'FalseEndpoint.lean').write_text(negative_source, encoding='utf8')
    receipt = {'schema': SCHEMA, 'profile_id': profile_id, 'profile_sha256': PROFILE_SHA256,
               'status': 'failed', 'scope': profile['scope'],
               'alignment_status': profile['alignment_status'], 'statements': profile['statements'],
               'source_hashes': profile['sources'], 'toolchain': tool,
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'artifact_dir': str(project), 'commands': [], 'full_theorem_certificates': []}
    env = _environment(root, project)
    flags = ['-DwarningAsError=true', '-M1024', '-j1', '-T2000000']
    def execute(label, executable, args):
        row = _bounded_run(label, [root / 'bin' / executable, *args], project, env)
        receipt['commands'].append(row)
        return row
    try:
        for module in profile['modules']:
            name = module.replace('.', '/')
            row = execute('elaborate:' + module, 'lean', [*flags, '-o', name + '.olean', name + '.lean'])
            if not _ok(row):
                raise IntegrityError('compiler rejected reviewed source or exceeded bound: ' + module)
        row = execute('axioms_and_formal_types', 'lean', [*flags, 'Audit.lean'])
        if not _ok(row) or row['stdout'].encode() != expected_audit or row['stderr']:
            raise IntegrityError('axiom audit or exact declaration/statement binding mismatch')
        row = execute('kernel_replay', 'leanchecker', ['-v', *profile['modules']])
        if not _ok(row) or any('replaying ' + name not in row['stdout'] for name in profile['modules']):
            raise IntegrityError('standalone kernel replay failed or exceeded bound')
        row = execute('false_endpoint_negative_control', 'lean', [*flags, 'FalseEndpoint.lean'])
        if row['exit_code'] in (None, 0) or row['timed_out'] or row['output_truncated'] or 'is false' not in row['stdout']:
            raise IntegrityError('false proposition was not rejected for mathematical falsity')
        # Validate again after execution to catch ordinary concurrent mutation.
        if inventory(root)['state'] != 'configured':
            raise IntegrityError('toolchain changed during execution')
        _sources(profile)
        receipt.update(status='scoped_kernel_checked', axiom_free_count=31,
                       false_proposition_rejected=True, reason=None, reason_code=None)
    except (OSError, ValueError) as exc:
        receipt.update(reason=str(exc), reason_code='execution_failed')
    receipt['artifacts'] = _artifacts(project)
    receipt['receipt_sha256'] = _digest(receipt)
    if receipt['status'] == 'scoped_kernel_checked':
        receipt_path = run_dir / 'receipt.json'
    else:
        receipt_path = run_dir / 'attempt.json'
    # receipt_path is a convenience locator and excluded from the content digest.
    receipt['receipt_path'] = str(receipt_path)
    _write_json(receipt_path, receipt)
    return receipt


def verify_receipt(receipt_or_path, toolchain=None):
    """Validate custody and replay current pinned Lean before returning valid=True.

    A hash is not a signature: untrusted receipt claims alone are never enough.
    Paths may be relocated with the whole receipt run directory. No command in
    the submitted receipt is executed; replay is generated from the fixed ID.
    """
    reasons = []
    try:
        if isinstance(receipt_or_path, (str, Path)):
            receipt_path = Path(receipt_or_path).resolve()
            if receipt_path.stat().st_size > 2 * 1024 * 1024:
                raise IntegrityError('receipt exceeds input bound')
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            project = receipt_path.parent / 'project'
        else:
            receipt = json.loads(json.dumps(receipt_or_path))
            project = Path(receipt.get('artifact_dir', '/nonexistent-ul-formal-receipt'))
        if not isinstance(receipt, dict):
            raise IntegrityError('receipt must be an object')
        claimed = receipt.pop('receipt_sha256', None)
        receipt.pop('receipt_path', None)
        if claimed != _digest(receipt):
            raise IntegrityError('receipt content digest mismatch (stale or tampered receipt)')
        profile = _profile()
        _sources(profile)
        expected_fields = {'schema': SCHEMA, 'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256,
            'status': 'scoped_kernel_checked', 'scope': profile['scope'],
            'alignment_status': profile['alignment_status'], 'statements': profile['statements'],
            'source_hashes': profile['sources'], 'axiom_free_count': 31,
            'false_proposition_rejected': True, 'full_theorem_certificates': []}
        for key, expected in expected_fields.items():
            if receipt.get(key) != expected:
                raise IntegrityError('receipt reviewed binding mismatch: ' + key)
        commands = receipt.get('commands')
        labels = ['elaborate:' + module for module in profile['modules']] + [
            'axioms_and_formal_types', 'kernel_replay', 'false_endpoint_negative_control']
        if not isinstance(commands, list) or len(commands) != len(labels):
            raise IntegrityError('recorded command inventory mismatch')
        for row, label in zip(commands, labels):
            if not isinstance(row, dict) or row.get('label') != label:
                raise IntegrityError('recorded command label mismatch')
            argv = row.get('command')
            if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
                raise IntegrityError('malformed recorded command vector')
        artifact_names = set(profile['sources']) | {'Audit.lean', 'FalseEndpoint.lean'} | {
            module.replace('.', '/') + '.olean' for module in profile['modules']}
        if set(receipt['artifacts']) != artifact_names:
            raise IntegrityError('artifact inventory differs from fixed profile')
        for name, expected in profile['sources'].items():
            if receipt['artifacts'][name].get('sha256') != expected:
                raise IntegrityError('artifact source differs from reviewed source: ' + name)
        if not project.is_dir() or project.is_symlink():
            raise IntegrityError('artifact directory missing or symlinked')
        for name, metadata in receipt['artifacts'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise IntegrityError('artifact path escaped receipt directory')
            path = project / relative
            if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
                raise IntegrityError('artifact path escaped receipt directory')
            if path.stat().st_size != metadata['bytes'] or _sha(path) != metadata['sha256']:
                raise IntegrityError('artifact integrity mismatch: ' + name)
        if _artifacts(project) != receipt['artifacts']:
            raise IntegrityError('artifact file inventory mismatch')
        tool = inventory(toolchain)
        if tool['state'] != 'configured':
            return {'valid': False, 'state': 'blocked', 'reasons': [tool['reason']],
                    'artifact_integrity': True, 'full_theorem_certificates': []}
        for key in ['toolchain_version', 'archive_sha256', 'pins_sha256', 'pinned_files', 'pinned_bytes']:
            if receipt['toolchain'].get(key) != tool[key]:
                raise IntegrityError('stale toolchain receipt binding: ' + key)
        # Independently build/check in a fresh private workspace. Never execute
        # uploaded .olean bytes or any command vector recorded in the receipt.
        with tempfile.TemporaryDirectory(prefix='ul-formal-verify-') as temp:
            replay = run_profile(PROFILE_ID, temp, toolchain=tool['toolchain_root'])
        if replay['status'] != 'scoped_kernel_checked':
            raise IntegrityError('current pinned replay did not pass: ' + str(replay.get('reason')))
        if replay['artifacts'] != receipt['artifacts']:
            raise IntegrityError('artifact differs from independently compiled fixed profile')
        def outcomes(commands):
            result = []
            for row in commands:
                item = {key: row.get(key) for key in ['label', 'exit_code', 'timed_out', 'output_truncated', 'stdout', 'stderr']}
                command = row['command']
                item['command'] = [Path(command[0]).name, *command[1:]]
                result.append(item)
            return result
        if outcomes(receipt['commands']) != outcomes(replay['commands']):
            raise IntegrityError('recorded command outcomes differ from fresh replay')
        return {'valid': True, 'state': 'scoped_kernel_checked', 'reasons': [],
                'artifact_integrity': True, 'fresh_replay': True, 'full_theorem_certificates': []}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        reasons.append(str(exc))
    return {'valid': False, 'state': 'invalid', 'reasons': reasons, 'full_theorem_certificates': []}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'run', 'verify'])
    parser.add_argument('--toolchain')
    parser.add_argument('--output', default='examples/formal/runs')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if args.action == 'inventory':
        result = inventory(args.toolchain)
    elif args.action == 'run':
        result = run_profile(PROFILE_ID, args.output, args.toolchain)
    else:
        if not args.receipt:
            parser.error('verify requires --receipt')
        result = verify_receipt(args.receipt, args.toolchain)
    print(json.dumps(result, indent=2, ensure_ascii=False))
