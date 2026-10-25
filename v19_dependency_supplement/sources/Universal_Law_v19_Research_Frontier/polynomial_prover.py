"""Bounded, fixed-goal Lean search and independent scalar-proof receipt replay.

This additive lane preserves formal_lane and its toolchain pins. Only reviewed
sources are compiled. Receipt commands and submitted artifacts are never executed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import types

import formal_lane as _formal_lane
from formal_lane import inventory, _bounded_run, _environment, _ok

SOURCE_ROOT = Path(__file__).resolve().parent / 'formal_frontier'
PROFILE_ID = 'scalar_variance_bounds_v1'
PROFILE_SHA256 = '94ef392a06467dfd19a1b62f38738068d47bd3b7399883ee456387bf71b8c228'
SCHEMA = 'ul-v19-scalar-proof-receipt-1'
HELPER_SOURCE_SHA256 = 'd037c1ed52aa6da44ba12911d8d6b49c0a397ffa7b6841a3e66ccff785990008'
MAX_FILE_BYTES = 2 * 1024**2
MAX_ARTIFACT_BYTES = 16 * 1024**2
MAX_ARTIFACT_FILES = 16
LIMITS = {'wall_seconds_per_command': 45, 'cpu_soft_seconds': 35,
          'cpu_hard_seconds': 36, 'address_space_bytes': 8 * 1024**3,
          'file_size_bytes': 16 * 1024**2, 'output_bytes_per_stream': 512 * 1024,
          'compiler_memory_megabytes': 1024, 'compiler_heartbeats': 2000000,
          'compiler_threads': 1, 'maximum_attempts': 2}
FLAGS = ['-DwarningAsError=true', '-M1024', '-j1', '-T2000000']
LABELS = ['attempt:decide_without_parameter_elimination',
          'attempt:monotonicity_and_distributivity',
          'axioms_and_formal_types', 'kernel_replay', 'false_proposition_negative_control']


class IntegrityError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@contextmanager
def _directory_fd(path):
    """Open each ancestor relative to an already-open directory, never a symlink."""
    absolute = Path(path).absolute()
    if '..' in absolute.parts:
        raise IntegrityError('parent traversal in input path')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                      dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _regular_stream(name, directory_fd, limit=MAX_FILE_BYTES, expected=None):
    # O_NONBLOCK prevents a racing FIFO replacement from blocking open().
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise IntegrityError('input is not a bounded regular file: ' + str(name))
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if expected is not None and identity(before) != identity(expected):
            raise IntegrityError('artifact changed after inventory: ' + str(name))
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            yield stream, before
        if identity(os.fstat(descriptor)) != identity(before):
            raise IntegrityError('input changed while reading: ' + str(name))
    finally:
        os.close(descriptor)


def _read_regular(path, limit=MAX_FILE_BYTES):
    path = Path(path).absolute()
    with _directory_fd(path.parent) as parent:
        with _regular_stream(path.name, parent, limit) as (stream, info):
            data = stream.read(limit + 1)
            if len(data) > limit or len(data) != info.st_size:
                raise IntegrityError('input exceeds bound or changed size: ' + path.name)
            return data


def _stream_digest(stream, size, limit=MAX_FILE_BYTES):
    digest = hashlib.sha256()
    total = 0
    while True:
        block = stream.read(min(65536, limit - total + 1))
        if not block:
            break
        total += len(block)
        if total > limit:
            raise IntegrityError('input grew beyond byte bound')
        digest.update(block)
    if total != size:
        raise IntegrityError('input changed size while hashing')
    return digest.hexdigest()


def _sha(path):
    path = Path(path).absolute()
    with _directory_fd(path.parent) as parent:
        with _regular_stream(path.name, parent) as (stream, info):
            return _stream_digest(stream, info.st_size)


def _implementation_value(value):
    """Serialize semantics, avoiding marshal's runtime-dependent intern flags."""
    if isinstance(value, types.CodeType):
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
                  'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names',
                  'co_varnames', 'co_freevars', 'co_cellvars', 'co_name', 'co_qualname',
                  'co_firstlineno', 'co_linetable', 'co_exceptiontable')
        return {'code': {name: _implementation_value(getattr(value, name)) for name in fields}}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, tuple):
        return {'tuple': [_implementation_value(item) for item in value]}
    if isinstance(value, frozenset):
        return {'frozenset': sorted((_implementation_value(item) for item in value), key=_canonical)}
    if isinstance(value, dict):
        return {key: _implementation_value(item) for key, item in value.items()}
    if value is Ellipsis:
        return {'ellipsis': True}
    if type(value) not in (str, int, bool, float, type(None)):
        raise IntegrityError('unexpected native helper implementation constant type')
    return value


def _helper_bindings():
    """Pin source AND check loaded helper functions/defaults/globals against it.

    Only the exact reviewed local helper bytes may be compiled into this audit
    namespace. No submitted Python/Lean source is evaluated by this operation.
    """
    helper_path = Path(_formal_lane.__file__).absolute()
    data = _read_regular(helper_path)
    if hashlib.sha256(data).hexdigest() != HELPER_SOURCE_SHA256:
        raise IntegrityError('native helper source differs from reviewed bytes')
    trusted = {'__name__': 'formal_lane', '__file__': str(helper_path)}
    exec(compile(data, str(helper_path), 'exec'), trusted)
    identities = {}
    for name, expected in trusted.items():
        if isinstance(expected, types.FunctionType) and expected.__module__ == 'formal_lane':
            actual = getattr(_formal_lane, name, None)
            if not isinstance(actual, types.FunctionType):
                raise IntegrityError('loaded native helper function differs: ' + name)
            def fingerprint(function):
                return _digest(_implementation_value((function.__code__,
                    function.__defaults__, function.__kwdefaults__)))
            if fingerprint(actual) != fingerprint(expected) or actual.__globals__ is not vars(_formal_lane):
                raise IntegrityError('loaded native helper implementation differs: ' + name)
            identities[name] = fingerprint(actual)
        elif not name.startswith('__') and (type(expected) in (str, int, float, bool) or isinstance(expected, Path)):
            if getattr(_formal_lane, name, None) != expected:
                raise IntegrityError('loaded native helper constant differs: ' + name)
    for name in ('inventory', '_bounded_run', '_environment', '_ok'):
        if globals()[name] is not getattr(_formal_lane, name):
            raise IntegrityError('worker native helper alias differs: ' + name)
    return {'formal_lane_sha256': HELPER_SOURCE_SHA256,
            'loaded_implementation_sha256': _digest(identities),
            'python_implementation': sys.implementation.name,
            'python_version': list(sys.version_info[:3])}


def _profile_and_sources():
    path = SOURCE_ROOT / 'profile.json'
    profile_bytes = _read_regular(path)
    if hashlib.sha256(profile_bytes).hexdigest() != PROFILE_SHA256:
        raise IntegrityError('fixed profile bytes differ')
    profile = json.loads(profile_bytes)
    sources = {}
    for name, digest in profile['sources'].items():
        path = SOURCE_ROOT / name
        if Path(name).name != name:
            raise IntegrityError('source escaped reviewed root')
        data = _read_regular(path)
        if hashlib.sha256(data).hexdigest() != digest:
            raise IntegrityError('source hash mismatch: ' + name)
        sources[name] = data
    # Both attempts must ask exactly the same question, independent of strategy.
    def headers(data):
        import re
        return re.findall(r'(?m)^theorem\s+(\w+)\s+([\s\S]*?) := by', data.decode())
    if headers(sources['AttemptDecide.lean']) != headers(sources['VarianceBounds.lean']):
        raise IntegrityError('frozen proof goals differ across strategies')
    return profile, sources


def _artifacts(project, expected_names=None, source_bytes=None):
    """Preflight the complete flat inventory and size budget BEFORE reading files."""
    with _directory_fd(project) as directory:
        entries = {}
        total = 0
        with os.scandir(directory) as children:
            for entry in children:
                if len(entries) >= MAX_ARTIFACT_FILES:
                    raise IntegrityError('artifact file count exceeds bound')
                if expected_names is not None and entry.name not in expected_names:
                    raise IntegrityError('unexpected artifact entry: ' + entry.name)
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                    raise IntegrityError('artifact is not a bounded regular file: ' + entry.name)
                if source_bytes is not None and entry.name in source_bytes and info.st_size != len(source_bytes[entry.name]):
                    raise IntegrityError('artifact source size mismatch: ' + entry.name)
                total += info.st_size
                if total > MAX_ARTIFACT_BYTES:
                    raise IntegrityError('aggregate artifact byte bound exceeded')
                entries[entry.name] = info
        if expected_names is not None and set(entries) != set(expected_names):
            raise IntegrityError('fixed artifact inventory mismatch')
        result = {}
        for name, info in sorted(entries.items()):
            with _regular_stream(name, directory, expected=info) as (stream, actual):
                digest = _stream_digest(stream, actual.st_size)
            if source_bytes is not None and name in source_bytes and digest != hashlib.sha256(source_bytes[name]).hexdigest():
                raise IntegrityError('artifact source differs from reviewed bytes: ' + name)
            result[name] = {'sha256': digest, 'bytes': info.st_size}
        return result


def profile_contract():
    """Return the pinned scalar question contract; raises on changed source bytes.

    This descriptor is suitable for conductor registration, never for closing an
    actual-field D5 obligation. Actual success still requires fresh native replay.
    """
    profile, _ = _profile_and_sources()
    helpers = _helper_bindings()
    from formal_lane import PINS_SHA256
    return {'schema': 'ul-v19-scalar-contract-1', 'profile_id': PROFILE_ID,
        'profile_sha256': PROFILE_SHA256, 'scope': profile['scope'],
        'alignment_status': profile['alignment_status'], 'statements': profile['statements'],
        'source_hashes': profile['sources'], 'allowed_axioms': profile['allowed_axioms'],
        'checker': 'polynomial_prover.verify_proof_receipt',
        'checker_bindings': {**helpers, 'worker_sha256': _sha(__file__), 'receipt_schema': SCHEMA,
                            'toolchain_pins_sha256': PINS_SHA256},
        'toolchain_pins_sha256': PINS_SHA256, 'requires_fresh_replay': True,
        'actual_field_applicable': False, 'full_theorem_certificates': []}


def _blocked(code, reason):
    return {'schema': SCHEMA, 'profile_id': PROFILE_ID, 'status': 'blocked',
            'reason_code': code, 'reason': reason, 'commands': [], 'attempts': [],
            'full_theorem_certificates': []}


def _selected_toolchain(toolchain):
    if toolchain is not None:
        return toolchain
    from toolchain_setup import default_toolchain
    return default_toolchain()


def prove_profile(output_dir, toolchain=None):
    """Execute two frozen approaches, audit types/axioms, kernel, and false control.

    No source, profile, import, flag or command argument is accepted from callers.
    A blocked configuration creates no receipt; execution failures retain attempt.json.
    """
    try:
        profile, sources = _profile_and_sources()
    except (OSError, ValueError, KeyError) as exc:
        return _blocked('source_integrity_mismatch', str(exc))
    try:
        helpers = _helper_bindings()
    except (OSError, ValueError) as exc:
        return _blocked('helper_integrity_mismatch', str(exc))
    tool = inventory(_selected_toolchain(toolchain))
    try:
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper binding changed during inventory')
    except (OSError, ValueError) as exc:
        return _blocked('helper_integrity_mismatch', str(exc))
    if tool['state'] != 'configured':
        return _blocked(tool['reason_code'], tool['reason'])
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=PROFILE_ID + '-', dir=destination))
    project = run_dir / 'project'
    project.mkdir()
    for name, data in sources.items():
        (project / name).write_bytes(data)
    root = Path(tool['toolchain_root'])
    env = _environment(root, project)
    receipt = {'schema': SCHEMA, 'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256,
        'status': 'failed', 'scope': profile['scope'], 'alignment_status': profile['alignment_status'],
        'worker_sha256': _sha(__file__),
        'helper_bindings': helpers,
        'statements': profile['statements'], 'source_hashes': profile['sources'],
        'toolchain': tool, 'resource_limits': dict(LIMITS), 'commands': [], 'attempts': [],
        'artifact_dir': str(project), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'actual_field_applicable': False, 'full_theorem_certificates': []}

    def execute(label, executable, args):
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper changed before execution')
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        row = _bounded_run(label, [root / 'bin' / executable, *args], project, env)
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper changed during execution')
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        row['resources'] = {'user_cpu_seconds': max(0, after.ru_utime - before.ru_utime),
            'system_cpu_seconds': max(0, after.ru_stime - before.ru_stime),
            'children_maxrss_kib': after.ru_maxrss,
            'maxrss_scope': 'process child high-water mark; cumulative across calls'}
        receipt['commands'].append(row)
        return row

    try:
        first = execute(LABELS[0], 'lean', [*FLAGS, 'AttemptDecide.lean'])
        receipt['attempts'].append({'strategy': profile['attempt_grammar'][0],
            'exit_code': first['exit_code'], 'goal_sha256': _digest(profile['statements']),
            'diagnosis': 'decision procedure does not eliminate free parameters'})
        if first['exit_code'] in (None, 0) or first['timed_out'] or first['output_truncated'] or 'Expected type must not contain free variables' not in first['stdout']:
            raise IntegrityError('first genuine strategy did not fail for the expected reason')
        second = execute(LABELS[1], 'lean', [*FLAGS, '-o', 'VarianceBounds.olean', 'VarianceBounds.lean'])
        receipt['attempts'].append({'strategy': profile['attempt_grammar'][1],
            'exit_code': second['exit_code'], 'goal_sha256': _digest(profile['statements']),
            'diagnosis': 'explicit multiplication monotonicity and arithmetic equality proof terms'})
        if not _ok(second) or second['stdout'] or second['stderr']:
            raise IntegrityError('universal scalar proof failed compilation')
        row = execute(LABELS[2], 'lean', [*FLAGS, 'Audit.lean'])
        if not _ok(row) or row['stdout'].encode() != sources['Audit.expected.txt'] or row['stderr']:
            raise IntegrityError('exact axiom or declaration type audit mismatch')
        if row['stdout'].count('does not depend on any axioms') != 2:
            raise IntegrityError('required axiom-free declarations absent')
        row = execute(LABELS[3], 'leanchecker', ['-v', 'VarianceBounds'])
        if not _ok(row) or row['stdout'] != 'replaying VarianceBounds\n' or row['stderr']:
            raise IntegrityError('standalone kernel rejected fresh module')
        row = execute(LABELS[4], 'lean', [*FLAGS, 'FalseControl.lean'])
        if row['exit_code'] in (None, 0) or row['timed_out'] or row['output_truncated'] or 'is false' not in row['stdout']:
            raise IntegrityError('false numerical specialization was not rejected')
        _profile_and_sources()
        if inventory(root)['state'] != 'configured':
            raise IntegrityError('pinned toolchain changed during work')
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper changed after work')
        receipt.update(status='scoped_kernel_checked', axiom_free_count=2,
                       false_proposition_rejected=True, reason=None, reason_code=None)
    except (OSError, ValueError) as exc:
        receipt.update(reason_code='execution_failed', reason=str(exc))
    receipt['artifacts'] = _artifacts(project)
    receipt['receipt_sha256'] = _digest(receipt)
    path = run_dir / ('receipt.json' if receipt['status'] == 'scoped_kernel_checked' else 'attempt.json')
    receipt['receipt_path'] = str(path)
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_proof_receipt(path, toolchain=None):
    """Recheck custody and independently recompile reviewed sources in a new directory.

    A self-consistent hash cannot certify a theorem. Never load submitted .olean
    files or execute receipt command vectors; compare them against fresh fixed work.
    """
    failure = {'valid': False, 'state': 'invalid', 'reasons': [], 'fresh_replay': False,
               'full_theorem_certificates': []}
    try:
        receipt_path = Path(path).absolute()
        receipt = json.loads(_read_regular(receipt_path))
        if not isinstance(receipt, dict):
            raise IntegrityError('receipt must be an object')
        claimed = receipt.pop('receipt_sha256', None)
        receipt.pop('receipt_path', None)
        if claimed != _digest(receipt):
            raise IntegrityError('receipt digest mismatch')
        profile, sources = _profile_and_sources()
        helpers = _helper_bindings()
        expected = {'schema': SCHEMA, 'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256,
            'status': 'scoped_kernel_checked', 'scope': profile['scope'],
            'worker_sha256': _sha(__file__),
            'helper_bindings': helpers,
            'alignment_status': profile['alignment_status'], 'statements': profile['statements'],
            'source_hashes': profile['sources'], 'resource_limits': LIMITS,
            'axiom_free_count': 2, 'false_proposition_rejected': True,
            'actual_field_applicable': False, 'full_theorem_certificates': []}
        for key, value in expected.items():
            if _canonical(receipt.get(key)) != _canonical(value):
                raise IntegrityError('reviewed receipt binding mismatch: ' + key)
        commands = receipt['commands']
        if not isinstance(commands, list) or [r.get('label') for r in commands] != LABELS:
            raise IntegrityError('fixed command inventory mismatch')
        for row in commands:
            if not isinstance(row['command'], list) or not row['command'] or not all(isinstance(x, str) for x in row['command']):
                raise IntegrityError('invalid recorded command')
            elapsed = row.get('elapsed_seconds')
            if type(elapsed) not in (int, float) or not 0 <= elapsed <= 50 or not math.isfinite(elapsed):
                raise IntegrityError('invalid command resource record')
        project = receipt_path.parent / 'project'
        names = set(sources) | {'VarianceBounds.olean'}
        if set(receipt['artifacts']) != names:
            raise IntegrityError('fixed artifact inventory mismatch')
        if _artifacts(project, names, sources) != receipt['artifacts']:
            raise IntegrityError('artifact hash/inventory mismatch')
        tool = inventory(_selected_toolchain(toolchain))
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper changed during verification inventory')
        if tool['state'] != 'configured':
            failure.update(state='blocked', reasons=[tool['reason']])
            return failure
        for key in ['toolchain_version', 'archive_sha256', 'pins_sha256', 'pinned_files', 'pinned_bytes']:
            if receipt['toolchain'].get(key) != tool[key]:
                raise IntegrityError('pinned toolchain receipt mismatch: ' + key)
        with tempfile.TemporaryDirectory(prefix='ul-v19-scalar-replay-') as temp:
            replay = prove_profile(temp, toolchain=tool['toolchain_root'])
        if _helper_bindings() != helpers:
            raise IntegrityError('native helper changed during independent replay')
        if replay['status'] != 'scoped_kernel_checked':
            raise IntegrityError('fresh native proof replay failed: ' + str(replay.get('reason')))
        if replay['artifacts'] != receipt['artifacts']:
            raise IntegrityError('artifact differs from independently compiled fixed profile')
        def outcomes(rows):
            result = []
            for row in rows:
                item = {key: row[key] for key in ['label', 'exit_code', 'timed_out', 'output_truncated', 'stdout', 'stderr']}
                item['command'] = [Path(row['command'][0]).name, *row['command'][1:]]
                result.append(item)
            return result
        if outcomes(commands) != outcomes(replay['commands']) or receipt['attempts'] != replay['attempts']:
            raise IntegrityError('recorded strategy/command outcomes differ from fresh execution')
        return {'valid': True, 'state': 'scoped_kernel_checked', 'reasons': [],
                'fresh_replay': True, 'artifact_integrity': True, 'axiom_free_count': 2,
                'actual_field_applicable': False, 'full_theorem_certificates': []}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        failure['reasons'].append(str(exc))
    return failure


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prove', 'verify'])
    parser.add_argument('--output', default='examples/frontier_math/runs')
    parser.add_argument('--receipt')
    parser.add_argument('--toolchain')
    args = parser.parse_args()
    if args.action == 'verify' and not args.receipt:
        parser.error('verify requires --receipt')
    result = prove_profile(args.output, args.toolchain) if args.action == 'prove' else verify_proof_receipt(args.receipt, args.toolchain)
    print(json.dumps(result, indent=2))
