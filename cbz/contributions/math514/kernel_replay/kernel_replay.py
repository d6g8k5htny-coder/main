#!/usr/bin/env python3
"""Bounded, serial, exact-module adapter to the installed LeanChecker routine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parent
SCHEMA = 'cbz-kernel-replay/v1'
# Linux poll uses signed-int milliseconds. Floor before passing seconds to any
# platform wait, and also respect CPython's documented blocking-time ceiling.
MAX_TIMEOUT_SECONDS = min(int(threading.TIMEOUT_MAX), ((1 << 31) - 1) // 1000)
CLEANUP_TIMEOUT_SECONDS = 1.0
PROCESS_ERRORS = (OSError, subprocess.SubprocessError, OverflowError, ValueError)
MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError(f'not a regular file: {p}')
    return {'path': str(p), 'sha256': sha256(p)}


def reject_artifact_symlinks(path):
    p = Path(path)
    # Do not canonicalize one filename and then inventory destination siblings:
    # upstream finds .server/.private siblings of the lexical selected filename.
    # Conservatively reject symlinks in every artifact/search-root component.
    try:
        canonical = p.resolve()
    except RuntimeError as e:
        raise ValueError(f'symlink loop: {p}') from e
    if not p.is_absolute() or p.is_symlink() or canonical != p:
        raise ValueError(f'symlink or noncanonical artifact/search path forbidden: {p}')


def exact_keys(obj, keys, what):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f'{what}: required exact fields {sorted(keys)}')


def check_binding(b):
    exact_keys(b, {'path', 'sha256'}, 'binding')
    p = Path(b['path'])
    if not p.is_absolute() or not re.fullmatch('[0-9a-f]{64}', b['sha256']):
        raise ValueError('binding requires absolute path and SHA-256')
    if binding(p) != b:
        raise ValueError(f'binding changed: {p}')


def tool_bindings(sysroot):
    upstream = Path(sysroot) / 'src/lean'
    library = Path(sysroot) / 'lib/lean'
    paths = [ROOT / 'kernel_replay.py', ROOT / 'ExactReplay.lean',
             upstream / 'LeanChecker.lean', upstream / 'Lean/Replay.lean',
             library / 'LeanChecker.olean', library / 'Lean/Replay.olean']
    # Bind any available data parts of both imported checker implementations.
    for stem in (library / 'LeanChecker.olean', library / 'Lean/Replay.olean'):
        paths.extend(p for p in (Path(str(stem) + '.server'), Path(str(stem) + '.private'))
                     if p.exists())
    return [binding(p) for p in paths]


def resolve_olean(name, lean_path, sysroot):
    if not isinstance(name, str) or not MODULE.fullmatch(name):
        raise ValueError(f'invalid exact module: {name!r}')
    package = name.split('.')[0]
    # Mirrors SearchPath.findWithExt: package shadowing is significant, even
    # when a matching later path contains the requested submodule.
    for raw in [str(Path(sysroot) / 'lib/lean'), *lean_path]:
        root = Path(raw)
        if (root / package).is_dir() or (root / (package + '.olean')).exists():
            p = root.joinpath(*name.split('.')).with_suffix('.olean')
            reject_artifact_symlinks(p)
            if not p.is_file():
                raise FileNotFoundError(f'exact module missing: {p}')
            return p.resolve()
    raise FileNotFoundError(f'exact module not found: {name}')


def parts_of(olean):
    parts = [Path(olean)]
    server, private = Path(str(olean) + '.server'), Path(str(olean) + '.private')
    for p in (Path(olean), server, private):
        reject_artifact_symlinks(p)
    if server.exists():
        parts.append(server)
        if private.exists():
            parts.append(private)
    elif private.exists():
        raise ValueError(f'orphan private part: {private}')
    return [binding(p) for p in parts]


def prepare_manifest(lean, lean_path, selections, context):
    lean = str(Path(lean).resolve(strict=True))
    sysroot = str(Path(lean).parent.parent)
    if not selections:
        raise ValueError('empty expected module coverage forbidden')
    paths = []
    for raw in lean_path:
        p = Path(raw).absolute()
        reject_artifact_symlinks(p)
        paths.append(str(p.resolve(strict=True)))
    if any(not Path(p).is_dir() for p in paths):
        raise ValueError('lean_path entries must be directories')
    names, modules = [], []
    for selection in selections:
        if '=' not in selection:
            raise ValueError('module selection must be NAME=SOURCE')
        name, source = selection.split('=', 1)
        if name in names:
            raise ValueError(f'duplicate module: {name}')
        olean = resolve_olean(name, paths, sysroot)
        names.append(name)
        modules.append({'name': name, 'source': binding(source), 'parts': parts_of(olean)})
    return {'schema': SCHEMA, 'lean': binding(lean), 'sysroot': sysroot,
            'lean_path': paths, 'expected_modules': names, 'modules': modules,
            'context': [binding(p) for p in context], 'tool': tool_bindings(sysroot)}


def validate_manifest(m):
    exact_keys(m, {'schema', 'lean', 'sysroot', 'lean_path', 'expected_modules',
                   'modules', 'context', 'tool'}, 'manifest')
    if m['schema'] != SCHEMA:
        raise ValueError('unsupported schema')
    names = m['expected_modules']
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError('empty or invalid expected coverage')
    if len(set(names)) != len(names):
        raise ValueError('duplicate expected coverage')
    if (not isinstance(m['modules'], list) or any(not isinstance(x, dict) for x in m['modules'])
            or [x.get('name') for x in m['modules']] != names):
        raise ValueError('expected module coverage mismatch')
    check_binding(m['lean'])
    if not Path(m['sysroot']).is_absolute():
        raise ValueError('sysroot must be absolute')
    reject_artifact_symlinks(Path(m['sysroot']))
    if not isinstance(m['lean_path'], list) or any(
            not Path(p).is_absolute() or not Path(p).is_dir() for p in m['lean_path']):
        raise ValueError('invalid lean_path')
    for p in m['lean_path']:
        reject_artifact_symlinks(Path(p))
    for entry in m['modules']:
        exact_keys(entry, {'name', 'source', 'parts'}, 'module')
        olean = resolve_olean(entry['name'], m['lean_path'], m['sysroot'])
        check_binding(entry['source'])
        if parts_of(olean) != entry['parts']:
            raise ValueError(f'compiled parts changed: {entry["name"]}')
    if m['tool'] != tool_bindings(m['sysroot']):
        raise ValueError('tool binding changed')
    if not isinstance(m['context'], list):
        raise ValueError('context must be a binding list')
    for b in m['context']:
        check_binding(b)


def limits(memory_mib):
    def apply():
        cap = memory_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def cleanup_child(p):
    """Kill the process group and reap without an implicit/unbounded wait."""
    cleanup = {'pid': p.pid, 'reaped': False, 'timeout_seconds': CLEANUP_TIMEOUT_SECONDS,
               'errors': []}
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Child/group may have exited immediately before the signal.
    except PROCESS_ERRORS as e:
        cleanup['errors'].append(f'group kill {type(e).__name__}: {e}')
        try:
            p.kill()
        except ProcessLookupError:
            pass
        except PROCESS_ERRORS as e:
            cleanup['errors'].append(f'child kill {type(e).__name__}: {e}')
    returncode = None
    try:
        returncode = p.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
    except PROCESS_ERRORS as e:
        cleanup['errors'].append(f'reap {type(e).__name__}: {e}')
        # poll is nonblocking and can still reap an already-exited child if a
        # platform wait rejected its timeout argument.
        try:
            returncode = p.poll()
        except PROCESS_ERRORS as e:
            cleanup['errors'].append(f'poll {type(e).__name__}: {e}')
    cleanup['reaped'] = returncode is not None
    return returncode, cleanup


def run_one(m, entry, memory_mib, timeout_seconds):
    env = os.environ.copy()
    # Pin the imported checker and Lean/Lake support to the installed sysroot.
    # User artifacts cannot replace the trusted checker through LEAN_PATH.
    env['LEAN_PATH'] = os.pathsep.join([str(Path(m['sysroot']) / 'lib/lean'), *m['lean_path']])
    env['LEAN_SYSROOT'] = m['sysroot']
    env['CBZ_KERNEL_REPLAY_MODULE'] = entry['name']
    env['CBZ_KERNEL_REPLAY_OLEAN'] = entry['parts'][0]['path']
    cmd = [m['lean']['path'], '-j1', str(ROOT / 'ExactReplay.lean')]
    result = {'module': entry['name'], 'command': cmd, 'started': now(),
              'status': 'ERROR', 'ok': False, 'returncode': None,
              'memory_mib': memory_mib, 'timeout_seconds': timeout_seconds,
              'source': entry['source'], 'parts': entry['parts']}
    started = time.monotonic()
    # File-backed output prevents a noisy or corrupted module exhausting the
    # orchestrator's memory. Only a bounded tail is included in JSON evidence.
    import tempfile
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        p = None
        try:
            # A Popen context manager calls wait() without a timeout on exit.
            # Keep ownership explicit so every abnormal wait has bounded cleanup.
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                 start_new_session=True, preexec_fn=limits(memory_mib))
            result['pid'] = p.pid
            try:
                result['returncode'] = p.wait(timeout=timeout_seconds)
                result['status'] = 'FINISHED'
            except subprocess.TimeoutExpired:
                result['status'] = 'TIMEOUT'
            except PROCESS_ERRORS as e:
                result['error'] = f'wait {type(e).__name__}: {e}'
        except PROCESS_ERRORS as e:
            result['error'] = f'spawn {type(e).__name__}: {e}'
        if p is not None and result['status'] != 'FINISHED':
            result['returncode'], result['cleanup'] = cleanup_child(p)
            if not result['cleanup']['reaped']:
                result['status'] = 'ERROR'
                detail = f'child PID {p.pid} not reaped within {CLEANUP_TIMEOUT_SECONDS} seconds'
                result['error'] = (result.get('error', '') + '; ' + detail).lstrip('; ')
        for key, stream in (('stdout', stdout), ('stderr', stderr)):
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 131072))
            result[key] = stream.read().decode('utf-8', errors='replace')
            result[key + '_bytes'] = size
    marker = 'CBZ_KERNEL_REPLAY_OK ' + entry['name']
    if result['status'] == 'FINISHED':
        result['ok'] = result['returncode'] == 0 and result['stdout'].splitlines().count(marker) == 1
        result['status'] = 'PASS' if result['ok'] else 'FAIL'
    result['ended'] = now()
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def run_manifest(m, memory_mib, timeout_seconds):
    report = {'schema': SCHEMA, 'gate': 'G4-scoped-adapter', 'started': now(), 'ok': False,
              'results': [], 'completed_count': 0, 'expected_count': 0, 'manifest': m,
              'g2_status': 'NOT_ASSESSED', 'full_cbz': False}
    try:
        # RLIMIT_AS is inherited; a child cannot raise a finite hard limit.
        # sys.maxsize is a conservative signed platform-integer byte ceiling,
        # below Linux rlim_t's unsigned sentinel/range. Never silently clamp.
        hard_memory = resource.getrlimit(resource.RLIMIT_AS)[1]
        memory_bytes_max = sys.maxsize
        if hard_memory != resource.RLIM_INFINITY:
            memory_bytes_max = min(memory_bytes_max, hard_memory)
        memory_mib_max = memory_bytes_max // (1024 * 1024)
        report['resource_limits'] = {'memory_mib_max': memory_mib_max,
                                     'timeout_seconds_max': MAX_TIMEOUT_SECONDS,
                                     'cleanup_timeout_seconds': CLEANUP_TIMEOUT_SECONDS}
        if type(memory_mib) is not int or not 64 <= memory_mib <= memory_mib_max:
            raise ValueError(f'memory_mib must be an integer in [64, {memory_mib_max}]')
        # Compare the upper bound before float conversion: math.isfinite(10**400)
        # itself overflows. Bool is not a resource quantity despite being an int.
        if (type(timeout_seconds) not in (int, float)
                or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS
                or not math.isfinite(timeout_seconds)):
            raise ValueError(f'timeout_seconds must be finite and in (0, {MAX_TIMEOUT_SECONDS}]')
        validate_manifest(m)
        report['expected_count'] = len(m['expected_modules'])
        for entry in m['modules']:
            validate_manifest(m)
            report['results'].append(run_one(m, entry, memory_mib, timeout_seconds))
            if report['results'][-1].get('cleanup', {}).get('reaped') is False:
                break  # Do not start another process while an earlier PID remains unreaped.
        validate_manifest(m)
        report['completed_count'] = sum(x['ok'] for x in report['results'])
        report['ok'] = (report['completed_count'] == report['expected_count']
                        and len(report['results']) == report['expected_count']
                        and [r['module'] for r in report['results']] == m['expected_modules'])
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError, OverflowError) as e:
        report['error'] = f'{type(e).__name__}: {e}'
    report['ended'] = now()
    return report


def emit(value, output):
    data = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if output:
        Path(output).write_text(data)
    else:
        print(data, end='')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('help', add_help=False, allow_abbrev=False)
    prepare = subs.add_parser('prepare', add_help=False, allow_abbrev=False)
    one = subs.add_parser('one', add_help=False, allow_abbrev=False)
    for p in (prepare, one):
        p.add_argument('--lean', required=True)
        p.add_argument('--lean-path', action='append', default=[])
        p.add_argument('--context', action='append', default=[])
        p.add_argument('--output')
    prepare.add_argument('--module', action='append', required=True)
    one.add_argument('module')
    one.add_argument('--source', required=True)
    run = subs.add_parser('run', add_help=False, allow_abbrev=False)
    run.add_argument('manifest')
    run.add_argument('--output')
    for p in (run, one):
        p.add_argument('--memory-mib', type=int, required=True)
        p.add_argument('--timeout-seconds', type=float, required=True)
    args = parser.parse_args(argv)
    if args.action == 'help':
        print(parser.format_help())
        print('prepare --lean ABS --lean-path DIR --module NAME=SOURCE --output MANIFEST')
        print('run MANIFEST --memory-mib MIB --timeout-seconds SECONDS --output REPORT')
        print('one NAME --lean ABS --lean-path DIR --source SOURCE --memory-mib MIB --timeout-seconds SECONDS')
        return 0
    try:
        if args.action == 'run':
            m = json.loads(Path(args.manifest).read_text())
        else:
            selections = args.module if args.action == 'prepare' else [args.module + '=' + args.source]
            m = prepare_manifest(args.lean, args.lean_path, selections, args.context)
        if args.action == 'prepare':
            emit(m, args.output)
            return 0
        report = run_manifest(m, args.memory_mib, args.timeout_seconds)
        emit(report, args.output)
        return 0 if report['ok'] else 1
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError, OverflowError) as e:
        print(f'kernel replay ERROR: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
