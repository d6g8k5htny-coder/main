#!/usr/bin/env python3
"""Bounded local CBZ evidence validator. No network or Git-ref writes.

PASS means only that the explicitly reported mechanical checks passed. This tool
cannot certify mathematical closure, complete Layer A, remote hygiene or CBZ-0.
Gate PASS means an observed, source-bound registered command exited zero.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

DEFAULT_CONFIG = 'cbz/evidence/website-foundation-20261029/baseline-config.json'
GATES = tuple(f'G{i}' for i in range(12))
STATES = {'PASS', 'FAIL', 'INCOMPLETE'}


class ValidationError(RuntimeError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValidationError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read(path):
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=_pairs)
    except (OSError, ValueError) as error:
        raise ValidationError('unreadable or invalid JSON: ' + str(path)) from error


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValidationError('expected relative source path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or '.git' in Path(relative).parts:
        raise ValidationError('source path escapes root or enters Git internals')
    return path


def _config(root, config_path=None):
    selected = config_path or DEFAULT_CONFIG
    path = _inside(root, selected)
    config = _read(path)
    if config.get('schema') != 'cbz.baseline.v1':
        raise ValidationError('unsupported baseline config schema')
    if not set(config.get('allowed_axioms', [])).issubset({'propext', 'Classical.choice', 'Quot.sound'}):
        raise ValidationError('custom axioms cannot be admitted by config')
    gates = config.get('gates', [])
    if not isinstance(gates, list) or len({g.get('id') for g in gates}) != len(gates):
        raise ValidationError('duplicate or invalid gate declarations')
    for gate in gates:
        if gate.get('id') not in GATES or not isinstance(gate.get('argv'), list) or not gate['argv']:
            raise ValidationError('invalid registered gate')
        if any(not isinstance(x, str) or not x or '\x00' in x for x in gate['argv']):
            raise ValidationError('invalid gate argv')
        if not isinstance(gate.get('sources'), list) or not gate['sources']:
            raise ValidationError('gate requires exact source inputs')
        for source in gate['sources']:
            _inside(root, source)
        timeout = gate.get('timeout_seconds', 60)
        memory = gate.get('memory_mb', 1024)
        if type(timeout) is not int or not 1 <= timeout <= 3600 or type(memory) is not int or not 64 <= memory <= 8192:
            raise ValidationError('gate resource bounds invalid')
    config['_config_path'] = selected
    return config


def _status(failures, unmet):
    return 'FAIL' if failures else ('INCOMPLETE' if unmet else 'PASS')


def _hash_entries(root, entries, label, failures, unmet):
    if not isinstance(entries, list) or not entries:
        unmet.append(label + ': no exact source/evidence hashes')
        return
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(entry.get('sha256', ''))):
            failures.append(label + ': invalid SHA-256 entry')
            continue
        path = _inside(root, entry.get('path'))
        if entry['path'] in seen:
            failures.append(label + ': duplicate source path')
        seen.add(entry['path'])
        if not path.is_file():
            unmet.append(label + ': missing ' + entry['path'])
        elif _sha(path) != entry['sha256']:
            failures.append(label + ': source/evidence hash mismatch ' + entry['path'])


def _proofs(root, config, failures, unmet):
    index = _read(_inside(root, config['proof_index']))
    records = index.get('records')
    if index.get('schema') != 'cbz.proof-index.v1' or not isinstance(records, list) or not records:
        failures.append('proof index schema/records invalid')
        return
    ids = [r.get('id') for r in records]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        failures.append('proof index IDs missing or duplicated')
        return
    by_id = dict(zip(ids, records))
    allowed = set(config.get('allowed_axioms', []))
    for record in records:
        label = 'proof ' + record['id']
        status = record.get('status')
        if status not in ('CLOSED', 'QUARANTINED'):
            failures.append(label + ': invalid status')
        _hash_entries(root, record.get('source_hashes'), label, failures, unmet)
        if not record.get('exact_boundary') or not record.get('scientific_closure_limits'):
            unmet.append(label + ': missing exact boundary or scientific limits')
        deps = record.get('dependencies')
        if not isinstance(deps, list) or any(dep not in by_id for dep in deps):
            failures.append(label + ': dependency list missing or unresolved')
            continue
        if status == 'QUARANTINED':
            if record.get('owner') != 'none' or any(not record.get(x) for x in ('reason', 'attempts', 'isolation')):
                failures.append(label + ': quarantine reason/attempts/isolation/owner incomplete')
        if status == 'CLOSED':
            axioms = record.get('axioms')
            if not isinstance(axioms, list):
                unmet.append(label + ': exact observed axiom list unavailable')
            elif any(axiom not in allowed for axiom in axioms):
                failures.append(label + ': custom or undeclared axiom in CLOSED record')
            _hash_entries(root, record.get('kernel_evidence'), label + ' kernel evidence', failures, unmet)
            todo, seen = list(deps), set()
            while todo:
                dep = todo.pop()
                if dep in seen:
                    continue
                seen.add(dep)
                if by_id[dep].get('status') == 'QUARANTINED':
                    failures.append(label + ': transitively depends on quarantined ' + dep)
                next_deps = by_id[dep].get('dependencies', [])
                if isinstance(next_deps, list):
                    todo.extend(x for x in next_deps if x in by_id)
    # Cycles cannot supply independent closure evidence.
    visiting, done = set(), set()
    def visit(name):
        if name in visiting:
            failures.append('proof dependency cycle: ' + name)
            return
        if name in done:
            return
        visiting.add(name)
        for dep in by_id[name].get('dependencies', []):
            if dep in by_id:
                visit(dep)
        visiting.remove(name)
        done.add(name)
    for name in ids:
        visit(name)


def _tools(root, config, failures, unmet):
    registry = _read(_inside(root, config['tool_registry']))
    if registry.get('schema') != 'cbz.tools.v1' or not isinstance(registry.get('tools'), list):
        failures.append('tool registry schema invalid')
        return
    listed = set()
    for tool in registry['tools']:
        path = tool.get('path')
        if path in listed:
            failures.append('duplicate registered tool: ' + str(path))
        listed.add(path)
        if not _inside(root, path).is_file():
            unmet.append('registered tool missing: ' + str(path))
        command = tool.get('self_test')
        if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
            unmet.append('registered tool has no executable self-test: ' + str(path))
        if 'source_hashes' in tool:
            _hash_entries(root, tool['source_hashes'], 'tool ' + str(path), failures, unmet)
    for path in (root / 'tools').glob('cbz_*.py'):
        if path.relative_to(root).as_posix() not in listed:
            failures.append('unlisted mission tool: ' + path.name)


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_NO_REPLACE_OBJECTS='1')
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, env=env, timeout=30)
    if p.returncode:
        raise ValidationError('local Git observation unavailable')
    return p.stdout


def _pins(root, config, failures, unmet):
    result = []
    pins = config.get('pins')
    if not isinstance(pins, list) or not pins:
        unmet.append('cross-repository pin inventory unavailable or empty')
        return result
    for pin in pins:
        local_f, local_u = [], []
        try:
            commit = pin['commit']
            if not re.fullmatch(r'[0-9a-f]{40}(?:[0-9a-f]{24})?', commit):
                local_f.append('pin commit must be exact full object ID')
            repo = (root / pin['repo']).resolve()
            if not repo.is_dir():
                local_u.append('offline dependency repository unavailable')
            elif not local_f:
                _git(repo, 'cat-file', '-e', commit + '^{commit}')
                objpath = pin['object_path']
                if not isinstance(objpath, str) or not objpath or objpath.startswith(('/', '-')) or '..' in Path(objpath).parts:
                    raise ValidationError('invalid pinned object path')
                content = _git(repo, 'show', commit + ':' + objpath)
                if hashlib.sha256(content).hexdigest() != pin['sha256']:
                    local_f.append('pinned object bytes mismatch')
            if 'pin_file' in pin:
                manifest = _read(_inside(root, pin['pin_file']))
                matches = [p for p in manifest.get('packages', []) if p.get('name') == pin.get('package')]
                if len(matches) != 1 or matches[0].get('rev') != commit:
                    local_f.append('package manifest pinned commit mismatch')
            else:
                local_u.append('no consuming package manifest binding')
        except (KeyError, TypeError, ValidationError, OSError, subprocess.TimeoutExpired):
            local_u.append('offline pin object/config evidence incomplete')
        failures.extend('pin: ' + x for x in local_f)
        unmet.extend('pin: ' + x for x in local_u)
        result.append({'repo': pin.get('repo'), 'status': _status(local_f, local_u), 'failures': local_f, 'unmet': local_u})
    return result


def _binding(root, config, gate):
    paths = [config['_config_path'], config['proof_index'], config['tool_registry'], *gate['sources']]
    binding = {name: _sha(_inside(root, name)) for name in sorted(set(paths))}
    if gate['id'] == 'G1':
        # G1 receipt semantics are implemented by this consumer; bind that code too.
        binding['tools/cbz_baseline.py'] = _sha(Path(__file__).resolve())
    return binding


def _scan_execution_identity(root, gate):
    argv = gate.get('argv')
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item or '\x00' in item for item in argv
    ):
        raise ValidationError('invalid registered G1 command')
    executable = shutil.which(argv[0])
    if not executable:
        raise ValidationError('registered G1 executable unavailable')
    path = Path(executable).resolve()
    if not path.is_file():
        raise ValidationError('registered G1 executable is not a file')
    environment = {
        'PATH': os.defpath,
        'LANG': 'C',
        'LC_ALL': 'C',
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_CONFIG_SYSTEM': os.devnull,
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_TERMINAL_PROMPT': '0',
        'GIT_NO_REPLACE_OBJECTS': '1',
        'GIT_OPTIONAL_LOCKS': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': '0',
    }
    git_executable = shutil.which('git', path=environment['PATH'])
    if not git_executable:
        raise ValidationError('registered G1 Git executable unavailable')
    git_path = Path(git_executable).resolve()
    if not git_path.is_file():
        raise ValidationError('registered G1 Git executable is not a file')
    try:
        git_sha256 = _sha(git_path)
        exclude_output = subprocess.run(
            [str(git_path), '-C', str(root), 'rev-parse', '--git-path', 'info/exclude'],
            cwd=root, capture_output=True, env=environment, timeout=10, check=True,
        ).stdout
        exclude_name = os.fsdecode(exclude_output.removesuffix(b'\n'))
        if not exclude_name:
            raise ValidationError('G1 Git info/exclude path unavailable')
        info_exclude = Path(exclude_name)
        if not info_exclude.is_absolute():
            info_exclude = (Path(root) / info_exclude).resolve()
        else:
            info_exclude = info_exclude.resolve()
        info_exclude_sha256 = _sha(info_exclude) if info_exclude.is_file() else None

        # Pin Git's implicit global-ignore lookup to private, repository-local
        # empty directories. Reject either default ignore file if it appears.
        git_dir = info_exclude.parent.parent
        home_dir = git_dir / 'cbz-g1-home'
        xdg_dir = git_dir / 'cbz-g1-xdg'
        for directory in (home_dir, xdg_dir):
            if directory.is_symlink():
                raise ValidationError('G1 isolated Git home/config path is a symlink')
            if directory.exists() and not directory.is_dir():
                raise ValidationError('G1 isolated Git home/config path is not a directory')
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.is_symlink():
                raise ValidationError('G1 isolated Git home/config path became a symlink')
        environment['HOME'] = str(home_dir)
        environment['XDG_CONFIG_HOME'] = str(xdg_dir)

        xdg_ignore = xdg_dir / 'git' / 'ignore'
        home_ignore = home_dir / '.config' / 'git' / 'ignore'
        for directory in (xdg_dir / 'git', home_dir / '.config', home_dir / '.config' / 'git'):
            if directory.is_symlink():
                raise ValidationError('G1 default Git ignore parent is a symlink')
            if directory.exists() and not directory.is_dir():
                raise ValidationError('G1 default Git ignore parent is not a directory')
        if xdg_ignore.is_symlink() or xdg_ignore.exists():
            raise ValidationError('G1 default XDG Git ignore file must be absent')
        if home_ignore.is_symlink() or home_ignore.exists():
            raise ValidationError('G1 default HOME Git ignore file must be absent')
        default_excludes = {
            'home_path_sha256': hashlib.sha256(os.fsencode(str(home_dir))).hexdigest(),
            'xdg_config_home_path_sha256': hashlib.sha256(os.fsencode(str(xdg_dir))).hexdigest(),
            'xdg_ignore_path_sha256': hashlib.sha256(os.fsencode(str(xdg_ignore))).hexdigest(),
            'xdg_ignore_sha256': None,
            'home_fallback_ignore_path_sha256': hashlib.sha256(os.fsencode(str(home_ignore))).hexdigest(),
            'home_fallback_ignore_sha256': None,
        }
        git_config = subprocess.run(
            [str(git_path), '-C', str(root), 'config', '--null', '--show-origin', '--list'],
            cwd=root, capture_output=True, env=environment, timeout=10, check=True,
        ).stdout
        core_excludes_output = subprocess.run(
            [str(git_path), '-C', str(root), 'config', '--path', '--null', '--get-all', 'core.excludesfile'],
            cwd=root, capture_output=True, env=environment, timeout=10,
        )
        if core_excludes_output.returncode not in (0, 1):
            raise ValidationError('G1 Git core.excludesfile configuration unavailable')
        core_excludesfiles = []
        for raw_path in core_excludes_output.stdout.split(b'\0'):
            if not raw_path:
                continue
            configured_path = Path(os.fsdecode(raw_path)).expanduser()
            if not configured_path.is_absolute():
                configured_path = (Path(root) / configured_path).resolve()
            else:
                configured_path = configured_path.resolve()
            core_excludesfiles.append({
                'path_sha256': hashlib.sha256(os.fsencode(str(configured_path))).hexdigest(),
                'sha256': _sha(configured_path) if configured_path.is_file() else None,
            })
        version = subprocess.run(
            [str(git_path), '--version'], cwd=root, capture_output=True,
            env=environment, timeout=10, check=True,
        ).stdout
        if _sha(git_path) != git_sha256:
            raise ValidationError('registered G1 Git executable changed during identity capture')
    except (OSError, subprocess.SubprocessError) as error:
        raise ValidationError('registered G1 Git execution identity unavailable') from error
    return {
        'argv': [str(path), *argv[1:]],
        'executable_path': str(path),
        'executable_sha256': _sha(path),
        'environment': environment,
        'git': {
            'executable_path': str(git_path),
            'executable_sha256': git_sha256,
            'version': os.fsdecode(version).strip(),
            'config_sha256': hashlib.sha256(git_config).hexdigest(),
            'info_exclude_sha256': info_exclude_sha256,
            'core_excludesfiles': core_excludesfiles,
            'default_excludes': default_excludes,
        },
    }

def _scan_snapshot(raw_output, exit_code, execution_identity):
    if type(exit_code) is not int:
        raise ValidationError('G1 scan exit code unavailable')
    git_identity = execution_identity.get('git') if isinstance(execution_identity, dict) else None
    def valid_hash(value):
        return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None
    core_excludesfiles = git_identity.get('core_excludesfiles') if isinstance(git_identity, dict) else None
    default_excludes = git_identity.get('default_excludes') if isinstance(git_identity, dict) else None
    environment = execution_identity.get('environment') if isinstance(execution_identity, dict) else None
    if (
        not isinstance(execution_identity, dict)
        or not valid_hash(execution_identity.get('executable_sha256'))
        or not isinstance(git_identity, dict)
        or not isinstance(git_identity.get('executable_path'), str)
        or not valid_hash(git_identity.get('executable_sha256'))
        or not valid_hash(git_identity.get('config_sha256'))
        or not isinstance(git_identity.get('version'), str)
        or not git_identity.get('version')
        or (git_identity.get('info_exclude_sha256') is not None and not valid_hash(git_identity.get('info_exclude_sha256')))
        or not isinstance(environment, dict)
        or not isinstance(environment.get('HOME'), str)
        or not Path(environment.get('HOME', '')).is_absolute()
        or not isinstance(environment.get('XDG_CONFIG_HOME'), str)
        or not Path(environment.get('XDG_CONFIG_HOME', '')).is_absolute()
        or not isinstance(default_excludes, dict)
        or not valid_hash(default_excludes.get('home_path_sha256'))
        or not valid_hash(default_excludes.get('xdg_config_home_path_sha256'))
        or not valid_hash(default_excludes.get('xdg_ignore_path_sha256'))
        or default_excludes.get('xdg_ignore_sha256') is not None
        or not valid_hash(default_excludes.get('home_fallback_ignore_path_sha256'))
        or default_excludes.get('home_fallback_ignore_sha256') is not None
        or default_excludes.get('home_path_sha256') != hashlib.sha256(os.fsencode(environment['HOME'])).hexdigest()
        or default_excludes.get('xdg_config_home_path_sha256') != hashlib.sha256(os.fsencode(environment['XDG_CONFIG_HOME'])).hexdigest()
        or default_excludes.get('xdg_ignore_path_sha256') != hashlib.sha256(os.fsencode(str(Path(environment['XDG_CONFIG_HOME']) / 'git' / 'ignore'))).hexdigest()
        or default_excludes.get('home_fallback_ignore_path_sha256') != hashlib.sha256(os.fsencode(str(Path(environment['HOME']) / '.config' / 'git' / 'ignore'))).hexdigest()
        or not isinstance(core_excludesfiles, list)
        or any(
            not isinstance(item, dict)
            or not valid_hash(item.get('path_sha256'))
            or (item.get('sha256') is not None and not valid_hash(item.get('sha256')))
            for item in core_excludesfiles
        )
    ):
        raise ValidationError('G1 scan executable or Git identity unavailable')
    try:
        parsed = json.loads(raw_output.decode('utf-8'), object_pairs_hook=_pairs)
        if not isinstance(parsed, dict):
            raise ValidationError('G1 scan output is not a JSON object')
        canonical = json.dumps(
            {'exit_code': exit_code, 'execution': execution_identity, 'output': parsed},
            sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False,
        ).encode('utf-8')
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValidationError('G1 scan output unavailable or invalid') from error
    return hashlib.sha256(canonical).hexdigest()


def _recompute_scan_snapshot(root, gate, expected_execution):
    execution = _scan_execution_identity(root, gate)
    if execution != expected_execution:
        raise ValidationError('G1 registered execution identity changed')
    timeout = gate.get('timeout_seconds', 60)
    memory = gate.get('memory_mb', 1024) * 1024 * 1024

    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))

    with tempfile.TemporaryDirectory(prefix='cbz-g1-recheck-') as directory:
        stdout_path = Path(directory) / 'stdout.json'
        stderr_path = Path(directory) / 'stderr.log'
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            process = subprocess.Popen(
                execution['argv'], cwd=root, stdout=stdout, stderr=stderr,
                env=execution['environment'], start_new_session=True, preexec_fn=limits,
            )
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise ValidationError('G1 scan timed out during receipt verification') from error
        if _scan_execution_identity(root, gate) != execution:
            raise ValidationError('G1 registered execution identity changed during recheck')
        return _scan_snapshot(stdout_path.read_bytes(), exit_code, execution)

def _receipt_statuses(root, config, receipt_path, failures, unmet):
    statuses = dict.fromkeys(GATES, 'INCOMPLETE')
    declared = {gate['id']: gate for gate in config.get('gates', [])}
    if receipt_path is None:
        unmet.append('no observed source-bound gate receipts supplied')
        return statuses
    try:
        receipt_path = Path(receipt_path).resolve()
        receipts = _read(receipt_path)
        if not isinstance(receipts, dict):
            raise ValidationError('receipt must be a JSON object')
        if receipts.get('schema') != 'cbz.receipts.v1' or receipts.get('config_sha256') != _sha(_inside(root, config['_config_path'])):
            raise ValidationError('stale or invalid receipt configuration binding')
        rows = receipts.get('gates')
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValidationError('receipt gates must be a list of objects')
        names = [row.get('id') for row in rows]
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValidationError('duplicate or invalid gate receipts')
        for row in rows:
            name = row['id']
            if name not in declared:
                raise ValidationError('receipt for unregistered gate')
            gate = declared[name]
            binding_before = _binding(root, config, gate)
            if row.get('argv') != gate['argv'] or row.get('source_sha256') != binding_before:
                unmet.append(name + ': stale receipt command/source identity')
                continue
            valid = True
            log_paths = {}
            for stream in ('stdout', 'stderr'):
                relative = row.get(stream + '_path')
                if not isinstance(relative, str):
                    valid = False
                    break
                log = (receipt_path.parent / relative).resolve()
                if not log.is_relative_to(receipt_path.parent) or not log.is_file() or _sha(log) != row.get(stream + '_sha256'):
                    valid = False
                    break
                log_paths[stream] = log
            exit_code = row.get('exit_code')
            observed = 'INCOMPLETE' if row.get('timed_out') or row.get('source_changed') else ('PASS' if exit_code == 0 else ('INCOMPLETE' if exit_code == 2 else 'FAIL'))
            if not valid or row.get('status') != observed or type(exit_code) not in (int, type(None)) or not isinstance(row.get('started_utc'), str) or not row.get('started_utc'):
                unmet.append(name + ': unverified/forged receipt or process logs')
                continue
            if name == 'G1':
                snapshot = row.get('scan_snapshot_sha256')
                execution = row.get('scan_execution')
                if not isinstance(snapshot, str) or not re.fullmatch(r'[0-9a-f]{64}', snapshot):
                    unmet.append('G1: missing or invalid scan snapshot; legacy receipt is INCOMPLETE')
                    continue
                try:
                    current_execution = _scan_execution_identity(root, gate)
                    if execution != current_execution:
                        unmet.append('G1: registered execution identity is missing or changed')
                        continue
                    stored_snapshot = _scan_snapshot(log_paths['stdout'].read_bytes(), exit_code, execution)
                    if stored_snapshot != snapshot:
                        unmet.append('G1: stored scan snapshot does not match preserved JSON output')
                        continue
                    current_snapshot = _recompute_scan_snapshot(root, gate, execution)
                    binding_after = _binding(root, config, gate)
                    execution_after = _scan_execution_identity(root, gate)
                except (ValidationError, OSError, subprocess.SubprocessError, TypeError, KeyError, AttributeError):
                    unmet.append('G1: current registered scan snapshot could not be reproduced')
                    continue
                if binding_after != binding_before or execution_after != execution:
                    unmet.append('G1: source or execution identity changed during receipt recheck')
                    continue
                if current_snapshot != snapshot:
                    unmet.append('G1: current scan snapshot differs from receipt')
                    continue
            statuses[name] = observed
    except (ValidationError, OSError, TypeError, KeyError, AttributeError):
        unmet.append('stale/invalid receipt or missing exact execution evidence')
        return dict.fromkeys(GATES, 'INCOMPLETE')
    for name, status in statuses.items():
        if status == 'INCOMPLETE':
            unmet.append(name + ': not observed passing on current registered inputs')
        elif status == 'FAIL':
            failures.append(name + ': observed command failed')
    return statuses

def cold_start(root, *, config_path=None):
    root = Path(root).resolve()
    config = _config(root, config_path)
    cold = config.get('cold_start', {})
    failures, unmet = [], []
    try:
        text = _inside(root, cold['status']).read_text()
        for answer in cold.get('required_answers', []):
            if answer not in text:
                failures.append('STATUS missing required answer heading: ' + answer)
        commands = cold.get('commands', [])
        if not commands:
            unmet.append('no cold-start commands registered')
        for command in commands:
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                failures.append('invalid cold-start command')
                continue
            if ' '.join(command) not in text:
                failures.append('registered cold-start argv absent from STATUS')
            if not shutil.which(command[0]) and not (root / command[0]).is_file():
                unmet.append('cold-start executable unavailable: ' + command[0])
            for arg in command[1:]:
                if arg.startswith(('tools/', 'tests/')) and not _inside(root, arg).is_file():
                    unmet.append('cold-start entrypoint missing: ' + arg)
    except (KeyError, OSError):
        unmet.append('STATUS cold-start entrypoint unavailable')
    return {'schema': 'cbz.cold-start.v1', 'status': _status(failures, unmet), 'failures': failures,
            'unmet': unmet, 'certifies_theorems': False,
            'scope': 'required answer headings, documented argv, and executable entrypoint availability only'}


def inspect(root, *, receipts=None, config_path=None):
    root = Path(root).resolve()
    config = _config(root, config_path)
    failures, unmet = [], []
    for operation in (_proofs, _tools):
        try:
            operation(root, config, failures, unmet)
        except (ValidationError, OSError, TypeError, KeyError):
            unmet.append(operation.__name__ + ': input schema/files incomplete')
    pins = _pins(root, config, failures, unmet)
    statuses = _receipt_statuses(root, config, receipts, failures, unmet)
    for name in GATES:
        if name not in {gate['id'] for gate in config.get('gates', [])}:
            unmet.append(name + ': no registered gate command')
    for doc in config.get('docs', []):
        try:
            text = _inside(root, doc).read_text()
            markers = re.findall(r'<!--\s*cbz-gates:\s*(\{.*?\})\s*-->', text, re.S)
            if len(markers) != 1:
                unmet.append(doc + ': missing/duplicate machine gate claims')
                continue
            claims = json.loads(markers[0], object_pairs_hook=_pairs)
            if set(claims) != set(GATES) or any(value not in STATES for value in claims.values()):
                failures.append(doc + ': invalid gate claims')
            elif claims != statuses:
                failures.append(doc + ': documented gate statuses disagree with observed current receipts')
        except (OSError, ValidationError, ValueError):
            unmet.append(doc + ': unavailable/invalid document claims')
    cold = cold_start(root, config_path=config_path)
    failures.extend(cold['failures'])
    unmet.extend(cold['unmet'])
    try:
        dirty = _git(root, 'status', '--porcelain=v1', '--untracked-files=all').decode().splitlines()
        stashes = _git(root, 'stash', 'list').decode().splitlines()
        heads = _git(root, 'for-each-ref', '--format=%(refname)', 'refs/heads').decode().splitlines()
        worktrees = _git(root, 'worktree', 'list', '--porcelain').decode().count('worktree ')
        if dirty:
            unmet.append('local Git worktree/index not clean')
        if stashes:
            unmet.append('local Git stashes remain')
        if len(heads) != 1:
            unmet.append('local Git branch disposition incomplete')
        if worktrees != 1:
            unmet.append('additional local Git worktrees require disposition')
        git = {'dirty_paths': len(dirty), 'stashes': len(stashes), 'local_heads': heads, 'worktrees': worktrees}
    except (ValidationError, subprocess.TimeoutExpired):
        unmet.append('local Git hygiene unavailable')
        git = {'status': 'INCOMPLETE'}
    return {'schema': 'cbz.inspect.v1', 'status': _status(failures, unmet), 'gate_statuses': statuses,
            'failures': failures, 'unmet': unmet, 'pins': pins, 'local_git': git,
            'certifies_cbz0': False, 'baseline_state': 'Partial Baseline',
            'scope': 'mechanical local evidence consistency; command exits do not establish full Layer A or theorem acceptance'}


def run(root, output, *, config_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValidationError('run output must be new and outside source root')
    config = _config(root, config_path)
    output.mkdir(parents=True)
    result = {'schema': 'cbz.receipts.v1', 'config_sha256': _sha(_inside(root, config['_config_path'])),
              'scope': 'observed registered command exits with exact source/log binding; not scientific certification', 'gates': []}
    for gate in sorted(config.get('gates', []), key=lambda g: int(g['id'][1:])):
        name = gate['id']
        row = {'id': name, 'argv': gate['argv'], 'status': 'INCOMPLETE', 'exit_code': None,
               'timed_out': False, 'source_changed': False, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'stdout_path': name + '.stdout', 'stderr_path': name + '.stderr'}
        execution = None
        try:
            row['source_sha256'] = _binding(root, config, gate)
            if name == 'G1':
                execution = _scan_execution_identity(root, gate)
                row['scan_execution'] = execution
            timeout = gate.get('timeout_seconds', 60)
            def limits():
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
                memory = gate.get('memory_mb', 1024) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
            command = execution['argv'] if execution is not None else gate['argv']
            environment = execution['environment'] if execution is not None else None
            with (output / row['stdout_path']).open('wb') as stdout, (output / row['stderr_path']).open('wb') as stderr:
                process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                           env=environment, start_new_session=True, preexec_fn=limits)
                try:
                    row['exit_code'] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    row['timed_out'] = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    row['exit_code'] = process.wait()
            if name == 'G1' and _scan_execution_identity(root, gate) != execution:
                row['source_changed'] = True
            row['source_changed'] = row['source_changed'] or row['source_sha256'] != _binding(root, config, gate)
            row['status'] = 'INCOMPLETE' if row['timed_out'] or row['source_changed'] else ('PASS' if row['exit_code'] == 0 else ('INCOMPLETE' if row['exit_code'] == 2 else 'FAIL'))
        except (OSError, ValidationError, subprocess.SubprocessError):
            row['status'] = 'INCOMPLETE'
            row['error'] = 'command/input unavailable; no passing execution claimed'
        for stream in ('stdout', 'stderr'):
            path = output / row[stream + '_path']
            path.touch(exist_ok=True)
            row[stream + '_sha256'] = _sha(path)
        if name == 'G1':
            try:
                if execution is None:
                    raise ValidationError('G1 executable identity unavailable')
                row['scan_snapshot_sha256'] = _scan_snapshot(
                    (output / row['stdout_path']).read_bytes(), row['exit_code'], execution)
            except (ValidationError, OSError):
                row['status'] = 'INCOMPLETE'
                row['scan_snapshot_error'] = 'G1 JSON output, execution identity or actual exit code unavailable'
        result['gates'].append(row)
        temp = output / 'receipts.json.tmp'
        temp.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        temp.replace(output / 'receipts.json')
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'cold-start', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--root', default='.')
        command.add_argument('--config', default=DEFAULT_CONFIG, help='root-relative namespaced config')
        if name == 'inspect':
            command.add_argument('--receipts')
        if name == 'run':
            command.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run(args.root, args.output, config_path=args.config)
            code = 0 if len(result['gates']) == 12 and all(r['status'] == 'PASS' for r in result['gates']) else 2
        else:
            result = inspect(args.root, receipts=args.receipts, config_path=args.config) if args.command == 'inspect' else cold_start(args.root, config_path=args.config)
            code = {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]
        print(json.dumps(result, sort_keys=True, indent=2))
        return code
    except (ValidationError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'INCOMPLETE', 'error': str(error), 'certifies_cbz0': False}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
