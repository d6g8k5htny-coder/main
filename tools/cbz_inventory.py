#!/usr/bin/env python3
"""Offline tracked-source/binding integrity; never kernel or CBZ certification.

Generate a deterministic inventory of an explicit set of at most twelve local
Git clones. Verify exactly those bytes, source states and declared classification
bindings. A valid CLOSED record remains an externally supplied assertion: this
tool checks its actual receipt/manifest bytes and target bindings, not whether
the claimed kernel execution happened or proved the intended mathematics.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

DEFAULT_REPOSITORIES = (
    'main', 'Math-', 'governance-', 'Universal-Law-Workspace', 'meta-framework',
    'query-', 'google-drive', 'trial', 'sandbox', 'd6g8k5htny-coder',
    'openclaw-custodian-validation', 'research-hosted-qualification',
)
SCOPE = 'tracked-source-and-declared-binding-integrity-only'
APPROVED_KERNEL_AXIOMS = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
REPO_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
FILE_KEYS = {'path', 'kind', 'mode', 'git_object', 'head_git_object', 'index_git_object',
             'head_mode', 'index_mode', 'working_mode',
             'working_git_blob', 'bytes', 'sha256', 'materialization', 'submodule_head'}
BINDING_KEYS = {'repository', 'commit', 'path', 'git_object', 'sha256'}
MANIFEST_KEYS = {'schema', 'certification_scope', 'semantic_claim_completeness',
                 'cbz_green', 'repositories', 'obligations'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _control_types(value):
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), 'control keys must be strings')
        for child in value.values():
            _control_types(child)
    elif isinstance(value, list):
        for child in value:
            _control_types(child)
    else:
        require(type(value) in (str, int, bool, type(None)), 'unsupported control value; floats refused')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def forbidden(value):
        raise ValueError('floating/nonfinite JSON token refused: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_float=forbidden,
                        parse_constant=forbidden)
    _control_types(result)
    return result


def _json_file(path):
    require(not path.is_symlink() and path.is_file(), 'missing or symlinked control file')
    with path.open('rb') as stream:
        raw = stream.read(256 * 1024 * 1024 + 1)
    require(len(raw) <= 256 * 1024 * 1024, 'control file exceeds 256 MiB bound')
    return strict_json(raw.decode('utf-8'))


def _names(names):
    require(isinstance(names, (list, tuple)) and 1 <= len(names) <= 12,
            'one to twelve explicit repository names required')
    require(all(type(name) is str and REPO_NAME.fullmatch(name) for name in names),
            'invalid repository directory name')
    require(len(set(names)) == len(names), 'duplicate repository name')
    return sorted(names)


def _path(value):
    require(type(value) is str and value and '\0' not in value and not value.startswith('/'),
            'invalid tracked path')
    require(all(part not in ('', '.', '..', '.git') for part in value.split('/')),
            'unsafe tracked path')
    # Filesystem decoding/encoding round-trips arbitrary Git filename bytes.
    require(os.fsdecode(os.fsencode(value)) == value, 'unrepresentable tracked path')
    return value


def _git(repo, *args, input_bytes=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_NO_LAZY_FETCH='1',
               GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0',
               GIT_ALLOW_PROTOCOL='', LC_ALL='C')
    command = ['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=' + os.devnull,
               '-c', 'core.untrackedCache=false', '-C', str(repo)]
    try:
        if args and args[0] == 'status':
            probe = subprocess.run([*command, 'config', '--null', '--get-regexp',
                                    r'^filter\..*\.(clean|process)$'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, env=env)
            require(probe.returncode in (0, 1), 'could not inspect repository filter configuration')
            require(probe.returncode == 1 and not probe.stdout,
                    'configured Git clean/process filters refused before execution')
        result = subprocess.run([*command, *args], input=input_bytes, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('local Git command failed: ' + str(error)) from error
    require(result.returncode == 0, 'local Git command failed: ' +
            result.stderr.decode('utf-8', 'backslashreplace')[:2000])
    return result.stdout


def _state(repo):
    head = _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}').decode('ascii').strip()
    require(HEX40.fullmatch(head), 'SHA-1 commit required')
    index = _git(repo, 'ls-files', '--stage', '-z', '--full-name')
    status_bytes = _git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all',
                        '--ignore-submodules=all')
    return head, index, status_bytes


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _parent_fd(repo, relative):
    parts = os.fsencode(_path(relative)).split(b'/')
    fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = following
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def _read_regular(repo, relative):
    parent, name = _parent_fd(repo, relative)
    fd = None
    try:
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), 'unsupported or substituted regular file: ' + relative)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        opened = os.fstat(fd)
        require(_fingerprint(before) == _fingerprint(opened), 'file changed while opening: ' + relative)
        require(opened.st_size <= 1024 * 1024 * 1024, 'tracked file exceeds 1 GiB bound: ' + relative)
        digest = hashlib.sha256()
        blob = hashlib.sha1(b'blob ' + str(opened.st_size).encode('ascii') + b'\0')
        count = 0
        while True:
            raw = os.read(fd, 1024 * 1024)
            if not raw:
                break
            count += len(raw)
            require(count <= opened.st_size, 'tracked file grew while reading: ' + relative)
            digest.update(raw)
            blob.update(raw)
        after = os.fstat(fd)
        require(count == opened.st_size and _fingerprint(opened) == _fingerprint(after),
                'tracked file changed while reading: ' + relative)
        require(_fingerprint(after) == _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                'tracked file replaced while reading: ' + relative)
        return {'working_mode': '100755' if opened.st_mode & stat.S_IXUSR else '100644',
                'working_git_blob': blob.hexdigest(), 'bytes': count,
                'sha256': digest.hexdigest()}, _fingerprint(after)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _read_symlink(repo, relative):
    parent, name = _parent_fd(repo, relative)
    try:
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISLNK(before.st_mode), 'substituted symlink: ' + relative)
        raw = os.fsencode(os.readlink(name, dir_fd=parent))
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(_fingerprint(before) == _fingerprint(after), 'symlink changed while reading: ' + relative)
        return {'working_mode': '120000', 'working_git_blob': hashlib.sha1(
                    b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest(),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, _fingerprint(after)
    finally:
        os.close(parent)


def _gitlink(repo, relative):
    parent, name = _parent_fd(repo, relative)
    try:
        try:
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return 'absent', None, None
        require(stat.S_ISDIR(info.st_mode), 'unsupported gitlink materialization: ' + relative)
        path = repo / relative
        head = None
        if (path / '.git').exists():
            actual_root = os.fsdecode(_git(path, 'rev-parse', '--show-toplevel').rstrip(b'\n'))
            require(Path(actual_root).resolve() == path.resolve(), 'gitlink repository root mismatch')
            head = _git(path, 'rev-parse', '--verify', 'HEAD^{commit}').decode('ascii').strip()
            require(HEX40.fullmatch(head), 'invalid materialized submodule HEAD')
        return 'directory', head, _fingerprint(info)
    finally:
        os.close(parent)


def _entries(index):
    entries = []
    seen = set()
    for raw in index.split(b'\0'):
        if not raw:
            continue
        header, raw_path = raw.split(b'\t', 1)
        mode, oid, stage = header.decode('ascii').split(' ')
        relative = _path(os.fsdecode(raw_path))
        require(stage == '0' and relative not in seen, 'unmerged or duplicate tracked path')
        require(mode in ('100644', '100755', '120000', '160000') and HEX40.fullmatch(oid),
                'unsupported tracked Git entry')
        seen.add(relative)
        entries.append((relative, mode, oid))
    return sorted(entries, key=lambda row: os.fsencode(row[0]))


def _confirm(checks):
    for repo, before, files in checks:
        for path, kind, fingerprint, submodule_head in files:
            if kind == 'gitlink':
                _, head, current = _gitlink(repo, path)
                require((current, head) == (fingerprint, submodule_head), 'gitlink changed during capture')
            else:
                parent, name = _parent_fd(repo, path)
                try:
                    current = _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
                finally:
                    os.close(parent)
                require(current == fingerprint, 'tracked file changed during capture: ' + path)
        require(_state(repo) == before, 'HEAD, index or dirty status changed during capture: ' + repo.name)


def _scan(root, names):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'invalid source root')
    root = root.resolve()
    records, checks = [], []
    for name in _names(names):
        repo = root / name
        require(repo.is_dir() and not repo.is_symlink(), 'missing or symlinked repository: ' + name)
        actual = os.fsdecode(_git(repo, 'rev-parse', '--show-toplevel').rstrip(b'\n'))
        require(Path(actual).resolve() == repo, 'repository root mismatch: ' + name)
        require(_git(repo, 'rev-parse', '--show-object-format').strip() == b'sha1', 'unsupported Git object format')
        before = _state(repo)
        index_entries = {path: (mode, oid) for path, mode, oid in _entries(before[1])}
        head_raw = _git(repo, 'ls-tree', '-r', '-z', '--full-tree', before[0])
        head_index = bytearray()
        for raw in head_raw.split(b'\0'):
            if not raw:
                continue
            header, raw_path = raw.split(b'\t', 1)
            mode, kind, oid = header.split(b' ')
            require((mode == b'160000' and kind == b'commit') or
                    (mode in (b'100644', b'100755', b'120000') and kind == b'blob'),
                    'unsupported HEAD tree entry')
            head_index.extend(mode + b' ' + oid + b' 0\t' + raw_path + b'\0')
        head_entries = {path: (mode, oid) for path, mode, oid in _entries(bytes(head_index))}
        entries = [(path, *(head_entries.get(path) or index_entries[path]))
                   for path in sorted(set(head_entries) | set(index_entries), key=os.fsencode)]
        oids = sorted({oid for mapping in (head_entries, index_entries)
                       for mode, oid in mapping.values() if mode != '160000'})
        if oids:
            reports = _git(repo, 'cat-file', '--batch-check',
                           input_bytes=('\n'.join(oids) + '\n').encode('ascii')).splitlines()
            require(len(reports) == len(oids), 'incomplete Git object availability result')
            for expected, report in zip(oids, reports):
                parts = report.decode('ascii').split()
                require(len(parts) == 3 and parts[0] == expected and parts[1] == 'blob'
                        and parts[2].isdigit(), 'missing or unsupported tracked Git object')
        rows, files = [], []
        for relative, mode, oid in entries:
            kind = 'gitlink' if mode == '160000' else 'symlink' if mode == '120000' else 'regular'
            row = {'path': relative, 'kind': kind, 'mode': mode, 'git_object': oid,
                   'head_git_object': head_entries.get(relative, (None, None))[1],
                   'index_git_object': index_entries.get(relative, (None, None))[1],
                   'head_mode': head_entries.get(relative, (None, None))[0],
                   'index_mode': index_entries.get(relative, (None, None))[0],
                   'working_mode': None, 'working_git_blob': None, 'bytes': None, 'sha256': None,
                   'materialization': 'file', 'submodule_head': None}
            if kind == 'gitlink':
                materialization, head, fingerprint = _gitlink(repo, relative)
                row.update(materialization=materialization, submodule_head=head)
            else:
                values, fingerprint = (_read_symlink if kind == 'symlink' else _read_regular)(repo, relative)
                row.update(values)
            rows.append(row)
            files.append((relative, kind, fingerprint, row['submodule_head']))
        status_bytes = before[2]
        records.append({'name': name, 'head': before[0], 'dirty': bool(status_bytes),
                        'status_porcelain_base64': base64.b64encode(status_bytes).decode('ascii'),
                        'files': rows})
        checks.append((repo, before, files))
    _confirm(checks)
    return records, checks


def _binding(binding, repos, root, cache, allow_historical=True):
    require(isinstance(binding, dict) and set(binding) in (BINDING_KEYS, BINDING_KEYS | {'scope'}),
            'invalid source binding fields')
    name = binding['repository']
    require(type(name) is str and name in repos, 'binding repository not inventoried')
    scope = binding.get('scope', 'current')
    require(scope in ('current', 'historical') and (allow_historical or scope == 'current'),
            'dormant historical scope cannot discharge an active CLOSED source')
    commit = binding['commit']
    require(type(commit) is str and HEX40.fullmatch(commit), 'immutable source commit required')
    require(type(binding['git_object']) is str and HEX40.fullmatch(binding['git_object'])
            and type(binding['sha256']) is str and HEX64.fullmatch(binding['sha256']),
            'invalid source object or SHA256 binding')
    path = _path(binding['path'])
    key = (name, commit, path)
    if key not in cache:
        directory = Path(root) / name
        resolved = _git(directory, 'rev-parse', '--verify', commit + '^{commit}').decode('ascii').strip()
        require(resolved == commit, 'immutable source commit resolution mismatch')
        tree = _git(directory, 'ls-tree', '-r', '-z', '--full-tree', commit, '--', ':(literal)' + path)
        entries = [raw for raw in tree.split(b'\0') if raw]
        require(len(entries) == 1, 'immutable source path missing or not a regular blob')
        metadata, raw_path = entries[0].split(b'\t', 1)
        mode, kind, oid = metadata.decode('ascii').split(' ')
        require(os.fsdecode(raw_path) == path and kind == 'blob' and mode in ('100644', '100755'),
                'immutable source path must be a regular Git blob')
        size = _git(directory, 'cat-file', '-s', oid).decode('ascii').strip()
        require(size.isdigit() and int(size) <= 256 * 1024 * 1024, 'bound blob exceeds 256 MiB limit')
        raw = _git(directory, 'cat-file', 'blob', oid)
        require(len(raw) == int(size) and hashlib.sha1(
            b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest() == oid,
            'immutable Git blob byte identity mismatch')
        cache[key] = {'path': path, 'kind': 'regular', 'git_object': oid,
                      'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), '_raw': raw}
    row = cache[key]
    require(binding['git_object'] == row['git_object'] and binding['sha256'] == row['sha256'],
            'immutable source binding byte/object identity mismatch')
    if scope == 'current':
        files = {record['path']: record for record in repos[name]['files']}
        require(path in files and files[path]['kind'] == 'regular', 'active source path is not currently tracked')
        current = files[path]
        require(current['head_git_object'] == current['working_git_blob'] == row['git_object']
                and current['sha256'] == row['sha256'], 'active source bytes differ from immutable binding')
    return row


def _bindings(values, repos, root, cache, allow_historical=True):
    require(isinstance(values, list) and values, 'nonempty source binding list required')
    seen = set()
    for binding in values:
        _binding(binding, repos, root, cache, allow_historical)
        identity = (binding['repository'], binding['commit'], binding['path'])
        require(identity not in seen, 'duplicate source binding')
        seen.add(identity)


def _bound_json(root, binding, repos, cache):
    row = _binding(binding, repos, root, cache)
    return strict_json(row['_raw'].decode('utf-8'))


def _obligations(values, repositories, root):
    require(isinstance(values, list), 'obligations must be a list')
    repos = {repo['name']: repo for repo in repositories}
    cache = {}
    seen = set()
    common = {'id', 'status', 'source_bindings'}
    quarantine = {'exact_boundary', 'blocking_reason', 'isolation', 'owner', 'maintenance'}
    for entry in values:
        require(isinstance(entry, dict), 'obligation must be an object')
        identifier = entry.get('id')
        require(type(identifier) is str and identifier.strip() and identifier not in seen,
                'missing or duplicate obligation id')
        seen.add(identifier)
        status_value = entry.get('status')
        require(status_value in ('CLOSED', 'QUARANTINED'), 'invalid obligation status')
        required = common | (quarantine if status_value == 'QUARANTINED' else {'kernel_evidence'})
        require(set(entry) == required, 'invalid obligation fields')
        _bindings(entry['source_bindings'], repos, root, cache,
                  allow_historical=status_value == 'QUARANTINED')
        if status_value == 'QUARANTINED':
            require(all(type(entry[key]) is str and entry[key].strip()
                        for key in ('exact_boundary', 'blocking_reason', 'isolation')),
                    'quarantine requires exact boundary, reason and isolation')
            require(entry['owner'] == 'none' and entry['maintenance'] == 'zero',
                    'quarantine must have owner none and maintenance zero')
        else:
            evidence = entry['kernel_evidence']
            kernel_keys = {'receipt', 'manifest', 'checked_commit', 'targets', 'source_bindings'}
            require(isinstance(evidence, dict) and set(evidence) in
                    (kernel_keys, kernel_keys | {'manifest_overlay'}),
                    'CLOSED requires actual kernel evidence bindings')
            _bindings(evidence['source_bindings'], repos, root, cache, allow_historical=False)
            require(all(binding in evidence['source_bindings'] for binding in entry['source_bindings']),
                    'kernel source bindings must cover declared obligation source bindings')
            checked = evidence['checked_commit']
            require(type(checked) is str and HEX40.fullmatch(checked), 'invalid checked kernel commit')
            targets = evidence['targets']
            require(isinstance(targets, list) and targets and
                    all(type(target) is str and target.strip() for target in targets) and
                    len(set(targets)) == len(targets), 'nonempty unique kernel target names required')
            manifest_row = _binding(evidence['manifest'], repos, root, cache)
            manifest_data = _bound_json(root, evidence['manifest'], repos, cache)
            manifest_inputs = [evidence['manifest']]
            manifest_digest = manifest_row['sha256']
            identity = manifest_data.get('source_identity') if isinstance(manifest_data, dict) else None
            rf312 = identity == {'schema': 'rf312-overlay-v1', 'overlay': 'manifest.overlay.json'}
            if identity is not None or 'manifest_overlay' in evidence:
                require(rf312 and 'manifest_overlay' in evidence,
                        'RF312 source identity requires its exact manifest overlay binding')
                overlay_binding = evidence['manifest_overlay']
                require(evidence['manifest']['repository'] == 'Math-' and
                        evidence['manifest']['path'] == 'formal/manifest.json' and
                        isinstance(overlay_binding, dict) and
                        overlay_binding.get('repository') == 'Math-' and
                        overlay_binding.get('path') == 'formal/manifest.overlay.json',
                        'only the exact Math RF312 manifest and sibling overlay are supported')
                require(isinstance(manifest_data.get('source_modules'), list) and
                        isinstance(manifest_data.get('files'), dict),
                        'RF312 base manifest requires its source-module and file inventories')
                overlay_row = _binding(overlay_binding, repos, root, cache)
                overlay_data = _bound_json(root, overlay_binding, repos, cache)
                require(isinstance(overlay_data, dict) and set(overlay_data) == {'schema_version', 'files'}
                        and type(overlay_data['schema_version']) is int and overlay_data['schema_version'] == 1
                        and isinstance(overlay_data['files'], dict), 'invalid RF312 overlay keys or schema')
                require(all(type(path) is str and type(digest) is str and HEX64.fullmatch(digest)
                            for path, digest in overlay_data['files'].items()),
                        'invalid RF312 overlay file binding')
                manifest_digest = hashlib.sha256(
                    manifest_row['_raw'] + b'\0RF312-OVERLAY\0' + overlay_row['_raw']).hexdigest()
                manifest_inputs.append(overlay_binding)
            receipt = _bound_json(root, evidence['receipt'], repos, cache)
            require(isinstance(receipt, dict) and receipt.get('formalization_status') == 'kernel-checked'
                    and receipt.get('checked_commit') == checked
                    and receipt.get('manifest_sha256') == manifest_digest,
                    'kernel receipt source/manifest binding mismatch')
            axioms = receipt.get('axioms')
            require(isinstance(axioms, dict) and all(target in axioms for target in targets),
                    'kernel target missing from actual receipt axiom-report keys')
            for target in targets:
                reported = axioms[target]
                require(type(reported) is list and all(type(name) is str and
                        name in APPROVED_KERNEL_AXIOMS for name in reported) and
                        len(set(reported)) == len(reported),
                        'invalid, duplicate or unapproved kernel axiom report: ' + target)
            declared = manifest_data.get('targets') if isinstance(manifest_data, dict) else None
            if rf312:
                require(isinstance(declared, list) and declared and
                        all(type(item) is str and item.strip() for item in declared),
                        'RF312 base manifest requires explicit target names')
                declared_names = declared
            else:
                require(isinstance(declared, list) and declared and all(
                    isinstance(item, dict) and type(item.get('name')) is str and item['name'].strip()
                    for item in declared), 'kernel manifest requires explicit target declarations')
                declared_names = [item['name'] for item in declared]
            require(len(set(declared_names)) == len(declared_names) and
                    all(target in declared_names for target in targets),
                    'kernel target missing or duplicated in actual manifest')
            kernel_repository = evidence['manifest']['repository']
            for binding in [*manifest_inputs, *evidence['source_bindings']]:
                require(binding['repository'] == kernel_repository,
                        'one kernel checked commit must bind one repository source cut')
                _binding({**binding, 'commit': checked, 'scope': 'historical'}, repos, root, cache)


def generate(root, names=DEFAULT_REPOSITORIES, obligations=None):
    """Capture source bytes and validate supplied declarations, without promoting them."""
    try:
        values = [] if obligations is None else obligations
        _control_types(values)
        repositories, checks = _scan(root, names)
        _obligations(values, repositories, root)
        _confirm(checks)
        return {'schema': 'cbz.inventory.v1', 'certification_scope': SCOPE,
                'semantic_claim_completeness': False, 'cbz_green': False,
                'repositories': repositories, 'obligations': sorted(values, key=lambda item: item['id'])}
    except (OSError, UnicodeError, KeyError, TypeError, OverflowError) as error:
        raise ValueError('inventory capture refused: ' + str(error)) from error


def verify(manifest, root, names=DEFAULT_REPOSITORIES):
    """Check complete declared repository/file sets and bound bytes; no semantic certification."""
    try:
        _control_types(manifest)
        require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS, 'invalid manifest fields')
        require(manifest['schema'] == 'cbz.inventory.v1' and manifest['certification_scope'] == SCOPE,
                'unsupported inventory schema or certification scope')
        require(manifest['semantic_claim_completeness'] is False and manifest['cbz_green'] is False,
                'inventory cannot claim semantic completeness or CBZ green')
        expected_names = _names(names)
        expected_repos = manifest['repositories']
        require(isinstance(expected_repos, list), 'repositories must be a list')
        actual_names = [repo.get('name') for repo in expected_repos if isinstance(repo, dict)]
        require(len(actual_names) == len(expected_repos) and len(set(actual_names)) == len(actual_names)
                and set(actual_names) == set(expected_names), 'missing, unexpected or duplicate repository')
        actual_repos, checks = _scan(root, expected_names)
        actual = {repo['name']: repo for repo in actual_repos}
        for repo in expected_repos:
            require(set(repo) == {'name', 'head', 'dirty', 'status_porcelain_base64', 'files'},
                    'invalid repository fields')
            name = repo['name']
            require(type(repo['dirty']) is bool, 'dirty state must be Boolean')
            for field in ('head', 'dirty', 'status_porcelain_base64'):
                require(repo[field] == actual[name][field], 'repository state mismatch: ' + name + '/' + field)
            require(isinstance(repo['files'], list), 'files must be a list')
            expected_files = {}
            for record in repo['files']:
                require(isinstance(record, dict) and set(record) == FILE_KEYS, 'invalid file record fields')
                path = _path(record['path'])
                require(path not in expected_files, 'duplicate inventory file path')
                if record['kind'] == 'gitlink':
                    require(record['bytes'] is None and record['sha256'] is None,
                            'gitlink cannot claim hydrated byte identities')
                else:
                    require(type(record['bytes']) is int and record['bytes'] >= 0,
                            'file byte count must be a nonnegative integer, not Boolean')
                expected_files[path] = record
            actual_files = {record['path']: record for record in actual[name]['files']}
            require(set(expected_files) == set(actual_files), 'missing or unexpected tracked file: ' + name)
            for path, record in expected_files.items():
                require(record == actual_files[path], 'file identity mismatch: ' + name + '/' + path)
        _obligations(manifest['obligations'], actual_repos, root)
        _confirm(checks)
        return {'inventory_verified': True, 'certification_scope': SCOPE,
                'semantic_claim_completeness': False, 'cbz_green': False,
                'repositories': len(actual_repos),
                'tracked_entries': sum(len(repo['files']) for repo in actual_repos)}
    except (OSError, UnicodeError, KeyError, TypeError, OverflowError) as error:
        raise ValueError('inventory verification refused: ' + str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    for operation in ('generate', 'verify'):
        sub = subparsers.add_parser(operation)
        sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--repositories', nargs='+', default=list(DEFAULT_REPOSITORIES))
        if operation == 'generate':
            sub.add_argument('--output', type=Path, required=True)
            sub.add_argument('--classifications', type=Path)
        else:
            sub.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        names = _names(args.repositories)
        if args.operation == 'verify':
            result = verify(_json_file(args.manifest), args.root, names)
        else:
            output = args.output.absolute()
            require(not output.exists() and not output.is_symlink(), 'output already exists')
            for name in names:
                require(not output.resolve().is_relative_to((args.root / name).resolve()),
                        'inventory output must be outside inventoried source repositories')
            obligations = []
            if args.classifications:
                classifications = _json_file(args.classifications)
                require(isinstance(classifications, dict) and set(classifications) == {'obligations'},
                        'classification input must contain exactly obligations')
                obligations = classifications['obligations']
            data = generate(args.root, names, obligations)
            raw = (json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + '\n').encode('ascii')
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.cbz-inventory-', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.link(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
            result = {'inventory_generated': True, 'cbz_green': False,
                      'semantic_claim_completeness': False, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'repositories': len(data['repositories']),
                      'tracked_entries': sum(len(repo['files']) for repo in data['repositories'])}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, UnicodeError, subprocess.SubprocessError, RecursionError) as error:
        print(json.dumps({'error': str(error), 'cbz_green': False}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
