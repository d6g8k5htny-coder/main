#!/usr/bin/env python3
"""Read-only recursive Git observations, not a clean-state or mathematical certificate.

Search every physical directory, including ignored/node_modules directories,
except Git metadata internals. Follow internal directory symlinks once by inode;
record external directory links as uncovered boundaries (incomplete/nonzero).
Git output uses NUL records. JSON ensure_ascii preserves arbitrary path bytes
through Python's filesystem surrogateescape convention. No network is needed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _text(raw):
    return os.fsdecode(raw)


def _line(raw):
    return _text(raw[:-1] if raw.endswith(b'\n') else raw)


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r'([A-Za-z][A-Za-z0-9+.-]*://)[^/\s]+@',
                   r'\1[REDACTED]@', value)
    value = re.sub(r'(?i)([?&](?:access_token|token|password|passwd|secret|signature|key|auth)=)[^&#\s]*',
                   r'\1[REDACTED]', value)
    return value


def _remote_url(value):
    value = _redact(value)
    if '://' not in value:
        value = re.sub(r'^[^/\s]+@(?=[^/:\s]+:)', '[REDACTED]@', value)
    return value


class Git:
    def __init__(self, git, command_timeout, total_timeout, max_command_bytes):
        self.executable = git
        self.command_timeout = command_timeout
        self.deadline = time.monotonic() + total_timeout
        self.max_bytes = max_command_bytes
        self.errors = []
        self.commands = 0
        self.metadata_cache = {}
        self.configuration_complete = True
        self.env = os.environ.copy()
        for name in ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE',
                     'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE']:
            self.env.pop(name, None)
        for name in list(self.env):
            if name in ('GIT_CONFIG_COUNT', 'GIT_CONFIG_PARAMETERS') or name.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')):
                self.env.pop(name)
        self.env.update(GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0', GIT_NO_LAZY_FETCH='1')
        self.base = ['core.fsmonitor=false', 'core.untrackedCache=false',
                     'core.hooksPath=/dev/null', 'maintenance.auto=false', 'gc.auto=0',
                     'core.quotePath=false', 'protocol.allow=never']
        self.base += ['protocol.' + name + '.allow=never'
                      for name in ['http', 'https', 'ssh', 'git', 'file', 'ext']]

    def error(self, kind, path, operation, **detail):
        diagnostics = {key: _redact(value) if key in ('message', 'stderr') else value
                       for key, value in detail.items()}
        self.errors.append({'kind': kind, 'path': str(path), 'operation': _redact(operation), **diagnostics})

    def expired(self):
        return time.monotonic() >= self.deadline

    def run(self, cwd, *args, allowed=(0,), settings=(), expected_stderr_on=()):
        operation = ' '.join(args[:3])
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.error('deadline', cwd, operation)
            return None
        command = [self.executable, '--no-optional-locks']
        # All override values here are fixed strings without '='. Split at the
        # LAST separator, then transport key/value separately (Git >= 2.31).
        # This also propagates intact through Git's recursive submodule status.
        environment = self.env.copy()
        overrides = [*self.base, *settings]
        environment['GIT_CONFIG_COUNT'] = str(len(overrides))
        for number, setting in enumerate(overrides):
            key, _, value = setting.rpartition('=')
            environment['GIT_CONFIG_KEY_' + str(number)] = key
            environment['GIT_CONFIG_VALUE_' + str(number)] = value
        command += ['-C', str(cwd), *args]
        self.commands += 1
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=environment, start_new_session=True)
        except OSError as exc:
            self.error('spawn', cwd, operation, message=str(exc))
            return None
        streams = {process.stdout: bytearray(), process.stderr: bytearray()}
        selector = selectors.DefaultSelector()
        for stream in streams:
            selector.register(stream, selectors.EVENT_READ)
        cutoff = time.monotonic() + min(self.command_timeout, remaining)
        failure = None
        try:
            while selector.get_map():
                if time.monotonic() >= cutoff:
                    failure = 'timeout'
                    break
                for key, _ in selector.select(min(.1, max(0, cutoff - time.monotonic()))):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if sum(len(b) for b in streams.values()) + len(chunk) > self.max_bytes:
                        failure = 'output_limit'
                        break
                    streams[key.fileobj].extend(chunk)
                if failure:
                    break
            if not failure:
                try:
                    code = process.wait(timeout=max(0, cutoff - time.monotonic()))
                except subprocess.TimeoutExpired:
                    failure = 'timeout'
            if failure:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.error('child_not_reaped', cwd, operation, pid=process.pid)
                    code = None
        finally:
            selector.close()
            for stream in streams:
                stream.close()
        out, err = bytes(streams[process.stdout]), bytes(streams[process.stderr])
        if failure:
            self.error(failure, cwd, operation, stderr=_text(err), output_bytes=len(out))
            return None
        if code not in allowed or (err and code not in expected_stderr_on):
            self.error('git_error' if code not in allowed else 'git_stderr', cwd, operation,
                       returncode=code, stderr=_text(err))
            return None
        return code, out


def _is_git_metadata(directory, git):
    try:
        directory = Path(directory).resolve()
    except (OSError, RuntimeError) as exc:
        git.error('filesystem', directory, 'metadata validation', message=str(exc))
        return False
    if directory not in git.metadata_cache:
        if not ((directory / 'HEAD').is_file() and
                ((directory / 'objects').is_dir() or (directory / 'commondir').is_file())):
            git.metadata_cache[directory] = False
            return False
        # --resolve-git-dir validates THIS candidate, without ancestor discovery.
        # Exit 128 is an expected negative probe, not permission to prune it.
        result = git.run(directory, 'rev-parse', '--resolve-git-dir', str(directory),
                         allowed=(0, 128), expected_stderr_on=(128,))
        exact = result is not None and result[0] == 0 and Path(_line(result[1])).resolve() == directory
        git.metadata_cache[directory] = exact
    return git.metadata_cache[directory]


def _discover(root, git):
    stack = [root]
    visited = set()
    roots = set()
    symlinks = []
    skipped_metadata = []
    while stack:
        if git.expired():
            git.error('deadline', root, 'recursive discovery', pending_directories=len(stack))
            break
        directory = stack.pop()
        try:
            stat = directory.stat()
            identity = stat.st_dev, stat.st_ino
            if identity in visited:
                continue
            visited.add(identity)
            if os.path.lexists(directory / '.git'):
                roots.add(str(directory.resolve()))
            # Prune only exact, validated Git metadata, including bare stores.
            if _is_git_metadata(directory, git):
                roots.add(str(directory.resolve()))
                skipped_metadata.append(str(directory))
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if git.expired():
                        git.error('deadline', directory, 'recursive discovery')
                        break
                    if entry.name == '.git' and _is_git_metadata(Path(entry.path), git):
                        skipped_metadata.append(entry.path)
                        continue
                    try:
                        if entry.is_symlink():
                            target = Path(entry.path).resolve()
                            is_directory = entry.is_dir()
                            record = {'path': entry.path, 'target': str(target),
                                      'directory': is_directory, 'exists': target.exists()}
                            symlinks.append(record)
                            if is_directory:
                                if any(_is_git_metadata(parent, git) for parent in target.parents):
                                    record['not_followed'] = 'Git metadata internals'
                                    skipped_metadata.append(str(target))
                                elif target.is_relative_to(root):
                                    stack.append(target)
                                else:
                                    git.error('external_directory_symlink', entry.path,
                                              'recursive discovery', target=str(target))
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except (OSError, RuntimeError) as exc:
                        git.error('filesystem', entry.path, 'recursive discovery', message=str(exc))
        except OSError as exc:
            git.error('filesystem', directory, 'recursive discovery', message=str(exc))
    return sorted(roots), symlinks, skipped_metadata, len(visited)


def _config(git, cwd, pattern, source=(), settings=()):
    result = git.run(cwd, 'config', *source, '--null', '--name-only', '--get-regexp', pattern,
                     allowed=(0, 1), settings=settings)
    if result is None:
        return {}
    values = {}
    for key in result[1].split(b'\0'):
        if not key:
            continue
        name = _text(key)
        value = git.run(cwd, 'config', *source, '--null', '--get-all', name, allowed=(0, 1), settings=settings)
        if value is not None:
            values[name] = [_text(v) for v in value[1].split(b'\0')[:-1]]
    return values


def _settings_and_remotes(git, cwd):
    # Reading config does not execute it. Disable filter programs for status;
    # URL overrides and protocol denial also prevent lazy partial-clone fetch.
    names = git.run(cwd, 'config', '--null', '--name-only', '--get-regexp',
                    r'^(filter\..*\.(clean|process|required)|remote\..*\.(url|pushurl|promisor)|protocol\..*\.allow)$',
                    allowed=(0, 1))
    settings = []
    remotes = {}
    if names is None:
        git.configuration_complete = False
        return settings, []
    for raw in names[1].split(b'\0'):
        if not raw:
            continue
        key = _text(raw)
        if key.startswith('filter.'):
            prefix = key.rsplit('.', 1)[0]
            settings += [prefix + '.clean=', prefix + '.process=', prefix + '.required=false']
        elif key.startswith('protocol.'):
            settings.append(key + '=never')
        elif key.startswith('remote.'):
            name, field = key[7:].rsplit('.', 1)
            settings += ['remote.' + name + '.url=', 'remote.' + name + '.pushurl=',
                         'remote.' + name + '.promisor=false']
            row = remotes.setdefault(name, {'name': name, 'fetch_urls': [], 'push_urls': []})
            if field in ('url', 'pushurl'):
                result = git.run(cwd, 'config', '--null', '--get-all', key, allowed=(0, 1))
                if result is not None:
                    row['fetch_urls' if field == 'url' else 'push_urls'] = [
                        _remote_url(_text(v)) for v in result[1].split(b'\0')[:-1]]
    return list(dict.fromkeys(settings)), list(remotes.values())


def _refs(git, cwd, settings, prefixes=()):
    result = git.run(cwd, 'for-each-ref',
        '--format=%(refname)%00%(objectname)%00%(objecttype)%00%(*objectname)%00%(*objecttype)%00%(symref)%00',
        *prefixes, settings=settings)
    rows = []
    if result is None:
        return rows
    fields = result[1].split(b'\0')
    if fields[-1] in (b'', b'\n'):
        fields.pop()
    if len(fields) % 6:
        git.error('parse', cwd, 'refs', message='Invalid NUL field count')
        return rows
    for i in range(0, len(fields), 6):
        names = ['name', 'object', 'type', 'peeled_object', 'peeled_type', 'symbolic_target']
        row = dict(zip(names, map(_text, fields[i:i + 6])))
        row['name'] = row['name'].removeprefix('\n')
        rows.append(row)
    return rows


def _registered_worktrees(git, cwd, settings):
    result = git.run(cwd, 'worktree', 'list', '--porcelain', '-z', settings=settings)
    rows = []
    if result is None:
        return rows
    for block in result[1].split(b'\0\0'):
        if not block:
            continue
        row = {}
        for field in block.split(b'\0'):
            if not field:
                continue
            key, _, value = field.partition(b' ')
            row[_text(key)] = (_redact(_text(value)) if key in (b'locked', b'prunable') else _text(value)) if value else True
        if 'worktree' not in row:
            git.error('parse', cwd, 'worktree list', message='Missing worktree path')
        else:
            rows.append(row)
    return rows


def _status(git, path, settings):
    if not git.configuration_complete:
        git.error('unsafe_configuration', path, 'status', message='Filter configuration discovery failed; status refused')
        return None
    # Fail closed if this Git does not honor separate config key/value transport,
    # including keys with '='. Inspect final effective values before any status.
    checks = ['core.fsmonitor=false', *(s for s in settings if s.startswith('filter.'))]
    for setting in dict.fromkeys(checks):
        key, _, expected = setting.rpartition('=')
        result = git.run(path, 'config', '--null', '--get-all', key, allowed=(0, 1), settings=settings)
        if (result is None or result[0] != 0 or not result[1].endswith(b'\0') or
                result[1].split(b'\0')[-2] != os.fsencode(expected)):
            git.error('unsafe_configuration', path, 'status', message='Configuration suppression could not be verified; status refused')
            return None
    result = git.run(path, 'status', '--porcelain=v1', '-z', '--untracked-files=all',
                     '--ignore-submodules=none', settings=settings)
    if result is None:
        return None
    fields = result[1].split(b'\0')
    if fields[-1] == b'':
        fields.pop()
    rows = []
    i = 0
    while i < len(fields):
        raw = fields[i]
        i += 1
        if len(raw) < 4 or raw[2:3] != b' ':
            git.error('parse', path, 'status', message='Invalid porcelain record')
            return None
        row = {'xy': _text(raw[:2]), 'path': _text(raw[3:])}
        if b'R' in raw[:2] or b'C' in raw[:2]:
            if i == len(fields):
                git.error('parse', path, 'status', message='Missing rename origin')
                return None
            row['original_path'] = _text(fields[i])
            i += 1
        rows.append(row)
    return rows


def _submodules(git, path, settings, bare, head):
    result = git.run(path, *(['ls-tree', '-r', '-z', 'HEAD'] if bare else ['ls-files', '--stage', '-z']),
                     settings=settings) if not bare or head else (0, b'')
    modules = {}
    if result is not None:
        for raw in result[1].split(b'\0'):
            if not raw:
                continue
            header, separator, name = raw.partition(b'\t')
            fields = header.split()
            if not separator or len(fields) != 3:
                git.error('parse', path, 'submodule index/tree', message='Invalid entry')
                continue
            if fields[0] == b'160000':
                modules[_text(name)] = {'path': _text(name), 'object': _text(fields[2] if bare else fields[1]),
                                       'index_stage': None if bare else int(fields[2])}
    source = ('--blob', 'HEAD:.gitmodules') if bare else ('--file', str(path / '.gitmodules'))
    if (bare and modules) or (not bare and (path / '.gitmodules').exists()):
        config = _config(git, path, r'^submodule\..*\.(path|url)$', source, settings)
        for key, values in config.items():
            if key.endswith('.path'):
                name = key[len('submodule.'):-len('.path')]
                for module_path in values:
                    row = modules.setdefault(module_path, {'path': module_path, 'object': None, 'index_stage': None})
                    row['name'] = name
                    row['urls'] = [_remote_url(url) for url in config.get('submodule.' + name + '.url', [])]
    for row in modules.values():
        row['initialized'] = False if bare else os.path.lexists(path / row['path'] / '.git')
    return sorted(modules.values(), key=lambda row: os.fsencode(row['path']))


def _preflight_submodule_settings(reader, candidates, settings):
    """Collect initialized submodule config before recursive worktree status."""
    all_settings = list(dict.fromkeys(settings))
    submodules_by_path = {}
    pending = list(sorted(candidates))
    visited = set()
    while pending:
        path = pending.pop()
        try:
            path = str(Path(path).resolve())
        except (OSError, RuntimeError) as exc:
            reader.error('filesystem', path, 'submodule configuration', message=str(exc))
            reader.configuration_complete = False
            continue
        if path in visited:
            continue
        visited.add(path)
        if reader.expired():
            reader.error('deadline', path, 'submodule configuration')
            reader.configuration_complete = False
            break
        if not Path(path).is_dir():
            continue
        bare_result = reader.run(path, 'rev-parse', '--is-bare-repository',
                                 settings=all_settings)
        head_result = reader.run(path, 'rev-parse', '--verify', '--quiet', 'HEAD',
                                 allowed=(0, 1), settings=all_settings)
        if bare_result is None or head_result is None:
            reader.configuration_complete = False
            continue
        bare = _line(bare_result[1]) == 'true'
        head = _line(head_result[1]) if head_result[0] == 0 else None
        errors_before = len(reader.errors)
        modules = _submodules(reader, Path(path), all_settings, bare, head)
        submodules_by_path[path] = modules
        if len(reader.errors) != errors_before:
            reader.configuration_complete = False
        for module in modules:
            if not module['initialized']:
                continue
            relative = Path(module['path'])
            if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
                reader.error('unsafe_submodule_path', path, 'submodule configuration',
                             module_path=module['path'])
                reader.configuration_complete = False
                continue
            child_path = Path(path) / relative
            try:
                child = str(child_path.resolve())
            except (OSError, RuntimeError) as exc:
                reader.error('filesystem', child_path, 'submodule configuration',
                             message=str(exc))
                reader.configuration_complete = False
                continue
            if child in visited:
                continue
            child_settings, _ = _settings_and_remotes(reader, child)
            all_settings = list(dict.fromkeys([*all_settings, *child_settings]))
            pending.append(child)
    return all_settings, submodules_by_path


def inventory(root, *, git='git', command_timeout=10, total_timeout=1800, max_command_bytes=32 * 1024 * 1024):
    reader = Git(git, command_timeout, total_timeout, max_command_bytes)
    try:
        root = Path(root).resolve()
    except (OSError, RuntimeError) as exc:
        reader.error('filesystem', root, 'resolve root', message=str(exc))
        root = Path(os.path.abspath(root))
    report = {'schema_version': 1, 'started_at': _now(), 'root': str(root),
        'purpose': 'Observed Git census only; not a clean-state or mathematical certification',
        'non_atomic_observations': True, 'complete': False, 'errors': reader.errors,
        'policy': {'walk_exclusions': ['Validated exact Git metadata internals only; node_modules is searched'],
                   'symlinks': 'Follow internal directory targets once by device/inode; record all links; external directory targets are uncovered/incomplete',
                   'paths': 'NUL Git records; filesystem surrogateescape; JSON ensure_ascii=true',
                   'read_only': 'Optional locks, hooks, filters, maintenance and lazy network fetch disabled; no Git mutation commands',
                   'status': 'porcelain v1 -z --untracked-files=all; Git ignore rules still apply',
                   'url_redaction': 'Remote URLs and diagnostic values only; filesystem identities preserved'},
        'limits': {'git_timeout_seconds': command_timeout, 'total_timeout_seconds': total_timeout,
                   'max_command_bytes': max_command_bytes}, 'stores': [], 'worktrees': []}
    roots, links, skipped, directory_count = _discover(root, reader)
    report.update(discovered_roots=roots, symlinks=links, git_metadata_not_traversed=skipped,
                  directories_visited=directory_count, empty_discovery=not roots)
    stores = {}
    candidates = {}
    for path in roots:
        if reader.expired():
            reader.error('deadline', path, 'repository census')
            break
        common = reader.run(path, 'rev-parse', '--path-format=absolute', '--git-common-dir')
        if common is None:
            continue
        common_path = str(Path(_line(common[1])).resolve())
        if common_path not in stores:
            settings, remotes = _settings_and_remotes(reader, path)
            bare = reader.run(common_path, 'rev-parse', '--is-bare-repository', settings=settings)
            shallow = reader.run(common_path, 'rev-parse', '--is-shallow-repository', settings=settings)
            refs = [ref for ref in _refs(reader, path, settings)
                    if not ref['name'].startswith(('refs/worktree/', 'refs/bisect/', 'refs/rewritten/'))]
            registered = _registered_worktrees(reader, path, settings)
            store = {'path': common_path, 'representative': path, 'refs': refs, 'remotes': remotes,
                     'bare': None if bare is None else _line(bare[1]) == 'true',
                     'shallow': None if shallow is None else _line(shallow[1]) == 'true',
                     'registered_worktrees': registered, 'stashes': []}
            if any(row['name'] == 'refs/stash' for row in refs):
                stashes = reader.run(path, 'stash', 'list', '-z', '--format=%H%x00%gd', settings=settings)
                if stashes is not None:
                    fields = stashes[1].split(b'\0')
                    if fields[-1] == b'':
                        fields.pop()
                    if len(fields) % 2:
                        reader.error('parse', path, 'stash list', message='Invalid NUL field count')
                    else:
                        store['stashes'] = [{'object': _text(fields[i]), 'selector': _text(fields[i + 1])}
                                            for i in range(0, len(fields), 2)]
            stores[common_path] = store, settings
            for row in registered:
                candidates[str(Path(row['worktree']).resolve())] = common_path
        inside = reader.run(path, 'rev-parse', '--is-inside-work-tree')
        if inside is not None and _line(inside[1]) == 'true':
            candidates[path] = common_path
    # Include worktree-specific config, then propagate safe filter/remote
    # overrides through Git's own recursive submodule status subprocesses too.
    per_worktree_settings = {}
    per_worktree_remotes = {}
    for path in sorted(candidates):
        if reader.expired():
            reader.error('deadline', path, 'worktree configuration')
            break
        if Path(path).is_dir():
            per_worktree_settings[path], per_worktree_remotes[path] = _settings_and_remotes(reader, path)
    all_settings = list(dict.fromkeys(setting for _, settings in stores.values()
                                      for setting in settings))
    all_settings += list(dict.fromkeys(setting for settings in per_worktree_settings.values()
                                       for setting in settings))
    all_settings, pre_status_submodules = _preflight_submodule_settings(
        reader, candidates, all_settings)
    for path, common in sorted(candidates.items()):
        if reader.expired():
            reader.error('deadline', path, 'worktree census')
            break
        work = {'path': path, 'common_store': common, 'observed_at': _now(),
                'exists': Path(path).is_dir(), 'head': None, 'branch': None,
                'status': None, 'submodules': [], 'remotes': per_worktree_remotes.get(path, [])}
        report['worktrees'].append(work)
        if not work['exists']:
            reader.error('missing_worktree', path, 'worktree census')
            continue
        settings = list(dict.fromkeys([*stores[common][1], *all_settings]))
        bare = reader.run(path, 'rev-parse', '--is-bare-repository', settings=settings)
        work['bare'] = None if bare is None else _line(bare[1]) == 'true'
        git_dir = reader.run(path, 'rev-parse', '--path-format=absolute', '--git-dir', settings=settings)
        work['git_dir'] = None if git_dir is None else _line(git_dir[1])
        head = reader.run(path, 'rev-parse', '--verify', '--quiet', 'HEAD', allowed=(0, 1), settings=settings)
        branch = reader.run(path, 'symbolic-ref', '--quiet', 'HEAD', allowed=(0, 1), settings=settings)
        work['head'] = _line(head[1]) if head and head[0] == 0 else None
        work['branch'] = _line(branch[1]) if branch and branch[0] == 0 else None
        work['detached'] = branch is not None and branch[0] == 1
        work['unborn'] = work['head'] is None and work['branch'] is not None
        work['local_refs'] = _refs(reader, path, settings,
                                   ('refs/worktree', 'refs/bisect', 'refs/rewritten'))
        if head is not None and work['head'] is None and work['branch'] is None:
            reader.error('unresolved_head', path, 'HEAD identity')
        if work['bare'] is False:
            work['status'] = _status(reader, Path(path), settings)
        work['submodules'] = pre_status_submodules.get(path)
        if work['submodules'] is None:
            work['submodules'] = _submodules(reader, Path(path), settings, work['bare'], work['head'])
        final_head = reader.run(path, 'rev-parse', '--verify', '--quiet', 'HEAD', allowed=(0, 1), settings=settings)
        if head is not None and final_head is not None and head != final_head:
            reader.error('changed_during_observation', path, 'HEAD identity')
    report['stores'] = [stores[path][0] for path in sorted(stores)]
    report.update(finished_at=_now(), git_commands=reader.commands, complete=not reader.errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--git', default='git')
    parser.add_argument('--command-timeout', type=float, default=10)
    parser.add_argument('--total-timeout', type=float, default=1800)
    parser.add_argument('--max-command-bytes', type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()
    if (not math.isfinite(args.command_timeout) or not math.isfinite(args.total_timeout) or
        args.command_timeout <= 0 or args.total_timeout <= 0 or args.max_command_bytes <= 0):
        parser.error('All resource limits must be positive and finite')
    report = inventory(args.root, git=args.git, command_timeout=args.command_timeout,
                       total_timeout=args.total_timeout, max_command_bytes=args.max_command_bytes)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(report, output, ensure_ascii=True, indent=2)
        output.write('\n')
    print(json.dumps({'output': str(args.output), 'complete': report['complete'],
        'stores': len(report['stores']), 'worktrees': len(report['worktrees']),
        'errors': len(report['errors'])}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
