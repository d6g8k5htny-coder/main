#!/usr/bin/env python3
"""Hosted-only exact asset rewrite proposal capture; never applies a proposal."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time

PRODUCER = 'tools/site_asset_release.py'
PRODUCER_SHA = '390ea41ad0b4426913757c186e341dfcc1d1abcdafce505f51607a3dd31c0ffd'
FREEZE = 'tests/fixtures/current_site_transition_candidate_20261022/production-source-freeze.json'
MODULES = {'docs/site/current-model.mjs', 'docs/site/current.mjs', 'docs/site/dependency-model.mjs'}
ROOTS = ('docs/site/', 'docs/public-math/')
EDITABLE = {'.html', '.js', '.mjs', '.css'}
FILE_LIMIT = 16 * 1024 * 1024
NAMESPACE_LIMIT = 256 * 1024 * 1024
BODY_LIMIT = 512 * 1024
COUNT_LIMIT = 128
CONTEXT_LIMIT = 65536
STDIO_LIMIT = 1024 * 1024
CARRIER_LIMIT = 950000
CHUNK = 3072

class Hold(Exception):
    pass

def require(condition, code):
    if not condition:
        raise Hold(code)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def pin(raw):
    return {'bytes': len(raw), 'sha256': digest(raw),
            'git_blob': hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()}

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result

def load_json(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object)

def clean_path(name):
    require(isinstance(name, str) and len(name) <= 4096 and
            re.fullmatch(r'[A-Za-z0-9_./-]+', name) is not None and
            all(part not in ('', '.', '..') for part in name.split('/')) and
            not PurePosixPath(name).is_absolute(), 'UNSUPPORTED_TRACKED_PATH')
    return name

def safe_read(repo, name, limit):
    """Open every ancestor relative to an owned directory fd, never a symlink."""
    parts = clean_path(name).split('/')
    directory = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 <= before.st_size <= limit, 'UNSAFE_OR_OVERSIZED_FILE')
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            require(size <= limit, 'PHYSICAL_READ_LIMIT')
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size,
                              s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and size == after.st_size, 'FILE_CHANGED_DURING_READ')
        return b''.join(chunks), identity(after)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)

def child(command, repo, work, label, limit, seconds, deadline):
    require(time.monotonic() < deadline, 'PARENT_DEADLINE')
    target = work / (label + '.stdout')
    errors = work / (label + '.stderr')
    stream_limit = limit // 2
    process = None
    status, code = 'SETUP_ERROR', None
    def physical_cap():
        resource.setrlimit(resource.RLIMIT_FSIZE, (stream_limit, stream_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    environment = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'TZ') if key in os.environ}
    environment.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
                        'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    started = time.monotonic()
    try:
        with target.open('xb') as stream, errors.open('xb') as error_stream:
            process = subprocess.Popen(command, cwd=repo, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=error_stream,
                                       start_new_session=True, preexec_fn=physical_cap)
            try:
                code = process.wait(timeout=min(seconds, max(0.01, deadline - time.monotonic())))
                status = 'COMPLETED' if code == 0 else 'CHILD_ERROR'
            except subprocess.TimeoutExpired:
                status = 'TIMEOUT'
    except OSError:
        status = 'SETUP_ERROR'
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
                code = process.returncode
            except subprocess.TimeoutExpired:
                status = 'TERMINATION_UNCONFIRMED'
        if target.exists():
            raw, _ = safe_read(work, target.name, stream_limit)
        else:
            raw = b''
        if errors.exists():
            stderr, _ = safe_read(work, errors.name, stream_limit)
        else:
            stderr = b''
    if len(raw) >= stream_limit or len(stderr) >= stream_limit:
        status = 'OUTPUT_CAP'
    result = {'phase': label, 'status': status, 'returncode': code,
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'stdout': {'bytes': len(raw), 'sha256': digest(raw)},
              'stderr': {'bytes': len(stderr), 'sha256': digest(stderr)},
              'stdio_emitted': False, 'deadline_seconds': seconds,
              'physical_stdio_combined_limit': limit, 'physical_stdio_stream_limit': stream_limit}
    return result, raw

def git(repo, work, arguments, serial, deadline):
    serial[0] += 1
    result, raw = child(['git', '-C', str(repo), *arguments], repo, work,
                        'git-%04d' % serial[0], 4 * 1024 * 1024, 10, deadline)
    require(result['status'] == 'COMPLETED', 'GIT_CAPTURE_FAILED')
    return raw

def tracked(repo, work, serial, deadline):
    raw = git(repo, work, ['ls-files', '--stage', '-z'], serial, deadline)
    approved, all_names, indexed = [], set(), {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        prefix, separator, name_raw = entry.partition(b'\t')
        require(separator == b'\t', 'INVALID_INDEX_RECORD')
        mode, blob, stage = prefix.decode('ascii').split(' ')
        name = name_raw.decode('utf-8')
        require(name not in all_names and stage == '0', 'DUPLICATE_OR_UNMERGED_INDEX')
        all_names.add(name)
        if name.startswith(ROOTS):
            clean_path(name)
            require(mode in ('100644', '100755') and re.fullmatch(r'[0-9a-f]{40}', blob), 'NONORDINARY_TRACKED_INPUT')
            approved.append(name)
            indexed[name] = blob
    require(approved and len(approved) <= 65536 and any(n.startswith('docs/site/') for n in approved), 'INVENTORY_LIMIT')
    return raw, sorted(approved), indexed

def snapshot(repo, names):
    result, total = {}, 0
    for name in names:
        raw, identity = safe_read(repo, name, FILE_LIMIT)
        total += len(raw)
        require(total <= NAMESPACE_LIMIT, 'NAMESPACE_BYTE_LIMIT')
        result[name] = {'pin': pin(raw), 'identity': identity}
    return result

def check_clean(repo, work, serial, deadline):
    require(not git(repo, work, ['diff', '--name-status', '-z', '--no-renames', 'HEAD', '--'], serial, deadline), 'DIRTY_TRACKED_BASELINE')
    require(not git(repo, work, ['ls-files', '--others', '-z'], serial, deadline), 'UNTRACKED_BASELINE')

def frame_set(context, records, bodies):
    context_raw, records_raw = encoded(context), encoded(records)
    require(len(context_raw) <= CONTEXT_LIMIT and len(records_raw) <= CONTEXT_LIMIT, 'CONTEXT_OR_RECORD_LIMIT')
    members = [('context.json', context_raw), ('changes.json', records_raw), *bodies]
    lines, aggregate, total = [], hashlib.sha256(b'CS_ASSET_SET_V1\0'), 0
    for name, raw in members:
        sha, count = digest(raw), math.ceil(len(raw) / CHUNK)
        header = 'name=%s bytes=%d sha256=%s chunks=%d' % (name, len(raw), sha, count)
        lines.append('CS_ASSET_BEGIN ' + header)
        for index in range(count):
            value = base64.b64encode(raw[index * CHUNK:(index + 1) * CHUNK]).decode('ascii')
            lines.append('CS_ASSET_CHUNK name=%s index=%d data=%s' % (name, index, value))
        lines.append('CS_ASSET_END ' + header)
        aggregate.update(('%s\0%d\0%s\0' % (name, len(raw), sha)).encode('ascii'))
        total += len(raw)
    lines.append('CS_ASSET_COMPLETE files=%d bytes=%d sha256=%s disposition=%s' %
                 (len(members), total, aggregate.hexdigest(), context['disposition']))
    carrier = ('\n'.join(lines) + '\n').encode('ascii')
    require(len(carrier) <= CARRIER_LIMIT, 'FRAME_CARRIER_LIMIT')
    return carrier

def observe(repo, work, context, deadline):
    serial = [0]
    expected = os.environ.get('SITE_ASSET_CANDIDATE_FREEZE_SHA256', '')
    require(re.fullmatch(r'[0-9a-f]{64}', expected), 'UNBOUND_CANDIDATE_FREEZE')
    context['candidate_freeze_sha256'] = expected
    immutable_sources = {}
    for label, source_path in (('capture_source', 'tools/site_asset_rewrite_observation.py'),
                               ('workflow_source', '.github/workflows/site-asset-rewrite-proposal.yml')):
        source_raw, _ = safe_read(repo, source_path, 32768)
        context[label] = {'path': source_path, **pin(source_raw)}
        immutable_sources[source_path] = source_raw
    require(os.environ.get('GITHUB_EVENT_NAME') == 'pull_request' and
            os.environ.get('GITHUB_REPOSITORY') == 'd6g8k5htny-coder/main', 'WRONG_EVENT_OR_REPOSITORY')
    event_path = Path(os.environ['GITHUB_EVENT_PATH'])
    require(event_path.is_absolute(), 'INVALID_EVENT_PATH')
    event, _ = safe_read(event_path.parent, event_path.name, 1024 * 1024)
    event = load_json(event)
    pr = event['pull_request']
    number, head, base = event['number'], pr['head']['sha'], pr['base']['sha']
    require(type(number) is int and 1 <= number < 10**12 and
            re.fullmatch(r'[0-9a-f]{40}', head) and re.fullmatch(r'[0-9a-f]{40}', base), 'INVALID_PR_CONTEXT')
    checked = git(repo, work, ['rev-parse', 'HEAD'], serial, deadline).decode('ascii').strip()
    require(checked == os.environ.get('GITHUB_SHA') and re.fullmatch(r'[0-9a-f]{40}', checked) and
            os.environ.get('GITHUB_REF') == 'refs/pull/%d/merge' % number, 'WRONG_TESTED_COMMIT')
    parents = git(repo, work, ['rev-list', '--parents', '-n', '1', 'HEAD'], serial, deadline).decode('ascii').split()
    require(parents == [checked, base, head], 'TESTED_MERGE_PARENTS_MISMATCH')
    context.update({'event_name': 'pull_request', 'repository': 'd6g8k5htny-coder/main',
                    'event_ref': os.environ['GITHUB_REF'], 'event_sha': os.environ['GITHUB_SHA'],
                    'pull_request_number': number, 'head_sha': head, 'base_sha': base,
                    'tested_commit': checked, 'tested_merge_parents': parents[1:]})
    for key in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        value = os.environ.get(key, '')
        require(re.fullmatch(r'[1-9][0-9]{0,19}', value), 'INVALID_RUN_CONTEXT')
        context[key.lower()] = value
    producer, _ = safe_read(repo, PRODUCER, 32768)
    require(len(producer) == 26305 and digest(producer) == PRODUCER_SHA and
            pin(producer)['git_blob'] == '6c7c1642745d7488ff42a1842ec0ed6b49b42951', 'PRODUCER_SOURCE_MISMATCH')
    context['producer'] = {'path': PRODUCER, **pin(producer)}
    freeze_raw, _ = safe_read(repo, FREEZE, 16384)
    require(digest(freeze_raw) == expected, 'CANDIDATE_FREEZE_MISMATCH')
    freeze = load_json(freeze_raw)
    require(set(freeze) == {'schema_version', 'candidate_base_commit', 'files'} and
            type(freeze['schema_version']) is int and freeze['schema_version'] == 1 and freeze['candidate_base_commit'] == base and
            isinstance(freeze['files'], list) and len(freeze['files']) == 3, 'INVALID_CANDIDATE_FREEZE')
    found = set()
    for entry in freeze['files']:
        require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256', 'git_blob'} and
                entry['path'] in MODULES and entry['path'] not in found and
                type(entry['bytes']) is int and 0 <= entry['bytes'] <= FILE_LIMIT and
                isinstance(entry['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) and
                isinstance(entry['git_blob'], str) and re.fullmatch(r'[0-9a-f]{40}', entry['git_blob']), 'INVALID_CANDIDATE_MEMBER')
        raw, _ = safe_read(repo, entry['path'], FILE_LIMIT)
        require(pin(raw) == {key: entry[key] for key in ('bytes', 'sha256', 'git_blob')}, 'CANDIDATE_SOURCE_MISMATCH')
        found.add(entry['path'])
    require(found == MODULES, 'CANDIDATE_MEMBER_SET_MISMATCH')
    context['candidate_freeze'] = {'path': FREEZE, **pin(freeze_raw)}
    context['candidate_modules'] = freeze['files']
    check_clean(repo, work, serial, deadline)
    before_index, names, indexed = tracked(repo, work, serial, deadline)
    before = snapshot(repo, names)
    require(all(before[name]['pin']['git_blob'] == indexed[name] for name in names), 'WORKTREE_INDEX_BYTES_MISMATCH')
    context['inventory_before_sha256'] = digest(encoded({name: before[name]['pin'] for name in names}))
    context['inventory_count'] = len(names)
    context['source_guard'] = 'passed'
    rewritten = None
    for phase, arguments in (('rewrite', []), ('check', ['--check'])):
        producer_now, _ = safe_read(repo, PRODUCER, 32768)
        freeze_now, _ = safe_read(repo, FREEZE, 16384)
        require(producer_now == producer and freeze_now == freeze_raw, 'PHASE_IMMUTABLE_INPUT_CHANGED')
        result, _ = child([sys.executable, '-I', '-B', '-S', PRODUCER, *arguments], repo, work,
                          phase, STDIO_LIMIT, 90, deadline)
        context['producer_phases'].append(result)
        require(result['status'] == 'COMPLETED', 'PRODUCER_' + phase.upper() + '_' + result['status'])
        phase_index, phase_names, _ = tracked(repo, work, serial, deadline)
        require(phase_index == before_index and phase_names == names, 'PHASE_MEMBERSHIP_OR_INDEX_CHANGED')
        require(not git(repo, work, ['ls-files', '--others', '-z'], serial, deadline), 'PHASE_UNTRACKED_OUTPUT')
        phase_state = snapshot(repo, names)
        phase_diff = git(repo, work, ['diff', '--name-status', '-z', '--no-renames', 'HEAD', '--'], serial, deadline).split(b'\0')
        if phase_diff and phase_diff[-1] == b'':
            phase_diff.pop()
        require(len(phase_diff) % 2 == 0, 'PHASE_INVALID_DIFF')
        phase_changes = sorted(name for name in names if phase_state[name]['pin'] != before[name]['pin'])
        require(len(phase_changes) <= COUNT_LIMIT and
                all(name.startswith('docs/site/') and PurePosixPath(name).suffix in EDITABLE for name in phase_changes),
                'PHASE_INELIGIBLE_CHANGE_OR_COUNT')
        require(all(before[name]['identity'][2] == phase_state[name]['identity'][2] for name in names), 'PHASE_MODE_CHANGED')
        require(all(phase_diff[index] == b'M' for index in range(0, len(phase_diff), 2)) and
                sorted(phase_diff[index + 1].decode('utf-8') for index in range(0, len(phase_diff), 2)) == phase_changes,
                'PHASE_ADD_DELETE_RENAME_TYPE_OR_OUTSIDE_CHANGE')
        producer_after, _ = safe_read(repo, PRODUCER, 32768)
        freeze_after, _ = safe_read(repo, FREEZE, 16384)
        require(producer_after == producer and freeze_after == freeze_raw, 'PHASE_IMMUTABLE_INPUT_CHANGED')
        for source_path, source_before in immutable_sources.items():
            source_after, _ = safe_read(repo, source_path, 32768)
            require(source_after == source_before, 'PHASE_CAPTURE_OR_WORKFLOW_CHANGED')
        context[phase + '_inventory_sha256'] = digest(encoded({name: phase_state[name]['pin'] for name in names}))
        if phase == 'rewrite':
            rewritten = phase_state
        else:
            require(phase_state == rewritten, 'CHECK_PHASE_CHANGED_SOURCE')
    after_index, after_names, _ = tracked(repo, work, serial, deadline)
    require(after_index == before_index and after_names == names, 'TRACKED_MEMBERSHIP_OR_INDEX_CHANGED')
    require(not git(repo, work, ['ls-files', '--others', '-z'], serial, deadline), 'UNTRACKED_OUTPUT')
    after = snapshot(repo, names)
    context['inventory_after_sha256'] = digest(encoded({name: after[name]['pin'] for name in names}))
    changed = []
    for name in names:
        require(before[name]['identity'][2] == after[name]['identity'][2], 'FILE_MODE_CHANGED')
        if before[name]['pin'] != after[name]['pin']:
            require(name.startswith('docs/site/') and PurePosixPath(name).suffix in EDITABLE, 'INELIGIBLE_CHANGED_FILE')
            changed.append(name)
    require(len(changed) <= COUNT_LIMIT, 'CHANGED_FILE_COUNT_LIMIT')
    difference = git(repo, work, ['diff', '--name-status', '-z', '--no-renames', 'HEAD', '--'], serial, deadline).split(b'\0')
    if difference and difference[-1] == b'':
        difference.pop()
    require(len(difference) % 2 == 0, 'INVALID_DIFF_RECORDS')
    diff_names = []
    for index in range(0, len(difference), 2):
        require(difference[index] == b'M', 'ADD_DELETE_RENAME_TYPE_OR_OUTSIDE_CHANGE')
        diff_names.append(difference[index + 1].decode('utf-8'))
    require(sorted(diff_names) == changed, 'OUTSIDE_OR_UNACCOUNTED_CHANGE')
    records, bodies, total = [], [], 0
    for index, name in enumerate(changed):
        raw, identity = safe_read(repo, name, min(FILE_LIMIT, BODY_LIMIT - total))
        require(pin(raw) == after[name]['pin'] and identity == after[name]['identity'], 'POST_SNAPSHOT_READBACK_MISMATCH')
        total += len(raw)
        require(total <= BODY_LIMIT, 'CHANGED_BODY_AGGREGATE_LIMIT')
        body_name = 'body-%04d.bin' % index
        records.append({'path': name, 'body_name': body_name,
                        'preimage': before[name]['pin'], 'proposed': after[name]['pin']})
        bodies.append((body_name, raw))
    # Revalidate complete membership and diff after all exact body readbacks.
    final_index, final_names, _ = tracked(repo, work, serial, deadline)
    require(final_index == before_index and final_names == names, 'FINAL_MEMBERSHIP_CHANGED')
    final = snapshot(repo, names)
    require(final == after, 'FINAL_NAMESPACE_CHANGED')
    require(not git(repo, work, ['ls-files', '--others', '-z'], serial, deadline), 'FINAL_UNTRACKED_OUTPUT')
    require(git(repo, work, ['diff', '--name-status', '-z', '--no-renames', 'HEAD', '--'], serial, deadline).split(b'\0')[:-1] == difference,
            'FINAL_DIFF_CHANGED')
    context.update({'changed_file_count': len(records), 'changed_body_bytes': total,
                    'complete_set': True, 'disposition': 'PROPOSAL_COMPLETE_REQUIRES_REVIEW'})
    return records, bodies

def main():
    context = {'schema_version': 1, 'scientific_effect': 'NONE', 'source_guard': 'not_started',
               'disposition': 'HOLD', 'complete_set': False, 'producer_phases': [],
               'limits': {'changed_bytes': BODY_LIMIT, 'changed_files': COUNT_LIMIT,
                          'context_bytes': CONTEXT_LIMIT, 'carrier_bytes': CARRIER_LIMIT,
                          'chunk_bytes': CHUNK, 'producer_seconds_each': 90, 'producer_stdio_bytes_each': STDIO_LIMIT},
               'python_version': '.'.join(map(str, sys.version_info[:3])),
               'producer_commands': [['python3', '-I', '-B', '-S', PRODUCER],
                                     ['python3', '-I', '-B', '-S', PRODUCER, '--check']]}
    records, bodies = [], []
    try:
        repo = Path.cwd().resolve(strict=True)
        temporary = Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
        require(not temporary.is_relative_to(repo), 'TEMP_INSIDE_CHECKOUT')
        with tempfile.TemporaryDirectory(prefix='site-asset-proposal-', dir=temporary) as directory:
            work = Path(directory)
            os.chmod(work, 0o700)
            records, bodies = observe(repo, work, context, time.monotonic() + 270)
    except Hold as error:
        context['failure_code'] = str(error)
    except Exception:
        context['failure_code'] = 'UNEXPECTED_SOURCE_OR_SETUP_ERROR'
    if 'failure_code' in context:
        context.update({'disposition': 'HOLD', 'complete_set': False,
                        'changed_file_count': 0, 'changed_body_bytes': 0})
        records, bodies = [], []
    try:
        carrier = frame_set(context, records, bodies)
    except Exception:
        context.update({'disposition': 'HOLD', 'complete_set': False, 'failure_code': 'REPORT_OR_CARRIER_LIMIT',
                        'changed_file_count': 0, 'changed_body_bytes': 0})
        carrier = frame_set(context, [], [])
    sys.stdout.buffer.write(carrier)
    sys.stdout.buffer.flush()
    return 0 if context['disposition'] == 'PROPOSAL_COMPLETE_REQUIRES_REVIEW' else 1

if __name__ == '__main__':
    raise SystemExit(main())
