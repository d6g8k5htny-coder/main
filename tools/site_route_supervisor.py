#!/usr/bin/env python3
"""External Linux process supervisor and complete bounded log carrier.

Source proposal: independent Source review and hosted controls are mandatory.
No receipt is authenticated offline. A browser result needs external Root joins.
"""
from __future__ import annotations
import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

MIB = 1024 * 1024
OUTPUT_MAX, REPORT_MAX, MANIFEST_MAX = 16 * MIB, 2 * MIB, 256 * 1024
RAW_MAX, WIRE_MAX, CHUNK = 19 * MIB, 28 * MIB, 3072
PIPE_MAX, MEMBER_MAX = 64 * 1024, 256
PREFIX = b'SITE_FRAME_V1 '
HEX40, HEX64 = re.compile(r'[0-9a-f]{40}\Z'), re.compile(r'[0-9a-f]{64}\Z')
HELPER_SHA = 'ba90f005bceebec28a249ebc29df220d0890b1173054396402871767b1fe8e78'
REQ_SHA = '05b3a77f3aa648cdcee6c1dea416ac85754a9b3b417456ba451984d16d10738a'
PAGES = ('cite', 'dependencies', 'explore', 'formal', 'index', 'measure', 'museum',
         'reproduce', 'research', 'workspace')
RECEIPT = re.compile(r'https://github\.com/d6g8k5htny-coder/main/(?:pull|issues)/[1-9][0-9]*#issuecomment-[1-9][0-9]*\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def packed(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def strict(raw):
    def pairs(rows):
        out = {}
        for k, v in rows:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}



def same_json(left, right):
    """Type-preserving JSON comparison; Python True == 1 is not identity."""
    return packed(left) == packed(right)


def typed_identity(value, maximum):
    require(type(value) is dict and set(value) == {'bytes','sha256'} and
        type(value['bytes']) is int and 0 <= value['bytes'] <= maximum and
        type(value['sha256']) is str and HEX64.fullmatch(value['sha256']),
        'strict typed byte identity')


def typed_context(value):
    keys = {'schema_version','head_sha','base_sha','tested_merge_sha',
        'loader_qualification','supervisor_qualification','source_manifest_sha256',
        'helper_sha256','requirements_sha256'}
    require(type(value) is dict and set(value) == keys and
        type(value['schema_version']) is int and value['schema_version'] == 1,
        'strict typed context schema')
    for key in ('head_sha','base_sha','tested_merge_sha'):
        require(type(value[key]) is str and HEX40.fullmatch(value[key]), 'strict typed Git identity')
    for key in ('source_manifest_sha256','helper_sha256','requirements_sha256'):
        require(type(value[key]) is str and HEX64.fullmatch(value[key]), 'strict typed source identity')
    for name, source_key, disposition in (
        ('loader_qualification','model_sha256','30_OF_30_PASS'),
        ('supervisor_qualification','source_sha256','ROOT_ADOPTED_CONTROLS_PASS')):
        gate=value[name]
        require(type(gate) is dict and set(gate)=={'disposition',source_key,'native_receipt'} and
            type(gate['disposition']) is str and gate['disposition']==disposition and
            type(gate[source_key]) is str and HEX64.fullmatch(gate[source_key]) and
            type(gate['native_receipt']) is str and RECEIPT.fullmatch(gate['native_receipt']),
            'strict typed qualification structure; offline shape is not native authentication')


def retained_inputs(value, context, manifest_identity, context_identity, source_files, source_bytes):
    require(type(value) is dict and set(value)=={'context','source_manifest','context_identity',
        'source_files','source_bytes','scope'}, 'strict retained-input schema')
    typed_context(context)
    typed_context(value['context'])
    typed_identity(manifest_identity,2*MIB)
    typed_identity(context_identity,16*1024)
    typed_identity(value['source_manifest'],2*MIB)
    typed_identity(value['context_identity'],16*1024)
    require(type(value['source_files']) is int and 0 <= value['source_files'] <= 20000 and
        type(value['source_bytes']) is int and 0 <= value['source_bytes'] <= 128*MIB and
        value['source_files']==source_files and value['source_bytes']==source_bytes,
        'strict retained input counts')
    require(type(value['scope']) is str and value['scope']==
        'Receipt supplied by separately qualified route; this program does not authenticate GitHub offline.',
        'retained input scope')
    require(same_json(value['context'],context) and
        same_json(value['source_manifest'],manifest_identity) and
        same_json(value['context_identity'],context_identity), 'retained binding type/byte mismatch')


def clean_case(value):
    """Error evidence dominates any rebound PASS/RED label or aggregate."""
    require(type(value) is dict and type(value.get('page_errors')) is list and
        not value['page_errors'] and type(value.get('page_error_count')) is int and
        value['page_error_count']==0, 'application error evidence; HOLD_INCOMPLETE')
    # The exact helper emits these optional keys only on an error/overflow.
    # Presence is rejected even when a replacement value is empty or false.
    require(not {'request_capture_overflow','capture_error','close_error','traceback'} & set(value),
        'capture/cleanup/overflow evidence; HOLD_INCOMPLETE')
    require(type(value.get('passed')) is bool and
        (('error' not in value) if value['passed'] else
         (type(value.get('error')) is str and bool(value['error']))),
        'case error field contradicts pass/assertion semantics')


def safe_name(name):
    return (isinstance(name, str) and len(name) <= 240 and
            re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name) and
            all(p not in ('.', '..') for p in name.split('/')))


def read_file(path, maximum):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= maximum,
            'unsafe or oversized member: ' + path.name)
    geometry = lambda a: (a.st_dev, a.st_ino, a.st_mode, a.st_uid, a.st_gid, a.st_nlink, a.st_size,
                          a.st_mtime_ns, a.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        require(geometry(s) == geometry(os.fstat(stream.fileno())), 'changed before read')
        raw = stream.read(maximum + 1)
        require(geometry(s) == geometry(os.fstat(stream.fileno())) == geometry(path.lstat()),
                'changed during read')
    require(len(raw) == s.st_size, 'short read')
    return raw


def write_new(path, raw):
    with path.open('xb') as out:
        require(out.write(raw) == len(raw), 'short write')
    path.chmod(0o600)


def replace_json(path, value):
    temp = path.with_name(path.name + '.next')
    write_new(temp, packed(value))
    os.replace(temp, path)


def precreate(root):
    root.mkdir(mode=0o700)
    write_new(root / 'pending.json', packed({'schema_version': 1, 'disposition': 'HOLD_INCOMPLETE',
        'reason': 'child has not completed; missing final status never qualifies',
        'scientific_effect': 'NONE'}))
    write_new(root / 'stdout.bin', b'')
    write_new(root / 'stderr.bin', b'')
    write_new(root / 'status.json', packed({'schema_version': 1, 'complete': False,
        'disposition': 'HOLD_INCOMPLETE', 'scientific_effect': 'NONE'}))


def proc_rows():
    """Bounded Linux records; start ticks fence PID reuse."""
    entries = list(Path('/proc').iterdir())
    require(len(entries) <= 16384, 'process inventory capacity exceeded')
    rows = {}
    for p in entries:
        if not p.name.isdigit():
            continue
        try:
            raw = (p / 'stat').read_text()
            fields = raw[raw.rindex(')') + 2:].split()
            rows[int(p.name)] = (int(fields[1]), int(fields[2]), int(fields[19]), fields[0])
        except (FileNotFoundError, ProcessLookupError):
            pass
    return rows


def subreaper():
    require(sys.platform == 'linux', 'Linux /proc and subreaper required')
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'PR_SET_CHILD_SUBREAPER failed')


def supervise(command, cwd, root, env, deadline=540.0, pipe_cap=PIPE_MAX):
    """Always precreate evidence. TERM then KILL adopted tree and process group.

    No preexec_fn or RLIMIT changes: descendants inherit no newly lowered hard
    file ceiling. Pipe capture is bounded independently of child resource limits.
    """
    precreate(root)
    started = time.monotonic()
    status = {'schema_version': 1, 'complete': False, 'disposition': 'HOLD_INCOMPLETE',
              'scientific_effect': 'NONE', 'deadline_seconds': deadline,
              'pipe_cap_each': pipe_cap, 'child_returncode': None, 'stop_reason': None,
              'cleanup_complete': False, 'signals': [], 'tracked_processes': [],
              'runtime': {k: os.environ.get(k) for k in
                          ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY')}}
    child, sel, tracked = None, selectors.DefaultSelector(), {}
    handles = {}
    def remember(pid, row):
        if pid in tracked:
            require(tracked[pid] == row[2], 'tracked PID reuse')
            return
        try:
            fd = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            return
        current = proc_rows().get(pid)
        if current is None or current[2] != row[2]:
            os.close(fd)
            return
        tracked[pid], handles[pid] = row[2], fd
    def send(pid, sig):
        if pid in handles:
            try:
                signal.pidfd_send_signal(handles[pid], sig, None, 0)
            except ProcessLookupError:
                pass
    def group_signal(sig):
        # An unreaped leader retains its PID even if it exits between checks.
        # Once Popen reaps it, target exact pidfds instead of a reused group ID.
        if child is not None and child.returncode is None:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
    pipes, counts = {}, {'stdout': 0, 'stderr': 0}
    stop_at, kill_at = None, None
    try:
        subreaper()
        require(sys.dont_write_bytecode, 'python -B required')
        require(0 < deadline <= 540 and 0 < pipe_cap <= PIPE_MAX, 'invalid process bounds')
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            close_fds=True)
        status['leader_pid'] = child.pid
        for name, pipe in (('stdout', child.stdout), ('stderr', child.stderr)):
            os.set_blocking(pipe.fileno(), False)
            sel.register(pipe, selectors.EVENT_READ, name)
            pipes[name] = (root / (name + '.bin')).open('ab', buffering=0)
        while True:
            now = time.monotonic()
            rows = proc_rows()
            # Every newly adopted child is ours: this process starts no unrelated
            # subprocesses while supervising. Descendants may create new groups.
            changed = True
            descendants = {child.pid, *tracked}
            while changed:
                before = len(descendants)
                descendants.update(pid for pid, row in rows.items()
                                   if row[0] in descendants or row[0] == os.getpid())
                changed = len(descendants) != before
            require(len(descendants) <= 1024, 'descendant capacity exceeded')
            for pid in descendants:
                if pid in rows:
                    remember(pid, rows[pid])
            child.poll()
            reaped = set()
            for pid in list(tracked):
                if pid != child.pid and pid in rows and rows[pid][3] == 'Z':
                    try:
                        waited, _ = os.waitpid(pid, os.WNOHANG)
                        if waited == pid:
                            reaped.add(pid)
                    except ChildProcessError:
                        pass
            alive = {pid for pid, tick in tracked.items()
                     if pid in rows and pid not in reaped and rows[pid][2] == tick and
                     not (pid == child.pid and child.returncode is not None)}
            if status['stop_reason'] is None:
                if now - started >= deadline:
                    status['stop_reason'] = 'WALL_TIMEOUT'
                elif child.returncode is not None and any(rows[pid][3] != 'Z' for pid in alive):
                    status['stop_reason'] = 'DESCENDANTS_AFTER_LEADER_EXIT'
            if status['stop_reason'] is not None and stop_at is None:
                stop_at, kill_at = now, now + 5.0
                status['signals'].append('TERM')
                group_signal(signal.SIGTERM)
                for pid in alive:
                    send(pid, signal.SIGTERM)
            if kill_at is not None and now >= kill_at:
                if 'KILL' not in status['signals']:
                    status['signals'].append('KILL')
                group_signal(signal.SIGKILL)
                for pid in alive:
                    send(pid, signal.SIGKILL)
            for key, _ in sel.select(timeout=0.05):
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    sel.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                retained = data[:max(0, pipe_cap - counts[name])]
                if retained:
                    require(pipes[name].write(retained) == len(retained), 'short raw capture')
                counts[name] += len(retained)
                if len(retained) != len(data) and status['stop_reason'] is None:
                    status['stop_reason'] = name.upper() + '_CAP'
            if child.returncode is not None and not alive and not sel.get_map():
                final_rows = proc_rows()
                # Leader wait and pipe EOF can race a newly adopted descendant
                # that closed its inherited descriptors. Scan again after wait.
                if any(row[0] == os.getpid() or
                       (pid in tracked and row[2] == tracked[pid] and pid != child.pid)
                       for pid, row in final_rows.items()):
                    continue
                status['cleanup_complete'] = True
                break
            if stop_at is not None and now - stop_at >= 10.0:
                status['stop_reason'] = status['stop_reason'] or 'CLEANUP_TIMEOUT'
                break
        status['child_returncode'] = child.poll()
        status['complete'] = status['cleanup_complete'] and status['stop_reason'] is None
        if status['complete']:
            status['disposition'] = 'CHILD_EXIT_CAPTURED'
    except Exception as error:
        status['error'] = type(error).__name__ + ': ' + str(error)[:1500]
        status['stop_reason'] = status['stop_reason'] or 'SUPERVISOR_OR_LAUNCH_ERROR'
    finally:
        # Exceptional failures also require tree cleanup. Normal-loop evidence
        # alone does not establish cleanup if this fallback is needed.
        if child is not None and not status['cleanup_complete']:
            finish = time.monotonic() + 5
            while time.monotonic() < finish:
                try:
                    rows = proc_rows()
                    candidates = {pid for pid, row in rows.items()
                                  if row[0] == os.getpid() or
                                  (pid in tracked and row[2] == tracked[pid])}
                    for pid in candidates:
                        remember(pid, rows[pid])
                        send(pid, signal.SIGKILL)
                    group_signal(signal.SIGKILL)
                    child.poll()
                    for pid in candidates - {child.pid}:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                    remaining = proc_rows()
                    if child.returncode is not None and not any(
                        row[0] == os.getpid() or (pid in tracked and row[2] == tracked[pid])
                        for pid, row in remaining.items() if pid != child.pid):
                        status['cleanup_complete'] = True
                        break
                except Exception as error:
                    status['cleanup_error'] = str(error)[:1000]
                    break
                time.sleep(0.05)
            status['child_returncode'] = child.poll()
        for stream in pipes.values():
            stream.close()
        for key in list(sel.get_map().values()):
            key.fileobj.close()
        sel.close()
        for fd in handles.values():
            os.close(fd)
        status['elapsed_seconds'] = round(time.monotonic() - started, 6)
        status['tracked_processes'] = [{'pid': p, 'start_ticks': t} for p, t in sorted(tracked.items())]
        status['retained_raw_bytes'] = counts
        replace_json(root / 'status.json', status)
    return status


def frozen_browser(checkout, manifest_raw, context_raw, expected_manifest, expected_context, expected_self):
    require(hashlib.sha256(read_file(Path(__file__), MIB)).hexdigest() == expected_self, 'supervisor source pin')
    require(hashlib.sha256(manifest_raw).hexdigest() == expected_manifest, 'manifest pin')
    require(hashlib.sha256(context_raw).hexdigest() == expected_context, 'context pin')
    manifest, context = strict(manifest_raw), strict(context_raw)
    typed_context(context)
    require(set(manifest) == {'schema_version', 'files'} and type(manifest['schema_version']) is int and manifest['schema_version'] == 1,
            'manifest schema')
    require(isinstance(manifest['files'], list) and len(manifest['files']) <= 20000, 'manifest count')
    expected_keys = {'schema_version', 'head_sha', 'base_sha', 'tested_merge_sha',
        'loader_qualification', 'supervisor_qualification', 'source_manifest_sha256',
        'helper_sha256', 'requirements_sha256'}
    require(set(context) == expected_keys and type(context['schema_version']) is int and context['schema_version'] == 1, 'strict context schema')
    for key in ('head_sha', 'base_sha', 'tested_merge_sha'):
        require(isinstance(context[key], str) and HEX40.fullmatch(context[key]), 'Git context pin')
    require(context['tested_merge_sha'] == os.environ.get('GITHUB_SHA'), 'native merge environment join')
    require(os.environ.get('GITHUB_EVENT_NAME') == 'pull_request' and
            os.environ.get('GITHUB_REPOSITORY') == 'd6g8k5htny-coder/main', 'native event/repository')
    require(context['head_sha'] == os.environ.get('SITE_HEAD_SHA') and
            context['base_sha'] == os.environ.get('SITE_BASE_SHA'), 'native head/base environment join')
    require(context['source_manifest_sha256'] == expected_manifest and
            context['helper_sha256'] == HELPER_SHA and context['requirements_sha256'] == REQ_SHA,
            'context source binding')
    required = set()
    for directory in ('docs', 'tools'):
        for p in (checkout / directory).rglob('*'):
            require(not p.is_symlink(), 'symlinked input tree')
            if p.is_file() and (directory == 'docs' or p.suffix == '.py'):
                required.add(p.relative_to(checkout).as_posix())
                require(len(required) <= 20000, 'source membership bound')
    require(not any(checkout.rglob('*.pyc')), 'preexisting project bytecode')
    required.add('tests/browser-requirements.txt')
    rows, total, previous = {}, 0, ''
    for row in manifest['files']:
        require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'source row schema')
        name = row['path']
        require(safe_name(name) and name > previous and name not in rows, 'source order/path')
        previous = name
        require(type(row['bytes']) is int and row['bytes'] >= 0 and
                isinstance(row['sha256'], str) and HEX64.fullmatch(row['sha256']), 'source identity types')
        p = checkout / name
        require(not any(q.is_symlink() for q in [p, *p.parents]), 'symlinked source ancestor')
        body = read_file(p, 128 * MIB)
        total += len(body)
        require(total <= 128 * MIB and identity(body) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                'source identity/cumulative bound')
        rows[name] = row
    require(set(rows) == required, 'complete docs/Python/requirements membership')
    require({f'docs/site/{p}.html' for p in PAGES} <= required, 'ten maintained pages absent')
    require({p.name for p in (checkout/'docs/site').glob('*.html')} == {p+'.html' for p in PAGES}, 'root page census')
    for name in ('docs/site/current.mjs', 'docs/site/current-model.mjs', 'docs/site/dependency-model.mjs'):
        require(name in rows, 'required served module')
    require(rows['tools/current_site_transition_browser_check.py']['sha256'] == HELPER_SHA and
            rows['tests/browser-requirements.txt']['sha256'] == REQ_SHA and
            rows['tests/browser-requirements.txt']['bytes'] == 7774, 'helper/requirements source')
    gate = context['loader_qualification']
    require(set(gate) == {'disposition', 'model_sha256', 'native_receipt'} and
            gate['disposition'] == '30_OF_30_PASS' and RECEIPT.fullmatch(gate['native_receipt']) and
            gate['model_sha256'] == rows['docs/site/current-model.mjs']['sha256'], 'loader gate')
    gate = context['supervisor_qualification']
    require(set(gate) == {'disposition', 'source_sha256', 'native_receipt'} and
            gate['disposition'] == 'ROOT_ADOPTED_CONTROLS_PASS' and
            gate['source_sha256'] == expected_self and RECEIPT.fullmatch(gate['native_receipt']), 'supervisor gate')
    # This route refuses mutable source modes; it cannot make owner-controlled
    # chmod an adversarial immutability boundary. Root independently qualifies
    # the clean hosted checkout and exact native source joins.
    require(not (checkout.lstat().st_mode & 0o222) and all(not (p.lstat().st_mode & 0o222) for p in checkout.rglob('*')
                if '.git' not in p.relative_to(checkout).parts), 'writable checkout source')
    return context


def validate_browser(output, status, context, manifest_identity, context_identity, source_files, source_bytes):
    require(status['complete'] and status['child_returncode'] in (0, 1, 2), 'incomplete supervised process')
    files = sorted(output.iterdir())
    require(len(files) <= 128 and all(p.is_file() and not p.is_symlink() for p in files), 'browser output shape')
    bodies = {p.name: read_file(p, MANIFEST_MAX if p.name == 'manifest.json' else REPORT_MAX) for p in files}
    require(sum(map(len, bodies.values())) <= OUTPUT_MAX, 'browser output cumulative cap')
    report, manifest = strict(bodies['report.json']), strict(bodies['manifest.json'])
    require(set(manifest) == {'schema_version', 'members', 'write_attempts', 'manifest_self_excluded',
        'bytes_before_manifest', 'maximum_bytes_including_manifest'} and manifest['schema_version'] == 1 and
        manifest['manifest_self_excluded'] is True and type(manifest['schema_version']) is int and
        type(manifest['bytes_before_manifest']) is int and type(manifest['maximum_bytes_including_manifest']) is int and manifest['maximum_bytes_including_manifest'] == OUTPUT_MAX,
        'helper manifest schema')
    rows = manifest['members']
    require(isinstance(rows, list) and len(rows) < 128, 'helper members bound')
    listed = {}
    for row in rows:
        require(set(row) == {'path', 'bytes', 'sha256'} and safe_name(row['path']) and
                '/' not in row['path'] and row['path'] not in listed and row['path'] != 'manifest.json', 'helper member row')
        require(type(row['bytes']) is int and row['bytes'] >= 0 and row['path'] in bodies and
                identity(bodies[row['path']]) == {'bytes': row['bytes'], 'sha256': row['sha256']}, 'helper member bytes')
        listed[row['path']] = row
    require(set(bodies) == set(listed) | {'manifest.json'}, 'helper output complete membership')
    require(manifest['bytes_before_manifest'] == sum(len(bodies[n]) for n in listed), 'helper byte total')
    attempts = manifest['write_attempts']
    require(isinstance(attempts, list) and len(attempts) == len(rows) and
        all(set(a) == {'path', 'bytes_reserved', 'complete'} and a['complete'] is True and
            a['path'] == row['path'] and a['bytes_reserved'] == row['bytes']
            for a, row in zip(attempts, rows)), 'partial/failed helper write')
    require(type(report.get('schema_version')) is int and report.get('schema_version') == 1 and
            type(report.get('exit_code')) is int and report.get('scientific_effect') == 'NONE' and
            report.get('exit_code') == status['child_returncode'], 'helper report exit/schema')
    require(type(report) is dict and not {'error','traceback','unretained_detailed_report'} & set(report),
            'report-level incomplete/error evidence')
    retained_inputs(report.get('inputs'),context,manifest_identity,context_identity,source_files,source_bytes)
    require(status['child_returncode'] in (0, 1), 'helper setup exit remains HOLD')
    modes, cases = report['page_modes'], report['cases']
    require(set(modes) == {p+'.html' for p in PAGES} and
            set(modes.values()) == {'static', 'interactive'}, 'runtime modes')
    expected = []
    for page in PAGES:
        scenarios = ['old-positive', 'new-positive', 'old-html-new-payload']
        if modes[page+'.html'] == 'interactive':
            scenarios += ['outer-budget-metadata', 'outer-budget-successor']
        expected += [page+'--'+s for s in scenarios]
    require([c['case'] for c in cases] == expected and all(type(c['passed']) is bool for c in cases), 'exact case order/census')
    require(all(c['passed'] for c in cases if c['case'].endswith('-positive')), 'positive control failure')
    assertion_count = 0
    for case, case_id in zip(cases, expected):
        clean_case(case)
        page, scenario = case_id.split('--',1)
        mode = modes[page+'.html']
        require(case['page'] == page+'.html' and case['mode'] == mode, 'case page/mode join')
        if mode == 'static':
            allowed = {'PASS_STATIC_COHERENT_OBSERVATION_NO_READER': True}
        elif scenario.startswith('outer-budget-'):
            allowed = {'PASS_ACTUAL_OUTER_15S_ABORT': True, 'OUTER_BUDGET_ASSERTION': False}
        elif scenario == 'old-html-new-payload':
            allowed = {'PASS_VISIBLE_CURRENT_COHERENCE': True, 'UI_COHERENCE_ASSERTION': False}
        else:
            allowed = {'PASS_VISIBLE_CURRENT_COHERENCE': True}
        require(case['disposition'] in allowed and case['passed'] is allowed[case['disposition']],
                'case scenario/disposition/passed join')
        assertion_count += case['disposition'] in ('UI_COHERENCE_ASSERTION','OUTER_BUDGET_ASSERTION')
    expected_members = {'old-current.json', 'new-current.json', 'old-current-meta.json',
        'new-current-meta.json', 'report.json', 'manifest.json'}
    expected_members |= {p+'--'+v+'.html' for p in PAGES for v in ('old', 'new')}
    expected_members |= {name+'.png' for name in expected}
    require(set(bodies) == expected_members, 'fixture/PNG/report complete census')
    passed = sum(c['passed'] for c in cases)
    require(isinstance(report['counts'],dict) and all(type(v) is int for v in report['counts'].values()), 'strict integer counts')
    require(report['counts'] == {'expected': len(expected), 'observed': len(expected),
        'passed': passed, 'failed': len(expected)-passed, 'setup_or_incomplete': 0, 'skipped': 0}, 'report counts')
    code = status['child_returncode']
    require((code == 0 and passed == len(expected) and report['disposition'] == 'CONTROLLED_BROWSER_CASES_PASS') or
            (code == 1 and assertion_count > 0 and passed < len(expected) and report['disposition'] == 'QUALIFIED_BEHAVIOR_FAILURE'), 'report aggregate')
    footer = strict(read_file(output.parent/'process/stdout.bin', PIPE_MAX))
    require(type(footer.get('exit_code')) is int and type(footer.get('cases')) is int and
            footer == {'disposition': report['disposition'], 'exit_code': code, 'cases': len(cases)}, 'strict helper footer')
    require(read_file(output.parent/'process/stderr.bin', PIPE_MAX) == b'', 'unexpected helper stderr')
    return code


def carrier_bytes(members, label, complete):
    require(len(members) <= MEMBER_MAX and list(members) == sorted(members), 'carrier member count/order')
    require(all(safe_name(n) for n in members), 'carrier names')
    require(sum(map(len, members.values())) <= RAW_MAX, 'carrier raw cap')
    index = [{'name': n, **identity(b), 'chunks': math.ceil(len(b)/CHUNK)} for n, b in members.items()]
    frames = [packed({'kind': 'BEGIN', 'schema_version': 1, 'label': label, 'payload_complete': complete,
        'chunk_bytes': CHUNK, 'members': index, 'raw_bytes': sum(map(len, members.values()))})]
    for i, (name, raw) in enumerate(members.items()):
        frames.append(packed({'kind': 'MEMBER', 'i': i, 'name': name}))
        for number, offset in enumerate(range(0, len(raw), CHUNK)):
            frames.append(packed({'kind': 'CHUNK', 'i': i, 'n': number,
                'base64': base64.b64encode(raw[offset:offset+CHUNK]).decode('ascii')}))
        frames.append(packed({'kind': 'END_MEMBER', 'i': i, **identity(raw)}))
    stream_sha = hashlib.sha256(b''.join(frames)).hexdigest()
    frames.append(packed({'kind': 'END', 'member_count': len(index), 'stream_sha256': stream_sha}))
    wire = b''.join(PREFIX + frame for frame in frames)
    require(len(wire) <= WIRE_MAX, 'carrier wire cap')
    return wire


def decode_carrier(wire):
    """Strict offline decoder used by controls; Root independently rechecks bytes."""
    require(len(wire) <= WIRE_MAX and wire.endswith(b'\n'), 'carrier truncated/capped')
    lines = wire.splitlines(keepends=True)
    require(lines and all(line.startswith(PREFIX) for line in lines), 'carrier foreign line')
    raws = [line[len(PREFIX):] for line in lines]
    values = [strict(raw) for raw in raws]
    require(all(packed(v) == raw for v, raw in zip(values, raws)), 'noncanonical frame')
    begin, end = values[0], values[-1]
    require(set(begin) == {'kind','schema_version','label','payload_complete','chunk_bytes','members','raw_bytes'} and
        begin['kind'] == 'BEGIN' and type(begin['schema_version']) is int and begin['schema_version'] == 1 and
        type(begin['chunk_bytes']) is int and begin['chunk_bytes'] == CHUNK and
        type(begin['raw_bytes']) is int and 0 <= begin['raw_bytes'] <= RAW_MAX and
        isinstance(begin['label'], str) and 0 < len(begin['label']) <= 80 and
        type(begin['payload_complete']) is bool and isinstance(begin['members'], list) and
        len(begin['members']) <= MEMBER_MAX, 'BEGIN schema')
    require(set(end) == {'kind','member_count','stream_sha256'} and end['kind'] == 'END' and
        type(end['member_count']) is int and end['member_count'] == len(begin['members']) and
        isinstance(end['stream_sha256'], str) and HEX64.fullmatch(end['stream_sha256']) and
        end['stream_sha256'] == hashlib.sha256(b''.join(raws[:-1])).hexdigest(), 'END/footer')
    pos, result = 1, {}
    previous = ''
    for i, row in enumerate(begin['members']):
        require(set(row) == {'name','bytes','sha256','chunks'} and safe_name(row['name']) and
            row['name'] > previous and type(row['bytes']) is int and 0 <= row['bytes'] <= RAW_MAX and
            type(row['chunks']) is int and row['chunks'] == math.ceil(row['bytes']/CHUNK) and
            isinstance(row['sha256'], str) and HEX64.fullmatch(row['sha256']), 'member index')
        previous = row['name']
        require(type(values[pos].get('i')) is int and
            values[pos] == {'kind':'MEMBER','i':i,'name':row['name']}, 'MEMBER order')
        pos += 1
        pieces = []
        for n in range(row['chunks']):
            frame = values[pos]; pos += 1
            require(set(frame) == {'kind','i','n','base64'} and frame['kind'] == 'CHUNK' and
                    type(frame['i']) is int and frame['i'] == i and type(frame['n']) is int and
                    frame['n'] == n and isinstance(frame['base64'], str), 'CHUNK schema/order')
            part = base64.b64decode(frame['base64'], validate=True)
            require(base64.b64encode(part).decode('ascii') == frame['base64'] and
                len(part) == min(CHUNK, row['bytes'] - n*CHUNK), 'CHUNK geometry/Base64')
            pieces.append(part)
        raw = b''.join(pieces)
        require(identity(raw) == {'bytes':row['bytes'],'sha256':row['sha256']} and
            type(values[pos].get('i')) is int and type(values[pos].get('bytes')) is int and
            values[pos] == {'kind':'END_MEMBER','i':i,**identity(raw)}, 'END_MEMBER/hash')
        pos += 1
        result[row['name']] = raw
    require(pos == len(values)-1 and sum(map(len,result.values())) == begin['raw_bytes'] and
        sum(map(len,result.values())) <= RAW_MAX, 'carrier full membership/raw total')
    return begin, result


def emit(root):
    try:
        paths = sorted(root.rglob('*'))
        geometry = [(p,p.lstat()) for p in paths]
        require(all(stat.S_ISDIR(info.st_mode) or
            (stat.S_ISREG(info.st_mode) and info.st_nlink == 1) for p,info in geometry),
            'unsafe evidence entry: special/symlink/multilink')
        files = [p for p,info in geometry if stat.S_ISREG(info.st_mode)]
        require(len(files) <= MEMBER_MAX, 'evidence member cap')
        require(sum(p.lstat().st_size for p in files) <= RAW_MAX, 'evidence raw cap before reads')
        members = {p.relative_to(root).as_posix(): read_file(p, OUTPUT_MAX) for p in files}
        complete = strict(members['route.json']).get('capture_complete') is True
        wire = carrier_bytes(members, 'site-route', complete)
    except Exception as error:
        # Explicit fallback is a COMPLETE frame stream of INCOMPLETE evidence.
        # It does not carry or qualify omitted original output bytes.
        record = {'schema_version':1, 'capture_complete':False,
            'disposition':'HOLD_INCOMPLETE','reason':type(error).__name__+': '+str(error)[:1500],
            'omitted_original_outputs':True,'scientific_effect':'NONE','fallback_read_errors':[]}
        fallback = {}
        # Even when browser outputs exceed carrier capacity, preserve the finite
        # raw/process/pending boundary. Missing members are stated explicitly.
        fixed = [('route.json',16*1024),('context.json',16*1024),
                 ('source-manifest.json',2*MIB),('process/pending.json',4096),
                 ('process/stdout.bin',PIPE_MAX),('process/stderr.bin',PIPE_MAX),
                 ('process/status.json',256*1024),('stdout.bin',PIPE_MAX),
                 ('stderr.bin',PIPE_MAX)]
        for name, maximum in fixed:
            try:
                fallback[name] = read_file(root/name, maximum)
            except Exception as failure:
                record['fallback_read_errors'].append({'name':name,
                    'error':type(failure).__name__+': '+str(failure)[:300]})
        fallback['fallback.json'] = packed(record)
        wire = carrier_bytes(dict(sorted(fallback.items())), 'site-route-fallback', False)
    sys.stdout.buffer.write(wire)
    sys.stdout.buffer.flush()
    return 0 if decode_carrier(wire)[0]['payload_complete'] else 2


def browser(args):
    root, checkout = args.evidence, args.checkout.resolve()
    root.mkdir(mode=0o700)
    write_new(root/'route.json', packed({'schema_version':1,'capture_complete':False,
        'disposition':'HOLD_INCOMPLETE','scientific_effect':'NONE'}))
    code, scratch, status = 2, None, None
    try:
        m, c = read_file(args.manifest, 2*MIB), read_file(args.context, 16*1024)
        write_new(root/'source-manifest.json', m); write_new(root/'context.json', c)
        context = frozen_browser(checkout, m, c, args.manifest_sha256, args.context_sha256, args.self_sha256)
        scratch = root.with_name(root.name+'-scratch')
        scratch.mkdir(mode=0o700)
        home = scratch/'home'; home.mkdir(mode=0o700)
        temp = scratch/'tmp'; temp.mkdir(mode=0o700)
        # No ambient token, user site, Python path or git credentials reach child.
        env = {'PATH': os.environ.get('PATH','/usr/bin:/bin'), 'HOME':str(home),
               'TMPDIR':str(temp),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
               'PYTHONDONTWRITEBYTECODE':'1'}
        command = [sys.executable, '-B', str(checkout/'tools/current_site_transition_browser_check.py'),
            '--source-manifest', str((root/'source-manifest.json').resolve()),
            '--source-manifest-sha256', args.manifest_sha256,
            '--context', str((root/'context.json').resolve()), '--context-sha256', args.context_sha256,
            '--output', str((root/'browser').resolve())]
        status = supervise(command, checkout, root/'process', env)
        # Reread actual carried binding files, then recheck the entire source.
        actual_m = read_file(root/'source-manifest.json',2*MIB)
        actual_c = read_file(root/'context.json',16*1024)
        require(actual_m == m and actual_c == c, 'carried binding bytes changed during child')
        actual_manifest = strict(actual_m)
        source_files = len(actual_manifest['files'])
        source_bytes = sum(row['bytes'] for row in actual_manifest['files'])
        code = validate_browser(root/'browser', status, context, identity(actual_m), identity(actual_c),
                                source_files,source_bytes)
        frozen_browser(checkout, actual_m, actual_c, args.manifest_sha256, args.context_sha256, args.self_sha256)
        replace_json(root/'route.json', {'schema_version':1,'capture_complete':True,
            'disposition':'CONTROLLED_PASS' if code == 0 else 'QUALIFIED_BEHAVIOR_FAILURE',
            'exit_code':code,'source_manifest':identity(m),'context':identity(c),
            'supervisor_sha256':args.self_sha256,'scientific_effect':'NONE'})
    except Exception as error:
        replace_json(root/'route.json', {'schema_version':1,'capture_complete':False,
            'disposition':'HOLD_INCOMPLETE','exit_code':2,
            'error':type(error).__name__+': '+str(error)[:2000],'scientific_effect':'NONE'})
        code = 2
    finally:
        if scratch is not None and status is not None and status['cleanup_complete']:
            import shutil
            try:
                shutil.rmtree(scratch)
            except Exception as error:
                replace_json(root/'route.json', {'schema_version':1,'capture_complete':False,
                    'disposition':'HOLD_INCOMPLETE','exit_code':2,
                    'error':'scratch cleanup: '+str(error)[:1000],'scientific_effect':'NONE'})
                code = 2
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('emit'); e.add_argument('--evidence', required=True, type=Path)
    b = commands.add_parser('browser')
    for name in ('checkout','evidence','manifest','context'):
        b.add_argument('--'+name, required=True, type=Path)
    for name in ('manifest-sha256','context-sha256','self-sha256'):
        b.add_argument('--'+name, required=True)
    args = parser.parse_args()
    return emit(args.evidence) if args.command == 'emit' else browser(args)


if __name__ == '__main__':
    raise SystemExit(main())
