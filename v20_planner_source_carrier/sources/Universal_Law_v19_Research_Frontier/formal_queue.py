"""Durable, fenced queue for the one reviewed arithmetic Lean profile.

No caller code, commands, receipts or certificate claims enter this interface.
Successful jobs retain the pending mathematical alignment review unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import formal_lane

SCHEMA = 'ul-formal-queue-1'
MAX_ATTEMPTS = 3
ACTIVE = {'queued', 'running'}
_LANE_SHA256 = hashlib.sha256(Path(formal_lane.__file__).read_bytes()).hexdigest()


class JournalError(ValueError):
    """The committed journal cannot be trusted; no work is allowed."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as stream:
        stream.write(_canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _sync_directory(path.parent)


def _bindings(profile_id):
    if profile_id != formal_lane.PROFILE_ID or not isinstance(profile_id, str):
        raise ValueError('Only the registered reviewed profile ID is accepted')
    if _sha(formal_lane.__file__) != _LANE_SHA256:
        raise ValueError('formal checker implementation changed since import')
    profile = formal_lane._profile()
    formal_lane._sources(profile)
    formal_lane._pins()
    return {'profile_id': profile_id, 'profile_sha256': formal_lane.PROFILE_SHA256,
            'statements_sha256': _digest(profile['statements']),
            'source_hashes': profile['sources'],
            'checker_pins_sha256': formal_lane.PINS_SHA256,
            'checker_implementation_sha256': _LANE_SHA256,
            'alignment_status': profile['alignment_status'], 'scope': profile['scope']}


class FormalQueue:
    def __init__(self, workdir, toolchain=None, max_workers=1):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 8:
            raise ValueError('max_workers must be an integer from 1 through 8')
        self.root = Path(workdir).resolve() / 'runtime' / 'formal_queue'
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / 'attempts'
        self.artifacts.mkdir(exist_ok=True)
        _sync_directory(self.root.parent)
        _sync_directory(self.root.parent.parent)
        self.journal = self.root / 'events.jsonl'
        self.head = self.root / 'head.json'
        self.toolchain = toolchain
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads = []
        self._lifecycle = threading.Lock()
        with self._locked():
            state = self._read()
            if not state['events']:
                self._append(state, 'configured', {'max_workers': max_workers})
            self.max_workers = state['max_workers']
            self._recover(state)

    @contextmanager
    def _locked(self):
        with (self.root / 'journal.lock').open('a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self):
        state = {'events': [], 'jobs': {}, 'last_sequence': 0, 'digest': '0' * 64, 'max_workers': 1}
        data = self.journal.read_bytes() if self.journal.exists() else b''
        complete = data.rfind(b'\n') + 1
        tail = data[complete:]
        try:
            for raw in data[:complete].splitlines():
                event = json.loads(raw)
                body = {k: v for k, v in event.items() if k != 'sha256'}
                if event.get('schema') != SCHEMA or event.get('sequence') != state['last_sequence'] + 1 or event.get('previous_sha256') != state['digest'] or event.get('sha256') != _digest(body):
                    raise JournalError('journal sequence, schema or digest mismatch')
                state['events'].append(event)
                state['last_sequence'] = event['sequence']
                state['digest'] = event['sha256']
                if event['kind'] == 'configured':
                    if event['sequence'] != 1:
                        raise JournalError('duplicate queue configuration')
                    state['max_workers'] = event['data']['max_workers']
                if 'job' in event['data']:
                    job = event['data']['job']
                    state['jobs'][job['id']] = job
            if self.head.exists():
                head = json.loads(self.head.read_bytes())
                sequence = head['sequence']
                if sequence > state['last_sequence'] or (sequence and state['events'][sequence - 1]['sha256'] != head['sha256']):
                    raise JournalError('committed journal was truncated or its checkpoint changed')
            elif state['events']:
                # This can occur after fsynced initial append and before checkpoint.
                pass
        except (KeyError, TypeError, IndexError, AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise JournalError('malformed journal: ' + str(exc)) from exc
        if tail:
            # Only uncommitted incomplete bytes are repaired; all complete events remain.
            tail_path = self.root / ('torn-tail-' + uuid.uuid4().hex)
            with tail_path.open('xb') as out:
                out.write(tail)
                out.flush()
                os.fsync(out.fileno())
            with self.journal.open('r+b') as out:
                out.truncate(complete)
                out.flush()
                os.fsync(out.fileno())
            self._append(state, 'torn_tail_recovered', {'sha256': hashlib.sha256(tail).hexdigest(), 'bytes': len(tail), 'path': tail_path.name})
        return state

    def _append(self, state, kind, data):
        event = {'schema': SCHEMA, 'sequence': state['last_sequence'] + 1,
                 'previous_sha256': state['digest'], 'created_utc': _now(), 'kind': kind, 'data': copy.deepcopy(data)}
        event['sha256'] = _digest(event)
        with self.journal.open('ab') as out:
            out.write(_canonical(event) + b'\n')
            out.flush()
            os.fsync(out.fileno())
        _atomic(self.head, {'sequence': event['sequence'], 'sha256': event['sha256']})
        state['events'].append(event)
        state['last_sequence'] = event['sequence']
        state['digest'] = event['sha256']
        if kind == 'configured':
            state['max_workers'] = data['max_workers']
        if 'job' in data:
            state['jobs'][data['job']['id']] = copy.deepcopy(data['job'])
        return event

    def _slot(self, index):
        stream = (self.root / ('slot-%d.lock' % index)).open('a+b')
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except BlockingIOError:
            stream.close()
            return None

    def _recover(self, state):
        for job in list(state['jobs'].values()):
            if job['status'] != 'running':
                continue
            slot = self._slot(job['lease']['slot'])
            if slot is None:
                continue
            try:
                job = copy.deepcopy(job)
                job['attempts'][-1].update(status='interrupted', reason_code='lease_recovered', reason='Worker no longer holds its OS lease; orphan output cannot be accepted', finished_utc=_now())
                job.update(status='queued' if len(job['attempts']) < MAX_ATTEMPTS else 'escalated', lease=None,
                           reason_code='lease_recovered', reason='Interrupted attempt fenced; fresh attempt required', updated_utc=_now())
                self._append(state, 'lease_recovered', {'job': job})
            finally:
                slot.close()

    def enqueue(self, profile_id, event_id=None):
        if event_id is not None and (not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200):
            raise ValueError('event_id must be a nonempty string of at most 200 characters')
        bindings = _bindings(profile_id)
        fingerprint = _digest(bindings)
        with self._locked():
            state = self._read()
            self._recover(state)
            for job in state['jobs'].values():
                if event_id and event_id in job['event_ids']:
                    if job['profile_id'] != profile_id or job['input_fingerprint'] != fingerprint:
                        raise ValueError('event_id already belongs to another input')
                    return copy.deepcopy(job)
            for job in state['jobs'].values():
                if job['status'] in ACTIVE and job['input_fingerprint'] == fingerprint:
                    if event_id and event_id not in job['event_ids']:
                        job = copy.deepcopy(job)
                        job['event_ids'].append(event_id)
                        self._append(state, 'enqueue_deduplicated', {'job': job})
                    return copy.deepcopy(job)
            job = {'id': uuid.uuid4().hex, 'profile_id': profile_id, 'event_ids': [event_id] if event_id else [],
                   'input_fingerprint': fingerprint, 'bindings': bindings, 'status': 'queued',
                   'created_utc': _now(), 'updated_utc': _now(), 'attempts': [], 'lease': None,
                   'reason_code': None, 'reason': None, 'receipt': None, 'verification': None,
                   'full_theorem_certificates': []}
            self._append(state, 'enqueued', {'job': job})
        self._wake.set()
        return copy.deepcopy(job)

    def snapshot(self):
        with self._locked():
            state = self._read()
            self._recover(state)
            return {'schema': SCHEMA, 'jobs': copy.deepcopy(list(state['jobs'].values())),
                    'last_sequence': state['last_sequence'], 'journal_sha256': state['digest'],
                    'max_workers': state['max_workers'], 'max_attempts': MAX_ATTEMPTS,
                    'full_theorem_certificates': []}

    def retry(self, job_id):
        with self._locked():
            state = self._read()
            self._recover(state)
            if not isinstance(job_id, str) or job_id not in state['jobs']:
                raise ValueError('Unknown formal job ID')
            job = copy.deepcopy(state['jobs'][job_id])
            if job['status'] not in {'blocked', 'failed', 'escalated'}:
                raise ValueError('Only a blocked or failed job can be explicitly retried')
            if len(job['attempts']) >= MAX_ATTEMPTS:
                job.update(status='escalated', reason_code='attempt_limit', reason='Three attempts exhausted; owner intervention and a new job are required')
            else:
                job.update(status='queued', reason_code=None, reason=None)
            job['updated_utc'] = _now()
            self._append(state, 'retry_requested', {'job': job})
        self._wake.set()
        return copy.deepcopy(job)

    def _claim(self):
        with self._locked():
            state = self._read()
            self._recover(state)
            pending = next((j for j in state['jobs'].values() if j['status'] == 'queued'), None)
            if pending is None:
                return None
            for index in range(state['max_workers']):
                slot = self._slot(index)
                if slot is not None:
                    break
            else:
                return None
            try:
                job = copy.deepcopy(pending)
                token = uuid.uuid4().hex
                attempt_dir = self.artifacts / job['id'] / token
                attempt_dir.mkdir(parents=True)
                _sync_directory(attempt_dir.parent)
                _sync_directory(self.artifacts)
                lease = {'token': token, 'slot': index, 'pid': os.getpid(), 'started_utc': _now(),
                         'custody': 'exclusive OS flock held until durable completion'}
                attempt = {'number': len(job['attempts']) + 1, 'token': token, 'status': 'running',
                           'started_utc': lease['started_utc'], 'artifact_dir': str(attempt_dir)}
                job['attempts'].append(attempt)
                job.update(status='running', lease=lease, updated_utc=_now())
                self._append(state, 'attempt_started', {'job': job})
                return job, slot, attempt_dir
            except BaseException:
                slot.close()
                raise

    def _selected_toolchain(self):
        if self.toolchain is not None:
            return self.toolchain
        if os.environ.get('UL_LEAN_TOOLCHAIN'):
            return os.environ['UL_LEAN_TOOLCHAIN']
        try:
            from toolchain_setup import default_toolchain
            return default_toolchain()
        except ImportError:
            return None

    def _run_one(self):
        claim = self._claim()
        if claim is None:
            return False
        job, slot, attempt_dir = claim
        verification = None
        try:
            try:
                current = _bindings(job['profile_id'])
                if _digest(current) != job['input_fingerprint']:
                    raise ValueError('Queued profile, statements, sources or checker pins changed')
            except (OSError, ValueError) as exc:
                result = {'status': 'blocked', 'reason_code': 'input_integrity_mismatch', 'reason': str(exc), 'full_theorem_certificates': []}
            else:
                toolchain = self._selected_toolchain()
                result = formal_lane.run_profile(job['profile_id'], attempt_dir, toolchain=toolchain)
                if result.get('status') == 'scoped_kernel_checked':
                    verification = formal_lane.verify_receipt(result, toolchain=toolchain)
            self._complete(job['id'], job['lease']['token'], result, verification)
        except Exception as exc:
            result = {'status': 'failed', 'reason_code': 'worker_exception', 'reason': type(exc).__name__ + ': ' + str(exc), 'full_theorem_certificates': []}
            self._complete(job['id'], job['lease']['token'], result, None)
        finally:
            slot.close()
        return True

    def _complete(self, job_id, token, result, verification=None):
        # Tokens are internal attempt capabilities. No HTTP API accepts results.
        with self._locked():
            state = self._read()
            job = state['jobs'].get(job_id)
            if not job or job['status'] != 'running' or job['lease']['token'] != token:
                self._append(state, 'late_result_rejected', {'job_id': job_id, 'token': token})
                return False
            job = copy.deepcopy(job)
            attempt = job['attempts'][-1]
            result = copy.deepcopy(result)
            verification = copy.deepcopy(verification)
            status = result.get('status')
            if status == 'scoped_kernel_checked':
                try:
                    if _digest(_bindings(job['profile_id'])) != job['input_fingerprint']:
                        raise ValueError('queued input changed before result acceptance')
                    expected = {'profile_id': job['profile_id'], 'profile_sha256': job['bindings']['profile_sha256'],
                                'source_hashes': job['bindings']['source_hashes'],
                                'alignment_status': job['bindings']['alignment_status'], 'full_theorem_certificates': []}
                    if any(result.get(key) != value for key, value in expected.items()) or _digest(result.get('statements')) != job['bindings']['statements_sha256']:
                        raise ValueError('receipt profile, statement, source or scope binding differs')
                    if result.get('scope') != job['bindings']['scope'] or result.get('toolchain', {}).get('pins_sha256') != job['bindings']['checker_pins_sha256']:
                        raise ValueError('receipt scope or checker pin binding differs')
                    if not verification or verification.get('valid') is not True or verification.get('fresh_replay') is not True or verification.get('state') != 'scoped_kernel_checked':
                        raise ValueError('receipt did not pass independent real pinned replay')
                    receipt_path = Path(result['receipt_path'])
                    if receipt_path.is_symlink() or not receipt_path.resolve().is_relative_to(Path(attempt['artifact_dir']).resolve()):
                        raise ValueError('receipt escaped its fenced attempt directory')
                    if json.loads(receipt_path.read_bytes()) != result:
                        raise ValueError('receipt file differs from worker result')
                    if result.get('receipt_sha256') != formal_lane._digest({k: v for k, v in result.items() if k not in {'receipt_sha256', 'receipt_path'}}):
                        raise ValueError('receipt digest mismatch')
                    job['receipt_ref'] = {'path': str(receipt_path.relative_to(self.root)), 'sha256': _sha(receipt_path), 'bytes': receipt_path.stat().st_size}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    status = 'failed'
                    result.update(status='failed', reason_code='receipt_rejected', reason=str(exc))
            elif status not in {'blocked', 'failed'}:
                status = 'failed'
                result.update(status='failed', reason_code='invalid_worker_status', reason='Worker returned an unsupported status')
            # Durable result custody precedes the terminal journal event. If the
            # process dies between these operations recovery fences this output.
            record = {'schema': SCHEMA, 'job_id': job_id, 'token': token, 'input_fingerprint': job['input_fingerprint'],
                      'result': result, 'verification': verification}
            result_path = Path(attempt['artifact_dir']) / 'result.json'
            _atomic(result_path, record)
            for path in result_path.parent.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            for path in sorted((p for p in result_path.parent.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                _sync_directory(path)
            _sync_directory(result_path.parent)
            attempt.update(status=status, finished_utc=_now(), reason_code=result.get('reason_code'), reason=result.get('reason'),
                           result_ref={'path': str(result_path.relative_to(self.root)), 'sha256': _sha(result_path), 'bytes': result_path.stat().st_size})
            job.update(status=status, lease=None, updated_utc=_now(), reason_code=result.get('reason_code'),
                       reason=result.get('reason'), receipt=result, verification=verification, full_theorem_certificates=[])
            self._append(state, 'attempt_finished', {'job': job})
            return True

    def run_pending(self):
        """Synchronously drain claimable work; global OS leases bound concurrency."""
        while not self._stop.is_set() and self._run_one():
            pass
        return self.snapshot()

    def _worker(self):
        while not self._stop.is_set():
            try:
                if self._run_one():
                    continue
            except JournalError:
                # Do not execute anything after an integrity failure.
                return
            self._wake.wait(0.2)
            self._wake.clear()

    def start(self):
        with self._lifecycle:
            if any(thread.is_alive() for thread in self._threads):
                return self
            self._stop.clear()
            self._threads = [threading.Thread(target=self._worker, name='formal-queue-%d' % i, daemon=True) for i in range(self.max_workers)]
            for thread in self._threads:
                thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join()
