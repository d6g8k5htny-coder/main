"""Durable orchestration for the fixed, supporting v19 mathematical experiment.

The native process lease serializes execution across processes. Engine locks precede
conductor locks, which precede the runtime job journal. No public method accepts
proof source, command vectors, profiles, IR, or submitted proof receipts.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

import flow  # Establish the retained engine module prefix before component imports.
from conductor import Journal
from belief_ledger import BeliefLedger
from collaboration import AnnotationStore
from discovery import rank_connections
from evolution import EvolutionLab
from frontier_gpu import encode_positions
from frontier_ir import bounded_json, digest
from hypergraph_learning import train_experiment
from moment_certificates import make_certificate
from moment_checker import verify_certificate
from polynomial_prover import profile_contract, prove_profile
from proof_conductor import ProofConductor, native_backend_identity

ROOT = Path(__file__).resolve().parent
MAX_JOBS = 128
MAX_EVENTS = 2048
MAX_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS = 3
TERMINAL = {'blocked', 'failed', 'scoped_kernel_checked'}
TOKEN = re.compile(r'[0-9a-f]{32}')
JOB_ID = re.compile(r'math-[0-9a-f]{32}')


def _event_id(value):
    if value is not None and (type(value) is not str or not 1 <= len(value) <= 180
            or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('event_id must be a bounded nonempty text identifier')
    return value


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 2**53:
        raise ValueError('invalid runtime event time')


def _relative(value):
    if type(value) is not str or not value or '\\' in value:
        raise ValueError('invalid relative artifact reference')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('artifact reference must stay relative to the frontier directory')
    return value


def _reason(code, message, diagnosis='checker_failure'):
    return {'code': str(code)[:160], 'message': str(message)[:2048], 'diagnosis': diagnosis}


class PublicFrontierRuntime:
    def __init__(self, workdir, coordinator, toolchain=None):
        self.workdir = Path(workdir).resolve()
        self.root = self.workdir / 'frontier'
        self.root.mkdir(parents=True, exist_ok=True)
        self.coordinator = coordinator
        self.toolchain = toolchain
        self.conductor = ProofConductor(self.root / 'conductor', coordinator, toolchain)
        self.beliefs = BeliefLedger(self.root / 'beliefs')
        self.evolution = EvolutionLab(self.root / 'evolution', coordinator)
        ir = self.conductor.snapshot()['ir']
        self.annotations = AnnotationStore(self.root / 'annotations', [n['id'] for n in ir['nodes']])
        self.journal = Journal(self.root / 'jobs')
        self._stop = threading.Event()
        self.thread = None
        self._thread_mutex = threading.Lock()
        self._derived = {}
        self._cursor_cache = None
        self._native_state = None
        self._geometry_mutex = threading.Lock()
        self._geometry_cache = None
        self._geometry_baseline_sha256 = digest(self.coordinator.base)
        with self._transaction(): self._read()

    @contextmanager
    def _transaction(self):
        with self.conductor._transaction(), self.journal.transaction(): yield

    def _selected_toolchain(self):
        from toolchain_setup import default_toolchain
        return Path(self.toolchain).expanduser() if self.toolchain is not None else default_toolchain()

    def _capability(self):
        # Share the conductor's complete native/import/native-shadow identity.
        # Filesystem metadata detects drift; actual byte pins remain native-checker work.
        identity = native_backend_identity(self.toolchain)
        signature = digest({'native_backend': identity['signature'], 'contract': profile_contract()})
        state = 'startup_pending' if identity['present'] else 'blocked'
        if self._native_state and self._native_state[0] == signature:
            state = self._native_state[1]
        return signature, state

    def _append(self, kind, payload):
        if len(self.journal.events) >= MAX_EVENTS:
            raise ValueError('runtime journal event budget exhausted')
        if self.journal.path.exists() and self.journal.path.stat().st_size + len(json.dumps(payload).encode()) > MAX_BYTES:
            raise ValueError('runtime journal byte budget exhausted')
        self.journal.append(kind, payload)

    def _read(self):
        if self.journal.path.exists() and self.journal.path.stat().st_size > MAX_BYTES:
            raise ValueError('runtime journal byte bound exceeded')
        self.journal._load()
        if len(self.journal.events) > MAX_EVENTS: raise ValueError('runtime event bound exceeded')
        jobs, requests, event_ids, learning = {}, {}, {}, None
        tokens = set()
        for event in self.journal.events:
            kind, row = event['type'], event['payload']
            if kind == 'MATH_QUEUED':
                fields = {'id', 'status', 'claim_id', 'task_id', 'scope', 'artifact_ref', 'request_key',
                          'event_id', 'capability_signature', 'profile_contract_sha256', 'queued_at',
                          'attempt', 'claim_history', 'actual_field_applicable', 'full_theorem_certificates'}
                if set(row) != fields or not JOB_ID.fullmatch(row.get('id', '')):
                    raise ValueError('invalid queued runtime job')
                if (row['id'] in jobs or row['request_key'] in requests or row['status'] != 'queued'
                        or row['attempt'] != 1 or row['claim_history'] != [row['claim_id']]
                        or row['actual_field_applicable'] is not False or row['full_theorem_certificates'] != []
                        or type(row['scope']) is not dict or row['scope'].get('kind') != 'scalar_pilot'):
                    raise ValueError('duplicate job or invalid scalar job authority')
                if not re.fullmatch(r'claim-[0-9a-f]{32}', row['claim_id']):
                    raise ValueError('invalid assigned claim')
                _event_id(row['event_id']); _time(row['queued_at']); _relative(row['artifact_ref'])
                if row['artifact_ref'] != 'conductor/artifacts/' + row['claim_id']:
                    raise ValueError('job artifact directory must be assigned by its claim')
                for field in ('request_key', 'capability_signature', 'profile_contract_sha256'):
                    if type(row[field]) is not str or not re.fullmatch(r'[0-9a-f]{64}', row[field]):
                        raise ValueError('invalid runtime context hash')
                expected = digest({'event_id': row['event_id'], 'profile': row['profile_contract_sha256'],
                                   'capability': row['capability_signature']})
                if row['request_key'] != expected: raise ValueError('request context mismatch')
                if row['event_id'] is not None:
                    if row['event_id'] in event_ids: raise ValueError('duplicate public request event_id')
                    event_ids[row['event_id']] = row['id']
                requests[row['request_key']] = row['id']
                jobs[row['id']] = copy.deepcopy(row)
                if len(jobs) > MAX_JOBS: raise ValueError('runtime job budget exhausted')
            elif kind == 'MATH_REQUEST_ALIASED':
                if set(row) != {'id', 'event_id', 'request_key', 'profile_contract_sha256', 'capability_signature'}:
                    raise ValueError('invalid request alias fields')
                job = jobs.get(row['id'])
                _event_id(row['event_id'])
                expected = digest({'event_id': row['event_id'], 'profile': row['profile_contract_sha256'],
                                   'capability': row['capability_signature']})
                if (job is None or job['status'] not in {'queued', 'running'}
                        or (row['event_id'] is not None and row['event_id'] in event_ids) or row['request_key'] in requests
                        or row['request_key'] != expected
                        or row['profile_contract_sha256'] != job['profile_contract_sha256']
                        or row['capability_signature'] != job['capability_signature']):
                    raise ValueError('duplicate alias or different active request context')
                if row['event_id'] is not None: event_ids[row['event_id']] = job['id']
                requests[row['request_key']] = job['id']
            elif kind == 'MATH_STARTED':
                if set(row) != {'id', 'claim_id', 'attempt', 'token', 'started_at'}:
                    raise ValueError('invalid native start event')
                job = jobs.get(row['id'])
                if (job is None or job['status'] != 'queued' or row['claim_id'] != job['claim_id']
                        or row['attempt'] != job['attempt'] or not TOKEN.fullmatch(row.get('token', ''))
                        or row['token'] in tokens):
                    raise ValueError('native start violates job transition or fencing token')
                _time(row['started_at'])
                tokens.add(row['token'])
                job.update(status='running', token=row['token'], started_at=row['started_at'])
            elif kind in {'MATH_FINISHED', 'MATH_INTERRUPTED', 'MATH_RECONCILED'}:
                expected = {'id', 'token', 'finished_at', 'result'}
                if set(row) != expected: raise ValueError('invalid native result event')
                job = jobs.get(row['id'])
                if job is None or job['status'] != 'running' or row['token'] != job['token']:
                    raise ValueError('late native result or invalid transition token')
                _time(row['finished_at'])
                result = row['result']
                if type(result) is not dict or set(result) - {'status', 'reason', 'verification', 'receipt_ref',
                        'receipt_sha256', 'moment', 'worker_attempts', 'failure_report', 'native_capability'}:
                    raise ValueError('unexpected native result fields')
                if kind == 'MATH_INTERRUPTED':
                    if result.get('status') != 'interrupted' or set(result) != {'status', 'reason'}:
                        raise ValueError('invalid interrupted result')
                elif kind == 'MATH_RECONCILED' and (result.get('status') != 'scoped_kernel_checked'
                        or result.get('reason', {}).get('code') != 'reconciled_verified_conductor_completion'):
                    raise ValueError('reconciliation requires an exact verified completion')
                elif result.get('status') not in TERMINAL:
                    raise ValueError('invalid native terminal status')
                if result.get('status') == 'scoped_kernel_checked':
                    verification = result.get('verification', {})
                    if (verification.get('valid') is not True or verification.get('fresh_replay') is not True
                            or verification.get('actual_field_applicable') is not False
                            or verification.get('full_theorem_certificates') != []
                            or result.get('moment', {}).get('valid') is not True):
                        raise ValueError('native success requires scoped independent replay')
                    _relative(result['receipt_ref'])
                    if not result['receipt_ref'].startswith(job['artifact_ref'] + '/'):
                        raise ValueError('receipt outside assigned job claim')
                if result.get('reason') is not None:
                    reason = result['reason']
                    if (type(reason) is not dict or set(reason) != {'code', 'message', 'diagnosis'}
                            or (reason['diagnosis'] not in {'checker_failure', 'timeout'}
                                and not (kind == 'MATH_RECONCILED' and reason['diagnosis'] == 'delivery_reconciliation'))):
                        raise ValueError('invalid structured checker failure')
                bounded_json(result, 'native result', 256 * 1024)
                job.update(copy.deepcopy(result), finished_at=row['finished_at'])
            elif kind == 'MATH_RETRIED':
                if set(row) != {'id', 'previous_claim_id', 'claim_id', 'artifact_ref', 'attempt'}:
                    raise ValueError('invalid recovery attempt fields')
                job = jobs.get(row['id'])
                if (job is None or job['status'] != 'interrupted' or row['previous_claim_id'] != job['claim_id']
                        or row['attempt'] != job['attempt'] + 1 or row['attempt'] > MAX_ATTEMPTS
                        or row['claim_id'] in job['claim_history']
                        or not re.fullmatch(r'claim-[0-9a-f]{32}', row['claim_id'])
                        or row['artifact_ref'] != 'conductor/artifacts/' + row['claim_id']):
                    raise ValueError('invalid recovery reservation or attempt cap')
                job.update(status='queued', claim_id=row['claim_id'], artifact_ref=row['artifact_ref'], attempt=row['attempt'])
                job['claim_history'].append(row['claim_id'])
                for field in ('token', 'reason', 'finished_at', 'started_at'): job.pop(field, None)
            elif kind == 'MATH_RECOVERY_STOPPED':
                if set(row) != {'id', 'claim_id', 'reason'}: raise ValueError('invalid recovery stop')
                job = jobs.get(row['id'])
                if job is None or job['status'] != 'interrupted' or row['claim_id'] != job['claim_id']:
                    raise ValueError('invalid recovery terminal transition')
                job.update(status='failed', reason=row['reason'])
            elif kind == 'LEARNING_RECORDED':
                if set(row) != {'ir_sha256', 'epochs', 'seed', 'report', 'report_sha256'}:
                    raise ValueError('invalid derived learning event')
                if (type(row['epochs']) is not int or not 1 <= row['epochs'] <= 200
                        or type(row['seed']) is not int or not -(2**31) <= row['seed'] < 2**31
                        or row['report_sha256'] != digest(row['report'])):
                    raise ValueError('derived learning context mismatch')
                bounded_json(row['report'], 'derived learning report', 512 * 1024)
                learning = copy.deepcopy(row['report'])
            else: raise ValueError('unknown runtime event; mathematical authority refused: ' + kind)
        # Jobs never establish proof authority by their own hash chain. Bind
        # every reservation and successful result back to the conductor ledger.
        authority = self.conductor._read()
        for job in jobs.values():
            task = authority['tasks'].get(job['task_id'])
            if (task is None or task['kind'] != 'scalar_pilot' or task['scope'] != job['scope']
                    or digest(task['profile_contract']) != job['profile_contract_sha256']):
                raise ValueError('runtime job lacks its exact conductor scalar question')
            for claim_id in job['claim_history']:
                claim = authority['claims'].get(claim_id)
                if (claim is None or claim['task_id'] != job['task_id']
                        or claim['strategy'] != 'native_scalar_profile'
                        or not claim['agent_id'].startswith('frontier:' + job['id'])):
                    raise ValueError('runtime reservation is missing or has another authority')
            if job['status'] == 'scoped_kernel_checked':
                completion = next((c for c in authority['completions'] if c['claim_id'] == job['claim_id']), None)
                if (completion is None or completion['verification'] != job['verification']
                        or completion['worker_attempts'] != job['worker_attempts']
                        or completion['receipt_file_sha256'] != job['receipt_sha256']
                        or completion['profile_contract_sha256'] != job['profile_contract_sha256']):
                    raise ValueError('runtime success lacks an independently checked conductor completion')
                receipt = completion['receipt_path']
                expected = self._ref(receipt) if Path(receipt).is_absolute() else 'conductor/' + receipt
                if job['receipt_ref'] != expected:
                    raise ValueError('runtime receipt does not match completed claim')
                if job['moment'] != verify_certificate(make_certificate()):
                    raise ValueError('runtime moment result differs from independent exact checking')
        return {'jobs': jobs, 'requests': requests, 'event_ids': event_ids, 'learning': learning}

    def queue_math(self, event_id=None):
        _event_id(event_id)
        signature, _ = self._capability()
        contract_hash = digest(profile_contract())
        key = digest({'event_id': event_id, 'profile': contract_hash, 'capability': signature})
        with self._transaction():
            state = self._read()
            if event_id is not None and event_id in state['event_ids']:
                old = state['jobs'][state['event_ids'][event_id]]
                if old['profile_contract_sha256'] != contract_hash:
                    raise ValueError('event_id is bound to a different fixed profile contract')
                return copy.deepcopy(old)
            if key in state['requests']:
                return copy.deepcopy(state['jobs'][state['requests'][key]])
            if len(state['jobs']) >= MAX_JOBS: raise ValueError('runtime job budget exhausted')
            snapshot = self.conductor.snapshot()
            active = next((j for j in state['jobs'].values() if j['status'] in {'queued', 'running', 'interrupted'}), None)
            if active:
                if (active['profile_contract_sha256'] == contract_hash
                        and active['capability_signature'] == signature
                        and active['status'] in {'queued', 'running'}):
                    self._append('MATH_REQUEST_ALIASED', {'id': active['id'], 'event_id': event_id,
                        'request_key': key, 'profile_contract_sha256': contract_hash,
                        'capability_signature': signature})
                    return copy.deepcopy(active)
                raise ValueError('fixed scalar work already has an active runtime reservation')
            limit = snapshot['budget']['limit'] or 20
            blueprint = self.conductor.blueprint(['scalar-variance-bounds'], budget=limit)
            snapshot = self.conductor.snapshot()
            task = next(t for t in snapshot['tasks'] if t['task_id'] in blueprint['task_ids'] and t['kind'] == 'scalar_pilot')
            job_id = 'math-' + uuid.uuid4().hex
            claim = self.conductor.claim(task['task_id'], 'frontier:' + job_id,
                                         strategy='native_scalar_profile', lease_seconds=600)
            job = {'id': job_id, 'status': 'queued', 'claim_id': claim['claim_id'], 'task_id': task['task_id'],
                'scope': task['scope'], 'artifact_ref': self._ref(claim['artifact_directory']),
                'request_key': key, 'event_id': event_id, 'capability_signature': signature,
                'profile_contract_sha256': contract_hash, 'queued_at': time.time(),
                'attempt': 1, 'claim_history': [claim['claim_id']], 'actual_field_applicable': False,
                'full_theorem_certificates': []}
            self._append('MATH_QUEUED', job)
            return copy.deepcopy(job)

    def _ref(self, path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root): raise ValueError('artifact outside frontier directory')
        return _relative(resolved.relative_to(self.root).as_posix())

    def _path(self, reference):
        _relative(reference)
        path = self.root / reference
        cursor = path
        while cursor != self.root:
            if cursor.is_symlink(): raise ValueError('symlink in runtime artifact reference')
            cursor = cursor.parent
        if not path.resolve().is_relative_to(self.root): raise ValueError('runtime path escaped root')
        return path

    @contextmanager
    def _retained_parent(self, reference):
        """Walk every parent through held directory descriptors, without symlinks."""
        reference = _relative(reference)
        parts = (self.root / reference).parts
        descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in parts[1:-1]:
                next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                          dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            yield descriptor, parts[-1]
        finally:
            os.close(descriptor)

    def _read_retained(self, reference, limit):
        """Read a bounded, stable regular file; even racing FIFOs cannot block open."""
        try:
            with self._retained_parent(reference) as (parent, name):
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                     dir_fd=parent)
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
                        raise ValueError('retained artifact is not a bounded regular file: ' + reference)
                    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                    data = bytearray()
                    while True:
                        block = os.read(descriptor, min(65536, limit - len(data) + 1))
                        if not block: break
                        data.extend(block)
                        if len(data) > limit:
                            raise ValueError('retained artifact grew beyond its byte bound: ' + reference)
                    if len(data) != before.st_size or identity(os.fstat(descriptor)) != identity(before):
                        raise ValueError('retained artifact changed while reading: ' + reference)
                    return bytes(data)
                finally:
                    os.close(descriptor)
        except OSError as exc:
            raise ValueError('retained artifact could not be opened safely: ' + reference) from exc

    def _write_moment_certificate(self, reference, data):
        """Create exclusively, or compare an existing bounded regular certificate."""
        if len(data) > 65536: raise ValueError('generated moment certificate exceeds byte bound')
        try:
            with self._retained_parent(reference) as (parent, name):
                try:
                    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                                         | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=parent)
                except FileExistsError:
                    descriptor = None
                if descriptor is not None:
                    try:
                        remaining = memoryview(data)
                        while remaining:
                            written = os.write(descriptor, remaining)
                            if written <= 0: raise ValueError('moment certificate write made no progress')
                            remaining = remaining[written:]
                        os.fsync(descriptor)
                        os.fsync(parent)
                    finally:
                        os.close(descriptor)
                    return
        except OSError as exc:
            raise ValueError('moment certificate could not be written safely') from exc
        if self._read_retained(reference, 65536) != data:
            raise ValueError('stored moment certificate differs from generated bytes')

    @contextmanager
    def _process_lease(self):
        import fcntl
        stream = (self.journal.workdir / '.native-worker.lock').open('a+b')
        acquired = False
        try:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError: pass
            yield acquired
        finally: stream.close()

    def _failure(self, job, reason):
        try:
            report = self.conductor.submit_failure(job['claim_id'], reason['diagnosis'],
                                                   {'runtime_job': job['id'], 'reason': reason})
            return {'recorded': True, 'attempt_id': report['attempt']['attempt_id']}
        except (ValueError, OSError) as exc:
            return {'recorded': False, 'reason': str(exc)[:2048]}

    def _reconcile_completion(self, job):
        # A committed conductor result is authority-bound evidence. A receipt
        # found on disk without that durable completion never reaches this path.
        authority = self.conductor._read()
        if not any(c['claim_id'] == job['claim_id'] for c in authority['completions']):
            return None
        certificate = json.loads(self._read_retained(job['artifact_ref'] + '/moment-certificate.json', 65536))
        moment = verify_certificate(certificate)
        if certificate != make_certificate() or moment.get('valid') is not True:
            raise ValueError('retained moment certificate failed independent recovery check')
        self.conductor._proof_cache.clear()
        current = self.conductor.snapshot()  # Force an actual current native replay.
        completion = next(c for c in current['completions'] if c['claim_id'] == job['claim_id'])
        verification = completion['verification']
        task = next(t for t in current['tasks'] if t['task_id'] == job['task_id'])
        claim = next(c for c in current['claims'] if c['claim_id'] == job['claim_id'])
        if (verification.get('valid') is not True or verification.get('fresh_replay') is not True
                or verification.get('actual_field_applicable') is not False
                or verification.get('full_theorem_certificates') != []):
            return None
        if (completion['task_id'] != job['task_id'] or completion['input_fingerprint'] != task['input_fingerprint']
                or claim['input_fingerprint'] != task['input_fingerprint'] or task['scope'] != job['scope']
                or completion['profile_contract_sha256'] != job['profile_contract_sha256']):
            raise ValueError('committed completion differs from interrupted runtime scope')
        receipt_ref = 'conductor/' + completion['receipt_relative_path']
        if not receipt_ref.startswith(job['artifact_ref'] + '/'):
            raise ValueError('committed receipt is outside interrupted claim directory')
        receipt_bytes = self._read_retained(receipt_ref, 2 * 1024 * 1024)
        if hashlib.sha256(receipt_bytes).hexdigest() != completion['receipt_file_sha256']:
            raise ValueError('committed receipt hash changed during recovery')
        return {'status': 'scoped_kernel_checked', 'receipt_ref': receipt_ref,
            'receipt_sha256': completion['receipt_file_sha256'], 'verification': verification,
            'worker_attempts': completion['worker_attempts'], 'moment': moment,
            'native_capability': 'configured', 'reason': _reason('reconciled_verified_conductor_completion',
                'Recovered a committed conductor completion after current independent native replay.',
                'delivery_reconciliation')}

    def _recover(self):
        # Called only while owning the OS process lease. A live worker's job is
        # never inferred dead from age, another process's PID, or an HTTP poll.
        with self._transaction():
            state = self._read()
            for original in state['jobs'].values():
                job = copy.deepcopy(original)
                if job['status'] == 'running':
                    try:
                        result = self._reconcile_completion(job)
                    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
                        reason = _reason('recovery_artifact_rejected', str(exc))
                        self._append('MATH_INTERRUPTED', {'id': job['id'], 'token': job['token'],
                            'finished_at': time.time(), 'result': {'status': 'interrupted', 'reason': reason}})
                        self._append('MATH_RECOVERY_STOPPED', {'id': job['id'], 'claim_id': job['claim_id'],
                            'reason': reason})
                        continue
                    if result is not None:
                        self._append('MATH_RECONCILED', {'id': job['id'], 'token': job['token'],
                            'finished_at': time.time(), 'result': result})
                        self._native_state = (self._capability()[0], 'configured')
                        continue
                    reason = _reason('interrupted_native_attempt', 'The prior native process lease is free; its output is not adopted.')
                    self._append('MATH_INTERRUPTED', {'id': job['id'], 'token': job['token'],
                        'finished_at': time.time(), 'result': {'status': 'interrupted', 'reason': reason}})
                    job['status'] = 'interrupted'
                if job['status'] != 'interrupted': continue
                reason = _reason('interrupted_native_attempt', 'Recovering with a separate globally budgeted claim.')
                self._failure(job, reason)
                if job['attempt'] >= MAX_ATTEMPTS:
                    self._append('MATH_RECOVERY_STOPPED', {'id': job['id'], 'claim_id': job['claim_id'],
                        'reason': _reason('runtime_attempt_cap', 'Three runtime attempts consumed; explicit new work is required.')})
                    continue
                try:
                    claim = self.conductor.claim(job['task_id'], 'frontier:' + job['id'] + ':retry:' + str(job['attempt'] + 1),
                                                  strategy='native_scalar_profile', lease_seconds=600)
                    self._append('MATH_RETRIED', {'id': job['id'], 'previous_claim_id': job['claim_id'],
                        'claim_id': claim['claim_id'], 'artifact_ref': self._ref(claim['artifact_directory']),
                        'attempt': job['attempt'] + 1})
                except (ValueError, OSError) as exc:
                    self._append('MATH_RECOVERY_STOPPED', {'id': job['id'], 'claim_id': job['claim_id'],
                        'reason': _reason('recovery_reservation_rejected', str(exc))})

    def _execute(self, job):
        certificate = make_certificate()
        moment = verify_certificate(certificate)
        if moment.get('valid') is not True:
            raise ValueError('independent exact Gaussian moment checker rejected generated certificate')
        directory = self._path(job['artifact_ref'])
        self._write_moment_certificate(job['artifact_ref'] + '/moment-certificate.json',
                                       (json.dumps(certificate, indent=2) + '\n').encode())
        result = prove_profile(directory, toolchain=self.toolchain)
        if result.get('status') != 'scoped_kernel_checked':
            diagnosis = 'timeout' if any(c.get('timed_out') for c in result.get('commands', [])) else 'checker_failure'
            reason = _reason(result.get('reason_code') or 'native_worker_failed', result.get('reason') or 'Fixed scalar worker failed.', diagnosis)
            return {'status': 'blocked' if result.get('status') == 'blocked' else 'failed', 'reason': reason,
                'moment': moment, 'worker_attempts': result.get('attempts', []),
                'failure_report': self._failure(job, reason), 'native_capability': 'blocked'}
        receipt = Path(result.get('receipt_path', ''))
        receipt_ref = self._ref(receipt)
        if not receipt_ref.startswith(job['artifact_ref'] + '/'):
            raise ValueError('worker returned receipt outside its assigned claim directory')
        completion = self.conductor.complete_claim(job['claim_id'], self._path(receipt_ref))
        verification = completion['verification']
        if verification.get('valid') is not True or verification.get('fresh_replay') is not True:
            raise ValueError('completion lacks fresh independent native verification')
        return {'status': 'scoped_kernel_checked', 'reason': None, 'receipt_ref': receipt_ref,
            'receipt_sha256': hashlib.sha256(self._read_retained(receipt_ref, 2 * 1024 * 1024)).hexdigest(),
            'verification': verification, 'moment': moment, 'worker_attempts': completion['worker_attempts'],
            'native_capability': 'configured'}

    def run_pending(self):
        processed = 0
        with self._process_lease() as acquired:
            if not acquired: return 0
            self._recover()
            with self._transaction():
                state = self._read()
                job = next((copy.deepcopy(j) for j in state['jobs'].values() if j['status'] == 'queued'), None)
                if job is None: return 0
                token = uuid.uuid4().hex
                self._append('MATH_STARTED', {'id': job['id'], 'claim_id': job['claim_id'],
                    'attempt': job['attempt'], 'token': token, 'started_at': time.time()})
            try:
                with self.conductor._transaction():
                    current = self.conductor.snapshot()
                    claim = next(c for c in current['claims'] if c['claim_id'] == job['claim_id'])
                    task = next(t for t in current['tasks'] if t['task_id'] == job['task_id'])
                    if claim['state'] != 'active' or task['stale'] or task['scope'] != job['scope']:
                        raise ValueError('runtime claim expired, was reported, or has stale scope/inputs')
                result = self._execute(job)
            except Exception as exc:
                reason = _reason('checker_execution_rejected', str(exc))
                result = {'status': 'failed', 'reason': reason, 'failure_report': self._failure(job, reason)}
            with self._transaction():
                self._append('MATH_FINISHED', {'id': job['id'], 'token': token,
                    'finished_at': time.time(), 'result': result})
                self._read()  # Validate the actual durable result before serving it.
            if 'native_capability' in result:
                self._native_state = (job['capability_signature'], result['native_capability'])
            processed += 1
        return processed

    def start(self):
        with self._thread_mutex:
            if self.thread is not None and self.thread.is_alive(): return
            self._stop.clear()
            def loop():
                while not self._stop.is_set():
                    try:
                        self.run_pending()
                        with self._transaction():
                            urgent = any(j['status'] in {'queued', 'running', 'interrupted'} for j in self._read()['jobs'].values())
                    except (ValueError, OSError):
                        # Fail closed on a corrupt journal; API reads expose the
                        # same error and no further native work is scheduled.
                        self._stop.set()
                        return
                    self._stop.wait(.1 if urgent else 2.)
            self.thread = threading.Thread(target=loop, name='frontier-fixed-math', daemon=True)
            self.thread.start()

    def stop(self):
        self._stop.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread(): thread.join(timeout=2)

    def learn(self, epochs=40, seed=0):
        if type(epochs) is not int or not 1 <= epochs <= 200:
            raise ValueError('epochs must be an integer between 1 and 200')
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError('seed must be a bounded signed integer')
        ir = self.conductor.snapshot()['ir']
        report = train_experiment(ir, seed=seed, epochs=epochs)
        with self._transaction():
            self._append('LEARNING_RECORDED', {'ir_sha256': ir['ir_sha256'], 'epochs': epochs,
                'seed': seed, 'report': report, 'report_sha256': digest(report)})
            self._read()
        return copy.deepcopy(report)

    def snapshot(self):
        with self._transaction():
            conductor = self.conductor.snapshot()
            state = self._read()
        ir = conductor['ir']
        suggestions = self._derived.get(ir['ir_sha256'])
        if suggestions is None:
            suggestions = rank_connections(ir, limit=10)
            self._derived = {ir['ir_sha256']: suggestions}
        jobs = copy.deepcopy(list(state['jobs'].values()))
        for job in jobs:
            if job['status'] == 'scoped_kernel_checked':
                job['historical_result'] = True
                completion = next((c for c in conductor['completions'] if c['claim_id'] == job['claim_id']), None)
                if completion is not None:
                    job['current_verification'] = copy.deepcopy(completion['verification'])
        urgent = any(j['status'] in {'queued', 'running', 'interrupted'} for j in jobs)
        signature, native = self._capability()
        if native == 'startup_pending':
            historical = next((j for j in reversed(jobs) if j['capability_signature'] == signature
                               and j.get('native_capability') in {'configured', 'blocked'}), None)
            if historical: native = historical['native_capability']
        return {'version': '19.0', 'math': {'canonical_records': len(ir['nodes']),
                    'prerequisites': sum(len(e['premises']) for e in ir['hyperedges']), 'full_theorem_certificates': 0},
            'conductor': conductor, 'beliefs': self.beliefs.snapshot(ir),
            'discovery': {'suggestions': copy.deepcopy(suggestions), 'learning': state['learning']},
            'evolution': self.evolution.snapshot(), 'annotations': self.annotations.snapshot(),
            'jobs': copy.deepcopy(jobs), 'tick': {'mode': 'urgent' if urgent else 'idle', 'interval_ms': 100 if urgent else 2000},
            'capabilities': {'native_lean': native, 'llm_backend': 'unconfigured', 'wasm': 'unconfigured',
                             'zk': 'unconfigured', 'remote_auth': 'unconfigured'}}

    def _geometry_key(self):
        try:
            info = self.coordinator.journal.path.stat()
            journal = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        except FileNotFoundError:
            journal = None
        return self._geometry_baseline_sha256, journal

    def positions(self):
        # Cache presentation bytes only. Proof/job/backend changes do not alter
        # geometry and must never force verifier work in the 10Hz stream path.
        with self._geometry_mutex:
            key = self._geometry_key()
            if self._geometry_cache is not None and self._geometry_cache[0] == key:
                return self._geometry_cache[1]
            with self.conductor._transaction():
                key = self._geometry_key()
                state = self.conductor.snapshot()
                ir = state['ir']
                sequence = len(self.coordinator.journal.events)
            nodes = sorted(ir['nodes'], key=lambda n: n['id'])
            parents = {e['conclusion']: e['premises'] for e in ir['hyperedges']}
            depths = {}
            def depth(record):
                if record not in depths: depths[record] = max((depth(p) + 1 for p in parents[record]), default=0)
                return depths[record]
            geometry = []
            for index, node in enumerate(nodes):
                level = depth(node['id'])
                angle = index * math.pi * (3 - math.sqrt(5))
                radius = 1 + .12 * level
                geometry.append({'x': radius * math.cos(angle), 'y': (level - 4) * .25,
                    'z': radius * math.sin(angle), 'size': .035 + .002 * len(parents[node['id']]),
                    'r': .2 + .06 * min(level, 8), 'g': .65, 'b': .95 - .05 * min(level, 8), 'a': 1.})
            frame = encode_positions(geometry, sequence=sequence % 2**32,
                                     graph_sha256=ir['ir_sha256'], synthetic=False)
            self._geometry_cache = (key, frame)
            return frame

    def cursor(self):
        paths = [self.journal.path, self.conductor.journal.path, self.evolution.journal.path,
                 self.annotations.journal.path, self.coordinator.journal.path, self.beliefs.path]
        stamps = []
        for path in paths:
            try:
                stat = path.stat(); stamps.append([str(path.relative_to(self.workdir)), stat.st_mtime_ns, stat.st_size])
            except (OSError, ValueError): stamps.append([path.name, None])
        stamp = digest(stamps)
        if self._cursor_cache is None or self._cursor_cache[0] != stamp:
            # Refresh only lightweight claim times when the journal changes; no
            # repeated IR compilation/native replay in the 250ms SSE poll path.
            with self.conductor.journal.transaction():
                self.conductor.journal._load()
                deadlines = sorted(e['payload']['expires_at'] for e in self.conductor.journal.events
                                   if e['type'] == 'TASK_CLAIMED')
            self._cursor_cache = (stamp, deadlines)
        expired = sum(deadline <= time.time() for deadline in self._cursor_cache[1])
        return digest([stamp, expired])

# Server import retained as a short public spelling.
FrontierRuntime = PublicFrontierRuntime
