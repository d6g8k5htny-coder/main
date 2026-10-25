"""Runtime contracts exercise real journals; mocked workers never certify math."""
import importlib
import json
import os
import threading
import subprocess
import sys
import time
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import flow  # Install the retained engine's import prefix before all components.
from conductor import Coordinator
from frontier_gpu import decode_positions

try:
    runtime_module = importlib.import_module('frontier_runtime')
except ImportError:
    runtime_module = None


class FrontierRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime_module, 'durable public frontier runtime missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.coord = Coordinator(self.directory / 'engine')
        self.runtime = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(self.runtime.stop)

    def test_public_snapshot_and_real_graph_transport(self):
        state = self.runtime.snapshot()
        self.assertEqual(set(state), {'version', 'math', 'conductor', 'beliefs', 'discovery',
                                      'evolution', 'annotations', 'jobs', 'tick', 'capabilities'})
        self.assertEqual(state['math'], {'canonical_records': 104, 'prerequisites': 108,
                                       'full_theorem_certificates': 0})
        frame = decode_positions(self.runtime.positions())
        self.assertEqual(frame['count'], 104)
        self.assertFalse(frame['synthetic'])
        self.assertEqual(frame['graph_sha256'], state['conductor']['ir']['ir_sha256'])
        self.assertEqual(self.runtime.positions(), self.runtime.positions())
        self.assertEqual(state['capabilities']['native_lean'], 'blocked')
        self.assertEqual(state['tick'], {'mode': 'idle', 'interval_ms': 2000})

    def test_positions_cache_avoids_replay_and_refreshes_after_engine_change(self):
        with patch.object(self.runtime.conductor, 'snapshot', wraps=self.runtime.conductor.snapshot) as wrapped:
            frames = [self.runtime.positions() for _ in range(10)]
            self.assertEqual(wrapped.call_count, 1)
        self.assertTrue(all(frame is frames[0] for frame in frames))
        before = decode_positions(frames[0])
        self.runtime.queue_math(event_id='geometry-ignores-proof-jobs')
        self.runtime.run_pending()
        with patch.object(self.runtime.conductor, 'snapshot', side_effect=AssertionError('geometry rechecked proof verdict')):
            self.assertIs(self.runtime.positions(), frames[0])
        record = next(r for r in self.coord.base['records'] if r['id'] == 'D1-A')
        proposal = self.coord.propose('statement', 'D1-A',
            record['statement'] + ' Cache invalidation test amendment.', actor='human:test')
        self.coord.decide(proposal['proposal_id'], True, actor='human:test')
        with patch.object(self.runtime.conductor, 'snapshot', wraps=self.runtime.conductor.snapshot) as wrapped:
            changed_frames = [self.runtime.positions() for _ in range(10)]
            self.assertEqual(wrapped.call_count, 1)
        after = decode_positions(changed_frames[0])
        self.assertNotEqual(before['graph_sha256'], after['graph_sha256'])
        self.assertGreater(after['sequence'], before['sequence'])
        self.assertEqual(after['count'], 104)
        self.assertFalse(after['synthetic'])
        self.assertEqual(after['graph_sha256'], self.runtime.conductor.snapshot()['ir']['ir_sha256'])

    def test_blocked_work_is_durable_deduplicated_and_budgeted(self):
        before = self.runtime.cursor()
        first = self.runtime.queue_math(event_id='missing-backend')
        self.assertEqual(first['status'], 'queued')
        self.assertTrue(first['claim_id'])
        self.assertNotEqual(before, self.runtime.cursor())
        self.assertEqual(self.runtime.queue_math(event_id='missing-backend')['id'], first['id'])
        self.assertEqual(self.runtime.snapshot()['tick']['interval_ms'], 100)
        self.runtime.run_pending()
        job = self.runtime.snapshot()['jobs'][0]
        self.assertEqual(job['status'], 'blocked')
        self.assertEqual(job['reason']['diagnosis'], 'checker_failure')
        self.assertNotIn('receipt_ref', job)
        restored = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(restored.stop)
        self.assertEqual(restored.queue_math(event_id='missing-backend')['id'], first['id'])
        state = restored.snapshot()
        self.assertEqual(len(state['conductor']['claims']), 1)
        self.assertEqual(state['conductor']['budget']['remaining'], 19)
        self.assertFalse(state['conductor']['completions'])
        self.assertEqual(state['math']['full_theorem_certificates'], 0)
        self.assertFalse(list(restored.root.rglob('receipt.json')))

    def test_no_argument_accepts_math_source_receipt_or_profile(self):
        for argument in ['source', 'receipt_path', 'profile_id', 'ir', 'code']:
            with self.assertRaises(TypeError):
                self.runtime.queue_math(**{argument: 'untrusted'})
        for event_id in ['', 4, 'x' * 181, '\n']:
            with self.assertRaises(ValueError):
                self.runtime.queue_math(event_id=event_id)

    def test_distinct_delivery_ids_alias_identical_live_job_durably(self):
        first = self.runtime.queue_math(event_id='first-delivery')
        alias = self.runtime.queue_math(event_id='second-delivery')
        self.assertEqual(alias['id'], first['id'])
        self.assertEqual(self.runtime.queue_math()['id'], first['id'])
        self.runtime.run_pending()
        restored = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(restored.stop)
        self.assertEqual(restored.queue_math(event_id='second-delivery')['id'], first['id'])
        self.assertEqual(restored.queue_math()['id'], first['id'])
        self.assertEqual(len(restored.snapshot()['conductor']['claims']), 1)

    def test_empty_native_environment_disables_bundle_and_never_executes(self):
        self.runtime.toolchain = None
        self.runtime.conductor.toolchain = None
        with patch.dict(os.environ, {'UL_LEAN_TOOLCHAIN': ''}):
            self.assertIsNone(self.runtime._selected_toolchain())
            self.assertEqual(self.runtime.snapshot()['capabilities']['native_lean'], 'blocked')
            self.runtime.queue_math(event_id='explicitly-disabled-native')
            with patch('subprocess.Popen', side_effect=AssertionError('native invocation forbidden')):
                self.runtime.run_pending()
            state = self.runtime.snapshot()
        self.assertEqual(state['jobs'][0]['status'], 'blocked')
        self.assertEqual(state['jobs'][0]['reason']['code'], 'toolchain_missing')
        self.assertFalse(list(self.runtime.root.rglob('receipt.json')))
        self.assertFalse(state['conductor']['completions'])

    def test_existing_budget_is_not_replenished(self):
        self.runtime.conductor.blueprint(['d5-second-factorial-upper'], budget=1)
        self.runtime.queue_math(event_id='one-reservation')
        self.runtime.run_pending()
        with self.assertRaises(ValueError):
            self.runtime.queue_math(event_id='another-reservation')
        self.assertEqual(len(self.runtime.snapshot()['jobs']), 1)

    def test_forged_worker_success_cannot_complete_claim(self):
        job = self.runtime.queue_math(event_id='forged')
        with patch.object(runtime_module, 'prove_profile', return_value={
                'status': 'scoped_kernel_checked', 'receipt_path': '/etc/passwd'}):
            self.runtime.run_pending()
        state = self.runtime.snapshot()
        self.assertEqual(state['jobs'][0]['status'], 'failed')
        self.assertFalse(state['conductor']['completions'])
        self.assertEqual(state['jobs'][0]['claim_id'], job['claim_id'])

    def test_learning_bound_persistence_and_component_cursor(self):
        for epochs in [0, 201, True]:
            with self.assertRaises(ValueError): self.runtime.learn(epochs=epochs)
        before = self.runtime.cursor()
        report = self.runtime.learn(epochs=2, seed=7)
        self.assertNotEqual(before, self.runtime.cursor())
        self.assertEqual(self.runtime.snapshot()['discovery']['learning'], report)
        restored = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(restored.stop)
        self.assertEqual(restored.snapshot()['discovery']['learning'], report)
        before = restored.cursor()
        restored.beliefs.assert_claim('test proposition', 'exact test scope', 'support', 'test')
        self.assertNotEqual(before, restored.cursor())

    def test_interrupted_worker_gets_new_budgeted_claim_without_adopting_output(self):
        first = self.runtime.queue_math(event_id='crash-recovery')
        token = 'a' * 32
        self.runtime.journal.append('MATH_STARTED', {'id': first['id'], 'claim_id': first['claim_id'],
            'attempt': 1, 'token': token, 'started_at': time.time()})
        orphan = self.runtime.root / first['artifact_ref'] / 'receipt.json'
        orphan.write_text(json.dumps({'status': 'scoped_kernel_checked', 'valid': True}))
        restored = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(restored.stop)
        restored.run_pending()
        state = restored.snapshot()
        job = state['jobs'][0]
        self.assertEqual(job['status'], 'blocked')
        self.assertEqual(job['attempt'], 2)
        self.assertNotEqual(job['claim_id'], first['claim_id'])
        self.assertEqual(job['claim_history'][0], first['claim_id'])
        self.assertEqual(len(state['conductor']['claims']), 2)
        self.assertEqual(state['conductor']['budget']['remaining'], 18)
        self.assertFalse(state['conductor']['completions'])
        self.assertEqual(json.loads(orphan.read_text())['valid'], True)
        self.assertNotIn('receipt_ref', job)

    def test_interrupted_attempts_stop_after_three_without_reclaiming_budget(self):
        first = self.runtime.queue_math(event_id='bounded-crashes')
        for attempt in range(1, 4):
            current = self.runtime.snapshot()['jobs'][0]
            self.runtime.journal.append('MATH_STARTED', {'id': first['id'], 'claim_id': current['claim_id'],
                'attempt': attempt, 'token': str(attempt) * 32, 'started_at': time.time()})
            with self.runtime._process_lease() as acquired:
                self.assertTrue(acquired)
                self.runtime._recover()
        state = self.runtime.snapshot()
        self.assertEqual(state['jobs'][0]['status'], 'failed')
        self.assertEqual(state['jobs'][0]['reason']['code'], 'runtime_attempt_cap')
        self.assertEqual(state['jobs'][0]['attempt'], 3)
        self.assertEqual(len(state['conductor']['claims']), 3)
        self.assertEqual(state['conductor']['budget']['remaining'], 17)
        self.assertFalse(state['conductor']['completions'])

    def test_process_lock_prevents_live_worker_recovery(self):
        entered, release = threading.Event(), threading.Event()
        second = runtime_module.PublicFrontierRuntime(
            self.directory, self.coord, toolchain=self.directory / 'missing-toolchain')
        self.addCleanup(second.stop)
        self.runtime.queue_math(event_id='one-live-process')
        def blocked_worker(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=10))
            return {'status': 'blocked', 'reason_code': 'test_backend_missing', 'reason': 'missing'}
        with patch.object(runtime_module, 'prove_profile', side_effect=blocked_worker):
            thread = threading.Thread(target=self.runtime.run_pending)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=10))
                self.assertEqual(second.run_pending(), 0)
                self.assertEqual(second.snapshot()['jobs'][0]['status'], 'running')
            finally:
                release.set()
                thread.join(timeout=15)
        state = second.snapshot()
        self.assertEqual(len(state['conductor']['claims']), 1)
        self.assertEqual(state['jobs'][0]['attempt'], 1)

    def test_queued_runtime_relocates_without_original_directory(self):
        first = self.runtime.queue_math(event_id='portable-queue')
        moved = self.directory / 'relocated'
        shutil.copytree(self.directory / 'frontier', moved / 'frontier')
        shutil.copytree(self.directory / 'engine', moved / 'engine')
        shutil.rmtree(self.directory / 'frontier')
        restored = runtime_module.PublicFrontierRuntime(
            moved, Coordinator(moved / 'engine'), toolchain=moved / 'missing-toolchain')
        self.addCleanup(restored.stop)
        self.assertEqual(restored.queue_math(event_id='portable-queue')['id'], first['id'])
        restored.run_pending()
        self.assertEqual(restored.snapshot()['jobs'][0]['status'], 'blocked')
        raw = (moved / 'frontier/jobs/events.jsonl').read_text()
        self.assertNotIn(str(self.directory / 'missing-toolchain'), raw)
        self.assertNotIn(str(self.directory / 'frontier'), raw)
        self.assertTrue((moved / 'frontier' / first['artifact_ref'] / 'moment-certificate.json').is_file())

    def test_expired_claim_changes_cursor_and_cannot_run(self):
        job = self.runtime.queue_math(event_id='expired')
        before = self.runtime.cursor()
        future = time.time() + 601
        with patch.object(runtime_module.time, 'time', return_value=future):
            self.assertNotEqual(before, self.runtime.cursor())
            self.runtime.run_pending()
        state = self.runtime.snapshot()
        self.assertEqual(state['jobs'][0]['status'], 'failed')
        self.assertFalse(state['conductor']['completions'])
        self.assertEqual(state['jobs'][0]['claim_id'], job['claim_id'])

    @unittest.skipUnless(os.environ.get('UL_RUN_FRONTIER_NATIVE_TESTS') == '1',
                         'set UL_RUN_FRONTIER_NATIVE_TESTS=1 for genuine native recovery')
    def test_committed_native_completion_reconciles_after_delivery_crash_once(self):
        toolchain = Path(__file__).resolve().parent / 'toolchains/lean-4.34.1-linux-x86_64'
        self.runtime.toolchain = toolchain
        self.runtime.conductor.toolchain = toolchain
        first = self.runtime.queue_math(event_id='committed-native-before-delivery-crash')
        append = self.runtime._append
        def crash_before_terminal(kind, payload):
            if kind == 'MATH_FINISHED':
                raise RuntimeError('simulated crash after conductor completion before runtime terminal event')
            return append(kind, payload)
        with patch.object(self.runtime, '_append', side_effect=crash_before_terminal):
            with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                self.runtime.run_pending()
        before = self.runtime.snapshot()
        self.assertEqual(before['jobs'][0]['status'], 'running')
        self.assertEqual(len(before['conductor']['completions']), 1)
        restored = runtime_module.PublicFrontierRuntime(self.directory, self.coord, toolchain=toolchain)
        self.addCleanup(restored.stop)
        self.assertEqual(restored.run_pending(), 0)
        state = restored.snapshot()
        job = state['jobs'][0]
        self.assertEqual(job['id'], first['id'])
        self.assertEqual(job['claim_id'], first['claim_id'])
        self.assertEqual(job['status'], 'scoped_kernel_checked')
        self.assertEqual(job['reason']['code'], 'reconciled_verified_conductor_completion')
        self.assertTrue(job['verification']['valid'])
        self.assertTrue(job['verification']['fresh_replay'])
        self.assertTrue(job['moment']['valid'])
        self.assertEqual(len(state['conductor']['claims']), 1)
        self.assertEqual(len(state['conductor']['completions']), 1)
        self.assertEqual(state['conductor']['budget']['spent'], 1)
        self.assertEqual(state['conductor']['budget']['remaining'], 19)
        self.assertEqual(state['math']['full_theorem_certificates'], 0)
        self.assertEqual(restored.run_pending(), 0)
        self.assertEqual(restored.queue_math(event_id='committed-native-before-delivery-crash')['id'], first['id'])
        events = restored.journal.since(0)
        self.assertEqual(sum(e['type'] == 'MATH_RECONCILED' for e in events), 1)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO required')
    def test_retained_reader_rejects_fifo_symlink_and_oversize_under_timeout(self):
        artifact = self.runtime.root / 'reader-controls'
        artifact.mkdir()
        (artifact / 'ordinary.json').write_bytes(b'{}')
        os.mkfifo(artifact / 'fifo.json')
        (artifact / 'link.json').symlink_to(artifact / 'ordinary.json')
        (artifact / 'oversize.json').write_bytes(b'x' * 17)
        script = r"""
import sys
from pathlib import Path
import flow
from conductor import Coordinator
from frontier_runtime import FrontierRuntime
root = Path(sys.argv[1])
runtime = FrontierRuntime(root, Coordinator(root / 'engine'), toolchain=root / 'missing-toolchain')
assert runtime._read_retained('reader-controls/ordinary.json', 16) == b'{}'
for name in ['fifo.json', 'link.json', 'oversize.json']:
    try:
        runtime._read_retained('reader-controls/' + name, 16)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted unbounded/nonregular file: ' + name)
print('all nonregular/oversize controls rejected')
"""
        result = subprocess.run([sys.executable, '-c', script, str(self.directory)],
            cwd=Path(__file__).resolve().parent, timeout=2, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('all nonregular/oversize controls rejected', result.stdout)

    @unittest.skipUnless(os.environ.get('UL_RUN_FRONTIER_NATIVE_TESTS') == '1' and hasattr(os, 'mkfifo'),
                         'set UL_RUN_FRONTIER_NATIVE_TESTS=1 for genuine committed FIFO recovery')
    def test_committed_native_fifo_moment_recovery_stops_under_restart_timeout(self):
        toolchain = Path(__file__).resolve().parent / 'toolchains/lean-4.34.1-linux-x86_64'
        self.runtime.toolchain = toolchain
        self.runtime.conductor.toolchain = toolchain
        first = self.runtime.queue_math(event_id='committed-native-fifo-moment')
        append = self.runtime._append
        def crash_before_terminal(kind, payload):
            if kind == 'MATH_FINISHED':
                raise RuntimeError('simulated committed native crash')
            return append(kind, payload)
        with patch.object(self.runtime, '_append', side_effect=crash_before_terminal):
            with self.assertRaisesRegex(RuntimeError, 'simulated committed native crash'):
                self.runtime.run_pending()
        self.assertEqual(len(self.runtime.conductor.snapshot()['completions']), 1)
        moment = self.runtime.root / first['artifact_ref'] / 'moment-certificate.json'
        moment.unlink()
        os.mkfifo(moment)
        script = r"""
import json
import sys
from pathlib import Path
import flow
from conductor import Coordinator
from frontier_runtime import FrontierRuntime
root = Path(sys.argv[1])
runtime = FrontierRuntime(root, Coordinator(root / 'engine'), toolchain=root / 'missing-toolchain')
runtime.run_pending()
state = runtime.snapshot()
job = state['jobs'][0]
assert job['status'] == 'failed', job
assert job['reason']['code'] == 'recovery_artifact_rejected', job
assert len(state['conductor']['completions']) == 1
assert len(state['conductor']['claims']) == 1
assert state['math']['full_theorem_certificates'] == 0
assert not any(e['type'] == 'MATH_RECONCILED' for e in runtime.journal.since(0))
print(json.dumps({'status': job['status'], 'reason': job['reason'], 'retained_completions': 1, 'full_theorem_certificates': 0}))
"""
        result = subprocess.run([sys.executable, '-c', script, str(self.directory)],
            cwd=Path(__file__).resolve().parent, timeout=2, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'failed')

    def test_success_flags_without_conductor_completion_are_rejected(self):
        from moment_certificates import make_certificate
        from moment_checker import verify_certificate
        job = self.runtime.queue_math(event_id='forged-journal')
        token = 'b' * 32
        self.runtime.journal.append('MATH_STARTED', {'id': job['id'], 'claim_id': job['claim_id'],
            'attempt': 1, 'token': token, 'started_at': time.time()})
        self.runtime.journal.append('MATH_FINISHED', {'id': job['id'], 'token': token,
            'finished_at': time.time(), 'result': {'status': 'scoped_kernel_checked',
                'verification': {'valid': True, 'fresh_replay': True, 'actual_field_applicable': False,
                                 'full_theorem_certificates': []},
                'receipt_ref': job['artifact_ref'] + '/fake/receipt.json', 'receipt_sha256': '0' * 64,
                'moment': verify_certificate(make_certificate()), 'worker_attempts': []}})
        with self.assertRaises(ValueError): self.runtime.snapshot()

    def test_unknown_transition_and_forged_completion_fail_closed(self):
        self.runtime.journal.append('APPROVE_MATHEMATICS', {'approved': True})
        with self.assertRaises(ValueError): self.runtime.snapshot()


if __name__ == '__main__': unittest.main()
