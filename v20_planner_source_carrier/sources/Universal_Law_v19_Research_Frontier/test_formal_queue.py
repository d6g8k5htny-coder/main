"""Queue custody/recovery tests; only real Lean can establish a positive pass."""
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import formal_lane
try:
    import formal_queue
except ImportError:
    formal_queue = None

HERE = Path(__file__).resolve().parent
PROFILE = formal_lane.PROFILE_ID
REAL = Path(os.environ.get('UL_LEAN_TEST_TOOLCHAIN') or
            os.environ.get('UL_LEAN_TOOLCHAIN') or
            Path(__file__).resolve().parent/'toolchains/lean-4.34.1-linux-x86_64')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(formal_queue, 'durable scoped queue is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = formal_queue.FormalQueue(self.root, toolchain=self.root / 'absent')

    def test_enqueue_is_durable_idempotent_and_deduplicates_active_input(self):
        with patch.object(formal_lane, 'run_profile', side_effect=AssertionError('enqueue ran kernel')):
            first = self.queue.enqueue(PROFILE, event_id='owner-click-1')
            self.assertEqual(first['status'], 'queued')
            same = self.queue.enqueue(PROFILE, event_id='owner-click-2')
            self.assertEqual(first['id'], same['id'])
            again = formal_queue.FormalQueue(self.root).enqueue(PROFILE, event_id='owner-click-1')
            self.assertEqual(again['id'], first['id'])
        snap = self.queue.snapshot()
        self.assertEqual(len(snap['jobs']), 1)
        self.assertIn('owner-click-2', snap['jobs'][0]['event_ids'])
        self.assertEqual(snap['full_theorem_certificates'], [])
        self.assertIn('statements_sha256', first['bindings'])
        self.assertIn('checker_pins_sha256', first['bindings'])

    def test_unknown_profiles_and_caller_payload_are_rejected_without_job(self):
        for value in ['../../pwn', {'source': 'by decide'}, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.queue.enqueue(value)
        self.assertEqual(self.queue.snapshot()['jobs'], [])

    def test_missing_toolchain_requires_explicit_retry_and_caps_attempts(self):
        job = self.queue.enqueue(PROFILE)
        self.queue.run_pending()
        blocked = self.queue.snapshot()['jobs'][0]
        self.assertEqual(blocked['status'], 'blocked')
        self.assertEqual(blocked['reason_code'], 'toolchain_missing')
        self.queue.run_pending()
        self.assertEqual(len(self.queue.snapshot()['jobs'][0]['attempts']), 1)
        for _ in range(2):
            self.queue.retry(job['id'])
            self.queue.run_pending()
        exhausted = self.queue.retry(job['id'])
        self.assertEqual(exhausted['status'], 'escalated')
        self.assertEqual(len(exhausted['attempts']), 3)
        self.assertEqual(exhausted['full_theorem_certificates'], [])

    def test_source_change_blocks_before_worker_executes(self):
        job = self.queue.enqueue(PROFILE)
        source = self.root / 'sources'
        shutil.copytree(HERE / 'formal_examples', source)
        target = source / 'UniversalLaw/Side24/Endpoints.lean'
        target.write_bytes(target.read_bytes() + b'\n-- changed\n')
        with patch.object(formal_lane, 'SOURCE_ROOT', source), patch.object(formal_lane, 'run_profile', side_effect=AssertionError('must block first')):
            self.queue.run_pending()
        result = self.queue.snapshot()['jobs'][0]
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason_code'], 'input_integrity_mismatch')

    def test_fake_exit_zero_toolchain_cannot_pass(self):
        tool = self.root / 'fake'
        (tool / 'bin').mkdir(parents=True)
        for name in ['lean', 'leanchecker']:
            file = tool / 'bin' / name
            file.write_text('#!/bin/sh\nexit 0\n')
            file.chmod(0o755)
        queue = formal_queue.FormalQueue(self.root, toolchain=tool)
        queue.enqueue(PROFILE)
        queue.run_pending()
        result = queue.snapshot()['jobs'][0]
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason_code'], 'toolchain_integrity_mismatch')

    def test_worker_exception_is_durable_failure_and_retry_is_fresh(self):
        self.queue.enqueue(PROFILE)
        with patch.object(formal_lane, 'run_profile', side_effect=RuntimeError('injected crash')):
            self.queue.run_pending()
        job = self.queue.snapshot()['jobs'][0]
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['reason_code'], 'worker_exception')
        old = job['attempts'][0]['token']
        self.queue.retry(job['id'])
        self.queue.run_pending()
        job = self.queue.snapshot()['jobs'][0]
        self.assertNotEqual(job['attempts'][1]['token'], old)
        self.assertEqual(job['attempts'][0]['reason_code'], 'worker_exception')

    def test_background_start_does_not_block_enqueue(self):
        gate = threading.Event()
        entered = threading.Event()
        def failure(*args, **kwargs):
            entered.set()
            gate.wait(5)
            return {'status': 'failed', 'reason_code': 'injected_failure', 'reason': 'negative', 'full_theorem_certificates': []}
        with patch.object(formal_lane, 'run_profile', side_effect=failure):
            self.queue.start()
            try:
                start = time.monotonic()
                self.queue.enqueue(PROFILE)
                self.assertLess(time.monotonic() - start, 1)
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.queue.snapshot()['jobs'][0]['status'], 'running')
            finally:
                gate.set()
                self.queue.stop()
        self.assertEqual(self.queue.snapshot()['jobs'][0]['status'], 'failed')

    def test_cross_instance_global_bound_and_running_dedup(self):
        other = formal_queue.FormalQueue(self.root, max_workers=4)
        gate = threading.Event()
        entered = threading.Event()
        count = []
        def failure(*args, **kwargs):
            count.append(1)
            entered.set()
            gate.wait(5)
            return {'status': 'failed', 'reason_code': 'negative', 'reason': 'control'}
        self.queue.enqueue(PROFILE)
        with patch.object(formal_lane, 'run_profile', side_effect=failure):
            worker = threading.Thread(target=self.queue.run_pending)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(other.snapshot()['max_workers'], 1)
                same = other.enqueue(PROFILE)
                other.run_pending()
                self.assertEqual(same['status'], 'running')
                self.assertEqual(len(count), 1)
            finally:
                gate.set()
                worker.join(5)

    def test_process_death_recovers_lease_and_fences_old_attempt(self):
        self.queue.enqueue(PROFILE)
        script = "import formal_queue, formal_lane, os; q=formal_queue.FormalQueue(%r); formal_lane.run_profile=lambda *a,**k: os._exit(19); q.run_pending()" % str(self.root)
        child = subprocess.run([sys.executable, '-c', script], cwd=HERE)
        self.assertEqual(child.returncode, 19)
        recovered = formal_queue.FormalQueue(self.root, toolchain=self.root / 'absent')
        job = recovered.snapshot()['jobs'][0]
        self.assertEqual(job['status'], 'queued')
        self.assertEqual(job['attempts'][0]['reason_code'], 'lease_recovered')
        old_token = job['attempts'][0]['token']
        self.assertFalse(recovered._complete(job['id'], old_token, {'status': 'scoped_kernel_checked'}, {'valid': True}))
        recovered.run_pending()
        self.assertEqual(recovered.snapshot()['jobs'][0]['status'], 'blocked')

    def test_crash_after_result_fsync_before_terminal_journal_fences_orphan(self):
        self.queue.enqueue(PROFILE)
        script = """import formal_queue, formal_lane, os
q = formal_queue.FormalQueue(%r)
formal_lane.run_profile = lambda *a, **k: {'status':'failed','reason_code':'negative_control','reason':'injected failure'}
original = q._append
def append(state, kind, data):
    if kind == 'attempt_finished':
        os._exit(23)
    return original(state, kind, data)
q._append = append
q.run_pending()
""" % str(self.root)
        child = subprocess.run([sys.executable, '-c', script], cwd=HERE)
        self.assertEqual(child.returncode, 23)
        recovered = formal_queue.FormalQueue(self.root, toolchain=self.root / 'absent')
        job = recovered.snapshot()['jobs'][0]
        self.assertEqual(job['status'], 'queued')
        old = job['attempts'][0]
        self.assertTrue((Path(old['artifact_dir']) / 'result.json').is_file())
        self.assertFalse(recovered._complete(job['id'], old['token'], {'status':'scoped_kernel_checked'}, {'valid': True}))
        recovered.run_pending()
        self.assertEqual(recovered.snapshot()['jobs'][0]['status'], 'blocked')
        self.assertEqual(len(recovered.snapshot()['jobs'][0]['attempts']), 2)

    def test_unverified_or_mismatched_receipt_is_rejected(self):
        self.queue.enqueue(PROFILE)
        job, slot, attempt_dir = self.queue._claim()
        try:
            # An apparently successful status and a caller boolean are no proof.
            claimed = {'status':'scoped_kernel_checked', 'profile_id':'different-profile', 'full_theorem_certificates':[]}
            self.assertTrue(self.queue._complete(job['id'], job['lease']['token'], claimed, {'valid':True}))
        finally:
            slot.close()
        result = self.queue.snapshot()['jobs'][0]
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason_code'], 'receipt_rejected')
        self.assertEqual(result['full_theorem_certificates'], [])

    def test_journal_tamper_and_complete_line_truncation_fail_closed(self):
        self.queue.enqueue(PROFILE)
        journal = self.root / 'runtime/formal_queue/events.jsonl'
        data = journal.read_text()
        changed = data.replace('queued', 'passed', 1)
        journal.write_text(changed)
        with self.assertRaises(formal_queue.JournalError):
            self.queue.snapshot()
        journal.write_text(data.rsplit('\n', 2)[0] + '\n')
        with self.assertRaises(formal_queue.JournalError):
            self.queue.snapshot()

    def test_malformed_utf8_journal_fails_closed_with_journal_error(self):
        self.queue.enqueue(PROFILE)
        journal = self.root / 'runtime/formal_queue/events.jsonl'
        data = journal.read_bytes()
        journal.write_bytes(b'\xff' + data[1:])
        with self.assertRaises(formal_queue.JournalError):
            self.queue.snapshot()

    def test_event_id_cannot_be_rebound_to_changed_checker_inputs(self):
        self.queue.enqueue(PROFILE, event_id='fixed-input-event')
        changed = formal_queue._bindings(PROFILE)
        changed['checker_implementation_sha256'] = '1' * 64
        with patch.object(formal_queue, '_bindings', return_value=changed):
            with self.assertRaises(ValueError):
                self.queue.enqueue(PROFILE, event_id='fixed-input-event')

    def test_partial_tail_recovery_preserves_events_and_sequence(self):
        self.queue.enqueue(PROFILE)
        before = self.queue.snapshot()['last_sequence']
        journal = self.root / 'runtime/formal_queue/events.jsonl'
        with journal.open('ab') as out:
            out.write(b'{"crashed-write":')
        reopened = formal_queue.FormalQueue(self.root)
        self.assertEqual(len(reopened.snapshot()['jobs']), 1)
        self.assertGreater(reopened.snapshot()['last_sequence'], before)
        self.assertTrue(list(journal.parent.glob('torn-tail-*')))


@unittest.skipUnless((REAL / 'bin/lean').is_file(), 'real pinned Lean unavailable')
class RealQueueTests(unittest.TestCase):
    def test_actual_scoped_kernel_job_has_real_verified_receipt_and_false_control(self):
        self.assertIsNotNone(formal_queue)
        with tempfile.TemporaryDirectory() as temp:
            queue = formal_queue.FormalQueue(os.environ.get('UL_FORMAL_QUEUE_EVIDENCE_WORKDIR', temp), toolchain=REAL)
            queue.enqueue(PROFILE, event_id='real-kernel-control')
            queue.run_pending()
            job = queue.snapshot()['jobs'][0]
            self.assertEqual(job['status'], 'scoped_kernel_checked', job)
            self.assertTrue(job['verification']['valid'])
            self.assertTrue(job['verification']['fresh_replay'])
            self.assertEqual(job['receipt']['axiom_free_count'], 31)
            self.assertEqual(job['receipt']['alignment_status'], 'pending_independent_review')
            self.assertEqual(job['full_theorem_certificates'], [])
            negative = next(row for row in job['receipt']['commands'] if row['label'] == 'false_endpoint_negative_control')
            self.assertNotEqual(negative['exit_code'], 0)
            self.assertIn('is false', negative['stdout'])
            self.assertIn('sha256', job['receipt_ref'])
            # A cached successful job is not an opportunity to import a late receipt.
            self.assertFalse(queue._complete(job['id'], job['attempts'][0]['token'], job['receipt'], job['verification']))


if __name__ == '__main__':
    unittest.main()
