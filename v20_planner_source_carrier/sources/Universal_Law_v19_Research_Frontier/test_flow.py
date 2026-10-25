"""Behavioral tests for evidence flow, integrity, recovery and branch isolation."""
import copy
import hashlib
import json
import tempfile
import threading
import unittest
import subprocess
import sys
import zipfile
from pathlib import Path

from flow import Intake, Coordinator, verify_runtime, snapshot_loop
from conductor import downstream_closure
from research_evidence import revision_id
from workers import verify_receipt


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.coord = Coordinator(self.root / "runtime", max_attempts=1)
        self.config = {"source_namespace": "test", "patterns": ["*.md"],
                       "auto_register_evidence": True,
                       "rules": {"finding.md": {"classification": "new_derivation",
                           "scope": "A scoped component, not a complete theorem.",
                           "limitations": ["Mathematical applicability remains unvalidated."],
                           "links": [{"record_id": "D1-KR-FAR", "relation": "context_only",
                                      "reason": "Review relevance only."}]}}}
        self.intake = Intake(self.coord, self.sources, self.config)
        self.file = self.sources / "finding.md"
        self.file.write_bytes(b"First discovery\r\n")

    def tearDown(self):
        self.coord.stop()
        self.temporary.cleanup()

    def test_capture_preserves_bytes_and_revision_history(self):
        first = self.intake.capture("finding.md")
        self.assertEqual((self.coord.journal.workdir / first["artifact_path"]).read_bytes(), b"First discovery\r\n")
        self.file.write_bytes(b"Second discovery\n")
        self.intake.capture("finding.md")
        self.assertEqual(self.coord.snapshot()["research_revision_count"], 2)
        self.assertEqual((self.coord.journal.workdir / first["artifact_path"]).read_bytes(), b"First discovery\r\n")

    def test_watcher_does_not_mark_another_source_namespace_missing(self):
        other = self.root / 'other-sources'
        other.mkdir()
        (other / 'other.md').write_text('Independently preserved mathematical note')
        config = {'source_namespace': 'other', 'patterns': ['*.md'],
                  'auto_register_evidence': False, 'rules': {}}
        Intake(self.coord, other, config).capture('other.md')
        self.intake.scan()
        self.assertNotIn('other/other.md', self.coord.snapshot()['research_failures'])

    def test_namespace_requires_a_nonempty_single_segment(self):
        for namespace in ('', ' ', 'test/review', '/test', 'test/', 'test\\review', '.', '..', None, 1):
            with self.subTest(namespace=namespace), self.assertRaisesRegex(ValueError, 'source_namespace'):
                config = dict(self.config, source_namespace=namespace)
                Intake(self.coord, self.sources, config)

    def legacy_nested_pending(self):
        capture = copy.deepcopy(self.intake.capture('finding.md'))
        capture['source_key'] = 'test/review/finding.md'
        capture['revision_id'] = revision_id(capture)
        self.coord.journal.append('RESEARCH_CAPTURED', capture)
        self.coord.propose('research_evidence', 'D1-KR-FAR', {
            'kind': 'research_evidence', 'record_id': 'D1-KR-FAR', 'capture': capture,
            'relation': 'context_only', 'applicability': 'unvalidated', 'formal_status': 'not_certified'})
        return capture

    def test_legacy_nested_capture_is_never_registered_by_parent_namespace(self):
        capture = self.legacy_nested_pending()
        self.intake.recover_links()
        state = self.coord.snapshot()
        foreign = [p for p in state['proposals']
                   if json.loads(p['replacement'])['capture']['revision_id'] == capture['revision_id']]
        self.assertTrue(foreign)
        self.assertFalse(any(p['status'] == 'approved' for p in foreign))
        self.assertFalse(any(x['capture']['revision_id'] == capture['revision_id']
                             for x in state['research_evidence_attachments'].get('D1-KR-FAR', [])))
        self.assertEqual(state['full_theorem_certificates'], 0)

    def test_legacy_nested_capture_is_not_marked_missing_by_parent_namespace(self):
        self.config['auto_register_evidence'] = False
        capture = self.legacy_nested_pending()
        self.intake.scan()
        self.assertNotIn(capture['source_key'], self.coord.snapshot()['research_failures'])

    def test_automatic_registration_exact_closure_and_unchanged_mathematics(self):
        before = self.coord.snapshot()
        self.intake.scan()
        after = self.coord.snapshot()
        expected = downstream_closure(self.coord.base, ["D1-KR-FAR"])
        self.assertEqual(after["changed_ids"], expected)
        self.assertEqual(after["effective_dataset_sha256"], before["effective_dataset_sha256"])
        self.assertEqual(len(after["jobs"]), len(expected))
        self.assertEqual(after["full_theorem_certificates"], 0)
        for old, new in zip(before["records"], after["records"]):
            self.assertEqual(old["statement"], new["statement"])
            self.assertEqual(old["acceptance"], new["acceptance"])
            if old["id"] not in expected:
                self.assertEqual(old["fingerprint"], new["fingerprint"])

    def test_duplicate_delivery_and_restart_do_not_duplicate_jobs(self):
        self.intake.scan()
        before = self.coord.snapshot()
        again = Coordinator(self.coord.journal.workdir)
        Intake(again, self.sources, self.config).scan()
        after = again.snapshot()
        self.assertEqual(before["last_sequence"], after["last_sequence"])
        self.assertEqual(before["jobs"], after["jobs"])

    def test_crash_after_capture_recovers_registration(self):
        self.intake.capture("finding.md")
        self.assertFalse(self.coord.snapshot()["jobs"])
        again = Coordinator(self.coord.journal.workdir)
        Intake(again, self.sources, self.config).recover_links()
        self.assertTrue(again.snapshot()["jobs"])

    def test_conflicting_intake_policies_cannot_autoapprove_another_namespace(self):
        before = self.coord.snapshot()
        manual_config = copy.deepcopy(self.config)
        # A shared textual prefix must not turn test-review/ into test/.
        manual_config.update(source_namespace="test-review", auto_register_evidence=False)
        manual = Intake(self.coord, self.sources, manual_config)
        manual.scan()
        pending = self.coord.snapshot()["proposals"][0]
        self.assertEqual(pending["ui_status"], "awaiting_approval")
        self.intake.scan()
        state = self.coord.snapshot()
        by_namespace = {json.loads(p["replacement"])["capture"]["source_key"].split('/')[0]: p
                        for p in state["proposals"]}
        # Own registration can advance the branch and mark the manual proposal
        # stale; it must never approve it under the other intake's policy.
        self.assertEqual(by_namespace["test-review"]["status"], "awaiting_approval")
        self.assertIn(by_namespace["test-review"]["ui_status"], {"awaiting_approval", "stale"})
        self.assertEqual(by_namespace["test"]["status"], "approved")
        attached = state["research_evidence_attachments"]["D1-KR-FAR"]
        self.assertEqual([x["capture"]["source_key"] for x in attached], ["test/finding.md"])
        self.assertEqual(state["effective_dataset_sha256"], before["effective_dataset_sha256"])
        self.assertEqual(state["full_theorem_certificates"], 0)
        for old, new in zip(before["records"], state["records"]):
            self.assertEqual(old["statement"], new["statement"])
            self.assertEqual(old["acceptance"], new["acceptance"])

    def test_restart_recovers_only_own_namespace_and_preserves_manual_outbox(self):
        manual_config = copy.deepcopy(self.config)
        manual_config.update(source_namespace="test-review", auto_register_evidence=False)
        manual = Intake(self.coord, self.sources, manual_config)
        other_capture = manual.capture("finding.md")
        self.intake.capture("finding.md")
        again = Coordinator(self.coord.journal.workdir)
        try:
            self.assertEqual(Intake(again, self.sources, self.config).recover_links(), 1)
            state = again.snapshot()
            self.assertEqual(len(state["proposals"]), 1)
            self.assertEqual(json.loads(state["proposals"][0]["replacement"])["capture"]["source_key"],
                             "test/finding.md")
            self.assertEqual(Intake(again, self.sources, manual_config).recover_links(), 1)
            pending = next(p for p in again.snapshot()["proposals"]
                           if json.loads(p["replacement"])["capture"]["revision_id"] == other_capture["revision_id"])
            self.assertEqual(pending["ui_status"], "awaiting_approval")
            count = again.snapshot()["last_sequence"]
            self.assertEqual(Intake(again, self.sources, self.config).recover_links(), 0)
            self.assertEqual(again.snapshot()["last_sequence"], count)
            self.assertEqual(again.snapshot()["full_theorem_certificates"], 0)
            self.assertEqual((again.journal.workdir / other_capture["artifact_path"]).read_bytes(),
                             b"First discovery\r\n")
        finally:
            again.stop()

    def test_new_revision_rejects_stale_proposal(self):
        self.config["auto_register_evidence"] = False
        self.intake.scan()
        old = self.coord.snapshot()["proposals"][0]
        self.file.write_bytes(b"Changed finding\n")
        self.intake.scan()
        with self.assertRaisesRegex(ValueError, "stale research"):
            self.coord.decide(old["proposal_id"], True)
        self.assertEqual(self.coord.snapshot()["full_theorem_certificates"], 0)

    def test_tampered_object_blocks_registration(self):
        self.config["auto_register_evidence"] = False
        self.intake.scan()
        state = self.coord.snapshot()
        capture = state["research_discoveries"][0]
        path = self.coord.journal.workdir / capture["artifact_path"]
        path.write_bytes(b"Tampered bytes")
        with self.assertRaisesRegex(ValueError, "research artifact"):
            self.coord.decide(state["proposals"][0]["proposal_id"], True)
        self.assertEqual(verify_runtime(self.coord)["result"], "failed")
        self.assertFalse(self.coord.snapshot()["jobs"])

    def test_worker_fails_after_registered_evidence_is_tampered(self):
        self.intake.scan()
        capture = self.coord.snapshot()["research_discoveries"][0]
        (self.coord.journal.workdir / capture["artifact_path"]).write_bytes(b"corrupted")
        receipt = self.coord.run_one_job()
        self.assertEqual(receipt["result"], "failed")
        state = self.coord.snapshot()
        self.assertEqual(state["failed_job_count"], 1)
        self.assertTrue(state["changed_ids"])
        self.assertEqual(state["full_theorem_certificates"], 0)

    def test_worker_captures_and_replays_actual_research_bytes(self):
        self.intake.scan()
        receipt = self.coord.run_one_job()
        self.assertEqual(receipt["result"], "diagnostic_completed", receipt)
        self.assertEqual(receipt["diagnostic"]["research_inputs_checked"], 1)
        self.assertTrue(any(p["path"].startswith("research/objects/") for p in receipt["input_files"]))
        self.assertTrue(verify_receipt(receipt, replay=True)["replayed"])
        self.assertTrue(self.coord.snapshot()["changed_ids"])
        self.assertEqual(self.coord.snapshot()["full_theorem_certificates"], 0)

    def test_missing_source_keeps_evidence_and_reports_failure(self):
        self.intake.scan()
        self.file.unlink()
        self.intake.scan()
        state = self.coord.snapshot()
        self.assertEqual(state["research_discoveries"][0]["capture_state"], "source_unavailable")
        self.assertEqual(verify_runtime(self.coord)["result"], "passed")
        self.assertTrue(state["research_failures"])

    def test_repeated_missing_and_restore_cycles_recover(self):
        self.intake.scan()
        for _ in range(2):
            self.file.unlink()
            self.intake.scan()
            self.file.write_bytes(b"First discovery\r\n")
            self.intake.scan()
            self.assertFalse(self.coord.snapshot()["research_failures"])

    def test_paths_symlinks_and_size_limits_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.intake.capture("../outside.md")
        outside = self.root / "outside.md"
        outside.write_text("outside")
        self.file.unlink()
        self.file.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.intake.capture("finding.md")
        self.file.unlink()
        with self.file.open("wb") as stream:
            stream.truncate(16777217)
        with self.assertRaisesRegex(ValueError, "bound"):
            self.intake.capture("finding.md")

    def test_concurrent_captures_are_deduplicated(self):
        errors = []
        def capture():
            try:
                self.intake.capture("finding.md")
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=capture) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(self.coord.snapshot()["research_revision_count"], 1)
        self.assertEqual(self.coord.snapshot()["last_sequence"], 1)

    def test_unknown_discoveries_are_saved_without_invented_links(self):
        self.file.unlink()
        (self.sources / "new-idea.md").write_text("A new investigation")
        self.intake.scan()
        state = self.coord.snapshot()
        self.assertEqual(len(state["research_discoveries"]), 1)
        self.assertFalse(state["research_discoveries"][0]["links"])
        self.assertFalse(state["jobs"])

    def test_invalid_target_and_false_certification_are_refused(self):
        self.config["auto_register_evidence"] = False
        self.intake.scan()
        proposal = self.coord.snapshot()["proposals"][0]
        value = json.loads(proposal["replacement"])
        value["formal_status"] = "formally_certified"
        with self.assertRaisesRegex(ValueError, "cannot assert"):
            self.coord.propose("research_evidence", proposal["target"], value)
        value["formal_status"] = "not_certified"
        value["capture"]["links"][0]["record_id"] = "DOES-NOT-EXIST"
        value["capture"]["revision_id"] = revision_id(value["capture"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.coord.propose("research_evidence", proposal["target"], value)

    def test_latest_revision_supersedes_active_attachment_preserving_history(self):
        self.intake.scan()
        first = self.coord.snapshot()["research_discoveries"][0]
        self.file.write_text("A revised investigation")
        self.intake.scan()
        state = self.coord.snapshot()
        active = state["research_evidence_attachments"]["D1-KR-FAR"]
        self.assertEqual(len(active), 1)
        self.assertNotEqual(first["revision_id"], active[0]["capture"]["revision_id"])
        self.assertEqual(state["research_revision_count"], 2)
        self.assertEqual(len([p for p in state["proposals"] if p["status"] == "approved"]), 2)

    def test_source_can_revert_to_previously_captured_bytes(self):
        first = self.intake.capture("finding.md")
        self.file.write_text("Revision B")
        self.intake.capture("finding.md")
        self.file.write_bytes(b"First discovery\r\n")
        self.intake.capture("finding.md")
        self.assertEqual(self.coord.snapshot()["research_discoveries"][0]["revision_id"], first["revision_id"])
        self.assertEqual(self.coord.snapshot()["research_revision_count"], 2)
        self.assertEqual(len(self.coord.journal.since(0)), 3)

    def test_missing_watch_root_recovers_without_losing_saved_evidence(self):
        self.intake.scan()
        moved = self.root / "temporarily-moved"
        self.sources.rename(moved)
        self.intake.scan()
        self.assertTrue(self.coord.snapshot()["research_failures"])
        moved.rename(self.sources)
        self.intake.scan()
        self.assertFalse(self.coord.snapshot()["research_failures"])

    def test_repeated_content_reversions_register_current_evidence(self):
        self.intake.scan()
        for text in ("Version B", "First discovery\r\n", "Version B"):
            self.file.write_bytes(text.encode())
            self.intake.scan()
            state = self.coord.snapshot()
            active = state["research_evidence_attachments"]["D1-KR-FAR"][0]
            self.assertEqual(active["capture"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(len([p for p in state["proposals"] if p["status"] == "approved"]), 4)

    def test_snapshot_verifies_and_replays_without_original_watch_directory(self):
        from release import release
        self.intake.scan()
        self.assertEqual(self.coord.run_one_job()["result"], "diagnostic_completed")
        output = self.root / "portable.zip"
        release(output, self.coord.journal.workdir)
        self.sources.rename(self.root / "source-unavailable")
        with zipfile.ZipFile(output) as archive:
            archive.extractall(self.root / "extracted")
        extracted = self.root / "extracted/Research_Flow"
        completed = subprocess.run([sys.executable, "-B", "verify_flow.py", "--replay"],
                                   cwd=extracted, capture_output=True, text=True, timeout=45)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["worker_receipts_replayed"], 1)
        self.assertGreater(report["release_files_hashed"], 100)
        self.assertEqual(report["custody"]["evidence_revisions_checked"], 1)

    def test_periodic_snapshot_loop_publishes_a_consistent_snapshot(self):
        self.intake.capture("finding.md")
        class OneIteration:
            calls = 0
            def wait(self, interval):
                self.calls += 1
                return self.calls > 1
        directory = self.root / "snapshots"
        snapshot_loop(self.coord, OneIteration(), directory, 0)
        latest = json.loads((directory / "LATEST.json").read_text())
        self.assertTrue(Path(latest["zip"]).is_file())
        self.assertEqual(latest["research_sources"], 1)
        self.assertEqual(hashlib.sha256(Path(latest["zip"]).read_bytes()).hexdigest(), latest["sha256"])

    def test_object_store_symlink_cannot_escape_runtime(self):
        outside = self.root / "outside-store"
        outside.mkdir()
        research = self.coord.journal.workdir / "research"
        research.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes runtime"):
            self.intake.capture("finding.md")
        self.assertFalse(list(outside.iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
