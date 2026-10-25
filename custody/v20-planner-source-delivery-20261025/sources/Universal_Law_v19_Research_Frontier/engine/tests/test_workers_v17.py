"""Actual subprocess, replay and custody regression controls for v17 workers."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from model import load
from workers import ROOT, WorkerError, execute_job, make_job, verify_receipt


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.data = load()
        self.temp = tempfile.TemporaryDirectory(prefix="ul-worker-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def job(self, worker="local-diagnostic", root=ROOT):
        return make_job(self.data, "D1-A", {"event_id": "worker-test:1"}, worker=worker, root=root)

    def test_actual_worker_pass_and_replay_preserve_uncertified_obligations(self):
        job = self.job()
        receipt = execute_job(job, self.data, self.base / "attempt")
        self.assertEqual(receipt["result"], "diagnostic_completed", receipt.get("reason"))
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["diagnostic"]["proof_state"], "needs_mathematical_reverification")
        self.assertEqual(receipt["theorem_certificates_awarded"], 0)
        self.assertTrue(receipt["diagnostic"]["unresolved_corrections"])
        self.assertTrue(verify_receipt(receipt, job=job, data=self.data, replay=True)["replayed"])

    def test_tampered_artifact_is_rejected(self):
        job = self.job()
        attempt = self.base / "attempt"
        receipt = execute_job(job, self.data, attempt)
        (attempt / "diagnostic.json").write_text('{"result":"diagnostic_completed"}')
        with self.assertRaisesRegex(WorkerError, "artifact hash"):
            verify_receipt(receipt, job=job)

    def test_worker_implementation_hash_mismatch_fails(self):
        job = self.job()
        job["worker"]["implementation_sha256"]["worker_process.py"] = "0" * 64
        receipt = execute_job(job, self.data, self.base / "attempt")
        self.assertEqual(receipt["result"], "failed")
        self.assertIn("implementation", receipt["reason"])
        self.assertEqual(receipt["theorem_certificates_awarded"], 0)

    def test_unconfigured_formal_adapter_is_blocked(self):
        job = self.job(worker="lean")
        receipt = execute_job(job, self.data, self.base / "attempt")
        self.assertEqual(receipt["result"], "blocked")
        self.assertIsNone(receipt["exit_code"])
        self.assertEqual(receipt["formal_certification"], "not_certified")
        self.assertTrue(verify_receipt(receipt, job=job)["valid"])

    def test_relocated_receipt_replays_after_original_directory_is_deleted(self):
        job = self.job()
        original = self.base / "original"
        receipt = execute_job(job, self.data, original)
        relocated = self.base / "fresh-extraction"
        shutil.copytree(original, relocated)
        shutil.rmtree(original)
        # The receipt retains its original absolute execution directory as history.
        self.assertEqual(json.loads((relocated / "receipt.json").read_text())["artifact_directory"], str(original))
        result = verify_receipt(relocated / "receipt.json", job=job, data=self.data, replay=True)
        self.assertTrue(result["valid"])
        self.assertTrue(result["replayed"])

    def test_unrelated_approval_before_execution_does_not_fail_snapshot_check(self):
        job = self.job()
        unrelated = copy.deepcopy(self.data)
        next(r for r in unrelated["records"] if r["id"] == "P15-F")["statement"] += " [unrelated working proposal]"
        receipt = execute_job(job, unrelated, self.base / "attempt")
        self.assertEqual(receipt["result"], "diagnostic_completed", receipt.get("reason"))
        self.assertNotEqual(receipt["execution_dataset_sha256"], job["inputs"]["effective_dataset_sha256"])
        self.assertTrue(verify_receipt(receipt, job=job, data=unrelated, replay=True)["valid"])


if __name__ == "__main__":
    unittest.main()
