"""Adverse A8 controls; first hosted run must fail against the old predicates."""
import importlib.util
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_harness(filename):
    spec = importlib.util.spec_from_file_location("admission_control_" + filename[:-3], ROOT / "tests" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(method="test_expected", assertion="self.assertEqual(actual, expected)",
           cause="AssertionError: 0 != 1"):
    return (
        "======================================================================\n"
        f"FAIL: {method} (tests.fixture.C.{method})\n"
        "----------------------------------------------------------------------\n"
        "Traceback (most recent call last):\n"
        f'  File "/fixture/test_fixture.py", line 12, in {method}\n'
        f"    {assertion}\n"
        f"{cause}\n\n"
        "----------------------------------------------------------------------\n"
        "Ran 3 tests in 0.001s\n\n"
        "FAILED (failures=1)\n"
    )


class AdmissionAdverseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harnesses = [
            load_harness("test_cbz_layer_a_scan_mutants.py"),
            load_harness("test_cbz_a3_float_census.py"),
        ]

    def check_rejected(self, transcript, returncode=1):
        proc = subprocess.CompletedProcess([], returncode, stdout="", stderr=transcript)
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__, returncode=returncode):
                self.assertFalse(mod.killed(proc, {"stderr": report().replace("0.001", "<ELAPSED>"),
                                                   "tests": 3, "failures": 1, "skipped": 0}))

    def test_wrong_exit_cannot_kill(self):
        for code in [0, 2, 7, -9]:
            self.check_rejected(report(), code)

    def test_foreign_failure_cannot_be_hidden_beside_expected(self):
        extra = report("test_foreign").split("Ran 3 tests", 1)[0]
        transcript = report().replace("Ran 3 tests", extra + "Ran 3 tests")
        self.check_rejected(transcript.replace("failures=1", "failures=2"))

    def test_error_cannot_be_hidden_by_failure_only_summary(self):
        transcript = report().replace("Ran 3 tests",
            "ERROR: test_foreign (tests.fixture.C.test_foreign)\n"
            "Traceback (most recent call last):\nRuntimeError: unrelated\n\nRan 3 tests")
        self.check_rejected(transcript)

    def test_same_method_wrong_assertion_cause_cannot_kill(self):
        self.check_rejected(report(assertion='raise AssertionError("unrelated setup")',
                                   cause="AssertionError: unrelated setup"))

    def test_inconsistent_failure_count_cannot_kill(self):
        self.check_rejected(report().replace("failures=1", "failures=9"))

    def test_duplicate_summary_cannot_kill(self):
        self.check_rejected(report() + report())

    def test_malformed_population_cannot_kill(self):
        for count in ["0", "-1", "x", "3garbage"]:
            self.check_rejected(report().replace("Ran 3 tests", "Ran " + count + " tests"))


    def test_complete_intended_report_is_accepted(self):
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr=report())
        profile = {"stderr": report().replace("0.001", "<ELAPSED>"),
                   "tests": 3, "failures": 1, "skipped": 0}
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertTrue(mod.killed(proc, profile))

    def test_complete_multiple_failure_profile_is_accepted(self):
        profile_for = load_harness("cbz_mutation_admission.py").profile_for
        profile = profile_for("census_14")
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr=profile["stderr"])
        self.assertEqual(profile["failures"], 7)
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertTrue(mod.killed(proc, profile))

    def test_same_site_wrong_mismatch_is_rejected(self):
        profile_for = load_harness("cbz_mutation_admission.py").profile_for
        profile = profile_for("census_15")
        self.assertIn("AssertionError: 0 != 2", profile["stderr"])
        bad = profile["stderr"].replace("AssertionError: 0 != 2", "AssertionError: 1 != 2")
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr=bad)
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertFalse(mod.killed(proc, profile))

    def test_same_named_foreign_class_is_rejected(self):
        profile_for = load_harness("cbz_mutation_admission.py").profile_for
        profile = profile_for("census_15")
        bad = profile["stderr"].replace("InputCoverage.", "ForeignCoverage.")
        self.assertNotEqual(bad, profile["stderr"])
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr=bad)
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertFalse(mod.killed(proc, profile))

    def test_missing_profile_never_admits_a_result(self):
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr=report())
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertFalse(mod.killed(proc))
                self.assertFalse(mod.killed(proc, "test_expected"))

    def test_output_in_wrong_channel_is_rejected(self):
        profile = {"stderr": report().replace("0.001", "<ELAPSED>"),
                   "tests": 3, "failures": 1, "skipped": 0}
        proc = subprocess.CompletedProcess([], 1, stdout=report(), stderr="")
        for mod in self.harnesses:
            with self.subTest(harness=mod.__name__):
                self.assertFalse(mod.killed(proc, profile))


if __name__ == "__main__":
    unittest.main()
