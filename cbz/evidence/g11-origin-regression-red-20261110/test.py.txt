"""Adverse checks for G11 selected-assertion origin attribution.

These exercise the real selected MutationAttribution test and the real parent
grader, with only the copied subject harness changed for each isolated child.
"""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_mutation_selftest as g11m

SELECTED = "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion"
FRAGMENT = "self.assertFalse(g11m.qualifies_kill(wrong_line, expected, 17))"
NEEDLE = "def qualifies_kill(outcome, expected_test, expected_line):\n"
OLD = '        and outcome.get("traceback_line") == expected_line\n        and outcome.get("exception_type")'
NEW = '        and True\n        and outcome.get("exception_type")'
INJECTIONS = {
    "subject": "    if outcome.get('traceback_line') == 18:\n        raise AssertionError('SUBJECT_ASSERT')\n",
    "helper": "    if outcome.get('traceback_line') == 18:\n        def helper():\n            raise AssertionError('HELPER_ASSERT')\n        return helper()\n",
    "callback": "    if outcome.get('traceback_line') == 18:\n        def invoke(callback):\n            return callback()\n        def callback():\n            raise AssertionError('CALLBACK_ASSERT')\n        return invoke(callback)\n",
    "runtime": "    if outcome.get('traceback_line') == 18:\n        raise RuntimeError('RUNTIME_ERROR')\n",
}

class ActualG11Origin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = [ROOT / g11m.TOOL, ROOT / g11m.TESTS, ROOT / g11m.HARNESS]
        cls.before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths]
        cls.tool, cls.tests, cls.harness = [p.read_text(encoding="utf-8") for p in cls.paths]
        if cls.harness.count(NEEDLE) != 1 or cls.harness.count(OLD) != 1:
            raise AssertionError("mutation target is missing or ambiguous")
        cls.line = g11m._assertion_line(cls.tests, SELECTED, FRAGMENT)

    @classmethod
    def tearDownClass(cls):
        after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths]
        if after != cls.before:
            raise AssertionError("original G11 inputs changed during adverse tests")

    def run_variant(self, variant):
        with tempfile.TemporaryDirectory(prefix="g11-origin-test-") as tmp:
            g11m._sandbox(self.tool, self.tests, variant, tmp)
            return g11m._run(tmp, SELECTED)

    def test_selected_body_and_runtime_controls(self):
        baseline = self.run_variant(self.harness)
        self.assertEqual((baseline["status"], baseline["exit"], baseline["ran"]), ("OK", 0, 1))
        self.assertFalse(g11m.qualifies_kill(baseline, SELECTED, self.line))
        mutant = self.run_variant(self.harness.replace(OLD, NEW, 1))
        self.assertTrue(g11m.qualifies_kill(mutant, SELECTED, self.line))
        runtime = self.run_variant(self.harness.replace(NEEDLE, NEEDLE + INJECTIONS["runtime"], 1))
        self.assertEqual((runtime["errors"], runtime["failures"]), (1, 0))
        self.assertFalse(g11m.qualifies_kill(runtime, SELECTED, self.line))

    def test_subject_helper_and_callback_assertions_are_not_kills(self):
        for name in ("subject", "helper", "callback"):
            with self.subTest(origin=name):
                variant = self.harness.replace(NEEDLE, NEEDLE + INJECTIONS[name], 1)
                outcome = self.run_variant(variant)
                self.assertEqual((outcome["errors"], outcome["failures"], outcome["ran"]), (0, 1, 1))
                self.assertFalse(g11m.qualifies_kill(outcome, SELECTED, self.line), name)

    def test_formatted_traceback_without_live_origin_is_not_a_kill(self):
        output = (
            "Ran 1 test in 0.001s\n\n"
            "FAIL: test_mutation_kill_requires_exact_target_assertion "
            "(tests.test_g11_census.MutationAttribution)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_g11_census.py", line 123, in test_mutation_kill_requires_exact_target_assertion\n'
            "AssertionError: expected\n\nFAILED (failures=1)\n"
        )
        proc = subprocess.CompletedProcess(["python"], 1, output, "")
        parsed = g11m._parse_run(proc)
        self.assertFalse(g11m.qualifies_kill(parsed, SELECTED, 123))

if __name__ == "__main__":
    unittest.main()
