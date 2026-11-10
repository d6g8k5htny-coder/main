"""Adverse checks for G11 selected-assertion origin attribution.

These exercise the real selected MutationAttribution test and the real parent
grader, with only the copied subject harness changed for each isolated child.
"""
import hashlib
import subprocess
import sys
import tempfile
import types
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

    def test_direct_subject_assertion_is_not_a_kill(self):
        variant = self.harness.replace(NEEDLE, NEEDLE + INJECTIONS["subject"], 1)
        outcome = self.run_variant(variant)
        self.assertEqual((outcome["errors"], outcome["failures"], outcome["ran"]), (0, 1, 1))
        self.assertFalse(g11m.qualifies_kill(outcome, SELECTED, self.line))

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

    def run_fixture(self, source):
        with tempfile.TemporaryDirectory(prefix="g11-origin-fixture-") as tmp:
            tests = Path(tmp) / "tests"
            tests.mkdir()
            (tests / "__init__.py").write_text("", encoding="utf-8")
            (tests / "test_g11_census.py").write_text(source, encoding="utf-8")
            target = "tests.test_g11_census.Probe.test_body"
            return g11m._run(tmp, target), target

    def test_direct_body_and_unittest_oracle_are_authorized(self):
        for body, fragment in (
            ("        raise AssertionError('SELECTED')\n", "AssertionError('SELECTED')"),
            ("        self.assertEqual(1, 2)\n", "self.assertEqual(1, 2)"),
        ):
            with self.subTest(body=body):
                source = "import unittest\nclass Probe(unittest.TestCase):\n    def test_body(self):\n" + body
                outcome, target = self.run_fixture(source)
                line = g11m._assertion_line(source, target, fragment)
                self.assertTrue(g11m.qualifies_kill(outcome, target, line), outcome)

    def test_lifecycle_and_spoofed_nested_code_are_not_authorized(self):
        fixtures = {
            "setup": ("import unittest\nclass Probe(unittest.TestCase):\n"
                      "    def setUp(self):\n        raise AssertionError('SETUP')\n"
                      "    def test_body(self):\n        pass\n"),
            "teardown": ("import unittest\nclass Probe(unittest.TestCase):\n"
                         "    def test_body(self):\n        pass\n"
                         "    def tearDown(self):\n        raise AssertionError('TEARDOWN')\n"),
            "cleanup_reenters_body": (
                "import unittest\nclass Probe(unittest.TestCase):\n"
                "    def setUp(self):\n        self.in_cleanup = False\n        self.addCleanup(self.cleanup)\n"
                "    def test_body(self):\n        if self.in_cleanup:\n            raise AssertionError('CLEANUP')\n"
                "    def cleanup(self):\n        self.in_cleanup = True\n        self.test_body()\n"),
            "nested_same_name": (
                "import unittest\nclass Probe(unittest.TestCase):\n"
                "    def test_body(self):\n"
                "        def nested():\n            raise AssertionError('NESTED')\n"
                "        nested.__code__ = nested.__code__.replace(co_name='test_body', co_filename=__file__)\n"
                "        nested()\n"),
        }
        for name, source in fixtures.items():
            with self.subTest(origin=name):
                outcome, target = self.run_fixture(source)
                self.assertEqual((outcome["failures"], outcome["errors"]), (1, 0), outcome)
                self.assertFalse(outcome.get("origin_authorized"), outcome)
                self.assertFalse(g11m.qualifies_kill(outcome, target, outcome.get("traceback_line")))

    def test_origin_report_rejects_inconsistent_and_non_oracle_frames(self):
        target = "tests.test_g11_census.Probe.test_body"
        frame = {"selected": True, "oracle": False, "name": "test_body",
                 "lineno": 7, "firstlineno": 4, "filename": "/tmp/test_g11_census.py"}
        subject = {"selected": False, "oracle": False, "name": "subject", "lineno": 9}
        event = {"test": "test_body", "kind": "failure", "authorized": True, "entered": 1, "phase": "body-exited",
                 "exception": "AssertionError", "frames": [frame, subject],
                 "selected": {"name": "test_body", "firstlineno": 4,
                              "filename": "/tmp/test_g11_census.py"}}
        report = {"schema": "reference-crawler-origin-v3", "tests_run": 1,
                  "failure_count": 1, "error_count": 0, "skip_count": 0,
                  "expected_failure_count": 0, "unexpected_success_count": 0,
                  "optimize": sys.flags.optimize, "ids": [target], "names": ["test_body"],
                  "started": [target], "stopped": [target], "events": [event],
                  "authorized_failures": ["test_body"], "unauthorized_failures": [], "errors": []}
        proc = types.SimpleNamespace(returncode=1)
        outcome = g11m._parse_origin_run(proc, report, target)
        self.assertFalse(g11m.qualifies_kill(outcome, target, 7))
        malformed = dict(report, started=[])
        self.assertFalse(g11m._parse_origin_run(proc, malformed, target)["parsed"])
        self.assertFalse(g11m._parse_origin_run(proc, None, target)["parsed"])

    def test_subject_cannot_precreate_a_forged_origin_report(self):
        source = '''import json, sys, unittest
class Probe(unittest.TestCase):
    def test_body(self):
        target = "tests.test_g11_census.Probe.test_body"
        selected = {"name": "test_body", "firstlineno": type(self).test_body.__code__.co_firstlineno, "filename": __file__}
        frame = dict(selected, selected=True, oracle=False, lineno=123)
        event = {"test": "test_body", "kind": "failure", "authorized": True, "entered": 1, "phase": "body-exited", "exception": "AssertionError", "frames": [frame], "selected": selected}
        report = {"schema": "reference-crawler-origin-v3", "tests_run": 1, "failure_count": 1, "error_count": 0, "skip_count": 0, "expected_failure_count": 0, "unexpected_success_count": 0, "optimize": sys.flags.optimize, "ids": [target], "names": ["test_body"], "started": [target], "stopped": [target], "events": [event], "authorized_failures": ["test_body"], "unauthorized_failures": [], "errors": []}
        with open(sys.argv[2], "w", encoding="utf-8") as output:
            json.dump(report, output)
'''
        outcome, target = self.run_fixture(source)
        self.assertFalse(g11m.qualifies_kill(outcome, target, 123), outcome)

    def test_precreated_report_without_runner_completion_is_rejected(self):
        source = '''import json, sys, unittest
class Probe(unittest.TestCase):
    def test_body(self):
        target = "tests.test_g11_census.Probe.test_body"
        selected = {"name": "test_body", "firstlineno": type(self).test_body.__code__.co_firstlineno, "filename": __file__}
        frame = dict(selected, selected=True, oracle=False, lineno=123)
        event = {"test": "test_body", "kind": "failure", "authorized": True, "entered": 1, "phase": "body-exited", "exception": "AssertionError", "frames": [frame], "selected": selected}
        report = {"schema": "reference-crawler-origin-v3", "tests_run": 1, "failure_count": 1, "error_count": 0, "skip_count": 0, "expected_failure_count": 0, "unexpected_success_count": 0, "optimize": sys.flags.optimize, "ids": [target], "names": ["test_body"], "started": [target], "stopped": [target], "events": [event], "authorized_failures": ["test_body"], "unauthorized_failures": [], "errors": []}
        with open(__import__("__main__")._g11_report_path, "w", encoding="utf-8") as output:
            json.dump(report, output)
'''
        outcome, target = self.run_fixture(source)
        self.assertFalse(g11m.qualifies_kill(outcome, target, 123), outcome)


    def test_live_failure_retains_actual_origin_frames(self):
        source = ("import unittest\nclass Probe(unittest.TestCase):\n"
                  "    def test_body(self):\n        self.assertEqual(1, 2)\n")
        outcome, target = self.run_fixture(source)
        self.assertIn("origin_provenance", outcome)
        event = outcome["origin_provenance"]
        self.assertEqual(event["entered"], 1)
        self.assertIn(event["phase"], ("body", "body-exited"))
        self.assertTrue(event["authorized"])
        frames = event["frames"]
        selected = [i for i, frame in enumerate(frames) if frame["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertTrue(all(frame["oracle"] for frame in frames[selected[0] + 1:]))
        line = g11m._assertion_line(source, target, "self.assertEqual(1, 2)")
        self.assertTrue(g11m.qualifies_kill(outcome, target, line))


    def test_false_origin_authorization_is_not_a_kill(self):
        target = "tests.test_g11_census.Probe.test_body"
        outcome = {"parsed": True, "status": "FAILED", "exit": 1, "ran": 1,
                   "failures": 1, "errors": 0, "skipped": 0, "test_id": target,
                   "traceback_method": "test_body", "traceback_line": 7,
                   "exception_type": "AssertionError", "origin_authorized": False,
                   "origin_provenance": {"authorized": True}}
        self.assertFalse(g11m.qualifies_kill(outcome, target, 7))

    def test_unapproved_origin_provenance_is_not_a_kill(self):
        target = "tests.test_g11_census.Probe.test_body"
        outcome = {"parsed": True, "status": "FAILED", "exit": 1, "ran": 1,
                   "failures": 1, "errors": 0, "skipped": 0, "test_id": target,
                   "traceback_method": "test_body", "traceback_line": 7,
                   "exception_type": "AssertionError", "origin_authorized": True,
                   "origin_provenance": {"authorized": False}}
        self.assertFalse(g11m.qualifies_kill(outcome, target, 7))

if __name__ == "__main__":
    unittest.main()
