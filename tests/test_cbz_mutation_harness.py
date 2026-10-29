"""Real-process controls for mutation-runner health and kill accounting."""

import importlib.util
from pathlib import Path
import tempfile
import textwrap
import unittest


RUNNER = Path(__file__).with_name("run_scanner_mutations.py")
spec = importlib.util.spec_from_file_location("cbz_mutation_harness", RUNNER)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class MutationHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cbz-harness-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project"
        (self.root / "tests").mkdir(parents=True)
        self.evidence = self.base / "evidence"
        self.target = self.base / "target.py"
        self.write_suite()

    def write_suite(self, body=None):
        if body is None:
            body = """
                report = json.loads(subprocess.check_output(
                    [sys.executable, os.environ['CBZ_AUDITOR_SCRIPT'],
                     '--manifest', 'unused'], text=True))
                self.assertEqual(report['aggregate_status'], 'NOT_ESTABLISHED')
            """
        code = (
            "import json, os, subprocess, sys, time, unittest\n"
            "class AuditorContract(unittest.TestCase):\n"
            "    def test_gate_remains_blocked(self):\n"
            + textwrap.indent(textwrap.dedent(body).strip() + "\n", "        ")
        )
        (self.root / "tests/test_cbz_source_audit.py").write_text(code)

    def evaluate(self, code, *, baseline_count=1, **options):
        self.target.write_text(textwrap.dedent(code))
        evaluator = getattr(harness, "evaluate_target", None)
        self.assertTrue(callable(evaluator), "missing operational mutant evaluation")
        return evaluator(self.target, baseline_count, "control.log", self.evidence,
                         suite_root=self.root, **options)

    def assert_harness_error(self, result, stage):
        self.assertEqual(result["result"], "HARNESS_ERROR", result)
        self.assertEqual(result["harness_error"]["stage"], stage, result)
        self.assertFalse(result["killed"], result)

    def test_startup_exception_cannot_be_an_assertion_kill(self):
        # Removing launch validation would let the CLI helper count this traceback.
        result = self.evaluate("raise RuntimeError('startup failure')\n")
        self.assert_harness_error(result, "smoke")
        self.assertEqual(result["tests_run"], 0)

    def test_non_json_output_cannot_be_an_assertion_kill(self):
        result = self.evaluate("print('not an audit JSON report')\n")
        self.assert_harness_error(result, "smoke")

    def test_nonobject_json_cannot_be_an_assertion_kill(self):
        for output in ("[]", "null", "1", '"report"'):
            with self.subTest(output=output):
                result = self.evaluate("print(" + repr(output) + ")\n")
                self.assert_harness_error(result, "smoke")

    def test_nonfinite_json_constants_do_not_pass_launch_health(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                output = '{"value":' + constant + '}'
                result = self.evaluate("print(" + repr(output) + ")\n")
                self.assert_harness_error(result, "smoke")

    def test_stderr_failure_cannot_be_hidden_by_json_stdout(self):
        result = self.evaluate("""
            import sys
            print('{"aggregate_status":"ESTABLISHED"}')
            print('Traceback: simulated launch failure', file=sys.stderr)
        """)
        self.assert_harness_error(result, "smoke")

    def test_smoke_timeout_is_not_a_kill(self):
        result = self.evaluate("import time\ntime.sleep(60)\n", smoke_timeout=0.2)
        self.assert_harness_error(result, "smoke")
        self.assertEqual(result["smoke"]["status"], "TIMEOUT")

    def test_compile_failure_is_recorded_without_running_the_suite(self):
        result = self.evaluate("def broken(:\n    pass\n")
        self.assert_harness_error(result, "compile")
        self.assertEqual(result["tests_run"], 0)

    def test_valid_semantic_failure_is_killed_even_with_exit_zero(self):
        result = self.evaluate('print(\'{"aggregate_status":"ESTABLISHED"}\')\n')
        self.assertEqual(result["result"], "KILLED", result)
        self.assertTrue(result["killed"], result)
        self.assertEqual(result["assertion_failures"], 1)
        self.assertEqual(result["test_errors"], 0)
        self.assertEqual(result["tests_run"], 1)
        self.assertTrue((self.evidence / "control.log").is_file())

    def test_unrelated_repository_tests_are_not_discovered(self):
        marker = self.base / "unrelated-ran"
        decoy = "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('ran')\nraise RuntimeError('unrelated suite was discovered')\n"
        (self.root / "tests/test_unrelated.py").write_text(decoy)
        result = self.evaluate('print(\'{"aggregate_status":"ESTABLISHED"}\')\n')
        self.assertEqual(result["result"], "KILLED", result)
        self.assertFalse(marker.exists(), "mutation run escaped its registered suite")
        self.assertEqual(result["tests_run"], 1)

    def test_incomplete_suite_is_not_a_kill(self):
        result = self.evaluate('print(\'{"aggregate_status":"ESTABLISHED"}\')\n',
                               baseline_count=2)
        self.assertEqual(result["result"], "NOT_KILLED", result)
        self.assertFalse(result["killed"], result)

    def test_suite_timeout_is_not_a_kill(self):
        self.write_suite("time.sleep(60)")
        result = self.evaluate('print(\'{"aggregate_status":"NOT_ESTABLISHED"}\')\n',
                               suite_timeout=0.2)
        self.assert_harness_error(result, "suite")

    def test_suite_errors_cannot_substitute_for_complete_assertion_evidence(self):
        self.write_suite("""
            with self.subTest(case='assertion'):
                self.fail('behavioral assertion')
            with self.subTest(case='execution'):
                raise RuntimeError('test execution error')
        """)
        result = self.evaluate('print(\'{"aggregate_status":"ESTABLISHED"}\')\n')
        self.assertFalse(result["killed"], result)
        self.assertEqual(result["result"], "NOT_KILLED", result)
        self.assertEqual(result["assertion_failures"], 1)
        self.assertEqual(result["test_errors"], 1)

    def test_skipped_suite_cases_cannot_establish_a_kill(self):
        self.write_suite("""
            with self.subTest(case='assertion'):
                self.fail('behavioral assertion')
            self.skipTest('unfinished evidence')
        """)
        result = self.evaluate('print(\'{"aggregate_status":"ESTABLISHED"}\')\n')
        self.assertFalse(result["killed"], result)
        self.assertEqual(result["assertion_failures"], 1)
        self.assertEqual(result["tests_skipped"], 1)

    def test_clean_negative_control_is_not_a_kill(self):
        result = self.evaluate('print(\'{"aggregate_status":"NOT_ESTABLISHED"}\')\n')
        self.assertEqual(result["result"], "NOT_KILLED", result)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["tests_run"], 1)


if __name__ == "__main__":
    unittest.main()
