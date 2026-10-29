"""The mutation harness must distinguish assertion kills from broken execution."""
import subprocess
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_check as mc


class KillClassification(unittest.TestCase):
    def result(self, body, code=1):
        return subprocess.CompletedProcess([], code, stdout=body)

    def good(self):
        return ("FAIL: test_bound (__main__.Checks.test_bound)\n"
                "AssertionError: bound accepted\nRan 4 tests in 0.1s\nFAILED (failures=1)\n")

    def test_expected_complete_assertion_failure_counts(self):
        self.assertTrue(mc.intended_failure(self.result(self.good()), "test_bound", 4))

    def test_syntax_word_in_assertion_context_is_not_launch_error(self):
        output = self.good().replace("bound accepted", "case 'SyntaxError: invalid source' accepted")
        self.assertTrue(mc.intended_failure(self.result(output), "test_bound", 4))

    def test_wrong_test_error_skip_incomplete_syntax_and_success_refused(self):
        variants = [self.good().replace("test_bound", "test_other"),
                    self.good().replace("failures=1", "errors=1"),
                    self.good().replace("failures=1", "failures=1, errors=1"),
                    self.good().replace("failures=1", "failures=1, skipped=1"),
                    self.good().replace("Ran 4 tests", "Ran 3 tests"),
                    self.good() + "\nERROR: launch failure\n",
                    self.good() + "\nSyntaxError: invalid source\n"]
        for output in variants:
            with self.subTest(output=output):
                self.assertFalse(mc.intended_failure(self.result(output), "test_bound", 4))
        self.assertFalse(mc.intended_failure(self.result(self.good(), 0), "test_bound", 4))
        self.assertFalse(mc.intended_failure(self.result(self.good(), 2), "test_bound", 4))


if __name__ == "__main__":
    unittest.main()
