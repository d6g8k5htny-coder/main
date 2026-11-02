"""Mutation accounting cannot turn errors, skips, or no tests into kills."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MutationHarnessTests(unittest.TestCase):
    def test_only_clean_assertion_failure_is_a_kill(self):
        spec = importlib.util.spec_from_file_location('mutations', ROOT / 'mutations.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        clean = dict(tests_run=1, assertion_failures=1, errors=0, skipped=0,
                     expected_failures=0, unexpected_successes=0, successful=False)
        self.assertEqual(mod.classify(clean, 1), 'KILLED')
        self.assertEqual(mod.classify({**clean, 'assertion_failures': 0, 'successful': True}, 0), 'SURVIVED')
        for changes, code in (({'tests_run': 0}, 1), ({'errors': 1}, 1),
                              ({'skipped': 1}, 1), ({'expected_failures': 1}, 1),
                              ({'unexpected_successes': 1}, 1), ({}, 2), ({}, -9)):
            with self.subTest(changes=changes, code=code):
                self.assertEqual(mod.classify({**clean, **changes}, code), 'ERROR')


if __name__ == '__main__':
    unittest.main(verbosity=2)
