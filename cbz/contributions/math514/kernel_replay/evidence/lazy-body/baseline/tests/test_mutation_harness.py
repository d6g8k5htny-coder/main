"""Real runner controls for body-only mutation kills and complete selected populations."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('mutations_under_test', ROOT / 'mutations.py')
mutations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mutations)


class MutationHarnessTests(unittest.TestCase):
    def run_fixture(self, methods, selection='fixture.Case', module_tail=''):
        source = 'import unittest\nclass Case(unittest.TestCase):\n' + textwrap.indent(textwrap.dedent(methods), '    ') + '\n' + module_tail
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tests').mkdir()
            shutil.copy2(ROOT / 'run_tests.py', root / 'run_tests.py')
            (root / 'tests/fixture.py').write_text(source)
            output = root / 'result.json'
            p = subprocess.run([sys.executable, '-B', '-S', str(root / 'run_tests.py'),
                                '--output', str(output), '--test', selection],
                               capture_output=True, text=True, timeout=10)
            self.assertTrue(output.exists(), p.stderr)
            value = json.loads(output.read_text())
            return mutations.classify(value, p.returncode), value, p.returncode

    def test_setup_assertion_is_not_a_kill(self):
        outcome, value, _ = self.run_fixture('''
            def setUp(self): self.fail('setup only')
            def test_one(self): pass
        ''')
        self.assertEqual(outcome, 'ERROR', value)

    def test_body_assertion_kills_after_all_selected_bodies_finish(self):
        outcome, value, _ = self.run_fixture('''
            def test_one(self): self.fail('body assertion')
            def test_two(self): pass
        ''')
        self.assertEqual(outcome, 'KILLED', value)
        self.assertEqual(value['body_entered_ids'], ['fixture.Case.test_one', 'fixture.Case.test_two'])
        self.assertEqual(value['body_finished_ids'], ['fixture.Case.test_one', 'fixture.Case.test_two'])
        self.assertEqual(value['body_assertion_ids'], ['fixture.Case.test_one'])

    def test_dispatch_hook_assertion_without_selected_method_is_error(self):
        outcome, value, _ = self.run_fixture("def _callTestMethod(self, method): self.fail('dispatch hook only')\ndef test_one(self): raise RuntimeError('selected body was entered')")
        self.assertEqual(outcome, 'ERROR', value)

    def test_selected_method_invoked_from_lifecycle_is_error(self):
        for source in ("def setUp(self): self.test_one()\ndef test_one(self): self.fail('called from setup')",
                       "def setUp(self): self._callTestMethod(self.test_one)\ndef test_one(self): self.fail('dispatch called from setup')",
                       "def tearDown(self): self.test_one()\ndef test_one(self): pass",
                       "def test_one(self):\n    if not hasattr(self, 'scheduled'):\n        self.scheduled = True\n        self.addCleanup(self.test_one)"):
            outcome, value, _ = self.run_fixture(source)
            self.assertEqual(outcome, 'ERROR', value)

    def test_clean_identity_survives(self):
        outcome, value, _ = self.run_fixture('def test_one(self): pass')
        self.assertEqual(outcome, 'SURVIVED', value)

    def test_lifecycle_failures_never_kill_even_with_body_assertion(self):
        cases = [
            "def tearDown(self): self.fail('teardown')\ndef test_one(self): pass",
            "def test_one(self): self.addCleanup(self.fail, 'cleanup')",
            "def tearDown(self): self.fail('teardown')\ndef test_one(self): self.fail('body')",
            "def test_one(self):\n    self.addCleanup(self.fail, 'cleanup')\n    self.fail('body')",
            "@classmethod\ndef setUpClass(cls): raise AssertionError('class setup')\ndef test_one(self): pass",
            "@classmethod\ndef tearDownClass(cls): raise AssertionError('class teardown')\ndef test_one(self): self.fail('body')",
            "@classmethod\ndef setUpClass(cls): cls.addClassCleanup(lambda: (_ for _ in ()).throw(AssertionError('class cleanup')))\ndef test_one(self): self.fail('body')",
        ]
        for source in cases:
            with self.subTest(source=source):
                outcome, value, _ = self.run_fixture(source)
                self.assertEqual(outcome, 'ERROR', value)

    def test_errors_skips_and_expected_failures_are_errors(self):
        cases = [
            "def test_one(self): raise RuntimeError('body runtime')",
            "def setUp(self): raise RuntimeError('setup runtime')\ndef test_one(self): self.fail('body')",
            "@unittest.skip('skip')\ndef test_one(self): pass",
            "@unittest.expectedFailure\ndef test_one(self): self.fail('expected')",
            "@unittest.expectedFailure\ndef test_one(self): pass",
        ]
        for source in cases:
            with self.subTest(source=source):
                outcome, value, _ = self.run_fixture(source)
                self.assertEqual(outcome, 'ERROR', value)

    def test_body_scoped_subtest_assertions_kill_without_changing_parent_population(self):
        for source, failures in (("def test_one(self):\n    with self.subTest(i=1): self.fail('first')\n    with self.subTest(i=2): self.fail('second')", 2),
                                 ("def test_one(self):\n    with self.subTest(i=1): pass\n    self.fail('body')", 1)):
            outcome, value, _ = self.run_fixture(source)
            self.assertEqual(outcome, 'KILLED', value)
            self.assertEqual(value['selected_ids'], ['fixture.Case.test_one'])
            self.assertEqual(value['assertion_failures'], failures)
        outcome, value, _ = self.run_fixture("def test_one(self):\n    with self.subTest(i=1): pass")
        self.assertEqual(outcome, 'SURVIVED', value)

    def test_foreign_subtest_parent_cannot_claim_selected_body_assertion(self):
        outcome, value, _ = self.run_fixture("def test_one(self):\n    import sys\n    foreign = unittest.case._SubTest(unittest.FunctionTestCase(lambda: None), 'foreign', {})\n    try: raise AssertionError('foreign parent assertion')\n    except AssertionError: self._outcome.result.addSubTest(self, foreign, sys.exc_info())")
        self.assertEqual(outcome, 'ERROR', value)

    def test_lifecycle_subtests_and_subtest_errors_are_errors(self):
        for source in ("def setUp(self):\n    with self.subTest(i=1): self.fail('setup subtest')\ndef test_one(self): pass",
                       "def tearDown(self):\n    with self.subTest(i=1): self.fail('teardown subtest')\ndef test_one(self): self.fail('body')",
                       "def test_one(self):\n    with self.subTest(i=1): raise RuntimeError('subtest runtime')"):
            outcome, value, _ = self.run_fixture(source)
            self.assertEqual(outcome, 'ERROR', value)

    def test_empty_duplicate_synthetic_and_spoofed_population_is_error(self):
        cases = [
            ('def test_one(self): pass', 'fixture', 'def load_tests(loader, tests, pattern): return unittest.TestSuite()'),
            ('def test_one(self): self.fail("body")', 'fixture', 'def load_tests(loader, tests, pattern): return unittest.TestSuite([Case("test_one"), Case("test_one")])'),
            ('def test_one(self): pass', 'fixture.Missing', ''),
            ('def id(self): return "fixture.Case.test_forged"\ndef test_one(self): self.fail("body")', 'fixture.Case', ''),
        ]
        for source, selection, tail in cases:
            with self.subTest(selection=selection, tail=tail):
                outcome, value, _ = self.run_fixture(source, selection, tail)
                self.assertEqual(outcome, 'ERROR', value)

    def test_population_listing_does_not_execute_fixtures_or_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tests').mkdir()
            shutil.copy2(ROOT / 'run_tests.py', root / 'run_tests.py')
            (root / 'tests/fixture.py').write_text("import unittest\nclass Case(unittest.TestCase):\n @classmethod\n def setUpClass(cls): raise AssertionError('fixture must not run')\n def test_one(self): raise RuntimeError('body must not run')\n")
            output = root / 'population.json'
            p = subprocess.run([sys.executable, '-B', '-S', str(root / 'run_tests.py'),
                                '--output', str(output), '--test', 'fixture.Case', '--list'],
                               capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(output.read_text()),
                             {'selection': ['fixture.Case'], 'selected_ids': ['fixture.Case.test_one'],
                              'invalid_population': []})

    def test_summary_without_lifecycle_receipt_fails_closed(self):
        old = dict(tests_run=1, assertion_failures=1, errors=0, skipped=0,
                   expected_failures=0, unexpected_successes=0, successful=False)
        self.assertEqual(mutations.classify(old, 1), 'ERROR')
        self.assertEqual(mutations.classify({}, 1), 'ERROR')
        self.assertEqual(mutations.classify(None, 1), 'ERROR')

    def test_forged_or_missing_accounting_is_error(self):
        _, receipt, code = self.run_fixture('def test_one(self): self.fail("body")')
        for key in ('receipt_schema', 'selected_ids', 'started_ids', 'finished_ids',
                    'body_entered_ids', 'body_finished_ids', 'body_assertion_ids',
                    'lifecycle_failures', 'subtests', 'invalid_population', 'subtest_events', 'assertion_events'):
            with self.subTest(missing=key):
                value = copy.deepcopy(receipt)
                value.pop(key, None)
                self.assertEqual(mutations.classify(value, code), 'ERROR')
        for changes in ({'tests_run': True}, {'tests_run': 1.0}, {'assertion_failures': '1'},
                        {'successful': 0}, {'lifecycle_failures': -1}, {'subtests': True},
                        {'body_entered_ids': []}, {'body_finished_ids': []},
                        {'finished_ids': []}, {'body_assertion_ids': []},
                        {'body_assertion_ids': ['fixture.Case.test_forged']},
                        {'selected_ids': ['synthetic']},
                        {'selected_ids': ['unittest.loader._FailedTest.test_one']},
                        {'selected_ids': ['fixture.Case.test_one', 'fixture.Case.test_one']},
                        {'started_ids': ['fixture.Case.test_one', 'fixture.Case.test_one']},
                        {'errors': 1}, {'skipped': 1}, {'expected_failures': 1},
                        {'unexpected_successes': 1}, {'assertion_failures': 2},
                        {'subtests': 1}, {'assertion_events': []},
                        {'subtest_events': [{'parent_id': 'fixture.Case.test_one', 'phase': 'body', 'outcome': 'assertion'}]}):
            with self.subTest(changes=changes):
                self.assertEqual(mutations.classify({**receipt, **changes}, code), 'ERROR')
        self.assertEqual(mutations.classify(receipt, code, ['fixture.Other']), 'ERROR')
        self.assertEqual(mutations.classify({**receipt, 'selection': ['fixture.Other']}, code, ['fixture.Other']), 'ERROR')
        self.assertEqual(mutations.classify(receipt, code, ['fixture.Case'], ['fixture.Case.test_other']), 'ERROR')
        for exit_code in (2, -9, True):
            self.assertEqual(mutations.classify(receipt, exit_code), 'ERROR')


if __name__ == '__main__':
    unittest.main(verbosity=2)
