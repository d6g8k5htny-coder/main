"""The runner must execute source mutants, and cannot infer a kill from broken syntax."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
try:
    import mutation_test
except ImportError:
    mutation_test=None


class MutationHarnessRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mutation_test,'mutation runner implementation is missing')

    def test_unchanged_checker_survives_and_assertion_failure_kills_real_mutant(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            survived=mutation_test.run_one('identity','source_checks.py',lambda s:s,out)
            self.assertEqual(survived['outcome'],'SURVIVED',survived)
            killed=mutation_test.run_one('drop-placeholders','source_checks.py',
                lambda s:s.replace("placeholders = {'sorry', 'admit', 'sorryAx', 'admitGoal', 'mkSorry'}","placeholders = set()"),out)
            self.assertEqual(killed['outcome'],'KILLED',killed)
            self.assertGreater(killed['assertion_failures'],0)
            self.assertEqual(killed['errors'],0)
            self.assertTrue((out/'drop-placeholders.log').is_file())

    def test_invalid_checker_syntax_is_invalid_mutant_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=mutation_test.run_one('broken-syntax','source_checks.py',lambda s:'def :',Path(tmp))
            self.assertEqual(r['outcome'],'INVALID',r)

    def test_mutant_that_only_crashes_is_not_counted_as_behavioral_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=mutation_test.run_one('crash','source_checks.py',lambda s:'raise RuntimeError("crash")\n'+s,Path(tmp))
            self.assertEqual(r['outcome'],'ERROR',r)


class LifecycleHarnessRules(unittest.TestCase):
    """Catch false kills from lifecycle assertions and incomplete body execution."""

    def run_fixture(self, source, transform=None):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'tests').mkdir()
            for name in ('auditor.py','source_checks.py','canonical_lean.py'):
                (root/name).write_text('fixture_source = True\n')
            (root/'tests/test_auditor.py').write_text(source)
            previous=mutation_test.HERE
            try:
                mutation_test.HERE=root
                return mutation_test.run_one('fixture','source_checks.py',transform or (lambda s:s),root/'logs')
            finally:
                mutation_test.HERE=previous

    def test_body_assertion_kills_and_identity_survives(self):
        passed=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self): pass\n')
        self.assertEqual(passed['outcome'],'SURVIVED',passed)
        killed=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self): self.fail("body sentinel")\n def test_two(self): pass\n')
        self.assertEqual(killed['outcome'],'KILLED',killed)
        self.assertEqual(killed['tests']['body_entered_ids'],['test_auditor.C.test_one','test_auditor.C.test_two'])
        self.assertEqual(killed['tests']['body_assertion_ids'],['test_auditor.C.test_one'])

    def test_setup_teardown_cleanup_assertions_cannot_kill(self):
        fixtures=[
            ' def setUp(self): self.fail("setup")\n def test_one(self): pass\n',
            ' def setUp(self): self.test_one()\n def test_one(self): self.fail("setup called body")\n',
            ' def setUp(self): self._callTestMethod(self.test_one)\n def test_one(self): self.fail("setup called dispatch")\n',
            ' def _callTestMethod(self, method): self.fail("dispatch hook")\n def test_one(self): pass\n',
            ' def test_one(self): pass\n def tearDown(self): self.fail("teardown")\n',
            ' def test_one(self): self.addCleanup(self.fail,"cleanup")\n',
            ' @classmethod\n def setUpClass(cls): raise AssertionError("class setup")\n def test_one(self): pass\n',
            ' def test_one(self): pass\n @classmethod\n def tearDownClass(cls): raise AssertionError("class teardown")\n',
            ' def test_one(self): self.addClassCleanup(self.fail,"class cleanup")\n',
            ' def test_one(self): self.addCleanup(self.fail,"cleanup"); self.fail("body")\n',
        ]
        for body in fixtures:
            with self.subTest(body=body):
                value=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n'+body)
                self.assertEqual(value['outcome'],'ERROR',value)

    def test_errors_skips_expected_failures_and_subtests_are_ambiguous(self):
        fixtures=[
            ' def test_one(self): raise RuntimeError("body error")\n',
            ' @unittest.skip("skip")\n def test_one(self): pass\n',
            ' @unittest.expectedFailure\n def test_one(self): self.fail("expected")\n',
            ' @unittest.expectedFailure\n def test_one(self): pass\n',
            ' def setUp(self):\n  with self.subTest(x=1): self.fail("setup subtest")\n def test_one(self): pass\n',
            ' def test_one(self):\n  with self.subTest(x=1): raise RuntimeError("subtest error")\n',
        ]
        for body in fixtures:
            with self.subTest(body=body):
                value=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n'+body)
                self.assertEqual(value['outcome'],'ERROR',value)

    def test_body_scoped_subtests_have_exact_parent_assertion_events(self):
        passed=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self):\n  with self.subTest(x=1): pass\n')
        self.assertEqual(passed['outcome'],'SURVIVED',passed)
        killed=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self):\n  for x in (1,2):\n   with self.subTest(x=x): self.fail("subtest body")\n def test_two(self): pass\n')
        self.assertEqual(killed['outcome'],'KILLED',killed)
        self.assertEqual(killed['tests']['subtests'],2)
        self.assertEqual(killed['tests']['body_assertion_ids'],['test_auditor.C.test_one']*2)
        self.assertEqual(killed['tests']['body_finished_ids'],['test_auditor.C.test_one','test_auditor.C.test_two'])

    def test_empty_duplicate_synthetic_and_rewritten_method_ids_are_rejected(self):
        fixtures=[
            'import unittest\ndef load_tests(loader, tests, pattern): return unittest.TestSuite()\n',
            'import unittest\nclass C(unittest.TestCase):\n def test_one(self): self.fail("body")\ndef load_tests(loader, tests, pattern): return unittest.TestSuite([C("test_one"), C("test_one")])\n',
            'raise RuntimeError("import sentinel")\n',
            'import unittest\nclass C(unittest.TestCase):\n def test_one(self): self._outcome.result.addFailure(self,(AssertionError,AssertionError("forged callback"),None))\n',
            'import unittest\nclass C(unittest.TestCase):\n def id(self): return "forged.method"\n def test_one(self): self.fail("body")\n',
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                value=self.run_fixture(fixture)
                self.assertEqual(value['outcome'],'ERROR',value)

    def test_assertion_event_provenance_rejects_duplicate_direct_failures(self):
        import copy
        value=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self): self.fail("body")\n')['tests']
        forged=copy.deepcopy(value);forged['assertion_failures']=2
        forged['body_assertion_ids']=['test_auditor.C.test_one']*2
        self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')
        forged['assertion_events']=[{'id':'test_auditor.C.test_one','source':'direct'}]*2
        self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')

    def test_expected_population_is_bound_before_mutation(self):
        source='import unittest\nimport source_checks\nclass C(unittest.TestCase):\n def test_one(self): pass\n def test_two(self): self.fail("wrong population")\ndef load_tests(loader, tests, pattern):\n return unittest.TestSuite([C("test_one" if source_checks.fixture_source else "test_two")])\n'
        result=self.run_fixture(source,lambda s:s.replace('fixture_source = True','fixture_source = False'))
        self.assertEqual(result['outcome'],'ERROR',result)
        self.assertEqual(result['expected_ids'],['test_auditor.C.test_one'])
        self.assertEqual(result['tests']['selected_ids'],['test_auditor.C.test_two'])

    def test_missing_forged_or_noninteger_receipts_fail_closed(self):
        import copy
        value=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self): self.fail("body")\n')
        trusted=value.get('tests',{})
        self.assertEqual(mutation_test.classify_test_result(trusted,1),'KILLED')
        self.assertEqual(mutation_test.classify_test_result(trusted,1,expected_ids=['test_auditor.C.test_other']),'ERROR')
        duplicate=copy.deepcopy(trusted)
        for key in ('selected_ids','started_ids','finished_ids','body_entered_ids','body_finished_ids','body_assertion_ids'):
            duplicate[key]*=2
        duplicate['tests_run']=2;duplicate['assertion_failures']=2
        duplicate['assertion_events']=[{'id':'test_auditor.C.test_one','source':'subtest'}]*2
        duplicate['subtests']=2
        duplicate['subtest_events']=[{'id':'test_auditor.C.test_one','phase':'body','outcome':'assertion'}]*2
        self.assertEqual(mutation_test.classify_test_result(duplicate,1),'ERROR')
        for field in ('receipt_schema','selected_ids','started_ids','finished_ids','body_entered_ids','body_finished_ids','body_assertion_ids','invalid_population','subtests','subtest_events','assertion_events','lifecycle_failures'):
            with self.subTest(missing=field):
                forged=copy.deepcopy(trusted);forged.pop(field,None)
                self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')
        for field in ('tests_run','assertion_failures','errors','skipped','expected_failures','unexpected_successes','lifecycle_failures','subtests'):
            with self.subTest(noninteger=field):
                forged=copy.deepcopy(trusted);forged[field]=True
                self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')
        for field, replacement in [('started_ids',[]),('body_finished_ids',[]),('body_entered_ids',['test_auditor.C.test_one']*2),('body_assertion_ids',['forged.method']),('selected_ids',['test_auditor.C.test_one']*2),('invalid_population',['forged']),('lifecycle_failures',1),('subtests',1),('successful',1)]:
            with self.subTest(forged=field):
                forged=copy.deepcopy(trusted);forged[field]=replacement
                self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')
        subtest=self.run_fixture('import unittest\nclass C(unittest.TestCase):\n def test_one(self):\n  with self.subTest(x=1): self.fail("body")\n')['tests']
        for replacement in ({'id':'test_auditor.C.test_one','phase':'cleanup','outcome':'assertion'},
                            {'id':'forged.method','phase':'body','outcome':'assertion'},
                            {'id':'test_auditor.C.test_one','phase':'body','outcome':'error'}):
            with self.subTest(subtest_event=replacement):
                forged=copy.deepcopy(subtest);forged['subtest_events']=[replacement]
                self.assertEqual(mutation_test.classify_test_result(forged,1),'ERROR')
        self.assertEqual(mutation_test.classify_test_result({'tests_run':1,'assertion_failures':1,'errors':0},1),'ERROR')


if __name__=='__main__':
    unittest.main()
