"""Real unittest lifecycle controls for finite mutation-credit classification."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL=Path(__file__).resolve().parents[1]/'tools/cbz_mutation.py'

def module():
    if not TOOL.is_file():
        raise AssertionError('lifecycle-aware mutation runner is missing')
    spec=importlib.util.spec_from_file_location('cbz_mutation_under_test',TOOL)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

class MutationLifecycleTests(unittest.TestCase):
    def run_case(self, phase=None, population=None):
        events=[]
        class Fixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                if phase=='class_setup': raise AssertionError('class setup')
            @classmethod
            def tearDownClass(cls):
                if phase=='class_teardown': raise AssertionError('class teardown')
            def setUp(self):
                if phase=='setup': self.fail('setup')
                if phase=='cleanup': self.addCleanup(self.fail,'cleanup')
                if phase=='skip': self.skipTest('skip')
            def tearDown(self):
                if phase=='teardown': self.fail('teardown')
            def test_guard(self):
                events.append('body')
                if phase in ('body','body_and_teardown'): self.fail('guard mismatch')
                if phase=='error': raise ValueError('body error')
                if phase=='subtest':
                    with self.subTest(value=1): self.fail('subtest assertion')
                if phase=='passing_subtest':
                    with self.subTest(value=1): self.assertTrue(True)
        if phase=='body_and_teardown':
            Fixture.tearDown=lambda self:self.fail('teardown after body failure')
        if phase=='dispatcher_impostor':
            Fixture._callTestMethod=lambda self,method:self.fail('no actual named body')
        if phase=='expected_failure': Fixture.test_guard=unittest.expectedFailure(Fixture.test_guard)
        if phase=='unexpected_success': Fixture.test_guard=unittest.expectedFailure(Fixture.test_guard)
        if phase=='expected_failure':
            @unittest.expectedFailure
            def expected(self): self.fail('expected')
            expected.__name__='test_guard';Fixture.test_guard=expected
        case=Fixture('test_guard');expected=population if population is not None else [case.id()]
        return module().run_suite(unittest.TestSuite([case]),io.StringIO(),expected),events

    def test_positive_body_and_complete_population(self):
        report,events=self.run_case();self.assertTrue(report['baseline_success']);self.assertFalse(report['body_kill']);self.assertEqual(events,['body'])
    def test_actual_direct_guard_body_assertion_is_a_kill(self):
        report,events=self.run_case('body');self.assertTrue(report['body_kill']);self.assertEqual(events,['body']);self.assertEqual(report['body_failures'],report['population'])
    def test_setup_assertion_never_counts_and_body_not_entered(self):
        report,events=self.run_case('setup');self.assertFalse(report['body_kill']);self.assertEqual(events,[])
    def test_teardown_assertion_never_counts(self):
        self.assertFalse(self.run_case('teardown')[0]['body_kill'])
    def test_cleanup_assertion_never_counts(self):
        self.assertFalse(self.run_case('cleanup')[0]['body_kill'])
    def test_class_setup_assertion_never_counts(self):
        self.assertFalse(self.run_case('class_setup')[0]['body_kill'])
    def test_class_teardown_assertion_never_counts(self):
        self.assertFalse(self.run_case('class_teardown')[0]['body_kill'])
    def test_body_failure_with_bad_teardown_is_not_credited(self):
        self.assertFalse(self.run_case('body_and_teardown')[0]['body_kill'])
    def test_skip_is_not_positive_or_kill(self):
        report,_=self.run_case('skip');self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
    def test_body_error_is_not_assertion_kill(self):
        self.assertFalse(self.run_case('error')[0]['body_kill'])
    def test_subtest_failure_is_not_direct_body_kill(self):
        self.assertFalse(self.run_case('subtest')[0]['body_kill'])
    def test_passing_subtest_remains_positive(self):
        self.assertTrue(self.run_case('passing_subtest')[0]['baseline_success'])
    def test_expected_failure_is_not_mutation_credit(self):
        report,_=self.run_case('expected_failure');self.assertFalse(report['body_kill']);self.assertFalse(report['baseline_success'])
    def test_unexpected_success_is_not_positive(self):
        self.assertFalse(self.run_case('unexpected_success')[0]['baseline_success'])
    def test_dispatcher_assertion_cannot_impersonate_actual_named_body(self):
        report,events=self.run_case('dispatcher_impostor');self.assertFalse(report['body_kill']);self.assertEqual(events,[])
    def test_expected_identity_mismatch_is_refused(self):
        report,_=self.run_case('body',['different.test_identity']);self.assertFalse(report['body_kill']);self.assertFalse(report['population_valid'])
    def test_duplicate_expected_population_is_refused(self):
        report,_=self.run_case('body',['same','same']);self.assertFalse(report['body_kill'])
    def test_case_instance_methods_restored_after_run(self):
        class Fixture(unittest.TestCase):
            def test_body(self): self.assertTrue(True)
        case=Fixture('test_body');before=dict(case.__dict__)
        report=module().run_suite(unittest.TestSuite([case]),io.StringIO())
        self.assertTrue(report['baseline_success']);self.assertEqual(case.__dict__,before)

    def test_actual_child_optimization_modes_and_population_binding(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'fixture.py';p.write_text('import unittest\nclass Fixture(unittest.TestCase):\n def test_body(self): self.assertTrue(True)\n')
            expected=Path(temp)/'expected.json';expected.write_text(json.dumps(['cbz_test_suite.Fixture.test_body']))
            for mode,flags in ((0,[]),(1,['-O']),(2,['-OO'])):
                child=subprocess.run([sys.executable,'-B',*flags,'-S',str(TOOL),'--tests',str(p),'--expected',str(expected)],capture_output=True,text=True,check=True)
                report=json.loads(child.stdout);self.assertTrue(report['baseline_success'])
                self.assertEqual(report['optimization'],mode);self.assertEqual(report['optimization_args'],flags)
            expected.write_text(json.dumps(['different.identity']))
            child=subprocess.run([sys.executable,'-B','-S',str(TOOL),'--tests',str(p),'--expected',str(expected)],capture_output=True,text=True,check=True)
            self.assertFalse(json.loads(child.stdout)['baseline_success'])

    def test_async_named_body_is_explicitly_unsupported(self):
        events=[]
        class Fixture(unittest.IsolatedAsyncioTestCase):
            async def test_body(self): events.append('executed');self.fail('actual async assertion')
        report=module().run_suite(unittest.TestSuite([Fixture('test_body')]),io.StringIO())
        self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
        self.assertIn('unsupported_named_body',report['invalid'])
        self.assertEqual(events,[])

    def test_generator_named_body_is_explicitly_unsupported(self):
        class Fixture(unittest.TestCase):
            def test_body(self): yield self.fail('generator assertion')
        report=module().run_suite(unittest.TestSuite([Fixture('test_body')]),io.StringIO())
        self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
        self.assertIn('unsupported_named_body',report['invalid'])

    def test_async_generator_named_body_is_explicitly_unsupported(self):
        class Fixture(unittest.TestCase):
            async def test_body(self): yield self.fail('async generator assertion')
        report=module().run_suite(unittest.TestSuite([Fixture('test_body')]),io.StringIO())
        self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
        self.assertIn('unsupported_named_body',report['invalid'])

    def test_sync_body_returning_coroutine_is_refused_and_closed(self):
        returned=[]
        async def hidden(): raise AssertionError('hidden async body')
        class Fixture(unittest.TestCase):
            def test_body(self):
                obj=hidden();returned.append(obj);return obj
        report=module().run_suite(unittest.TestSuite([Fixture('test_body')]),io.StringIO())
        self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
        self.assertIn('non_none_body_return',report['invalid'])
        self.assertIsNone(returned[0].cr_frame)

    def test_sync_body_returning_generator_is_refused(self):
        def hidden(): yield 1
        class Fixture(unittest.TestCase):
            def test_body(self): return hidden()
        report=module().run_suite(unittest.TestSuite([Fixture('test_body')]),io.StringIO())
        self.assertFalse(report['baseline_success']);self.assertFalse(report['body_kill'])
        self.assertIn('non_none_body_return',report['invalid'])

    def test_optimization_flag_is_preserved_exactly(self):
        obj=module();expected=['-O'] if sys.flags.optimize==1 else ['-OO'] if sys.flags.optimize==2 else []
        self.assertEqual(obj.optimization_flags(),expected)

if __name__=='__main__': unittest.main()
