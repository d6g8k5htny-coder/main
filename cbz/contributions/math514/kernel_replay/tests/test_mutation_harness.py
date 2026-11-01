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

    def test_checker_origin_assertion_is_not_a_kill(self):
        # Code under test compiled from another file: its AssertionError is a crash, not an oracle.
        tail = ("CHECKER = {}\n"
                "exec(compile('def check(value):\\n    raise AssertionError(\"checker runtime crash\")\\n',"
                " 'checker.py', 'exec'), CHECKER)\n")
        for methods in ("def test_one(self): CHECKER['check'](1)",
                        "def test_one(self):\n    with self.subTest(i=1): CHECKER['check'](1)"):
            with self.subTest(methods=methods):
                outcome, value, _ = self.run_fixture(methods, module_tail=tail)
                self.assertEqual(outcome, 'ERROR', value)
                self.assertEqual(value['body_assertion_ids'], [])
        outcome, value, _ = self.run_fixture(
            "def test_one(self):\n    try: CHECKER['check'](1)\n    except AssertionError: pass\n    self.fail('oracle')",
            module_tail=tail)
        self.assertEqual(outcome, 'KILLED', value)
        outcome, value, _ = self.run_fixture(
            "def test_one(self):\n    from unittest import mock\n    m = mock.Mock(); m(1); m.assert_called_once_with(2)")
        self.assertEqual(outcome, 'KILLED', value)

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


class LazyBodyHarnessTests(unittest.TestCase):
    """A call returning suspended work must never establish complete test execution."""

    def run_lazy_fixture(self, methods, definitions='', listing=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tests').mkdir()
            shutil.copy2(ROOT / 'run_tests.py', root / 'run_tests.py')
            events = root / 'events'
            source = ('import unittest, functools\nfrom pathlib import Path\n'
                      'def record(label):\n    with Path(' + repr(str(events)) + ').open("a") as out: out.write(label + "\\n")\n'
                      + textwrap.dedent(definitions) + '\nclass Case(unittest.TestCase):\n'
                      + textwrap.indent(textwrap.dedent(methods), '    '))
            (root / 'tests/fixture.py').write_text(source)
            output = root / 'result.json'
            command = [sys.executable, '-B', '-S', str(root / 'run_tests.py'),
                       '--output', str(output), '--test', 'fixture.Case']
            if listing:
                command.append('--list')
            p = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertTrue(output.is_file(), p.stderr)
            value = json.loads(output.read_text())
            observed = events.read_text().splitlines() if events.exists() else []
            return mutations.classify(value, p.returncode), value, observed, p.stderr

    def test_ordinary_synchronous_body_positives_remain_supported(self):
        survived, value, events, _ = self.run_lazy_fixture("def test_one(self): record('ordinary')")
        self.assertEqual(survived, 'SURVIVED', value)
        self.assertEqual(events, ['ordinary'])
        killed, value, events, _ = self.run_lazy_fixture("def test_one(self):\n    record('assertion')\n    self.fail('real body assertion')")
        self.assertEqual(killed, 'KILLED', value)
        self.assertEqual(events, ['assertion'])

    def test_declared_lazy_method_shapes_refuse_mixed_population_before_execution(self):
        for lazy in ("def test_a_lazy(self):\n    record('lazy advanced')\n    yield None",
                     "async def test_a_lazy(self):\n    record('lazy advanced')\n    yield None",
                     "async def test_a_lazy(self): record('lazy advanced')"):
            with self.subTest(shape=lazy):
                methods = lazy + "\ndef test_b_assertion(self):\n    record('ordinary assertion')\n    self.fail('body assertion')"
                outcome, value, events, _ = self.run_lazy_fixture(methods)
                self.assertEqual(outcome, 'ERROR', value)
                self.assertEqual(events, [], value)
                self.assertTrue(value['invalid_population'], value)
                _, population, events, _ = self.run_lazy_fixture(methods, listing=True)
                self.assertTrue(population['invalid_population'], population)
                self.assertEqual(events, [])

    def test_wrapped_lazy_method_shapes_refuse_before_execution(self):
        definitions = '''
            def wrap(method):
                @functools.wraps(method)
                def wrapper(*args, **kwargs):
                    record('decorator executed')
                    return method(*args, **kwargs)
                return wrapper
        '''
        for lazy in ("def test_a_lazy(self):\n    record('lazy advanced')\n    yield None",
                     "async def test_a_lazy(self):\n    record('lazy advanced')\n    yield None",
                     "async def test_a_lazy(self): record('lazy advanced')"):
            with self.subTest(shape=lazy):
                outcome, value, events, _ = self.run_lazy_fixture('@wrap\n' + lazy + "\ndef test_b_assertion(self): self.fail('body')", definitions)
                self.assertEqual(outcome, 'ERROR', value)
                self.assertEqual(events, [], value)

    def test_sync_lazy_returns_refuse_without_advancing_suspended_work(self):
        definitions = '''
            def generator():
                record('generator advanced')
                yield None
            async def coroutine(): record('coroutine advanced')
            async def async_generator():
                record('async generator advanced')
                yield None
            class Awaitable:
                def __await__(self):
                    record('custom awaitable advanced')
                    yield None
        '''
        for expression in ('generator()', 'coroutine()', 'async_generator()', 'Awaitable()'):
            with self.subTest(expression=expression):
                methods = "def test_a_return(self):\n    record('outer method')\n    return " + expression + "\ndef test_b_assertion(self):\n    record('ordinary assertion')\n    self.fail('body')"
                outcome, value, events, stderr = self.run_lazy_fixture(methods, definitions)
                self.assertEqual(outcome, 'ERROR', value)
                self.assertEqual(events, ['outer method', 'ordinary assertion'], value)
                self.assertTrue(value['invalid_population'], value)
                self.assertNotIn('was never awaited', stderr)

    def test_decorator_without_wrapped_metadata_cannot_hide_lazy_return(self):
        definitions = '''
            def hide(method):
                def wrapper(*args, **kwargs):
                    record('decorator executed')
                    return method(*args, **kwargs)
                return wrapper
        '''
        outcome, value, events, _ = self.run_lazy_fixture("@hide\ndef test_a_lazy(self):\n    record('lazy advanced')\n    yield None\ndef test_b_assertion(self):\n    record('ordinary assertion')\n    self.fail('body')", definitions)
        self.assertEqual(outcome, 'ERROR', value)
        self.assertEqual(events, ['decorator executed', 'ordinary assertion'], value)

    def test_all_non_none_returns_refuse_without_iterating_custom_objects(self):
        definitions = "class DeferredIterable:\n    def __iter__(self):\n        record('unknown iterable advanced')\n        yield None"
        for expression in ('False', '0', '[]', 'DeferredIterable()'):
            with self.subTest(expression=expression):
                outcome, value, events, _ = self.run_lazy_fixture("def test_a_return(self):\n    record('outer method')\n    return " + expression + "\ndef test_b_assertion(self):\n    record('ordinary assertion')\n    self.fail('body')", definitions)
                self.assertEqual(outcome, 'ERROR', value)
                self.assertEqual(events, ['outer method', 'ordinary assertion'], value)


import os
import shutil
import subprocess

class IdentityHarnessTests(unittest.TestCase):
    """Real ordinary-body controls for loader and in-process identity binding."""

    def run_identity_fixture(self, label, source, selectors=()):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'tests').mkdir()
            runner=root/'run_tests.py'
            shutil.copy2(ROOT/'run_tests.py',runner)
            events=root/'events'
            header=('import unittest\nfrom pathlib import Path\n'
                    +'def record(label):\n with Path('+repr(str(events))+').open("a") as out: out.write(label+"\\n")\n')
            fixture=root/'tests'/'test_identity.py'
            fixture.write_text(header+source)
            output=root/'receipt.json'
            command=[sys.executable,'-B','-S',str(runner),*[],'--output',str(output)]
            for selector in selectors:command.extend(['--test',selector])
            p=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertTrue(output.exists(),p.stderr)
            value=json.loads(output.read_text())
            outcome=mutations.classify(value,p.returncode)
            observed=events.read_text().splitlines() if events.exists() else []
            record={'label':label,'optimize':sys.flags.optimize,'fixture_source':header+source,
                    'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    'command':command,'exit_code':p.returncode,'outcome':outcome,
                    'events':observed,'receipt':value,'stdout':p.stdout,'stderr':p.stderr}
            destination=os.environ.get('CBZ_IDENTITY_CONTROL_EVIDENCE')
            if destination:
                out=Path(destination);out.mkdir(parents=True,exist_ok=True)
                (out/(label+'.json')).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
            return outcome,value,observed

    def test_module_alias_subclass_and_population_hooks_refused_before_execution(self):
        base=('class C(unittest.TestCase):\n'
              +' def test_first(self): record("first")\n'
              +' def test_second(self):\n  record("second")\n  self.fail("retained second body")\n')
        hooks=[('alias','def load_tests(loader, tests, pattern):\n record("hook")\n C.test_second=C.test_first\n return tests\n'),
               ('alias-assertion','def load_tests(loader, tests, pattern):\n record("hook")\n C.test_second=C.test_first\n return tests\n'),
               ('subclass','def load_tests(loader, tests, pattern):\n record("hook")\n def replacement(self): record("replacement")\n R=type("C",(C,),{"test_first":replacement,"test_second":replacement,"__module__":__name__,"__qualname__":"C"})\n globals()["C"]=R\n return loader.loadTestsFromTestCase(R)\n'),
               ('omission','def load_tests(loader, tests, pattern):\n record("hook")\n return unittest.TestSuite([C("test_first")])\n')]
        for label,hook in hooks:
            with self.subTest(shape=label):
                source=base+hook
                if label=='alias-assertion':source=source.replace('def test_first(self): record("first")','def test_first(self):\n  record("first")\n  self.fail("first body assertion")')
                outcome,value,events=self.run_identity_fixture('hook-'+label,source)
                self.assertEqual(outcome,'ERROR',value)
                self.assertEqual(events,[],value)
                self.assertTrue(value['invalid_population'],value)

        source=('class Hidden(unittest.TestCase):\n def test_hidden(self):\n  record("hidden")\n  self.fail("hidden body retained")\n'
                +'class Visible(unittest.TestCase):\n def test_visible(self): record("visible")\n'
                +'def __dir__():\n record("module-dir")\n return ["Visible"]\n')
        outcome,value,events=self.run_identity_fixture('module-dir-cannot-hide',source)
        self.assertEqual(outcome,'KILLED',value)
        self.assertEqual(events,['hidden','visible'],value)
        self.assertEqual(len(value['selected_ids']),2,value)

    def test_named_class_and_method_selection_preserve_exact_population(self):
        source='class C(unittest.TestCase):\n def test_first(self): record("first")\n def test_second(self): record("second")\n'
        for label,selectors,wanted in [('class',('test_identity.C',),['first','second']),
                                        ('method',('test_identity.C.test_second',),['second'])]:
            with self.subTest(selector=label):
                outcome,value,events=self.run_identity_fixture('named-'+label,source,selectors)
                self.assertEqual(outcome,'SURVIVED',value)
                self.assertEqual(events,wanted,value)
        outcome,value,events=self.run_identity_fixture('named-assertion',source.replace('record("first")','record("first"); self.fail("body")'),('test_identity.C',))
        self.assertEqual(outcome,'KILLED',value)
        self.assertEqual(events,['first','second'],value)

    def test_callable_factory_selector_is_refused_before_factory_or_body(self):
        source=('class C(unittest.TestCase):\n def test_one(self): record("body")\n'
                +'def factory():\n record("factory")\n return unittest.TestSuite([C("test_one")])\n')
        outcome,value,events=self.run_identity_fixture('callable-factory',source,('test_identity.factory',))
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,[],value)
        self.assertTrue(value['invalid_population'],value)

        proxy_source=('class C(unittest.TestCase):\n def test_one(self): record("body")\n'
                      +'class Proxy:\n def __getattr__(self,name):\n  record("proxy")\n  return C\nproxy=Proxy()\n')
        outcome,value,events=self.run_identity_fixture('selector-proxy',proxy_source,('test_identity.proxy.C.test_one',))
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,[],value)
        self.assertTrue(value['invalid_population'],value)
        constructor_fixtures=[
            ('init','class C(unittest.TestCase):\n def __init__(self,*a,**k):\n  record("constructor")\n  super().__init__(*a,**k)\n  type(self).test_second=type(self).test_first\n def test_first(self): record("first")\n def test_second(self):\n  record("second")\n  self.fail("retained second body")\n'),
            ('new','class C(unittest.TestCase):\n def __new__(cls,*a,**k):\n  record("constructor")\n  return super().__new__(cls)\n def test_one(self): record("body")\n'),
            ('metaclass','class Meta(type):\n def __call__(cls,*a,**k):\n  record("metaclass")\n  return super().__call__(*a,**k)\nclass C(unittest.TestCase,metaclass=Meta):\n def test_one(self): record("body")\n'),
            ('getattribute','class C(unittest.TestCase):\n def __getattribute__(self,name):\n  record("attribute")\n  return super().__getattribute__(name)\n def test_one(self): record("body")\n'),
            ('setattr','class C(unittest.TestCase):\n def __setattr__(self,name,value):\n  record("attribute")\n  return super().__setattr__(name,value)\n def test_one(self): record("body")\n'),
        ]
        constructor_fixtures.append(('descriptor','class Descriptor:\n def __init__(self,original): self.original=original\n def __get__(self,instance,owner):\n  record("descriptor")\n  owner.test_second=owner.test_first\n  return owner.test_first\nclass C(unittest.TestCase):\n def test_first(self): record("first")\n def test_second(self):\n  record("second")\n  self.fail("retained second body")\n test_second=Descriptor(test_second)\n'))
        for kind in ('staticmethod','classmethod'):
            constructor_fixtures.append(('descriptor-'+kind,'class Alias('+kind+'):\n def __get__(self,instance,owner):\n  record("descriptor-subclass")\n  owner.test_second=owner.test_first\n  return owner.test_first.__get__(instance,owner)\nclass C(unittest.TestCase):\n def test_first(self): record("first")\n def test_second(self):\n  record("second")\n  self.fail("retained second body")\n test_second=Alias(test_second)\n'))
        constructor_fixtures.append(('special-descriptor','class InitDescriptor:\n def __get__(self,instance,owner):\n  record("special-descriptor")\n  owner.__init__=unittest.TestCase.__init__\n  return owner.__init__.__get__(instance,owner)\nclass C(unittest.TestCase):\n __init__=InitDescriptor()\n def test_one(self): record("body")\n'))
        for label,source in constructor_fixtures:
            for selection in [(),('test_identity.C',),('test_identity.C.'+('test_first' if label=='init' else 'test_second' if label.startswith('descriptor') else 'test_one'),)]:
                with self.subTest(constructor=label,selector=selection):
                    outcome,value,events=self.run_identity_fixture('construction-'+label+'-'+str(len(selection))+'-'+('method' if selection and selection[0].count('.')==2 else 'class'),source,selection)
                    self.assertEqual(outcome,'ERROR',value)
                    self.assertEqual(events,[],value)
                    self.assertTrue(value['invalid_population'],value)

        helper_source=('class Alias(staticmethod):\n def __get__(self,instance,owner):\n  record("helper-descriptor")\n  return owner.test_one.__get__(instance,owner)\n'
                       +'class C(unittest.TestCase):\n def test_one(self): record("test")\n def helper(self):\n  record("retained-helper")\n  self.fail("original helper body")\n helper=Alias(helper)\n')
        outcome,value,events=self.run_identity_fixture('selector-helper-descriptor',helper_source,('test_identity.C.helper',))
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,[],value)
        self.assertTrue(value['invalid_population'],value)

    def test_class_fixture_function_and_code_substitutions_are_refused(self):
        base=('def replacement(self): record("replacement")\n'
              +'class C(unittest.TestCase):\n'
              +' @classmethod\n def setUpClass(cls):\n  record("class-setup")\n  CHANGE\n'
              +' def test_one(self):\n  record("original")\n  self.fail("original body retained")\n')
        for label,change in [('function','cls.test_one=replacement'),('code','cls.test_one.__code__=replacement.__code__'),
                             ('same-code-function','cls.test_one=__import__("types").FunctionType(cls.test_one.__code__,cls.test_one.__globals__)')]:
            with self.subTest(change=label):
                outcome,value,events=self.run_identity_fixture('class-fixture-'+label,base.replace('CHANGE',change))
                self.assertEqual(outcome,'ERROR',value)
                self.assertEqual(events,['class-setup'],value)
                self.assertTrue(value['invalid_population'],value)

        source=('class C(unittest.TestCase):\n'
                +' def setUp(self):\n  record("setup")\n  self.__class__=type("C",(type(self),),{"__module__":__name__,"__qualname__":"C"})\n'
                +' def test_one(self):\n  record("original")\n  self.fail("original body retained")\n')
        outcome,value,events=self.run_identity_fixture('instance-class-change',source)
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,['setup'],value)
        self.assertTrue(value['invalid_population'],value)

    def test_postload_substitution_refused_before_any_fixture(self):
        spec=importlib.util.spec_from_file_location('identity_runtime_probe',ROOT/'run_tests.py')
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        events=[]
        class Case(unittest.TestCase):
            @classmethod
            def setUpClass(cls):events.append('class-setup')
            def setUp(self):events.append('setup')
            def test_one(self):events.append('original')
        loader=runtime.IdentityLoader();suite=loader.loadTestsFromTestCase(Case)
        def replacement(self):events.append('replacement')
        Case.test_one=replacement
        value=runtime.run_suite(suite,['probe.Case'],loader.bindings,loader.invalid)
        self.assertTrue(value['invalid_population'],value)
        self.assertEqual(events,[],value)
        self.assertEqual(value['tests_run'],0,value)

    def test_custom_testcase_run_is_refused_before_execution(self):
        source=('class C(unittest.TestCase):\n'
                +' def run(self,result=None):\n  record("custom-run")\n  return super().run(result)\n'
                +' def test_one(self): record("body")\n')
        outcome,value,events=self.run_identity_fixture('custom-run',source,('test_identity.C',))
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,[],value)
        self.assertTrue(value['invalid_population'],value)

    def test_identity_change_during_body_is_not_a_clean_kill(self):
        source=('class C(unittest.TestCase):\n'
                +' def test_first(self):\n  record("first")\n  type(self).test_second=type(self).test_first\n'
                +' def test_second(self):\n  record("second")\n  self.fail("retained second body")\n')
        outcome,value,events=self.run_identity_fixture('during-body-change',source)
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,['first'],value)
        self.assertTrue(value['invalid_population'],value)

if __name__ == '__main__':
    unittest.main(verbosity=2)
