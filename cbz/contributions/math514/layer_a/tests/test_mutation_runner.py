"""The runner must execute source mutants, and cannot infer a kill from broken syntax."""
import importlib.util
import json
import os
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
        # Canonical declarations are frozen before the unsupported hook.
        # Both declared bodies remain required; the hook is never evaluated.
        self.assertEqual(result['expected_ids'],['test_auditor.C.test_one','test_auditor.C.test_two'])
        self.assertEqual(result['pristine_population']['invalid_population'],['unsupported module load_tests hook: test_auditor'])
        self.assertEqual(result['tests'],{})

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


class LazyBodyHarnessRules(unittest.TestCase):
    """Lazy call creation cannot satisfy synchronous selected-body execution."""

    def run_lazy_fixture(self,label,body):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel=Path(temporary)/'lazy-executed'
            ordinary=Path(temporary)/'ordinary-executed'
            source=('from pathlib import Path\nimport unittest\nimport functools\n'
                    +'sentinel=Path('+repr(str(sentinel))+')\n'
                    +'ordinary=Path('+repr(str(ordinary))+')\n'+body)
            value=LifecycleHarnessRules.run_fixture(self,source)
            record={'label':label,'optimize':sys.flags.optimize,'fixture_source':source,
                    'lazy_sentinel_executed':sentinel.exists(),
                    'ordinary_sentinel_executed':ordinary.exists(),'receipt':value}
            destination=os.environ.get('CBZ_LAZY_CONTROL_EVIDENCE')
            if destination:
                out=Path(destination);out.mkdir(parents=True,exist_ok=True)
                (out/(label+'.json')).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
            return record

    def test_sync_bodies_require_none_and_refuse_other_returns_without_iteration(self):
        passed=self.run_lazy_fixture('ordinary-pass','class C(unittest.TestCase):\n def test_one(self): pass\n')
        self.assertEqual(passed['receipt']['outcome'],'SURVIVED',passed)
        killed=self.run_lazy_fixture('ordinary-none-kill',
            'class C(unittest.TestCase):\n def test_a_return(self): return None\n'
            ' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
        self.assertEqual(killed['receipt']['outcome'],'KILLED',killed)
        for label,expression in [('zero','0'),('false','False'),('empty-string',"''"),('scalar','7')]:
            with self.subTest(value=label):
                record=self.run_lazy_fixture('return-'+label,'class C(unittest.TestCase):\n'
                    +' def test_a_return(self): return '+expression+'\n'
                    +' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
                self.assertEqual(record['receipt']['outcome'],'ERROR',record)
                self.assertFalse(record['lazy_sentinel_executed'],record)
                self.assertTrue(record['ordinary_sentinel_executed'],record)
        unknown=self.run_lazy_fixture('ordinary-unknown-iterable',
            'class UnknownIterable:\n def __iter__(self):\n  sentinel.write_text("iterated")\n  yield 1\n'
            'class C(unittest.TestCase):\n def test_a_return(self): return UnknownIterable()\n'
            ' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
        self.assertEqual(unknown['receipt']['outcome'],'ERROR',unknown)
        self.assertFalse(unknown['lazy_sentinel_executed'],unknown)
        self.assertTrue(unknown['ordinary_sentinel_executed'],unknown)

    def test_mixed_lazy_function_shapes_are_refused_before_any_body(self):
        methods=[('generator',' def test_a_lazy(self):\n  sentinel.write_text("generator")\n  yield 1\n'),
                 ('coroutine',' async def test_a_lazy(self):\n  sentinel.write_text("coroutine")\n'),
                 ('async-generator',' async def test_a_lazy(self):\n  sentinel.write_text("asyncgen")\n  yield 1\n')]
        for label,method in methods:
            with self.subTest(shape=label):
                record=self.run_lazy_fixture('method-'+label,'class C(unittest.TestCase):\n'+method+
                    ' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
                self.assertEqual(record['receipt']['outcome'],'ERROR',record)
                self.assertFalse(record['lazy_sentinel_executed'],record)
                self.assertFalse(record['ordinary_sentinel_executed'],record)
                self.assertEqual(record['receipt']['pristine_population']['invalid_population'],['test_auditor.C.test_a_lazy'])

    def test_wraps_decorated_lazy_functions_are_refused_at_population(self):
        decorator=('def transparent(fn):\n @functools.wraps(fn)\n def wrapper(*args,**kwargs):\n'
                   '  sentinel.write_text("wrapper executed")\n  return fn(*args,**kwargs)\n return wrapper\n'
                   'def lazy_middle(fn):\n @functools.wraps(fn)\n def wrapper(*args,**kwargs):\n'
                   '  sentinel.write_text("middle executed")\n  yield fn(*args,**kwargs)\n return wrapper\n')
        methods=[('generator',' def test_a_lazy(self):\n  yield 1\n'),
                 ('coroutine',' async def test_a_lazy(self):\n  return None\n'),
                 ('async-generator',' async def test_a_lazy(self):\n  yield 1\n'),
                 ('lazy-middle',' @lazy_middle\n def test_a_lazy(self): pass\n')]
        for label,method in methods:
            with self.subTest(shape=label):
                record=self.run_lazy_fixture('wrapped-'+label,decorator+'class C(unittest.TestCase):\n @transparent\n'+method+
                    ' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
                self.assertEqual(record['receipt']['outcome'],'ERROR',record)
                self.assertFalse(record['lazy_sentinel_executed'],record)
                self.assertFalse(record['ordinary_sentinel_executed'],record)
                self.assertEqual(record['receipt']['pristine_population']['invalid_population'],['test_auditor.C.test_a_lazy'])

    def test_sync_lazy_returns_are_rejected_without_advancing_objects(self):
        returns=[('generator','def make_lazy():\n sentinel.write_text("generator")\n yield 1\n','make_lazy()'),
                 ('coroutine','async def make_lazy():\n sentinel.write_text("coroutine")\n','make_lazy()'),
                 ('async-generator','async def make_lazy():\n sentinel.write_text("asyncgen")\n yield 1\n','make_lazy()'),
                 ('custom-awaitable','class CustomAwaitable:\n def __await__(self):\n  sentinel.write_text("awaited")\n  yield 1\n def close(self): sentinel.write_text("custom close")\n','CustomAwaitable()')]
        for label,definitions,expression in returns:
            with self.subTest(shape=label):
                record=self.run_lazy_fixture('return-'+label,definitions+'class C(unittest.TestCase):\n'
                    +' def test_a_lazy(self): return '+expression+'\n'
                    +' def test_b_assertion(self):\n  ordinary.write_text("body")\n  self.fail("ordinary body assertion")\n')
                self.assertEqual(record['receipt']['outcome'],'ERROR',record)
                self.assertFalse(record['lazy_sentinel_executed'],record)
                self.assertTrue(record['ordinary_sentinel_executed'],record)
                self.assertTrue(record['receipt']['tests']['invalid_population'],record)


import os
import shutil
import subprocess

class IdentityHarnessRules(unittest.TestCase):
    """Real ordinary-body controls for loader and in-process identity binding."""

    def run_identity_fixture(self, label, source, selectors=()):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'tests').mkdir()
            runner=root/'mutation_test.py'
            shutil.copy2(HERE/'mutation_test.py',runner)
            events=root/'events'
            header=('import unittest\nfrom pathlib import Path\n'
                    +'def record(label):\n with Path('+repr(str(events))+').open("a") as out: out.write(label+"\\n")\n')
            fixture=root/'tests'/'test_auditor.py'
            fixture.write_text(header+source)
            output=root/'receipt.json'
            command=[sys.executable,'-B','-S',str(runner),*['--internal-test-runner','--test-root',str(root/'tests')],'--output',str(output)]
            for selector in selectors:command.extend(['--test',selector])
            p=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertTrue(output.exists(),p.stderr)
            value=json.loads(output.read_text())
            outcome=mutation_test.classify_test_result(value,p.returncode)
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
        for label,selectors,wanted in [('class',('test_auditor.C',),['first','second']),
                                        ('method',('test_auditor.C.test_second',),['second'])]:
            with self.subTest(selector=label):
                outcome,value,events=self.run_identity_fixture('named-'+label,source,selectors)
                self.assertEqual(outcome,'SURVIVED',value)
                self.assertEqual(events,wanted,value)
        outcome,value,events=self.run_identity_fixture('named-assertion',source.replace('record("first")','record("first"); self.fail("body")'),('test_auditor.C',))
        self.assertEqual(outcome,'KILLED',value)
        self.assertEqual(events,['first','second'],value)

    def test_callable_factory_selector_is_refused_before_factory_or_body(self):
        source=('class C(unittest.TestCase):\n def test_one(self): record("body")\n'
                +'def factory():\n record("factory")\n return unittest.TestSuite([C("test_one")])\n')
        outcome,value,events=self.run_identity_fixture('callable-factory',source,('test_auditor.factory',))
        self.assertEqual(outcome,'ERROR',value)
        self.assertEqual(events,[],value)
        self.assertTrue(value['invalid_population'],value)

        proxy_source=('class C(unittest.TestCase):\n def test_one(self): record("body")\n'
                      +'class Proxy:\n def __getattr__(self,name):\n  record("proxy")\n  return C\nproxy=Proxy()\n')
        outcome,value,events=self.run_identity_fixture('selector-proxy',proxy_source,('test_auditor.proxy.C.test_one',))
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
            for selection in [(),('test_auditor.C',),('test_auditor.C.'+('test_first' if label=='init' else 'test_second' if label.startswith('descriptor') else 'test_one'),)]:
                with self.subTest(constructor=label,selector=selection):
                    outcome,value,events=self.run_identity_fixture('construction-'+label+'-'+str(len(selection))+'-'+('method' if selection and selection[0].count('.')==2 else 'class'),source,selection)
                    self.assertEqual(outcome,'ERROR',value)
                    self.assertEqual(events,[],value)
                    self.assertTrue(value['invalid_population'],value)

        helper_source=('class Alias(staticmethod):\n def __get__(self,instance,owner):\n  record("helper-descriptor")\n  return owner.test_one.__get__(instance,owner)\n'
                       +'class C(unittest.TestCase):\n def test_one(self): record("test")\n def helper(self):\n  record("retained-helper")\n  self.fail("original helper body")\n helper=Alias(helper)\n')
        outcome,value,events=self.run_identity_fixture('selector-helper-descriptor',helper_source,('test_auditor.C.helper',))
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
        spec=importlib.util.spec_from_file_location('identity_runtime_probe',HERE/'mutation_test.py')
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
        value=runtime.execute_test_suite(suite,loader.bindings,loader.invalid)
        self.assertTrue(value['invalid_population'],value)
        self.assertEqual(events,[],value)
        self.assertEqual(value['tests_run'],0,value)

    def test_custom_testcase_run_is_refused_before_execution(self):
        source=('class C(unittest.TestCase):\n'
                +' def run(self,result=None):\n  record("custom-run")\n  return super().run(result)\n'
                +' def test_one(self): record("body")\n')
        outcome,value,events=self.run_identity_fixture('custom-run',source,('test_auditor.C',))
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

if __name__=='__main__':
    unittest.main()
