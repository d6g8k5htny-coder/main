"""CLI and false-success controls for the census mutation self-test runner."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MutationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        (self.source / 'tools').mkdir(parents=True)
        (self.source / 'tests').mkdir()
        for path in ['tools/workspace_git_inventory.py', 'tools/workspace_git_inventory_mutations.py', 'tests/test_workspace_git_inventory.py']:
            shutil.copyfile(ROOT / path, self.source / path)
        self.runner = self.source / 'tools/workspace_git_inventory_mutations.py'
        self.output = self.root / 'evidence'

    def run_cli(self, *, override=None, env=None, output=True):
        interpreter = [sys.executable, '-S', '-B']
        if sys.flags.optimize:
            interpreter.append('-' + 'O' * sys.flags.optimize)
        if override is None:
            command = [*interpreter, str(self.runner)]
        else:
            driver = (
                'import importlib.util,sys\n'
                'p=sys.argv[1];s=importlib.util.spec_from_file_location("runner",p)\n'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
                + override + '\n'
                'sys.argv=[p]+sys.argv[2:];raise SystemExit(m.main())\n')
            command = [*interpreter, '-c', driver, str(self.runner)]
        if output:
            command += ['--output', str(self.output)]
        return subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, timeout=90)

    def report(self):
        return json.loads((self.output / 'results.json').read_bytes())

    def reject_lifecycle_assertion(self, stage):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        control = '\n# Owned temporary lifecycle negative control; original assertions retained.\n'
        control += '_original_setup=InventoryTests.setUp\n_original_teardown=InventoryTests.tearDown\n'
        if stage == 'tearDown':
            control += ('def _controlled_teardown(self):\n'
                        '    _original_teardown(self)\n'
                        '    if TOOL.name != "workspace_git_inventory.py":self.fail("CONTROL tearDown assertion")\n'
                        'InventoryTests.tearDown=_controlled_teardown\n')
        else:
            control += ('def _cleanup_assertion(case):case.fail("CONTROL cleanup assertion")\n'
                        'def _controlled_setup(self):\n'
                        '    _original_setup(self)\n'
                        '    if TOOL.name != "workspace_git_inventory.py":\n')
            control += ('        self.fail("CONTROL setUp assertion")\n' if stage == 'setUp' else
                        '        self.addCleanup(_cleanup_assertion,self)\n')
            control += 'InventoryTests.setUp=_controlled_setup\n'
        with copied.open('a') as stream:
            stream.write(control)
        # All thirteen scratch sources deliberately survive. Only the lifecycle
        # assertion fails; no canonical test or production source is weakened.
        result = self.run_cli(override='m.CASES=[(n,o,o,t) for n,o,_new,t in m.CASES]')
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(len(report['mutations']), 13)
        self.assertFalse(report['all_detected'])
        self.assertFalse(report['complete'])
        for row in report['mutations']:
            self.assertEqual(row['failures'], 1)
            self.assertEqual(row['errors'], 0)
            self.assertFalse(row['detected_by_assertion'])
            self.assertEqual(row['body_calls'], 0 if stage == 'setUp' else 1)
            self.assertEqual(row['issues'][0]['phase'], stage)
            self.assertFalse(row['issues'][0]['origin_in_test_body'])

    def test_setup_assertion_cannot_count_as_mutation_detection(self):
        self.reject_lifecycle_assertion('setUp')

    def test_teardown_assertion_cannot_count_as_mutation_detection(self):
        self.reject_lifecycle_assertion('tearDown')

    def test_cleanup_assertion_cannot_count_as_mutation_detection(self):
        self.reject_lifecycle_assertion('cleanup')

    def test_child_optimization_matches_companion(self):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        with copied.open('a') as stream:
            stream.write('\nimport sys\n_optimization_setup=InventoryTests.setUp\n'
                         'def _require_optimization(self):\n'
                         '    _optimization_setup(self)\n'
                         '    self.assertEqual(sys.flags.optimize,' + str(sys.flags.optimize) + ')\n'
                         'InventoryTests.setUp=_require_optimization\n')
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = self.report()
        self.assertTrue(report['complete'])
        self.assertEqual(report['baseline']['optimization'], sys.flags.optimize)
        self.assertTrue(all(row['optimization'] == sys.flags.optimize for row in report['mutations']))

    def test_deferred_test_bodies_cannot_pass_baseline(self):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        original = copied.read_text()
        controls = {
            'coroutine': 'async def _unsupported(self):\n    self.fail("unexecuted coroutine body")\n',
            'generator': 'def _unsupported(self):\n    yield None\n    self.fail("unexecuted generator body")\n',
            'async_generator': 'async def _unsupported(self):\n    yield None\n    self.fail("unexecuted async-generator body")\n',
            'returned_generator': 'def _deferred(self):\n    yield None\n    self.fail("unexecuted returned generator")\ndef _unsupported(self):\n    return _deferred(self)\n',
        }
        for kind, control in controls.items():
            copied.write_text(original + '\n' + control +
                              'InventoryTests.test_unborn_and_detached_heads_are_observations_not_errors=_unsupported\n')
            self.output = self.root / ('unsupported-' + kind)
            result = self.run_cli()
            self.assertEqual(result.returncode, 1, kind + ': ' + result.stdout + result.stderr)
            report = self.report()
            self.assertFalse(report['baseline']['passed'], kind)
            self.assertEqual(report['baseline']['unsupported_bodies'], 1, kind)
            self.assertEqual(report['mutations'], [], kind)
            self.assertFalse(report['complete'], kind)

    def reject_population(self, control, *, loaded_valid):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        with copied.open('a') as stream:
            stream.write('\n' + control)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        report = self.report()
        self.assertFalse(report['baseline']['passed'])
        self.assertFalse(report['baseline']['population_valid'])
        self.assertEqual(report['baseline']['loaded_population_valid'], loaded_valid)
        self.assertEqual(report['mutations'], [])
        self.assertFalse(report['complete'])
        return report['baseline']

    def callback_control(self, body):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        with copied.open('a') as stream:
            stream.write('\nimport sys\ndef _callback_control(original):\n'
                         '    def controlled(self):\n'
                         '        if TOOL.name=="workspace_git_inventory.py":return original(self)\n' + body +
                         '    return controlled\n'
                         'for _name in list(vars(InventoryTests)):\n'
                         '    if _name.startswith("test_"):setattr(InventoryTests,_name,_callback_control(getattr(InventoryTests,_name)))\n')
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['mutations']), 13)
        self.assertTrue(all(not r['detected_by_assertion'] for r in report['mutations']))
        return report

    def test_callback_without_escaping_assertion_cannot_kill(self):
        report = self.callback_control(
            '        try:self.fail("CONTROL swallowed callback assertion")\n'
            '        except AssertionError:self._outcome.result.addFailure(self,sys.exc_info())\n')
        for row in report['mutations']:
            self.assertTrue(row['population_valid'])
            self.assertEqual(row['failures'], 1)
            self.assertEqual(row['errors'], 0)
            self.assertFalse(row['issues'][0]['escaping_assertion'])

    def test_native_and_foreign_subtest_reports_are_unsupported(self):
        copied = self.source / 'tests/test_workspace_git_inventory.py'
        original = copied.read_text()
        for kind, body in [
            ('native', '        with self.subTest(control="native"):self.fail("CONTROL subtest")\n'),
            ('foreign', '        foreign=unittest.FunctionTestCase(lambda:None)\n'
                        '        subtest=unittest.case._SubTest(foreign,"CONTROL foreign",{})\n'
                        '        try:self.fail("CONTROL foreign subtest")\n'
                        '        except AssertionError:self._outcome.result.addSubTest(foreign,subtest,sys.exc_info())\n')]:
            copied.write_text(original)
            self.output = self.root / ('subtest-' + kind)
            report = self.callback_control(body)
            self.assertTrue(all(row['subtests'] == 1 for row in report['mutations']), kind)
            self.assertTrue(all(row['failures'] == 1 and row['errors'] == 0 for row in report['mutations']), kind)
        copied.write_text(original + '\n_original_setup=InventoryTests.setUp\n'
                          'def _passing_subtest(self):\n'
                          '    _original_setup(self)\n'
                          '    with self.subTest(control="passing fixture subtest"):pass\n'
                          'InventoryTests.setUp=_passing_subtest\n')
        self.output = self.root / 'passing-baseline-subtests'
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.report()['baseline']['subtests'], 20)
        self.assertFalse(self.report()['baseline']['passed'])
        self.assertEqual(self.report()['mutations'], [])

    def test_loader_cannot_replace_population_with_duplicate_method(self):
        baseline = self.reject_population(
            'def load_tests(loader,tests,pattern):\n'
            '    return unittest.TestSuite(InventoryTests("test_unborn_and_detached_heads_are_observations_not_errors") for _ in range(20))\n',
            loaded_valid=False)
        self.assertEqual(baseline['tests_run'], 20)
        self.assertEqual(len(set(baseline['loaded_tests'])), 1)
        self.assertEqual(len(set(baseline['body_completed_tests'])), 1)

    def test_loader_cannot_substitute_unexpected_class(self):
        self.reject_population(
            'class OtherInventoryTests(InventoryTests):pass\n'
            'def load_tests(loader,tests,pattern):\n'
            '    return loader.loadTestsFromTestCase(OtherInventoryTests)\n', loaded_valid=False)

    def test_loader_cannot_rebind_pristine_class_identity(self):
        self.reject_population(
            'def load_tests(loader,tests,pattern):\n'
            '    class OtherInventoryTests(InventoryTests):pass\n'
            '    OtherInventoryTests.__module__=InventoryTests.__module__\n'
            '    OtherInventoryTests.__qualname__=InventoryTests.__qualname__\n'
            '    globals()["InventoryTests"]=OtherInventoryTests\n'
            '    return loader.loadTestsFromTestCase(OtherInventoryTests)\n', loaded_valid=False)

    def test_loader_cannot_alias_expected_methods_to_one_body(self):
        self.reject_population(
            'def load_tests(loader,tests,pattern):\n'
            '    body=InventoryTests.test_unborn_and_detached_heads_are_observations_not_errors\n'
            '    for name in loader.getTestCaseNames(InventoryTests):setattr(InventoryTests,name,body)\n'
            '    return tests\n', loaded_valid=False)

    def test_loaded_population_must_actually_execute_each_expected_body(self):
        baseline = self.reject_population(
            'class RepeatedExecution(unittest.TestSuite):\n'
            '    def run(self,result,debug=False):\n'
            '        case=next(c for c in self._tests[0] if c._testMethodName=="test_unborn_and_detached_heads_are_observations_not_errors")\n'
            '        for _ in range(20):case.run(result)\n'
            '        return result\n'
            'def load_tests(loader,tests,pattern):return RepeatedExecution(tests)\n', loaded_valid=True)
        self.assertEqual(baseline['tests_run'], 20)
        self.assertEqual(baseline['body_calls'], 20)
        self.assertEqual(len(set(baseline['loaded_tests'])), 20)
        self.assertEqual(len(set(baseline['started_tests'])), 1)
        self.assertEqual(len(set(baseline['body_completed_tests'])), 1)

    def test_requires_new_output_and_keeps_private_success_evidence(self):
        missing = self.run_cli(output=False)
        self.assertEqual(missing.returncode, 2)
        self.assertFalse(self.output.exists())
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = self.report()
        self.assertTrue(report['complete'])
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['baseline']['tests_run'], 20)
        self.assertEqual(len(report['mutations']), 13)
        self.assertTrue(all(row['detected_by_assertion'] for row in report['mutations']))
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.output.iterdir()}
        repeated = self.run_cli()
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.output.iterdir()})

    def test_failed_baseline_never_runs_mutations(self):
        fake = self.root / 'bin'
        fake.mkdir()
        git = fake / 'git'
        git.write_text('#!/bin/sh\nexit 7\n')
        git.chmod(0o700)
        env = os.environ.copy()
        env['PATH'] = str(fake) + os.pathsep + env['PATH']
        result = self.run_cli(env=env)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertFalse(report['baseline']['passed'])
        self.assertEqual(report['mutations'], [])
        self.assertFalse(report['complete'])

    def test_zero_test_baseline_cannot_report_success(self):
        result = self.run_cli(override='m.DRIVER=m.DRIVER.replace("unittest.defaultTestLoader.loadTestsFromModule(module)","unittest.TestSuite()")')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertEqual(report['baseline']['tests_run'], 0)
        self.assertTrue(report['baseline']['successful'])
        self.assertFalse(report['baseline']['passed'])
        self.assertEqual(report['mutations'], [])
        self.assertFalse(report['complete'])

    def test_surviving_mutant_is_failure(self):
        result = self.run_cli(override='n,o,_new,t=m.CASES[0];m.CASES[0]=(n,o,o,t)')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['mutations'][0]['returncode'], 0)
        self.assertFalse(report['mutations'][0]['detected_by_assertion'])
        self.assertFalse(report['all_detected'])
        self.assertFalse(report['complete'])

    def test_test_error_cannot_count_as_mutation_detection(self):
        result = self.run_cli(override='n,o,new,_t=m.CASES[0];m.CASES[0]=(n,o,new,"nonexistent_test")')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['mutations'][0]['errors'], 1)
        self.assertFalse(report['mutations'][0]['detected_by_assertion'])
        self.assertFalse(report['all_detected'])
        self.assertFalse(report['complete'])


if __name__ == '__main__':
    unittest.main()
