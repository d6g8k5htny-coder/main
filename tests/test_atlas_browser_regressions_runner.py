"""Controls for exact two-test execution; these never launch a browser."""
import ast
import importlib.util
import io
import json
import tempfile
import types
import hashlib
from unittest import mock
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tools/run_atlas_browser_regressions.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('atlas_browser_runner', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AtlasBrowserRunnerControls(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def suite(self, behavior='pass', ids=None):
        ids = self.runner.EXPECTED_IDS if ids is None else ids
        cases = []
        for index, identity in enumerate(ids):
            def test(case, selected=index == 0):
                if not selected:
                    return
                if behavior == 'skip':
                    case.skipTest('synthetic missing prerequisite')
                elif behavior in ('failure', 'expected_failure'):
                    case.fail('synthetic assertion failure')
                elif behavior == 'error':
                    raise RuntimeError('synthetic runtime error')
                elif behavior == 'subtest_failure':
                    with case.subTest(value=1):
                        case.fail('synthetic subtest failure')
            if behavior in ('expected_failure', 'unexpected_success') and index == 0:
                test = unittest.expectedFailure(test)
            cls = type('SyntheticCase', (unittest.TestCase,), {
                'runTest': test, 'id': lambda case, value=identity: value})
            cases.append(cls())
        return unittest.TestSuite(cases)

    def test_exact_two_successes_pass(self):
        result = self.runner.run_suite(self.suite(), io.StringIO())
        self.assertTrue(result['passed'])
        self.assertEqual(result['tests_run'], 2)
        self.assertEqual(result['successful_ids'], list(self.runner.EXPECTED_IDS))

    def test_empty_inventory_refused(self):
        with self.assertRaisesRegex(ValueError, 'test inventory'):
            self.runner.run_suite(self.suite(ids=[]), io.StringIO())

    def test_duplicate_inventory_refused(self):
        with self.assertRaisesRegex(ValueError, 'test inventory'):
            self.runner.run_suite(self.suite(ids=[self.runner.EXPECTED_IDS[0]] * 2), io.StringIO())

    def test_wrong_and_extra_ids_refused(self):
        for ids in [('wrong', self.runner.EXPECTED_IDS[1]), (*self.runner.EXPECTED_IDS, 'extra')]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'test inventory'):
                self.runner.run_suite(self.suite(ids=ids), io.StringIO())

    def test_skip_is_failure(self):
        result = self.runner.run_suite(self.suite('skip'), io.StringIO())
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['skips']), 1)

    def test_failures_errors_and_subtests_are_not_success(self):
        for kind in ('failure', 'error', 'subtest_failure'):
            with self.subTest(kind=kind):
                self.assertFalse(self.runner.run_suite(self.suite(kind), io.StringIO())['passed'])

    def test_expected_failure_is_not_success(self):
        result = self.runner.run_suite(self.suite('expected_failure'), io.StringIO())
        self.assertFalse(result['passed'])
        self.assertEqual(result['expected_failures'], 1)

    def test_unexpected_success_is_not_success(self):
        result = self.runner.run_suite(self.suite('unexpected_success'), io.StringIO())
        self.assertFalse(result['passed'])
        self.assertEqual(result['unexpected_successes'], 1)

    def test_missing_execution_is_not_success(self):
        class NoExecution(unittest.TestSuite):
            def run(self, result, debug=False):
                return result
        suite = NoExecution(list(self.suite()))
        self.assertFalse(self.runner.run_suite(suite, io.StringIO())['passed'])

    def invoke_main(self, directory, behavior='pass', identities=None):
        identities = identities or [{'commit': 'a', 'tree': 'b'}] * 2
        with mock.patch.object(self.runner, 'git_identity', side_effect=identities), \
                mock.patch.object(self.runner, 'runtime_identity', return_value={'synthetic': True}), \
                mock.patch.object(self.runner.unittest.defaultTestLoader, 'loadTestsFromNames', return_value=self.suite(behavior)), \
                mock.patch.dict(self.runner.os.environ), \
                mock.patch.object(self.runner.sys, 'path', list(self.runner.sys.path)), \
                mock.patch.object(self.runner.sys, 'stderr', io.StringIO()):
            return self.runner.main(['--output', str(directory)])

    def test_main_success_and_failure_exit_match_new_report(self):
        for behavior, expected in [('pass', 0), ('skip', 1), ('error', 1), ('expected_failure', 1)]:
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'attempt'
                self.assertEqual(self.invoke_main(output, behavior), expected)
                report = json.loads((output / 'result.json').read_text())
                self.assertEqual(report['passed'], expected == 0)
                self.assertEqual(report['scientific_effect'], 'NONE')

    def test_reused_output_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            previous = b'original retained bytes\n'
            (output / 'result.json').write_bytes(previous)
            with self.assertRaises(FileExistsError):
                self.invoke_main(output)
            self.assertEqual((output / 'result.json').read_bytes(), previous)

    def test_source_drift_cannot_leave_successful_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'attempt'
            identities = [{'commit': 'a', 'tree': 'b'}, {'commit': 'c', 'tree': 'd'}]
            self.assertEqual(self.invoke_main(output, identities=identities), 1)
            report = json.loads((output / 'result.json').read_text())
            self.assertFalse(report['passed'])
            self.assertIn('source identity changed', report['error'])

    def test_six_runner_source_mutants_are_killed_by_assertions(self):
        source = PATH.read_text()
        mutations = (
            ('inventory', 'if tuple(test_ids(suite)) != EXPECTED_IDS:', 'if False:', 'test_empty_inventory_refused'),
            ('green', "'passed': passed", "'passed': True", 'test_skip_is_failure'),
            ('drift', "if report['source_before'] != report['source_after']:", 'if False:', 'test_source_drift_cannot_leave_successful_report'),
            ('overwrite', 'output.mkdir(parents=True, exist_ok=False)', 'output.mkdir(parents=True, exist_ok=True)', 'test_reused_output_is_refused_without_overwrite'),
            ('exit', "return 0 if report['passed'] else 1", 'return 0', 'test_main_success_and_failure_exit_match_new_report'),
            ('error', "report['passed'] = False", 'pass', 'test_source_drift_cannot_leave_successful_report'),
        )
        for name, before, after, method in mutations:
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temp:
                self.assertIn(before, source)
                path = Path(temp) / 'mutant.py'
                path.write_text(source.replace(before, after, 1))
                spec = importlib.util.spec_from_file_location('mutant_' + name, path)
                mutant = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mutant)
                with mock.patch(__name__ + '.load_runner', return_value=mutant):
                    result = unittest.TextTestRunner(stream=io.StringIO()).run(
                        unittest.TestSuite([type(self)(method)]))
                self.assertGreater(len(result.failures), 0, name)
                self.assertEqual(result.errors, [], name)
                self.assertEqual(result.skipped, [], name)

    def test_focus_and_reader_explicitly_enable_sandbox(self):
        for name in ('test_atlas_focus_geometry.py', 'test_atlas_reader_scroll.py'):
            tree = ast.parse((ROOT / 'tests' / name).read_text())
            launches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == 'launch']
            self.assertEqual(len(launches), 1)
            values = {item.arg: item.value for item in launches[0].keywords}
            self.assertIn('chromium_sandbox', values)
            self.assertIs(values['chromium_sandbox'].value, True)

    def browser_selection(self, filename, attribute, override, available):
        tree = ast.parse((ROOT / 'tests' / filename).read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == attribute
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        launches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'launch']
        executable = next(item.value for item in launches[0].keywords
                          if item.arg == 'executable_path')
        self.assertEqual(ast.dump(executable), ast.dump(ast.Name(id=attribute, ctx=ast.Load())))
        environment = {} if override is None else {'ATLAS_BROWSER_EXECUTABLE': override}
        return eval(compile(ast.Expression(assignments[0].value), filename, 'eval'), {
            'os': types.SimpleNamespace(environ=environment),
            'shutil': types.SimpleNamespace(which=available.get)})

    def test_both_tests_select_override_then_chrome_then_chromium(self):
        cases = (
            ('/explicit/browser', {'google-chrome': '/chrome', 'chromium': '/chromium'}, '/explicit/browser'),
            (None, {'google-chrome': '/chrome', 'chromium': '/chromium'}, '/chrome'),
            ('', {'google-chrome': '/chrome', 'chromium': '/chromium'}, '/chrome'),
            (None, {'chromium': '/chromium'}, '/chromium'),
            (None, {}, None),
        )
        for filename, attribute in (('test_atlas_focus_geometry.py', 'CHROMIUM'),
                                    ('test_atlas_reader_scroll.py', 'BROWSER')):
            for override, available, expected in cases:
                with self.subTest(filename=filename, override=override, available=available):
                    self.assertEqual(self.browser_selection(filename, attribute, override, available), expected)

    def test_runtime_identity_records_each_test_selected_launcher(self):
        # Version probing is stubbed: no real executable or browser is launched.
        with tempfile.TemporaryDirectory() as temp:
            modules = {}
            expected = []
            for filename, attribute in (('test_atlas_focus_geometry.py', 'CHROMIUM'),
                                        ('test_atlas_reader_scroll.py', 'BROWSER')):
                launcher = Path(temp) / filename
                launcher.write_bytes(filename.encode())
                selected = self.browser_selection(filename, attribute, str(launcher), {})
                self.assertEqual(selected, str(launcher))
                name = 'tests.' + filename.removesuffix('.py')
                modules[name] = types.SimpleNamespace(**{attribute: selected})
                expected.append({'module': name, 'path': str(launcher.resolve()),
                                 'version': 'controlled version',
                                 'sha256': hashlib.sha256(launcher.read_bytes()).hexdigest()})
            with mock.patch.dict(self.runner.sys.modules, modules), \
                    mock.patch.object(self.runner.subprocess, 'check_output', return_value='controlled version\n') as probe, \
                    mock.patch.object(self.runner.importlib.metadata, 'version', return_value='controlled playwright'):
                result = self.runner.runtime_identity()
            self.assertEqual(result['browsers'], expected)
            self.assertEqual(probe.call_args_list, [mock.call([item['path'], '--version'], text=True,
                             stderr=self.runner.subprocess.STDOUT, timeout=15) for item in expected])

    def test_workflow_runs_both_modes_and_retains_evidence(self):
        source = (ROOT / '.github/workflows/public-shop.yml').read_text()
        self.assertIn('python -B tools/run_atlas_browser_regressions.py --output browser-evidence/atlas-regressions/normal', source)
        self.assertIn('python -B -O tools/run_atlas_browser_regressions.py --output browser-evidence/atlas-regressions/optimized', source)
        self.assertIn('tests.test_atlas_browser_regressions_runner', source)
        self.assertIn('path: browser-evidence', source)
        self.assertIn('if: always()', source)


if __name__ == '__main__':
    unittest.main()
