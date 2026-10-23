"""Check numerical-test admission without importing or running certificate code.

The exact source decorators are applied to synthetic test bodies in isolated
stdlib-only children. Temporary NumPy/GUDHI stubs model discovery/import states;
these are selection controls, not numerical-runtime or certificate evidence.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/periodic_h0/test_finite_certificate.py'
TARGETS = (
    'test_extractor_captures_the_actual_generator_input',
    'test_regeneration_writes_fixed_lf_under_windows_text_translation',
)
PROBE = r'''
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

source, targets, available, broken = json.loads(sys.argv[1])
tree = ast.parse(Path(source).read_text(encoding='utf-8'), filename=source)
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CertificateTests']
if len(classes) != 1:
    raise ValueError('Expected one CertificateTests class')
methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
for name in ('numpy', 'gudhi'):
    if name in sys.modules or importlib.util.find_spec(name) is not None:
        raise RuntimeError('Child numerical dependency isolation failed: ' + name)
entered, successes, errors = [], [], []
class Result(unittest.TestResult):
    def addSuccess(self, test):
        successes.append(test._testMethodName)
        super().addSuccess(test)
    def addError(self, test, err):
        errors.append([test._testMethodName, err[0].__name__, str(err[1])])
        super().addError(test, err)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name in available:
        text = ('raise RuntimeError("synthetic broken ' + name + '")\n'
                if name == broken else 'SYNTHETIC = True\n')
        (root / (name + '.py')).write_text(text, encoding='utf-8')
    sys.path.insert(0, directory)
    importlib.invalidate_caches()
    namespace = {}
    for name in targets:
        method = methods[name]
        if len(method.decorator_list) != 1:
            raise ValueError('Expected one admission decorator: ' + name)
        decorator = method.decorator_list[0]
        if not (isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == 'unittest'
                and decorator.func.attr == 'skipUnless'):
            raise ValueError('Unexpected admission decorator: ' + name)
        def body(self, label=name):
            entered.append(label)
            importlib.import_module('numpy')
            importlib.import_module('gudhi')
        body.__name__ = name
        expression = ast.Expression(body=decorator)
        wrapper = eval(compile(expression, source, 'eval'),
                       {'__builtins__': {}, 'unittest': unittest, 'importlib': importlib})
        namespace[name] = wrapper(body)
    case = type('SyntheticAdmissionCase', (unittest.TestCase,), namespace)
    suite = unittest.TestSuite(case(name) for name in targets)
    result = Result()
    suite.run(result)
    payload = {
        'run': result.testsRun,
        'skipped': [[test._testMethodName, reason] for test, reason in result.skipped],
        'errors': errors,
        'failures': len(result.failures),
        'entered': entered,
        'successes': successes,
        'optimized': sys.flags.optimize,
        'fixture_files': sorted(p.name for p in root.iterdir()),
    }
print(json.dumps(payload, sort_keys=True))
'''


class PeriodicCertificateDependencies(unittest.TestCase):
    def probe(self, available=(), broken=None):
        before = SOURCE.read_bytes()
        command = [sys.executable, '-I', '-B', '-S']
        if sys.flags.optimize:
            command.append('-O')
        arguments = json.dumps([str(SOURCE), TARGETS, list(available), broken])
        run = subprocess.run(command + ['-c', PROBE, arguments],
                             capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stderr, '')
        self.assertEqual(SOURCE.read_bytes(), before, 'source was modified by admission probe')
        result = json.loads(run.stdout)
        self.assertEqual(result['optimized'], sys.flags.optimize)
        self.assertEqual(result['fixture_files'], sorted(name + '.py' for name in available))
        self.assertEqual(result['run'], 2)
        self.assertEqual(result['failures'], 0)
        return result

    def test_missing_either_or_both_dependencies_skips_numerical_methods(self):
        for available in ((), ('numpy',), ('gudhi',)):
            with self.subTest(available=available):
                result = self.probe(available)
                self.assertEqual([name for name, _ in result['skipped']], list(TARGETS), result)
                self.assertEqual(result['entered'], [], result)
                self.assertEqual(result['errors'], [], result)

    def test_both_discoverable_dependencies_admit_synthetic_bodies(self):
        result = self.probe(('numpy', 'gudhi'))
        self.assertEqual(result['skipped'], [])
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['entered'], list(TARGETS))
        self.assertEqual(result['successes'], list(TARGETS))

    def test_present_but_broken_dependency_errors_remain_visible(self):
        for broken in ('numpy', 'gudhi'):
            with self.subTest(broken=broken):
                result = self.probe(('numpy', 'gudhi'), broken)
                self.assertEqual(result['skipped'], [], result)
                self.assertEqual(result['entered'], list(TARGETS), result)
                self.assertEqual(result['successes'], [], result)
                self.assertEqual(result['errors'], [
                    [name, 'RuntimeError', 'synthetic broken ' + broken] for name in TARGETS
                ], result)

    def test_skips_explain_the_required_numerical_environment(self):
        result = self.probe()
        self.assertEqual(len(result['skipped']), 2)
        for name, reason in result['skipped']:
            self.assertIn('pinned numerical environment', reason, name)

    def test_core_certificate_methods_are_not_whole_module_skipped(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CertificateTests']
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].decorator_list, [])
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        self.assertGreater(len(methods), len(TARGETS), 'missing standard-library core methods')
        decorated = {n.name for n in methods if n.decorator_list}
        self.assertEqual(decorated, set(TARGETS))


if __name__ == '__main__':
    unittest.main()
