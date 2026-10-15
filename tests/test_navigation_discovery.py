"""Execute the real navigation shell block with inert process-result fixtures.

The fixtures model unittest summaries and exit statuses, including Python 3.11's
successful empty suite. They do not execute project checks or emulate GitHub CI.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NORMAL = ['-B', '-S', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']
OPTIMIZED = ['-B', '-O', '-S', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']
GOOD = 'Ran 2 tests in 0.001s\n\nOK\n'
ZERO = 'Ran 0 tests in 0.000s\n\nOK\n'
DRIVER = '''import json, os, sys
from pathlib import Path
kind, *args = sys.argv[1:]
with Path(os.environ['FIXTURE_TRACE']).open('a') as stream:
    stream.write(json.dumps([kind, *args]) + '\\n')
if kind == 'git' and args == ['rev-parse', 'HEAD']:
    print('inert-fixture-commit')
elif kind == 'python' and args == ['--version']:
    print('inert process-result fixture; not a Python-version assertion')
elif kind == 'python' and '-m' in args:
    mode = 'optimized' if '-O' in args else 'normal'
    data = json.loads(Path(os.environ['FIXTURE_RESULTS']).read_text())[mode]
    sys.stderr.write(data['output'])
    raise SystemExit(data['exit'])
elif kind == 'python' and args in (
        ['-B', '-S', 'tools/workspace_landing_check.py'],
        ['-B', '-S', 'tools/navigation_check.py', '--verify-public']):
    print('{}')
else:
    raise SystemExit('unexpected command in isolated workflow fixture: ' + repr([kind, *args]))
'''


def workflow_block():
    text = (ROOT / '.github/workflows/navigation.yml').read_text(encoding='utf-8')
    marker = '      - name: Current landing and navigation controls\n        run: |\n'
    if text.count(marker) != 1:
        raise ValueError('expected exactly one navigation controls run block')
    block = text.split(marker, 1)[1].split('\n      - ', 1)[0]
    lines = block.splitlines()
    if not lines or any(line and not line.startswith('          ') for line in lines):
        raise ValueError('unsupported navigation run-block indentation')
    return '\n'.join(line[10:] if line else '' for line in lines) + '\n'


class NavigationDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which('bash')
        if not cls.bash:
            raise RuntimeError('Bash is required for navigation shell controls')
        cls.block = workflow_block()

    def run_block(self, normal=GOOD, optimized=GOOD, normal_exit=0, optimized_exit=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'bin'
            binary.mkdir()
            driver = root / 'fixture.py'
            driver.write_text(DRIVER, encoding='utf-8')
            for name in ('python', 'git'):
                shim = binary / name
                shim.write_text('#!/bin/sh\nexec "$FIXTURE_INTERPRETER" -B -S '
                                '"$FIXTURE_DRIVER" ' + name + ' "$@"\n', encoding='utf-8')
                shim.chmod(0o700)
            results = root / 'results.json'
            results.write_text(json.dumps({
                'normal': {'output': normal, 'exit': normal_exit},
                'optimized': {'output': optimized, 'exit': optimized_exit},
            }), encoding='utf-8')
            script = root / 'run.sh'
            script.write_text(self.block, encoding='utf-8')
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('PYTHON') and key not in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS')}
            env.update(PATH=str(binary) + os.pathsep + os.defpath,
                       FIXTURE_INTERPRETER=sys.executable, FIXTURE_DRIVER=str(driver),
                       FIXTURE_TRACE=str(root / 'trace.jsonl'), FIXTURE_RESULTS=str(results))
            run = subprocess.run([self.bash, '--noprofile', '--norc', '-e', str(script)],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, encoding='utf-8', timeout=15)
            trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
            logs = {path.name: path.read_text(encoding='utf-8')
                    for path in (root / 'navigation-evidence').iterdir()}
            return run, trace, logs

    def assert_rejected(self, mode, output, exit_status=0):
        kwargs = {mode: output, mode + '_exit': exit_status}
        run, trace, logs = self.run_block(**kwargs)
        self.assertEqual(run.returncode, exit_status or 1, run.stdout + run.stderr)
        self.assertEqual(logs['tests-' + mode + '.log'], output)
        self.assertNotIn(['python', '-B', '-S', 'tools/navigation_check.py', '--verify-public'], trace)
        self.assertNotIn('navigation.json', logs)
        self.assertEqual(trace.count(['python', *NORMAL]), 1)
        self.assertEqual(trace.count(['python', *OPTIMIZED]), int(mode == 'optimized'))
        if exit_status == 0:
            self.assertIn('::error::navigation ' + mode + ' discovery collected 0 tests or returned no count',
                          run.stdout + run.stderr)

    def test_positive_plural_summaries_reach_navigation(self):
        run, trace, logs = self.run_block()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(trace.count(['python', *NORMAL]), 1)
        self.assertEqual(trace.count(['python', *OPTIMIZED]), 1)
        self.assertEqual(logs['tests-normal.log'], GOOD)
        self.assertEqual(logs['tests-optimized.log'], GOOD)
        self.assertIn('navigation.json', logs)

    def test_positive_singular_summaries_are_accepted(self):
        singular = 'Ran 1 test in 0.001s\n\nOK\n'
        run, _, logs = self.run_block(normal=singular, optimized=singular)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('navigation.json', logs)

    def test_normal_zero_count_exit_zero_is_rejected(self):
        self.assert_rejected('normal', ZERO)

    def test_optimized_zero_count_exit_zero_is_rejected(self):
        self.assert_rejected('optimized', ZERO)

    def test_missing_normal_summary_is_rejected(self):
        self.assert_rejected('normal', 'OK\n')

    def test_missing_optimized_summary_is_rejected(self):
        self.assert_rejected('optimized', 'OK\n')

    def test_nonzero_normal_exit_is_not_masked(self):
        self.assert_rejected('normal', GOOD + 'fixture failure\n', 23)

    def test_nonzero_optimized_exit_is_not_masked(self):
        self.assert_rejected('optimized', GOOD + 'fixture failure\n', 23)

    def test_positive_skipped_count_is_not_an_empty_suite(self):
        skipped = 'Ran 1 test in 0.001s\n\nOK (skipped=1)\n'
        run, _, logs = self.run_block(normal=skipped, optimized=skipped)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(logs['tests-normal.log'], skipped)
        self.assertEqual(logs['tests-optimized.log'], skipped)

    def test_malformed_summary_does_not_supply_a_positive_count(self):
        for output in ('prefix Ran 2 tests in 0.001s\n', 'Ran -1 tests in 0.001s\n',
                       'Ran 01 tests in 0.001s\n', 'Ran 2 testcases in 0.001s\n'):
            for mode in ('normal', 'optimized'):
                with self.subTest(output=output, mode=mode):
                    self.assert_rejected(mode, output)


if __name__ == '__main__':
    unittest.main()
