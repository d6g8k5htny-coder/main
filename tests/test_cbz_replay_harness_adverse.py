"""Executable counterexamples against the frozen historical replay harness.

These tests intentionally assert the historical defects are observable. A pass
is evidence of the defect, never certification of the historical runner.
"""
import contextlib
import io
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / 'cbz/evidence/kernel/additional/replay-tool'


def historical_runner():
    path = ROOT / 'run_mutations.py'
    module = types.ModuleType('historical_run_mutations')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


class HistoricalCounterexamples(unittest.TestCase):
    def run_harness(self, scenario):
        runner = historical_runner()

        def fake_run(command, cwd, env, capture_output):
            name = Path(env['REPLAY_TOOL_MODULE']).stem
            if name == 'replay_tool':
                count = 0 if scenario == 'empty-baseline' else 31
                stderr = f'Ran {count} tests in 0.001s\n\nOK\n'
                return subprocess.CompletedProcess(command, 0, b'', stderr.encode())
            tests = {mutation[0]: mutation[3] for mutation in runner.MUTATIONS}
            test = tests[name]
            witness = ('"owned names differ" does not match '
                       '"module assignment differs from complete compiled census"'
                       if name == 'owned-names' else 'ValueError not raised')
            if scenario == 'wrong-assertion':
                witness = 'unrelated assertion'
            stderr = (f'FAIL: {test} (test_replay_tool.TranscriptTests.{test})\n'
                      f'AssertionError: {witness}\n')
            if scenario == 'mixed-error':
                stderr += 'ERROR: foreign_test (other.Other.foreign_test)\nRuntimeError: foreign failure\n'
            stderr += ('\nRan 31 tests in 0.001s\n\nFAILED (failures=1, errors=1)\n'
                       if scenario == 'mixed-error' else
                       '\nRan 31 tests in 0.001s\n\nFAILED (failures=1)\n')
            return subprocess.CompletedProcess(command, 1, b'', stderr.encode())

        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run_mutations.py', '--output', str(Path(tmp) / 'result')]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(runner.subprocess, 'run', side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            return (Path(tmp) / 'result' / 'REPORT.json').read_text()

    def test_zero_test_baseline_is_accepted(self):
        self.assertIn('PASS_ALL_DECLARED_MUTANTS_KILLED', self.run_harness('empty-baseline'))

    def test_wrong_assertion_is_accepted(self):
        self.assertIn('PASS_ALL_DECLARED_MUTANTS_KILLED', self.run_harness('wrong-assertion'))

    def test_mixed_failure_and_error_are_accepted(self):
        self.assertIn('PASS_ALL_DECLARED_MUTANTS_KILLED', self.run_harness('mixed-error'))

    def test_poisoned_bytecode_is_executed_despite_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'replay_tool.py'
            original = (ROOT / 'replay_tool.py').read_bytes()
            poison = b"raise RuntimeError('REPLAY_TOOL_POISONED')\n"
            self.assertLess(len(poison), len(original))
            target.write_bytes(poison + b' ' * (len(original) - len(poison)))
            os.utime(target, (1700000000, 1700000000))
            previous = sys.pycache_prefix
            try:
                sys.pycache_prefix = None
                py_compile.compile(str(target), doraise=True, optimize=sys.flags.optimize,
                                   invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            finally:
                sys.pycache_prefix = previous
            target.write_bytes(original)
            os.utime(target, (1700000000, 1700000000))
            testfile = root / 'test_replay_tool.py'
            testfile.write_bytes((ROOT / 'test_replay_tool.py').read_bytes())
            code = ("from pathlib import Path\n"
                    "p=Path(" + repr(str(testfile)) + ")\n"
                    "g={'__name__':'counterexample','__file__':str(p)}\n"
                    "exec(compile(p.read_bytes(),str(p),'exec'),g)\n")
            environment = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
            environment['REPLAY_TOOL_MODULE'] = str(target)
            command = [sys.executable, '-B', '-S']
            if sys.flags.optimize:
                command.append('-O')
            result = subprocess.run(command + ['-c', code], env=environment, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'REPLAY_TOOL_POISONED', result.stderr)


if __name__ == '__main__':
    unittest.main()
