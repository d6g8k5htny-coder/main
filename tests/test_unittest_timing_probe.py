"""Synthetic subprocess contracts; no research or repository test bodies run."""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/unittest_timing_probe.py'
PASS = 'import unittest\nclass Case(unittest.TestCase):\n    def test_ok(self): self.assertEqual(2 + 2, 4)\n'
MIXED = '''import unittest
class Case(unittest.TestCase):
    def test_a_pass(self): pass
    def test_b_failure(self): self.fail('synthetic failure')
    def test_c_error(self): raise RuntimeError('synthetic error')
    @unittest.skip('synthetic skip')
    def test_d_skip(self): self.fail('not entered')
    @unittest.expectedFailure
    def test_e_expected(self): self.fail('expected')
    @unittest.expectedFailure
    def test_f_unexpected(self): pass
    def test_g_subtest(self):
        for value in (0, 1):
            with self.subTest(value=value): self.assertEqual(value, 0)
'''


def python():
    return [sys.executable, '-I', '-B', '-S'] + (['-O'] if sys.flags.optimize else [])


def summary(stderr):
    return [re.sub(r'in [0-9.]+s', 'in <elapsed>s', line)
            for line in stderr.splitlines()
            if line.startswith(('Ran ', 'OK', 'FAILED'))]


class TimingProbeTests(unittest.TestCase):
    def fixture(self, root, text=PASS):
        tests = root / 'cases'
        tests.mkdir()
        (tests / 'test_sample.py').write_text(text, encoding='utf-8')
        return ['discover', '-s', str(tests), '-p', 'test_*.py', '-v']

    def invoke(self, root, args, baseline=False, journal=None):
        journal = journal or root / 'timing.jsonl'
        if baseline or os.environ.get('TIMING_PROBE_BASELINE') == '1':
            command = python() + ['-m', 'unittest', *args]
        else:
            command = python() + [str(TOOL), str(journal), *args]
        return subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=12)

    def rows(self, root):
        path = root / 'timing.jsonl'
        self.assertTrue(path.is_file(), 'missing per-test timing journal')
        data = path.read_bytes()
        self.assertTrue(data.endswith(b'\n'), 'partial journal line')
        rows = [json.loads(line) for line in data.splitlines()]
        self.assertEqual([r['seq'] for r in rows], list(range(len(rows))))
        clocks = [r['monotonic_ns'] for r in rows]
        self.assertEqual(clocks, sorted(clocks))
        self.assertTrue(all(isinstance(n, int) for n in clocks))
        self.assertEqual(len({r['pid'] for r in rows}), 1)
        self.assertTrue(all(r['schema'] == 'unittest-timing/v1' for r in rows))
        return rows

    def test_success_has_live_start_stop_and_native_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root)
            original = self.invoke(root, args, baseline=True)
            candidate = self.invoke(root, args)
            self.assertEqual(candidate.returncode, original.returncode)
            self.assertEqual(summary(candidate.stderr), summary(original.stderr))
            rows = self.rows(root)
            self.assertEqual([r['event'] for r in rows],
                             ['invocation_start', 'run_start', 'test_start', 'outcome',
                              'test_stop', 'run_end', 'invocation_exit'])
            self.assertEqual(rows[2]['test_id'], 'test_sample.Case.test_ok')
            self.assertEqual(rows[3]['outcome'], 'success')
            self.assertGreaterEqual(rows[4]['elapsed_ns'], 0)
            self.assertEqual(rows[-1]['exit_code'], original.returncode)
            self.assertEqual(rows[0]['optimize'], sys.flags.optimize)

    def test_all_native_outcomes_and_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, MIXED)
            original = self.invoke(root, args, baseline=True)
            candidate = self.invoke(root, args)
            self.assertEqual((candidate.returncode, summary(candidate.stderr)),
                             (original.returncode, summary(original.stderr)))
            self.assertEqual(candidate.returncode, 1)
            rows = self.rows(root)
            starts = [r['test_id'] for r in rows if r['event'] == 'test_start']
            stops = [r['test_id'] for r in rows if r['event'] == 'test_stop']
            self.assertEqual(starts, stops)
            self.assertEqual(len(starts), 7)
            kinds = [r['outcome'] for r in rows if r['event'] == 'outcome']
            self.assertEqual(kinds, ['success', 'failure', 'error', 'skip',
                                    'expected_failure', 'unexpected_success',
                                    'subtest_success', 'subtest_failure'])
            self.assertFalse(next(r for r in rows if r['event'] == 'run_end')['successful'])

    def test_failfast_and_name_selection_match_native_cli(self):
        for extra in (['-f'], ['-k', 'a_pass']):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args = self.fixture(root, MIXED) + extra
                original = self.invoke(root, args, baseline=True)
                candidate = self.invoke(root, args)
                self.assertEqual((candidate.returncode, summary(candidate.stderr)),
                                 (original.returncode, summary(original.stderr)))
                rows = self.rows(root)
                count = next(r['tests_run'] for r in rows if r['event'] == 'run_end')
                self.assertEqual(count, 2 if extra == ['-f'] else 1)

    def test_import_and_class_setup_failures_are_not_hidden(self):
        specimens = ["raise RuntimeError('synthetic import failure')\n",
                     "import unittest\nclass Case(unittest.TestCase):\n"
                     "    @classmethod\n    def setUpClass(cls): raise RuntimeError('setup')\n"
                     "    def test_never(self): pass\n"]
        for text in specimens:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args = self.fixture(root, text)
                original = self.invoke(root, args, baseline=True)
                candidate = self.invoke(root, args)
                self.assertEqual(candidate.returncode, original.returncode)
                self.assertNotEqual(candidate.returncode, 0)
                self.assertEqual(summary(candidate.stderr), summary(original.stderr))
                rows = self.rows(root)
                self.assertTrue(any(r.get('outcome') == 'error' for r in rows))

    def test_empty_discovery_retains_native_version_specific_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, '')
            original = self.invoke(root, args, baseline=True)
            candidate = self.invoke(root, args)
            self.assertEqual((candidate.returncode, summary(candidate.stderr)),
                             (original.returncode, summary(original.stderr)))
            rows = self.rows(root)
            self.assertEqual(next(r['tests_run'] for r in rows if r['event'] == 'run_end'), 0)
            self.assertEqual(rows[-1]['exit_code'], original.returncode)

    def test_journal_is_exclusive_and_fixture_is_not_executed(self):
        text = "from pathlib import Path\nPath('entered').write_text('bad')\n" + PASS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, text)
            journal = root / 'timing.jsonl'; journal.write_bytes(b'original evidence\n')
            result = self.invoke(root, args)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(journal.read_bytes(), b'original evidence\n')
            self.assertFalse((root / 'entered').exists())

    def test_missing_parent_is_failure_before_discovery(self):
        text = "from pathlib import Path\nPath('entered').write_text('bad')\n" + PASS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, text)
            result = self.invoke(root, args, journal=root / 'absent' / 'timing.jsonl')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'entered').exists())

    def test_interrupted_test_leaves_flushed_unmatched_start(self):
        text = "import unittest,time\nclass Case(unittest.TestCase):\n    def test_wait(self): time.sleep(30)\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, text)
            command = python() + [str(TOOL), str(root / 'timing.jsonl'), *args]
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            observed = []
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and process.poll() is None:
                    if (root / 'timing.jsonl').exists():
                        lines = (root / 'timing.jsonl').read_bytes().splitlines()
                        try:
                            observed = [json.loads(line) for line in lines]
                        except json.JSONDecodeError:
                            observed = []
                        if any(r['event'] == 'test_start' for r in observed):
                            break
                    time.sleep(0.01)
                self.assertTrue(any(r['event'] == 'test_start' for r in observed), observed)
                self.assertIsNone(process.poll(), 'start not observed while test was live')
                process.terminate()
                process.communicate(timeout=5)
                rows = self.rows(root)
                self.assertEqual([r['event'] for r in rows],
                                 ['invocation_start', 'run_start', 'test_start'])
                self.assertNotEqual(process.returncode, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_setup_teardown_module_skip_and_subtest_errors_match_native(self):
        specimens = [
            "import unittest\nclass Case(unittest.TestCase):\n"
            "    def setUp(self): raise RuntimeError('setup')\n    def test_never(self): pass\n",
            "import unittest\nclass Case(unittest.TestCase):\n"
            "    def tearDown(self): raise RuntimeError('teardown')\n    def test_ok(self): pass\n",
            "import unittest\ndef setUpModule(): raise unittest.SkipTest('module skip')\n"
            "class Case(unittest.TestCase):\n    def test_never(self): pass\n",
            "import unittest\nclass Case(unittest.TestCase):\n    def test_sub(self):\n"
            "        with self.subTest(part='error'): raise ValueError('subtest')\n",
        ]
        for text in specimens:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args = self.fixture(root, text)
                original = self.invoke(root, args, baseline=True)
                candidate = self.invoke(root, args)
                self.assertEqual((candidate.returncode, summary(candidate.stderr)),
                                 (original.returncode, summary(original.stderr)))
                rows = self.rows(root)
                self.assertEqual(rows[-1]['exit_code'], original.returncode)
                expected = 'subtest_error' if 'test_sub' in text else 'skip' if 'SkipTest' in text else 'error'
                self.assertTrue(any(r.get('outcome') == expected for r in rows), rows)

    def test_buffered_output_and_failure_reporting_are_retained(self):
        text = ("import unittest\nclass Case(unittest.TestCase):\n"
                "    def test_print(self): print('synthetic-output'); self.fail('failure')\n")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root, text) + ['-b']
            original = self.invoke(root, args, baseline=True)
            candidate = self.invoke(root, args)
            self.assertEqual(candidate.returncode, original.returncode)
            self.assertEqual(candidate.stdout, original.stdout)
            self.assertEqual(candidate.stderr.count('synthetic-output'),
                             original.stderr.count('synthetic-output'))
            self.assertEqual(summary(candidate.stderr), summary(original.stderr))
            self.rows(root)

    def test_nonisolated_named_module_imports_match_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'named_case.py').write_text(PASS, encoding='utf-8')
            flags = [sys.executable, '-B', '-S'] + (['-O'] if sys.flags.optimize else [])
            args = ['named_case.Case.test_ok', '-v']
            original = subprocess.run(flags + ['-m', 'unittest', *args], cwd=root,
                                      capture_output=True, text=True, timeout=12)
            candidate = subprocess.run(flags + [str(TOOL), str(root / 'timing.jsonl'), *args],
                                       cwd=root, capture_output=True, text=True, timeout=12)
            self.assertEqual(original.returncode, 0, original.stderr)
            self.assertEqual((candidate.returncode, summary(candidate.stderr)),
                             (original.returncode, summary(original.stderr)))
            rows = self.rows(root)
            self.assertEqual([r['test_id'] for r in rows if r['event'] == 'test_start'],
                             ['named_case.Case.test_ok'])

    def test_existing_symlink_journal_does_not_touch_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root)
            target = root / 'retained'; target.write_bytes(b'original')
            (root / 'timing.jsonl').symlink_to(target)
            run = self.invoke(root, args)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(target.read_bytes(), b'original')

    def test_cli_usage_errors_keep_native_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ['--deliberately-unknown-synthetic-option']
            original = self.invoke(root, args, baseline=True)
            candidate = self.invoke(root, args)
            self.assertEqual(candidate.returncode, original.returncode)
            self.assertNotEqual(candidate.returncode, 0)
            rows = self.rows(root)
            self.assertEqual([r['event'] for r in rows], ['invocation_start', 'invocation_exit'])
            self.assertEqual(rows[-1]['exit_code'], original.returncode)

    def test_midrun_flush_error_cannot_return_success(self):
        spec = importlib.util.spec_from_file_location('timing_proposal_flush', TOOL)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class BrokenFlush(io.StringIO):
            count = 0
            def flush(self):
                self.count += 1
                if self.count == 3:
                    raise OSError('synthetic full journal')
                return super().flush()
        class Synthetic(unittest.TestCase):
            def test_pass(self): pass
        journal = module.Journal(BrokenFlush())
        runner = module.runner_type(journal)(stream=io.StringIO())
        with self.assertRaisesRegex(OSError, 'synthetic full journal'):
            runner.run(unittest.TestSuite([Synthetic('test_pass')]))
        self.assertIsNotNone(journal.failure)
        with self.assertRaisesRegex(OSError, 'synthetic full journal'):
            journal.emit('must_not_resume')

    def test_journal_write_failure_is_visible_and_latched(self):
        spec = importlib.util.spec_from_file_location('timing_proposal', TOOL)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class Broken(io.StringIO):
            def write(self, value): raise OSError('synthetic journal failure')
        writer = module.Journal(Broken())
        with self.assertRaisesRegex(OSError, 'synthetic journal failure'):
            writer.emit('probe')
        self.assertIsNotNone(writer.failure)


if __name__ == '__main__':
    unittest.main()
