"""Acceptance tests for the bounded exact-module adapter.

Changes that suppress manifest validation, failure propagation, time limits,
sequential waiting, or binding checks must make these tests fail.
"""
import importlib.util
import json
import os
import resource
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LEAN = Path(os.environ.get('CBZ_TEST_LEAN', '/home/agent/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean'))


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('kernel_replay', ROOT / 'kernel_replay.py')
        cls.kr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.kr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'evidence')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'Toy.lean'
        self.source.write_text('theorem toy : True := True.intro\n')
        (self.base / 'Toy.olean').write_bytes(b'opaque bytes for wrapper test')

    def manifest(self, names=('Toy',)):
        for name in names:
            (self.base / (name + '.lean')).write_text('theorem toy : True := True.intro\n')
            (self.base / (name + '.olean')).write_bytes(b'opaque bytes for wrapper test')
        return self.kr.prepare_manifest(str(LEAN), [str(self.base)],
            [f'{n}={self.base / (n + ".lean")}' for n in names], [])

    def fake(self, code):
        path = self.base / 'fake-lean'
        path.write_text('#!/usr/bin/env python3\n' + code)
        path.chmod(0o700)
        return path

    def run_fake(self, code, names=('Toy',), timeout=4):
        manifest = self.manifest(names)
        fake = self.fake(code)
        # Tests replace only executable binding, retaining the real sysroot:
        manifest['lean']['path'] = str(fake)
        manifest['lean']['sha256'] = self.kr.sha256(fake)
        return self.kr.run_manifest(manifest, 256, timeout)

    def test_empty_coverage_rejected(self):
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(self.base)], [], [])

    def test_duplicate_modules_rejected(self):
        with self.assertRaises(ValueError):
            self.manifest(('Toy', 'Toy'))

    def test_missing_module_rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.kr.prepare_manifest(str(LEAN), [str(self.base)], [f'Missing={self.source}'], [])

    def test_individual_olean_symlink_rejected(self):
        linked = self.base / 'linked'
        linked.mkdir()
        (linked / 'Toy.olean').symlink_to(self.base / 'Toy.olean')
        # Upstream selects lexical siblings. Canonicalizing a file symlink
        # could bind destination private parts while replaying other parts.
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(linked)], [f'Toy={self.source}'], [])

    def test_search_root_and_part_symlinks_rejected(self):
        linked = self.base / 'linked-root'
        linked.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(linked)], [f'Toy={self.source}'], [])
        part = self.base / 'Toy.olean.server'
        part.symlink_to(self.base / 'Toy.olean')
        with self.assertRaises(ValueError):
            self.manifest()

    def test_expected_coverage_mismatch_rejected(self):
        m = self.manifest()
        m['expected_modules'].append('Missing')
        self.assertFalse(self.kr.run_manifest(m, 256, 4)['ok'])

    def test_source_change_rejected(self):
        m = self.manifest()
        self.source.write_text('changed')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('binding changed', r.get('error', ''))

    def test_part_change_rejected(self):
        m = self.manifest()
        (self.base / 'Toy.olean').write_bytes(b'changed')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('compiled parts changed', r.get('error', ''))

    def test_new_private_part_rejected(self):
        m = self.manifest()
        (self.base / 'Toy.olean.server').write_bytes(b'added')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('compiled parts changed', r.get('error', ''))

    def test_tool_binding_change_rejected(self):
        m = self.manifest()
        m['tool'][0]['sha256'] = '0' * 64
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('tool binding changed', r.get('error', ''))

    def test_nonzero_exit_fatal_even_with_marker(self):
        r = self.run_fake("import os\nprint('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\nraise SystemExit(7)\n")
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'][0]['returncode'], 7)

    def test_missing_or_wrong_marker_fatal(self):
        for code in ("print('no replay')\n", "print('CBZ_KERNEL_REPLAY_OK Other')\n"):
            with self.subTest(code=code):
                self.assertFalse(self.run_fake(code)['ok'])

    def test_timeout_fatal(self):
        r = self.run_fake('import time\ntime.sleep(5)\n', timeout=0.1)
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'][0]['status'], 'TIMEOUT')

    def test_memory_limit_applied(self):
        r = self.run_fake("import json, os, resource\nprint(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))\nprint('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\n")
        self.assertTrue(r['ok'], r)
        self.assertEqual(json.loads(r['results'][0]['stdout'].splitlines()[0]), [268435456, 268435456])

    def test_one_child_at_a_time_and_exact_coverage(self):
        lock = self.base / 'active.lock'
        code = ("import os, time\nfrom pathlib import Path\n"
            + f"p = Path({str(lock)!r})\n"
            + "fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n"
            + "time.sleep(0.05)\nos.close(fd)\np.unlink()\n"
            + "print('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\n")
        r = self.run_fake(code, ('Toy', 'Other'))
        self.assertTrue(r['ok'], r)
        self.assertEqual([x['module'] for x in r['results']], ['Toy', 'Other'])
        self.assertEqual(r['completed_count'], 2)


    def resource_report(self, m, memory=256, timeout=4):
        # Turn escaped production exceptions into assertion failures, so mutation
        # accounting never confuses a harness error with a regression kill.
        try:
            return self.kr.run_manifest(m, memory, timeout)
        except Exception as exc:
            self.fail(f'resource boundary escaped structured report: {type(exc).__name__}: {exc}')

    def test_resource_memory_upper_bounds_before_spawn(self):
        m = self.manifest()
        for memory in (10**400, sys.maxsize // (1024*1024) + 1):
            with self.subTest(memory_digits=len(str(memory))):
                with mock.patch.object(self.kr, 'run_one', return_value={'ok': True, 'module': 'Toy'}):
                    r = self.resource_report(m, memory=memory)
                self.assertFalse(r['ok'])
                self.assertEqual(r['results'], [])
                self.assertIn('memory_mib', r.get('error', ''))

    def test_resource_memory_hard_limit_before_spawn(self):
        m = self.manifest()
        with mock.patch.object(self.kr.resource, 'getrlimit', return_value=(128*1024*1024, 128*1024*1024)):
            with mock.patch.object(self.kr, 'run_one', return_value={'ok': True, 'module': 'Toy'}):
                r = self.resource_report(m, memory=256)
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'], [])
        self.assertIn('memory_mib', r.get('error', ''))

    def test_resource_boolean_arguments_rejected(self):
        m = self.manifest()
        for memory, timeout in ((True, 4), (256, True)):
            with self.subTest(memory=memory, timeout=timeout):
                with mock.patch.object(self.kr, 'run_one', return_value={'ok': True, 'module': 'Toy'}):
                    r = self.resource_report(m, memory, timeout)
                self.assertFalse(r['ok'])
                self.assertEqual(r['results'], [])

    def test_resource_timeout_upper_bounds_before_spawn(self):
        m = self.manifest()
        for timeout in (10**400, 1e100, 2147484, float('inf'), float('nan')):
            with self.subTest(timeout_type=type(timeout).__name__):
                with mock.patch.object(self.kr, 'run_one', return_value={'ok': True, 'module': 'Toy'}):
                    r = self.resource_report(m, timeout=timeout)
                self.assertFalse(r['ok'])
                self.assertEqual(r['results'], [])
                self.assertIn('timeout_seconds', r.get('error', ''))

    def test_resource_valid_integer_and_float_timeouts(self):
        code = "import os\nprint('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\n"
        for timeout in (4, 4.5):
            with self.subTest(timeout=timeout):
                self.assertTrue(self.run_fake(code, timeout=timeout)['ok'])

    def test_resource_spawn_exceptions_are_structured_errors(self):
        m = self.manifest()
        for exception in (subprocess.SubprocessError, OverflowError, ValueError, OSError):
            with self.subTest(exception=exception.__name__):
                with mock.patch.object(self.kr.subprocess, 'Popen', side_effect=exception('spawn rejected')):
                    r = self.resource_report(m)
                self.assertFalse(r['ok'])
                self.assertEqual(len(r['results']), 1, r)
                self.assertEqual(r['results'][0]['status'], 'ERROR')
                self.assertIn(exception.__name__, r['results'][0].get('error', ''))

    def test_resource_real_preexec_failure_is_structured(self):
        m = self.manifest()
        fake = self.fake("print('must not execute')\n")
        m['lean'] = self.kr.binding(fake)
        def reject_limit():
            raise ValueError('injected setrlimit failure')
        with mock.patch.object(self.kr, 'limits', return_value=reject_limit):
            r = self.resource_report(m)
        self.assertFalse(r['ok'])
        self.assertEqual(len(r['results']), 1, r)
        self.assertEqual(r['results'][0]['status'], 'ERROR')
        self.assertIn('SubprocessError', r['results'][0].get('error', ''))
        self.assertNotIn('must not execute', r['results'][0]['stdout'])

    def test_resource_wait_errors_cannot_become_success(self):
        m = self.manifest()
        for exception in (OverflowError, ValueError, subprocess.SubprocessError, OSError):
            with self.subTest(exception=exception.__name__):
                child = mock.Mock(pid=12345, returncode=0)
                child.__enter__ = mock.Mock(return_value=child)
                child.__exit__ = mock.Mock(return_value=False)
                child.wait.side_effect = [exception('wait rejected'), 0]
                child.poll.return_value = 0
                def spawn(*args, **kwargs):
                    kwargs['stdout'].write(b'CBZ_KERNEL_REPLAY_OK Toy\n')
                    return child
                with mock.patch.object(self.kr.subprocess, 'Popen', side_effect=spawn), mock.patch.object(self.kr.os, 'killpg'):
                    r = self.resource_report(m)
                self.assertFalse(r['ok'])
                self.assertEqual(len(r['results']), 1, r)
                self.assertEqual(r['results'][0]['status'], 'ERROR')
                self.assertIn(exception.__name__, r['results'][0].get('error', ''))
                self.assertEqual(r['results'][0]['returncode'], 0)
                self.assertFalse(child.__exit__.called, 'implicit Popen cleanup must not run')

    def test_resource_wait_error_kills_and_reaps_real_child(self):
        m = self.manifest()
        fake = self.fake('import time\ntime.sleep(0.2)\n')
        m['lean'] = self.kr.binding(fake)
        actual_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            children.append(child)
            actual_wait = child.wait
            first = True
            def wait(timeout=None):
                nonlocal first
                if first:
                    first = False
                    raise OverflowError('injected platform wait overflow')
                return actual_wait(timeout=timeout)
            child.wait = wait
            return child
        with mock.patch.object(self.kr.subprocess, 'Popen', side_effect=spawn):
            r = self.resource_report(m)
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'][0]['status'], 'ERROR')
        self.assertEqual(children[0].poll(), -signal.SIGKILL)
        self.assertTrue(r['results'][0]['cleanup']['reaped'])

    def test_resource_timeout_kills_and_reaps_real_child(self):
        started = time.monotonic()
        r = self.run_fake('import time\ntime.sleep(30)\n', timeout=0.05)
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(r['ok'])
        result = r['results'][0]
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertEqual(result['returncode'], -signal.SIGKILL)
        self.assertTrue(result.get('cleanup', {}).get('reaped'))
        with self.assertRaises(ProcessLookupError):
            os.kill(result['pid'], 0)

    def test_resource_unreapable_child_cleanup_is_bounded(self):
        m = self.manifest()
        child = mock.Mock(pid=12345, returncode=None)
        child.__enter__ = mock.Mock(return_value=child)
        child.__exit__ = mock.Mock(return_value=False)
        waits = []
        def wait(timeout=None):
            waits.append(timeout)
            if timeout is None:
                self.fail('unbounded child cleanup wait')
            raise subprocess.TimeoutExpired('fake-child', timeout)
        child.wait.side_effect = wait
        child.poll.return_value = None
        child.kill.side_effect = PermissionError('kill denied')
        started = time.monotonic()
        with mock.patch.object(self.kr.subprocess, 'Popen', return_value=child), mock.patch.object(self.kr.os, 'killpg', side_effect=PermissionError('group kill denied')):
            r = self.resource_report(m, timeout=0.05)
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(r['ok'])
        result = r['results'][0]
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['pid'], 12345)
        self.assertFalse(result['cleanup']['reaped'])
        self.assertIn('not reaped', result.get('error', ''))
        self.assertTrue(all(t is not None and 0 < t <= 1 for t in waits))
        self.assertFalse(child.__exit__.called, 'implicit Popen cleanup must not run')

    def test_resource_unreaped_child_stops_further_modules(self):
        m = self.manifest(('Toy', 'Other'))
        child = mock.Mock(pid=12345, returncode=None)
        child.__enter__ = mock.Mock(return_value=child)
        child.__exit__ = mock.Mock(return_value=False)
        child.wait.side_effect = subprocess.TimeoutExpired('fake-child', 0.05)
        child.poll.return_value = None
        with mock.patch.object(self.kr.subprocess, 'Popen', return_value=child), mock.patch.object(self.kr.os, 'killpg'):
            r = self.resource_report(m, timeout=0.05)
        self.assertFalse(r['ok'])
        self.assertEqual(len(r['results']), 1, 'must stop serial scheduling when child cleanup failed')
        self.assertEqual(r['expected_count'], 2)
        self.assertEqual(r['results'][0]['status'], 'ERROR')

    def test_resource_cleanup_exceptions_remain_structured(self):
        m = self.manifest()
        for exception in (OverflowError, ValueError, subprocess.SubprocessError, OSError):
            with self.subTest(exception=exception.__name__):
                child = mock.Mock(pid=12345, returncode=None)
                child.__enter__ = mock.Mock(return_value=child)
                child.__exit__ = mock.Mock(return_value=False)
                child.wait.side_effect = [subprocess.TimeoutExpired('fake-child', 0.05), exception('cleanup rejected')]
                child.poll.return_value = None
                with mock.patch.object(self.kr.subprocess, 'Popen', return_value=child), mock.patch.object(self.kr.os, 'killpg'):
                    r = self.resource_report(m, timeout=0.05)
                self.assertFalse(r['ok'])
                self.assertEqual(len(r['results']), 1, r)
                self.assertEqual(r['results'][0]['status'], 'ERROR')
                self.assertIn(exception.__name__, str(r['results'][0].get('cleanup', {})))

    def test_resource_oversized_cli_arguments_emit_non_green_report(self):
        manifest = self.base / 'manifest.json'
        manifest.write_text(json.dumps(self.manifest()))
        output = self.base / 'report.json'
        for memory, timeout in (('1208925819614629174706176', '4'), ('256', '1e100')):
            with self.subTest(memory=memory, timeout=timeout):
                proc = subprocess.run([sys.executable, str(ROOT / 'kernel_replay.py'), 'run',
                    str(manifest), '--memory-mib', memory, '--timeout-seconds', timeout,
                    '--output', str(output)], capture_output=True, text=True, timeout=5)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                value = json.loads(output.read_text())
                self.assertFalse(value['ok'])
                self.assertEqual(value['results'], [])
                self.assertIn('error', value)
                self.assertEqual(value['g2_status'], 'NOT_ASSESSED')

    def test_strict_cli(self):
        for args in (['--unknown'], ['run', '--help', '--bogus'], ['one', 'Toy', 'Other']):
            p = subprocess.run([sys.executable, str(ROOT / 'kernel_replay.py'), *args],
                capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_module_names_and_unknown_fields_rejected(self):
        for name in ('', '-v', 'Toy.*', 'Toy..Other', 'Toy/Other'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.kr.prepare_manifest(str(LEAN), [str(self.base)], [f'{name}={self.source}'], [])
        m = self.manifest()
        m['ignored_flag'] = True
        self.assertFalse(self.kr.run_manifest(m, 256, 4)['ok'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
