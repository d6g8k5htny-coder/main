"""Finite process/scheduling guards for the output-only Math execution adapter.

Imports no Math/project code and runs only disposable stdlib child processes.
These tests do not prove a theorem or establish whole-federation CBZ compliance.
"""
import ast
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'run_math_sequential_gate.py'


def functions(source=None):
    tree = ast.parse(DRIVER.read_text() if source is None else source)
    selected = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {'__name__': 'cbz_driver_fixture'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DRIVER), 'exec'), ns)
    return ns


class DriverGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cbz-math-driver-test-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.ns = functions()
        self.ns.update(OUT=self.out, ENV={'PATH':os.defpath,'LANG':'C.UTF-8'}, RSS_LIMIT=128 * 1024**2,
                       TIME_LIMIT=5, records=[], R=self.out, MODULES=['Root.A', 'Root.B', 'Root'],
                       TEMPLATE='replayFromImports `CBZ_EXACT_MODULE\n',
                       T=self.out/'toolchain/bin', LAKE_ENVIRONMENTS={})

    def lake_environment(self):
        return dict(LAKE=str(self.ns['T']/'lake'), LAKE_HOME=str(self.ns['T'].parent),
                    LEAN_SYSROOT=str(self.ns['T'].parent), LEAN_AR='ar', LEAN_CC=None,
                    LEAN_PATH='exact-lean-path', LEAN_SRC_PATH='exact-source-path',
                    PATH='exact-path', DYLD_LIBRARY_PATH=None, LD_LIBRARY_PATH='exact-library-path')

    def test_lake_environment_launch_preserves_pinned_binary_args_and_values(self):
        self.assertTrue('lean_launch' in self.ns,'documented Lake environment launcher is missing')
        captured=self.lake_environment();self.ns['ENV']={'PRESERVE_GUARD':'keep','LEAN_CC':'old'}
        calls=[]
        self.ns['run_process']=lambda args,label,cwd:(calls.append((args,label,cwd))or(0,self.ns['json'].dumps(captured)))
        command,env,fields=self.ns['lean_launch'](['lake','env','lean','-o','result.olean','source.lean'],'phase',self.out)
        self.assertEqual(command,[str(self.ns['T']/'lean'),'-o','result.olean','source.lean'])
        self.assertEqual(fields,captured)
        self.assertEqual(env['LEAN_PATH'],'exact-lean-path')
        self.assertEqual(env['LD_LIBRARY_PATH'],'exact-library-path')
        self.assertEqual(env['PRESERVE_GUARD'],'keep')
        self.assertNotIn('LEAN_CC',env)
        self.assertEqual(calls[0][0][:2],[str(self.ns['T']/'lake'),'env'])

    def test_lake_environment_unknown_fields_fail_closed(self):
        captured=self.lake_environment();captured['UNEXPECTED_FIELD']='inert-fixture'
        self.ns['run_process']=lambda *args:(0,self.ns['json'].dumps(captured))
        with self.assertRaisesRegex(RuntimeError,'environment field set'):
            self.ns['lean_launch'](['lake','env','lean','source.lean'],'phase',self.out)

    def test_lake_environment_duplicate_keys_fail_closed(self):
        captured=self.lake_environment();raw=self.ns['json'].dumps(captured)
        duplicate='{"LAKE":'+self.ns['json'].dumps(captured['LAKE'])+','+raw[1:]
        self.ns['run_process']=lambda *args:(0,duplicate)
        with self.assertRaisesRegex(RuntimeError,'duplicate Lake environment key'):
            self.ns['lean_launch'](['lake','env','lean','source.lean'],'phase',self.out)

    def test_lake_environment_capture_failure_cannot_launch(self):
        self.ns['run_process']=lambda *args:(1,self.ns['json'].dumps(self.lake_environment()))
        with self.assertRaisesRegex(RuntimeError,'Lake environment capture failed'):
            self.ns['lean_launch'](['lake','env','lean','source.lean'],'phase',self.out)

    def test_lake_environment_sysroot_must_match_pinned_toolchain(self):
        captured=self.lake_environment();captured['LEAN_SYSROOT']='/inert-other-toolchain'
        self.ns['run_process']=lambda *args:(0,self.ns['json'].dumps(captured))
        with self.assertRaisesRegex(RuntimeError,'Lake environment toolchain'):
            self.ns['lean_launch'](['lake','env','lean','source.lean'],'phase',self.out)

    def test_lake_environment_capture_cached_by_working_directory(self):
        calls=[]
        self.ns['run_process']=lambda *args:(calls.append(args)or(0,self.ns['json'].dumps(self.lake_environment())))
        self.ns['lean_launch'](['lake','env','lean','one.lean'],'one',self.out)
        self.ns['lean_launch'](['lake','env','lean','two.lean'],'two',self.out)
        self.assertEqual(len(calls),1)
        self.ns['lean_launch'](['lake','env','lean','three.lean'],'three',self.out/'other')
        self.assertEqual(len(calls),2)

    def test_non_lean_launch_is_unchanged(self):
        self.ns['run_process']=lambda *args:self.fail('non-Lean launch must not capture an environment')
        command,env,fields=self.ns['lean_launch'](['lake','build'],'build',self.out)
        self.assertEqual(command,['lake','build'])
        self.assertEqual(env,self.ns['ENV'])
        self.assertIsNone(fields)

    def test_success_is_observed_and_log_hash_bound(self):
        code, log = self.ns['run_process']([sys.executable, '-c', 'print("actual-positive")'], 'positive', self.out)
        self.assertEqual(code, 0)
        self.assertEqual(log, 'actual-positive\n')
        row = self.ns['records'][0]
        self.assertIsNone(row['stop_reason'])
        self.assertEqual(row['exit_code'], 0)
        self.assertEqual(row['log_sha256'], self.ns['hashlib'].sha256(log.encode()).hexdigest())
        self.assertTrue((self.out / 'commands.json').is_file())

    def test_failure_exit_is_retained(self):
        code, log = self.ns['run_process']([sys.executable, '-c', 'print("actual-negative");raise SystemExit(7)'], 'negative', self.out)
        self.assertEqual(code, 7)
        self.assertEqual(log, 'actual-negative\n')
        self.assertEqual(self.ns['records'][0]['exit_code'], 7)

    def test_monitor_error_cleans_direct_child_before_reraise(self):
        real_popen = subprocess.Popen
        children = []
        def capture(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            if kwargs.get('start_new_session'):
                children.append(child)
            return child
        def failed_monitor(*args):
            raise OSError('planted monitor failure')
        self.ns['group_usage'] = failed_monitor
        caught = None
        try:
            with mock.patch.object(subprocess, 'Popen', capture):
                try:
                    self.ns['run_process']([sys.executable, '-c', 'import time;time.sleep(20)'], 'monitor-failure', self.out)
                except BaseException as exc:
                    caught = exc
            self.assertIsInstance(caught, OSError)
            self.assertIn('planted monitor failure', str(caught))
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll(), 'owned child survived failed process monitor')
        finally:
            for child in children:
                try:
                    os.killpg(child.pid, self.ns['signal'].SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)

    def test_refused_final_seal_retires_inner_success_receipt(self):
        live = self.out / '.lake/formal-evidence'
        live.mkdir(parents=True)
        result = {'status': 'RUNNING'}
        class FakeGate:
            def execute(inner, *args):
                value = {'formalization_status': 'kernel-checked', 'checked_commit': 'fixture-only'}
                (live / 'receipt.json').write_text(self.ns['json'].dumps(value))
                return value
        tree = ast.parse(DRIVER.read_text())
        final = [n for n in tree.body if isinstance(n, ast.Try)][-1]
        self.ns.update(gate=FakeGate(), m={}, digest='fixture',
                       hashes=lambda root: {'changed': 'new'}, before={'changed': 'old'},
                       result=result, records=[])
        exec(compile(ast.Module(body=[final], type_ignores=[]), str(DRIVER), 'exec'), self.ns)
        self.assertEqual(result['status'], 'INCOMPLETE_OR_FAILED')
        self.assertFalse((live / 'receipt.json').exists(), 'refused outer seal left an inner success receipt live')
        preserved = self.out / 'refused-inner-receipt.json'
        self.assertTrue(preserved.is_file(), 'refused evidence must be preserved')
        self.assertEqual(self.ns['json'].loads(preserved.read_text())['checked_commit'], 'fixture-only')

    def test_timeout_refuses_and_records_only_own_process_group(self):
        self.ns['TIME_LIMIT'] = 0
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'], start_new_session=True)
        try:
            with self.assertRaisesRegex(RuntimeError, 'COMMAND_TIMEOUT'):
                self.ns['run_process']([sys.executable, '-c', 'import time;time.sleep(20)'], 'timeout', self.out)
            self.assertIsNone(sentinel.poll(), 'unrelated owned sentinel must survive another group cleanup')
            self.assertEqual(self.ns['records'][0]['stop_reason'], 'COMMAND_TIMEOUT')
            self.assertNotEqual(self.ns['records'][0]['exit_code'], 0)
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)

    def test_direct_child_exit_with_live_descendant_is_refused_and_cleaned(self):
        original_popen = subprocess.Popen
        groups = []
        def capture(*args, **kwargs):
            proc = original_popen(*args, **kwargs)
            if kwargs.get('start_new_session'):
                groups.append(proc.pid)
            return proc
        script = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]);print(p.pid,flush=True)'
        try:
            with mock.patch.object(subprocess, 'Popen', capture):
                with self.assertRaisesRegex(RuntimeError, 'DESCENDANTS_AFTER_EXIT'):
                    self.ns['run_process']([sys.executable, '-c', script], 'orphan', self.out)
            self.assertEqual(self.ns['records'][0]['stop_reason'], 'DESCENDANTS_AFTER_EXIT')
            self.assertEqual(self.ns['records'][0]['process_group_id'], groups[0])
        finally:
            for group in groups:
                try:
                    os.killpg(group, self.ns['signal'].SIGKILL)
                except ProcessLookupError:
                    pass

    def test_term_resistant_direct_child_requires_kill_escalation(self):
        self.ns['TIME_LIMIT'] = 0.3
        original_popen = subprocess.Popen
        groups = []
        def capture(*args, **kwargs):
            proc = original_popen(*args, **kwargs)
            if kwargs.get('start_new_session'):
                groups.append(proc.pid)
            return proc
        script = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'
        try:
            with mock.patch.object(subprocess, 'Popen', capture):
                with self.assertRaisesRegex(RuntimeError, 'COMMAND_TIMEOUT'):
                    self.ns['run_process']([sys.executable, '-c', script], 'term-resistant', self.out)
            self.assertEqual(self.ns['records'][0]['exit_code'], -self.ns['signal'].SIGKILL)
        finally:
            for group in groups:
                try:
                    os.killpg(group, self.ns['signal'].SIGKILL)
                except ProcessLookupError:
                    pass

    def test_rss_guard_is_observed(self):
        self.ns['RSS_LIMIT'] = 16 * 1024**2
        with self.assertRaisesRegex(RuntimeError, 'RSS_LIMIT'):
            self.ns['run_process']([sys.executable, '-c', 'import time;x=bytearray(64*1024**2);time.sleep(20)'], 'rss', self.out)
        row = self.ns['records'][0]
        self.assertEqual(row['stop_reason'], 'RSS_LIMIT')
        self.assertGreater(row['peak_group_rss_bytes'], 16 * 1024**2)
        self.assertNotEqual(row['exit_code'], 0)

    def test_build_schedules_every_registered_module_and_default(self):
        calls = []
        def run(args, label, cwd):
            calls.append((args, label, cwd))
            return 0, label
        self.ns['run_process'] = run
        self.ns['sequential_run'](['lake', 'build'], 'build', self.out)
        self.assertEqual([x[0] for x in calls], [
            ['lake', 'build', '+Root.A'], ['lake', 'build', '+Root.B'],
            ['lake', 'build', '+Root'], ['lake', 'build']])
        self.assertTrue((self.out / 'build.log').is_file())

    def test_exact_kernel_schedules_every_module_without_prefix_cli(self):
        calls = []
        def replay(module, label, cwd):
            calls.append((module, label, cwd))
            return 0, module
        self.ns['exact_replay'] = replay
        self.ns['sequential_run'](['lake', 'env', 'leanchecker', 'ResearchFormalCoreR1'], 'kernel', self.out)
        self.assertEqual([x[0] for x in calls], ['Root.A', 'Root.B', 'Root'])
        self.assertTrue((self.out / 'kernel.log').is_file())

    def test_failed_exact_kernel_cannot_be_passing(self):
        self.ns['exact_replay'] = lambda *args: (1, 'actual-negative')
        with self.assertRaisesRegex(RuntimeError, 'kernel replay failed'):
            self.ns['sequential_run'](['lake', 'env', 'leanchecker', 'ResearchFormalCoreR1'], 'kernel', self.out)

    def test_failed_module_build_cannot_be_passing(self):
        self.ns['run_process'] = lambda *args: (1, 'actual-negative')
        with self.assertRaisesRegex(RuntimeError, 'module build failed'):
            self.ns['sequential_run'](['lake', 'build'], 'build', self.out)

    def test_original_negative_phase_expectation_is_preserved(self):
        self.ns['run_process'] = lambda *args: (1, 'expected original negative')
        text = self.ns['sequential_run'](['lake', 'env', 'lean', 'negative.lean'], 'negative', self.out, False)
        self.assertIn('expected original negative', text)
        with self.assertRaisesRegex(RuntimeError, 'unexpected original gate phase outcome'):
            self.ns['sequential_run'](['lake', 'env', 'lean', 'negative.lean'], 'negative', self.out, True)

    def test_exact_module_name_validation_and_source(self):
        calls = []
        self.ns['run_process'] = lambda args, label, cwd: (calls.append(args) or (0, 'ok'))
        for name in ['Root.A; IO.println "bad"', '', '`Root', 'Root A', 'Root/A']:
            with self.assertRaisesRegex(RuntimeError, 'unsafe module name'):
                self.ns['exact_replay'](name, 'unsafe', self.out)
        self.assertEqual(calls, [])
        self.ns['exact_replay']('Root.A', 'exact', self.out)
        self.assertEqual((self.out / 'exact.lean').read_text(), 'replayFromImports `Root.A\n')
        self.assertEqual(calls, [['lake', 'env', 'lean', str(self.out / 'exact.lean')]])


if __name__ == '__main__':
    unittest.main()
