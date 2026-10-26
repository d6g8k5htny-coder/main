"""Durable Source-only actual-wrapper regressions: all results NOT_RUN here.

Exercises the exact PYNODE body extracted from the supplied workflow, unmodified.
Real stdlib Python children stand in for Node; no project/Node/Git helper runs.
The wrapper's actual source inventory, physical cap, cleanup and context code run.
Fault adapters are at OS/file/process boundaries and retain real cleanup effects.
"""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / 'tests/fixtures/site_transition_capture_regression_20261025'
BASELINE_WORKFLOW = HARNESS_DIR / 'baseline-current-site-transition-candidate.yml.txt'
DEFAULT_WORKFLOW = PROJECT_ROOT / '.github/workflows/current-site-transition-candidate.yml'
WORKFLOW = Path(os.environ.get('INLINE_WRAPPER_WORKFLOW', str(DEFAULT_WORKFLOW)))
CAPTURE_OBSERVER = globals().get('__capture_observer__')
CAPTURE_CONTRACT = globals().get('__capture_contract__')
REAL_POPEN = subprocess.Popen
REAL_KILLPG = os.killpg
REAL_OS_OPEN = os.open
REAL_WRITE_BYTES = Path.write_bytes
REAL_WRITE_TEXT = Path.write_text
REAL_MONOTONIC = time.monotonic


def blocks(workflow=WORKFLOW):
    raw = workflow.read_text()
    # The durable workflow also contains the independent regression job. Only
    # the actual core job supplies these four code-under-test stdin bodies.
    jobs = list(re.finditer(r'^  frozen-reader-transition-candidate:\n(.*?)(?=^  [A-Za-z0-9_-]+:|\Z)',
                           raw, re.M | re.S))
    if len(jobs) != 1:
        raise AssertionError('Expected one actual frozen core job')
    result = {}
    for match in re.finditer(r"^          (python3[^\n]+) <<'(PY[A-Z]+)'\n(.*?)^          \2$", jobs[0].group(1), re.M | re.S):
        result[match.group(2)] = (shlex.split(match.group(1)), textwrap.dedent(match.group(3)))
    if set(result) != {'PYCONTEXT', 'PYGUARD', 'PYNODE', 'PYEMIT'}:
        raise AssertionError('Expected all four real workflow stdin bodies')
    return result



def case_id(test):
    identity = test.id()
    prefix = '__main__.'
    if not identity.startswith(prefix):
        raise ValueError('HOLD_RUNTIME_CASE_ID')
    return identity[len(prefix):]


def compact_state(context):
    if type(context) is not dict:
        raise ValueError('HOLD_CONTEXT_OBSERVATION_TYPE')
    return {name: context.get(name) for name in (
        'node_phase', 'capture_status', 'immutable_source_recheck',
        'node_returncode', 'capture_cleanup_diagnostics')}


def emit_observation(kind, **values):
    if not callable(CAPTURE_OBSERVER):
        raise ValueError('HOLD_AUTHENTICATED_LAUNCHER_REQUIRED')
    # The authenticated launcher enforces the complete event schema/4096-byte
    # row and 64KiB stream bounds. No expected outcome is inserted here.
    CAPTURE_OBSERVER(kind, **values)


class InlineCaptureTests(unittest.TestCase):
    def run_case(self, scenario):
        # Independent hand-built source ledger; the production recheck must read it.
        with tempfile.TemporaryDirectory(prefix='inline-wrapper-focused-') as temporary:
            root = Path(temporary)
            output = root / 'runner/current-site-transition-candidate'
            output.mkdir(parents=True)
            entries = json.loads((HARNESS_DIR / 'ledger.json').read_bytes())
            self.assertEqual(len(entries), 16)
            for entry in entries:
                path = root / entry['path']
                path.parent.mkdir(exist_ok=True)
                data = (HARNESS_DIR / entry['path']).read_bytes()
                self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (entry['bytes'], entry['sha256']))
                path.write_bytes(data)
            initial = {
                'schema_version': 1, 'scientific_effect': 'NONE',
                'candidate_source_guard': 'passed', 'node_phase': 'not_started',
                'immutable_source_recheck': 'pending', 'immutable_source_inputs': entries,
                'node_version': {'status': 'OK'}, 'python_command_version': {'status': 'OK'},
                'event_name': 'pull_request', 'pull_request_head_sha': '1' * 40,
                'pull_request_base_sha': '2' * 40,
            }
            context_path = output / 'context.json'
            context_path.write_text(json.dumps(initial))
            for name in ['tap.stdout.tap', 'tap.stderr.txt']:
                (output / name).write_bytes(b'')
            (output / 'node-exit-code.txt').write_text('NOT_RUN\n')
            source = blocks()['PYNODE'][1]
            # Only parse/compile the acquired actual body; no copied wrapper algorithm.
            code = compile(ast.parse(source, filename=str(WORKFLOW) + ':PYNODE'), str(WORKFLOW) + ':PYNODE', 'exec')
            children, source_opens, save_attempts = [], [], []
            launch_attempts, source_phase_counts = [0], [0, 0]
            child_name = {'nonzero': 'nonzero', 'timeout': 'timeout',
                'outputcap': 'outputcap', 'missingcapture': 'missingcapture',
                'sourcechange': 'sourcechange', 'capturefifo': 'capturefifo',
                'capturedirectory': 'capturedirectory', 'capturesymlink': 'capturesymlink',
                'capturehardlink': 'capturehardlink', 'sourcefifo': 'sourcefifo'}.get(scenario, 'success')
            child_script = (HARNESS_DIR / 'children' / (child_name + '.py')).read_text()

            def launch(command, **kwargs):
                launch_attempts[0] += 1
                if command != ['node', '--test', '--test-reporter=tap', 'tests/test_current_site_transition.mjs']:
                    raise AssertionError('Unexpected production child command')
                if scenario == 'setupfailure':
                    raise OSError('injected spawn failure')
                child = REAL_POPEN([sys.executable, '-I', '-B', '-S', '-c', child_script], **kwargs)
                children.append(child)
                if scenario == 'reapfailure':
                    real_wait = child.wait
                    def failed_wait(timeout=None):
                        real_wait(timeout=timeout)
                        raise subprocess.TimeoutExpired(command, timeout)
                    child.wait = failed_wait
                return child

            def kill_group(pid, sig):
                try:
                    REAL_KILLPG(pid, sig)
                except ProcessLookupError:
                    if scenario not in ['killfailure', 'cleanupinterrupt']:
                        raise
                if scenario == 'killfailure':
                    raise PermissionError('injected kill diagnostic after actual cleanup')
                if scenario == 'cleanupinterrupt':
                    raise KeyboardInterrupt('injected interruption after actual cleanup')

            def open_source(path, *args, **kwargs):
                if scenario == 'unreadablecapture' and Path(path).name == 'tap.stderr.txt' and args and args[0] & os.O_NONBLOCK:
                    raise PermissionError('injected unreadable descriptor capture')
                if path == '.':
                    source_opens.append(path)
                    source_phase_counts[bool(launch_attempts[0])] += 1
                return REAL_OS_OPEN(path, *args, **kwargs)

            def save_bytes(path, data):
                if path == context_path:
                    save_attempts.append(json.loads(data))
                    if scenario == 'contextsavefailure' and len(save_attempts) == 2:
                        raise PermissionError('injected final context write failure')
                return REAL_WRITE_BYTES(path, data)

            def write_text(path, data, *args, **kwargs):
                if scenario == 'exitwritefailure' and path.name == 'node-exit-code.txt':
                    raise PermissionError('injected integer exit-file write failure')
                return REAL_WRITE_TEXT(path, data, *args, **kwargs)

            clock_calls = [0]
            def monotonic():
                if scenario == 'timeout':
                    clock_calls[0] += 1
                    return 100.0 * (clock_calls[0] - 1)
                return REAL_MONOTONIC()

            previous = Path.cwd()
            caught = None
            began = REAL_MONOTONIC()
            try:
                os.chdir(root)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, {'RUNNER_TEMP': str(root / 'runner')}))
                    stack.enter_context(patch('subprocess.Popen', launch))
                    stack.enter_context(patch('os.killpg', kill_group))
                    stack.enter_context(patch('os.open', open_source))
                    stack.enter_context(patch.object(Path, 'write_bytes', save_bytes))
                    stack.enter_context(patch.object(Path, 'write_text', write_text))
                    stack.enter_context(patch('time.monotonic', monotonic))
                    try:
                        exec(code, {'__name__': '__main__'})
                    except BaseException as error:
                        caught = error
            finally:
                os.chdir(previous)
                # Harness-only safety cleanup still delegates to actual OS/process methods.
                for child in children:
                    try: REAL_KILLPG(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    subprocess.Popen.wait(child, timeout=5)
            # Observe the actual failed/successful wrapper path BEFORE assertions.
            # This records attempted writes separately from physically stored context.
            context_read_error, witness_stored = None, None
            try:
                witness_stored = json.loads(context_path.read_bytes())
            except Exception as error:
                context_read_error = type(error).__name__
            attempt = save_attempts[-1] if save_attempts else {}
            emit_observation('CASE_WITNESS', case_id=case_id(self), scenario=scenario,
                caught_type=type(caught).__name__ if caught is not None else None,
                caught_exit=caught.code if isinstance(caught, SystemExit) and type(caught.code) is int else None,
                pre_source_opens=source_phase_counts[0], post_source_opens=source_phase_counts[1],
                total_source_opens=len(source_opens),
                save_attempts=[compact_state(value) for value in save_attempts],
                stored=compact_state(witness_stored) if witness_stored is not None else None,
                capture_errors=attempt.get('capture_errors', []),
                cleanup_errors=attempt.get('cleanup_errors', []),
                stdout_missing=not (output / 'tap.stdout.tap').exists(),
                child_count=len(children), elapsed_seconds=round(REAL_MONOTONIC() - began, 6),
                context_read_error=context_read_error)
            self.assertLess(REAL_MONOTONIC() - began, 12, 'Focused case exceeded its safety budget')
            self.assertGreaterEqual(len(source_opens), 17, 'Actual post-child source read was skipped')
            self.assertEqual(len(save_attempts), 2, 'Starting and outer-finally context writes must both be attempted')
            final_attempt = save_attempts[-1]
            self.assertIn(final_attempt['immutable_source_recheck'], ['passed', 'failed'])
            stored = json.loads(context_path.read_bytes())
            files = {name: (output / name).read_bytes() for name in ['node-exit-code.txt']}
            return caught, final_attempt, stored, files

    def test_success_preserves_integer_zero_and_complete_capture(self):
        # Break caught: cleanup/custody diagnostics turn a real successful capture into HOLD.
        error, context, _, files = self.run_case('success')
        self.assertIsInstance(error, SystemExit)
        self.assertEqual(error.code, 0)
        self.assertEqual(context['capture_status'], 'COMPLETED')
        self.assertEqual(context['node_returncode'], 0)
        self.assertEqual(files['node-exit-code.txt'], b'0\n')
        self.assertEqual(context['immutable_source_recheck'], 'passed')

    def test_nonzero_retains_child_failure_and_still_attempts_custody(self):
        # Break caught: nonzero child exit is mislabeled as successful test evidence or lost.
        error, context, _, files = self.run_case('nonzero')
        self.assertEqual(error.code, 7)
        self.assertEqual(context['capture_status'], 'COMPLETED')
        self.assertEqual(context['node_returncode'], 7)
        self.assertEqual(files['node-exit-code.txt'], b'7\n')

    def test_timeout_remains_hold_after_outer_custody(self):
        error, context, _, _ = self.run_case('timeout')
        self.assertEqual(error.code, 1)
        self.assertEqual(context['capture_status'], 'TIMEOUT')
        self.assertEqual(context['node_capture_outcome'], 'TIMEOUT')

    def test_output_cap_remains_hold_after_outer_custody(self):
        error, context, _, _ = self.run_case('outputcap')
        self.assertEqual(error.code, 1)
        self.assertEqual(context['capture_status'], 'REFUSED_OUTPUT_LIMIT')

    def test_missing_tap_does_not_skip_source_recheck_or_context_attempt(self):
        # Direct regression for reviewed P2-2: old final stat raises before recheck/save.
        _, context, _, _ = self.run_case('missingcapture')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertTrue(any(row['operation'].startswith('capture:') and row['error'] == 'FileNotFoundError' for row in context['capture_errors']))

    def test_unreadable_tap_is_recorded_and_keeps_hold(self):
        _, context, _, _ = self.run_case('unreadablecapture')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertTrue(any(row['operation'] == 'capture:tap.stderr.txt' and row['error'] == 'PermissionError' for row in context['capture_errors']))

    def test_exit_file_failure_does_not_skip_outer_custody(self):
        _, context, _, files = self.run_case('exitwritefailure')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertEqual(context['node_returncode'], 0)
        self.assertEqual(files['node-exit-code.txt'], b'NOT_RUN\n')
        self.assertIn({'operation': 'exit_write', 'error': 'PermissionError'}, context['capture_errors'])

    def test_kill_error_is_separate_cleanup_hold(self):
        _, context, _, _ = self.run_case('killfailure')
        self.assertEqual(context['capture_status'], 'REFUSED_CLEANUP')
        self.assertIn({'operation': 'killpg', 'error': 'PermissionError'}, context['cleanup_errors'])

    def test_reap_error_is_separate_cleanup_hold(self):
        _, context, _, _ = self.run_case('reapfailure')
        self.assertEqual(context['capture_status'], 'REFUSED_REAP')
        self.assertIn({'operation': 'wait', 'error': 'TimeoutExpired'}, context['cleanup_errors'])

    def test_interrupted_cleanup_never_claims_confirmed_completion(self):
        error, context, _, _ = self.run_case('cleanupinterrupt')
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertEqual(context['capture_status'], 'REFUSED_CLEANUP')
        self.assertEqual(context['capture_cleanup_diagnostics'], 'incomplete')

    def test_source_change_refuses_even_after_completed_child(self):
        _, context, _, _ = self.run_case('sourcechange')
        self.assertEqual(context['capture_status'], 'REFUSED_SOURCE_CHANGED')
        self.assertEqual(context['immutable_source_recheck'], 'failed')
        self.assertEqual(context['node_capture_outcome'], 'COMPLETED')

    def test_spawn_setup_failure_never_becomes_completed(self):
        _, context, _, _ = self.run_case('setupfailure')
        self.assertEqual(context['capture_status'], 'REFUSED_SETUP')
        self.assertEqual(context['node_phase'], 'not_started')
        self.assertIn({'operation': 'capture', 'error': 'OSError'}, context['capture_errors'])

    def test_final_context_write_failure_is_attempted_and_propagated(self):
        error, attempted, stored, _ = self.run_case('contextsavefailure')
        self.assertIsInstance(error, PermissionError)
        self.assertEqual(attempted['immutable_source_recheck'], 'passed')
        self.assertEqual(stored['capture_status'], 'NOT_RUN')
        self.assertEqual(stored['node_phase'], 'starting')

    def test_fifo_capture_refuses_without_blocking_and_attempts_custody(self):
        # Break caught: a replaced FIFO blocks a pathname open before outer-finally custody.
        error, context, _, _ = self.run_case('capturefifo')
        self.assertEqual(error.code, 1)
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertEqual(context['immutable_source_recheck'], 'passed')
        self.assertTrue(any(row['operation'] == 'capture:tap.stdout.tap' for row in context['capture_errors']))

    def test_directory_capture_refuses_and_attempts_custody(self):
        _, context, _, _ = self.run_case('capturedirectory')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertEqual(context['immutable_source_recheck'], 'passed')

    def test_symlink_capture_refuses_and_attempts_custody(self):
        _, context, _, _ = self.run_case('capturesymlink')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertEqual(context['immutable_source_recheck'], 'passed')

    def test_hardlinked_capture_refuses_without_touching_source_ledger(self):
        _, context, _, _ = self.run_case('capturehardlink')
        self.assertEqual(context['capture_status'], 'REFUSED_CAPTURE')
        self.assertEqual(context['immutable_source_recheck'], 'passed')

    def test_source_input_fifo_refuses_without_blocking_and_attempts_context(self):
        # Break caught: recheck_sources blocks opening FIFO before its regular-file fstat.
        error, context, _, _ = self.run_case('sourcefifo')
        self.assertEqual(error.code, 1)
        self.assertEqual(context['capture_status'], 'REFUSED_SOURCE_CHANGED')
        self.assertEqual(context['immutable_source_recheck'], 'failed')
        self.assertEqual(context['immutable_source_recheck_error'], 'ValueError')

    def test_real_stdin_entrypoints_exclude_cwd_and_pythonpath_shadow(self):
        # Break caught: removing isolation admits an actual hostile root/PYTHONPATH import.
        with tempfile.TemporaryDirectory(prefix='inline-isolation-focused-') as temporary:
            root = Path(temporary)
            for module in ['hashlib', 'json', 'pathlib', 'selectors', 'base64']:
                (root / (module + '.py')).write_bytes((HARNESS_DIR / 'shadow' / (module + '.py')).read_bytes())
            probe = (HARNESS_DIR / 'isolation_probe.py').read_text()
            for name, (arguments, _) in blocks().items():
                with self.subTest(block=name):
                    result = subprocess.run(arguments, input=probe.encode(), cwd=root,
                        env={**os.environ, 'PYTHONPATH': str(root)}, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, b'STDLIB_BOUNDARY_OK\n')


    def test_actual_baseline_emitter_imports_shadow_and_candidate_emitter_holds_without_it(self):
        # Causal P2-1 control: actual pre-fix body admits shadow before its qualification;
        # actual isolated candidate reaches its real pending-context HOLD without marker.
        with tempfile.TemporaryDirectory(prefix='inline-import-causal-') as temporary:
            root = Path(temporary)
            output = root / 'runner/current-site-transition-candidate'
            output.mkdir(parents=True)
            (root / 'hashlib.py').write_bytes((HARNESS_DIR / 'shadow/hashlib.py').read_bytes())
            (output / 'context.json').write_text(json.dumps({
                'schema_version': 1, 'candidate_source_guard': 'not_started',
                'composition_binding': 'PENDING_FINAL_CANONICAL_COMPOSITION',
                'node_phase': 'not_started', 'immutable_source_recheck': 'pending'}))
            for name in ['tap.stdout.tap', 'tap.stderr.txt']:
                (output / name).write_bytes(b'')
            (output / 'node-exit-code.txt').write_text('NOT_RUN\n')
            environment = {**os.environ, 'PYTHONPATH': str(root),
                'RUNNER_TEMP': str(root / 'runner'), 'GITHUB_OUTPUT': str(root / 'github-output')}
            old_argv, old_body = blocks(BASELINE_WORKFLOW)['PYEMIT']
            old = subprocess.run(old_argv, input=old_body.encode(), cwd=root,
                env=environment, capture_output=True, timeout=5)
            self.assertEqual(old.returncode, 1)
            self.assertIn(b'unqualified shadow imported', old.stderr)
            marker = root / '.import-shadow-marker'
            self.assertEqual(marker.read_bytes(), b'SHADOW_IMPORT\n')
            marker.unlink()
            new_argv, new_body = blocks()['PYEMIT']
            new = subprocess.run(new_argv, input=new_body.encode(), cwd=root,
                env=environment, capture_output=True, timeout=5)
            self.assertEqual(new.returncode, 1)
            self.assertFalse(marker.exists())
            self.assertIn(b'classification=HOLD', new.stdout)
            self.assertEqual(new.stdout.count(b'CS_TRANSITION_FILE_BEGIN '), 4)
            self.assertEqual(new.stderr, b'')


class ObservedTextResult(unittest.TextTestResult):
    """Actual unittest callbacks, with full originals left in stderr."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_order = []
        self.observed_ends = []
        self.observed_subtests = 0
        self._terminal = {}

    def startTest(self, test):
        super().startTest(test)
        identity = case_id(test)
        self.observed_order.append(identity)
        emit_observation('CASE_BEGIN', case_id=identity, runtime_id=test.id())

    def _outcome(self, test, status, error=None, formatted=None, message=''):
        identity = case_id(test)
        prior = self._terminal.get(identity)
        if prior is not None and prior['status'] in ('FAIL', 'ERROR') and status == 'PASS':
            return
        raw = formatted.encode('utf-8') if formatted is not None else b''
        self._terminal[identity] = {
            'case_id': identity, 'runtime_id': test.id(), 'status': status,
            'exception_type': error[0].__name__ if error is not None else None,
            'message': (str(error[1]) if error is not None else message)[:512],
            'traceback_bytes': len(raw),
            'traceback_sha256': hashlib.sha256(raw).hexdigest() if formatted is not None else None,
        }

    def addSuccess(self, test):
        super().addSuccess(test)
        self._outcome(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._outcome(test, 'FAIL', err, self.failures[-1][1])

    def addError(self, test, err):
        super().addError(test, err)
        self._outcome(test, 'ERROR', err, self.errors[-1][1])

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._outcome(test, 'SKIP', message=str(reason))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._outcome(test, 'EXPECTED_FAILURE', err, self.expectedFailures[-1][1])

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._outcome(test, 'UNEXPECTED_SUCCESS')

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        status = 'PASS' if err is None else 'FAIL' if issubclass(err[0], test.failureException) else 'ERROR'
        self.observed_subtests += 1
        emit_observation('CASE_SUBTEST', case_id=case_id(test),
            subtest_id=str(subtest)[:512], status=status,
            exception_type=err[0].__name__ if err is not None else None,
            message=str(err[1])[:512] if err is not None else '')
        if err is not None:
            formatted = self.failures[-1][1] if status == 'FAIL' else self.errors[-1][1]
            self._outcome(test, status, err, formatted)

    def stopTest(self, test):
        identity = case_id(test)
        if identity not in self._terminal:
            raise ValueError('HOLD_MISSING_ACTUAL_CASE_OUTCOME')
        row = self._terminal.pop(identity)
        self.observed_ends.append(row)
        emit_observation('CASE_END', **row)
        super().stopTest(test)


def main():
    # This file has no standalone/ambient wrapper-selection route. Only the
    # authenticated fixed launcher supplies its contract and event callback.
    if sys.platform != 'linux' or type(CAPTURE_CONTRACT) is not dict or not callable(CAPTURE_OBSERVER):
        raise ValueError('HOLD_SUPPORTED_AUTHENTICATED_LAUNCHER_REQUIRED')
    contract = CAPTURE_CONTRACT
    if contract.get('status') != 'ROOT_FROZEN_EXECUTION_CONTRACT':
        raise ValueError('HOLD_PENDING_EXECUTION_CONTRACT')
    selection = os.environ.get('SITE_CAPTURE_SELECTION')
    if selection not in ('baseline', 'candidate'):
        raise ValueError('HOLD_FIXED_SELECTION_REQUIRED')
    expected_workflow = BASELINE_WORKFLOW if selection == 'baseline' else DEFAULT_WORKFLOW
    if WORKFLOW != expected_workflow or str(WORKFLOW) != contract['workflows'][selection]['path']:
        raise ValueError('HOLD_WORKFLOW_OVERRIDE')
    if Path.cwd() != PROJECT_ROOT or str(PROJECT_ROOT) != contract['cwd']:
        raise ValueError('HOLD_ACTUAL_HARNESS_CWD')
    if str(Path(__file__).resolve()) != contract['harness']['path']:
        raise ValueError('HOLD_ACTUAL_HARNESS_SOURCE_PATH')
    actual_candidate = sorted('InlineCaptureTests.' + name for name, value in InlineCaptureTests.__dict__.items()
                              if name.startswith('test_') and callable(value))
    baseline = ['InlineCaptureTests.test_missing_tap_does_not_skip_source_recheck_or_context_attempt']
    if contract['selectors']['candidate'] != actual_candidate or contract['selectors']['baseline'] != baseline:
        raise ValueError('HOLD_EXACT_SELECTOR_CENSUS')
    selectors = contract['selectors'][selection]
    if sys.argv[1:] != selectors or len(actual_candidate) != 20:
        raise ValueError('HOLD_EXPLICIT_CANONICAL_SELECTOR_ORDER')
    suite = unittest.defaultTestLoader.loadTestsFromNames(selectors, module=sys.modules[__name__])
    if suite.countTestCases() != len(selectors):
        raise ValueError('HOLD_ACTUAL_CASE_COUNT')
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, resultclass=ObservedTextResult).run(suite)
    counts = {status: sum(row['status'] == status for row in result.observed_ends)
              for status in ('PASS', 'FAIL', 'ERROR', 'SKIP', 'EXPECTED_FAILURE', 'UNEXPECTED_SUCCESS')}
    emit_observation('HARNESS_RESULT', tests_run=result.testsRun, passed=counts['PASS'],
        failed=counts['FAIL'], errors=counts['ERROR'], skipped=counts['SKIP'],
        expected_failures=counts['EXPECTED_FAILURE'], unexpected_successes=counts['UNEXPECTED_SUCCESS'],
        ordered_cases=result.observed_order, subtests=result.observed_subtests)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
