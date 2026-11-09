"""Narrow synchronous unittest child provenance for inventory mutation controls.

The child captures live traceback code identities before unittest formats them.
Only the selected test body and a finite set of unittest assertion helpers can
authorize an assertion failure. This is not a sandbox for hostile test code.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import unittest.case


SCHEMA = 'cbz-inventory-origin-v1'


def _code(value):
    return getattr(getattr(value, '__func__', value), '__code__', None)


def _assertion_codes():
    classes = [unittest.TestCase, unittest.case._AssertRaisesContext,
               unittest.case._AssertRaisesBaseContext]
    base = getattr(unittest.case, '_BaseTestCaseContext', None)
    if base is not None:
        classes.append(base)
    return {code for cls in classes for name, value in vars(cls).items()
            if (name.startswith('assert') or name in
                ('fail', '_baseAssertEqual', '_formatMessage', '__exit__', '_raiseFailure'))
            for code in [_code(value)] if code is not None}


def _leaves(suite):
    if isinstance(suite, unittest.TestSuite):
        for child in suite:
            yield from _leaves(child)
    else:
        yield suite


def _child(directory, selector):
    sys.path.insert(0, str(Path(directory).resolve()))
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        suite = unittest.defaultTestLoader.loadTestsFromName(selector)
        cases = list(_leaves(suite))
        ids = [case.id() for case in cases]
        if not cases or len(ids) != len(set(ids)):
            raise ValueError('test inventory must be nonempty and unique')
        allowed = _assertion_codes()
        states = {}
        for case in cases:
            selected = getattr(case, getattr(case, '_testMethodName', ''), None)
            code = _code(selected)
            if (code is None or inspect.iscoroutinefunction(selected) or
                    inspect.isgeneratorfunction(selected) or inspect.isasyncgenfunction(selected)):
                raise ValueError('test inventory requires synchronous test bodies')
            state = {'code': code, 'entered': 0, 'phase': 'initial'}
            states[case.id()] = state
            for name, phase in (('_callSetUp', 'setup'), ('_callTearDown', 'teardown'),
                                ('_callCleanup', 'cleanup')):
                original = getattr(case, name)
                def wrapped(*args, _original=original, _state=state, _phase=phase, **kwargs):
                    _state['phase'] = _phase
                    return _original(*args, **kwargs)
                setattr(case, name, wrapped)

            def call_body(method, _state=state):
                if _code(method) is not _state['code']:
                    raise RuntimeError('selected test body changed')
                _state['entered'] += 1
                _state['phase'] = 'body'
                try:
                    returned = method()
                    if returned is not None:
                        if inspect.iscoroutine(returned) or inspect.isgenerator(returned):
                            returned.close()
                        raise RuntimeError('unsupported test-body return value')
                finally:
                    _state['phase'] = 'body-exited'
            case._callTestMethod = call_body

        class Result(unittest.TextTestResult):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.started = []
                self.stopped = []
                self.succeeded = []
                self.events = []

            def startTest(self, test):
                self.started.append(test.id())
                super().startTest(test)

            def stopTest(self, test):
                self.stopped.append(test.id())
                super().stopTest(test)

            def addSuccess(self, test):
                self.succeeded.append(test.id())
                super().addSuccess(test)

            def capture(self, test, err, kind):
                case = getattr(test, 'test_case', test)
                state = states.get(case.id(), {})
                frames = []
                tb = err[2]
                while tb is not None:
                    frames.append((tb.tb_frame.f_code, tb.tb_lineno))
                    tb = tb.tb_next
                selected_code = state.get('code')
                positions = [i for i, (code, _) in enumerate(frames) if code is selected_code]
                authorized = (kind == 'failure' and type(err[1]) is AssertionError and
                              state.get('entered') == 1 and
                              state.get('phase') in ('body', 'body-exited') and
                              len(positions) == 1 and
                              all(code in allowed for code, _ in frames[positions[0] + 1:]))
                self.events.append({'id': case.id(), 'kind': kind, 'authorized': authorized,
                                    'exception': err[0].__name__,
                                    'frames': [{'name': code.co_name, 'filename': code.co_filename,
                                                'line': line, 'selected': code is selected_code,
                                                'oracle': code in allowed}
                                               for code, line in frames]})

            def addFailure(self, test, err):
                self.capture(test, err, 'failure')
                super().addFailure(test, err)

            def addError(self, test, err):
                self.capture(test, err, 'error')
                super().addError(test, err)

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    kind = 'failure' if issubclass(err[0], test.failureException) else 'error'
                    self.capture(subtest, err, kind)
                super().addSubTest(test, subtest, err)

        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, resultclass=Result).run(suite)
        return {'schema': SCHEMA, 'selector': selector, 'ids': ids,
                'tests_run': result.testsRun, 'started': result.started, 'stopped': result.stopped,
                'succeeded': result.succeeded,
                'entered': {identity: state['entered'] for identity, state in states.items()},
                'failure_count': len(result.failures), 'error_count': len(result.errors),
                'skip_count': len(result.skipped),
                'skips': [{'id': test.id(), 'reason': reason} for test, reason in result.skipped],
                'expected_failure_count': len(result.expectedFailures),
                'unexpected_success_count': len(result.unexpectedSuccesses),
                'events': result.events, 'captured_stdout': captured.getvalue()}


def classify(report, expected_ids, target, allowed_skips):
    """Return baseline, kill or invalid, rejecting absent or inconsistent fields."""
    if (not isinstance(report, dict) or not isinstance(expected_ids, (list, tuple)) or
            not expected_ids or any(type(x) is not str for x in expected_ids) or
            len(set(expected_ids)) != len(expected_ids) or target not in expected_ids or
            type(allowed_skips) is not dict or
            any(type(k) is not str or type(v) is not str for k, v in allowed_skips.items()) or
            not set(allowed_skips).issubset(expected_ids)):
        return 'invalid'
    counts = ('tests_run', 'failure_count', 'error_count', 'skip_count',
              'expected_failure_count', 'unexpected_success_count', 'child_exit')
    if (report.get('protocol_error') is not False or report.get('schema') != SCHEMA or
            report.get('selected_id') != target or
            any(type(report.get(key)) is not int for key in counts) or
            report.get('ids') != list(expected_ids) or
            report.get('started') != list(expected_ids) or
            report.get('stopped') != list(expected_ids) or
            report['tests_run'] != len(expected_ids) or report['error_count'] != 0 or
            report['expected_failure_count'] != 0 or report['unexpected_success_count'] != 0):
        return 'invalid'
    skips = report.get('skips')
    wanted_skips = [{'id': identity, 'reason': allowed_skips[identity]}
                    for identity in expected_ids if identity in allowed_skips]
    if skips != wanted_skips or report['skip_count'] != len(wanted_skips):
        return 'invalid'
    entered = report.get('entered')
    if (type(entered) is not dict or set(entered) != set(expected_ids) or
            any(type(entered[x]) is not int or entered[x] != (0 if x in allowed_skips else 1)
                for x in expected_ids)):
        return 'invalid'
    successes, events = report.get('succeeded'), report.get('events')
    if (type(successes) is not list or
            any(type(x) is not str or x not in expected_ids for x in successes) or
            len(set(successes)) != len(successes) or
            type(events) is not list or len(events) != report['failure_count'] or
            any(type(event) is not dict or event.get('kind') != 'failure' or
                event.get('authorized') is not True or event.get('id') not in expected_ids
                for event in events)):
        return 'invalid'
    for event in events:
        frames = event.get('frames')
        if (event.get('exception') != 'AssertionError' or type(frames) is not list or
                any(type(frame) is not dict or type(frame.get('name')) is not str or
                    type(frame.get('filename')) is not str or type(frame.get('line')) is not int or
                    type(frame.get('selected')) is not bool or type(frame.get('oracle')) is not bool
                    for frame in frames)):
            return 'invalid'
        selected = [index for index, frame in enumerate(frames) if frame['selected']]
        if len(selected) != 1 or not all(frame['oracle'] for frame in frames[selected[0] + 1:]):
            return 'invalid'
    failed = {event['id'] for event in events}
    if (set(successes) & failed or set(successes) & set(allowed_skips) or
            failed & set(allowed_skips) or
            set(successes) | failed | set(allowed_skips) != set(expected_ids)):
        return 'invalid'
    if report['child_exit'] == 0 and not events:
        return 'baseline'
    if report['child_exit'] == 1 and events and target in failed:
        return 'kill'
    return 'invalid'


def run(directory, selector, expected_ids, target, allowed_skips=None, env=None, timeout=180):
    allowed_skips = {} if allowed_skips is None else allowed_skips
    command = [sys.executable, '-B', *(['-O'] if sys.flags.optimize else []), '-S',
               str(Path(__file__).resolve()), '--child', str(directory), selector]
    try:
        child = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, text=True,
                               capture_output=True, timeout=timeout)
        report = json.loads(child.stdout)
        if type(report) is not dict:
            raise ValueError('report is not an object')
        if report.get('selector') != selector:
            raise ValueError('child selector mismatch')
        report.update(protocol_error=False, child_exit=child.returncode,
                      selected_id=target, stderr=child.stderr)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        report = {'protocol_error': True, 'error': repr(error)}
    report['classification'] = classify(report, expected_ids, target, allowed_skips)
    return report


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--child':
        raise SystemExit('internal child invocation required')
    report = _child(sys.argv[2], sys.argv[3])
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report['failure_count'] == 0 and report['error_count'] == 0 and
                     report['unexpected_success_count'] == 0 else 1)
