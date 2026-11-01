#!/usr/bin/env python3
"""Structured unittest execution; only complete selected test bodies grade mutants."""
import argparse
import functools
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
RECEIPT_SCHEMA = 'cbz-unittest-lifecycle-v1'


def method_id(test):
    """Use the actual selected method, rather than an overridable display id."""
    cls = type(test)
    name = getattr(test, '_testMethodName', None)
    if (not isinstance(test, unittest.TestCase) or cls.__module__ == 'unittest.loader'
            or not isinstance(name, str) or not callable(getattr(test, name, None))):
        return None
    return cls.__module__ + '.' + cls.__qualname__ + '.' + name


def supported_method_shape(method):
    """Only synchronous ordinary functions/methods are supported test bodies.

    Inspect every advertised decorator link. Returning suspended work is checked
    separately, since decorators may intentionally omit __wrapped__ metadata.
    """
    try:
        inspect.unwrap(method)  # Reject cyclic decorator metadata.
    except ValueError:
        return False
    current = method
    while True:
        if (not (inspect.isfunction(current) or inspect.ismethod(current))
                or inspect.isgeneratorfunction(current)
                or inspect.iscoroutinefunction(current)
                or inspect.isasyncgenfunction(current)):
            return False
        if not hasattr(current, '__wrapped__'):
            return True
        current = current.__wrapped__


def selected_cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from selected_cases(test)
        else:
            yield test


class LifecycleResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_ids = []
        self.finished_ids = []
        self.body_entered_ids = []
        self.body_finished_ids = []
        self.body_assertion_ids = []
        self.lifecycle_failures = 0
        self.subtests = 0
        self.subtest_events = []
        self.assertion_events = []
        self.phase = {}
        self.body_exception = {}
        self.body_active = set()
        self.dispatch_active = set()
        self.lifecycle_depth = {}
        self.invalid_execution = []

    def startTest(self, test):
        self.started_ids.append(method_id(test))
        self.phase[id(test)] = 'setup'
        super().startTest(test)

    def stopTest(self, test):
        self.finished_ids.append(method_id(test))
        super().stopTest(test)

    def addFailure(self, test, err):
        if (self.phase.get(id(test)) == 'body'
                and self.body_exception.get(id(test)) is err[1]
                and isinstance(err[1], AssertionError)):
            self.body_assertion_ids.append(method_id(test))
            self.assertion_events.append({'parent_id': method_id(test), 'origin': 'body'})
        else:
            self.lifecycle_failures += 1
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        self.subtests += 1
        phase = self.phase.get(id(test))
        if phase == 'body' and id(test) not in self.body_active:
            phase = 'outside-body'
        if getattr(subtest, 'test_case', None) is not test:
            phase = 'foreign-parent'
        outcome = 'success' if err is None else 'assertion' if isinstance(err[1], AssertionError) else 'error'
        self.subtest_events.append({'parent_id': method_id(test), 'phase': phase, 'outcome': outcome})
        if err is not None and outcome == 'assertion':
            if phase == 'body':
                self.body_assertion_ids.append(method_id(test))
                self.assertion_events.append({'parent_id': method_id(test), 'origin': 'subtest'})
            else:
                self.lifecycle_failures += 1
        super().addSubTest(test, subtest, err)


def instrument(test, result):
    """Keep body exception identity until unittest reports it after this call exits."""
    for name, phase in (('_callSetUp', 'setup'), ('_callTearDown', 'teardown'),
                        ('doCleanups', 'cleanup')):
        original = getattr(test, name)

        def lifecycle(*args, _original=original, _phase=phase, **kwargs):
            result.phase[id(test)] = _phase
            result.lifecycle_depth[id(test)] = result.lifecycle_depth.get(id(test), 0) + 1
            try:
                return _original(*args, **kwargs)
            finally:
                result.lifecycle_depth[id(test)] -= 1
        setattr(test, name, lifecycle)
    original_dispatch = test._callTestMethod

    def dispatch(*args, **kwargs):
        if result.lifecycle_depth.get(id(test), 0):
            result.invalid_execution.append('test dispatch invoked inside a lifecycle hook')
            return original_dispatch(*args, **kwargs)
        result.dispatch_active.add(id(test))
        try:
            return original_dispatch(*args, **kwargs)
        finally:
            result.dispatch_active.discard(id(test))
    test._callTestMethod = dispatch
    original_body = getattr(test, test._testMethodName)

    @functools.wraps(original_body)
    def body(*args, **kwargs):
        if id(test) not in result.dispatch_active or result.lifecycle_depth.get(id(test), 0):
            result.invalid_execution.append('selected method invoked outside body dispatch')
            return original_body(*args, **kwargs)
        result.phase[id(test)] = 'body'
        result.body_entered_ids.append(method_id(test))
        result.body_active.add(id(test))
        try:
            value = original_body(*args, **kwargs)
            if value is not None:
                result.invalid_execution.append('selected synchronous method must return None')
                if inspect.iscoroutine(value) and inspect.getcoroutinestate(value) == inspect.CORO_CREATED:
                    value.close()  # Closing a native unstarted coroutine executes no body.
                return None
            return value
        except BaseException as exc:
            result.body_exception[id(test)] = exc
            raise
        finally:
            result.body_active.discard(id(test))
            result.body_finished_ids.append(method_id(test))
            # _Outcome calls addFailure *after* this wrapper unwinds.
    setattr(test, test._testMethodName, body)


def describe_suite(suite, selection):
    cases = list(selected_cases(suite))
    selected = [method_id(test) for test in cases]
    invalid = []
    if not selected:
        invalid.append('empty selected population')
    if any(item is None for item in selected):
        invalid.append('synthetic or unsupported selected test')
    if len(set(selected)) != len(selected):
        invalid.append('duplicate selected method identity')
    for test, identity in zip(cases, selected):
        if identity is not None and test.id() != identity:
            invalid.append('display identity does not match selected method')
        if identity is not None and not supported_method_shape(getattr(test, test._testMethodName)):
            invalid.append('unsupported selected method shape: ' + identity)
    return cases, {'selection': selection, 'selected_ids': selected, 'invalid_population': invalid}


def run_suite(suite, selection):
    cases, population = describe_suite(suite, selection)
    stream = io.StringIO()
    result = LifecycleResult(stream=unittest.runner._WritelnDecorator(stream),
                             descriptions=True, verbosity=2)
    if not population['invalid_population']:
        for test in cases:
            instrument(test, result)
        suite.run(result)
    population['invalid_population'].extend(result.invalid_execution)
    # Render the ordinary summary for humans; grading uses the structured fields.
    result.printErrors()
    stream.write(f'Ran {result.testsRun} tests\n')
    value = {'receipt_schema': RECEIPT_SCHEMA, 'selection': selection,
             **population,
             'tests_run': result.testsRun, 'assertion_failures': len(result.failures),
             'errors': len(result.errors), 'skipped': len(result.skipped),
             'expected_failures': len(result.expectedFailures),
             'unexpected_successes': len(result.unexpectedSuccesses),
             'successful': result.wasSuccessful(), 'log': stream.getvalue()}
    for name in ('started_ids', 'finished_ids', 'body_entered_ids',
                 'body_finished_ids', 'body_assertion_ids', 'lifecycle_failures', 'subtests',
                 'subtest_events', 'assertion_events'):
        value[name] = getattr(result, name)
    return value


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test', action='append', default=[])
    parser.add_argument('--list', action='store_true', help='discover identities without running fixtures or test bodies')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'tests'))
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromNames(args.test)
    else:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    selection = args.test or ['discover:test_*.py']
    if args.list:
        _, population = describe_suite(suite, selection)
        Path(args.output).write_text(json.dumps(population, indent=2) + '\n')
        return 0 if population['selected_ids'] and not population['invalid_population'] else 1
    value = run_suite(suite, selection)
    Path(args.output).write_text(json.dumps(value, indent=2) + '\n')
    print(value['log'], end='')
    complete = (not value['invalid_population'] and value['selected_ids']
                and all(value[key] == value['selected_ids'] for key in
                        ('started_ids', 'finished_ids', 'body_entered_ids', 'body_finished_ids')))
    return 0 if (complete and value['successful'] and not value['skipped']
                 and not value['expected_failures'] and not value['unexpected_successes']
                 and all(event['phase'] == 'body' and event['outcome'] == 'success'
                         for event in value['subtest_events'])
                 and not value['lifecycle_failures']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
