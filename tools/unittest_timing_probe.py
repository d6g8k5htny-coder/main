#!/usr/bin/env python3
"""Run native unittest with a flushed JSONL timing sidecar.

Usage: python -B -S tools/unittest_timing_probe.py NEW_JOURNAL discover ...
The journal must not exist. All following arguments belong to unittest.
No timeout, retry, test filter, signal handler, or scientific verdict is added.
An unmatched test_start means no stop was recorded, not that a hang is proven.
Flush is not fsync: abrupt host/storage loss may still lose records. This is a
trusted-process diagnostic, not hostile-test containment or an attestation.
"""
import json
import os
import sys
import time
import unittest


class Journal:
    """Write one ordered record at a time; do not conceal an I/O failure."""
    def __init__(self, stream):
        self.stream = stream
        self.sequence = 0
        self.failure = None

    def emit(self, event, **fields):
        if self.failure is not None:
            raise self.failure
        record = dict(schema='unittest-timing/v1', seq=self.sequence,
                      pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                      event=event, **fields)
        try:
            self.stream.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + '\n')
            self.stream.flush()
        except (OSError, ValueError) as error:
            self.failure = error
            raise
        self.sequence += 1
        return record


def runner_type(journal):
    """Bind this invocation's journal without global unittest monkey-patching."""
    class Result(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.started = {}

        def startTest(self, test):
            super().startTest(test)
            row = journal.emit('test_start', test_id=test.id())
            self.started[id(test)] = row['monotonic_ns']

        def stopTest(self, test):
            super().stopTest(test)
            started = self.started.pop(id(test), None)
            elapsed = None if started is None else time.monotonic_ns() - started
            journal.emit('test_stop', test_id=test.id(), elapsed_ns=elapsed)

        def addSuccess(self, test):
            super().addSuccess(test)
            journal.emit('outcome', test_id=test.id(), outcome='success')

        def addFailure(self, test, err):
            super().addFailure(test, err)
            journal.emit('outcome', test_id=test.id(), outcome='failure')

        def addError(self, test, err):
            super().addError(test, err)
            journal.emit('outcome', test_id=test.id(), outcome='error')

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            journal.emit('outcome', test_id=test.id(), outcome='skip')

        def addExpectedFailure(self, test, err):
            super().addExpectedFailure(test, err)
            journal.emit('outcome', test_id=test.id(), outcome='expected_failure')

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            journal.emit('outcome', test_id=test.id(), outcome='unexpected_success')

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            kind = ('subtest_success' if err is None else
                    'subtest_failure' if issubclass(err[0], test.failureException) else
                    'subtest_error')
            journal.emit('outcome', test_id=subtest.id(), parent_test_id=test.id(), outcome=kind)

    class Runner(unittest.TextTestRunner):
        resultclass = Result

        def run(self, test):
            journal.emit('run_start')
            result = super().run(test)
            journal.emit('run_end', tests_run=result.testsRun,
                         failures=len(result.failures), errors=len(result.errors),
                         skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
                         unexpected_successes=len(result.unexpectedSuccesses),
                         successful=result.wasSuccessful())
            return result

    return Runner


def main():
    if len(sys.argv) < 2:
        print('usage: unittest_timing_probe.py NEW_JOURNAL [unittest arguments]', file=sys.stderr)
        raise SystemExit(2)
    destination, arguments = sys.argv[1], sys.argv[2:]
    # Match the ordinary -m invocation's import root, not this tools directory.
    # -I / -P deliberately omit that automatic entry; discovery retains its own rules.
    if not sys.flags.safe_path:
        sys.path[0] = os.getcwd()
    sys.argv = [os.path.basename(sys.executable) + ' -m unittest', *arguments]
    # Exclusive creation happens before discovery and never truncates old evidence.
    with open(destination, 'x', encoding='utf-8', newline='\n') as stream:
        journal = Journal(stream)
        journal.emit('invocation_start', argv=sys.argv, python=sys.version,
                     optimize=sys.flags.optimize, no_site=sys.flags.no_site,
                     isolated=sys.flags.isolated)
        try:
            unittest.main(module=None, testRunner=runner_type(journal))
        except SystemExit as error:
            journal.emit('invocation_exit', exit_code=error.code)
            raise


if __name__ == '__main__':
    main()
