#!/usr/bin/env python3
"""Verify the Git census baseline and seven adverse mutations, not Git cleanliness.

Requires a NEW --output directory. Retains private logs, hashes and results there;
mutant sources exist only in an owned temporary directory. Uses Python stdlib.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


REQUIRED_BASELINE_TESTS = 14
REQUIRED_MUTATIONS = 7
EXPECTED_BASELINE_METHODS = (
    'test_nested_linked_state_refs_stash_and_redacted_remotes',
    'test_submodule_bare_store_shallow_and_internal_symlink',
    'test_broken_marker_and_external_directory_boundary_are_not_silent',
    'test_cli_private_output_and_nonzero_incomplete',
    'test_hanging_git_is_killed_and_reported',
    'test_closed_streams_do_not_leave_a_hanging_child',
    'test_rename_arbitrary_path_bytes_and_symlink_cycle',
    'test_status_does_not_execute_configured_clean_filter',
    'test_output_limit_and_total_deadline_are_explicit',
    'test_metadata_directory_alias_does_not_escape_git_exclusion',
    'test_unborn_and_detached_heads_are_observations_not_errors',
    'test_equals_filter_keys_are_suppressed_in_direct_and_recursive_status',
    'test_ordinary_head_objects_assets_do_not_hide_nested_repositories',
    'test_remote_redaction_preserves_all_filesystem_identity_fields',
)

CASES = [
    ('exclude-node-modules', "if entry.name == '.git' and _is_git_metadata(Path(entry.path), git):",
     "if entry.name == 'node_modules' or (entry.name == '.git' and _is_git_metadata(Path(entry.path), git)):",
     'test_nested_linked_state_refs_stash_and_redacted_remotes'),
    ('disable-redaction', 'def _redact(value):\n', 'def _redact(value):\n    return value\n',
     'test_nested_linked_state_refs_stash_and_redacted_remotes'),
    ('execute-clean-filter', "settings += [prefix + '.clean=', prefix + '.process=', prefix + '.required=false']",
     'settings += []', 'test_status_does_not_execute_configured_clean_filter'),
    ('false-complete', 'complete=not reader.errors', 'complete=True',
     'test_broken_marker_and_external_directory_boundary_are_not_silent'),
    ('ambiguous-filter-key', "key, _, value = setting.rpartition('=')", "key, _, value = setting.partition('=')",
     'test_equals_filter_keys_are_suppressed_in_direct_and_recursive_status'),
    ('unvalidated-metadata',
     "exact = result is not None and result[0] == 0 and Path(_line(result[1])).resolve() == directory",
     'exact = True', 'test_ordinary_head_objects_assets_do_not_hide_nested_repositories'),
    ('redact-path-identities', '    return report\n', '    return _redact(report)\n',
     'test_remote_redaction_preserves_all_filesystem_identity_fields'),
]

DRIVER = r'''
import collections, importlib.util, inspect, json, os, pathlib, sys, unittest
test, tool, result, name, population = sys.argv[1:]
manifest = json.loads(population)
spec = importlib.util.spec_from_file_location('census_tests', test)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TOOL = pathlib.Path(tool)
inventory_class = module.InventoryTests
canonical_prefix = spec.name + '.InventoryTests.'
pristine_methods = {method: getattr(inventory_class, method) for method in manifest['methods']}
pristine_codes = {method: getattr(function, '__code__', None) for method, function in pristine_methods.items()}
suite = (unittest.defaultTestLoader.loadTestsFromName('InventoryTests.' + name, module)
         if name else unittest.defaultTestLoader.loadTestsFromModule(module))
records = {}
loaded_tests, started_tests, body_started_tests, body_completed_tests = [], [], [], []
loaded_classes, execution_classes = [], []
expected_methods = manifest['methods']
manifest_valid = (len(expected_methods) == manifest['count'] > 0 and
                  len(set(expected_methods)) == len(expected_methods) and
                  (not name or name in expected_methods))
expected_tests = [canonical_prefix + method for method in ([name] if name else expected_methods)]

def identity(case):
    return type(case).__module__ + '.' + type(case).__qualname__ + '.' + str(getattr(case, '_testMethodName', '<unknown>'))

def pristine_method(case, method):
    function = getattr(method, '__func__', method)
    expected = pristine_methods.get(case._testMethodName)
    return function is expected and expected is not None and function.__code__ is pristine_codes[case._testMethodName]

def observe(destination, case, method=None):
    destination.append(identity(case))
    actual = method if method is not None else getattr(case, case._testMethodName)
    execution_classes.append(type(case) is inventory_class and pristine_method(case, actual))

def instrument(tests):
    for case in tests:
        if isinstance(case, unittest.TestSuite):
            instrument(case)
            continue
        loaded_tests.append(identity(case))
        loaded_classes.append(type(case) is inventory_class)
        method = getattr(case, case._testMethodName)
        loaded_classes[-1] = loaded_classes[-1] and pristine_method(case, method)
        function = getattr(method, '__func__', method)
        record = {'phase': 'outside', 'raised': {}, 'body_calls': 0, 'escaping_assertion': None,
                  'code': getattr(function, '__code__', None),
                  'unsupported': (inspect.iscoroutinefunction(method) or inspect.isgeneratorfunction(method) or
                                  inspect.isasyncgenfunction(method)),
                  'inventory_case': type(case) is inventory_class,
                  'designated': type(case) is inventory_class and case._testMethodName == name}
        records[id(case)] = record
        for attribute, phase in [('_callSetUp', 'setUp'), ('_callTestMethod', 'test_body'),
                                 ('_callTearDown', 'tearDown'), ('_callCleanup', 'cleanup')]:
            original = getattr(case, attribute)
            def tracked(*args, _original=original, _phase=phase, _record=record, _case=case, **kwargs):
                previous = _record['phase']
                _record['phase'] = _phase
                if _phase == 'test_body':
                    body = args[0]
                    def invoke_body():
                        if _record['unsupported']:
                            raise TypeError('Unsupported coroutine/generator test body')
                        _record['body_calls'] += 1
                        observe(body_started_tests, _case, body)
                        try:
                            returned = body()
                        except BaseException as exc:
                            if isinstance(exc, AssertionError):
                                _record['escaping_assertion'] = exc
                            observe(body_completed_tests, _case, body)
                            raise
                        if inspect.isawaitable(returned) or inspect.isgenerator(returned) or inspect.isasyncgen(returned):
                            _record['unsupported'] = True
                            if inspect.iscoroutine(returned) or inspect.isgenerator(returned):
                                returned.close()
                            raise TypeError('Unsupported deferred test-body result')
                        observe(body_completed_tests, _case, body)
                        return returned
                    args = (invoke_body, *args[1:])
                try:
                    return _original(*args, **kwargs)
                except BaseException as exc:
                    # Result.addFailure runs after unittest catches the exception,
                    # so retain the phase on this exception instead of a last-phase flag.
                    _record['raised'][id(exc)] = _phase
                    raise
                finally:
                    _record['phase'] = previous
            setattr(case, attribute, tracked)

class AttributedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.issues = []
        self.subtests = 0

    def startTest(self, case):
        observe(started_tests, case)
        super().startTest(case)

    def record_issue(self, case, err, kind):
        record = records.get(id(case))
        phase = record['raised'].get(id(err[1]), record['phase']) if record else 'unknown_fixture'
        traceback = err[2]
        origin = False
        while traceback is not None:
            if (record and record['inventory_case'] and record['code'] is not None and
                    traceback.tb_frame.f_code is record['code']):
                origin = True
            traceback = traceback.tb_next
        self.issues.append({'test': case.id(), 'kind': kind, 'phase': phase,
                            'origin_in_test_body': origin,
                            'escaping_assertion': bool(record and isinstance(err[1], AssertionError) and
                                                       record['escaping_assertion'] is err[1]),
                            'designated': bool(record and record['designated'])})

    def addFailure(self, case, err):
        self.record_issue(case, err, 'failure')
        super().addFailure(case, err)

    def addError(self, case, err):
        self.record_issue(case, err, 'error')
        super().addError(case, err)

    def addSubTest(self, case, subtest, err):
        # This finite InventoryTests verifier supports no subtest protocol,
        # including successful reports and callbacks with foreign parents.
        self.subtests += 1
        if err is not None:
            self.record_issue(case, err, 'failure' if issubclass(err[0], case.failureException) else 'error')
        super().addSubTest(case, subtest, err)

instrument(suite)
run = unittest.TextTestRunner(verbosity=2, resultclass=AttributedResult).run(suite)
loaded_population_valid = (manifest_valid and all(loaded_classes) and
                           collections.Counter(loaded_tests) == collections.Counter(expected_tests))
execution_population_valid = (all(execution_classes) and
                              all(collections.Counter(actual) == collections.Counter(expected_tests)
                                  for actual in (started_tests, body_started_tests, body_completed_tests)))
summary = {'tests_run': run.testsRun, 'successful': run.wasSuccessful(),
           'failures': len(run.failures), 'errors': len(run.errors),
           'skipped': len(run.skipped), 'expected_failures': len(run.expectedFailures),
           'unexpected_successes': len(run.unexpectedSuccesses), 'optimization': sys.flags.optimize,
           'unsupported_bodies': sum(record['unsupported'] for record in records.values()),
           'body_calls': sum(record['body_calls'] for record in records.values()),
           'designated_body_calls': sum(record['body_calls'] for record in records.values() if record['designated']),
           'body_failures': sum(issue['kind'] == 'failure' and issue['phase'] == 'test_body' and
                                issue['origin_in_test_body'] and issue['designated'] and issue['escaping_assertion'] for issue in run.issues),
           'subtests': run.subtests,
           'fixture_issues': sum(issue['phase'] != 'test_body' for issue in run.issues),
           'expected_tests': expected_tests, 'loaded_tests': loaded_tests, 'started_tests': started_tests,
           'body_started_tests': body_started_tests, 'body_completed_tests': body_completed_tests,
           'loaded_population_valid': loaded_population_valid,
           'execution_population_valid': execution_population_valid,
           'population_valid': loaded_population_valid and execution_population_valid,
           'issues': run.issues}
with os.fdopen(os.open(result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
    json.dump(summary, output)
raise SystemExit(0 if run.wasSuccessful() else 1)
'''


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   'w', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=True, indent=2)
        output.write('\n')


def _trial(test, tool, output, label, name='', timeout=20):
    metrics = output / (label + '.json')
    log = output / (label + '.log')
    row = {'test': name or 'entire suite', 'log': log.name, 'metrics': metrics.name,
           'timeout_seconds': timeout}
    interpreter = [sys.executable, '-S', '-B']
    if sys.flags.optimize:
        interpreter.append('-' + 'O' * sys.flags.optimize)
    with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        child = subprocess.Popen([*interpreter, '-c', DRIVER,
                                  str(test), str(tool), str(metrics), name,
                                  json.dumps({'methods': EXPECTED_BASELINE_METHODS, 'count': REQUIRED_BASELINE_TESTS})],
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            row['returncode'] = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            row.update(returncode=child.returncode, error='timeout')
    if 'error' not in row:
        try:
            row.update(json.loads(metrics.read_bytes()))
            if row.get('optimization') != sys.flags.optimize:
                row['error'] = 'Child optimization does not match companion'
        except (OSError, ValueError) as exc:
            row['error'] = 'Missing/invalid test metrics: ' + str(exc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New private evidence directory')
    args = parser.parse_args()
    output = args.output.absolute()
    try:
        output.mkdir(mode=0o700)
        output.chmod(0o700)
    except OSError as exc:
        parser.error('Cannot exclusively create output directory: ' + str(exc))
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    tool = root / 'tools/workspace_git_inventory.py'
    test = root / 'tests/test_workspace_git_inventory.py'
    report = {'schema_version': 1, 'purpose': 'Git census self-tests only; no mathematical certification or Git cleanliness claim',
              'started_at': _now(), 'python': sys.version, 'python_executable': sys.executable,
              'sources': {}, 'baseline': {}, 'mutations': [], 'errors': [],
              'complete': False, 'all_detected': False}
    try:
        original = tool.read_bytes()
        for label, path in [('collector', tool), ('tests', test), ('runner', Path(__file__).resolve())]:
            raw = path.read_bytes()
            report['sources'][label] = {'path': str(path), 'sha256': _sha(raw), 'bytes': len(raw)}
        baseline = _trial(test, tool, output, 'baseline', timeout=60)
        baseline['passed'] = (baseline.get('returncode') == 0 and baseline.get('tests_run') == REQUIRED_BASELINE_TESTS and
                              baseline.get('population_valid') is True and
                              baseline.get('body_calls') == REQUIRED_BASELINE_TESTS and
                              baseline.get('unsupported_bodies') == 0 and
                              baseline.get('subtests') == 0 and
                              baseline.get('optimization') == sys.flags.optimize and 'error' not in baseline and
                              baseline.get('successful') is True and
                              all(baseline.get(key) == 0 for key in
                                  ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes')))
        report['baseline'] = baseline
        if baseline['passed']:
            text = original.decode('utf-8')
            with tempfile.TemporaryDirectory(prefix='workspace-git-inventory-mutations-') as temporary:
                for name, old, new, selected in CASES:
                    if text.count(old) != 1:
                        raise ValueError('Mutation target is not unique: ' + name)
                    mutant = Path(temporary) / (name + '.py')
                    raw = text.replace(old, new).encode('utf-8')
                    mutant.write_bytes(raw)
                    row = _trial(test, mutant, output, name, selected)
                    row.update(mutation=name, source_sha256=_sha(original), mutant_sha256=_sha(raw),
                               replacement={'old': old, 'new': new})
                    row['detected_by_assertion'] = (
                        row.get('returncode') == 1 and row.get('tests_run') == 1 and
                        row.get('population_valid') is True and
                        row.get('failures') == 1 and row.get('successful') is False and
                        row.get('body_calls') == 1 and row.get('designated_body_calls') == 1 and
                        row.get('unsupported_bodies') == 0 and
                        row.get('subtests') == 0 and
                        row.get('body_failures') == 1 and row.get('fixture_issues') == 0 and
                        len(row.get('issues', [])) == 1 and row.get('optimization') == sys.flags.optimize and
                        all(row.get(key) == 0 for key in
                            ('errors', 'skipped', 'expected_failures', 'unexpected_successes')) and 'error' not in row)
                    report['mutations'].append(row)
        else:
            report['errors'].append('Baseline did not pass all ' + str(REQUIRED_BASELINE_TESTS) + ' tests without skips/errors; no mutations run')
        for label, source in report['sources'].items():
            if _sha(Path(source['path']).read_bytes()) != source['sha256']:
                report['errors'].append('Source changed during verification: ' + label)
        report['all_detected'] = (len(report['mutations']) == REQUIRED_MUTATIONS and
                                  all(row['detected_by_assertion'] for row in report['mutations']))
        report['complete'] = bool(baseline['passed'] and report['all_detected'] and not report['errors'])
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['finished_at'] = _now()
    _write_json(output / 'results.json', report)
    print(json.dumps({'output': str(output), 'baseline_passed': report['baseline'].get('passed', False),
                      'mutations': len(report['mutations']), 'all_detected': report['all_detected'],
                      'complete': report['complete']}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
