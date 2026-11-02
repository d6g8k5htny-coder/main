#!/usr/bin/env python3
"""Bounded source-omission experiment for the Main lifecycle classifier.

Run from the repository root; --output must be outside it. The untouched
classifier certifies real outer control bodies against exact IDs/failures.
This is Python harness evidence only, with no Lean or kernel execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests import test_formal_compiled_inventory as golden

SOURCE = ROOT / 'tests/test_formal_compiled_inventory.py'


def body_failure(test):
    test.fail('intended assertion')


def clean(test):
    pass


async def coroutine(test):
    test.fail('dormant coroutine')


async def async_generator(test):
    yield 1


def generator(test):
    yield 1


def run_inner(module, body=body_failure, **methods):
    control = type('InnerControl', (unittest.TestCase,), {'test_expected': body, **methods})
    return module.run_policy_controls(unittest.TestSuite([control('test_expected')]))


def must_refuse(test, module, scenario):
    baseline = False
    if scenario == 'fixture':
        result = run_inner(module, clean, tearDown=body_failure)
    elif scenario == 'event-phase':
        result = run_inner(module)
        result.events[0]['phase'] = 'teardown'
    elif scenario == 'callback':
        def callback(case):
            try:
                case.fail('caught assertion callback')
            except AssertionError:
                case._outcome.result.addFailure(case, sys.exc_info())
        result = run_inner(module, callback)
    elif scenario in ('fixture-method', 'fixture-dispatch'):
        def setup(case):
            if scenario == 'fixture-dispatch':
                case._callTestMethod(case.test_expected)
            else:
                case.test_expected()
        result = run_inner(module, clean, setUp=setup)
        baseline = True
    elif scenario == 'clean-subtest':
        def body(case):
            with case.subTest(clean=True):
                pass
        result = run_inner(module, body)
        baseline = True
    elif scenario in ('coroutine', 'async-generator', 'generator'):
        result = run_inner(module, {'coroutine': coroutine, 'async-generator': async_generator,
                                   'generator': generator}[scenario])
        baseline = True
    elif scenario in ('entered', 'completed', 'stopped', 'outcome'):
        result = run_inner(module)
        result.records[0][scenario] = None if scenario == 'outcome' else False
    elif scenario == 'records':
        result = run_inner(module, clean)
        result.records.clear()
        baseline = True
    elif scenario == 'selected':
        result = run_inner(module)
        result.selected_ids = ('unknown.test_expected',)
    elif scenario == 'duplicate':
        control = type('DuplicateControl', (unittest.TestCase,), {'test_expected': clean})
        result = module.run_policy_controls(unittest.TestSuite([
            control('test_expected'), control('test_expected')]))
        baseline = True
    elif scenario == 'empty':
        result = module.run_policy_controls(unittest.TestSuite())
        baseline = True
    elif scenario == 'unknown-parent':
        result = run_inner(module)
        result.events[0]['test'] = type(result.records[0]['test'])('test_expected')
    elif scenario == 'unmapped-failure':
        result = run_inner(module)
        result.events[0]['failure_test'] = type(result.records[0]['test'])('test_expected')
    elif scenario == 'subtest-parent':
        def wrong_parent(case):
            try:
                case.fail('subtest from a different parent')
            except AssertionError:
                other = unittest.FunctionTestCase(lambda: None)
                subtest = unittest.case._SubTest(other, None, {})
                case._outcome.result.addSubTest(case, subtest, sys.exc_info())
        result = run_inner(module, wrong_parent)
    elif scenario == 'named-failures':
        result = run_inner(module)
    else:
        raise ValueError('unknown scenario: ' + scenario)
    expected_ids = tuple(record['id'] for record in result.records)
    # The record-removal scenario still uses the frozen selection, as production does.
    if scenario == 'records':
        expected_ids = result.selected_ids
    with test.assertRaises(AssertionError):
        if baseline:
            module.require_policy_control(result, expected_ids)
        else:
            names = {'missing'} if scenario == 'named-failures' else {'test_expected'}
            module.require_policy_mutation_kill(result, expected_ids, names)


SUPPORTED_REMOVAL = [
    ("if not record['supported']:", 'if False:'),
    ("record['supported'] is not True or ", ''),
]
MUTATIONS = [
    ('omit-fixture-phase', [("event['phase'] != 'body' or ", '')], 'event-phase'),
    ('omit-fixture-phase-and-provenance', [("event['phase'] != 'body' or ", ''),
        ("'failure' if proven else 'unproven-failure'", "'failure'")], 'fixture'),
    ('omit-callback-provenance', [("'failure' if proven else 'unproven-failure'", "'failure'")], 'callback'),
    ('omit-body-entry', [("or record['entered'] is not True", '')], 'entered'),
    ('omit-body-completion', [("or record['completed'] is not True", '')], 'completed'),
    ('omit-test-stop', [("or record['stopped'] is not True", '')], 'stopped'),
    ('omit-body-outcome', [("or record['outcome'] not in ('normal', 'assertion')", '')], 'outcome'),
    ('omit-record-population', [("\n            or tuple(record['id'] for record in result.records) != expected_ids", '')], 'records'),
    ('omit-selected-population', [("or result.selected_ids != expected_ids", '')], 'selected'),
    ('omit-unique-population', [("or len(expected_ids) != len(set(expected_ids))", '')], 'duplicate'),
    ('omit-nonempty-population', [("not expected_ids or ", '')], 'empty'),
    ('omit-event-parent', [("\n           or not any(event['test'] is record['test'] for record in result.records)", '')], 'unknown-parent'),
    ('omit-failure-association', [("(len(result.events) != len(result.failures)\n            or any(event['failure_test'] is not failure[0]\n                   for event, failure in zip(result.events, result.failures)))", 'False')], 'unmapped-failure'),
    ('omit-subtest-refusal', [("self.event(test, 'subtest', subtest)", "self.event(test, 'failure', subtest)")], 'subtest-parent'),
    ('omit-clean-subtest-refusal', [("self.event(test, 'subtest', subtest)", "if err is not None: self.event(test, 'subtest', subtest)")], 'clean-subtest'),
    *[('omit-authorization-' + scenario, [
        ("if (not record['dispatching'] or record['fixture_depth'] != 0\n                    or record['phase'] != 'body' or record['entered']):", 'if False:'),
        ("if (record['fixture_depth'] != 0 or record['phase'] != 'setup'\n                        or record['entered']):", 'if False:'),
      ], scenario) for scenario in ('fixture-method', 'fixture-dispatch')],
    ('omit-named-failures', [("if not named or any(len(identities) != 1 for identities in named.values()):", 'if False:'),
                              ("(len(failures) != len(set(failures))\n            or set(failures) != {identities[0] for identities in named.values()})", 'False')], 'named-failures'),
    *[('omit-unsupported-' + scenario, SUPPORTED_REMOVAL, scenario)
      for scenario in ('coroutine', 'async-generator', 'generator')],
]


def load(source):
    module = types.ModuleType('policy_lifecycle_source_mutant')
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    return module


def observe(module, scenario):
    def body(test):
        must_refuse(test, module, scenario)
    control = type('SafeguardControl', (unittest.TestCase,), {'test_rejects_regression': body})
    suite = unittest.TestSuite([control('test_rejects_regression')])
    ids = golden.policy_test_ids(suite)
    result = golden.run_policy_controls(suite)
    return result, ids


def summary(result):
    return dict(testsRun=result.testsRun, failures=[t.id() for t, _ in result.failures],
                errors=[t.id() for t, _ in result.errors], skipped=result.skipped,
                expectedFailures=len(result.expectedFailures), unexpectedSuccesses=len(result.unexpectedSuccesses),
                records=[{k: v for k, v in record.items() if k != 'test'} for record in result.records],
                events=[dict(id=e['test'].id(), kind=e['kind'], phase=e['phase']) for e in result.events])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        parser.error('--output must be a fresh directory outside the repository')
    output.mkdir(parents=True)
    source = SOURCE.read_text()
    records = []
    for label, replacements, scenario in MUTATIONS:
        baseline, ids = observe(load(source), scenario)
        golden.require_policy_control(baseline, ids)
        mutated = source
        for old, new in replacements:
            if mutated.count(old) != 1:
                raise AssertionError('stale/nonunique source anchor: ' + label)
            mutated = mutated.replace(old, new)
        result, actual_ids = observe(load(mutated), scenario)
        golden.require_policy_mutation_kill(result, ids, {'test_rejects_regression'})
        if actual_ids != ids:
            raise AssertionError('source-mutant control identity changed')
        (output / (label + '-baseline.log')).write_text(baseline.stream.stream.getvalue())
        (output / (label + '.log')).write_text(result.stream.stream.getvalue())
        records.append(dict(name=label, scenario=scenario, replacements=replacements,
                            baseline=summary(baseline), mutant=summary(result), qualified=True))
        print(label + ': exact intended real body assertion', flush=True)
    (output / 'summary.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        interpreter=sys.version, optimized=sys.flags.optimize, mutations=records), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
