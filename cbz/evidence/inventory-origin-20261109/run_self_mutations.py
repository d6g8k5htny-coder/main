#!/usr/bin/env python3
"""Adverse source mutations of the inventory provenance recorder.

The separately qualified node-freshness runner records the actual selected
test/oracle traceback before unittest cleanup. This experiment is bounded to
the listed synchronous tests and mutations; it is not a hostile-code sandbox.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / 'tests'
HELPER = TESTS / 'cbz_inventory_origin.py'
CONTROLS = TESTS / 'test_cbz_inventory_origin.py'
CHECKER = ROOT / 'cbz/evidence/node-status-freshness/run_mutations.py'
CONTROLLER = TESTS / 'test_cbz_inventory.py'
INVENTORY_TOOL = ROOT / 'tools/cbz_inventory.py'

MUTANTS = {
    'accept_boolean_counts': (
        [('any(type(report.get(key)) is not int for key in counts)', 'False')],
        'test_missing_extra_and_forged_protocol_fields_fail_closed'),
    'ignore_protocol_error': (
        [("report.get('protocol_error') is not False", 'False')],
        'test_missing_extra_and_forged_protocol_fields_fail_closed'),
    'ignore_selected_target': (
        [('and target in failed:', 'and True:')],
        'test_full_suite_requires_only_declared_skips_and_authorized_origins'),
    'accept_subject_traceback': (
        [('all(_is_oracle(code, allowed) for code, _ in frames[positions[0] + 1:]))', 'True)'),
         ("not all(frame['oracle'] for frame in frames[selected[0] + 1:])", 'False')],
        'test_selected_subject_assertion_is_refused'),
    'compare_oracle_code_by_equality': (
        [('any(code is oracle for oracle in allowed)', 'any(code == oracle for oracle in allowed)')],
        'test_equal_but_distinct_oracle_code_is_refused'),
    'allow_skipped_target': (
        [(' or target in allowed_skips):', '):')],
        'test_declared_skip_cannot_be_the_target'),
    'allow_missing_message': (
        [("type(event.get('message')) is not str", 'False')],
        'test_missing_extra_and_forged_protocol_fields_fail_closed'),
    'remove_callable_assertion': (
        [("'__exit__', '_raiseFailure', 'handle'))",
          "'__exit__', '_raiseFailure', '__unsupported__'))")],
        'test_callable_raises_is_authorized'),
    'remove_mock_assertion': (
        [('unittest._log._AssertLogsContext, unittest.mock.NonCallableMock]',
          'unittest._log._AssertLogsContext]')],
        'test_mock_assertion_is_authorized'),
    'remove_log_assertion': (
        [('unittest._log._AssertLogsContext, unittest.mock.NonCallableMock]',
          'unittest.mock.NonCallableMock]')],
        'test_log_assertion_is_authorized'),
    'remove_warning_assertion': (
        [('unittest.case._AssertRaisesBaseContext, unittest.case._AssertWarnsContext,',
          'unittest.case._AssertRaisesBaseContext,')],
        'test_warning_assertion_is_authorized'),
}

CONTROLLER_MUTANTS = {
    'drop_status_failure_count': (
        "self.assertEqual(mutant['failure_count'], 1, mutant)",
        'self.assertTrue(True)'),
    'drop_status_event_message': (
        "self.assertEqual(mutant['events'][0]['message'],\n"
        "                                     'True is not false : a late repository filter must never execute')",
        'self.assertTrue(True)'),
    'drop_status_formatted_message': (
        "self.assertIn('AssertionError: True is not false : a late repository filter must never execute',\n"
        "                                  mutant['stderr'])",
        'self.assertTrue(True)'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('node_freshness_origin_checker', CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original = HELPER.read_text(encoding='utf-8')
    controls = CONTROLS.read_bytes()
    controller_original = CONTROLLER.read_text(encoding='utf-8')
    rows = []
    with tempfile.TemporaryDirectory(prefix='cbz-inventory-origin-self-') as tmp:
        root = Path(tmp)
        tests = root / 'tests'
        tests.mkdir()
        (tests / '__init__.py').write_text('', encoding='utf-8')
        (tests / CONTROLS.name).write_bytes(controls)
        candidate = tests / HELPER.name
        controller = tests / CONTROLLER.name
        (root / 'tools').mkdir()
        shutil.copyfile(INVENTORY_TOOL, root / 'tools/cbz_inventory.py')
        for optimized in (False, True):
            controller.write_text(controller_original, encoding='utf-8')
            for name, (edits, method) in MUTANTS.items():
                identity = 'tests.test_cbz_inventory_origin.InventoryOriginControls.' + method
                candidate.write_text(original, encoding='utf-8')
                baseline = checker.run_selected(root, identity, optimized)
                if not checker.baseline_ok(baseline, identity):
                    raise RuntimeError('self-test baseline failed: ' + json.dumps(baseline))
                changed = original
                for before, after in edits:
                    if changed.count(before) != 1:
                        raise RuntimeError('nonunique mutation anchor: ' + name)
                    changed = changed.replace(before, after)
                candidate.write_text(changed, encoding='utf-8')
                mutant = checker.run_selected(root, identity, optimized)
                killed = checker.qualified_kill(mutant, identity)
                rows.append({'name': name, 'optimized': optimized, 'test_id': identity,
                             'edits': len(edits), 'mutant_sha256': digest(changed.encode()),
                             'baseline': baseline, 'mutant': mutant, 'killed': killed})
                if not killed:
                    raise RuntimeError('self-mutation not killed by selected oracle: ' +
                                       json.dumps(rows[-1]))
            candidate.write_text(original, encoding='utf-8')
            method = 'test_status_isolation_requires_its_exact_failure_and_count'
            identity = 'tests.test_cbz_inventory_origin.InventoryOriginControls.' + method
            for name, (before, after) in CONTROLLER_MUTANTS.items():
                controller.write_text(controller_original, encoding='utf-8')
                baseline = checker.run_selected(root, identity, optimized)
                if not checker.baseline_ok(baseline, identity):
                    raise RuntimeError('controller self-test baseline failed: ' + json.dumps(baseline))
                if controller_original.count(before) != 1:
                    raise RuntimeError('nonunique controller mutation anchor: ' + name)
                changed = controller_original.replace(before, after, 1)
                controller.write_text(changed, encoding='utf-8')
                mutant = checker.run_selected(root, identity, optimized)
                killed = (checker.qualified_kill(mutant, identity) and
                          'AssertionError: 0 != 1 : ([], [])' in mutant.get('output', ''))
                rows.append({'name': name, 'optimized': optimized, 'test_id': identity,
                             'edits': 1, 'mutant_sha256': digest(changed.encode()),
                             'baseline': baseline, 'mutant': mutant, 'killed': killed})
                if not killed:
                    raise RuntimeError('controller mutation not killed by selected oracle: ' +
                                       json.dumps(rows[-1]))
    print(json.dumps({'schema': 'cbz-inventory-origin-self-mutations-v1',
                      'helper_sha256': digest(original.encode()),
                      'controls_sha256': digest(controls),
                      'controller_sha256': digest(controller_original.encode()),
                      'checker_sha256': digest(CHECKER.read_bytes()),
                      'cases': len(rows), 'results': rows}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
