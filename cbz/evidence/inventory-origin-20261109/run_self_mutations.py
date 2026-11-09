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
import tempfile


ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / 'tests'
HELPER = TESTS / 'cbz_inventory_origin.py'
CONTROLS = TESTS / 'test_cbz_inventory_origin.py'
CHECKER = ROOT / 'cbz/evidence/node-status-freshness/run_mutations.py'

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
        [('all(code in allowed for code, _ in frames[positions[0] + 1:]))', 'True)'),
         ("not all(frame['oracle'] for frame in frames[selected[0] + 1:])", 'False')],
        'test_subject_helper_nested_and_non_oracle_failures_are_refused'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('node_freshness_origin_checker', CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original = HELPER.read_text(encoding='utf-8')
    controls = CONTROLS.read_bytes()
    rows = []
    with tempfile.TemporaryDirectory(prefix='cbz-inventory-origin-self-') as tmp:
        root = Path(tmp)
        tests = root / 'tests'
        tests.mkdir()
        (tests / '__init__.py').write_text('', encoding='utf-8')
        (tests / CONTROLS.name).write_bytes(controls)
        candidate = tests / HELPER.name
        for optimized in (False, True):
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
    print(json.dumps({'schema': 'cbz-inventory-origin-self-mutations-v1',
                      'helper_sha256': digest(original.encode()),
                      'controls_sha256': digest(controls),
                      'checker_sha256': digest(CHECKER.read_bytes()),
                      'cases': len(rows), 'results': rows}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
