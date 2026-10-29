"""Finite adversarial mutation controls; only disposable local fixtures execute."""
import argparse
import hashlib
import shutil
import subprocess
import sys
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOL_NAME = 'cbz_baseline.py'
TEST_NAME = 'test_cbz_baseline.py'
CLASS_NAME = 'BaselineTests'
SCALE = False


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutations(root, output):
    source = (root / 'tools' / TOOL_NAME).read_text()
    cases = [
        ('live_hash', "elif _sha(path) != entry['sha256']:", 'elif False:', 'test_live_proof_tamper_fails'),
        ('closed_quarantine', "if by_id[dep].get('status') == 'QUARANTINED':", 'if False:', 'test_closed_cannot_depend_transitively_on_quarantine'),
        ('receipt_binding', "if row.get('argv') != gate['argv'] or row.get('source_sha256') != _binding(root, config, gate):", 'if False:', 'test_receipts_bind_actual_execution_and_stale_source_is_refused'),
        ('receipt_logs', "or _sha(log) != row.get(stream + '_sha256')", '', 'test_receipt_log_tampering_is_refused'),
        ('custom_axiom', 'elif any(axiom not in allowed for axiom in axioms):', 'elif False:', 'test_custom_axiom_cannot_be_closed'),
        ('unlisted_tool', 'if path.relative_to(root).as_posix() not in listed:', 'if False:', 'test_unlisted_mission_tool_refused'),
        ('doc_truth', 'elif claims != statuses:', 'elif False:', 'test_receipts_bind_actual_execution_and_stale_source_is_refused'),
    ]
    rows = []
    for name, old, new, test in cases:
        if source.count(old) != 1:
            raise RuntimeError('ambiguous mutation site: ' + name)
        with tempfile.TemporaryDirectory(prefix='cbz-baseline-mutation-') as temp:
            path = Path(temp) / 'mutant.py'
            path.write_text(source.replace(old, new))
            tests = load(root / 'tests' / TEST_NAME, 'baseline_tests_' + name)
            tests.TOOL = path
            with (output / ('mutation-' + name + '.log')).open('w') as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite([tests.BaselineTests(test)]))
            rows.append({'mutation': name, 'test': test, 'detected': result.testsRun == 1 and len(result.failures) == 1 and not result.errors and not result.skipped,
                         'failures': len(result.failures), 'errors': len(result.errors)})
    (output / 'mutations.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def positive(root, output, test_name, class_name):
    tests = load(root / 'tests' / test_name, 'unmutated_positive')
    tests.TOOL = root / 'tools' / ('cbz_refs.py' if class_name == 'RefSafetyTests' else 'cbz_baseline.py')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests, class_name))
    with (output / 'positive.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped),
              'passed': result.wasSuccessful() and result.testsRun > 0 and not result.skipped}
    (output / 'positive.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('unmutated positive harness failed; mutations cannot be credited')
    return report


def cli_controls(root, output):
    script = str(Path(__file__).resolve())
    missing_root = output / 'missing-source-root'
    existing = output / 'cli-existing-output'
    existing.mkdir()
    marker = existing / 'preserve'
    marker.write_text('keep exact bytes\n')
    inside = root / ('.cbz-control-must-not-create-' + output.name)
    if inside.exists() or inside.is_symlink():
        raise RuntimeError('finite inside-root control target unexpectedly exists')
    cases = [
        ('missing_output', ['--root', str(root)]),
        ('missing_root', ['--root', str(missing_root), '--output', str(output / 'must-not-create')]),
        ('existing_output', ['--root', str(root), '--output', str(existing)]),
        ('inside_root_output', ['--root', str(root), '--output', str(inside)]),
    ]
    rows = []
    for name, args in cases:
        result = subprocess.run([sys.executable, '-B', '-S', script, *args],
                                capture_output=True, timeout=15)
        rows.append({'control': name, 'exit_code': result.returncode, 'rejected': result.returncode == 2})
    if (marker.read_text() != 'keep exact bytes\n' or inside.exists()
            or (output / 'must-not-create').exists() or not all(r['rejected'] for r in rows)):
        raise RuntimeError('finite CLI/output preservation controls failed')
    shutil.rmtree(existing)
    (output / 'cli-controls.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='actual source checkout containing tools/ and tests/')
    parser.add_argument('--output', required=True, help='NEW output directory outside the source checkout')
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    original_output = Path(args.output)
    output = original_output.resolve()
    try:
        if (not root.is_dir() or not (root / 'tools' / TOOL_NAME).is_file()
                or not (root / 'tests' / TEST_NAME).is_file()):
            raise ValueError('source root/tool/test unavailable')
        if output.is_relative_to(root) or original_output.is_symlink() or output.exists():
            raise ValueError('output must be NEW and outside the source root')
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'REFUSED', 'error': str(error)}), file=sys.stderr)
        return 2
    sources = [root / 'tools' / TOOL_NAME, root / 'tests' / TEST_NAME]
    before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {'schema': 'cbz.portable-controls.v1', 'status': 'FAIL', 'source_sha256': before,
              'control_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'finite engineering controls only; no global G0..G11, Layer A or CBZ-0 certification'}
    try:
        report['cli_controls'] = cli_controls(root, output)
        report['positive'] = positive(root, output, TEST_NAME, CLASS_NAME)
        report['mutations'] = mutations(root, output)
        if not all(row['detected'] and row['failures'] == 1 and row['errors'] == 0 for row in report['mutations']):
            raise RuntimeError('mutation survived or harness error occurred; no mutation credit')
        if SCALE:
            report['scale'] = scale(root, output)
        after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        if after != before:
            raise RuntimeError('source/test changed during controls; evidence is stale')
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
