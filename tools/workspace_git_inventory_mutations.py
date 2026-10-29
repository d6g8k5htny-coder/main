#!/usr/bin/env python3
"""Verify the Git census baseline and four adverse mutations, not Git cleanliness.

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


CASES = [
    ('exclude-node-modules', "if entry.name == '.git':", "if entry.name in ('.git', 'node_modules'):",
     'test_nested_linked_state_refs_stash_and_redacted_remotes'),
    ('disable-redaction', 'def _redact(value):\n', 'def _redact(value):\n    return value\n',
     'test_nested_linked_state_refs_stash_and_redacted_remotes'),
    ('execute-clean-filter', "settings += [prefix + '.clean=', prefix + '.process=', prefix + '.required=false']",
     'settings += []', 'test_status_does_not_execute_configured_clean_filter'),
    ('false-complete', 'complete=not reader.errors', 'complete=True',
     'test_broken_marker_and_external_directory_boundary_are_not_silent'),
]

DRIVER = r'''
import importlib.util, json, os, pathlib, sys, unittest
test, tool, result, name = sys.argv[1:]
spec = importlib.util.spec_from_file_location('census_tests', test)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TOOL = pathlib.Path(tool)
suite = (unittest.defaultTestLoader.loadTestsFromName('InventoryTests.' + name, module)
         if name else unittest.defaultTestLoader.loadTestsFromModule(module))
run = unittest.TextTestRunner(verbosity=2).run(suite)
summary = {'tests_run': run.testsRun, 'successful': run.wasSuccessful(),
           'failures': len(run.failures), 'errors': len(run.errors),
           'skipped': len(run.skipped), 'expected_failures': len(run.expectedFailures),
           'unexpected_successes': len(run.unexpectedSuccesses)}
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
    with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        child = subprocess.Popen([sys.executable, '-S', '-B', '-c', DRIVER,
                                  str(test), str(tool), str(metrics), name],
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
        baseline['passed'] = (baseline.get('returncode') == 0 and baseline.get('tests_run') == 11 and
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
                        row.get('failures') == 1 and row.get('successful') is False and
                        all(row.get(key) == 0 for key in
                            ('errors', 'skipped', 'expected_failures', 'unexpected_successes')) and 'error' not in row)
                    report['mutations'].append(row)
        else:
            report['errors'].append('Baseline did not pass all 11 tests without skips/errors; no mutations run')
        for label, source in report['sources'].items():
            if _sha(Path(source['path']).read_bytes()) != source['sha256']:
                report['errors'].append('Source changed during verification: ' + label)
        report['all_detected'] = (len(report['mutations']) == 4 and
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
