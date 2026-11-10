"""Real child-process origin controls for the Lean and Python auditor graders.

Only source copies in a temporary directory are modified. The selected body,
subject and standard unittest assertion helpers remain distinguishable by the
live traceback recorder; filenames and exception text are not authority.
"""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tests import cbz_auditor_origin

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('tests.test_cbz_layer_a_scan_mutants', 'tools/cbz/layer_a_scan.py',
     'test_native_decide_options', 'scan_text', 'return []'),
    ('tests.test_cbz_a3_float_census', 'tools/cbz/a3_float_census.py',
     'test_float_and_complex_calls', 'census_text',
     "return {'class': 'NO_FLOAT_SIGNAL', 'signals': []}"),
)


class AuditorOriginControls(unittest.TestCase):
    def test_real_subject_assertions_cannot_count_as_oracle_failures(self):
        for module, source, name, function, _ in CASES:
            grader = importlib.import_module(module)
            test_id = grader.designated_test_ids()[name]
            with self.subTest(grader=module), tempfile.TemporaryDirectory() as tmp:
                tests_dir = Path(tmp) / 'tests'
                tests_dir.mkdir()
                copy = tests_dir / 'subject.py'
                copy.write_bytes((ROOT / source).read_bytes() +
                    ('\n\ndef ' + function + '(*args, **kwargs):\n'
                     '    raise AssertionError("SUBJECT_ORIGIN")\n').encode())
                result = grader.run_suite(str(copy), test_id)
                self.assertFalse(grader.killed(result, name), result)

    def test_baselines_and_actual_selected_body_oracles(self):
        for module, source, name, function, replacement in CASES:
            grader = importlib.import_module(module)
            test_id = grader.designated_test_ids()[name]
            with self.subTest(grader=module), tempfile.TemporaryDirectory() as tmp:
                tests_dir = Path(tmp) / 'tests'
                tests_dir.mkdir()
                copy = tests_dir / 'subject.py'
                raw = (ROOT / source).read_bytes()
                copy.write_bytes(raw)
                baseline = grader.run_suite(str(copy), test_id)
                self.assertEqual(cbz_auditor_origin.outcome(baseline, name), 'baseline', baseline)
                self.assertFalse(grader.killed(baseline, name))
                copy.write_bytes(raw + ('\n\ndef ' + function + '(*args, **kwargs):\n'
                                        '    ' + replacement + '\n').encode())
                self.assertTrue(grader.killed(grader.run_suite(str(copy), test_id), name))

    def test_nested_subject_and_changed_source_are_refused(self):
        for module, source, name, function, replacement in CASES:
            grader = importlib.import_module(module)
            test_id = grader.designated_test_ids()[name]
            with self.subTest(grader=module), tempfile.TemporaryDirectory() as tmp:
                tests_dir = Path(tmp) / 'tests'
                tests_dir.mkdir()
                subject = tests_dir / 'subject.py'
                raw = (ROOT / source).read_bytes()
                nested = ('\n\ndef ' + function + '(*args, **kwargs):\n'
                          '    def callback():\n'
                          '        raise AssertionError("NESTED_SUBJECT")\n'
                          '    return callback()\n')
                subject.write_bytes(raw + nested.encode())
                report = grader.run_suite(str(subject), test_id)
                self.assertFalse(grader.killed(report, name), report)
                changed = ('\n\ndef ' + function + '(*args, **kwargs):\n'
                           '    with open(__file__, "ab") as f:\n'
                           '        f.write(b"\\n# source changed during child\\n")\n'
                           '    ' + replacement + '\n')
                subject.write_bytes(raw + changed.encode())
                report = grader.run_suite(str(subject), test_id)
                self.assertNotEqual(report.get('source_before'), report.get('source_after'))
                self.assertFalse(grader.killed(report, name), report)

    def test_evidence_is_source_bound_and_write_failures_are_not_hidden(self):
        module, source, name, function, replacement = CASES[0]
        grader = importlib.import_module(module)
        test_id = grader.designated_test_ids()[name]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / 'subject.py'
            subject.write_bytes((ROOT / source).read_bytes() +
                ('\n\ndef ' + function + '(*args, **kwargs):\n    ' + replacement + '\n').encode())
            evidence = root / 'evidence'
            evidence.mkdir()
            with mock.patch.dict(os.environ, {'CBZ_AUDITOR_EVIDENCE_DIR': str(evidence)}):
                report = grader.run_suite(str(subject), test_id)
            self.assertTrue(grader.killed(report, name), report)
            nonce = report['evidence_nonce']
            saved = json.loads((evidence / (nonce + '.json')).read_text())
            self.assertEqual(saved, report)
            for suffix in ('.before.py', '.after.py'):
                self.assertEqual(hashlib.sha256((evidence / (nonce + suffix)).read_bytes()).hexdigest(),
                                 report['source_before'])
            with mock.patch.dict(os.environ, {'CBZ_AUDITOR_EVIDENCE_DIR': str(root / 'missing')}):
                with self.assertRaises(FileNotFoundError):
                    grader.run_suite(str(subject), test_id)
            subject.write_bytes((ROOT / source).read_bytes())
            with mock.patch.dict(os.environ, {'CBZ_AUDITOR_EVIDENCE_DIR': str(evidence)}):
                full = grader.run_suite(str(subject))
            self.assertEqual(full.returncode, 0, full.stderr)
            records = list((evidence / 'full').glob('*.json'))
            self.assertEqual(len(records), 1)
            full_record = json.loads(records[0].read_text())
            self.assertEqual(full_record['exit_code'], full.returncode)
            self.assertEqual(full_record['stderr_sha256'], hashlib.sha256(full.stderr.encode()).hexdigest())
            self.assertEqual((evidence / 'full' / (full_record['nonce'] + '.stderr')).read_text(), full.stderr)

    def test_missing_or_inconsistent_report_provenance_is_refused(self):
        module, source, name, function, replacement = CASES[0]
        grader = importlib.import_module(module)
        test_id = grader.designated_test_ids()[name]
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / 'subject.py'
            subject.write_bytes((ROOT / source).read_bytes() +
                ('\n\ndef ' + function + '(*args, **kwargs):\n    ' + replacement + '\n').encode())
            good = grader.run_suite(str(subject), test_id)
            self.assertTrue(grader.killed(good, name), good)
            variants = []
            for field in ('source_before', 'source_after', 'selected_id', 'events'):
                bad = copy.deepcopy(good)
                bad.pop(field)
                variants.append(bad)
            bad = copy.deepcopy(good)
            bad['source_after'] = '0' * 64
            variants.append(bad)
            for fake in ('g' * 64, '0' * 63):
                bad = copy.deepcopy(good)
                bad['source_before'] = bad['source_after'] = fake
                variants.append(bad)
            bad = copy.deepcopy(good)
            bad['events'][0]['authorized'] = False
            variants.append(bad)
            bad = copy.deepcopy(good)
            bad['classification'] = 'baseline'
            variants.append(bad)
            for bad in variants:
                with self.subTest(bad=bad):
                    self.assertFalse(grader.killed(bad, name))

    def test_traceback_text_without_live_provenance_is_not_a_kill(self):
        fake = subprocess.CompletedProcess([], 1, stdout='', stderr=(
            '=' * 70 + '\nFAIL: test_expected (fixture.Test.test_expected)\n'
            '  File "/tmp/tests/fake.py", line 1, in test_expected\n'
            'AssertionError: fake\n' + '-' * 70 +
            '\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n'))
        for module, _, _, _, _ in CASES:
            with self.subTest(grader=module):
                self.assertFalse(importlib.import_module(module).killed(fake, 'test_expected'))


if __name__ == '__main__':
    unittest.main()
