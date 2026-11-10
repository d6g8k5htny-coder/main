"""Adverse controls for the one-off post-merge pin-log observation."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'verify_pin_log.py').read_text()
LOG = (HERE / 'pin-job-129740862312.log').read_text(encoding='utf-8')
PIN = 'info: repos/Math-: pin equals remote default main tip'
CHECKOUT = '5ee815d60d5e11cb590b2a3b274fd13104bd915d'


def execute(log, source=SOURCE):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'verify_pin_log.py').write_text(source)
        (root / 'pin-job-129740862312.log').write_text(log, encoding='utf-8')
        return subprocess.run([sys.executable, '-O' if not __debug__ else '-B',
                               str(root / 'verify_pin_log.py')],
                              capture_output=True, text=True)


class PinTranscriptControls(unittest.TestCase):
    def test_original_is_accepted(self):
        result = execute(LOG)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_checkout_is_rejected(self):
        self.assertNotEqual(execute(LOG.replace(CHECKOUT, '0' * 40)).returncode, 0)

    def test_duplicated_pin_is_rejected(self):
        self.assertIn(PIN, LOG)
        duplicate = LOG.replace(PIN, 'info: repos/main: pin equals remote default main tip')
        self.assertNotEqual(execute(duplicate).returncode, 0)

    def test_wrong_test_count_is_rejected(self):
        self.assertIn('Ran 258 tests in', LOG)
        self.assertNotEqual(execute(LOG.replace('Ran 258 tests in', 'Ran 257 tests in')).returncode, 0)

    def test_skip_is_rejected(self):
        adverse = LOG + "\n2026-11-10T09:25:11Z test_example ... skipped 'missing'\n"
        self.assertNotEqual(execute(adverse).returncode, 0)

    def test_missing_canonical_inputs_is_rejected(self):
        self.assertNotEqual(execute(LOG.replace('4 canonical inputs', '3 canonical inputs')).returncode, 0)

    def test_pin_guard_removal_reopens_duplicate(self):
        anchor = 'if len(actual) != 8 or set(actual) != expected:'
        self.assertEqual(SOURCE.count(anchor), 1)
        adverse = LOG.replace(PIN, 'info: repos/main: pin equals remote default main tip')
        result = execute(adverse, SOURCE.replace(anchor, 'if False:', 1))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_skip_guard_removal_reopens_skip(self):
        anchor = "if re.search(r'(?m)^\\S+ .* \\.\\.\\. (?:skipped|FAIL|ERROR)\\b|^\\S+ FAILED\\s*\\(|^\\S+ OK \\(skipped=', log):"
        self.assertEqual(SOURCE.count(anchor), 1)
        adverse = LOG + "\n2026-11-10T09:25:11Z test_example ... skipped 'missing'\n"
        result = execute(adverse, SOURCE.replace(anchor, 'if False:', 1))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
