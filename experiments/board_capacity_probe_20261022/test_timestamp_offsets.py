"""Timestamp-boundary regression for the isolated board-capacity prototype.

Run: python -B -S test_timestamp_offsets.py path/to/board_capacity_probe.py
Only local imports/temporary inputs and bounded child processes are used.
The 20,000 signed HH:MM forms are an exhaustive two-digit lexical grid, not
20,000 live board observations or a claim of full RFC3339 implementation.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(sys.argv.pop(1)).resolve()
ORIGINAL_BYTES = SOURCE.read_bytes()
spec = importlib.util.spec_from_file_location('capacity_under_test', SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
URL = 'https://github.com/d6g8k5htny-coder/main/issues/475'
NOW = '2026-10-21T22:30:00Z'
REPORT = {'source_sha256': hashlib.sha256(ORIGINAL_BYTES).hexdigest(),
          'optimized': bool(sys.flags.optimize)}


def snapshot(observed='2026-10-21T22:29:00Z'):
    return {'observed_at': observed, 'issue': {
        'url': 'https://api.github.com/repos/d6g8k5htny-coder/main/issues/475',
        'repository_url': 'https://api.github.com/repos/d6g8k5htny-coder/main',
        'html_url': URL, 'id': 5966843259, 'number': 475,
        'state': 'open', 'locked': False, 'comments': 193}}


class TimestampBoundaries(unittest.TestCase):
    def assert_unknown_time(self, result):
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['capacity'], 'UNKNOWN')
        self.assertEqual(result['reasons'], ['INVALID_TIME'])
        self.assertTrue(result['attention_required'])
        self.assertEqual(result['posting_permission'], 'NOT_ESTABLISHED')
        self.assertEqual(probe.exit_code(result), 2)

    def cli(self, value, now=NOW, extra=()):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.json'
            raw = (json.dumps(value, sort_keys=True) + '\n').encode()
            path.write_bytes(raw)
            command = [sys.executable, '-B', '-S']
            if sys.flags.optimize:
                command.append('-O')
            command += [str(SOURCE), str(path), '--active-url', URL, '--now', now, *extra]
            result = subprocess.run(command, capture_output=True, timeout=5,
                                    env={'PATH': os.defpath, 'LC_ALL': 'C.UTF-8'})
            self.assertEqual(result.stderr, b'')
            self.assertTrue(result.stdout.strip(), 'missing JSON output')
            data = json.loads(result.stdout)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(data['input_sha256'], hashlib.sha256(raw).hexdigest())
            return result, data

    def test_exhaustive_two_digit_signed_offset_grid(self):
        false_accepts, false_rejects, wrong_instants = [], [], []
        valid = invalid = 0
        local = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
        for sign, direction in [('+', 1), ('-', -1)]:
            for hours in range(100):
                for minutes in range(100):
                    offset = f'{sign}{hours:02}:{minutes:02}'
                    stamp = '2026-10-21T12:00:00' + offset
                    allowed = hours < 24 and minutes < 60
                    valid += int(allowed)
                    invalid += int(not allowed)
                    try:
                        actual = probe._time(stamp)
                    except (ValueError, OverflowError):
                        if allowed:
                            false_rejects.append(offset)
                    else:
                        if not allowed:
                            false_accepts.append(offset)
                        elif actual != local - direction * timedelta(hours=hours, minutes=minutes):
                            wrong_instants.append(offset)
        REPORT['grid'] = {'total': valid + invalid, 'valid': valid, 'invalid': invalid,
                          'false_accepts': len(false_accepts), 'false_rejects': len(false_rejects),
                          'wrong_instants': len(wrong_instants),
                          'first_false_accepts': false_accepts[:8]}
        self.assertEqual((valid, invalid), (2880, 17120))
        self.assertEqual((len(false_accepts), len(false_rejects), len(wrong_instants)),
                         (0, 0, 0), REPORT['grid'])

    def test_invalid_observation_offset_is_not_normalized(self):
        for observed, now in [('2026-10-21T23:29:00+00:60', NOW),
                              ('2026-10-21T23:29:00+00:99', '2026-10-21T21:51:00Z'),
                              ('2026-10-21T21:29:00-00:60', NOW)]:
            with self.subTest(observed=observed):
                self.assert_unknown_time(probe.assess(snapshot(observed), URL, now))

    def test_invalid_now_offset_is_not_normalized(self):
        for now in ('2026-10-21T23:30:00+00:60', '2026-10-21T21:30:00-00:60'):
            with self.subTest(now=now):
                self.assert_unknown_time(probe.assess(snapshot(), URL, now))

    def test_valid_instants_and_freshness_edges_are_preserved(self):
        for observed in ('2026-10-21T22:29:00Z', '2026-10-21T23:29:00+01:00',
                         '2026-10-21T21:29:00-01:00', '2026-10-21T22:29:00-00:00',
                         '2026-10-21T23:28:00+00:59', '2026-10-21T21:30:00-00:59'):
            with self.subTest(observed=observed):
                got = probe.assess(snapshot(observed), URL, NOW)
                self.assertEqual((got['status'], got['capacity'], got.get('age_seconds')),
                                 ('OK', 'NORMAL', 60))
                self.assertEqual(got['posting_permission'], 'NOT_ESTABLISHED')
        exact = probe.assess(snapshot('2026-10-21T22:15:00Z'), URL, NOW)
        stale = probe.assess(snapshot('2026-10-21T22:14:59Z'), URL, NOW)
        future = probe.assess(snapshot('2026-10-21T22:30:01Z'), URL, NOW)
        self.assertEqual(exact['status'], 'OK')
        self.assertEqual(stale['reasons'], ['STALE_SNAPSHOT'])
        self.assertEqual(future['reasons'], ['FUTURE_SNAPSHOT'])

    def test_cli_refuses_both_timestamp_inputs(self):
        for value, now in [(snapshot('2026-10-21T23:29:00+00:60'), NOW),
                           (snapshot(), '2026-10-21T23:30:00+00:60'),
                           (snapshot('2026-10-21T21:29:00-00:60'), NOW),
                           (snapshot(), '2026-10-21T21:30:00-00:60')]:
            with self.subTest(value=value, now=now):
                result, got = self.cli(value, now)
                self.assertEqual(result.returncode, 2)
                self.assert_unknown_time(got)

    def test_cli_valid_offset_and_unknown_count_boundary_unchanged(self):
        result, got = self.cli(snapshot('2026-10-21T23:28:00+00:59'))
        self.assertEqual(result.returncode, 0)
        self.assertEqual((got['status'], got['age_seconds']), ('OK', 60))
        value = snapshot(); value['issue']['comments'] = True
        result, got = self.cli(value)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(got['reasons'], ['INVALID_ISSUE'])

    def test_argparse_non_json_limit_is_retained(self):
        command = [sys.executable, '-B', '-S']
        if sys.flags.optimize:
            command.append('-O')
        command += [str(SOURCE), 'not-opened.json']
        result = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'--active-url', result.stderr)

    def test_existing_narrow_timestamp_subset_unchanged(self):
        self.assertEqual(probe._time('2026-10-21T22:29:00.123456Z').microsecond, 123456)
        for stamp in ('2026-10-21T22:29:60Z', '2026-10-21t22:29:00z',
                      '2026-10-21T22:29:00.1234567Z', '2026-10-21T22:29:00Z\n',
                      '2026-02-30T22:29:00Z', '2026-10-21T22:29:00+24:00'):
            with self.subTest(stamp=stamp):
                with self.assertRaises((ValueError, OverflowError)):
                    probe._time(stamp)
        self.assertEqual(SOURCE.read_bytes(), ORIGINAL_BYTES)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TimestampBoundaries)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(tests=outcome.testsRun, failures=len(outcome.failures),
                  errors=len(outcome.errors), skipped=len(outcome.skipped),
                  source_unchanged=SOURCE.read_bytes() == ORIGINAL_BYTES)
    print(json.dumps(REPORT, sort_keys=True))
    raise SystemExit(0 if outcome.wasSuccessful() and REPORT['source_unchanged'] else 1)
