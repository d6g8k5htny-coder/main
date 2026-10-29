"""Tests for the scanner mutation self-test itself (the auditor's auditor).

They check that the harness fails closed: a mutation that no longer applies
(stale) and a mutation that the scanner tests do not detect (survivor) both
make the gate fail.  The real mutant list is exercised by running the tool.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'cbz_scanner_mutation_check_under_test', ROOT / 'tools' / 'cbz_scanner_mutation_check.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HarnessFailsClosed(unittest.TestCase):
    def test_stale_mutation_fails_the_gate(self):
        with mock.patch.object(harness, 'MUTANTS', [('stale', 'text that is not in the scanner', 'x')]):
            self.assertEqual(harness.main(['--json']), 1)

    def test_surviving_mutant_fails_the_gate(self):
        # Rewording the module docstring changes no behaviour, so the tests pass on it.
        with mock.patch.object(harness, 'MUTANTS',
                               [('docstring-only', 'Layer A static scanner for Lean sources',
                                 'Layer A static scanner (reworded)')]):
            self.assertEqual(harness.main(['--json']), 1)

    def test_real_mutant_is_killed(self):
        with mock.patch.object(harness, 'MUTANTS', [harness.MUTANTS[0]]):
            self.assertEqual(harness.main(['--json']), 0)

    def test_every_listed_mutant_applies_exactly_once(self):
        source = harness.SCANNER.read_text(encoding='utf-8')
        for name, old, _ in harness.MUTANTS:
            with self.subTest(name=name):
                self.assertEqual(source.count(old), 1)


if __name__ == '__main__':
    unittest.main()
