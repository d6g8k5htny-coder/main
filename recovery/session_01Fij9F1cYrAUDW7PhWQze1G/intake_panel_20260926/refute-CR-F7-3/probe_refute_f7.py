"""Skeptic controls for CR-F7. No network. INTAKE_MODE=current|patched selects the checker copy.

A: the exact names CONTRIBUTING and the house suite protect stay refused in both modes (positive control).
B: ordinary names that merely START with a protected stem are legitimate package content; the finder's
   prefix patch refuses them (a fail-closed regression introduced by the patch).
C: the package id itself (parts[1]) is untouched by the finder's patch: incoming/claims/ and incoming/status/
   are green in BOTH modes, so the patch does not even close the finding's own second observation.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()


class Controls(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = house.IntakeTests.install_source
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def test_A_exact_protected_names_refused_both_modes(self):
        for name in ['STATUS.md', 'status.txt', 'Status.json', 'PROOF_INDEX.md', 'proof_index.csv',
                     'LANDING_CLAIMS.json', 'claims/graph.json', 'data/CLAIMS/x.md', 'STATUS', 'status/run1.txt']:
            with self.subTest(name=name):
                old = self.raw.copy()
                try:
                    self.raw[name] = b'{"x": 1}\n' if name.endswith('.json') else b'x\n'
                    self.save()
                    self.reject('protected')
                finally:
                    self.raw = old

    def test_B_legitimate_prefix_names(self):
        # Plausible artifact names in a replay package. Current: green. Finder's patch: refused.
        for name in ['status_log.txt', 'statuses.csv', 'claims_summary.md', 'claims_of_the_paper.md',
                     'proof_index_notes.md', 'landing_claims_diff.md']:
            with self.subTest(name=name):
                old = self.raw.copy()
                try:
                    self.raw[name] = b'x\n'
                    self.save()
                    if MODE == 'current':
                        self.green()
                    else:
                        self.reject('protected')
                finally:
                    self.raw = old

    def test_C_package_id_unchecked_in_both_modes(self):
        for pkg in ['claims', 'status', 'proof-index', 'landing-claims']:
            with self.subTest(pkg=pkg):
                self.package = f'incoming/{pkg}/'
                self.save()
                result = self.green()
                self.assertEqual(result['package'], f'incoming/{pkg}')


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
