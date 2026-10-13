"""Skeptic re-probe of CR-F7 (protected-name check is exact-stem). No network.

INTAKE_TREE   -> directory holding tools/public_intake_check.py and tests/test_public_intake.py
INTAKE_EXPECT -> 'green'   : assert the current checker accepts the finder's variants (and the
                             legitimate English names that the finder's prefix patch would refuse)
                 'refused' : assert a prefix-patched checker refuses both groups

The house fixture (FakeAPI, setUp/save/install/run_check/reject) is reused verbatim.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

TREE = Path(os.environ['INTAKE_TREE']).resolve()
EXPECT = os.environ.get('INTAKE_EXPECT', 'green')
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == TREE / 'tools/public_intake_check.py', intake.__file__

FINDER_VARIANTS = ['STATUS_v2.md', 'STATUS-final.md', 'STATUS .md', 'claims.json', 'CLAIMS.md',
                   'PROOF_INDEX2.md', 'LANDING_CLAIMS_new.json']
# Ordinary research-shop names that a prefix rule on STATUS/CLAIMS also refuses.
LEGITIMATE_PREFIXED = ['status-notes.md', 'status_of_replay.txt', 'claims-checked.csv',
                       'claimsummary.md']
EXACT_PROTECTED = ['STATUS.md', 'STATUS.txt', 'STATUS.tar.gz.md', 'PROOF_INDEX.md', 'LANDING_CLAIMS.json',
                   'claims/graph.json', 'data/claims/x.md', 'sub/STATUS.md', 'claims', 'status/summary.txt', 'Claims/x.md']


class F7(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = getattr(house.IntakeTests, 'install_source', None)
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def outcome(self):
        if EXPECT == 'green':
            result = self.run_check()
            self.assertEqual(result['lane'], 'incoming')
            self.assertEqual(result['scientific_effect'], 'NONE')
        else:
            self.reject('protected')

    def with_file(self, name, raw=b'ACCEPT\n'):
        old = self.raw.copy()
        try:
            self.raw[name] = b'{"status": "ACCEPT"}\n' if name.endswith('.json') else raw
            self.save()
            self.outcome()
        finally:
            self.raw = old

    def test_finder_variants(self):
        for name in FINDER_VARIANTS:
            with self.subTest(name=name):
                self.with_file(name)

    def test_legitimate_prefixed_names(self):
        # green on the current checker; a prefix patch turns these into wrong rejections.
        for name in LEGITIMATE_PREFIXED:
            with self.subTest(name=name):
                self.with_file(name, b'notes\n')

    def test_exact_protected_refused_in_both_modes(self):
        for name in EXACT_PROTECTED:
            with self.subTest(name=name):
                old = self.raw.copy()
                try:
                    self.raw[name] = b'changed\n'
                    self.save()
                    self.reject('protected')
                finally:
                    self.raw = old

    def test_package_id_claims_is_green_in_both_modes(self):
        # parts[1] is never name-checked, before or after the finder's patch.
        self.package = 'incoming/claims/'
        self.save()
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['package'], 'incoming/claims')

    def test_real_root_status_paths_refused_in_both_modes(self):
        # The stated guarantee: an intake PR cannot write the real status paths.
        rows = self.api.routes[f'/repos/{house.REPO}/pulls/7/files?per_page=100&page=1']
        for path in ['STATUS.md', 'PROOF_INDEX.md', 'claims/LANDING_CLAIMS.json', 'claims/graph.json', 'STATUS_v2.md']:
            with self.subTest(path=path):
                saved = rows[0]['filename']
                rows[0]['filename'] = path
                try:
                    self.reject('incoming-only')
                finally:
                    rows[0]['filename'] = saved


if __name__ == '__main__':
    print(f'[probe] tree={TREE} expect={EXPECT}', file=sys.stderr)
    unittest.main()
