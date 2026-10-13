"""Skeptic re-probe of TB-F3 against origin/main's checker. No network.

Run with INTAKE_CHECKER=<path to a copy of tools/public_intake_check.py>.
Reuses the fixture of tests/test_public_intake.py at origin/main (which serves
the /compare and /branches routes the reachability gate now requires).
"""
import importlib.util
import os
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = Path(os.environ['INTAKE_CHECKER'])
spec = importlib.util.spec_from_file_location('intake', CHECKER)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

# Borrow the current test module's fixture, bound to the same checker copy.
tspec = importlib.util.spec_from_file_location('test_public_intake_orig', HERE / 'test_public_intake_orig.py')
tmod = importlib.util.module_from_spec(tspec)
# The test module loads tools/public_intake_check.py relative to itself; redirect it.
import importlib.util as _iu
_real = _iu.spec_from_file_location
def _redirect(name, location, *a, **k):
    if name == 'intake':
        return _real(name, CHECKER, *a, **k)
    return _real(name, location, *a, **k)
_iu.spec_from_file_location = _redirect
try:
    tspec.loader.exec_module(tmod)
finally:
    _iu.spec_from_file_location = _real
assert tmod.intake.__file__ == str(CHECKER), tmod.intake.__file__

FINDER_VARIANTS = ['STATUS_v2.md', 'STATUSv2.md', 'claims.json', 'PROOF_INDEX-2.md']
DOCUMENTED = ['STATUS.md', 'PROOF_INDEX.md', 'LANDING_CLAIMS.json', 'claims/graph.json',
              'status.md', 'Status.MD', 'proof_index.json', 'landing_claims.csv', 'CLAIMS/x.json', 'notes/claims/y.md']
LEGITIMATE_PREFIX_NAMES = ['status_table.csv', 'status-notes.md', 'statuses.md', 'proof_index_review.md',
                           'landing_claims_commentary.md', 'claims_review.md', 'STATUS_OF_RUN.txt']


class ReProbe(tmod.IntakeTests):
    def _with(self, name, content=b'{}\n'):
        old = self.raw.copy()
        self.raw[name] = content
        self.save()
        try:
            return self.run_check()
        finally:
            self.raw = old

    def probe_documented_names_are_refused(self):
        # The names CONTRIBUTING actually lists (STATUS.md, PROOF_INDEX, LANDING_CLAIMS, claims/), any case.
        for name in DOCUMENTED:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                self.reject('protected')
                self.raw = old

    def probe_finder_variants_accepted(self):
        # The finder's four names: not in the documented list; accepted on the fresh checker.
        for name in FINDER_VARIANTS:
            with self.subTest(name=name):
                r = self._with(name)
                self.assertEqual(r['lane'], 'incoming')
                self.assertEqual(r['scientific_effect'], 'NONE')

    def probe_legitimate_prefix_names_accepted(self):
        # Collateral check for the proposed prefix patch: ordinary contributor names sharing a prefix.
        for name in LEGITIMATE_PREFIX_NAMES:
            with self.subTest(name=name):
                r = self._with(name, b'a,b\n1,2\n' if name.endswith('.csv') else b'note\n')
                self.assertEqual(r['lane'], 'incoming')


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(ReProbe(n) for n in sorted(n for n in dir(ReProbe) if n.startswith('probe_')))
    return suite


if __name__ == '__main__':
    unittest.main()
