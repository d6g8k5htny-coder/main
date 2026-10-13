"""Refutation probe for TB-F3 (skeptic #3, lens trust-boundary). No network.

Placed next to tests/test_public_intake.py so it reuses that file's fixture and
therefore the checker at ../tools/public_intake_check.py of whichever layout it
runs in (current copy from origin/main, or the copy with the finder's patch).
"""
import unittest
from test_public_intake import IntakeTests

VARIANTS = ['STATUS_v2.md', 'STATUSv2.md', 'claims.json', 'PROOF_INDEX-2.md']            # finder's four
EXACT = ['STATUS.md', 'PROOF_INDEX.md', 'LANDING_CLAIMS.json', 'claims/graph.json',      # existing test coverage
         'STATUS.v2.md', 'status.MD', 'proof_index.txt', 'CLAIMS/x.json', 'sub/STATUS.md']  # case/first-dot/nested
LEGITIMATE = ['status_table.csv', 'STATUS-of-replay.md', 'statuses.json',
              'proof_index_notes.md', 'landing_claims_replay.csv', 'claims_summary.md']


class F3Probe(IntakeTests):
    def lane_with(self, name):
        old = self.raw.copy()
        self.raw[name] = b'{}\n'
        self.save()
        try:
            return self.run_check()['lane']
        finally:
            self.raw = old
            self.save()

    def test_f3_variant_names_accepted(self):
        for name in VARIANTS:
            with self.subTest(name=name):
                self.assertEqual(self.lane_with(name), 'incoming')

    def test_f3_exact_and_case_variants_refused(self):
        for name in EXACT:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                self.reject('protected')
                self.raw = old
                self.save()

    def test_f3_legitimate_names_accepted(self):
        for name in LEGITIMATE:
            with self.subTest(name=name):
                self.assertEqual(self.lane_with(name), 'incoming')


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(F3Probe(n) for n in sorted(dir(F3Probe)) if n.startswith('test_f3_'))
    return suite
