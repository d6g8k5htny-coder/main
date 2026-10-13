"""Skeptic #2 (trust-boundary) re-check of TB-F3. No network.

Run from a folder that holds copies of tools/public_intake_check.py and
tests/test_public_intake.py:   cd <copy>; python3 -B -m unittest -v probe_refute_tb_f3
It reuses that copy's IntakeTests fixture (fake API dict of routes) so it also
serves the reachability routes the origin/main checker asks for.
"""
import os
import sys
import unittest

sys.path.insert(0, os.getcwd())
from tests.test_public_intake import IntakeTests, intake, REPO  # noqa: E402


def check(fixture):
    return intake.check(fixture.event, fixture.api, REPO)


class Observation(IntakeTests):
    """Reproduces the finder's observation on this copy: stem variants pass."""

    def test_F3_variant_names_pass_on_current(self):
        for name in ['STATUS_v2.md', 'STATUSv2.md', 'claims.json', 'PROOF_INDEX-2.md']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                result = check(self)
                self.assertEqual(result['lane'], 'incoming')
                self.assertEqual(result['scientific_effect'], 'NONE')
                self.raw = old


class ContractAsWritten(IntakeTests):
    """What the code, its introducing commit (349e036) and the existing control
    test_protected_status_and_claim_paths_refused actually promise: the exact
    stems of the repository's scientific status files, in any case, with any
    extension or multi-dot suffix, as a file or directory, at any depth inside
    the package, plus any path segment named claims."""

    def test_exact_stems_refused_any_case_extension_depth(self):
        for name in ['STATUS.md', 'status.md', 'Status.json', 'STATUS.csv', 'STATUS.v2.md',
                     'STATUS/notes.md', 'status/summary.md', 'notes/STATUS.md', 'PROOF_INDEX.txt', 'proof_index.md',
                     'LANDING_CLAIMS.md', 'landing_claims.json', 'claims/x.md', 'CLAIMS/x.md',
                     'a/b/claims/x.md']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                with self.assertRaisesRegex(ValueError, 'protected scientific status path'):
                    check(self)
                self.raw = old

    def test_repository_level_protected_files_are_unreachable_from_a_package(self):
        # The names CONTRIBUTING lists are repository-level paths; every one is
        # refused earlier by the incoming-only rule regardless of the stem rule.
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        for path in ['STATUS.md', 'PROOF_INDEX.md', 'claims/LANDING_CLAIMS.json', 'claims/graph.json',
                     'STATUS_v2.md', 'docs/site/status.json']:
            with self.subTest(path=path):
                rows[0]['filename'] = path
                with self.assertRaisesRegex(ValueError, 'incoming-only edits required'):
                    check(self)


class LegitimateNames(IntakeTests):
    """Ordinary result-package names that merely start with a protected word.
    They pass on the current checker; the finder's optional prefix patch would
    refuse every one of them (a wrong rejection, MINOR by the house scale)."""

    def test_prefixed_ordinary_names_accepted_on_current(self):
        for name in ['status_table.csv', 'status-by-lemma.md', 'statuses.json', 'statusline.txt',
                     'proof_index_notes.md', 'landing_claims_review.md']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                self.assertEqual(check(self)['lane'], 'incoming')
                self.raw = old


if __name__ == '__main__':
    unittest.main()
