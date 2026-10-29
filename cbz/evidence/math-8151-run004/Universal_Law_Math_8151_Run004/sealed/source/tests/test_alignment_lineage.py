"""Declared-lineage validation only; synthetic fixtures do not authenticate agents."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('lineage_gate_under_test', ROOT / 'gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

TARGETS = ['ResearchFormalCoreR1.first', 'ResearchFormalCoreR1.second']
MANIFEST, SCOPE = 'a' * 64, 'b' * 64


def valid_review():
    return {
        'disposition': 'ACCEPTED', 'manifest_sha256': MANIFEST,
        'scope_sha256': SCOPE, 'targets': list(TARGETS),
        'author': {'provider': 'Author Provider', 'family': 'Author Family', 'agent': 'author-session'},
        'reviewer': {'provider': 'Reviewer Provider', 'family': 'Reviewer Family', 'agent': 'review-session'},
        'evidence': {'repository': 'fixture/repository', 'path': 'reviews/fixture.md',
                     'commit': 'c' * 40, 'sha256': 'd' * 64},
    }


def proposer():
    return {'provider': 'Proposal Provider', 'family': 'Proposal Family', 'agent': 'proposal-session',
            'targets': [TARGETS[0]], 'pr': 'fixture#1'}


class AlignmentLineageTests(unittest.TestCase):
    def check(self, review):
        return gate.check_alignment(review, MANIFEST, TARGETS, SCOPE)

    def reject(self, review, reason='lineage'):
        with self.assertRaisesRegex(ValueError, reason):
            self.check(review)

    def test_legacy_single_author_record(self):
        self.check(valid_review())

    def test_known_distinct_proposer(self):
        r = valid_review(); r['proposal_authors'] = [proposer()]
        self.check(r)

    def test_explicit_empty_proposer_list(self):
        r = valid_review(); r['proposal_authors'] = []
        self.check(r)

    def test_unscoped_known_proposer(self):
        r = valid_review(); p = proposer(); del p['targets']; r['proposal_authors'] = [p]
        self.check(r)

    def test_every_declared_proposer_is_checked(self):
        r = valid_review(); p = proposer(); p['provider'] = r['reviewer']['provider']
        r['proposal_authors'] = [proposer(), p]
        self.reject(r)

    def test_same_proposer_provider(self):
        r = valid_review(); p = proposer(); p['provider'] = r['reviewer']['provider']
        r['proposal_authors'] = [p]; self.reject(r)

    def test_same_proposer_family(self):
        r = valid_review(); p = proposer(); p['family'] = r['reviewer']['family']
        r['proposal_authors'] = [p]; self.reject(r)

    def test_same_proposer_agent(self):
        r = valid_review(); p = proposer(); p['agent'] = r['reviewer']['agent']
        r['proposal_authors'] = [p]; self.reject(r)

    def test_proposer_identity_normalization(self):
        r = valid_review(); p = proposer(); p['provider'] = '  REVIEWER\tPROVIDER  '
        r['proposal_authors'] = [p]; self.reject(r)

    def test_primary_identity_normalization(self):
        r = valid_review(); r['author']['provider'] = '  REVIEWER\tPROVIDER  '
        self.reject(r)

    def test_placeholder_lineage_in_all_roles(self):
        for role in ('author', 'reviewer', 'proposal'):
            for key in ('provider', 'family', 'agent'):
                for value in ('UNKNOWN', ' unknown ', 'Unverified', 'UNSPECIFIED', 'TBD',
                              'N/A', 'NONE', 'null', '?', 'not known', 'Not Available'):
                    with self.subTest(role=role, key=key, value=value):
                        r = valid_review(); p = proposer(); r['proposal_authors'] = [p]
                        (p if role == 'proposal' else r[role])[key] = value
                        self.reject(r)

    def test_missing_proposer_identity_field(self):
        for key in ('provider', 'family', 'agent'):
            with self.subTest(key=key):
                r = valid_review(); p = proposer(); del p[key]; r['proposal_authors'] = [p]
                self.reject(r)

    def test_blank_or_nonstring_proposer_identity(self):
        for value in ('', ' \t\n ', None, 7, [], {}):
            with self.subTest(value=value):
                r = valid_review(); p = proposer(); p['provider'] = value
                r['proposal_authors'] = [p]; self.reject(r)

    def test_malformed_primary_and_reviewer_objects(self):
        for role in ('author', 'reviewer'):
            for value in (None, [], 'provider', 7):
                with self.subTest(role=role, value=value):
                    r = valid_review(); r[role] = value; self.reject(r)

    def test_malformed_proposer_collection(self):
        for value in (None, {}, 'provider', 7):
            with self.subTest(value=value):
                r = valid_review(); r['proposal_authors'] = value; self.reject(r)

    def test_malformed_proposer_object(self):
        for value in (None, [], 'provider', 7):
            with self.subTest(value=value):
                r = valid_review(); r['proposal_authors'] = [value]; self.reject(r)

    def test_invalid_proposer_target_scope(self):
        for value in ([], [TARGETS[0], TARGETS[0]], ['Unreviewed.target'],
                      None, TARGETS[0], [1], [{}]):
            with self.subTest(value=value):
                r = valid_review(); p = proposer(); p['targets'] = value
                r['proposal_authors'] = [p]; self.reject(r, 'proposal target')

    def test_withheld_record_remains_rejected(self):
        r = valid_review(); r['disposition'] = 'WITHHELD_PROPOSER_LINEAGE_UNKNOWN'
        p = proposer(); p['provider'] = 'UNKNOWN'; r['proposal_authors'] = [p]
        self.reject(r, 'alignment not accepted')

    def test_disposition_only_edit_cannot_hide_unknown_proposer(self):
        r = valid_review(); p = proposer(); p['provider'] = p['family'] = 'UNKNOWN'
        r['proposal_authors'] = [p]
        self.reject(r)

    def test_input_not_mutated(self):
        r = valid_review(); r['proposal_authors'] = [proposer()]; original = copy.deepcopy(r)
        self.check(r); self.assertEqual(r, original)

    def test_existing_digest_scope_and_coverage_checks(self):
        for key, value in (('manifest_sha256', 'e' * 64), ('scope_sha256', 'f' * 64),
                           ('targets', [TARGETS[0]])):
            with self.subTest(key=key):
                r = valid_review(); r[key] = value
                with self.assertRaises(ValueError): self.check(r)


    def assert_required_lineage_file(self, omitted):
        # Fail at the required-file inventory before accessing any source bytes.
        required = {'gate.py', 'tests/test_gate.py', 'lean-toolchain', 'lakefile.toml',
                    'lake-manifest.json', 'ResearchFormalCoreR1.lean', 'SCOPE.md',
                    'GLOSSARY.md', 'README.md', 'blueprint/src/content.tex',
                    'tests/test_alignment_lineage.py', 'LINEAGE_VALIDATION.md'}
        files = {name: '0' * 64 for name in required if name != omitted}
        files.update({
            'originals/Algebra.lean.txt': '4c196820c4db8fafc288dd35642828ba577e3544d60aba7d1d6d24f14ad1e8ae',
            'originals/ProbabilityCompanions.lean.txt': '4ace6600a476859982c8851ae9097c89b082d3c96291b3c8330bd4cc00bae65d',
            'COMPATIBILITY.md': '0' * 64,
        })
        manifest = {'schema_version': 1, 'scientific_effect': 'NONE',
                    'formalization_status': 'proved', 'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
                    'files': files}
        old_root = gate.ROOT
        with tempfile.TemporaryDirectory() as directory:
            gate.ROOT = Path(directory)
            try:
                (gate.ROOT / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'unbound control or scope file'):
                    gate.source_check()
            finally:
                gate.ROOT = old_root

    def test_lineage_tests_cannot_be_unbound(self):
        self.assert_required_lineage_file('tests/test_alignment_lineage.py')

    def test_lineage_scope_cannot_be_unbound(self):
        self.assert_required_lineage_file('LINEAGE_VALIDATION.md')

if __name__ == '__main__':
    unittest.main()
