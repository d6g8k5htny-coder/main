"""Source-identity assembly, never mathematical acceptance."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from tools.atlas_formal_evidence_build import build_connection

ROOT = Path(__file__).resolve().parents[1]
CLAIM = 'math.side24-coefficient'
FIXTURE = ROOT/'tests/fixtures/atlas_formal_build_3b39'
HISTORICAL_MANIFEST = '8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090'
HISTORICAL_FILES = {
    'tools/formal_gate_check.py': ('formal_gate_check.py.txt',
        '21c355e02ac6e01337d10465a3105b34b93f832586ee60aa2430f662a90cd8b7'),
    'tests/test_formal_gate.py': ('test_formal_gate.py.txt',
        '4286136d8b24eca9b26d1e7b12933bed5b0961809b362a60c3af7bf46e6ed4b6'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FormalConnectionAssembly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        manifest_raw = (FIXTURE/'manifest.json').read_bytes()
        self.assertEqual(digest(manifest_raw), HISTORICAL_MANIFEST)
        manifest = json.loads(manifest_raw)
        for relative, expected in manifest['files'].items():
            fixture = HISTORICAL_FILES.get(relative)
            raw = (FIXTURE/fixture[0]).read_bytes() if fixture else (ROOT/relative).read_bytes()
            self.assertEqual(digest(raw), expected, relative)
            destination = self.root/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (self.root/'formal/manifest.json').write_bytes(manifest_raw)
        for relative in (
            'docs/site/dependency-source/GRAPH.json',
            'docs/site/evidence/side24-main-3b39/review.json',
            'docs/site/evidence/side24-main-3b39/review.md',
        ):
            raw = (ROOT/relative).read_bytes()
            destination = self.root/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        self.snapshot = json.loads((ROOT/'tests/fixtures/atlas_formal_projection.json').read_bytes())
        self.proof = (ROOT/'formal/sources/side24_v1/PROOF.md').read_bytes()
        self.artifact = (ROOT/'docs/site/evidence/side24-main-3b39/verified-execution.zip').read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_connection(self.root, self.snapshot, self.proof, self.artifact)

    def test_live_manifest_drift_is_refused_before_historical_evidence_is_rebound(self):
        with self.assertRaisesRegex(ValueError, 'formal manifest changed'):
            build_connection(ROOT, self.snapshot, self.proof, self.artifact)

    def test_exact_proof_source_connects_to_original_checked_closed_declaration(self):
        result = self.build()
        self.assertIsInstance(result, dict, 'A verified source connection is not assembled yet')
        self.assertEqual(result['claim']['id'], CLAIM)
        self.assertEqual(result['declaration']['name'], 'UniversalLaw.Side24.pairing_sum_six')
        self.assertIn('pairingSum 6 = 76', result['declaration']['source_statement'])
        self.assertIn('@Eq Nat', result['declaration']['elaborated_type'])
        self.assertEqual(result['declaration']['hypothesis_binders'], [])
        self.assertEqual(result['execution']['checked_commit'], '3b39c42d6edcc98ae165270be346f0e31c57f81d')
        self.assertEqual(result['execution']['receipt_sha256'], '1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0')
        self.assertEqual(result['formal']['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
        self.assertIn('derivative bound', result['declaration']['does_not_claim'])

    def test_actual_adverse_review_is_retained_without_turning_one_aligned_row_into_package_acceptance(self):
        review = self.build()['alignment']
        self.assertIsNotNone(review, 'The actual adverse review must not be hidden')
        self.assertEqual(review['record']['disposition'], 'REJECTED')
        self.assertEqual(review['selected_target_verdict'], 'ALIGNED')
        self.assertEqual(len(review['record']['targets']), 31)
        self.assertEqual(review['record']['reviewer']['provider'], 'xAI')

    def test_changed_proof_bytes_are_refused(self):
        self.proof += b'\n'
        with self.assertRaisesRegex(ValueError, 'proof'):
            self.build()

    def test_changed_graph_scope_is_refused_even_with_same_fingerprint(self):
        self.snapshot['graph']['data']['nodes'][CLAIM]['scope'] = 'Full theorem'
        with self.assertRaisesRegex(ValueError, 'graph record'):
            self.build()

    def test_same_name_or_fingerprint_with_wrong_blob_is_refused(self):
        source = next(s for s in self.snapshot['sources'] if s['path'] == 'coefficients/side24_v1/PROOF.md')
        source['blob'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'proof inventory'):
            self.build()

    def test_modified_execution_archive_is_not_trusted(self):
        self.artifact += b'\n'
        with self.assertRaisesRegex(ValueError, 'artifact identity'):
            self.build()

    def test_missing_named_claim_cannot_use_an_equally_named_source(self):
        del self.snapshot['graph']['data']['nodes'][CLAIM]
        with self.assertRaisesRegex(ValueError, 'graph record'):
            self.build()


if __name__ == '__main__':
    unittest.main()
