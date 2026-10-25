"""Evidence reports must remain durable, scoped observations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from frontier_ir import compile_ir
try:
    from belief_ledger import BeliefLedger
except ImportError:
    BeliefLedger = None


class BeliefLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = compile_ir(json.loads((Path(__file__).parent / 'engine/dataset.json').read_text()))

    def setUp(self):
        self.assertIsNotNone(BeliefLedger, 'durable four-valued evidence ledger missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = BeliefLedger(self.tmp.name)

    def claim(self, polarity, **extra):
        return self.ledger.assert_claim('P', 'fixed domain', polarity, 'reviewer', **extra)

    def test_four_values_and_restart_preserve_provenance_without_certifying(self):
        snap = self.ledger.snapshot(self.ir)
        self.assertTrue(all(b['state'] == 'neither' for b in snap['beliefs']))
        self.claim('support', evidence={'report': 'first'})
        self.assertEqual(self.ledger.snapshot()['beliefs'][0]['state'], 'support')
        self.claim('refute', evidence={'report': 'second'})
        snap = BeliefLedger(self.tmp.name).snapshot()
        self.assertEqual(snap['beliefs'][0]['state'], 'both')
        self.assertEqual(len(snap['beliefs'][0]['evidence']), 2)
        self.assertFalse(snap['beliefs'][0]['certified'])
        self.assertEqual(snap['full_theorem_certificates'], 0)
        self.ledger.assert_claim('Q', 'fixed domain', 'refute', 'other')
        self.assertEqual({b['proposition']: b['state'] for b in self.ledger.snapshot()['beliefs']}['Q'], 'refute')

    def test_duplicate_evidence_is_idempotent_and_majority_does_not_resolve_both(self):
        first = self.claim('support', evidence={'x': 1})
        self.assertEqual(first['claim_id'], self.claim('support', evidence={'x': 1})['claim_id'])
        self.assertEqual(len(self.ledger.snapshot()['beliefs'][0]['evidence']), 1)
        self.claim('refute')
        for n in range(4):
            self.claim('support', evidence={'report': n})
        self.assertEqual(self.ledger.snapshot()['beliefs'][0]['state'], 'both')

    def test_identity_separates_scope_and_input_version(self):
        self.claim('support', input_fingerprint='a' * 64)
        self.claim('refute', input_fingerprint='b' * 64)
        self.ledger.assert_claim('P', 'other domain', 'refute', 'reviewer', input_fingerprint='a' * 64)
        self.assertEqual(sorted(b['state'] for b in self.ledger.snapshot()['beliefs']), ['refute', 'refute', 'support'])

    def test_only_exact_explicit_binding_affects_descendants(self):
        node = next(n for n in self.ir['nodes'] if n['id'] == 'D1-ELDER-MEAS')
        for polarity in ('support', 'refute'):
            self.ledger.assert_claim(node['statement'], node['scope'], polarity, 'reviewer',
                input_fingerprint=node['input_fingerprint'], record_id=node['id'])
        snap = self.ledger.snapshot(self.ir)
        investigation = snap['investigations'][0]
        self.assertIn('D1-A', investigation['affected_records'])
        self.assertNotIn('C6-F', investigation['affected_records'])
        self.assertTrue(all(not b['certified'] for b in snap['beliefs']))

    def test_unbound_and_mismatched_reports_never_invalidate_graph(self):
        node = self.ir['nodes'][0]
        for polarity in ('support', 'refute'):
            self.ledger.assert_claim(node['statement'], node['scope'], polarity, 'unbound', input_fingerprint=node['input_fingerprint'])
            self.ledger.assert_claim('different claim', node['scope'], polarity, 'wrong',
                input_fingerprint=node['input_fingerprint'], record_id=node['id'])
        self.assertTrue(all(not i['affected_records'] for i in self.ledger.snapshot(self.ir)['investigations']))

    def test_rejects_invalid_json_hash_polarity_and_modified_ir(self):
        for extra in ({'evidence': float('nan')}, {'input_fingerprint': 'bad'}, {'evidence': {'n': 2**60}}):
            with self.assertRaises(ValueError): self.claim('support', **extra)
        with self.assertRaises(ValueError): self.claim('theorem')
        bad = copy.deepcopy(self.ir)
        bad['nodes'][0]['certified'] = True
        with self.assertRaises(ValueError): self.ledger.snapshot(bad)


if __name__ == '__main__': unittest.main()
