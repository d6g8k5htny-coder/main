import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from collaboration import AnnotationCRDT, AnnotationStore
except ImportError:
    AnnotationCRDT = AnnotationStore = None


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(AnnotationCRDT, 'annotation-only CRDT is missing')

    def crdt(self, actor): return AnnotationCRDT(actor, record_ids=['D1', 'D5'])

    def test_concurrent_texts_and_later_observed_replacement(self):
        a, b = self.crdt('alice'), self.crdt('bob')
        a.set('note1', 'D1', 'First argument')
        b.set('note1', 'D1', 'Alternative argument')
        a.merge(b)
        self.assertEqual(len(a.snapshot()['annotations'][0]['versions']), 2)
        a.set('note1', 'D1', 'Both checked')
        versions = a.snapshot()['annotations'][0]['versions']
        self.assertEqual([v['text'] for v in versions], ['Both checked'])

    def test_tombstone_observes_only_prior_versions_add_wins(self):
        a, b = self.crdt('alice'), self.crdt('bob')
        a.set('n', 'D1', 'old'); b.merge(a)
        a.delete('n'); b.set('n', 'D1', 'concurrent new')
        a.merge(b); b.merge(a)
        self.assertEqual(a.snapshot()['annotations'], b.snapshot()['annotations'])
        self.assertEqual(a.snapshot()['annotations'][0]['versions'][0]['text'], 'concurrent new')
        a.delete('n'); a.merge(b)
        self.assertEqual(a.snapshot()['annotations'], [])

    def test_union_commutative_associative_idempotent_and_json(self):
        a, b, c = [self.crdt(x) for x in ('a', 'b', 'c')]
        for replica in (a, b, c): replica.set('n', 'D5', replica.actor)
        x, y = self.crdt('x'), self.crdt('y')
        x.merge(a); x.merge(b); x.merge(c)
        b.merge(c); y.merge(b); y.merge(a); y.merge(a)
        self.assertEqual(x.snapshot()['operations'], y.snapshot()['operations'])
        self.assertEqual(x.snapshot()['annotations'], y.snapshot()['annotations'])
        json.dumps(x.snapshot(), allow_nan=False)

    def test_duplicate_actor_counter_different_payload_rejects_atomically(self):
        a, b = self.crdt('shared'), self.crdt('shared')
        a.set('n', 'D1', 'one'); b.set('n', 'D1', 'two')
        before = a.snapshot()
        with self.assertRaisesRegex(ValueError, 'dot|counter|duplicate'): a.merge(b)
        self.assertEqual(before, a.snapshot())

    def test_idempotent_merge_near_operation_limit_deduplicates_before_budget(self):
        a = self.crdt('alice')
        a.set('one', 'D1', 'one'); a.set('two', 'D5', 'two')
        before = a.snapshot()
        with patch('collaboration.MAX_OPERATIONS', 3): a.merge(before)
        self.assertEqual(before, a.snapshot())

    def test_serialized_snapshot_byte_budget_is_enforced_atomically(self):
        a = self.crdt('alice'); before = a.snapshot()
        with patch('collaboration.MAX_STATE_BYTES', 1200):
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                a.set('n', 'D1', 'x' * 700)
        self.assertEqual(before, a.snapshot())

    def test_record_namespace_cannot_be_repointed_even_after_delete(self):
        a, b = self.crdt('alice'), self.crdt('bob')
        a.set('n', 'D1', 'a'); a.delete('n'); b.set('n', 'D5', 'b')
        with self.assertRaisesRegex(ValueError, 'record|namespace'): a.merge(b)
        with self.assertRaises(ValueError): a.set('n', 'D5', 'repoint')

    def test_scope_write_attempts_unknown_records_and_size_rejected(self):
        a = self.crdt('alice')
        for state in ({'schema': 'UL-annotation-crdt-1', 'operations': [], 'certified': True},
                      {'schema': 'UL-annotation-crdt-1', 'operations': [{'kind': 'statement'}]}):
            with self.assertRaises(ValueError): a.merge(state)
        with self.assertRaises(ValueError): a.set('n', 'unknown', 'text')
        with self.assertRaises(ValueError): a.set('../escape', 'D1', 'text')
        with self.assertRaises(ValueError): a.set('n', 'D1', 'é' * 8193)
        with self.assertRaises(ValueError): a.set('n', 'D1', {'acceptance': 'primary'})

    def test_fabricated_remove_and_extra_operation_fields_rejected(self):
        a = self.crdt('alice'); a.set('n', 'D1', 'a')
        bad = copy.deepcopy(a.snapshot()); bad['operations'][0]['observed'] = [['other', 1]]
        with self.assertRaises(ValueError): self.crdt('b').merge(bad)

    def test_remote_authentication_claims_and_causal_cycles_rejected(self):
        a, b = self.crdt('alice'), self.crdt('bob')
        a.set('n', 'D1', 'a'); b.set('n', 'D1', 'b')
        a.merge(b)
        bad = copy.deepcopy(a.snapshot()); bad['actor_identity'] = 'authenticated'
        with self.assertRaises(ValueError): self.crdt('c').merge(bad)
        bad = copy.deepcopy(a.snapshot())
        bad['operations'][0]['observed'] = [['bob', 1]]
        bad['operations'][1]['observed'] = [['alice', 1]]
        with self.assertRaisesRegex(ValueError, 'cyclic'): self.crdt('c').merge(bad)
        bad = copy.deepcopy(a.snapshot()); bad['operations'][0]['job'] = 'approve'
        with self.assertRaises(ValueError): self.crdt('b').merge(bad)

    def test_durable_store_restart_and_scope_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(tmp, ['D1', 'D5'])
            store.apply({'action': 'set', 'annotation_id': 'n', 'record_id': 'D1',
                         'text': 'durable note'}, actor='owner')
            self.assertEqual(store.snapshot()['operations'],
                             AnnotationStore(tmp, ['D1', 'D5']).snapshot()['operations'])
            replica = self.crdt('remote'); replica.set('m', 'D5', 'untrusted observation')
            store.merge(replica.snapshot())
            self.assertEqual(len(store.snapshot()['annotations']), 2)
            with self.assertRaises(ValueError):
                store.apply({'action': 'set', 'annotation_id': 'x', 'record_id': 'D1',
                             'text': 'x', 'acceptance': 'primary'})
            self.assertEqual(store.snapshot()['remote_multiuser'], 'unconfigured')
            store.apply({'action': 'delete', 'annotation_id': 'n'}, actor='owner')
            self.assertEqual(len(store.snapshot()['annotations']), 1)


if __name__ == '__main__': unittest.main()
