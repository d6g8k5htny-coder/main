"""Structural scene selection must not depend on mathematical proof names."""
import unittest

from tools.atlas_spatial_admission import choose_seed


class SpatialSeedTests(unittest.TestCase):
    def test_prefers_small_source_bearing_connected_neighborhood(self):
        graph = {'nodes': {'hub': {'source': 'hub.md'}, 'a': {'source': 'a.md'},
                           'b': {'source': 'b.md'}, 'isolated': {'source': 'i.md'}},
                 'edges': [{'from': 'hub', 'to': 'a'}, {'from': 'b', 'to': 'hub'}]}
        self.assertEqual(choose_seed(graph), 'hub')

    def test_ignores_dangling_edges_and_self_loops(self):
        graph = {'nodes': {'a': {}, 'b': {}, 'z': {}}, 'edges': [
            {'from': 'z', 'to': 'missing'}, {'from': 'z', 'to': 'z'},
            {'from': 'a', 'to': 'b'}]}
        self.assertEqual(choose_seed(graph), 'a')

    def test_selection_is_stable_under_input_reordering(self):
        graph = {'nodes': {'z': {}, 'a': {}, 'b': {}}, 'edges': [
            {'from': 'z', 'to': 'a'}, {'from': 'a', 'to': 'b'}]}
        reversed_graph = {'nodes': dict(reversed(list(graph['nodes'].items()))),
                          'edges': list(reversed(graph['edges']))}
        self.assertEqual(choose_seed(graph), choose_seed(reversed_graph))

    def test_fails_without_a_pickable_recorded_neighbor(self):
        with self.assertRaisesRegex(ValueError, 'connected'):
            choose_seed({'nodes': {'only': {}}, 'edges': []})


if __name__ == '__main__':
    unittest.main()
