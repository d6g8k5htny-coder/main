"""Bounded exact interface and fail-closed regression tests (including -O)."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import os
import shutil
import tempfile
import unittest

import check
from finite import (colorable, complete_witnesses, cover_cost, family, lift,
                    occupancy_cover, obstruction_cover, primitive_palette,
                    require_complete_actual, singleton_restriction, subsets,
                    verify_composition, verify_scalar_sandwich)

def locate_repo():
    if os.environ.get('P15_REPO'):
        return Path(os.environ['P15_REPO']).resolve()
    for candidate in Path(__file__).resolve().parents:
        if ((candidate / 'drive/inventory.jsonl').is_file()
                and (candidate / 'quarantine/EXCLUSIONS.json').is_file()):
            return candidate
    raise RuntimeError('Set P15_REPO to the research checkout; no repository ancestor was found')


REPO = locate_repo()


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.blocks = {0: frozenset([0, 1]), 1: frozenset([2, 3]), 2: frozenset([4, 5])}
        self.local = {i: family([b]) for i, b in self.blocks.items()}
        self.macro = family([{0, 1, 2}])
        self.prices = {v: F(1, 2) for v in range(6)}

    def test_all_exact_cases(self):
        self.assertEqual(len(check.run_cases()), 13)

    def test_true_hypergraph_two_color_obstruction(self):
        h = family(combinations(range(5), 3))
        self.assertFalse(colorable(range(5), h, 2))
        self.assertTrue(colorable(range(4), h, 2))

    def test_global_empty_downset_empty_set_zero_parts(self):
        self.assertTrue(colorable([], family([[]]), 1))
        self.assertFalse(colorable([0], family([[]]), 99))

    def test_missing_local_cover_rejected(self):
        with self.assertRaisesRegex(ValueError, 'uncovered obstruction'):
            verify_composition(self.blocks, self.local, self.macro, 1,
                               {i: family([]) for i in self.blocks},
                               {i: 0 for i in self.blocks}, self.macro, self.prices)

    def test_missing_macro_cover_rejected(self):
        with self.assertRaisesRegex(ValueError, 'monochromatic macro edge'):
            verify_composition(self.blocks, self.local, self.macro, 1,
                               self.local, {i: 0 for i in self.blocks},
                               family([]), self.prices)

    def test_overlapping_blocks_rejected(self):
        b = dict(self.blocks)
        b[2] = frozenset([0, 5])
        with self.assertRaisesRegex(ValueError, 'overlapping blocks'):
            complete_witnesses(b, self.local, self.macro)

    def test_empty_block_must_be_removed(self):
        b = dict(self.blocks)
        b[2] = frozenset()
        with self.assertRaisesRegex(ValueError, 'empty block'):
            complete_witnesses(b, self.local, self.macro)

    def test_zero_cost_singletons_not_dropped(self):
        c = occupancy_cover(frozenset([0, 1]), {0: F(0), 1: F(0)})
        self.assertEqual(c, family([[0], [1]]))
        self.assertEqual(cover_cost(c, {0: F(0), 1: F(0)}), 0)

    def test_empty_generator_not_empty_family(self):
        self.assertEqual(cover_cost(family([[]]), {}), 1)
        self.assertEqual(cover_cost(family([]), {}), 0)
        with self.assertRaisesRegex(ValueError, 'uncovered obstruction'):
            obstruction_cover([0], family([[0]]), 1, family([]))

    def test_lift_deduplication_never_increases_cost(self):
        occ = {i: family([[]]) for i in range(3)}
        lifted, descriptions = lift(family([[0, 1], [1, 2]]), occ)
        self.assertEqual(len(descriptions), 2)
        self.assertEqual(lifted, family([[]]))
        self.assertEqual(cover_cost(lifted, {}), 1)

    def test_empty_macro_generator_lifts_to_empty(self):
        lifted, descriptions = lift(family([[]]), {})
        self.assertEqual(lifted, family([[]]))
        self.assertEqual(descriptions, (frozenset(),))

    def test_negative_prices_rejected(self):
        with self.assertRaisesRegex(ValueError, 'negative price'):
            occupancy_cover([0], {0: F(-1)})

    def test_incomplete_lifting_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete-transversal'):
            require_complete_actual(self.blocks, {i: family([]) for i in self.blocks},
                                    self.macro, family([[0, 2, 4]]))

    def test_macro_singleton_forbids_whole_block(self):
        removed, b, local, macro = singleton_restriction(
            self.blocks, {i: family([]) for i in self.blocks}, family([[0], [0, 1, 2]]))
        self.assertEqual(removed, frozenset([0, 1]))
        self.assertEqual(set(b), {1, 2})
        self.assertEqual(macro, family([]))

    def test_empty_downset_precedes_singleton_reduction(self):
        with self.assertRaisesRegex(ValueError, 'globally empty family'):
            singleton_restriction(self.blocks, self.local, family([[]]))

    def test_strict_scalar_outer_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'strict outer'):
            verify_scalar_sandwich([0, 1], family([]), {0: F(1, 2), 1: F(1, 2)}, F(1))

    def test_scalar_inner_implication(self):
        with self.assertRaisesRegex(ValueError, 'inner scalar'):
            verify_scalar_sandwich([0, 1], family([[0, 1]]), {0: F(1, 3), 1: F(1, 3)}, F(1))

    def test_scalar_singleton_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'scalar singleton'):
            verify_scalar_sandwich([0], family([]), {0: F(1)}, F(2))

    def test_recurrence_and_resource_caps(self):
        self.assertEqual(primitive_palette(3), 4706657280256)
        with self.assertRaisesRegex(ValueError, 'bounded recurrence'):
            primitive_palette(9)
        with self.assertRaisesRegex(ValueError, 'exhaustion cap'):
            subsets(range(11))

    def test_exact_source_guard(self):
        self.assertEqual(len(check.source_guard(REPO)), 9)

    def test_copied_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix='p15-source-test-') as d:
            artifact = Path(d)
            shutil.copy2(check.HERE / 'SOURCES.json', artifact / 'SOURCES.json')
            shutil.copytree(check.HERE / 'sources', artifact / 'sources')
            victim = artifact / 'sources/P15_NEXT_WORK.md'
            victim.write_bytes(victim.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'copied source identity'):
                check.source_guard(REPO, artifact)

    def test_source_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix='p15-manifest-test-') as d:
            artifact = Path(d)
            (artifact / 'SOURCES.json').write_bytes((check.HERE / 'SOURCES.json').read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'source manifest identity'):
                check.source_guard(REPO, artifact)


if __name__ == '__main__':
    unittest.main()
