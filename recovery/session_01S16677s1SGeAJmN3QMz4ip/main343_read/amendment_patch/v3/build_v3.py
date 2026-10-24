"""Build the v3 amended test file from the v2 amended file: one test appended at the
end of EnclosureTests (C270-A4 regression). No existing test or line is changed."""
import sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
text = src.read_text()

enclosure_anchor = """            self.assertLess(lo, hi)
        self.assertEqual(pc.poisson_target(), pc.poisson_target(pc.coefficient_interval()))
"""
enclosure_block = enclosure_anchor + r'''
    def test_tail_upper_endpoint_dominates_a_finer_midpoint_lower_bound(self):
        """Kills M06_CEILING_ONLY (C270-A4): the trapezoid ceiling at population_certificate.py
        line 392 replaced by an outward-rounded midpoint sum while tail_upper_floor and the
        positive tail diagnostic stay intact, which the component replay above cannot see.

        Oracle: the tail integrand C*(9*t^(-11/4) - rhi^2*t^(-3/4)) is convex on [1, 1/rhi]
        (CONTROLLED_FALSIFICATION.md, direct tail: "It is convex; f'' <= 99/32*t^(-19/4)";
        gates tail_convexity and tail_second_derivative), so on every octave each trapezoid
        sum bounds the integral from above, each midpoint sum from below, and midpoint sums
        increase under dyadic refinement. With outward rounding the declared 512-panel tail
        upper endpoint therefore dominates the 1024-panel tail lower endpoint. The damaged
        ceiling is itself a 512-panel midpoint sum and falls below that finer lower bound by
        about 7.5e-8 at j=1; the pristine ceiling exceeds it by about 2.2e-7. The core is cut
        to 64 panels because the tail sums read only rhi, the tail coefficient and
        TAIL_PANELS, never the core; the row is then INCONCLUSIVE_PRECISION but still reports
        its tail interval. The finer sum is an oracle only; the campaign keeps 512 panels.
        """
        def tail_row(panels):
            with mock.patch.object(pc, 'CORE_PANELS', 64), mock.patch.object(pc, 'TAIL_PANELS', panels):
                row = pc.direct_mass(1)
            self.assertEqual(row['status'], 'INCONCLUSIVE_PRECISION')
            self.assertTrue(row['gates']['tail_convexity'] and row['gates']['tail_second_derivative'])
            # 65 core nodes; two tail octaves of 2*panels+1 nodes sharing the node t=2.
            self.assertEqual(row['evaluations'], 65 + 2 * (2 * panels + 1) - 1)
            return row
        self.assertEqual(pc.TAIL_PANELS, 512)
        coarse, finer = tail_row(pc.TAIL_PANELS), tail_row(2 * pc.TAIL_PANELS)
        self.assertLessEqual(finer['tail_interval'][0], finer['tail_interval'][1])
        # Refinement raises the midpoint sum; the finer floor loses at most its own arithmetic width.
        self.assertLessEqual(coarse['tail_interval'][0],
                             finer['tail_interval'][0] + finer['arithmetic_components']['tail_lower'])
        self.assertGreaterEqual(coarse['tail_interval'][1], finer['tail_interval'][0])
'''
assert text.count(enclosure_anchor) == 1
text = text.replace(enclosure_anchor, enclosure_block)
dst.write_text(text)
print('wrote', dst, len(text), 'bytes')
