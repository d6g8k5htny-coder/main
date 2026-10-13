"""Build the v2 amended test file from the v1 amended file: three tests appended
at the end of EnclosureTests (two) and PredicateFixtureTests (one). No existing
test or line is changed."""
import sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
text = src.read_text()

enclosure_anchor = """        lo, hi = pc.cdf_mass(Q(9, 4) - Q(1, 1 << 40))
        self.assertLess(Q(1, 5) - lo, Q(1, 1 << 80))
        self.assertLessEqual(hi, Q(1, 5))
"""
enclosure_block = enclosure_anchor + r'''
    def test_void_interval_replays_its_outward_log_exp_composition(self):
        """Kills M03a (void endpoints taken inward); containment of (1-m)^n alone cannot see it."""
        def exponent(m, n):
            lo, hi = pc.log_bounds(1 - m, bits=pc.BITS)
            return pc._outward((-n * hi, -n * lo))
        # Normalized mass n*m in (1/7, 1/5) at n = 4 and at the row-8 count, plus a point mass.
        for mass, n in (((Q(1, 7), Q(1, 5)), 4), ((Q(1, 7 * 4 ** 8), Q(1, 5 * 4 ** 8)), 4 ** 8),
                        ((Q(1, 3), Q(1, 3)), 4)):
            lo, hi = pc.void_interval(mass, n)
            self.assertEqual((lo, hi), (pc.exp_neg_bounds(exponent(mass[1], n)[1])[0],
                                        pc.exp_neg_bounds(exponent(mass[0], n)[0])[1]))
            self.assertTrue(lo < (1 - mass[1]) ** n <= (1 - mass[0]) ** n < hi, (mass, n))

    def test_poisson_target_replays_its_outward_exp_composition(self):
        """Kills M03b (Poisson target endpoints taken inward); exp(-B) is contained either way."""
        for b in (pc.coefficient_interval(), (Q(1, 3), Q(1, 2))):
            lo, hi = pc.poisson_target(b)
            self.assertEqual((lo, hi), (pc.exp_neg_bounds(b[1])[0], pc.exp_neg_bounds(b[0])[1]))
            self.assertLess(lo, hi)
        self.assertEqual(pc.poisson_target(), pc.poisson_target(pc.coefficient_interval()))
'''
assert text.count(enclosure_anchor) == 1
text = text.replace(enclosure_anchor, enclosure_block)

predicate_anchor = """        self.assertEqual(runner._overall(['FALSIFIED', 'FAIL_IMPLEMENTATION']), 'FAIL_IMPLEMENTATION')
        self.assertEqual(runner._overall([]), 'INCONCLUSIVE_PRECISION')
"""
predicate_block = predicate_anchor + r'''
    def test_overall_requires_every_status_to_pass(self):
        """Kills M21 (PASS if any status passes): an undeclared status beside PASS is no pass."""
        self.assertEqual(runner._overall(['PASS', 'NOT_A_DECLARED_STATUS']), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(runner._overall(['NOT_RUN', 'PASS', 'PASS']), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(runner._overall(['PASS', 'PASS']), 'PASS')
        self.assertEqual(runner._overall([]), 'INCONCLUSIVE_PRECISION')
'''
assert text.count(predicate_anchor) == 1
text = text.replace(predicate_anchor, predicate_block)
dst.write_text(text)
print('wrote', dst, len(text), 'bytes')
