"""Catch exponent/normalization errors with hand-derived exact examples."""
from fractions import Fraction as Q
import unittest
import fold_controls as f


class FoldControls(unittest.TestCase):
    def test_same_cubic_fold_has_three_sampling_exponents(self):
        # Omitting the radial counting weight would conflate these three laws.
        self.assertEqual(f.density_exponent(Q(0), Q(3)), Q(-2, 3))
        self.assertEqual(f.density_exponent(Q(1), Q(3)), Q(-1, 3))
        self.assertEqual(f.density_exponent(Q(2), Q(3)), Q(0))

    def test_cumulative_exponent_is_two_thirds(self):
        self.assertEqual(f.cumulative_exponent(Q(1), Q(3)), Q(2, 3))

    def test_exact_fold_height_gap_includes_four_thirds(self):
        self.assertEqual(f.fold_gap(Q(1, 2)), Q(1, 6))

    def test_bin_mass_retains_jacobian_and_mark_coefficient(self):
        # ell=8*r^3 sends [1,8) to r in [1/2,1); integral 3*r dr=9/8.
        self.assertEqual(f.lifetime_bin_mass(Q(1), Q(8), gap_coefficient=Q(8),
                                            weight=Q(3)), Q(9, 8))

    def test_radial_mass_and_lifetime_mass_agree_independently(self):
        # For ell=r^3, integral_1^2 r dr=3/2; no assumed 1/3 fit is used.
        self.assertEqual(f.lifetime_bin_mass(Q(1), Q(8)), Q(3, 2))
        self.assertEqual(f.lifetime_bin_mass(Q(0), Q(8)), Q(2))

    def test_nonrational_root_is_refused_in_exact_calculation(self):
        with self.assertRaises(ValueError):
            f.lifetime_bin_mass(Q(1), Q(2))

    def test_invalid_domain_refused_even_under_optimization(self):
        for a, p in ((Q(-1), Q(3)), (Q(1), Q(0)), (Q(1), Q(-2))):
            with self.assertRaises(ValueError):
                f.density_exponent(a, p)
        for r in (Q(0), Q(-1), True, 0.5):
            with self.assertRaises(ValueError):
                f.fold_gap(r)
        for args in ((Q(2), Q(1)), (Q(-1), Q(8)), (Q(1), Q(1))):
            with self.assertRaises(ValueError):
                f.lifetime_bin_mass(*args)
        with self.assertRaises(ValueError):
            f.lifetime_bin_mass(Q(1), Q(8), gap_coefficient=Q(0))


if __name__ == '__main__':
    unittest.main()
