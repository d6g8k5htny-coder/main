"""Exact identities for the shrinking witness-pair reconnaissance.

Rational and integer checks only. No Gaussian sampling and no claim about the
periodized pinned contact kernel.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import unittest


def det(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    total = 0
    for col, entry in enumerate(matrix[0]):
        minor = [row[:col] + row[col + 1 :] for row in matrix[1:]]
        total += ((-1) ** col) * entry * det(minor)
    return total


def jacobian_matrix(dim, delta):
    """Blocks [[I, 0], [I, delta I]] for (G0, G1) -> (G0, G0+delta G1)."""
    size = 2 * dim
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(dim):
        matrix[i][i] = 1
        matrix[dim + i][i] = 1
        matrix[dim + i][dim + i] = delta
    return matrix


def axial_from_higher(delta, mu, nu, rho):
    """Impose h(0)=h(delta)=0 on a degree-four axial derivative and return jets."""
    alpha = -mu * delta / 2 - nu * delta ** 2 / 6 - rho * delta ** 3 / 24
    alpha_far = alpha + mu * delta + nu * delta ** 2 / 2 + rho * delta ** 3 / 6
    gap = (
        alpha * delta ** 2 / 2
        + mu * delta ** 3 / 6
        + nu * delta ** 4 / 24
        + rho * delta ** 5 / 120
    )
    return alpha, alpha_far, gap


class ShrinkingWitnessIdentities(unittest.TestCase):
    def test_divided_difference_jacobian(self):
        for dim in (2, 3, 4):
            for delta in (Fraction(1, 3), Fraction(2, 5), Fraction(1)):
                self.assertEqual(det(jacobian_matrix(dim, delta)), delta ** dim)

    def test_singular_value_controls_determinant(self):
        # |det H| <= ||Hu||_2 ||H||_F, compared in squares for u=(1,0).
        for alpha, beta, gamma in (
            (Fraction(1, 7), Fraction(1, 7), Fraction(3)),
            (Fraction(-2, 5), Fraction(1, 4), Fraction(-6)),
            (Fraction(0), Fraction(0), Fraction(4)),
        ):
            determinant = alpha * gamma - beta ** 2
            hu2 = alpha ** 2 + beta ** 2
            frobenius2 = alpha ** 2 + 2 * beta ** 2 + gamma ** 2
            self.assertLessEqual(determinant ** 2, hu2 * frobenius2)

    def test_axial_expansion_and_height_gap(self):
        delta, mu, nu, rho = Fraction(2, 7), Fraction(3), Fraction(-5), Fraction(11)
        alpha, alpha_far, gap = axial_from_higher(delta, mu, nu, rho)
        self.assertEqual(
            alpha,
            -(delta / 2) * mu - (delta ** 2 / 6) * nu - (delta ** 3 / 24) * rho,
        )
        self.assertEqual(
            alpha_far,
            (delta / 2) * mu + (delta ** 2 / 3) * nu + (delta ** 3 / 8) * rho,
        )
        self.assertEqual(
            gap,
            -(mu * delta ** 3) / 12 - (nu * delta ** 4) / 24 - (rho * delta ** 5) / 80,
        )

    def test_leading_axial_signs_and_fourth_derivative_window(self):
        # Third derivative alone forces opposite axial curvatures.
        alpha, alpha_far, _gap = axial_from_higher(Fraction(1, 5), Fraction(4), 0, 0)
        self.assertLess(alpha * alpha_far, 0)
        # mu = delta*mu_1 with mu_1=-3, nu=6 puts both scaled curvatures at +1/2.
        delta = Fraction(1, 4)
        alpha, alpha_far, _gap = axial_from_higher(delta, delta * Fraction(-3), Fraction(6), 0)
        self.assertEqual(alpha / delta ** 2, Fraction(1, 2))
        self.assertEqual(alpha_far / delta ** 2, Fraction(1, 2))
        # The same fourth derivative with mu_1=0 returns opposite signs.
        alpha, alpha_far, _gap = axial_from_higher(delta, 0, Fraction(6), 0)
        self.assertLess(alpha * alpha_far, 0)

    def test_radial_integral_is_independent_of_small_eta(self):
        # ∫_η^{δ_*} s^{2-d} s^{d-1} ds = ∫_η^{δ_*} s ds for every d>=2.
        for dim in (2, 3, 5):
            self.assertEqual((2 - dim) + (dim - 1), 1)
        eta, radius = Fraction(1, 10 ** 6), Fraction(1, 8)
        integral = (radius ** 2 - eta ** 2) / 2
        self.assertLess(integral, radius ** 2 / 2)
        self.assertGreater(integral, 0)

    def test_bargmann_fock_third_derivative_survives_gradient_conditioning(self):
        # Var(∂xxx f)=15 and Cov(∂x f, ∂xxx f)=-3 at a point of exp(-|z|^2/2).
        # Hu is uncorrelated with the third derivative, so conditioning on it
        # does not change this residual variance.
        variance = 15 - ((-3) ** 2) / 1
        self.assertEqual(variance, 6)

    def test_note_cites_the_reviewed_outer_boundary(self):
        note = Path(__file__).resolve().parents[1] / "docs" / "rn_d5_shrinking_witness_20260926.md"
        text = note.read_text(encoding="utf-8")
        self.assertIn(
            "a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7",
            text,
        )
        self.assertIn("191ea7d541a486736ba7bbddfd4eac25a6c4567b", text)
        self.assertIn("E_{Q_r^W}[T_{r,j,2}(η)] ≤ C k r^3", text)
        self.assertIn("does not edit", text)


if __name__ == "__main__":
    unittest.main()
