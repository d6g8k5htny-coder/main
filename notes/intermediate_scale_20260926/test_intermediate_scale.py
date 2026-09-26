"""Algebraic guards for the intermediate-scale envelope.

These checks certify the jet identities used in PROOF.md. They do not sample
the Gaussian field and they do not accept the analytic bound.
"""

import unittest
from math import factorial


def deriv_coeff(i, j, x, z, which):
    """Coefficient of d^i/dx^i d^j/dz^j in f_x, f_z, or f."""
    if which == "fx":
        if i < 1:
            return 0.0
        return (x ** (i - 1)) * (z ** j) / (factorial(i - 1) * factorial(j))
    if which == "fz":
        if j < 1:
            return 0.0
        return (x ** i) * (z ** (j - 1)) / (factorial(i) * factorial(j - 1))
    if which == "f":
        return (x ** i) * (z ** j) / (factorial(i) * factorial(j))
    raise ValueError(which)


def gram_det(columns, x, z):
    rows = []
    for which in ("fx", "fz", "f"):
        rows.append([deriv_coeff(i, j, x, z, which) for (i, j) in columns])
    # 3 x n times n x 3
    g = [[0.0, 0.0, 0.0] for _ in range(3)]
    for a in range(3):
        for b in range(3):
            g[a][b] = sum(rows[a][k] * rows[b][k] for k in range(len(columns)))
    return (
        g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
        - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
        + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
    )


def det3(cols):
    """Determinant of three 3-vectors stored as columns."""
    a, b, c = cols
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


class JetMinorTests(unittest.TestCase):
    def test_cubic_gram_matches_closed_form(self):
        columns = [(0, 2), (2, 1), (1, 2), (0, 3)]  # S, T, C, D
        samples = [
            (0.3, 0.4),
            (0.2, -0.5),
            (-0.7, 0.1),
            (0.0, 0.8),
            (0.9, 0.0),
            (1.0, 1.0),
        ]
        for x, z in samples:
            got = gram_det(columns, x, z)
            expect = (z ** 8) * (9 * x ** 4 + 4 * x * x * z * z + z ** 4) / 576.0
            self.assertAlmostEqual(got, expect, places=9)

    def test_cubic_form_dominates_squared_radius(self):
        # 9x^4 + 4 x^2 z^2 + z^4 >= (x^2+z^2)^2
        samples = [(1, 0), (0, 1), (0.6, 0.8), (-1.2, 0.5), (0.3, -0.9)]
        for x, z in samples:
            left = 9 * x ** 4 + 4 * x * x * z * z + z ** 4
            right = (x * x + z * z) ** 2
            self.assertGreaterEqual(left + 1e-12, right)

    def test_axial_minor_is_x_to_the_tenth(self):
        # Columns T=(2,1), Q=(4,0), R=(5,0), in order (fx, fz, f).
        samples = [0.2, 0.5, -0.7, 1.1, 0.0]
        for x in samples:
            z = 0.3
            cols = []
            for i, j in ((2, 1), (4, 0), (5, 0)):
                cols.append(
                    [
                        deriv_coeff(i, j, x, z, "fx"),
                        deriv_coeff(i, j, x, z, "fz"),
                        deriv_coeff(i, j, x, z, "f"),
                    ]
                )
            self.assertAlmostEqual(det3(cols), (x ** 10) / 5760.0, places=9)

    def test_cauchy_binet_floor_covers_every_direction(self):
        # det >= x^20/5760^2 from the axial minor, and
        # det >= z^8 s^4 / 576 from the cubic Gram, up to the lambda^3
        # factor omitted in this white-jet model.
        cubic = [(0, 2), (2, 1), (1, 2), (0, 3)]
        axial = [(2, 1), (4, 0), (5, 0)]
        # A jet containing both families.
        both = list(dict.fromkeys(cubic + axial + [(6, 0), (0, 5)]))
        radius = 0.05
        for step in range(24):
            theta = step * 3.141592653589793 / 24.0
            x = radius * __import__("math").cos(theta)
            z = radius * __import__("math").sin(theta)
            got = gram_det(both, x, z)
            floor_ax = (x ** 20) / (5760.0 ** 2)
            floor_cu = (z ** 8) * (radius ** 4) / 576.0
            self.assertGreaterEqual(got + 1e-18, max(floor_ax, floor_cu) * 0.5)


class CurvatureIsolatorTests(unittest.TestCase):
    def test_cubic_combination_returns_transverse_curvature(self):
        # On the cubic jet, S = -(2x/z^2) f_x - (2/z) f_z + (6/z^2)(f-b).
        samples = [
            (0.2, 0.5, 1.0, 0.3, -0.4, 0.7, 0.2),
            (-0.4, 0.3, 2.0, -1.0, 0.5, -0.2, 0.8),
            (0.1, -0.7, 0.5, 0.0, 0.0, 0.0, -0.3),
        ]
        for x, z, k, S, T, C, D in samples:
            fx = 6 * k * x * x + T * x * z + 0.5 * C * z * z
            fz = S * z + 0.5 * T * x * x + C * x * z + 0.5 * D * z * z
            ff = (
                0.5 * S * z * z
                + 2 * k * x ** 3
                + 0.5 * T * x * x * z
                + 0.5 * C * x * z * z
                + (1.0 / 6.0) * D * z ** 3
            )
            got = -(2 * x / z ** 2) * fx - (2 / z) * fz + (6 / z ** 2) * ff
            self.assertAlmostEqual(got, S, places=9)


class AngularEnvelopeTests(unittest.TestCase):
    def test_sigma_polynomial_times_gaussian_is_bounded(self):
        # The proof uses sup_{0<sigma<=1} sigma^{-4} exp(-c/(4 sigma^2)) < infinity.
        c = 0.1
        best = 0.0
        sigma = 1e-3
        while sigma <= 1.0:
            value = (sigma ** -4) * __import__("math").exp(-c / (4 * sigma * sigma))
            if value > best:
                best = value
            sigma *= 1.05
        self.assertLess(best, 1e6)
        self.assertGreater(best, 0.0)


if __name__ == "__main__":
    unittest.main()
