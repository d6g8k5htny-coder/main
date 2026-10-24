"""Exact checks for the zeta-fold exploration packet (standard library only).

Exploratory packet; scientific effect NONE. Everything here is exact rational arithmetic
(fractions.Fraction, plus a tiny Q(sqrt 2) class), so it runs under `python3 -B -E -S`:

    python3 -B -E -S -m unittest discover -s experiments/zeta_fold_exploration -p 'test_exact*.py'

Covered:
  * fold normal form: the gap of x^3/3 - h^2 x is r^3/6 with r = 2h (k = |F'''|/12 = 1/6), and the
    local collision form A(x^3/3 - 2 tau x) gives |A| r^3/6 = (8 sqrt2/3)|A| tau^{3/2};
  * the lobe-kernel moments r^3/6, 0, r^5/120 used in the kappa_ell derivation;
  * backward heat polynomials v_n satisfy d_tau v_n = -d_x^2 v_n exactly and are homogeneous;
  * kappa_r, kappa_ell, kappa_t and the midpoint drift 2B/A from exact power series in
    sigma = sqrt(tau) with rational A, B, C (and higher terms), for A > 0 and A < 0;
  * density pushforward exponents (alpha+1)/3 - 1 = -2/3, -1/3, 0 for alpha = 0, 1, 2, and the
    small-mark phase diagram including the log(1/ell) case.
"""
import unittest
from fractions import Fraction as Fr
from math import factorial


# ---------------------------------------------------------------------------------------------
# Q(sqrt 2) and truncated power series in sigma
# ---------------------------------------------------------------------------------------------
class Q2:
    """a + b*sqrt(2) with rational a, b."""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(self, o):
        o = q2(o)
        return Q2(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return Q2(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-q2(o))

    def __rsub__(self, o):
        return q2(o) - self

    def __mul__(self, o):
        o = q2(o)
        return Q2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def inv(self):
        n = self.a * self.a - 2 * self.b * self.b
        return Q2(self.a / n, -self.b / n)

    def __truediv__(self, o):
        return self * q2(o).inv()

    def __eq__(self, o):
        o = q2(o)
        return self.a == o.a and self.b == o.b

    def __repr__(self):
        return f"({self.a} + {self.b}*sqrt2)"


def q2(x):
    return x if isinstance(x, Q2) else Q2(x, 0)


SQRT2 = Q2(0, 1)
NS = 4  # series kept to sigma^NS


def s_const(c):
    return [q2(c)] + [Q2() for _ in range(NS)]


def s_add(p, q):
    return [x + y for x, y in zip(p, q)]


def s_sub(p, q):
    return [x - y for x, y in zip(p, q)]


def s_scale(p, c):
    return [x * c for x in p]


def s_mul(p, q):
    r = [Q2() for _ in range(NS + 1)]
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            if i + j <= NS:
                r[i + j] = r[i + j] + x * y
    return r


def s_inv(p):
    r = [Q2() for _ in range(NS + 1)]
    r[0] = p[0].inv()
    for n in range(1, NS + 1):
        acc = Q2()
        for k in range(1, n + 1):
            acc = acc + p[k] * r[n - k]
        r[n] = -(acc * r[0])
    return r


def s_poly_eval(coeffs, y):
    """coeffs[j] is a series (coefficient of y^j); y is a series. Horner in y."""
    out = s_const(0)
    for c in reversed(coeffs):
        out = s_add(s_mul(out, y), c)
    return out


def v_coeffs(n):
    """Backward heat polynomial v_n(x, tau) = sum_k n!/(k!(n-2k)!) (-tau)^k x^(n-2k), as {(i, j): c}
    for the monomial x^i tau^j."""
    return {(n - 2 * k, k): Fr(factorial(n), factorial(k) * factorial(n - 2 * k)) * (-1) ** k
            for k in range(n // 2 + 1)}


def h_series_coeffs(c):
    """h(y, sigma) = sigma^-2 H(t0 + sigma^2, z0 + sigma y) = sum_n c_n sigma^(n-2) v_n(y, 1),
    returned as a list over powers of y of series in sigma."""
    deg = max(c)
    out = [s_const(0) for _ in range(deg + 1)]
    for n, cn in c.items():
        if n - 2 > NS:
            continue
        for (i, j), v in v_coeffs(n).items():
            out[i][n - 2] = out[i][n - 2] + cn * v
    return out


def root_series(hc, start):
    """Newton iteration in series arithmetic for h(y(sigma), sigma) = 0 with y(0) = start."""
    dh = [s_scale(hc[j], j) for j in range(1, len(hc))]
    y = s_const(start)
    for _ in range(NS + 2):
        y = s_sub(y, s_mul(s_poly_eval(hc, y), s_inv(s_poly_eval(dh, y))))
    return y


def fold_coefficients(A, B, C, extra=None):
    c = {2: Fr(A), 3: Fr(B), 4: Fr(C)}
    c.update(extra or {})
    hc = h_series_coeffs(c)
    yp = root_series(hc, SQRT2)
    ym = root_series(hc, -SQRT2)
    R = s_sub(yp, ym)                                # r = sigma * R
    M = s_scale(s_add(yp, ym), Fr(1, 2))             # midpoint offset = sigma * M
    G = [s_const(0)] + [s_scale(hc[j], Fr(1, j + 1)) for j in range(len(hc))]   # antiderivative in y
    J = s_sub(s_poly_eval(G, yp), s_poly_eval(G, ym))   # ell = sigma^3 |J|
    R2over8 = s_scale(s_mul(R, R), Fr(1, 8))                         # r^2/(8 tau)
    R3 = s_mul(s_mul(R, R), R)
    ell_ratio = s_mul(s_scale(J, Fr(-6) / Fr(A)), s_inv(R3))         # ell / (|A| r^3/6)
    J0 = Q2(0, Fr(-8, 3) * Fr(A))                                     # -(8 sqrt2/3) A
    t_ratio = s_scale(J, J0.inv())                                    # ell / ((8 sqrt2/3)|A| tau^1.5)
    return R2over8, M, ell_ratio, t_ratio, J


# ---------------------------------------------------------------------------------------------
class FoldNormalForm(unittest.TestCase):
    def test_gap_is_r_cubed_over_six(self):
        for h in (Fr(1, 3), Fr(2), Fr(7, 5), Fr(1, 1000)):
            f = lambda x: x ** 3 / 3 - h * h * x
            gap = f(-h) - f(h)
            r = 2 * h
            self.assertEqual(gap, Fr(4, 3) * h ** 3)
            self.assertEqual(gap, r ** 3 / 6)
        # project convention ell = k r^3 with k = |F'''|/12; here F''' = 2
        self.assertEqual(Fr(2, 12), Fr(1, 6))

    def test_local_collision_form(self):
        # F = A (x^3/3 - 2 tau x), critical points +-sqrt(2 tau); tau = 1/8 gives sqrt(2 tau) = 1/2.
        tau = Fr(1, 8)
        for A in (Fr(3), Fr(-5, 7)):
            F = lambda x: A * (x ** 3 / 3 - 2 * tau * x)
            s = Fr(1, 2)
            self.assertEqual(s * s, 2 * tau)
            gap = abs(F(-s) - F(s))
            r = 2 * s
            self.assertEqual(gap, abs(A) * r ** 3 / 6)
            # (8 sqrt2/3)|A| tau^(3/2) with tau^(3/2) = (1/8)^(3/2) = sqrt2/32
            tau32 = Q2(0, Fr(1, 32))
            self.assertEqual(tau32 * tau32, Q2(tau ** 3))
            self.assertEqual(Q2(0, Fr(8, 3)) * abs(A) * tau32, Q2(gap))

    def test_lobe_kernel_moments(self):
        # int_{-r/2}^{r/2} (r^2/4 - s^2) s^m ds for m = 0, 1, 2 (exact polynomial integration)
        def moment(r, m):
            a = r / 2
            prim = lambda s: r * r / 4 * s ** (m + 1) / (m + 1) - s ** (m + 3) / (m + 3)
            return prim(a) - prim(-a)
        for r in (Fr(1), Fr(3, 7), Fr(5, 2)):
            self.assertEqual(moment(r, 0), r ** 3 / 6)
            self.assertEqual(moment(r, 1), 0)
            self.assertEqual(moment(r, 2), r ** 5 / 120)


class BackwardHeatPolynomials(unittest.TestCase):
    def test_explicit_low_orders(self):
        self.assertEqual(v_coeffs(2), {(2, 0): 1, (0, 1): -2})
        self.assertEqual(v_coeffs(3), {(3, 0): 1, (1, 1): -6})
        self.assertEqual(v_coeffs(4), {(4, 0): 1, (2, 1): -12, (0, 2): 12})

    def test_backward_heat_equation_exact(self):
        for n in range(2, 10):
            v = v_coeffs(n)
            dtau, dxx = {}, {}
            for (i, j), c in v.items():
                if j:
                    dtau[(i, j - 1)] = dtau.get((i, j - 1), 0) + j * c
                if i >= 2:
                    dxx[(i - 2, j)] = dxx.get((i - 2, j), 0) + i * (i - 1) * c
            keys = set(dtau) | set(dxx)
            for k in keys:
                self.assertEqual(dtau.get(k, 0) + dxx.get(k, 0), 0, (n, k))

    def test_parabolic_homogeneity(self):
        # v_n(sigma y, sigma^2) = sigma^n v_n(y, 1): every monomial x^i tau^j has i + 2j = n
        for n in range(2, 10):
            self.assertTrue(all(i + 2 * j == n for (i, j) in v_coeffs(n)))


class FoldCoefficients(unittest.TestCase):
    CASES = [
        (Fr(1), Fr(3, 10), Fr(1, 5), {5: Fr(-3, 20)}),                     # kappa_check.py
        (Fr(-7, 10), Fr(9, 20), Fr(-31, 100), {5: Fr(9, 10), 6: Fr(-13, 10)}),  # A < 0, v5 and v6 terms
        (Fr(2), Fr(-5, 3), Fr(1, 7), {5: Fr(4), 6: Fr(-1, 3)}),
    ]

    def test_kappas_and_midpoint(self):
        for A, B, C, extra in self.CASES:
            R2over8, M, ell_ratio, t_ratio, J = fold_coefficients(A, B, C, extra)
            kr = 4 * C / A - 2 * B * B / (A * A)
            kl = 6 * B * B / (A * A) - Fr(48, 5) * C / A
            kt = 3 * B * B / (A * A) - Fr(18, 5) * C / A
            # r^2 = 8 tau (1 + kappa_r tau + O(tau^2)); r/sigma is even in sigma
            self.assertEqual(R2over8[0], 1)
            self.assertEqual(R2over8[1], 0)
            self.assertEqual(R2over8[2], Q2(kr))
            self.assertEqual(R2over8[3], 0)
            # ell = (|A| r^3/6)(1 + kappa_ell tau + O(tau^2))
            self.assertEqual(ell_ratio[0], 1)
            self.assertEqual(ell_ratio[1], 0)
            self.assertEqual(ell_ratio[2], Q2(kl))
            # ell = (8 sqrt2/3)|A| tau^(3/2) (1 + kappa_t tau + O(tau^2)), kappa_t = kappa_ell + 3/2 kappa_r
            self.assertEqual(t_ratio[0], 1)
            self.assertEqual(t_ratio[1], 0)
            self.assertEqual(t_ratio[2], Q2(kt))
            self.assertEqual(kt, kl + Fr(3, 2) * kr)
            # midpoint z0 + 2(B/A) tau + O(tau^2): midpoint offset = sigma * M
            self.assertEqual(M[0], 0)
            self.assertEqual(M[1], Q2(2 * B / A))
            self.assertEqual(M[2], 0)
            # J(0) = int_{-sqrt2}^{sqrt2} A (y^2 - 2) dy = -(8 sqrt2/3) A
            self.assertEqual(J[0], Q2(0, Fr(-8, 3) * A))

    def test_reference_numbers(self):
        # kappa_check.py: (A, B, C) = (1, 0.3, 0.2) predicts 0.62, -1.38, -0.45, 0.6
        A, B, C = Fr(1), Fr(3, 10), Fr(1, 5)
        self.assertEqual(4 * C / A - 2 * B * B / A ** 2, Fr(62, 100))
        self.assertEqual(6 * B * B / A ** 2 - Fr(48, 5) * C / A, Fr(-138, 100))
        self.assertEqual(3 * B * B / A ** 2 - Fr(18, 5) * C / A, Fr(-45, 100))
        self.assertEqual(2 * B / A, Fr(6, 10))


class DensityPushforward(unittest.TestCase):
    def test_fixed_mark_exponents(self):
        # separation density (alpha+1) r^alpha on (0,1), ell = k r^3 with k fixed (k = 1):
        # P(ell < x) = x^((alpha+1)/3). Exact on cubes: P(ell < (p/q)^3) = (p/q)^(alpha+1).
        expected = {0: Fr(-2, 3), 1: Fr(-1, 3), 2: Fr(0)}
        for alpha, dens in expected.items():
            cdf = lambda rmax: rmax ** (alpha + 1)          # = int_0^rmax (alpha+1) r^alpha dr
            for rmax in (Fr(1, 2), Fr(3, 10)):
                # ell -> 8 ell multiplies the CDF by 2^(alpha+1) = 8^((alpha+1)/3)
                self.assertEqual(cdf(2 * rmax) / cdf(rmax), Fr(2) ** (alpha + 1))
            beta = Fr(alpha + 1, 3)
            self.assertEqual(beta - 1, dens)

    @staticmethod
    def leading(alpha, q):
        """nu(ell) = (q+1) beta ell^(beta-1) int_ell^1 k^(q-beta) dk for mark density (q+1) k^q on (0,1)
        and separation density (alpha+1) r^alpha; beta = (alpha+1)/3. Returns the leading small-ell
        term as (exponent, power of log(1/ell), coefficient), by exact monomial integration."""
        beta = Fr(alpha + 1, 3)
        e = q - beta
        pref = (q + 1) * beta
        if e + 1 == 0:
            return beta - 1, 1, pref
        terms = {beta - 1: pref / (e + 1), beta - 1 + e + 1: -pref / (e + 1)}
        ex = min(terms)
        return ex, 0, terms[ex]

    def test_mark_phase_diagram(self):
        # Math- collision_mechanism subsection D2: exponent beta-1 if q+1 > beta, extra log if
        # q+1 = beta, exponent q if q+1 < beta. Coefficients must come out positive.
        cases = [
            (2, Fr(1), (Fr(0), 0)), (2, Fr(0), (Fr(0), 1)), (2, Fr(-1, 2), (Fr(-1, 2), 0)),
            (1, Fr(0), (Fr(-1, 3), 0)), (1, Fr(-1, 3), (Fr(-1, 3), 1)), (1, Fr(-1, 2), (Fr(-1, 2), 0)),
            (0, Fr(0), (Fr(-2, 3), 0)),
        ]
        for alpha, q, (ex, lg) in cases:
            got = self.leading(alpha, q)
            self.assertEqual(got[:2], (ex, lg), (alpha, q))
            self.assertGreater(got[2], 0, (alpha, q))

    def test_log_case_coefficient(self):
        # alpha = 2 with k ~ U(0,1): nu(ell) = int_ell^1 dk/k = log(1/ell) exactly (check_gue_transfer.py)
        self.assertEqual(self.leading(2, Fr(0)), (Fr(0), 1, Fr(1)))


if __name__ == "__main__":
    unittest.main()
