"""Exact and one-dimensional parts of J(k) = 3k^2 E[H] = 3k^2 (E[2(B_-)^3] + E[2T] + E[V]), and tail bounds."""
from fractions import Fraction as Fr
import math
from ia import (dn, up, add, sub, mul, div, scal, divc, sqr, pw, isqrt, exp_neg, Phi, SQRT2PI, from_frac, SQRT2)
from fcore import moments


def E2T(k):
    """E[2T] = 6 E[u^2] = 6 (6k + 1/(4k) + 5/(216 k^3)), exact rational, k rational."""
    k = Fr(k)
    return 6 * (6 * k + Fr(1) / (4 * k) + Fr(5) / (216 * k ** 3))


def g_iv(mi):
    """g(m) = E[(m - beta)_+^3], beta ~ N(0, 2): (m^3 + 6m) Phi(m/sqrt2) + sqrt2 (m^2 + 4) phi(m/sqrt2), m >= 0 interval."""
    t = div(mi, SQRT2)
    ph = div(exp_neg(divc(sqr(mi), 4.0)), SQRT2PI)
    return add(mul(add(pw(mi, 3), scal(6.0, mi)), Phi(t)), mul(mul(SQRT2, add(sqr(mi), (4.0, 4.0))), ph))


def E2B(k, L, N):
    """Enclosure of E[2(B_-)^3] = 4 int_0^inf phi_sa(a) g(a^2/12) da, sa^2 = 2/k: g increasing in a >= 0, so each cell
    [a_i, a_{i+1}] is bounded by its endpoints; plus a tail bound beyond L."""
    s2 = (2.0 / k, 2.0 / k)
    s = isqrt(s2)
    lo = hi = 0.0
    prev_a = 0.0
    prev_g = g_iv((0.0, 0.0))
    prev_F = Phi((0.0, 0.0))
    for i in range(1, N + 1):
        a = L * i / N
        F = Phi(div((a, a), s))
        gi = g_iv(divc(sqr((a, a)), 12.0))
        mass = (max(0.0, dn(F[0] - prev_F[1])), up(F[1] - prev_F[0]))
        lo = dn(lo + dn(mass[0] * prev_g[0]))
        hi = up(hi + up(mass[1] * gi[1]))
        prev_g, prev_F = gi, F
    # tail a > L: g(m) <= m^3 + 6m + (m^2 + 4)/sqrt(pi), m = a^2/12
    M = tail_moments(L, s, s2, 6)
    inv_sqrtpi = div((1.0, 1.0), isqrt(div(scal(2.0, SQRT2PI), (2.0, 2.0)) if False else isqrt((3.141592653589793, 3.1415926535897936))))
    t = add(add(divc(M[6], 1728.0), divc(M[2], 2.0)), mul(inv_sqrtpi, add(divc(M[4], 144.0), scal(4.0, M[0]))))
    return (scal(4.0, (lo, hi))[0], up(4.0 * up(hi + t[1])))


def tail_moments(L, s, s2, nmax):
    """int_L^inf x^n phi_s(x) dx, n = 0..nmax (outward)."""
    from fcore import phis
    pL = phis(L, s, s2)
    F = Phi(div((L, L), s))
    M = [(max(0.0, dn(1.0 - F[1])), up(1.0 - F[0]))]
    M.append(mul(s2, pL))
    Li = (L, L)
    for n in range(2, nmax + 1):
        M.append(add(mul(scal(float(n - 1), s2), M[n - 2]), mul(s2, mul(pw(Li, n - 1), pL))))
    return M
