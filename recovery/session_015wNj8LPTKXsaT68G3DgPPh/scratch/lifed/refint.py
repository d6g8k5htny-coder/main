"""Certified reference integrals: int k^q exp(-12 k^2) dk (incomplete gamma series) and
int exp(-3 b^2/4) m(b) db (composite midpoint rule with a second-derivative bound)."""
import math
from fractions import Fraction as Fr

import lifecore as LC

E2, E3 = LC.E2, LC.E3
PREC = 240


def iroot_int(N, p):
    """Largest y >= 0 with y^p <= N (integers)."""
    if N < 2:
        return N
    y = 1 << ((N.bit_length() + p - 1) // p)          # y^p >= N
    while True:
        z = ((p - 1) * y + N // y ** (p - 1)) // p
        if z >= y:
            break
        y = z
    while y ** p > N:
        y -= 1
    while (y + 1) ** p <= N:
        y += 1
    return y


def root_iv(M, x, p):
    """Enclosure of x^(1/p), rational x > 0."""
    x = Fr(x)
    S = 1 << (p * PREC)
    lo_n = (x.numerator * S) // x.denominator
    hi_n = -((-x.numerator * S) // x.denominator)
    ylo = iroot_int(lo_n, p)
    yhi = iroot_int(hi_n, p)
    if yhi ** p < hi_n:
        yhi += 1
    return M.IV(Fr(ylo, 1 << PREC), Fr(yhi, 1 << PREC))


def pow_frac(M, x, num, den):
    """Enclosure of x^(num/den) for rational x > 0, integers num, den > 0."""
    return root_iv(M, Fr(x) ** num, den)


def lower_gamma(M, a_num, a_den, x):
    """Enclosure of gamma(a, x) = int_0^x t^(a-1) e^-t dt, a = a_num/a_den > 0, rational x > 0:
    x^a e^-x sum_n x^n / (a (a+1) ... (a+n))."""
    a = Fr(a_num, a_den)
    x = Fr(x)
    term = 1 / a
    s = term
    n = 0
    while True:
        n += 1
        term = term * x / (a + n)
        s += term
        if a + n + 1 > 2 * x and term < Fr(1, 10 ** 70) * s:
            break
    rho = x / (a + n + 1)
    tail = term * rho / (1 - rho)
    series = M.IV(s, s + tail)
    return pow_frac(M, x, a_num, a_den) * M.exp_neg_point(x) * series


def k_integral(M, q_num, q_den, k0, k1):
    """Enclosure of int_{k0}^{k1} k^q exp(-12 k^2) dk = (1/2) 12^(-a) [gamma(a, 12 k1^2) - gamma(a, 12 k0^2)],
    a = (q + 1)/2 = (q_num + q_den)/(2 q_den)."""
    an, ad = q_num + q_den, 2 * q_den
    g = math.gcd(an, ad)
    an, ad = an // g, ad // g
    G1 = lower_gamma(M, an, ad, 12 * Fr(k1) ** 2)
    G0 = lower_gamma(M, an, ad, 12 * Fr(k0) ** 2)
    pre = pow_frac(M, Fr(1, 12), an, ad) * Fr(1, 2)
    return pre * (G1 - G0)


# ------------------------------------------------------------------------------------------- the b-integrals
def m_d2(b, up=True):
    """F(b, sqrt2) = E[(b + sqrt2 z)_+^2] (interval): F increases in sigma."""
    s = E2.isqrt_iv(E2.IV(2))
    lo = E2.elam2_pos(Fr(b), s.lo).lo
    hi = E2.elam2_pos(Fr(b), s.hi).hi
    return E2.IV(lo, hi)


def derivs_d2(b):
    """Upper bounds of m, m', m'' at b (all increasing in b): m' = 2 F_1(b, sqrt2), m'' = 2 Phi(b/sqrt2) <= 2."""
    s = E2.isqrt_iv(E2.IV(2)).hi
    return E2.elam2_pos(Fr(b), s).hi, 2 * E2.elam_pos(Fr(b), s).hi, Fr(2)


def m_d3(b):
    return E3.contact_moment(2, 0, Fr(b))


def derivs_d3(b):
    cm = E3.contact_moment
    return cm(2, 0, Fr(b)).hi, 2 * cm(1, 1, Fr(b)).hi, 2 * cm(0, 2, Fr(b)).hi + 4 * cm(1, 0, Fr(b)).hi


def b_cells(d, nh=4096, block=16):
    """Enclosures of int_{j h}^{(j+1) h} exp(-3 b^2/4) m(b) db, h = 1/nh, j = 0..nh-1 (on [0, 1])."""
    M = E2 if d == 2 else E3
    mfun = m_d2 if d == 2 else m_d3
    dfun = derivs_d2 if d == 2 else derivs_d3
    h = Fr(1, nh)
    out = []
    for blk in range(nh // block):
        x0 = blk * block * h
        x1 = (blk + 1) * block * h
        m, m1, m2 = dfun(x1)
        g2 = M.exp_neg_point(3 * x0 * x0 / 4).hi * ((Fr(9, 4) * x1 * x1 + Fr(3, 2)) * m + 3 * x1 * m1 + m2)
        err = h ** 3 / 24 * g2
        for j in range(block):
            c = x0 + (j + Fr(1, 2)) * h
            val = M.exp_neg_point(3 * c * c / 4) * mfun(c) * h
            out.append(val + M.IV(-err, err))
    return out


def b_integral(M, cells, nh, b0, b1):
    i0, i1 = Fr(b0) * nh, Fr(b1) * nh
    assert i0.denominator == 1 and i1.denominator == 1
    tot = M.IV(0)
    for j in range(int(i0), int(i1)):
        tot = tot + cells[j]
    return tot


def K_const(M, d):
    """|S^(d-1)| 144 (2 pi)^(-n/2) / sqrt12, n = 2(d + 1): A_0^ref / (3 k^p) = K_n k^(2-p) e^(-12k^2) e^(-3b^2/4) m(b)."""
    n = 2 * (d + 1)
    two_pi = 2 * M.PI
    pw = M.IV(1)
    for _ in range(n // 2):
        pw = pw * two_pi
    sphere = two_pi if d == 2 else 2 * two_pi
    return sphere * 144 / pw / M.isqrt_iv(M.IV(12))
