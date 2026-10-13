"""One-dimensional and exact parts of J_fail(k) = 3k^2 (E[2(B_-)^3] + E[2T] + E[V]) and tail bounds."""
from fractions import Fraction as Fr
from ia import (dn, up, add, sub, neg, mul, div, scal, divc, sqr, pw, isqrt, exp_neg, Phi, SQRT2PI, from_frac)
import fcore2 as F

SQRT2 = isqrt((2.0, 2.0))


def E2T_exact(k):
    """E[2T] = 6 E[u^2] = 6 (6k + 1/(4k) + 5/(216 k^3)) as an exact rational (k rational)."""
    k = Fr(k)
    return 6 * (6 * k + Fr(1) / (4 * k) + Fr(5) / (216 * k ** 3))


def _gs(mi):
    """(g, g', g'') at an interval m >= 0, beta ~ N(0, 2), t = m/sqrt2:
    g = E[(m - beta)_+^3] = (m^3 + 6m) Phi(t) + sqrt2 (m^2 + 4) phi(t),
    g' = 3 E[(m - beta)_+^2] = 3 [(m^2 + 2) Phi(t) + sqrt2 m phi(t)],   g'' = 6 E[(m - beta)_+] = 6 [m Phi(t) + sqrt2 phi(t)];
    all three are increasing in m."""
    t = div(mi, SQRT2)
    P = F.Phi(t) if hasattr(F, 'Phi') else None
    P = Phi(t)
    ph = div(exp_neg(divc(sqr(mi), 4.0)), SQRT2PI)
    g = add(mul(add(pw(mi, 3), scal(6.0, mi)), P), mul(mul(SQRT2, add(sqr(mi), (4.0, 4.0))), ph))
    g1 = scal(3.0, add(mul(add(sqr(mi), (2.0, 2.0)), P), mul(mul(SQRT2, mi), ph)))
    g2 = scal(6.0, add(mul(mi, P), mul(SQRT2, ph)))
    return g, g1, g2


def E2B(k, L=12.0, N=4000):
    """Enclosure of E[2(B_-)^3] = 4 int_0^inf phi_sa(a) G(a) da, G(a) = g(a^2/12), sa^2 = 2/k: on each cell a second-order
    Taylor expansion about the midpoint with G'' = g''(m) a^2/36 + g'(m)/6 (increasing in a >= 0) bounded at the cell ends;
    plus the tail a > L sa with g(m) <= m^3 + 6m + sqrt2 (m^2 + 4)/sqrt(2 pi)."""
    s2 = (2.0 / k, 2.0 / k)
    assert s2[0] * k == 2.0
    s = isqrt(s2)
    top = L * s[1]
    acc = (0.0, 0.0)
    for i in range(N):
        a0, a1 = top * i / N, top * (i + 1) / N
        M = F.moments(a0, a1, s, s2, 2)
        ac = 0.5 * (a0 + a1)
        aci = (ac, ac)
        mc = divc(sqr(aci), 12.0)
        g, g1, _ = _gs(mc)
        G0, G1 = g, mul(g1, divc(aci, 6.0))
        lo_m, hi_m = divc(sqr((a0, a0)), 12.0), divc(sqr((a1, a1)), 12.0)
        _, g1l, g2l = _gs(lo_m)
        _, g1h, g2h = _gs(hi_m)
        G2lo = add(divc(mul(g2l, sqr((a0, a0))), 36.0), divc(g1l, 6.0))[0]
        G2hi = add(divc(mul(g2h, sqr((a1, a1))), 36.0), divc(g1h, 6.0))[1]
        M1c = sub(M[1], mul(aci, M[0]))
        M2c = F._nonneg(add(sub(M[2], scal(2.0, mul(aci, M[1]))), mul(sqr(aci), M[0])))
        cell = add(add(mul(G0, M[0]), mul(G1, M1c)), scal(0.5, mul((G2lo, G2hi), M2c)))
        acc = add(acc, cell)
    T = tail_moments(top, s, s2, 6)
    inv_sqrt2pi = div((1.0, 1.0), SQRT2PI)
    # a > top: m = a^2/12, g(m) <= m^3 + 6m + sqrt2 (m^2 + 4) / sqrt(2 pi)
    tail = add(add(divc(T[6], 1728.0), divc(T[2], 2.0)),
               mul(mul(SQRT2, inv_sqrt2pi), add(divc(T[4], 144.0), scal(4.0, T[0]))))
    return (scal(4.0, acc)[0], scal(4.0, add(acc, (0.0, tail[1])))[1])


def tail_moments(L, s, s2, nmax):
    """int_L^inf x^n phi_s(x) dx, n = 0..nmax (outward), L > 0."""
    pL = F.phis(L, s, s2)
    Fm = Phi(div((L, L), s))
    M = [(max(0.0, dn(1.0 - Fm[1])), up(1.0 - Fm[0]))]
    M.append(mul(s2, pL))
    Li = (L, L)
    for n in range(2, nmax + 1):
        M.append(add(mul(scal(float(n - 1), s2), M[n - 2]), mul(s2, mul(pw(Li, n - 1), pL))))
    return M


def abs_moments(s, s2, nmax):
    """E|X|^n for X ~ N(0, s^2), n = 0..nmax: 2 int_0^inf x^n phi_s (recurrence; first two exact)."""
    half = tail_moments(0.0, s, s2, nmax) if False else None
    M = [(1.0, 1.0), scal(2.0, mul(s2, F.phis(0.0, s, s2)))]
    for n in range(2, nmax + 1):
        M.append(mul(scal(float(n - 1), s2), M[n - 2]))
    return M


def tail_bound_3d(k, L):
    """Upper bound of E[(68|beta|^3 + (68/1728 + 96/5184) a^6 + 48 c^2 + 6 a^2 beta^2) 1{outside the box}] for the scaled
    law, box [-L sa, L sa] x [-L sb, L sb] x [-L sc, L sc]; this polynomial dominates 17|B|^3 + 8T >= |V|, H, H_2."""
    sig2 = [(2.0 / k, 2.0 / k), (2.0, 2.0), (6.0 * k, 6.0 * k)]
    assert sig2[0][0] * k == 2.0 and sig2[2][0] / k == 6.0
    sig = [isqrt(v) for v in sig2]
    full = [abs_moments(sig[i], sig2[i], 6) for i in range(3)]
    tl = [[scal(2.0, t) for t in tail_moments(L * sig[i][1], sig[i], sig2[i], 6)] for i in range(3)]
    c6 = add(divc((68.0, 68.0), 1728.0), divc((96.0, 96.0), 5184.0))
    terms = [((68.0, 68.0), (0, 3, 0)), (c6, (6, 0, 0)), ((48.0, 48.0), (0, 0, 2)), ((6.0, 6.0), (2, 2, 0))]
    tot = (0.0, 0.0)
    for i in range(3):                 # union bound over the three slab complements
        for cf, pw_ in terms:
            v = cf
            for j in range(3):
                v = mul(v, tl[j][pw_[j]] if j == i else full[j][pw_[j]])
            tot = add(tot, v)
    return tot[1]
