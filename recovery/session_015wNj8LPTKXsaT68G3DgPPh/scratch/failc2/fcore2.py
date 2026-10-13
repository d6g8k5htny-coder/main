"""Box engine for the failure integrals in the scaled jet coordinates (a, beta, c), a ~ N(0, 2/k), beta ~ N(0, 2),
c ~ N(0, 6k) (independent):  B = beta - a^2/12,  u = c + d(a, beta),  d = -a beta/4 + a^3/72,  w = |u|,  T = 3 w^2.

Big root x of (x + B)^2 (2x - B) = T with x >= max(-B, B/2), r = B/x in [-1, 2]:
    V    = -3B(2x - B)(x + B)                          V_B  = x^2 (-6 + 9r^2/2 - 3r^3/2)
    V_w  = -sqrt3 x^(3/2) r (4 + r) (2 - r)^(1/2)
    V_BB = x p(r),  p(r) = 6 + 3r - 9r^2/2 - 3r^3/4 + 3r^4/4
    V_Bw = -(sqrt3/2) x^(1/2) e(r),  e(r) = (2 - r)^(3/2) (4 + 2r + r^2)      (e decreasing on r <= 2)
    V_ww = g(r) = -r (4 - r + r^2)                                               (g decreasing)
Middle root (B < 0, T < |B|^3/2, x in (|B|/2, |B|], r in [-2, -1)): same V, V_B, V_BB, V_ww; V_w and V_Bw change sign.
H_2 = 2|B|^3 + 2T + V(middle root).

Per box: a first-order enclosure (Gaussian mass x pointwise range from monotonicity at the rectangle corners) is
intersected with a second-order Taylor enclosure in (B, w) about a float point (B*, w*) of the image rectangle:
    int V = V* m + V_B* int dB + V_w* int dw + 1/2 int (h_BB dB^2 + 2 h_Bw dB dw + h_ww dw^2),
where the averaged Hessian entries h_ij lie in the interval ranges of the second derivatives over the rectangle; the
second moments int dB^2, int dw^2, int dB dw are exact Gaussian polynomial moments.  Requires the box to lie on one
side of u = 0 (V depends on w = |u|) and x > 0 on the rectangle."""
import math
from ia import (dn, up, add, sub, neg, mul, div, scal, divc, sqr, pw, isqrt, exp_neg, Phi, Q_point, SQRT2PI)

SQRT3 = isqrt((3.0, 3.0))
HSQRT3 = scal(0.5, SQRT3)


# ---------------------------------------------------------------- 1D Gaussian partial moments
def phis(x, s, s2):
    """phi_sigma(x) for a float x, sigma interval s, sigma^2 interval s2."""
    e = exp_neg(div(sqr((x, x)), scal(2.0, s2)))
    return div(e, mul(s, SQRT2PI))


def moments(x0, x1, s, s2, nmax):
    """Intervals M_n = int_{x0}^{x1} x^n phi_sigma(x) dx, n = 0..nmax (floats x0 < x1), by the recurrence
    M_n = (n - 1) s^2 M_{n-2} + s^2 (x0^(n-1) phi(x0) - x1^(n-1) phi(x1))."""
    p0, p1 = phis(x0, s, s2), phis(x1, s, s2)
    m0 = cell_mass(x0, x1, s)
    M = [(max(0.0, m0[0]), m0[1])]
    if nmax >= 1:
        M.append(mul(s2, sub(p0, p1)))
    x0i, x1i = (x0, x0), (x1, x1)
    for n in range(2, nmax + 1):
        t = sub(mul(pw(x0i, n - 1), p0), mul(pw(x1i, n - 1), p1))
        M.append(add(mul(scal(float(n - 1), s2), M[n - 2]), mul(s2, t)))
    return M


def _Qiv(t):
    """Q = 1 - Phi over an interval t >= 0 (decreasing)."""
    return (Q_point(t[1])[0], Q_point(t[0])[1])


def cell_mass(x0, x1, s):
    """Gaussian mass of [x0, x1] (floats) for sigma in the interval s, as a difference of upper tails on the side away
    from 0 (relative precision far out)."""
    t0, t1 = div((x0, x0), s), div((x1, x1), s)
    if x0 >= 0.0:
        return sub(_Qiv(t0), _Qiv(t1))
    if x1 <= 0.0:
        return sub(_Qiv(neg(t1)), _Qiv(neg(t0)))
    return sub(sub((1.0, 1.0), _Qiv(t1)), _Qiv(neg(t0)))


class Moments:
    """Cache of 1D partial moments per interval."""
    def __init__(self, sig, sig2, nmax):
        self.sig, self.sig2, self.nmax, self.c = sig, sig2, nmax, {}

    def get(self, x0, x1):
        key = (x0, x1)
        r = self.c.get(key)
        if r is None:
            r = moments(x0, x1, self.sig, self.sig2, self.nmax)
            self.c[key] = r
        return r


# ---------------------------------------------------------------- verified roots of P(x) = (x + B)^2 (2x - B) = T
_U = 2.0 ** -53


def _sgn_f(x, B, T):
    """Rigorous sign of P(x) - T at floats x, B, T >= 0: +1, -1, or 0 if undecided.  The binary64 rounding error of
    p = x + B, q = 2x - B, p*p*q and the final subtraction is below 5u (p^2 |q| + T) + u |f|; the test uses
    8u (p^2 |q| + T) + 4u |f| + 1e-300."""
    p = x + B
    q = 2.0 * x - B
    r = p * p * q
    f = r - T
    e = 8.0 * _U * (abs(r) + T) + 4.0 * _U * abs(f) + 1e-300
    if f > e:
        return 1
    if f < -e:
        return -1
    return 0


def _guess_big(B, T):
    aB = abs(B)
    if B < 0 and T < aB ** 3:
        c = -1.0 + 2.0 * T / aB ** 3
        ph = math.acos(max(-1.0, min(1.0, c)))
        return aB * (math.cos(ph / 3.0) + 0.5)
    h = B ** 3 / 8.0 + T / 4.0
    sd = math.sqrt(max(0.0, T * (B ** 3 + T) / 16.0))
    A, C = h + sd, h - sd
    y = math.copysign(abs(A) ** (1.0 / 3.0), A) + math.copysign(abs(C) ** (1.0 / 3.0), C)
    return y - B / 2.0


def _guess_mid(B, T):
    aB = -B
    c = -1.0 + 2.0 * T / aB ** 3
    ph = math.acos(max(-1.0, min(1.0, c)))
    return aB * (math.cos((2.0 * math.pi - ph) / 3.0) + 0.5)


def _newton_big(B, T, xm):
    lo = xm
    hi = xm + 2.0 * (T / 2.0) ** (1.0 / 3.0) + abs(B) + 1.0      # heuristic safeguard only (the result is verified)
    x = _guess_big(B, T)
    if not (lo < x < hi):
        x = 0.5 * (lo + hi)
    for _ in range(60):
        fx = (x + B) * (x + B) * (2 * x - B) - T
        d = 6 * x * (x + B)
        if fx > 0:
            hi = min(hi, x)
        else:
            lo = max(lo, x)
        nx = x - fx / d if d > 0 else 0.5 * (lo + hi)
        if not (lo < nx < hi):
            nx = 0.5 * (lo + hi)
        if abs(nx - x) <= 2e-15 * max(1.0, abs(x)):
            return nx
        x = nx
    return x


def xbig(B, Tlo, Thi=None):
    """Interval containing the big root x(B, T) for every T in [Tlo, Thi] (P is increasing on [xm, inf),
    xm = max(-B, B/2), P(xm) = 0): the lower end a has a = xm or P(a) < Tlo, the upper end b has P(b) > Thi."""
    if Thi is None:
        Thi = Tlo
    xm = -B if B < 0 else B / 2.0       # exact
    if Thi <= 0.0:
        return (xm, xm)
    x = _newton_big(B, 0.5 * (Tlo + Thi), xm)
    delta = 8e-16 * max(1.0, abs(x))
    for _ in range(60):
        a, b = max(xm, x - delta), x + delta
        if (a <= xm or _sgn_f(a, B, Tlo) == -1) and _sgn_f(b, B, Thi) == 1:
            return (a, b)
        delta *= 4.0
    raise RuntimeError('big-root bracket failed: B=%r T=[%r, %r]' % (B, Tlo, Thi))


def xmid(B, Tlo, Thi=None):
    """Interval containing the middle root x(B, T) in [|B|/2, |B|] for every T in [Tlo, Thi] subset [0, |B|^3/2]
    (B < 0; P is decreasing on [|B|/2, |B|] from |B|^3/2 to 0): the lower end a has a = |B|/2 or P(a) > Thi,
    the upper end b has b = |B| or P(b) < Tlo."""
    if Thi is None:
        Thi = Tlo
    aB = -B
    lo0, hi0 = aB / 2.0, aB
    if Thi <= 0.0:
        return (hi0, hi0)
    T = 0.5 * (Tlo + Thi)
    lo, hi = lo0, hi0
    x = _guess_mid(B, T)
    if not (lo < x < hi):
        x = 0.5 * (lo + hi)
    for _ in range(80):
        fx = (x + B) * (x + B) * (2 * x - B) - T
        if fx > 0:
            lo = x
        else:
            hi = x
        d = 6 * x * (x + B)
        nx = x - fx / d if d < 0 else 0.5 * (lo + hi)
        if not (lo < nx < hi):
            nx = 0.5 * (lo + hi)
        if abs(nx - x) <= 2e-15 * max(1.0, abs(x)):
            x = nx
            break
        x = nx
    delta = 8e-16 * max(1.0, abs(x))
    for _ in range(60):
        a, b = max(lo0, x - delta), min(hi0, x + delta)
        if (a <= lo0 or _sgn_f(a, B, Thi) == 1) and (b >= hi0 or _sgn_f(b, B, Tlo) == -1):
            return (a, b)
        delta *= 4.0
    raise RuntimeError('middle-root bracket failed: B=%r T=[%r, %r]' % (B, Tlo, Thi))


def _cbrt_lo(y):
    """A float c >= 0 with c^3 <= y (y >= 0)."""
    if y <= 0:
        return 0.0
    c = y ** (1.0 / 3.0)
    while pw((c, c), 3)[1] > y:
        c = dn(c)
    return c


# ---------------------------------------------------------------- pointwise pieces
def V_iv(Bi, xi):
    """V = -3B(2x - B)(x + B) for intervals B, x."""
    return mul(scal(-3.0, Bi), mul(sub(scal(2.0, xi), Bi), add(xi, Bi)))


def _nonneg(a):
    return (max(0.0, a[0]), max(0.0, a[1]))


def VB_iv(Bi, xi):
    """V_B = -6x^2 + 9B^2/2 - 3B^3/(2x) (any root, x > 0)."""
    return add(add(scal(-6.0, sqr(xi)), scal(4.5, sqr(Bi))), neg(div(scal(1.5, pw(Bi, 3)), xi)))


def Vw_big_iv(Bi, xi):
    """V_w = -sqrt3 B (4x + B) (2x - B)^(1/2) / x on the big root."""
    q = _nonneg(sub(scal(2.0, xi), Bi))
    return neg(div(mul(mul(SQRT3, mul(Bi, add(scal(4.0, xi), Bi))), isqrt(q)), xi))


_P = (6.0, 3.0, -4.5, -0.75, 0.75)          # p(r) coefficients, increasing degree
_DP = (3.0, -9.0, -2.25, 3.0)               # p'(r)


def _horner(coef, R):
    acc = (coef[-1], coef[-1])
    for c in reversed(coef[:-1]):
        acc = add(mul(acc, R), (c, c))
    return acc


def p_range(R):
    """Mean-value enclosure of p over the interval R."""
    rc = 0.5 * (R[0] + R[1])
    return add(_horner(_P, (rc, rc)), mul(_horner(_DP, R), sub(R, (rc, rc))))


def _g_pt(r):
    """g(r) = -r(4 - r + r^2) at a float point (interval)."""
    ri = (r, r)
    return neg(mul(ri, add(sub((4.0, 4.0), ri), sqr(ri))))


def g_range(R):
    """g is strictly decreasing on R (g' = -4 + 2r - 3r^2 < 0)."""
    return (_g_pt(R[1])[0], _g_pt(R[0])[1])


def _e_pt(r):
    t = _nonneg(sub((2.0, 2.0), (r, r)))
    ri = (r, r)
    return mul(mul(t, isqrt(t)), add(add((4.0, 4.0), scal(2.0, ri)), sqr(ri)))


def e_range(R):
    """e(r) = (2 - r)^(3/2)(4 + 2r + r^2) is decreasing on r <= 2 (e' = (2-r)^(1/2)(-2 - r - 7r^2/2) < 0)."""
    return (_e_pt(R[1])[0], _e_pt(R[0])[1])


# ---------------------------------------------------------------- ranges over a box
def d_iv(ai, bi):
    return add(divc(neg(mul(ai, bi)), 4.0), divc(pw(ai, 3), 72.0))


def d_range(a0, a1, b0, b1):
    """Outward range of d(a, beta) = -a beta/4 + a^3/72 over a rectangle: d is linear in beta, so the extremes lie on
    beta = b0, b1; for fixed beta the a-extremes are at a0, a1 or at a = +-sqrt(6 beta) (beta > 0), where
    d = -+ beta^(3/2)/sqrt6."""
    cands = []
    for b in (b0, b1):
        bi = (b, b)
        for a in (a0, a1):
            cands.append(d_iv((a, a), bi))
        if b > 0:
            r = isqrt(scal(6.0, bi))
            val = div(mul(bi, isqrt(bi)), isqrt((6.0, 6.0)))
            if r[0] < a1 and a0 < r[1]:
                cands.append(neg(val))
            if -r[1] < a1 and a0 < -r[0]:
                cands.append(val)
    return (min(c[0] for c in cands), max(c[1] for c in cands))


def box_geometry(box):
    """(Bl, Bh, ul, uh): outward ranges of B and u over the box."""
    (a0, a1), (b0, b1), (c0, c1) = box
    A2 = sqr((a0, a1))
    Bl = dn(b0 - up(A2[1] / 12.0))
    Bh = up(b1 - dn(A2[0] / 12.0))
    dl, dh = d_range(a0, a1, b0, b1)
    return Bl, Bh, dn(c0 + dl), up(c1 + dh)


def _wT(ul, uh):
    """(s, wl, wh, Tl, Th): s = sign of u on the box (0 if the box meets u = 0), w = |u| range, T = 3w^2 range."""
    if ul > 0.0:
        s, wl, wh = 1, ul, uh
    elif uh < 0.0:
        s, wl, wh = -1, -uh, -ul
    else:
        s, wl, wh = 0, 0.0, max(-ul, uh)
    Tl = max(0.0, dn(3.0 * max(0.0, dn(wl * wl)))) if wl > 0.0 else 0.0
    Th = up(3.0 * up(wh * wh))
    return s, wl, wh, Tl, Th


# ---------------------------------------------------------------- exact box moments of the Taylor increments
def taylor_moments(A, Bm, C, Bs, us):
    """m = int 1, and with dB = B - Bs, du = u - us:  (m, int dB, int du, int dB^2, int du^2, int dB du)
    over the box (A, Bm, C: partial moment lists, A up to order 6, Bm and C up to order 2)."""
    ab = lambda i, j: mul(A[i], Bm[j])
    ab00 = ab(0, 0)
    ab01, ab02 = ab(0, 1), ab(0, 2)
    ab11, ab12 = ab(1, 1), ab(1, 2)
    ab20, ab21, ab22 = ab(2, 0), ab(2, 1), ab(2, 2)
    ab30, ab31 = ab(3, 0), ab(3, 1)
    ab40, ab41 = ab(4, 0), ab(4, 1)
    ab50, ab60 = ab(5, 0), ab(6, 0)
    Bsi = (Bs, Bs)
    usi = (us, us)
    gB0 = sub(sub(ab01, divc(ab20, 12.0)), mul(Bsi, ab00))
    gBB = add(add(add(ab02, divc(ab40, 144.0)), mul(sqr(Bsi), ab00)),
              add(sub(divc(mul(Bsi, ab20), 6.0), divc(ab21, 6.0)), scal(-2.0, mul(Bsi, ab01))))
    Id1 = add(divc(neg(ab11), 4.0), divc(ab30, 72.0))
    Id2 = add(add(divc(ab22, 16.0), divc(neg(ab41), 144.0)), divc(ab60, 5184.0))
    IBd = sub(add(add(divc(neg(ab12), 4.0), divc(scal(5.0, ab31), 144.0)), divc(neg(ab50), 864.0)), mul(Bsi, Id1))
    C0, C1, C2 = C[0], C[1], C[2]
    Cs1 = sub(C1, mul(usi, C0))
    Cs2 = add(sub(C2, scal(2.0, mul(usi, C1))), mul(sqr(usi), C0))
    m = mul(ab00, C0)
    E1B = mul(gB0, C0)
    E1u = add(mul(ab00, Cs1), mul(Id1, C0))
    QB = mul(gBB, C0)
    Qu = add(add(mul(ab00, Cs2), scal(2.0, mul(Id1, Cs1))), mul(Id2, C0))
    QBu = add(mul(gB0, Cs1), mul(IBd, C0))
    return m, E1B, E1u, _nonneg(QB), _nonneg(Qu), QBu


def _expansion_point(box, Bl, Bh, s, wl, wh):
    (a0, a1), (b0, b1), (c0, c1) = box
    ac, bc, cc = 0.5 * (a0 + a1), 0.5 * (b0 + b1), 0.5 * (c0 + c1)
    Bs = min(max(bc - ac * ac / 12.0, Bl), Bh)
    ws = min(max(s * (cc - ac * bc / 4.0 + ac ** 3 / 72.0), wl), wh)
    return Bs, ws


def _quad(SBB, SBw, Sww, QB, Qu, QBu, s):
    """1/2 (SBB QB + Sww Qu) + s SBw_c QBu +- dBw sqrt(QB Qu) (averaged Hessian entries in the given ranges)."""
    c = 0.5 * (SBw[0] + SBw[1])
    dBw = up(max(up(SBw[1] - c), up(c - SBw[0])))
    cross = scal(float(s), mul((c, c), QBu))
    rad = up(dBw * up(math.sqrt(up(QB[1] * Qu[1]))) * (1.0 + 4 * _U))
    q = add(add(scal(0.5, mul(SBB, QB)), scal(0.5, mul(Sww, Qu))), cross)
    return (dn(q[0] - rad), up(q[1] + rad))


# ---------------------------------------------------------------- box integral of V (big root)
def V_box(box, Ma, Mb, Mc, second=True):
    (a0, a1), (b0, b1), (c0, c1) = box
    A, Bm, C = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1)
    Bl, Bh, ul, uh = box_geometry(box)
    s, wl, wh, Tl, Th = _wT(ul, uh)
    xLL, xLH, xHL, xHH = xbig(Bl, Tl), xbig(Bl, Th), xbig(Bh, Tl), xbig(Bh, Th)
    # pointwise range: V non-increasing in B; in T non-increasing for B > 0, non-decreasing for B < 0; V(0, T) = 0
    if Bh > 0:
        vmin = V_iv((Bh, Bh), xHH)[0]
    elif Bh < 0:
        vmin = V_iv((Bh, Bh), xHL)[0]
    else:
        vmin = 0.0
    if Bl < 0:
        vmax = V_iv((Bl, Bl), xLH)[1]
    elif Bl > 0:
        vmax = V_iv((Bl, Bl), xLL)[1]
    else:
        vmax = 0.0
    m0 = mul(mul(A[0], Bm[0]), C[0])
    I1 = mul(m0, (vmin, vmax))
    if not second or s == 0:
        return I1
    # x-range: increasing in T; for fixed T unimodal in B with minimum x = B at B = (T/4)^(1/3)
    XL, XH = min(xLL[0], xHL[0]), max(xLH[1], xHH[1])
    if Bh > 0:
        bs = _cbrt_lo(max(0.0, dn(Tl / 4.0)))
        if bs <= Bh:
            XL = min(XL, bs)
    if XL <= 0.0:
        return I1
    # r = B/x: non-decreasing in B; in T increasing for B < 0, decreasing for B > 0
    rmin = div((Bl, Bl), xLL)[0] if Bl < 0 else (div((Bl, Bl), xLH)[0] if Bl > 0 else 0.0)
    rmax = div((Bh, Bh), xHL)[1] if Bh > 0 else (div((Bh, Bh), xHH)[1] if Bh < 0 else 0.0)
    R = (max(-1.0, rmin), min(2.0, rmax))
    X = (XL, XH)
    SBB = mul(X, p_range(R))
    SBw = neg(mul(HSQRT3, mul(isqrt(X), e_range(R))))
    Sww = g_range(R)
    Bs, ws = _expansion_point(box, Bl, Bh, s, wl, wh)
    us = s * ws
    w2 = (max(0.0, dn(ws * ws)), up(ws * ws))
    xs = xbig(Bs, dn(3.0 * w2[0]), up(3.0 * w2[1]))
    Bsi = (Bs, Bs)
    V0, VB, Vw = V_iv(Bsi, xs), VB_iv(Bsi, xs), Vw_big_iv(Bsi, xs)
    m, E1B, E1u, QB, Qu, QBu = taylor_moments(A, Bm, C, Bs, us)
    lin = add(add(mul(V0, m), mul(VB, E1B)), scal(float(s), mul(Vw, E1u)))
    I2 = add(lin, _quad(SBB, SBw, Sww, QB, Qu, QBu, s))
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if lo > hi:
        raise RuntimeError('inconsistent V enclosures on box %r: %r vs %r' % (box, I1, I2))
    return (lo, hi)


# ---------------------------------------------------------------- box integral of H_2 (middle root, B < 0)
def _half_cube(B):
    """Interval of |B|^3/2 for float B < 0."""
    return scal(0.5, pw((-B, -B), 3))


def H2_point(B, T):
    """Enclosure of H_2(B, T) = 2|B|^3 + 2T + V(B, T; middle root) on the support {B < 0, T < |B|^3/2}, 0 outside;
    near the support boundary 0 <= H_2(B, T) <= H_2(B, T') for T' <= T inside (H_2 non-increasing in T)."""
    if B >= 0.0:
        return (0.0, 0.0)
    half = _half_cube(B)
    if T >= half[1]:
        return (0.0, 0.0)
    if T > half[0]:
        Ts = dn(dn(half[0]) * (1.0 - 1e-12))
        return (0.0, H2_point(B, Ts)[1])
    xm = xmid(B, T)
    v = add(add(scal(2.0, pw((-B, -B), 3)), (dn(2 * T), up(2 * T))), V_iv((B, B), xm))
    return (max(0.0, v[0]), max(0.0, v[1]))


def H2_box(box, Ma, Mb, Mc, second=True):
    (a0, a1), (b0, b1), (c0, c1) = box
    Bl, Bh, ul, uh = box_geometry(box)
    if Bl >= 0.0:
        return (0.0, 0.0)
    s, wl, wh, Tl, Th = _wT(ul, uh)
    if Tl >= _half_cube(Bl)[1]:
        return (0.0, 0.0)            # every point has T >= |B|^3/2 (support threshold is largest at B = Bl)
    A, Bm, C = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1)
    m0 = mul(mul(A[0], Bm[0]), C[0])
    inside = Bh < 0.0 and Th < _half_cube(Bh)[0]
    if not inside:
        I1 = mul(m0, (H2_point(Bh, Th)[0], H2_point(Bl, Tl)[1]))
        if not second or s == 0:
            return I1
        return _H2_straddle(box, A, Bm, C, Bl, Bh, s, wl, wh, Tl, I1)
    xLL, xHH = xmid(Bl, Tl), xmid(Bh, Th)
    hmin = max(0.0, add(add(scal(2.0, pw((-Bh, -Bh), 3)), (dn(2 * Th), up(2 * Th))), V_iv((Bh, Bh), xHH))[0])
    hmax = add(add(scal(2.0, pw((-Bl, -Bl), 3)), (dn(2 * Tl), up(2 * Tl))), V_iv((Bl, Bl), xLL))[1]
    I1 = mul(m0, (hmin, max(0.0, hmax)))
    if not second or s == 0:
        return I1
    # middle root: x and r = B/x are non-increasing in B and in T
    X = (xHH[0], xLL[1])
    R = (max(-2.0, div((Bh, Bh), xHH)[0]), min(-1.0, div((Bl, Bl), xLL)[1]))
    SBB = add(scal(-12.0, (Bl, Bh)), mul(X, p_range(R)))
    SBw = mul(HSQRT3, mul(isqrt(X), e_range(R)))
    Sww = add((12.0, 12.0), g_range(R))
    Bs, ws = _expansion_point(box, Bl, Bh, s, wl, wh)
    us = s * ws
    w2 = (max(0.0, dn(ws * ws)), up(ws * ws))
    T2 = (dn(3.0 * w2[0]), up(3.0 * w2[1]))
    xs = xmid(Bs, T2[0], T2[1])
    Bsi = (Bs, Bs)
    H0 = add(add(scal(-2.0, pw(Bsi, 3)), scal(2.0, T2)), V_iv(Bsi, xs))
    HB = add(scal(-6.0, sqr(Bsi)), VB_iv(Bsi, xs))
    Hw = add(scal(12.0, (ws, ws)), neg(Vw_big_iv(Bsi, xs)))
    m, E1B, E1u, QB, Qu, QBu = taylor_moments(A, Bm, C, Bs, us)
    lin = add(add(mul(H0, m), mul(HB, E1B)), scal(float(s), mul(Hw, E1u)))
    I2 = add(lin, _quad(SBB, SBw, Sww, QB, Qu, QBu, s))
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if lo > hi:
        raise RuntimeError('inconsistent H2 enclosures on box %r: %r vs %r' % (box, I1, I2))
    return (lo, hi)


# ---------------------------------------------------------------- the full failure integrand H = 2(B_-)^3 + 2T + V
def _Bneg3(Bi):
    """(B_-)^3 for an interval B (non-increasing in B)."""
    lo = pw((max(0.0, -Bi[1]),) * 2, 3)[0]
    hi = pw((max(0.0, -Bi[0]),) * 2, 3)[1]
    return (max(0.0, lo), hi)


def _Bneg(Bi):
    return (max(0.0, -Bi[1]), max(0.0, -Bi[0]))


def H_corner(B, T, xi):
    """H = 2(B_-)^3 + 2T + V at a float corner (B, T) with big-root enclosure xi."""
    return add(add(scal(2.0, _Bneg3((B, B))), (dn(2.0 * T), up(2.0 * T))), V_iv((B, B), xi))


def H_parts(geom, A, Bm, C, Bs_ws, mom_fn, second=True):
    """Shared H enclosure given the image rectangle; mom_fn(Bs, us) returns the six Taylor moments."""
    Bl, Bh, ul, uh = geom
    s, wl, wh, Tl, Th = _wT(ul, uh)
    xLL, xLH, xHL, xHH = xbig(Bl, Tl), xbig(Bl, Th), xbig(Bh, Tl), xbig(Bh, Th)
    # H is non-increasing in B and non-decreasing in T
    hmin = max(0.0, H_corner(Bh, Tl, xHL)[0])
    hmax = H_corner(Bl, Th, xLH)[1]
    return s, wl, wh, Tl, Th, (xLL, xLH, xHL, xHH), (hmin, hmax)


def H_box(box, Ma, Mb, Mc, second=True):
    """Enclosure of int_box H dmu (3D, scaled coordinates)."""
    (a0, a1), (b0, b1), (c0, c1) = box
    A, Bm, C = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1)
    Bl, Bh, ul, uh = box_geometry(box)
    m0 = mul(mul(A[0], Bm[0]), C[0])
    s, wl, wh, Tl, Th, (xLL, xLH, xHL, xHH), hr = H_parts((Bl, Bh, ul, uh), A, Bm, C, None, None)
    I1 = mul(m0, hr)
    if not second or s == 0:
        return I1
    Bs, ws = _expansion_point(box, Bl, Bh, s, wl, wh)
    mom = taylor_moments(A, Bm, C, Bs, s * ws)
    I2 = H_second_order(Bl, Bh, Tl, Th, s, ws, Bs, (xLL, xLH, xHL, xHH), mom)
    if I2 is None:
        return I1
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if lo > hi:
        raise RuntimeError('inconsistent H enclosures on box %r: %r vs %r' % (box, I1, I2))
    return (lo, hi)


def H_second_order(Bl, Bh, Tl, Th, s, ws, Bs, corners, mom):
    """Second-order enclosure of int H over a box whose image lies in [Bl, Bh] x [Tl, Th] on one side of u = 0, about
    the float point (Bs, ws); mom = (m, int dB, int du, int dB^2, int du^2, int dB du).  None if x reaches 0."""
    xLL, xLH, xHL, xHH = corners
    XL, XH = min(xLL[0], xHL[0]), max(xLH[1], xHH[1])
    if Bh > 0:
        bs = _cbrt_lo(max(0.0, dn(Tl / 4.0)))
        if bs <= Bh:
            XL = min(XL, bs)
    if XL <= 0.0:
        return None
    rmin = div((Bl, Bl), xLL)[0] if Bl < 0 else (div((Bl, Bl), xLH)[0] if Bl > 0 else 0.0)
    rmax = div((Bh, Bh), xHL)[1] if Bh > 0 else (div((Bh, Bh), xHH)[1] if Bh < 0 else 0.0)
    R = (max(-1.0, rmin), min(2.0, rmax))
    X = (XL, XH)
    SBB = add(mul(X, p_range(R)), scal(12.0, _Bneg((Bl, Bh))))
    SBw = neg(mul(HSQRT3, mul(isqrt(X), e_range(R))))
    Sww = add(g_range(R), (12.0, 12.0))
    w2 = (max(0.0, dn(ws * ws)), up(ws * ws))
    T2 = (dn(3.0 * w2[0]), up(3.0 * w2[1]))
    xs = xbig(Bs, T2[0], T2[1])
    Bsi = (Bs, Bs)
    H0 = add(add(scal(2.0, _Bneg3(Bsi)), scal(2.0, T2)), V_iv(Bsi, xs))
    HB = sub(VB_iv(Bsi, xs), scal(6.0, sqr(_Bneg(Bsi))))
    Hw = add(Vw_big_iv(Bsi, xs), scal(12.0, (ws, ws)))
    m, E1B, E1u, QB, Qu, QBu = mom
    lin = add(add(mul(H0, m), mul(HB, E1B)), scal(float(s), mul(Hw, E1u)))
    return add(lin, _quad(SBB, SBw, Sww, QB, Qu, QBu, s))


def _hull0(a):
    return (min(0.0, a[0]), max(0.0, a[1]))


def _H2_straddle(box, A, Bm, C, Bl, Bh, s, wl, wh, Tl, I1):
    """Second-order enclosure for a box meeting the support boundary of H_2 (one side of u = 0).  The zero extension of
    H_2 is C^1 with zero value and gradient on the boundary {T = |B|^3/2} (and piecewise C^2), so the Taylor remainder
    holds with averaged Hessian entries in the hull of {0} and the ranges over rectangle and support.  There the middle
    root has |B|/2 <= x <= x(Bl, Tl) and r = B/x in [-2, r(Bl, Tl)] (x, r non-increasing in B and T)."""
    if Tl >= _half_cube(Bl)[0]:
        return I1
    Bhe = min(Bh, 0.0)
    xLL = xmid(Bl, Tl)
    X = (max(0.0, -Bhe / 2.0), xLL[1])
    R = (-2.0, min(-1.0, div((Bl, Bl), xLL)[1]))
    SBB = _hull0(add(scal(-12.0, (Bl, Bhe)), mul(X, p_range(R))))
    SBw = _hull0(mul(HSQRT3, mul(isqrt(X), e_range(R))))
    Sww = _hull0(add((12.0, 12.0), g_range(R)))
    Bs, ws = _expansion_point(box, Bl, Bh, s, wl, wh)
    us = s * ws
    w2 = (max(0.0, dn(ws * ws)), up(ws * ws))
    T2 = (dn(3.0 * w2[0]), up(3.0 * w2[1]))
    if Bs < 0.0 and T2[1] < _half_cube(Bs)[0]:
        xs = xmid(Bs, T2[0], T2[1])
        Bsi = (Bs, Bs)
        H0 = add(add(scal(-2.0, pw(Bsi, 3)), scal(2.0, T2)), V_iv(Bsi, xs))
        HB = add(scal(-6.0, sqr(Bsi)), VB_iv(Bsi, xs))
        Hw = add(scal(12.0, (ws, ws)), neg(Vw_big_iv(Bsi, xs)))
    elif Bs >= 0.0 or T2[0] > _half_cube(Bs)[1]:
        H0 = HB = Hw = (0.0, 0.0)
    else:
        return I1
    m, E1B, E1u, QB, Qu, QBu = taylor_moments(A, Bm, C, Bs, us)
    lin = add(add(mul(H0, m), mul(HB, E1B)), scal(float(s), mul(Hw, E1u)))
    I2 = add(lin, _quad(SBB, SBw, Sww, QB, Qu, QBu, s))
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if lo > hi:
        raise RuntimeError('inconsistent H2 straddle enclosures on box %r: %r vs %r' % (box, I1, I2))
    return (lo, hi)
