"""Core of the failure-coefficient certificate: Gaussian box moments, verified cubic roots, V = H - 2(B_-)^3 - 2T and H_2,
first- and second-order box enclosures in the scaled jet coordinates (a, beta, c) (a ~ N(0, 2/k), beta ~ N(0, 2),
c ~ N(0, 6k)), with B = beta - a^2/12, u = c + d(a, beta), d = -a beta/4 + a^3/72, T = 3 u^2."""
import math
from ia import (dn, up, iv, add, sub, neg, mul, div, scal, divc, sqr, pw, hull, absmax, isqrt, exp_neg, Phi, SQRT2PI)


# ---------------------------------------------------------------- 1D Gaussian partial moments
def phis(x, s, s2):
    """phi_sigma(x) for a float x, sigma interval s, sigma^2 interval s2."""
    e = exp_neg(div(sqr((x, x)), scal(2.0, s2)))
    return div(e, mul(s, SQRT2PI))


def moments(x0, x1, s, s2, nmax):
    """Intervals M_n = int_{x0}^{x1} x^n phi_sigma(x) dx, n = 0..nmax (x0 < x1 floats)."""
    p0, p1 = phis(x0, s, s2), phis(x1, s, s2)
    F1, F0 = Phi(div((x1, x1), s)), Phi(div((x0, x0), s))
    m0 = sub(F1, F0)
    M = [(max(0.0, m0[0]), m0[1])]
    if nmax >= 1:
        M.append(mul(s2, sub(p0, p1)))
    x0i, x1i = (x0, x0), (x1, x1)
    for n in range(2, nmax + 1):
        t = sub(mul(pw(x0i, n - 1), p0), mul(pw(x1i, n - 1), p1))
        M.append(add(mul(scal(float(n - 1), s2), M[n - 2]), mul(s2, t)))
    return M


# ---------------------------------------------------------------- the cubic f(x) = 2x^3 + 3Bx^2 - B^3 - T
def f_iv(x, B, T):
    """Interval value of f at float point x with float B, T, in the factored (well-conditioned) form
    2x^3 + 3Bx^2 - B^3 - T = (x + B)^2 (2x - B) - T."""
    p = add((x, x), (B, B))
    q = sub(scal(2.0, (x, x)), (B, B))
    return sub(mul(sqr(p), q), (T, T))


def _cbrt_lo(y):
    """A float c with c^3 <= y (y >= 0)."""
    if y <= 0:
        return 0.0
    c = y ** (1.0 / 3.0)
    while pw((c, c), 3)[1] > y:
        c = dn(c)
    return c


def _cbrt_hi(y):
    if y <= 0:
        return 0.0
    c = y ** (1.0 / 3.0)
    while pw((c, c), 3)[0] < y:
        c = up(c)
    return c


def _guess_big(B, T):
    """Float initial guess for the largest root of (x + B)^2 (2x - B) = T (Cardano / trigonometric form)."""
    aB = abs(B)
    if B < 0 and T < aB ** 3:
        c = -1.0 + 2.0 * T / aB ** 3
        phi = math.acos(max(-1.0, min(1.0, c)))
        return aB * (math.cos(phi / 3.0) + 0.5)
    h = B ** 3 / 8.0 + T / 4.0
    sd = math.sqrt(max(0.0, T * (B ** 3 + T) / 16.0))
    A, C = h + sd, h - sd
    y = math.copysign(abs(A) ** (1.0 / 3.0), A) + math.copysign(abs(C) ** (1.0 / 3.0), C)
    return y - B / 2.0


def _guess_mid(B, T):
    """Float initial guess for the middle root (B < 0, 0 <= T <= |B|^3/2)."""
    aB = -B
    c = -1.0 + 2.0 * T / aB ** 3
    phi = math.acos(max(-1.0, min(1.0, c)))
    return aB * (math.cos((2.0 * math.pi - phi) / 3.0) + 0.5)


_U8 = 8.0 * 2.0 ** -53


def _sgn_f(x, B, T):
    """Rigorous sign of f(x) = (x + B)^2 (2x - B) - T at float x, B, T >= 0: +1, -1, or 0 if undecided.
    Binary64 evaluation; the rounding error of p = x + B, q = 2x - B (2x exact), p*p*q and the final subtraction is at most
    5u (p^2 |q| + T) + u |f| (u = 2^-53), below the threshold 8u (p^2 |q| + T) + 4u |f| used here."""
    p = x + B
    q = 2.0 * x - B
    r = p * p * q
    f = r - T
    e = _U8 * (abs(r) + T) + 4.0 * 2.0 ** -53 * abs(f) + 1e-300
    if f > e:
        return 1
    if f < -e:
        return -1
    return 0


def xbig(B, T):
    """Enclosure of the largest root x >= max(-B, B/2) of f = (x + B)^2 (2x - B) - T = 0 (float B, T >= 0); f is
    increasing there and f(max(-B, B/2)) = -T. Newton from the Cardano guess (unverified, only a starting point), then a
    bracket whose endpoint signs are certified by _sgn_f."""
    xm = -B if B < 0 else B / 2.0       # exact
    if T <= 0.0:
        return (xm, xm)
    lo = xm
    hi = xm + 2.0 * (T / 2.0) ** (1.0 / 3.0) + abs(B) + 1.0     # heuristic safeguard only
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
            x = nx
            break
        x = nx
    delta = 8e-16 * max(1.0, abs(x))
    for _ in range(60):
        a, b = max(xm, x - delta), x + delta
        sa = -1 if a <= xm else _sgn_f(a, B, T)
        if sa == -1 and _sgn_f(b, B, T) == 1:
            return (a, b)
        delta *= 4.0
    raise RuntimeError('root bracket failed: B=%r T=%r' % (B, T))


def xmid(B, T):
    """Enclosure of the middle root in [|B|/2, |B|] of f = T for B < 0, 0 <= T <= |B|^3/2 (f decreasing there)."""
    aB = -B
    lo, hi = aB / 2.0, aB      # f(lo) = |B|^3/2 - T >= 0, f(hi) = -T <= 0
    if T <= 0.0:
        return (hi, hi)
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
        a, b = max(aB / 2.0, x - delta), min(aB, x + delta)
        sa = 1 if a <= aB / 2.0 and 0.5 * aB ** 3 - T > 0 else _sgn_f(a, B, T)
        sb = -1 if b >= aB and T > 0 else _sgn_f(b, B, T)
        if sa == 1 and sb == -1:
            return (a, b)
        delta *= 4.0
    raise RuntimeError('middle root bracket failed: B=%r T=%r' % (B, T))


# ---------------------------------------------------------------- V and H_2 at points and their derivatives
def V_iv(Bi, xi):
    """V = -3B(2x - B)(x + B) for interval B, x."""
    return mul(scal(-3.0, Bi), mul(sub(scal(2.0, xi), Bi), add(xi, Bi)))


def V_point(B, T):
    """Enclosure of V(B, T) at a float point (big root)."""
    return V_iv((B, B), xbig(B, T))


def H2_point(B, T):
    """Enclosure of H_2(B, T) = 2|B|^3 + 2T + V(B, T; middle root) on the support, 0 outside."""
    if B >= 0.0:
        return (0.0, 0.0)
    half = scal(0.5, pw((-B, -B), 3))
    if T >= half[1]:
        return (0.0, 0.0)
    if T > half[0]:
        # within rounding of the support boundary: 0 <= H_2(B, T) <= H_2(B, T') for any T' <= T inside the support
        Ts = dn(dn(half[0]) * (1.0 - 1e-12))
        return (0.0, H2_point(B, Ts)[1])
    xm = xmid(B, T)
    Bi = (B, B)
    v = add(add(scal(2.0, pw((-B, -B), 3)), (2 * T, 2 * T)), V_iv(Bi, xm))
    return (max(0.0, v[0]), max(0.0, v[1]))


def cbrt_iv(a):
    return (_cbrt_lo(a[0]), _cbrt_hi(a[1]))


# ---------------------------------------------------------------- ranges of B, d, u, T over a box
def d_iv(ai, bi):
    """d = -a b/4 + a^3/72 for intervals a, b (natural extension)."""
    return add(divc(neg(mul(ai, bi)), 4.0), divc(pw(ai, 3), 72.0))


def d_range(a0, a1, b0, b1):
    """Outward range of d(a, beta) = -a beta/4 + a^3/72 over the rectangle: d is linear in beta, so extremes lie on
    beta = b0, b1; for fixed beta the a-extremes are at a0, a1 or at a = +-sqrt(6 beta) (beta > 0), where
    d = -+ beta^(3/2)/sqrt6."""
    cands = []
    for b in (b0, b1):
        bi = (b, b)
        for a in (a0, a1):
            cands.append(d_iv((a, a), bi))
        if b > 0:
            r = isqrt(scal(6.0, bi))                     # sqrt(6 b)
            val = div(mul(bi, isqrt(bi)), isqrt((6.0, 6.0)))   # b^(3/2)/sqrt6
            if r[0] < a1 and a0 < r[1]:
                cands.append(neg(val))
            if -r[1] < a1 and a0 < -r[0]:
                cands.append(val)
    return (min(c[0] for c in cands), max(c[1] for c in cands))


def box_ranges(box):
    """Outward float ranges (Bl, Bh, Tl, Th) of B and T over the box, plus the u-range."""
    (a0, a1), (b0, b1), (c0, c1) = box
    A2 = sqr((a0, a1))
    Bl = dn(b0 - up(A2[1] / 12.0))
    Bh = up(b1 - dn(A2[0] / 12.0))
    dl, dh = d_range(a0, a1, b0, b1)
    ul, uh = dn(c0 + dl), up(c1 + dh)
    U2 = sqr((ul, uh))
    return Bl, Bh, dn(3.0 * U2[0]) if U2[0] > 0 else 0.0, up(3.0 * U2[1]), (ul, uh)


# ---------------------------------------------------------------- V: first-order (corner) enclosure of the pointwise range
def V_range_box(Bl, Bh, Tl, Th):
    """[min V, max V] over the rectangle [Bl, Bh] x [Tl, Th] (V non-increasing in B; in T non-increasing for B > 0,
    non-decreasing for B < 0)."""
    vmin = V_point(Bh, Th if Bh >= 0 else Tl)[0] if Bh != 0 else 0.0
    vmax = V_point(Bl, Th if Bl <= 0 else Tl)[1] if Bl != 0 else 0.0
    return (vmin, vmax)


def x_range_rect(Bl, Bh, Tl, Th):
    """Range of the big root over the rectangle: increasing in T; for fixed T unimodal in B with minimum x = B at
    B* = (T/4)^(1/3)."""
    xh = max(xbig(Bl, Th)[1], xbig(Bh, Th)[1])
    xl = min(xbig(Bl, Tl)[0], xbig(Bh, Tl)[0])
    if Bh > 0:
        bs_lo, bs_hi = _cbrt_lo(dn(Tl / 4.0)), _cbrt_hi(up(Tl / 4.0))
        if bs_lo <= Bh and Bl <= bs_hi:
            xl = min(xl, bs_lo)
    return (xl, xh)


def V_second_derivs(Bl, Bh, Tl, Th):
    """Upper bounds of |V_BB|, |V_BT|, |V_TT| over the rectangle (requires Tl > 0), or None."""
    X = x_range_rect(Bl, Bh, Tl, Th)
    if X[0] <= 0:
        return None
    xll = xbig(Bl, Tl)
    xhh = xbig(Bh, Th)
    Y = (dn(xll[0] + Bl), up(xhh[1] + Bh))          # x + B: increasing in B and T
    if Y[0] <= 0:
        return None
    xlh = xbig(Bh, Tl)
    xhl = xbig(Bl, Th)
    W = (dn(2 * xlh[0] - Bh), up(2 * xhl[1] - Bl))   # 2x - B: decreasing in B, increasing in T
    Bi = (Bl, Bh)
    Q = add(add(scal(4.0, sqr(X)), scal(2.0, mul(Bi, X))), sqr(Bi))
    X3 = pw(X, 3)
    VTT = div(mul(Bi, Q), scal(12.0, mul(X3, pw(Y, 3))))
    VBT = div(mul(Q, W), scal(4.0, mul(X3, Y)))
    VBB = add(add(scal(3.0, Bi), scal(6.0, X)),
              add(neg(div(scal(4.5, sqr(Bi)), X)), add(neg(div(scal(0.75, pw(Bi, 3)), sqr(X))), div(scal(0.75, pw(Bi, 4)), X3))))
    return absmax(VBB), absmax(VBT), absmax(VTT)


def V_center(Bc, Tc):
    """(V, V_B, V_T) at an interval centre (Bc, Tc) (narrow intervals)."""
    X = x_range_rect(Bc[0], Bc[1], Tc[0], Tc[1])
    xll = xbig(Bc[0], Tc[0])
    xhh = xbig(Bc[1], Tc[1])
    Y = (dn(xll[0] + Bc[0]), up(xhh[1] + Bc[1]))
    V = V_iv(Bc, X)
    VT = neg(div(mul(Bc, add(scal(4.0, X), Bc)), scal(2.0, mul(X, Y))))
    VB = add(add(neg(scal(6.0, sqr(X))), scal(4.5, sqr(Bc))), neg(div(scal(1.5, pw(Bc, 3)), X)))
    return V, VB, VT


# ---------------------------------------------------------------- box integral of V
class Moments:
    """Cache of 1D partial moments per dimension and interval."""
    def __init__(self, sig, sig2, nmax):
        self.sig, self.sig2, self.nmax, self.c = sig, sig2, nmax, {}

    def get(self, x0, x1):
        key = (x0, x1)
        r = self.c.get(key)
        if r is None:
            r = moments(x0, x1, self.sig, self.sig2, self.nmax)
            self.c[key] = r
        return r


def V_box_integral(box, Ma, Mb, Mc, second=True):
    """Enclosure of int_box V dmu (product Gaussian measure), first order (mass x pointwise range) intersected with
    the second-order Taylor enclosure about the box centre."""
    (a0, a1), (b0, b1), (c0, c1) = box
    A, Bm, C = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1)
    m = mul(mul(A[0], Bm[0]), C[0])
    Bl, Bh, Tl, Th, U = box_ranges(box)
    vr = V_range_box(Bl, Bh, Tl, Th)
    I1 = mul(m, vr)
    if not second or Tl <= 0.0:
        return I1
    sd = V_second_derivs(Bl, Bh, Tl, Th)
    if sd is None:
        return I1
    ac, bc, cc = 0.5 * (a0 + a1), 0.5 * (b0 + b1), 0.5 * (c0 + c1)
    aci, bci, cci = (ac, ac), (bc, bc), (cc, cc)
    Bc = sub(bci, divc(sqr(aci), 12.0))
    uc = add(cci, d_iv(aci, bci))
    Tc = scal(3.0, sqr(uc))
    if Tc[0] <= 0:
        return I1
    V0, VB, VT = V_center(Bc, Tc)
    # first moments of (B - Bc) and (T - Tc) over the box
    E1B = sub(mul(mul(sub(Bm[1], mul(bci, Bm[0])), A[0]), C[0]),
              divc(mul(mul(sub(A[2], mul(sqr(aci), A[0])), Bm[0]), C[0]), 12.0))
    Id0 = add(divc(neg(mul(A[1], Bm[1])), 4.0), divc(mul(A[3], Bm[0]), 72.0))
    Id02 = add(add(divc(mul(A[2], Bm[2]), 16.0), divc(neg(mul(A[4], Bm[1])), 144.0)), divc(mul(A[6], Bm[0]), 5184.0))
    Eu2 = add(add(mul(mul(C[2], A[0]), Bm[0]), scal(2.0, mul(C[1], Id0))), mul(C[0], Id02))
    E1T = scal(3.0, sub(Eu2, mul(sqr(uc), m)))
    dB = up(max(Bh - Bc[0], Bc[1] - Bl))
    dT = up(max(Th - Tc[0], Tc[1] - Tl))
    SBB, SBT, STT = sd
    R = up(0.5 * up(up(SBB * dB * dB) + up(2.0 * SBT * dB * dT) + up(STT * dT * dT)))
    mid = add(add(mul(V0, m), mul(VB, E1B)), mul(VT, E1T))
    I2 = sub(mid, mul((R, R), (m[1], m[1]))) , add(mid, mul((R, R), (m[1], m[1])))
    lo2, hi2 = I2[0][0], I2[1][1]
    lo, hi = max(I1[0], lo2), min(I1[1], hi2)
    if lo > hi:
        raise RuntimeError('inconsistent enclosures on box %r: %r vs %r' % (box, I1, (lo2, hi2)))
    return (lo, hi)


# ---------------------------------------------------------------- H_2 (two extra saddles): box integral
def H2_second_derivs_and_ranges(Bl, Bh, Tl, Th):
    """For a rectangle inside the support {B < 0, T < |B|^3/2} with Tl > 0: ranges of the middle root x, of
    y = x + B (< 0), w = 2x - B, and bounds of |H2_BB|, |H2_BT|, |H2_TT|. All of x, y, w are non-increasing in B and T."""
    xll, xhh = xmid(Bl, Tl), xmid(Bh, Th)
    X = (xhh[0], xll[1])
    Y = (dn(xhh[0] + Bh), up(xll[1] + Bl))
    if Y[1] >= 0:
        return None
    W = (dn(2 * xhh[0] - Bh), up(2 * xll[1] - Bl))
    Bi = (Bl, Bh)
    Q = add(add(scal(4.0, sqr(X)), scal(2.0, mul(Bi, X))), sqr(Bi))
    X3 = pw(X, 3)
    VTT = div(mul(Bi, Q), scal(12.0, mul(X3, pw(Y, 3))))
    VBT = div(mul(Q, W), scal(4.0, mul(X3, Y)))
    VBB = add(add(scal(3.0, Bi), scal(6.0, X)),
              add(neg(div(scal(4.5, sqr(Bi)), X)), add(neg(div(scal(0.75, pw(Bi, 3)), sqr(X))), div(scal(0.75, pw(Bi, 4)), X3))))
    HBB = add(VBB, scal(-12.0, Bi))           # + d^2/dB^2 of 2|B|^3 = -12 B for B < 0
    return absmax(HBB), absmax(VBT), absmax(VTT)


def H2_center(Bc, Tc):
    """(H_2, dH_2/dB, dH_2/dT) at a narrow interval centre inside the support."""
    xll, xhh = xmid(Bc[0], Tc[0]), xmid(Bc[1], Tc[1])
    X = (xhh[0], xll[1])
    Y = (dn(xhh[0] + Bc[1]), up(xll[1] + Bc[0]))
    H = add(add(scal(-2.0, pw(Bc, 3)), scal(2.0, Tc)), V_iv(Bc, X))
    HT = add((2.0, 2.0), neg(div(mul(Bc, add(scal(4.0, X), Bc)), scal(2.0, mul(X, Y)))))
    HB = add(scal(-6.0, sqr(Bc)), add(add(neg(scal(6.0, sqr(X))), scal(4.5, sqr(Bc))), neg(div(scal(1.5, pw(Bc, 3)), X))))
    return H, HB, HT


def H2_box_integral(box, Ma, Mb, Mc, second=True):
    (a0, a1), (b0, b1), (c0, c1) = box
    A, Bm, C = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1)
    m = mul(mul(A[0], Bm[0]), C[0])
    Bl, Bh, Tl, Th, U = box_ranges(box)
    if Bl >= 0:
        return (0.0, 0.0)
    hr = (H2_point(Bh, Th)[0], H2_point(Bl, Tl)[1])
    I1 = mul(m, hr)
    if not second or Tl <= 0.0 or Bh >= 0:
        return I1
    edge = scal(0.5, pw((-Bh, -Bh), 3))
    if Th >= edge[0]:
        return I1
    sd = H2_second_derivs_and_ranges(Bl, Bh, Tl, Th)
    if sd is None:
        return I1
    ac, bc, cc = 0.5 * (a0 + a1), 0.5 * (b0 + b1), 0.5 * (c0 + c1)
    aci, bci, cci = (ac, ac), (bc, bc), (cc, cc)
    Bc = sub(bci, divc(sqr(aci), 12.0))
    uc = add(cci, d_iv(aci, bci))
    Tc = scal(3.0, sqr(uc))
    if Tc[0] <= 0 or Bc[1] >= 0:
        return I1
    H0, HB, HT = H2_center(Bc, Tc)
    E1B = sub(mul(mul(sub(Bm[1], mul(bci, Bm[0])), A[0]), C[0]),
              divc(mul(mul(sub(A[2], mul(sqr(aci), A[0])), Bm[0]), C[0]), 12.0))
    Id0 = add(divc(neg(mul(A[1], Bm[1])), 4.0), divc(mul(A[3], Bm[0]), 72.0))
    Id02 = add(add(divc(mul(A[2], Bm[2]), 16.0), divc(neg(mul(A[4], Bm[1])), 144.0)), divc(mul(A[6], Bm[0]), 5184.0))
    Eu2 = add(add(mul(mul(C[2], A[0]), Bm[0]), scal(2.0, mul(C[1], Id0))), mul(C[0], Id02))
    E1T = scal(3.0, sub(Eu2, mul(sqr(uc), m)))
    dB = up(max(Bh - Bc[0], Bc[1] - Bl))
    dT = up(max(Th - Tc[0], Tc[1] - Tl))
    SBB, SBT, STT = sd
    R = up(0.5 * up(up(SBB * dB * dB) + up(2.0 * SBT * dB * dT) + up(STT * dT * dT)))
    mid = add(add(mul(H0, m), mul(HB, E1B)), mul(HT, E1T))
    lo2 = sub(mid, (up(R * m[1]), up(R * m[1])))[0]
    hi2 = add(mid, (up(R * m[1]), up(R * m[1])))[1]
    lo, hi = max(I1[0], lo2), min(I1[1], hi2)
    if lo > hi:
        raise RuntimeError('inconsistent H2 enclosures on box %r' % (box,))
    return (lo, hi)
