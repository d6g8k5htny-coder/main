"""Four-dimensional box engine for I_k = int_{1/2}^{2} e^{-12k^2} k^{-5/3} J_fail(k) dk, J_fail(k) = 3k^2 E[H]:

    I_k = int_{K} int_{R^3} W(k) H(B, T) phi(a) phi(beta) phi(c) da dbeta dc dk,     W(k) = 3 k^(1/3) e^{-12k^2},

in the original jet coordinates (a, beta, c) ~ N(0, diag(2, 2, 6)) (law free of k), with
    B = beta - a^2/(12k),    u = sqrt(k) c - a beta/(4 sqrt k) + a^3/(72 k^(3/2)),    T = 3u^2
(u = c^ + d(a^, beta) in the scaled coordinates a^ = a/sqrt k, c^ = sqrt(k) c, so the (B, w) Taylor machinery of
fcore2 applies to the image of each 4D box unchanged).  The k-moments K_p = int_cell W(k) k^(p/2) dk, p = -6..2, are
enclosed on dyadic cells of [1/2, 2] by the midpoint and trapezoid rules on convex integrands."""
import math
from ia import dn, up, add, sub, mul, div, scal, divc, sqr, pw, isqrt, exp_neg
import fcore2 as F

K_LO, K_HI = 0.5, 2.0
PMIN, PMAX = -6, 2
KLEVEL = 12                       # elementary dyadic level of k-cells: width 1.5 / 2^12


def _cbrt_iv(x):
    """Enclosure of x^(1/3) for a float x > 0."""
    c = x ** (1.0 / 3.0)
    lo, hi = c, c
    while pw((lo, lo), 3)[1] > x:
        lo = dn(lo)
    while pw((hi, hi), 3)[0] < x:
        hi = up(hi)
    return (lo, hi)


def _fvals(k):
    """Intervals of f_p(k) = 3 k^(1/3) k^(p/2) e^{-12k^2}, p = PMIN..PMAX, at a float k > 0."""
    base = mul(scal(3.0, _cbrt_iv(k)), exp_neg((dn(12.0 * dn(k * k)), up(12.0 * up(k * k)))))
    sk = isqrt((k, k))
    isk = div((1.0, 1.0), sk)
    out = {}
    v = base
    for p in range(0, PMAX + 1):
        out[p] = v
        v = mul(v, sk)
    v = mul(base, isk)
    for p in range(-1, PMIN - 1, -1):
        out[p] = v
        v = mul(v, isk)
    return out


def elementary_kmoments(nsub=24):
    """K_p on the 2^KLEVEL elementary cells: each f_p is convex on [1/2, 2] (f''/f = (24k - q/k)^2 - q/k^2 - 24 > 0 for
    q = 1/3 + p/2 in [-8/3, 4/3]), so on every sub-cell  h f(mid) <= int f <= h (f(l) + f(r))/2."""
    n = 2 ** KLEVEL
    width = (K_HI - K_LO) / n
    cells = []
    prev = _fvals(K_LO)
    for i in range(n):
        k0 = K_LO + width * i
        acc = {p: (0.0, 0.0) for p in range(PMIN, PMAX + 1)}
        for j in range(nsub):
            l = k0 + width * j / nsub
            r = k0 + width * (j + 1) / nsub
            h = (dn(r - l), up(r - l))
            mid = _fvals(0.5 * (l + r))
            fr = _fvals(r)
            for p in acc:
                lo = mul(h, mid[p])[0]
                hi = mul(h, scal(0.5, add(prev[p], fr[p])))[1]
                acc[p] = (dn(acc[p][0] + lo), up(acc[p][1] + hi))
            prev = fr
        cells.append(acc)
    return cells


class KMoments:
    """K_p on dyadic k-cells (unions of elementary cells), cached."""
    def __init__(self, elem):
        self.elem, self.c = elem, {}
        self.n = len(elem)
        self.width = (K_HI - K_LO) / self.n

    def get(self, k0, k1):
        key = (k0, k1)
        r = self.c.get(key)
        if r is None:
            i0 = round((k0 - K_LO) / self.width)
            i1 = round((k1 - K_LO) / self.width)
            if abs(K_LO + i0 * self.width - k0) > 0 or abs(K_LO + i1 * self.width - k1) > 0:
                raise ValueError('k-cell is not a union of elementary cells: %r' % (key,))
            r = {}
            for p in range(PMIN, PMAX + 1):
                lo = hi = 0.0
                for i in range(i0, i1):
                    e = self.elem[i][p]
                    lo, hi = dn(lo + e[0]), up(hi + e[1])
                r[p] = (lo, hi)
            self.c[key] = r
        return r


# ---------------------------------------------------------------- exact polynomial moments of the Taylor increments
def _integrate(terms, A, Bm, C, K):
    acc = (0.0, 0.0)
    for (cf, i, j, l, p) in terms:
        acc = add(acc, mul(cf, mul(mul(A[i], Bm[j]), mul(C[l], K[p]))))
    return acc


def _prod(t1, t2):
    out = {}
    for (c1, i1, j1, l1, p1) in t1:
        for (c2, i2, j2, l2, p2) in t2:
            key = (i1 + i2, j1 + j2, l1 + l2, p1 + p2)
            c = mul(c1, c2)
            out[key] = add(out[key], c) if key in out else c
    return [(c,) + key for key, c in out.items()]


_C12 = div((1.0, 1.0), (12.0, 12.0))
_C4 = (0.25, 0.25)
_C72 = div((1.0, 1.0), (72.0, 72.0))


def taylor_moments4(A, Bm, C, K, Bs, us):
    """(m, int dB, int du, int dB^2, int du^2, int dB du) over a 4D box against W(k) dk x Gaussian, dB = B - Bs,
    du = u - us (terms: coefficient, power of a, beta, c, and p with k^(p/2))."""
    dB = [((1.0, 1.0), 0, 1, 0, 0), (scal(-1.0, _C12), 2, 0, 0, -2), ((-Bs, -Bs), 0, 0, 0, 0)]
    du = [((1.0, 1.0), 0, 0, 1, 1), (scal(-1.0, _C4), 1, 1, 0, -1), (_C72, 3, 0, 0, -3), ((-us, -us), 0, 0, 0, 0)]
    m = _integrate([((1.0, 1.0), 0, 0, 0, 0)], A, Bm, C, K)
    E1B = _integrate(dB, A, Bm, C, K)
    E1u = _integrate(du, A, Bm, C, K)
    QB = _integrate(_prod(dB, dB), A, Bm, C, K)
    Qu = _integrate(_prod(du, du), A, Bm, C, K)
    QBu = _integrate(_prod(dB, du), A, Bm, C, K)
    return m, E1B, E1u, F._nonneg(QB), F._nonneg(Qu), QBu


# ---------------------------------------------------------------- geometry and box integral
def _scaled_range(x0, x1, f0, f1):
    """Range of x * f over [x0, x1] x [f0, f1] with f0 > 0 intervals given as floats (outward)."""
    p = (x0 * f0, x0 * f1, x1 * f0, x1 * f1)
    return dn(min(p)), up(max(p))


def scaled_box(box):
    """Outer box of the scaled coordinates (a^, beta, c^) = (a/sqrt k, beta, sqrt(k) c) over a 4D box."""
    (a0, a1), (b0, b1), (c0, c1), (k0, k1) = box
    sk0, sk1 = isqrt((k0, k0)), isqrt((k1, k1))
    isk = (div((1.0, 1.0), sk1)[0], div((1.0, 1.0), sk0)[1])       # 1/sqrt(k) range
    sk = (sk0[0], sk1[1])
    return (_scaled_range(a0, a1, isk[0], isk[1]), (b0, b1), _scaled_range(c0, c1, sk[0], sk[1]))


def H_box4(box, Ma, Mb, Mc, Km, second=True):
    (a0, a1), (b0, b1), (c0, c1), (k0, k1) = box
    A, Bm, C, K = Ma.get(a0, a1), Mb.get(b0, b1), Mc.get(c0, c1), Km.get(k0, k1)
    sbox = scaled_box(box)
    Bl, Bh, ul, uh = F.box_geometry(sbox)
    s, wl, wh, Tl, Th, corners, hr = F.H_parts((Bl, Bh, ul, uh), None, None, None, None, None)
    m0 = mul(mul(A[0], Bm[0]), mul(C[0], K[0]))
    I1 = mul(m0, hr)
    if not second or s == 0:
        return I1
    ac, bc, cc, kc = 0.5 * (a0 + a1), 0.5 * (b0 + b1), 0.5 * (c0 + c1), 0.5 * (k0 + k1)
    skc = math.sqrt(kc)
    ah, ch = ac / skc, cc * skc
    Bs = min(max(bc - ah * ah / 12.0, Bl), Bh)
    ws = min(max(s * (ch - ah * bc / 4.0 + ah ** 3 / 72.0), wl), wh)
    mom = taylor_moments4(A, Bm, C, K, Bs, s * ws)
    I2 = F.H_second_order(Bl, Bh, Tl, Th, s, ws, Bs, corners, mom)
    if I2 is None:
        return I1
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if lo > hi:
        raise RuntimeError('inconsistent H enclosures on 4D box %r: %r vs %r' % (box, I1, I2))
    return (lo, hi)
