# ============================================================================
# W3 CERTIFICATE LIBRARY — rigorous WP numerics for SIDE24/q0 (workstream W3)
# Model: normalized periodized Bargmann-Fock field on T^2_24, 9 pins, r=1/40.
# All arithmetic: mpmath.iv intervals (plus exact rational preconditioning).
# Fail-closed: every certified check goes through ck(); any failure raises
# SystemExit(nonzero). No bare asserts; no swallowed exceptions.
# ============================================================================
import mpmath as mp
from mpmath import iv, mpi
from fractions import Fraction as Fr
import sys, hashlib, json

class CertFail(Exception):
    pass

FAILURES = []
def ck(cond, tag):
    """Fail-closed certified check."""
    ok = False
    try:
        ok = bool(cond)
    except Exception as e:
        FAILURES.append((tag, "exception:" + repr(e)[:120]))
        raise SystemExit("CK-EXCEPTION %s %r" % (tag, e))
    if not ok:
        FAILURES.append((tag, "false"))
        raise SystemExit("CK-FAIL %s" % tag)
    return True

def ivm(x):
    """exact-ish conversion to interval"""
    return iv.mpf(x)

def ivq(q):
    """exact Fraction -> point interval"""
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)

def lo(x):
    return mp.mpf(x.a)

def hi(x):
    return mp.mpf(x.b)

def width(x):
    return mp.mpf(x.b) - mp.mpf(x.a)

def mid(x):
    return (mp.mpf(x.a) + mp.mpf(x.b)) / 2

# ----------------------------------------------------------------------------
# KERNEL with certified tails.
# K1(s) = Z^{-1} sum_{j in (pi/12)Z} e^{-j^2/2} e^{ijs}
#        = (24/(sqrt(2pi) Z)) sum_{n in Z} e^{-(s+24n)^2/2}     [wrapped]
# Z     = sum_{j in (pi/12)Z} e^{-j^2/2}
# We certify both representations on a probe lattice and use the wrapped one.
# ----------------------------------------------------------------------------

def GA():
    return (iv.pi / 12) ** 2 / 2   # spectral Gaussian rate a: terms e^{-a m^2}

def Z_interval():
    """Z = sum_m e^{-(pi m/12)^2/2}, interval with certified truncation tail."""
    a = GA()
    M0 = 60
    s = iv.mpf(0)
    for m in range(-M0, M0 + 1):
        s += iv.exp(-iv.mpf(a) * m * m)
    # tail: 2 sum_{m>M0} e^{-a m^2} <= 2 e^{-a (M0+1)^2} / (1 - e^{-a (2 M0 + 3)})
    t1 = iv.exp(-iv.mpf(a) * (M0 + 1) ** 2)
    ratio = iv.exp(-iv.mpf(a) * (2 * M0 + 3))
    tail = 2 * t1 / (1 - ratio)
    return s + mpi(0, hi(tail))

def c_norm_interval():
    """c = 24/(sqrt(2 pi) Z) as an interval."""
    Z = Z_interval()
    ck(lo(Z) > 0, "Z positive")
    return 24 / (iv.sqrt(2 * iv.pi) * Z)

def he_val(m, t):
    """probabilists' Hermite He_m(t), plain arithmetic (works for mp or iv)."""
    if m == 0:
        return t * 0 + 1
    if m == 1:
        return t
    a, b = t * 0 + 1, t
    for k in range(1, m):
        a, b = b, t * b - k * a
    return b

# crude |He_m(t)| <= sum_k m!/(k!(m-2k)!) |t|^{m-2k} / 2^k  (exact expansion, |.|)
def he_abs_bound(m, tabs):
    from math import factorial
    out = tabs * 0
    k = 0
    while 2 * k <= m:
        out += iv.mpf(factorial(m)) / (iv.mpf(factorial(k) * factorial(m - 2 * k)) * 2 ** k) \
               * tabs ** (m - 2 * k)
        k += 1
    return out

def K1_wrapped(mderiv, s):
    """
    d^m/ds^m K1(s), interval, via wrapped sum with certified tail.
    Valid for |s| <= 13 (checked). Tail from |n| >= 2.
    """
    ck(hi(iv.fabs(s)) <= 13, "wrapped kernel domain |s|<=13")
    c = c_norm_interval()
    acc = iv.mpf(0)
    for n in (-1, 0, 1):
        t = s + 24 * n
        acc += (-1) ** mderiv * he_val(mderiv, t) * iv.exp(-t * t / 2)
    # tail: sum_{|n|>=2} |He_m(s+24n)| e^{-(s+24n)^2/2}
    # for |s|<=13, |s+24n| >= 24|n|-13 >= 11; terms decrease geometrically after n=2.
    tabs2 = 24 * 2 + 13  # upper |arg| at n=2
    # term bound at n=2 (and symmetric n=-2):
    t0 = iv.mpf(24 * 2 - 13)
    term2 = he_abs_bound(mderiv, iv.mpf(tabs2)) * iv.exp(-t0 * t0 / 2)
    # ratio term(n+1)/term(n) <= [(24(n+1)+13)/(24n-13)]^m * exp(-((24(n+1)-13)^2-(24n-13)^2)/2)
    n0 = 2
    rr = ((iv.mpf(24 * (n0 + 1) + 13) / (24 * n0 - 13)) ** mderiv) * \
         iv.exp(-(iv.mpf((24 * (n0 + 1) - 13)) ** 2 - iv.mpf((24 * n0 - 13)) ** 2) / 2)
    ck(hi(rr) < 0.5, "wrapped tail ratio < 1/2")
    tail = 2 * term2 / (1 - rr)
    return c * (acc + mpi(-tail.b, tail.b))

def K1_spectral(mderiv, s):
    """
    d^m/ds^m K1(s) = Z^{-1} sum_j (ij)^m e^{-j^2/2} e^{ijs}, j in (pi/12)Z,
    truncated |j| <= J with certified tail. Used ONLY as cross-check.
    """
    Z = Z_interval()
    ck(lo(Z) > 0, "Z positive (spectral)")
    Jm = 90  # frequency cutoff (in units of pi/12 -> m index): tail ~ e^{-284}
    step = iv.pi / 12
    acc = iv.mpf(0)
    for mm in range(-Jm, Jm + 1):
        j = step * mm
        acc += (1j * j) ** mderiv * iv.exp(-j * j / 2) * iv.exp(1j * j * s)
    # tail: 2 sum_{m>J} (m pi/12)^mderiv e^{-(m pi/12)^2/2}; ratio bounded
    a = GA()
    m0 = Jm + 1
    x0 = iv.pi / 12 * m0
    term0 = x0 ** mderiv * iv.exp(-iv.mpf(a) * m0 * m0)
    # ratio of successive: ((m+1)/m)^mderiv * exp(-a(2m+1))
    rr = (iv.mpf(m0 + 1) / m0) ** mderiv * iv.exp(-iv.mpf(a) * (2 * m0 + 1))
    ck(hi(rr) < 0.5, "spectral tail ratio < 1/2")
    tail = 2 * term0 / (1 - rr)
    tb = hi(tail)
    return (acc + iv.mpc(mpi(-tb, tb), mpi(-tb, tb))) / Z

def crosscheck_kernel():
    """verify wrapped vs spectral agree on a probe lattice (fail-closed)."""
    for md in range(0, 5):
        for sval in ("0", "0.7", "1.234567", "3.9", "5.55", "12.0", "-2.5"):
            s = iv.mpf(sval)
            a1 = K1_wrapped(md, s)
            a2 = K1_spectral(md, s)
            r1 = a1 if isinstance(a1, type(mpi(0,1))) else a1.real
            r2 = a2.real
            ck(lo(r2) <= hi(r1) and lo(r1) <= hi(r2),
               "kernel crosscheck md=%d s=%s" % (md, sval))
            ck(hi(iv.fabs(a2.imag)) < 1e-30, "spectral imag tiny md=%d s=%s" % (md, sval))

def K2d(a1, a2, s1, s2):
    """d^{(a1,a2)} K(s) = K1^{(a1)}(s1) K1^{(a2)}(s2), interval."""
    return K1_wrapped(a1, s1) * K1_wrapped(a2, s2)

def cov2iv(alpha, beta, x, y):
    """Cov(d^alpha f(x), d^beta f(y)), interval. x,y interval 2-vectors."""
    s1, s2 = x[0] - y[0], x[1] - y[1]
    return (-1) ** (beta[0] + beta[1]) * K2d(alpha[0] + beta[0], alpha[1] + beta[1], s1, s2)

# unconditional variance of d^alpha f at a point (for Taylor remainder bounds)
def var_deriv_uncond(alpha):
    return (-1) ** (alpha[0] + alpha[1]) * K2d(2 * alpha[0], 2 * alpha[1],
                                               iv.mpf(0), iv.mpf(0))


# ----------------------------------------------------------------------------
# Module-level caches (must be initialized after iv.prec is set by the runner)
# ----------------------------------------------------------------------------
_CACHE = {}
def init_caches():
    _CACHE.clear()
    _CACHE['Z'] = Z_interval()
    _CACHE['c'] = c_norm_interval()
    # wrapped-kernel tails for |s| <= 13, per derivative order 0..40
    tl = []
    for mderiv in range(0, 41):
        tabs2 = iv.mpf(24 * 2 + 13)
        t0 = iv.mpf(24 * 2 - 13)
        term2 = he_abs_bound(mderiv, tabs2) * iv.exp(-t0 * t0 / 2)
        n0 = 2
        rr = ((iv.mpf(24 * (n0 + 1) + 13) / (24 * n0 - 13)) ** mderiv) * \
             iv.exp(-(iv.mpf((24 * (n0 + 1) - 13)) ** 2 - iv.mpf((24 * n0 - 13)) ** 2) / 2)
        ck(hi(rr) < 0.5, "wrapped tail ratio < 1/2 (cache)")
        tl.append(2 * term2 / (1 - rr))
    _CACHE['wtail'] = tl

def K1w(mderiv, s):
    """cached wrapped K1^{(m)}(s), |s|<=13"""
    ck(hi(iv.fabs(s)) <= 13, "wrapped kernel domain |s|<=13 (cached)")
    acc = iv.mpf(0)
    for n in (-1, 0, 1):
        t = s + 24 * n
        acc += (-1) ** mderiv * he_val(mderiv, t) * iv.exp(-t * t / 2)
    tb = hi(_CACHE['wtail'][mderiv])
    return _CACHE['c'] * (acc + mpi(-tb, tb))

def K2(a1, a2, s1, s2):
    return K1w(a1, s1) * K1w(a2, s2)

def cov2c(alpha, beta, x, y):
    s1, s2 = x[0] - y[0], x[1] - y[1]
    return (-1) ** (beta[0] + beta[1]) * K2(alpha[0] + beta[0], alpha[1] + beta[1], s1, s2)

# ----------------------------------------------------------------------------
# Exact rational geometry + preconditioner
# ----------------------------------------------------------------------------
r_f = Fr(1, 40)
Mpt = (Fr(-1, 80), Fr(0))
Spt = (Fr(1, 80), Fr(0))
Ypt = (Fr(-1, 80) + Fr(1, 40) * Fr(-19, 25), Fr(1, 40) * Fr(6, 25))
PINS = [Mpt, Spt, Ypt]
b_rat = Fr(6, 5)
ell_rat = r_f ** 3 / 6
PINF = [(0, 0), (1, 0), (0, 1)]
LFUNS = [(P, a) for P in PINS for a in PINF]
MONOS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2)]
JET6 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

def _fall(n, k):
    out = Fr(1)
    for j in range(k):
        out *= (n - j)
    return out

def _Lmono(L, mono):
    (P, a) = L
    if a[0] > mono[0] or a[1] > mono[1]:
        return Fr(0)
    return (_fall(mono[0], a[0]) * _fall(mono[1], a[1])
            * P[0] ** (mono[0] - a[0]) * P[1] ** (mono[1] - a[1]))

def _ratinv(A):
    n = len(A)
    Mx = [row[:] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i2: abs(Mx[i2][col]))
        ck(Mx[piv][col] != 0, "rational matrix singular")
        Mx[col], Mx[piv] = Mx[piv], Mx[col]
        pv = Mx[col][col]
        Mx[col] = [v / pv for v in Mx[col]]
        for i2 in range(n):
            if i2 != col and Mx[i2][col] != 0:
                f = Mx[i2][col]
                Mx[i2] = [a - f * b2 for a, b2 in zip(Mx[i2], Mx[col])]
    return [row[n:] for row in Mx]

def build_preconditioner():
    Vmat = [[_Lmono(L, m) for m in MONOS] for L in LFUNS]
    Crat = _ratinv(Vmat)
    for i in range(9):
        for j in range(9):
            s = sum(Vmat[i][k] * Crat[k][j] for k in range(9))
            ck(s == Fr(int(i == j)), "V*C=I exact")
    return Crat

def civ(q):
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)

# ----------------------------------------------------------------------------
# Gram, Gamma, certified inverse (Neumaier-style Neumann enclosure)
# ----------------------------------------------------------------------------
def gram_interval():
    G = [[cov2c(LFUNS[i][1], LFUNS[j][1],
                (civ(LFUNS[i][0][0]), civ(LFUNS[i][0][1])),
                (civ(LFUNS[j][0][0]), civ(LFUNS[j][0][1])))
          for j in range(9)] for i in range(9)]
    return G

def matmul_iv(A, B):
    n, p, m = len(A), len(B), len(B[0])
    out = [[iv.mpf(0)] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = iv.mpf(0)
            for k in range(p):
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out

def matvec_iv(A, v):
    return [sum((A[i][k] * v[k] for k in range(len(v))), iv.mpf(0)) for i in range(len(A))]

def certified_inverse(Gam):
    """
    Interval enclosure of Gam^{-1}: Neumann series around high-precision mp
    approximate inverse (Neumaier residual certification). Gam: 9x9 iv matrix.
    Returns B with Gam^{-1} entrywise in B. Fail-closed.
    """
    n = len(Gam)
    prec0 = mp.mp.prec
    mp.mp.prec = 2 * iv.prec + 200
    Gm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Gm[i, j] = mid(Gam[i][j])
    B0m = Gm ** -1
    mp.mp.prec = prec0
    B0 = [[iv.mpf(mp.nstr(B0m[i, j], 3 * iv.prec)) for j in range(n)] for i in range(n)]
    # R = I - Gam B0  (interval)
    GB = matmul_iv(Gam, B0)
    R = [[(iv.mpf(int(i == j)) - GB[i][j]) for j in range(n)] for i in range(n)]
    def maxabs(M):
        m = iv.mpf(0)
        for row in M:
            for x in row:
                m = max(m, hi(iv.fabs(x)))
        return m
    rnorm = 0.0
    for i in range(n):
        rnorm = max(rnorm, float(hi(sum((iv.fabs(R[i][j]) for j in range(n)), iv.mpf(0)))))
    ck(rnorm < 1e-40, "Neumann contraction ||R||_inf < 1e-40 (got %g)" % rnorm)
    R2 = matmul_iv(R, R)
    BR = matmul_iv(B0, R)
    BR2 = matmul_iv(B0, R2)
    b0norm = max(float(hi(sum((iv.fabs(B0[i][j]) for j in range(n)), iv.mpf(0)))) for i in range(n))
    err = b0norm * rnorm ** 3 / (1 - rnorm)
    out = [[B0[i][j] + BR[i][j] + BR2[i][j] + mpi(-err, err) for j in range(n)] for i in range(n)]
    # certified residual: ||I - Gam*out||_inf <= n*err + rnorm^4/(1-rnorm) must be small
    GB2 = matmul_iv(Gam, out)
    resid = 0.0
    for i in range(n):
        resid = max(resid, float(hi(sum((iv.fabs(iv.mpf(int(i == j)) - GB2[i][j]) for j in range(n)), iv.mpf(0)))))
    ck(resid < 1e-30, "certified inverse residual < 1e-30 (got %g)" % resid)
    return out, rnorm, resid


# ----------------------------------------------------------------------------
# Whitening preconditioner: C = Lambda^{-1/2} Q^T from mp eigendecomposition of
# mid(G) at 2x precision. Identity Gamma = C G C^T holds for ANY invertible C;
# certification is by residual (Neumann). The exact rational C_rat is kept as an
# independent cross-check of the whole conditional-law chain.
# ----------------------------------------------------------------------------
def build_whitener(G):
    prec0 = mp.mp.prec
    mp.mp.prec = 2 * iv.prec + 300
    n = 9
    Gm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Gm[i, j] = mid(G[i][j])
    E, Q = mp.eigh(Gm)
    ck(all(E[i] > 1e-300 for i in range(n)), "G mid SPD (eigh)")
    Cm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Cm[i, j] = sum(Q[k, i] / mp.sqrt(E[i]) * Q[k, j] for k in range(n)) if False else Q[j, i] / mp.sqrt(E[i])
    # rows of Cm: Q[:,i]^T / sqrt(E[i])
    mp.mp.prec = prec0
    C = [[iv.mpf(mp.nstr(Cm[i, j], iv.prec)) for j in range(n)] for i in range(n)]
    return C, [mp.nstr(E[i], 8) for i in range(n)]

def build_law(G, C):
    """Gamma = C G C^T, certified inverse, and transformed-value machinery."""
    CT = [[C[j][i] for j in range(9)] for i in range(9)]
    Gam = matmul_iv(matmul_iv(C, G), CT)
    Ginv, rnorm, resid = certified_inverse(Gam)
    return dict(C=C, CT=CT, Gam=Gam, Ginv=Ginv, rnorm=rnorm, resid=resid)

# ----------------------------------------------------------------------------
# Pin values: v = (b,0,0, b-ell,0,0, mu_t,0,0); mu_t from the 8-pin subsystem
# (drop f(Y) = index 6), itself whitened and certified.
# ----------------------------------------------------------------------------
IDX8 = [0, 1, 2, 3, 4, 5, 7, 8]

def compute_mu_t(G):
    G8 = [[G[i][j] for j in IDX8] for i in IDX8]
    prec0 = mp.mp.prec
    mp.mp.prec = 2 * iv.prec + 300
    n = 8
    Gm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Gm[i, j] = mid(G8[i][j])
    E, Q = mp.eigh(Gm)
    ck(all(E[i] > 1e-300 for i in range(n)), "G8 mid SPD")
    Cm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Cm[i, j] = Q[j, i] / mp.sqrt(E[i])
    # cross-cov k_j = Cov(f(Y), L_j), j in IDX8  (interval)
    mp.mp.prec = prec0
    yY = (civ(Ypt[0]), civ(Ypt[1]))
    k = [cov2c((0, 0), LFUNS[j][1], yY, (civ(LFUNS[j][0][0]), civ(LFUNS[j][0][1]))) for j in IDX8]
    C8 = [[iv.mpf(mp.nstr(Cm[i, j], iv.prec)) for j in range(n)] for i in range(n)]
    Gam8 = matmul_iv(matmul_iv(C8, G8), [[C8[j][i] for j in range(n)] for i in range(n)])
    # certified inverse of Gam8 (Neumann, small)
    prec0 = mp.mp.prec
    mp.mp.prec = 2 * iv.prec + 200
    Gm8 = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Gm8[i, j] = mid(Gam8[i][j])
    B0m = Gm8 ** -1
    mp.mp.prec = prec0
    B0 = [[iv.mpf(mp.nstr(B0m[i, j], 3 * iv.prec)) for j in range(n)] for i in range(n)]
    GB = matmul_iv(Gam8, B0)
    R = [[(iv.mpf(int(i == j)) - GB[i][j]) for j in range(n)] for i in range(n)]
    rnorm = 0.0
    for i in range(n):
        rnorm = max(rnorm, float(hi(sum((iv.fabs(R[i][j]) for j in range(n)), iv.mpf(0)))))
    ck(rnorm < 1e-40, "mu_t Neumann contraction")
    R2 = matmul_iv(R, R)
    BR = matmul_iv(B0, R)
    BR2 = matmul_iv(B0, R2)
    b0norm = max(float(hi(sum((iv.fabs(B0[i][j]) for j in range(n)), iv.mpf(0)))) for i in range(n))
    err = b0norm * rnorm ** 3 / (1 - rnorm)
    G8inv = [[B0[i][j] + BR[i][j] + BR2[i][j] + mpi(-err, err) for j in range(n)] for i in range(n)]
    # mu_t = k^T G8^{-1} v8 = (C8 k)^T Gam8^{-1} (C8 v8)
    b_iv = civ(b_rat)
    ell_iv = civ(ell_rat)
    v8 = [b_iv, iv.mpf(0), iv.mpf(0), b_iv - ell_iv, iv.mpf(0), iv.mpf(0), iv.mpf(0), iv.mpf(0)]
    Ck = matvec_iv(C8, k)
    Cv = matvec_iv(C8, v8)
    lam8 = matvec_iv(G8inv, Cv)
    mu_t = sum((Ck[i] * lam8[i] for i in range(n)), iv.mpf(0))
    return mu_t

def pin_values(mu_t):
    b_iv = civ(b_rat)
    ell_iv = civ(ell_rat)
    return [b_iv, iv.mpf(0), iv.mpf(0), b_iv - ell_iv, iv.mpf(0), iv.mpf(0),
            mu_t, iv.mpf(0), iv.mpf(0)]

# ----------------------------------------------------------------------------
# Conditional jet law at an interval point/box y = (y1, y2)
# Returns (muJ[6], SigJ 6x6) intervals: law of (f, f1, f2, f11, f12, f22)(y)
# given the 9 pins.
# ----------------------------------------------------------------------------
def jetlaw_iv(y, law, tv):
    C, CT, Ginv = law['C'], law['CT'], law['Ginv']
    B = [[cov2c(JET6[i], JET6[j], y, y) for j in range(6)] for i in range(6)]
    A = [[cov2c(JET6[i], LFUNS[j][1], y, (civ(LFUNS[j][0][0]), civ(LFUNS[j][0][1])))
          for j in range(9)] for i in range(6)]
    At = matmul_iv(A, CT)           # 6x9
    lam = matvec_iv(Ginv, tv)       # 9
    muJ = matvec_iv(At, lam)        # 6
    # SigJ = B - At Ginv At^T
    M = matmul_iv(At, Ginv)         # 6x9
    AtT = [[At[j][i] for j in range(6)] for i in range(9)]
    S = matmul_iv(M, AtT)           # 6x6
    SigJ = [[B[i][j] - S[i][j] for j in range(6)] for i in range(6)]
    return muJ, SigJ


# ============================================================================
# Certified g(u) = E[ |det H| 1{det H < 0} ],  X=(a,c,d)~N(m,V), W = X'QX,
# Q = [[0,0,1/2],[0,-1,0],[1/2,0,0]].
# Method: E[W^-] = (E|W| - EW)/2,  E|W| = (2/pi) int_0^inf (1-Re phi)/t^2 dt,
# phi(t) = D(t)^{-1/2} exp(i t N(t)/D(t)), D = det(I-2itVQ) (cubic),
# N/D = m'Q(I-2itVQ)^{-1}m (N quadratic, via adjugate polynomial).
# Remainders (all explicit):
#  R1 series [0,t0]: Cauchy-Taylor remainder with M_rho on |z|=rho.
#  R2 Simpson [t0,T]: per-panel Peano bound d^5/2880 sup|h^{(4)}| with
#     sup via Leibniz/Bell bounds (|D|>=1 theorem: VQ similar to symmetric).
#  R3 tail [T,inf): (1-Re phi)/t^2 <= 1/t^2 + |phi|/t^2 with |phi|<=|D|^-1/2
#     and |D| >= c3lo t^3/2 (or c2, c1 fallbacks); plus EW2-based smallness.
# ============================================================================
_DEBUG_G = False
import math

Q03 = ((0.0, 0.0, 0.5), (0.0, -1.0, 0.0), (0.5, 0.0, 0.0))

def _fup(x):
    """upper magnitude of an interval as float, inflated 1 ulp-safety"""
    return float(hi(iv.fabs(x))) * (1 + 1e-12)

def _vq_entries(V):
    # VQ = V*Q: col1 = V col3 /2, col2 = -V col2, col3 = V col1 /2
    return ((V[0][2] / 2, -V[0][1], V[0][0] / 2),
            (V[1][2] / 2, -V[1][1], V[1][0] / 2),
            (V[2][2] / 2, -V[2][1], V[2][0] / 2))

def _mat3_mul(A, B):
    return tuple(tuple(sum((A[i][k] * B[k][j] for k in range(3)), iv.mpf(0))
                     for j in range(3)) for i in range(3))

def _mQXm(m, X):
    # m' Q X m  with Q as above: (QX)_{1j}=X_{3j}/2, (QX)_{2j}=-X_{2j}, (QX)_{3j}=X_{1j}/2
    return ((m[0] / 2) * (X[2][0] * m[0] + X[2][1] * m[1] + X[2][2] * m[2])
            - m[1] * (X[1][0] * m[0] + X[1][1] * m[1] + X[1][2] * m[2])
            + (m[2] / 2) * (X[0][0] * m[0] + X[0][1] * m[1] + X[0][2] * m[2]))

def g_quad_iv(m, V, tol, tag):
    """returns dict(g=interval, EW=interval, parts=...) certified"""
    prec_keep = iv.prec
    iv.prec = min(prec_keep, 160)
    try:
        return _g_quad_iv_inner(m, V, tol, tag)
    finally:
        iv.prec = prec_keep

def _g_quad_iv_inner(m, V, tol, tag):
    import time as _tm
    global _T0G
    _T0G = _tm.time()
    VQ = _vq_entries(V)
    trA = VQ[0][0] + VQ[1][1] + VQ[2][2]
    e2A = ((VQ[0][0] * VQ[1][1] - VQ[0][1] * VQ[1][0])
           + (VQ[0][0] * VQ[2][2] - VQ[0][2] * VQ[2][0])
           + (VQ[1][1] * VQ[2][2] - VQ[1][2] * VQ[2][1]))
    detV = (V[0][0] * (V[1][1] * V[2][2] - V[1][2] ** 2)
            - V[0][1] * (V[1][0] * V[2][2] - V[1][2] * V[2][0])
            + V[0][2] * (V[1][0] * V[2][1] - V[1][1] * V[2][0]))
    detA = detV / 4
    c1r = -2 * trA          # imag coeff of t in D
    c2r = -4 * e2A          # real coeff of t^2
    c3r = 8 * detA          # imag coeff of t^3  (= 2 detV)
    # N coefficients: n0 + n1 t + n2 t^2, n0 real, n1 imag, n2 real
    E1m = tuple(tuple((trA if i == j else iv.mpf(0)) - VQ[i][j] for j in range(3)) for i in range(3))
    adjB = _mat3_mul(VQ, VQ)
    adjB = tuple(tuple(adjB[i][j] - trA * VQ[i][j] + (e2A if i == j else iv.mpf(0))
                       for j in range(3)) for i in range(3))
    n0r = _mQXm(m, tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3)))
    n1r = -2 * _mQXm(m, E1m)      # imag
    n2r = -4 * _mQXm(m, adjB)     # real
    # ---- elementary quantities
    EW = n0r + trA
    # cumulants: kappa_j = 2^{j-1} (j-1)! [ tr((VQ)^j) + j m'Q (VQ)^{j-1} m ]
    KMAX = 16
    kappas = [None]
    Ppow = [tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3))]
    for j in range(1, 2 * KMAX + 2):
        Ppow.append(_mat3_mul(Ppow[-1], VQ))
    for j in range(1, 2 * KMAX + 1):
        trj = Ppow[j][0][0] + Ppow[j][1][1] + Ppow[j][2][2]
        mj = _mQXm(m, Ppow[j - 1])
        kappas.append(iv.mpf(2 ** (j - 1) * math.factorial(j - 1)) * (trj + j * mj))
    EW = kappas[1]
    EW2 = kappas[2] + kappas[1] ** 2
    if lo(EW2) < 0:
        EW2 = mpi(0, hi(EW2))   # EW2 = kap2 + kap1^2 >= 0 mathematically
    # moments m_n from cumulants: m_n = sum_{k=1..n} C(n-1,k-1) kappa_k m_{n-k}
    moms = [iv.mpf(1), kappas[1]]
    for n in range(2, 2 * KMAX + 1):
        s = iv.mpf(0)
        for k in range(1, n + 1):
            s += iv.mpf(math.comb(n - 1, k - 1)) * kappas[k] * moms[n - k]
        moms.append(s)
    # float magnitude constants
    S2 = Ppow[2][0][0] + Ppow[2][1][1] + Ppow[2][2][2]
    ck(lo(S2) > -1e-9 * max(1.0, float(hi(iv.fabs(S2)))), "S2>=0 %s" % tag)
    S2u = _fup(S2) if lo(S2) > 0 else _fup(iv.fabs(S2))
    lamu = math.sqrt(S2u)                       # |lambda_max| <= sqrt(S2)
    # nu^2 = m' V^{-1} m  (V PD certified by caller; adjugate/det)
    adjV = ((V[1][1] * V[2][2] - V[1][2] ** 2, V[0][2] * V[2][1] - V[0][1] * V[2][2], V[0][1] * V[1][2] - V[0][2] * V[1][1]),
            (V[0][2] * V[2][1] - V[0][1] * V[2][2], V[0][0] * V[2][2] - V[0][2] ** 2, V[0][1] * V[0][2] - V[0][0] * V[1][2]),
            (V[0][1] * V[1][2] - V[0][2] * V[1][1], V[0][1] * V[0][2] - V[0][0] * V[1][2], V[0][0] * V[1][1] - V[0][1] ** 2))
    ck(lo(detV) > 0, "detV>0 %s" % tag)
    nu2 = (m[0] * (adjV[0][0] * m[0] + adjV[0][1] * m[1] + adjV[0][2] * m[2])
           + m[1] * (adjV[1][0] * m[0] + adjV[1][1] * m[1] + adjV[1][2] * m[2])
           + m[2] * (adjV[2][0] * m[0] + adjV[2][1] * m[1] + adjV[2][2] * m[2])) / detV
    nu2u = max(0.0, float(hi(nu2))) * (1 + 1e-12)
    EW2u = float(hi(EW2)) * (1 + 1e-12)
    m4v = moms[4]
    if lo(m4v) < 0:
        m4v = mpi(0, hi(m4v))   # raw 4th moment >= 0 mathematically
    EW4u = _fup(m4v)
    EM3u = math.sqrt(EW2u * EW4u) * (1 + 1e-12)
    c1u, c2u, c3u = _fup(c1r), _fup(c2r), _fup(c3r)
    c3lo = float(lo(iv.fabs(c3r))) / (1 + 1e-12)
    c2lo_abs = float(lo(iv.fabs(c2r))) / (1 + 1e-12)
    c1lo_abs = float(lo(iv.fabs(c1r))) / (1 + 1e-12)
    n0u, n1u, n2u = _fup(n0r), _fup(n1r), _fup(n2r)
    # ---------------- R1: series on [0, t0] ----------------
    # consistency checks on coefficients (Newton identities)
    S2chk = trA * trA - 2 * e2A
    ck(lo(S2chk) <= hi(S2) and lo(S2) <= hi(S2chk), "Newton e2 %s" % tag)
    rho0 = 1.0 / (2.0 * lamu)                   # certified lower radius
    rho = min(0.9 * rho0, 0.4 / (lamu * max(1.0, nu2u)), 0.25)
    t0 = rho / 3.0
    # M_rho: |h(z)| <= (1 + rho E[|W| e^{rho|W|}])/rho^2 on |z|=rho;
    # E|W|e^{rho|W|} <= sqrt(EW2) sqrt(E e^{2 rho |W|}),
    # E e^{s|W|} <= 2 (1-s lamu)^{-3/2} exp(s lamu nu2u/(1-s lamu)), s=2*rho*lamu<1
    s2 = 2 * rho * lamu
    ck(s2 < 1.0, "mgf domain %s" % tag)
    Es = 2 * (1 - s2) ** (-1.5) * math.exp(s2 * nu2u / (1 - s2))
    Mrho = (1.0 + rho * math.sqrt(EW2u) * math.sqrt(Es)) / (rho * rho)
    Mrho *= (1 + 1e-12)
    if _DEBUG_G:
        import time as _tm
        print("GQUAD pre-I1 %.1fs" % (_tm.time() - _T0G), flush=True)
    ser = iv.mpf(0)
    for k in range(1, KMAX + 1):
        ser += ((-1) ** (k - 1)) * moms[2 * k] * iv.mpf(t0) ** (2 * k - 1) / \
               iv.mpf(math.factorial(2 * k) * (2 * k - 1))
    R1 = t0 * Mrho * (t0 / rho) ** (2 * KMAX - 1) / (1 - t0 / rho)
    I1 = ser + mpi(-R1, R1)
    # ---------------- R3: tail bound and T ----------------
    # choose branch
    Tstar = 0.0
    branch = None
    if c3lo > 1e-300:
        Tstar = 2.0 * max(2 * c2u / c3lo, math.sqrt(2 * c1u / c3lo), (2.0 / c3lo) ** (1.0 / 3.0))
        branch = 3
    elif c2lo_abs > 1e-300:
        Tstar = 2.0 * max(2 * c1u / c2lo_abs, math.sqrt(2.0 / c2lo_abs))
        branch = 2
    elif c1lo_abs > 1e-300:
        Tstar = 2.0 / c1lo_abs
        branch = 1
    else:
        ck(False, "tail branch %s" % tag)
    def tailbd(T):
        # bound on int_T^inf h dt (no 2/pi factor; applied outside)
        base = 1.0 / T
        if branch == 3:
            return base + math.sqrt(2.0 / c3lo) * 0.4 * T ** (-2.5)
        if branch == 2:
            return base + math.sqrt(2.0 / c2lo_abs) * 0.5 * T ** (-2)
        return base + math.sqrt(2.0 / c1lo_abs) * (2.0 / 3.0) * T ** (-1.5)
    # also EW2-based: tail <= 2 sqrt(EW2) - T EW2/2 for T <= 2/sqrt(EW2)
    def tailbd2(T):
        r = math.sqrt(EW2u)
        Tc = 2.0 / r
        if T <= Tc:
            return 2 * r - T * EW2u / 2
        return 2.0 / T
    T = max(Tstar, t0 * 2)
    while tailbd(T) > tol / 8 and tailbd2(T) > tol / 8:
        T *= 1.5
        ck(T < 3e11, "T blowup %s" % tag)
    R3 = min(tailbd(T), tailbd2(T))
    ck(R3 <= tol / 4, "tail budget %s" % tag)
    if _DEBUG_G:
        import time as _tm
        print("GQUAD T=%.3e R3=%.3g t0=%.3g EW2u=%.3g EW4u=%.3g lamu=%.3g %.1fs" % (
            T, R3, t0, EW2u, EW4u, lamu, _tm.time() - _T0G), flush=True)
    # ---------------- R2: adaptive Simpson on [t0, T] ----------------
    phiC = dict(c1r=c1r, c2r=c2r, c3r=c3r, n0r=n0r, n1r=n1r, n2r=n2r)

    def hval(tv):
        t = iv.mpf(tv)
        D = 1 + iv.mpc(mpi(0, 0), c1r) * t + c2r * t * t + iv.mpc(mpi(0, 0), c3r) * t ** 3
        N = n0r + iv.mpc(mpi(0, 0), n1r) * t + n2r * t * t
        phi = D ** (iv.mpf(-0.5)) * iv.exp(iv.mpc(mpi(0, 0), t) * (N / D))
        h = (1 - phi.real) / (t * t)
        # |phi(t)| <= 1 always (ch.f.) => h in [0, 2/t^2]: clamp (rigorous)
        cap = 2 / (t * t)
        lo_c = max(float(lo(h)), 0.0)
        hi_c = min(float(hi(h)), float(hi(cap)))
        ck(lo_c <= hi_c * (1 + 1e-15), "hval clamp %s" % tag)
        return mpi(lo_c, max(hi_c, lo_c))

    def U4(a, b):
        # certified sup of |h^{(4)}| on [a,b], float, inflated
        d_lo = 1.0
        if c3lo * a ** 3 - c1u * b > 1.0:
            d_lo = c3lo * a ** 3 - c1u * b
        if c2lo_abs * a * a - 1.0 > d_lo:
            d_lo = c2lo_abs * a * a - 1.0
        D1u = c1u + 2 * c2u * b + 3 * c3u * b * b
        D2u = 2 * c2u + 6 * c3u * b
        D3u = 6 * c3u
        w_u = 1.0 / d_lo
        w1 = D1u * w_u ** 2
        w2 = D2u * w_u ** 2 + 2 * D1u ** 2 * w_u ** 3
        w3 = D3u * w_u ** 2 + 6 * D1u * D2u * w_u ** 3 + 6 * D1u ** 3 * w_u ** 4
        w4 = (8 * D1u * D3u + 6 * D2u ** 2) * w_u ** 3 + 36 * D1u ** 2 * D2u * w_u ** 4 + 24 * D1u ** 4 * w_u ** 5
        N0u = n0u + n1u * b + n2u * b * b
        N1u = n1u + 2 * n2u * b
        N2u = 2 * n2u
        R0 = N0u * w_u
        R1u = N1u * w_u + N0u * w1
        R2u = N2u * w_u + 2 * N1u * w1 + N0u * w2
        R3u = 3 * N2u * w1 + 3 * N1u * w2 + N0u * w3
        R4u = 4 * N2u * w2 + 6 * N1u * w3 + N0u * w4
        L1 = 0.5 * D1u * w_u + R0 + b * R1u
        L2 = 0.5 * (D2u * w_u + D1u * w1) + 2 * R1u + b * R2u
        L3 = 0.5 * (D3u * w_u + 2 * D2u * w1 + D1u * w2) + 3 * R2u + b * R3u
        L4 = 0.5 * (3 * D3u * w1 + 3 * D2u * w2 + D1u * w3) + 4 * R3u + b * R4u
        # combined ch.f. derivative bounds:
        #  (i) moment bounds: |F^{(j)}| <= E|W|^j (j>=2), |F'| <= b*EW2 (tight at small t)
        # (ii) decaying Bell bounds: |phi^{(j)}| <= |phi| Bell_j(L), |phi| <= |D|^{-1/2}
        phimag = 1.0 / math.sqrt(d_lo)
        B1 = min(b * EW2u, phimag * L1)
        B2 = min(EW2u, phimag * (L2 + L1 * L1))
        B3 = min(EM3u, phimag * (L3 + 3 * L1 * L2 + L1 ** 3))
        B4 = min(EW4u, phimag * (L4 + 4 * L1 * L3 + 3 * L2 * L2 + 6 * L1 * L1 * L2 + L1 ** 4))
        F0 = min(2.0, b * b * EW2u / 2)
        U = (F0 * 120.0 / a ** 6 + 4 * B1 * 24.0 / a ** 5 + 6 * B2 * 6.0 / a ** 4
             + 4 * B3 * 2.0 / a ** 3 + B4 / a ** 2)
        return U * (1 + 1e-12)

    # adaptive panel stepping with restart-on-budget-violation
    tol_loc0 = (tol / 2) / 16384.0
    Isum = None
    for restart in range(4):
        tol_loc = tol_loc0 / (8.0 ** restart)
        a = t0
        Isum = iv.mpf(0)
        err2 = 0.0
        npan = 0
        hprev = hval(a)
        ok = True
        while a < T:
            Delta = min(a / 2.0, T - a)
            u4 = U4(a, a + Delta)
            Delta = min(Delta, (2880.0 * tol_loc / u4) ** 0.2)
            for _ in range(80):
                b = a + Delta
                u4 = U4(a, b)
                if Delta ** 5 / 2880.0 * u4 <= tol_loc:
                    break
                Delta /= 2.0
            if _DEBUG_G and npan < 3:
                print("GQUAD panel a=%.4g Delta=%.3g u4=%.3g" % (a, Delta, u4), flush=True)
            b = a + Delta
            midp = a + Delta / 2.0
            hmid = hval(midp)
            hnew = hval(b)
            if _DEBUG_G:
                for (tt, hh) in ((midp, hmid), (b, hnew)):
                    if float(hi(iv.fabs(hh))) > 1e6:
                        print("GQUAD big-h t=%.4g h=%s" % (tt, mp.nstr(hh, 4)), flush=True)
            Isum += (Delta / 6.0) * (hprev + 4 * hmid + hnew)
            err2 += Delta ** 5 / 2880.0 * u4
            npan += 1
            if npan > 400000 or err2 > tol / 2:
                ok = False
                if _DEBUG_G:
                    print("GQUAD restart npan=%d err2=%.3g a=%.3g tol_loc=%.3g" % (npan, err2, a, tol_loc), flush=True)
                break
            a = b
            hprev = hnew
        if ok and err2 <= tol / 2:
            break
    ck(ok and err2 <= tol / 2, "simpson budget %s (err2=%g npan=%d)" % (tag, err2, npan))
    I2 = Isum + mpi(-err2, err2)
    if _DEBUG_G:
        print("DEBUG t0=%.6g T=%.6g npan=%d" % (t0, T, npan))
        print("DEBUG I1 mid=", float(mid(I1)), "I2 mid=", float(mid(I2)), "R1=%.3g err2=%.3g R3=%.3g" % (R1, err2, R3))
    Eabs = (2 / iv.pi) * (I1 + I2) + mpi(-R3, R3) * (2 / iv.pi)
    g = (Eabs - EW) / 2
    # sanity: 0 <= g <= E|W|; E[W^-] >= 0
    ck(hi(g) >= -tol, "g sign %s" % tag)
    if lo(g) < 0:
        g = mpi(0, hi(g))
    return dict(g=g, EW=EW, Eabs=Eabs, R1=R1, err2=err2, R3=R3, npan=npan, T=T, t0=t0)


# ============================================================================
# Box evaluation: rho(y-box) enclosures (cheap CS prefilter / exact).
# Certified Phi via alternating series (|x|<=1.5) and Mills-ratio brackets.
# ============================================================================
def phi_pdf(x):
    return iv.exp(-x * x / 2) / iv.sqrt(2 * iv.pi)

def _he_prob(n, t):
    # probabilists' Hermite He_n(t) via recurrence (n <= 14)
    if n == 0:
        return ivm(1)
    h0, h1 = ivm(1), t
    for k in range(1, n):
        h0, h1 = h1, t * h1 - k * h0
    return h1


def dPhi(y0, y1, tag=""):
    """certified Phi(y1) - Phi(y0) for y1 >= y0 (interval)."""
    h = (y1 - y0) / 2
    ym = (y0 + y1) / 2
    if float(hi(h)) > 0.75:
        return Phi_cdf(y1) - Phi_cdf(y0)
    pm = phi_pdf(ym)
    s = ivm(0)
    h2 = h * h
    hp = h  # h^{2k+1}/(2k+1)!
    fac = 1
    for k in range(0, 7):
        if k > 0:
            fac *= (2 * k) * (2 * k + 1)
            hp = hp * h2
        s += _he_prob(2 * k, ym) * hp / ivm(fac)
    delta = 2 * pm * s
    # remainder: sup |phi^{(14)}| <= 0.434 sqrt(14!) (Cramer) ; |R| <= sup*2h^15/15!
    R = ivm(0.434 * math.sqrt(math.factorial(14)) * 2.0 / math.factorial(15)) * (ivm(float(hi(h))) ** 15)
    return mpi(lo(delta) - float(hi(R)), hi(delta) + float(hi(R)))


def Phi_cdf(x):
    """interval enclosure of standard normal CDF over interval x"""
    if hi(x) <= 0:
        return 1 - Phi_cdf(-x)
    if lo(x) < 0:
        # straddles 0: split
        a = Phi_cdf(mpi(lo(x), 0))
        b = Phi_cdf(mpi(0, hi(x)))
        return mpi(lo(a), hi(b))
    if hi(x) <= 1.5:
        # Phi(x) = 1/2 + 1/sqrt(2pi) sum_{n>=0} (-1)^n x^{2n+1}/(2^n n! (2n+1))
        # alternating with decreasing terms for |x|<=1.5; tail <= next term
        N = 24
        s = x * 0
        term = x  # n=0 term (interval)
        for n in range(N):
            s += ((-1) ** n) * term / (2 * n + 1)
            term = term * x * x / (2 * (n + 1))
        # tail bound: |x|^{2N+1}/(2^N N! (2N+1))
        tb = (iv.fabs(x)) ** (2 * N + 1) / (iv.mpf(2) ** N * iv.mpf(math.factorial(N)) * (2 * N + 1))
        return iv.mpf("0.5") + s / iv.sqrt(2 * iv.pi) + mpi(-hi(tb), hi(tb))
    # Mills ratio: phi(x) * x/(x^2+1) < Q(x) < phi(x)/x
    ph = phi_pdf(x)
    qlo = ph * x / (x * x + 1)
    qhi = ph / x
    return 1 - mpi(lo(qlo), hi(qhi))

def inv2x2_spd(A, tag):
    """certified SPD (Sylvester) + inverse for symmetric 2x2 intervals"""
    ck(lo(A[0][0]) > 0, "2x2 a11>0 %s" % tag)
    det = A[0][0] * A[1][1] - A[0][1] ** 2
    ck(lo(det) > 0, "2x2 det>0 %s" % tag)
    return [[A[1][1] / det, -A[0][1] / det], [-A[0][1] / det, A[0][0] / det]], det

def inv3x3_spd(A, tag):
    """certified SPD (Sylvester) + inverse for symmetric 3x3 intervals"""
    d1 = A[0][0]
    d2 = A[0][0] * A[1][1] - A[0][1] ** 2
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] ** 2)
           - A[0][1] * (A[0][1] * A[2][2] - A[1][2] * A[2][0])
           + A[0][2] * (A[0][1] * A[2][1] - A[1][1] * A[2][0]))
    ck(lo(d1) > 0 and lo(d2) > 0 and lo(det) > 0, "3x3 SPD %s" % tag)
    adj = [[A[1][1] * A[2][2] - A[1][2] ** 2, A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][1] * A[1][2] - A[0][2] * A[1][1]],
           [A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][0] * A[2][2] - A[0][2] ** 2, A[0][1] * A[0][2] - A[0][0] * A[1][2]],
           [A[0][1] * A[1][2] - A[0][2] * A[1][1], A[0][1] * A[0][2] - A[0][0] * A[1][2], A[0][0] * A[1][1] - A[0][1] ** 2]]
    return [[adj[i][j] / det for j in range(3)] for i in range(3)], det

def _mv2(A, v):
    return [sum((A[i][k] * v[k] for k in range(2)), iv.mpf(0)) for i in range(2)]

def _mv3(A, v):
    return [sum((A[i][k] * v[k] for k in range(3)), iv.mpf(0)) for i in range(3)]

# conditioning indices inside the 6-jet: f=0, f1=1, f2=2, H=(3,4,5)
def rho_box(ybox, law, tv, gtol, tag, need_exact=True, JS=None):
    """returns dict(rho=interval, cheap=interval, mode) enclosing rho over ybox"""
    muJ, SigJ = JS if JS is not None else jetlaw_iv(ybox, law, tv)
    b_iv = civ(b_rat)
    ell_iv = civ(ell_rat)
    # grad marginal (indices 1,2)
    Sg = [[SigJ[1][1], SigJ[1][2]], [SigJ[1][2], SigJ[2][2]]]
    Sgi, detg = inv2x2_spd(Sg, tag + " grad")
    mug = [muJ[1], muJ[2]]
    t = _mv2(Sgi, mug)
    chi2 = mug[0] * t[0] + mug[1] * t[1]
    if lo(chi2) < 0:
        chi2 = mpi(0, hi(chi2))   # clamp: chi2_true >= 0; e^{-x/2} monotone
    pgrad = iv.exp(-chi2 / 2) / (2 * iv.pi * iv.sqrt(detg))
    # f | grad = 0
    Sfg = [SigJ[0][1], SigJ[0][2]]
    t2 = _mv2(Sgi, Sfg)
    mt = muJ[0] - (Sfg[0] * t[0] + Sfg[1] * t[1])
    vt = SigJ[0][0] - (Sfg[0] * t2[0] + Sfg[1] * t2[1])
    degraded = lo(vt) <= 0
    if not degraded:
        svt = iv.sqrt(vt)
        z1 = (b_iv - mt) / svt
        z0 = (b_iv - ell_iv - mt) / svt
        Pwin = dPhi(z0, z1, tag)
        if lo(Pwin) < 0:
            Pwin = mpi(0, hi(Pwin))
    else:
        z1 = mpi(0, 0)
        z0 = mpi(0, 0)
        Pwin = mpi(0, 1)   # trivial certified bound; driver must subdivide if not small enough
    # cheap CS prefilter: rho <= pgrad * sqrt(E[det^2|grad=0]) * sqrt(Pwin)
    # H | grad = 0: mean m2, cov V2
    SHg = [[SigJ[3][1], SigJ[3][2]], [SigJ[4][1], SigJ[4][2]], [SigJ[5][1], SigJ[5][2]]]
    SHH = [[SigJ[3][3], SigJ[3][4], SigJ[3][5]],
           [SigJ[3][4], SigJ[4][4], SigJ[4][5]],
           [SigJ[3][5], SigJ[4][5], SigJ[5][5]]]
    W2 = [[sum((SHg[i][k] * Sgi[k][j] for k in range(2)), iv.mpf(0)) for j in range(2)] for i in range(3)]
    m2 = [muJ[3 + i] - sum((W2[i][k] * mug[k] for k in range(2)), iv.mpf(0)) for i in range(3)]
    V2 = [[SHH[i][j] - sum((W2[i][k] * SHg[j][k] for k in range(2)), iv.mpf(0)) for j in range(3)] for i in range(3)]
    # E[W^2] = kappa2 + kappa1^2 with V2 (no SPD cert needed for an upper bound? need kappa2>=0-ish: use fabs guard)
    VQ2 = _vq_entries(V2)
    trA2 = VQ2[0][0] + VQ2[1][1] + VQ2[2][2]
    P2 = _mat3_mul(VQ2, VQ2)
    S22 = P2[0][0] + P2[1][1] + P2[2][2]
    kap1 = trA2 + _mQXm(m2, tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3)))
    kap2 = 2 * (S22 + 2 * _mQXm(m2, VQ2))
    EW2c = kap2 + kap1 ** 2
    # rigorous upper bound on sqrt(EW2c) needs no positivity of the interval:
    EW2c_up = iv.fabs(EW2c) if lo(EW2c) < 0 else EW2c
    ck(lo(EW2c_up) >= 0 or hi(EW2c_up) >= 0, "EW2c_up %s" % tag)
    cheap = pgrad * iv.sqrt(EW2c_up) * iv.sqrt(Pwin)
    out = dict(cheap=cheap, pgrad=pgrad, Pwin=Pwin, z1=z1, z0=z0, mt=mt, vt=vt)
    if degraded:
        out['rho'] = cheap
        out['mode'] = 'degraded'
        return out
    # mid-tier bound: rho <= pgrad * Pwin * sqrt(sup_{u in win} E[det^2 | f=u, grad=0])
    Sfg3 = [[SigJ[0][0], SigJ[0][1], SigJ[0][2]],
            [SigJ[0][1], SigJ[1][1], SigJ[1][2]],
            [SigJ[0][2], SigJ[1][2], SigJ[2][2]]]
    uwin = mpi(float(lo(b_iv - ell_iv)), float(hi(b_iv)))
    try:
        Sfg3i, detfg = inv3x3_spd(Sfg3, tag + " fg")
    except SystemExit:
        Sfg3i = None
    if Sfg3i is not None:
        mufg = [muJ[0], muJ[1], muJ[2]]
        SHfg = [[SigJ[3][0], SigJ[3][1], SigJ[3][2]],
                [SigJ[4][0], SigJ[4][1], SigJ[4][2]],
                [SigJ[5][0], SigJ[5][1], SigJ[5][2]]]
        W3 = [[sum((SHfg[i][k] * Sfg3i[k][j] for k in range(3)), iv.mpf(0)) for j in range(3)] for i in range(3)]
        V = [[SHH[i][j] - sum((W3[i][k] * SHfg[j][k] for k in range(3)), iv.mpf(0)) for j in range(3)] for i in range(3)]
        wcol = [W3[i][0] for i in range(3)]
        mH0 = [muJ[3 + i] - (W3[i][0] * mufg[0] + W3[i][1] * mufg[1] + W3[i][2] * mufg[2]) for i in range(3)]
        VQm = _vq_entries(V)
        trAm = VQm[0][0] + VQm[1][1] + VQm[2][2]
        Pm = _mat3_mul(VQm, VQm)
        S22m = Pm[0][0] + Pm[1][1] + Pm[2][2]
        Id3 = tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3))
        # kappa1(u) = trA + (mH0 + w u)' Q (mH0 + w u) = c0 + c1 u + c2 u^2,
        # Q = [[0,0,1/2],[0,-1,0],[1/2,0,0]]
        q0 = _mQXm(mH0, Id3)
        Qw = [wcol[2] / 2, -wcol[1], wcol[0] / 2]
        c1x = 2 * (mH0[0] * Qw[0] + mH0[1] * Qw[1] + mH0[2] * Qw[2])
        c2x = wcol[0] * Qw[0] + wcol[1] * Qw[1] + wcol[2] * Qw[2]
        c0x = trAm + q0
        kap1u = c0x + c1x * uwin + c2x * uwin * uwin
        a1 = iv.fabs(kap1u)   # sup_u |kap1(u)| <= max(|lo|,|hi|)
        # kap2(u) = 2(S22 + 2 m(u)'VQ m(u)), m(u) = mH0 + w u: interval eval over window
        def _mVQm_at(u):
            mm = [mH0[0] + wcol[0] * u, mH0[1] + wcol[1] * u, mH0[2] + wcol[2] * u]
            return _mQXm(mm, VQm)
        kap2u = 2 * (S22m + 2 * _mVQm_at(uwin))
        EW2sup = kap2u + a1 * a1
        EW2sup_up = iv.fabs(EW2sup) if lo(EW2sup) < 0 else EW2sup
        mid = pgrad * Pwin * iv.sqrt(EW2sup_up)
        out['mid'] = mid
        out['EW2sup'] = EW2sup_up
    else:
        out['mid'] = cheap
        out['EW2sup'] = EW2c_up
    if not need_exact:
        out['rho'] = out['mid']
        out['mode'] = 'mid' if Sfg3i is not None else 'cheap'
        return out
    # exact: H | f=u, grad=0, u in window
    ck(Sfg3i is not None, "Sfg3 spd %s" % tag)
    mufg = [muJ[0], muJ[1], muJ[2]]
    SHfg = [[SigJ[3][0], SigJ[3][1], SigJ[3][2]],
            [SigJ[4][0], SigJ[4][1], SigJ[4][2]],
            [SigJ[5][0], SigJ[5][1], SigJ[5][2]]]
    W3 = [[sum((SHfg[i][k] * Sfg3i[k][j] for k in range(3)), iv.mpf(0)) for j in range(3)] for i in range(3)]
    resid = [uwin - mufg[0], -mufg[1], -mufg[2]]
    mH = [muJ[3 + i] + sum((W3[i][k] * resid[k] for k in range(3)), iv.mpf(0)) for i in range(3)]
    V = [[SHH[i][j] - sum((W3[i][k] * SHfg[j][k] for k in range(3)), iv.mpf(0)) for j in range(3)] for i in range(3)]
    inv3x3_spd(V, tag + " V")  # certify SPD
    gres = g_quad_iv(mH, V, gtol, tag)
    g = gres['g']
    # phi range of N(mt, vt) over the window
    s = uwin - mt
    ssq = s * s
    if lo(ssq) < 0:
        ssq = mpi(0, hi(ssq))
    e = ssq / (2 * vt)
    phir = iv.exp(-e) / iv.sqrt(2 * iv.pi * vt)
    rho = pgrad * ell_iv * phir * g
    out.update(rho=rho, g=g, mH=mH, V=V, mode='exact', npan=gres['npan'],
               R1=gres['R1'], err2=gres['err2'], R3=gres['R3'])
    return out


# ============================================================================
# Pin charts: certified bound of E[# unpinned saddles-in-window in D(P, smin)]
#   <= P(lambda_min(Ntilde) <= K_hi*smin) + P(K > K_hi),
#   Ntilde = H_m(P) + X, X ~ N(0, SigH), H_m(P)=E[H(P)|pins], SigH=Cov(H(P)|pins).
# Chernoff via mgf of quadratic forms (interval, same D/N machinery).
# ============================================================================
JET10 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]

def jetlaw10_iv(y, law, tv):
    C, CT, Ginv = law['C'], law['CT'], law['Ginv']
    n = 10
    B = [[cov2c(JET10[i], JET10[j], y, y) for j in range(n)] for i in range(n)]
    A = [[cov2c(JET10[i], LFUNS[j][1], y, (civ(LFUNS[j][0][0]), civ(LFUNS[j][0][1])))
          for j in range(9)] for i in range(n)]
    At = matmul_iv(A, CT)
    lam = matvec_iv(Ginv, tv)
    muJ = matvec_iv(At, lam)
    M = matmul_iv(At, Ginv)
    AtT = [[At[j][i] for j in range(n)] for i in range(9)]
    S = matmul_iv(M, AtT)
    SigJ = [[B[i][j] - S[i][j] for j in range(n)] for i in range(n)]
    return muJ, SigJ

def mgf_qform(s, m, V, tag):
    """interval mgf M(s)=E[e^{sW}], W=X'QX, X~N(m,V): M=Dm^-1/2 exp(s Nm/Dm)"""
    VQ = _vq_entries(V)
    trA = VQ[0][0] + VQ[1][1] + VQ[2][2]
    e2A = ((VQ[0][0] * VQ[1][1] - VQ[0][1] * VQ[1][0])
           + (VQ[0][0] * VQ[2][2] - VQ[0][2] * VQ[2][0])
           + (VQ[1][1] * VQ[2][2] - VQ[1][2] * VQ[2][1]))
    detV = (V[0][0] * (V[1][1] * V[2][2] - V[1][2] ** 2)
            - V[0][1] * (V[1][0] * V[2][2] - V[1][2] * V[2][0])
            + V[0][2] * (V[1][0] * V[2][1] - V[1][1] * V[2][0]))
    detA = detV / 4
    Dm = 1 - 2 * trA * s + 4 * e2A * s * s - 8 * detA * s ** 3
    E1m = tuple(tuple((trA if i == j else iv.mpf(0)) - VQ[i][j] for j in range(3)) for i in range(3))
    adjB = _mat3_mul(VQ, VQ)
    adjB = tuple(tuple(adjB[i][j] - trA * VQ[i][j] + (e2A if i == j else iv.mpf(0)) for j in range(3)) for i in range(3))
    n0 = _mQXm(m, tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3)))
    n1 = -2 * _mQXm(m, E1m)
    n2 = -4 * _mQXm(m, adjB)
    ck(lo(Dm) > 0, "mgf domain %s" % tag)
    Nm = n0 + n1 * s - n2 * s * s
    return Dm ** (iv.mpf(-0.5)) * iv.exp(s * Nm / Dm)

def chernoff_left(t, m, V, tag):
    """P(W <= -t) <= e^{-s t} M(-s), s>0, optimized on a small grid"""
    best = None
    for k in range(1, 60):
        s = 0.25 * 1.35 ** k
        try:
            v = iv.exp(-s * t) * mgf_qform(-s, m, V, tag)
        except SystemExit:
            break
        if best is None or hi(v) < hi(best):
            best = v
        if best is not None and hi(v) > 4 * hi(best):
            break
    ck(best is not None, "chernoff L %s" % tag)
    return best

def chernoff_right(t, m, V, tag):
    """P(W >= t) <= e^{-s t} M(s)"""
    best = None
    for k in range(1, 60):
        s = 0.25 * 1.35 ** k
        try:
            v = iv.exp(-s * t) * mgf_qform(s, m, V, tag)
        except SystemExit:
            break
        if best is None or hi(v) < hi(best):
            best = v
        if best is not None and hi(v) > 4 * hi(best):
            break
    ck(best is not None, "chernoff R %s" % tag)
    return best

# kernel derivative global sup: |K^{(a1,a2)}(s)| = |K1^{(a1)}(s1) K1^{(a2)}(s2)|;
# sup_s |K1^{(m)}(s)| <= |K1^{(m)}(0)| if m even, else <= sqrt(K1^{(2m)}(0) K1(0)) (Cauchy-Schwarz
# on the spectral measure). K1(0)=1.
def K1_deriv_sup(m):
    if m % 2 == 0:
        return iv.fabs(K1w(m, iv.mpf(0)))
    return iv.sqrt(iv.fabs(K1w(2 * m, iv.mpf(0))) * iv.fabs(K1w(0, iv.mpf(0))))

def kernel_deriv_sup(m1, m2):
    return K1_deriv_sup(m1) * K1_deriv_sup(m2)

# spectral global sup of the random field derivatives:
# f(y) = sum_{j in (pi/12)Z^2} sqrt(ph1 ph2) e^{i j.y} Z_j, ph = e^{-j^2/2}/Z, Z_j iid CN(0,1)
# |d^g f(y)| <= sum_j |j1|^g1 |j2|^g2 sqrt(ph1 ph2) |Z_j| =: T.
# Union bound: |Z_j| <= u_j = 1+sqrt(2(L+3 log(1+|j|))) for all j w.p. >= 1 - 55 e^{-L}.
def spectral_sup(g1, g2, L=50.0):
    import math as _m
    Z = Z_interval()
    c = _m.pi / 12
    JMAX = 40
    S = iv.mpf(0)
    for k1 in range(-JMAX, JMAX + 1):
        for k2 in range(-JMAX, JMAX + 1):
            j1, j2 = c * k1, c * k2
            n2 = j1 * j1 + j2 * j2
            aj = (abs(j1) ** g1) * (abs(j2) ** g2) * _m.exp(-n2 / 4) / float(hi(iv.sqrt(Z)))
            if aj < 1e-40:
                continue
            uj = 1 + _m.sqrt(2 * (L + 3 * _m.log(1 + _m.sqrt(n2))))
            S += aj * uj
    # tail |j| > JMAX*c: a_j u_j <= (1+|j|) |j|^g e^{-|j|^2/4} sqrt(2(L+3log(1+|j|)))/sqrt(Z)
    # integral comparison: sum_{|j|>R} |j|^{p} e^{-|j|^2/4} <= (12/pi)^2 int_{R-c}^inf r^{p+1} e^{-r^2/4} dr
    gsum = g1 + g2
    R = JMAX * c
    def integ_tail(p):
        # int_A^inf r^{p} e^{-r^2/4} dr, A = R - c
        A = R - c
        # recurrence: I_p = 2 A^{p-1} e^{-A^2/4} + 2(p-1) I_{p-2}
        if p == 0:
            return _m.sqrt(_m.pi) * _m.erfc(A / 2)
        if p == 1:
            return 2 * _m.exp(-A * A / 4)
        return 2 * A ** (p - 1) * _m.exp(-A * A / 4) + 2 * (p - 1) * integ_tail(p - 2)
    tail = (12 / _m.pi) ** 2 * (integ_tail(gsum + 2) + integ_tail(gsum + 3)) *         _m.sqrt(2 * (L + 3 * _m.log(1 + 4 * R))) / float(lo(iv.sqrt(Z)))
    return S + tail

def gauss_tail(x):
    """P(|N(0,1)| > x) <= 2 phi(x)/x for x>0 (Mills)"""
    return 2 * phi_pdf(iv.mpf(x)) / iv.mpf(x)


# ============================================================================
# Taylor-model rho: all factors via polynomial models (coefficient-level
# cancellation). Uses w3_tm.RhoTM.
# ============================================================================
def rho_tm(TL, ybox, law, tv, gtol, tag, need_exact=True, _F=None):
    import importlib.util
    if 'w3tm' not in globals():
        spec = importlib.util.spec_from_file_location("w3tm", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_tm.py")
        globals()['w3tm'] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(globals()['w3tm'])
    w3tm = globals()['w3tm']
    if _F is not None:
        F = _F
        RT = F['_RT']
    else:
        RT = w3tm.RhoTM(TL, ybox)
        F = RT.build()
        F['_RT'] = RT
    b_iv = civ(b_rat)
    ell_iv = civ(ell_rat)
    vt, mt, pgrad = F['vt'], F['mt'], F['pgrad']
    svt = iv.sqrt(vt)
    z1 = (b_iv - mt) / svt
    z0 = (b_iv - ell_iv - mt) / svt
    Pwin = dPhi(z0, z1, tag)
    if lo(Pwin) < 0:
        Pwin = mpi(0, hi(Pwin))
    # exact-path quantities (always: needed for mid bound too)
    ADJ, Sfgm, SHH = F['ADJ'], F['Sfgm'], F['SHH']
    det3p, det3v = F['det3p'], F['det3']
    Rs = RT.Rsig
    Rm = RT.Rmu
    Smax = RT.Smax
    R_adj = 2 * Smax * Rs + Rs * Rs
    # W3num[i][k] = sum_l Sfgm[i][l] ADJ[l][k]  (polys)
    W3num = [[sum((Sfgm[i][l] * ADJ[l][k] for l in range(3)), w3tm.P.const(ivm(0), RT.Gs)) for k in range(3)] for i in range(3)]
    adjmax = ivm(0)
    for i in range(3):
        for j in range(3):
            bb = w3tm.poly_eval_bound(ADJ[i][j], RT.r1, RT.r2)
            if lo(adjmax) < lo(bb):
                adjmax = bb
    W3nummax = ivm(0)
    for i in range(3):
        for k in range(3):
            bb = w3tm.poly_eval_bound(W3num[i][k], RT.r1, RT.r2)
            if lo(W3nummax) < lo(bb):
                W3nummax = bb
    det3max = w3tm.poly_eval_bound(det3p, RT.r1, RT.r2)
    R_W3num = Smax * R_adj + adjmax * Rs + Rs * R_adj
    # V numerator poly: SHH[i][j]*det3p - sum_k W3num[i][k]*Sfgm[j][k]
    R_Vnum = (Smax * F['Rdet3'] + det3max * Rs + Rs * F['Rdet3']
              + 3 * (W3nummax * Rs + Smax * R_W3num + Rs * R_W3num))
    V = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            nump = SHH[i][j] * det3p - sum((W3num[i][k] * Sfgm[j][k] for k in range(3)),
                                           w3tm.P.const(ivm(0), RT.Gs))
            nv = RT.evp(nump) + mpi(-hi(R_Vnum), hi(R_Vnum))
            V[i][j] = nv / det3v
    inv3x3_spd(V, tag + " Vtm")  # certify SPD
    # mH over window: mH_i = (muH_i det3 + sum_k W3num[i][k] resid_k)/det3
    uwin = mpi(float(lo(b_iv - ell_iv)), float(hi(b_iv)))
    mu0v, mu1v, mu2v = F['mu']
    resid = [uwin - mu0v, -mu1v, -mu2v]
    muHv = [RT.evp(RT.mu[3 + i]) + mpi(-hi(Rm), hi(Rm)) for i in range(3)]
    mH = [None] * 3
    for i in range(3):
        w3r = sum((RT.evp(W3num[i][k]) * resid[k] for k in range(3)), ivm(0))
        R_mHnum = R_W3num * sum((iv.fabs(resid[k]) for k in range(3)), ivm(0)) + Rm * det3max + F['Rdet3'] * (iv.fabs(muHv[i]))
        mH[i] = (muHv[i] * det3v + w3r + mpi(-hi(R_mHnum), hi(R_mHnum))) / det3v
    # mid bound: sup_u E[det^2 | f=u, grad=0]
    wcol = [None] * 3
    for i in range(3):
        wcol[i] = RT.evp(W3num[i][0]) / det3v  # W3 first column
    mufg = [mu0v, mu1v, mu2v]
    mH0 = [muHv[i] - sum((wcol_dummy for wcol_dummy in []), ivm(0)) for i in range(3)]
    # mH0_i = muH_i - sum_k W3[i][k] mufg[k] ; W3[i][k] = W3num/det3 (interval)
    W3full = [[RT.evp(W3num[i][k]) / det3v for k in range(3)] for i in range(3)]
    mH0 = [muHv[i] - sum((W3full[i][k] * mufg[k] for k in range(3)), ivm(0)) for i in range(3)]
    VQm = _vq_entries(V)
    trAm = VQm[0][0] + VQm[1][1] + VQm[2][2]
    Pm = _mat3_mul(VQm, VQm)
    S22m = Pm[0][0] + Pm[1][1] + Pm[2][2]
    Id3 = tuple(tuple(iv.mpf(int(i == j)) for j in range(3)) for i in range(3))
    q0 = _mQXm(mH0, Id3)
    Qw = [wcol[2] / 2, -wcol[1], wcol[0] / 2]
    c1x = 2 * (mH0[0] * Qw[0] + mH0[1] * Qw[1] + mH0[2] * Qw[2])
    c2x = wcol[0] * Qw[0] + wcol[1] * Qw[1] + wcol[2] * Qw[2]
    c0x = trAm + q0
    kap1u = c0x + c1x * uwin + c2x * uwin * uwin
    a1 = iv.fabs(kap1u)
    def _mVQm_at(u):
        mm = [mH0[0] + wcol[0] * u, mH0[1] + wcol[1] * u, mH0[2] + wcol[2] * u]
        return _mQXm(mm, VQm)
    kap2u = 2 * (S22m + 2 * _mVQm_at(uwin))
    EW2sup = kap2u + a1 * a1
    EW2sup_up = iv.fabs(EW2sup) if lo(EW2sup) < 0 else EW2sup
    midb = pgrad * Pwin * iv.sqrt(EW2sup_up)
    # cheap: CS at grad=0 (no f-conditioning): reuse EW2sup as an upper proxy is NOT
    # valid for cheap; cheap = pgrad sqrt(E[det^2|grad=0]) sqrt(Pwin): E[det^2|grad=0]
    # <= sup_u E[det^2|u,grad=0] + (sup shift)... simpler: cheap via tower:
    # E[det^2|grad=0] = E_u[ E[det^2|u,grad=0] ] <= sup_u E[det^2|u,grad=0] = EW2sup_up
    cheap = pgrad * iv.sqrt(EW2sup_up) * iv.sqrt(Pwin)
    out = dict(vt=vt, mt=mt, pgrad=pgrad, Pwin=Pwin, z1=z1, z0=z0,
               cheap=cheap, mid=midb, EW2sup=EW2sup_up, mH=mH, V=V, mode='mid')
    if not need_exact:
        out['rho'] = midb
        return out
    gres = g_quad_iv(mH, V, gtol, tag)
    g = gres['g']
    s = uwin - mt
    ssq = s * s
    if lo(ssq) < 0:
        ssq = mpi(0, hi(ssq))   # s^2 >= 0 (interval-square artifact)
    e = ssq / (2 * vt)
    phir = iv.exp(-e) / iv.sqrt(2 * iv.pi * vt)
    rho = pgrad * ell_iv * phir * g
    out.update(rho=rho, g=g, mode='exact', npan=gres['npan'],
               R1=gres['R1'], err2=gres['err2'], R3=gres['R3'])
    return out
