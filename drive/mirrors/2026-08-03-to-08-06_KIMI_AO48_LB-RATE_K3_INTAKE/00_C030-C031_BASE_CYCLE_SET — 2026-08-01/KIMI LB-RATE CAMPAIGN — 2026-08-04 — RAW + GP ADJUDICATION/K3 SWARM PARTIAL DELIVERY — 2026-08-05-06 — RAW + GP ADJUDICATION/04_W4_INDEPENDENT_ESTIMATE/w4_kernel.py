#!/usr/bin/env python3
"""
W4 INDEPENDENT - wrapped-kernel real-space representation of the periodized
Bargmann-Fock kernel on T^2_24 = R^2/(24 Z)^2.

Representation (independent choice, NOT the spectral-lattice/Gauss-Hermite stack):
    K(x,y) = k1(x1-y1) * k1(x2-y2),
    k1(s)   = theta(s) / theta(0),
    theta(s) = sum_{n in Z} exp(-(s+24 n)^2 / 2).

Derivatives: d^m/ds^m exp(-s^2/2) = (-1)^m He_m(s) exp(-s^2/2),
He_m = probabilists' Hermite polynomial (He_0=1, He_1=s, He_{m+1}=s He_m - m He_{m-1}).

Covariance of derivative jets:
    Cov( d^a f(x), d^b f(y) ) = (-1)^(|b|) k1^{(a1+b1)}(s1) k1^{(a2+b2)}(s2),
    s = x - y,   a=(a1,a2), b=(b1,b2) multi-indices.

Certified tails: for |s| <= S0, the n != 0 wraps contribute
    |theta^{(m)}(s) - (-1)^m He_m(s) e^{-s^2/2}| <= T(m, S0)
with T computed below; we keep n in {-1,0,1} explicitly anyway (costs nothing)
and certify the |n|>=2 remainder rigorously.

Fail-closed: ck(cond, msg) aborts the program on any failed check.
No `assert` statements anywhere (byte-identical output under -O).
Deterministic: no randomness, no wall-clock dependence in values.
"""
import sys
from mpmath import mp, mpf, exp as mexp, sqrt as msqrt, pi as mpi, factorial

DPS = 60
mp.dps = DPS

PERIOD = mpf(24)


def ck(cond, msg):
    """Fail-closed check: abort if cond is not truthy."""
    if not cond:
        sys.stderr.write("CK-FAIL: %s\n" % msg)
        raise SystemExit(2)
    return True


def He(m, x):
    """Probabilists' Hermite polynomial He_m(x), mpmath-exact recursion."""
    if m == 0:
        return mpf(1)
    if m == 1:
        return x
    h0, h1 = mpf(1), x
    for k in range(1, m):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def gauss_der(m, s):
    """d^m/ds^m exp(-s^2/2) at s (single Gaussian, no wrap)."""
    return (-1) ** m * He(m, s) * mexp(-s * s / 2)


# ---------------- wrapped kernel with certified remainder ----------------

WRAP_N = 1  # keep n = -1, 0, +1 explicitly (verification default)


def set_production_wrap(smax):
    """Switch to n=0-only kernel with certified |n|>=1 remainder for |s|<=smax.
    Fail-closed: refuses if the certified bound exceeds 1e-40."""
    global WRAP_N, _SMAX_PROD
    worst = mpf(0)
    for m in range(7):
        tb = tail_bound0(m, smax)
        worst = max(worst, tb)
    ck(worst < mpf(10) ** (-40),
       "production wrap remainder too large: %s" % mp.nstr(worst, 3))
    WRAP_N = 0
    _SMAX_PROD = mpf(smax)
    return worst


def tail_bound0(m, S0):
    """Rigorous bound on |sum_{|n|>=1} gauss_der(m, s+24n)| for |s|<=S0."""
    coeffs = [mpf(1)]
    if m > 0:
        p0 = [mpf(1)]
        p1 = [mpf(0), mpf(1)]
        for k in range(1, m):
            pnew = [mpf(0)] * (len(p1) + 1)
            for i, c in enumerate(p1):
                pnew[i + 1] += c
            for i, c in enumerate(p0):
                pnew[i] -= k * c
            p0, p1 = p1, pnew
        coeffs = p1
    CM = sum(abs(c) for c in coeffs)

    def term(absn):
        t = PERIOD * absn - S0
        return CM * (t ** m) * mexp(-t * t / 2)

    t1 = PERIOD * 1 - S0
    t2 = PERIOD * 2 - S0
    q = (t2 / t1) ** m * mexp(-(t2 * t2 - t1 * t1) / 2)
    ck(q < 1, "geometric ratio < 1 required")
    return 2 * term(1) / (1 - q)


def theta_der(m, s):
    """sum over kept wraps of d^m/ds^m exp(-(s+24n)^2/2)."""
    tot = mpf(0)
    for n in range(-WRAP_N, WRAP_N + 1):
        tot += gauss_der(m, s + PERIOD * n)
    return tot


def tail_bound(m, S0):
    """
    Rigorous bound R(m, S0) on |sum_{|n|>=2} gauss_der(m, s+24n)| for |s|<=S0.
    For |n|>=2 and |s|<=S0: |s+24n| >= 24|n|-S0 >= 48-S0.
    |He_m(t)| e^{-t^2/2}: for |t| >= t0 >= m+1, |He_m(t)| <= |t|^m * c_m with
    c_m = sum of |coeffs|, and t^m e^{-t^2/2} is decreasing for t > sqrt(m).
    We bound each term by CM(m) * |t|^m e^{-t^2/2} evaluated at |t|=24|n|-S0,
    and sum the geometrically decaying series with an explicit ratio bound.
    """
    # coefficient-sum bound for |He_m|
    coeffs = [mpf(0)] * (m + 1)
    # build He_m coefficients
    # He_0=1, He_1=x, He_{k+1} = x He_k - k He_{k-1}
    p0 = [mpf(1)]
    if m == 0:
        p1 = p0
    else:
        p1 = [mpf(0), mpf(1)]
        for k in range(1, m):
            pnew = [mpf(0)] * (len(p1) + 1)
            for i, c in enumerate(p1):
                pnew[i + 1] += c
            for i, c in enumerate(p0):
                pnew[i] -= k * c
            p0, p1 = p1, pnew
    CM = sum(abs(c) for c in p1)

    def term(absn):
        t = PERIOD * absn - S0
        return CM * (t ** m) * mexp(-t * t / 2)

    # ratio between successive terms for |n| and |n|+1 is <= q < 1
    t2 = PERIOD * 2 - S0
    t3 = PERIOD * 3 - S0
    q = (t3 / t2) ** m * mexp(-(t3 * t3 - t2 * t2) / 2)
    ck(q < 1, "geometric ratio < 1 required")
    tot = 2 * term(2) / (1 - q)  # both signs of n
    return tot


THETA0 = theta_der(0, mpf(0))  # = sum_n exp(-(24n)^2/2)


def k1_der(m, s):
    """m-th derivative of k1(s) = theta(s)/theta(0)."""
    return theta_der(m, s) / THETA0


def k1_der_certified(m, s, S0):
    """Return (value, rigorous absolute error bound incl. |n|>=2 remainder)."""
    v = theta_der(m, s) / THETA0
    r = tail_bound(m, S0) / THETA0
    return v, r


_SMAX_PROD = None


def cov_jet(a, b, s1, s2):
    """Cov(d^a f(x), d^b f(y)) with s = x-y. a,b : 2-tuples of ints."""
    if _SMAX_PROD is not None:
        ck(abs(s1) <= _SMAX_PROD and abs(s2) <= _SMAX_PROD,
           "kernel argument outside certified wrap range")
    m1 = a[0] + b[0]
    m2 = a[1] + b[1]
    return (-1) ** (b[0] + b[1]) * k1_der(m1, s1) * k1_der(m2, s2)


# ---------------- self-verification ----------------

def verify():
    ck(abs(THETA0 - 1) < mpf(10) ** (-60),
       "theta(0) = 1 + 2 e^{-288} + ... must equal 1 to 60 digits")
    # closed-form identity: wrapped theta vs spectral-lattice representation.
    # Spectral form: k1_spec(s) = Z^{-1} sum_{j in (pi/12) Z} e^{-j^2/2} e^{i j s},
    # Z = sum_{j in (pi/12)Z} e^{-j^2/2}. Poisson summation gives the wrapped
    # form identically; we verify numerically at several s with the spectral
    # sum truncated at |j| <= J and a certified spectral tail bound.
    J = 80  # frequency cutoff j = (pi/12) k, |k| <= J
    a = mpi / 12
    Zspec = mpf(0)
    k = -J
    while k <= J:
        Zspec += mexp(-(a * k) ** 2 / 2)
        k += 1
    # spectral tail: 2 * sum_{k>J} e^{-(a k)^2/2} <= 2 e^{-(a(J+1))^2/2}/(1-q2)
    tj = (a * (J + 1)) ** 2 / 2
    q2 = mexp(-((a * (J + 2)) ** 2 - (a * (J + 1)) ** 2) / 2)
    spec_tail_Z = 2 * mexp(-tj) / (1 - q2)
    ck(spec_tail_Z < mpf(10) ** (-50), "spectral Z tail certified small")
    maxdiff = mpf(0)
    for num, den in [(3, 7), (11, 13), (5, 3), (17, 5), (23, 11), (1, 1), (7, 2)]:
        s = mpf(num) / den
        spec = mpf(0)
        k = -J
        while k <= J:
            spec += mexp(-(a * k) ** 2 / 2) * mpmath_cos(a * k * s)
            k += 1
        spec /= Zspec
        wrap = k1_der(0, s)
        d = abs(spec - wrap) + spec_tail_Z / Zspec + tail_bound(0, s) / THETA0
        maxdiff = max(maxdiff, abs(spec - wrap))
        ck(d < mpf(10) ** (-45), "spectral vs wrapped agree at s=%s" % str(s))
    # derivative spot-check against complex-step / finite-difference-free route:
    # compare k1_der(m,s) with mpmath diff of theta/theta0
    from mpmath import diff as mdiff
    for m in (1, 2, 3, 4, 5):
        for s in (mpf(3) / 11, mpf(7) / 5):
            v1 = k1_der(m, s)
            v2 = mdiff(lambda t: theta_der(0, t) / THETA0, s, m)
            ck(abs(v1 - v2) < mpf(10) ** (-40), "derivative check m=%d" % m)
    # standard BF moment checks at coincidence
    ck(abs(k1_der(0, 0) - 1) < mpf(10) ** (-50), "k1(0)=1")
    ck(abs(k1_der(2, 0) + 1) < mpf(10) ** (-50), "k1''(0)=-1")
    ck(abs(k1_der(4, 0) - 3) < mpf(10) ** (-50), "k1''''(0)=3")
    ck(abs(k1_der(6, 0) + 15) < mpf(10) ** (-50), "k1^(6)(0)=-15")
    # jet covariance sanity: Var(grad)=1, Var(H11)=3, Var(H12)=1, Cov(f,H11)=-1
    z = mpf(0)
    ck(abs(cov_jet((1, 0), (1, 0), z, z) - 1) < mpf(10) ** (-50), "Var f1")
    ck(abs(cov_jet((2, 0), (2, 0), z, z) - 3) < mpf(10) ** (-50), "Var f11")
    ck(abs(cov_jet((1, 1), (1, 1), z, z) - 1) < mpf(10) ** (-50), "Var f12")
    ck(abs(cov_jet((0, 0), (2, 0), z, z) + 1) < mpf(10) ** (-50), "Cov f,f11")
    return maxdiff


def mpmath_cos(x):
    from mpmath import cos
    return cos(x)


if __name__ == "__main__":
    md = verify()
    print("W4-KERNEL-VERIFY PASS")
    print("dps =", DPS)
    print("theta(0) - 1 =", mp.nstr(THETA0 - 1, 5))
    print("max |spectral - wrapped| at 7 test points =", mp.nstr(md, 5))
    for m in (0, 1, 2, 3, 4, 5, 6):
        print("certified |n|>=2 remainder m=%d, |s|<=13: %s"
              % (m, mp.nstr(tail_bound(m, mpf(13)) / THETA0, 5)))
