#!/usr/bin/env python3
"""
cov_exact.py -- H2 foundations LIBRARY 1: the exact-periodic covariance library
for the 2D upper-theorem campaign (work-order items 176-197).

THE LAW. The exact normalized periodized side-24 2D Bargmann-Fock field:
spectral lattice (pi/12)Z^2, masses w(k) = exp(-|k|^2/2), Z-normalized,

    K(x, y) = (1/Z^2) sum_{k in (pi/12)Z^2} w(k) e^{i k.(x-y)},
    Z = Z_1 = sum_{j in (pi/12)Z} e^{-j^2/2}   (1D normalization).

The kernel factorizes, K(x,y) = K1(x1-y1) K1(x2-y2), with the 1D periodized
kernel K1(s) = (1/Z) sum_{k in (pi/12)Z} w(k) e^{i k s} (K1(0) = 1 exactly by
the normalization; K1 is exactly 24-periodic).

DERIVATIVE PARITY DECOMPOSITION (derived, not copied). For real s,

    K1(s) = (1/Z) sum_k w(k) cos(k s)                 (imaginary parts cancel),

so with E_n(s) = (1/Z) sum_k w(k) k^n cos(k s) and S_n(s) = (1/Z) sum_k w(k)
k^n sin(k s),

    K1^{(n)}(s) = (1/Z) sum_k w(k) k^n cos(k s + n pi/2)
    n even:  cos(ks + n pi/2) = (-1)^{n/2} cos(ks)      -> K1^{(n)} = (-1)^{n/2} E_n(s)
    n odd :  cos(ks + n pi/2) = (-1)^{(n+1)/2} sin(ks)  -> K1^{(n)} = (-1)^{(n+1)/2} S_n(s)

(the odd case: n=1 gives cos(ks+pi/2) = -sin(ks); n=3 gives cos(ks+3pi/2) =
+sin(ks)).  The odd-order derivatives are NONZERO SINE SUMS at s != 0.  Zeroing
them is the RB_R2 rung-table defect of record (REVIEW_LPW/RB_R2_RUNGTABLE_ERRATA.md);
this library never special-cases them away.

COVARIANCE BY MULTI-INDEX.  For multi-indices alpha, beta in N^2 and
separation (dx, dy) = x - y,

    Cov(d^alpha f(x), d^beta f(y)) = (-1)^{|beta|} d^{alpha+beta} K(x-y)
        = (-1)^{|beta|} K1^{(alpha1+beta1)}(dx) * K1^{(alpha2+beta2)}(dy).

SECOND IMPLEMENTATION (no spectral sums).  Poisson summation gives the
wrapped-kernel image sum

    K1(s) = (24/(sqrt(2 pi) Z)) sum_{n in Z} exp(-(s + 24 n)^2 / 2),

and with probabilists' Hermite polynomials He_n (He_0=1, He_1=t,
He_{n+1} = t He_n - n He_{n-1}),  (d/ds)^n e^{-s^2/2} = (-1)^n He_n(s) e^{-s^2/2},

    K1^{(n)}(s) = (24/(sqrt(2 pi) Z)) sum_{n in Z} (-1)^n He_n(s+24n) e^{-(s+24n)^2/2}.

CERTIFIED TAILS (computed in-program, interval arithmetic; see spectral_tail_1d
and image_tail_1d for the proved bounds).  At the default cutoff KMAX = 30 the
spectral truncation tail is < 1e-60 for every derivative order n = 0..8; at the
default image window JMAX = 3 the image tail is < 1e-60 for n = 0..8 on the
whole post-wrap domain |s| <= 12 (and a fortiori on the adversarial set
|s| <= 1.62).

TWO PATHS.  mode='mpf' is the fast path (mpmath mpf at the configured dps).
mode='iv' is the certified path (mpmath iv.mpf interval arithmetic) and CARRIES
THE TRUNCATION TAILS: every interval returned by the iv path encloses the exact
periodized value, sum rounding + certified tail included.  Theorem constants
must be consumed from the iv path (or from the mpf path plus the exposed tail
functions); the mpf path alone is diagnostics grade.

FINITE-TORUS MOMENT DISCIPLINE.  a_{2m} = (1/Z) sum_k w(k) k^{2m} are the exact
finite-torus moments.  They are NOT the planar double factorials (2m-1)!!: e.g.
1 - a_2 = 576 * (sum_n n^2 e^{-288 n^2})/(sum_n e^{-288 n^2}) > 0 (~9.65e-123).
Planar values are exposed ONLY as labeled diagnostics (planar_moment).  Consumers
must call moment_a (exact torus) and must not silently substitute planar values.

FAIL-CLOSED CONTRACT.  Every public entry validates its inputs and raises
CovInputError on violation; no bare asserts anywhere; nothing in this module
depends on wall clock, iteration order of unsorted containers, or environment.
Importing this module sets mp.dps and iv.dps to DEFAULT_DPS (100 >= 60); use
configure(dps=...) to change (dps < 60 is refused).

Self-test: selftest_cov_exact.py (fail-closed; byte-identical under python and
python -O).  Reference pattern consulted for cross-checking only:
SIDE24_gap_fill/scripts/verify_wp_witness_v1.py (kL/kE/kS); this implementation
is independently derived above.
"""
import mpmath
from mpmath import mp, mpf, iv

SIDE = 24                       # torus side
MIN_DPS = 60                    # house floor
DEFAULT_DPS = 100
DEFAULT_KMAX = 30               # spectral cutoff |k| <= KMAX
DEFAULT_JMAX = 3                # image window |j| <= JMAX
MAX_DERIV = 8                   # certified derivative-order ceiling
TAIL_TARGET = mpf('1e-60')      # certified-tail requirement at defaults


class CovInputError(ValueError):
    """Raised on any contract violation (fail-closed input validation)."""


class CertificationError(RuntimeError):
    """Raised when a certified bound cannot be established (fail-closed)."""


# --------------------------------------------------------------------------
# configuration and lattice state
# --------------------------------------------------------------------------
_CFG = {
    'dps': DEFAULT_DPS,
    'kmax': DEFAULT_KMAX,
    'jmax': DEFAULT_JMAX,
    'sine_drop': False,         # MUTATION HOOK (self-test only; see selftest)
}
_LATT = None                    # (h, jmax_lattice, ks, ws, Z1) mpf
_LATT_IV = None                 # interval twin


def configure(dps=None, kmax=None, jmax=None):
    """Reconfigure the engine.  dps < MIN_DPS is refused (fail-closed).  After
    reconfiguration the certified-tail requirement (< 1e-60 for n = 0..MAX_DERIV)
    is re-verified in-program; failure raises CertificationError."""
    global _LATT, _LATT_IV
    if dps is not None:
        if not (isinstance(dps, int) and dps >= MIN_DPS):
            raise CovInputError("dps must be an int >= %d" % MIN_DPS)
        _CFG['dps'] = dps
    if kmax is not None:
        if not (isinstance(kmax, int) and kmax >= 1):
            raise CovInputError("kmax must be a positive int")
        _CFG['kmax'] = kmax
    if jmax is not None:
        if not (isinstance(jmax, int) and jmax >= 0):
            raise CovInputError("jmax must be a nonnegative int")
        _CFG['jmax'] = jmax
    mp.dps = _CFG['dps']
    iv.dps = _CFG['dps']
    _LATT = None
    _LATT_IV = None
    _build_lattice()
    # fail-closed certification of the default truncation tails
    for n in range(MAX_DERIV + 1):
        t = spectral_tail_1d(n)
        if not t < TAIL_TARGET:
            raise CertificationError(
                "spectral tail %.3e >= 1e-60 at n=%d under this configuration" % (float(t), n))
        ti = image_tail_1d(n, SIDE / 2)
        if not ti < TAIL_TARGET:
            raise CertificationError(
                "image tail %.3e >= 1e-60 at n=%d, |s|<=12 under this configuration" % (float(ti), n))
    return dict(_CFG)


def _build_lattice():
    global _LATT, _LATT_IV
    if _LATT is not None:
        return _LATT
    h = mp.pi / 12
    J = int(mp.ceil(mpf(_CFG['kmax']) / h)) + 1   # lattice index bound; |k| > KMAX omitted
    ks = [j * h for j in range(-J, J + 1)]
    ws = [mp.e ** (-k * k / 2) for k in ks]
    Z1 = sum(ws)
    _LATT = (h, J, ks, ws, Z1)
    hiv = iv.pi / 12
    ks_iv = [j * hiv for j in range(-J, J + 1)]
    ws_iv = [iv.exp(-k * k / 2) for k in ks_iv]
    Z1_iv = sum(ws_iv, iv.mpf(0))
    _LATT_IV = (hiv, J, ks_iv, ws_iv, Z1_iv)
    return _LATT


def get_config():
    _build_lattice()
    return dict(_CFG)


def lattice():
    """Return (h, J, ks, ws, Z1): step pi/12, index bound J, lattice points,
    masses, and the exact 1D normalization Z1 (mpf)."""
    return _build_lattice()


# --------------------------------------------------------------------------
# certified spectral truncation tail
# --------------------------------------------------------------------------
def spectral_tail_1d(n):
    """Rigorous upper bound (returned as mpf, computed with iv.mpf) on

        T_n = (1/Z1) sum_{|k| > KMAX, k in (pi/12)Z} e^{-k^2/2} |k|^n .

    Bound: let J0 = J+1 be the first omitted lattice index and h = pi/12.
    For j >= J0 the term ratio obeys

        t_{j+1}/t_j = exp(-(2j+1) h^2/2) (1 + 1/j)^n
                    <= exp(-(2 J0+1) h^2/2) (1 + 1/J0)^n =: rho,

    because exp(-(2j+1)h^2/2) decreases in j and (1+1/j)^n decreases in j.
    The routine VERIFIES rho < 1 in interval arithmetic (fail-closed), whence
    the geometric series gives

        T_n <= 2 t_{J0} / (1 - rho) / Z1_lower,
        t_{J0} = exp(-(J0 h)^2/2) (J0 h)^n .

    (Z1_lower >= 1 makes the division safe.)"""
    if not (isinstance(n, int) and 0 <= n <= MAX_DERIV):
        raise CovInputError("spectral_tail_1d: n must be an int in [0, %d]" % MAX_DERIV)
    _build_lattice()
    hiv, J, _, _, Z1_iv = _LATT_IV
    J0 = J + 1
    k0 = J0 * hiv                       # first omitted |k|
    rho = iv.exp(-(2 * J0 + 1) * hiv * hiv / 2) * (1 + 1 / iv.mpf(J0)) ** n
    if not rho.b < 1:
        raise CertificationError("spectral tail: ratio bound rho >= 1; cutoff not certified")
    t0 = iv.exp(-k0 * k0 / 2) * k0 ** n
    tail = 2 * t0 / (1 - rho) / Z1_iv.a   # Z1_iv.a >= 1 is the safe divisor
    return mpf(tail.b)


# --------------------------------------------------------------------------
# certified image-sum truncation tail
# --------------------------------------------------------------------------
def _he_absum(n):
    """Sum of absolute values of the coefficients of the probabilists' Hermite
    polynomial He_n, so |He_n(t)| <= _he_absum(n) * |t|^n for |t| >= 1.
    Computed from the recurrence on absolute-coefficient polynomials:
    A_0 = 1, A_1 = t, A_{k+1} = t A_k + k A_{k-1}; return A_n(1)."""
    a, b = 1, 1
    if n == 0:
        return 1
    for k in range(1, n):
        a, b = b, b + k * a
    return b


def image_tail_1d(n, smax):
    """Rigorous upper bound (mpf, computed with iv.mpf) on the omitted-image
    contribution to K1^{(n)}(s), uniform over |s| <= smax <= 12:

        T = (24/(sqrt(2 pi) Z1)) sum_{|j| > JMAX} |He_n(s+24j)| e^{-(s+24j)^2/2}.

    With A = 24 (JMAX+1) - smax every omitted image has |t| = |s+24j| >= A.
    |He_n(t)| <= C_n |t|^n for |t| >= 1 with C_n = _he_absum(n).  Consecutive
    images on one side differ by 24, so for t >= A the term ratio obeys

        ((t+24)/t)^n exp(-((t+24)^2 - t^2)/2) <= (1 + 24/A)^n exp(-24 A) =: rho

    (since (t+24)^2 - t^2 = 48 t + 576 >= 48 A).  The routine VERIFIES A >= 1
    and rho < 1 in interval arithmetic (fail-closed), whence

        T <= (24/(sqrt(2 pi) Z1_lower)) * 2 * C_n A^n e^{-A^2/2} / (1 - rho)."""
    if not (isinstance(n, int) and 0 <= n <= MAX_DERIV):
        raise CovInputError("image_tail_1d: n must be an int in [0, %d]" % MAX_DERIV)
    smax = mpf(smax)
    if not (0 <= smax <= SIDE / 2):
        raise CovInputError("image_tail_1d: require 0 <= smax <= 12 (post-wrap domain)")
    _build_lattice()
    JIMG = _CFG['jmax']
    A = iv.mpf(24 * (JIMG + 1)) - iv.mpf(smax)
    if not A.a >= 1:
        raise CertificationError("image tail: A < 1; image window not certified")
    Cn = iv.mpf(_he_absum(n))
    rho = (1 + 24 / A) ** n * iv.exp(-24 * A)
    if not rho.b < 1:
        raise CertificationError("image tail: ratio bound rho >= 1; window not certified")
    first = Cn * A ** n * iv.exp(-A * A / 2)
    Z1_iv = _LATT_IV[4]
    norm = 24 / (iv.sqrt(2 * iv.pi) * Z1_iv.a)
    tail = norm * 2 * first / (1 - rho)
    return mpf(tail.b)


# --------------------------------------------------------------------------
# 1D kernel derivatives: implementation 1 (spectral lattice sums)
# --------------------------------------------------------------------------
def _check_order_s(n, s):
    if not (isinstance(n, int) and 0 <= n <= MAX_DERIV):
        raise CovInputError("derivative order n must be an int in [0, %d]" % MAX_DERIV)
    return mpf(s)


def e_sum(n, s, mode='mpf'):
    """E_n(s) = (1/Z) sum_k w(k) k^n cos(k s)  (raw cosine sum, n >= 0)."""
    s = _check_order_s(n, s)
    _build_lattice()
    if mode == 'mpf':
        _, _, ks, ws, Z1 = _LATT
        return sum(wj * (kj ** n) * mp.cos(kj * s) for kj, wj in zip(ks, ws)) / Z1
    if mode == 'iv':
        _, _, ks, ws, Z1 = _LATT_IV
        sv = iv.mpf(s)
        tot = sum((wj * (kj ** n) * iv.cos(kj * sv) for kj, wj in zip(ks, ws)), iv.mpf(0))
        tau = spectral_tail_1d(n)
        return tot / Z1 + iv.mpf([-tau, tau])
    raise CovInputError("mode must be 'mpf' or 'iv'")


def s_sum(n, s, mode='mpf'):
    """S_n(s) = (1/Z) sum_k w(k) k^n sin(k s)  (raw sine sum, n >= 0)."""
    s = _check_order_s(n, s)
    _build_lattice()
    if mode == 'mpf':
        _, _, ks, ws, Z1 = _LATT
        return sum(wj * (kj ** n) * mp.sin(kj * s) for kj, wj in zip(ks, ws)) / Z1
    if mode == 'iv':
        _, _, ks, ws, Z1 = _LATT_IV
        sv = iv.mpf(s)
        tot = sum((wj * (kj ** n) * iv.sin(kj * sv) for kj, wj in zip(ks, ws)), iv.mpf(0))
        tau = spectral_tail_1d(n)
        return tot / Z1 + iv.mpf([-tau, tau])
    raise CovInputError("mode must be 'mpf' or 'iv'")


def k1_spectral(n, s, mode='mpf'):
    """K1^{(n)}(s) by the spectral lattice sum, with the derived parity
    decomposition: even n -> (-1)^{n/2} E_n(s); odd n -> (-1)^{(n+1)/2} S_n(s).
    The odd (sine) branch is NEVER dropped (RB_R2 errata of record)."""
    s = _check_order_s(n, s)
    if n % 2 == 0:
        if n % 4 == 0:
            return e_sum(n, s, mode)
        return -e_sum(n, s, mode)
    # odd branch: sine sum with sign (-1)^{(n+1)/2}
    if _CFG['sine_drop']:
        # MUTATION HOOK (self-test only): the RB_R2 defect.  Never active in
        # library service; the self-test flips it to prove the guard rails.
        return mpf(0) if mode == 'mpf' else iv.mpf(0)
    if (n + 1) % 4 == 0:
        return s_sum(n, s, mode)
    return -s_sum(n, s, mode)


# --------------------------------------------------------------------------
# 1D kernel derivatives: implementation 2 (wrapped-kernel image sums)
# --------------------------------------------------------------------------
def hermite_he(n, t, mode='mpf'):
    """Probabilists' Hermite polynomial He_n(t) by the recurrence
    He_0 = 1, He_1 = t, He_{k+1} = t He_k - k He_{k-1}."""
    if not (isinstance(n, int) and n >= 0):
        raise CovInputError("hermite_he: n must be a nonnegative int")
    if mode == 'mpf':
        t = mpf(t)
    elif mode == 'iv':
        t = iv.mpf(t)
    else:
        raise CovInputError("mode must be 'mpf' or 'iv'")
    if n == 0:
        return t ** 0
    a, b = t ** 0, t
    if n == 1:
        return b
    for k in range(1, n):
        a, b = b, t * b - k * a
    return b


def _wrap24(s, mode):
    """Reduce s modulo 24 into [-12, 12).  K1 is exactly 24-periodic (every
    lattice frequency k in (pi/12)Z satisfies e^{i k 24} = 1), so this is exact.
    In iv mode the integer shift is verified by interval containment (fail-closed)
    and the subtraction s - 24 n is exact in interval arithmetic."""
    if mode == 'mpf':
        s = mpf(s)
        n = int(mp.floor((s + 12) / 24))
        return s - 24 * n
    sv = iv.mpf(s)
    mid = mpf((sv.a + sv.b) / 2)
    n = int(mp.floor((mid + 12) / 24))
    frac = (sv + 12) / 24
    if not (frac.a >= n and frac.b < n + 1):
        raise CertificationError("wrap24: cannot certify the integer shift in iv mode")
    return sv - 24 * n


def k1_image(n, s, mode='mpf'):
    """K1^{(n)}(s) by the wrapped-kernel image sum (NO spectral sums):
    (24/(sqrt(2 pi) Z)) sum_{|j| <= JMAX} (-1)^n He_n(t) e^{-t^2/2},
    t = s_wrapped + 24 j.  The iv path adds the certified image tail."""
    s = _check_order_s(n, s)
    _build_lattice()
    JIMG = _CFG['jmax']
    if mode == 'mpf':
        sw = _wrap24(s, 'mpf')
        Z1 = _LATT[4]
        tot = mpf(0)
        for j in range(-JIMG, JIMG + 1):
            t = sw + 24 * j
            tot += hermite_he(n, t) * mp.e ** (-t * t / 2)
        if n % 2 == 1:
            tot = -tot          # (d/ds)^n e^{-s^2/2} = (-1)^n He_n(s) e^{-s^2/2}
        return tot * 24 / (mp.sqrt(2 * mp.pi) * Z1)
    if mode == 'iv':
        sw = _wrap24(s, 'iv')
        Z1 = _LATT_IV[4]
        tot = iv.mpf(0)
        for j in range(-JIMG, JIMG + 1):
            t = sw + 24 * j
            tot = tot + hermite_he(n, t, 'iv') * iv.exp(-t * t / 2)
        if n % 2 == 1:
            tot = -tot
        tau = image_tail_1d(n, SIDE / 2)
        return tot * 24 / (iv.sqrt(2 * iv.pi) * Z1) + iv.mpf([-tau, tau])
    raise CovInputError("mode must be 'mpf' or 'iv'")


def k1(n, s, impl='spectral', mode='mpf'):
    """K1^{(n)}(s); impl in {'spectral', 'image'}, mode in {'mpf', 'iv'}."""
    if impl == 'spectral':
        return k1_spectral(n, s, mode)
    if impl == 'image':
        return k1_image(n, s, mode)
    raise CovInputError("impl must be 'spectral' or 'image'")


# --------------------------------------------------------------------------
# 2D covariance by multi-index
# --------------------------------------------------------------------------
def _check_multi(idx, name):
    if not (isinstance(idx, (tuple, list)) and len(idx) == 2
            and all(isinstance(v, int) and v >= 0 for v in idx)):
        raise CovInputError("%s must be a pair of nonnegative ints" % name)
    return (idx[0], idx[1])


def cov(alpha, beta, dx, dy, impl='spectral', mode='mpf'):
    """Cov(d^alpha f(x), d^beta f(y)) at separation (dx, dy) = x - y:
    (-1)^{|beta|} K1^{(alpha1+beta1)}(dx) K1^{(alpha2+beta2)}(dy).
    Cosine AND sine contributions are automatic via the parity decomposition.
    Total order |alpha+beta| must be <= MAX_DERIV (certified-tail ceiling)."""
    alpha = _check_multi(alpha, 'alpha')
    beta = _check_multi(beta, 'beta')
    n1, n2 = alpha[0] + beta[0], alpha[1] + beta[1]
    if n1 + n2 > MAX_DERIV:
        raise CovInputError("total order %d exceeds certified ceiling %d"
                            % (n1 + n2, MAX_DERIV))
    v = k1(n1, dx, impl, mode) * k1(n2, dy, impl, mode)
    if (beta[0] + beta[1]) % 2 == 1:
        v = -v
    return v


# --------------------------------------------------------------------------
# finite-torus moments (NEVER silently planar)
# --------------------------------------------------------------------------
def moment_a(m, impl='spectral', mode='mpf'):
    """Exact finite-torus moment a_{2m} = (1/Z) sum_k w(k) k^{2m}
    = (-1)^m K1^{(2m)}(0).  This is the ONLY moment entry point for theorem
    use; planar double factorials are diagnostics (planar_moment)."""
    if not (isinstance(m, int) and 0 <= 2 * m <= MAX_DERIV):
        raise CovInputError("moment_a: need 0 <= 2m <= %d" % MAX_DERIV)
    v = k1(2 * m, 0, impl, mode)
    return v if m % 2 == 0 else -v


def moment_a_poisson(m, mode='mpf'):
    """a_{2m} by the real-space Poisson identity
        a_{2m} = (-1)^m [sum_n He_{2m}(24 n) e^{-288 n^2}] / [sum_n e^{-288 n^2}]
    (|n| <= 50; the first omitted image has |t| = 1224, so omitted terms are
    below e^{-749088} up to polynomial factors, certified by
    moment_poisson_tail).  Independent of the spectral lattice path."""
    if not (isinstance(m, int) and 0 <= 2 * m <= MAX_DERIV):
        raise CovInputError("moment_a_poisson: need 0 <= 2m <= %d" % MAX_DERIV)
    if mode == 'mpf':
        num = mpf(0); den = mpf(0)
        for j in range(-50, 51):
            t = mpf(24 * j)
            e = mp.e ** (-t * t / 2)
            num += hermite_he(2 * m, t) * e
            den += e
        v = num / den
    elif mode == 'iv':
        num = iv.mpf(0); den = iv.mpf(0)
        for j in range(-50, 51):
            t = iv.mpf(24 * j)
            e = iv.exp(-t * t / 2)
            num = num + hermite_he(2 * m, t, 'iv') * e
            den = den + e
        tau = moment_poisson_tail(m)
        # omitted |n|>50 terms enter BOTH num and den; |num|/den = a_{2m} and
        # C_{2m} >= 1, so 2*tau rigorously covers the ratio perturbation.
        v = num / den + 2 * iv.mpf([-tau, tau])
    else:
        raise CovInputError("mode must be 'mpf' or 'iv'")
    return v if m % 2 == 0 else -v


def moment_poisson_tail(m):
    """Certified bound on the omitted |n| > 50 terms of the Poisson moment
    sums: |He_{2m}(24n)| e^{-(24n)^2/2} <= C_{2m} |24n|^{2m} e^{-(24n)^2/2},
    first omitted |t| = 1224, ratio <= (1+24/1224)^{2m} e^{-24*1224} < 1.
    The numerator tail is normalized by sum_n e^{-288 n^2} >= 1 (safe)."""
    if not (isinstance(m, int) and 0 <= 2 * m <= MAX_DERIV):
        raise CovInputError("moment_poisson_tail: need 0 <= 2m <= %d" % MAX_DERIV)
    A = iv.mpf(1224)
    Cn = iv.mpf(_he_absum(2 * m))
    rho = (1 + 24 / A) ** (2 * m) * iv.exp(-24 * A)
    if not rho.b < 1:
        raise CertificationError("moment Poisson tail: rho >= 1")
    return mpf((2 * Cn * A ** (2 * m) * iv.exp(-A * A / 2) / (1 - rho)).b)


def one_minus_a2_poisson(mode='mpf'):
    """The exact identity  1 - a_2 = 576 (sum_n n^2 e^{-288 n^2})/(sum_n e^{-288 n^2})
    (real-space form; > 0 strictly: the torus is never planar)."""
    if mode == 'mpf':
        num = mpf(0); den = mpf(0)
        for j in range(-50, 51):
            e = mp.e ** (mpf(-288) * j * j)
            num += j * j * e
            den += e
        return 576 * num / den
    if mode == 'iv':
        num = iv.mpf(0); den = iv.mpf(0)
        for j in range(-50, 51):
            e = iv.exp(iv.mpf(-288) * j * j)
            num = num + j * j * e
            den = den + e
        # omitted |n|>50 terms enter BOTH num and den; C_4 >= C_2 = 1 makes the
        # m=2 tail dominate each, and 3x covers the ratio perturbation.
        tau = moment_poisson_tail(2)
        return 576 * num / den + 3 * iv.mpf([-tau, tau])
    raise CovInputError("mode must be 'mpf' or 'iv'")


def planar_moment(m):
    """DIAGNOSTIC ONLY (labeled): the planar moment (2m-1)!! = E[N(0,1)^{2m}].
    Forbidden as a theorem input on the finite torus; the exact finite-torus
    moment is moment_a(m)."""
    if not (isinstance(m, int) and m >= 0):
        raise CovInputError("planar_moment: m must be a nonnegative int")
    v = 1
    for j in range(1, m + 1):
        v *= (2 * j - 1)
    return mpf(v)


# apply default configuration at import (sets mp.dps/iv.dps, builds lattice,
# and certifies the default tails fail-closed)
configure(dps=DEFAULT_DPS, kmax=DEFAULT_KMAX, jmax=DEFAULT_JMAX)
