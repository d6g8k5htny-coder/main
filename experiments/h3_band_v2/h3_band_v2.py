#!/usr/bin/env python3
"""H3 band certificate, numbered successor v2: a ceiling and a floor for E[G_r], r in (0, 0.05].

Successor of the frozen programs on main, under
incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/UPPER2D/H3_closure/:
h3_band_ceil.py (sha256 b97c5428...) and h3_band_floor.py (sha256 a907eeed...), author Kimi.
Those two files share their law pipeline (CB0-CB6) byte for byte; this program keeps that
pipeline once, stage by stage, and runs either envelope. README.md lists every change and why.

Arithmetic. mpmath iv (outward-rounded interval arithmetic) at dps 100, as in the frozen
programs. Every load-bearing endpoint comes from an interval. Round-to-nearest mp is used only
where the result is re-checked in iv (the midpoint preconditioner, the search for the majorant
growth R) or is not load-bearing (the CB7 direct point law, which is a cross-check). The exact
pole-structure lemma (CB2s) uses fractions.Fraction only. Certified decimals are rounded outward
exactly. The Monte Carlo diagnostic (--mc) is NON-CERTIFYING and writes to stderr only.

Usage:
  h3_band_v2.py structure                         the exact pole-structure lemma alone
  h3_band_v2.py ceil  [--transcript F] [--record R | --verify R] [--mc]
  h3_band_v2.py floor [--transcript F] [--record R | --verify R] [--mc]
  h3_band_v2.py verify-record --transcript F --results R
Negative controls: --mutation NAME, one of MUTATIONS below. A checker prints one summary line
and exits nonzero on failure.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from pathlib import Path

try:
    import mpmath
    from mpmath import mp, iv, mpf
except ImportError:  # the certificate needs mpmath; the structure lemma does not
    mpmath = None

HERE = Path(__file__).resolve().parent
COV_EXACT_SHA256 = 'f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783'
PROGRAM = 'experiments/h3_band_v2/h3_band_v2.py'

MUTATIONS = {
    'structure_pin_pole': 'drop the -1/2 weight of S(1,0) in pin V5; CB2s must find a pole',
    'structure_window': 'conditioning vector with zV[3] = 1; CB2s must find a mu pole',
    'planar_moments': 'planar moments in place of the exact torus moments; CB1 must fail',
    'wrong_window': 'zV[5] = -1/12; the CB3 limit identities must fail',
    'signed_maxima': 'B1 regression (signed upper endpoints as majorants); the CB4 dominance guard must fail',
    'wide_first_cell': 'R-majorant first cell (0, 0.0025] as in the frozen programs; the CB4 convergence guard must fail',
    'one_tail': 'B3 regression (one Gaussian tail for a two-sided event); the CB5 tail self-test must fail',
    'no_preconditioner': 'identity in place of the midpoint inverse; the CB6 contraction guard must fail',
    'cap_raised': 'p and pbar doubled; the ceiling consumption cap must fail',
    'drop_fail_piece': 'ceiling without its type-fail pieces; not detectable inside the run, --verify must fail',
    'old_remainder': 'B2 regression (the frozen dGinv law correction); --verify must fail',
}
MUT = ''
OUT = []


def emit(s):
    OUT.append(s)
    print(s)


def note(s):
    # stderr only: timings and diagnostics that are not part of the transcript
    print(s, file=sys.stderr)


class CheckFailed(Exception):
    pass


def ck(cond, msg):
    if not cond:
        emit('CK_FAIL: ' + msg)
        raise CheckFailed(msg)


# ------------------------------------------------------------------ exact helpers
def mpf_to_fraction(x):
    """Exact value of a finite mpf as a Fraction (mpf is a binary float: (-1)^sign man 2^exp)."""
    x = mpf(x)
    if not mpmath.isfinite(x):
        raise ValueError('nonfinite mpf')
    sign, man, exp, _ = x._mpf_
    man = -int(man) if sign else int(man)
    return Fraction(man * 2 ** exp) if exp >= 0 else Fraction(man, 2 ** (-exp))


def decimal_outward(x, direction, sig=15):
    """Round x (mpf or Fraction) to sig significant digits, up ('up') or down ('down'), exactly."""
    q = x if isinstance(x, Fraction) else mpf_to_fraction(x)
    if q == 0:
        return '0'
    neg = q < 0
    a = -q if neg else q
    e10 = len(str(a.numerator // a.denominator)) - 1 if a >= 1 else -len(str(a.denominator // a.numerator))
    while Fraction(10) ** e10 > a:
        e10 -= 1
    while Fraction(10) ** (e10 + 1) <= a:
        e10 += 1
    scale = Fraction(10) ** (sig - 1 - e10)
    v = a * scale
    up = (direction == 'up') != neg
    n = -((-v.numerator) // v.denominator) if up else v.numerator // v.denominator
    digits = str(n)
    point = len(digits) - (sig - 1 - e10)
    if point <= 0:
        s = '0.' + '0' * (-point) + digits
    elif point >= len(digits):
        s = digits + '0' * (point - len(digits))
    else:
        s = digits[:point] + '.' + digits[point:]
    return ('-' if neg else '') + s


def frac_label(q):
    return str(q.numerator) if q.denominator == 1 else '%d/%d' % (q.numerator, q.denominator)


# ------------------------------------------------- CB2s: exact Laurent pole structure
# The functionals of record, as (station, multi-index, exact weight, power of z). M sits at the
# origin and S at z e_x (DISP below: S - M = +z e_x). The interval series of CB2 are generated
# from these tables, so the lemma is about exactly the functionals the certificate uses.
F = Fraction
PINS_EXACT = [
    [('M', (0, 0), F(1), 0)],
    [('M', (1, 0), F(1), 0)],
    [('M', (0, 1), F(1), 0)],
    [('S', (1, 0), F(1), -1), ('M', (1, 0), F(-1), -1)],
    [('S', (0, 1), F(1), -1), ('M', (0, 1), F(-1), -1)],
    [('S', (0, 0), F(1), -3), ('M', (0, 0), F(-1), -3), ('S', (1, 0), F(-1, 2), -2), ('M', (1, 0), F(-1, 2), -2)],
]
SOFT_EXACT = [
    [('M', (0, 2), F(1), 0)],
    [('S', (0, 2), F(1), 0)],
    [('M', (2, 0), F(1), -1)],
    [('S', (2, 0), F(1), -1)],
    [('M', (1, 1), F(1), -1)],
    [('S', (1, 1), F(1), -1)],
]
ZV_EXACT = [F(6, 5), F(0), F(0), F(0), F(0), F(-1, 6)]


def laurent_expand(func, top):
    """Exact Laurent expansion in z of a functional, through z^top: {power: {beta: weight}}.
    S(alpha) = d^alpha f(z e_x) = sum_q z^q / q! d^{alpha + (q, 0)} f(0)."""
    out = defaultdict(lambda: defaultdict(Fraction))
    for station, (ax, ay), w, k in func:
        if station == 'M':
            out[k][(ax, ay)] += w
        else:
            for q in range(0, top - k + 1):
                out[k + q][(ax + q, ay)] += w / factorial(q)
    return {p: {b: c for b, c in row.items() if c != 0} for p, row in out.items()}


def solve_in_span(rows, target):
    """Exact coefficients x with sum_k x_k rows[k] = target (dicts beta -> Fraction), or None."""
    keys = sorted(set().union(*[set(r) for r in rows], set(target)))
    n = len(rows)
    A = [[rows[k].get(b, F(0)) for k in range(n)] + [target.get(b, F(0))] for b in keys]
    piv_cols, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        A[r] = [v / A[r][c] for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv_cols.append(c)
        r += 1
    if any(all(v == 0 for v in A[i][:n]) and A[i][n] != 0 for i in range(len(A))):
        return None
    x = [F(0)] * n
    for i, c in enumerate(piv_cols):
        x[c] = A[i][n]
    return x


def structure_lemma(pins, soft, zv):
    """CB2s. Proves, in exact rational arithmetic, the pole structure the certificate relies on:
      (a) every pin functional V_k has no negative powers of z (so the pin Gram G(z) is analytic);
      (b) every soft functional Y_a has no power below z^-1;
      (c) the z^-1 part of Y equals E V^(0) for an exact matrix E, where V^(0) is the z^0 part of V;
      (d) E zV = 0.
    Then Ytilde = Y - E V / z is regular, E[Y | V = zV] = E[Ytilde | V = zV] + E zV / z has no
    pole, and Cov(Y | V) = Cov(Ytilde | V) is analytic wherever G(z) is invertible. So the
    z^-1 coefficient of mu and the z^-2, z^-1 coefficients of Sigma vanish exactly."""
    pe = [laurent_expand(f, 1) for f in pins]
    se = [laurent_expand(f, 1) for f in soft]
    for k, e in enumerate(pe):
        for p, row in e.items():
            ck(not (p < 0 and row), 'CB2s pin V%d has a z^%d term' % (k, p))
    for a, e in enumerate(se):
        for p, row in e.items():
            ck(not (p < -1 and row), 'CB2s soft Y%d has a z^%d term' % (a, p))
    v0 = [e.get(0, {}) for e in pe]
    E = []
    for a, e in enumerate(se):
        x = solve_in_span(v0, e.get(-1, {}))
        ck(x is not None, 'CB2s soft Y%d pole part lies in the span of V^(0)' % a)
        E.append(x)
    for a in range(len(soft)):
        ck(sum(E[a][k] * zv[k] for k in range(len(zv))) == 0, 'CB2s E zV = 0, row %d' % a)
    return E


# ------------------------------------------------------------------ interval helpers
def ub(x):
    """Exact upper bound on |x| for an iv interval or an mpf."""
    if hasattr(x, 'b'):
        return max(abs(mpf(x.a)), abs(mpf(x.b)))
    return abs(mpf(x))


def up(x):
    """Upper endpoint of an iv value, as an exact mpf."""
    return mpf(x.b)


def iv_frac(q):
    """Enclosure of an exact rational."""
    return iv.mpf(q.numerator) / q.denominator


def run_certificate(mode, h2_dir, mc=False):
    """Build the law pipeline (CB0-CB6) and evaluate the ceiling or the floor on every cell."""
    t_start = time.time()
    emit('H3 BAND V2 %s (successor of the frozen h3_band_%s.py)' % (mode.upper(), 'ceil' if mode == 'ceil' else 'floor'))
    emit('mode dps=100 MUTATION=' + (MUT if MUT else 'none'))

    # ----------------------------------------------------------- CB2s: exact structure
    pins_exact = [list(f) for f in PINS_EXACT]
    zv_exact = list(ZV_EXACT)
    if MUT == 'structure_pin_pole':
        pins_exact[5] = [pins_exact[5][0], pins_exact[5][1], ('S', (1, 0), F(-1), -2), pins_exact[5][3]]
    if MUT == 'structure_window':
        zv_exact[3] = F(1)
    E_sel = structure_lemma(pins_exact, SOFT_EXACT, zv_exact)
    emit('CB2s PASS: exact Laurent structure: pins regular, soft poles z^-1 only, Y^(-1) = E V^(0), E zV = 0')

    mp.dps = 100
    iv.dps = 100
    # ------------------------------------------------------------------ CB0: hashes
    h2 = Path(h2_dir)
    digest = hashlib.sha256((h2 / 'cov_exact.py').read_bytes()).hexdigest()
    manifest = {}
    for ln in (h2 / 'MANIFEST.sha256').read_text().splitlines():
        hh, fn = ln.split()
        manifest[fn] = hh
    ck(digest == COV_EXACT_SHA256 and manifest.get('cov_exact.py') == COV_EXACT_SHA256, 'CB0 cov_exact hash')
    spec = importlib.util.spec_from_file_location('cov_exact', h2 / 'cov_exact.py')
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)   # sets mp.dps = iv.dps = 100 and certifies its own tails
    ck(mp.dps == 100 and iv.dps == 100, 'CB0 precision dps=100')
    emit('CB0 PASS: H2 library cov_exact.py sha256 %s... matches the pinned digest' % COV_EXACT_SHA256[:16])

    REQ = mpf('1.615489267643502474048123284509081575382')   # c_Z of record
    RMAX = F(1, 20)

    # ------------------------------------------------------ CB1: exact-torus moments
    # a_{2m} = (-1)^m sum_n He_{2m}(24 n) e^{-288 n^2} / sum_n e^{-288 n^2}   (Poisson)
    def He_iv(n, x):
        if n == 0:
            return iv.mpf(1)
        a, b = iv.mpf(1), iv.mpf(x)
        for k in range(2, n + 1):
            a, b = b, x * b - (k - 1) * a
        return b

    def moment_a2m(m):
        # |n| <= 3 exactly, |n| >= 4 by a certified tail:
        # |He_k(x)| <= (|x| + sqrt k)^k, and for n >= 4 consecutive terms (24n + c)^{2m} e^{-288 n^2}
        # shrink by at least (5/4)^{2m} e^{-2592} < 1/2, so each tail is at most twice its first
        # term, on both sides of 0.
        num = iv.mpf(0)
        den = iv.mpf(0)
        for n in range(-3, 4):
            w = iv.exp(-iv.mpf(288) * n * n)
            num += He_iv(2 * m, iv.mpf(24 * n)) * w
            den += w
        ck(((iv.mpf(5) / 4) ** (2 * m) * iv.exp(-iv.mpf(2592))).b < mpf(1) / 2, 'CB1 moment tail ratio m=%d' % m)
        first = iv.exp(-iv.mpf(288 * 16))
        tn = 4 * (96 + iv.sqrt(iv.mpf(2 * m))) ** (2 * m) * first
        td = 4 * first
        num = num + iv.mpf((-up(tn), up(tn)))
        den = den + iv.mpf((0, up(td)))
        return ((-1) ** m) * num / den

    MMAX = 40
    A2 = [moment_a2m(m) for m in range(MMAX + 1)]
    DEV2 = ce.one_minus_a2_poisson(mode='iv')
    num4 = iv.mpf(0)
    den4 = iv.mpf(0)
    for n in range(-3, 4):
        w = iv.exp(-iv.mpf(288) * n * n)
        t = iv.mpf(24 * n)
        num4 += (He_iv(4, t) - 3) * w   # He_4(t) - 3 = t^2 (t^2 - 6) >= 0 at t in 24Z
        den4 += w
    first = iv.exp(-iv.mpf(288 * 16))
    t4 = 4 * ((96 + iv.mpf(2)) ** 4 + 3) * first
    DEV4 = (num4 + iv.mpf((0, up(t4)))) / (den4 + iv.mpf((0, up(4 * first))))
    if MUT == 'planar_moments':
        A2 = [iv.mpf(factorial(2 * m)) / iv.mpf(factorial(m) * 2 ** m) for m in range(MMAX + 1)]
        DEV2 = iv.mpf(0)
        DEV4 = iv.mpf(0)
        emit('CB1 MUTATION planar_moments applied (must be caught)')
    ck(DEV2.a > 0 and DEV2.b < mpf('1e-100'), 'CB1a exact-torus 0 < 1-a2 < 1e-100')
    ck(DEV4.a > 0 and DEV4.b < mpf('1e-100'), 'CB1b exact-torus 0 < a4-3 < 1e-100')
    ck(all(ub(A2[m] - ce.moment_a_poisson(m, mode='iv')) < mpf('1e-90') for m in range(5)),
       'CB1c Poisson moments agree with cov_exact for 2m <= 8')
    emit('CB1 PASS: moments a_{2m}, m<=%d, exact finite torus with certified image tails (1-a2 = %s)' % (MMAX, mp.nstr(up(DEV2), 3)))

    # ------------------------------------------------------------ CB5 (early): Phi and tails
    SQRT2PI = iv.sqrt(2 * iv.pi)

    def iv_phi(x):
        return iv.exp(-(x * x) / 2) / SQRT2PI

    def iv_Phi(x):
        # standard normal CDF on an interval x. |x| <= 6: power series with a certified geometric
        # tail. |x| > 6: Gordon-Mills bounds (both monotone). Intervals that straddle +-6 are split
        # by monotonicity. Endpoints stay mpf (O2: the frozen float() conversions are removed).
        if (x.a < 6 < x.b) or (x.a < -6 < x.b):
            lo = iv_Phi(iv.mpf(x.a))
            hi = iv_Phi(iv.mpf(x.b))
            return iv.mpf((mpf(lo.a), mpf(hi.b)))
        if x.a >= 6:
            hi = iv_phi(iv.mpf(x.a)) / iv.mpf(x.a)
            lo = iv_phi(iv.mpf(x.b)) * (1 / iv.mpf(x.b) - 1 / iv.mpf(x.b) ** 3)
            return iv.mpf(1) - iv.mpf((mpf(lo.a), mpf(hi.b)))
        if x.b <= -6:
            hi = iv_phi(iv.mpf(x.b)) / iv.mpf(-x.b)
            lo = iv_phi(iv.mpf(x.a)) * (1 / iv.mpf(-x.a) - 1 / iv.mpf(-x.a) ** 3)
            return iv.mpf((mpf(lo.a), mpf(hi.b)))
        ck(x.a >= -6 and x.b <= 6, 'CB5 Phi series domain')
        # term ratio -x^2 (2k+1) / (2 (k+1) (2k+3)): modulus < 0.9 for k >= 21 when |x| <= 6
        tot = iv.mpf(0)
        K = 0
        term = x
        while K < 120:
            tot += term
            K += 1
            term = -term * x * x * (2 * K - 1) / (2 * K * (2 * K + 1))
            if K >= 22 and abs(term.b) < 1e-90 and abs(term.a) < 1e-90:
                break
        tail = up(iv.mpf(ub(term)) / (1 - iv.mpf('0.9')))
        return iv.mpf(1) / 2 + (tot + iv.mpf((-tail, tail))) / SQRT2PI

    def gauss_tail_upper(t0_lo, sides):
        """Upper bound on P(Z > t0) (sides=1) or P(|Z| > t0) (sides=2), Z ~ N(0,1), t0 >= t0_lo."""
        p = up(iv_Phi(iv.mpf(-t0_lo)))
        if MUT == 'one_tail':
            return p
        return p * sides

    for t in ('2.5', '4', '6.5', '9'):
        tt = iv.mpf(t)
        two = 1 - iv_Phi(tt) + iv_Phi(-tt)
        one = 1 - iv_Phi(tt)
        ck(gauss_tail_upper(mpf(t), 2) >= mpf(two.a), 'CB5 two-sided tail self-test at %s' % t)
        ck(gauss_tail_upper(mpf(t), 1) >= mpf(one.a), 'CB5 one-sided tail self-test at %s' % t)
    emit('CB5 PASS: certified Phi; Gaussian tail helpers dominate independent enclosures')

    # ------------------------------------------------------------- CB2: series machinery
    J = 60          # Taylor order kept for mu, Sigma
    NK = J + 2      # Gram blocks kept to order J+2; Cauchy tails start at J+3
    LMIN = -8
    RHO = mpf('0.11')   # majorant circle radius (a choice; every bound below is certified on it)

    def ser_zero():
        return defaultdict(lambda: iv.mpf(0))

    def ser_monom(c, k):
        s = ser_zero()
        s[k] = c
        return s

    def ser_mul(A, B):
        C = ser_zero()
        for ja, ca in A.items():
            if ca == 0:
                continue
            for jb, cb in B.items():
                if cb == 0:
                    continue
                C[ja + jb] += ca * cb
        return C

    def ser_add(A, B):
        C = ser_zero()
        for j, c in A.items():
            C[j] += c
        for j, c in B.items():
            C[j] += c
        return C

    def ser_scale(A, c):
        C = ser_zero()
        for j, v in A.items():
            C[j] = v * c
        return C

    def ser_trim(A):
        return defaultdict(lambda: iv.mpf(0),
                           {j: c for j, c in A.items() if LMIN <= j <= NK and not (c.a == 0 and c.b == 0)})

    # kernel series K1^{(n)}(+-z): coefficient j is (-1)^{(n+j)/2} a_{n+j} / j! (n+j even).
    # A Gram entry at order k uses kernel orders up to k + 6 - n (the largest pair weight, pole
    # shift plus x-order, is 6), so order NK needs j <= NK + 6 - n: range(0, J + 9 - n).
    def kernel_ser(n, parity):
        s = ser_zero()
        for j in range(0, J + 9 - n):
            m = n + j
            if m % 2 == 1:
                continue
            ck(m // 2 <= MMAX, 'CB2 kernel order within the moment table')
            val = ((-1) ** (m // 2)) * A2[m // 2] / factorial(j)
            if parity < 0 and j % 2 == 1:
                val = -val
            s[j] = val
        return s

    KSER = {}

    def get_kser(n, parity):
        if (n, parity) not in KSER:
            KSER[(n, parity)] = kernel_ser(n, parity)
        return KSER[(n, parity)]

    def cov_ser(alpha, beta, d):
        # cov(alpha, beta, d z, 0) = (-1)^{|beta|} K1^{(n1)}(d z) K1^{(n2)}(0)
        n1 = alpha[0] + beta[0]
        n2 = alpha[1] + beta[1]
        sgn = (-1) ** (beta[0] + beta[1])
        if d == 0:
            v1 = ser_zero()
            if n1 % 2 == 0:
                v1[0] = ((-1) ** (n1 // 2)) * A2[n1 // 2]
        else:
            v1 = get_kser(n1, d)
        k0 = iv.mpf(0)
        if n2 % 2 == 0:
            k0 = ((-1) ** (n2 // 2)) * A2[n2 // 2]
        return ser_scale(v1, sgn * k0)

    def to_iv_func(func):
        return [(st, al, ser_monom(iv_frac(w), k)) for st, al, w, k in func]

    PinsV = [to_iv_func(f) for f in PINS_EXACT]
    SoftY = [to_iv_func(f) for f in SOFT_EXACT]
    DISP = {('M', 'M'): 0, ('S', 'S'): 0, ('S', 'M'): 1, ('M', 'S'): -1}

    def gram_entry_ser(FA, FB):
        tot = ser_zero()
        for (xa, aa, ca) in FA:
            for (xb, ab, cb) in FB:
                tot = ser_add(tot, ser_mul(ser_mul(ca, cb), cov_ser(aa, ab, DISP[(xa, xb)])))
        return ser_trim(tot)

    t1 = time.time()
    GP_ser = [[gram_entry_ser(PinsV[i], PinsV[j]) for j in range(6)] for i in range(6)]
    GY_ser = [[gram_entry_ser(SoftY[i], SoftY[j]) for j in range(6)] for i in range(6)]
    GYV_ser = [[gram_entry_ser(SoftY[i], PinsV[j]) for j in range(6)] for i in range(6)]
    note('CB2 series built in %.1fs' % (time.time() - t1))
    bad = 0
    for i in range(6):
        for j in range(6):
            for k, c in GP_ser[i][j].items():
                if k < 0 and ub(c) > 1e-60:
                    bad += 1
            for k, c in GYV_ser[i][j].items():
                if k <= -2 and ub(c) > 1e-60:
                    bad += 1
            for k, c in GY_ser[i][j].items():
                if k <= -3 and ub(c) > 1e-60:
                    bad += 1
    ck(bad == 0, 'CB2 numeric Laurent pole pattern agrees with CB2s')
    emit('CB2 PASS: Gram series to order %d; numeric pole pattern agrees with the exact lemma CB2s' % NK)

    # ---------------------------------------------------- CB3: series division -> mu, Sigma
    def mat_coeff(Gser, j):
        return [[Gser[a][b].get(j, iv.mpf(0)) for b in range(6)] for a in range(6)]

    def iv_solve6(GG, rhs):
        # interval Gauss-Jordan; every step is an interval extension, so the result encloses
        # the exact solution (a division by an interval containing 0 would surface downstream)
        A = [GG[i][:] + [rhs[i]] for i in range(6)]
        for c in range(6):
            p = max(range(c, 6), key=lambda rr: abs(A[rr][c].a + A[rr][c].b))
            A[c], A[p] = A[p], A[c]
            for rr in range(6):
                if rr != c:
                    f = A[rr][c] / A[c][c]
                    A[rr] = [A[rr][j] - f * A[c][j] for j in range(7)]
        return [A[i][6] / A[i][i] for i in range(6)]

    def matvec(G, v):
        return [sum((G[i][k] * v[k] for k in range(6)), iv.mpf(0)) for i in range(6)]

    def matmul(A, B):
        return [[sum((A[i][k] * B[k][j] for k in range(6)), iv.mpf(0)) for j in range(6)] for i in range(6)]

    def matadd(A, B):
        return [[A[i][j] + B[i][j] for j in range(6)] for i in range(6)]

    def matneg(A):
        return [[-A[i][j] for j in range(6)] for i in range(6)]

    def transpose(A):
        return [[A[j][i] for j in range(6)] for i in range(6)]

    def matzero():
        return [[iv.mpf(0)] * 6 for _ in range(6)]

    def eye():
        return [[iv.mpf(1) if i == j else iv.mpf(0) for j in range(6)] for i in range(6)]

    G0 = mat_coeff(GP_ser, 0)
    zV = [iv_frac(q) for q in ZV_EXACT]
    if MUT == 'wrong_window':
        zV[5] = iv.mpf(-1) / 12
        emit('CB3 MUTATION wrong_window applied (must be caught)')
    cols = [iv_solve6(G0, [iv.mpf(1) if l == k else iv.mpf(0) for l in range(6)]) for k in range(6)]
    Ginv0 = [[cols[k][i] for k in range(6)] for i in range(6)]
    res = max(ub(sum((G0[i][k] * Ginv0[k][j] for k in range(6)), iv.mpf(0)) - (1 if i == j else 0))
              for i in range(6) for j in range(6))
    ck(res < 1e-70, 'CB3 G0 inverse residual')
    emit('CB3 G0 inverse enclosure, residual %s' % mp.nstr(res, 3))

    Gj = [mat_coeff(GP_ser, j) for j in range(NK + 1)]
    GYj = [mat_coeff(GY_ser, j) for j in range(NK + 1)]
    Aj = [mat_coeff(GYV_ser, j) for j in range(NK + 1)]     # A = GYV, the soft-pin cross Gram
    A_m1 = mat_coeff(GYV_ser, -1)
    GY_m2 = mat_coeff(GY_ser, -2)
    GY_m1 = mat_coeff(GY_ser, -1)

    W = [None] * (NK + 1)     # Taylor coefficients of w = G^-1 zV
    H = [None] * (NK + 1)     # Taylor coefficients of G^-1
    W[0] = matvec(Ginv0, zV)
    H[0] = Ginv0
    for j in range(1, NK + 1):
        acc = matzero()
        for i in range(1, j + 1):
            acc = matadd(acc, matmul(Gj[i], H[j - i]))
        H[j] = matneg(matmul(Ginv0, acc))
        W[j] = matvec(Ginv0, [-sum((Gj[i][a][l] * W[j - i][l] for i in range(1, j + 1) for l in range(6)), iv.mpf(0))
                              for a in range(6)])

    def Ac(i):
        return A_m1 if i == -1 else Aj[i]

    t1 = time.time()
    # mu = A w: X[k] = sum_{i+l=k} A_i W_l, i in [-1, NK], l in [0, NK]; mu_k = X[k] for k <= J
    X = {}
    for k in range(-1, 2 * NK + 1):
        acc = [iv.mpf(0)] * 6
        for i in range(max(-1, k - NK), min(k, NK) + 1):
            v = matvec(Ac(i), W[k - i])
            acc = [acc[a] + v[a] for a in range(6)]
        X[k] = acc
    mu_ser = [X[k] for k in range(J + 1)]
    # Sigma = GY - A H A^T: AH[k] = sum_{a+b=k} A_a H_b; S[k] = sum_{m+c=k} AH_m A_c^T
    AH = {}
    for k in range(-1, 2 * NK + 1):
        acc = matzero()
        for a in range(max(-1, k - NK), min(k, NK) + 1):
            acc = matadd(acc, matmul(Ac(a), H[k - a]))
        AH[k] = acc
    AcT = {i: transpose(Ac(i)) for i in range(-1, NK + 1)}
    D = {}
    for k in range(-2, 3 * NK + 1):
        acc = matzero()
        for c in range(max(-1, k - 2 * NK), min(k + 1, NK) + 1):
            acc = matadd(acc, matmul(AH[k - c], AcT[c]))
        gy = GY_m2 if k == -2 else GY_m1 if k == -1 else GYj[k] if k <= NK else None
        D[k] = [[(gy[x][y] if gy is not None else 0) - acc[x][y] for y in range(6)] for x in range(6)]
    Sig_ser = [D[k] for k in range(J + 1)]
    # residual polynomials of the truncated inverse: zV - G_P W_P and I - G_P H_P (orders 0..2NK)
    RW = {}
    RH = {}
    for k in range(0, 2 * NK + 1):
        accv = [iv.mpf(0)] * 6
        accm = matzero()
        for i in range(max(0, k - NK), min(k, NK) + 1):
            v = matvec(Gj[i], W[k - i])
            accv = [accv[a] + v[a] for a in range(6)]
            accm = matadd(accm, matmul(Gj[i], H[k - i]))
        RW[k] = [(zV[a] if k == 0 else 0) - accv[a] for a in range(6)]
        RH[k] = [[(1 if (k == 0 and x == y) else 0) - accm[x][y] for y in range(6)] for x in range(6)]
    note('CB3 product and residual polynomials in %.1fs' % (time.time() - t1))
    emit('CB3 series division done: mu, Sigma to order %d; product polynomials to order %d' % (J, 3 * NK))

    LR = [[1, 0, 0, 0, 0, 0], [-1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    mu2_ser = [[sum((iv.mpf(LR[i][k]) * mu_ser[j][k] for k in range(6)), iv.mpf(0)) for i in range(6)] for j in range(J + 1)]
    Sig2_ser = [[[sum((iv.mpf(LR[i][k]) * Sig_ser[j][k][l] * iv.mpf(LR[x][l]) for k in range(6) for l in range(6)), iv.mpf(0))
                  for x in range(6)] for i in range(6)] for j in range(J + 1)]

    # the exact lemma CB2s says these vanish; the enclosures must agree
    for a in range(6):
        ck(ub(X[-1][a]) < 1e-55, 'CB3 mu pole z^-1 enclosure contains only tiny values, coord %d' % a)
    for x in range(6):
        for y in range(6):
            ck(ub(D[-2][x][y]) < 1e-50 and ub(D[-1][x][y]) < 1e-50, 'CB3 Sigma pole enclosures tiny %d %d' % (x, y))
    # E V^(0) = Y^(-1) at the Gram level: A_{-1} = E G0 (the matrix E is exact, from CB2s)
    for a in range(6):
        for k in range(6):
            ev = sum((iv_frac(E_sel[a][l]) * G0[l][k] for l in range(6)), iv.mpf(0))
            ck(ub(A_m1[a][k] - ev) < 1e-80, 'CB3 A_{-1} = E G0 entry %d %d' % (a, k))
    emit('CB3 poles of mu (z^-1) and Sigma (z^-2, z^-1): exact zeros by CB2s; enclosures agree')
    ck(ub(mu_ser[0][0] + mpf(6) / 5) < 1e-60, 'CB3 mu_qM(0) = -6/5')
    ck(ub(mu_ser[0][1] + mpf(6) / 5) < 1e-60, 'CB3 mu_qS(0) = -6/5')
    ck(ub(mu_ser[0][2] + 1) < 1e-60, 'CB3 mu_aM(0) = -1 (window identity)')
    ck(ub(mu_ser[0][3] - 1) < 1e-60, 'CB3 mu_aS(0) = +1')
    ck(ub(Sig_ser[0][2][2]) < 1e-60, 'CB3 Var(aM)(0) = 0')
    ck(ub(Sig_ser[0][0][0] - 2) < 1e-55, 'CB3 Var(qM)(0) = a4 - a2^2 = 2')
    emit('CB3 PASS: limit identities mu(0), Sigma(0) match the independent blow-up analysis')

    # ------------------------------------------------------------------ CB4: majorants
    # |K1^{(n)}(w)| <= A_n(rho) for complex |w| <= rho:
    # A_n = (1/Z) sum_{k in (pi/12)Z} w(k) |k|^n e^{|k| rho}; the lattice holds |j| <= Jl, and for
    # |k| >= k0 = (Jl+1) pi/12 >= 4 rho, w(k) e^{|k| rho} <= e^{-k^2/4}; consecutive terms
    # |k|^n e^{-k^2/4} then shrink by at most q = e^{-(2 Jl + 3) h^2 / 4} (1 + 1/(Jl+1))^n < 1.
    hstep, Jl, ks_iv, ws_iv, Z1_iv = ce._LATT_IV

    def abs_sum_A(n, rho_mpf):
        rho = iv.mpf(rho_mpf)
        tot = iv.mpf(0)
        for kk, ww in zip(ks_iv, ws_iv):
            ak = iv.mpf(ub(kk))
            tot += ww * ak ** n * iv.exp(ak * rho)
        j0 = Jl + 1
        k0 = j0 * hstep
        ck(k0.a >= 4 * rho_mpf, 'CB4 kernel tail cut k0 >= 4 rho')
        q = iv.exp(-(2 * j0 + 1) * hstep * hstep / 4) * (1 + iv.mpf(1) / j0) ** n
        ck(q.b < 1, 'CB4 kernel tail ratio < 1, n=%d' % n)
        tail = 2 * iv.exp(-k0 * k0 / 4) * k0 ** n / (1 - q)
        return (tot + iv.mpf((0, up(tail)))) / Z1_iv

    A_rho = [abs_sum_A(n, RHO) for n in range(9)]

    def entry_max2(FA, FB):
        tot = iv.mpf(0)
        for (xa, aa, ca) in FA:
            for (xb, ab, cb) in FB:
                n1 = aa[0] + ab[0]
                n2 = aa[1] + ab[1]
                ma = sum((iv.mpf(ub(c)) * iv.mpf(RHO) ** j for j, c in ca.items()), iv.mpf(0))
                mb = sum((iv.mpf(ub(c)) * iv.mpf(RHO) ** j for j, c in cb.items()), iv.mpf(0))
                tot += ma * mb * up(A_rho[n1]) * up(A_rho[n2])
        return up(tot)

    # sup of each Gram block over |z| = rho; Laurent coefficients obey |c_j| <= M rho^-j
    M_GP = max(entry_max2(PinsV[i], PinsV[j]) for i in range(6) for j in range(6))
    M_GY = max(entry_max2(SoftY[i], SoftY[j]) for i in range(6) for j in range(6))
    M_GYV = max(entry_max2(SoftY[i], PinsV[j]) for i in range(6) for j in range(6))
    emit('CB4 Gram block maxima on |z|=%s: GP %s GY %s GYV %s' % (mp.nstr(RHO, 3), mp.nstr(M_GP, 6), mp.nstr(M_GY, 6), mp.nstr(M_GYV, 6)))

    def matmax(M):
        if MUT == 'signed_maxima':
            return max(mpf(M[i][j].b) for i in range(6) for j in range(6))
        return max(ub(M[i][j]) for i in range(6) for j in range(6))

    def vecmax(v):
        if MUT == 'signed_maxima':
            return max(mpf(x.b) for x in v)
        return max(ub(x) for x in v)

    G_m = [matmax(Gj[j]) for j in range(NK + 1)]
    GY_m = [matmax(GYj[j]) for j in range(NK + 1)]
    A_m = [matmax(Aj[j]) for j in range(NK + 1)]
    H_m = [matmax(H[j]) for j in range(NK + 1)]
    W_m = [vecmax(W[j]) for j in range(NK + 1)]
    Ginv0_m = matmax(Ginv0)
    A1_m = matmax(A_m1)
    # B1 guard: every majorant dominates every entry of the matrix or vector it majorizes
    dom = all(G_m[j] >= ub(Gj[j][a][b]) and GY_m[j] >= ub(GYj[j][a][b]) and A_m[j] >= ub(Aj[j][a][b])
              and H_m[j] >= ub(H[j][a][b]) for j in range(NK + 1) for a in range(6) for b in range(6))
    dom = dom and all(W_m[j] >= ub(W[j][a]) for j in range(NK + 1) for a in range(6))
    dom = dom and all(Ginv0_m >= ub(Ginv0[a][b]) and A1_m >= ub(A_m1[a][b]) for a in range(6) for b in range(6))
    ck(dom, 'CB4 majorants dominate every |entry| (B1)')

    RHOi = iv.mpf(RHO)

    def theta(R):
        # 36 ||Ginv0|| sum_{i>=1} g_i R^-i, with g_i = G_m[i] (i <= J) and the Cauchy bound beyond
        Ri = iv.mpf(R)
        x = 1 / (RHOi * Ri)
        s = sum((iv.mpf(G_m[i]) / Ri ** i for i in range(1, J + 1)), iv.mpf(0)) + iv.mpf(M_GP) * x ** (J + 1) / (1 - x)
        return 36 * iv.mpf(Ginv0_m) * s

    # majorant growth R: |H_j|, |W_j| <= C R^j for all j once theta(R) <= 1/2 (induction on the
    # recursion H_j = -Ginv0 sum_i G_i H_{j-i}, base j <= NK from the computed coefficients)
    R = 1 / RHO * mpf('1.001')
    for _ in range(600):
        if up(theta(R)) > 0.5:
            R *= mpf('1.02')
        else:
            break
    ck(up(theta(R)) <= 0.5, 'CB4 majorant growth R found')
    for _ in range(600):
        Rl = R / mpf('1.02')
        if (RHOi * iv.mpf(Rl)).a > 1 and up(theta(Rl)) <= 0.5:
            R = Rl
        else:
            break
    ck((RHOi * iv.mpf(R)).a > 1 and up(theta(R)) <= 0.5, 'CB4 rho R > 1 and theta(R) <= 1/2 in interval arithmetic')
    emit('CB4 majorant growth R = %s (rho R = %s, theta = %s)' % (mp.nstr(R, 8), mp.nstr(R * RHO, 4), mp.nstr(up(theta(R)), 4)))
    Ri = iv.mpf(R)
    C_W = max([mpf(1)] + [up(iv.mpf(W_m[j]) / Ri ** j) for j in range(NK + 1)])
    C_H = max([mpf(1)] + [up(iv.mpf(H_m[j]) / Ri ** j) for j in range(NK + 1)])

    class Maj:
        """Majorant of a nonnegative sequence: c[j] for j <= N, then C q^j."""

        def __init__(self, coeffs, C, q):
            self.c = [iv.mpf(x) for x in coeffs]
            self.N = len(coeffs) - 1
            self.C = iv.mpf(C)
            self.q = iv.mpf(q)
            self._t = None

        def coef(self, j):
            return self.c[j] if j <= self.N else self.C * self.q ** j

        def tails(self, t):
            # T[m] = sum_{j >= m} a_j t^j for m = 0 .. N+1 (suffix sums of nonnegative terms)
            if self._t is not None and self._t[0] is t:
                return self._t[1]
            qt = self.q * t
            ck(qt.b < 1, 'CB4 majorant series converges (q t < 1)')
            T = [None] * (self.N + 2)
            T[self.N + 1] = self.C * qt ** (self.N + 1) / (1 - qt)
            for j in range(self.N, -1, -1):
                T[j] = T[j + 1] + self.c[j] * t ** j
            self._t = (t, T)
            return T

        def tail_from(self, m, t):
            m = max(m, 0)
            if m <= self.N + 1:
                return self.tails(t)[m]
            qt = self.q * t
            return self.C * qt ** m / (1 - qt)

        def full(self, t):
            return self.tail_from(0, t)

    def P2(Aq, Bq, m, t):
        # sum over i + l >= m of a_i b_l t^{i+l}
        s = sum((Aq.coef(i) * t ** i * Bq.tail_from(m - i, t) for i in range(0, max(m, 0))), iv.mpf(0))
        return s + Aq.tail_from(m, t) * Bq.full(t)

    def P3(Aq, Bq, Cq, m, t):
        # sum over a + b + c >= m of a_a b_b c_c t^{a+b+c}
        s = sum((Aq.coef(a) * t ** a * P2(Bq, Cq, m - a, t) for a in range(0, max(m, 0))), iv.mpf(0))
        return s + Aq.tail_from(m, t) * Bq.full(t) * Cq.full(t)

    majW = Maj(W_m, C_W, R)
    majH = Maj(H_m, C_H, R)
    majA = Maj(A_m, M_GYV, 1 / RHO)
    majGY = Maj(GY_m, M_GY, 1 / RHO)
    R_STAR = F(1, 16384) if MUT != 'wide_first_cell' else F(1, 400)
    ck(up(Ri * iv_frac(R_STAR)) < mpf('0.2'), 'CB4 R-majorant converges on the first cell (0, %s]' % frac_label(R_STAR))
    tS = iv.mpf(up(iv_frac(R_STAR)))

    def mu_tail(t):
        # entries of sum_{k > J} mu_k t^k: mu_k = sum_{i=-1}^{k} A_i W_{k-i}
        main = 6 * P2(majA, majW, J + 1, t)
        pole = 6 * iv.mpf(A1_m) * majW.tail_from(J + 2, t) / t
        return up(main + pole)

    def Sig_tail(t):
        # entries of sum_{k > J} Sigma_k t^k: Sigma_k = GY_k - sum_{a+b+c=k} A_a H_b A_c^T
        gy = majGY.tail_from(J + 1, t)
        prod = 36 * P3(majA, majH, majA, J + 1, t)
        pole1 = 2 * 36 * iv.mpf(A1_m) * P2(majH, majA, J + 2, t) / t
        pole2 = 36 * iv.mpf(A1_m) ** 2 * majH.tail_from(J + 3, t) / (t * t)
        return up(gy + prod + pole1 + pole2)

    ST = Sig_tail(tS)
    MT = mu_tail(tS)
    ck(ST < 1e-5 and ST >= 0, 'CB4 Sigma tail small on the first cell')
    ck(MT < 1e-5 and MT >= 0, 'CB4 mu tail small on the first cell')
    emit('CB4 PASS: first cell (0, %s]: R t = %s, Sigma tail %s, mu tail %s' % (
        frac_label(R_STAR), mp.nstr(up(Ri * tS), 4), mp.nstr(ST, 3), mp.nstr(MT, 3)))

    def gram_tail(M, r_up):
        x = iv.mpf(r_up) / RHOi
        return up(iv.mpf(M) * x ** (NK + 1) / (1 - x))

    # ------------------------------------------------------ CB5: truncated moments and Wick
    def trunc_mom_iv(mu, var, a, b, kmax):
        # E[X^k 1_{a <= X <= b}], X ~ N(mu, var), every argument an interval
        s = iv.sqrt(var)
        al = (a - mu) / s
        be = (b - mu) / s
        I = [iv.mpf(0)] * (kmax + 1)
        I[0] = iv_Phi(be) - iv_Phi(al)
        I[1] = iv_phi(al) - iv_phi(be)
        for e in range(2, kmax + 1):
            I[e] = al ** (e - 1) * iv_phi(al) - be ** (e - 1) * iv_phi(be) + (e - 1) * I[e - 2]
        out = []
        for k in range(kmax + 1):
            tot = iv.mpf(0)
            for j in range(k + 1):
                tot += comb(k, j) * s ** j * I[j] * mu ** (k - j)
            out.append(tot)
        return out

    def make_wick_iv(mu, Sig):
        @lru_cache(maxsize=None)
        def wick(state):
            if len(state) == 0:
                return iv.mpf(1)
            if len(state) == 1:
                return mu[state[0]]
            i = state[0]
            rest = state[1:]
            tot = mu[i] * wick(rest)
            for k, j in enumerate(rest):
                tot += Sig[i][j] * wick(rest[:k] + rest[k + 1:])
            return tot
        return wick

    def poly_expect_iv(P, wick):
        tot = iv.mpf(0)
        for e, c in P.items():
            st = []
            for i, pp in enumerate(e):
                st += [i] * pp
            tot += c * wick(tuple(st))
        return tot

    def p_affine(idx):
        P = defaultdict(lambda: iv.mpf(0))
        P[tuple(1 if k == idx else 0 for k in range(6))] = iv.mpf(1)
        return P

    def p_mul(A, B):
        C = defaultdict(lambda: iv.mpf(0))
        for ea, ca in A.items():
            for eb, cb in B.items():
                C[tuple(x + y for x, y in zip(ea, eb))] += ca * cb
        return C

    def p_scale(A, s):
        C = defaultdict(lambda: iv.mpf(0))
        for e, c in A.items():
            C[e] = c * s
        return C

    def p_add(A, B):
        C = defaultdict(lambda: iv.mpf(0))
        for e, c in A.items():
            C[e] += c
        for e, c in B.items():
            C[e] += c
        return C

    def tp_add(A, B):
        n = max(len(A), len(B))
        return [(A[i] if i < len(A) else iv.mpf(0)) + (B[i] if i < len(B) else iv.mpf(0)) for i in range(n)]

    def tp_mul(A, B):
        C = [iv.mpf(0)] * (len(A) + len(B) - 1)
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                C[i + j] += a * b
        return C

    def tp_scale(A, s):
        return [s * a for a in A]

    def conditional_wick_iv(mu, Sig, cond):
        i0 = cond
        s00 = Sig[i0][i0]
        means = [[mu[i] - Sig[i][i0] / s00 * mu[i0], Sig[i][i0] / s00] for i in range(6)]
        Sc = [[Sig[i][j] - Sig[i][i0] * Sig[i0][j] / s00 for j in range(6)] for i in range(6)]

        @lru_cache(maxsize=None)
        def wick(state):
            if len(state) == 0:
                return (iv.mpf(1),)
            if len(state) == 1:
                return tuple(means[state[0]])
            i = state[0]
            rest = state[1:]
            tot = tp_mul(list(means[i]), list(wick(rest)))
            for k, j in enumerate(rest):
                tot = tp_add(tot, tp_scale(list(wick(rest[:k] + rest[k + 1:])), Sc[i][j]))
            return tuple(tot)
        return wick

    def cond_expect_poly_iv(P, cw):
        tot = (iv.mpf(0),)
        for e, c in P.items():
            st = []
            for i, pp in enumerate(e):
                st += [i] * pp
            tot = tp_add(list(tot), tp_scale(list(cw(tuple(st))), c))
        return tot

    # ---------------------------------------------------------- CB6: law on each cell
    def horner(coeffs, t_iv):
        v = iv.mpf(0)
        for c in reversed(coeffs):
            v = v * t_iv + c
        return v

    def ser_eval(coeffs, t_iv, tail):
        return horner(coeffs, t_iv) + iv.mpf((-tail, tail))

    def mat_poly_eval(mats, t_iv, shift=0):
        # sum_j mats[j] t^(j + shift), entrywise
        out = [[horner([m[a][b] for m in mats], t_iv) for b in range(6)] for a in range(6)]
        if shift:
            f = t_iv ** shift
            out = [[out[a][b] * f for b in range(6)] for a in range(6)]
        return out

    def vec_poly_eval(vecs, t_iv, shift=0):
        out = [horner([v[a] for v in vecs], t_iv) for a in range(6)]
        if shift:
            f = t_iv ** shift
            out = [x * f for x in out]
        return out

    def frob_up(M):
        return up(iv.sqrt(sum((iv.mpf(ub(M[i][j])) ** 2 for i in range(6) for j in range(6)), iv.mpf(0))))

    def l2_up(v):
        return up(iv.sqrt(sum((iv.mpf(ub(x)) ** 2 for x in v), iv.mpf(0))))

    def mp_inverse6(GG):
        A = [GG[i][:] + [mpf(1) if l == i else mpf(0) for l in range(6)] for i in range(6)]
        for c in range(6):
            p = max(range(c, 6), key=lambda rr: abs(A[rr][c]))
            A[c], A[p] = A[p], A[c]
            for rr in range(6):
                if rr != c:
                    f = A[rr][c] / A[c][c]
                    A[rr] = [A[rr][j] - f * A[c][j] for j in range(12)]
        return [[A[i][6 + j] / A[i][i] for j in range(6)] for i in range(6)]

    Xhi = [X[k] for k in range(J + 1, 2 * NK + 1)]
    Dhi = [D[k] for k in range(J + 1, 3 * NK + 1)]
    RWl = [RW[k] for k in range(0, 2 * NK + 1)]
    RHl = [RH[k] for k in range(0, 2 * NK + 1)]
    SQRT6 = iv.sqrt(iv.mpf(6))

    def neumann_law(c1, c2):
        """Certified (mu, Sigma) on the cell [c1, c2] (exact rationals, c1 > 0).
        Heavy part: the order-J Taylor polynomials, interval Horner. Correction (B2): exact identities
          G (w - W_P) = (zV - G_P W_P) - (G - G_P) W_P,
          G (G^-1 - H_P) = (I - G_P H_P) - (G - G_P) H_P,
          mu = A_P W_P + (A - A_P) W_P + A (w - W_P),
          Sigma = [GY_P - A_P H_P A_P^T] + (GY - GY_P) - [cross terms in A - A_P] - A (G^-1 - H_P) A^T,
        where A_P W_P and GY_P - A_P H_P A_P^T are explicit Laurent polynomials whose orders 0..J are
        the heavy part; their pole and high orders, the residual polynomials and the Cauchy tails of
        the Gram blocks (from order J+3) are bounded on the cell."""
        lo = iv_frac(c1)
        hi = iv_frac(c2)
        t_iv = iv.mpf((mpf(lo.a), mpf(hi.b)))
        r_lo = mpf(lo.a)
        r_hi = mpf(hi.b)
        ck(r_lo > 0, 'CB6 Neumann cell away from 0')
        tailG = gram_tail(M_GP, r_hi)
        tailA = gram_tail(M_GYV, r_hi)
        tailY = gram_tail(M_GY, r_hi)
        GP_iv = mat_poly_eval(Gj, t_iv)
        GP_iv = [[GP_iv[a][b] + iv.mpf((-tailG, tailG)) for b in range(6)] for a in range(6)]
        mid = [[(mpf(GP_iv[i][j].a) + mpf(GP_iv[i][j].b)) / 2 for j in range(6)] for i in range(6)]
        Bm = mp_inverse6(mid) if MUT != 'no_preconditioner' else [[mpf(1) if i == j else mpf(0) for j in range(6)] for i in range(6)]
        Biv = [[iv.mpf(Bm[i][j]) for j in range(6)] for i in range(6)]
        Emat = [[(1 if i == j else 0) - sum((Biv[i][k] * GP_iv[k][j] for k in range(6)), iv.mpf(0)) for j in range(6)] for i in range(6)]
        rhoE = frob_up(Emat)
        ck(rhoE < 0.5, 'CB6 Neumann contraction on [%s,%s]' % (frac_label(c1), frac_label(c2)))
        BF = frob_up(Biv)
        Gnorm = up(iv.mpf(BF) / (1 - iv.mpf(rhoE)))        # ||G(t)^-1||_2 for every t in the cell
        if MUT == 'old_remainder':
            # the frozen correction, kept only as a negative control
            GinvPnorm = 6 * sum(H_m[j] * r_hi ** j for j in range(len(H_m)))
            Gn = BF + 6 * BF * rhoE / (1 - rhoE)
            dGinv = Gn * 6 * tailG * GinvPnorm
            APo = mat_poly_eval(Aj, t_iv)
            APo = [[APo[a][b] + A_m1[a][b] / t_iv for b in range(6)] for a in range(6)]
            GYVmax = max(ub(APo[a][b]) for a in range(6) for b in range(6))
            zVmax = max(ub(x) for x in zV) + mpf('1e-50')
            T_mu = 6 * GYVmax * dGinv * zVmax + 6 * tailA * Gn * zVmax
            T_Sig = tailY + 36 * GYVmax ** 2 * dGinv + 2 * 36 * tailA * Gn * GYVmax + 36 * tailA ** 2 * Gn
        else:
            WP = vec_poly_eval(W, t_iv)
            WPn = l2_up(WP)
            WPmax = max(ub(x) for x in WP)
            rWn = l2_up(vec_poly_eval(RWl, t_iv))
            Ew = up(iv.mpf(Gnorm) * (iv.mpf(rWn) + 6 * iv.mpf(tailG) * iv.mpf(WPn)))
            AP = mat_poly_eval(Aj, t_iv)
            AP = [[AP[a][b] + A_m1[a][b] / t_iv for b in range(6)] for a in range(6)]
            AmaxP = max(ub(AP[a][b]) for a in range(6) for b in range(6))
            Amax = up(iv.mpf(AmaxP) + iv.mpf(tailA))
            ext_mu = max(ub(x) for x in vec_poly_eval(Xhi, t_iv, shift=J + 1))
            pole_mu = up(iv.mpf(max(ub(x) for x in X[-1])) / iv.mpf(r_lo))
            T_mu = up(iv.mpf(ext_mu) + iv.mpf(pole_mu) + 6 * iv.mpf(tailA) * iv.mpf(WPmax) + SQRT6 * iv.mpf(Amax) * iv.mpf(Ew))
            HP = mat_poly_eval(H, t_iv)
            HPF = frob_up(HP)
            RHF = frob_up(mat_poly_eval(RHl, t_iv))
            EH = up(iv.mpf(Gnorm) * (iv.mpf(RHF) + 6 * iv.mpf(tailG) * iv.mpf(HPF)))
            Dext = mat_poly_eval(Dhi, t_iv, shift=J + 1)
            ext_S = max(ub(Dext[a][b]) for a in range(6) for b in range(6))
            rl = iv.mpf(r_lo)
            pole_S = up(iv.mpf(max(ub(D[-2][a][b]) for a in range(6) for b in range(6))) / (rl * rl)
                        + iv.mpf(max(ub(D[-1][a][b]) for a in range(6) for b in range(6))) / rl)
            tA = iv.mpf(tailA)
            T_Sig = up(iv.mpf(ext_S) + iv.mpf(pole_S) + iv.mpf(tailY) + 12 * tA * iv.mpf(HPF) * iv.mpf(AmaxP)
                       + 6 * tA * tA * iv.mpf(HPF) + 6 * iv.mpf(Amax) ** 2 * iv.mpf(EH))
        ck(T_mu < 1e-2 and T_Sig < 1e-2, 'CB6 law correction small on [%s,%s]' % (frac_label(c1), frac_label(c2)))
        mu = [ser_eval([mu_ser[j][a] for j in range(J + 1)], t_iv, T_mu) for a in range(6)]
        Sig = [[ser_eval([Sig_ser[j][a][b] for j in range(J + 1)], t_iv, T_Sig) for b in range(6)] for a in range(6)]
        return mu, Sig, T_mu, T_Sig, t_iv

    def first_cell_law():
        t_iv = iv.mpf((0, up(iv_frac(R_STAR))))
        mu = [ser_eval([mu_ser[j][a] for j in range(J + 1)], t_iv, MT) for a in range(6)]
        Sig = [[ser_eval([Sig_ser[j][a][b] for j in range(J + 1)], t_iv, ST) for b in range(6)] for a in range(6)]
        return mu, Sig, MT, ST, t_iv

    I_LO = mpf('-3.9')
    I_HI = mpf('-0.4')
    ETA = mpf('0.3')
    BB = mpf('2')
    AB = mpf('0.8')
    EDGE_AM = mpf('-0.6')
    CAP_CONS = mpf(4)   # consumption cap of LPW_CONSTANT v4

    def polys(r_iv):
        aqM, aqS, aaM, aaS, abM, abS = [p_affine(i) for i in range(6)]
        DM = p_add(p_mul(aaM, aqM), p_scale(p_mul(abM, abM), -r_iv))
        DS = p_add(p_mul(aaS, aqS), p_scale(p_mul(abS, abS), -r_iv))
        P = p_scale(p_mul(DM, DS), -1)
        one = defaultdict(lambda: iv.mpf(0))
        one[(0,) * 6] = iv.mpf(1)

        def ubar(aa, qq, bb):
            u = p_add(p_scale(p_mul(p_mul(aa, qq), p_mul(aa, qq)), iv.mpf(1) / 2), p_scale(one, iv.mpf(1) / 2))
            return p_add(u, p_scale(p_mul(bb, bb), r_iv))
        PBAR = p_mul(ubar(aaM, aqM, abM), ubar(aaS, aqS, abS))
        aqS2 = p_add(p_affine(0), p_affine(1))
        PBAR2 = p_mul(ubar(aaM, aqM, abM), ubar(aaS, aqS2, abS))
        return P, PBAR, PBAR2

    def t0_lower(num_iv, var_iv):
        return mpf((num_iv / iv.sqrt(iv.mpf(up(var_iv)))).a)

    def ceil_hi(mu, Sig, r_iv, label):
        # UPPER envelope (direction analysis in README.md, unchanged from the frozen program):
        #   E[G_r] = E[p 1_type] <= E[p 1_{q_M<0}] + E[pbar 1_{q_M<0, D_M<=0}] + E[pbar 1_{D_S>=0}],
        #   {D_M<=0, q_M<0} in {a_M>-0.6} u {|b_M|>2} u {q_M in (-0.4,0)}, {D_S>=0} in {q_S>=0} u {a_S<=0}.
        q = mpf_to_fraction
        ck(q(-EDGE_AM) * q(-I_HI) > q(up(r_iv)) * q(BB) ** 2, 'CC6 D_M cover margin ' + label)
        P, PBAR, _ = polys(r_iv)
        if MUT == 'cap_raised':
            PBAR = p_scale(PBAR, iv.mpf(2))
            P = p_scale(P, iv.mpf(2))
            emit('CC6 MUTATION cap_raised applied (must be caught at consumption)')
        wick = make_wick_iv(tuple(mu), tuple(tuple(row) for row in Sig))
        Epbar2 = up(poly_expect_iv(p_mul(PBAR, PBAR), wick))
        Ep2 = up(poly_expect_iv(p_mul(P, P), wick))
        ck(Epbar2 >= 0 and Ep2 >= 0, 'CC6 second moments nonnegative ' + label)

        def trunc_expect(PP, cond, a, b):
            cw = conditional_wick_iv(mu, Sig, cond)
            cp = cond_expect_poly_iv(PP, cw)
            tm = trunc_mom_iv(mu[cond], Sig[cond][cond], iv.mpf(a), iv.mpf(b), len(cp) - 1)
            return sum((c * t for c, t in zip(cp, tm)), iv.mpf(0))

        def cs_piece(second, prob):
            return iv.mpf((0, up(iv.sqrt(iv.mpf(second) * iv.mpf(prob)))))

        def chern_gt(cond, edge):
            # E[pbar 1_{X > edge}] <= sqrt(E[pbar^2] P(X > edge)), edge above the mean
            t0 = t0_lower(iv.mpf(edge) - iv.mpf(up(mu[cond])), Sig[cond][cond])
            ck(t0 > 2, 'CC6 Chernoff margin ' + label)
            return cs_piece(Epbar2, gauss_tail_upper(t0, 1))

        def chern_lt(cond, edge):
            t0 = t0_lower(iv.mpf(mpf(mu[cond].a)) - iv.mpf(edge), Sig[cond][cond])
            ck(t0 > 2, 'CC6 Chernoff margin ' + label)
            return cs_piece(Epbar2, gauss_tail_upper(t0, 1))

        Wp1 = trunc_expect(P, 0, I_LO, I_HI)
        Wp2 = trunc_expect(P, 0, I_HI, mpf(0))
        Wp3 = trunc_expect(P, 0, mpf(-100), I_LO)
        t0l = t0_lower(iv.mpf(100) + iv.mpf(mpf(mu[0].a)), Sig[0][0])
        ck(t0l > 20, 'CC6 left residue margin ' + label)
        RESp = cs_piece(Ep2, gauss_tail_upper(t0l, 1))
        BaM = chern_gt(2, EDGE_AM)                                       # E[pbar 1_{a_M > -0.6}]
        BbM = trunc_expect(PBAR, 4, mpf('-100'), -BB) + trunc_expect(PBAR, 4, BB, mpf('100'))
        t0b = t0_lower(iv.mpf(100) - iv.mpf(ub(mu[4])), Sig[4][4])
        ck(t0b > 20, 'CC6 beta far-tail margin ' + label)
        BbM = BbM + cs_piece(Epbar2, gauss_tail_upper(t0b, 2))          # |b_M| > 100: two tails
        BqM = trunc_expect(PBAR, 0, I_HI, mpf(0))                        # q_M in (-0.4, 0)
        BqS = trunc_expect(PBAR, 1, mpf(0), mpf('100'))                  # q_S >= 0
        t0q = t0_lower(iv.mpf(100) - iv.mpf(up(mu[1])), Sig[1][1])
        ck(t0q > 20, 'CC6 q_S far-tail margin ' + label)
        BqS = BqS + cs_piece(Epbar2, gauss_tail_upper(t0q, 1))
        BaS = chern_lt(3, mpf('0'))                                      # E[pbar 1_{a_S <= 0}]
        if MUT == 'drop_fail_piece':
            hi = Wp1 + Wp2 + Wp3 + RESp
        else:
            hi = Wp1 + Wp2 + Wp3 + RESp + BaM + BbM + BqM + BqS + BaS
        emit('CC6 %-18s hi.b = %s  (p-win %s %s %s; fails %s %s %s %s %s)' % (
            label, mp.nstr(up(hi), 20), mp.nstr(up(Wp1), 10), mp.nstr(up(Wp2), 10), mp.nstr(up(Wp3), 10),
            mp.nstr(up(BaM), 3), mp.nstr(up(BbM), 6), mp.nstr(up(BqM), 6), mp.nstr(up(BqS), 6), mp.nstr(up(BaS), 3)))
        return hi

    def band_lo(mu, Sig, Tm, Ts, t_iv, label):
        # LOWER envelope: B_cert = {q_M in [I_LO, I_HI]} {|delta| <= eta} {a_M <= -0.8} {a_S >= 0.8} {|b_M| <= 2}
        # lies inside the type event, p > 0 there, and pbar >= |p|, so
        #   E[G_r] >= E[p 1_{q_M in I'}] - E[pbar 1_{|delta|>eta}] - E[pbar 1_{|b_M|>2}]
        #             - E[pbar 1_{a_M>-0.8}] - E[pbar 1_{a_S<0.8}].
        mu2 = [ser_eval([mu2_ser[j][a] for j in range(J + 1)], t_iv, up(2 * iv.mpf(Tm))) for a in range(6)]
        Sig2 = [[ser_eval([Sig2_ser[j][a][b] for j in range(J + 1)], t_iv, up(4 * iv.mpf(Ts))) for b in range(6)] for a in range(6)]
        r_hi = mpf_to_fraction(up(t_iv))
        q = mpf_to_fraction
        ck(q(AB) * q(-I_HI) - r_hi * q(BB) ** 2 > F(1, 20), 'CB6 type margin D_M>0 ' + label)
        ck(q(AB) * (q(I_HI) + q(ETA)) < F(-1, 20), 'CB6 type margin D_S<0 ' + label)
        ck(q(I_HI) + q(ETA) < 0, 'CB6 q_S<0 ' + label)
        P, PBAR, PBAR2 = polys(t_iv)
        wick = make_wick_iv(tuple(mu), tuple(tuple(row) for row in Sig))
        cw0 = conditional_wick_iv(mu, Sig, 0)
        cp0 = cond_expect_poly_iv(P, cw0)
        tm0 = trunc_mom_iv(mu[0], Sig[0][0], iv.mpf(I_LO), iv.mpf(I_HI), len(cp0) - 1)
        EqI = sum((c * t for c, t in zip(cp0, tm0)), iv.mpf(0))
        Ebar = poly_expect_iv(PBAR, wick)
        wick2 = make_wick_iv(tuple(mu2), tuple(tuple(row) for row in Sig2))
        Ebar2 = poly_expect_iv(PBAR2, wick2)
        ck(Ebar.a <= Ebar2.b and Ebar2.a <= Ebar.b, 'CB6 pbar enclosures intersect ' + label)
        Epbar2 = up(poly_expect_iv(p_mul(PBAR, PBAR), wick))
        ck(Epbar2 >= 0, 'CB6 second moment nonnegative ' + label)

        def piece_outside(PP, muX, SigX, cond, a, b, Etot, oneside=None, force_chern=False):
            s00 = SigX[cond][cond]
            if (not force_chern) and s00.a > 1e-4:
                cw = conditional_wick_iv(muX, SigX, cond)
                cp = cond_expect_poly_iv(PP, cw)
                tm = trunc_mom_iv(muX[cond], s00, iv.mpf(a), iv.mpf(b), len(cp) - 1)
                return Etot - sum((c * t for c, t in zip(cp, tm)), iv.mpf(0))
            # exact Gaussian tail (certified Phi) and Cauchy-Schwarz
            if oneside == 'gt':     # P(X > b), b above the mean
                t0 = t0_lower(iv.mpf(b) - iv.mpf(up(muX[cond])), s00)
                sides = 1
            elif oneside == 'lt':   # P(X < a), a below the mean
                t0 = t0_lower(iv.mpf(mpf(muX[cond].a)) - iv.mpf(a), s00)
                sides = 1
            else:                   # P(|X| > b) with a = -b: two tails (B3)
                t0 = t0_lower(iv.mpf(b) - iv.mpf(ub(muX[cond])), s00)
                sides = 2
            ck(t0 > 2, 'CB6 Chernoff tail margin ' + label)
            return iv.mpf((0, up(iv.sqrt(iv.mpf(Epbar2) * iv.mpf(gauss_tail_upper(t0, sides))))))

        Edelta = piece_outside(PBAR2, mu2, Sig2, 1, -ETA, ETA, Ebar2, force_chern=True)
        Ebeta = piece_outside(PBAR, mu, Sig, 4, -BB, BB, Ebar)
        EaM = piece_outside(PBAR, mu, Sig, 2, mpf('-100'), -AB, Ebar, oneside='gt', force_chern=True)
        EaS = piece_outside(PBAR, mu, Sig, 3, AB, mpf('100'), Ebar, oneside='lt', force_chern=True)
        lo = EqI - Edelta - Ebeta - EaM - EaS
        emit('CB6 %-18s lo.a = %s  (EqI %s; pieces %s %s %s %s)' % (
            label, mp.nstr(mpf(lo.a), 20), mp.nstr(mpf(EqI.a), 10),
            mp.nstr(up(Edelta), 3), mp.nstr(up(Ebeta), 6), mp.nstr(up(EaM), 3), mp.nstr(up(EaS), 3)))
        return lo

    CELLS = [(F(0), R_STAR), (R_STAR, F(1, 400)), (F(1, 400), F(1, 200)), (F(1, 200), F(1, 100)),
             (F(1, 100), F(1, 50)), (F(1, 50), F(3, 100)), (F(3, 100), F(1, 25)), (F(1, 25), RMAX)]
    ck(all(CELLS[i][1] == CELLS[i + 1][0] for i in range(len(CELLS) - 1)) and CELLS[0][0] == 0 and CELLS[-1][1] == RMAX,
       'CB6 cells tile (0, 1/20]')
    results = []
    for (c1, c2) in CELLS:
        label = '(%s,%s]' % (frac_label(c1), frac_label(c2))
        if c1 == 0:
            mu, Sig, Tm, Ts, t_iv = first_cell_law()
            method = 'R-majorant'
        else:
            mu, Sig, Tm, Ts, t_iv = neumann_law(c1, c2)
            method = 'Neumann'
        emit('CB6 %-18s law %s: correction mu %s Sigma %s' % (label, method, mp.nstr(Tm, 3), mp.nstr(Ts, 3)))
        if mode == 'ceil':
            hi = ceil_hi(mu, Sig, t_iv, label)
            ck(up(hi) < CAP_CONS, 'CC6 consumption cap (U <= 4) on ' + label)
            bound = decimal_outward(up(hi), 'up')
            emit('CERT cell=%s method=%s E[G_r] <= %s' % (label, method, bound))
        else:
            lo = band_lo(mu, Sig, Tm, Ts, t_iv, label)
            ck(lo.a > REQ, 'CB6 band lo > c_Z on ' + label)
            bound = decimal_outward(mpf(lo.a), 'down')
            emit('CERT cell=%s method=%s E[G_r] >= %s' % (label, method, bound))
        results.append((label, method, bound))
    pick = max if mode == 'ceil' else min
    uniform = pick((Fraction(b) for _, _, b in results))
    rel = '<=' if mode == 'ceil' else '>='
    emit('CERT uniform=(0,1/20] E[G_r] %s %s' % (rel, decimal_outward(uniform, 'up' if mode == 'ceil' else 'down')))

    # ----------------------------------------------- CB7: rung cross-check (not load-bearing)
    # direct point law at r = 0.05 from ce.cov in mp arithmetic, with no series machinery
    def direct_point_law(rpt):
        Mx = (-rpt / 2, mpf(0))
        Sx = (rpt / 2, mpf(0))
        V = [[(Mx, (0, 0), mpf(1))], [(Mx, (1, 0), mpf(1))], [(Mx, (0, 1), mpf(1))],
             [(Sx, (1, 0), 1 / rpt), (Mx, (1, 0), -1 / rpt)], [(Sx, (0, 1), 1 / rpt), (Mx, (0, 1), -1 / rpt)],
             [(Sx, (0, 0), 1 / rpt ** 3), (Mx, (0, 0), -1 / rpt ** 3), (Sx, (1, 0), -1 / (2 * rpt ** 2)), (Mx, (1, 0), -1 / (2 * rpt ** 2))]]
        Yv = [[(Mx, (0, 2), mpf(1))], [(Sx, (0, 2), mpf(1))],
              [(Mx, (2, 0), 1 / rpt)], [(Sx, (2, 0), 1 / rpt)],
              [(Mx, (1, 1), 1 / rpt)], [(Sx, (1, 1), 1 / rpt)]]

        def lc(FA, FB):
            tot = mpf(0)
            for (xa, aa, ca) in FA:
                for (xb, ab, cb) in FB:
                    tot += ca * cb * ce.cov(aa, ab, xa[0] - xb[0], xa[1] - xb[1], mode='mpf')
            return tot
        GP = [[lc(V[i], V[j]) for j in range(6)] for i in range(6)]
        GY = [[lc(Yv[i], Yv[j]) for j in range(6)] for i in range(6)]
        GYV = [[lc(Yv[i], V[j]) for j in range(6)] for i in range(6)]
        Gi = mp_inverse6(GP)
        zVd = [mpf(6) / 5, 0, 0, 0, 0, mpf(-1) / 6]
        csol = [sum(Gi[i][k] * zVd[k] for k in range(6)) for i in range(6)]
        muD = [sum(GYV[i][k] * csol[k] for k in range(6)) for i in range(6)]
        SigD = [[GY[i][j] - sum(GYV[i][k] * sum(Gi[k][l] * GYV[j][l] for l in range(6)) for k in range(6)) for j in range(6)] for i in range(6)]
        return muD, SigD

    muD, SigD = direct_point_law(mpf(1) / 20)
    muN, SigN, TmN, TsN, _ = neumann_law(RMAX, RMAX)
    for a in range(6):
        ck(muN[a].a <= muD[a] <= muN[a].b, 'CB7 Neumann mu contains the direct mu at the rung, coord %d' % a)
        for b in range(6):
            ck(SigN[a][b].a <= SigD[a][b] <= SigN[a][b].b, 'CB7 Neumann Sigma contains the direct Sigma, %d %d' % (a, b))
    dev_mu = max(abs(muD[a] - (mpf(muN[a].a) + mpf(muN[a].b)) / 2) for a in range(6))
    dev_S = max(abs(SigD[a][b] - (mpf(SigN[a][b].a) + mpf(SigN[a][b].b)) / 2) for a in range(6) for b in range(6))
    emit('CB7 PASS: Neumann (mu, Sigma) at r=1/20 contain the independent direct point law (cross-check, not load-bearing); '
         'observed deviation mu %s Sigma %s against corrections %s %s' % (mp.nstr(dev_mu, 3), mp.nstr(dev_S, 3), mp.nstr(TmN, 3), mp.nstr(TsN, 3)))

    if mc:
        monte_carlo(neumann_law, results, mode)
    note('total runtime %.1fs' % (time.time() - t_start))
    return results


def monte_carlo(neumann_law, results, mode):
    """NON-CERTIFYING diagnostic: Monte Carlo E[G_r] at r = 0.05 and 0.01, to stderr only."""
    try:
        import numpy as np
    except ImportError:
        note('CC7 MC DIAGNOSTIC skipped: numpy absent (NON-CERTIFYING in any case)')
        return
    rng = np.random.default_rng(90210)
    for c in (F(1, 20), F(1, 100)):
        mu, Sig, _, _, _ = neumann_law(c, c)
        m = np.array([float((mpf(x.a) + mpf(x.b)) / 2) for x in mu])
        S = np.array([[float((mpf(Sig[a][b].a) + mpf(Sig[a][b].b)) / 2) for b in range(6)] for a in range(6)])
        S = (S + S.T) / 2
        L = np.linalg.cholesky(S + 1e-14 * np.eye(6))
        Y = m[None, :] + rng.standard_normal((400000, 6)) @ L.T
        qM, qS, aM, aS, bM, bS = Y.T
        r = float(c)
        DM = aM * qM - r * bM ** 2
        DS = aS * qS - r * bS ** 2
        G = np.where((qM < 0) & (DM > 0) & (DS < 0), -DM * DS, 0.0)
        note('CC7 MC DIAGNOSTIC (NON-CERTIFYING) r=%g: E[G] = %.5f +- %.5f' % (r, G.mean(), G.std(ddof=1) / np.sqrt(len(G))))


# ------------------------------------------------------------------ records and replay
def parse_certified(text):
    cells, uniform = [], None
    for ln in text.splitlines():
        if ln.startswith('CERT cell='):
            head, rel, bound = ln.rsplit(' ', 2)
            cell = head.split()[1][len('cell='):]
            method = head.split()[2][len('method='):]
            cells.append({'cell': cell, 'method': method, 'relation': rel, 'bound': bound})
        elif ln.startswith('CERT uniform='):
            head, rel, bound = ln.rsplit(' ', 2)
            uniform = {'interval': head.split()[1][len('uniform='):], 'relation': rel, 'bound': bound}
    return cells, uniform


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_record(mode, body):
    cells, uniform = parse_certified(body)
    return {
        'schema': 1,
        'program': PROGRAM,
        'program_sha256': file_sha256(__file__),
        'cov_exact_sha256': COV_EXACT_SHA256,
        'mode': mode,
        'environment': {'python': platform.python_version(), 'mpmath': mpmath.__version__ if mpmath else None},
        'transcript_sha256': hashlib.sha256(body.encode()).hexdigest(),
        'cells': cells,
        'uniform': uniform,
        'scientific_effect': 'NONE',
        'review_status': 'REVIEW_REQUIRED',
    }


def compare_record(record, expected):
    """Return a list of mismatches between a fresh record and the reviewed one."""
    bad = []
    for key in ('schema', 'program', 'program_sha256', 'cov_exact_sha256', 'mode', 'transcript_sha256', 'cells', 'uniform',
                'scientific_effect', 'review_status'):
        if record.get(key) != expected.get(key):
            bad.append(key)
    if record['environment'].get('mpmath') != expected.get('environment', {}).get('mpmath'):
        bad.append('environment.mpmath')
    if not expected.get('cells') or not expected.get('uniform'):
        bad.append('empty record')
    return bad


def main(argv=None):
    global MUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['structure', 'ceil', 'floor', 'verify-record'])
    ap.add_argument('--h2-dir', default=str(HERE / 'h2'))
    ap.add_argument('--mutation', default='', choices=[''] + sorted(MUTATIONS))
    ap.add_argument('--transcript', help='write the transcript here (ceil, floor) or read it (verify-record)')
    ap.add_argument('--record', help='write the replay record (JSON) here')
    ap.add_argument('--verify', help='compare the run with this reviewed record; exit 1 on any difference')
    ap.add_argument('--results', help='the reviewed record (verify-record)')
    ap.add_argument('--mc', action='store_true', help='NON-CERTIFYING Monte Carlo diagnostic to stderr (needs numpy)')
    args = ap.parse_args(argv)
    MUT = args.mutation
    if args.mode == 'structure':
        pins = [list(f) for f in PINS_EXACT]
        zv = list(ZV_EXACT)
        if MUT == 'structure_pin_pole':
            pins[5] = [pins[5][0], pins[5][1], ('S', (1, 0), F(-1), -2), pins[5][3]]
        if MUT == 'structure_window':
            zv[3] = F(1)
        try:
            E = structure_lemma(pins, SOFT_EXACT, zv)
        except CheckFailed as e:
            print('H3_BAND_V2 STRUCTURE FAIL: %s' % e)
            return 1
        rows = ['[' + ' '.join(frac_label(x) for x in row) + ']' for row in E]
        print('H3_BAND_V2 STRUCTURE PASS: pins regular; soft poles z^-1 only; Y^(-1) = E V^(0) with E = %s; E zV = 0' % ' '.join(rows))
        return 0
    if args.mode == 'verify-record':
        if not (args.transcript and args.results):
            print('H3_BAND_V2 VERIFY_FAIL: verify-record needs --transcript and --results')
            return 2
        body = Path(args.transcript).read_text()
        expected = json.loads(Path(args.results).read_text())
        bad = compare_record(make_record(expected.get('mode'), body), expected)
        if bad:
            print('H3_BAND_V2 VERIFY_FAIL: record mismatch in %s' % ', '.join(bad))
            return 1
        print('H3_BAND_V2 VERIFY PASS: transcript matches the reviewed %s record (%d cells)' % (expected['mode'], len(expected['cells'])))
        return 0
    if mpmath is None:
        print('H3_BAND_V2 ERROR: mpmath is required for the certificate')
        return 2
    try:
        run_certificate(args.mode, args.h2_dir, mc=args.mc)
    except CheckFailed as e:
        print('H3_BAND_V2 FAIL: %s' % e)
        return 1
    emit('ALL_CHECKS_PASS')
    body = '\n'.join(OUT) + '\n'
    print('sha256(transcript-above) = ' + hashlib.sha256(body.encode()).hexdigest())
    if args.transcript:
        Path(args.transcript).write_text(body)
    record = make_record(args.mode, body)
    if args.record:
        Path(args.record).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    if args.verify:
        bad = compare_record(record, json.loads(Path(args.verify).read_text()))
        if bad:
            print('H3_BAND_V2 VERIFY_FAIL: record mismatch in %s' % ', '.join(bad))
            return 1
    u = record['uniform']
    print('H3_BAND_V2 PASS %s: E[G_r] %s %s on (0, 1/20]%s' % (args.mode, u['relation'], u['bound'],
                                                             '; matches the reviewed record' if args.verify else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
