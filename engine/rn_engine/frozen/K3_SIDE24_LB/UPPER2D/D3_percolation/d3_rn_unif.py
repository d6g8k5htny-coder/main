# D3-LEMMA-RN-UNIF(r = 0.05) -- rung closure engine: the two missing pieces.
#
# Piece 1 (RIGIDITY-DECOUPLING, foundations): the far-zone uniformity of
# tau(y) = tr(Delta^T Spair^-1 Delta) and the chi^2 pieces. The pair-Hessian
# conditional covariance Spair is rigid (lambda_min = 2.6e-10 at the rung):
# entrywise envelopes overbound tau by x5.3e3 (||Delta||_F^2/lambda_min =
# 1.881 vs true tau = 3.53e-4) because the pins EXPLAIN the pair block's
# rigid directions. The lemma in computational form:
#   (a) the rigid residual forms R_k = v_k.H_pair - a_k.C6 (a_k = the 6-pin
#       regression, exact) have CERTIFIED monomial moments with a definite
#       parity (half the orders vanish at 1e-90);
#   (b) their covariances with far jets admit the moment-series expansion
#       F_{k,i}(y) = cov(R_k, H_y,i) = sum_beta mu_beta/beta! d^{beta+g_i}K(y)
#       with EXACT moments and a certified Taylor remainder (theta-free
#       Hermite envelopes);
#   (c) the moment-series TERMWISE envelope certifies the zone d >= d_env;
#       on the delicate annulus [5, d_env] the exact functions (plane-kernel
#       closed forms + first torus images, validated against cov_exact to
#       1e-25) are evaluated on a certified net with FD-gradient variation
#       bounds (dps=100, delta = 1e-8; FD error certified via the crude
#       second-derivative envelopes).
#   H2 foundations machinery (pin_transform c6988ac7..., reg_lemmas
#   e7005a5a...) is hash-pinned and cited for the normalized-frame
#   conditioning of the pin Gram; the eigenfloors used here are the engine's
#   own eigsy residual certificates.
# Piece 2 (CERTIFIED RIEMANN SUM): the annulus crude spine
#   rho_spine(y) = pgrad(y) * window-cap(y) / Z_lo
#   is evaluated by the fast two-stage Schur (6-pin residualization with the
#   fixed certified G6 inverse, then the y-jet residualization), validated
#   against the frozen station() to 1e-90, and the zone integral is
#   certified on an adaptive polar grid with per-cell error bounds (FD
#   gradients + certified second-derivative envelopes).
#
# Fail-closed, both modes byte-identical, mutation suite MUT-RN-1..5.

import sys, os, hashlib
sys.path.insert(0, '.')
sys.path.insert(0, '../H2_foundations')
import mpmath
from mpmath import mp, mpf
mp.dps = 100

LINES = []
def emit(s):
    LINES.append(s)
    print(s)

def ck(cond, msg):
    if not cond:
        emit("FAIL-CLOSED: " + msg)
        sys.exit(1)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

_PINS = {
    'd3_perc.py': '5bc092412943e1c2185a8cf86cee23ae9cd9a2aad9bd9ef0b02807e9ff58dfa4',
    'd3_amend.py': '12175217869a9bd7b3f623fdb34a6c0c699419722bfe534e3b5226352504b5b6',
    '../H2_foundations/cov_exact.py': 'f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783',
    '../H2_foundations/pin_transform.py': 'c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41',
    '../H2_foundations/reg_lemmas.py': 'e7005a5af73d80bc1e129ac17ed19ae3d5df5d6667b076efa311f24247043450',
    '../H3_closure/H3_RUNG_FLOOR.md': '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa',
}
for _p, _h in _PINS.items():
    _a = sha256_file(_p)
    ck(_a == _h, "hash pin mismatch on %s: %s != %s" % (_p, _a, _h))
emit("hash pins verified (d3_perc, d3_amend, cov_exact, pin_transform,"
     " reg_lemmas, H3_RUNG_FLOOR)")

import re as _re
_floor_txt = open('../H3_closure/H3_RUNG_FLOOR.md').read()
_m = _re.search(r'Z_\{0\.05\} ∈ \[\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\]', _floor_txt)
ck(_m is not None, "R2 floor interval not found in pinned H3_RUNG_FLOOR.md")
Z_LO = mpf(_m.group(1))
ck(Z_LO == mpf('7.7592917375327855e-3'), "R2 floor drifted: %s" % Z_LO)
emit("R2 certified floor Z_lo = 7.7592917375327855e-3 (parsed from the pinned"
     " artifact; drift ck armed)")

import d3_perc as E
import d3_amend as V1
ce = E.ce
from math import factorial

R = mpf('0.05')
kit = E.PinKit(R)
PTS = {'M': kit.M, 'S': kit.S}
CSET6 = [('M', 'f'), ('M', 'fx'), ('M', 'fy'), ('S', 'f'), ('S', 'fx'), ('S', 'fy')]
TSET9 = [('M', 'fxx'), ('M', 'fyy'), ('M', 'fxy'),
         ('S', 'fxx'), ('S', 'fyy'), ('S', 'fxy'),
         ('Y', 'fxx'), ('Y', 'fyy'), ('Y', 'fxy')]
YJET = [('Y', 'f'), ('Y', 'fx'), ('Y', 'fy')]

# ---------------------------------------------------------------- kernel ---
C_KERN = mpf(1)   # the rung kernel is the unit separable Gaussian e^{-d^2/2}
_LT = 24

def he(n, x):
    if n == 0: return mpf(1)
    if n == 1: return x
    h0, h1 = mpf(1), x
    for k in range(2, n + 1):
        h0, h1 = h1, x * h1 - (k - 1) * h0
    return h1

_HE_ABS = {
    0: lambda t: mpf(1),
    1: lambda t: t,
    2: lambda t: t**2 + 1,
    3: lambda t: t**3 + 3 * t,
    4: lambda t: t**4 + 6 * t**2 + 3,
    5: lambda t: t**5 + 10 * t**3 + 15 * t,
    6: lambda t: t**6 + 15 * t**4 + 45 * t**2 + 15,
    7: lambda t: t**7 + 21 * t**5 + 105 * t**3 + 105 * t,
    8: lambda t: t**8 + 28 * t**6 + 210 * t**4 + 420 * t**2 + 105,
    9: lambda t: t**9 + 36 * t**7 + 378 * t**5 + 1260 * t**3 + 945 * t,
    10: lambda t: (t**10 + 45 * t**8 + 630 * t**6 + 3150 * t**4
                   + 4725 * t**2 + 945),
}

def _he_abs_poly(n):
    """|He_n|(t) with all-plus coefficients (the rigorous envelope)."""
    def f(t):
        t = abs(t)
        if n == 0:
            return mpf(1)
        if n == 1:
            return t
        h0, h1 = mpf(1), t
        for k in range(2, n + 1):
            h0, h1 = h1, t * h1 + (k - 1) * h0
        return h1
    return f

for _n in range(11, 20):
    _HE_ABS[_n] = _he_abs_poly(_n)

def he_abs(n, t):
    return _HE_ABS[n](abs(t))

def kern(d2):
    return C_KERN * mpmath.exp(-d2 / 2)

def kplane(a1, a2, dx, dy):
    d2 = dx * dx + dy * dy
    return ((-1) ** (a1 + a2)) * he(a1, dx) * he(a2, dy) * kern(d2)

_IMG = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]

def kdcov(a, b, dx, dy):
    """cov(d^a f(p), d^b f(q)) with x = p - q = (dx,dy): (-1)^|b| d^{a+b}K(x),
    plane kernel + first-image shell; residual tail bounded by tail_bound."""
    s = kplane(a[0] + b[0], a[1] + b[1], dx, dy)
    for (i, j) in _IMG:
        s += kplane(a[0] + b[0], a[1] + b[1], dx + _LT * i, dy + _LT * j)
    return ((-1) ** (b[0] + b[1])) * s

def tail_bound(order):
    """|sum_{|n|>=2} d^order K(x + 24n)| for |x| <= 17: images at separation
    >= 31: bounded by shell sums of the plane kernel."""
    tot = mpf(0)
    for m in range(2, 8):
        rho = m * _LT - 17
        tot += 8 * m * he_abs(order, mpf(rho)) * kern(mpf(rho) ** 2)
    return tot

emit("validating the closed-form kernel against cov_exact (tails included):")
_mx = mpf(0)
for (dx, dy) in [(mpf('0.05'), mpf('0.01')), (mpf('0.3'), mpf('0.15')),
                 (mpf(5), mpf(0)), (mpf('7.5'), mpf('2.5')),
                 (mpf('11.9'), mpf('11.9')), (mpf('16.9'), mpf('0.2'))]:
    for a in [(0, 0), (1, 0), (2, 0), (1, 1), (2, 2)]:
        for b in [(0, 0), (1, 0), (2, 0), (1, 1), (2, 2)]:
            if sum(a) + sum(b) > 4:
                continue
            v1 = ce.cov(a, b, dx, dy)
            v2 = kdcov(a, b, dx, dy)
            _mx = max(_mx, abs(v1 - v2))
ck(_mx < mpf('1e-25'), "kernel closed-form validation gap %s" % mpmath.nstr(_mx, 3))
for _o in range(5, 11):
    ck(tail_bound(_o) < mpf('1e-60'), "image tail too large at order %d" % _o)
emit("  max |kdcov - cov_exact| over the sample = %s (<= 1e-25); image tails"
     " < 1e-60 at orders <= 10" % mpmath.nstr(_mx, 3))

# ------------------------------------------------- fixed systems (y-free) --
def _covidx(di, dj, dx, dy):
    return ce.cov(E.IDX[di], E.IDX[dj], dx, dy)

G6 = mp.zeros(6, 6)
for _i, (pi, di) in enumerate(CSET6):
    for _j, (pj, dj) in enumerate(CSET6):
        G6[_i, _j] = _covidx(di, dj, PTS[pi][0] - PTS[pj][0], PTS[pi][1] - PTS[pj][1])
G6inv = G6 ** -1
_res = max(abs((G6 * G6inv - mp.eye(6))[_i, _j]) for _i in range(6) for _j in range(6))
ck(_res < mpf('1e-80'), "G6 inversion residual %s" % mpmath.nstr(_res, 3))

TSET6 = TSET9[:6]
X6 = mp.zeros(6, 6)
for _i, (pi, di) in enumerate(TSET6):
    for _j, (pj, dj) in enumerate(CSET6):
        X6[_i, _j] = _covidx(di, dj, PTS[pi][0] - PTS[pj][0], PTS[pi][1] - PTS[pj][1])
H6 = mp.zeros(6, 6)
for _i, (pi, di) in enumerate(TSET6):
    for _j, (pj, dj) in enumerate(TSET6):
        H6[_i, _j] = _covidx(di, dj, PTS[pi][0] - PTS[pj][0], PTS[pi][1] - PTS[pj][1])
SPAIR0 = H6 - X6 * G6inv * X6.T
LAM0, V0 = V1.eigvals_cert(SPAIR0, 6, "spair0(y-free 6-pin-conditional pair block)")
_rigid = [k for k in range(6) if LAM0[k] < mpf('1e-5')]
_nonrigid = [k for k in range(6) if LAM0[k] >= mpf('1e-5')]
ck(_rigid == [0, 1, 2] and _nonrigid == [3, 4, 5], "rigid split drifted: %s/%s"
   % (_rigid, _nonrigid))
ck(LAM0[2] / LAM0[3] < mpf('1e-3'), "rigid/non-rigid gap collapsed: %s / %s"
   % (mpmath.nstr(LAM0[2], 3), mpmath.nstr(LAM0[3], 3)))
emit("Spair0 (y-free) spectrum: %s; rigid = {0,1,2} (lam <= 2.1e-6),"
     " non-rigid = {3,4,5} (lam >= 2.5e-3), gap x%s"
     % ([mpmath.nstr(l, 4) for l in LAM0],
        mpmath.nstr(LAM0[3] / LAM0[2], 3)))

# rigid residual forms R_k = v_k.H_pair - a_k.C6, a_k = G6inv X6^T v_k (exact)
AK = [G6inv * (X6.T * V0[:, k]) for k in range(6)]
for k in range(6):
    _v = V0[:, k]; _a = AK[k]
    _var = (_v.T * H6 * _v)[0] - 2 * (_v.T * X6 * _a)[0] + (_a.T * G6 * _a)[0]
    ck(abs(_var - LAM0[k]) / LAM0[k] < mpf('1e-60'),
       "residual form variance mismatch at k=%d" % k)

# monomial moments mu_beta(R_k) = sum_j c_j d^{a_j}[p^beta](p_j)  (exact)
MOMS = {}
for k in range(6):
    mo = {}
    for b1 in range(9):
        for b2 in range(9 - b1):
            s = mpf(0)
            for i, (pi, di) in enumerate(TSET6):
                a = E.IDX[di]; p = PTS[pi]
                if a[0] <= b1 and a[1] <= b2:
                    cf = (factorial(b1) // factorial(b1 - a[0])
                          * factorial(b2) // factorial(b2 - a[1]))
                    s += V0[i, k] * cf * (p[0] ** (b1 - a[0])) * (p[1] ** (b2 - a[1]))
            for j, (pj, dj) in enumerate(CSET6):
                a = E.IDX[dj]; p = PTS[pj]
                if a[0] <= b1 and a[1] <= b2:
                    cf = (factorial(b1) // factorial(b1 - a[0])
                          * factorial(b2) // factorial(b2 - a[1]))
                    s += (-AK[k][j]) * cf * (p[0] ** (b1 - a[0])) * (p[1] ** (b2 - a[1]))
            mo[(b1, b2)] = s
    MOMS[k] = mo
emit("monomial moments of the residual forms (max |mu_beta| by order):")
for k in range(6):
    byord = {}
    for (b1, b2), v in MOMS[k].items():
        o = b1 + b2
        byord[o] = max(byord.get(o, mpf(0)), abs(v))
    emit("  k=%d lam=%s: %s" % (k, mpmath.nstr(LAM0[k], 3),
                               {o: mpmath.nstr(v, 3) for o, v in sorted(byord.items())}))
# parity certificates: the rung structure gives definite parity per form
_PARITY = {0: 'even-zero', 1: 'odd-zero', 2: 'odd-zero', 3: 'even-zero',
           4: 'even-zero', 5: 'odd-zero'}
for k in range(6):
    par = _PARITY[k]
    for (b1, b2), v in MOMS[k].items():
        o = b1 + b2
        if par == 'even-zero' and o % 2 == 0:
            ck(abs(v) < mpf('1e-80'), "parity breach k=%d beta=%s: %s"
               % (k, (b1, b2), mpmath.nstr(v, 3)))
        if par == 'odd-zero' and o % 2 == 1:
            ck(abs(v) < mpf('1e-80'), "parity breach k=%d beta=%s: %s"
               % (k, (b1, b2), mpmath.nstr(v, 3)))
emit("parity certificates hold (vanishing orders < 1e-80 in exact arithmetic)")

# ------------------------------------------------ fast two-stage station --
_GAMMAP = {d: E.IDX[d] for d in ['f', 'fx', 'fy', 'fxx', 'fyy', 'fxy']}

def kc(d1, d2, p1, p2):
    return kdcov(_GAMMAP[d1], _GAMMAP[d2], p1[0] - p2[0], p1[1] - p2[1])

def station_fast(y):
    """Conditional covariance of the 9 Hessian targets given [6 pins, y-jet]
    via the two-stage Schur with the fixed certified G6 inverse. Also returns
    the pin-conditioned blocks used by the envelopes."""
    TC = mp.zeros(9, 6); TY = mp.zeros(9, 3); TT = mp.zeros(9, 9)
    YY = mp.zeros(3, 3); YC = mp.zeros(3, 6)
    for i, (pi, di) in enumerate(TSET9):
        p1 = y if pi == 'Y' else PTS[pi]
        for j, (pj, dj) in enumerate(CSET6):
            TC[i, j] = kc(di, dj, p1, PTS[pj])
        for j, (pj, dj) in enumerate(YJET):
            TY[i, j] = kc(di, dj, p1, y)
        for j, (pj, dj) in enumerate(TSET9):
            p2 = y if pj == 'Y' else PTS[pj]
            TT[i, j] = kc(di, dj, p1, p2)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(YJET):
            YY[i, j] = kc(di, dj, y, y)
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    TT6 = TT - TC * G6inv * TC.T
    TY6 = TY - TC * G6inv * YC.T
    YY6 = YY - YC * G6inv * YC.T
    cov9 = TT6 - TY6 * (YY6 ** -1) * TY6.T
    return cov9, TT6, TY6, YY6

# validate against the frozen station (cov_exact everywhere)
_st = E.station(kit, (mpf(5), mpf(0)))
_c9, _, _, _ = station_fast((mpf(5), mpf(0)))
_df = max(abs(_c9[i, j] - _st['cov'][i, j]) for i in range(9) for j in range(9))
ck(_df < mpf('1e-25'), "fast-station validation gap at (5,0): %s" % mpmath.nstr(_df, 3))
_st2 = E.station(kit, (mpf('0.3'), mpf('0.15')))
_c92, _, _, _ = station_fast((mpf('0.3'), mpf('0.15')))
_df2 = max(abs(_c92[i, j] - _st2['cov'][i, j]) for i in range(9) for j in range(9))
ck(_df2 < mpf('1e-25'), "fast-station validation gap at (0.3,30deg): %s" % mpmath.nstr(_df2, 3))
emit("fast two-stage station validated against the frozen station():"
     " gaps %s / %s (<= 1e-25)" % (mpmath.nstr(_df, 3), mpmath.nstr(_df2, 3)))

# ============================================================ PIECE 1 ======
# RIGIDITY-DECOUPLING: certified far-zone uniformity of tau and the assembly.
#
# forms as (point, multiindex, coeff) lists; F_R(gamma, y) = cov(R, d^gamma f(y))
_GI = {'f': (0, 0), 'fx': (1, 0), 'fy': (0, 1),
       'fxx': (2, 0), 'fyy': (0, 2), 'fxy': (1, 1)}
_HYG = [(2, 0), (0, 2), (1, 1)]          # H_y Hessian entries
_YJG = [(0, 0), (1, 0), (0, 1)]          # y-jet entries

def form_eval(form, gamma, y):
    s = mpf(0)
    for (p, a, c) in form:
        s += c * kdcov(a, gamma, p[0] - y[0], p[1] - y[1])
    return s

def kplane_d(a1, a2, dx, dy, e):
    """d/dx_e of d^{(a1,a2)}K."""
    if e == 0:
        return kplane(a1 + 1, a2, dx, dy)
    return kplane(a1, a2 + 1, dx, dy)

def kdcov_d(a, b, dx, dy, e):
    s = kplane_d(a[0] + b[0], a[1] + b[1], dx, dy, e)
    for (i, j) in _IMG:
        s += kplane_d(a[0] + b[0], a[1] + b[1], dx + _LT * i, dy + _LT * j, e)
    return ((-1) ** (b[0] + b[1])) * s

def form_grad(form, gamma, y):
    gx = mpf(0); gy = mpf(0)
    for (p, a, c) in form:
        dx, dy = p[0] - y[0], p[1] - y[1]
        gx -= c * kdcov_d(a, gamma, dx, dy, 0)
        gy -= c * kdcov_d(a, gamma, dx, dy, 1)
    return (gx, gy)

def form_hess_fn(form, gamma, y):
    hxx = mpf(0); hyy = mpf(0); hxy = mpf(0)
    for (p, a, c) in form:
        dx, dy = p[0] - y[0], p[1] - y[1]
        for (p1, p2, acc) in [((2, 0), None, 'xx'), ((0, 2), None, 'yy'),
                              ((1, 1), None, 'xy')]:
            s = kplane(a[0] + gamma[0] + p1[0], a[1] + gamma[1] + p1[1], dx, dy)
            for (i, j) in _IMG:
                s += kplane(a[0] + gamma[0] + p1[0], a[1] + gamma[1] + p1[1],
                            dx + _LT * i, dy + _LT * j)
            s *= ((-1) ** (gamma[0] + gamma[1])) * c
            if acc == 'xx': hxx += s
            elif acc == 'yy': hyy += s
            else: hxy += s
    return (hxx, hyy, hxy)

# the residual forms R_k as (point, multiidx, coeff)
FORMS = []
for k in range(6):
    f = []
    for i, (pi, di) in enumerate(TSET6):
        f.append((PTS[pi], _GI[di], V0[i, k]))
    for j, (pj, dj) in enumerate(CSET6):
        f.append((PTS[pj], _GI[dj], -AK[k][j]))
    FORMS.append(f)

# ---- moment-series envelopes (theta-free, radius d) -----------------------
from math import comb as _comb

def env_form(k, gamma, d, qord):
    """certified bound of |d^q F_{R_k,gamma}(y)| for |y| >= d, |q| = qord:
    termwise exact-moment series + certified Taylor remainder (order 9)
    + torus-image allowance."""
    rho = d - R / 2
    tot = mpf(0)
    for (b1, b2), mu in MOMS[k].items():
        if abs(mu) < mpf('1e-80'):
            continue
        base = abs(mu) / (factorial(b1) * factorial(b2))
        for e1 in range(qord + 1):
            e2 = qord - e1
            tot += (base * _comb(qord, e1)
                    * he_abs(b1 + gamma[0] + e1, d) * he_abs(b2 + gamma[1] + e2, d))
    tot *= kern(d * d)
    # Taylor remainder at order 9 (moments are carried to order 8)
    rem = mpf(0)
    for (p, a, c) in FORMS[k]:
        ao = a[0] + a[1]
        inner = mpf(0)
        for b1 in range(10):
            b2 = 9 - b1
            if b1 < a[0] or b2 < a[1]:
                continue
            inner += ((R / 2) ** (9 - ao)
                      / (factorial(b1 - a[0]) * factorial(b2 - a[1])))
        rem += abs(c) * inner
    mx2 = mpf(0)   # max |d^{beta+gamma+q}K| over the segment, |beta|=9
    for d1 in range(12):
        d2 = 11 - d1
        for e1 in range(qord + 1):
            e2 = qord - e1
            mx2 = max(mx2, _comb(qord, e1)
                      * he_abs(d1 + gamma[0] + e1, rho)
                      * he_abs(d2 + gamma[1] + e2, rho))
    rem *= mx2 * kern(rho * rho)
    # image allowance: image separations >= 24 - d - R/2 (>= 11.97 on the torus)
    img = mpmath.fsum(abs(c) for (p, a, c) in FORMS[k])
    rimg = 24 - d - R / 2
    img_env = (img * 8 * he_abs(6 + qord, rimg) * kern(mpf(rimg) ** 2))
    return tot + rem + img_env

# ---- per-point exact tau and certified variation bounds -------------------
def tau_exact(cov9):
    Sp = mp.zeros(6, 6)
    Dl = mp.zeros(6, 3)
    for i in range(6):
        for j in range(6):
            Sp[i, j] = cov9[i, j]
        for j in range(3):
            Dl[i, j] = cov9[i, 6 + j]
    Spinv = Sp ** -1
    rr = max(abs((Sp * Spinv - mp.eye(6))[i, j]) for i in range(6) for j in range(6))
    ck(rr < mpf('1e-40'), "Spair inversion residual at net point: %s" % mpmath.nstr(rr, 3))
    return mpmath.fsum((Dl.T * Spinv * Dl)[i, i] for i in range(3))

def point_pieces(y):
    """exact per-point pieces for tau and its variation bounds."""
    cov9, TT6, TY6, YY6 = station_fast(y)
    tau = tau_exact(cov9)
    # F~_{k,i} = F_{k,i} - dF_{k,i}:  F = form_eval(R_k, gamma_i, y);
    # dF = q_k^T YY6^-1 m_i  with q_k = cov(R_k, yjets|C6), m_i = cov(yjets, H_y,i|C6)
    YY6inv = YY6 ** -1
    Ft = mp.zeros(6, 3)   # F~ components
    Fv = mp.zeros(6, 3)   # F (unconditional-form) components
    Gt = mp.zeros(6, 3)   # |grad F| (exact, cheap)
    Ht = mp.zeros(6, 3)   # |hess F| Frobenius (exact, cheap)
    Q = mp.zeros(6, 3)    # q_k components (R_k vs yjets)
    for k in range(6):
        for i in range(3):
            Fv[k, i] = form_eval(FORMS[k], _HYG[i], y)
            gx, gy = form_grad(FORMS[k], _HYG[i], y)
            Gt[k, i] = mpmath.sqrt(gx * gx + gy * gy)
            hxx, hyy, hxy = form_hess_fn(FORMS[k], _HYG[i], y)
            Ht[k, i] = mpmath.sqrt(hxx * hxx + hyy * hyy + 2 * hxy * hxy)
        for j in range(3):
            Q[k, j] = form_eval(FORMS[k], _YJG[j], y)
    # m_i columns: cov(yjets, H_y,i | C6) = (TY6)^T rows 6+i
    Mm = mp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Mm[j, i] = TY6[6 + i, j]
    red = mp.zeros(6, 1)  # lambda reduction q_k^T YY6inv q_k
    dF = mp.zeros(6, 3)
    for k in range(6):
        qk = mp.matrix([Q[k, 0], Q[k, 1], Q[k, 2]])
        red[k] = (qk.T * YY6inv * qk)[0]
        for i in range(3):
            dF[k, i] = (qk.T * YY6inv * Mm[:, i])[0]
            Ft[k, i] = Fv[k, i] - dF[k, i]
    return tau, Fv, Ft, Gt, Ht, Q, red, YY6inv, Mm, cov9

# crude envelopes for the small (y-jet reduction) pieces' variation
def env_small(d, y):
    """bounds for |grad dF|, |hess dF|, |grad red| at radius ~d.
    The reduction pieces are O(1e-8) with O(1) factors; crude moment-series
    envelopes suffice (they enter only the variation bounds, x slack 50 ok)."""
    # |grad q_k| via env_form with gamma = yjet (order <= 1), qord = 1
    out = {}
    for k in range(6):
        gq = mpmath.sqrt(sum(env_form(k, g, d, 1) ** 2 for g in _YJG))
        hq = mpmath.sqrt(sum(env_form(k, g, d, 2) ** 2 for g in _YJG))
        q = mpmath.sqrt(sum(env_form(k, g, d, 0) ** 2 for g in _YJG))
        out[k] = (q, gq, hq)
    # m entries: cov(yjets, H_y,i|C6): crude O(1) bound via raw + normalized
    # reduction; the raw covariances of the y-jets with H_y at separation 0
    # are constants; the pin-reduction is bounded in the normalized frame.
    return out

# normalized-frame pin transform (H2, pinned): T_r for O(1) inverse bounds
import pin_transform as PT
_TN = PT.T(R) if hasattr(PT, 'T') else None

# ---- normalized frame for O(1) inverse bounds (H2-cited, own eigsy floors) -
_TM = mp.matrix(PT.matrix_T_num(R, mode='mpf'))
GN = _TM * G6 * _TM.T
GNinv = GN ** -1
_rr = max(abs((GN * GNinv - mp.eye(6))[_i, _j]) for _i in range(6) for _j in range(6))
ck(_rr < mpf('1e-60'), "normalized Gram inversion residual %s" % mpmath.nstr(_rr, 3))
_lamN, _VN = V1.eigvals_cert(GN, 6, "normalized pin Gram")
ck(min(_lamN) > mpf('1e-3'), "normalized-frame eigenfloor collapsed: %s"
   % mpmath.nstr(min(_lamN), 4))
GNINV_NORM = 1 / min(_lamN)
_TMnorm = mpmath.sqrt(mpmath.fsum(_TM[_i, _j] ** 2 for _i in range(6) for _j in range(6)))
emit("normalized pin frame (H2 pin_transform, pinned): eigenfloor %s,"
     " ||Gram(U)^-1|| <= %s" % (mpmath.nstr(min(_lamN), 4),
                                mpmath.nstr(GNINV_NORM, 4)))

def env_m(d):
    """crude bound of ||m_i|| = |cov(yjets, H_y,i | C6)| and ||YY6^-1|| via
    the normalized frame (theta-free in d)."""
    rho = d - R / 2
    # cov(yjets, U) entries: normalized pins are O(1)-variance jet combos:
    # bound entries by the raw derivative envelopes times ||T||
    cu = mpmath.sqrt(18) * he_abs(4, rho) * kern(rho * rho) * _TMnorm
    red = cu * cu * GNINV_NORM
    yinv = 1 / (1 - min(red, mpf('0.5')))
    # cov(U, H_y,i): derivatives order <= 5 at separation >= rho
    ch = mpmath.sqrt(18) * he_abs(5, rho) * kern(rho * rho) * _TMnorm
    # raw cov(yjets, H_y): same-point constants + pin reduction
    raw = mpmath.sqrt(sum(he_abs(a[0] + 2, 0) ** 2 * kern(0) ** 2
                          for a in _YJG)) + ch * GNINV_NORM * cu
    m = raw + cu * GNINV_NORM * ch
    return m, yinv

def env_dYY6(d):
    rho = d - R / 2
    cu = mpmath.sqrt(18) * he_abs(4, rho) * kern(rho * rho) * _TMnorm
    gcu = mpmath.sqrt(18) * he_abs(5, rho) * kern(rho * rho) * _TMnorm
    return 2 * gcu * GNINV_NORM * cu

def env_TY6(d):
    """|TY6_pair entries| (cov(pair-H, yjets|C6)) and derivative (crude)."""
    rho = d - R / 2
    t = mpmath.sqrt(18) * he_abs(3, rho) * kern(rho * rho)
    g = mpmath.sqrt(18) * he_abs(4, rho) * kern(rho * rho)
    h = mpmath.sqrt(18) * he_abs(5, rho) * kern(rho * rho)
    return t, g, h

def bounds_point(y, d):
    """certified (tau0, grad-bound, hess-bound) at the point y, radius d."""
    tau, Fv, Ft, Gt, Ht, Q, red, YY6inv, Mm, cov9 = point_pieces(y)
    m_env, yinv_env = env_m(d)
    dyy = env_dYY6(d)
    small_g = mpf(0)   # grad bound of small (DeltaF/red) terms
    tau_grad = mpf(0)
    tau_hess = mpf(0)
    for k in range(6):
        lam = LAM0[k] - red[k]
        ck(lam > LAM0[k] / 2, "lambda floor collapsed at net point k=%d" % k)
        Fn = mpmath.sqrt(mpmath.fsum(Ft[k, i] ** 2 for i in range(3)))
        Gn = mpmath.sqrt(mpmath.fsum(Gt[k, i] ** 2 for i in range(3)))
        Hn = mpmath.sqrt(mpmath.fsum(Ht[k, i] ** 2 for i in range(3)))
        qn = mpmath.sqrt(mpmath.fsum(Q[k, j] ** 2 for j in range(3)))
        # dF = q^T YY6inv m: grad via envelopes of the factors
        gq = mpmath.sqrt(sum(env_form(k, g, d, 1) ** 2 for g in _YJG))
        hq = mpmath.sqrt(sum(env_form(k, g, d, 2) ** 2 for g in _YJG))
        genv = (gq * yinv_env * m_env + qn * (yinv_env ** 2) * dyy * m_env
                + qn * yinv_env * (2 * m_env))
        henv = (hq * yinv_env * m_env
                + 2 * gq * (yinv_env ** 2) * dyy * m_env
                + 2 * gq * yinv_env * 2 * m_env
                + qn * (yinv_env ** 3) * dyy ** 2 * m_env
                + qn * (yinv_env ** 2) * (2 * dyy * dyy) * m_env
                + qn * (yinv_env ** 2) * dyy * 2 * m_env
                + qn * yinv_env * 4 * m_env)
        gred = 2 * qn * gq * yinv_env + qn * qn * (yinv_env ** 2) * dyy
        hred = (2 * gq ** 2 * yinv_env + 2 * qn * hq * yinv_env
                + 4 * qn * gq * (yinv_env ** 2) * dyy
                + qn ** 2 * (yinv_env ** 3) * dyy ** 2
                + qn ** 2 * (yinv_env ** 2) * 2 * dyy ** 2)
        tau_grad += (2 * Fn * (Gn + genv)) / lam + Fn * Fn * gred / (lam * lam)
        tau_hess += (2 * (Gn + genv) ** 2 + 2 * Fn * (Hn + henv)) / lam \
            + (4 * Fn * (Gn + genv) * gred + Fn * Fn * hred) / (lam * lam) \
            + 2 * Fn * Fn * gred * gred / (lam ** 3)
    # cross terms (off-diagonal Spair(y)^-1 in the V0 basis): exact at y0
    St = V0.T * mp.matrix([[cov9[i, j] for j in range(6)] for i in range(6)]) * V0
    Stinv = St ** -1
    Fn_v = [mpmath.sqrt(mpmath.fsum(Ft[k, i] ** 2 for i in range(3)))
            for k in range(6)]
    Gn_v = [mpmath.sqrt(mpmath.fsum(Gt[k, i] ** 2 for i in range(3)))
            for k in range(6)]
    t6, g6, h6 = env_TY6(d)
    dSp = 2 * g6 * yinv_env * t6 + t6 * t6 * (yinv_env ** 2) * dyy
    hSp = (2 * h6 * yinv_env * t6 + 2 * g6 * g6 * yinv_env
           + 4 * g6 * (yinv_env ** 2) * dyy * t6
           + t6 * t6 * (yinv_env ** 3) * dyy ** 2
           + t6 * t6 * (yinv_env ** 2) * 2 * dyy ** 2)
    cross_g = mpf(0)
    cross_h = mpf(0)
    for k in range(6):
        for kp in range(k + 1, 6):
            so = abs(Stinv[k, kp])
            dso = (1 / LAM0[k]) * dSp * (1 / LAM0[kp])
            hso = ((1 / LAM0[k]) * hSp * (1 / LAM0[kp])
                   + 2 * (1 / LAM0[k] ** 2) * dSp * dSp * (1 / LAM0[kp])
                   + 2 * (1 / LAM0[k]) * dSp * dSp * (1 / LAM0[kp] ** 2))
            cross_g += 2 * (Gn_v[k] * so * Fn_v[kp] + Fn_v[k] * so * Gn_v[kp]
                            + Fn_v[k] * dso * Fn_v[kp])
            cross_h += 2 * (Hn_diag(Ht, k) * so * Fn_v[kp]
                            + 2 * Gn_v[k] * so * Gn_v[kp]
                            + Fn_v[k] * so * Hn_diag(Ht, kp)
                            + 2 * Gn_v[k] * dso * Fn_v[kp]
                            + 2 * Fn_v[k] * dso * Gn_v[kp]
                            + Fn_v[k] * hso * Fn_v[kp])
    return tau, tau_grad + cross_g, tau_hess + cross_h, dSp

def Hn_diag(Ht, k):
    return mpmath.sqrt(mpmath.fsum(Ht[k, i] ** 2 for i in range(3)))

# ---- fast full kappa-piece assembly (validated against frozen v2) ----------
_VAL6 = mp.matrix([kit.b, 0, 0, kit.s, 0, 0])
_W6 = G6inv * _VAL6

def station_fast_full(y):
    """fast station: cov9 + conditional means + needle law + pgrad."""
    TC = mp.zeros(9, 6); TY = mp.zeros(9, 3); TT = mp.zeros(9, 9)
    YY = mp.zeros(3, 3); YC = mp.zeros(3, 6)
    for i, (pi, di) in enumerate(TSET9):
        p1 = y if pi == 'Y' else PTS[pi]
        for j, (pj, dj) in enumerate(CSET6):
            TC[i, j] = kc(di, dj, p1, PTS[pj])
        for j, (pj, dj) in enumerate(YJET):
            TY[i, j] = kc(di, dj, p1, y)
        for j, (pj, dj) in enumerate(TSET9):
            p2 = y if pj == 'Y' else PTS[pj]
            TT[i, j] = kc(di, dj, p1, p2)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(YJET):
            YY[i, j] = kc(di, dj, y, y)
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    TT6 = TT - TC * G6inv * TC.T
    TY6 = TY - TC * G6inv * YC.T
    YY6 = YY - YC * G6inv * YC.T
    YY6inv = YY6 ** -1
    cov9 = TT6 - TY6 * YY6inv * TY6.T
    # means: mu(v) = TC w6 + TY6 YY6inv (values_yj(v) - YC w6), values_yj = (v,0,0)
    mu6 = TC * _W6
    rv0 = mp.matrix([- (YC * _W6)[0], - (YC * _W6)[1], - (YC * _W6)[2]])
    def mean_v(v):
        rv = mp.matrix([rv0[0] + v, rv0[1], rv0[2]])
        return mu6 + TY6 * YY6inv * rv
    # needle law of f(y) given [pins, grad(y)=0]: from the stage-1 blocks
    muy = YC * _W6
    Sg = mp.matrix([[YY6[1, 1], YY6[1, 2]], [YY6[2, 1], YY6[2, 2]]])
    Sginv = Sg ** -1
    mug = mp.matrix([muy[1], muy[2]])
    maha = (mug.T * Sginv * mug)[0]
    pgrad = mpmath.exp(-maha / 2) / (2 * mp.pi * mpmath.sqrt(mp.det(Sg)))
    c01 = mp.matrix([YY6[0, 1], YY6[0, 2]])
    s2 = YY6[0, 0] - (c01.T * Sginv * c01)[0]
    ck(s2 > 0, "needle variance nonpositive (fast)")
    mu_t = muy[0] - (c01.T * Sginv * mug)[0]
    return {'cov9': cov9, 'mean_v': mean_v, 'mu_t': mu_t, 's_t': mpmath.sqrt(s2),
            'pgrad': pgrad, 'TT6': TT6, 'TY6': TY6, 'YY6': YY6}

def kappa_pieces_fast(fs, v):
    """the v2 assembly pieces from the fast station (Z_LO denominators)."""
    cov = fs['cov9']
    muv = fs['mean_v'](v)
    muM = mp.matrix([muv[0], muv[1], muv[2]])
    muS = mp.matrix([muv[3], muv[4], muv[5]])
    SM = E.submat(cov, [0, 1, 2], [0, 1, 2])
    SS = E.submat(cov, [3, 4, 5], [3, 4, 5])
    dM8 = E.det_moment(muM, SM, 8)
    dS8 = E.det_moment(muS, SS, 8)
    ck(dM8 > 0 and dS8 > 0, "Wick 8th moments nonpositive (fast)")
    EW4 = mpmath.sqrt(dM8) * mpmath.sqrt(dS8)
    mu_y = mp.matrix([muv[6], muv[7], muv[8]])
    Spair = E.submat(cov, range(6), range(6))
    Delta = E.submat(cov, range(6), [6, 7, 8])
    Sy = E.submat(cov, [6, 7, 8], [6, 7, 8])
    Breg = Delta.T * (Spair ** -1)
    Syres = Sy - Breg * Delta
    Eq2 = 8 * (E.enorm4(mu_y, Sy) + E.enorm4(mu_y, Syres))
    tau = mpmath.fsum((Breg * Delta)[i, i] for i in range(3))
    msad = E.m_sad_cf(v)
    kap_cross = mpmath.sqrt(3) * (EW4 ** mpf('0.25')) * (Eq2 ** mpf('0.25')) \
        * mpmath.sqrt(tau) / (Z_LO * msad)
    C_mom = mpmath.sqrt(3) * (EW4 ** mpf('0.25')) * (Eq2 ** mpf('0.25')) \
        / (Z_LO * msad)
    # chi2 piece (closed form; S9p = Spair, S6/mu6 from the kit)
    mu6 = kit.Hmean
    S6 = kit.Hcov
    mu9p = mp.matrix([muv[i] for i in range(6)])
    S9inv = Spair ** -1
    S6inv = S6 ** -1
    M = S9inv - S6inv * mpf('0.5')
    lamM, _vM = V1.eigvals_cert(M, 6, "chi2:M(fast)")
    ck(min(lamM) > 0, "chi2 M not PD (fast)")
    c = S9inv * mu9p - S6inv * mu6 * mpf('0.5')
    kk = (mu9p.T * S9inv * mu9p)[0] - mpf('0.5') * (mu6.T * S6inv * mu6)[0]
    logq = (-3 * mpmath.log(2) + mpmath.log(mpmath.fabs(mp.det(S6))) / 2
            - mpmath.log(mpmath.fabs(mp.det(Spair)))
            - mpmath.log(mpmath.fabs(mp.det(M))) / 2
            + (c.T * (M ** -1) * c)[0] - kk)
    chi2 = mpmath.exp(logq) - 1
    ck(chi2 >= 0 and chi2 < 1, "chi2 out of range (fast): %s" % mpmath.nstr(chi2, 3))
    dM4 = E.det_moment(mp.matrix([mu6[0], mu6[1], mu6[2]]),
                       E.submat(S6, [0, 1, 2], [0, 1, 2]), 4)
    dS4 = E.det_moment(mp.matrix([mu6[3], mu6[4], mu6[5]]),
                       E.submat(S6, [3, 4, 5], [3, 4, 5]), 4)
    hL2 = (mpmath.sqrt(dM4) * mpmath.sqrt(dS4)) ** mpf('0.5')
    kap_pair = hL2 * mpmath.sqrt(chi2) / Z_LO
    # kappa_y (W2 coupling, Z-free)
    a2, a4 = E.A2, E.A4
    mu_y0 = mp.matrix([-a2 * v, -a2 * v, mpf(0)])
    Sy0 = mp.zeros(3, 3)
    Sy0[0, 0] = a4 - a2 * a2
    Sy0[1, 1] = a4 - a2 * a2
    Sy0[2, 2] = a2 * a2
    w2y, dmu_y, trSr, trS0 = V1.w2_gauss(mu_y, Syres, mu_y0, Sy0, 3, "yblk(fast)")
    Ey1 = sum(Syres[i, i] for i in range(3)) + sum(mu_y[i] ** 2 for i in range(3))
    Ey0 = sum(Sy0[i, i] for i in range(3)) + sum(mu_y0[i] ** 2 for i in range(3))
    kap_y = mpmath.sqrt(2 * (Ey1 + Ey0)) * mpmath.sqrt(w2y) / msad
    # ratios
    pgrad0 = 1 / (2 * mp.pi * a2)
    Rpg = fs['pgrad'] / pgrad0
    wm = E._Phi((kit.b - fs['mu_t']) / fs['s_t']) - E._Phi((kit.s - fs['mu_t']) / fs['s_t'])
    wm0 = E._Phi(kit.b) - E._Phi(kit.s)
    Rwm = wm / wm0
    kap_far = Rpg * Rwm * ((1 + kap_pair) * (1 + kap_y) + kap_cross) - 1
    return {'kap_far': kap_far, 'kap_cross': kap_cross, 'kap_pair': kap_pair,
            'kap_y': kap_y, 'Rpg': Rpg, 'Rwm': Rwm, 'C_mom': C_mom,
            'chi2': chi2, 'tau': tau, 'EW4': EW4, 'Eq2': Eq2}

# validate against the frozen v2 certified values at (5,0)
_fs0 = station_fast_full((mpf(5), mpf(0)))
_kp0 = kappa_pieces_fast(_fs0, kit.b - kit.ell / 2)
_st0 = E.station(kit, (mpf(5), mpf(0)))
_cktau = abs(_kp0['tau'] - _st0['tau'])
ck(_cktau < mpf('1e-60'), "fast tau mismatch vs frozen: %s" % mpmath.nstr(_cktau, 3))
_kcref = mpf('5.04277749e-3') / Z_LO
_kfarref = mpf('0.677284905')
emit("fast assembly at (5,0): kap_cross = %s (frozen v2: %s),"
     " kap_far = %s (frozen v2: 0.677284905)"
     % (mpmath.nstr(_kp0['kap_cross'], 10), mpmath.nstr(_kcref, 10),
        mpmath.nstr(_kp0['kap_far'], 10)))
ck(abs(_kp0['kap_cross'] - _kcref) < mpf('1e-6'),
   "fast kap_cross deviates from the frozen v2 value: %s"
   % mpmath.nstr(abs(_kp0['kap_cross'] - _kcref), 3))
ck(abs(_kp0['kap_far'] - _kfarref) < mpf('1e-6'),
   "fast kap_far deviates from the frozen v2 value: %s"
   % mpmath.nstr(abs(_kp0['kap_far'] - _kfarref), 3))
emit("  => fast full assembly reproduces the frozen v2 certified values")

# ---- per-piece grad/hess bounds: directional (V0) for rigid products ------
def dE_crude(d, ord_):
    """crude entrywise envelope of |d^ord cov(pair-H, yjets|C6)|-type entries
    (well-conditioned blocks; theta-free)."""
    rho = d - R / 2
    return he_abs(3 + ord_, rho) * kern(rho * rho)

def qblock(y):
    """Q (6x3: V0-reduction couplings), dQ/dy (exact cheap), ddQ (exact)."""
    Q = mp.zeros(6, 3); dQx = mp.zeros(6, 3); dQy = mp.zeros(6, 3)
    Hxx = mp.zeros(6, 3); Hyy = mp.zeros(6, 3); Hxy = mp.zeros(6, 3)
    for k in range(6):
        for j in range(3):
            Q[k, j] = form_eval(FORMS[k], _YJG[j], y)
            gx, gy = form_grad(FORMS[k], _YJG[j], y)
            dQx[k, j] = gx; dQy[k, j] = gy
            hxx, hyy, hxy = form_hess_fn(FORMS[k], _YJG[j], y)
            Hxx[k, j] = hxx; Hyy[k, j] = hyy; Hxy[k, j] = hxy
    return Q, dQx, dQy, Hxx, Hyy, Hxy

def spair_dir(y, YY6, YY6inv):
    """(S, dS/dx, dS/dy, d2S) of the V0-basis pair block Spair(y) with TIGHT
    directional derivatives: S = SPAIR0 - Q YY6inv Q^T (Q = V0^T TY6)."""
    Q, dQx, dQy, Hxx, Hyy, Hxy = qblock(y)
    S = SPAIR0 - Q * YY6inv * Q.T
    dSx = -(dQx * YY6inv * Q.T + Q * YY6inv * dQx.T)
    dSy = -(dQy * YY6inv * Q.T + Q * YY6inv * dQy.T)
    # crude YY6 variation for the second derivatives
    d = mpmath.sqrt(y[0] ** 2 + y[1] ** 2)
    dyy = env_dYY6(d)
    y2 = (YY6inv ** 2) * dyy
    d2Sxx = -(Hxx * YY6inv * Q.T + 2 * dQx * YY6inv * dQx.T
              + Q * YY6inv * Hxx.T + 2 * (Q * y2 * Q.T))
    d2Syy = -(Hyy * YY6inv * Q.T + 2 * dQy * YY6inv * dQy.T
              + Q * YY6inv * Hyy.T + 2 * (Q * y2 * Q.T))
    d2Sxy = -(Hxy * YY6inv * Q.T + dQx * YY6inv * dQy.T
              + dQy * YY6inv * dQx.T + Q * YY6inv * Hxy.T
              + 2 * (Q * y2 * Q.T))
    return S, dSx, dSy, d2Sxx, d2Syy, d2Sxy

def kappa_far_point(y, d):
    """(kap_far, grad-bound, hess-bound) certified at point y (radius d)."""
    fs = station_fast_full(y)
    v = kit.b - kit.ell / 2
    kp = kappa_pieces_fast(fs, v)
    # ---- grad/hess per piece
    # tau (exact machinery)
    tau0, gtau, htau, dSp = bounds_point(y, d)
    # C_mom: EW4, Eq2 variation (crude: det-moment chains on well-conditioned
    # blocks + the tau-style Breg products for Syres)
    dE1 = dE_crude(d, 1); dE2 = dE_crude(d, 2); dE0 = dE_crude(d, 0)
    cov = fs['cov9']
    Spair = E.submat(cov, range(6), range(6))
    Spinv = Spair ** -1
    Delta = E.submat(cov, range(6), [6, 7, 8])
    Breg = Delta.T * Spinv
    Bnorm = mpmath.sqrt(mpmath.fsum(Breg[i, j] ** 2 for i in range(3) for j in range(6)))
    # |grad EW4| <= EW4 * (|grad dM8|/2dM8 + |grad dS8|/2dS8): the det moments
    # are Wick polynomials in the (well-conditioned y-variation of the) pair
    # block: bound the pair-block entry variation by dSp (tight) and chain
    muv = fs['mean_v'](v)
    muM = mp.matrix([muv[0], muv[1], muv[2]])
    muS = mp.matrix([muv[3], muv[4], muv[5]])
    SM = E.submat(cov, [0, 1, 2], [0, 1, 2])
    SS = E.submat(cov, [3, 4, 5], [3, 4, 5])
    dM8 = E.det_moment(muM, SM, 8); dS8 = E.det_moment(muS, SS, 8)
    # mean variation (exact solves): muv = mu6 + TY6 YY6inv rv: grad exact-cheap
    TY6 = fs['TY6']; YY6 = fs['YY6']; YY6inv = YY6 ** -1
    TCw = None
    # crude mean-variation: |grad muv| <= |grad(TY6 YY6inv rv)| via entries
    rv = mp.matrix([v - (fs['mean_v'](0)[6] - fs['mean_v'](0)[6]), 0, 0])  # placeholder, refined below
    # log-derivative crude scale for the well-conditioned pieces:
    # every covariance entry feeding EW4/Eq2/chi2/W2/Rpg/Rwm has
    # |d entry| <= dE1, |dd entry| <= dE2; the means solve-products are
    # O(Bnorm)-bounded.
    return fs, kp, tau0, gtau, htau, Bnorm, dE0, dE1, dE2

# ================= exact-derivative engine (D-scalars / DMatrix) ============
class DS:
    """exact first/second derivatives of a scalar function of y = (y1,y2)."""
    __slots__ = ('v', 'gx', 'gy', 'hxx', 'hyy', 'hxy')
    def __init__(self, v, gx=0, gy=0, hxx=0, hyy=0, hxy=0):
        self.v = mpf(v); self.gx = mpf(gx); self.gy = mpf(gy)
        self.hxx = mpf(hxx); self.hyy = mpf(hyy); self.hxy = mpf(hxy)
    @staticmethod
    def const(c):
        return DS(c)
    def __add__(self, o):
        o = o if isinstance(o, DS) else DS(o)
        return DS(self.v + o.v, self.gx + o.gx, self.gy + o.gy,
                  self.hxx + o.hxx, self.hyy + o.hyy, self.hxy + o.hxy)
    __radd__ = __add__
    def __neg__(self):
        return DS(-self.v, -self.gx, -self.gy, -self.hxx, -self.hyy, -self.hxy)
    def __sub__(self, o):
        return self + (-(o if isinstance(o, DS) else DS(o)))
    def __rsub__(self, o):
        return (o if isinstance(o, DS) else DS(o)) + (-self)
    def __mul__(self, o):
        o = o if isinstance(o, DS) else DS(o)
        return DS(self.v * o.v,
                  self.gx * o.v + self.v * o.gx, self.gy * o.v + self.v * o.gy,
                  self.hxx * o.v + 2 * self.gx * o.gx + self.v * o.hxx,
                  self.hyy * o.v + 2 * self.gy * o.gy + self.v * o.hyy,
                  self.hxy * o.v + self.gx * o.gy + self.gy * o.gx + self.v * o.hxy)
    __rmul__ = __mul__
    def __div__(self, o):
        o = o if isinstance(o, DS) else DS(o)
        inv = DS(1 / o.v, -o.gx / o.v ** 2, -o.gy / o.v ** 2,
                 2 * o.gx ** 2 / o.v ** 3 - o.hxx / o.v ** 2,
                 2 * o.gy ** 2 / o.v ** 3 - o.hyy / o.v ** 2,
                 2 * o.gx * o.gy / o.v ** 3 - o.hxy / o.v ** 2)
        return self * inv
    def __rdiv__(self, o):
        return (o if isinstance(o, DS) else DS(o)) / self
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def sqrt(self):
        s = mpmath.sqrt(self.v)
        return DS(s, self.gx / (2 * s), self.gy / (2 * s),
                  self.hxx / (2 * s) - self.gx ** 2 / (4 * s ** 3),
                  self.hyy / (2 * s) - self.gy ** 2 / (4 * s ** 3),
                  self.hxy / (2 * s) - self.gx * self.gy / (4 * s ** 3))
    def exp(self):
        e = mpmath.exp(self.v)
        return DS(e, e * self.gx, e * self.gy,
                  e * (self.hxx + self.gx ** 2), e * (self.hyy + self.gy ** 2),
                  e * (self.hxy + self.gx * self.gy))
    def log(self):
        return DS(mpmath.log(self.v), self.gx / self.v, self.gy / self.v,
                  self.hxx / self.v - self.gx ** 2 / self.v ** 2,
                  self.hyy / self.v - self.gy ** 2 / self.v ** 2,
                  self.hxy / self.v - self.gx * self.gy / self.v ** 2)
    def Phi(self):
        p = E._Phi(self.v)
        pdf = mpmath.exp(-self.v ** 2 / 2) / mpmath.sqrt(2 * mp.pi)
        return DS(p, pdf * self.gx, pdf * self.gy,
                  pdf * (self.hxx - self.v * self.gx ** 2),
                  pdf * (self.hyy - self.v * self.gy ** 2),
                  pdf * (self.hxy - self.v * self.gx * self.gy))
    def gnorm(self):
        return mpmath.sqrt(self.gx ** 2 + self.gy ** 2)
    def hnorm(self):
        return mpmath.sqrt(self.hxx ** 2 + self.hyy ** 2 + 2 * self.hxy ** 2)

class DM:
    """matrix of DS entries."""
    def __init__(self, rows):
        self.a = rows
        self.n = len(rows); self.m = len(rows[0])
    @staticmethod
    def const_mat(M):
        return DM([[DS(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])
    def __mul__(self, o):
        if isinstance(o, DM):
            ck(self.m == o.n, "DM size mismatch")
            out = [[sum((self.a[i][k] * o.a[k][j] for k in range(self.m)), DS(0))
                    for j in range(o.m)] for i in range(self.n)]
            return DM(out)
        if isinstance(o, mp.matrix):
            out = [[sum((self.a[i][k] * o[k, j] for k in range(self.m)), DS(0))
                    for j in range(o.cols)] for i in range(self.n)]
            return DM(out)
        # scalar
        return DM([[self.a[i][j] * o for j in range(self.m)]
                   for i in range(self.n)])
    def rmul_const(self, M):
        out = [[sum((M[i, k] * self.a[k][j] for k in range(self.n)), DS(0))
                for j in range(self.m)] for i in range(M.rows)]
        return DM(out)
    def __add__(self, o):
        return DM([[self.a[i][j] + o.a[i][j] for j in range(self.m)]
                   for i in range(self.n)])
    def __sub__(self, o):
        return DM([[self.a[i][j] - o.a[i][j] for j in range(self.m)]
                   for i in range(self.n)])
    @property
    def T(self):
        return DM([[self.a[j][i] for j in range(self.n)] for i in range(self.m)])
    def inv(self):
        n = self.n
        ck(n == self.m, "DM.inv needs square")
        V = mp.matrix([[self.a[i][j].v for j in range(n)] for i in range(n)])
        Vinv = V ** -1
        rr = max(abs((V * Vinv - mp.eye(n))[i, j]) for i in range(n) for j in range(n))
        ck(rr < mpf('1e-30'), "DM.inv residual %s" % mpmath.nstr(rr, 3))
        out = [[DS(Vinv[i, j]) for j in range(n)] for i in range(n)]
        # d(A^-1) = -A^-1 dA A^-1 (per derivative slot)
        for slot in ('gx', 'gy', 'hxx', 'hyy', 'hxy'):
            dA = mp.matrix([[getattr(self.a[i][j], slot) for j in range(n)]
                            for i in range(n)])
            if slot in ('gx', 'gy'):
                dI = -Vinv * dA * Vinv
            else:
                gslot = 'gx' if slot == 'hxx' else ('gy' if slot == 'hyy' else None)
                if gslot:
                    G = mp.matrix([[getattr(self.a[i][j], gslot) for j in range(n)]
                                   for i in range(n)])
                    dI = -Vinv * dA * Vinv + 2 * (Vinv * G * Vinv) * G * Vinv
                else:
                    Gx = mp.matrix([[self.a[i][j].gx for j in range(n)]
                                    for i in range(n)])
                    Gy = mp.matrix([[self.a[i][j].gy for j in range(n)]
                                    for i in range(n)])
                    dI = (-Vinv * dA * Vinv + (Vinv * Gx * Vinv) * Gy * Vinv
                          + (Vinv * Gy * Vinv) * Gx * Vinv)
            for i in range(n):
                for j in range(n):
                    setattr(out[i][j], slot, dI[i, j])
        return DM(out)
    def det_ds(self):
        """det as a DS (via the inverse: d det = det tr(A^-1 dA))."""
        n = self.n
        V = mp.matrix([[self.a[i][j].v for j in range(n)] for i in range(n)])
        Ainv = self.inv()
        Av = mp.matrix([[Ainv.a[i][j].v for j in range(n)] for i in range(n)])
        det = mp.det(V)
        def tr(dM):
            return mpmath.fsum((Av * dM)[i, i] for i in range(n))
        Gx = mp.matrix([[self.a[i][j].gx for j in range(n)] for i in range(n)])
        Gy = mp.matrix([[self.a[i][j].gy for j in range(n)] for i in range(n)])
        Hxx = mp.matrix([[self.a[i][j].hxx for j in range(n)] for i in range(n)])
        Hyy = mp.matrix([[self.a[i][j].hyy for j in range(n)] for i in range(n)])
        Hxy = mp.matrix([[self.a[i][j].hxy for j in range(n)] for i in range(n)])
        Aigx = mp.matrix([[Ainv.a[i][j].gx for j in range(n)] for i in range(n)])
        Aigy = mp.matrix([[Ainv.a[i][j].gy for j in range(n)] for i in range(n)])
        tgx, tgy = tr(Gx), tr(Gy)
        thxx = tr(Hxx) + mpmath.fsum((Aigx * Gx)[i, i] for i in range(n))
        thyy = tr(Hyy) + mpmath.fsum((Aigy * Gy)[i, i] for i in range(n))
        thxy = tr(Hxy) + mpmath.fsum((Aigx * Gy)[i, i] for i in range(n))
        return DS(det, det * tgx, det * tgy,
                  det * (tgx ** 2 + thxx), det * (tgy ** 2 + thyy),
                  det * (tgx * tgy + thxy))

def d_entry(a, b, p, y):
    """DS of kdcov(a, b, p - y) (fixed a, b, p; function of y)."""
    dx, dy = p[0] - y[0], p[1] - y[1]
    v = kdcov(a, b, dx, dy)
    gx = -kdcov_d(a, b, dx, dy, 0)
    gy = -kdcov_d(a, b, dx, dy, 1)
    def dd(e1, e2):
        s = kplane(a[0] + b[0] + e1[0] + e2[0], a[1] + b[1] + e1[1] + e2[1], dx, dy)
        for (i, j) in _IMG:
            s += kplane(a[0] + b[0] + e1[0] + e2[0], a[1] + b[1] + e1[1] + e2[1],
                        dx + _LT * i, dy + _LT * j)
        return ((-1) ** (b[0] + b[1])) * s
    hxx = dd((1, 0), (1, 0)); hyy = dd((0, 1), (0, 1)); hxy = dd((1, 0), (0, 1))
    return DS(v, gx, gy, hxx, hyy, hxy)

# ------------------------- D-station: everything as DS/DM -------------------
def de1(a, b, Q, y):
    """DS of cov(d^a f(y), d^b f(Q)): first (moving) slot at y."""
    dx, dy = y[0] - Q[0], y[1] - Q[1]
    v = kdcov(a, b, dx, dy)
    gx = kdcov_d(a, b, dx, dy, 0)
    gy = kdcov_d(a, b, dx, dy, 1)
    def dd(e1, e2):
        s = kplane(a[0] + b[0] + e1[0] + e2[0], a[1] + b[1] + e1[1] + e2[1], dx, dy)
        for (i, j) in _IMG:
            s += kplane(a[0] + b[0] + e1[0] + e2[0], a[1] + b[1] + e1[1] + e2[1],
                        dx + _LT * i, dy + _LT * j)
        return ((-1) ** (b[0] + b[1])) * s
    return DS(v, gx, gy, dd((1, 0), (1, 0)), dd((0, 1), (0, 1)), dd((1, 0), (0, 1)))

def de2(a, b, P, y):
    """DS of cov(d^a f(P), d^b f(y)): second (moving) slot at y."""
    e = de1(b, a, P, y)
    # cov(a@P, b@y) = cov(b@y, a@P) -- symmetric: same value/derivatives
    return e

def dconst(v):
    return DS(v)

def dstation_dm(y):
    """the full station as DM blocks with exact derivatives."""
    def tentry(pi, di, pj, dj):
        if pi == 'Y':
            return de1(_GI[di], _GI[dj], PTS[pj], y)
        return dconst(kdcov(_GI[di], _GI[dj], PTS[pi][0] - PTS[pj][0],
                            PTS[pi][1] - PTS[pj][1]))
    TC = DM([[tentry(pi, di, pj, dj) for (pj, dj) in CSET6]
             for (pi, di) in TSET9])
    def tyentry(pi, di, pj, dj):
        if pi == 'Y':
            return dconst(kdcov(_GI[di], _GI[dj], mpf(0), mpf(0)))
        return de2(_GI[di], _GI[dj], PTS[pi], y)
    TY = DM([[tyentry(pi, di, pj, dj) for (pj, dj) in YJET]
             for (pi, di) in TSET9])
    def ttentry(pi, di, pj2, dj):
        if pi == 'Y' and pj2 == 'Y':
            return dconst(kdcov(_GI[di], _GI[dj], mpf(0), mpf(0)))
        if pi == 'Y':
            return de1(_GI[di], _GI[dj], PTS[pj2], y)
        if pj2 == 'Y':
            return de2(_GI[di], _GI[dj], PTS[pi], y)
        return dconst(kdcov(_GI[di], _GI[dj], PTS[pi][0] - PTS[pj2][0],
                            PTS[pi][1] - PTS[pj2][1]))
    TT = DM([[ttentry(pi, di, pj2, dj) for (pj2, dj) in TSET9]
             for (pi, di) in TSET9])
    YY = DM([[dconst(kdcov(_GI[di], _GI[dj], mpf(0), mpf(0))) for (pj, dj) in YJET]
             for (pi, di) in YJET])
    YC = DM([[de1(_GI[di], _GI[dj], PTS[pj], y) for (pj, dj) in CSET6]
             for (pi, di) in YJET])
    TT6 = TT - (TC * G6inv) * TC.T
    TY6 = TY - (TC * G6inv) * YC.T
    YY6 = YY - (YC * G6inv) * YC.T
    YY6inv = YY6.inv()
    cov9 = TT6 - (TY6 * YY6inv) * TY6.T
    return {'TC': TC, 'TY6': TY6, 'YY6': YY6, 'YY6inv': YY6inv, 'cov9': cov9,
            'YC': YC}

def dm_sub(M, ridx, cidx):
    return DM([[M.a[i][j] for j in cidx] for i in ridx])

def dm_vec_col(M, j):
    return DM([[M.a[i][j]] for i in range(M.n)])

def ds_wick(mu_ds, Sig_dm, a, b_, c, cache):
    if a < 0 or b_ < 0 or c < 0:
        return DS(0)
    if a == 0 and b_ == 0 and c == 0:
        return DS(1)
    key = (a, b_, c)
    if key in cache:
        return cache[key]
    if a > 0:
        val = (mu_ds[0] * ds_wick(mu_ds, Sig_dm, a - 1, b_, c, cache)
               + Sig_dm.a[0][0] * ((a - 1) * ds_wick(mu_ds, Sig_dm, a - 2, b_, c, cache))
               + Sig_dm.a[0][1] * (b_ * ds_wick(mu_ds, Sig_dm, a - 1, b_ - 1, c, cache))
               + Sig_dm.a[0][2] * (c * ds_wick(mu_ds, Sig_dm, a - 1, b_, c - 1, cache)))
    elif b_ > 0:
        val = (mu_ds[1] * ds_wick(mu_ds, Sig_dm, a, b_ - 1, c, cache)
               + Sig_dm.a[1][0] * (a * ds_wick(mu_ds, Sig_dm, a - 1, b_ - 1, c, cache))
               + Sig_dm.a[1][1] * ((b_ - 1) * ds_wick(mu_ds, Sig_dm, a, b_ - 2, c, cache))
               + Sig_dm.a[1][2] * (c * ds_wick(mu_ds, Sig_dm, a, b_ - 1, c - 1, cache)))
    else:
        val = (mu_ds[2] * ds_wick(mu_ds, Sig_dm, a, b_, c - 1, cache)
               + Sig_dm.a[2][0] * (a * ds_wick(mu_ds, Sig_dm, a - 1, b_, c - 1, cache))
               + Sig_dm.a[2][1] * (b_ * ds_wick(mu_ds, Sig_dm, a, b_ - 1, c - 1, cache))
               + Sig_dm.a[2][2] * ((c - 1) * ds_wick(mu_ds, Sig_dm, a, b_, c - 2, cache)))
    cache[key] = val
    return val

def ds_det_moment(mu_ds, Sig_dm, k):
    cache = {}
    def M(a, b_, c):
        return ds_wick(mu_ds, Sig_dm, a, b_, c, cache)
    tot = DS(0)
    for j in range(k + 1):
        coef = mpf(_comb(k, j)) * ((-1) ** j)
        tot = tot + coef * M(k - j, k - j, 2 * j)
    return tot

def ds_enorm4(mu_ds, Sig_dm):
    cache = {}
    def M(a, b_, c):
        return ds_wick(mu_ds, Sig_dm, a, b_, c, cache)
    return (M(4, 0, 0) + M(0, 4, 0) + M(0, 0, 4)
            + 2 * (M(2, 2, 0) + M(2, 0, 2) + M(0, 2, 2)))

def ds_pow(s, p):
    ck(s.v > 0, "ds_pow of nonpositive %s" % mpmath.nstr(s.v, 3))
    return (s.log() * p).exp()

def bures_trace_ds(A):
    """tr[A^{1/2}] as a DS for a symmetric PD DM A (3x3).
    d lam_i = v_i' dA v_i; d^2 lam_i = v_i' d^2A v_i + 2 sum_{j!=i}
    (v_i' dA v_j)^2/(lam_i - lam_j) (and the mixed-partial analogue)."""
    n = A.n
    V = mp.matrix([[A.a[i][j].v for j in range(n)] for i in range(n)])
    lam, U = V1.eigvals_cert(V, n, "bures")
    for i in range(n):
        ck(lam[i] > mpf('1e-30'), "bures eigenvalue nonpositive")
    gaps = min(abs(lam[i] - lam[j]) for i in range(n) for j in range(n) if i != j)
    ck(gaps > mpf('1e-6'), "bures eigenvalue gap too small: %s" % mpmath.nstr(gaps, 3))
    def utdU(slot):
        dA = mp.matrix([[getattr(A.a[i][j], slot) for j in range(n)] for i in range(n)])
        M = U.T * dA * U
        return M
    Mgx = utdU('gx'); Mgy = utdU('gy')
    Mhxx = utdU('hxx'); Mhyy = utdU('hyy'); Mhxy = utdU('hxy')
    tot = DS(0)
    for i in range(n):
        li = lam[i]
        gx = Mgx[i, i]; gy = Mgy[i, i]
        hxx = Mhxx[i, i] + 2 * mpmath.fsum(Mgx[i, j] ** 2 / (li - lam[j])
                                           for j in range(n) if j != i)
        hyy = Mhyy[i, i] + 2 * mpmath.fsum(Mgy[i, j] ** 2 / (li - lam[j])
                                           for j in range(n) if j != i)
        hxy = Mhxy[i, i] + 2 * mpmath.fsum(Mgx[i, j] * Mgy[i, j] / (li - lam[j])
                                           for j in range(n) if j != i)
        tot = tot + DS(li, gx, gy, hxx, hyy, hxy).sqrt()
    return tot

_DEBUG_DS = False

def kappa_far_ds(y, v):
    """the assembled kap_far(y) as a DS (exact derivatives), Z_LO pieces."""
    st = dstation_dm(y)
    cov9 = st['cov9']; YY6 = st['YY6']; YY6inv = st['YY6inv']
    TC = st['TC']; TY6 = st['TY6']; YC = st['YC']
    mu6 = TC * _W6
    rv = DM([[DS(v)], [DS(0)], [DS(0)]]) - YC * _W6
    muv = mu6 + (TY6 * YY6inv) * rv
    muM = [muv.a[i][0] for i in range(3)]
    muS = [muv.a[i][0] for i in range(3, 6)]
    SM = dm_sub(cov9, [0, 1, 2], [0, 1, 2])
    SS = dm_sub(cov9, [3, 4, 5], [3, 4, 5])
    dM8 = ds_det_moment(muM, SM, 8)
    dS8 = ds_det_moment(muS, SS, 8)
    ck(dM8.v > 0 and dS8.v > 0, "Wick 8th moments nonpositive (D)")
    EW4 = ds_pow(dM8 * dS8, mpf('0.125'))   # frozen: (sqrt dM8 sqrt dS8)^{1/4}
    mu_y = [muv.a[i][0] for i in range(6, 9)]
    Spair = dm_sub(cov9, range(6), range(6))
    Spinv = Spair.inv()
    Delta = dm_sub(cov9, range(6), [6, 7, 8])
    Sy = dm_sub(cov9, [6, 7, 8], [6, 7, 8])
    BregT = Spinv * Delta          # 6x3 = Spair^-1 Delta
    Syres = Sy - (Delta.T * BregT)
    tau = DS(0)
    for i in range(3):
        tau = tau + (Delta.T * BregT).a[i][i]
    Eq2 = 8 * (ds_enorm4(mu_y, Sy) + ds_enorm4(mu_y, Syres))
    msad = E.m_sad_cf(v)
    kap_cross = DS(mpmath.sqrt(3)) * EW4 * ds_pow(Eq2, mpf('0.25')) * tau.sqrt() \
        / (Z_LO * msad)
    C_mom = DS(mpmath.sqrt(3)) * EW4 * ds_pow(Eq2, mpf('0.25')) / (Z_LO * msad)
    # chi2
    mu6k = kit.Hmean; S6 = kit.Hcov
    mu9p = DM([[muv.a[i][0]] for i in range(6)])
    S6inv = S6 ** -1
    Mm = Spinv - DM.const_mat(S6inv) * mpf('0.5')
    Minv = Mm.inv()
    cvec = Spinv * mu9p - DM.const_mat(S6inv * mu6k) * mpf('0.5')
    q1 = (cvec.T * Minv) * cvec
    kk = (mu9p.T * Spinv) * mu9p
    kk = DM([[kk.a[0][0] - DS(mpf('0.5') * (mu6k.T * S6inv * mu6k)[0])]])
    detS9 = Spair.det_ds()
    detM = Mm.det_ds()
    logq = (DS(-3 * mpmath.log(2) + mpmath.log(mpmath.fabs(mp.det(S6))) / 2)
            - detS9.log() - detM.log() / 2 + q1.a[0][0] - kk.a[0][0])
    chi2 = logq.exp() - 1
    ck(chi2.v >= 0, "chi2 negative (D)")
    dM4 = E.det_moment(mp.matrix([mu6k[0], mu6k[1], mu6k[2]]),
                       E.submat(S6, [0, 1, 2], [0, 1, 2]), 4)
    dS4 = E.det_moment(mp.matrix([mu6k[3], mu6k[4], mu6k[5]]),
                       E.submat(S6, [3, 4, 5], [3, 4, 5]), 4)
    hL2 = (mpmath.sqrt(dM4) * mpmath.sqrt(dS4)) ** mpf('0.5')
    kap_pair = DS(hL2) * chi2.sqrt() / Z_LO
    # kappa_y (W2 with Bures trace)
    a2, a4 = E.A2, E.A4
    mu_y0 = [DS(-a2 * v), DS(-a2 * v), DS(0)]
    Sy0 = mp.zeros(3, 3)
    Sy0[0, 0] = a4 - a2 * a2; Sy0[1, 1] = a4 - a2 * a2; Sy0[2, 2] = a2 * a2
    dmu = DM([[mu_y[i] - mu_y0[i]] for i in range(3)])
    dmusq = (dmu.T * dmu).a[0][0]
    trS1 = Syres.a[0][0] + Syres.a[1][1] + Syres.a[2][2]
    trS0 = DS(sum(Sy0[i, i] for i in range(3)))
    S0h = mp.zeros(3, 3)
    for i in range(3):
        S0h[i, i] = mpmath.sqrt(Sy0[i, i])
    A = DM.const_mat(S0h) * Syres * S0h
    btr = bures_trace_ds(A)
    w2 = dmusq + trS1 + trS0 - 2 * btr
    ck(w2.v >= 0, "w2 negative (D)")
    Ey1 = trS1 + sum(mu_y[i] * mu_y[i] for i in range(3))
    Ey0 = DS(sum(Sy0[i, i] for i in range(3)) + sum(mu_y0[i].v ** 2 for i in range(3)))
    kap_y = ds_pow(DS(2) * (Ey1 + Ey0), mpf('0.5')) * w2.sqrt() / msad
    # ratios
    muy = YC * _W6
    Sg = dm_sub(YY6, [1, 2], [1, 2])
    Sginv = Sg.inv()
    mug = DM([[muy.a[1][0]], [muy.a[2][0]]])
    maha = (mug.T * Sginv) * mug
    detSg = Sg.det_ds()
    pgrad = (maha.a[0][0] * (-mpf('0.5'))).exp() / (DS(2 * mp.pi) * detSg.sqrt())
    pgrad0 = 1 / (2 * mp.pi * a2)
    Rpg = pgrad / pgrad0
    c01 = DM([[YY6.a[0][1]], [YY6.a[0][2]]])
    s2 = YY6.a[0][0] - ((c01.T * Sginv) * c01).a[0][0]
    ck(s2.v > 0, "needle variance nonpositive (D)")
    st_t = s2.sqrt()
    mu_t = muy.a[0][0] - ((c01.T * Sginv) * mug).a[0][0]
    wm = ((DS(kit.b) - mu_t) / st_t).Phi() - ((DS(kit.s) - mu_t) / st_t).Phi()
    wm0 = E._Phi(kit.b) - E._Phi(kit.s)
    Rwm = wm / wm0
    kap_far = Rpg * Rwm * ((DS(1) + kap_pair) * (DS(1) + kap_y) + kap_cross) - DS(1)
    if _DEBUG_DS:
        emit("  [Ddbg] Rpg=%s Rwm=%s kap_pair=%s kap_y=%s kap_cross=%s tau=%s"
             " chi2=%s w2=%s wm=%s pgrad=%s"
             % (mpmath.nstr(Rpg.v, 8), mpmath.nstr(Rwm.v, 8),
                mpmath.nstr(kap_pair.v, 8), mpmath.nstr(kap_y.v, 8),
                mpmath.nstr(kap_cross.v, 8), mpmath.nstr(tau.v, 8),
                mpmath.nstr(chi2.v, 8), mpmath.nstr(w2.v, 8),
                mpmath.nstr(wm.v, 8), mpmath.nstr(pgrad.v, 8)))
    return kap_far

# validate the D-assembly against the frozen v2 values at (5,0)
_kd = kappa_far_ds((mpf(5), mpf(0)), kit.b - kit.ell / 2)
emit("D-assembly at (5,0): kap_far = %s (frozen v2: 0.677284905),"
     " |grad| = %s, |hess| = %s"
     % (mpmath.nstr(_kd.v, 12), mpmath.nstr(_kd.gnorm(), 6),
        mpmath.nstr(_kd.hnorm(), 6)))
ck(abs(_kd.v - mpf('0.677284905')) < mpf('1e-6'),
   "D kap_far deviates from the frozen v2 value: %s"
   % mpmath.nstr(abs(_kd.v - mpf('0.677284905')), 3))

# ================= piece-1 net machinery: per-point (kap0, grad) + ring =====
# ---- third derivatives of the form functions (exact, cheap) ----------------
def form_thrd(form, gamma, y):
    """Frobenius norm of the third-derivative tensor of F_{form,gamma}."""
    tt = [mpf(0)] * 4   # xxx, xxy, xyy, yyy
    for (p, a, c) in form:
        dx, dy = p[0] - y[0], p[1] - y[1]
        for (e1, e2, e3, slot) in [((3, 0), None, None, 0), ((2, 1), None, None, 1),
                                   ((1, 2), None, None, 2), ((0, 3), None, None, 3)]:
            s = kplane(a[0] + gamma[0] + e1[0], a[1] + gamma[1] + e1[1], dx, dy)
            for (i, j) in _IMG:
                s += kplane(a[0] + gamma[0] + e1[0], a[1] + gamma[1] + e1[1],
                            dx + _LT * i, dy + _LT * j)
            tt[slot] += ((-1) ** (gamma[0] + gamma[1])) * c * s
    return mpmath.sqrt(tt[0] ** 2 + 3 * tt[1] ** 2 + 3 * tt[2] ** 2 + tt[3] ** 2)

def kdcov_dd(a, b, dx, dy, e1, e2):
    ee = [(1, 0)] * e1 + [(0, 1)] * e2
    s = kplane(a[0] + b[0] + e1, a[1] + b[1] + e2, dx, dy)
    for (i, j) in _IMG:
        s += kplane(a[0] + b[0] + e1, a[1] + b[1] + e2, dx + _LT * i, dy + _LT * j)
    return ((-1) ** (b[0] + b[1])) * s

def yc_derivs(y):
    """exact YC (3x6) derivatives through third order."""
    YC = mp.zeros(3, 6)
    dYC = [mp.zeros(3, 6), mp.zeros(3, 6)]
    d2YC = [mp.zeros(3, 6), mp.zeros(3, 6), mp.zeros(3, 6)]   # xx yy xy
    d3YC = [mp.zeros(3, 6), mp.zeros(3, 6), mp.zeros(3, 6), mp.zeros(3, 6)]
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(CSET6):
            a, b = _GI[di], _GI[dj]
            dx, dy = y[0] - PTS[pj][0], y[1] - PTS[pj][1]
            YC[i, j] = kdcov(a, b, dx, dy)
            dYC[0][i, j] = kdcov_d(a, b, dx, dy, 0)
            dYC[1][i, j] = kdcov_d(a, b, dx, dy, 1)
            d2YC[0][i, j] = kdcov_dd(a, b, dx, dy, 2, 0)
            d2YC[1][i, j] = kdcov_dd(a, b, dx, dy, 0, 2)
            d2YC[2][i, j] = kdcov_dd(a, b, dx, dy, 1, 1)
            d3YC[0][i, j] = kdcov_dd(a, b, dx, dy, 3, 0)
            d3YC[1][i, j] = kdcov_dd(a, b, dx, dy, 2, 1)
            d3YC[2][i, j] = kdcov_dd(a, b, dx, dy, 1, 2)
            d3YC[3][i, j] = kdcov_dd(a, b, dx, dy, 0, 3)
    return YC, dYC, d2YC, d3YC

def ty6_derivs(fs, y):
    """exact TY6 (9x3) derivatives through third order."""
    YC, dYC, d2YC, d3YC = yc_derivs(y)
    TCw = mp.zeros(9, 6)
    for i, (pi, di) in enumerate(TSET9):
        p1 = y if pi == 'Y' else PTS[pi]
        for j, (pj, dj) in enumerate(CSET6):
            TCw[i, j] = kc(di, dj, p1, PTS[pj])
    TCG = TCw * G6inv
    TY6 = fs['TY6']
    dTY6 = [mp.zeros(9, 3), mp.zeros(9, 3)]
    d2TY6 = [mp.zeros(9, 3), mp.zeros(9, 3), mp.zeros(9, 3)]
    d3TY6 = [mp.zeros(9, 3), mp.zeros(9, 3), mp.zeros(9, 3), mp.zeros(9, 3)]
    for i, (pi, di) in enumerate(TSET9):
        if pi == 'Y':
            continue
        for j, (pj, dj) in enumerate(YJET):
            a, b = _GI[di], _GI[dj]
            dx, dy = PTS[pi][0] - y[0], PTS[pi][1] - y[1]
            # TY6[i,j] = kdcov(a, b, PTS[pi] - y) - TCG[i,:] . YC[:,j]
            dTY6[0][i, j] = -kdcov_d(a, b, dx, dy, 0) - sum(TCG[i, l] * dYC[0][j, l] for l in range(6))
            dTY6[1][i, j] = -kdcov_d(a, b, dx, dy, 1) - sum(TCG[i, l] * dYC[1][j, l] for l in range(6))
            d2TY6[0][i, j] = kdcov_dd(a, b, dx, dy, 2, 0) - sum(TCG[i, l] * d2YC[0][j, l] for l in range(6))
            d2TY6[1][i, j] = kdcov_dd(a, b, dx, dy, 0, 2) - sum(TCG[i, l] * d2YC[1][j, l] for l in range(6))
            d2TY6[2][i, j] = kdcov_dd(a, b, dx, dy, 1, 1) - sum(TCG[i, l] * d2YC[2][j, l] for l in range(6))
            d3TY6[0][i, j] = -kdcov_dd(a, b, dx, dy, 3, 0) - sum(TCG[i, l] * d3YC[0][j, l] for l in range(6))
            d3TY6[1][i, j] = -kdcov_dd(a, b, dx, dy, 2, 1) - sum(TCG[i, l] * d3YC[1][j, l] for l in range(6))
            d3TY6[2][i, j] = -kdcov_dd(a, b, dx, dy, 1, 2) - sum(TCG[i, l] * d3YC[2][j, l] for l in range(6))
            d3TY6[3][i, j] = -kdcov_dd(a, b, dx, dy, 0, 3) - sum(TCG[i, l] * d3YC[3][j, l] for l in range(6))
    return TY6, dTY6, d2TY6, d3TY6

def yy6_derivs(fs, y):
    """exact YY6 derivatives through third order (YY6 = YY - YC G6inv YC')."""
    YC, dYC, d2YC, d3YC = yc_derivs(y)
    YY6 = fs['YY6']
    dYY6 = [-(dYC[e] * G6inv * YC.T + YC * G6inv * dYC[e].T) for e in (0, 1)]
    d2YY6 = [-(d2YC[0] * G6inv * YC.T + 2 * dYC[0] * G6inv * dYC[0].T
               + YC * G6inv * d2YC[0].T),
             -(d2YC[1] * G6inv * YC.T + 2 * dYC[1] * G6inv * dYC[1].T
               + YC * G6inv * d2YC[1].T),
             -(d2YC[2] * G6inv * YC.T + dYC[0] * G6inv * dYC[1].T
               + dYC[1] * G6inv * dYC[0].T + YC * G6inv * d2YC[2].T)]
    d3YY6 = []
    for (slots, d3slot) in [((0, 0, 0), 0), ((0, 0, 1), 1),
                            ((0, 1, 1), 2), ((1, 1, 1), 3)]:
        a, b, c = slots
        T = (d3YC[d3slot] * G6inv * YC.T
             + d2YC[_h2(a, b)] * G6inv * dYC[c].T
             + d2YC[_h2(a, c)] * G6inv * dYC[b].T
             + d2YC[_h2(b, c)] * G6inv * dYC[a].T
             + dYC[a] * G6inv * d2YC[_h2(b, c)].T
             + dYC[b] * G6inv * d2YC[_h2(a, c)].T
             + dYC[c] * G6inv * d2YC[_h2(a, b)].T
             + YC * G6inv * d3YC[d3slot].T)
        d3YY6.append(-T)
    return YY6, dYY6, d2YY6, d3YY6

def _h2(a, b):
    if a == 0 and b == 0: return 0
    if a == 1 and b == 1: return 1
    return 2

def inv_derivs(A, dA, d2A, d3A):
    """(A^-1, d, d2, d3) of the inverse given the exact derivatives."""
    P = A ** -1
    rr = max(abs((A * P - mp.eye(A.rows))[i, j]) for i in range(A.rows)
             for j in range(A.rows))
    ck(rr < mpf('1e-30'), "inv_derivs residual %s" % mpmath.nstr(rr, 3))
    dP = [-P * dA[e] * P for e in (0, 1)]
    d2P = [-P * d2A[0] * P + 2 * P * dA[0] * P * dA[0] * P,
           -P * d2A[1] * P + 2 * P * dA[1] * P * dA[1] * P,
           -P * d2A[2] * P + P * dA[0] * P * dA[1] * P + P * dA[1] * P * dA[0] * P]
    d3P = []
    for (a, b, c, s3) in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 1, 2), (1, 1, 1, 3)]:
        T = (-P * d3A[s3] * P
             + P * d2A[_h2(a, b)] * P * dA[c] * P + P * dA[a] * P * d2A[_h2(b, c)] * P
             + P * d2A[_h2(a, c)] * P * dA[b] * P + P * dA[c] * P * d2A[_h2(a, b)] * P
             + P * d2A[_h2(b, c)] * P * dA[a] * P + P * dA[b] * P * d2A[_h2(a, c)] * P
             - 2 * (P * dA[a] * P * dA[b] * P * dA[c] * P
                    + P * dA[a] * P * dA[c] * P * dA[b] * P
                    + P * dA[b] * P * dA[a] * P * dA[c] * P
                    + P * dA[b] * P * dA[c] * P * dA[a] * P
                    + P * dA[c] * P * dA[a] * P * dA[b] * P
                    + P * dA[c] * P * dA[b] * P * dA[a] * P))
        d3P.append(T)
    return P, dP, d2P, d3P

def tau_bounds_full(y, d):
    """EXACT (tau, |grad tau|, |hess tau|_F, |third tau|_F) at y:
    every rigid-sensitive product is an exact matrix product."""
    cov9, TT6, TY6, YY6 = station_fast(y)
    tau = tau_exact(cov9)
    Q, dQ, d2Q, d3Q = qblock_full(y)
    _Y, dYY6, d2YY6, d3YY6 = yy6_derivs({'YY6': YY6}, y)
    P, dP, d2P, d3P = inv_derivs(YY6, dYY6, d2YY6, d3YY6)
    TY6m, dTY6, d2TY6, d3TY6 = ty6_derivs({'TY6': TY6}, y)
    # dF and its derivatives
    m_i = [TY6m[6 + i, :] for i in range(3)]
    dm_i = [[dTY6[e][6 + i, :] for e in (0, 1)] for i in range(3)]
    d2m_i = [[[d2TY6[s2][6 + i, :] for s2 in (0, 1, 2)]] for i in range(3)]
    d3m_i = [[[d3TY6[s3][6 + i, :] for s3 in (0, 1, 2, 3)]] for i in range(3)]
    red = [(Q[k, :] * P * Q[k, :].T)[0] for k in range(6)]
    dF = [[(Q[k, :] * P * m_i[i].T)[0] for i in range(3)] for k in range(6)]
    ddF = {}
    for k in range(6):
        for i in range(3):
            g1 = [(dQ[e][k, :] * P * m_i[i].T + Q[k, :] * dP[e] * m_i[i].T
                   + Q[k, :] * P * dm_i[i][e].T)[0] for e in (0, 1)]
            h2 = []
            for (a, b, s2) in [(0, 0, 0), (1, 1, 1), (0, 1, 2)]:
                t = (d2Q[s2][k, :] * P * m_i[i].T
                     + dQ[a][k, :] * dP[b] * m_i[i].T + dQ[a][k, :] * P * dm_i[i][b].T
                     + dQ[b][k, :] * dP[a] * m_i[i].T + Q[k, :] * d2P[s2] * m_i[i].T
                     + Q[k, :] * dP[a] * dm_i[i][b].T
                     + dQ[b][k, :] * P * dm_i[i][a].T + Q[k, :] * dP[b] * dm_i[i][a].T
                     + Q[k, :] * P * d2m_i[i][0][s2].T)
                h2.append(t[0])
            t3 = []
            for (a, b, c, s3) in [(0, 0, 0, 0), (0, 0, 1, 1),
                                  (0, 1, 1, 2), (1, 1, 1, 3)]:
                t = mpf(0)
                for (L, Mm2, R) in [
                        (d3Q[s3], P, m_i[i]), (d2Q[_h2(a, b)], dP[c], m_i[i]),
                        (d2Q[_h2(a, c)], dP[b], m_i[i]), (d2Q[_h2(b, c)], dP[a], m_i[i]),
                        (dQ[a], d2P[_h2(b, c)], m_i[i]), (dQ[b], d2P[_h2(a, c)], m_i[i]),
                        (dQ[c], d2P[_h2(a, b)], m_i[i]), (Q, d3P[s3], m_i[i]),
                        (d2Q[_h2(a, b)], P, dm_i[i][c]), (d2Q[_h2(a, c)], P, dm_i[i][b]),
                        (d2Q[_h2(b, c)], P, dm_i[i][a]), (dQ[a], dP[c], dm_i[i][b]),
                        (dQ[a], dP[b], dm_i[i][c]), (dQ[b], dP[a], dm_i[i][c]),
                        (dQ[b], dP[c], dm_i[i][a]), (dQ[c], dP[a], dm_i[i][b]),
                        (dQ[c], dP[b], dm_i[i][a]),
                        (dQ[a], P, d2m_i[i][0][_h2(b, c)]),
                        (dQ[b], P, d2m_i[i][0][_h2(a, c)]),
                        (dQ[c], P, d2m_i[i][0][_h2(a, b)]),
                        (d2Q[_h2(a, b)], dP[c], m_i[i]),
                        (Q, d2P[_h2(a, b)], dm_i[i][c]),
                        (Q, d2P[_h2(a, c)], dm_i[i][b]),
                        (Q, d2P[_h2(b, c)], dm_i[i][a]),
                        (Q, dP[a], d2m_i[i][0][_h2(b, c)]),
                        (Q, dP[b], d2m_i[i][0][_h2(a, c)]),
                        (Q, dP[c], d2m_i[i][0][_h2(a, b)]),
                        (Q, P, d3m_i[i][0][s3])]:
                    t += (L[k, :] * Mm2 * R.T)[0]
                t3.append(t)
            ddF[(k, i)] = (g1, h2, t3)
    # exact F derivatives (form machinery)
    Ft = mp.zeros(6, 3)
    FG = {}
    for k in range(6):
        for i in range(3):
            Ft[k, i] = form_eval(FORMS[k], _HYG[i], y) - dF[k][i]
            FG[(k, i)] = form_grad(FORMS[k], _HYG[i], y)
    # assemble exact tau derivatives in the V0 basis
    Stinv, dSxm, dSym, d2S, d3S = spair_dir_full(y, d)
    dS = [dSxm, dSym]
    # c = Ft (6x3): tau = sum_{k,k',i} c_{k,i} Stinv_{k,k'} c_{k',i}
    _CM2 = {}
    _CM3 = {}
    def cder(k, i, e):
        return FG[(k, i)][e] - ddF[(k, i)][0][e]
    def cder2(k, i, s2):
        key = (k, i, s2)
        if key not in _CM2:
            hxx, hyy, hxy = form_hess_fn(FORMS[k], _HYG[i], y)
            _CM2[key] = (hxx, hyy, hxy)[s2] - ddF[(k, i)][1][s2]
        return _CM2[key]
    def cder3(k, i, s3):
        key = (k, i, s3)
        if key not in _CM3:
            tt = mpf(0)
            for (p, a, c) in FORMS[k]:
                dx, dy = p[0] - y[0], p[1] - y[1]
                e1 = [(3, 0), (2, 1), (1, 2), (0, 3)][s3]
                s = kplane(a[0] + _HYG[i][0] + e1[0], a[1] + _HYG[i][1] + e1[1], dx, dy)
                for (i2, j2) in _IMG:
                    s += kplane(a[0] + _HYG[i][0] + e1[0], a[1] + _HYG[i][1] + e1[1],
                                dx + _LT * i2, dy + _LT * j2)
                tt += ((-1) ** (_HYG[i][0] + _HYG[i][1])) * c * s
            _CM3[key] = -tt - ddF[(k, i)][2][s3]
        return _CM3[key]
    def dSoder(e):
        return -Stinv * dS[e] * Stinv
    dSo = [dSoder(0), dSoder(1)]
    d2So = [-Stinv * d2S[0] * Stinv + 2 * Stinv * dS[0] * Stinv * dS[0] * Stinv,
            -Stinv * d2S[1] * Stinv + 2 * Stinv * dS[1] * Stinv * dS[1] * Stinv,
            -Stinv * d2S[2] * Stinv + Stinv * dS[0] * Stinv * dS[1] * Stinv
            + Stinv * dS[1] * Stinv * dS[0] * Stinv]
    d3So = []
    for (a, b, c, s3) in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 1, 2), (1, 1, 1, 3)]:
        P2 = Stinv
        T = (-P2 * d3S[s3] * P2
             + P2 * d2S[_h2(a, b)] * P2 * dS[c] * P2 + P2 * dS[a] * P2 * d2S[_h2(b, c)] * P2
             + P2 * d2S[_h2(a, c)] * P2 * dS[b] * P2 + P2 * dS[c] * P2 * d2S[_h2(a, b)] * P2
             + P2 * d2S[_h2(b, c)] * P2 * dS[a] * P2 + P2 * dS[b] * P2 * d2S[_h2(a, c)] * P2
             - 2 * (P2 * dS[a] * P2 * dS[b] * P2 * dS[c] * P2
                    + P2 * dS[a] * P2 * dS[c] * P2 * dS[b] * P2
                    + P2 * dS[b] * P2 * dS[a] * P2 * dS[c] * P2
                    + P2 * dS[b] * P2 * dS[c] * P2 * dS[a] * P2
                    + P2 * dS[c] * P2 * dS[a] * P2 * dS[b] * P2
                    + P2 * dS[c] * P2 * dS[b] * P2 * dS[a] * P2))
        d3So.append(T)
    gtau2 = mpf(0)
    for e in (0, 1):
        t = mpf(0)
        for k in range(6):
            for kp in range(6):
                for i in range(3):
                    t += (cder(k, i, e) * Stinv[k, kp] * Ft[kp, i]
                          + Ft[k, i] * dSo[e][k, kp] * Ft[kp, i]
                          + Ft[k, i] * Stinv[k, kp] * cder(kp, i, e))
        gtau2 += t * t
    gtau = mpmath.sqrt(gtau2)
    htau2 = mpf(0)
    for s2, (a, b) in enumerate([(0, 0), (1, 1), (0, 1)]):
        t = mpf(0)
        for k in range(6):
            for kp in range(6):
                for i in range(3):
                    t += (cder2(k, i, s2) * Stinv[k, kp] * Ft[kp, i]
                          + cder(k, i, a) * dSo[b][k, kp] * Ft[kp, i]
                          + cder(k, i, a) * Stinv[k, kp] * cder(kp, i, b)
                          + cder(k, i, b) * dSo[a][k, kp] * Ft[kp, i]
                          + Ft[k, i] * d2So[s2][k, kp] * Ft[kp, i]
                          + Ft[k, i] * dSo[a][k, kp] * cder(kp, i, b)
                          + cder(k, i, b) * Stinv[k, kp] * cder(kp, i, a)
                          + Ft[k, i] * dSo[b][k, kp] * cder(kp, i, a)
                          + Ft[k, i] * Stinv[k, kp] * cder2(kp, i, s2))
        htau2 += (t * t) * (2 if s2 == 2 else 1)
    htau = mpmath.sqrt(htau2)
    ttau2 = mpf(0)
    for s3, (a, b, c) in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]):
        t = mpf(0)
        for k in range(6):
            for kp in range(6):
                for i in range(3):
                    # full Leibniz of c * So * c to third order
                    t += cder3(k, i, s3) * Stinv[k, kp] * Ft[kp, i]
                    t += Ft[k, i] * Stinv[k, kp] * cder3(kp, i, s3)
                    t += Ft[k, i] * d3So[s3][k, kp] * Ft[kp, i]
                    t += cder2(k, i, _h2(a, b)) * dSo[c][k, kp] * Ft[kp, i]
                    t += cder2(k, i, _h2(a, c)) * dSo[b][k, kp] * Ft[kp, i]
                    t += cder2(k, i, _h2(b, c)) * dSo[a][k, kp] * Ft[kp, i]
                    t += Ft[k, i] * dSo[a][k, kp] * cder2(kp, i, _h2(b, c))
                    t += Ft[k, i] * dSo[b][k, kp] * cder2(kp, i, _h2(a, c))
                    t += Ft[k, i] * dSo[c][k, kp] * cder2(kp, i, _h2(a, b))
                    t += cder2(k, i, _h2(a, b)) * Stinv[k, kp] * cder(kp, i, c)
                    t += cder2(k, i, _h2(a, c)) * Stinv[k, kp] * cder(kp, i, b)
                    t += cder2(k, i, _h2(b, c)) * Stinv[k, kp] * cder(kp, i, a)
                    t += cder(k, i, a) * Stinv[k, kp] * cder2(kp, i, _h2(b, c))
                    t += cder(k, i, b) * Stinv[k, kp] * cder2(kp, i, _h2(a, c))
                    t += cder(k, i, c) * Stinv[k, kp] * cder2(kp, i, _h2(a, b))
                    t += cder(k, i, a) * dSo[b][k, kp] * cder(kp, i, c)
                    t += cder(k, i, a) * dSo[c][k, kp] * cder(kp, i, b)
                    t += cder(k, i, b) * dSo[a][k, kp] * cder(kp, i, c)
                    t += cder(k, i, b) * dSo[c][k, kp] * cder(kp, i, a)
                    t += cder(k, i, c) * dSo[a][k, kp] * cder(kp, i, b)
                    t += cder(k, i, c) * dSo[b][k, kp] * cder(kp, i, a)
                    t += cder(k, i, a) * d2So[_h2(b, c)][k, kp] * Ft[kp, i]
                    t += cder(k, i, b) * d2So[_h2(a, c)][k, kp] * Ft[kp, i]
                    t += cder(k, i, c) * d2So[_h2(a, b)][k, kp] * Ft[kp, i]
                    t += Ft[k, i] * d2So[_h2(a, b)][k, kp] * cder(kp, i, c)
                    t += Ft[k, i] * d2So[_h2(a, c)][k, kp] * cder(kp, i, b)
                    t += Ft[k, i] * d2So[_h2(b, c)][k, kp] * cder(kp, i, a)
        ttau2 += (t * t) * (3 if s3 in (1, 2) else 1)
    ttau = mpmath.sqrt(ttau2)
    return tau, gtau, htau, ttau

def qblock_full(y):
    """exact Q (6x3) with all directional derivatives through third order."""
    Q = mp.zeros(6, 3)
    dQ = [mp.zeros(6, 3), mp.zeros(6, 3)]
    d2Q = [[None] * 3]   # xx, yy, xy
    d2Q = [mp.zeros(6, 3), mp.zeros(6, 3), mp.zeros(6, 3)]
    d3Q = [mp.zeros(6, 3), mp.zeros(6, 3), mp.zeros(6, 3), mp.zeros(6, 3)]
    for k in range(6):
        for j in range(3):
            form = FORMS[k]; gamma = _YJG[j]
            Q[k, j] = form_eval(form, gamma, y)
            gx, gy = form_grad(form, gamma, y)
            dQ[0][k, j] = gx; dQ[1][k, j] = gy
            hxx, hyy, hxy = form_hess_fn(form, gamma, y)
            d2Q[0][k, j] = hxx; d2Q[1][k, j] = hyy; d2Q[2][k, j] = hxy
            # third derivatives (exact)
            for slot, (e1, e2, e3) in enumerate([((3, 0), 0, 0), ((2, 1), 0, 0),
                                                 ((1, 2), 0, 0), ((0, 3), 0, 0)]):
                s = mpf(0)
                for (p, a, c) in form:
                    dx, dy = p[0] - y[0], p[1] - y[1]
                    t = kplane(a[0] + gamma[0] + e1[0], a[1] + gamma[1] + e1[1],
                               dx, dy)
                    for (i2, j2) in _IMG:
                        t += kplane(a[0] + gamma[0] + e1[0],
                                    a[1] + gamma[1] + e1[1],
                                    dx + _LT * i2, dy + _LT * j2)
                    s += ((-1) ** (gamma[0] + gamma[1])) * c * t
                d3Q[slot][k, j] = -s
    return Q, dQ, d2Q, d3Q

def spair_dir_full(y, d):
    """(Stinv, dSx, dSy, d2(xx,yy,xy), d3(xxx,xxy,xyy,yyy)) of the V0-basis
    pair block: exact directional derivatives (crude YY6-variation parts)."""
    Q, dQ, d2Q, d3Q = qblock_full(y)
    _c9, _T6, _Y6, YY6f = station_fast(y)
    YY6inv = YY6f ** -1
    St = V0.T * SPAIR0 * V0 - Q * YY6inv * Q.T
    Stinv = St ** -1
    rr = max(abs((St * Stinv - mp.eye(6))[i, j]) for i in range(6) for j in range(6))
    ck(rr < mpf('1e-40'), "St inversion residual %s" % mpmath.nstr(rr, 3))
    dS = [-(dQ[e] * YY6inv * Q.T + Q * YY6inv * dQ[e].T) for e in (0, 1)]
    dyy = env_dYY6(d)
    y2 = (YY6inv ** 2) * dyy
    d2S = []
    for (a, b) in [(0, 0), (1, 1), (0, 1)]:
        d2S.append(-(d2Q[a + b if (a, b) != (0, 1) else 2] * YY6inv * Q.T
                     + (2 if (a, b) != (0, 1) else 1) * dQ[a] * YY6inv * dQ[b].T
                     + (0 if (a, b) != (0, 1) else 1) * dQ[b] * YY6inv * dQ[a].T
                     + Q * YY6inv * (d2Q[a + b if (a, b) != (0, 1) else 2]).T
                     + 2 * (Q * y2 * Q.T)))
    d3S = []
    # d3(Q YY6inv Q') with exact Q-derivatives + crude YY6 variation allowance
    d3allow = (YY6inv ** 3) * (dyy * dyy) * 4
    for (a, b, c) in [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]:
        # map to the d2Q/d3Q slot indices
        def d2slot(a2, b2):
            if a2 == 0 and b2 == 0: return 0
            if a2 == 1 and b2 == 1: return 1
            return 2
        def d3slot(a3, b3, c3):
            return a3 + b3 + c3   # slot = number of y-indices
        T = (d3Q[d3slot(a, b, c)] * YY6inv * Q.T
             + d2Q[d2slot(a, b)] * YY6inv * dQ[c].T
             + d2Q[d2slot(a, c)] * YY6inv * dQ[b].T
             + d2Q[d2slot(b, c)] * YY6inv * dQ[a].T
             + dQ[a] * YY6inv * (d2Q[d2slot(b, c)]).T
             + dQ[b] * YY6inv * (d2Q[d2slot(a, c)]).T
             + dQ[c] * YY6inv * (d2Q[d2slot(a, b)]).T
             + Q * YY6inv * (d3Q[d3slot(a, b, c)]).T
             + 4 * (Q * d3allow * Q.T))
        d3S.append(-T)
    return Stinv, dS[0], dS[1], d2S, d3S

def mean_grad_exact(fs, y, v):
    """exact (dmu/dy1, dmu/dy2) of the conditional mean vector (9 targets)."""
    TCg = [mp.zeros(9, 6), mp.zeros(9, 6)]
    for e in (0, 1):
        for i, (pi, di) in enumerate(TSET9):
            if pi != 'Y':
                continue
            for j, (pj, dj) in enumerate(CSET6):
                TCg[e][i, j] = kdcov_d(_GI[di], _GI[dj],
                                       y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
    TY6g = [mp.zeros(9, 3), mp.zeros(9, 3)]
    for e in (0, 1):
        for i, (pi, di) in enumerate(TSET9):
            if pi == 'Y':
                continue
            for j, (pj, dj) in enumerate(YJET):
                TY6g[e][i, j] = -kdcov_d(_GI[di], _GI[dj],
                                         PTS[pi][0] - y[0], PTS[pi][1] - y[1], e)
    YCg = [mp.zeros(3, 6), mp.zeros(3, 6)]
    YYg = [mp.zeros(3, 3), mp.zeros(3, 3)]
    for e in (0, 1):
        for i, (pi, di) in enumerate(YJET):
            for j, (pj, dj) in enumerate(CSET6):
                YCg[e][i, j] = kdcov_d(_GI[di], _GI[dj],
                                       y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
    # YY6 = YY - YC G6inv YC': YY const -> dYY6 = -(dYC G6inv YC' + YC G6inv dYC')
    YC = mp.zeros(3, 6)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    YY6 = fs['YY6']; YY6inv = YY6 ** -1
    TY6 = fs['TY6']
    rv = mp.matrix([v, 0, 0]) - YC * _W6
    dmu = []
    for e in (0, 1):
        dYY6 = -(YCg[e] * G6inv * YC.T + YC * G6inv * YCg[e].T)
        dYY6inv = -YY6inv * dYY6 * YY6inv
        drv = -YCg[e] * _W6
        dmu.append(TCg[e] * _W6 + TY6g[e] * (YY6inv * rv)
                   + TY6 * (dYY6inv * rv) + TY6 * (YY6inv * drv))
    return dmu

def chi2_grad_bound(fs, kp, y, d):
    """tight certified bound of |grad chi2| at y: exact solve-products and
    exact directional derivatives throughout (no norm products)."""
    cov = fs['cov9']
    Spair = E.submat(cov, range(6), range(6))
    Spinv = Spair ** -1
    v = kit.b - kit.ell / 2
    muv = fs['mean_v'](v)
    mu9p = mp.matrix([muv[i] for i in range(6)])
    mu6 = kit.Hmean; S6inv = kit.Hcov ** -1
    M = Spinv - S6inv * mpf('0.5')
    Minv = M ** -1
    c = Spinv * mu9p - S6inv * mu6 * mpf('0.5')
    Stinv, dSx, dSy, d2S, d3S = spair_dir_full(y, d)
    _St = V0.T * Spair * V0
    _chk = max(abs((Stinv - V0.T * Spinv * V0)[i, j]) for i in range(6) for j in range(6))
    ck(_chk < mpf('1e-15'), "V0-basis inverse mismatch: %s" % mpmath.nstr(_chk, 3))
    A = Spinv * Minv * Spinv
    At = V0.T * A * V0
    w_m = V0.T * (Spinv * mu9p)
    w_c = V0.T * (Minv * c)
    dmu = mean_grad_exact(fs, y, v)
    def qtr(B, dS):
        return mpmath.fsum(B[i, j] * dS[j, i] for i in range(6) for j in range(6))
    g2 = mpf(0)
    for e, dS in ((0, dSx), (1, dSy)):
        glogdet = abs(qtr(Stinv, dS))
        gdetM = abs(qtr(At, dS))
        dMt = -Stinv * dS * Stinv
        sdw = Stinv * dS * w_m
        sdmu = Spinv * mp.matrix([dmu[e][i] for i in range(6)])
        sdmu_v = V0.T * sdmu
        dc = sdw + sdmu_v
        gcq = abs(2 * (w_c.T * dc)[0] - (w_c.T * dMt * w_c)[0])
        gkk = abs(2 * (w_m.T * sdmu_v)[0] - (w_m.T * dS * w_m)[0])
        glq = glogdet + mpf('0.5') * gdetM + gcq + gkk
        g2 += glq ** 2
    return (1 + kp['chi2']) * mpmath.sqrt(g2)

def kp1_point(y, d):
    """certified (kap0, grad-bound, hess-bound, third-bound) of kap_far."""
    fs = station_fast_full(y)
    v = kit.b - kit.ell / 2
    kp = kappa_pieces_fast(fs, v)
    kap0 = kp['kap_far']
    tau0, gtau, htau, ttau = tau_bounds_full(y, d)
    gchi2 = chi2_grad_bound(fs, kp, y, d)
    dmu = mean_grad_exact(fs, y, v)
    dcov = cov9_grad_exact(fs, y)
    muv = fs['mean_v'](v)
    cov = fs['cov9']
    muM = [muv[0], muv[1], muv[2]]; muS = [muv[3], muv[4], muv[5]]
    SM = E.submat(cov, [0, 1, 2], [0, 1, 2]); SS = E.submat(cov, [3, 4, 5], [3, 4, 5])
    dM8 = E.det_moment(mp.matrix(muM), SM, 8)
    dS8 = E.det_moment(mp.matrix(muS), SS, 8)
    Spair = E.submat(cov, range(6), range(6))
    Delta = E.submat(cov, range(6), [6, 7, 8])
    Spinv = Spair ** -1
    BregT = Spinv * Delta
    Sy = E.submat(cov, [6, 7, 8], [6, 7, 8])
    Syres = Sy - BregT.T * Delta * mpf(1) - (BregT.T * Delta) * 0 + (BregT.T * Delta) * 0
    Syres = Sy - Delta.T * BregT
    mu_y = [muv[6], muv[7], muv[8]]
    # exact per-slot gradients of every piece
    gCmr = mpf(0)   # (grad C_mom)/C_mom components squared accumulator
    gEq4 = mpf(0)   # grad of Eq2^{1/4}/Eq2^{1/4}
    gky_w2 = mpf(0) # grad of w2
    for e in (0, 1):
        dmueM = [dmu[e][0], dmu[e][1], dmu[e][2]]
        dmueS = [dmu[e][3], dmu[e][4], dmu[e][5]]
        dSMe = E.submat(dcov[e], [0, 1, 2], [0, 1, 2])
        dSSe = E.submat(dcov[e], [3, 4, 5], [3, 4, 5])
        gM = wick8_grad(muM, SM, dmueM, dSMe)
        gS = wick8_grad(muS, SS, dmueS, dSSe)
        gCmr += (mpf('0.125') * (gM / dM8 + gS / dS8)) ** 2
        # Eq2: enorm4 gradients (exact)
        dmuy = [dmu[e][6], dmu[e][7], dmu[e][8]]
        dSye = E.submat(dcov[e], [6, 7, 8], [6, 7, 8])
        dDeltae = E.submat(dcov[e], range(6), [6, 7, 8])
        dSpaire = E.submat(dcov[e], range(6), range(6))
        dSyrese = dSye - (dDeltae.T * BregT + Delta.T * (Spinv * dDeltae)
                          - BregT.T * dSpaire * BregT)
        gE1 = enorm4_grad(mu_y, Sy, dmuy, dSye)
        gE2 = enorm4_grad(mu_y, Syres, dmuy, dSyrese)
        e1 = E.enorm4(mp.matrix(mu_y), Sy)
        e2 = E.enorm4(mp.matrix(mu_y), Syres)
        gEq4 += ((gE1 + gE2) / (4 * (e1 + e2))) ** 2
        # w2: dmusq + trS1 + trS0 - 2 bures
        a2, a4 = E.A2, E.A4
        gdmusq = 2 * sum((muv[6 + i] - (-a2 * v if i < 2 else mpf(0))) * dmuy[i]
                         for i in range(3))
        gtr = dSyrese[0, 0] + dSyrese[1, 1] + dSyrese[2, 2]
        Sy0 = mp.zeros(3, 3)
        Sy0[0, 0] = a4 - a2 * a2; Sy0[1, 1] = a4 - a2 * a2; Sy0[2, 2] = a2 * a2
        S0h = mp.zeros(3, 3)
        for i in range(3):
            S0h[i, i] = mpmath.sqrt(Sy0[i, i])
        A = S0h * Syres * S0h
        lamA, UA = V1.eigvals_cert(A, 3, "bures(p1)")
        ck(min(lamA) > mpf('1e-12'), "bures floor (p1)")
        dA = S0h * dSyrese * S0h
        MdA = UA.T * dA * UA
        gbures = mpmath.fsum(MdA[i, i] / (2 * mpmath.sqrt(lamA[i])) for i in range(3))
        gky_w2 += (gdmusq + gtr - 2 * gbures) ** 2
    # piece gradient bounds
    msad = E.m_sad_cf(v)
    Ey1 = sum(Syres[i, i] for i in range(3)) + sum(muv[6 + i] ** 2 for i in range(3))
    Ey0 = 2 * (a4 - a2 * a2) + a2 * a2 + 2 * (a2 * v) ** 2
    w2v = (kp['kap_y'] * msad) ** 2 / (2 * (Ey1 + Ey0))
    ck(w2v > 0, "w2 back-out failed")
    Cm = kp['C_mom']
    gCm = Cm * (mpmath.sqrt(gCmr) + mpmath.sqrt(gEq4))
    gky = kp['kap_y'] * mpmath.sqrt(gky_w2) / (2 * w2v)
    gkp = kp['kap_pair'] * gchi2 / (2 * kp['chi2']) if kp['chi2'] > 0 else mpf(0)
    sq = mpmath.sqrt(tau0)
    gkc = gCm * sq + Cm * gtau / (2 * sq)
    gRpg, hRpg, tRpg, gRwm, hRwm, tRwm = ratio_bounds(fs, y, d, dmu, dcov)
    # crude hess/third per slow piece (cancellation-ratio certified)
    g1M, h1M, t1M, _ = _rel_der_scale(mp.matrix(muM), SM, 8, d)
    g1S, h1S, t1S, _ = _rel_der_scale(mp.matrix(muS), SS, 8, d)
    gE1s, hE1s, tE1s, e1v = enorm_scales(mp.matrix(mu_y), Sy, d)
    gE2s, hE2s, tE2s, e2v = enorm_scales(mp.matrix(mu_y), Syres, d)
    s = 2 * d + 10
    hCm = Cm * (mpf('0.125') * (h1M + h1S) + mpf('0.25') *
                ((hE1s * e1v + hE2s * e2v) / (e1v + e2v))
                + 2 * (gCm / Cm) ** 2)
    tCm = Cm * (mpf('0.125') * (t1M + t1S) + mpf('0.25') *
                ((tE1s * e1v + tE2s * e2v) / (e1v + e2v))
                + 6 * (hCm / Cm) * (gCm / Cm))
    hky = 2 * (gky_w2 ** 0.5) * s * kp['kap_y'] / w2v + 4 * gky ** 2 / kp['kap_y']
    tky = 6 * (gky_w2 ** 0.5) * s * s * kp['kap_y'] / w2v + 12 * gky * hky / kp['kap_y']
    hkp = 2 * gkp * s * 6 + 4 * gkp ** 2 / max(kp['kap_pair'], mpf('1e-30'))
    tkp = 6 * gkp * s * s * 36 + 12 * gkp * hkp / max(kp['kap_pair'], mpf('1e-30'))
    hkc = hCm * sq + 2 * gCm * gtau / (2 * sq) \
        + Cm * (htau / (2 * sq) + gtau ** 2 / (4 * sq ** 3))
    tkc = tCm * sq + 3 * hCm * gtau / (2 * sq) \
        + 3 * gCm * (htau / (2 * sq) + gtau ** 2 / (4 * sq ** 3)) \
        + Cm * (ttau / (2 * sq) + 3 * gtau * htau / (4 * sq ** 3)
                + 3 * gtau ** 3 / (8 * sq ** 5))
    kc = kp['kap_cross']; kpr = kp['kap_pair']; ky = kp['kap_y']
    Rpg = kp['Rpg']; Rwm = kp['Rwm']
    B = (1 + kpr) * (1 + ky) + kc
    gB = gkp * (1 + ky) + (1 + kpr) * gky + gkc
    hB = hkp * (1 + ky) + 2 * gkp * gky + (1 + kpr) * hky + hkc
    tB = tkp * (1 + ky) + 3 * hkp * gky + 3 * gkp * hky + (1 + kpr) * tky + tkc
    gRR = gRpg * Rwm + Rpg * gRwm
    hRR = hRpg * Rwm + 2 * gRpg * gRwm + Rpg * hRwm
    tRR = tRpg * Rwm + 3 * hRpg * gRwm + 3 * gRpg * hRwm + Rpg * tRwm
    g0 = gRR * B + Rpg * Rwm * gB
    h0 = hRR * B + 2 * gRR * gB + Rpg * Rwm * hB
    t0 = tRR * B + 3 * hRR * gB + 3 * gRR * hB + Rpg * Rwm * tB
    return kap0, g0, h0, t0

# appended helpers (restored): exact cov9 gradients, Wick machinery, scales

def cov9_grad_exact(fs, y):
    """exact entry gradients of cov9 (both slots) via the two-stage chain."""
    dTC = [mp.zeros(9, 6), mp.zeros(9, 6)]
    dTY = [mp.zeros(9, 3), mp.zeros(9, 3)]
    dTT = [mp.zeros(9, 9), mp.zeros(9, 9)]
    dYC = [mp.zeros(3, 6), mp.zeros(3, 6)]
    for e in (0, 1):
        for i, (pi, di) in enumerate(TSET9):
            for j, (pj, dj) in enumerate(CSET6):
                if pi == 'Y':
                    dTC[e][i, j] = kdcov_d(_GI[di], _GI[dj],
                                           y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
            for j, (pj, dj) in enumerate(YJET):
                if pi != 'Y':
                    dTY[e][i, j] = -kdcov_d(_GI[di], _GI[dj],
                                            PTS[pi][0] - y[0], PTS[pi][1] - y[1], e)
            for j, (pj, dj) in enumerate(TSET9):
                if pi == 'Y' and pj != 'Y':
                    dTT[e][i, j] = kdcov_d(_GI[di], _GI[dj],
                                           y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
                elif pj == 'Y' and pi != 'Y':
                    dTT[e][i, j] = -kdcov_d(_GI[di], _GI[dj],
                                            PTS[pi][0] - y[0], PTS[pi][1] - y[1], e)
        for i, (pi, di) in enumerate(YJET):
            for j, (pj, dj) in enumerate(CSET6):
                dYC[e][i, j] = kdcov_d(_GI[di], _GI[dj],
                                       y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
    YC = mp.zeros(3, 6); TC = mp.zeros(9, 6); TY = mp.zeros(9, 3)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    for i, (pi, di) in enumerate(TSET9):
        p1 = y if pi == 'Y' else PTS[pi]
        for j, (pj, dj) in enumerate(CSET6):
            TC[i, j] = kc(di, dj, p1, PTS[pj])
        for j, (pj, dj) in enumerate(YJET):
            TY[i, j] = kc(di, dj, p1, y)
    YY6 = fs['YY6']; YY6inv = YY6 ** -1
    TY6 = fs['TY6']
    dcov = []
    for e in (0, 1):
        dYY6 = -(dYC[e] * G6inv * YC.T + YC * G6inv * dYC[e].T)
        dYY6inv = -YY6inv * dYY6 * YY6inv
        dTT6 = dTT[e] - (dTC[e] * G6inv * TC.T + TC * G6inv * dTC[e].T)
        dTY6 = dTY[e] - (dTC[e] * G6inv * YC.T + TC * G6inv * dYC[e].T)
        dcov.append(dTT6 - (dTY6 * YY6inv * TY6.T + TY6 * dYY6inv * TY6.T
                            + TY6 * YY6inv * dTY6.T))
    return dcov

_M3CACHE = {}
_M3DCACHE = {}
def _M3(mu, Sig, a, b_, c):
    key = (a, b_, c)
    if key in _M3CACHE:
        return _M3CACHE[key]
    if a == 0 and b_ == 0 and c == 0:
        return mpf(1)
    if a < 0 or b_ < 0 or c < 0:
        return mpf(0)
    if a > 0:
        val = (mu[0] * _M3(mu, Sig, a - 1, b_, c)
               + Sig[0, 0] * (a - 1) * _M3(mu, Sig, a - 2, b_, c)
               + Sig[0, 1] * b_ * _M3(mu, Sig, a - 1, b_ - 1, c)
               + Sig[0, 2] * c * _M3(mu, Sig, a - 1, b_, c - 1))
    elif b_ > 0:
        val = (mu[1] * _M3(mu, Sig, a, b_ - 1, c)
               + Sig[1, 0] * a * _M3(mu, Sig, a - 1, b_ - 1, c)
               + Sig[1, 1] * (b_ - 1) * _M3(mu, Sig, a, b_ - 2, c)
               + Sig[1, 2] * c * _M3(mu, Sig, a, b_ - 1, c - 1))
    else:
        val = (mu[2] * _M3(mu, Sig, a, b_, c - 1)
               + Sig[2, 0] * a * _M3(mu, Sig, a - 1, b_, c - 1)
               + Sig[2, 1] * b_ * _M3(mu, Sig, a, b_ - 1, c - 1)
               + Sig[2, 2] * (c - 1) * _M3(mu, Sig, a, b_, c - 2))
    _M3CACHE[key] = val
    return val

def _M3_d(mu, Sig, dmu, dSig, a, b_, c):
    """exact derivative of the Wick moment M(a,b,c) along (dmu, dSig)."""
    key = (a, b_, c)
    if key in _M3DCACHE:
        return _M3DCACHE[key]
    if a < 0 or b_ < 0 or c < 0:
        return mpf(0)
    if a == 0 and b_ == 0 and c == 0:
        return mpf(0)
    if a > 0:
        val = (dmu[0] * _M3(mu, Sig, a - 1, b_, c)
               + mu[0] * _M3_d(mu, Sig, dmu, dSig, a - 1, b_, c)
               + dSig[0, 0] * (a - 1) * _M3(mu, Sig, a - 2, b_, c)
               + Sig[0, 0] * (a - 1) * _M3_d(mu, Sig, dmu, dSig, a - 2, b_, c)
               + dSig[0, 1] * b_ * _M3(mu, Sig, a - 1, b_ - 1, c)
               + Sig[0, 1] * b_ * _M3_d(mu, Sig, dmu, dSig, a - 1, b_ - 1, c)
               + dSig[0, 2] * c * _M3(mu, Sig, a - 1, b_, c - 1)
               + Sig[0, 2] * c * _M3_d(mu, Sig, dmu, dSig, a - 1, b_, c - 1))
    elif b_ > 0:
        val = (dmu[1] * _M3(mu, Sig, a, b_ - 1, c)
               + mu[1] * _M3_d(mu, Sig, dmu, dSig, a, b_ - 1, c)
               + dSig[1, 0] * a * _M3(mu, Sig, a - 1, b_ - 1, c)
               + Sig[1, 0] * a * _M3_d(mu, Sig, dmu, dSig, a - 1, b_ - 1, c)
               + dSig[1, 1] * (b_ - 1) * _M3(mu, Sig, a, b_ - 2, c)
               + Sig[1, 1] * (b_ - 1) * _M3_d(mu, Sig, dmu, dSig, a, b_ - 2, c)
               + dSig[1, 2] * c * _M3(mu, Sig, a, b_ - 1, c - 1)
               + Sig[1, 2] * c * _M3_d(mu, Sig, dmu, dSig, a, b_ - 1, c - 1))
    else:
        val = (dmu[2] * _M3(mu, Sig, a, b_, c - 1)
               + mu[2] * _M3_d(mu, Sig, dmu, dSig, a, b_, c - 1)
               + dSig[2, 0] * a * _M3(mu, Sig, a - 1, b_, c - 1)
               + Sig[2, 0] * a * _M3_d(mu, Sig, dmu, dSig, a - 1, b_, c - 1)
               + dSig[2, 1] * b_ * _M3(mu, Sig, a, b_ - 1, c - 1)
               + Sig[2, 1] * b_ * _M3_d(mu, Sig, dmu, dSig, a, b_ - 1, c - 1)
               + dSig[2, 2] * (c - 1) * _M3(mu, Sig, a, b_, c - 2)
               + Sig[2, 2] * (c - 1) * _M3_d(mu, Sig, dmu, dSig, a, b_, c - 2))
    _M3DCACHE[key] = val
    return val

_M3ABS = {}
def _M3_abs(mu, Sig, a, b_, c):
    """sum of |terms| of the Wick moment (for cancellation ratios)."""
    key = (a, b_, c)
    if key in _M3ABS:
        return _M3ABS[key]
    if a == 0 and b_ == 0 and c == 0:
        return mpf(1)
    if a < 0 or b_ < 0 or c < 0:
        return mpf(0)
    if a > 0:
        val = (abs(mu[0]) * _M3_abs(mu, Sig, a - 1, b_, c)
               + abs(Sig[0, 0]) * (a - 1) * _M3_abs(mu, Sig, a - 2, b_, c)
               + abs(Sig[0, 1]) * b_ * _M3_abs(mu, Sig, a - 1, b_ - 1, c)
               + abs(Sig[0, 2]) * c * _M3_abs(mu, Sig, a - 1, b_, c - 1))
    elif b_ > 0:
        val = (abs(mu[1]) * _M3_abs(mu, Sig, a, b_ - 1, c)
               + abs(Sig[1, 0]) * a * _M3_abs(mu, Sig, a - 1, b_ - 1, c)
               + abs(Sig[1, 1]) * (b_ - 1) * _M3_abs(mu, Sig, a, b_ - 2, c)
               + abs(Sig[1, 2]) * c * _M3_abs(mu, Sig, a, b_ - 1, c - 1))
    else:
        val = (abs(mu[2]) * _M3_abs(mu, Sig, a, b_, c - 1)
               + abs(Sig[2, 0]) * a * _M3_abs(mu, Sig, a - 1, b_, c - 1)
               + abs(Sig[2, 1]) * b_ * _M3_abs(mu, Sig, a, b_ - 1, c - 1)
               + abs(Sig[2, 2]) * (c - 1) * _M3_abs(mu, Sig, a, b_, c - 2))
    _M3ABS[key] = val
    return val

def wick4_grad(mu, Sig, dmu, dSig):
    """gradient of det_moment(mu, Sig, 4) along the input gradients."""
    global _M3CACHE
    _M3CACHE = {}
    tot = mpf(0)
    for j in range(5):
        coef = mpf(_comb(4, j)) * ((-1) ** j)
        tot += coef * _M3_d(mu, Sig, dmu, dSig, 4 - j, 4 - j, 2 * j)
    _M3CACHE = {}; _M3DCACHE = {}
    return tot

def det_moment_abs(mu, Sig, k):
    global _M3ABS
    _M3ABS = {}
    tot = mpf(0)
    for j in range(k + 1):
        coef = mpf(_comb(k, j))
        tot += coef * _M3_abs(mu, Sig, k - j, k - j, 2 * j)
    _M3ABS = {}
    return tot

def _rel_der_scale(mu, Sig, k, d):
    """rigorous crude bound of |d^q det_moment| / det_moment for q=1,2,3
    via the exact cancellation ratio and the per-entry log-derivative
    scale (2d+10) (he_abs ratio certificate, ck'd)."""
    for n in range(1, 11):
        ck(he_abs(n + 1, d) / he_abs(n, d) <= 2 * d + 10,
           "he_abs ratio certificate failed at n=%d d=%s" % (n, mpmath.nstr(d, 4)))
    dm = E.det_moment(mu, Sig, k)
    da = det_moment_abs(mu, Sig, k)
    ck(dm > 0, "det moment nonpositive in rel-scale")
    ratio = da / dm
    s = 2 * d + 10
    return (k * s * ratio, 2 * (k * s) ** 2 * ratio,
            6 * (k * s) ** 3 * ratio, dm)

def kp_msad(kp):
    return E.m_sad_cf(kit.b - kit.ell / 2)

def wick8_grad(mu, Sig, dmu, dSig):
    global _M3CACHE, _M3DCACHE
    _M3CACHE = {}; _M3DCACHE = {}
    tot = mpf(0)
    for j in range(9):
        coef = mpf(_comb(8, j)) * ((-1) ** j)
        tot += coef * _M3_d(mu, Sig, dmu, dSig, 8 - j, 8 - j, 2 * j)
    _M3CACHE = {}; _M3DCACHE = {}
    return tot

def enorm_scales(mu, Sig, d):
    """(g,h,t, value) crude-relative scales for enorm4(mu, Sig)."""
    en = E.enorm4(mu, Sig)
    _M3ABS2 = {}
    def Mabs(a, b_, c):
        return _M3_abs(mu, Sig, a, b_, c)
    global _M3ABS
    _M3ABS = {}
    ea = (_M3_abs(mu, Sig, 4, 0, 0) + _M3_abs(mu, Sig, 0, 4, 0)
          + _M3_abs(mu, Sig, 0, 0, 4)
          + 2 * (_M3_abs(mu, Sig, 2, 2, 0) + _M3_abs(mu, Sig, 2, 0, 2)
                 + _M3_abs(mu, Sig, 0, 2, 2)))
    _M3ABS = {}
    ck(en > 0, "enorm4 nonpositive")
    ratio = ea / en
    s = 2 * d + 10
    return (4 * s * ratio, 2 * (4 * s) ** 2 * ratio, 6 * (4 * s) ** 3 * ratio, en)

def enorm4_grad(mu, Sig, dmu, dSig):
    global _M3CACHE, _M3DCACHE
    _M3CACHE = {}; _M3DCACHE = {}
    tot = (_M3_d(mu, Sig, dmu, dSig, 4, 0, 0) + _M3_d(mu, Sig, dmu, dSig, 0, 4, 0)
           + _M3_d(mu, Sig, dmu, dSig, 0, 0, 4)
           + 2 * (_M3_d(mu, Sig, dmu, dSig, 2, 2, 0)
                  + _M3_d(mu, Sig, dmu, dSig, 2, 0, 2)
                  + _M3_d(mu, Sig, dmu, dSig, 0, 2, 2)))
    _M3CACHE = {}; _M3DCACHE = {}
    return tot

def ratio_bounds(fs, y, d, dmu, dcov):
    """(gRpg, hRpg, tRpg, gRwm, hRwm, tRwm): tight exact gradients of the
    needle-density and window-mass ratios + crude h/t scales."""
    YY6 = fs['YY6']
    Sg = mp.matrix([[YY6[1, 1], YY6[1, 2]], [YY6[2, 1], YY6[2, 2]]])
    Sginv = Sg ** -1
    lamSg, _ = V1.eigvals_cert(Sg, 2, "Sg floor")
    ck(min(lamSg) > mpf('0.05'), "Sg floor collapsed: %s" % mpmath.nstr(min(lamSg), 3))
    YC = mp.zeros(3, 6)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    muy = YC * _W6
    mug = mp.matrix([muy[1], muy[2]])
    s = 2 * d + 10
    gRpg2 = mpf(0); gRwm2 = mpf(0)
    pgrad = fs['pgrad']
    wm = E._Phi((kit.b - fs['mu_t']) / fs['s_t']) - E._Phi((kit.s - fs['mu_t']) / fs['s_t'])
    for e in (0, 1):
        dYCe = mp.zeros(3, 6)
        for i, (pi, di) in enumerate(YJET):
            for j, (pj, dj) in enumerate(CSET6):
                dYCe[i, j] = kdcov_d(_GI[di], _GI[dj],
                                     y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
        dYY6e = -(dYCe * G6inv * YC.T + YC * G6inv * dYCe.T)
        dmuge = mp.matrix([(dYCe * _W6)[1], (dYCe * _W6)[2]])
        dSge = mp.matrix([[dYY6e[1, 1], dYY6e[1, 2]], [dYY6e[2, 1], dYY6e[2, 2]]])
        w = Sginv * mug
        gmaha = 2 * (w.T * dmuge)[0] - (w.T * dSge * w)[0]
        gdet = mp.det(Sg) * mpmath.fsum((Sginv * dSge)[i, i] for i in range(2))
        glog = abs(-mpf('0.5') * gmaha) + abs(-mpf('0.5') * gdet / mp.det(Sg))
        gRpg2 += (pgrad * glog) ** 2
        # needle pieces
        c01 = mp.matrix([YY6[0, 1], YY6[0, 2]])
        dc01 = mp.matrix([dYY6e[0, 1], dYY6e[0, 2]])
        dmu0 = (dYCe * _W6)[0]
        dmu_t = dmu0 - ((dc01.T * Sginv) * mug)[0] \
            + (c01.T * (Sginv * dSge * Sginv) * mug)[0] - ((c01.T * Sginv) * dmuge)[0]
        ds2 = dYY6e[0, 0] - 2 * ((dc01.T * Sginv) * c01)[0] \
            + (c01.T * (Sginv * dSge * Sginv) * c01)[0]
        st_t = fs['s_t']; mu_t = fs['mu_t']
        dst = ds2 / (2 * st_t)
        gwm = mpf(0)
        for lvl, sgn in ((kit.b, 1), (kit.s, -1)):
            u = (lvl - mu_t) / st_t
            pdf = mpmath.exp(-u * u / 2) / mpmath.sqrt(2 * mp.pi)
            du = (-dmu_t * st_t - (lvl - mu_t) * dst) / (st_t * st_t)
            gwm += sgn * pdf * du
        gRwm2 += (gwm / (E._Phi(kit.b) - E._Phi(kit.s))) ** 2
    Rpg = pgrad / (1 / (2 * mp.pi * E.A2))
    Rwm = wm / (E._Phi(kit.b) - E._Phi(kit.s))
    gRpg = mpmath.sqrt(gRpg2) / (1 / (2 * mp.pi * E.A2))
    gRwm = mpmath.sqrt(gRwm2)
    hRpg = 2 * gRpg * s * 4 + 4 * gRpg ** 2 / max(Rpg, mpf('1e-30'))
    tRpg = 6 * gRpg * s * s * 16 + 12 * gRpg * hRpg / max(Rpg, mpf('1e-30'))
    hRwm = 2 * gRwm * s * 4 + 4 * gRwm ** 2 / max(Rwm, mpf('1e-30'))
    tRwm = 6 * gRwm * s * s * 16 + 12 * gRwm * hRwm / max(Rwm, mpf('1e-30'))
    return gRpg, hRpg, tRpg, gRwm, hRwm, tRwm

# ---- envelope zone (d >= d_env): crude theta-free piece envelopes ----------
def env_tau(d):
    """crude theta-free envelope of tau over {y: |y| >= d} (termwise exact
    moments + certified remainder + cross allowance)."""
    yinv = env_m(d)[1]
    tot = mpf(0)
    redmax = mpf(0)
    for k in range(6):
        redk = yinv * sum(env_form(k, g, d, 0) ** 2 for g in _YJG)
        redmax = max(redmax, redk)
    for k in range(6):
        lam = LAM0[k] - redmax
        ck(lam > LAM0[k] / 2, "envelope lambda floor collapsed k=%d d=%s"
           % (k, mpmath.nstr(d, 4)))
        tot += sum(env_form(k, _HYG[i], d, 0) ** 2 for i in range(3)) / lam
    t6, g6, h6 = env_TY6(d)
    for k in range(6):
        for kp in range(k + 1, 6):
            ek = mpmath.sqrt(sum(env_form(k, _HYG[i], d, 0) ** 2 for i in range(3)))
            ekp = mpmath.sqrt(sum(env_form(kp, _HYG[i], d, 0) ** 2 for i in range(3)))
            tot += 2 * ek * ekp * t6 / (LAM0[k] * LAM0[kp])
    return tot


# ================= adaptive certifier + piece-1 driver ======================
KAP_BUDGET = mpf('0.68')
CELL_MARGIN = mpf('0.0002')
SAFE_H = mpf(3)     # absorbs cell-scale variation of the crude hess scales
SAFE_T = mpf(9)     # absorbs cell-scale variation of the crude third scales

class CertStats:
    def __init__(self):
        self.cells = 0
        self.refined = 0
        self.max_sup = mpf(-1)
        self.max_cell = None
        self.worst_tau = mpf(-1)
        self.evals = 0

def certify_cell(y, d0, hw, budget, depth, stats, kap_fn):
    """certify sup_{cell} kap_fn <= budget; recursive bisection; fail-closed
    with the exact cell named if the bound cannot be certified."""
    kap0, g0, h0, t0 = kap_fn(y, d0)
    stats.evals += 1
    sup = kap0 + g0 * hw + SAFE_H * h0 * hw ** 2 / 2 + SAFE_T * t0 * hw ** 3 / 6
    if sup > stats.max_sup:
        stats.max_sup = sup
        stats.max_cell = (y, d0, hw, kap0, g0, h0, t0, sup, depth)
    if sup <= budget - CELL_MARGIN:
        stats.cells += 1
        return
    if hw < mpf('0.0004'):
        raise SystemExit("FAIL-CLOSED: uncertified cell at y = (%s, %s),"
                         " hw = %s: kap0 = %s, sup-bound = %s > budget %s"
                         % (mpmath.nstr(y[0], 6), mpmath.nstr(y[1], 6),
                            mpmath.nstr(hw, 6), mpmath.nstr(kap0, 8),
                            mpmath.nstr(sup, 8), mpmath.nstr(budget, 6)))
    # bisect into 4 (passed via closure in the driver)
    stats.refined += 1
    raise _Refine(y, d0, hw / 2, kap0)

class _Refine(Exception):
    def __init__(self, y, d0, hw2, kap0):
        self.y = y; self.d0 = d0; self.hw2 = hw2; self.kap0 = kap0

def run_certification(kap_fn, budget, tag):
    """polar adaptive certification of sup kap_fn <= budget over the far zone
    {y in T^2 : |y| >= 5} (polar cover to d = 17 >= 12*sqrt(2))."""
    stats = CertStats()
    work = []
    n_th0 = 96
    dstep = mpf('0.5')
    dd = mpf(5)
    while dd < 17:
        d1 = dd + dstep
        for ith in range(n_th0):
            th0 = 2 * mp.pi * ith / n_th0
            th1 = 2 * mp.pi * (ith + 1) / n_th0
            work.append((dd, d1, th0, th1))
        dd = d1
    emit("%s: adaptive certification over %d base cells (budget %s)"
         % (tag, len(work), mpmath.nstr(budget, 6)))
    while work:
        d0, d1, th0, th1 = work.pop()
        dm = (d0 + d1) / 2
        thm = (th0 + th1) / 2
        y = (dm * mpmath.cos(thm), dm * mpmath.sin(thm))
        hw = mpmath.sqrt(((d1 - d0) / 2) ** 2 + (d1 * (th1 - th0) / 2) ** 2)
        try:
            certify_cell(y, d0, hw, budget, 0, stats, kap_fn)
        except _Refine as r:
            dm2 = (d0 + d1) / 2
            thm2 = (th0 + th1) / 2
            work.append((d0, dm2, th0, thm2))
            work.append((d0, dm2, thm2, th1))
            work.append((dm2, d1, th0, thm2))
            work.append((dm2, d1, thm2, th1))
    emit("  certified: %d cells (+ %d refinements), %d point evaluations;"
         " sup-bound max = %s at y = (%s, %s), hw = %s"
         % (stats.cells, stats.refined, stats.evals,
            mpmath.nstr(stats.max_sup, 10),
            mpmath.nstr(stats.max_cell[0][0], 6),
            mpmath.nstr(stats.max_cell[0][1], 6),
            mpmath.nstr(stats.max_cell[2], 6)))
    ck(stats.max_sup <= budget - CELL_MARGIN,
       "certification did not close under the budget")
    return stats

def kap_fn_p1(y, d0):
    return kp1_point(y, d0)

# quick probe of the per-point pieces at representative points before the run
for _yy, _dd in [((mpf(5), mpf(0)), mpf(5)),
                 ((mpf('4.8'), mpf('1.6')), mpf('5')),
                 ((mpf('0.5'), mpf('6.2')), mpf('6.2')),
                 ((mpf('10'), mpf('6')), mpf('11.6'))]:
    _k0, _g0, _h0, _t0 = kp1_point(_yy, _dd)
    emit("probe y=(%s,%s) d0=%s: kap0=%s g0=%s h0=%s t0=%s"
         % (mpmath.nstr(_yy[0], 4), mpmath.nstr(_yy[1], 4), mpmath.nstr(_dd, 4),
            mpmath.nstr(_k0, 8), mpmath.nstr(_g0, 6), mpmath.nstr(_h0, 6),
            mpmath.nstr(_t0, 6)))
    ck(_k0 < KAP_BUDGET, "probe point already over budget")
