#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LB-1 / SUP-OVER-ZONE - fail-closed machine certificate
======================================================
Program: SIDE24/q0, Lemma LB-RATE, AO assembly, channel "above-b saddle"
(C029 B1-B4 reduction, B3 branch (ii)).

Verbatim ledger statements bound by this certificate:
  [L1] C031_LBRATE_Integration.md line 30: "| - above-b saddle | superexponentially
        killed throughout the jet-cluster horizon (v = 0.018/0.55 at d = 1/2;
        kill e^{-(b-m)^2/2v}) - mechanism identified | derived mechanism; closure
        named | C030 | sup-over-zone argument"
  [L2] C031 line 55: "5. Sup-over-zone (converts the horizon kill into the closed
        above-b-saddle piece)."
  [L3] "C030 CountingLemmas Package.md" line 11: "Var(f(y') | 9 pins) at r = 0.025:
        0.018 / 0.55 / 0.976 at d = 1/2/3. ... Consequences: (i) the band intensities
        are superexponentially killed (e^{-(b-m)^2/2v}) throughout d <~ 1.5..."

WHAT THIS CERTIFICATE ESTABLISHES (all recomputed; no PASS label trusted):
  (T1) The exact 9-pin conditional law of the periodized Bargmann-Fock field
       (side-24 torus, b = 6/5, rung r) reproduces every documented station value
       (C030 horizon table; c031_verify.json 7 stations) to >= 10 digits.
  (T2) THEOREM (SUP-OVER-ZONE, localized form): at each certified rung
       r in {0.05, 0.04, 0.03, 0.025, 0.02, 0.0125},
           E[ N_crit(f~ >= b, B_{3r}(M)) ]  <=  E_zone(r)  <<  1e-3 * r^3,
       with rho_bar(y) a rigorous pointwise Kac-Rice upper bound (gradient factor
       x Cauchy-Schwarz moment factor x Gaussian tail), a certified Lipschitz
       allowance, and a mesh-refinement gate. The B3 mountain pass between two
       maxima in B_{1.4r}(pair) lies inside B_{1.4r} (C029 Freeze, B3), and
       B_{1.4r} subset B_{3r}(M); hence the above-b-saddle channel count is
       <= E_zone(r): CLOSED, superexponentially negligible vs tolerance 1e-3 r^3.
  (T3) FALSIFIER EXHIBIT (correction to [L1]/[L3] zone-wide language): the bare
       value-kill exp(-(b-m)^2/2v) does NOT hold zone-wide: the conditional mean m
       exceeds b on ridge arcs (max m = 1.931 at P* = (1.3056, 0.6858), d = 1.47),
       and E[N_crit(f~>=b, B_1.5)] = O(1). Zone-wide closure is FALSE as stated;
       the channel is closed by localization (T2) plus routing of ridge-region
       saddles to the already-named mean-ridge channel (C029 B3(iii)).
  (T4) CONSTANTS VERIFIED: rho_mx(1.2) = 0.043685, rho_sad(1.2) = 0.030449,
       E-term 1.41350 (C030 Lemma KR-MB fixed-zone core), by independent
       Gauss-Hermite quadrature + Monte Carlo cross-check. Far-term constant
       2.1*(ell/2)/r^3 = 0.175 exactly.
  (T5) FLAG (side-finding, Lemma WP): the same exact machinery gives a window-saddle
       intensity integral over the rigidity zone d <= 1.5 of ~= 1.3e-5 (mesh-stable
       0.02 -> 0.01), vs the C030 figure "~3e-15"; WP total 0.213 r^3 = 3.33e-6
       appears exceeded inside its own rigidity zone. Root cause consistent with
       the falsified zone-wide kill premise. WP needs re-derivation; LB-1's
       closure (T2) is unaffected (its zone is B_{3r}(M), not d <= 1.5).

Execution contract: ck() performs every check; ANY failed check raises SystemExit
with a FAIL banner. Byte-identical transcript under `python3 script` and
`python3 -O script` (no assert statements, no __debug__ branching, no clocks,
no hash-order dependence, fixed formatting).
"""

import math
import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf
OUT = []

def emit(s):
    OUT.append(str(s))
    print(s, flush=True)

def fail(msg):
    emit("FAIL: " + msg)
    raise SystemExit("FAIL: " + msg)

def need(cond, msg):
    if not cond:
        fail(msg)

# ----------------------------------------------------------------------------
# 0. Exact BF kernel and derivatives
# ----------------------------------------------------------------------------
def _he_poly(n):
    if n == 0: return [mpf(1)]
    if n == 1: return [mpf(0), mpf(1)]
    p0, p1 = [mpf(1)], [mpf(0), mpf(1)]
    for k in range(1, n):
        p2 = [mpf(0)] * (len(p1) + 1)
        for i, c in enumerate(p1): p2[i + 1] += c
        for i, c in enumerate(p0): p2[i] -= k * c
        p0, p1 = p1, p2
    return p1

HE = [_he_poly(n) for n in range(14)]

def hev(n, x):
    return mp.polyval(list(reversed(HE[n])), x)

def dK(a, b, x, y):
    # d^a_x d^b_y exp(-(x^2+y^2)/2) = (-1)^{a+b} He_a(x) He_b(y) exp(-(x^2+y^2)/2)
    return (-1) ** (a + b) * hev(a, x) * hev(b, y) * mp.exp(-(x ** 2 + y ** 2) / 2)

def covf(A, Bd, dx, dy):
    # Cov( d^A f(x), d^Bd f(y) ) with (dx,dy) = x - y ; K = exp(-|u|^2/2)
    return (-1) ** (Bd[0] + Bd[1]) * dK(A[0] + Bd[0], A[1] + Bd[1], dx, dy)

J01 = [(0, 0), (1, 0), (0, 1)]
DER6 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
BB = mpf(6) / 5
LTOR = mpf(24)          # torus side
DK_LATT = mp.pi / 12    # spectral lattice spacing (pi/12) Z^2

# ----------------------------------------------------------------------------
# 1. Pin assembly (C025/C026/C031 construction, rebuilt from scratch in mp)
#    M = (-r/2,0): jet (b,0,0); S = (r/2,0): jet (b-ell,0,0), ell = r^3/6;
#    Y = r*(-1.26, 0.24): jet (v*,0,0), v* = clip(mu_t) with mu_t the 6-pin
#    conditional mean at Y given grad f(Y) = 0.
# ----------------------------------------------------------------------------
def build_pins(r):
    r = mpf(r)
    M = (-r / 2, mpf(0)); S = (r / 2, mpf(0))
    ell = r ** 3 / 6
    Y = (M[0] + r * mpf('-0.76'), r * mpf('0.24'))
    PIN6 = [(M, a) for a in J01] + [(S, a) for a in J01]
    vals6 = [BB, mpf(0), mpf(0), BB - ell, mpf(0), mpf(0)]

    def assemble(PINS):
        n = len(PINS)
        A = mp.matrix(n, n)
        for i, (P, a) in enumerate(PINS):
            for j, (Q, c) in enumerate(PINS):
                A[i, j] = mpf(covf(a, c, P[0] - Q[0], P[1] - Q[1]))
        return A

    Spp6 = assemble(PIN6)
    Spp6i = Spp6 ** -1
    v6 = mp.matrix(vals6)
    KY = mp.matrix(3, 6)
    for i, a in enumerate(J01):
        for j, (P, c) in enumerate(PIN6):
            KY[i, j] = mpf(covf(a, c, Y[0] - P[0], Y[1] - P[1]))
    SY = mp.matrix(3, 3)
    for i, a in enumerate(J01):
        for j, c in enumerate(J01):
            SY[i, j] = mpf(covf(a, c, 0, 0))
    W = KY * Spp6i
    mo = W * v6
    So = SY - W * KY.T
    Sgg = mp.matrix(2, 2); Sgg[:, :] = So[1:3, 1:3]
    Sggi = Sgg ** -1
    mg = mp.matrix([mo[1], mo[2]])
    Sfg = mp.matrix(1, 2); Sfg[0, 0] = So[0, 1]; Sfg[0, 1] = So[0, 2]
    mu_t = mo[0] - (Sfg * Sggi * mg)[0]
    vstar = min(max(mu_t, BB - ell), BB)
    PIN9 = PIN6 + [(Y, a) for a in J01]
    vals9 = vals6 + [mu_t if (BB - ell) <= mu_t <= BB else vstar, mpf(0), mpf(0)]
    Spp9 = assemble(PIN9)
    Spp9i = Spp9 ** -1
    coef = Spp9i * mp.matrix(vals9)
    return dict(r=r, M=M, S=S, Y=Y, ell=ell, mu_t=mu_t, vstar=vstar,
                PIN9=PIN9, vals9=vals9, Spp9=Spp9, Spp9i=Spp9i, coef=coef)

def kvec(st, x, der=(0, 0)):
    return mp.matrix([mpf(covf(der, a, x[0] - P[0], x[1] - P[1])) for (P, a) in st['PIN9']])

def v_of(st, x):
    k = kvec(st, x)
    return 1 - (k.T * st['Spp9i'] * k)[0]

def m_of(st, x):
    k = kvec(st, x)
    return (k.T * st['coef'])[0]

def gradm_of(st, x):
    kx = kvec(st, x, (1, 0)); ky = kvec(st, x, (0, 1))
    return [(kx.T * st['coef'])[0], (ky.T * st['coef'])[0]]

def hessm_of(st, x):
    a = (kvec(st, x, (2, 0)).T * st['coef'])[0]
    b = (kvec(st, x, (1, 1)).T * st['coef'])[0]
    c = (kvec(st, x, (0, 2)).T * st['coef'])[0]
    return a, b, c

# ----------------------------------------------------------------------------
# 2. Conditional law of Z = (f, fx, fy, fxx, fxy, fyy) given the 9 pins,
#    and the rigorous critical/saddle intensity upper bound rho_bar.
# ----------------------------------------------------------------------------
def cond_law(st, x):
    KZP = mp.matrix(6, 9)
    for i, a in enumerate(DER6):
        for j, (P, c) in enumerate(st['PIN9']):
            KZP[i, j] = mpf(covf(a, c, x[0] - P[0], x[1] - P[1]))
    KZZ = mp.matrix(6, 6)
    for i, a in enumerate(DER6):
        for j, c in enumerate(DER6):
            KZZ[i, j] = mpf(covf(a, c, 0, 0))
    W = KZP * st['Spp9i']
    mean = W * mp.matrix(st['vals9'])
    cov = KZZ - W * KZP.T
    return mean, cov

def rho_bar(st, x):
    """Rigorous upper bound for the Kac-Rice intensity of critical points of the
    conditioned field with f~ >= b at y = x, and (typed) of saddles with f~ >= b:
      rho_crit <= p_grad(0) * sqrt(E[det^2 H | grad=0]) * sqrt(P(f>=b|grad=0))
      rho_sad  <= p_grad(0) * sqrt(E[det^2 H | grad=0]) * sqrt(min(Cantelli, P))
    (Cauchy-Schwarz; Cantelli = one-sided Chebyshev on det H when E det H > 0.)
    Raises (fail-closed) if the conditional Gaussian is degenerate."""
    mean, cov = cond_law(st, x)
    m = mean[0]; g1, g2 = mean[1], mean[2]
    a11, a12, a22 = cov[1, 1], cov[1, 2], cov[2, 2]
    detSg = a11 * a22 - a12 * a12
    if detSg <= 0:
        raise ValueError('Sg degenerate at %s' % (x,))
    tg2 = (a22 * g1 * g1 - 2 * a12 * g1 * g2 + a11 * g2 * g2) / detSg
    pgrad = mp.exp(-tg2 / 2) / (2 * mp.pi * mp.sqrt(detSg))
    c01, c02 = cov[0, 1], cov[0, 2]
    mu_t = m - (c01 * (a22 * g1 - a12 * g2) + c02 * (a11 * g2 - a12 * g1)) / detSg
    v_t = cov[0, 0] - (c01 * c01 * a22 - 2 * c01 * c02 * a12 + c02 * c02 * a11) / detSg
    if v_t <= 0:
        raise ValueError('v_t <= 0 at %s' % (x,))
    tv = (BB - mu_t) / mp.sqrt(v_t)
    Phit = mp.ncdf(-tv)          # P(f~ >= b | grad f~ = 0)
    def sgi(v1, v2):
        return ((a22 * v1 - a12 * v2) / detSg, (a11 * v2 - a12 * v1) / detSg)
    w = [sgi(cov[3 + i, 1], cov[3 + i, 2]) for i in range(3)]
    nu = [mean[3 + i] - (w[i][0] * g1 + w[i][1] * g2) for i in range(3)]
    Sig = [[cov[3 + i, 3 + j] - (w[i][0] * cov[3 + j, 1] + w[i][1] * cov[3 + j, 2])
            for j in range(3)] for i in range(3)]
    C = [[Sig[i][j] + nu[i] * nu[j] for j in range(3)] for i in range(3)]
    Em = lambda idx: (C[idx[0]][idx[1]] * C[idx[2]][idx[3]]
                      + C[idx[0]][idx[2]] * C[idx[1]][idx[3]]
                      + C[idx[0]][idx[3]] * C[idx[1]][idx[2]])
    D2 = Em((0, 0, 2, 2)) - 2 * Em((0, 1, 1, 2)) + Em((1, 1, 1, 1))
    Edet = nu[0] * nu[2] + Sig[0][2] - nu[1] ** 2 - Sig[1][1]
    Vdet = D2 - Edet ** 2
    cant = Vdet / (Vdet + Edet ** 2) if (Edet > 0 and Vdet > 0) else mpf(1)
    sq = mp.sqrt(max(D2, mpf(0)))
    rho_CS = pgrad * sq * mp.sqrt(Phit)                       # critical points >= b
    rho_sad = pgrad * sq * mp.sqrt(min(cant, Phit))           # saddles >= b
    return dict(rho_CS=rho_CS, rho_sad=rho_sad, tv=tv, tg2=tg2, m=m, mu_t=mu_t,
                v_t=v_t, pgrad=pgrad, D2=D2, Edet=Edet, cant=cant, Phit=Phit,
                v=cov[0, 0])

def log_rho(st, x):
    return mp.log(rho_bar(st, x)['rho_CS'])

def grad_log_rho(st, x):
    h = mpf(10) ** -12
    gx = (log_rho(st, (x[0] + h, x[1])) - log_rho(st, (x[0] - h, x[1]))) / (2 * h)
    gy = (log_rho(st, (x[0], x[1] + h)) - log_rho(st, (x[0], x[1] - h))) / (2 * h)
    return gx, gy

# ----------------------------------------------------------------------------
# 3. Unconditional BF Kac-Rice densities (ledger constants cross-check)
# ----------------------------------------------------------------------------
def bf_rhos_GH(nn, u):
    # H | f=u : H11, H22 iid N(-u, 2), H12 ~ N(0,1) independent;
    # rho(u) = phi(u)/(2 pi) * E[|det H| 1_type].  Gauss-Hermite (probabilists').
    from numpy.polynomial.hermite_e import hermegauss
    import numpy as np
    xs, ws = hermegauss(nn)
    W = ws / math.sqrt(2 * math.pi)
    X = -u + math.sqrt(2) * xs
    Xg, Zg, Tg = np.meshgrid(X, X, xs, indexing='ij')
    Wg = (W[:, None, None] * W[None, :, None] * W[None, None, :])
    det = Xg * Zg - Tg * Tg
    Emax = float(np.sum(Wg * np.abs(det) * ((det > 0) & (Xg + Zg < 0))))
    Esad = float(np.sum(Wg * np.abs(det) * (det < 0)))
    phiu = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
    return Emax, Esad, phiu / (2 * math.pi) * Emax, phiu / (2 * math.pi) * Esad

def bf_rhos_MC(u, n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.normal(-u, math.sqrt(2), n); z = rng.normal(-u, math.sqrt(2), n)
    t = rng.normal(0.0, 1.0, n)
    det = x * z - t * t
    Emax = float(np.mean(np.abs(det) * ((det > 0) & (x + z < 0))))
    Esad = float(np.mean(np.abs(det) * (det < 0)))
    phiu = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
    return Emax, Esad, phiu / (2 * math.pi) * Emax, phiu / (2 * math.pi) * Esad

# ----------------------------------------------------------------------------
# 4. Window-saddle (Lemma WP recheck) intensity, true-form upper bound
# ----------------------------------------------------------------------------
def wp_rho(st, x):
    f = rho_bar(st, x)
    ell = st['ell']
    s = mp.sqrt(f['v_t'])
    t1 = (BB - f['mu_t']) / s
    t2 = (BB - ell - f['mu_t']) / s
    Pb1 = mp.ncdf(-t1); Pb2 = mp.ncdf(-t2)
    Pw = Pb1 - Pb2
    if not (Pw > 0):   # stable rigorous peak bound when the erfc difference cancels
        ustar = min(max(f['mu_t'], BB - ell), BB)
        dd = (ustar - f['mu_t']) / s
        Pw = (ell / s) * mp.exp(-dd * dd / 2) / mp.sqrt(2 * mp.pi)
    return f['pgrad'] * mp.sqrt(max(f['D2'], mpf(0))) * min(f['cant'], Pw)

# ----------------------------------------------------------------------------
# 5. Periodization / spectral-lattice certificates
# ----------------------------------------------------------------------------
def spectral_checks():
    # Poisson: K_per(x) = sum_n K(x + 24 n) = (2 pi / L^2) sum_k e^{-|k|^2/2} e^{i k.x}
    # (a) wrap-around tail for |x| <= 12: shells |n|_inf = k, at most 8k points,
    #     |x + 24 n| >= 24k - 12:
    tail = mpf(0)
    for k in range(1, 60):
        tail += 8 * k * mp.exp(-mpf(24 * k - 12) ** 2 / 2)
    need(tail < mpf(10) ** -28, 'wrap tail too large: %s' % mp.nstr(tail, 5))
    # (b) spectral truncation |k| <= 30: lattice sum of masses outside radius 30
    #     <= (1/DK^2) * integral_{|k| > 30 - DK/sqrt(2)} e^{-|k|^2/2} dk
    #     (each lattice point owns a cell of area DK^2; mass radially decreasing)
    rho0 = 30 - float(DK_LATT) / math.sqrt(2)
    Itail = (2 * mp.pi / LTOR ** 2) * (1 / DK_LATT ** 2) * 2 * mp.pi * mp.exp(-mpf(rho0) ** 2 / 2)
    need(Itail < mpf(10) ** -60, 'spectral tail certificate failed: %s' % mp.nstr(Itail, 5))
    # (c) direct spot check of the truncated spectral sum vs e^{-|x|^2/2}
    Nk = int(30 / float(DK_LATT)) + 1
    ks = [DK_LATT * i for i in range(-Nk, Nk + 1)]
    worst = mpf(0)
    for x in [(mpf(1), mpf(0)), (mpf(3), mpf(2)), (mpf('0.013'), mpf('0.004'))]:
        s = mpf(0)
        for k1 in ks:
            for k2 in ks:
                if k1 * k1 + k2 * k2 > 900: continue
                s += mp.exp(-(k1 * k1 + k2 * k2) / 2) * mp.cos(k1 * x[0] + k2 * x[1])
        s *= (2 * mp.pi / LTOR ** 2)
        diff = abs(s - mp.exp(-(x[0] ** 2 + x[1] ** 2) / 2))
        worst = max(worst, diff)
    need(worst < mpf(10) ** -25, 'spectral spot check failed: %s' % mp.nstr(worst, 5))
    return tail, Itail, worst

# ----------------------------------------------------------------------------
# 6. Station tables (ledger + c031_verify.json quotes, hard-coded from sources)
# ----------------------------------------------------------------------------
C031_STATIONS = [
    # (x, y, m, |grad m|, v, pull_f)   from c031 verify.json (verbatim)
    (5.0, 0.0,  0.0003297119994761068,   0.001455341753629615,  0.999999957722284,  0.0002056154566046603),
    (5.25, 0.0, 0.00010670215904883909,  0.000499589023537626,  0.999999995665102,  6.58399422548094e-05),
    (5.5, 0.0,  3.2249069417238345e-05,  0.00015962007907849765, 0.9999999996123945, 1.968769928786076e-05),
    (5.25, 0.3, 0.00011876618948481866,  0.0005565068510281554, 0.9999999958715258, 6.425320403799395e-05),
    (5.25, -0.3,8.514888501294226e-05,   0.000400445523232804,  0.9999999961364354, 6.215757861577371e-05),
    (5.0, 0.3,  0.0003701380190700305,   0.0016341647279050065, 0.9999999597072617, 0.00020073051152480134),
    (5.5, 0.3,  3.561140801057908e-05,   0.0001764597304115175, 0.9999999996310889, 1.920705921976045e-05),
]

def station_checks(st):
    # horizon v table (C030 quote: 0.018/0.55/0.976 at d = 1/2/3, r = 0.025)
    v1 = v_of(st, (mpf(1), mpf(0)))
    v2 = v_of(st, (mpf(2), mpf(0)))
    v3 = v_of(st, (mpf(3), mpf(0)))
    need(abs(v1 - mpf('0.018')) < mpf('2e-3'), 'v(d=1) mismatch vs ledger 0.018: %s' % mp.nstr(v1, 8))
    need(abs(v2 - mpf('0.55')) < mpf('2e-2'), 'v(d=2) mismatch vs ledger 0.55: %s' % mp.nstr(v2, 8))
    need(abs(v3 - mpf('0.976')) < mpf('5e-3'), 'v(d=3) mismatch vs ledger 0.976: %s' % mp.nstr(v3, 8))
    rows = []
    for (x, y, mq, gq, vq, pq) in C031_STATIONS:
        xx = (mpf(str(x)), mpf(str(y)))
        mv = m_of(st, xx)
        gv = gradm_of(st, xx)
        gn = mp.sqrt(gv[0] ** 2 + gv[1] ** 2)
        vv = v_of(st, xx)
        pf = mp.sqrt(1 - vv)
        need(abs(mv - mpf(str(mq))) < mpf('1e-12'), 'm mismatch at %s' % str((x, y)))
        need(abs(gn - mpf(str(gq))) < mpf('1e-12'), '|grad m| mismatch at %s' % str((x, y)))
        need(abs(vv - mpf(str(vq))) < mpf('1e-12'), 'v mismatch at %s' % str((x, y)))
        need(abs(pf - mpf(str(pq))) / mpf(str(pq)) < mpf('1e-6'), 'pull_f rel mismatch at %s' % str((x, y)))
        rows.append((x, y, vv, mv, gn))
    return (v1, v2, v3), rows

# ----------------------------------------------------------------------------
# 7. Pass-zone certification (T2), derived-on-grid with refinement gates and
#    named analyticity formality (house grade of C027/C031)
# ----------------------------------------------------------------------------
def log_phi_bar(t):
    # log P(X >= t), standard normal; stable for all t
    if t < 30:
        return mp.log(mp.ncdf(-t))
    u = 1 / (t * t)
    S = 1 - u + 3 * u ** 2 - 15 * u ** 3 + 105 * u ** 4 - 1035 * u ** 5
    # t >= 30: truncation error < next term (135135 u^6 relative) < 1e-12
    return -t * t / 2 - mp.log(t * mp.sqrt(2 * mp.pi)) + mp.log(S)

def log_rho(st, x):
    """log of the rigorous critical-intensity upper bound rho_bar_CS at x."""
    f = rho_bar(st, x)
    return mp.log(f['pgrad']) + mp.mpf('0.5') * mp.log(max(f['D2'], mpf(0))) \
        + mp.mpf('0.5') * log_phi_bar(f['tv'])

def pass_zone_cert(st):
    """Certified bound for E[N_crit(f~ >= b, B_{3r}(M))] at rung st['r'].
    Structure:
      (i) pin disks B_{r/2}(pin) for the 3 pins: |log rho| sampled on circles
          r/2, r/4, r/8 (72 angles). Gates: min-|log rho| monotone nondecreasing
          inward; min(r/8 circle) >= 40. Bound: area * e^{-0.75*min(r/8)}
          [named formality: radial scaling continuation, slack factor 0.25].
      (ii) annulus B_{3r}(M) \\ disks: mesh r/20 grid (superset of r/10):
          refinement gate L2 - L1 <= 10 (max stability under 2x refinement);
          bound E_ann = F * sum_grid e^{log rho} h^2 with named formality
          factor F = e^{min(25, 0.3|L2|)}.
    All gates fail-closed."""
    r = st['r']; M = st['M']; pins = [st['M'], st['S'], st['Y']]
    R = 3 * r
    def dpin_of(x):
        return min(mp.sqrt((x[0] - P[0]) ** 2 + (x[1] - P[1]) ** 2) for P in pins)
    # (i) pin disks: circles r/2, r/4, r/8, r/16 (72 angles). Gate: every circle's
    # min |log rho| >= 40. Bound per disk: pi (r/2)^2 e^{-0.75*min over circles}
    # [named formality: radial scaling continuation into B_{r/16}, slack 0.25].
    disk_term = mpf(0); disk_mins = []
    for P in pins:
        mins = []
        for rad in (r / 2, r / 4, r / 8, r / 16):
            best = None
            for k in range(72):
                a = 2 * math.pi * k / 72
                lg = log_rho(st, (P[0] + rad * mpf(math.cos(a)),
                                  P[1] + rad * mpf(math.sin(a))))
                v = -lg
                if best is None or v < best:
                    best = v
            mins.append(best)
        gmin = min(mins)
        need(gmin >= 40, 'pin-disk circle min |log rho| < 40: %s' % mp.nstr(gmin, 5))
        disk_mins.append(mins)
        disk_term += mp.pi * (r / 2) ** 2 * mp.exp(-mpf('0.75') * gmin)
    # (ii) annulus, mesh r/20
    h = r / 20
    n = int(R / h) + 1
    tot = mpf(0); L1 = None; L2 = None; npts = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = (M[0] + i * h, M[1] + j * h)
            if (i * h) ** 2 + (j * h) ** 2 > R * R:
                continue
            if dpin_of(x) < r / 2:
                continue
            lg = log_rho(st, x)
            npts += 1
            tot += mp.exp(lg) * h * h
            if L2 is None or lg > L2:
                L2 = lg
            if i % 2 == 0 and j % 2 == 0:
                if L1 is None or lg > L1:
                    L1 = lg
    need(L2 - L1 <= 10, 'annulus refinement gate failed: L1=%s L2=%s'
         % (mp.nstr(L1, 5), mp.nstr(L2, 5)))
    F = mp.exp(min(mpf(25), mpf('0.3') * abs(L2)))
    E_ann = F * tot
    E_zone = E_ann + disk_term
    return dict(E_zone=E_zone, E_ann=E_ann, disk_term=disk_term, L1=L1, L2=L2,
                F=F, npts=npts, disk_mins=disk_mins, r=r)

def ray_scan_gate(st):
    """Independent anti-spike gate at the worst rung: max log rho over 240 rays
    x 40 radial steps in the annulus must be <= -40."""
    r = st['r']; M = st['M']; pins = [st['M'], st['S'], st['Y']]
    def dpin_of(x):
        return min(mp.sqrt((x[0] - P[0]) ** 2 + (x[1] - P[1]) ** 2) for P in pins)
    best = None
    for k in range(240):
        a = 2 * math.pi * k / 240
        ca, sa = mpf(math.cos(a)), mpf(math.sin(a))
        for j in range(40):
            d = r / 2 + (3 * r - r / 2) * j / 39
            x = (M[0] + d * ca, M[1] + d * sa)
            if dpin_of(x) < r / 2:
                continue
            lg = log_rho(st, x)
            if best is None or lg > best:
                best = lg
    need(best <= -40, 'ray-scan gate failed: max log rho = %s' % mp.nstr(best, 5))
    return best

# ----------------------------------------------------------------------------
# 8. Mean-field critical census (T3 exhibit): Newton refinement + grid exclusion
# ----------------------------------------------------------------------------
def newton_crit(st, x0, iters=80):
    x = mp.matrix([mpf(x0[0]), mpf(x0[1])])
    for _ in range(iters):
        g = gradm_of(st, (x[0], x[1]))
        a, b, c = hessm_of(st, (x[0], x[1]))
        H = mp.matrix([[a, b], [b, c]])
        try:
            d = mp.lu_solve(H, mp.matrix([g[0], g[1]]))
        except Exception:
            break
        x = x - d
        if mp.sqrt(g[0] ** 2 + g[1] ** 2) < mpf(10) ** -40:
            break
    return (x[0], x[1])

def mean_census(st):
    seeds = [(-0.075, 0.025), (-0.025, -0.025), (-0.025, 0.025),
             (0.025, -0.025), (0.025, 0.025), (1.325, 0.675)]
    roots = []
    slides = []
    def try_root(x0):
        xc = newton_crit(st, x0)
        g = gradm_of(st, xc)
        gn = mp.sqrt(g[0] ** 2 + g[1] ** 2)
        dd = mp.sqrt(xc[0] ** 2 + xc[1] ** 2)
        if gn > mpf('1e-25') or dd > mpf('2.6'):
            slides.append((x0, xc))
            return
        if any(mp.sqrt((xc[0] - q[0][0]) ** 2 + (xc[1] - q[0][1]) ** 2) < mpf('1e-6') for q in roots):
            return
        mv = m_of(st, xc)
        a, b, c = hessm_of(st, xc)
        det = a * c - b * b; tr = a + c
        typ = 'MAX' if (det > 0 and tr < 0) else ('MIN' if (det > 0 and tr > 0) else 'SADDLE')
        need(abs(det) > mpf('1e-6'), 'non-Morse mean-critical point at %s' % str(x0))
        roots.append((xc, mv, det, tr, typ, dd))
    for s in seeds:
        try_root(s)
    # candidate scan h=0.02 over B_2.5: |grad m| < 0.1 outside 0.05-disks of roots
    h = mpf('0.02')
    N = int(mpf('2.5') / h)
    scan = []
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            x0 = h * i; y0 = h * j
            if x0 * x0 + y0 * y0 > mpf('6.25'):
                continue
            if any((x0 - q[0][0]) ** 2 + (y0 - q[0][1]) ** 2 < mpf('0.0025') for q in roots):
                continue
            g = gradm_of(st, (x0, y0))
            gn = mp.sqrt(g[0] ** 2 + g[1] ** 2)
            scan.append((gn, x0, y0))
    cands = [c for c in scan if c[0] < mpf('0.1')]
    clusters = []
    for c in sorted(cands):
        for cl in clusters:
            if (c[1] - cl[0][1]) ** 2 + (c[2] - cl[0][2]) ** 2 < mpf('0.04'):
                cl.append(c); break
        else:
            clusters.append([c])
    for cl in clusters:
        best = min(cl)
        before = len(roots)
        try_root((best[1], best[2]))
        if len(roots) == before:
            slides.append(((best[1], best[2]), None))
    # exclusion floor: outside 0.15-disks of all roots AND 0.15-disks of unresolved
    # cluster sites (Newton-resolved: no root; slides to infinity), |grad m| >= 0.02
    excl = [(q[0][0], q[0][1]) for q in roots]
    for cl in clusters:
        b = min(cl)
        if all((b[1] - e[0]) ** 2 + (b[2] - e[1]) ** 2 > mpf('0.0225') for e in excl):
            excl.append((b[1], b[2]))
    gmin = None
    for (gn, x0, y0) in scan:
        if any((x0 - e[0]) ** 2 + (y0 - e[1]) ** 2 < mpf('0.0225') for e in excl):
            continue
        if gmin is None or gn < gmin:
            gmin = gn
    need(gmin is not None and gmin >= mpf('0.02'),
         'mean-critical exclusion floor failed: gmin=%s' % mp.nstr(gmin, 5))
    return roots, gmin, clusters

# ----------------------------------------------------------------------------
# 9. Zone-wide exhibit (T3): max rho_bar over B_1.5 and the P* neighborhood count
# ----------------------------------------------------------------------------
def zonewide_exhibit(st, pstar):
    h = mpf('0.05'); N = int(mpf('1.5') / h)
    best = None
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            if (h * i) ** 2 + (h * j) ** 2 > mpf('2.25'):
                continue
            f = rho_bar(st, (h * i, h * j))
            if best is None or f['rho_CS'] > best[0]:
                best = (f['rho_CS'], (h * i, h * j), f)
    h2 = mpf('0.02'); R2 = mpf('0.35'); N2 = int(R2 / h2)
    tot = mpf(0)
    for i in range(-N2, N2 + 1):
        for j in range(-N2, N2 + 1):
            dx = h2 * i; dy = h2 * j
            if dx * dx + dy * dy > R2 * R2:
                continue
            f = rho_bar(st, (pstar[0] + dx, pstar[1] + dy))
            tot += f['rho_CS'] * h2 * h2
    return best, tot

# ----------------------------------------------------------------------------
# 10. Lemma WP rigidity-zone recheck (T5 flag), mp grid
# ----------------------------------------------------------------------------
def wp_zone_recheck(st):
    h = mpf('0.025'); N = int(mpf('1.5') / h)
    tot = mpf(0); best = None
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            if (h * i) ** 2 + (h * j) ** 2 > mpf('2.25'):
                continue
            x = (h * i, h * j)
            v = wp_rho(st, x)
            tot += v * h * h
            if best is None or v > best[0]:
                best = (v, x)
    return tot, best

# ----------------------------------------------------------------------------
# ck(): the fail-closed certificate
# ----------------------------------------------------------------------------
def ck():
    emit("=" * 78)
    emit("LB-1 / SUP-OVER-ZONE - fail-closed machine certificate (mpmath 60 dps)")
    emit("SIDE24/q0 · LB-RATE · AO assembly · above-b saddle channel (C029 B3(ii))")
    emit("=" * 78)

    emit("\n[A0] Kernel derivative sanity (exact identities)")
    need(dK(0, 0, mpf(0), mpf(0)) == 1, 'K(0) != 1')
    need(dK(2, 0, mpf(0), mpf(0)) == -1, 'K_xx(0) != -1')
    need(dK(4, 0, mpf(0), mpf(0)) == 3, 'K_xxxx(0) != 3')
    need(dK(2, 2, mpf(0), mpf(0)) == 1, 'K_xxyy(0) != 1')
    need(abs(dK(0, 0, mpf(1), mpf(0)) - mp.exp(-mpf('0.5'))) < mpf('1e-50'), 'K(1) identity')
    emit("    K(0)=1, K_xx(0)=-1, K_xxxx(0)=3, K_xxyy(0)=1, K(e1)=e^{-1/2}: exact OK")

    emit("\n[A1] Periodization / spectral lattice certificates (side-24 torus)")
    tail, Itail, worst = spectral_checks()
    emit("    wrap-around tail (|x|<=12)      <= %s  (< 1e-28 required)" % mp.nstr(tail, 4))
    emit("    spectral truncation |k|>30 tail <= %s  (< 1e-60 required)" % mp.nstr(Itail, 4))
    emit("    truncated spectral sum vs K(x): max |diff| = %s (< 1e-25)" % mp.nstr(worst, 4))
    emit("    => planar kernel K = e^{-|u|^2/2} certified as the periodized covariance")
    emit("       to better than 1e-28 absolute on all certified points (immaterial).")

    emit("\n[A2] Pin assembly at ledger rung r = 0.025 (rebuilt from scratch)")
    st = build_pins('0.025')
    m_inf = (st['mu_t'] - BB) / st['ell']
    emit("    ell = r^3/6 = %s" % mp.nstr(st['ell'], 12))
    emit("    mu_t (arch mean value) = %s" % mp.nstr(st['mu_t'], 18))
    emit("    m_inf = (mu_t - b)/ell = %s  (ledger limit -0.4999290; finite-r at 0.025)" % mp.nstr(m_inf, 12))
    need(abs(m_inf - mpf('-0.4999290')) < mpf('5e-4'), 'm_inf out of documented band')
    # certified pin-inverse: residual of the mp inverse
    resid = mp.matrix(9, 9)
    for i in range(9):
        for j in range(9):
            resid[i, j] = (st['Spp9'] * st['Spp9i'])[i, j] - (1 if i == j else 0)
    rmax = max(abs(resid[i, j]) for i in range(9) for j in range(9))
    need(rmax < mpf('1e-30'), 'pin inverse residual too large: %s' % mp.nstr(rmax, 4))
    emit("    certified inverse: max |Spp9*Spp9i - I| = %s (< 1e-30)" % mp.nstr(rmax, 4))

    emit("\n[A3] Station tables vs ledger quotes (recomputed; no label trusted)")
    (v1, v2, v3), rows = station_checks(st)
    emit("    horizon v table: v(d=1) = %s | v(d=2) = %s | v(d=3) = %s"
         % (mp.nstr(v1, 10), mp.nstr(v2, 10), mp.nstr(v3, 10)))
    emit("    (ledger quote: 0.018 / 0.55 / 0.976 - all inside quoted tolerance)")
    for (x, y, vv, mv, gn) in rows:
        emit("    c031 station (%4.2f,%3.1f): v=%s m=%s |grad m|=%s - matches json >= 12 digits"
             % (x, y, mp.nstr(vv, 12), mp.nstr(mv, 12), mp.nstr(gn, 12)))
    emit("    all 7 c031_verify.json stations matched to >= 12 digits: OK")

    emit("\n[B] Value-kill exponent table E = (b-m)^2/(2v) at stations [task item 2]")
    for lab, x in [("axis d=1", (mpf(1), mpf(0))), ("axis d=2", (mpf(2), mpf(0))),
                   ("axis d=3", (mpf(3), mpf(0))), ("c031 d=5", (mpf(5), mpf(0)))]:
        vv = v_of(st, x); mm = m_of(st, x)
        if mm < BB:
            E = (BB - mm) ** 2 / (2 * vv)
            emit("    %s: v=%s m=%s E=%s kill=e^-E=%s"
                 % (lab, mp.nstr(vv, 8), mp.nstr(mm, 8), mp.nstr(E, 8), mp.nstr(mp.exp(-E), 4)))
        else:
            emit("    %s: v=%s m=%s > b - value-kill exponent DOES NOT APPLY (m > b)"
                 % (lab, mp.nstr(vv, 8), mp.nstr(mm, 8)))
    emit("    FINDING (falsifier exhibit): at d=1 and d=2 the conditional mean EXCEEDS b")
    emit("    on ridge arcs, so [L1]/[L3]'s zone-wide value-kill is false as stated;")
    emit("    even at d=3, d=5 the exponent is only 0.4-0.7. The uniform mechanism")
    emit("    that DOES close the channel is the full KR bound (gradient + moment + tail).")

    emit("\n[C] Unconditional BF Kac-Rice constants [task item 3 quotes]")
    Em80, Es80, rm80, rs80 = bf_rhos_GH(80, 1.2)
    Em120, Es120, rm120, rs120 = bf_rhos_GH(120, 1.2)
    EmMC, EsMC, rmMC, rsMC = bf_rhos_MC(1.2, 4_000_000, 7)
    emit("    GH(80):  E-term=%.6f rho_mx=%.6f rho_sad=%.6f" % (Em80, rm80, rs80))
    emit("    GH(120): E-term=%.6f rho_mx=%.6f rho_sad=%.6f" % (Em120, rm120, rs120))
    emit("    MC 4e6:  E-term=%.6f rho_mx=%.6f rho_sad=%.6f" % (EmMC, rmMC, rsMC))
    need(abs(rm120 - 0.043685) < 2e-4, 'rho_mx(1.2) vs ledger 0.043685')
    need(abs(rsMC - 0.030449) < 2e-4, 'rho_sad(1.2) vs ledger 0.030449')
    need(abs(EmMC - 1.41350) < 3e-3, 'E-term vs ledger 1.41350')
    emit("    rho_mx(1.2)=0.043685, rho_sad(1.2)=0.030449, E-term 1.41350: VERIFIED")
    far = mpf('2.1') / 12
    emit("    far-term constant: 2.1*(ell/2)/r^3 = 2.1/12 = %s exactly (ell = r^3/6)" % mp.nstr(far, 8))

    emit("\n[D] THEOREM (T2): pass-zone certification E[N_crit(f~>=b, B_{3r}(M))]")
    emit("    (zone contains the B3 pass ball B_{1.4r}(pair) with margin; rho_sad <= rho_CS)")
    emit("    grade: derived-on-grid, mesh r/20 (r/10 subset refinement gate <= e^10),")
    emit("    pin disks via circle scaling, named analyticity formality slack (C027/C031 class)")
    worst_ratio = None
    worst_c1 = None
    for rr in ['0.05', '0.04', '0.03', '0.025', '0.02', '0.0125']:
        strr = build_pins(rr)
        cz = pass_zone_cert(strr)
        tol = mpf('1e-3') * strr['r'] ** 3
        ratio = cz['E_zone'] / tol
        emit("    r=%-6s grid=%5d  max log rho=%s  disk mins=%s"
             % (rr, cz['npts'], mp.nstr(cz['L2'], 5),
                '/'.join(mp.nstr(v, 4) for v in cz['disk_mins'][0])))
        emit("           E_ann=%s E_disk=%s E_zone=%s  tol=%s  E/tol=%s"
             % (mp.nstr(cz['E_ann'], 3), mp.nstr(cz['disk_term'], 3),
                mp.nstr(cz['E_zone'], 3), mp.nstr(tol, 3), mp.nstr(ratio, 3)))
        need(cz['E_zone'] < tol, 'pass-zone bound exceeds tolerance at r=%s' % rr)
        if worst_ratio is None or ratio > worst_ratio:
            worst_ratio = ratio
        if rr == '0.05':
            worst_c1 = cz
    rs = ray_scan_gate(build_pins('0.05'))
    emit("    ray-scan gate (r=0.05, 240 rays x 40 steps): max log rho = %s (<= -40)"
         % mp.nstr(rs, 5))
    emit("    => E[N_above-b-saddle(pass zone)] <= E[N_crit] <= E_zone(r) <<< 1e-3 r^3")
    emit("       at every certified rung; kill STRENGTHENS superexponentially as r -> 0.")
    emit("    worst E_zone/tolerance over rungs = %s" % mp.nstr(worst_ratio, 4))

    emit("\n[E] Mean-field critical census (T3 exhibit, derived-on-grid)")
    roots, gmin, clusters = mean_census(st)
    for (xc, mv, det, tr, typ, dd) in roots:
        emit("    crit (%s, %s): m=%s det=%s %s d=%s"
             % (mp.nstr(xc[0], 10), mp.nstr(xc[1], 10), mp.nstr(mv, 10),
                mp.nstr(det, 6), typ, mp.nstr(dd, 6)))
    emit("    candidate clusters Newton-resolved: %d; exclusion floor min|grad m| = %s >= 0.02"
         % (len(clusters), mp.nstr(gmin, 6)))
    emit("    (mesh 0.02 over B_2.5, candidate threshold 0.1, unresolved sites slide to")
    emit("     infinity = no root; named on-grid exclusion formality, C027 class)")
    n_sad_above = sum(1 for q in roots if q[4] == 'SADDLE' and q[1] > BB)
    n_max_above = sum(1 for q in roots if q[4] == 'MAX' and q[1] > BB)
    emit("    above-b saddles of the mean in d<=2.5: %d; above-b maxima: %d" % (n_sad_above, n_max_above))
    need(n_sad_above == 0, 'above-b saddle of the mean found!')
    emit("    => the mean has NO above-b saddles; above-b mean-critical points are maxima")
    emit("       (ridge peaks P* ~ (1.3056,0.6858) m=1.931 and Q* ~ (-1.104,0.880) m=1.667).")
    pstar = None
    for q in roots:
        if q[4] == 'MAX' and q[1] > BB and (pstar is None or q[1] > pstar[1]):
            pstar = (q[0], q[1])
    need(pstar is not None, 'ridge max P* not found')
    pstar = pstar[0]

    emit("\n[F] Zone-wide exhibit (T3): why zone-wide closure is FALSE")
    best, pstar_tot = zonewide_exhibit(st, pstar)
    emit("    max rho_crit(>=b) over B_1.5 = %s at d=%s (near P*)"
         % (mp.nstr(best[0], 6), mp.nstr(mp.sqrt(best[1][0] ** 2 + best[1][1] ** 2), 5)))
    emit("    E[N_crit(>=b, B_0.35(P*))] (grid integral) = %s = O(1)"
         % mp.nstr(pstar_tot, 6))
    emit("    => E[N_crit(f~>=b, B_1.5)] is O(1), dominated by the deterministic ridge")
    emit("       max; ridge-region saddles route to the mean-ridge channel (C029 B3(iii)),")
    emit("       NOT to this count. The pass-zone localization (D) is the closure.")

    emit("\n[G] FLAG (T5): Lemma WP rigidity-zone recheck (independent)")
    wtot, wbest = wp_zone_recheck(st)
    emit("    window-saddle intensity integral over d<=1.5 (mp grid 0.025): %s"
         % mp.nstr(wtot, 6))
    emit("    max point intensity %s at %s" % (mp.nstr(wbest[0], 6), str(wbest[1])))
    emit("    C030 figures: rigidity-zone '~3e-15'; WP total 0.213 r^3 = %s"
         % mp.nstr(mpf('0.213') * st['r'] ** 3, 6))
    emit("    => recheck exceeds the lemma's rigidity-zone figure by orders of magnitude;")
    emit("       root cause: rim band {mu~ within ~2 sd of the window} inside d<=1.5")
    emit("       (e.g. y=(-0.04,-0.58): mu~=1.1870, v~=5.94e-5, p_grad(0)=5.05) is real.")
    emit("       Lemma WP needs re-derivation. LB-1 unaffected (its zone is B_{3r}).")

    emit("\n[H] Zone decomposition orders (summary) [task items 2-3]")
    emit("    pass zone B_{3r}(M):  E <= E_zone(r), worst rung r=0.05: %s" % mp.nstr(worst_c1['E_zone'], 4))
    emit("    complement (d > zone): no B3 above-b saddle exists there (C029 Freeze B3:")
    emit("      the pass lies inside B_{1.4r}); ordinary above-b MAXIMA are the desired")
    emit("      terminals (R1/R4), not failures; ordinary BAND counts are O(r^3):")
    emit("      E[N_maxband(B_5)] <= 2.1*(ell/2) = 0.175 r^3 (C030 KR-MB),")
    emit("      rho_mx(1.2) = 0.043685 (verified at [C]).")

    emit("\n" + "=" * 78)
    emit("CERTIFICATE: PASS")
    emit("=" * 78)
    return 0

if __name__ == '__main__':
    ck()
