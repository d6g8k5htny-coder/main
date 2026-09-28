"""rnu_chi2_white.py -- step 1 of CL-RNU-001 SS5: EXACT gradient of chi2 in the whitened frame
(chain rule with the engine's exact dcov9/dy, dmu/dy), validated against dps-100 central FD.
No absolute-value sums: the cancellation that destroyed chi2_grad_bound happens exactly here."""
import sys, io, contextlib
sys.path.insert(0, "."); sys.path.insert(0, "../H2_foundations")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import d3_rn_unif as R
from mpmath import mp, mpf
import mpmath
E = R.E; kit = R.kit
ns = lambda x, n=8: mpmath.nstr(x, n)
mu6 = kit.Hmean
W = mp.zeros(6, 6)
for k in range(6):
    for j in range(6):
        W[k, j] = R.V0[j, k] / mpmath.sqrt(R.LAM0[k])

def tr(A): return mpmath.fsum(A[i, i] for i in range(A.rows))

def mean_grad_fixed(fs, y, v):
    """corrected exact dmu/dy: includes the TC G6inv dYC^T term of dTY6 for all rows and the
    dTC G6inv YC^T term for Y-target rows (both missing in the engine's mean_grad_exact)."""
    TSET9, CSET6, YJET, PTS, _GI, G6inv, _W6, kc, kdcov_d = R.TSET9, R.CSET6, R.YJET, R.PTS, R._GI, R.G6inv, R._W6, R.kc, R.kdcov_d
    TC = mp.zeros(9, 6)
    for i, (pi, di) in enumerate(TSET9):
        p1 = y if pi == 'Y' else PTS[pi]
        for j, (pj, dj) in enumerate(CSET6):
            TC[i, j] = kc(di, dj, p1, PTS[pj])
    YC = mp.zeros(3, 6)
    for i, (pi, di) in enumerate(YJET):
        for j, (pj, dj) in enumerate(CSET6):
            YC[i, j] = kc(di, dj, y, PTS[pj])
    YY6 = fs['YY6']; YY6inv = YY6 ** -1; TY6 = fs['TY6']
    rv = mp.matrix([v, 0, 0]) - YC * _W6
    out = []
    for e in (0, 1):
        dTC = mp.zeros(9, 6); dTY = mp.zeros(9, 3); dYC = mp.zeros(3, 6)
        for i, (pi, di) in enumerate(TSET9):
            if pi == 'Y':
                for j, (pj, dj) in enumerate(CSET6):
                    dTC[i, j] = kdcov_d(_GI[di], _GI[dj], y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
            else:
                for j, (pj, dj) in enumerate(YJET):
                    dTY[i, j] = -kdcov_d(_GI[di], _GI[dj], PTS[pi][0] - y[0], PTS[pi][1] - y[1], e)
        for i, (pi, di) in enumerate(YJET):
            for j, (pj, dj) in enumerate(CSET6):
                dYC[i, j] = kdcov_d(_GI[di], _GI[dj], y[0] - PTS[pj][0], y[1] - PTS[pj][1], e)
        dTY6 = dTY - dTC * G6inv * YC.T - TC * G6inv * dYC.T
        dYY6 = -(dYC * G6inv * YC.T + YC * G6inv * dYC.T)
        dYY6inv = -YY6inv * dYY6 * YY6inv
        drv = -dYC * _W6
        out.append(dTC * _W6 + dTY6 * (YY6inv * rv) + TY6 * (dYY6inv * rv) + TY6 * (YY6inv * drv))
    return out

def chi2_white_exact(y):
    """returns chi2, (d chi2/dy1, d chi2/dy2) EXACT, plus whitened smallness diagnostics."""
    fs = R.station_fast_full(y); v = kit.b - kit.ell / 2
    muv = fs['mean_v'](v); cov = fs['cov9']
    Sp = E.submat(cov, range(6), range(6)); mu9p = mp.matrix([muv[i] for i in range(6)])
    dmu = mean_grad_fixed(fs, y, v); dcov = R.cov9_grad_exact(fs, y)
    Sw = W * Sp * W.T; m = W * (mu9p - mu6)
    Swi = Sw ** -1
    Mw = Swi - mp.eye(6) * mpf('0.5'); Mwi = Mw ** -1
    c = Swi * m
    logq = (-3 * mpmath.log(2) - mpmath.log(mp.det(Sw)) - mpmath.log(mp.det(Mw)) / 2
            + (c.T * Mwi * c)[0] - (m.T * Swi * m)[0])
    chi2 = mpmath.exp(logq) - 1
    grads = []
    for e in (0, 1):
        dSp = E.submat(dcov[e], range(6), range(6)); dSw = W * dSp * W.T
        dm = W * mp.matrix([dmu[e][i] for i in range(6)])
        dSwi = -Swi * dSw * Swi
        dMw = dSwi
        dc = dSwi * m + Swi * dm
        dlogq = (-tr(Swi * dSw) - tr(Mwi * dMw) / 2
                 + 2 * (dc.T * Mwi * c)[0] - (c.T * Mwi * dMw * Mwi * c)[0]
                 - 2 * (dm.T * Swi * m)[0] - (m.T * dSwi * m)[0])
        grads.append((1 + chi2) * dlogq)
    A = mp.eye(6) - Sw
    smallness = dict(normA=mpmath.sqrt(mpmath.fsum(A[i, j] ** 2 for i in range(6) for j in range(6))),
                     normm=mpmath.sqrt(mpmath.fsum(m[i] ** 2 for i in range(6))))
    return chi2, grads, smallness

for y in [(mpf(5), mpf(0)), (mpf('4.8'), mpf('1.6')), (mpf('0.5'), mpf('6.2')), (mpf('5.1'), mpf('-0.3'))]:
    chi2, g, sm = chi2_white_exact(y)
    h = mpf('1e-20'); fd = []
    for e in (0, 1):
        yp = (y[0] + (h if e == 0 else 0), y[1] + (h if e == 1 else 0)); ym = (y[0] - (h if e == 0 else 0), y[1] - (h if e == 1 else 0))
        cp = chi2_white_exact(yp)[0]; cm = chi2_white_exact(ym)[0]
        fd.append((cp - cm) / (2 * h))
    gn = mpmath.sqrt(g[0] ** 2 + g[1] ** 2); fdn = mpmath.sqrt(fd[0] ** 2 + fd[1] ** 2)
    print(f"y=({ns(y[0],4)},{ns(y[1],4)}): chi2={ns(chi2,10)}  |grad|exact={ns(gn,10)}  |grad|FD={ns(fdn,10)}  rel.diff={ns(abs(gn-fdn)/gn if gn>0 else 0,3)}  ||A||={ns(sm['normA'],3)} ||m||={ns(sm['normm'],3)}")
    fs = R.station_fast_full(y); v = kit.b - kit.ell / 2; kp = R.kappa_pieces_fast(fs, v)
    print(f"      engine chi2={ns(kp['chi2'],10)}  => exact |grad kap_pair| = {ns(kp['kap_pair']*gn/(2*kp['chi2']),6)}   (engine crude bound: {ns(R.chi2_grad_bound(fs,kp,y,mpf(5))*kp['kap_pair']/(2*kp['chi2']),3)})")
