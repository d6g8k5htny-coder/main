import sys, io, contextlib
sys.path.insert(0, "."); sys.path.insert(0, "../H2_foundations")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import d3_rn_unif as R
from mpmath import mp, mpf
import mpmath
E = R.E; kit = R.kit; ns = lambda x, n=6: mpmath.nstr(x, n)

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

v = kit.b - kit.ell / 2; h = mpf('1e-20')
for y in [(mpf(5), mpf(0)), (mpf('4.8'), mpf('1.6')), (mpf('0.5'), mpf('6.2'))]:
    fs = R.station_fast_full(y); dmu = mean_grad_fixed(fs, y, v)
    worst = mpf(0); scale = mpf(0)
    for e in (0, 1):
        yp = (y[0] + (h if e == 0 else 0), y[1] + (h if e == 1 else 0)); ym = (y[0] - (h if e == 0 else 0), y[1] - (h if e == 1 else 0))
        mup = R.station_fast_full(yp)['mean_v'](v); mum = R.station_fast_full(ym)['mean_v'](v)
        for i in range(9):
            fd = (mup[i] - mum[i]) / (2 * h); worst = max(worst, abs(fd - dmu[e][i])); scale = max(scale, abs(fd))
    print(f"y=({ns(y[0],4)},{ns(y[1],4)}): FIXED mean grad max|FD-exact| = {ns(worst,3)}  (scale {ns(scale,3)})")
