import sys, random
sys.path.insert(0, '/home/user/wt_cusp/frontiers/cusp_coefficient_certified_20261001')
sys.dont_write_bytecode = True
import mpmath as mp
import elem as E
mp.mp.dps = 40
random.seed(1)
def chk(name, f_iv, f_mp, pts):
    bad = 0; worst = 0.0
    for t in pts:
        iv = f_iv(t)
        v = f_mp(mp.mpf(t))
        if not (mp.mpf(iv[0]) <= v <= mp.mpf(iv[1])):
            bad += 1
            if bad < 4: print('  FAIL', name, t, iv, v)
        w = (iv[1] - iv[0]) / max(abs(float(v)), 1e-300)
        worst = max(worst, w)
    print('%-8s %5d pts  fails %d  max rel width %.2e' % (name, len(pts), bad, worst))
P = [random.uniform(-3, 3) for _ in range(300)] + [random.uniform(-1e-6, 1e-6) for _ in range(50)] + [random.uniform(-40, 40) for _ in range(100)] + [0.5, -0.5, 1.0, -1.0, 3.0, 1e-300, 2.5e-8]
Ppos = [abs(x) for x in P if x != 0] + [1e-10, 1e10, 7.3e5, 0.7071067811865476, 1.4142135623730951]
chk('exp', lambda t: E.exp_pt(t), mp.exp, [x for x in P if x <= 700])
chk('log', lambda t: E.log_pt(t), mp.log, Ppos)
chk('log1p', lambda t: E.log1p_pt(t), mp.log1p, [x for x in P if x > -1] + [-0.9, -0.99, 1e5])
chk('atan', lambda t: E.atan_pt(t), mp.atan, P + [1e8, -1e8])
chk('sin', lambda t: E.sincos_pt(t)[0], mp.sin, P + [1000.0, 12345.6])
chk('cos', lambda t: E.sincos_pt(t)[1], mp.cos, P + [1000.0, 12345.6])
chk('E1', lambda t: E.E1_pt(t), lambda x: mp.expm1(x)/x if x != 0 else mp.mpf(1), [x for x in P if x <= 700] + [-700.0, -50.0])
chk('L', lambda t: E.L_pt(t), lambda x: mp.log1p(x)/x if x != 0 else mp.mpf(1), [x for x in P if x > -1] + [-0.9, 1e6])
chk('A', lambda t: E.A_pt(t), lambda x: mp.atan(x)/x if x != 0 else mp.mpf(1), P + [1e6])
chk('Sinc', lambda t: E.Sinc_pt(t), lambda x: mp.sin(x)/x if x != 0 else mp.mpf(1), P)
# interval extensions: random intervals, sample points inside
def chk_iv(name, f_iv, f_mp, gen, n=300):
    bad = 0
    for _ in range(n):
        a, b = gen()
        X = (min(a, b), max(a, b))
        Y = f_iv(X)
        for t in (X[0], X[1], 0.5*(X[0]+X[1]), X[0] + 0.3*(X[1]-X[0])):
            v = f_mp(mp.mpf(t))
            if not (mp.mpf(Y[0]) <= v <= mp.mpf(Y[1])):
                bad += 1
                if bad < 4: print('  FAIL', name, X, Y, t, v)
    print('%-8s intervals %d fails %d' % (name, n, bad))
g = lambda: (random.uniform(-5, 5), random.uniform(-5, 5))
chk_iv('E1_iv', E.E1_iv, lambda x: mp.expm1(x)/x if x != 0 else mp.mpf(1), g)
chk_iv('L_iv', E.L_iv, lambda x: mp.log1p(x)/x if x != 0 else mp.mpf(1), lambda: (random.uniform(-0.9, 5), random.uniform(-0.9, 5)))
chk_iv('A_iv', E.A_iv, lambda x: mp.atan(x)/x if x != 0 else mp.mpf(1), g)
chk_iv('Sinc_iv', E.Sinc_iv, lambda x: mp.sin(x)/x if x != 0 else mp.mpf(1), lambda: (random.uniform(-12, 12), random.uniform(-12, 12)))
chk_iv('sin_iv', E.sin_iv, mp.sin, lambda: (random.uniform(-9, 9), random.uniform(-9, 9)))
chk_iv('cos_iv', E.cos_iv, mp.cos, lambda: (random.uniform(-9, 9), random.uniform(-9, 9)))
chk_iv('cosh_iv', E.cosh_iv, mp.cosh, g)
chk_iv('sinh_iv', E.sinh_iv, mp.sinh, g)
chk_iv('atan_iv', E.atan_iv, mp.atan, g)
