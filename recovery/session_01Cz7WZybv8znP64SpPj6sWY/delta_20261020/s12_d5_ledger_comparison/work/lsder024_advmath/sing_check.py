# Independent check of LS-DER-024 (27)-(28) at fixed distance and near t~r, one or two angles.
# lambda_crit(y,u)=p_{(f,grad f)(y)|pins}(u,0) E[W|det H_y| | pins,f(y)=u,grad f(y)=0]/Z  (024 eq. (4))
import mpmath as mp, random, sys
sys.path.insert(0, sys.argv[1])
from collar_mc import condition, sqrtm_float, HESS
mp.mp.dps = 80

def gdens(mean, cov, x):
    d = mp.matrix([x[i] - mean[i] for i in range(mean.rows)])
    return mp.exp(-(d.T * mp.lu_solve(cov, d))[0] / 2) / mp.sqrt((2 * mp.pi) ** cov.rows * mp.det(cov))

def lam(r, t, th, N, b=mp.mpf('1.2'), kap=mp.mpf(1), alpha=mp.mpf('0.5'), seed=11):
    M = (-r / 2, mp.mpf(0)); S = (r / 2, mp.mpf(0)); y = (t * mp.cos(th), t * mp.sin(th))
    pins = [(M, (0, 0)), (M, (1, 0)), (M, (0, 1)), (S, (0, 0)), (S, (1, 0)), (S, (0, 1))]
    tg = [b, 0, 0, b - kap * r ** 3 / 6, 0, 0]
    u = b - alpha * kap * r ** 3 / 6
    mZ, cZ, _ = condition(pins, tg, [(M, a) for a in HESS] + [(S, a) for a in HESS])
    m3, c3, _ = condition(pins, tg, [(y, (0, 0)), (y, (1, 0)), (y, (0, 1))])
    dens = gdens(m3, c3, [u, 0, 0])
    obs = pins + [(y, (0, 0)), (y, (1, 0)), (y, (0, 1))]
    mH, cH, _ = condition(obs, tg + [u, 0, 0], [(M, a) for a in HESS] + [(S, a) for a in HESS] + [(y, a) for a in HESS])
    LZ = sqrtm_float(cZ); LH = sqrtm_float(cH); mZf = [float(v) for v in mZ]; mHf = [float(v) for v in mH]
    rnd = random.Random(seed); aZ = 0.0; aN = 0.0; aU = 0.0
    for _ in range(N):
        g = [rnd.gauss(0, 1) for _ in range(9)]
        v = [mZf[i] + sum(LZ[i][j] * g[j] for j in range(6)) for i in range(6)]
        dM = v[0] * v[2] - v[1] ** 2; dS = v[3] * v[5] - v[4] ** 2
        if dM > 0 and v[0] < 0 and dS < 0:
            aZ += abs(dM * dS)
        w = [mHf[i] + sum(LH[i][j] * g[j] for j in range(9)) for i in range(9)]
        eM = w[0] * w[2] - w[1] ** 2; eS = w[3] * w[5] - w[4] ** 2; eY = w[6] * w[8] - w[7] ** 2
        aU += abs(eM * eS * eY)
        if eM > 0 and w[0] < 0 and eS < 0:
            aN += abs(eM * eS * eY)
    Z = aZ / N; num = aN / N; unt = aU / N
    L = float(dens) * num / Z
    return L, num, unt, Z, float(dens)

if __name__ == '__main__':
    N = int(sys.argv[2])
    for th in [mp.pi / 2, mp.pi / 4]:
        for r in [mp.mpf('0.04'), mp.mpf('0.02'), mp.mpf('0.01')]:
            for lab, t in [('t=2.5r', 2.5 * r), ('t=r^(2/3)', r ** (mp.mpf(2) / 3)), ('t=0.4', mp.mpf('0.4'))]:
                L, num, unt, Z, dens = lam(r, t, th, N)
                rf = float(r); tf = float(t)
                print(f"th={float(th):.3f} r={rf:.3f} {lab:>10} (t={tf:.4f}): lambda={L:.3e} lambda/(r^4 t^-6)={L/(rf**4*tf**-6):.3e} "
                      f"typed num/r^2={num/rf**2:.3e} typed num/r^6={num/rf**6:.3e} untyped/r^6={unt/rf**6:.3e} dens*t^6={dens*tf**6:.3e}", flush=True)
