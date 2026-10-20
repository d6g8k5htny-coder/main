# Independent adversarial check (not reusing the lens code).
# Planar Bargmann-Fock kernel K(h)=exp(-|h|^2/2); periodization ignored (side 24).
# Conditional Gaussian law given the six pins of LS-DER-024 / P2 plus grad f(X)=0,
# X = M + r(p,q).  Estimates E|det H_M det H_S det H_X| (untyped, as in 024 (13) and
# P2 (P16)-(P17)) and the typed weighted intensity rho^W(X) of (P18), summed over j.
import mpmath as mp, random, sys
mp.mp.dps = 80

def hermite(n, x):
    h0, h1 = mp.mpf(1), x
    if n == 0:
        return h0
    for m in range(1, n):
        h0, h1 = h1, x * h1 - m * h0
    return h1

def kcov(P, a, R, b):
    h1 = P[0] - R[0]; h2 = P[1] - R[1]
    sgn = -1 if (a[0] + a[1]) % 2 else 1
    return sgn * hermite(a[0] + b[0], h1) * hermite(a[1] + b[1], h2) * mp.exp(-(h1 * h1 + h2 * h2) / 2)

def condition(obs, tgt, qry):
    n, m = len(obs), len(qry)
    S = mp.matrix(n, n); C = mp.matrix(m, n); Q = mp.matrix(m, m)
    for i in range(n):
        for j in range(n):
            S[i, j] = kcov(obs[i][0], obs[i][1], obs[j][0], obs[j][1])
    for i in range(m):
        for j in range(n):
            C[i, j] = kcov(qry[i][0], qry[i][1], obs[j][0], obs[j][1])
        for j in range(m):
            Q[i, j] = kcov(qry[i][0], qry[i][1], qry[j][0], qry[j][1])
    Sinv = mp.inverse(S)
    mean = C * (Sinv * mp.matrix(tgt))
    cov = Q - C * Sinv * C.T
    return mean, cov, S

def sqrtm_float(cov):
    m = cov.rows
    cov = (cov + cov.T) / 2
    E, V = mp.eigsy(cov)
    L = [[0.0] * m for _ in range(m)]
    for j in range(m):
        s = mp.sqrt(max(E[j], 0))
        for i in range(m):
            L[i][j] = float(V[i, j] * s)
    return L

def gauss_density0(mean, cov):
    # density of a Gaussian vector at 0
    d = mp.matrix([-mean[i] for i in range(mean.rows)])
    return mp.exp(-(d.T * mp.lu_solve(cov, d))[0] / 2) / mp.sqrt((2 * mp.pi) ** cov.rows * mp.det(cov))

HESS = [(2, 0), (1, 1), (0, 2)]

def run(r, p, q, N, seed=7, b=mp.mpf('1.2'), kap=mp.mpf(1)):
    M = (-r / 2, mp.mpf(0)); S = (r / 2, mp.mpf(0))
    X = (M[0] + r * p, r * q)
    pins = [(M, (0, 0)), (M, (1, 0)), (M, (0, 1)), (S, (0, 0)), (S, (1, 0)), (S, (0, 1))]
    tg = [b, 0, 0, b - kap * r ** 3 / 6, 0, 0]
    # normaliser Z (endpoint law only)
    mZ, cZ, _ = condition(pins, tg, [(M, a) for a in HESS] + [(S, a) for a in HESS])
    # gradient density at X under pins
    mg, cg, _ = condition(pins, tg, [(X, (1, 0)), (X, (0, 1))])
    pg = gauss_density0(mg, cg)
    # Hessians given pins and grad f(X)=0
    obs = pins + [(X, (1, 0)), (X, (0, 1))]
    mH, cH, _ = condition(obs, tg + [0, 0], [(M, a) for a in HESS] + [(S, a) for a in HESS] + [(X, a) for a in HESS])
    LZ = sqrtm_float(cZ); LH = sqrtm_float(cH)
    mZf = [float(v) for v in mZ]; mHf = [float(v) for v in mH]
    rnd = random.Random(seed)
    accZ = 0.0; accU = 0.0; accT = 0.0
    for _ in range(N):
        g = [rnd.gauss(0, 1) for _ in range(9)]
        # Z sample (6-dim)
        v = [mZf[i] + sum(LZ[i][j] * g[j] for j in range(6)) for i in range(6)]
        dM = v[0] * v[2] - v[1] ** 2; dS = v[3] * v[5] - v[4] ** 2
        if dM > 0 and v[0] < 0 and dS < 0:
            accZ += abs(dM * dS)
        u = [mHf[i] + sum(LH[i][j] * g[j] for j in range(9)) for i in range(9)]
        eM = u[0] * u[2] - u[1] ** 2; eS = u[3] * u[5] - u[4] ** 2; eX = u[6] * u[8] - u[7] ** 2
        trip = abs(eM * eS * eX)
        accU += trip
        if eM > 0 and u[0] < 0 and eS < 0:
            accT += trip
    Z = accZ / N; EU = accU / N; ET = accT / N
    rho2 = float(p * p + q * q)
    rhoW = float(pg) * ET / Z      # sum_j rho_j^W per physical area
    return dict(Z_over_r2=Z / float(r) ** 2, EU=EU, EU_r6rho2=EU / (float(r) ** 6 * rho2),
                EU_r3rho2=EU / (float(r) ** 3 * rho2), rhoW=rhoW, rhoW_over_r=rhoW / float(r),
                pg=float(pg), Hmean_M=[float(mH[i]) for i in range(3)], Hmean_X=[float(mH[i]) for i in range(6, 9)])

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    for pp in [mp.mpf('0.15'), mp.mpf('-0.15')]:
        for r in [mp.mpf('0.04'), mp.mpf('0.02'), mp.mpf('0.01')]:
            cases = [('axis q=0', mp.mpf(0)), ('II q=rp/10', r * abs(pp) / 10),
                     ('I-bdry q=rp (t=1/r)', r * abs(pp)), ('I q=3rp', 3 * r * abs(pp)),
                     ('I q=sqrt(r)p', mp.sqrt(r) * abs(pp)), ('I q=p (t=1)', abs(pp))]
            for lab, qq in cases:
                o = run(r, pp, qq, N)
                print(f"p={float(pp):+.2f} r={float(r):.3f} {lab:>20}: E|trip|={o['EU']:.3e} /(r^6rho^2)={o['EU_r6rho2']:.3e} "
                      f"/(r^3rho^2)={o['EU_r3rho2']:.3e} rhoW/r={o['rhoW_over_r']:.3e} Z/r^2={o['Z_over_r2']:.3f} "
                      f"H_M mean={['%.2e'%x for x in o['Hmean_M']]}", flush=True)
