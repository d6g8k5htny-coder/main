import math, random
from proto_law import law
from engine import TARGET_NAMES
IDX = {n: i for i, n in enumerate(TARGET_NAMES)}
def phi(x): return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
def Phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def chol(A):
    n = len(A); L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][l] * L[j][l] for l in range(j))
            L[i][j] = math.sqrt(max(s, 1e-300)) if i == j else s / L[j][j]
    return L

def bound(b, k, r, n=100000, seed=1):
    mu, C, E = law(r, b, k)
    iw = IDX['w']
    mw, sw2 = mu[iw], C[iw][iw]; sw = math.sqrt(sw2)
    names = [n_ for n_ in TARGET_NAMES if n_ != 'w']
    beta = {n_: C[IDX[n_]][iw] / sw2 for n_ in names}
    Cg = [[C[IDX[a]][IDX[c]] - C[IDX[a]][iw] * C[IDX[c]][iw] / sw2 for c in names] for a in names]
    # X40 and T nearly collinear: jitter for float cholesky
    for i in range(len(names)): Cg[i][i] += 1e-12
    L = chol(Cg)
    a = 4 / (3 * k)
    pbar = phi(max(0.0, mw) / sw) / sw
    # crude Taylor tail T4 ~ small; ignore in the prototype (O(r^2))
    Xn = ['X40', 'X31', 'X22', 'X13', 'X04']
    bbar = max(2 * abs(beta['v']), abs(beta['om']), abs(beta['Y'])) + 4.5 * r * max(abs(beta[x]) for x in Xn)
    rng = random.Random(seed)
    acc = 0.0; acc0 = 0.0
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in names]
        g = {nm: mu[IDX[nm]] + sum(L[i][j] * z[j] for j in range(i + 1)) for i, nm in enumerate(names)}
        m = max(12 * k, 2 * abs(g['v']), abs(g['om']), abs(g['Y'])) + 4.5 * r * max(abs(g[x]) for x in Xn)
        mp = m + bbar * abs(mw)
        x = a * r * bbar * mp
        if x > 1 / 8: continue
        U = a * mp * mp / (1 - 3 * x)
        lam = r * U
        Tb = abs(g['T']) + abs(beta['T']) * (abs(mw) + lam)
        tb = abs(g['tau']) + abs(beta['tau']) * (abs(mw) + lam)
        nb = abs(g['nu']) + abs(beta['nu']) * (abs(mw) + lam)
        ob = abs(g['om']) + abs(beta['om']) * (abs(mw) + lam)
        vb = abs(g['v']) + abs(beta['v']) * (abs(mw) + lam)
        c1 = 6 * k + r * Tb; c2 = c1 + r * r * tb; e = (r * nb + vb) ** 2
        acc += c1 * (c2 * (U ** 3 / 3 + ob * U * U / 2) + e * U * U / 2)
    EPhi = acc / n
    lam_m = -mw
    El2 = (lam_m ** 2 + sw2) * Phi(lam_m / sw) + lam_m * sw * phi(lam_m / sw)
    z0 = 36 * k * k * El2
    return pbar * EPhi / z0, pbar, EPhi, z0

cG = {(0, 2): 5324360.4426, (0, 0.5): 85120.537, (1, 0.5): 24370.77, (1, 2): 1524412.12, (0, 1): 665544.26}
for (b, k), c in cG.items():
    for r in [1e-6, 1 / 1024, 1 / 512, 1 / 256]:
        B, pbar, EPhi, z0 = bound(b, k, r, n=40000)
        print('b=%g k=%g r=%.5f  bound=%.6g  bound/cG=%.4f' % (b, k, r, B, B / c))
