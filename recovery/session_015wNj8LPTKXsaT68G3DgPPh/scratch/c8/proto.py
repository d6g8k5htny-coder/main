import math, random, sys
from math import erf, sqrt, exp

def he(n, x):
    a, b = 1.0, x
    if n == 0: return a
    for k in range(1, n):
        a, b = b, x * b - k * a
    return b

def kd(g, y):
    v = 1.0
    for gi, yi in zip(g, y):
        v *= (-1) ** gi * he(gi, yi) * exp(-yi * yi / 2)
    return v

def cov(al, s, be, t):
    g = tuple(a + b for a, b in zip(al, be))
    return (-1) ** sum(al) * kd(g, [ti - si for si, ti in zip(s, t)])

def chol(S):
    n = len(S); L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = S[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0: raise ValueError('not spd %r' % s)
                L[i][i] = sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L

def solve(S, cols):
    L = chol(S); n = len(S); out = []
    for c in cols:
        y = [0.0] * n
        for i in range(n): y[i] = (c[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        x = [0.0] * n
        for i in reversed(range(n)): x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        out.append(x)
    return out

def conditional(r, b, k):
    M = (-r / 2, 0.0); S = (r / 2, 0.0)
    obs = [((0, 0), M), ((1, 0), M), ((0, 1), M), ((0, 0), S), ((1, 0), S), ((0, 1), S)]
    vals = [b, 0, 0, b - k * r ** 3, 0, 0]
    tg = [((2, 0), M), ((1, 1), M), ((0, 2), M), ((2, 0), S), ((1, 1), S), ((0, 2), S)]
    Soo = [[cov(a, s, c, t) for (c, t) in obs] for (a, s) in obs]
    Sto = [[cov(a, s, c, t) for (c, t) in obs] for (a, s) in tg]
    Stt = [[cov(a, s, c, t) for (c, t) in tg] for (a, s) in tg]
    W = solve(Soo, Sto)
    mu = [sum(W[i][j] * vals[j] for j in range(6)) for i in range(6)]
    C = [[Stt[i][j] - sum(W[i][l] * Sto[j][l] for l in range(6)) for j in range(6)] for i in range(6)]
    return mu, C

def mc(mu, C, n, seed):
    L = chol([[C[i][j] + (1e-18 if i == j else 0) for j in range(6)] for i in range(6)])
    rng = random.Random(seed); acc = acc2 = 0.0
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(6)]
        h = [mu[i] + sum(L[i][j] * z[j] for j in range(6)) for i in range(6)]
        dM = h[0] * h[2] - h[1] ** 2; dS = h[3] * h[5] - h[4] ** 2
        v = abs(dM) * abs(dS) if (h[0] < 0 and dM > 0 and dS < 0) else 0.0
        acc += v; acc2 += v * v
    m = acc / n
    return m, sqrt(max(acc2 / n - m * m, 0) / n)

def box_bound(mu, C, t):
    L = chol([[C[i][j] + (1e-18 if i == j else 0) for j in range(6)] for i in range(6)])
    rho = [sum(abs(L[i][j]) * t[j] for j in range(6)) for i in range(6)]
    lo = [mu[i] - rho[i] for i in range(6)]; hi = [mu[i] + rho[i] for i in range(6)]
    if hi[0] >= 0 or hi[2] >= 0: return 0.0
    detM = (-hi[0]) * (-hi[2]) - max(abs(lo[1]), abs(hi[1])) ** 2
    if detM <= 0: return 0.0
    if lo[3] <= 0 or hi[5] >= 0: return 0.0
    detS = lo[3] * (-hi[5])
    p = 1.0
    for tj in t: p *= erf(tj / sqrt(2))
    return detM * detS * p

def optimize(mu, C):
    t = [1.0] * 6; best = box_bound(mu, C, t)
    step = 0.5
    for it in range(60):
        improved = False
        for j in range(6):
            for s in (1 + step, 1 / (1 + step)):
                t2 = t[:]; t2[j] *= s
                v = box_bound(mu, C, t2)
                if v > best: best, t, improved = v, t2, True
        if not improved: step /= 2
        if step < 1e-3: break
    return best, t

def z0(b, k):
    # 36 k^2 E[A^2 1{A<0}], A ~ N(-b, 2)
    s = sqrt(2); m = -b
    Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2))); phi = lambda x: exp(-x * x / 2) / sqrt(2 * math.pi)
    a = -m / s  # P(A<0) = Phi(a)
    # E[A^2 1{A<0}] with A = m + s Z: = (m^2 + s^2) Phi(a) - s(2m) phi(a) ... compute: E[(m+sZ)^2 1{Z<a}] = m^2 Phi(a) + 2 m s (-phi(a)) + s^2 (Phi(a) - a phi(a))
    e = m * m * Phi(a) - 2 * m * s * phi(a) + s * s * (Phi(a) - a * phi(a))
    return 36 * k * k * e

if __name__ == '__main__':
  for r in (0.2, 0.3, 0.5):
    for b in (0.0, 1.0):
        for k in (0.5, 1.0, 2.0):
            mu, C = conditional(r, b, k)
            m, se = mc(mu, C, 100000, 1)
            best, t = optimize(mu, C)
            print('r=%.1f b=%.0f k=%.1f  Z/r2 MC %.4f (%.4f)  z0 %.4f  box %.4f  ratio %.3f  t=%s' % (r, b, k, m / r ** 2, se / r ** 2, z0(b, k), best / r ** 2, best / m, [round(x, 2) for x in t]))
            if r == 0.3 and b == 0.0 and k == 1.0:
                print(' mu', [round(x, 4) for x in mu]); print(' sd', [round(sqrt(C[i][i]), 5) for i in range(6)])
