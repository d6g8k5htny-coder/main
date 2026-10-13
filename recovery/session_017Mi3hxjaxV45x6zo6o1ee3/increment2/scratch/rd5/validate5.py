# Float validation of the m = 4 (d = 5) cone moment by three routes.
import math, random, itertools, sys
from math import erf, erfc, exp, pi, sqrt, gamma

def Z_u(m):
    return 2 ** (m / 2 + m * (m - 1) / 4) * (2 * pi) ** (m / 2) * math.prod(gamma(1 + j / 2) for j in range(1, m + 1)) / gamma(1.5) ** m

def orthant(m, T, n):
    coef = sqrt(3 / (m + 3)) / Z_u(m)
    h = T / n
    grid = [-(i + 0.5) * h for i in range(n)]
    total = 0.0
    for mu in itertools.product(grid, repeat=m):
        S = sum(mu)
        w = 1.0
        for x in mu:
            w *= x * x
        for i in range(m):
            for j in range(i + 1, m):
                w *= abs(mu[i] - mu[j])
        total += w * exp(-sum(x * x for x in mu) / 4 + S * S / (4 * (m + 3)))
    return coef * total * h ** m

def det4(A):
    # Laplace expansion via 3x3
    def det3(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    d = 0.0
    for j in range(4):
        minor = [[A[i][k] for k in range(4) if k != j] for i in range(1, 4)]
        d += (-1) ** j * A[0][j] * det3(minor)
    return d

def negdef(A):
    # Sylvester on -A: all leading principal minors positive
    B = [[-x for x in row] for row in A]
    m1 = B[0][0]
    m2 = B[0][0]*B[1][1] - B[0][1]**2
    M3 = [row[:3] for row in B[:3]]
    m3 = (M3[0][0]*(M3[1][1]*M3[2][2]-M3[1][2]*M3[2][1]) - M3[0][1]*(M3[1][0]*M3[2][2]-M3[1][2]*M3[2][0]) + M3[0][2]*(M3[1][0]*M3[2][1]-M3[1][1]*M3[2][0]))
    return m1 > 0 and m2 > 0 and m3 > 0 and det4(B) > 0

def mc(n, seed=7):
    rng = random.Random(seed); c = sqrt(2 / 3); s = s2 = 0.0
    for _ in range(n):
        Q = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            Q[i][i] = rng.gauss(0, sqrt(2))
            for j in range(i+1, 4):
                Q[i][j] = Q[j][i] = rng.gauss(0, 1)
        z = rng.gauss(0, 1)
        A = [[Q[i][j] + (c*z if i == j else 0.0) for j in range(4)] for i in range(4)]
        v = det4(A) ** 2 if negdef(A) else 0.0
        s += v; s2 += v * v
    mean = s / n
    return mean, sqrt(max(s2 / n - mean * mean, 0) / n)

def a_integrated_3d(T=26.0, n=90):
    """m = 4: after the exact a-integral, a 3-D midpoint quadrature over (p1, p2, p3) > 0 of the sector integrand;
    D_4 = sqrt(3/7) . 4! / Z_4 . I_4."""
    m = 4
    alpha = 3 * m / (4 * (m + 3))           # 3/7
    def G_list(beta, nmax=8):
        G0 = 0.5 * sqrt(pi / alpha) * exp(beta * beta / (4 * alpha)) * erfc(beta / (2 * sqrt(alpha)))
        G = [G0, (1 - beta * G0) / (2 * alpha)]
        for k in range(1, nmax):
            G.append((k * G[k - 1] - beta * G[k]) / (2 * alpha))
        return G
    def pm(u, v):
        out = [0.0] * (len(u) + len(v) - 1)
        for i, x in enumerate(u):
            for j, y in enumerate(v):
                out[i + j] += x * y
        return out
    h = T / n
    tot = 0.0
    for i in range(n):
        p1 = (i + 0.5) * h
        for j in range(n):
            p2 = (j + 0.5) * h
            for k in range(n):
                p3 = (k + 0.5) * h
                t1, t2, t3 = p1, p1 + p2, p1 + p2 + p3
                Tsum = t1 + t2 + t3
                beta = 3 * Tsum / (2 * (m + 3))       # coefficient of a: a T . 3/(2(m+3))
                rest = (t1*t1 + t2*t2 + t3*t3) / 4 - Tsum * Tsum / (4 * (m + 3))
                G = G_list(beta)
                poly = pm(pm(pm([0, 0, 1.0], [t1*t1, 2*t1, 1.0]), [t2*t2, 2*t2, 1.0]), [t3*t3, 2*t3, 1.0])   # a^2 (a+t1)^2 (a+t2)^2 (a+t3)^2
                val = sum(poly[q] * G[q] for q in range(9))
                vand = p1 * p2 * p3 * (p1 + p2) * (p2 + p3) * (p1 + p2 + p3)
                tot += val * vand * exp(-rest)
    I4 = tot * h ** 3
    return sqrt(3 / 7) * 24 / Z_u(4) * I4

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("orthant", "all"):
        print("D_4 orthant n=34:", orthant(4, 14.0, 34), flush=True)
    if which in ("aint", "all"):
        print("D_4 a-integrated 3-D n=90:", a_integrated_3d(26.0, 90), flush=True)
    if which in ("mc", "all"):
        print("D_4 MC 200k:", mc(200000), flush=True)
