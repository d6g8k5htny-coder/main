# Float validation of the reduction D_m = sqrt(3/(m+3)) / Z_m^u * int_{mu<0} prod mu_i^2 |Delta| exp(-|mu|^2/4 + S^2/(4(m+3))) dmu
import math, random, itertools
from math import erf, erfc, exp, pi, sqrt, gamma

def Z_u(m):            # Mehta: int_R^m |Delta| e^{-|l|^2/4} dl
    return 2 ** (m / 2 + m * (m - 1) / 4) * (2 * pi) ** (m / 2) * math.prod(gamma(1 + j / 2) for j in range(1, m + 1)) / gamma(1.5) ** m

def D_direct_mc(m, n=400000, seed=1):
    rng = random.Random(seed); c = sqrt(2 / 3); acc = 0.0
    import numpy as np
    rs = np.random.default_rng(seed)
    G = rs.standard_normal((n, m, m)); Q = (G + np.transpose(G, (0, 2, 1))) / np.sqrt(2)   # diag var 2, offdiag var 1
    Z = rs.standard_normal(n)
    A = Q + c * Z[:, None, None] * np.eye(m)[None]
    ev = np.linalg.eigvalsh(A)
    neg = ev[:, -1] < 0
    det = np.prod(ev, axis=1)
    vals = det ** 2 * neg
    return vals.mean(), vals.std() / sqrt(n)

def D_reduced_quadrature(m, T=14.0, n=140):
    # brute-force nested midpoint quadrature of the m-dim orthant integral (m <= 3), just to validate the formula
    coef = sqrt(3 / (m + 3)) / Z_u(m)
    h = T / n
    grid = [-(i + 0.5) * h for i in range(n)]
    total = 0.0
    for mu in itertools.product(grid, repeat=m):
        S = sum(mu)
        w = math.prod(x * x for x in mu)
        for i in range(m):
            for j in range(i + 1, m):
                w *= abs(mu[i] - mu[j])
        total += w * exp(-sum(x * x for x in mu) / 4 + S * S / (4 * (m + 3)))
    return coef * total * h ** m

def D3_onedim(T=40.0, n=4000):
    """m = 3: I_3 = (1/2) int_0^inf ds [E-part + R-part] with the exact a- and q-integrations; here the q-integral is done numerically
    to validate the (s,q) decoupling; D_3 = I_3/(16 pi)."""
    alpha = 3 / 8
    def G_list(beta, nmax=6):
        G0 = 0.5 * sqrt(pi / alpha) * exp(beta * beta / (4 * alpha)) * erfc(beta / (2 * sqrt(alpha)))
        G = [G0, (1 - beta * G0) / (2 * alpha)]
        for k in range(1, nmax):
            G.append((k * G[k - 1] - beta * G[k]) / (2 * alpha))
        return G
    def inner(p, q):    # int_0^inf a^2 (a+p)^2 (a+p+q)^2 e^{-Q} da  times p q (p+q) e^{-rest}
        beta = p / 2 + q / 4
        G = G_list(beta)
        # expand a^2 (a+p)^2 (a+p+q)^2 in powers of a
        poly = [0.0] * 7
        A = [0, 0, 1.0]                                  # a^2
        B = [p * p, 2 * p, 1.0]                          # (a+p)^2
        C = [(p + q) ** 2, 2 * (p + q), 1.0]             # (a+p+q)^2
        def pm(u, v):
            out = [0.0] * (len(u) + len(v) - 1)
            for i, x in enumerate(u):
                for j, y in enumerate(v):
                    out[i + j] += x * y
            return out
        full = pm(pm(A, B), C)
        val = sum(full[k] * G[k] for k in range(7))
        return val * p * q * (p + q) * exp(-(p * p / 3 + 5 * q * q / 24 + p * q / 3))
    h = T / n
    tot = 0.0
    for i in range(n):
        p = (i + 0.5) * h
        for j in range(n):
            q = (j + 0.5) * h
            tot += inner(p, q)
    I3 = tot * h * h
    return I3 / (16 * pi)

if __name__ == "__main__":
    print("D_1 formula check (m=1):", D_reduced_quadrature(1, T=40, n=40000), " exact 4/3 =", 4 / 3)
    print("D_2 formula check (m=2):", D_reduced_quadrature(2, T=16, n=400), " exact 29/6 - sqrt6 =", 29 / 6 - sqrt(6))
    d3q = D_reduced_quadrature(3, T=14, n=110)
    print("D_3 orthant quadrature (m=3):", d3q)
    d3r = D3_onedim(T=30.0, n=1200)
    print("D_3 via a-integration (2-D quadrature in p,q):", d3r)
    try:
        mc, se = D_direct_mc(3, n=1000000)
        print("D_3 Monte Carlo:", mc, "+-", se)
        mc2, se2 = D_direct_mc(2, n=1000000)
        print("D_2 Monte Carlo:", mc2, "+-", se2)
    except Exception as e:
        print("MC skipped:", e)
