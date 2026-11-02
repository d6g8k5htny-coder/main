"""Section 5 checks: C4 row-norm constant, perturbation ledger, count-cap transfer arithmetic,
and a numerical evaluation of the restricted fixed-base coefficient c_W of eq. (8)."""
import sys, json
from fractions import Fraction as Q
import numpy as np
sys.path.insert(0, sys.argv[1])
from jetcov import modes, gamma

out = {}
# (a) row norm of d^alpha H_N at a point = sqrt(sum_n q_n xi^(2 alpha)), |alpha|<=4
rn = {}
for N in (4, 8, 32):
    n1, n2, _ = modes(2 * N); q = 1.0 / (4 * N + 1) ** 2
    x1 = n1 / N; x2 = n2 / N
    alphas = [(a, b) for s in range(5) for a in range(s + 1) for b in [s - a]]
    assert len(alphas) == 15
    norms = [float(np.sqrt(np.sum(q * x1 ** (2 * a) * x2 ** (2 * b)))) for a, b in alphas]
    rn[N] = dict(max_row_norm=max(norms), claimed_bound=64, n_multi=len(alphas))
    assert max(norms) <= 64
out["C4_row_norms"] = rn
# (b) ledger arithmetic with exact rationals
led = []
for N in range(4, 200):
    eps = Q(1, N ** 12)
    e6 = 960 ** 2 * eps * (4 * N + 1) ** 2
    assert e6 <= Q(960 ** 2 * 25, N ** 10)
    assert 1 - (1 - float(eps)) ** 0.5 <= float(eps) * (1 + 1e-12)
    cap = (16 * (2 * N) ** 2) ** 2; assert cap == 4096 * N ** 4
    assert (cap + 4096 * N * N) * Q(1, N ** 10) <= Q(8192, N ** 6)
    # radius in window: ell<=t_N=N^-3, k>=1  =>  r=(ell/k)^(1/3)<=1/N
    assert (Q(1, N ** 3)) <= Q(1, N) ** 3
led.append("exact ledger checks passed for N=4..199")
out["ledger"] = led
# also: N^2 * t_N^(2/3) = 1
out["N2_tN23"] = [float(N ** 2 * (N ** -3.0) ** (2 / 3)) for N in (4, 10, 100)]

# (c) restricted coefficient c_W for the base G (uniform 1/25, K=2, L=2pi)
wb = np.full(25, 1 / 25)
gb, gwb = np.polynomial.legendre.leggauss(24)
B = gb; WB = gwb                      # b in [-1,1]
Kk = 1.5 + 0.5 * gb; WK = 0.5 * gwb   # k in [1,2]
Aa = -1.5 + 0.5 * gb; WA = 0.5 * gwb  # A in [-2,-1]
def theta_integrand(t, window=True):
    G = gamma(wb, 2, 1.0, t)
    GU = G[:6, :6]; GAU = G[6, :6]; GAA = G[6, 6]
    Gi = np.linalg.inv(GU); det = np.linalg.det(GU)
    s2 = GAA - GAU @ Gi @ GAU; s = np.sqrt(s2)
    tot = 0.0
    for b, wbq in zip(B, WB):
        for k, wk in zip(Kk, WK):
            v = np.array([b, 0, 0, 12 * k, 0, 0])
            pi0 = np.exp(-0.5 * v @ Gi @ v) / ((2 * np.pi) ** 3 * np.sqrt(det))
            mu = GAU @ Gi @ v
            if window:
                dens = np.exp(-0.5 * ((Aa - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
                EA = np.sum(WA * Aa ** 2 * dens)
            z = 36 * k * k * EA
            tot += wbq * wk * k ** (-2 / 3) * pi0 * z
    return tot, s2
ths = np.linspace(0, 2 * np.pi, 361)[:-1]
vals = [theta_integrand(t) for t in ths]
I = np.mean([v[0] for v in vals]) * 2 * np.pi  # unnormalized arclength
V0 = (2 * np.pi) ** 2
cW = 6 * V0 * I
out["c_W"] = dict(c_W=cW, limiting_mean=cW * (1 - 2 ** (-2 / 3)),
                  min_cond_var_A=float(min(v[1] for v in vals)),
                  min_theta_integrand=float(min(v[0] for v in vals)))
print(json.dumps(out, indent=1, default=float))
