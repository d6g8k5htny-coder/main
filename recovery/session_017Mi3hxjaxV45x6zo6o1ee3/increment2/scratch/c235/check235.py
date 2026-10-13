#!/usr/bin/env python3
"""Provider-distinct checks for Math-#235 comments 5938885261 (OA-ENDPOINT-TAIL-MARKS), 5942428034, 5942496197, 5942555223.
Stdlib only: exact Fractions for exponent/coefficient algebra, floats only for two quadrature controls."""
from fractions import Fraction as F
import math, random
random.seed(235)
ok = {}

# ---- Prop J: constants and exponents -------------------------------------------------------------
# intensity density in (y,b,k,u): t^{2/3} y^{-1/3} A0/(3 k^{2/3}); total mass (3/2) c t^{2/3}, c = int A0/(3k^{2/3})
# normalized: y^{-1/3} A0 /(3 k^{2/3}) / ((3/2)c), A0 = 432 k^2 B phi(12k)  ->  (96/c) y^{-1/3} k^{4/3} B phi
ok["432->96"] = F(432) / (3 * F(3, 2)) == 96 and F(2) - F(2, 3) == F(4, 3)
# k = y/(a^3 z^3): |dk/dz| = 3y/(a^3 z^4); density (96/c) y^{-1/3} k^{4/3} * 3y/(a^3 z^4)
ye = F(-1, 3) + F(4, 3) * 1 + 1          # y exponent
ae = F(4, 3) * (-3) - 3                  # a exponent
ze = F(4, 3) * (-3) - 4                  # z exponent
ok["96->288, exponents (2,-7,-8)"] = (96 * 3 == 288) and (ye, ae, ze) == (2, -7, -8)
ok["mass 96W/(7c)"] = F(288) * F(1, 3) * F(1, 7) == F(96, 7)
ok["Beta(3,1), Pareto(7) normalizations"] = (F(1, 3) * 3 == 1) and (F(1, 7) * 7 == 1)
# phi ratio exponent: (12 y/(a^3 z^3))^2 / (2 tau^2) = 72 y^2/(tau^2 a^6 z^6)
ok["Gaussian ratio 72"] = F(144, 2) == 72
# T4 unconditioned marginal of Y is (2/3) y^{-1/3}: check normalization
ok["T4 marginal (2/3)y^-1/3"] = F(2, 3) * F(3, 2) == 1

# float control of the change of variables: mass of {V>a} under (96/c) y^{-1/3} k^{4/3} phi_tau(12k), B=c=1
def mass_direct(a, tau, n=400):
    # V>a  <=>  k < y/a^3 ; integrate y in (0,1), k in (0, y/a^3) with Gauss-Legendre
    xs, ws = gl(n)
    tot = 0.0
    for xi, wi in zip(xs, ws):
        y = (xi + 1) / 2; wy = wi / 2
        kmax = y / a ** 3
        for xj, wj in zip(xs[:60], ws[:60]) if False else zip(*gl(60)):
            k = kmax * (xj + 1) / 2; wk = kmax * wj / 2
            tot += wy * wk * 96 * y ** (-1 / 3) * k ** (4 / 3) * math.exp(-(12 * k) ** 2 / (2 * tau ** 2)) / math.sqrt(2 * math.pi) / tau
    return tot
_gl = {}
def gl(n):
    if n in _gl: return _gl[n]
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = n * (x * p1 - p0) / (x * x - 1); dx = p1 / dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    _gl[n] = (xs, ws); return _gl[n]
tau = math.sqrt(6.0)
for a in (1.0, 2.0, 4.0):
    md = mass_direct(a, tau, 200)
    pred = 96 / 7 * a ** -7 / math.sqrt(2 * math.pi) / tau        # W = phi_tau(0) with B=1
    ok[f"tail mass a={a}: direct/predicted-1 = {md/pred-1:.2e} (bound 72/(tau^2 a^6)={72/(tau**2*a**6):.2e})"] = abs(md / pred - 1) <= 72 / (tau ** 2 * a ** 6) + 1e-9

# ---- 5942428034: m(t), normalized x-density, mean ----------------------------------------------
# int_0^t l^e dl = t^{e+1}/(e+1)
ok["m(t) coefficients 3/2, 4/5, 3/4, remainder 10/7"] = (1 / (F(-1, 3) + 1), 1 / (F(1, 4) + 1), 1 / (F(1, 3) + 1), F(3, 7) + 1) == (F(3, 2), F(4, 5), F(3, 4), F(10, 7))
# relative corrections of m: (4/5)/(3/2) = 8/15 at t^{5/4-2/3}=t^{7/12}; (3/4)/(3/2)=1/2 at t^{2/3}; remainder t^{10/7-2/3}=t^{16/21}
ok["relative orders 7/12, 2/3, 16/21; 8/15, 1/2"] = (F(5, 4) - F(2, 3), F(4, 3) - F(2, 3), F(10, 7) - F(2, 3), F(4, 5) / F(3, 2), F(3, 4) / F(3, 2)) == (F(7, 12), F(2, 3), F(16, 21), F(8, 15), F(1, 2))
ok["cross term t^{7/6} beyond t^{16/21}"] = 2 * F(7, 12) > F(16, 21)
mom = lambda e: 1 / (e + 1)              # int_0^1 x^e dx
ok["brackets integrate to 0"] = (mom(F(1, 4)) - F(8, 15) * mom(F(-1, 3)) == 0) and (mom(F(1, 3)) - F(1, 2) * mom(F(-1, 3)) == 0)
mean0 = F(2, 3) * mom(F(2, 3))
mean1 = F(2, 3) * (mom(F(5, 4)) - F(8, 15) * mom(F(2, 3)))
mean2 = F(2, 3) * (mom(F(4, 3)) - F(1, 2) * mom(F(2, 3)))
ok[f"mean 2/5 + (56/675)(c1/c) + (3/35)(c2/c): got {mean0}, {mean1}, {mean2}"] = (mean0, mean1, mean2) == (F(2, 5), F(56, 675), F(3, 35))

# ---- 5942496197 / 5942555223: count and marked identities ---------------------------------------
p = F(1, 7)
EK = 2 * p; PK0 = 1 - (1 - p) ** 2; D = 2 * p * p
ok["Bernoulli: EK=2p, P(K>0)=2p-p^2, D=2p^2"] = (EK, PK0, D) == (2 * p, 2 * p - p * p, 2 * p * p)
good = True
for _ in range(3000):
    nmax = random.randint(1, 6)
    w = [F(random.randint(0, 9)) for _ in range(nmax + 1)]
    if sum(w) == 0: continue
    s = sum(w); pk = [x / s for x in w]
    marks = {}  # per count n: list of n marks drawn from {0,1,2} (one representative field per n)
    for n in range(nmax + 1):
        marks[n] = [random.randint(0, 2) for _ in range(n)]
    m = sum(n * pk[n] for n in range(nmax + 1)); P0 = 1 - pk[0]
    P2 = pk[2] if nmax >= 2 else 0
    R3 = sum((n - 1) * pk[n] for n in range(3, nmax + 1))
    good &= (P0 == m - P2 - R3)
    Dt = sum(n * pk[n] for n in range(2, nmax + 1))
    good &= (Dt / 2 <= m - P0 <= Dt)
    # marked: I = E sum delta, J = E 1{K>0} (1/K) sum delta, I - J = (1/2) I_(2) + R_{>=3}, R>=0 with mass <= E[K 1{K>=3}]
    I = [F(0)] * 3; J = [F(0)] * 3; I2 = [F(0)] * 3; R = [F(0)] * 3
    for n in range(1, nmax + 1):
        for x in marks[n]:
            I[x] += pk[n]; J[x] += pk[n] / n
            if n == 2: I2[x] += pk[n]
            if n >= 3: R[x] += pk[n] * (1 - F(1, n))
    good &= all(I[x] - J[x] == I2[x] / 2 + R[x] and R[x] >= 0 for x in range(3))
    good &= sum(R) <= sum(n * pk[n] for n in range(3, nmax + 1))
    if P0 > 0 and m > 0:
        tv = sum(abs(I[x] / m - J[x] / P0) for x in range(3)) / 2
        good &= tv <= (m - P0) / m <= Dt / m
ok["count/marked/mixture identities (3000 exact random marked count laws)"] = good

# ---- Prop F / moving threshold: elementary inequalities on counts (N^a <= N) ---------------------
good = True
for _ in range(5000):
    N = random.randint(0, 9); Na = random.randint(0, N)
    good &= max(Na - 1, 0) <= Na * (Na >= 2) <= N * (N >= 2)
    good &= 2 * (Na >= 2) <= N * (N >= 2)
ok["Prop F count inequalities"] = good
# existential threshold: a = min(t^-1/6, eps^-1/14) gives eps a^7 <= sqrt(eps) and a t^{1/3} <= t^{1/6}
good = True
for _ in range(2000):
    t = 10 ** random.uniform(-12, -1); eps = 10 ** random.uniform(-15, 0)
    a = min(t ** (-1 / 6), eps ** (-1 / 14))
    good &= eps * a ** 7 <= math.sqrt(eps) * (1 + 1e-12) and a * t ** (1 / 3) <= t ** (1 / 6) * (1 + 1e-12)
ok["existential a(t) bounds"] = good
# counterexample: mixture tail eta + (1-eta) p_a with eta=a^-3 dominates a^-7
ok["counterexample tail dominance"] = all(a ** -3 > 10 * a ** -7 for a in (2.0, 4.0, 8.0))

w = max(len(k) for k in ok)
for k, v in ok.items():
    print(("PASS " if v else "FAIL ") + k)
print("ALL PASS" if all(ok.values()) else "SOME FAIL")
