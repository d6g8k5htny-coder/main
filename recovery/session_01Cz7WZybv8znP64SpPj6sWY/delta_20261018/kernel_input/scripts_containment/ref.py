"""Independent mpmath reference for the original-law spectral sums.
S_k(L) = sum_{m in Z} (h m)^{2k} exp(-alpha m^2), h = 2 pi/L, alpha = h^2/2.
z = S_0; mu_{2k} = S_k / S_0.  Two independent evaluations:
 (A) direct lattice summation with an explicit rigorous integral tail bound;
 (B) Poisson-dual (Hermite) series S_k = L/sqrt(2pi) sum_n (-1)^k He_{2k}(nL) exp(-(nL)^2/2).
"""
import mpmath as mp
from fractions import Fraction

def direct(L, dps=110):
    with mp.workdps(dps + 20):
        L = mp.mpf(L.numerator) / L.denominator if isinstance(L, Fraction) else mp.mpf(L)
        h = 2 * mp.pi / L
        a = h * h / 2
        target = (dps + 30) * mp.log(10)
        N = int(mp.ceil(mp.sqrt((target + 60) / a))) + 5
        N = max(N, int(mp.ceil(mp.sqrt(3 / a))) + 1)
        S = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        for m in range(1, N + 1):
            e = 2 * mp.exp(-a * m * m)
            hm2 = (h * m) ** 2
            p = mp.mpf(1)
            for k in range(4):
                S[k] += p * e
                p *= hm2
        # rigorous tail bound (same elementary integral lemma, evaluated in mpmath with big margin)
        E = mp.exp(-a * N * N)
        I = [E / (2 * a * N)]
        for k in range(1, 4):
            I.append((mp.mpf(N) ** (2 * k - 1) * E + (2 * k - 1) * I[-1]) / (2 * a))
        tails = [2 * h ** (2 * k) * I[k] for k in range(4)]
        return {"S": S, "tails": tails, "N": N,
                "z": S[0], "mu2": S[1] / S[0], "mu4": S[2] / S[0], "mu6": S[3] / S[0]}

def He(n, x):
    if n == 0: return mp.mpf(1)
    if n == 2: return x**2 - 1
    if n == 4: return x**4 - 6*x**2 + 3
    if n == 6: return x**6 - 15*x**4 + 45*x**2 - 15
    raise ValueError

def dual(L, dps=400):
    with mp.workdps(dps):
        L = mp.mpf(L.numerator) / L.denominator if isinstance(L, Fraction) else mp.mpf(L)
        pref = L / mp.sqrt(2 * mp.pi)
        target = dps * mp.log(10) + 60
        N = int(mp.ceil(mp.sqrt(2 * target) / L)) + 5
        S = []
        for k in range(4):
            s = (-1) ** k * He(2 * k, mp.mpf(0))
            for n in range(1, N + 1):
                x = n * L
                s += 2 * (-1) ** k * He(2 * k, x) * mp.exp(-x * x / 2)
            S.append(pref * s)
        return {"S": S, "z": S[0], "mu2": S[1] / S[0], "mu4": S[2] / S[0], "mu6": S[3] / S[0]}
