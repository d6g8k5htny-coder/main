"""Shared high-precision evaluation of the de Bruijn--Newman family.

Conventions (Rodgers--Tao 2020, Polymath15 2019):
    Phi(u) = sum_{n>=1} (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) exp(-pi n^2 e^{4u})
    H_t(z) = int_0^inf e^{t u^2} Phi(u) cos(z u) du
    Xi(x)  = xi(1/2 + i x),  xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s)
Claimed: H_0(z) = Xi(z/2)/8.

H_t is evaluated by composite Gauss--Legendre quadrature on [0, U]; Phi is
precomputed once at the nodes, so many (t, z) evaluations are cheap.

This is one of three independent H_t implementations in this packet (the others are
the fixed-point trapezoid rule in hflow.py and the tanh-sinh quadrature in
indep_ht.py); they are kept deliberately so that each cross-checks the others.
"""
from mpmath import mp
from mpmath.calculus.quadrature import GaussLegendre

mp.dps = 60


def Phi(u, nmax=40):
    u = mp.mpf(u) if not isinstance(u, mp.mpc) else u
    e4 = mp.exp(4 * u)
    s = 0
    for n in range(1, nmax + 1):
        term = (2 * mp.pi**2 * n**4 * mp.exp(9 * u) - 3 * mp.pi * n**2 * mp.exp(5 * u)) * mp.exp(-mp.pi * n**2 * e4)
        s += term
        if n > 3 and abs(term) < mp.mpf(10) ** (-mp.dps - 10):
            break
    return s


def xi(s):
    return s * (s - 1) / 2 * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def Xi(x):
    return xi(mp.mpf(1) / 2 + 1j * x)


class HFlow:
    def __init__(self, U=1.3, pieces=52, degree=5):
        gl = GaussLegendre(mp)
        base = gl.calc_nodes(degree, mp.prec)  # nodes on [-1, 1]
        h = mp.mpf(U) / pieces
        self.u = []
        self.w = []
        for p in range(pieces):
            a = p * h
            for (x, wt) in base:
                self.u.append(a + (x + 1) * h / 2)
                self.w.append(wt * h / 2)
        self.phi = [Phi(u) for u in self.u]
        self.u2 = [u * u for u in self.u]
        self._cache_t = None

    def _weights(self, t):
        t = mp.mpf(t)
        if self._cache_t != t:
            self._wt = [w * mp.exp(t * u2) * ph for w, u2, ph in zip(self.w, self.u2, self.phi)]
            self._cache_t = t
        return self._wt

    def H(self, t, z):
        wt = self._weights(t)
        return mp.fsum(c * mp.cos(z * u) for c, u in zip(wt, self.u))

    def Hz(self, t, z):  # d/dz
        wt = self._weights(t)
        return -mp.fsum(c * u * mp.sin(z * u) for c, u in zip(wt, self.u))

    def Hzz(self, t, z):
        wt = self._weights(t)
        return -mp.fsum(c * u * u * mp.cos(z * u) for c, u in zip(wt, self.u))

    def Hzzz(self, t, z):
        wt = self._weights(t)
        return mp.fsum(c * u**3 * mp.sin(z * u) for c, u in zip(wt, self.u))
