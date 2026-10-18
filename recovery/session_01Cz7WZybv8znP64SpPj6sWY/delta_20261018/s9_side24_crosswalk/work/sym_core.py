"""Core symbolic identities for the S9 crosswalk (own check, not a review)."""
import sympy as sp
from mpmath import mp, mpf, gamma as G, sqrt as msqrt, pi as mpi, exp as mexp
s6, s3 = sp.sqrt(6), sp.sqrt(3)
g16 = sp.gamma(sp.Rational(1, 6))
g76 = sp.gamma(sp.Rational(7, 6))
pi = sp.pi
r = sp.Rational
out = {}
# PROOF.md formula (1), d=3 and d=2
D = {1: r(4, 3), 2: r(29, 6) - s6}
c_ref = {d: g76 * r(3, 2)**r(1, 3) * D[d-1] / (2*s3*pi**(d-1)*sp.sqrt(pi)) for d in (2, 3)}
# LS-DER-075 eq (21)
c_bf_075 = 2**r(1, 6) * 3**r(1, 3) * (29*s6 - 36) * g16 / (432 * pi**r(5, 2))
# LS-CLS-077 closed form
c_inf_077 = 2**r(-23, 6) * 3**r(-8, 3) * (29*s6 - 36) * g16 * pi**r(-5, 2)
def same(a, b):
    q = sp.nsimplify(sp.simplify(sp.gammasimp(a / b)))
    return q
out['(1)/075eq21'] = same(c_ref[3], c_bf_075)
out['(1)/077'] = same(c_ref[3], c_inf_077)
out['075eq21/077'] = same(c_bf_075, c_inf_077)
# (15.2) with the reference law -> (1)
tau2 = 15 - 9  # Var(t)-Cov(t,G1)^2/Var(G1)
for d in (2, 3):
    pG = (2*pi)**(-r(d, 2))
    pV = (2*pi)**(-r(d, 2)) / s3
    S = {2: 2*pi, 3: 4*pi}[d]
    c152 = g76 / (24**r(1, 3) * sp.sqrt(pi)) * S * pG * pV * sp.Integer(tau2)**r(2, 3) * D[d-1]
    out[f'(15.2)ref/(1) d={d}'] = same(c152, c_ref[d])
# P's scalar gamma integral: 144 int_0^oo k^(4/3) phi_tau(12k) dk
k, tau = sp.symbols('k tau', positive=True)
phi = sp.exp(-(12*k)**2/(2*tau**2)) / (sp.sqrt(2*pi)*tau)
I = sp.integrate(144 * k**r(4, 3) * phi, (k, 0, sp.oo))
out['P scalar integral / stated'] = same(I, g76 * tau**r(4, 3) / (24**r(1, 3)*sp.sqrt(pi)))
# LS-DER-075 recombination (15)-(19) at d=3 BF: a=1, m4=3, Omega=6, chi=15
a, m4, chi = 1, 3, 15
Om = a*chi - m4**2
n = 2
beta = r(m4, 3)
B = D[2] * (2*pi*m4)**r(-1, 2) * (2*pi)**(-n) * sp.Integer(a)**(-r(n, 2)) * beta**r(n, 2)
K = (3*g76/(pi*sp.sqrt(Om))) * (r(Om, 2*a))**r(7, 6)
c_rec = r(1, 6) * 6**r(2, 3) / 3 * 4*pi * B * K
out['075 recombination/eq21'] = same(c_rec, c_bf_075)
Ajet = 4*pi*D[2]*g16 / (2**5 * 3**r(14, 6) * pi**r(7, 2))
out['075 eq18-19/eq21'] = same(Ajet * sp.Integer(Om)**r(2, 3) * sp.Integer(m4)**r(1, 2) / sp.Integer(a)**r(13, 6), c_bf_075)
for kk, v in out.items():
    print(f'{kk}: {v}')
mp.dps = 60
c3 = G(mpf(7)/6) * (mpf(3)/2)**(mpf(1)/3) * (mpf(29)/6 - msqrt(6)) / (2*msqrt(3)*mpi**mpf(2.5))
print('c_3,ref =', mp.nstr(c3, 40))
lo, hi = mpf('0.04177593184059834334'), mpf('0.04177593184059834335')
print('inside PROOF enclosure:', lo < c3 < hi)
P324 = mpf(-620813376)/35
rel = P324 * mexp(-288)
print('P_3(24) e^-288 =', mp.nstr(rel, 10), ' |.| < 1e-106:', abs(rel) + mpf(10)**-180 < mpf(10)**-106)
c324 = c3 * (1 + rel)
print('c_(3,24) (chain, eps dropped) =', mp.nstr(c324, 40), ' inside:', lo < c324 < hi)
print('digits agreeing c_ref vs chain c_(3,24): first diff at ~1e', mp.nstr(mp.log10(abs(c324 - c3)), 6))
