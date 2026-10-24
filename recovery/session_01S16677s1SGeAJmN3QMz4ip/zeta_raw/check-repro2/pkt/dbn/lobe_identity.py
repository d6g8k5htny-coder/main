"""Lobe-area identity for consecutive zeros a < c of H_t (here t = 0, a = 2 gamma_1, c = 2 gamma_2):

    int_a^c H_t = (1/2) int_a^c (x - a)(c - x) dH_t/dt dx,

i.e. the lobe area is a weighted integral of d_t H. It is the integration-by-parts identity
int_a^c H = -(1/2) int_a^c (x-a)(c-x) H'' combined with the backward heat equation.
Uses the independent tanh-sinh implementation in indep_ht.py. Both sides print
-6.31861166707933e-4. Stdout saved as results/dbn/lobe_identity.out.
"""
import mpmath as mp

from indep_ht import Phi, pts, Hder

mp.mp.dps = 40
a = 2*mp.zetazero(1).imag; c = 2*mp.zetazero(2).imag
Fz = lambda z: mp.quad(lambda u: Phi(u)*mp.sin(z*u)/u, pts(z))
lhs = Fz(c) - Fz(a)
# rhs: 1/2 int (x-a)(c-x) dH/dt dx, with dH/dt = -H_zz
rhs = mp.quad(lambda x: (x-a)*(c-x)*(-Hder(0, x, 2))/2, [a, (a+c)/2, c])
print(mp.nstr(lhs, 15), mp.nstr(rhs, 15))
