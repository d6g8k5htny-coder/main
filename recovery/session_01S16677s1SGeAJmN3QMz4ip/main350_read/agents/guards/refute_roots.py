from fractions import Fraction as Q
import sympy as sp
from mpmath import mp, mpf, exp
r1 = Q(1,32); a_star = r1/4; delta = Q(1,2**26); b = Q(4,9)
u0 = Q(3,2**32); v0 = Q(1,2**47)
s = sp.symbols('s')
corners = ((u0, v0), (u0, -v0), (-u0, v0), (-u0, -v0), (u0, Q(0)), (Q(0), v0), (Q(0), Q(0)), (u0/2, v0/3))
for (u, v) in corners:
    poly = sp.Poly(-s**3 + sp.Rational(u.numerator, u.denominator)*s + sp.Rational(v.numerator, v.denominator), s)
    roots = poly.real_roots()
    # exact-ish checks via high-precision evaluation plus rational interval isolation
    ok_cont = all(abs(sp.N(r, 60)) < sp.Rational(a_star.numerator, a_star.denominator) for r in roots)
    heights = [sp.N((-r**4/4 + sp.Rational(u.numerator,u.denominator)*r**2/2 + sp.Rational(v.numerator,v.denominator)*r), 60) for r in roots]
    ok_h = all(abs(h) < sp.Rational(delta.numerator, delta.denominator)/4 for h in heights)
    print(f'u={u} v={v}: n_real={len(roots)} roots~{[sp.N(r,12) for r in roots]} |s|<a*:{ok_cont} |P|<delta/4:{ok_h} rho<r1/2:{all(abs(sp.N(r,60)) < sp.Rational(1,64) for r in roots)}')
# Section 6 tied case v=0,u>0: heights b+u^2/4 twice and b
u = u0
poly = sp.Poly(-s**3 + sp.Rational(u.numerator,u.denominator)*s, s)
print('v=0 roots:', poly.real_roots(), ' P at roots:', [sp.nsimplify(-r**4/4 + sp.Rational(u.numerator,u.denominator)*r**2/2) for r in poly.real_roots()], ' u^2/4=', u**2/4)
# endpoint behaviour of psi' (G3), 60 digits
mp.dps = 60
def q(t): return exp(-1/t) if t > 0 else mpf(0)
def qp(t): return t**-2*exp(-1/t) if t > 0 else mpf(0)
def psip(t):
    if t <= 0 or t >= 1: return mpf(0)
    D = q(t)+q(1-t); return -(qp(1-t)*q(t) + q(1-t)*qp(t))/D**2
for t in [mpf(10)**-k for k in (1,2,3,6,12,30)] + [1-mpf(10)**-k for k in (1,2,3,6,12,30)]:
    D = q(t)+q(1-t)
    print(f"t={sp.N(t,8)}: |psi'|={sp.N(abs(psip(t)),10)}  M/D={sp.N(4*exp(-2)/D,10)}  D>=e^-2:{D >= exp(-2)}  |psi'|<=4:{abs(psip(t))<=4}")
r_, p_ = sp.symbols('r p')
print('CF.md c-formula == (G7):', sp.expand((r_/4 + (1-r_/4)*(1-p_)) - (1 - (1-r_/4)*p_)) == 0)
