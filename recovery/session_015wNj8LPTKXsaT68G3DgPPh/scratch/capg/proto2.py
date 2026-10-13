# check the 1-D decomposition J = [(512/3) I1 - 8 I2 - I3] / (216 k^3) against the 2-D quadrature of proto.py
import math
from math import erf, exp, sqrt, pi
from proto import J_quad
def Phi(x): return 0.5*(1+erf(x/sqrt(2)))
def phi(x): return exp(-x*x/2)/sqrt(2*pi)
def F1(t): return 2*Phi(t/sqrt(2))-1   # |N(0,2)| cdf : 2|v| and |om|
def F3(t): return 2*Phi(t/sqrt(6))-1   # |Y|
def integ(f,a,b,n=200000):
    h=(b-a)/n; return sum(f(a+(i+0.5)*h) for i in range(n))*h
def inc_mom(j, lo, hi, sig):
    # E[|X|^j 1{lo<|X|<hi}] for X~N(0,sig^2), via numeric (float control only)
    return integ(lambda x: 2*x**j*phi(x/sig)/sig, lo, hi, 20000)
def decomposition(k):
    m=12*k; T=m+40
    I1 = m**6 + integ(lambda t: 6*t**5*(1-F1(t)**2*F3(t)), m, T)
    H12 = m*m + integ(lambda t: 2*t*(1-F1(t)*F3(t)), m, T)
    sv=sqrt(0.5)
    Ev4_in = inc_mom(4, 0, 6*k, sv); Ev6_out = inc_mom(6, 6*k, 6*k+40, sv)
    B = lambda t: inc_mom(4, 6*k, t/2, sv) if t/2>6*k else 0.0
    I2 = H12*Ev4_in + 4*Ev6_out + integ(lambda t: 2*t*(1-F1(t)*F3(t))*B(t), m, T, 2000)
    # I3 = E[F(lo)]
    so=sqrt(2)
    def G(v):
        th=v*v/(3*k); u=th/so
        E0le=Phi(u); E1le=-so*phi(u)
        E0gt=1-Phi(u); E1gt=so*phi(u); E2gt=so**2*(u*phi(u)+1-Phi(u)); E3gt=so**3*(u*u*phi(u)+2*phi(u))
        piece1 = -(2/3)*v**6*E0le + 3*k*v**4*E1le
        # c = 6k om - v^2 ; c^2 (3 v^2 - c)/6 = (3 v^2 c^2 - c^3)/6
        # c^2 = 36k^2 om^2 -12 k v^2 om + v^4 ; c^3 = 216k^3 om^3 - 108 k^2 v^2 om^2 + 18 k v^4 om - v^6
        c2 = 36*k*k*E2gt - 12*k*v*v*E1gt + v**4*E0gt
        c3 = 216*k**3*E3gt - 108*k*k*v*v*E2gt + 18*k*v**4*E1gt - v**6*E0gt
        piece2 = (3*v*v*c2 - c3)/6
        return piece1+piece2
    I3 = 2*integ(lambda v: G(v)*phi(v/sv)/sv, 0, 10, 20000)
    J = ((512/3)*I1 - 8*I2 - I3)/(216*k**3)
    return J, I1, I2, I3
for k in [1/6,1/2,2]:
    J,I1,I2,I3 = decomposition(k)
    print('k=%g J_dec=%.8g  J_quad=%.8g  I1=%.6g I2=%.6g I3=%.6g  rel(I2,I3)=%.2e %.2e' % (k,J,J_quad(k,300,300),I1,I2,I3, 8*I2/((512/3)*I1), abs(I3)/((512/3)*I1)))
