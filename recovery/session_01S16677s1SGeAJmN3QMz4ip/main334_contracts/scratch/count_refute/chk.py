import math
from scipy import integrate
# A = a R, R ~ alpha r^(alpha-1) on (0,1); E A^2 = 1
def a_of(alpha): return math.sqrt((alpha+2)/alpha)
def EAneg(alpha, m):
    a=a_of(alpha)
    f=lambda u: alpha*a**(-alpha)*u**(alpha-1-m)
    try:
        v,err=integrate.quad(f,0,a,limit=200)
        return v if alpha>m else float('inf')
    except Exception as e: return float('inf')
d=2
for alpha in [0.5,2/3,1,3,4,4.5,6]:
    print(alpha, "E[A^-2d]=",EAneg(alpha,2*d), "E[A^-(2d+2)]=",EAneg(alpha,2*d+2))
# transform check: nu_H(s)=s^(-1/3) exp(-s) ; compare E[A^-1 nu_H(ell/A)] with AMP (9)
nuH=lambda s: s**(-1/3)*math.exp(-s)
for alpha in [0.5,1,3]:
    a=a_of(alpha); ell=0.1
    lhs,_=integrate.quad(lambda u: alpha*a**(-alpha)*u**(alpha-1)*nuH(ell/u)/u,0,a,limit=200)
    rhs=alpha*a**(-alpha)*ell**(alpha-1)*integrate.quad(lambda s: s**(-alpha)*nuH(s),ell/a,math.inf,limit=200)[0]
    print(alpha,lhs,rhs)
# bounded A-integrand for p_O^F at ell>0: A^-(2d+2) exp(-c ell^2/A^2) sup over (0,a)
import numpy as np
A=np.linspace(1e-4,2,100000); c=1.0; ell=0.05
print("sup A^-(2d+2)exp(-c ell^2/A^2) =", (A**(-(2*d+2))*np.exp(-c*ell**2/A**2)).max())
