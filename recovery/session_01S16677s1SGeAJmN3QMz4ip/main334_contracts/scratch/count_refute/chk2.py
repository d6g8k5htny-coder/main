import math
import numpy as np
def a_of(alpha): return math.sqrt((alpha+2)/alpha)
def trap(f,lo,hi,n=400000):
    x=np.linspace(lo,hi,n); y=f(x); return np.trapezoid(y,x)
d=2
for alpha in [0.5,2/3,1,3,4,4.5,6]:
    a=a_of(alpha)
    for m in (2*d,2*d+2):
        # E A^-m = alpha a^-alpha int_0^a u^(alpha-1-m) du: finite iff alpha>m
        print(alpha,m,"finite" if alpha>m else "DIVERGES (exponent %.2f)"%(alpha-1-m))
nuH=lambda s: s**(-1/3)*np.exp(-s)
for alpha in [0.5,1,3]:
    a=a_of(alpha); ell=0.1
    lhs=trap(lambda u: alpha*a**(-alpha)*u**(alpha-1)*nuH(ell/u)/u,1e-6,a)
    rhs=alpha*a**(-alpha)*ell**(alpha-1)*trap(lambda s: s**(-alpha)*nuH(s),ell/a,60)
    print("alpha",alpha,"E[A^-1 nu_H(ell/A)]=",lhs," AMP(9)=",rhs)
A=np.linspace(1e-4,2,100000); ell=0.05
print("sup_A A^-(2d+2) exp(-ell^2/A^2) =",(A**(-(2*d+2))*np.exp(-ell**2/A**2)).max(), " (bounded, max at A^2=ell^2/(d+1))")
