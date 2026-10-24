# toy far density nu_H,far(s)=1 on (0,1], 0.5*exp(-(s-1)) beyond (bounded, finite mass)
import math
def nuH(s): return 1.0 if s<=1 else 0.5*math.exp(-(s-1))
def I(ell,alpha,a):
    # integral_{ell/a}^inf s^-alpha nuH(s) ds, numeric
    lo=ell/a; n=200000; tot=0.0
    # piece on [lo,1]
    if lo<1:
        h=(1-lo)/n
        for i in range(n):
            s=lo+(i+0.5)*h; tot+=s**(-alpha)*h
    # tail
    h=20.0/n
    for i in range(n):
        s=max(lo,1)+(i+0.5)*h; tot+=s**(-alpha)*nuH(s)*h
    return tot
for alpha in [0.5,2/3,0.8,1.0,1.5]:
    a=math.sqrt((alpha+2)/alpha)
    for ell in [1e-2,1e-4,1e-6]:
        nuF=alpha*a**(-alpha)*ell**(alpha-1)*I(ell,alpha,a)
        print(f"alpha={alpha:.3f} ell={ell:.0e} ell^(1/3)nuF={ell**(1/3)*nuF:.4g} nuF={nuF:.4g}")
