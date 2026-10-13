import numpy as np
from scipy import integrate
# fast grid version of the ratio check: substitute b=A b', k=A k' analytically is what we test, so do it numerically on grids
B0=lambda b,k:(1+np.abs(b)+k)**4*np.exp(-(b*b+k*k))
b=np.linspace(-12,12,2401); k=np.linspace(1e-6,12,2401)
Bg,Kg=np.meshgrid(b,k,indexing='ij')
IH=np.trapz(np.trapz(Kg**(-2/3)*B0(Bg,Kg),k,axis=1),b)
for alpha in [0.7,0.8,1.0,1.5,3.0]:
    a=np.sqrt((alpha+2)/alpha)
    s=np.linspace(1e-6,a,4001)
    w=alpha*a**(-alpha)*s**(alpha-1)*s**(-2)
    BF=np.zeros_like(Bg)
    for si,wi in zip(s,w):
        BF+=wi*B0(Bg/si,Kg/si)
    BF*= (s[1]-s[0])
    IF=np.trapz(np.trapz(Kg**(-2/3)*BF,k,axis=1),b)
    pred=a**(-2/3)*alpha/(alpha-2/3)
    print(f"alpha={alpha}: closed E A^-2/3={pred:.6f}  ratio={IF/IH:.6f}")
# far density behaviour with nu_far^H=1{s<=1}
for alpha in [0.5,2/3,0.8,1.0,1.5]:
    a=np.sqrt((alpha+2)/alpha)
    def nuF(l):
        lo=l/a
        if lo>=1: return 0.0
        return alpha*a**(-alpha)*l**(alpha-1)*integrate.quad(lambda s:s**(-alpha),lo,1)[0]
    print(alpha,[(l, round(nuF(l),3), round(l**(1/3)*nuF(l),4)) for l in [1e-2,1e-4,1e-6,1e-8]])
