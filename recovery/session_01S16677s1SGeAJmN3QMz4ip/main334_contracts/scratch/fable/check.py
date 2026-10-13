import numpy as np
from scipy import integrate
# toy H-side limit integrand B0(b,k) with Gaussian decay (shape of SG (19)), d=2
B0=lambda b,k:(1+abs(b)+k)**4*np.exp(-(b*b+k*k))
for alpha in [0.8,1.0,1.5,3.0]:
    a=np.sqrt((alpha+2)/alpha)
    pA=lambda s: alpha*a**(-alpha)*s**(alpha-1)
    EAm23=integrate.quad(lambda s:pA(s)*s**(-2/3),0,a)[0]
    pred=a**(-2/3)*alpha/(alpha-2/3)
    IH=integrate.dblquad(lambda b,k:k**(-2/3)*B0(b,k),0,np.inf,-np.inf,np.inf)[0]
    # F-side: integrate k^{-2/3} E[A^{-2} B0(b/A,k/A)]
    def BF(b,k):
        return integrate.quad(lambda s:pA(s)*s**(-2)*B0(b/s,k/s),0,a,limit=200)[0]
    IF=integrate.dblquad(lambda b,k:k**(-2/3)*BF(b,k),0,8,-8,8)[0]
    print(alpha,'E A^-2/3 num',EAm23,'closed',pred,' IF/IH',IF/IH)
# far: nu_far^H = 1 on (0,1], total mass 1; nu_far^F(l)=alpha a^-alpha l^(alpha-1) int_{l/a}^inf s^-alpha nu
for alpha in [0.5,2/3,0.8,1.0,1.5]:
    a=np.sqrt((alpha+2)/alpha)
    def nuF(l):
        lo=l/a
        if lo>=1: return 0.0
        return alpha*a**(-alpha)*l**(alpha-1)*integrate.quad(lambda s:s**(-alpha),lo,1)[0]
    print(alpha,[ (l, l**(1/3)*nuF(l)) for l in [1e-2,1e-4,1e-6,1e-8]])
