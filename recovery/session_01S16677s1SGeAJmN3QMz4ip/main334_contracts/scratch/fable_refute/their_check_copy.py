# Sanity check of the scaling identity and the Mellin exponents (numeric, toy B^H_0).
import math
from scipy import integrate
import numpy as np
def run(alpha):
    a=math.sqrt((alpha+2)/alpha)
    pA=lambda s: alpha*a**(-alpha)*s**(alpha-1)
    # toy B^H_0(b,k) = k^2 exp(-(b^2+k^2)) (shape of 12 pi_0 z_0 under parity: 36k^2 phi(12k)*...)
    B=lambda b,k: k**2*math.exp(-(b*b+k*k))
    IH=integrate.dblquad(lambda k,b: k**(-2/3)*B(b,k), -8,8, 0,8)[0]
    # B^F_0(b,k)=E[A^-2 B(b/A,k/A)]
    def BF(b,k):
        return integrate.quad(lambda s: pA(s)*s**(-2)*B(b/s,k/s),0,a,limit=200)[0]
    IF=integrate.dblquad(lambda k,b: k**(-2/3)*BF(b,k), -4,4, 0,4, epsabs=1e-6)[0]
    EAm=integrate.quad(lambda s: pA(s)*s**(-2/3),0,a)[0] if alpha>2/3 else float('inf')
    closed=a**(-2/3)*alpha/(alpha-2/3) if alpha>2/3 else float('inf')
    print(f"alpha={alpha}: int k^-2/3 B^F = {IF:.5f}, E[A^-2/3]*int B^H = {EAm*IH:.5f}, closed form E A^-beta={closed:.5f} vs quad {EAm:.5f}")
for al in (1.0,1.5,3.0):
    run(al)
# divergence check at alpha<=2/3
for al in (0.5,2/3):
    a=math.sqrt((al+2)/al)
    print(al, integrate.quad(lambda s: al*a**(-al)*s**(al-1-2/3),1e-12,a)[0], "(grows without bound as lower limit -> 0)")
