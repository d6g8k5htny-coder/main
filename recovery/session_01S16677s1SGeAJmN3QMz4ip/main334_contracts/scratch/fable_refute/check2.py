# Independent check of C_F = E[A^-2/3] C_H with a NON-separable toy B^H_0, and of the
# small-k law of the b-integrated F amplitude (int B^F_0 db ~ c k^(alpha-1)).
import math
import numpy as np
from scipy import integrate

def run(alpha, B, bb=10.0, kk=10.0):
    a = math.sqrt((alpha+2)/alpha)
    pA = lambda s: alpha*a**(-alpha)*s**(alpha-1)
    IH = integrate.dblquad(lambda k,b: k**(-2/3)*B(b,k), -bb,bb, 0,kk, epsabs=1e-9, epsrel=1e-9)[0]
    def BF(b,k):
        # E[A^-2 B(b/A,k/A)], substitute s=k/kp to avoid the s->0 endpoint
        f = lambda s: pA(s)*s**(-2)*B(b/s,k/s)
        return integrate.quad(f, 0, a, limit=400, points=[min(a,k), min(a,2*k)])[0]
    IF = integrate.dblquad(lambda k,b: k**(-2/3)*BF(b,k), -bb,bb, 0,kk, epsabs=1e-7, epsrel=1e-7)[0]
    EAm = a**(-2/3)*alpha/(alpha-2/3)
    EAm_q = integrate.quad(lambda s: pA(s)*s**(-2/3), 0, a)[0]
    print(f"alpha={alpha:5.3f}  int k^-2/3 B^F_0 = {IF:.6f}   E[A^-2/3]*int k^-2/3 B^H_0 = {EAm*IH:.6f}   rel.diff={abs(IF-EAm*IH)/(EAm*IH):.2e}   (E A^-2/3 closed {EAm:.6f}, quad {EAm_q:.6f})")
    # small-k law of the b-integrated F amplitude
    def rowF(k):
        return integrate.quad(lambda b: BF(b,k), -bb, bb, limit=200)[0]
    ks = [0.02, 0.01, 0.005]
    vals = [rowF(k) for k in ks]
    slopes = [math.log(vals[i+1]/vals[i])/math.log(ks[i+1]/ks[i]) for i in range(len(ks)-1)]
    print(f"           small-k exponent of int B^F_0 db: {slopes}  (predicted alpha-1 = {alpha-1:.3f})")

B1 = lambda b,k: k**2*math.exp(-(b*b+0.7*b*k+k*k))          # non-separable
B2 = lambda b,k: k**2*math.exp(-(b*b+k*k))/(1+0.3*b*b*k)      # another non-separable, non-Gaussian in b
for al in (0.8, 1.0, 1.5, 3.0):
    run(al, B1)
run(1.0, B2); run(2.5, B2)
# divergence at alpha<=2/3: E[A^-2/3] = alpha a^-alpha int_0^a s^(alpha-5/3) ds
for al in (0.5, 2/3):
    a = math.sqrt((al+2)/al)
    for eps in (1e-3,1e-6,1e-9):
        v = al*a**(-al)*integrate.quad(lambda s: s**(al-5/3), eps, a)[0]
        print(f"alpha={al:.4f} eps={eps:.0e}  truncated E A^-2/3 = {v:.4f}")
