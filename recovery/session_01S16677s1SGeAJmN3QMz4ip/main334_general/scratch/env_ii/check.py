# Numeric check of the k^(-2/3)-mass identity  int k^-2/3 E[A^-2 H_SG(b/A,k/A)] db dk = E[A^-2/3] * I_SG
import numpy as np
from scipy import integrate
d, C, c = 2, 1.0, 0.7
H = lambda b,k: C*(1+abs(b)+k)**(2*d)*np.exp(-c*(b*b+k*k))
I_SG = integrate.dblquad(lambda k,b: k**(-2/3)*H(b,k), -np.inf, np.inf, 0, np.inf)[0]
# A = power law density alpha a^-alpha-1? use AMP's law: density alpha a^-alpha x^(alpha-1) on (0,a)
for alpha,a in [(1.0,1.0),(2.0,0.5),(0.8,1.3)]:
    m = a**(-2/3)*alpha/(alpha-2/3)
    def HF(b,k):
        # E[A^-2 H(b/A,k/A)] with density alpha a^-alpha x^(alpha-1) on (0,a)
        f = lambda x: alpha*a**(-alpha)*x**(alpha-1)*x**(-2)*H(b/x,k/x)
        return integrate.quad(f,0,a,limit=200)[0]
    lhs = integrate.dblquad(lambda k,b: k**(-2/3)*HF(b,k), -8, 8, 1e-9, 8, epsabs=1e-6)[0]
    print(alpha,a,"lhs",lhs,"m*I_SG",m*I_SG)
# pointwise finiteness: sup_a a^-2 H(b/a,k/a) for a small (b,k)
for (b,k) in [(0.0,1e-3),(1e-4,1e-4),(3.0,5.0)]:
    aa = np.logspace(-6,6,2000)
    print((b,k), (aa**-2*H(b/aa,k/aa)).max())
