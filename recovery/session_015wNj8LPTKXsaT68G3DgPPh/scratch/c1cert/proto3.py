# c1 (d=3) via the 4D reduction (a1, a2 > 0 = -eigenvalues of A; gamma' ~ N(0, 2 I)).
import math
import numpy as np
from proto2b import M
pref = (-(192/7)*2**0.25*4*math.pi*(2*math.pi)**-4*12**-0.5*6**(-7/8)*(2*math.pi)**-1.5*0.5
        *math.sqrt(4*math.pi/5)*(23/20)**(7/8)*math.pi*0.5)
c = math.sqrt(15/46)
xg, wg = np.polynomial.hermite.hermgauss(60)
g2 = (2*xg)**2; wgn = wg/math.sqrt(math.pi)          # gamma_i ~ N(0,2): gamma = 2x, weight e^-x^2/sqrt(pi)
G1, G2 = np.meshgrid(g2, g2, indexing='ij'); W12 = np.outer(wgn, wgn)
def Eg(a1, a2):
    t = c*(G1/a1 + G2/a2 - (a1 + a2)/5)
    return float(np.sum(W12*M(t.ravel()).reshape(t.shape)))
def run(n, smax):
    xs, ws = np.polynomial.legendre.leggauss(n)
    s = 0.5*smax*(xs+1); w = 0.5*smax*ws
    a = s**4; ja = 4*s**3*w
    tot = 0.0
    for i in range(n):
        for j in range(i+1, n):     # symmetric: 2 * sum over i<j; diagonal has |a1-a2| = 0
            a1, a2 = a[i], a[j]
            f = abs(a1-a2)*(a1*a2)**2*math.exp((a1+a2)**2/20 - (a1*a1+a2*a2)/4)*Eg(a1, a2)
            tot += 2*ja[i]*ja[j]*f
    return tot
for n, smax in ((60, 2.4), (80, 2.6)):
    I = run(n, smax)
    print(n, smax, 'I =', I, 'c1(d=3) =', pref*I)
