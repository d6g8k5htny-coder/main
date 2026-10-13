import math, sys
import numpy as np
from proto2b import M
pref = (-(192/7)*2**0.25*4*math.pi*(2*math.pi)**-4*12**-0.5*6**(-7/8)*(2*math.pi)**-1.5*0.5
        *math.sqrt(4*math.pi/5)*(23/20)**(7/8)*math.pi*0.5)
c = math.sqrt(15/46)
def run(n, m, ng, smax):
    xg, wg = np.polynomial.hermite.hermgauss(ng)
    g2 = (2*xg)**2; wgn = wg/math.sqrt(math.pi)
    G1, G2 = np.meshgrid(g2, g2, indexing='ij'); W12 = np.outer(wgn, wgn)
    xs, ws = np.polynomial.legendre.leggauss(n)
    s = 0.5*smax*(xs+1); wsn = 0.5*smax*ws
    xw, ww = np.polynomial.legendre.leggauss(m)
    w = 0.5*(xw+1); wwn = 0.5*ww
    tot = 0.0
    for si, wi in zip(s, wsn):
        a1 = si**4
        for vi, vw in zip(w, wwn):
            a2 = a1*vi**4
            jac = 4*si**3*a1*4*vi**3
            t = c*(G1/a1 + G2/a2 - (a1 + a2)/5)
            E = float(np.sum(W12*M(t.ravel()).reshape(t.shape)))
            f = (a1-a2)*(a1*a2)**2*math.exp((a1+a2)**2/20 - (a1*a1+a2*a2)/4)*E
            tot += 2*wi*vw*jac*f
    return tot
for n, m, ng, smax in ((40, 30, 80, 2.4), (60, 40, 120, 2.6)):
    I = run(n, m, ng, smax)
    print(n, m, ng, smax, 'I =', I, 'c1(d=3) =', pref*I, flush=True)
