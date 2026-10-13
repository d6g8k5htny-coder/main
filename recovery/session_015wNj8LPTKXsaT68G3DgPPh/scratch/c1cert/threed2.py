import math
import numpy as np
p = 1.75
Kp = math.pi/(2*math.gamma(1+p)*math.sin(math.pi*p/2))
CA = (2*math.pi)**-4*12**-0.5*(2*math.pi)**-1.5*0.5*math.sqrt(4*math.pi/5)
pref = -(192/7)*2**0.25*4*math.pi*CA*math.pi/Kp
def T3(nS, nz, nu, Smax, Umax, u0=0.1):
    xs, ws = np.polynomial.legendre.leggauss(nS)
    S = 0.5*Smax*(xs+1); wS = 0.5*Smax*ws
    xz, wz = np.polynomial.legendre.leggauss(nz)
    z = 0.5*(xz+1); wz = 0.5*wz
    xu, wu = np.polynomial.legendre.leggauss(nu)
    u = u0 + 0.5*(Umax-u0)*(xu+1); wu = 0.5*(Umax-u0)*wu
    w = u**4
    tot = 0.0
    for Si, wSi in zip(S, wS):
        y = z**4
        P = Si*Si/4*y
        jac = Si*Si/4*4*z**3
        base = P**0.25*np.exp(-Si*Si/5 + P/2)*jac*wz
        W, PP = np.meshgrid(w, P, indexing='ij')
        Phi = (1 - 1j*W*Si - W*W*PP)**(-0.5)*np.exp(-1j*W*PP*Si/20 - 23*W*W*PP*PP/240)
        inner = 4*u[:, None]**(-8)*(1 - Phi.real)
        m1 = Si/2 - P*Si/20
        c2 = P/2 - Si*Si/4 - 23*P*P/240
        lead = 4*(m1*m1/2 - c2)                       # limit of 4u^-8 (1 - Re Phi(u^4)) at u -> 0
        Iw = (wu[:, None]*inner).sum(axis=0) + u0*lead + (4/7)*Umax**-7
        tot += wSi*np.sum(base*Iw)
    return tot
for args in ((60, 40, 200, 20.0, 30.0), (90, 60, 300, 24.0, 40.0)):
    t = T3(*args)
    print(args, 'T3 =', t, 'c1(d=3) =', pref*t, flush=True)
