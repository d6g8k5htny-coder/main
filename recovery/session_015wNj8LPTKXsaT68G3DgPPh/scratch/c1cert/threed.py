# c1(d=3) via T3 = int dS int_0^{S^2/4} dP P^{1/4} e^{-S^2/5+P/2} int_0^inf w^{-1-p}[1 - Re Phi] dw,
# Phi = (1 - i w S - w^2 P)^{-1/2} exp(-i w P S/20 - 23 w^2 P^2/240).
import math
import numpy as np
p = 1.75
Kp = math.pi/(2*math.gamma(1+p)*math.sin(math.pi*p/2))
CA = (2*math.pi)**-4*12**-0.5*(2*math.pi)**-1.5*0.5*math.sqrt(4*math.pi/5)
pref = -(192/7)*2**0.25*4*math.pi*CA*math.pi/Kp
def T3(nS, nz, nu, Smax, Umax):
    xs, ws = np.polynomial.legendre.leggauss(nS)
    S = 0.5*Smax*(xs+1); wS = 0.5*Smax*ws
    xz, wz = np.polynomial.legendre.leggauss(nz)
    z = 0.5*(xz+1); wz = 0.5*wz
    xu, wu = np.polynomial.legendre.leggauss(nu)
    u = 0.5*Umax*(xu+1); wu = 0.5*Umax*wu
    w = u**4
    tot = 0.0
    for Si, wSi in zip(S, wS):
        y = z**4
        P = Si*Si/4*y
        jac = Si*Si/4*4*z**3          # dP = S^2/4 dy, dy = 4 z^3 dz
        base = P**0.25*np.exp(-Si*Si/5 + P/2)*jac*wz   # over z
        W, PP = np.meshgrid(w, P, indexing='ij')         # (nu, nz)
        Phi = (1 - 1j*W*Si - W*W*PP)**(-0.5)*np.exp(-1j*W*PP*Si/20 - 23*W*W*PP*PP/240)
        inner = 4*u[:, None]**(-8)*(1 - Phi.real)
        # small-u cancellation: for u < 0.05 use the analytic limit via a tiny-w series is unnecessary here (u nodes > 0)
        Iw = (wu[:, None]*inner).sum(axis=0) + (4/7)*Umax**-7   # tail of the '1' term beyond Umax (Phi part negligible)
        tot += wSi*np.sum(base*Iw)
    return tot
for args in ((40, 40, 200, 8.0, 30.0), (60, 60, 300, 9.0, 40.0)):
    t = T3(*args)
    print(args, 'T3 =', t, 'c1(d=3) =', pref*t, flush=True)
