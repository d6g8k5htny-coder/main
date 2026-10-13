# T3 = 64 int_0^inf ds int_0^1 dv int_0^inf dt s^20 v^4 (1-v^4) e^{-s^8 E(v)} Gt(t; a1 = s^4, v)
# Gt = -E1(R) Rt + e^R Sinc(I/2)^2 It^2/2, R = t^8 Rt, I = t^4 It,
# Rt = -(1/4)[L(t^8) + v^8 L(t^8 v^8)] - 23 a1^2 v^8/240, It = (1/2)[A(t^4) + v^4 A(t^4 v^4)] - a1^2 v^4 (1+v^4)/20
import math
import numpy as np
p = 1.75
Kp = math.pi/(2*math.gamma(1+p)*math.sin(math.pi*p/2))
CA = (2*math.pi)**-4*12**-0.5*(2*math.pi)**-1.5*0.5*math.sqrt(4*math.pi/5)
pref = -(192/7)*2**0.25*4*math.pi*CA*math.pi/Kp
def E1(x):
    x = np.asarray(x, float); out = np.empty_like(x); sm = np.abs(x) < 1e-5
    out[sm] = 1 + x[sm]/2 + x[sm]**2/6; out[~sm] = np.expm1(x[~sm])/x[~sm]; return out
def L(x):
    x = np.asarray(x, float); out = np.empty_like(x); sm = np.abs(x) < 1e-5
    out[sm] = 1 - x[sm]/2 + x[sm]**2/3; out[~sm] = np.log1p(x[~sm])/x[~sm]; return out
def A(x):
    x = np.asarray(x, float); out = np.empty_like(x); sm = np.abs(x) < 1e-5
    out[sm] = 1 - x[sm]**2/3; out[~sm] = np.arctan(x[~sm])/x[~sm]; return out
def Sinc(x):
    x = np.asarray(x, float); out = np.empty_like(x); sm = np.abs(x) < 1e-5
    out[sm] = 1 - x[sm]**2/6; out[~sm] = np.sin(x[~sm])/x[~sm]; return out
def Gt(t, a1, v):
    t4 = t**4; t8 = t4*t4; v4 = v**4; v8 = v4*v4
    Rt = -0.25*(L(t8) + v8*L(t8*v8)) - 23*a1*a1*v8/240
    It = 0.5*(A(t4) + v4*A(t4*v4)) - a1*a1*v4*(1+v4)/20
    R = t8*Rt; I = t4*It
    return -E1(R)*Rt + np.exp(R)*Sinc(I/2)**2*It*It/2
def gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n); return a + (b-a)*(x+1)/2, w*(b-a)/2
def T3(ns, nv, nt, smax, tmax):
    s, ws = gl(ns, 0, smax); v, wv = gl(nv, 0, 1)
    tp = [(0, 0.9), (0.9, 1.2), (1.2, 2), (2, 4), (4, tmax)]
    tot = 0.0
    for si, wsi in zip(s, ws):
        a1 = si**4
        for vi, wvi in zip(v, wv):
            inner = 0.0
            for (ta, tb) in tp:
                t, wt = gl(nt, ta, tb)
                inner += np.dot(wt, Gt(t, a1, vi))
            inner += tmax**-7/7
            tot += wsi*wvi*si**20*vi**4*(1-vi**4)*math.exp(-si**8*((1+vi**8)/4 - (1+vi**4)**2/20))*inner
    return 64*tot
for args in ((40, 30, 30, 2.6, 12.0), (60, 40, 40, 2.8, 16.0)):
    T = T3(*args)
    print(args, 'T3 =', T, 'c1(d=3) =', pref*T, flush=True)
