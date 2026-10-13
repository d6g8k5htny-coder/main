# c1 (d=2) via the 2D reduction:
# c1 = -(192/7) 2^(1/4) 2pi (2pi)^-3 12^-1/2 (1/2) (19/16)^(7/8) 6^(-7/8) * int_0^inf a^2 e^(-3a^2/16) E_g[ M( sqrt(6/19)(a/4 - g^2/a) ) ] da,
# g ~ N(0,2), M(t) = E|Z+t|^(7/4).
import math
import numpy as np
p = 1.75
Cp = 2**(p/2) * math.gamma((p+1)/2) / math.sqrt(math.pi)
def M(t):
    # Kummer: 1F1(-p/2;1/2;-z) = e^-z 1F1((1+p)/2; 1/2; z), z = t^2/2
    t = np.asarray(t, dtype=float)
    z = 0.5*t*t
    out = np.empty_like(t)
    big = np.abs(t) > 12
    # series for |t| <= 12
    zs = z[~big]
    term = np.ones_like(zs); s = np.ones_like(zs)
    a_, b_ = (1+p)/2, 0.5
    for n in range(400):
        term = term * (a_ + n)/(b_ + n) * zs/(n+1)
        s = s + term
        if np.all(term < 1e-17*s): break
    out[~big] = Cp*np.exp(-zs)*s
    # asymptotic for |t| > 12: |t|^p (1 + p(p-1)/(2t^2) + p(p-1)(p-2)(p-3)/(8 t^4) + ...)
    tb = np.abs(t[big])
    c2 = p*(p-1)/2; c4 = p*(p-1)*(p-2)*(p-3)/8; c6 = p*(p-1)*(p-2)*(p-3)*(p-4)*(p-5)/48
    out[big] = tb**p*(1 + c2/tb**2 + c4/tb**4 + c6/tb**6)
    return out
# checks: M(0) = E|Z|^p, M large
print('M(0)', M(np.array([0.0]))[0], 'vs', 2**(p/2)*math.gamma((p+1)/2)/math.sqrt(math.pi))
print('M(11.9), M(12.1)', M(np.array([11.9, 12.1])), (11.9**p*(1+p*(p-1)/2/11.9**2)), 12.1**p*(1+p*(p-1)/2/12.1**2))
# Gauss-Hermite for g ~ N(0,2): g = 2 x (weight e^-x^2) ... use x nodes: g = sqrt(2)*sqrt2*x? N(0,2): g = 2 * x with weight e^{-x^2}/sqrt(pi)
xg, wg = np.polynomial.hermite.hermgauss(200)
g2 = (2*xg)**2; wgn = wg/math.sqrt(math.pi)
# a-integral: substitute a = s^4 to tame a -> 0 (integrand ~ a^(1/4) there): da = 4 s^3 ds
xs, ws = np.polynomial.legendre.leggauss(400)
def Ia(smax):
    s = 0.5*smax*(xs+1); w = 0.5*smax*ws
    a = s**4
    tot = 0.0
    for si, ai, wi in zip(s, a, w):
        t = math.sqrt(6/19)*(ai/4 - g2/ai)
        Eg = np.dot(wgn, M(t))
        tot += wi * 4*si**3 * ai**2*math.exp(-3*ai*ai/16) * Eg
    return tot
for smax in (2.6, 2.8, 3.0):
    I = Ia(smax)
    pref = -(192/7)*2**0.25*2*math.pi*(2*math.pi)**-3*12**-0.5*0.5*(19/16)**(7/8)*6**(-7/8)
    print('smax', smax, 'I', I, 'c1(d=2) =', pref*I)
