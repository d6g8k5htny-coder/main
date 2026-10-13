import sympy as sp, mpmath as mp
from fractions import Fraction as Q
th = sp.symbols('theta', positive=True)
g_core = sp.Rational(1,15)*th*(9-th**2)*(3/(3+th**2))**sp.Rational(5,2)
total = sp.integrate(g_core, (th, 0, 1))
print('(i) sympy exact  int_0^1 g_core dtheta =', sp.nsimplify(sp.simplify(total)), '  float', float(total))
mp.mp.dps = 60
gc = lambda t: t*(9-t**2)*(3/(3+t**2))**mp.mpf('2.5')/15
print('    mpmath 60dps  int_0^1 g_core        =', mp.quad(gc, [0, 1]))
def closed(tau):
    tau = mp.mpf(tau)
    s = mp.findroot(lambda s: 36*s**3 - tau*(3+s**2)**2, mp.mpf('0.5'), tol=mp.mpf(10)**-58)
    return (3 - 3*(1-s**2)/(1+s**2/3)**mp.mpf('1.5') + (tau/4)**mp.mpf('1.25')*(mp.mpf(36)/7*(s**mp.mpf('-1.75')-1) - 4*(1-s**mp.mpf('0.25'))))/15, s
tau40 = mp.mpf(9)/4 - mp.mpf(2)**-40
m40, s40 = closed(tau40)
print('(ii) closed form at tau=9/4-2^-40: s=%s  m=%s  1/5-m=%s' % (mp.nstr(s40,25), mp.nstr(m40,40), mp.nstr(mp.mpf(1)/5-m40,8)))
# (iii) at tau=9/4, s=1 exactly: 36*1 = (9/4)*16 = 36; closed form = (3 - 0 + c*(0-0))/15 = 1/5
print('(iii) at tau=9/4: 36*1^3 == (9/4)*(3+1)^2 ->', Q(36) == Q(9,4)*16, '; closed form -> (3-0+anything*0)/15 = 1/5 exactly')
# (iv) route cross-check at tau=1/8 vs PR value 0.05700365661620024
tau = mp.mpf(1)/8
m, s = closed(tau)
gt = lambda t: (tau/4)**mp.mpf('1.25')*t**mp.mpf('-2.75')*(9-t**2)/15
direct = mp.quad(gc, [0, s]) + mp.quad(gt, [s, 1])
print('(iv) tau=1/8: closed=%s direct=%s PR cdf_mass(1/8)=0.05700365661620024' % (mp.nstr(m,20), mp.nstr(direct,20)))
