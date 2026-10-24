import sympy as sp
r,t,d = sp.symbols('r t delta', positive=True)
eta = r/t
a = sp.sqrt(15)/2
# Claimed LS-DER-033 envelope: per unit value per unit angle  Z^-1 * t^-6 * t^6 eta^6 = r^4 t^-6
claimed = r**-2 * eta**6
# Corrected envelope from the quartic layer: dets ~ t^2*eta*(eta+t) (endpoints), t^2*(eta^2+t) (third point)
corr = r**-2 * eta**2*(eta+t)**2*(eta**2+t)
win = r**3   # window width ~ kappa r^3/6
for nm,lam in (('claimed',claimed),('corrected',corr)):
    I = sp.integrate(sp.expand(win*lam*t), (t, a*r, d))
    I = sp.expand(I)
    print(nm, 'count =', I)
    print('   leading as r->0 :', sp.limit(I/r**3, r, 0), '* r^3')
# regime split for corrected envelope
for lo,hi,lab in ((a*r, r**sp.Rational(2,3),'I: t<=r^(2/3)'),(r**sp.Rational(2,3), r**sp.Rational(1,2),'II'),(r**sp.Rational(1,2), d,'III: t>=r^(1/2)')):
    I = sp.expand(sp.integrate(sp.expand(win*corr*t),(t,lo,hi)))
    print(lab, ' -> leading r^3-normalized limit:', sp.limit(I/r**3, r, 0), '; r^(13/3)-normalized:', sp.limit(I/r**sp.Rational(13,3), r, 0))
# Exact r->0 behaviour of the claimed per-angle intensity at fixed t vs the corrected one
print('claimed at fixed t, r->0 :', sp.limit(claimed, r, 0), '; corrected at fixed t:', sp.limit(corr, r, 0))
# P2 ledgers
print('P2 Region II exponent:', -2-3+3-2-6)
c = sp.symbols('c', positive=True); x = c/(2*r**2)
print('6!(2/c)^6 r^12 == 6!/x^6 :', sp.simplify(sp.factorial(6)*(2/c)**6*r**12 - sp.factorial(6)/x**6)==0)
p,q,Delta = sp.symbols('p q Delta', positive=True)
print('P2 Region I power: Z^-1 * r^-3 Delta^-2 * r^6 q^2 ->', sp.simplify(r**-2*r**-3*Delta**-2*r**6*q**2))
rho = sp.symbols('rho', positive=True)
print('LS-024/C100 collar: Z^-1 * r^-3 rho^-2 * r^6 rho^2 ->', sp.simplify(r**-2*r**-3*rho**-2*r**6*rho**2))
# P16 at d = r*rho: L^6 r^4 d^2 / sigma^4
sig = sp.symbols('sigma', positive=True)
print('P16 with d=r*rho:', sp.simplify(r**4*(r*rho)**2/sig**4))
print('P17 with d=r*rho:', sp.simplify(r*(r*rho)**2))
# P2-style deterministic bounds applied in LS geometry (|S-M|=r, |y-M|~|y-S|~t): det_M,det_S <~ r t, det_y <~ t (column bound)
det_P2style = (r*t)*(r*t)*t
I = sp.integrate(win*r**-2*t**-6*det_P2style*t,(t,a*r,d))
print('P2-style deterministic product in LS region gives count ~', sp.simplify(I), ' leading power:', sp.limit(I/r**2,r,0),'* r^2')
