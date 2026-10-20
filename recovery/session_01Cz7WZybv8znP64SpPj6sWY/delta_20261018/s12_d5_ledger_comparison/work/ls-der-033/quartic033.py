# Finite-t check of LS-DER-033 (8.1): add the pin-preserving quartic layer t*P4 to the leading cubic.
import sympy as sp
c,s,eta,kap,al,a,Q,w,z,X,Y,t = sp.symbols('c s eta kappa alpha a Q w z X Y t', real=True)
A,B,C,D,E = sp.symbols('A B C D E', real=True)
P3 = kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
P4 = (A/24)*(X**2-eta**2/4)**2+(B/6)*X*(X**2-eta**2/4)*Y+(C/4)*X**2*Y**2+(D/6)*X*Y**3+(E/24)*Y**4
P = P3 + t*P4
at = lambda e,x,y: e.subs({X:x,Y:y})
PX,PY = sp.diff(P,X),sp.diff(P,Y)
# pins still exact
assert all(sp.simplify(at(g,x0,0))==0 for g in (PX,PY) for x0 in (-eta/2,eta/2))
assert sp.simplify(at(P,-eta/2,0))==0 and sp.simplify(at(P,eta/2,0)+kap*eta**3/6)==0
sol = sp.solve([at(P,c,s)+al*kap*eta**3/6, at(PX,c,s), at(PY,c,s)],[a,Q,z],dict=True)[0]
Hs = sp.hessian(P,(X,Y))
dets = {nm: sp.together(Hs.subs({X:x0,Y:y0}).det().subs(sol)) for nm,(x0,y0) in
        {'M':(-eta/2,0),'S':(eta/2,0),'y':(c,s)}.items()}
# Expand each det as polynomial in t, then report lowest eta-order of each t-coefficient
for nm,d in dets.items():
    num,den = sp.fraction(sp.factor(d))
    pt = sp.Poly(sp.expand(num), t)
    print(f'det D2P({nm}): denominator = {sp.factor(den)}')
    for (k,),coef in sorted(pt.terms()):
        pe = sp.Poly(sp.expand(coef), eta)
        low = min(m[0] for m in pe.monoms())
        lowcoef = sp.factor(pe.coeff_monomial(eta**low))
        print(f'   t^{k}: lowest eta power {low}; coefficient {lowcoef}')
# Q (transverse quadratic at midpoint) order: Q = O(eta^2) + O(t)
Qn = sp.factor(sol[Q])
num,den = sp.fraction(Qn); pt = sp.Poly(sp.expand(num), t)
for (k,),coef in sorted(pt.terms()):
    pe = sp.Poly(sp.expand(coef), eta); low=min(m[0] for m in pe.monoms())
    print(f'Q: t^{k} lowest eta power {low}: {sp.factor(pe.coeff_monomial(eta**low))} / {sp.factor(den)}')
# Numerical scan: physical triple product t^6*dM*dS*dy divided by r^6=(t*eta)^6 at generic jets, t=r^beta
import random
random.seed(1)
vals = {kap:1, al:sp.Rational(1,2), w:sp.Rational(3,10), A:sp.Rational(7,10), B:sp.Rational(-11,10), C:sp.Rational(9,10), D:sp.Rational(1,3), E:sp.Rational(-1,2),
        c:sp.Rational(3,5), s:sp.Rational(4,5)}
prod = sp.lambdify((t,eta), (dets['M']*dets['S']*dets['y']).subs(vals), 'mpmath')
import mpmath as mp
mp.mp.dps = 50
print('\nratio |t^6 dM dS dy| / r^6 at fixed generic jets (c,s)=(3/5,4/5):')
for beta in (mp.mpf(1)/2, mp.mpf(3)/5, mp.mpf(2)/3, mp.mpf(3)/4, mp.mpf(9)/10):
    row=[]
    for r in (mp.mpf('1e-2'), mp.mpf('1e-4'), mp.mpf('1e-6'), mp.mpf('1e-8')):
        tt = r**beta; ee = r/tt
        row.append(mp.nstr(abs(prod(tt,ee))*tt**6/r**6, 4))
    print(f'  t=r^{mp.nstr(beta,3)}:', row)
# leading-order (t=0) check: ratio must be bounded
prod0 = sp.lambdify((eta,), (dets['M']*dets['S']*dets['y']).subs(vals).subs(t,0)/eta**6, 'mpmath')
print('t=0 (pure cubic) det-product/eta^6 at eta=1e-3,1e-6:', mp.nstr(prod0(mp.mpf('1e-3')),6), mp.nstr(prod0(mp.mpf('1e-6')),6))
