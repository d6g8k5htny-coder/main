import sympy as sp
X,Y,eta,kap,a,Q,w,z,al,c,s=sp.symbols('X Y eta kappa a Q w z alpha c s',real=True)
P=kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
sol=sp.solve([P.subs({X:c,Y:s})+al*kap*eta**3/6, sp.diff(P,X).subs({X:c,Y:s}), sp.diff(P,Y).subs({X:c,Y:s})],[a,Q,z],dict=True)[0]
Hs=sp.hessian(P,(X,Y))
K=2*(w/kap)*s**2-4*c**2-4*c*eta-eta**2
pref=kap**2*eta**2/(64*c**2*s**2)
ref={'M':pref*(64*al*c**2*eta**2-K**2),'S':pref*(64*(1-al)*c**2*eta**2-(K+8*c*eta)**2),'X':-pref*((K+8*al*c*eta)**2+64*al*(1-al)*c**2*eta**2)}
for nm,pt in (('M',(-eta/2,0)),('S',(eta/2,0)),('X',(c,s))):
    d=Hs.subs({X:pt[0],Y:pt[1]}).det().subs(sol)
    diff=sp.simplify((d-ref[nm]).subs(s**2,1-c**2).subs(s**3,s*(1-c**2)).subs(s**4,(1-c**2)**2))
    # numeric spot check on the unit circle as well
    import random
    ok=all(abs(sp.N((d-ref[nm]).subs({c:sp.Rational(3,5),s:sp.Rational(4,5),eta:sp.Rational(1,3),al:sp.Rational(2,7),w:sp.Rational(5,11),kap:sp.Rational(9,10)})))<1e-30 for _ in [0])
    print(nm,'matches LS-DER-033 (3.%s) on c^2+s^2=1:'%{'M':1,'S':2,'X':3}[nm], diff==0 or ok)
