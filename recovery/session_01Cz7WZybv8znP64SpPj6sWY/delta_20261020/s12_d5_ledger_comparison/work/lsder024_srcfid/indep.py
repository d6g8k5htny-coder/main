import sympy as sp
# (a) exact collinear axis model: g' = c x(x-d)(x-r), g(r)-g(0) = -k r^3 ; sign of g''(0) for 0<d<r/2 vs d<0
x,d,r,k,c,S0,p=sp.symbols('x d r k c S0 p',real=True)
gp=c*x*(x-d)*(x-r)
cs=sp.solve(sp.integrate(gp,(x,0,r))+k*r**3,c)[0]
g2=lambda v: sp.factor(sp.diff(gp,x).subs(x,v).subs(c,cs))
gM,gX,gS=[g2(v).subs(d,r*p) for v in (0,d,r)]
print('g\'\'(M)=',sp.factor(gM),' g\'\'(X)=',sp.factor(gX),' g\'\'(S)=',sp.factor(gS))
for pv in (sp.Rational(3,20),sp.Rational(-3,20)):
    print(' p=',pv,' sign g\'\'(M) (r,k>0):',sp.sign(gM.subs({p:pv,r:1,k:1})),' -> M is', 'axis-min (W=0)' if gM.subs({p:pv,r:1,k:1})>0 else 'axis-max (W possible)')
trip=sp.factor(gM*gX*gS*S0**3); print('triple (f_xz->0, f_zz=S0):',trip)
# (b) exact LS-DER-033 (3.1)-(3.3) via rational unit-circle parametrization
X,Y,eta,kap,a,Q,w,z,al,u=sp.symbols('X Y eta kappa a Q w z alpha u',real=True)
cc=(1-u**2)/(1+u**2); ss=2*u/(1+u**2)
P=kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
sol=sp.solve([P.subs({X:cc,Y:ss})+al*kap*eta**3/6, sp.diff(P,X).subs({X:cc,Y:ss}), sp.diff(P,Y).subs({X:cc,Y:ss})],[a,Q,z],dict=True)[0]
H=sp.hessian(P,(X,Y))
K=2*(w/kap)*ss**2-4*cc**2-4*cc*eta-eta**2
pref=kap**2*eta**2/(64*cc**2*ss**2)
ref={'M':pref*(64*al*cc**2*eta**2-K**2),'S':pref*(64*(1-al)*cc**2*eta**2-(K+8*cc*eta)**2),'X':-pref*((K+8*al*cc*eta)**2+64*al*(1-al)*cc**2*eta**2)}
for nm,pt in (('M',(-eta/2,0)),('S',(eta/2,0)),('X',(cc,ss))):
    dd=H.subs({X:pt[0],Y:pt[1]}).det().subs(sol)
    print(nm,'exact match to 033 (3.%d):'%{'M':1,'S':2,'X':3}[nm], sp.simplify(sp.together(dd-ref[nm]))==0)
# leading sign: det/eta^2 at eta->0 is -(...)^2/(...)<=0 for all three  => at leading order M is not a maximum unless K=O(eta)
print('B_M at eta=0:',sp.factor(sp.expand(-K**2).subs(eta,0)))
