import sympy as sp
r,s,eta,q=sp.symbols('r s eta q',positive=True)
c,t,a,b,lam,rho=sp.symbols('c t a b lam rho',real=True)
X,Y=sp.symbols('X Y')
J={(i,j):sp.Symbol(f'J{i}{j}') for i in range(6) for j in range(6) if i+j<=5}
F=sum(J[k]*X**k[0]*Y**k[1]/(sp.factorial(k[0])*sp.factorial(k[1])) for k in J)
Fx,Fy=sp.diff(F,X),sp.diff(F,Y)
ev=lambda e,x,y: e.subs({X:x,Y:y},simultaneous=True)
def rows(aa,bb):
    g=lambda u: ev(F,u,0); gp=lambda u: ev(Fx,u,0)
    U1=(g(r/2)-g(-r/2))/r; U2=(gp(r/2)-gp(-r/2))/r
    U3=6*(gp(-r/2)+gp(r/2)-2*U1)/r**2; V0=(ev(Fy,-r/2,0)+ev(Fy,r/2,0))/2; V1=(ev(Fy,r/2,0)-ev(Fy,-r/2,0))/r
    x0,y0=s*c,s*t; x1,y1=s*(c+eta*aa),s*(t+eta*bb)
    Gu,Gv=ev(Fx,x0,y0),ev(Fy,x0,y0)
    Du=(ev(Fx,x1,y1)-Gu)/eta; Dv=(ev(Fy,x1,y1)-Gv)/eta
    Ru=Gu-U1-s*(c*U2+t*V1)-s**2*c**2/2*U3
    Rv=Gv-V0-s*c*V1
    Su=Du-s*(aa*U2+bb*V1)-s**2*aa*c*U3
    Sv=Dv-s*aa*V1
    W1=Ru/s**2; W2=Su/s**2; W3=(Sv-(bb/t)*Rv)/s**2
    return U3,W1,W2,W3
D4=c**3*J[(4,0)]+3*c**2*t*J[(3,1)]+3*c*t**2*J[(2,2)]+t**3*J[(1,3)]
def limit0(e):
    e=sp.expand(sp.expand(e).subs(r,q*s))
    return sp.simplify(e.subs({q:0,eta:0}).subs(s,0)), e
# pin-parallel b=0
for aa in (1,-1):
    U3,W1,W2,W3=rows(aa,0)
    W1h=W1+(q**2/24)*U3
    K=(2*aa*W1h-c*W2-t*W3+(eta*c/2)*U3+(eta/(2*aa))*W2)/s
    K=sp.expand(sp.expand(K).subs(r,q*s))
    neg=[m for m in sp.Poly(K,q,s,eta).monoms() if min(m)<0] if True else None
    lim=sp.simplify(K.subs({q:0,eta:0}).subs(s,0))
    print('pin a=%+d: K polynomial (no neg powers):'%aa, neg==[], '; limit == -(a/6) D4:', sp.simplify(lim+sp.Rational(aa,6)*D4)==0)
    # third-order null relation
    L=[sp.simplify(sp.expand(x).subs(r,q*s).subs({q:0,eta:0}).subs(s,0)) for x in (W1,W2,W3)]
    print('   null 2aL1-cL2-tL3 =',sp.simplify(2*aa*L[0]-c*L[1]-t*L[2]))
# radial: a=lam c/rho, b=lam t/rho with rho^2=c^2+t^2
for lv in (1,-1):
    rr=sp.sqrt(c**2+t**2)
    U3,W1,W2,W3=rows(lv*c/rr,lv*t/rr)
    W1h=W1+(q**2/24)*U3
    K=(W2-(2*lv/rr)*W1h-(eta*c**2/(2*rr**2))*U3-(eta/rr**2)*W1h)/s
    K=sp.expand(sp.expand(K).subs(r,q*s))
    lim=sp.simplify(K.subs({q:0,eta:0}).subs(s,0))
    # check no 1/s survives: series in s at q=eta=0 has no negative powers
    Ks=sp.simplify(K.subs({q:0,eta:0}))
    print('rad lam=%+d: limit == lam/(6 rho) D4:'%lv, sp.simplify(lim-sp.Rational(lv,6)/rr*D4)==0, '; finite at s->0:', sp.limit(Ks,s,0)!=sp.oo)
    L=[sp.simplify(sp.expand(x).subs(r,q*s).subs({q:0,eta:0}).subs(s,0)) for x in (W1,W2)]
    print('   L2-(2lam/rho)L1 =',sp.simplify(L[1]-(2*lv/rr)*L[0]))
# check eta/q dependence at s=0 vanishes (uniform limit): K at s=0 independent of q, eta
