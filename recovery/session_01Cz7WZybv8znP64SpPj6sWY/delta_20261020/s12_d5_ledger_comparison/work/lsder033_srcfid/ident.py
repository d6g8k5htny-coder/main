import sympy as sp
c,s,eta,kap,al,a,Q,w,z,X,Y,t,r,dl = sp.symbols('c s eta kappa alpha a Q w z X Y t r delta', real=True)
P = kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
at=lambda e,x,y: e.subs({X:x,Y:y},simultaneous=True)
PX,PY=P.diff(X),P.diff(Y)
sol=sp.solve([at(P,c,s)+al*kap*eta**3/6,at(PX,c,s),at(PY,c,s)],[a,Q,z],dict=True)
print('nsol',len(sol)); sol=sol[0]
ag=(-c**2*kap+eta**2*kap/4-s**2*w/2)/(c*s)
Qg=eta**2*(-al*c*eta*kap+c**2*kap/2+c*eta*kap/2+eta**2*kap/8-s**2*w/4)/(c*s**2)
zg=(32*al*c*eta**3*kap+16*c**4*kap-24*c**2*eta**2*kap-24*c**2*s**2*w-16*c*eta**3*kap-3*eta**4*kap+6*eta**2*s**2*w)/(16*c*s**3)
print('a,Q,z free-cs:',[sp.simplify(sol[v]-g) for v,g in ((a,ag),(Q,Qg),(z,zg))])
H=sp.hessian(P,(X,Y))
W=w/kap;K=2*W*s**2-4*c**2-4*c*eta-eta**2
pref=kap**2*eta**2/(64*c**2*s**2)
BM=64*al*c**2*eta**2-K**2;BS=64*(1-al)*c**2*eta**2-(K+8*c*eta)**2;BX=(K+8*al*c*eta)**2+64*al*(1-al)*c**2*eta**2
dM=at(H.det(),-eta/2,0).subs(sol);dS=at(H.det(),eta/2,0).subs(sol);dY=at(H.det(),c,s).subs(sol)
print('3.1',sp.simplify(dM-pref*BM),'3.2',sp.simplify(dS-pref*BS),'3.3',sp.simplify(dY+pref*BX))
print('3.4',sp.simplify(dM*dS*dY+kap**6*eta**6/(64**3*c**6*s**6)*BM*BS*BX))
# transverse
Ab=sp.Symbol('Abar')
for sg in (1,-1):
    st=sp.solve([at(P,0,sg)+al*kap*eta**3/6,at(PX,0,sg),at(PY,0,sg)],[w,Q,z],dict=True)[0]
    rep=lambda e: sp.simplify(e.subs(st).subs(a,sg*Ab*kap))
    tM=rep(at(H.det(),-eta/2,0));tS=rep(at(H.det(),eta/2,0));tY=rep(at(H.det(),0,sg))
    print('s=',sg,'w=',st[w],sp.simplify(tM-kap**2*eta**2/4*(4*al*eta**2-(Ab+eta)**2)),sp.simplify(tS-kap**2*eta**2/4*(4*(1-al)*eta**2-(Ab-eta)**2)),sp.simplify(tY+kap**2*eta**2/4*((Ab+(1-2*al)*eta)**2+4*al*(1-al)*eta**2)))
S0=sp.Matrix([[s**2*(3*c**4+3*c**2*s**2+s**4)/24,-c*s**2*(2*c**2+s**2)/4,0],[-c*s**2*(2*c**2+s**2)/4,s**2*(4*c**2+s**2)/2,0],[0,0,2*s**2]])
print('6.2',sp.factor(S0.det()-s**8*(c**2+s**2)*(3*c**2+s**2)/24))
print('6.3',sp.simplify(S0[0,0]-S0[0,1]**2/S0[1,1]-s**4*(c**2+s**2)*(3*c**2+s**2)/(24*(4*c**2+s**2))))
print('2s^2-VarY2 on circle',sp.factor(sp.expand((2*s**2-S0[1,1]).subs(c**2,1-s**2))))
# (8.3)
rp,tp,dp=sp.symbols('r t delta',positive=True)
I=sp.integrate(sp.Rational(5,24)*rp**3*rp**4*tp**-6*tp,(tp,sp.sqrt(15)*rp/2,dp))
print('8.3 with 5/24 window, C=1:',sp.expand(sp.simplify(I)))
I1=sp.integrate(rp**3*rp**4*tp**-6*tp,(tp,sp.sqrt(15)*rp/2,dp))
print('8.3 with r^3 window, C=1:',sp.expand(sp.simplify(I1)))
# cubic bridge to P2 H
k=sp.Symbol('k');x=sp.Symbol('x')
Hx=sp.Symbol('b')+k*(2*x**3-3*r*x**2)
print('bridge',sp.simplify(Hx.subs(x,r/2+t*X).subs(r,t*eta).subs(k,kap/6)-sp.Symbol('b')-t**3*kap*(X**3/3-eta**2*X/4-eta**3/12)))
