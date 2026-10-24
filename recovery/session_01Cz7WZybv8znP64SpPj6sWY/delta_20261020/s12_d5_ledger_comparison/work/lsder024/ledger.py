import sympy as sp
X,Y,eta,kap,a,Q,w,z,al,c,s,t,r,b,k,p,q,d,S0=sp.symbols('X Y eta kappa a Q w z alpha c s t r b k p q d S0',real=True)
out=[]
def chk(name,cond):
    out.append((name,bool(cond))); print(('PASS ' if cond else 'FAIL ')+name)

# 1. mark conversion: 024 f(S)=b-kappa r^3/6  vs P2 f(S)=b-k r^3 ; Hermite cubic H(x)=b+k(2x^3-3r x^2)
x=sp.symbols('x',real=True)
H=b+k*(2*x**3-3*r*x**2)
chk('P2 Hermite: H(0)=b,H(r)=b-kr^3,H\'=0 at 0,r', sp.simplify(H.subs(x,0)-b)==0 and sp.simplify(H.subs(x,r)-(b-k*r**3))==0 and sp.diff(H,x).subs(x,0)==0 and sp.simplify(sp.diff(H,x).subs(x,r))==0)
chk('f_xxx=12k = 2kappa when k=kappa/6 (024 eq.8)', sp.simplify(sp.diff(H,x,3).subs(k,kap/6)-2*kap)==0)
print('k-range from K_*=[3/4,5/4]:', sp.Rational(3,4)/6, sp.Rational(5,4)/6)
# H''(0)=-6kr=-kappa r ; P13 det H_M/r=-6k S0 = -kappa S0
chk('f_xx(M)=-6kr=-kappa r', sp.simplify(sp.diff(H,x,2).subs(x,0)+kap*r).subs(k,kap/6)==0)

# 2. 024 (22) leading cubic satisfies the six pins at M=(-eta/2,0), S=(eta/2,0) (t-units)
P=kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
M=(-eta/2,0); S=(eta/2,0)
ev=lambda F,pt: sp.simplify(F.subs({X:pt[0],Y:pt[1]}))
chk('(22) pins: P(M)=0, P(S)=-kappa eta^3/6, grad=0 at M,S',
    ev(P,M)==0 and sp.simplify(ev(P,S)+kap*eta**3/6)==0 and all(ev(sp.diff(P,v),pt)==0 for v in (X,Y) for pt in (M,S)))

# 3. re-derive the eta^2 factor of each leading determinant (eq.25) by station elimination
st=[sp.Eq(P.subs({X:c,Y:s}),-al*kap*eta**3/6), sp.Eq(sp.diff(P,X).subs({X:c,Y:s}),0), sp.Eq(sp.diff(P,Y).subs({X:c,Y:s}),0)]
sol=sp.solve(st,[a,Q,z],dict=True)[0]
Hs=sp.hessian(P,(X,Y))
dets={nm:sp.factor(sp.simplify(Hs.subs({X:pt[0],Y:pt[1]}).det().subs(sol).subs(s**2,1-c**2))) for nm,pt in (('M',M),('S',S),('X',(c,s)))}
for nm,dv in dets.items():
    lim=sp.simplify(sp.limit(dv/eta**2,eta,0))
    print(nm,'det/eta^2 at eta->0 =',sp.factor(lim))
    chk(f'det D2P({nm}) vanishes to order eta^2 (eq.25)', sp.simplify(dv.subs(eta,0))==0 and sp.simplify(sp.diff(dv,eta).subs(eta,0))==0 and lim!=0)
prod=sp.factor(dets['M']*dets['S']*dets['X'])
chk('triple product = kappa^6 eta^6 * R (eq.25)', sp.simplify(sp.limit(prod/(kap**6*eta**6),eta,0))!=0 and sp.simplify((prod/eta**6).subs(eta,0))!=sp.nan)

# 4. collar exclusion geometry (21)
rp_=sp.symbols("rp_",positive=True); tmin=sp.sqrt(4*rp_**2-(rp_/2)**2); r_=r; r=rp_
chk("(21) t_min=sqrt(15)/2 r, eta^2<=4/15", sp.simplify(tmin-sp.sqrt(15)/2*r)==0 and sp.simplify((r/tmin)**2-sp.Rational(4,15))==0); r=r_
# (18) window
chk('(18) kappa r^3/6 <= 5/24 r^3 on K_*', sp.Rational(5,4)/6==sp.Rational(5,24))
# 5. (29) integral
tt,de=sp.symbols('tt delta',positive=True); rr=sp.symbols('rr',positive=True)
I=sp.integrate(tt**-5,(tt,sp.sqrt(15)/2*rr,de))
lead=sp.limit(rr**7*I/rr**3,rr,0)
print('(29): r^7 * int t^-5 / r^3 -> ',lead)
chk('(29) r^7 int_{sqrt15 r/2}^delta t^-5 dt = (4/225) r^3 + O(r^7)', lead==sp.Rational(4,225))

# 6. power ledgers (exponent of r)
collar = -2 + -3 + 6          # Z^-1, gradient Jacobian, r^6 triple det (024 eqs 10,11,13)
chk('024 collar ledger: intensity r^{+1}; x area r^2 -> r^3 (16)-(17)', collar==1 and collar+2==3)
P2_I  = -2 + -3 + 6           # (P19): Z, Jacobian r^-3, (P16) r^6 ; Delta^-2*q^2 bounded
P2_II = -2 + -3 + 3 + -2 + -6 # (P20) ledger at l.365
chk('P2 Region I ledger r^{+1} (P19)', P2_I==1)
chk('P2 Region II ledger r^{-10} (P20, l.365)', P2_II==-10)
sing = -2 + 6 + 3             # Z^-1, r^6 dets, window r^3 ; density t^-6, area t dt
chk('024 singular ledger r^7 t^-5 dt (28)-(29)', sing==7)

# 7. Region II / axis: exact collinear model, three zeros of g' at 0, d=rp, r
cc=sp.symbols('cc')
gp=cc*x*(x-d)*(x-r)
csol=sp.solve(sp.Eq(sp.integrate(gp,(x,0,r)),-k*r**3),cc)[0]
print('cubic coefficient c =',sp.factor(csol))
g2=[sp.factor(sp.diff(gp,x).subs(x,v).subs(cc,csol)) for v in (0,d,r)]
print('g\'\'(M),g\'\'(X),g\'\'(S)=',g2)
trip=sp.factor((g2[0]*g2[1]*g2[2]*S0**3).subs(d,r*p))
print('axis triple det (f_zz=S0, f_xz->0):',trip)
chk('axis triple det = -(12k)^3 r^3 p^2 (1-p)^2/(1-2p)^3 S0^3  [order r^3 rho^2, not r^6 rho^2]',
    sp.simplify(trip+(12*k)**3*r**3*p**2*(1-p)**2/(1-2*p)**3*S0**3)==0)
# required A4 on axis from (P4): g'(rp)=0
A4=sp.symbols('A4')
A4s=sp.solve(sp.Eq(6*k*r**2*p*(p-1)+r**3*A4/12*p*(p-1)*(2*p-1),0),A4)[0]
print('A4 forced on axis (P4):',sp.factor(A4s),' -> order 1/r: Gaussian cost exp(-c/r^2)')
# P2 (P16) at Region boundary q=r p versus (P17)
P16=r**6*q**2*(1+(p/q)**2)**3
print('(P16) at q=r p:',sp.factor(P16.subs(q,r*p)),'   (P17): r^3 (p^2+q^2) ->',sp.factor((r**3*(p**2+q**2)).subs(q,r*p)))

# 8. fixed-distance failure of (27)/(28): add pin-preserving quartic t*P4 and re-eliminate
A,B,e1,e2,e3=sp.symbols('A B e1 e2 e3',real=True)
P4=A*(X**2-eta**2/4)**2+B*X*(X**2-eta**2/4)*Y+Y**2*(e1*X**2+e2*X*Y+e3*Y**2)
chk('quartic correction preserves the six pins', all(ev(P4,pt)==0 and ev(sp.diff(P4,X),pt)==0 and ev(sp.diff(P4,Y),pt)==0 for pt in (M,S)))
Pt=P+t*P4
st2=[sp.Eq(Pt.subs({X:c,Y:s}),-al*kap*eta**3/6), sp.Eq(sp.diff(Pt,X).subs({X:c,Y:s}),0), sp.Eq(sp.diff(Pt,Y).subs({X:c,Y:s}),0)]
sol2=sp.solve(st2,[a,Q,z],dict=True)[0]
Ht=sp.hessian(Pt,(X,Y))
dX=sp.simplify(Ht.subs({X:c,Y:s}).det().subs(sol2))
d0=sp.simplify(dX.subs(eta,0))    # eta->0 with t>0: pure quartic-correction part
print('third det at eta=0 (t-units):',sp.factor(d0))
num={c:sp.Rational(3,5),s:sp.Rational(4,5),kap:1,w:sp.Rational(1,3),A:1,B:sp.Rational(1,2),e1:sp.Rational(1,3),e2:sp.Rational(-1,4),e3:sp.Rational(1,5)}
val=sp.simplify(sp.diff(d0,t).subs(t,0).subs(num))
print('d/dt [third det](eta=0,t=0) at a sample point =',val)
chk('third Hessian det has a nonzero O(t) term at eta=0 => physical det H_y ~ t^2(eta^2 + O(t)) = r^2 + O(t^3)', val!=0)
dM=sp.simplify(Ht.subs({X:M[0],Y:M[1]}).det().subs(sol2))
serM=sp.series(dM.subs(num).subs(al,sp.Rational(1,2)),eta,0,2).removeO()
print('endpoint det at M, expansion in eta to O(eta):',sp.factor(sp.expand(serM)))
cM=sp.simplify(sp.expand(serM).coeff(eta,1))
chk('endpoint det has eta*t term (physical r t^2): leading cubic r^2 scaling fails once t^2>r', sp.simplify(cM.subs(t,0))==0 and sp.simplify(cM)!=0)
print(out)
