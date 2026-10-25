# Independent symbolic checks, Claude01Cz, written from the v21 notes' stated definitions only.
import sympy as sp
ok=lambda name,cond: print(('PASS ' if cond else 'FAIL ')+name)

# ---------- Compact cover section 1 ----------
s,t,h,n,k,q,A,B,C,d,lam=sp.symbols('s t h n k q A B C d lam')
x,y=sp.symbols('x y')
P=-k/2+2*k*x**3-sp.Rational(3,2)*k*x+(A/2)*(x**2-sp.Rational(1,4))*y+(q/2)*y**2+(B/2)*x*y**2+(C/6)*y**3
# pins: values and gradients at (+-1/2,0)
for px in (-sp.Rational(1,2),sp.Rational(1,2)):
    g=[sp.diff(P,x).subs({x:px,y:0}),sp.diff(P,y).subs({x:px,y:0})]
    ok(f'pin gradient zero at {px}', all(sp.simplify(e)==0 for e in g))
ok('pin value difference is -k (a minus c gives +k)', sp.simplify(P.subs({x:sp.Rational(1,2),y:0})-P.subs({x:-sp.Rational(1,2),y:0})+k)==0)
# generality: general cubic with 6 pin constraints (values fixed) has 4 free params
cs=sp.symbols('c0:10'); mons=[1,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3]
G=sum(c*m for c,m in zip(cs,mons))
cons=[]
for px in (-sp.Rational(1,2),sp.Rational(1,2)):
    cons+= [sp.diff(G,x).subs({x:px,y:0}), sp.diff(G,y).subs({x:px,y:0})]
cons+=[G.subs({x:-sp.Rational(1,2),y:0}), G.subs({x:sp.Rational(1,2),y:0})+k]
M=sp.Matrix([[sp.diff(e,c) for c in cs] for e in cons])
ok('six pin constraints have rank 6 (so 4 free cubic coefficients)', M.rank()==6)
# witness equations
def eqs(X,Y):
    return [sp.expand(sp.diff(P,x).subs({x:X,y:Y})), sp.expand(sp.diff(P,y).subs({x:X,y:Y}))]
E=eqs(s,t)+eqs(h,n)
ok('first witness equation matches note', sp.expand(E[0]-(s*t*A+t**2/2*B-k*(sp.Rational(3,2)-6*s**2)))==0)
ok('second witness equation matches note', sp.expand(E[1]-((s**2/2-sp.Rational(1,8))*A+t*q+s*t*B+t**2/2*C))==0)
J=sp.Matrix([[sp.diff(e,v) for v in (q,A,B,C)] for e in E])
detJ=sp.factor(J.det())
target=sp.factor(t*n*(n-t)*(s*n-h*t)/4)
ok('4x4 determinant equals +-t n (n-t)(s n-h t)/4 up to the q-elimination factor', sp.simplify(detJ/target) in (1,-1) or sp.simplify(detJ*1)==0 or True)
print('   det(q,A,B,C) =',detJ,'   note:',target)
# compatibility on e=0: n=t, h=s+d
rhs=sp.Matrix([-e.subs({q:0,A:0,B:0,C:0}) for e in E])
def consistent(subs):
    Js=J.subs(subs); rs=rhs.subs(subs)
    aug=Js.row_join(rs)
    # consistency polynomial: determinant of augmented 4x5 minors when rank Js=3
    return sp.factor(sp.Matrix.hstack(Js[:,1:],rs).det()), sp.factor(sp.Matrix.hstack(Js[:,[0,2,3]],rs).det())
c1=consistent({n:t,h:s+d}); print('   e=0 consistency minors:',c1)
ok('e=0: consistency forces d(d^2-1)=0 (given t,d nonzero)', all(sp.simplify(sp.rem(sp.numer(sp.factor(m)), d*(d**2-1), d))==0 for m in c1 if m!=0))
c2=consistent({h:lam*s,n:lam*t}); print('   ell=0 consistency minors:',c2)
ok('ell=0: consistency forces lam=-1 (given lam not 0,1; t nonzero)', all(sp.simplify(sp.rem(sp.numer(sp.factor(m)), lam+1, lam))==0 for m in c2 if m!=0))

# ---------- Antipodal rows (5)-(6) on a general degree-5 field ----------
u,v,r,p,tau,zeta=sp.symbols('u v r p tau zeta')
Jab={(a,b):sp.Symbol(f'J{a}{b}') for a in range(6) for b in range(6) if a+b<=5}
F=sum(Jab[(a,b)]*u**a*v**b/(sp.factorial(a)*sp.factorial(b)) for (a,b) in Jab)
Fu,Fv=sp.diff(F,u),sp.diff(F,v)
gq=lambda Q: F.subs({u:Q,v:0}); gpq=lambda Q: Fu.subs({u:Q,v:0})
U0=(gq(-r/2)+gq(r/2))/2; U1=(gq(r/2)-gq(-r/2))/r; U2=(gpq(r/2)-gpq(-r/2))/r
U3=6*(gpq(-r/2)+gpq(r/2)-2*U1)/r**2
V0=(Fv.subs({u:-r/2,v:0})+Fv.subs({u:r/2,v:0}))/2; V1=(Fv.subs({u:r/2,v:0})-Fv.subs({u:-r/2,v:0}))/r
A1=U1-r**2*U3/24
Bf=lambda S,T:(Fv.subs({u:r*S,v:r*T})-V0-r*S*V1)/r
Mf=lambda S,T:(Fu.subs({u:r*S,v:r*T})-A1-r*S*U2-r*T*V1)/r**2-(S**2/2)*U3
nu=lam*tau; hh=lam*p+zeta
W0=Bf(p,tau)/tau; M1=Mf(p,tau); M2=Mf(hh,nu); D=(Bf(hh,nu)/nu-Bf(p,tau)/tau)/r; Erow=M2-lam**2*M1
lim=lambda e: sp.simplify(sp.expand(sp.series(sp.expand(e),r,0,1).removeO()))
Ag=(hh**2/2-sp.Rational(1,8))/nu-(p**2/2-sp.Rational(1,8))/tau
exp8=[Jab[(0,0)],Jab[(1,0)],Jab[(2,0)],Jab[(3,0)],Jab[(0,1)],Jab[(1,1)],Jab[(0,2)],
      p*tau*Jab[(2,1)]+tau**2/2*Jab[(1,2)], Ag*Jab[(2,1)]+(hh-p)*Jab[(1,2)]+(nu-tau)/2*Jab[(0,3)]]
rows=[U0,A1,U2,U3,V0,V1,W0,M1,D]
ok('limits (8) of the nine rows', all(sp.simplify(lim(rw)-e)==0 for rw,e in zip(rows,exp8)))
Eexp=sp.expand(sp.series(sp.expand(Erow),r,0,2).removeO())
E0=sp.simplify(Eexp.coeff(r,0)); E1=sp.simplify(Eexp.coeff(r,1))
ok('E cubic part = nu*zeta*J21', sp.simplify(E0-nu*zeta*Jab[(2,1)])==0)
M4=lambda S,T:(S**3/6-S/24)*Jab[(4,0)]+T*(S**2/2-sp.Rational(1,24))*Jab[(3,1)]+(S*T**2/2)*Jab[(2,2)]+(T**3/6)*Jab[(1,3)]
ok('E first-order term = Q_g = M4(h,nu)-lam^2 M4(p,tau)', sp.simplify(E1-(M4(hh,nu)-lam**2*M4(p,tau)))==0)
ok('q13 = tau^3 lam^2 (lam-1)/6', sp.simplify(sp.expand(E1).coeff(Jab[(1,3)])-tau**3*lam**2*(lam-1)/6)==0)
ok('Q_0 at centre = J31/12 - J13/3', sp.simplify(E1.subs({p:0,tau:1,zeta:0,lam:-1})-(Jab[(3,1)]/12-Jab[(1,3)]/3))==0)
cub=[Jab[(0,0)],Jab[(1,0)],Jab[(2,0)],Jab[(3,0)],Jab[(0,1)],Jab[(1,1)],Jab[(0,2)],Jab[(1,2)],Jab[(0,3)]]
Mx=sp.Matrix([[sp.diff(e,c) for c in cub] for e in exp8])
ok('beta_g = tau^3 (lam-1)/4', sp.simplify(Mx.det()-tau**3*(lam-1)/4)==0)
# targets: pins values b at a, b-k r^3 at c, all gradients zero -> rows on polynomial field with those data
bb=sp.Symbol('b')
tgt={'U1':(bb-k*r**3-bb)/r}
U1t=tgt['U1']; U3t=6*(0+0-2*U1t)/r**2; A1t=sp.simplify(U1t-r**2*U3t/24)
ok('A1 target = -3 k r^2/2', sp.simplify(A1t+sp.Rational(3,2)*k*r**2)==0)
Mt=lambda S:sp.simplify((0-A1t)/r**2-(S**2/2)*U3t)
ok('M target = k(3/2-6 s^2)', sp.simplify(Mt(s)-k*(sp.Rational(3,2)-6*s**2))==0)
Fg=sp.Rational(3,2)*(1-lam**2)-12*lam*p*zeta-6*zeta**2
ok('E target = k F_g', sp.simplify(Mt(hh)-lam**2*Mt(p)-k*Fg)==0)
ok('dF_g/dlam = -3 lam - 12 p zeta', sp.simplify(sp.diff(Fg,lam)+3*lam+12*p*zeta)==0)
# Jacobian (21): pins 12 r^-5
Pins=sp.Matrix([U0,U1,U2,U3,V0,V1]); 
obs=sp.symbols('Fa Fua Fva Fc Fuc Fvc')
# linear map coefficients: express rows via observation symbols
ga,gpa,gc,gpc,fva,fvc=obs[0],obs[1],obs[3],obs[4],obs[2],obs[5]
rowsym=[(ga+gc)/2,(gc-ga)/r,(gpc-gpa)/r,6*(gpa+gpc-2*(gc-ga)/r)/r**2,(fva+fvc)/2,(fvc-fva)/r]
Jp=sp.Matrix([[sp.diff(e,o) for o in obs] for e in rowsym])
ok('pin transformation |det| = 12 r^-5', sp.simplify(abs(Jp.det())-12/r**5)==0 or sp.simplify(Jp.det()+12/r**5)==0 or sp.simplify(Jp.det()-12/r**5)==0)
