import sympy as sp
r,s,eta,q=sp.symbols('r s eta q',positive=True); c=sp.symbols('c',real=True)
X,Y=sp.symbols('X Y')
J={(i,j):sp.Symbol(f'J{i}{j}') for i in range(7) for j in range(7) if i+j<=6}
F=sum(J[k]*X**k[0]*Y**k[1]/(sp.factorial(k[0])*sp.factorial(k[1])) for k in J)
Fx,Fy=sp.diff(F,X),sp.diff(F,Y); ev=lambda e,x,y: e.subs({X:x,Y:y},simultaneous=True)
g_=lambda u: ev(F,u,0); gp=lambda u: ev(Fx,u,0)
U1=(g_(r/2)-g_(-r/2))/r; U2=(gp(r/2)-gp(-r/2))/r; U3=6*(gp(-r/2)+gp(r/2)-2*U1)/r**2
V0=(ev(Fy,-r/2,0)+ev(Fy,r/2,0))/2; V1=(ev(Fy,r/2,0)-ev(Fy,-r/2,0))/r; A1=U1-r**2*U3/24
def negs(e):
    out={}
    for term in sp.Add.make_args(e):
        if term.as_coeff_exponent(s)[1]<0: out.setdefault(term.as_coeff_exponent(s)[1],0); out[term.as_coeff_exponent(s)[1]]+=term
    return {k:sp.factor(v) for k,v in out.items() if sp.simplify(v)!=0}
for a in (1,-1):
    x0=s*c; x1=s*(c+eta*a)
    Gu,Gv=ev(Fx,x0,0),ev(Fy,x0,0); Du=(ev(Fx,x1,0)-Gu)/eta; Dv=(ev(Fy,x1,0)-Gv)/eta
    Rus=Gu-A1-s*c*U2-s**2*c**2/2*U3; Rv=Gv-V0-s*c*V1
    Su=Du-s*a*U2-s**2*a*c*U3; Sv=Dv-s*a*V1; Sus=Su-eta*s**2*a**2/2*U3
    h=c**2/2-q**2/8; g=c**3/6-q**2*c/24; m=a*c+eta*a**2/2
    k=a*(c**2/2-q**2/24)+eta*a**2*c/2+eta**2*a**3/6
    A=Rus/s**3; Bv=Rv/s**2; C=(Sus/s**3-(k/g)*A)/s; D=(Sv/s**2-(m/h)*Bv)/s
    out=[]
    for nm,row,lim in [('A',A,c**3/6*J[(4,0)]),('B',Bv,c**2/2*J[(2,1)]),('C',C,a*c**3/24*J[(5,0)]),('D',D,a*c**2/6*J[(3,1)])]:
        e=sp.expand(sp.together(sp.expand(row).subs(r,q*s)))
        num,den=sp.fraction(sp.together(e)); num=sp.expand(num)
        # negative powers: check num divisible appropriately -> compute series via substitution
        ex=sp.expand(sp.cancel(e))
        ng=negs(sp.expand(sp.apart(ex,s)) if False else ex)
        L=sp.simplify(sp.limit(sp.limit(sp.limit(ex,q,0),eta,0),s,0)) if not ng else None
        out.append((nm,ng, None if L is None else sp.simplify(L-lim)==0))
    print('a=%+d'%a,out)
# Jacobian (symbolic a=1)
raw=sp.symbols('Fm D1m D2m Fp D1p D2p gu gv hu hv'); Fm,D1m,D2m,Fp,D1p,D2p,gu,gv,hu,hv=raw
a=1
u0=(Fm+Fp)/2;u1=(Fp-Fm)/r;u2=(D1p-D1m)/r;u3=6*(D1m+D1p-2*u1)/r**2;v0=(D2m+D2p)/2;v1=(D2p-D2m)/r;a1=u1-r**2*u3/24
du=(hu-gu)/eta;dv=(hv-gv)/eta
rus=gu-a1-s*c*u2-s**2*c**2/2*u3; rv=gv-v0-s*c*v1; su=du-s*a*u2-s**2*a*c*u3; sv=dv-s*a*v1; sus=su-eta*s**2*a**2/2*u3
qq=r/s; h=c**2/2-qq**2/8; g=c**3/6-qq**2*c/24; m=a*c+eta*a**2/2; k=a*(c**2/2-qq**2/24)+eta*a**2*c/2+eta**2*a**3/6
vec=[u0,u1,u2,u3,v0,v1,rus/s**3,rv/s**2,(sus/s**3-(k/g)*rus/s**3)/s,(sv/s**2-(m/h)*rv/s**2)/s]
Jm=sp.Matrix([[sp.diff(v,x) for x in raw] for v in vec]); d=sp.factor(sp.simplify(Jm.det()))
print('Jacobian:',d,'; ==12 r^-5 eta^-2 s^-12 in abs:',sp.simplify(sp.Abs(d)-12/(r**5*eta**2*s**12))==0)
