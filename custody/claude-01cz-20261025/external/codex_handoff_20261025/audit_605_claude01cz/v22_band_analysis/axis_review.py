import sympy as sp
r,s,eta,q=sp.symbols('r s eta q',positive=True)
c,a,b=sp.symbols('c a b',real=True)
X,Y=sp.symbols('X Y')
J={(i,j):sp.Symbol(f'J{i}{j}') for i in range(6) for j in range(6) if i+j<=5}
F=sum(J[k]*X**k[0]*Y**k[1]/(sp.factorial(k[0])*sp.factorial(k[1])) for k in J)
Fx,Fy=sp.diff(F,X),sp.diff(F,Y); ev=lambda e,x,y: e.subs({X:x,Y:y},simultaneous=True)
g=lambda u: ev(F,u,0); gp=lambda u: ev(Fx,u,0)
U1=(g(r/2)-g(-r/2))/r; U2=(gp(r/2)-gp(-r/2))/r; U3=6*(gp(-r/2)+gp(r/2)-2*U1)/r**2
V0=(ev(Fy,-r/2,0)+ev(Fy,r/2,0))/2; V1=(ev(Fy,r/2,0)-ev(Fy,-r/2,0))/r
x0,y0=s*c,0; x1,y1=s*(c+eta*a),s*eta*b
Gu,Gv=ev(Fx,x0,y0),ev(Fy,x0,y0); Du=(ev(Fx,x1,y1)-Gu)/eta; Dv=(ev(Fy,x1,y1)-Gv)/eta
Ru=Gu-U1-s*c*U2-s**2*c**2/2*U3; Rv=Gv-V0-s*c*V1; Su=Du-s*(a*U2+b*V1)-s**2*a*c*U3; Sv=Dv-s*a*V1
hq=c**2/2-q**2/8
W1s=Ru/s**2+(q**2/24)*U3
E=Sv/(s*b); Bv=Rv/s**2; A=W1s/s
N=Su/s**2-(eta*a**2/2)*U3-(b*(c+eta*a)/hq)*Bv
sub=lambda e: sp.expand(sp.expand(e).subs(r,q*s))
def negs(e):
    out={}
    for term in sp.Add.make_args(e):
        es=term.as_coeff_exponent(s)[1]; ee=term.as_coeff_exponent(eta)[1]
        if es<0 or ee<0: out[(es,ee)]=out.get((es,ee),0)+term
    return {k:sp.simplify(v) for k,v in out.items() if sp.simplify(v)!=0}
L=lambda e: sp.simplify(sub(e).subs({q:0}).subs({eta:0}).subs({s:0}))
print('B J21 coeff at finite q:',sp.simplify(sub(Bv).coeff(J[(2,1)]).subs({s:0,eta:0})-hq)==0 or sp.factor(sub(Bv).coeff(J[(2,1)])))
print('E->J02:',sp.simplify(L(E)-J[(0,2)])==0,'  B->c^2/2 J21:',sp.simplify(L(Bv)-c**2/2*J[(2,1)])==0,'  A limit:',sp.factor(L(A)))
Nx=sub(N); print('N negative powers:',negs(Nx))
N00=sp.simplify(Nx.subs({s:0,eta:0}) if not negs(Nx) else None); print('N(s=0,eta=0) for all q:',N00)
Xc=sp.simplify(sp.diff(Nx,s).subs({s:0,eta:0,q:0})); Yc=sp.simplify(sp.diff(Nx,eta).subs({s:0,eta:0,q:0}))
print('X =',sp.expand(Xc)); print('Y =',sp.expand(Yc))
# Jacobian
raw=sp.symbols('Fm D1m D2m Fp D1p D2p gu gv hu hv'); Fm,D1m,D2m,Fp,D1p,D2p,gu,gv,hu,hv=raw
u0=(Fm+Fp)/2;u1=(Fp-Fm)/r;u2=(D1p-D1m)/r;u3=6*(D1m+D1p-2*u1)/r**2;v0=(D2m+D2p)/2;v1=(D2p-D2m)/r
du=(hu-gu)/eta;dv=(hv-gv)/eta
ru=gu-u1-s*c*u2-s**2*c**2/2*u3; rv=gv-v0-s*c*v1; su=du-s*(a*u2+b*v1)-s**2*a*c*u3; sv=dv-s*a*v1
dl=s+eta; hq_r=c**2/2-(r/s)**2/8
vec=[u0,u1,u2,u3,v0,v1,sv/(s*b),rv/s**2,(ru/s**2+((r/s)**2/24)*u3)/s,(su/s**2-(eta*a**2/2)*u3-(b*(c+eta*a)/hq_r)*rv/s**2)/dl]
Jm=sp.Matrix([[sp.diff(v,x) for x in raw] for v in vec]); d=sp.factor(sp.simplify(Jm.det()))
print('Jacobian:',d,' ; |.|==12 r^-5 eta^-2 |b|^-1 s^-8 (s+eta)^-1:',sp.simplify(sp.Abs(d)-12/(r**5*eta**2*sp.Abs(b)*s**8*(s+eta)))==0)
