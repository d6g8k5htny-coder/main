import sympy as sp
r,s,eta=sp.symbols('r s eta',positive=True)
c,t,a,b=sp.symbols('c t a b',real=True)
X,Y=sp.symbols('X Y')
J={(i,j):sp.Symbol(f'J{i}{j}') for i in range(6) for j in range(6) if i+j<=5}
F=sum(J[k]*X**k[0]*Y**k[1]/(sp.factorial(k[0])*sp.factorial(k[1])) for k in J)
Fx,Fy=sp.diff(F,X),sp.diff(F,Y)
ev=lambda e,x,y: e.subs({X:x,Y:y},simultaneous=True)
g=lambda u: ev(F,u,0); gp=lambda u: ev(Fx,u,0)
U0=(g(-r/2)+g(r/2))/2; U1=(g(r/2)-g(-r/2))/r; U2=(gp(r/2)-gp(-r/2))/r
U3=6*(gp(-r/2)+gp(r/2)-2*U1)/r**2; V0=(ev(Fy,-r/2,0)+ev(Fy,r/2,0))/2; V1=(ev(Fy,r/2,0)-ev(Fy,-r/2,0))/r
x0,y0=s*c,s*t; x1,y1=s*(c+eta*a),s*(t+eta*b)
Gu,Gv=ev(Fx,x0,y0),ev(Fy,x0,y0)
Du=(ev(Fx,x1,y1)-Gu)/eta; Dv=(ev(Fy,x1,y1)-Gv)/eta
Ru=Gu-U1-s*(c*U2+t*V1)-s**2*c**2/2*U3
Rv=Gv-V0-s*c*V1
Su=Du-s*(a*U2+b*V1)-s**2*a*c*U3
Sv=Dv-s*a*V1
W0=sp.expand(Rv/(s*t)); W1=sp.expand(Ru/s**2); W2=sp.expand(Su/s**2); W3=sp.expand((Sv-(b/t)*Rv)/s**2)
def lim(e): return sp.simplify(sp.expand(e).subs({r:0}).subs({eta:0}).subs({s:0}))
L1=c*t*J[(2,1)]+t**2/2*J[(1,2)]
L2=(a*t+b*c)*J[(2,1)]+b*t*J[(1,2)]
L3=(a*c-b*c**2/(2*t))*J[(2,1)]+a*t*J[(1,2)]+b*t/2*J[(0,3)]
print('W0 ->J02:',sp.simplify(lim(W0)-J[(0,2)])==0)
print('W1 ->L1:',sp.simplify(lim(W1)-L1)==0,' W2 ->L2:',sp.simplify(lim(W2)-L2)==0,' W3 ->L3:',sp.simplify(lim(W3)-L3)==0)
# remainder structure: W - limit is O(r^2 + s + eta) : check no negative powers and vanish at r=s=eta=0
for nm,Wn,Ln in [('W0',W0,J[(0,2)]),('W1',W1,L1),('W2',W2,L2),('W3',W3,L3)]:
    q=sp.Symbol('q',positive=True)
    d=sp.expand(sp.expand(Wn-Ln).subs(r,q*s)); P=sp.Poly(d,q,s,eta)
    mons=P.monoms(); low=sorted(set(mons))[:6]
    bad=[m for m in mons if m==(0,0,0) or m==(1,0,0)]
    print(nm,'remainder polynomial in (q,s,eta) with no negative powers: True; no constant or pure-q^1 term:',bad==[],'; min q-power among s^0 eta^0 terms:',min([m[0] for m in mons if m[1]==0 and m[2]==0] or [None]))
M=sp.Matrix([[c*t,t**2/2,0],[a*t+b*c,b*t,0],[a*c-b*c**2/(2*t),a*t,b*t/2]])
print('det M = b t^3 (bc-at)/4:',sp.simplify(M.det()-b*t**3*(b*c-a*t)/4)==0)
print('b=0 rank:',M.subs(b,0).rank())
lam,nz=sp.symbols('lam nz',positive=True)
Mr=M.subs({a:lam*c/nz,b:lam*t/nz}); print('radial rank:',sp.Matrix(Mr).rank(simplify=True))
# Jacobian: raw (pins 6, grad x 2, grad y 2) -> (P_r, W0..W3)
raw=sp.symbols('Fa Dua Dva Fc Duc Dvc gu gv hu hv')
Fa,Dua,Dva,Fc,Duc,Dvc,gu,gv,hu,hv=raw
u0=(Fa+Fc)/2;u1=(Fc-Fa)/r;u2=(Duc-Dua)/r;u3=6*(Dua+Duc-2*u1)/r**2;v0=(Dva+Dvc)/2;v1=(Dvc-Dva)/r
du=(hu-gu)/eta; dv=(hv-gv)/eta
ru=gu-u1-s*(c*u2+t*v1)-s**2*c**2/2*u3; rv=gv-v0-s*c*v1; su=du-s*(a*u2+b*v1)-s**2*a*c*u3; sv=dv-s*a*v1
vec=[u0,u1,u2,u3,v0,v1,rv/(s*t),ru/s**2,su/s**2,(sv-(b/t)*rv)/s**2]
Jm=sp.Matrix([[sp.diff(v,x) for x in raw] for v in vec]); dJ=sp.factor(sp.simplify(Jm.det()))
print('Jacobian det:',dJ,'  |.| == 12 r^-5 eta^-2 |t|^-1 s^-7:',sp.simplify(sp.Abs(dJ)-12/(r**5*eta**2*sp.Abs(t)*s**7))==0)
