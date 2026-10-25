import sympy as sp
r,e,p,t,q,v=sp.symbols('r epsilon p tau q v',positive=True)
nu=t+e*v
# raw obs in order (4): Fa, DuFa, DvFa, Fc, DuFc, DvFc, DuFx, DvFx, DuFy, DvFy
X=sp.symbols('Fa Dua Dva Fc Duc Dvc Dux Dvx Duy Dvy'); Fa,Dua,Dva,Fc,Duc,Dvc,Dux,Dvx,Duy,Dvy=X
U0=(Fa+Fc)/2; U1=(Fc-Fa)/r; U2=(Duc-Dua)/r; U3=6*(Dua+Duc-2*U1)/r**2
V0=(Dva+Dvc)/2; V1=(Dvc-Dva)/r; A1=U1-r**2*U3/24
B=lambda s,tt,Dv: (Dv-V0-r*s*V1)/r
M=lambda s,tt,Du: (Du-A1-r*s*U2-r*tt*V1)/r**2-(s**2/2)*U3
h=p+e*q
W0=B(p,t,Dvx)/t; M1=M(p,t,Dux); M2=M(h,nu,Duy); N=(M2-M1)/e
G=(B(h,nu,Dvy)/nu-B(p,t,Dvx)/t)/(r*e)
Svec=[U0,A1,U2,U3,V0,V1,W0,M1,N,G]
J=sp.Matrix([[sp.diff(s,x) for x in X] for s in Svec])
d=sp.simplify(J.det()); print('det =',sp.factor(d)); print('matches -12/(r^12 eps^2 tau nu):',sp.simplify(d+12/(r**12*e**2*t*nu))==0)
