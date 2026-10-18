# Independent first-order periodization correction of P (15.2), d=3.
# Field K_L(z) = [sum_n exp(-|z+Ln|^2/2)]/[sum_n exp(-|Ln|^2/2)]  (P.md L13-14).
# First order in q=exp(-L^2/2): images +/-L e_i, normalizer 1+6q.
# In the frame (u,y1,y2) a lattice image L e_i has coordinates a = L v, v = R^T e_i.
# Integrand of (15.2): p_G(0) p_V(0) tau^(4/3) D_u ; compute first-order log-variation
# pointwise in u (frame u=e1), then average over the Haar frame = 3 * E_{v in S^2}[F(Lv)].
import sympy as sp
from sympy import Rational as Q
a1,a2,a3,L = sp.symbols('a1 a2 a3 L', real=True)
A=(a1,a2,a3)
w=sp.symbols('w1 w2 w3', real=True)
_X=sp.Symbol('_X')
_HE={n: sp.expand(sp.simplify((-1)**n*sp.exp(_X**2/2)*sp.diff(sp.exp(-_X**2/2),_X,n))) for n in range(7)}
def He(n,x):  # probabilists' Hermite
    return _HE[n].subs(_X,x)
def dK_ref(alpha):  # d^alpha exp(-|z|^2/2) at 0
    return sp.Integer(1) if False else sp.prod([ (-1)**k*He(k,0) for k in alpha])
def dImg(alpha,a):  # d^alpha exp(-|z+a|^2/2) at z=0, divided by exp(-|a|^2/2)
    return sp.prod([(-1)**k*He(k,a[i]) for i,k in enumerate(alpha)])
def tup(*idx):
    al=[0,0,0]
    for i in idx: al[i]+=1
    return tuple(al)
# perturbation (coefficient of q) of d^alpha K(0) from the pair +/-a and the 2q share of normalizer
def delta(alpha):
    if sum(alpha)%2: return sp.Integer(0)
    neg=[ -x for x in A]
    return sp.expand(dImg(alpha,A)+dImg(alpha,neg) - 2*dK_ref(alpha))
def ref(alpha): return dK_ref(alpha)
# covariance of d^alpha f, d^beta f = (-1)^|beta| d^(alpha+beta) K(0)
def cov(al,be,fun):
    s=tuple(x+y for x,y in zip(al,be)); return (-1)**sum(be)*fun(s)
G=[tup(i) for i in range(3)]
t=tup(0,0,0)
V=[tup(0,0),tup(0,1),tup(0,2)]          # H u with u=e1
Aent=[tup(1,1),tup(1,2),tup(2,2)]       # transverse a11,a12,a22
def M(rows,cols,fun): return sp.Matrix(len(rows),len(cols),lambda i,j: cov(rows[i],cols[j],fun))
# reference blocks
CG0=M(G,G,ref); CV0=M(V,V,ref); CA0=M(Aent,Aent,ref); CAV0=M(Aent,V,ref)
ctG0=M([t],G,ref); ctt0=M([t],[t],ref)
dCG=M(G,G,delta); dCV=M(V,V,delta); dCA=M(Aent,Aent,delta); dCAV=M(Aent,V,delta)
dctG=M([t],G,delta); dctt=M([t],[t],delta)
print('CG0',CG0,'CV0',CV0,'ctt0',ctt0,'ctG0',ctG0)
tau2_0=(ctt0-ctG0*CG0.inv()*ctG0.T)[0]
dtau2=(dctt-2*dctG*CG0.inv()*ctG0.T+ctG0*CG0.inv()*dCG*CG0.inv()*ctG0.T)[0]
print('tau2_0',tau2_0)
S0=CA0-CAV0*CV0.inv()*CAV0.T
dS=dCA-dCAV*CV0.inv()*CAV0.T-CAV0*CV0.inv()*dCAV.T+CAV0*CV0.inv()*dCV*CV0.inv()*CAV0.T
print('Sigma_A|V ref',S0)
# Gaussian cone moments under reference A=[[s+x,y],[y,s-x]], s~N(0,5/3), x,y~N(0,1)
s,z=sp.symbols('s z',real=True)
# E[h g] for g quadratic in (a11,a12,a22). Use E[h*s^2], E[h*R^2], E[h]
# with R^2=z ~ density e^{-z/2}/2 on (0,s^2), h=(s^2-z)^2 1{s<0, z<s^2}
vs=Q(5,3)
phis=sp.exp(-s**2/(2*vs))/sp.sqrt(2*sp.pi*vs)
def Es_neg(expr):  # E[expr(s) 1{s<0}]
    return sp.simplify(sp.integrate(expr*phis,(s,-sp.oo,0)))
inner0=sp.integrate((s**2-z)**2*sp.exp(-z/2)/2,(z,0,s**2))
innerR=sp.integrate(z*(s**2-z)**2*sp.exp(-z/2)/2,(z,0,s**2))
Eh=Es_neg(inner0); Ehs2=Es_neg(s**2*inner0); EhR2=Es_neg(innerR)
print('D=',sp.nsimplify(sp.simplify(Eh)), ' check 29/6-sqrt6:', sp.simplify(Eh-(Q(29,6)-sp.sqrt(6))))
# second-moment matrix E[h a a^T] for a=(a11,a12,a22)=(s+x, y, s-x); E[h x^2]=E[h y^2]=EhR2/2, cross terms 0
Ex2=EhR2/2
Mh=sp.Matrix([[Ehs2+Ex2, 0, Ehs2-Ex2],[0, Ex2, 0],[Ehs2-Ex2,0,Ehs2+Ex2]])
S0i=S0.inv()
dlogD=sp.Rational(1,2)*((S0i*dS*S0i*Mh).trace()/Eh - (S0i*dS).trace())
dlogpG=-sp.Rational(1,2)*(CG0.inv()*dCG).trace()
dlogpV=-sp.Rational(1,2)*(CV0.inv()*dCV).trace()
dlogtau43=sp.Rational(2,3)*dtau2/tau2_0
F=sp.expand(dlogD+dlogpG+dlogpV+dlogtau43)
# average over v uniform on S^2 with a=L v ; times 3 axes
Fv=sp.expand(F.subs({a1:L*w[0],a2:L*w[1],a3:L*w[2]}))
poly=sp.Poly(Fv,*w)
def sphmom(e):
    if any(k%2 for k in e): return 0
    num=sp.prod([sp.factorial2(k-1) for k in e]); den=sp.factorial2(sum(e)+1)
    return sp.Rational(num,den)
avg=sum(c*sphmom(m) for m,c in poly.terms())
P3=sp.factor(sp.simplify(3*avg))
print('P3_from_P152 =',P3)
P3chain=-L**2*(10*L**4-147*L**2+315)/105
print('difference vs chain LS-DER-055 (27):', sp.simplify(P3-P3chain))
print('P3(24)=',P3.subs(L,24))
