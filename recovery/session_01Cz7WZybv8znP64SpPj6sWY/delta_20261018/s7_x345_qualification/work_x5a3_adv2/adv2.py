import sympy as sp, random
r,k,b,u,s=sp.symbols('r k b u s',positive=True)
# generic degree-7 g on [0,r] with pins g(0)=b, g(r)=b-k r^3, g'(0)=g'(r)=0
cs=sp.symbols('c0:8'); g=sum(cs[i]*u**i for i in range(8))
sol=sp.solve([g.subs(u,0)-b, g.subs(u,r)-(b-k*r**3), sp.diff(g,u).subs(u,0), sp.diff(g,u).subs(u,r)],[cs[0],cs[1],cs[2],cs[3]],dict=True)[0]
g=sp.expand(g.subs(sol)); g3=sp.diff(g,u,3)
ok=[]
ok.append(sp.simplify(sp.integrate(sp.diff(g,u,2),(u,0,r)))==0)
ok.append(sp.simplify(sp.integrate(u*(r-u)*g3,(u,0,r))-2*k*r**3)==0)
ok.append(sp.simplify(sp.integrate(u*(r-u),(u,0,r))-r**3/6)==0)
ok.append(sp.simplify(sp.diff(g,u,2).subs(u,0)+sp.integrate((r-u)*g3,(u,0,r))/r)==0)
ok.append(sp.simplify(sp.diff(g,u,2).subs(u,r)-sp.integrate(u*g3,(u,0,r))/r)==0)
# N1 double integral: r g''(0) = -int_0^r int_0^s g'''(u) du ds
inner=sp.integrate(g3,(u,0,s)); ok.append(sp.simplify(r*sp.diff(g,u,2).subs(u,0)+sp.integrate(inner,(s,0,r)))==0)
# U3 equals weighted mean for general (non-pinned) g
h=sum(sp.Symbol('d%d'%i)*u**i for i in range(8))
U3=6/r**2*(sp.diff(h,u).subs(u,0)+sp.diff(h,u).subs(u,r)-2*(h.subs(u,r)-h.subs(u,0))/r)
ok.append(sp.simplify(U3-sp.integrate(u*(r-u)*sp.diff(h,u,3),(u,0,r))/(r**3/6))==0)
print('1D identities',ok)
# (5.3) and congruence for m=1..3
for m in (1,2,3):
    al=sp.Symbol('al'); be=sp.Matrix(sp.symbols('be0:%d'%m))
    A=sp.Matrix(m,m,lambda i,j: sp.Symbol('a%d%d'%(min(i,j),max(i,j))))
    H=sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r*al]]), r*be.T],[r*be, A]]))
    adj=A.adjugate() if m>1 else sp.Matrix([[1]])
    e1=sp.expand(H.det()/r-(al*A.det()-r*(be.T*adj*be)[0]))==0
    Dr=sp.diag(r**sp.Rational(-1,2),*([1]*m))
    S=Dr*H*Dr
    tgt=sp.Matrix(sp.BlockMatrix([[sp.Matrix([[al]]), sp.sqrt(r)*be.T],[sp.sqrt(r)*be, A]]))
    e2=sp.simplify(S-tgt)==sp.zeros(m+1,m+1)
    e3=sp.simplify(S.det()-H.det()/r)==0
    print('m',m,'(5.3)',e1,'D_r H D_r',e2,'det/r',e3)
# numeric sanity of (5.1)/(5.2) bounds with random pinned polynomials
import math
random.seed(1); worst1=0; worst2=0
for t in range(300):
    rv=random.choice([0.05,0.2,1.0]); kv=random.uniform(0.2,3)
    sub={r:rv,k:kv,b:0}
    for i in range(4,8): sub[cs[i]]=random.uniform(-5,5)/rv**(i-3)
    gg=sp.lambdify(u,g.subs(sub)); g2=sp.lambdify(u,sp.diff(g,u,2).subs(sub)); g3f=sp.lambdify(u,g3.subs(sub)); g4f=sp.lambdify(u,sp.diff(g,u,4).subs(sub))
    xs=[rv*i/2000 for i in range(2001)]
    M3=max(abs(g3f(x)) for x in xs); M4=max(abs(g4f(x)) for x in xs)
    worst1=max(worst1,abs(g2(0))/rv/(M3/2), abs(g2(rv))/rv/(M3/2))
    if M4>0: worst2=max(worst2,abs(g2(0)/rv+6*kv)/(rv*M4/2), abs(g2(rv)/rv-6*kv)/(rv*M4/2))
print('max ratio to (5.1) bound',worst1,' to (5.2) bound',worst2)
