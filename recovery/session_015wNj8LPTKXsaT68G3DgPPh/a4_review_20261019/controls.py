import sympy as sp, itertools, math, random
from fractions import Fraction as Fr
xi=sp.symbols('xi')
al,be,q=sp.symbols('alpha beta q',positive=True)
P=-al*xi**2/2+be*xi**3/6-sp.Rational(3,2)*xi**4+q*xi**5/120
P0=P.subs({al:6,be:22,q:24})
print('C1 P at base =',sp.expand(P0),' P\' =',sp.factor(sp.diff(P0,xi)))
print('C2 heights P(0..3) =',[P0.subs(xi,k) for k in range(4)])
# C3: open conditions on a box of half-width eps, checked at all 27 grid points per axis-triple (corners+centers)
def conds(a,b,c):
    Pn=sp.lambdify(xi,P.subs({al:a,be:b,q:c}))
    dP=sp.Poly(sp.diff(P.subs({al:a,be:b,q:c}),xi)/xi,xi)
    rts=sorted([complex(r) for r in dP.nroots()],key=lambda z:z.real)
    if any(abs(r.imag)>1e-12 for r in rts): return False,'complex'
    r=[z.real for z in rts]
    ok=(0.75<r[0]<1.25) and (1.75<r[1]<2.25) and (2.75<r[2]<3.25)
    h=[Pn(r[2]),Pn(r[0]),Pn(r[1]),0.0]   # strict order P(xi3)<P(xi1)<P(xi2)<P(0)
    gaps=[h[i+1]-h[i] for i in range(3)]
    ok=ok and min(gaps)>1/8 and 0.5<-Pn(r[2])<1.5
    return ok,(r,gaps,-Pn(r[2]))
for eps in (0.01,0.05,0.1):
    vals=[-eps,0,eps]
    res=[conds(6+d1,22+d2,24+d3) for d1,d2,d3 in itertools.product(vals,vals,vals)]
    print(f'C3 eps={eps}: all 27 box points satisfy the §6 conditions:',all(o for o,_ in res),
          ' min gap=%.4f'%min(min(x[1][1]) for x in res if x[0]),' L0 range=[%.4f,%.4f]'%(min(x[1][2] for x in res),max(x[1][2] for x in res)))
# C4: Jacobian (10) at random theta, lam, a
th,lm,aa=sp.symbols('theta lam a')
R=sp.Matrix([[sp.cos(th),-sp.sin(th)],[sp.sin(th),sp.cos(th)]])
H=R*sp.diag(-lm,aa)*R.T
J=sp.Matrix([[sp.diff(H[0,0],v) for v in (lm,aa,th)],[sp.diff(H[0,1],v) for v in (lm,aa,th)],[sp.diff(H[1,1],v) for v in (lm,aa,th)]])
print('C4 |d(H11,H12,H22)/d(lam,a,theta)| =',sp.simplify(J.det()),'(abs value = |a+lam|)')
# C5: Jacobian (13)
r,a_,b_=sp.symbols('r alpha_ beta_')
J13=sp.Matrix([[sp.diff(e,v) for v in (a_,b_,r)] for e in (a_*r**3,b_*r**2,-36*r)])
print('C5 |d(lam,T,Q4)/d(alpha,beta,r)| =',sp.simplify(J13.det()))
# C6: SO(2) action on Sym^n has determinant one (n=3,4,5), on raw derivative coordinates F_{u^i v^(n-i)}
for n in (3,4,5):
    s,t,x1,x2=sp.symbols('s t x1 x2')
    # a homogeneous degree-n form sum c_i x1^i x2^(n-i)/(i!(n-i)!); rotate the frame by theta, read new raw coefficients
    cs=sp.symbols(f'c0:{n+1}')
    form=sum(cs[i]*x1**i*x2**(n-i)/(sp.factorial(i)*sp.factorial(n-i)) for i in range(n+1))
    ct,st_=sp.cos(th),sp.sin(th)
    rot=sp.expand(form.subs({x1:ct*s-st_*t,x2:st_*s+ct*t},simultaneous=True))
    newc=[sp.diff(rot,s,i,t,n-i).subs({s:0,t:0}) if i>0 and n-i>0 else (sp.diff(rot,s,n).subs({s:0,t:0}) if i==n else sp.diff(rot,t,n).subs({s:0,t:0})) for i in range(n+1)]
    M=sp.Matrix([[sp.diff(nc,cv) for cv in cs] for nc in newc])
    print(f'C6 det of SO(2) action on raw Sym^{n} coordinates =',sp.simplify(M.det().subs(th,sp.Rational(7,10))))
# C7: exponent ledger for A_k: cumulative degree (k-1)(k+2)/2 over height k+1 equals k/2-1/(k+1)
k=sp.symbols('k',positive=True)
print('C7 ledger identity:',sp.simplify((k-1)*(k+2)/(2*(k+1))-(k/2-1/(k+1))),[ (kk/2-sp.Rational(1,kk+1)) for kk in (2,3,4)])
