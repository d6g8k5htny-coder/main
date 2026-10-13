from fractions import Fraction as F
import itertools
# multivariate polynomials over Q in variables: X,Z,s,a,b,c,k (exponent tuples)
V=['X','Z','s','a','b','c','k']; n=len(V)
def var(name, coef=F(1)):
    e=[0]*n; e[V.index(name)]=1; return {tuple(e):coef}
def const(v): return {tuple([0]*n):F(v)} if v!=0 else {}
def add(p,q,sg=1):
    r=dict(p)
    for kk,v in q.items(): r[kk]=r.get(kk,F(0))+sg*v
    return {kk:v for kk,v in r.items() if v!=0}
def mul(p,q):
    r={}
    for e1,u in p.items():
        for e2,v in q.items():
            e=tuple(x+y for x,y in zip(e1,e2)); r[e]=r.get(e,F(0))+u*v
    return {kk:v for kk,v in r.items() if v!=0}
def smul(c,p): return {kk:F(c)*v for kk,v in p.items()}
def powr(p,m):
    r=const(1)
    for _ in range(m): r=mul(r,p)
    return r
def diff(p,name):
    i=V.index(name); r={}
    for e,v in p.items():
        if e[i]>0:
            e2=list(e); e2[i]-=1; r[tuple(e2)]=r.get(tuple(e2),F(0))+v*e[i]
    return {kk:v for kk,v in r.items() if v!=0}
def subs(p,name,q):
    i=V.index(name); r={}
    for e,v in p.items():
        e2=list(e); m=e2[i]; e2[i]=0
        term=mul({tuple(e2):v},powr(q,m))
        r=add(r,term)
    return r
X,Z,s,a,b,c,k=[var(t) for t in V]
half=F(1,2)
F0=add(add(add(add(add(add(add(smul(2,mul(k,powr(X,3))), smul(-F(3,2),mul(k,X))), smul(-half,k)), smul(half,mul(s,powr(Z,2)))), smul(half,mul(a,mul(add(powr(X,2),const(-F(1,4))),Z)))), smul(half,mul(b,mul(X,powr(Z,2))))), smul(F(1,6),mul(c,powr(Z,3)))), {})
gx,gz=diff(F0,'X'),diff(F0,'Z')
def at(p,x,z): return subs(subs(p,'X',const(x)),'Z',const(z))
print('pins: F0(M)=',at(F0,-half,0),' F0(S)=',at(F0,half,0),' gx(M)=',at(gx,-half,0),' gz(M)=',at(gz,-half,0),' gx(S)=',at(gx,half,0),' gz(S)=',at(gz,half,0))
# Hessian at pins
Hxx,Hxz,Hzz=diff(gx,'X'),diff(gx,'Z'),diff(gz,'Z')
for nm,x in (('M',-half),('S',half)):
    print(nm,'Hxx=',at(Hxx,x,0),' Hxz=',at(Hxz,x,0),' Hzz=',at(Hzz,x,0))
# resultant in Z: gx = A2 Z^2 + A1 Z + A0, gz = B2 Z^2 + B1 Z + B0 (coefficients polys in X,s,a,b,c,k)
def coeffZ(p,j):
    i=V.index('Z'); r={}
    for e,v in p.items():
        if e[i]==j:
            e2=list(e); e2[i]=0; r[tuple(e2)]=v
    return r
A=[coeffZ(gx,j) for j in range(3)]; B=[coeffZ(gz,j) for j in range(3)]
# Sylvester 4x4 determinant for two quadratics: res = (A2B0-A0B2)^2 - (A2B1-A1B2)(A1B0-A0B1)
t1=add(mul(A[2],B[0]),mul(A[0],B[2]),-1); t2=add(mul(A[2],B[1]),mul(A[1],B[2]),-1); t3=add(mul(A[1],B[0]),mul(A[0],B[1]),-1)
res=add(mul(t1,t1),mul(t2,t3),-1)
# divide by (X^2-1/4) exactly: polynomial division in X
def degX(p): return max((e[0] for e in p),default=-1)
def divX(p,d):  # d = X^2 - 1/4
    q={}; r=dict(p)
    while degX(r)>=2:
        m=degX(r); lead={e:v for e,v in r.items() if e[0]==m}
        # shift lead by X^-2
        sh={}
        for e,v in lead.items():
            e2=list(e); e2[0]-=2; sh[tuple(e2)]=v
        q=add(q,sh); r=add(r,mul(sh,d),-1)
    return q,r
d=add(powr(X,2),const(-F(1,4)))
Q2,rem=divX(res,d)
print('resultant divisible by X^2-1/4:',rem=={})
def pretty(p):
    out=[]
    for e,v in sorted(p.items(),reverse=True):
        mono='*'.join(f'{V[i]}^{e[i]}' if e[i]>1 else V[i] for i in range(n) if e[i]>0)
        out.append(f'({v})'+('*'+mono if mono else ''))
    return ' + '.join(out)
print('Q2 (quadratic in X whose roots are the extra critical X):'); print(pretty(Q2))
# group by X power
for j in (2,1,0):
    print(f'  X^{j}:', pretty({tuple([0]+list(e[1:])):v for e,v in Q2.items() if e[0]==j}))
# block determinant expansion check (d=3): H = [[-6k r, -a r/2, r c1],[-a r/2, r(s-b/2), r c2],[r c1, r c2, D]] : det = r^2 D det(M2) + r^3(...)
# use variables: reuse 'X'->r, 'Z'->c1, 'c'->c2, 'k'->D? do fresh small check with Fractions numerically at random values and polynomial in r
import random
def det3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
random.seed(1)
for _ in range(3):
    kk,aa,ss,bb,c1,c2,D=[F(random.randint(-9,9),random.randint(1,5)) for _ in range(7)]
    if D==0: D=F(1)
    vals=[]
    for rr in (F(1,7),F(1,11),F(1,13),F(1,17)):
        H=[[-6*kk*rr,-aa*rr/2,rr*c1],[-aa*rr/2,rr*(ss-bb/2),rr*c2],[rr*c1,rr*c2,D]]
        vals.append((rr,det3(H)))
    detM2=(-6*kk)*(ss-bb/2)-aa*aa/4
    # det = r^2 D detM2 + r^3 * cubic-term(c1,c2..) exactly: check (det - r^2 D detM2)/r^3 is constant
    consts=set((v - rr*rr*D*detM2)/rr**3 for rr,v in vals)
    print('block det: r^2 coefficient = D*det(M2)? residual/r^3 constant across r:', len(consts)==1, ' value', consts)
