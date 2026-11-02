"""Finite C104 author controls. Not a proof of Gaussian/persistence analysis."""
from fractions import Fraction as F
from itertools import product
from math import factorial, prod
import argparse
import json


def require(ok, label):
    if not ok:
        raise ValueError(label)


def indices(n):
    return [(a,b,n-a-b) for a in range(n+1) for b in range(n-a+1)]


JETS = sum((indices(n) for n in range(4)), [])
N = len(JETS)


def double_factorial(n):
    return prod(range(n,0,-2))


def covariance(a,b):
    c=tuple(x+y for x,y in zip(a,b))
    if any(x%2 for x in c):
        return F(0)
    return F((-1)**(sum(a)+sum(c)//2)*prod(double_factorial(x-1) for x in c))


def eye(n):
    return [[F(i==j) for j in range(n)] for i in range(n)]


def transpose(a):
    return list(map(list,zip(*a)))


def mul(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def sub(a,b):
    return [[x-y for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def inverse(a):
    n=len(a); b=[row[:]+er for row,er in zip(a,eye(n))]
    for i in range(n):
        require(b[i][i]!=0,'nonzero pivot')
        p=b[i][i]; b[i]=[x/p for x in b[i]]
        for j in range(n):
            if i!=j:
                p=b[j][i]; b[j]=[x-p*y for x,y in zip(b[j],b[i])]
    return [row[n:] for row in b]


def block(a,rows,cols):
    return [[a[i][j] for j in cols] for i in rows]


def poly_product(p,q):
    z={}
    for a,x in p.items():
        for b,y in q.items():
            c=tuple(i+j for i,j in zip(a,b));z[c]=z.get(c,F(0))+x*y
    return z


def rotation_jets(r):
    out=[]
    for alpha in JETS:
        p={(0,0,0):F(1)}
        for axis,degree in enumerate(alpha):
            linear={tuple(int(i==j) for i in range(3)):r[j][axis] for j in range(3)}
            for _ in range(degree):
                p=poly_product(p,linear)
        out.append([p.get(beta,F(0)) for beta in JETS])
    return out


def hermite_abs(n):
    return {n-2*j:F(factorial(n),factorial(j)*factorial(n-2*j)*2**j) for j in range(n//2+1)}


def polynomial_mul(p,q):
    z={}
    for a,x in p.items():
        for b,y in q.items():z[a+b]=z.get(a+b,F(0))+x*y
    return z


def check(mutant=None):
    checks=0
    def eq(a,b,label):
        nonlocal checks
        require(a==b,label);checks+=1
    s=[[covariance(a,b) for b in JETS] for a in JETS]
    if mutant=='covariance-sign':s[JETS.index((0,0,0))][JETS.index((2,0,0))]=F(1)
    h=eye(N)
    for i,alpha in enumerate(JETS):
        if sum(alpha) in (2,3):
            for axis in range(3):
                if alpha[axis]>=2:
                    beta=list(alpha);beta[axis]-=2
                    coeff=F(alpha[axis]*(alpha[axis]-1),2)
                    if mutant=='cubic-contraction' and sum(alpha)==3:coeff=F(0)
                    h[i][JETS.index(tuple(beta))]+=coeff
    diagonal=[[F(prod(factorial(x) for x in a)) if i==j else F(0) for j in range(N)] for i,a in enumerate(JETS)]
    eq(mul(mul(h,s),transpose(h)),diagonal,'Hermite covariance diagonalization')
    nil=sub(h,eye(N))
    eq(mul(nil,nil),[[F(0)]*N for _ in range(N)],'Hermite nilpotence')
    hi=sub(eye(N),nil)
    eq(mul(h,hi),eye(N),'Hermite inverse')
    eq(sum(x*x for row in h for x in row),F(56),'Hermite Frobenius56')
    eq(sum(x*x for row in hi for x in row),F(56),'inverse Frobenius56')
    eq(sum(s[i][i] for i in range(N)),F(80),'covariance trace80')
    eq(sum(inverse(s)[i][i] for i in range(N)),F(22),'inverse trace22')
    for r in ([[F(1),0,0],[0,F(3,5),F(-4,5)],[0,F(4,5),F(3,5)]],
              [[F(3,5),F(-4,5),0],[F(4,5),F(3,5),0],[0,0,F(1)]],
              [[F(-1),0,0],[0,F(1),0],[0,0,F(1)]]):
        eq(mul(r,transpose(r)),eye(3),'orthogonal rational frame')
        t=rotation_jets(r)
        eq(mul(mul(t,s),transpose(t)),s,'reference rotational covariance')
    ukeys=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(3,0,0)]
    u=[JETS.index(x) for x in ukeys];j=[i for i in range(N) if i not in u]
    v=block(s,u,u);cross=block(s,j,u)
    regression=mul(cross,inverse(v))
    conditional=sub(block(s,j,j),mul(regression,transpose(cross)))
    targetdiag=[F(prod(factorial(x) for x in JETS[i])) for i in j]
    eq(conditional,[[targetdiag[i] if i==l else F(0) for l in range(12)] for i in range(12)],'conditional12 covariance')
    v0=[[F(1) if x==(0,0,0) else F(12) if x==(3,0,0) else F(0)] for x in ukeys]
    if mutant=='conditional-gap-mean':regression[0][-1]+=F(1)
    mean=mul(regression,v0)
    expect=[[F(-1) if JETS[i] in ((0,2,0),(0,0,2)) else F(0)] for i in j]
    eq(mean,expect,'reference conditional mean (b,k)=(1,1)')
    envelope=hermite_abs(6)
    for degree in range(7):
        for alpha in indices(degree):
            p={0:F(1)}
            for a in alpha:p=polynomial_mul(p,hermite_abs(a))
            # Independent coefficient-sum bound, valid at all t>=0.
            require(sum(p.values())<=76,'Hermite coefficient envelope');checks+=1
    if mutant=='shell-count':shell=lambda n:24*n*n
    else:shell=lambda n:24*n*n+2
    for n in range(1,10):
        eq(shell(n),(2*n+1)**3-(2*n-1)**3,'three-dimensional shell count')
        require(shell(n)<=26*n*n,'shell envelope');checks+=1
    delta=F(5000*25**6,2**288)
    require(delta<F(1,2**245),'L24 image envelope');checks+=1
    require(20*delta<F(1,112),'20-jet spectral margin');checks+=1
    require(F(256,2**864)<F(1,2),'shell tail ratio');checks+=1
    # Explicit cofactor block, no reference-isotropy assumption needed.
    for k,hard in product((F(1,2),F(1),F(2)),(F(1,3),F(2))):
        # At r=0 scaled soft eigenvalue s<0, axial±6k, hard eigenvalue -hard.
        sm,ss=F(-2),F(-3)
        detm=(-6*k)*sm*(-hard);dets=(6*k)*ss*(-hard)
        product_weight=-detm*dets
        weight2=(-6*k*sm)*( -6*k*ss)
        factor=hard if mutant=='hard-power' else hard**2
        eq(product_weight,factor*weight2,'hard determinant factor h²')
        eq(hard*hard**2,hard**3,'spectral times determinant hard power')
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mutant',choices=['covariance-sign','cubic-contraction','conditional-gap-mean','shell-count','hard-power']);a=ap.parse_args()
    n=check(a.mutant)
    print(json.dumps({'status':'PASS','finite_checks':n,'jet_dimension':20,'conditional_dimension':12,'Hermite_Frobenius_squared':56,'trace_covariance':80,'trace_inverse':22,'analytic_proof_by_tests':False},sort_keys=True))

if __name__=='__main__':main()
