"""Decimal regression falsification check; analytic argument is in the report."""
from decimal import Decimal as D, getcontext
getcontext().prec = 85
L = D(4)
def hermite(n,x):
    a,b=D(1),x
    if n==0:return a
    for j in range(1,n):a,b=b,x*b-j*a
    return b
norm=sum((-(L*j)**2/2).exp() for j in range(-12,13))
def kd(n,z):
    return sum((-1)**n*hermite(n,z+L*j)*(-(z+L*j)**2/2).exp() for j in range(-12,13))/norm
def cov(p,q):
    x,alpha=p;y,beta=q
    return (-1)**sum(beta)*kd(alpha[0]+beta[0],x-y)*kd(alpha[1]+beta[1],D(0))
def solve(a,b):
    a=[row[:]+[v] for row,v in zip(a,b)];n=len(b)
    for i in range(n):
        j=max(range(i,n),key=lambda j:abs(a[j][i]));a[i],a[j]=a[j],a[i]
        v=a[i][i];a[i]=[x/v for x in a[i]]
        for j in range(n):
            if j!=i:
                v=a[j][i];a[j]=[x-v*y for x,y in zip(a[j],a[i])]
    return [row[-1] for row in a]
a=-kd(2,D(0));sig2=kd(4,D(0))-a*a
print('a=',a,'sigma2=',sig2,'limit r^2 E_Q W=',18*sig2)
for r in map(D,['.1','.03','.01','.003']):
    pins=[(x,alpha) for x in [-r/2,r/2] for alpha in [(0,0),(1,0),(0,1)]]
    mat=[[cov(p,q) for q in pins] for p in pins]
    v=[D(0),D(0),D(0),-r,D(0),D(0)]
    coeff=solve(mat,v)
    hs=[(x,alpha) for x in [-r/2,r/2] for alpha in [(2,0),(1,1),(0,2)]]
    means=[sum(cov(h,p)*z for p,z in zip(pins,coeff)) for h in hs]
    am=[cov(hs[2],p) for p in pins]
    va=cov(hs[2],hs[2])-sum(x*y for x,y in zip(am,solve(mat,am)))
    print('r=',r,'r Hxx means=',float(r*means[0]),float(r*means[3]),'A means=',float(means[2]),float(means[5]),'Var A_M=',float(va),'cap necessary lambda>',float(192/r),'(r/k)^3=',r**9)
