from decimal import Decimal as D, getcontext
import math, sys
getcontext().prec = 160
def He(n,x):
    a,b = D(1), x
    if n==0: return a
    for k in range(2,n+1): a,b = b, x*b-(k-1)*a
    return b
def expo(x): return x.exp()
MULTI=[(a,b) for s in range(4) for a in range(s,-1,-1) for b in [s-a]]  # 10 multi-indices |beta|<=3
assert len(MULTI)==10
def cov(points):
    n=10*len(points); S=[[D(0)]*n for _ in range(n)]
    for i,(xi,yi) in enumerate(points):
        for j,(xj,yj) in enumerate(points):
            h1,h2 = xi-xj, yi-yj
            g = expo(-(h1*h1+h2*h2)/2)
            for bi,(b1,b2) in enumerate(MULTI):
                for gi,(c1,c2) in enumerate(MULTI):
                    v = He(b1+c1,h1)*He(b2+c2,h2)*g
                    if (b1+b2)%2: v=-v
                    S[10*i+bi][10*j+gi]=v
    return S
def inv_trace(S):
    n=len(S); L=[[D(0)]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=S[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j:
                if s<=0: raise ValueError("not PD")
                L[i][i]=s.sqrt()
            else: L[i][j]=s/L[j][j]
    # trace(S^-1) = ||L^-1||_F^2 ; solve L X = I column by column
    tot=D(0)
    for c in range(n):
        x=[D(0)]*n
        for i in range(c,n):
            s=(D(1) if i==c else D(0))-sum(L[i][k]*x[k] for k in range(c,i))
            x[i]=s/L[i][i]
        tot+=sum(v*v for v in x[c:])
    return tot
def lam_bounds(points):
    tr=inv_trace(cov(points)); n=10*len(points)
    return 1/tr, n/tr
def fit(name, cfg, etas, pred):
    rows=[]
    for e in etas:
        lo,hi=lam_bounds(cfg(D(e)))
        rows.append((e,lo,hi))
    print(f"== {name} (predicted floor exponent {pred})")
    for k,(e,lo,hi) in enumerate(rows):
        s=""
        if k:
            e0,lo0,_=rows[k-1]
            slope=(math.log(float(lo))-math.log(float(lo0)))/(math.log(e)-math.log(e0))
            s=f"  local exponent of 1/tr ~ {slope:.3f}"
        print(f"   eta={e:<9} lambda_min in [{float(lo):.3e}, {float(hi):.3e}]  lambda_min/eta^{pred} >= {float(lo)/e**pred:.3e}{s}")
etas=[0.5,0.25,0.125,0.0625,0.03125]
fit("p=2 pair (0,0),(eta,0)", lambda e:[(D(0),D(0)),(e,D(0))], etas, 14)
fit("p=3 equilateral side eta", lambda e:[(D(0),D(0)),(e,D(0)),(e/2,e*D(3).sqrt()/2)], etas[:4], 22)
fit("p=3 collinear spacing eta", lambda e:[(D(0),D(0)),(e,D(0)),(2*e,D(0))], etas[:4], 22)
for Dist in ["3","10","40"]:
    fit(f"p=3 close pair + far third at distance {Dist}", lambda e,Dist=Dist:[(D(0),D(0)),(e,D(0)),(D(Dist),D(0))], etas, 22)
