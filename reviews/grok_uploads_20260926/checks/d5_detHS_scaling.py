"""NON-CERTIFYING Monte Carlo check of the D5 on-axis cone determinant powers.

Scope: planar Bargmann-Fock field, E[f(x)f(y)] = exp(-|x-y|^2/2); six pins
f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0 with M=(0,0), S=(r,0); plus the
Kac-Rice event grad f(X)=0 at X = M + r(0,q). Gaussian conditioning is exact
(mpmath, 60 digits); expectations are Monte Carlo (4e5 samples per point).

Purpose: test the claim in Math- PR #87 harper/D5_OBSTRUCTION_LEDGER.md that
det H_S = O(k r) (unslaved) and that the on-axis product is O(k^3 r^5 q^2).

Recorded output (q=0.25, b=k=1, seed 0), 2026-09-27:
    fitted slope of E|det H_S| in r: 1.917, 1.977, 1.993  (r = 0.4 -> 0.05)
    fitted slope of E|det H_M det H_X det H_S| in r: 5.874, 5.972, 5.994
    sd f_ss(S) given the conditioning: 0.540, 0.278, 0.140, 0.070 (~1.4 r)
i.e. det H_S = Theta(r^2) and the product = Theta(r^6 q^2) on this slice,
consistent with Math- issue #58, not with the ledger. Floating-point and Monte
Carlo output: NON-CERTIFYING. Requires numpy and mpmath.

Run: python3 d5_detHS_scaling.py
"""
import mpmath as mp, numpy as np
mp.mp.dps = 60
def He(n,x):
    return mp.hermite(n, x/mp.sqrt(2))*mp.power(2,-mp.mpf(n)/2)  # probabilists' Hermite
def cov(a,x,b,y):
    u=(x[0]-y[0], x[1]-y[1])
    return (-1)**(a[0]+a[1])*He(a[0]+b[0],u[0])*He(a[1]+b[1],u[1])*mp.e**(-(u[0]**2+u[1]**2)/2)
def run(r,q,b=1,k=1,N=400000,seed=0):
    r=mp.mpf(r);q=mp.mpf(q)
    M=(mp.mpf(0),mp.mpf(0)); S=(r,mp.mpf(0)); X=(mp.mpf(0),r*q)
    pins=[((0,0),M),((1,0),M),((0,1),M),((0,0),S),((1,0),S),((0,1),S),((1,0),X),((0,1),X)]
    vals=[b,0,0,b-k*r**3,0,0,0,0]
    tg=[(d,P) for P in (M,X,S) for d in ((2,0),(1,1),(0,2))]
    Kpp=mp.matrix(len(pins),len(pins)); Ktp=mp.matrix(len(tg),len(pins)); Ktt=mp.matrix(len(tg),len(tg))
    for i,(a,x) in enumerate(pins):
        for j,(c,y) in enumerate(pins): Kpp[i,j]=cov(a,x,c,y)
    for i,(a,x) in enumerate(tg):
        for j,(c,y) in enumerate(pins): Ktp[i,j]=cov(a,x,c,y)
        for j,(c,y) in enumerate(tg): Ktt[i,j]=cov(a,x,c,y)
    Ki=Kpp**-1
    mu=Ktp*Ki*mp.matrix(vals); Sig=Ktt-Ktp*Ki*Ktp.T
    mu=np.array([float(v) for v in mu]); Sig=np.array([[float(Sig[i,j]) for j in range(9)] for i in range(9)])
    w,V=np.linalg.eigh((Sig+Sig.T)/2); w=np.clip(w,0,None)
    rng=np.random.default_rng(seed); Z=rng.standard_normal((N,9))
    H=mu+ (Z*np.sqrt(w))@V.T
    det=lambda o: H[:,o]*H[:,o+2]-H[:,o+1]**2
    dM,dX,dS=det(0),det(3),det(6)
    return np.mean(abs(dM)),np.mean(abs(dX)),np.mean(abs(dS)),np.mean(abs(dM*dX*dS)),float(mu[6]),float(mu[8]),float(np.sqrt(Sig[8,8]))
print("q=0.25; columns: r E|detHM| E|detHX| E|detHS| E|prod| mean f_tt(S) mean f_ss(S) sd f_ss(S)")
rows=[]
for r in (0.4,0.2,0.1,0.05):
    res=run(r,0.25); rows.append((r,)+res); print(r, *["%.4e"%v for v in res])
import math
for j,name in ((3,'detHS'),(4,'prod')):
    for a,b2 in zip(rows,rows[1:]):
        print(name,"slope r=%.3f->%.3f: %.3f"%(a[0],b2[0], math.log(a[j]/b2[j])/math.log(a[0]/b2[0])))
