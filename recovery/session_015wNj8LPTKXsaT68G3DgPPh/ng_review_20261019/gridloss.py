# Expected fraction of N((0,t]) lost by the grid detector in simulate*.py / gauss_matched.py.
# A max/min pair is missed when both roots of F' fall in one grid cell (no sign change on the grid).
# Cubic pair: ell = |T| delta^3/12, delta = root separation. With cell width h = 2 pi/J and a uniform offset,
# P(miss | delta) = max(0, 1 - delta/h). Under the leading law the lifetime density is ~ ell^(-1/3), i.e. the
# separation density is ~ delta on [0, delta_t], delta_t = (12 t/|T|)^(1/3). With r = ell_h/t, ell_h = |T| h^3/12:
#   g(r) = r^(2/3)/3            (r <= 1)
#   g(r) = 1 - (2/3) r^(-1/3)   (r >  1)
# Loss(t) = E[w |T|^(4/3) g(r(T))] / E[w |T|^(4/3)], over the same literal mixture kernel as kappa.py
# (w = p_(G',G'')(-W',-W''); T = F''' given F'=F''=0), or over the exact Gaussian law for the matched model.
import numpy as np, json, sys
N=25; n=np.arange(N+1)
def spec(s):
    w=np.exp(-n**2/(2*s**2)); return w/w.sum()
wG,wH=spec(5.0),spec(4.0)
lam=lambda w,k: float((w*n**(2*k)).sum())
l2,l4,l6=lam(wG,1),lam(wG,2),lam(wG,3); m2,m4,m6=lam(wH,1),lam(wH,2),lam(wH,3)
s2=l6-l4**2/l2
ts=np.array([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2])
def g(r):
    return np.where(r<=1, r**(2/3)/3, 1-(2/3)*np.maximum(r,1e-300)**(-1/3))
def loss(T,w,J):
    h=2*np.pi/J; a=w*np.abs(T)**(4/3); out=[]
    for t in ts:
        r=np.abs(T)*h**3/12/t; out.append(float((a*g(r)).sum()/a.sum()))
    return out
rng=np.random.default_rng(777); M=4_000_000
C=np.array([[1,0,-m2,0],[0,m2,0,-m4],[-m2,0,m4,0],[0,-m4,0,m6]],float); Lc=np.linalg.cholesky(C)
res={}
for eps in (0.0,1.0,2.0,3.0):
    Z=rng.standard_normal((M,4))@Lc.T; H,H1,H2,H3=Z.T
    W1=2*eps*H*H1; W2=2*eps*(H1**2+H*H2); W3=2*eps*(3*H1*H2+H*H3)
    w=np.exp(-W1**2/(2*l2)-W2**2/(2*l4))
    T=W3+(l4/l2)*W1+np.sqrt(s2)*rng.standard_normal(M)
    for J in (4096,8192,16384):
        res['ng eps=%g J=%d'%(eps,J)]=loss(T,w,J)
# matched Gaussian: T ~ N(0, L6-L4^2/L2) exactly, weight constant
for eps in (1.0,3.0):
    L2=l2+4*eps**2*m2; L4=l4+eps**2*(12*m2**2+4*m4); L6=l6+4*eps**2*(15*m2*m4+m6)
    T=np.sqrt(L6-L4**2/L2)*rng.standard_normal(M)
    res['gm eps=%g J=4096'%eps]=loss(T,np.ones(M),4096)
print(json.dumps(dict(ts=ts.tolist(),loss=res),indent=1))
