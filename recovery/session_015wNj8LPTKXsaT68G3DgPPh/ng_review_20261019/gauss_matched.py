# Gaussian field with exactly the covariance of F=G+eps(H^2-1): K_F = K_G + 2 eps^2 K_H^2. Simulate its short H0 bars and
# compare with its own exact stationary-Gaussian prediction.
import numpy as np, sys, json, time, mpmath as mp
N=25; n=np.arange(N+1)
def spec(s):
    w=np.exp(-n**2/(2*s**2)); w/=w.sum(); return w
wG,wH=spec(5.0),spec(4.0); eps=float(sys.argv[1]); R=int(sys.argv[2]); seed=int(sys.argv[3]); J=4096
# cosine-series weights of K_H(tau)^2: cos(m t)cos(n t) = [cos((m+n)t)+cos((m-n)t)]/2
K=2*N; u=np.zeros(K+1)
for m in range(N+1):
    for q in range(N+1):
        u[m+q]+=wH[m]*wH[q]/2; u[abs(m-q)]+=wH[m]*wH[q]/2
wF=np.zeros(K+1); wF[:N+1]+=wG; wF+=2*eps**2*u
nn=np.arange(K+1); lam=lambda k: float((wF*nn**(2*k)).sum())
L2,L4,L6=lam(1),lam(2),lam(3)
S=np.sqrt(L6-L4**2/L2); absm=float(S**(4/3)*2**(2/3)*mp.gamma(7/6)/mp.sqrt(mp.pi))
kappa=absm/(2*np.pi*np.sqrt(L2*L4)); pred=2*np.pi*1.5*12**(-1/3)*kappa
ts=np.array([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]); rng=np.random.default_rng(seed)
x=2*np.pi*np.arange(J)/J; CS=np.cos(np.outer(nn,x)); SN=np.sin(np.outer(nn,x))
def ser(ac,as_,xr,d):
    nx=xr[:,None]*nn[None,:]; c,s=np.cos(nx),np.sin(nx); k=d%4
    v={0:ac*c+as_*s,1:-ac*s+as_*c,2:-ac*c-as_*s,3:ac*s-as_*c}[k]
    return ((nn[None,:]**d)*v).sum(axis=1)
counts=np.zeros(len(ts),dtype=np.int64); nreal=0; skipped=0; t0=time.time()
while nreal<R:
    b=min(1000,R-nreal)
    zc=rng.standard_normal((b,K+1)); zs=rng.standard_normal((b,K+1)); zs[:,0]=0
    ac,as_=zc*np.sqrt(wF),zs*np.sqrt(wF)
    F1=(-ac*nn)@SN+(as_*nn)@CS
    sg=np.sign(F1); rows,cols=np.nonzero(sg!=np.roll(sg,-1,axis=1))
    A,Bs=ac[rows],as_[rows]
    lo=x[cols].copy(); hi=lo+2*np.pi/J; flo=F1[rows,cols].copy()
    for _ in range(30):
        mid=(lo+hi)/2; fm=ser(A,Bs,mid,1); left=np.sign(fm)==np.sign(flo)
        lo=np.where(left,mid,lo); flo=np.where(left,fm,flo); hi=np.where(left,hi,mid)
    xr=(lo+hi)/2; Fv=ser(A,Bs,xr,0)
    o=np.lexsort((xr,rows)); rows,Fv=rows[o],Fv[o]
    st=np.searchsorted(rows,np.arange(b)); en=np.searchsorted(rows,np.arange(b),side='right')
    for i in range(b):
        v=Fv[st[i]:en[i]]; m=len(v)
        if m<2: continue
        ismax=(v>np.roll(v,1))&(v>np.roll(v,-1))
        if ismax.sum()*2!=m: skipped+=1; continue
        par={j:j for j in np.nonzero(ismax)[0]}; top={j:v[j] for j in par}
        def find(a):
            while par[a]!=a: par[a]=par[par[a]]; a=par[a]
            return a
        mins=sorted([j for j in range(m) if not ismax[j]],key=lambda j:-v[j])
        for j in mins[:-1]:
            a,c=find((j-1)%m),find((j+1)%m)
            if a==c: continue
            if top[a]<top[c]: a,c=c,a
            counts+=(top[c]-v[j]<=ts); par[c]=a
    nreal+=b
print(json.dumps(dict(model='cov-matched Gaussian',eps=eps,R=nreal,seed=seed,spectral=(L2,L4,L6),pred=pred,ts=ts.tolist(),
      counts=counts.tolist(),rate=(counts/nreal/ts**(2/3)).tolist(),skipped=skipped,seconds=round(time.time()-t0,1))))
