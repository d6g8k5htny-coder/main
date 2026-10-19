# Monte Carlo count of short superlevel-H0 bars on the circle for F=G+eps(H^2-1) (same spectra as kappa.py).
# Critical points: sign changes of the exact F' on a grid, refined by Newton on the exact series; critical values from
# the exact series; exact circle H0 persistence (elder rule) on the cyclic extrema list.
import numpy as np, sys, json, time
N=25; n=np.arange(N+1)
def spec(scale):
    w=np.exp(-n**2/(2*scale**2)); w/=w.sum(); return w
wG=spec(5.0); wH=spec(4.0)
eps=float(sys.argv[1]); R=int(sys.argv[2]); seed=int(sys.argv[3]); J=int(sys.argv[4]) if len(sys.argv)>4 else 4096
ts=np.array([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2])
rng=np.random.default_rng(seed)
x=2*np.pi*np.arange(J)/J; CS=np.cos(np.outer(n,x)); SN=np.sin(np.outer(n,x))
counts=np.zeros(len(ts),dtype=np.int64); nbars=0; nreal=0; skipped=0; t0=time.time()
def series(ac,as_,xr,deriv):
    # value of sum_n ac_n cos(n x)+as_n sin(n x), derivative order deriv, at points xr (per-row coefficient arrays)
    ph=np.outer(np.ones(1),n)  # placeholder
    nx=xr[:,None]*n[None,:]
    c,s=np.cos(nx),np.sin(nx); f=n[None,:]**deriv
    k=deriv%4
    if k==0: v=ac*c+as_*s
    elif k==1: v=-ac*s+as_*c
    elif k==2: v=-ac*c-as_*s
    else: v=ac*s-as_*c
    return (f*v).sum(axis=1)
B=1000
while nreal<R:
    b=min(B,R-nreal)
    zc=rng.standard_normal((b,N+1)); zs=rng.standard_normal((b,N+1)); zs[:,0]=0
    gc,gs=zc*np.sqrt(wG),zs*np.sqrt(wG)
    yc=rng.standard_normal((b,N+1)); ys=rng.standard_normal((b,N+1)); ys[:,0]=0
    hc,hs=yc*np.sqrt(wH),ys*np.sqrt(wH)
    Hg=hc@CS+hs@SN; H1g=(-hc*n)@SN+(hs*n)@CS; G1g=(-gc*n)@SN+(gs*n)@CS
    F1=G1g+2*eps*Hg*H1g
    sgn=np.sign(F1); ch=(sgn!=np.roll(sgn,-1,axis=1))
    rows,cols=np.nonzero(ch)
    # linear interpolation start
    f0=F1[rows,cols]; f1=F1[rows,(cols+1)%J]
    xr=x[cols]+(2*np.pi/J)*f0/(f0-f1)
    GC,GS,HC,HS=gc[rows],gs[rows],hc[rows],hs[rows]
    # bracketed bisection on the exact series F' inside each grid cell (no Newton failures)
    lo=x[cols].copy(); hi=lo+2*np.pi/J; flo=f0.copy()
    for _ in range(30):
        mid=(lo+hi)/2
        fm=series(GC,GS,mid,1)+2*eps*series(HC,HS,mid,0)*series(HC,HS,mid,1)
        left=(np.sign(fm)==np.sign(flo))
        lo=np.where(left,mid,lo); flo=np.where(left,fm,flo); hi=np.where(left,hi,mid)
    xr=(lo+hi)/2
    h0=series(HC,HS,xr,0)
    Fv=series(GC,GS,xr,0)+eps*(h0*h0-1)
    # per realisation: extrema in angular order, then exact circle persistence
    order=np.lexsort((np.mod(xr,2*np.pi),rows))
    rows,Fv=rows[order],Fv[order]
    starts=np.searchsorted(rows,np.arange(b)); ends=np.searchsorted(rows,np.arange(b),side='right')
    for i in range(b):
        v=Fv[starts[i]:ends[i]]; m=len(v)
        if m<2: continue
        # alternate max/min: classify by neighbours
        ismax=(v>np.roll(v,1))&(v>np.roll(v,-1))
        if ismax.sum()*2!=m:
            skipped+=1; continue
        # union-find over maxima; minima processed in decreasing order merge their two neighbour maxima
        idx=np.arange(m); par={}; top={}
        for j in np.nonzero(ismax)[0]: par[j]=j; top[j]=v[j]
        def find(a):
            while par[a]!=a: par[a]=par[par[a]]; a=par[a]
            return a
        mins=[j for j in range(m) if not ismax[j]]
        mins.sort(key=lambda j:-v[j])
        for j in mins[:-1]:          # last (global) minimum closes the circle: no H0 death
            a,c=find((j-1)%m),find((j+1)%m)
            if a==c: continue
            if top[a]<top[c]: a,c=c,a
            life=top[c]-v[j]; nbars+=1
            counts+= (life<=ts)
            par[c]=a
    nreal+=b
print(json.dumps(dict(eps=eps,realisations=nreal,J=J,seed=seed,ts=ts.tolist(),counts=counts.tolist(),
      finite_bars=nbars,skipped_realisations=skipped,rate_per_t23=(counts/nreal/ts**(2/3)).tolist(),seconds=round(time.time()-t0,1))))
