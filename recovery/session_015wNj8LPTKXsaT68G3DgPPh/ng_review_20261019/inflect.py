# Cubic-model diagnostic on the same fields as simulate_small.py (same spectra and synthesis).
# At every inflection point m (F''(m)=0) with F'(m)F'''(m)<0, the cubic jet b+d s+T s^3/6 has a max/min pair of height
#   ell_cubic = (2/3) 2^(3/2) |d|^(3/2) |T|^(-1/2).
# Rice at inflection points gives E #{ell_cubic<=t} = V int |T| p_(F',F'',F''')(d,0,T) 1{ell_cubic<=t} dd dT
#   ~ V (3/2) 12^(-1/3) kappa t^(2/3)  as t->0,
# with corrections only from the d-dependence of the jet density (no Taylor remainder beyond the cubic).
# So this count isolates the jet-law part of formula (6) from the finite-t bar geometry.
import numpy as np, sys, json, time
N=25; n=np.arange(N+1)
def spec(scale):
    w=np.exp(-n**2/(2*scale**2)); w/=w.sum(); return w
wG=spec(5.0); wH=spec(4.0)
eps=float(sys.argv[1]); R=int(sys.argv[2]); seed=int(sys.argv[3]); J=int(sys.argv[4]) if len(sys.argv)>4 else 4096
ts=np.array([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2])
rng=np.random.default_rng(seed)
x=2*np.pi*np.arange(J)/J; CS=np.cos(np.outer(n,x)); SN=np.sin(np.outer(n,x))
def trig(xr):
    nx=xr[:,None]*n[None,:]; return np.cos(nx),np.sin(nx)
def series(ac,as_,cs,deriv):
    c,s=cs; f=n[None,:]**deriv; k=deriv%4
    v={0:ac*c+as_*s,1:-ac*s+as_*c,2:-ac*c-as_*s,3:ac*s-as_*c}[k]
    return (f*v).sum(axis=1)
def F2(GC,GS,HC,HS,xr):
    cs=trig(xr); h0,h1,h2=series(HC,HS,cs,0),series(HC,HS,cs,1),series(HC,HS,cs,2)
    return series(GC,GS,cs,2)+2*eps*(h1*h1+h0*h2)
counts=np.zeros(len(ts),dtype=np.int64); ninfl=0; nreal=0; t0=time.time()
while nreal<R:
    b=min(1000,R-nreal)
    zc=rng.standard_normal((b,N+1)); zs=rng.standard_normal((b,N+1)); zs[:,0]=0
    gc,gs=zc*np.sqrt(wG),zs*np.sqrt(wG)
    yc=rng.standard_normal((b,N+1)); ys=rng.standard_normal((b,N+1)); ys[:,0]=0
    hc,hs=yc*np.sqrt(wH),ys*np.sqrt(wH)
    Hg=hc@CS+hs@SN; H1g=(-hc*n)@SN+(hs*n)@CS; H2g=(-hc*n**2)@CS+(-hs*n**2)@SN; G2g=(-gc*n**2)@CS+(-gs*n**2)@SN
    Fpp=G2g+2*eps*(H1g**2+Hg*H2g)
    sg=np.sign(Fpp); rows,cols=np.nonzero(sg!=np.roll(sg,-1,axis=1))
    GC,GS,HC,HS=gc[rows],gs[rows],hc[rows],hs[rows]
    lo=x[cols].copy(); hi=lo+2*np.pi/J; flo=Fpp[rows,cols].copy()
    for _ in range(30):
        mid=(lo+hi)/2; fm=F2(GC,GS,HC,HS,mid)
        left=(np.sign(fm)==np.sign(flo))
        lo=np.where(left,mid,lo); flo=np.where(left,fm,flo); hi=np.where(left,hi,mid)
    xr=(lo+hi)/2
    cs=trig(xr); h0,h1,h2,h3=(series(HC,HS,cs,k) for k in range(4))
    d=series(GC,GS,cs,1)+2*eps*h0*h1
    T=series(GC,GS,cs,3)+2*eps*(3*h1*h2+h0*h3)
    ok=(d*T<0)
    ell=(2/3)*2**1.5*np.abs(d[ok])**1.5/np.sqrt(np.abs(T[ok]))
    counts+=(ell[:,None]<=ts[None,:]).sum(axis=0)
    ninfl+=len(xr); nreal+=b
print(json.dumps(dict(model='cubic-jet at inflection points',eps=eps,realisations=nreal,J=J,seed=seed,ts=ts.tolist(),
      counts=counts.tolist(),inflection_points=ninfl,rate_per_t23=(counts/nreal/ts**(2/3)).tolist(),seconds=round(time.time()-t0,1))))
