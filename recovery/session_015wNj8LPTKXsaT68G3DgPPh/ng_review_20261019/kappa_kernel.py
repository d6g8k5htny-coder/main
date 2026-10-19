# Independent check of kappa = p_(F',F'')(0,0) E[|F'''|^(4/3) | F'=F''=0] by kernel smoothing of joint jet samples
import numpy as np, json, sys
N=25; n=np.arange(N+1)
def spec(s):
    w=np.exp(-n**2/(2*s**2)); w/=w.sum(); return w
wG,wH=spec(5.0),spec(4.0); eps=float(sys.argv[1])
lam=lambda w,k: float((w*n**(2*k)).sum())
l2,l4,l6=lam(wG,1),lam(wG,2),lam(wG,3); m2,m4,m6=lam(wH,1),lam(wH,2),lam(wH,3)
CG=np.array([[l2,0,-l4],[0,l4,0],[-l4,0,l6]]); LG=np.linalg.cholesky(CG)
CH=np.array([[1,0,-m2,0],[0,m2,0,-m4],[-m2,0,m4,0],[0,-m4,0,m6]],float); LH=np.linalg.cholesky(CH)
rng=np.random.default_rng(2024)
hs=[0.02,0.04,0.08]; acc={h:[] for h in hs}
s1=np.sqrt(l2+4*eps**2*m2); s2=np.sqrt(l4+4*eps**2*(3*m2**2-2*m2**2+m4+2*m2**2))
for chunk in range(40):
    M=1_000_000
    g=rng.standard_normal((M,3))@LG.T; h=rng.standard_normal((M,4))@LH.T
    H,H1,H2,H3=h.T
    F1=g[:,0]+2*eps*H*H1; F2=g[:,1]+2*eps*(H1**2+H*H2); F3=g[:,2]+2*eps*(3*H1*H2+H*H3)
    a=np.abs(F3)**(4/3)
    for hh in hs:
        b1,b2=hh*s1,hh*s2
        k=np.exp(-F1**2/(2*b1**2)-F2**2/(2*b2**2))/(2*np.pi*b1*b2)
        acc[hh].append((a*k).mean())
res={str(hh):(float(np.mean(v)),float(np.std(v)/np.sqrt(len(v)))) for hh,v in acc.items()}
# quadratic-in-h extrapolation to h=0 from the three bandwidths
x=np.array(hs)**2; y=np.array([res[str(h)][0] for h in hs]); A=np.vstack([np.ones(3),x]).T
c0=np.linalg.lstsq(A,y,rcond=None)[0][0]
print(json.dumps(dict(eps=eps,kernel_estimates=res,extrapolated_h0=float(c0))))
