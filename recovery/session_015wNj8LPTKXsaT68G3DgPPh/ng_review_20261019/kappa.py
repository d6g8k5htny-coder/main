# Predicted short-bar constant for d=1, p=0 on the circle X=R/(2 pi Z), V=2 pi, for F=G+W, W=eps(H^2-1).
# Formula (6) at d=1: C_0 = 12^(-1/3) * kappa, kappa = p_(F',F'')(0,0) E[|F'''|^(4/3) | F'=F''=0] (stationary),
# and E N((0,t]) ~ V (3/2) C_0 t^(2/3).  kappa is evaluated through the literal mixture kernel (9):
# condition on W first; given W, (G',G'',G''') is Gaussian.
import numpy as np, mpmath as mp, json, sys
N=25; n=np.arange(N+1)
def spec(scale):
    w=np.exp(-n**2/(2*scale**2)); w/=w.sum(); return w      # Var=1, covariance sum_n w_n cos(n tau)
wG=spec(5.0); wH=spec(4.0); eps=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
lam=lambda w,k: float((w*n**(2*k)).sum())
l2,l4,l6=lam(wG,1),lam(wG,2),lam(wG,3); m2,m4,m6=lam(wH,1),lam(wH,2),lam(wH,3)
s2=l6-l4**2/l2                                   # Var(G''' | G')
def absmom(m,s,p=4/3):                           # E|m+sZ|^p, exact via 1F1
    return float(s**p*2**(p/2)*mp.gamma((p+1)/2)/mp.sqrt(mp.pi)*mp.hyp1f1(-p/2,0.5,-(m*m)/(2*s*s)))
rng=np.random.default_rng(12345)
# joint Gaussian (H,H',H'',H''') at a point for stationary H
C=np.array([[1,0,-m2,0],[0,m2,0,-m4],[-m2,0,m4,0],[0,-m4,0,m6]],float)
Lc=np.linalg.cholesky(C)
M=2_000_000; Z=rng.standard_normal((M,4))@Lc.T; H,H1,H2,H3=Z.T
W1=2*eps*H*H1; W2=2*eps*(H1**2+H*H2); W3=2*eps*(3*H1*H2+H*H3)
phi=lambda y,v: np.exp(-y*y/(2*v))/np.sqrt(2*np.pi*v)
mean3=W3+(l4/l2)*W1                               # F''' given G'=-W', G''=-W'' : W''' + E[G'''|G'] + residual
# vectorised E|m+sZ|^{4/3}: tabulate on a grid of m/s and interpolate (1F1 is smooth)
grid=np.linspace(0,60,6001); tab=np.array([absmom(g,1.0) for g in grid])
q=np.interp(np.abs(mean3)/np.sqrt(s2),grid,tab)*s2**(2/3)
weights=phi(W1,l2)*phi(W2,l4)
vals=weights*q
kappa=vals.mean(); kerr=vals.std()/np.sqrt(M)
# covariance-matched stationary Gaussian with F's spectral moments (W stationary: Cov(F',F''')=-Var(F''))
L2=l2+np.var(W1); L4=l4+np.var(W2); L6=l6+np.var(W3)
kg=(1/(2*np.pi*np.sqrt(L2*L4)))*absmom(0,np.sqrt(L6-L4**2/L2))
V=2*np.pi; pref=V*1.5*12**(-1/3)
out=dict(eps=eps,lam=(l2,l4,l6),mu=(m2,m4,m6),kappa=kappa,kappa_se=kerr,kappa_cov_matched_gauss=kg,
         ratio=kappa/kg,pred_coeff=pref*kappa,pred_coeff_gauss=pref*kg,
         third_cumulant_F=float(np.mean((eps*(H**2-1))**3)))
print(json.dumps(out,indent=1))
