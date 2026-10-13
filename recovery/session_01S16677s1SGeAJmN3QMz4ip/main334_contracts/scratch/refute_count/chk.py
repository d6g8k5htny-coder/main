import numpy as np
rng=np.random.default_rng(0)
# toy: d=1, two sites. "gradient" g=(g1,g2)~N(0,I2) (2d=2 dims), "Hessian" h=(h1,h2)~N(0,I2) indep, Delta=|h1 h2|.
# F = A*(g,h), A=a R, R~ density alpha r^(alpha-1) on (0,1), a=sqrt((alpha+2)/alpha).
def EAneg(alpha,m):
    a=np.sqrt((alpha+2)/alpha)
    # E A^-m = alpha a^-alpha int_0^a s^(alpha-1-m) ds
    return np.inf if alpha<=m else alpha*a**(-alpha)*a**(alpha-m)/(alpha-m)
for alpha in [0.5,2/3,1,1.5,2,2.5,3,5]:
    print(alpha, "E A^-2 (2d, d=1):",EAneg(alpha,2), " E A^-4 (2d+2):",EAneg(alpha,4))
# check (3) scaling with Rice-like identity in toy: E sum_{g=0} Delta w  ~ p_gg(0) E[Delta w| g=0]; here we just check the
# density of A*g at 0 is E[A^-2] p_g(0): numerically estimate P(|A g|<eps)/(pi eps^2) for alpha=3 vs 1.5
alpha=3; a=np.sqrt((alpha+2)/alpha); N=4_000_000
A=a*rng.random(N)**(1/alpha); g=rng.standard_normal((N,2))
for eps in [0.2,0.1,0.05]:
    r=np.hypot(*(A[:,None]*g).T); print("alpha=3 eps",eps, (r<eps).mean()/(np.pi*eps**2), "pred",EAneg(3,2)/(2*np.pi))
alpha=1.5; a=np.sqrt((alpha+2)/alpha)
A=a*rng.random(N)**(1/alpha)
for eps in [0.2,0.1,0.05,0.02]:
    r=np.hypot(*(A[:,None]*g).T); print("alpha=1.5 eps",eps, (r<eps).mean()/(np.pi*eps**2), "pred inf")
# mixture normalizability at (b,s)!=(0,0): weight(a')=p_A(a') a'^-(2d+2) p^H_O((b,0,s,0)/a') with p^H_O standard Gaussian in R^4
from scipy import integrate
def w(ap,alpha,b,s):
    a=np.sqrt((alpha+2)/alpha); d=1
    return alpha*a**(-alpha)*ap**(alpha-1)*ap**(-(2*d+2))*(2*np.pi)**(-2)*np.exp(-(b*b+s*s)/(2*ap*ap))
for alpha in [0.3,1,3]:
    for (b,s) in [(0.0,-0.01),(1.0,0.99),(0.0,0.0)]:
        a=np.sqrt((alpha+2)/alpha)
        val,err=integrate.quad(w,0,a,args=(alpha,b,s),limit=500)
        print("alpha",alpha,"(b,s)",(b,s),"p^F_O=",val)
