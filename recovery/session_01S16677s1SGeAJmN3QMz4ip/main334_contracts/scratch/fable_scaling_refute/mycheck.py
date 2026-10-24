import numpy as np
from scipy import integrate
np.random.seed(1)
# 1. E[A^{-2/3}] closed form and integral ratio with toy B0 (d=2 shape)
B0=lambda b,k:(1+abs(b)+k)**4*np.exp(-(b*b+k*k))
IH=integrate.dblquad(lambda b,k:k**(-2/3)*B0(b,k),0,np.inf,-np.inf,np.inf)[0]
for alpha in [0.7,0.8,1.0,1.5,3.0]:
    a=np.sqrt((alpha+2)/alpha)
    pA=lambda s: alpha*a**(-alpha)*s**(alpha-1)
    EAm23=integrate.quad(lambda s:pA(s)*s**(-2/3),0,a)[0]
    pred=a**(-2/3)*alpha/(alpha-2/3)
    BF=lambda b,k: integrate.quad(lambda s:pA(s)*s**(-2)*B0(b/s,k/s),0,a,limit=300)[0]
    IF=integrate.dblquad(lambda b,k:k**(-2/3)*BF(b,k),0,10,-10,10,epsabs=1e-9)[0]
    print(f"alpha={alpha}: E A^-2/3 num={EAm23:.7f} closed={pred:.7f} ratio IF/IH={IF/IH:.7f}")

# 2. mixture density identity for a 4-dim Gaussian row vector (d=1): pi^F(v)=E[A^-4 pi^H(v/A)], Monte Carlo check
from scipy.stats import multivariate_normal
Sig=np.array([[1,.3,.1,0],[.3,2,.2,.1],[.1,.2,1.5,.4],[0,.1,.4,3.0]])
alpha=1.5; a=np.sqrt((alpha+2)/alpha)
pH=lambda w: multivariate_normal(mean=np.zeros(4),cov=Sig).pdf(w)
v=np.array([0.5,-0.2,0.0,1.2])
piF=integrate.quad(lambda s: alpha*a**(-alpha)*s**(alpha-1)*s**(-4)*pH(v/s),0,a,limit=300)[0]
# MC estimate of density of A*U at v via kernel in small ball
N=4000000
U=np.random.multivariate_normal(np.zeros(4),Sig,size=N)
A=a*np.random.rand(N)**(1/alpha)
X=A[:,None]*U
h=0.25
inball=np.sum(np.all(np.abs(X-v)<h,axis=1))
print("mixture density",piF," MC box estimate",inball/N/(2*h)**4)

# 3. far density behaviour: nu_far^H = 1{s<=1} (bounded, mass 1)
for alpha in [0.5,2/3,0.8,1.0,1.5]:
    a=np.sqrt((alpha+2)/alpha)
    def nuF(l):
        lo=l/a
        if lo>=1: return 0.0
        return alpha*a**(-alpha)*l**(alpha-1)*integrate.quad(lambda s:s**(-alpha),lo,1)[0]
    print(alpha,[(l, round(nuF(l),3), round(l**(1/3)*nuF(l),4)) for l in [1e-2,1e-4,1e-6,1e-8]])
