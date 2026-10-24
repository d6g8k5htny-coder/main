"""Independent check of gauss_proc zero/lobe accuracy using direct trig sums (one sample)."""
import numpy as np, gauss_proc as gp
seed=10_000; M=1000; Ng=2**19
rng=np.random.default_rng(seed); kmax=Ng//2; w=1.0/M
k=np.arange(kmax+1); q=np.exp(-0.5*(w*k)**2); q/=q[0]+2*q[1:].sum()
xi=rng.standard_normal(kmax+1); eta=rng.standard_normal(kmax+1)
K=12000
def g(x):
    ph=np.outer(x, w*k[1:K]); return np.sqrt(q[0])*xi[0]+np.cos(ph)@(np.sqrt(2*q[1:K])*xi[1:K])+np.sin(ph)@(np.sqrt(2*q[1:K])*eta[1:K])
area,gap,peak,info=gp.one_sample(seed)
# reconstruct zeros: cumulative unfolded gaps -> need absolute positions; recompute here
L=2*np.pi*M; dx=L/Ng; x=dx*np.arange(Ng)
G=np.zeros(kmax+1,complex); G[0]=Ng*np.sqrt(q[0])*xi[0]; G[1:]=(Ng/2)*np.sqrt(2*q[1:])*(xi[1:]-1j*eta[1:]); G[-1]=0
gg=np.fft.irfft(G,n=Ng); s=np.sign(gg); i=np.nonzero(s[:-1]*s[1:]<0)[0]
# bisection directly on trig sum for first 40 zeros
a=x[i[:41]]; b=x[i[:41]+1]
fa=g(a)
for _ in range(45):
    m=0.5*(a+b); fm=g(m); same=np.sign(fm)==np.sign(fa); a=np.where(same,m,a); fa=np.where(same,fm,fa); b=np.where(same,b,m)
z=0.5*(a+b)
X,W=np.polynomial.legendre.leggauss(40)
ar=[(0.5*(z[j+1]-z[j]))*(g(0.5*(z[j]+z[j+1])+0.5*(z[j+1]-z[j])*X)@W) for j in range(40)]
print("max rel err of lobe areas vs direct trig-sum quadrature:", np.max(np.abs(np.array(ar)-area[:40])/np.abs(ar)))
print("max abs err of unfolded gaps:", np.max(np.abs(np.diff(z)/np.pi-gap[:40])))
