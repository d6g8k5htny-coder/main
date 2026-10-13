import numpy as np
from scipy import integrate
rng=np.random.default_rng(0)
# (1) T3/T4 consistency: nu_F(t)=E[A^-1 nu_H(t/A)], m_F(t)=E[m_H(t/A)] and m_F = int_0^t nu_F
def nu_H(u): return u**(-1/3)*np.exp(-u**2)   # toy canonical density, g(u)=exp(-u^2)
def m_H(t): return integrate.quad(nu_H,0,t)[0]
As=rng.lognormal(0,1.0,200000)
for t in [0.1,1.0,3.0]:
    nuF=lambda u: np.mean(nu_H(u/As)/As)
    mF=np.mean([m_H(t/a) for a in As[:3000]])
    mF2=integrate.quad(nuF,0,t,limit=200)[0]
    print("t",t,"m_F via m_H:",mF,"m_F via int nu_F:",mF2)
# (2) P5: sup_a phi(a) finite, phi(a)=a^-N (1+(|b|+k)/a)^{2d} exp(-c(b^2+k^2)/a^2)
d=2;N=2;c=0.5;b=0.3;k=0.05
a=np.logspace(-4,4,200001)
phi=a**(-N)*(1+(abs(b)+k)/a)**(2*d)*np.exp(-c*(b*b+k*k)/a**2)
print("sup phi =",phi.max(),"at a=",a[phi.argmax()],"phi(1e-4)=",phi[0],"phi(1e4)=",phi[-1])
# (3) P4(c) slip: sup_u u^{1/3} * far bound vs sup_u far bound (G_far as written)
Lam=5.0;Cfar=1.0
u=np.linspace(1e-6,40,400001)
far=Cfar*(1+u)**(2*d)*np.exp(-u**2/(4*Lam))
print("G_far:=sup far bound =",far.max()," sup u^{1/3} far bound =",(u**(1/3)*far).max())
