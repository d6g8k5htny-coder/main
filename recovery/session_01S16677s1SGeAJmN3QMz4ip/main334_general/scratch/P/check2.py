import math, random
random.seed(0)
# (1) T3/T4 consistency with a toy canonical density nu_H(u)=u^{-1/3} e^{-u^2}  (g(u)=e^{-u^2})
def nu_H(u): return u**(-1/3)*math.exp(-u*u)
def simpson(f,a,b,n=4000):
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+= (4 if i%2 else 2)*f(a+i*h)
    return s*h/3
def m_H(t):  # substitute u=w^{3/2} to remove the integrable singularity
    return simpson(lambda w: (3/2)*math.exp(-w**3), 0, t**(2/3))
As=[math.exp(random.gauss(0,1.0)) for _ in range(4000)]
for t in [0.1,1.0,3.0]:
    mF=sum(m_H(t/a) for a in As)/len(As)
    nuF=lambda u: sum(nu_H(u/a)/a for a in As)/len(As)
    mF2=simpson(lambda w: (3/2)*w**(1/2)*nuF(w**1.5),1e-9,t**(2/3),n=600)
    print("t",t," m_F=E[m_H(t/A)]:",round(mF,5),"  int_0^t nu_F:",round(mF2,5))
# (2) P5 majorant: sup_a phi(a) finite for k>0, with phi(a)=a^-N (1+(|b|+k)/a)^{2d} exp(-c(b^2+k^2)/a^2)
d=2;N=2;c=0.5;b=0.3;k=0.05
best=(0,None)
for i in range(0,200001):
    a=10**(-4+8*i/200000)
    phi=a**(-N)*(1+(abs(b)+k)/a)**(2*d)*math.exp(-c*(b*b+k*k)/a**2)
    if phi>best[0]: best=(phi,a)
print("sup phi =",best)
# (3) P4(c) notational slip: G_far:=sup nu_far-bound vs sup u^{1/3} * bound
Lam=5.0
sup_far=0;sup_g=0
for i in range(1,400001):
    u=40*i/400000
    far=(1+u)**(2*d)*math.exp(-u*u/(4*Lam))
    sup_far=max(sup_far,far); sup_g=max(sup_g,u**(1/3)*far)
print("sup far bound =",sup_far,"  sup u^{1/3} far bound =",sup_g)
