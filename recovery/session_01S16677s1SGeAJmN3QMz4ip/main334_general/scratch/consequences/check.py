import math
def quad(f,a,b,n=200000):
    # composite Simpson on log-spaced grid for singular endpoints: substitute u = a*(b/a)^x? use plain midpoint on log scale
    la,lb=math.log(a),math.log(b); h=(lb-la)/n; s=0.0
    for i in range(n):
        x=la+(i+0.5)*h; u=math.exp(x); s+=f(u)*u*h
    return s
# (1) E[A^(-2/3)] for AMP's power law vs a^(-2/3) alpha/(alpha-2/3)
for alpha in [0.7,1.0,1.5,3.0]:
    a=math.sqrt((alpha+2)/alpha)
    dens=lambda u: alpha*a**(-alpha)*u**(alpha-1)
    num=quad(lambda u: dens(u)*u**(-2/3),1e-12,a)
    print("alpha",alpha,"num",num,"formula",a**(-2/3)*alpha/(alpha-2/3),"mass",quad(dens,1e-12,a),"EA2",quad(lambda u:dens(u)*u*u,1e-12,a))
# (2) toy: nu_H(u)=C u^(-1/3) exp(-u), C=1; A ~ power law alpha=0.8 (m<inf): check t^(1/3) nu_F(t) -> m*C and t^(-2/3) m_F(t) -> 1.5 m C
alpha=0.8; a=math.sqrt((alpha+2)/alpha); m=a**(-2/3)*alpha/(alpha-2/3)
nuH=lambda u: u**(-1/3)*math.exp(-u)
# m_H(u)=∫_0^u s^(-1/3)e^{-s} ds  (lower incomplete gamma(2/3,u))
def mH(u): return quad(lambda s: nuH(s),1e-14,u,20000)
for t in [1e-2,1e-4,1e-6,1e-8]:
    nuF=quad(lambda A: alpha*a**(-alpha)*A**(alpha-1)*A**(-1)*nuH(t/A),1e-14,a)
    mF=quad(lambda A: alpha*a**(-alpha)*A**(alpha-1)*mH(t/A),1e-14,a,4000)
    print("t",t,"t^(1/3)nuF",t**(1/3)*nuF,"target",m,"| t^(-2/3)mF",t**(-2/3)*mF,"target",1.5*m)
