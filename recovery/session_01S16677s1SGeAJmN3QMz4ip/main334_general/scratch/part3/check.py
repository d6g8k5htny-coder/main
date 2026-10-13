import numpy as np
from scipy import integrate
# Toy base density with SG-type behaviour: near part ~ C u^{-1/3}, far part bounded with Gaussian decay
C_H=1.0
def nu_H(u): return C_H*u**(-1/3)*np.exp(-u) + 0.7*(1+u)**4*np.exp(-0.5*u**2)
def g(u): return u**(1/3)*nu_H(u)
# global sup of g
us=np.logspace(-8,3,200000); G=g(us).max(); print("sup g =",G, "at u=",us[g(us).argmax()])
# power law A on (0,a): density alpha a^-alpha A^(alpha-1); m=E A^{-2/3}
for alpha in [0.8,1.0,2.0]:
    a=np.sqrt((alpha+2)/alpha); m=a**(-2/3)*alpha/(alpha-2/3)
    def nuF(t):
        f=lambda A: A**(-1)*nu_H(t/A)*alpha*a**(-alpha)*A**(alpha-1)
        return integrate.quad(f,0,a,limit=400)[0]
    for t in [1e-2,1e-4,1e-6]:
        print(f"alpha={alpha} t={t:g}  t^(1/3)nu_F(t)/(m C_H)={t**(1/3)*nuF(t)/(m*C_H):.5f}")
# atomic A accumulating at 0: a_j=2^-j, p_j ∝ 2^{-j*0.7} => E A^{-2/3}=sum p_j 2^{2j/3} < inf
j=np.arange(0,60); p=2.0**(-0.7*j); p/=p.sum(); aj=2.0**(-j); m=(p*aj**(-2/3)).sum()
for t in [1e-2,1e-4,1e-6,1e-8]:
    val=(p*aj**(-1)*nu_H(t/aj)).sum()
    print(f"atomic t={t:g} t^(1/3)nu_F/(m C_H)={t**(1/3)*val/(m*C_H):.5f}")
# far Gaussian bound: int (1+|b|+l)^(2d) exp(-c(b^2+(b-l)^2)) db  <= C (1+l)^(2d) e^{-c l^2/2}
d=2;c=1.0
for l in [0.5,1,3,6,10]:
    I=integrate.quad(lambda b:(1+abs(b)+l)**(2*d)*np.exp(-c*(b*b+(b-l)**2)),-50,50)[0]
    print(f"l={l}: far integrand int={I:.3e}, ratio to (1+l)^(2d)e^(-c l^2/2)={I/((1+l)**(2*d)*np.exp(-c*l*l/2)):.3f}")
