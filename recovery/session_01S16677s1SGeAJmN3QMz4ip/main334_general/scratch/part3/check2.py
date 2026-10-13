import math
C_H=1.0
def nu_H(u): return C_H*u**(-1/3)*math.exp(-u) + 0.7*(1+u)**4*math.exp(-0.5*u**2)
def g(u): return u**(1/3)*nu_H(u)
# sup of g on a log grid
G=0;ua=None
for i in range(200000):
    u=10**(-8+11*i/199999); v=g(u)
    if v>G: G,ua=v,u
print("sup g =",G,"at u=",ua)
def quad(f,lo,hi,n=200000):
    # midpoint rule in log-space for (0,hi]
    s=0.0; L=math.log(lo); U=math.log(hi); h=(U-L)/n
    for i in range(n):
        x=math.exp(L+(i+0.5)*h); s+=f(x)*x*h
    return s
for alpha in [0.8,1.0,2.0]:
    a=math.sqrt((alpha+2)/alpha); m=a**(-2/3)*alpha/(alpha-2/3)
    for t in [1e-2,1e-4,1e-6]:
        val=quad(lambda A: A**(-1)*nu_H(t/A)*alpha*a**(-alpha)*A**(alpha-1), t*1e-9, a)
        print(f"alpha={alpha} t={t:g}  t^(1/3)nu_F(t)/(m C_H)={t**(1/3)*val/(m*C_H):.5f}")
# atomic A accumulating at 0
p=[2.0**(-0.7*j) for j in range(60)]; Z=sum(p); p=[x/Z for x in p]; aj=[2.0**(-j) for j in range(60)]
m=sum(pj*a**(-2/3) for pj,a in zip(p,aj))
for t in [1e-2,1e-4,1e-6,1e-8]:
    val=sum(pj/a*nu_H(t/a) for pj,a in zip(p,aj))
    print(f"atomic t={t:g} t^(1/3)nu_F/(m C_H)={t**(1/3)*val/(m*C_H):.5f}")
d=2;c=1.0
for l in [0.5,1,3,6,10]:
    n=200000; s=0.0
    for i in range(n):
        b=-50+100*(i+0.5)/n; s+=(1+abs(b)+l)**(2*d)*math.exp(-c*(b*b+(b-l)**2))*100/n
    print(f"l={l}: far int={s:.3e}, ratio to (1+l)^(2d)e^(-c l^2/2)={s/((1+l)**(2*d)*math.exp(-c*l*l/2)):.3f}")
