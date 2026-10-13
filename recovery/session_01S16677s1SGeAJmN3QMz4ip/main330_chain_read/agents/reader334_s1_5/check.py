# Numerical sanity check of PROOF.md (9),(10),(12),(14),(16),(17) with a toy base density
# nu_H(s) = c s^(beta-1) on (0,1], plus a far bump 0.5 on [2,3] (bounded, finite mass).
import math, random
beta=2/3.; c=1.3
def nu(s):
    v = c*s**(beta-1) if 0<s<=1 else 0.0
    if 2<=s<=3: v+=0.5
    return v
def integ(f,lo,hi,n=200000):
    # midpoint on log grid for lo>0
    if hi<=lo: return 0.0
    a=math.log(lo); b=math.log(hi); h=(b-a)/n; tot=0.0
    for i in range(n):
        x=math.exp(a+(i+0.5)*h); tot+=f(x)*x*h
    return tot
def M(alpha,lo): return integ(lambda s: s**(-alpha)*nu(s), lo, 3.0)
def mH(t): return integ(nu,1e-12,min(t,3.0))
for alpha in (0.4, 2/3., 1.5):
    a=math.sqrt((alpha+2)/alpha)
    for t in (1e-3,1e-5):
        G=M(alpha,t/a)
        nuF=alpha*a**(-alpha)*t**(alpha-1)*G
        mF=mH(t/a)+(t/a)**alpha*G
        # Monte-Carlo cross-check of nu_F via E[A^-1 nu_H(t/A)] and m_F via E min
        random.seed(1); N=400000
        s1=0.0; 
        for _ in range(N):
            A=a*random.random()**(1/alpha); s1+=nu(t/A)/A
        mc=s1/N
        if alpha<beta:
            pred_nu=alpha*a**(-alpha)*M(alpha,1e-12)*t**(alpha-1); pred_m=a**(-alpha)*M(alpha,1e-12)*t**alpha
        elif alpha==beta:
            pred_nu=beta*a**(-beta)*c*t**(beta-1)*math.log(1/t); pred_m=a**(-beta)*c*t**beta*math.log(1/t)
        else:
            EA=a**(-beta)*alpha/(alpha-beta)
            pred_nu=c*EA*t**(beta-1); pred_m=(c/beta)*EA*t**beta
        print(f"alpha={alpha:.3f} a={a:.4f} t={t:g}  nuF(9)={nuF:.5g} MC={mc:.5g} ratio_to_(12/14/16)={nuF/pred_nu:.4f}   mF(10)={mF:.5g} ratio={mF/pred_m:.4f}")
# Jensen check: E A^-beta > 1 for alpha>beta
for alpha in (0.7,1.0,2.0,10.0,1000.0):
    a=math.sqrt((alpha+2)/alpha); print("alpha",alpha,"E A^-beta=",a**(-beta)*alpha/(alpha-beta))
