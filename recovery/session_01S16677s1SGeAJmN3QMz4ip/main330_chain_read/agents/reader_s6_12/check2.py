import math
def simpson(f,a,b,n=20000):
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+= (4 if i%2 else 2)*f(a+i*h)
    return s*h/3
def pow_int(p,g,a,n=20000):
    # integral_0^a A^p g(A) dA, p>-1, via A=a*s^(1/(p+1)); A^p dA = a^(p+1)/(p+1) ds
    q=p+1
    return a**q/q*simpson(lambda s: g(a*s**(1/q)) if s>0 else g(0.0),0.0,1.0,n)
beta=2/3
for alpha in (0.8,1.0,2.0):
    a=math.sqrt((alpha+2)/alpha)
    num=alpha*a**(-alpha)*pow_int(alpha-1-beta,lambda A:1.0,a)
    tilt=(alpha-beta)*a**(-(alpha-beta))*pow_int(alpha-beta-1,lambda A:1.0,a)
    print("alpha",alpha,"E A^-beta num",round(num,6),"closed",round(a**(-beta)*alpha/(alpha-beta),6),"tilt mass",round(tilt,6))
# (19)
alpha=0.5; a=math.sqrt((alpha+2)/alpha); K=0.7; t=0.01
Psi=lambda z:1-math.exp(-K*min(1.0,z**(-beta))) if z>0 else 1-math.exp(-K)
lhs=alpha*a**(-alpha)*pow_int(alpha-1,lambda A:Psi(A/t),a)
rhs=alpha*a**(-alpha)*t**alpha*pow_int(alpha-1,Psi,a/t,n=200000)
print("(19) lhs rhs",round(lhs,8),round(rhs,8))
# (28) critical: use exact pieces: integral_0^1 z^(beta-1)Psi + integral_1^{a/t} z^-1 [z^beta Psi]; substitute z=e^w on [1,a/t]
alpha=beta; a=2.0
low=pow_int(beta-1,Psi,1.0)
for t in (1e-3,1e-6,1e-9,1e-12):
    W=math.log(a/t)
    up=simpson(lambda w: math.exp(beta*w)*Psi(math.exp(w)),0.0,W,n=200000)
    D=beta*a**(-beta)*t**beta*(low+up)
    print("t",t,"ratio",round(D/(t**beta*math.log(1/t)),6),"target",round(beta*a**(-beta)*K,6))
# (24)-(25)
alpha=0.5; a=math.sqrt((alpha+2)/alpha); lam=1.0; T=1.0; ells=(0.3,0.5)
Lam=lam*alpha*a**(-alpha)*pow_int(alpha-1,lambda z:1.0,T/min(ells))
print("Lambda_T num",round(Lam,8),"(24)",round(lam*a**(-alpha)*T**alpha*min(ells)**(-alpha),8))
ms=lam*alpha*a**(-alpha)*sum(pow_int(alpha-1,lambda z:1.0,T/l) for l in ells)
print("m*_T num",round(ms,8),"(25)",round(lam*a**(-alpha)*T**alpha*sum(l**(-alpha) for l in ells),8))
