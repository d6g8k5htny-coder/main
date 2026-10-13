import math
def integ(f,a,b,n=200000):
    h=(b-a)/n; s=0.5*(f(a)+f(b))
    for i in range(1,n): s+=f(a+i*h)
    return s*h
beta=2/3
# a=2 at alpha=beta
print("a(alpha=beta)=",math.sqrt((beta+2)/beta))
# (16): E A^-beta closed form vs numeric, alpha>beta
for alpha in (0.8,1.0,2.0):
    a=math.sqrt((alpha+2)/alpha)
    num=integ(lambda A: alpha*a**(-alpha)*A**(alpha-1)*A**(-beta),1e-9,a)
    print("alpha",alpha,"E A^-beta numeric",round(num,6),"closed",round(a**(-beta)*alpha/(alpha-beta),6),
          "tilted density mass",round(integ(lambda A:(alpha-beta)*a**(-(alpha-beta))*A**(alpha-beta-1),1e-9,a),6),
          "Jensen >1:",num>1)
# (19) identity: toy Psi_f(z)=1-exp(-K*min(1,z**-beta)); direct A-integral vs z-form
alpha=0.5; a=math.sqrt((alpha+2)/alpha); K=0.7; t=0.01
Psi=lambda z:1-math.exp(-K*min(1.0,z**(-beta)))
lhs=integ(lambda A: alpha*a**(-alpha)*A**(alpha-1)*Psi(A/t),1e-9,a)
rhs=alpha*a**(-alpha)*t**alpha*integ(lambda z:z**(alpha-1)*Psi(z),1e-9,a/t)
print("(19) lhs,rhs",lhs,rhs)
# (28): critical limit D_t/(t^beta log(1/t)) -> beta a^-beta K_f with z^beta Psi->K
alpha=beta; a=2.0
for t in (1e-3,1e-5,1e-7,1e-9):
    D=alpha*a**(-alpha)*t**alpha*(integ(lambda z:z**(alpha-1)*Psi(z),1e-9,1.0)+integ(lambda z:z**(alpha-1)*Psi(z),1.0,a/t,n=400000))
    print("t",t,"ratio",D/(t**beta*math.log(1/t)),"target",beta*a**(-beta)*K)
# (24)-(26) toy: deterministic barcode ell=(0.3,0.5), alpha=0.5, lam=1, T=1
alpha=0.5; a=math.sqrt((alpha+2)/alpha); lam=1.0; T=1.0; ells=(0.3,0.5)
Lam=lam*alpha*a**(-alpha)*integ(lambda z:z**(alpha-1),1e-12,T/min(ells))
print("Lambda_T numeric",Lam,"formula (24)",lam*a**(-alpha)*T**alpha*min(ells)**(-alpha))
mstar=lam*alpha*a**(-alpha)*integ(lambda z:z**(alpha-1)*sum(1 for l in ells if z*l<=T),1e-12,T/min(ells))
print("m*_T numeric",mstar,"formula (25)",lam*a**(-alpha)*T**alpha*sum(l**(-alpha) for l in ells))
# (26) first intensity on lifetime window (0,T] from density: integrate v^(alpha-1) over (0,T]
print("(26) window mass",lam*alpha*a**(-alpha)*sum(l**(-alpha) for l in ells)*T**alpha/alpha)
