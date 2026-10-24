import math
d=2; c=0.5
As=[10**(-6+12*i/200000) for i in range(200001)]
def sup(f): return max(f(A) for A in As)
for b,k in [(0.0,1.0),(3.0,0.01),(-2.0,5.0),(0.0,1e-3)]:
    dens=lambda A: A**(-(2*d+2))*math.exp(-c*(b*b+k*k)/A**2)
    wgt=lambda A: A**(-2)*(1+(abs(b)+k)/A)**(2*d)*math.exp(-c*(b*b+k*k)/A**2)
    print(f"b={b},k={k}: sup dens={sup(dens):.3e} sup wgt={sup(wgt):.3e}")
ell=0.3
def nuH(u): return u**(-1/3) if u<=1 else (1+u)**(2*d)*math.exp(-c*u*u)
print("sup_A A^-1 nu_H(ell/A) =", sup(lambda A: nuH(ell/A)/A))
def v_r(b,k,r): return [b-k*r**3/2,-k*r**2,0.0,12*k]
b,k,r,s=1.7,0.4,0.05,2.3
print("v_r homogeneous:", all(abs(x/s-y)<1e-12 for x,y in zip(v_r(b,k,r),v_r(b/s,k/s,r))))
