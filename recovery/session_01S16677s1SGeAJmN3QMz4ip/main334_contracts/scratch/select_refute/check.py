import sympy as sp
b,k,r,s,d = sp.symbols('b k r s d', positive=True)
# v_r(b,k)/s == v_r(b/s,k/s): first four rows (transverse rows are zero)
vr = lambda B,K: sp.Matrix([B-K*r**3/2, -K*r**2, 0, 12*K])
print("mark rescaling exact:", sp.simplify(vr(b,k)/s - vr(b/s,k/s)) == sp.zeros(4,1))
# scaling exponent: density of s*Y in R^n is s^-n p(v/s), n=2d+2; W scales as s^(2d)
n = 2*d+2
print("net exponent A^-(2d+2) * A^(2d) =", sp.simplify(-n+2*d))
# dominator  s^-2 H_SG(b/s,k/s) bounded on (0,a) for k>0:
import numpy as np
dd=2; C=1.0; c=0.5
H=lambda B,K: C*(1+abs(B)+K)**(2*dd)*np.exp(-c*(B**2+K**2))
for (B,K) in [(0.0,1.0),(3.0,0.1),(-2.0,5.0),(0.0,0.01)]:
    ss=np.logspace(-4,np.log10(2.0),20000)
    vals=ss**-2*H(B/ss,K/ss)
    print(f"sup_s s^-2 H(b/s,k/s) at (b,k)=({B},{K}):", vals.max(), " value as s->0:", vals[0])
# F-envelope integrability: int k^-2/3 E_A[A^-2 H(b/A,k/A)] db dk = E[A^-2/3] * const, finite iff alpha>2/3
alpha=sp.symbols('alpha',positive=True); a=sp.sqrt((alpha+2)/alpha); A=sp.symbols('A',positive=True)
EAm23=sp.integrate(A**(-sp.Rational(2,3))*alpha*a**(-alpha)*A**(alpha-1),(A,0,a),conds='none')
print("E[A^-2/3] =", sp.simplify(EAm23), " (finite iff alpha>2/3)")
