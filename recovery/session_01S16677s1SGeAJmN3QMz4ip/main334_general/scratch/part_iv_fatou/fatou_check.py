# Toy check of (iv): nu_H(u) = C u^(-1/3) on (0,1], 0 beyond (canonical representative
# with lim u^(1/3) nu_H(u) = C). A ~ power law on (0,a]: P(A<=s)=(s/a)^alpha.
# m = E[A^(-2/3)] finite iff alpha > 2/3. Compute ell^(1/3) nu_F(ell) = E[A^(-2/3) g(ell/A)]
# exactly: g(u)=C*1{u<=1}, so ell^(1/3) nu_F(ell) = C * E[A^(-2/3) 1{A>=ell}].
import math
C, a = 1.0, 1.0
def ratio(ell, alpha):
    # E[A^(-2/3) 1{A>=ell}] = alpha a^(-alpha) int_ell^a s^(alpha-1-2/3) ds
    p = alpha - 2/3
    if abs(p) < 1e-12:
        return C*alpha*a**(-alpha)*math.log(a/ell)
    return C*alpha*a**(-alpha)*(a**p - ell**p)/p
for alpha in (0.4, 2/3, 0.9):
    print("alpha", round(alpha,3), [round(ratio(10.0**(-j), alpha),4) for j in range(1,9)])
