"""Exact arithmetic behind MF4-MF9 of Math-#157 (reviewer's independent check, stdlib)."""
from fractions import Fraction as F
import json, math
out = {}
# MF6 line 1: Markov with E V <= C Phi h^{2d}(1+log 1/h), V >= c eta delta^{2d-1}, delta = h/lambda,
# h = exp(-a m^2), lambda = exp(a m^2/(4d)):  h^{2d} / delta^{2d-1} = h * lambda^{2d-1}  -> exponent -a + a(2d-1)/(4d) = -a(2d+1)/(4d)
rates = {}
for d in range(2, 9):
    a = F(1)  # exponent per unit a m^2
    e1 = -a + a*F(2*d-1, 4*d)          # line 1
    assert e1 == -a*F(2*d+1, 4*d)
    e2 = -4*a + 2*a                    # line 2: E R_m^2 / h_m^2
    e3 = -2*a*F(1, 4*d)                # line 3: E L^2 / lambda_m^2 = exp(-a m^2/(2d))
    assert e3 == -a/F(2*d)
    binding = max(e1, e2, e3)          # least negative = binding rate
    assert binding == e3               # the derivative-sup tail is binding for every d >= 2
    rates[d] = [str(e1), str(e2), str(e3)]
    # Psi_F > x => some k_j > (x/((2d)^d J))^{1/d}; exp(-c m^2) at m ~ x^{1/d} gives exp(-c' x^{2/d})
    assert F(2, d) == 2*F(1, d)
out["MF6_exponents_per_unit_a_m2"] = rates
# retained radii count: interval length eta_j, step 2 delta, slab width 2 zeta_0 <= eta_j/4, delta <= eta_j/8:
#   eta_j/(2 delta) - zeta_0/delta - 1 >= eta_j/(4 delta)  <=>  eta_j/4 - zeta_0 >= delta   (P's Lemma M'(b))
for eta in (F(1), F(1,8), F(3,7)):
    for zeta0 in (eta/8, eta/16, eta/64):
        for delta in (eta/8, eta/16, eta/100):
            assert eta/4 - zeta0 >= delta
            assert eta/(2*delta) - zeta0/delta - 1 >= eta/(4*delta)
out["retained_radii_count"] = True
# Cauchy-Schwarz prefactor: sqrt(1+beta^2+Phi) <= 1 + beta + sqrt(Phi)
for beta in (F(0), F(1,3), F(7), F(100)):
    for Phi in (F(1), F(4), F(10**6)):
        lhs2 = 1 + beta*beta + Phi
        rhs = 1 + beta + math.isqrt(int(Phi)) if Phi.denominator == 1 and math.isqrt(int(Phi))**2 == Phi else None
        if rhs is not None:
            assert lhs2 <= rhs*rhs
out["CS_prefactor"] = True
# MF8: E e^{uY} = 1 + int_0^inf u e^{ut} P(Y>t) dt <= 1 + C A u/(c-u) for u < c  (exact integral of u e^{(u-c)t})
for c in (F(1), F(3,10)):
    for u in (c/2, c*F(9,10)):
        assert u/(c-u) > 0
out["MF8_ratio_positive"] = True
# 2 theta < c admissibility (Hölder needs E exp(2 theta Psi^alpha))
for c in (F(1), F(3,10)):
    theta = c/3
    assert 2*theta < c and not (2*(c/2) < c)
out["holder_two_theta"] = True
print(json.dumps(out, indent=1))
