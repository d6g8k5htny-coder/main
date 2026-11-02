#!/usr/bin/env python3
"""Exact exponent ledger for main#259/6004452810 sections 4-5 (Fraction only).
nu_soft,p(ell; rho) <= C ell^(-e_p) rho^(5 e_p + 1 - s), where s is the extra r-power in the
dictionary density (s = 0 for 12 pi_r E_Q[W_r/r^4] dr db dsigma dell, per P (10.2) with dk = dell/r^3).
--dict-power s applies a negative control. One summary line; nonzero exit on failure."""
import sys
from fractions import Fraction as F
s = F(sys.argv[sys.argv.index("--dict-power") + 1]) if "--dict-power" in sys.argv else F(0)
fails = []
def ck(c, m):
    if not c: fails.append(m)
for p in (1, 2, 3, 4):
    a = F(p * (p + 4), 2)
    ck(a == p + F(p, 2) + p + F(p * (p - 1), 2), "soft exponent p=%d" % p)
    ck(F(p * (p + 1), 2) + p == F(p * (p + 3), 2), "codimension p=%d" % p)
    e = a + p                                   # epsilon >= r: (eps + r)^p <= (2 eps)^p
    ck(e == F(p * (p + 6), 2), "e_p p=%d" % p)
def power(e, rho_exp):                          # ell-exponent of ell^-e * rho^(5e+1-s), rho = ell^rho_exp
    return -e + rho_exp * (5 * e + 1 - s)
tab = {1: F(11, 18), 2: F(10, 9), 3: F(31, 18)}
for p, want in tab.items():
    ck(power(F(p * (p + 6), 2), F(2, 9)) == want, "table p=%d" % p)
ck(power(F(7, 2), F(2, 9)) > F(1, 2), "one-soft sector o(ell^(1/2))")
ck(power(F(7, 2), F(3, 13)) == F(10, 13) and F(10, 13) > F(2, 3), "quintic corner 10/13 > 2/3")
# ranges: ell^(1/4) <= r <= rho <= ell^(1/5)  (kappa <= 1 and r <= kappa); smaller exponent = larger power of ell < 1
ck(F(1, 5) < F(2, 9) < F(1, 4) and F(1, 5) < F(3, 13) < F(1, 4), "rho exponents inside [1/5, 1/4]")
# witness scales: r=t^3, kappa=t, k=t^4, ell=t^13 -> r = ell^(3/13), kappa r^4 = ell
ck(F(3, 13) * 13 == 3 and 1 + 4 * 3 == 13 and 4 + 3 * 3 == 13, "witness scales")
print("ledger259 dict-power=%s: %s (%d failing%s)" % (s, "PASS" if not fails else "FAIL", len(fails), "" if not fails else "; " + "; ".join(fails[:3])))
sys.exit(0 if not fails else 1)
