# Exact-rational checks of Section 2 guards with ANALYTIC_GUARDS parameters (d=2, L=1, R=1/8)
from fractions import Fraction as Fr
d = 2
w = [Fr(1,3)**i for i in range(1, d+1)]
b = sum(w)
Delta = 2*w[-1]
R2 = Fr(1,64); r1 = Fr(1,32); a_star = r1/4; delta = r1**4/64
print("R^2 < min(w_d,1):", R2 < min(w[-1], 1), " R^2=",R2," w_d=",w[-1])
print("a_*^4 == delta/4:", a_star**4 == delta/4)
print("delta == R^4/16384:", delta == Fr(1,8)**4/16384, " delta < Delta:", delta < Delta, delta, Delta)
u0 = 3*Fr(1,2**32); v0 = Fr(1,2**47)
print("(14) u0 < a*^2/4:", u0 < a_star**2/4, " v0 < a*^3/4:", v0 < a_star**3/4)
# v0 >= eta_c u0^{3/2}: eta_c^2 u0^3 = (4/27) u0^3 vs v0^2
print("(2) v0^2 >= (4/27) u0^3:", v0**2 >= Fr(4,27)*u0**3, " equal:", v0**2 == Fr(4,27)*u0**3)
# (16) bound
bound16 = a_star**4/4 + u0*a_star**2/2 + v0*a_star
print("(16) a*^4/4 + u0 a*^2/2 + v0 a* =", bound16, " < 5delta/32:", bound16 < 5*delta/32, " 5delta/32 < delta/4:", 5*delta/32 < delta/4)
# generic: with u0 < a*^2/4 and v0 < a*^3/4, bound < a*^4(1/4+1/8+1/4) = 5/8 a*^4 = 5 delta/32
print("5/8 a*^4 == 5 delta/32:", Fr(5,8)*a_star**4 == 5*delta/32)
# root bound: |s|>=a* => |s|^3 <= u0|s| + v0 < |s|^3/4 + |s|^3/4 ; check at s = a*
print("root-bound at |s|=a*: u0 a* + v0 =", u0*a_star + v0, " < a*^3/2:", u0*a_star + v0 < a_star**3/2)
# (17) outside core: b - F0 >= (r1^2/16)|z|^2 >= delta at |z| = r1/2
print("(17) (r1^2/16)(r1/2)^2 == delta:", (r1**2/16)*(r1/2)**2 == delta, " R^2 > delta:", R2 > delta)
# first-differing-weight dominance: 2 w_i > 2 sum_{j>i} w_j
print("weight dominance:", all(2*w[i] > 2*sum(w[i+1:]) for i in range(d)))
# check for d up to 6 as well
for dd in [3,4,6]:
    ww = [Fr(1,3)**i for i in range(1, dd+1)]
    print("  d=%d dominance:" % dd, all(2*ww[i] > 2*sum(ww[i+1:]) for i in range(dd)))
# polynomial P(s) outside core: P(s) < -delta/4 for |s| >= r1/2 (my own check of axial-section containment)
import math
def P(s, u, v): return -s**4/4 + u*s**2/2 + v*s
worst = max(P(Fr(k, 2**12), u0, v0) for k in range(int(r1/2*2**12), int(r1*2**12)+1))
print("sup_{r1/2<=s<=r1} P(s;u0,v0) =", float(worst), " < -delta/4 =", float(-delta/4), ":", worst < -delta/4)
