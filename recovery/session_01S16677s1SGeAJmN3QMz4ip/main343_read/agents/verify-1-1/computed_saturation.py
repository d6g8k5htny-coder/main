# Feasibility of the proposed amendment: evaluate the closed-form at s=(1,1), tau=9/4
# through the module's own interval operations (bypassing root_bracket's tau<9/4 guard).
import sys
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import population_certificate as pc
tau = Q(9, 4)
s = (Q(1), Q(1))
square = s[0] ** 2, s[1] ** 2
base = pc._add((Q(1), Q(1)), pc._scale(square, Q(1, 3)))
denominator = pc._multiply(base, pc._sqrt(base))
first = pc._subtract((Q(3), Q(3)), pc._scale(pc._multiply(pc._subtract((Q(1), Q(1)), square), pc._reciprocal(denominator)), Q(3)))
quarter = pc._sqrt(pc._sqrt(s))
inverse_seventh = pc._reciprocal((quarter[0] ** 7, quarter[1] ** 7))
second = pc._subtract(pc._scale(pc._subtract(inverse_seventh, (Q(1), Q(1))), Q(36, 7)), pc._scale(pc._subtract((Q(1), Q(1)), quarter), Q(4)))
coefficient = pc._scale(pc.quarter_power_bounds(tau / 4), tau / 4)
print('first =', first, ' second =', second, ' quarter =', quarter)
try:
    result = pc._outward(pc._scale(pc._add(first, pc._multiply(coefficient, second)), Q(1, 15)))
    print('computed closed-form mass at s=1:', result, ' contains 1/5:', result[0] <= Q(1,5) <= result[1])
except ValueError as e:
    print('closed-form at s=1 raised:', e)
# Direct core route with rlo := 1 (whole wedge), n := 1, Darboux lower/upper, using the module's integrand shape.
from fractions import Fraction
N = 1 << 12
lo_sum = Q(0); hi_sum = Q(0)
def f(theta):
    theta2 = theta ** 2
    b = Q(3) / (3 + theta2)
    sq = pc.sqrt_bounds(b, pc.BITS)
    w = Q(1, 15) * theta * (9 - theta2) * b ** 2
    return sq[0] * w, sq[1] * w
vals = [f(Q(i, N)) for i in range(N + 1)]
lower = sum(v[0] for v in vals[:-1]) / N
upper = sum(v[1] for v in vals[1:]) / N
print('direct wedge Darboux (N=%d): [%s, %s] contains 1/5: %s' % (N, float(lower), float(upper), lower <= Q(1,5) <= upper))
