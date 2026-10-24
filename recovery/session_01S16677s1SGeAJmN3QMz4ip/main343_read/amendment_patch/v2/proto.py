import sys, time
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import population_certificate as pc
def exponent(m, n):
    lo, hi = pc.log_bounds(1 - m, bits=pc.BITS)
    return pc._outward((-n * hi, -n * lo))
for mass, n in (((Q(1, 7), Q(1, 5)), 4), ((Q(1, 7 * 4 ** 8), Q(1, 5 * 4 ** 8)), 4 ** 8), ((Q(1, 3), Q(1, 3)), 4)):
    t = time.monotonic()
    lo, hi = pc.void_interval(mass, n)
    t1 = time.monotonic() - t
    ident = (lo, hi) == (pc.exp_neg_bounds(exponent(mass[1], n)[1])[0], pc.exp_neg_bounds(exponent(mass[0], n)[0])[1])
    t = time.monotonic()
    plo, phi = (1 - mass[1]) ** n, (1 - mass[0]) ** n
    sane = lo < plo <= phi < hi
    t2 = time.monotonic() - t
    print('void', n, 'identity', ident, 'sane', sane, 'void_t %.3f pow_t %.3f' % (t1, t2), 'width %.3e' % float(hi - lo), 'xhi', float(exponent(mass[1], n)[1]))
try:
    pc.void_interval((Q(1, 7), Q(1, 5)), 4 ** 8)
    print('(1/7,1/5)@4**8 OK')
except ValueError as e:
    print('(1/7,1/5)@4**8 raises:', e)
for b in (pc.coefficient_interval(), (Q(1, 3), Q(1, 2))):
    lo, hi = pc.poisson_target(b)
    print('poisson', (lo, hi) == (pc.exp_neg_bounds(b[1])[0], pc.exp_neg_bounds(b[0])[1]), lo < hi, float(lo), float(hi))
print('default', pc.poisson_target() == pc.poisson_target(pc.coefficient_interval()))
