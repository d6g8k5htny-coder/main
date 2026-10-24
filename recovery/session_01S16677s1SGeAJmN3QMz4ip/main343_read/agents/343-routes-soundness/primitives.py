from fractions import Fraction as Q
from mpmath import mp, mpf, log, exp, sqrt, nstr
import sys
sys.path.insert(0, '.')
from finite_certificate import sqrt_bounds
from gaussian_tail import exp_neg_bounds
from mean_adaptive_confidence import log_bounds
mp.dps = 120
def mq(f): return mpf(f.numerator)/mpf(f.denominator)
ok = True
for x in (Q(2), Q(3,7), Q(1,3)+Q(1,10**9), Q(4,3), Q(1,2**40)):
    lo, hi = sqrt_bounds(x, 128); t = sqrt(mq(x)); ok &= mq(lo) <= t <= mq(hi)
for x in (Q(1)-Q(1,200000), Q(1,2), Q(3,2), Q(7), Q(1)-Q(2551,10**7)):
    lo, hi = log_bounds(x, bits=128); t = log(mq(x)); ok &= mq(lo) <= t <= mq(hi)
    print('log', x, 'encl', mq(lo) <= t <= mq(hi), 'width', nstr(mq(hi-lo), 3))
for x in (Q(2551,10000), Q(1,8), Q(1,3), Q(4000), Q(1,10**6)):
    lo, hi = exp_neg_bounds(x); t = exp(-mq(x)); ok &= mq(lo) <= t <= mq(hi)
    print('exp-', x, 'encl', mq(lo) <= t <= mq(hi), 'width', nstr(mq(hi-lo), 3))
print('ALL primitives enclose:', ok)
