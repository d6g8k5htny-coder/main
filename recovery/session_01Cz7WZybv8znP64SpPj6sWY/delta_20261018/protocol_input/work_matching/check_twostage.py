"""Two-stage data-side bracket check: F <-delta-> P <-e-> Q (grid).
Claim: N_Q([a+2delta+d, b-2delta-d)) <= N_F([a,b)) <= N_Q([a-2delta-d, b+2delta+d))
when a > 2delta + d, d = 2e. Exact rationals, explicit matchings, short unmatched bars.
Pure combinatorics; no field/sampler/driver/experiment."""
from fractions import Fraction as F
import random
random.seed(7)

def cnt(xs, lo, hi):
    return 0 if lo >= hi else sum(1 for x in xs if lo <= x < hi)

def perturb(life, rad, step):
    # move birth and death each by a multiple of step within [-rad, rad]
    k = int(rad / step)
    return life + step * random.randint(-k, k) - step * random.randint(-k, k)

bad = 0; trials = 0
for _ in range(20000):
    delta = F(random.randint(1, 4), 40); e = F(random.randint(1, 4), 80); d = 2 * e
    step = F(1, 80)
    Fl, Pl, Ql = [], [], []
    for _ in range(random.randint(0, 20)):
        lf = step * random.randint(1, 240)
        lp = perturb(lf, delta, step)
        if lp <= 0:
            # P side vanished: then F bar must be short (<=2delta) to be unmatched legally
            if lf <= 2 * delta:
                Fl.append(lf)
            continue
        Fl.append(lf); Pl.append(lp)
    # P bars unmatched to F: lifetime <= 2delta
    for _ in range(random.randint(0, 4)):
        Pl.append(step * random.randint(1, int(2 * delta / step)))
    # second stage P -> Q
    for lp in Pl:
        lq = perturb(lp, e, step)
        if lq <= 0:
            if lp > 2 * e:
                lq = None  # illegal: would need diagonal match of a long bar; force keep
                Ql.append(lp)
            continue
        Ql.append(lq)
    for _ in range(random.randint(0, 6)):
        Ql.append(step * random.randint(1, max(1, int(2 * e / step))))
    a = 2 * delta + d + step * random.randint(1, 40)
    b = a + step * random.randint(1, 80)
    lo = cnt(Ql, a + 2 * delta + d, b - 2 * delta - d)
    mid = cnt(Fl, a, b)
    hi = cnt(Ql, a - 2 * delta - d, b + 2 * delta + d)
    trials += 1
    if not (lo <= mid <= hi):
        bad += 1
print("two-stage trials", trials, "violations", bad)
