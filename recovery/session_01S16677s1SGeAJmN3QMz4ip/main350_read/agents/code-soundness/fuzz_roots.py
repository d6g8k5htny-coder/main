"""Root-isolation completeness/multiplicity fuzz with planted rational roots (stdlib only)."""
import random, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf
import global_h0_selector as g

random.seed(777)
def budget():
    return sf.Budget(time.monotonic(), seconds=900)

def poly_from_roots(roots, scalar):
    p = (scalar,)
    for r, m in roots:
        for _ in range(m):
            p = g._mul(p, (-r, Q(1)))
    return p

fails = 0
cases = 0
for trial in range(400):
    # domain with dyadic endpoints of random width
    lo = Q(random.randint(-64, 64), 2**random.randint(0, 6))
    width = Q(1, 2**random.randint(0, 8)) * random.randint(1, 8)
    hi = lo + width
    k = random.randint(1, 5)
    roots = {}
    for _ in range(k):
        kind = random.random()
        if kind < 0.2:
            r = random.choice([lo, hi])                      # endpoint root
        elif kind < 0.45:
            depth = random.randint(1, 6)                      # bisection midpoint root
            r = lo + width * Q(random.randint(0, 2**depth), 2**depth)
        elif kind < 0.7:
            r = lo + width * Q(random.randint(0, 999), 1000) # interior non-dyadic
        elif kind < 0.85:
            r = lo - Q(random.randint(1, 100), 7)            # outside left
        else:
            r = hi + Q(random.randint(1, 100), 7)            # outside right
        m = random.choice([1, 1, 1, 2, 3])
        roots[r] = roots.get(r, 0) + m
    roots = sorted(roots.items())
    scalar = Q(random.choice([-7, -1, 1, 3, 5]), random.choice([1, 2, 3]))
    p = poly_from_roots(roots, scalar)
    b = budget(); b.begin_fixture('f')
    res = g.isolate_roots(p, (lo, hi), budget=b)
    cases += 1
    inside = [(r, m) for r, m in roots if lo <= r <= hi]
    if res['status'] != 'CERTIFIED':
        # Acceptable only as INCONCLUSIVE; record
        print('non-certified', res['status'], res.get('failure'), 'roots', roots, 'domain', (lo, hi))
        continue
    got = res['roots']
    ok = len(got) == len(inside)
    for rec, (r, m) in zip(got, inside):
        a, c = rec['isolator']
        ok &= a <= r <= c and rec['multiplicity'] == m
        if rec['exact_value'] is not None:
            ok &= rec['exact_value'] == r
        else:
            ok &= a < r < c
    if not ok:
        fails += 1
        print('FAIL', 'planted', inside, 'got', [(x['isolator'], x['exact_value'], x['multiplicity']) for x in got], 'poly', p, 'domain', (lo, hi))
    # verifier replay must pass on the producer's own record
    b2 = budget(); b2.begin_fixture('v')
    checks = {}
    try:
        g._verify_roots(res, p, (lo, hi), b2, checks, 'verify')
    except Exception as exc:
        fails += 1; print('VERIFY FAIL', repr(exc), checks, p, (lo, hi))
print('root fuzz cases', cases, 'failures', fails)
