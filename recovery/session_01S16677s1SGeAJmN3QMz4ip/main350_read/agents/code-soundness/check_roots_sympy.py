"""Root isolation vs sympy real_roots on random integer polynomials (irrational roots)."""
import random, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf
import global_h0_selector as g
import sympy as sp, mpmath as mp
mp.mp.dps = 80
random.seed(99)
x = sp.symbols('x')
fails = cases = 0
for trial in range(250):
    deg = random.randint(1, 6)
    coeffs = [random.randint(-9, 9) for _ in range(deg+1)]
    if coeffs[-1] == 0: coeffs[-1] = random.choice([-3, 2])
    if random.random() < 0.3:  # square a factor to make repeated irrational roots
        f = [random.randint(-5, 5), random.randint(-5,5), random.choice([1, 2, -1])]
        pf = tuple(Q(c) for c in f)
        base = tuple(Q(c) for c in coeffs)
        p = g._mul(g._mul(pf, pf), base)
    else:
        p = tuple(Q(c) for c in coeffs)
    expr = sum(sp.Rational(c.numerator, c.denominator)*x**i for i, c in enumerate(p))
    lo, hi = Q(-3), Q(3)
    if random.random() < 0.5: lo, hi = Q(-5, 2), Q(7, 4)
    b = sf.Budget(time.monotonic()); b.begin_fixture('f')
    res = g.isolate_roots(p, (lo, hi), budget=b)
    cases += 1
    if res['status'] != 'CERTIFIED':
        print('non-certified', res['status'], res.get('failure'), p); continue
    sroots = sp.Poly(expr, x).real_roots()  # with multiplicity, sorted
    from collections import Counter
    distinct = []
    for r in sroots:
        if not distinct or distinct[-1][0] != r: distinct.append([r, 0])
        distinct[-1][1] += 1
    inside = [(r, m) for r, m in distinct if sp.Rational(lo.numerator, lo.denominator) <= r <= sp.Rational(hi.numerator, hi.denominator)]
    got = res['roots']
    ok = len(got) == len(inside)
    if ok:
        for rec, (r, m) in zip(got, inside):
            a, c = rec['isolator']
            rv = mp.mpf(str(sp.N(r, 70)))
            ok &= mp.mpf(a.numerator)/a.denominator <= rv <= mp.mpf(c.numerator)/c.denominator and rec['multiplicity'] == m
    if not ok:
        fails += 1
        print('FAIL', p, (lo, hi), 'sympy', [(sp.N(r, 20), m) for r, m in inside], 'got', [(float(r['isolator'][0]), r['exact_value'], r['multiplicity']) for r in got])
print('sympy root cases', cases, 'failures', fails)
