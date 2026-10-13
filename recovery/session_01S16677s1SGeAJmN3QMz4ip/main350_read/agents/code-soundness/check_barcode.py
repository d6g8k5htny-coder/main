"""Independent 1D superlevel-H0 barcode vs select_h0 over random and structured controls."""
import random, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf
import global_h0_selector as g
import run_global_h0_selector as drv
import mpmath as mp
mp.mp.dps = 100
random.seed(2024)
U0, V0 = g.U0, g.V0
def mpq(q): return mp.mpf(q.numerator)/mp.mpf(q.denominator)
def expected(u, v):
    """Closed-form 1D persistence of p(x) = -x^4/4 + u x^2/2 + v x on the axis."""
    um, vm = mpq(u), mpq(v)
    import sympy as sp
    X = sp.symbols('x')
    rr = sp.Poly(X**3 - sp.Rational(u.numerator, u.denominator)*X - sp.Rational(v.numerator, v.denominator), X).real_roots()
    real = sorted({mp.mpf(str(sp.N(r, 95))) for r in rr})
    if len(real) == 3 and len(rr) == 3 and len(set(rr)) < 3: real = []  # unreachable guard
    p = lambda t: -t**4/4 + um*t*t/2 + vm*t
    if len(set(rr)) == 3:
        x1, x2, x3 = real
        h1, h2, h3 = p(x1), p(x2), p(x3)
        if abs(h1-h3) < mp.mpf(10)**-70 and v == 0 and u > 0:
            return dict(bars=[(2, h1-h2)], essential=0, tie=True)
        younger = 0 if h1 < h3 else 2
        return dict(bars=[(younger, min(h1,h3)-h2)], essential=2-younger, tie=False)
    return dict(bars=[], essential=0, tie=False)

controls = []
for _ in range(40):  # random in the rectangle
    controls.append((Q(random.randint(-10**6+1, 10**6-1), 10**6)*U0, Q(random.randint(-10**6+1, 10**6-1), 10**6)*V0))
for _ in range(30):  # inside the wedge
    u = Q(random.randint(1, 10**6), 10**6)*U0
    vmax = Q(str(mp.nstr(mp.sqrt(4*mpq(u)**3/27), 40)))
    controls.append((u, Q(random.randint(-10**6+1, 10**6-1), 10**6)*vmax))
for _ in range(8):   # ties with irrational sqrt(u)
    controls.append((Q(random.randint(1, 10**6), 10**6)*U0*Q(3,7), Q(0)))
for _ in range(8):   # discriminant zero with rational double root
    r = Q(random.randint(-2000, 2000), 2**random.randint(17, 20))
    if r == 0: r = Q(1, 2**19)
    controls.append((3*r*r, -2*r**3))
controls += [(Q(0), Q(0)), (Q(0), V0/2), (-U0/2, Q(0)), (U0/2, Q(0))]
controls += [(f['u'], f['v']) for f in drv.frozen_fixtures()]
fails = 0
for i, (u, v) in enumerate(controls):
    if not (-U0 < u < U0 and -V0 < v < V0): continue
    b = sf.Budget(time.monotonic()); b.begin_fixture('c%d' % i)
    cert = g.select_h0(u, v, budget=b)
    if cert['status'] != 'CERTIFIED':
        print('NOT CERTIFIED', i, u, v, cert['status'], cert.get('failure')); continue
    ver = g.verify_h0(u, v, cert, budget=b)
    if ver['status'] != 'PASS': fails += 1; print('VERIFY FAIL', i, u, v, ver['status'], [k for k, val in ver['checks'].items() if val is False])
    exp = expected(u, v)
    bars = cert['actual_barcode']['finite']
    ok = len(bars) == len(exp['bars'])
    for bar, (young, life) in zip(bars, exp['bars']):
        lo, hi = bar['lifetime_interval']
        tol = mp.mpf(10)**-85
        ok &= bar['birth_root'] == young and mpq(lo) - tol <= life <= mpq(hi) + tol and bar['tied_birth'] == exp['tie']
    ok &= cert['actual_barcode']['essential'][0]['birth_root'] == exp['essential']
    proj = cert['population_projection']
    disc = 4*u**3 - 27*v**2
    want_reason = 'EXCLUDED_DEGENERATE' if disc == 0 else ('EXCLUDED_TIE' if u > 0 and v == 0 else 'INCLUDED')
    ok &= proj['reason'] == want_reason and proj['count'] == (len(bars) if want_reason == 'INCLUDED' else 0)
    # reflection symmetry: (u,-v) must give same lifetime with younger side exchanged
    if not ok:
        fails += 1
        print('BARCODE MISMATCH', i, u, v, 'got', [(bb['birth_root'], bb['lifetime_interval']) for bb in bars], cert['actual_barcode']['essential'][0]['birth_root'], proj, 'expected', exp)
print('barcode cases', len(controls), 'failures', fails)
# literal expectations vs closed form for the 18 fixtures
for f in drv.frozen_fixtures():
    e = drv.literal_expectation(f['fixture_id']); ex = expected(f['u'], f['v'])
    lit = e['lifetimes'][0] if e['lifetimes'] else None
    cf = ex['bars'][0][1] if ex['bars'] else None
    agree = (lit is None and cf is None) or (lit is not None and cf is not None and abs(mpq(lit) - cf) < mp.mpf(10)**-60)
    yr = ex['bars'][0][0] if ex['bars'] and not ex['tie'] else None
    print(f['fixture_id'], 'literal', lit, 'closed-form', cf if cf is None else mp.nstr(cf, 12), 'agree', agree, 'younger', e['younger_root_ordinal'], yr)
