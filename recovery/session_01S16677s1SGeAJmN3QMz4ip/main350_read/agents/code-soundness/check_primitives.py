"""Primitive and full-field enclosures vs independent mpmath evaluation."""
import random, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf
import mpmath as mp
mp.mp.dps = 120
random.seed(4242)
def mpq(q): return mp.mpf(q.numerator)/mp.mpf(q.denominator)
def contains(iv, val): return mpq(iv[0]) <= val <= mpq(iv[1])
def budget(): 
    b = sf.Budget(time.monotonic()); b.begin_fixture('x'); return b
bad = 0
# pi, sqrt
b = budget()
assert contains(sf._pi_bounds(budget=b), mp.pi)
for _ in range(200):
    q = Q(random.randint(0, 10**9), random.randint(1, 10**9))
    if not contains(sf._sqrt_bounds(q, budget=b), mp.sqrt(mpq(q))): bad += 1; print('sqrt fail', q)
# sin at points and intervals in [-2,2]
for _ in range(300):
    a = Q(random.randint(-2*10**6, 2*10**6), 10**6); c = a if random.random()<0.5 else min(Q(2), a + Q(random.randint(0, 10**5), 10**6))
    iv = sf._sin_bounds((a, c), budget=b)
    for t in (a, c, (a+c)/2):
        if not contains(iv, mp.sin(mpq(t))): bad += 1; print('sin fail', a, c, t, iv)
# exp(-1/t) via _q_point and psi
for _ in range(300):
    t = Q(random.randint(1, 10**6), random.randint(1, 10**6)) * Q(random.choice([1, 1, 1, Q(1,5000)]))
    iv = sf._q_point(t, budget=b)
    val = mp.exp(-1/mpq(t))
    if not contains(iv, val): bad += 1; print('q fail', t, iv, val)
for _ in range(300):
    t = Q(random.randint(1, 999), 1000)
    iv = sf._psi_point(t, budget=b)
    q = lambda x: mp.exp(-1/x) if x > 0 else mp.mpf(0)
    val = q(1-mpq(t))/(q(mpq(t))+q(1-mpq(t)))
    if not contains(iv, val): bad += 1; print('psi fail', t, iv, val)
# psi/q on intervals (monotone endpoint use)
for _ in range(200):
    a = Q(random.randint(-200, 1200), 1000); c = a + Q(random.randint(0, 300), 1000)
    iv = sf._psi_bounds((a, c), budget=b); qiv = sf._q_bounds((a, c), budget=b)
    q = lambda x: mp.exp(-1/x) if x > 0 else mp.mpf(0)
    psi = lambda x: mp.mpf(1) if x <= 0 else (mp.mpf(0) if x >= 1 else q(1-x)/(q(x)+q(1-x)))
    for t in (a, c, (a+c)/2, a + (c-a)/3):
        if not contains(iv, psi(mpq(t))): bad += 1; print('psi iv fail', a, c, t)
        if not contains(qiv, q(mpq(t))): bad += 1; print('q iv fail', a, c, t)
print('primitive failures', bad)

# Full field: independent mp formula
B_, R, R1, U0, V0 = sf.B, sf.R, sf.R1, sf.U0, sf.V0
def field_mp(u, v, s, z):
    u, v, s, z = map(mpq, (u, v, s, z))
    q = lambda x: mp.exp(-1/x) if x > 0 else mp.mpf(0)
    psi = lambda x: mp.mpf(1) if x <= 0 else (mp.mpf(0) if x >= 1 else q(1-x)/(q(x)+q(1-x)))
    r = s*s + z*z
    chi = psi((r - mpq(R1**2/4))/mpq(3*R1**2/4))
    step = psi((r - mpq(R1**2))/mpq(R**2/2 - R1**2))
    c = 1 - (1 - r/4)*step
    return mpq(B_) - s*s*c - z*z + chi*(u*s*s/2 + v*s)
bad = 0; widths = []
for _ in range(60):
    u = Q(random.randint(-10**6+1, 10**6-1), 10**6) * U0; v = Q(random.randint(-10**6+1, 10**6-1), 10**6) * V0
    # points spread over the chart incl. both transition annuli
    rad = random.choice([Q(0), R1/8, R1/2, R1*Q(3,5), R1, R1*Q(5,4), R*Q(7,10)]) * Q(random.randint(90, 100), 100)
    ang = Q(random.randint(0, 999), 1000)
    s = rad * Q(random.randint(-100, 100), 100); z2 = rad*rad - s*s
    z = Q(0) if z2 <= 0 else Q(str(mp.nstr(mp.sqrt(mpq(z2)), 30)))  # rational approx inside chart
    if s*s + z*z > R*R: z = Q(0)
    bb = budget()
    try:
        iv = sf.chart_field_bounds(u, v, (s, s), (z, z), budget=bb)
    except sf.Inconclusive as exc:
        print('inconclusive at', s, z, exc); continue
    val = field_mp(u, v, s, z)
    widths.append(iv[1]-iv[0])
    if not contains(iv, val): bad += 1; print('FIELD FAIL', u, v, s, z, iv, val)
# torus route incl. canonicalization seam
def torus_mp(u, v, y):
    y = [mpq(t) for t in y]
    cy = [t - mp.floor(t + mp.mpf(1)/2) for t in y]
    s = mp.sqrt(mp.mpf(2)/3)*mp.sin(mp.pi*cy[0]); z = mp.sqrt(mp.mpf(2)/9)*mp.sin(mp.pi*cy[1])
    q = lambda x: mp.exp(-1/x) if x > 0 else mp.mpf(0)
    psi = lambda x: mp.mpf(1) if x <= 0 else (mp.mpf(0) if x >= 1 else q(1-x)/(q(x)+q(1-x)))
    r = s*s + z*z
    chi = psi((r - mpq(R1**2/4))/mpq(3*R1**2/4)); step = psi((r - mpq(R1**2))/mpq(R**2/2 - R1**2))
    c = 1 - (1 - r/4)*step
    return mpq(B_) - s*s*c - z*z + chi*(mpq(u)*s*s/2 + mpq(v)*s)
for y in [(Q(0),Q(0)),(Q(1,2),Q(1,2)),(Q(-1,2),Q(1,8)),(Q(3,2),Q(-7,2)),(Q(1,1000),Q(0)),(Q(-1,1000),Q(1,1000)),(Q(1,100),Q(1,300)),(Q(999,1000),Q(0)),(Q(1,250),Q(0))]:
    u, v = U0/3, -V0/3
    bb = budget()
    iv = sf.torus_field_bounds(u, v, y, budget=bb)
    val = torus_mp(u, v, y)
    if not contains(iv, val): bad += 1; print('TORUS FAIL', y, iv, val)
print('field failures', bad, 'max width log2', max(widths).numerator.bit_length() - max(widths).denominator.bit_length())
