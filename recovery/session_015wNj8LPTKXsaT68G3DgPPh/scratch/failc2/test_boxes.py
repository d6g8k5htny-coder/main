import math, random, sys, time
import numpy as np
import fcore2 as F
from ia import isqrt
k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
sa2, sb2, sc2 = 2.0 / k, 2.0, 6.0 * k
sa, sb, sc = isqrt((sa2, sa2)), isqrt((sb2, sb2)), isqrt((sc2, sc2))
Ma, Mb, Mc = F.Moments(sa, (sa2, sa2), 6), F.Moments(sb, (sb2, sb2), 2), F.Moments(sc, (sc2, sc2), 2)
def bigroot(B, T):
    xm = np.where(B < 0, -B, B / 2)
    x = xm + np.abs(B) + (T / 2) ** (1 / 3) + 1.0
    lo = xm.copy(); hi = x.copy()
    for _ in range(200):
        m = 0.5 * (lo + hi)
        P = (m + B) ** 2 * (2 * m - B) - T
        hi = np.where(P > 0, m, hi); lo = np.where(P > 0, lo, m)
    return 0.5 * (lo + hi)
def midroot(B, T):
    aB = -B
    lo = aB / 2; hi = aB.copy()
    for _ in range(200):
        m = 0.5 * (lo + hi)
        P = (m + B) ** 2 * (2 * m - B) - T
        lo = np.where(P > 0, m, lo); hi = np.where(P > 0, hi, m)
    return 0.5 * (lo + hi)
def ref(box, which, n=24):
    xs, ws = np.polynomial.legendre.leggauss(n)
    pts = []; wts = []
    for (l, h), s2 in zip(box, (sa2, sb2, sc2)):
        p = 0.5 * (h - l) * xs + 0.5 * (h + l)
        wt = 0.5 * (h - l) * ws * np.exp(-p * p / (2 * s2)) / math.sqrt(2 * math.pi * s2)
        pts.append(p); wts.append(wt)
    a, b, c = np.meshgrid(*pts, indexing='ij')
    W = np.einsum('i,j,k->ijk', *wts)
    B = b - a * a / 12; u = c - a * b / 4 + a ** 3 / 72; T = 3 * u * u
    if which == 'V':
        x = bigroot(B, T); val = -3 * B * (2 * x - B) * (x + B)
    else:
        sup = (B < 0) & (T < np.abs(B) ** 3 / 2)
        Bc = np.where(sup, B, -1.0); Tc = np.where(sup, T, 0.0)
        x = midroot(Bc, Tc); val = np.where(sup, 2 * np.abs(Bc) ** 3 + 2 * Tc - 3 * Bc * (2 * x - Bc) * (x + Bc), 0.0)
    return float((W * val).sum())
random.seed(7)
bad = 0; n2 = 0; t0 = time.time(); widths = []
for trial in range(600):
    which = 'V' if trial % 2 == 0 else 'H2'
    sz = 10 ** random.uniform(-2.5, -0.3)
    a0 = random.uniform(-3, 3) * math.sqrt(sa2); b0 = random.uniform(-3, 3) * math.sqrt(sb2); c0 = random.uniform(-3, 3) * math.sqrt(sc2)
    if which == 'H2':
        b0 = random.uniform(-4, 0.2)
    box = ((a0, a0 + sz * math.sqrt(sa2)), (b0, b0 + sz * math.sqrt(sb2)), (c0, c0 + sz * math.sqrt(sc2)))
    I = (F.V_box if which == 'V' else F.H2_box)(box, Ma, Mb, Mc)
    I1 = (F.V_box if which == 'V' else F.H2_box)(box, Ma, Mb, Mc, second=False)
    r = ref(box, which)
    # reference error allowance: GL on smooth boxes is ~exact; on kinked boxes allow a relative slack
    slack = 1e-9 * (abs(r) + 1e-300)
    ok = I[0] - slack <= r <= I[1] + slack
    if not ok:
        r2 = ref(box, which, 60)
        ok = I[0] - 1e-7 * abs(r2) <= r2 <= I[1] + 1e-7 * abs(r2)
        if not ok:
            bad += 1
            print('FAIL', which, box, I, r, r2)
    if I[1] - I[0] < I1[1] - I1[0]:
        n2 += 1
        if r != 0:
            widths.append(((I[1] - I[0]) / max(1e-300, abs(r)), (I1[1] - I1[0]) / max(1e-300, abs(r)), sz))
print('k', k, 'bad', bad, 'second-order improved', n2, 'of 600', 'time %.1fs' % (time.time() - t0))
widths.sort(key=lambda t: t[2])
for wd in widths[::max(1, len(widths) // 12)]:
    print('  size %.4f  rel width 2nd %.3e   1st %.3e' % (wd[2], wd[0], wd[1]))
