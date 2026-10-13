import math, random, time, sys
import numpy as np
import fcore2 as F, fcore4 as G
from ia import isqrt
t = time.time(); elem = G.elementary_kmoments(); print('elementary k-moments %.1fs' % (time.time() - t))
Km = G.KMoments(elem)
# check K-moments widths and values against numpy
w = (G.K_HI - G.K_LO) / 2 ** G.KLEVEL
for (k0, k1) in ((0.5, 2.0), (0.5, 0.5 + 64 * w), (0.5 + 1365 * w, 0.5 + 1366 * w)):
    K = Km.get(k0, k1)
    xs, ws = np.polynomial.legendre.leggauss(60)
    kk = 0.5 * (k1 - k0) * xs + 0.5 * (k1 + k0); ww = 0.5 * (k1 - k0) * ws
    for p in (-6, 0, 2):
        ref = float((ww * 3 * kk ** (1 / 3 + p / 2) * np.exp(-12 * kk * kk)).sum())
        ok = K[p][0] <= ref * (1 + 1e-12) and ref * (1 - 1e-12) <= K[p][1]
        print('  K cell [%g,%g] p=%d  rel width %.2e  contains %s' % (k0, k1, p, (K[p][1] - K[p][0]) / ref, ok))
sig = [isqrt((2.0, 2.0)), isqrt((2.0, 2.0)), isqrt((6.0, 6.0))]
Ma, Mb, Mc = F.Moments(sig[0], (2.0, 2.0), 6), F.Moments(sig[1], (2.0, 2.0), 2), F.Moments(sig[2], (6.0, 6.0), 2)
def bigroot(B, T):
    xm = np.where(B < 0, -B, B / 2); lo = xm.copy(); hi = xm + np.abs(B) + (T / 2) ** (1 / 3) + 1.0
    for _ in range(200):
        m = 0.5 * (lo + hi); P = (m + B) ** 2 * (2 * m - B) - T
        hi = np.where(P > 0, m, hi); lo = np.where(P > 0, lo, m)
    return 0.5 * (lo + hi)
def ref4(box, n=16):
    xs, ws = np.polynomial.legendre.leggauss(n)
    pts = []; wts = []
    for (l, h), s2 in zip(box[:3], (2.0, 2.0, 6.0)):
        p = 0.5 * (h - l) * xs + 0.5 * (h + l)
        pts.append(p); wts.append(0.5 * (h - l) * ws * np.exp(-p * p / (2 * s2)) / math.sqrt(2 * math.pi * s2))
    l, h = box[3]; kp = 0.5 * (h - l) * xs + 0.5 * (h + l); pts.append(kp)
    wts.append(0.5 * (h - l) * ws * 3 * kp ** (1 / 3) * np.exp(-12 * kp * kp))
    a, b, c, k = np.meshgrid(*pts, indexing='ij')
    W = np.einsum('i,j,k,l->ijkl', *wts)
    B = b - a * a / (12 * k); u = np.sqrt(k) * c - a * b / (4 * np.sqrt(k)) + a ** 3 / (72 * k ** 1.5); T = 3 * u * u
    x = bigroot(B, T)
    return float((W * (np.abs(B) ** 3 + 4 * x ** 3 - 3 * B * B * x)).sum())
random.seed(3); bad = 0; used = 0; rels = []
for trial in range(250):
    sz = 10 ** random.uniform(-2.3, -0.5)
    a0 = random.uniform(-3, 3) * 1.414; b0 = random.uniform(-3, 3) * 1.414; c0 = random.uniform(-3, 3) * 2.449
    lev = random.randint(4, 10); j = random.randrange(2 ** lev)
    kw = 1.5 / 2 ** lev; k0 = 0.5 + kw * j
    box = ((a0, a0 + sz * 1.414), (b0, b0 + sz * 1.414), (c0, c0 + sz * 2.449), (k0, k0 + kw))
    I = G.H_box4(box, Ma, Mb, Mc, Km); I1 = G.H_box4(box, Ma, Mb, Mc, Km, second=False)
    r = ref4(box)
    if not (I[0] - 1e-9 * abs(r) <= r <= I[1] + 1e-9 * abs(r)):
        r2 = ref4(box, 30)
        if not (I[0] - 1e-8 * abs(r2) <= r2 <= I[1] + 1e-8 * abs(r2)):
            bad += 1; print('FAIL', box, I, r, r2)
    if I[1] - I[0] < I1[1] - I1[0]:
        used += 1; rels.append(((I[1] - I[0]) / abs(r), sz, kw))
print('4D boxes bad', bad, 'second-order used', used, 'of 250')
rels.sort(key=lambda t: t[1])
for x in rels[::max(1, len(rels) // 8)]:
    print('  size %.4f kw %.4f rel width %.2e' % (x[1], x[2], x[0]))
