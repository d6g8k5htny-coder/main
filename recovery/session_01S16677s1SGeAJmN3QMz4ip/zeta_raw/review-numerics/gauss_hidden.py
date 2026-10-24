"""Resolve the Gaussian-control cells flagged as possible hidden zero pairs (same sign at both
grid nodes) using the cubic Hermite interpolant on a 400-point subgrid, and estimate the effect
of the missing short bars on the Hill CDF exponents."""
import sys, numpy as np
sys.dont_write_bytecode = True
ST = '/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats'
sys.path.insert(0, ST)
from analyze import barcode

def regen(seed, M=1000, Ng=2 ** 19):
    rng = np.random.default_rng(seed)
    L = 2 * np.pi * M; w = 1.0 / M; kmax = Ng // 2; k = np.arange(kmax + 1)
    q = np.exp(-0.5 * (w * k) ** 2); q = q / (q[0] + 2 * q[1:].sum())
    xi = rng.standard_normal(kmax + 1); eta = rng.standard_normal(kmax + 1)
    G = np.zeros(kmax + 1, dtype=complex)
    G[0] = Ng * np.sqrt(q[0]) * xi[0]
    G[1:] = (Ng / 2) * np.sqrt(2 * q[1:]) * (xi[1:] - 1j * eta[1:]); G[-1] = 0
    g = np.fft.irfft(G, n=Ng); dg = np.fft.irfft(G * (1j * w * k), n=Ng)
    return g, dg, L / Ng

hidden = []  # (seed, gap_unfolded, area)
ncand = 0
for seed in range(10000, 10500):
    g, dg, dx = regen(seed)
    s = np.sign(g); ag = np.abs(g)
    # candidate cells [i, i+1] with same sign at both ends and Hermite derivative sign change
    same = s[:-1] == s[1:]
    d0 = dg[:-1] * dx; d1 = dg[1:] * dx
    cand = np.nonzero(same & (np.sign(d0) != np.sign(d1)) & (np.minimum(ag[:-1], ag[1:]) < 0.5 * np.maximum(np.abs(d0), np.abs(d1))))[0]
    ncand += cand.size
    tt = np.linspace(0, 1, 401)
    for i in cand:
        g0, g1, a0, a1 = g[i], g[i + 1], d0[i], d1[i]
        t2 = tt * tt; t3 = t2 * tt
        p = (2 * t3 - 3 * t2 + 1) * g0 + (t3 - 2 * t2 + tt) * a0 + (-2 * t3 + 3 * t2) * g1 + (t3 - t2) * a1
        sc = np.nonzero(np.sign(p[:-1]) != np.sign(p[1:]))[0]
        if sc.size >= 2:
            za = tt[sc[0]]; zb = tt[sc[-1]]
            m = (tt >= za) & (tt <= zb)
            area = np.trapezoid(p[m], tt[m]) * dx if m.sum() > 2 else 0.0
            hidden.append((seed, (zb - za) * dx / np.pi, abs(area)))
print('candidate cells examined:', ncand, ' resolved hidden pairs:', len(hidden))
for h in hidden:
    print('  seed %d  unfolded gap %.5f  lobe area %.3e' % h)

d = np.load(ST + '/gauss.npz')
area, gap, sid = d['area'], d['gap'], d['sid']
ell, _, _ = barcode(area, sid)
med = np.median(ell)
L = ell / med
extra = np.array([h[2] for h in hidden]) / med
def hill(x, c):
    m = x < c; return m.sum() / np.sum(np.log(c / x[m])), int(m.sum())
for c in (0.01, 0.003):
    s0, n0 = hill(L, c); s1, n1 = hill(np.r_[L, extra], c)
    print('bars Hill cut %g: as published %.4f (n=%d); with hidden-pair bars added %.4f (n=%d); shift %+.4f' % (c, s0, n0, s1, n1, s1 - s0))
r = gap / gap.mean()
rx = np.array([h[1] for h in hidden]) / gap.mean()
for c in (0.2, 0.15):
    s0, n0 = hill(r, c); s1, n1 = hill(np.r_[r, rx], c)
    print('gaps Hill cut %g: %.4f -> %.4f (shift %+.4f)' % (c, s0, s1, s1 - s0))
# expected number of hidden pairs from the empirical small-gap law F(r) ~ a r^2
a = np.mean(r < 0.1) / 0.1 ** 2
h = (2 * np.pi * 1000 / 2 ** 19) / np.pi
print('expected hidden pairs ~ N*a*h^2/3 = %.1f (h_unfolded=%.5f)' % (r.size * a * h * h / 3, h))
