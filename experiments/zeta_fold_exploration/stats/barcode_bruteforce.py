"""Independent test of stats/analyze.py:barcode against a brute-force superlevel H0
elder-rule computation (O(n^2), no union-find) on random alternating sequences, multi-path
inputs, and a known analytic function (numerics review).
Stdout saved as results/stats/barcode_bruteforce.out. A smaller version runs in test_numeric.py."""
import numpy as np
from analyze import barcode

def brute(vals):
    """superlevel H0 bars of a path graph with vertex values vals (distinct).
    For each local max i (interior or endpoint): it dies when its component meets a component
    containing a higher vertex. Death level = max over sides of (min on the path segment to the
    nearest strictly higher vertex on that side). Returns list of (birth_idx, death_val)."""
    n = len(vals); out = []
    for i in range(n):
        left_ok = i == 0 or vals[i - 1] < vals[i]
        right_ok = i == n - 1 or vals[i + 1] < vals[i]
        if not (left_ok and right_ok):
            continue
        deaths = []
        # left
        m = np.inf
        for j in range(i - 1, -1, -1):
            m = min(m, vals[j])
            if vals[j] > vals[i]:
                deaths.append(m); break
        m = np.inf
        for j in range(i + 1, n):
            m = min(m, vals[j])
            if vals[j] > vals[i]:
                deaths.append(m); break
        if deaths:
            out.append((i, max(deaths)))
    return out

rng = np.random.default_rng(12345)
nbad = 0; ntot = 0
for trial in range(400):
    npath = rng.integers(1, 4)
    areas, sids = [], []
    for p in range(npath):
        K = int(rng.integers(2, 60))
        mag = rng.lognormal(0, 1.5, K)
        sgn = rng.choice([-1, 1]) * (-1) ** np.arange(K)
        areas.append(sgn * mag); sids.append(np.full(K, p))
    area = np.concatenate(areas); sid = np.concatenate(sids)
    ell, lobe, bml = barcode(area, sid)
    ref = []
    for p in range(npath):
        a = areas[p]
        vals = np.r_[0.0, np.cumsum(a)]
        for (i, d) in brute(vals):
            if i == 0 or i == len(vals) - 1:
                continue   # dropped: born at endpoint
            ref.append(vals[i] - d)
    ntot += 1
    if len(ref) != len(ell) or not np.allclose(np.sort(ref), np.sort(ell), rtol=1e-12, atol=1e-12):
        nbad += 1
        if nbad < 4:
            print('MISMATCH', trial, len(ref), len(ell))
print('random multi-path trials:', ntot, 'mismatches:', nbad)

# known function: F(x) = sin(x) + eps*sin(7.3x) on [x0, x1]; compare barcode of lobes of g=F'
from math import pi
x = np.linspace(0.3, 40.0, 2_000_001)
eps = 0.12
g = np.cos(x) + eps * 7.3 * np.cos(7.3 * x)
F = np.sin(x) + eps * np.sin(7.3 * x)
s = np.sign(g); idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
# zeros by linear interpolation, F at zeros by interpolation (adequate for test)
z = x[idx] - g[idx] * (x[idx + 1] - x[idx]) / (g[idx + 1] - g[idx])
Fz = np.sin(z) + eps * np.sin(7.3 * z)
area = np.diff(Fz)
ell, lobe, _ = barcode(area, np.zeros(area.size, dtype=np.int64))
ref = [Fz[i] - d for (i, d) in brute(Fz) if 0 < i < len(Fz) - 1]
print('analytic test: n bars', len(ell), 'brute', len(ref), 'max abs diff', float(np.max(np.abs(np.sort(ell) - np.sort(ref)))))
# sanity: tiny bars come from the eps-wiggles; long bars from the sin(x) maxima
print('5 longest bars', np.round(np.sort(ell)[-5:], 6), ' 5 shortest', np.round(np.sort(ell)[:5], 6))
