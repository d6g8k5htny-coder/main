"""Gap-CDF Hill slope at cuts 0.4/0.3/0.2 for CUE of small N, to place the zeta values
(A: 3.05/3.09/3.11, B: 2.90/2.94/2.93) and GUE two-term (2.84/2.91/2.96) (numerics review).
Stdout saved as results/stats/cue_gap_shape.out."""
import numpy as np
rng = np.random.default_rng(2026)
def angles(B, N):
    A = (rng.standard_normal((B, N, N)) + 1j * rng.standard_normal((B, N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(A); d = np.diagonal(R, axis1=1, axis2=2); Q = Q * (d / np.abs(d))[:, None, :]
    return np.sort(np.mod(np.angle(np.linalg.eigvals(Q)), 2 * np.pi), axis=1)
def hill(x, c):
    m = x < c; return m.sum() / np.sum(np.log(c / x[m])), int(m.sum())
for N in (2, 3, 4, 6, 12):
    nm = max(1, 1_200_000 // N)
    th = angles(nm, N)
    g = np.diff(np.concatenate([th, th[:, :1] + 2 * np.pi], axis=1), axis=1).ravel() * N / (2 * np.pi)
    # crude bootstrap-free SE ~ s/sqrt(n)
    out = []
    for c in (0.4, 0.3, 0.2):
        s, n = hill(g, c); out.append('cut %.1f: %.3f (+-%.3f, n=%d)' % (c, s, 1.96 * s / np.sqrt(n), n))
    print('CUE N=%-3d' % N, ' | '.join(out))
