"""Float Monte Carlo of the true Z(r)/r^2 at r=1/20. NON-CERTIFYING.

Samples the conditional law described in `engine/solver_pilots/h3_20260921/PROOF.md`
directly, via the sector decomposition the proof itself uses (X = f(x,0),
Y = f_y(x,0), E = f_yy(x,0) + m2 f(x,0) mutually independent), and reports the
sample mean of the typed integrand. It exists to answer a different question
from `reconstruct_bound.py`: not "is the arithmetic reproducible" but "is the
claimed inequality consistent with the quantity it is about".

The conditional means and covariances are formed in mpmath and only then cast
to float for sampling, because the X-sector conditional covariance is nearly
degenerate at small r.

WHAT THIS DOES NOT ESTABLISH. A Monte Carlo estimate is explicitly named in
CLAUDE.md rule 3 as NOT a certificate, and this path is float throughout, so it
is NON-CERTIFYING in both respects. A sample mean above the candidate
coefficient is not a proof of the candidate, and could not become one by adding
samples; it only means no counterexample to the claimed inequality was found at
the sampled radius. It bounds nothing, verifies no register, moves no gate,
discharges no premise and awards zero organizational independence credit. The
pilot it cross-checks is NOT DEPLOYED.

Run: python3 reviews/independent/h3_20260921/sample_true_value.py
"""
from __future__ import annotations

import sys

try:
    from mpmath import mp, mpf, exp, matrix, lu_solve
    import numpy as np
except ImportError:  # pragma: no cover - neither dependency is guaranteed
    print("sample_true_value: mpmath/numpy absent; NON-CERTIFYING sampling skipped.")
    sys.exit(0)

mp.dps = 60

R = mpf(1) / 20
BATCHES = 8
PER_BATCH = 1_000_000
SEED = 20260921
CANDIDATE = 1.747


def he(n, x):
    p, q = mpf(1), x
    if n == 0:
        return p
    for k in range(1, n):
        p, q = q, x * q - k * p
    return q


def kd(n, s):
    """n-th derivative of k(s) = exp(-s^2/2); the |j|>=1 images are below 1e-120."""
    return (-1) ** n * he(n, s) * exp(-s ** 2 / 2)


def conditional(sigma, observed_idx, target_idx, observed):
    """Gaussian conditional mean and covariance of target given observed."""
    n_obs, n_tar = len(observed_idx), len(target_idx)
    a = matrix(n_obs, n_obs)
    for i, p in enumerate(observed_idx):
        for j, q in enumerate(observed_idx):
            a[i, j] = sigma[p][q]
    b = matrix(n_tar, n_obs)
    for i, p in enumerate(target_idx):
        for j, q in enumerate(observed_idx):
            b[i, j] = sigma[p][q]
    c = matrix(n_tar, n_tar)
    for i, p in enumerate(target_idx):
        for j, q in enumerate(target_idx):
            c[i, j] = sigma[p][q]
    w = matrix(n_tar, n_obs)
    for i in range(n_tar):
        row = lu_solve(a.T, matrix([b[i, j] for j in range(n_obs)]))
        for j in range(n_obs):
            w[i, j] = row[j]
    return w * matrix(observed), c - w * b.T


def law(r):
    """Conditional laws of (f_xx, f_xy, f_yy) at M and S under the six pins."""
    h = r / 2
    m2 = mpf(1)
    sigma_e2 = mpf(2)

    # X sector: (X(-h), X(h), X'(-h), X'(h), X''(-h), X''(h)).
    pts = [(-h, 0), (h, 0), (-h, 1), (h, 1), (-h, 2), (h, 2)]
    sig_x = [[(-1) ** b * kd(a + b, u - v) for (v, b) in pts] for (u, a) in pts]
    mu_x, cov_x = conditional(sig_x, [0, 1, 2, 3], [4, 5],
                              [mpf(6) / 5, mpf(6) / 5 - r ** 3 / 6, 0, 0])

    # Y sector: (Y(-h), Y(h), Y'(-h), Y'(h)) with covariance m2 * k structure.
    pts_y = [(-h, 0), (h, 0), (-h, 1), (h, 1)]
    sig_y = [[m2 * (-1) ** b * kd(a + b, u - v) for (v, b) in pts_y] for (u, a) in pts_y]
    mu_y, cov_y = conditional(sig_y, [0, 1], [2, 3], [0, 0])

    # E sector is untouched by the pins; f_yy = E - m2 * (pinned f).
    kr = exp(-r ** 2 / 2)
    cov_e = [[sigma_e2, sigma_e2 * kr], [sigma_e2 * kr, sigma_e2]]
    shift = [m2 * mpf(6) / 5, m2 * (mpf(6) / 5 - r ** 3 / 6)]
    return mu_x, cov_x, mu_y, cov_y, cov_e, shift


def factor(cov):
    w, v = np.linalg.eigh(cov)
    return v * np.sqrt(np.clip(w, 0, None))


def main():
    mu_x, cov_x, mu_y, cov_y, cov_e, shift = law(R)
    to_m = lambda m: np.array([[float(m[i, j]) for j in range(2)] for i in range(2)])
    to_l = lambda m: np.array([float(m[i]) for i in range(2)])
    lx, ly = factor(to_m(cov_x)), factor(to_m(cov_y))
    le = factor(np.array([[float(cov_e[i][j]) for j in range(2)] for i in range(2)]))
    mx, sh = to_l(mu_x), np.array([float(s) for s in shift])
    r = float(R)

    rng = np.random.default_rng(SEED)
    total = sq = 0.0
    n = typed_n = 0
    for _ in range(BATCHES):
        fxx = mx + rng.standard_normal((PER_BATCH, 2)) @ lx.T
        fxy = rng.standard_normal((PER_BATCH, 2)) @ ly.T
        fyy = rng.standard_normal((PER_BATCH, 2)) @ le.T - sh
        det_m = fxx[:, 0] * fyy[:, 0] - fxy[:, 0] ** 2
        det_s = fxx[:, 1] * fyy[:, 1] - fxy[:, 1] ** 2
        typed = (det_m > 0) & (fxx[:, 0] + fyy[:, 0] < 0) & (det_s < 0)
        val = np.where(typed, np.abs(det_m * det_s), 0.0) / r ** 2
        total += val.sum()
        sq += (val ** 2).sum()
        n += PER_BATCH
        typed_n += int(typed.sum())

    mean = total / n
    sd = float(np.sqrt(max(sq / n - mean ** 2, 0.0) / n))
    print(f"  samples={n:,} typed_rate={typed_n / n:.6f}")
    print(f"  sample mean Z(r)/r^2 = {mean:.6f} +/- {3 * sd:.6f} (3 sigma, r=1/20)")
    print(f"  candidate coefficient = {CANDIDATE}")
    print(f"  no counterexample at this radius: {mean - 3 * sd > CANDIDATE}")
    print("sample_true_value: NON-CERTIFYING float Monte Carlo. Not a bound, not a "
          "certificate, no status change, no independence credit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
