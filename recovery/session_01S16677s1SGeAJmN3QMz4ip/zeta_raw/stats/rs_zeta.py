"""Vectorized Riemann-Siegel evaluation of Hardy's Z(t), zero finding and lobe integrals.

Z(t) = 2 sum_{n<=N} n^{-1/2} cos(theta(t) - t log n)
       + (-1)^{N-1} (t/2pi)^{-1/4} [C0(p) + C1(p) a^{-1} + C2(p) a^{-2}],
a = sqrt(t/2pi), N = floor(a), p = a - N,
Psi(p) = cos(2pi(p^2 - p - 1/16)) / cos(2pi p),
C0 = Psi, C1 = -Psi'''/(96 pi^2), C2 = Psi''/(64 pi^2) + Psi^(6)/(18432 pi^4)
(Riemann-Siegel formula with the first three correction terms; Edwards, "Riemann's
Zeta Function", ch. 7 -- formula quoted from memory and validated numerically below
against mpmath.siegelz).

C0..C2 are tabulated once with mpmath (high precision derivatives) and fitted by
Chebyshev series on p in [0,1]; the fit residual is checked.
"""
import os, sys, time, json
import numpy as np
from numpy.polynomial import chebyshev as cheb
import mpmath

HERE = os.path.dirname(os.path.abspath(__file__))
TWOPI = 2 * np.pi


def _fit_coeffs(deg=60):
    path = os.path.join(HERE, "rs_cheb_coeffs.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["c0"], d["c1"], d["c2"]
    mpmath.mp.dps = 50
    pi = mpmath.pi

    def psi(p):
        return mpmath.cos(2 * pi * (p * p - p - mpmath.mpf(1) / 16)) / mpmath.cos(2 * pi * p)

    M = deg + 20
    k = np.arange(M)
    x = np.cos(np.pi * (k + 0.5) / M)  # Chebyshev nodes on [-1,1]
    vals = {0: [], 1: [], 2: []}
    for xi in x:
        p = (mpmath.mpf(float(xi)) + 1) / 2
        d = [mpmath.diff(psi, p, j) for j in (0, 2, 3, 6)]
        vals[0].append(float(d[0]))
        vals[1].append(float(-d[2] / (96 * pi ** 2)))
        vals[2].append(float(d[1] / (64 * pi ** 2) + d[3] / (18432 * pi ** 4)))
    out = []
    for j in range(3):
        c = cheb.chebfit(x, np.array(vals[j]), deg)
        res = np.max(np.abs(cheb.chebval(x, c) - np.array(vals[j])))
        print(f"C{j} cheb fit residual on nodes: {res:.2e}", flush=True)
        out.append(c)
    np.savez(path, c0=out[0], c1=out[1], c2=out[2])
    return tuple(out)


C0c, C1c, C2c = _fit_coeffs()


def theta(t):
    return 0.5 * t * np.log(t / TWOPI) - 0.5 * t - np.pi / 8 + 1.0 / (48 * t) + 7.0 / (5760 * t ** 3)


def Z(t, nterms_corr=3, block=4_000_000):
    """Hardy Z(t) for an array of t >= ~200 (double precision)."""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    flat = t.ravel()
    res = out.ravel()
    Nmax_all = int(np.floor(np.sqrt(flat.max() / TWOPI)))
    step = max(1, block // max(Nmax_all, 1))
    for s in range(0, flat.size, step):
        tt = flat[s:s + step]
        a = np.sqrt(tt / TWOPI)
        N = np.floor(a).astype(np.int64)
        Nmax = int(N.max())
        n = np.arange(1, Nmax + 1, dtype=np.float64)
        logn = np.log(n)
        w = n ** -0.5
        th = theta(tt)
        ph = th[:, None] - tt[:, None] * logn[None, :]
        c = np.cos(ph)
        if N.min() != Nmax:
            c *= (n[None, :] <= N[:, None])
        main = 2.0 * (c @ w)
        p = a - N
        xx = 2 * p - 1
        corr = cheb.chebval(xx, C0c)
        if nterms_corr >= 2:
            corr = corr + cheb.chebval(xx, C1c) / a
        if nterms_corr >= 3:
            corr = corr + cheb.chebval(xx, C2c) / a ** 2
        sign = np.where(N % 2 == 1, 1.0, -1.0)  # (-1)^(N-1)
        res[s:s + step] = main + sign * a ** -0.5 * corr
    return out


def validate(rng, lo, hi, m=60):
    mpmath.mp.dps = 25
    ts = np.sort(rng.uniform(lo, hi, m))
    ref = np.array([float(mpmath.siegelz(float(x))) for x in ts])
    rows = {}
    for k in (1, 2, 3):
        v = Z(ts, nterms_corr=k)
        rows[f"maxabs_err_corr{k}"] = float(np.max(np.abs(v - ref)))
    rows["max_abs_Z"] = float(np.max(np.abs(ref)))
    rows["n_points"] = m
    rows["window"] = [lo, hi]
    return rows


def mean_spacing(t):
    return TWOPI / np.log(t / TWOPI)


def Nbar(t):
    """Smooth zero-counting function theta(t)/pi + 1 (unfolding map)."""
    return theta(t) / np.pi + 1.0


# ---------------------------------------------------------------------------
# zero finding on a chunk [t0, t1)
# ---------------------------------------------------------------------------

def _refine(a, b, fa, fb, tol=2e-9, maxit=80):
    """Vectorized Illinois/bisection hybrid on brackets with fa*fb<0."""
    a = a.copy(); b = b.copy(); fa = fa.copy(); fb = fb.copy()
    side = np.zeros(a.size, dtype=np.int8)
    for it in range(maxit):
        width = b - a
        act = width > tol
        if not act.any():
            break
        c = np.where(act, (a * fb - b * fa) / (fb - fa), a)
        # safeguard: fall back to bisection if c not strictly inside
        bad = ~((c > a) & (c < b))
        c = np.where(bad, 0.5 * (a + b), c)
        if it % 4 == 3:  # periodic bisection guarantees shrinking
            c = np.where(act, 0.5 * (a + b), c)
        idx = np.nonzero(act)[0]
        fc = Z(c[idx])
        same_a = np.sign(fc) == np.sign(fa[idx])
        ia = idx[same_a]; ib = idx[~same_a]
        a[ia] = c[ia]; fa[ia] = fc[same_a]
        # Illinois: halve the retained endpoint value when same side twice
        fb[ia] = np.where(side[ia] == 1, fb[ia] * 0.5, fb[ia]); side[ia] = 1
        b[ib] = c[ib]; fb[ib] = fc[~same_a]
        fa[ib] = np.where(side[ib] == -1, fa[ib] * 0.5, fa[ib]); side[ib] = -1
    return 0.5 * (a + b)


def zeros_chunk(args):
    t0, t1, per = args
    h = mean_spacing(t0) / per  # spacing decreases with t; use finer of chunk ends
    h = min(h, mean_spacing(t1) / per)
    n = int(np.ceil((t1 - t0) / h))
    grid = t0 + h * np.arange(n + 2)  # one point beyond t1 for overlap
    vals = Z(grid)
    s = np.sign(vals)
    br = np.nonzero(s[:-1] * s[1:] < 0)[0]
    a = grid[br]; b = grid[br + 1]; fa = vals[br]; fb = vals[br + 1]
    # near-miss detection: local minimum of |Z| on the grid with no sign change on either side
    av = np.abs(vals)
    i = np.arange(1, n + 1)
    cand = i[(av[i] < av[i - 1]) & (av[i] <= av[i + 1]) & (s[i - 1] == s[i]) & (s[i] == s[i + 1])]
    extra_a, extra_b, extra_fa, extra_fb = [], [], [], []
    n_nearmiss_pairs = 0
    if cand.size:
        sub = 100
        for ci in cand:
            g2 = np.linspace(grid[ci - 1], grid[ci + 1], 2 * sub + 1)
            v2 = Z(g2)
            s2 = np.sign(v2)
            k = np.nonzero(s2[:-1] * s2[1:] < 0)[0]
            if k.size:
                n_nearmiss_pairs += 1
                extra_a.append(g2[k]); extra_b.append(g2[k + 1])
                extra_fa.append(v2[k]); extra_fb.append(v2[k + 1])
    if extra_a:
        a = np.concatenate([a] + extra_a); b = np.concatenate([b] + extra_b)
        fa = np.concatenate([fa] + extra_fa); fb = np.concatenate([fb] + extra_fb)
    z = _refine(a, b, fa, fb)
    z = np.sort(z[(z >= t0) & (z < t1)])
    return z, int(cand.size), n_nearmiss_pairs


# ---------------------------------------------------------------------------
# lobe integrals between consecutive zeros (Gauss-Legendre)
# ---------------------------------------------------------------------------
GL_X, GL_W = np.polynomial.legendre.leggauss(16)


def lobes_chunk(z):
    """For consecutive zeros z_k<z_{k+1}: signed integral of Z, of the normalized
    unfolded integrand Zt(u) = Z(t)/sqrt(log(t/2pi)) du with du = log(t/2pi)/(2pi) dt,
    and peak |Z| / sqrt(log(t/2pi)) estimated on the quadrature nodes + midpoint."""
    a = z[:-1]; b = z[1:]
    mid = 0.5 * (a + b); half = 0.5 * (b - a)
    nodes = mid[:, None] + half[:, None] * GL_X[None, :]
    nodes_all = np.concatenate([nodes, mid[:, None]], axis=1)
    v = Z(nodes_all.ravel()).reshape(nodes_all.shape)
    L = np.log(nodes / TWOPI)
    raw = half * (v[:, :-1] @ GL_W)
    unf = half * ((v[:, :-1] * np.sqrt(L) / TWOPI) @ GL_W)
    peak = np.max(np.abs(v) / np.sqrt(np.log(nodes_all / TWOPI)), axis=1)
    return raw, unf, peak


def run(window, per=20, nchunks=None, procs=4, outname=None):
    from multiprocessing import Pool
    lo, hi = window
    t_start = time.time()
    if nchunks is None:
        nchunks = max(8, int((hi - lo) / 2000))
    edges = np.linspace(lo, hi, nchunks + 1)
    jobs = [(edges[i], edges[i + 1], per) for i in range(nchunks)]
    with Pool(procs) as pool:
        res = pool.map(zeros_chunk, jobs, chunksize=1)
    z = np.concatenate([r[0] for r in res])
    z.sort()
    ncand = sum(r[1] for r in res); nnear = sum(r[2] for r in res)
    t_zero = time.time() - t_start
    # exact count check
    mpmath.mp.dps = 20
    n_exact = int(mpmath.nzeros(hi)) - int(mpmath.nzeros(lo))
    dup = int(np.sum(np.diff(z) < 1e-8))
    # lobes
    parts = np.array_split(np.arange(z.size), procs * 8)
    zparts = [z[p[0]:p[-1] + 2] for p in parts if p.size]
    with Pool(procs) as pool:
        lres = pool.map(lobes_chunk, zparts, chunksize=1)
    raw = np.concatenate([r[0] for r in lres])
    unf = np.concatenate([r[1] for r in lres])
    peak = np.concatenate([r[2] for r in lres])
    assert raw.size == z.size - 1
    info = dict(window=[lo, hi], grid_points_per_mean_spacing=per, n_zeros_found=int(z.size),
                n_zeros_exact_mpmath_nzeros=n_exact, n_duplicate=dup,
                n_nearmiss_candidates=int(ncand), n_nearmiss_with_hidden_pair=int(nnear),
                seconds_zero_finding=round(t_zero, 1), seconds_total=round(time.time() - t_start, 1),
                sign_alternation_ok=bool(np.all(np.sign(raw[1:]) == -np.sign(raw[:-1]))))
    if outname:
        np.savez_compressed(os.path.join(HERE, outname), zeros=z, lobe_raw=raw, lobe_unf=unf,
                            peak=peak, unfold=Nbar(z))
    return z, info


if __name__ == "__main__":
    rng = np.random.default_rng(20261012)
    report = {}
    which = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if which == "validate":
        t0 = time.time()
        for lo, hi in [(1e4, 1e5), (1e5, 1e6)]:
            report[f"{lo:g}-{hi:g}"] = validate(rng, lo, hi)
        # spot-check zeros against mpmath.zetazero
        mpmath.mp.dps = 25
        chk = []
        for nidx in [10143, 54321, 138070, 777777]:
            g = float(mpmath.zetazero(nidx).imag)
            zz, _, _ = zeros_chunk((g - 0.3, g + 0.3, 20))
            chk.append(dict(n=nidx, mpmath=g, ours=[float(x) for x in zz],
                            min_abs_diff=float(np.min(np.abs(zz - g))) if zz.size else None))
        report["zetazero_check"] = chk
        # timing
        tt = np.linspace(9e5, 9e5 + 100, 200000)
        t1 = time.time(); Z(tt); report["sec_per_1e5_evals_at_9e5"] = (time.time() - t1) / 2
        print(json.dumps(report, indent=1))
        with open(os.path.join(HERE, "rs_validation.json"), "w") as f:
            json.dump(report, f, indent=1)
    else:
        lo, hi = float(sys.argv[2]), float(sys.argv[3])
        per = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        z, info = run((lo, hi), per=per, outname=f"zeta_{which}.npz")
        print(json.dumps(info, indent=1))
        with open(os.path.join(HERE, f"zeta_{which}_info.json"), "w") as f:
            json.dump(info, f, indent=1)
