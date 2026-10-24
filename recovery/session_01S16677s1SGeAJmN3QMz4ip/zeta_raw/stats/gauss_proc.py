"""Stationary smooth Gaussian process sample: the 1D periodized squared-exponential field
on a circle of length L = 2 pi M with Fourier weights q_n = exp(-w^2 n^2/2)/sum_j exp(-w^2 j^2/2),
w = 2 pi / L = 1/M (the 1D case of the project's periodized Gaussian field; covariance is the
periodization of exp(-tau^2/2)).

g(x) = sqrt(q0) xi_0 + sum_{n>=1} sqrt(2 q_n) (xi_n cos(w n x) + eta_n sin(w n x)), Var g = 1.
F(x) = int_0^x g (exact spectral antiderivative + linear part from the n=0 mode).
Zeros of g located by cubic Hermite interpolation of (g, g') on a fine FFT grid;
F at zeros by integrating the same Hermite cubic from the left grid node.
Output per lobe (between consecutive zeros): signed area, gap (unfolded: mean spacing 1),
peak |g| on the grid, sample id.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def one_sample(seed, M=1000, Ng=2 ** 19):
    rng = np.random.default_rng(seed)
    L = 2 * np.pi * M
    w = 1.0 / M
    kmax = Ng // 2
    k = np.arange(kmax + 1)
    q = np.exp(-0.5 * (w * k) ** 2)
    Zs = q[0] + 2 * q[1:].sum()
    q = q / Zs
    lam2 = 2 * np.sum(q[1:] * (w * k[1:]) ** 2)
    xi = rng.standard_normal(kmax + 1); eta = rng.standard_normal(kmax + 1)
    G = np.zeros(kmax + 1, dtype=complex)
    G[0] = Ng * np.sqrt(q[0]) * xi[0]
    G[1:] = (Ng / 2) * np.sqrt(2 * q[1:]) * (xi[1:] - 1j * eta[1:])
    G[-1] = 0.0  # Nyquist (weight ~ 0 anyway)
    dx = L / Ng
    x = dx * np.arange(Ng)
    g = np.fft.irfft(G, n=Ng)
    dg = np.fft.irfft(G * (1j * w * k), n=Ng)
    Fh = np.zeros_like(G)
    Fh[1:] = G[1:] / (1j * w * k[1:])
    c0 = np.sqrt(q[0]) * xi[0]
    F = np.fft.irfft(Fh, n=Ng) + c0 * x
    # sign changes (no wrap: treat [0, L) as an interval)
    s = np.sign(g)
    i = np.nonzero(s[:-1] * s[1:] < 0)[0]
    # cubic Hermite on [0,1]: p(t) = h00 g0 + h10 dx d0 + h01 g1 + h11 dx d1
    g0, g1, d0, d1 = g[i], g[i + 1], dg[i] * dx, dg[i + 1] * dx

    def herm(t):
        t2 = t * t; t3 = t2 * t
        return (2 * t3 - 3 * t2 + 1) * g0 + (t3 - 2 * t2 + t) * d0 + (-2 * t3 + 3 * t2) * g1 + (t3 - t2) * d1

    lo = np.zeros(i.size); hi = np.ones(i.size)
    flo = g0.copy()
    for _ in range(50):  # bisection on the Hermite cubic (robust)
        mid = 0.5 * (lo + hi)
        fm = herm(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    tz = 0.5 * (lo + hi)
    # integral of Hermite cubic from 0 to tz (times dx)
    t = tz; t2 = t * t; t3 = t2 * t; t4 = t3 * t
    I = (t4 / 2 - t3 + t) * g0 + (t4 / 4 - 2 * t3 / 3 + t2 / 2) * d0 + (-t4 / 2 + t3) * g1 + (t4 / 4 - t3 / 3) * d1
    Fz = F[i] + dx * I
    xz = x[i] + dx * tz
    area = np.diff(Fz)
    gap = np.diff(xz) * np.sqrt(lam2) / np.pi  # unfolded (Rice density sqrt(lam2)/pi)
    # peak |g| on grid nodes strictly inside each lobe
    ag = np.abs(g)
    starts = i[:-1] + 1; ends = i[1:] + 1
    peak = np.maximum.reduceat(ag, np.r_[starts, ends[-1]])[:-1]
    # reduceat with consecutive starts gives max over [starts[j], starts[j+1]) = lobe nodes
    return area, gap, peak, dict(lam2=lam2, n_zeros=int(i.size), dx=dx,
                                 hidden_cells=int(np.sum((s[1:-1] == s[:-2]) & (s[1:-1] == s[2:]) &
                                                         (ag[1:-1] < ag[:-2]) & (ag[1:-1] <= ag[2:]) &
                                                         (ag[1:-1] < 0.5 * np.abs(dg[1:-1]) * dx))))


def worker(seeds):
    out = []
    for sd in seeds:
        out.append(one_sample(sd))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t0 = time.time()
    seeds = [10_000 + j for j in range(nsamp)]
    chunks = [seeds[j::procs * 4] for j in range(procs * 4)]
    with Pool(procs) as pool:
        res = pool.map(worker, chunks)
    flat = [r for ch in res for r in ch]
    area = np.concatenate([r[0] for r in flat])
    gap = np.concatenate([r[1] for r in flat])
    peak = np.concatenate([r[2] for r in flat])
    sid = np.concatenate([np.full(r[0].size, j) for j, r in enumerate(flat)])
    info = dict(n_samples=nsamp, M=1000, L="2*pi*1000", grid=2 ** 19, n_lobes=int(area.size),
                lam2=flat[0][3]["lam2"], mean_gap_unfolded=float(gap.mean()),
                sign_alternation_ok=bool(np.all((np.sign(area[1:]) == -np.sign(area[:-1])) | (sid[1:] != sid[:-1]))),
                possible_hidden_pairs_flagged=int(sum(r[3]["hidden_cells"] for r in flat)),
                seconds=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(HERE, "gauss.npz"), area=area, gap=gap, peak=peak, sid=sid, cid=sid)
    print(json.dumps(info, indent=1))
    with open(os.path.join(HERE, "gauss_info.json"), "w") as f:
        json.dump(info, f, indent=1)
