"""CUE sample: Haar unitary U (N x N, N even), eigenangles theta_1<...<theta_N.
Real rotated characteristic polynomial on the circle (Hardy-Z analog):
    Z_U(theta) = prod_j 2 sin((theta - theta_j)/2),
evaluated directly as a product (no logarithms; |Z_U| <= 2^N fits in double for N <= 1000).
For N even Z_U is 2pi-periodic. Lobes: (theta_k, theta_{k+1}), k=1..N with theta_{N+1}=theta_1+2pi.
Lobe integrals by 16-point Gauss-Legendre (checked against 32 points on a subset).
Unfolding: gap * N/(2pi). The amplitude is left raw (no normalization); the analysis normalizes
bar lengths by their median and separately reports a local-amplitude diagnostic.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GLX, GLW = np.polynomial.legendre.leggauss(16)
GLX32, GLW32 = np.polynomial.legendre.leggauss(32)


def haar_angles(rng, B, N):
    A = (rng.standard_normal((B, N, N)) + 1j * rng.standard_normal((B, N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(A)
    d = np.diagonal(R, axis1=1, axis2=2)
    Q = Q * (d / np.abs(d))[:, None, :]
    ev = np.linalg.eigvals(Q)
    return np.sort(np.mod(np.angle(ev), 2 * np.pi), axis=1)


def lobes(th, glx=GLX, glw=GLW):
    """th: (B, N) sorted angles. Returns signed lobe integrals (B,N), gaps (B,N), peaks (B,N)."""
    B, N = th.shape
    right = np.concatenate([th[:, 1:], th[:, :1] + 2 * np.pi], axis=1)
    mid = 0.5 * (th + right); half = 0.5 * (right - th)
    nodes = mid[:, :, None] + half[:, :, None] * np.concatenate([glx, [0.0]])[None, None, :]  # (B,N,n+1)
    # product over j of 2 sin((x - th_j)/2)
    diff = nodes[:, :, :, None] - th[:, None, None, :]  # (B,N,n+1,N)
    val = np.prod(2.0 * np.sin(0.5 * diff), axis=3)
    area = half * (val[:, :, :-1] @ glw)
    peak = np.max(np.abs(val), axis=2)
    return area, right - th, peak


def worker(args):
    seed, B, N, nb = args
    rng = np.random.default_rng(seed)
    out_a, out_g, out_p = [], [], []
    for _ in range(nb):
        th = haar_angles(rng, B, N)
        a, g, p = lobes(th)
        out_a.append(a); out_g.append(g); out_p.append(p)
    return np.concatenate(out_a), np.concatenate(out_g), np.concatenate(out_p)


if __name__ == "__main__":
    from multiprocessing import Pool
    N = int(sys.argv[1]); nmat = int(sys.argv[2]); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    t0 = time.time()
    # quadrature check
    rng = np.random.default_rng(1)
    th = haar_angles(rng, 20, N)
    a16, _, _ = lobes(th)
    a32, _, _ = lobes(th, GLX32, GLW32)
    qerr = float(np.max(np.abs(a16 - a32) / np.abs(a32)))
    B = max(1, min(256, 4_000_000 // (N * N * 17)))
    nbatch = int(np.ceil(nmat / B))
    per = int(np.ceil(nbatch / (procs * 4)))
    jobs = [(777 + j, B, N, per) for j in range(procs * 4)]
    with Pool(procs) as pool:
        res = pool.map(worker, jobs)
    area = np.concatenate([r[0] for r in res]); gap = np.concatenate([r[1] for r in res])
    peak = np.concatenate([r[2] for r in res])
    M = area.shape[0]
    sid = np.repeat(np.arange(M), N)
    area = area.ravel(); gap = gap.ravel() * N / (2 * np.pi); peak = peak.ravel()
    info = dict(N=N, n_matrices=int(M), n_lobes=int(area.size), quad_rel_err_16_vs_32=qerr,
                sign_alternation_ok=bool(np.all((np.sign(area[1:]) == -np.sign(area[:-1])) | (sid[1:] != sid[:-1]))),
                mean_gap_unfolded=float(gap.mean()), seconds=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(HERE, f"cue{N}.npz"), area=area, gap=gap, peak=peak, sid=sid, cid=sid)
    print(json.dumps(info, indent=1))
    with open(os.path.join(HERE, f"cue{N}_info.json"), "w") as f:
        json.dump(info, f, indent=1)
