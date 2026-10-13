"""Adaptive driver: initial grid over [-L sigma, L sigma]^3, depth-first bisection of boxes whose enclosure is wider
than tol (along the standardized-widest side), outward accumulation; the initial cells are dealt round-robin to
worker processes and the partial sums are combined in a fixed order (deterministic)."""
import sys, time, math
from multiprocessing import Pool
import fcore2 as F
from ia import isqrt, dn, up

def setup(k):
    sa2, sb2, sc2 = 2.0 / k, 2.0, 6.0 * k
    assert sa2 * k == 2.0 and sc2 / k == 6.0
    sig = (isqrt((sa2, sa2)), isqrt((sb2, sb2)), isqrt((sc2, sc2)))
    M = (F.Moments(sig[0], (sa2, sa2), 6), F.Moments(sig[1], (sb2, sb2), 2), F.Moments(sig[2], (sc2, sc2), 2))
    return sig, M

def work(args):
    k, which, tol, cells, maxdepth = args
    sig, (Ma, Mb, Mc) = setup(k)
    fn = {'V': F.V_box, 'H2': F.H2_box, 'H': F.H_box}[which]
    sd = (sig[0][1], sig[1][1], sig[2][1])
    lo = hi = 0.0; nleaf = 0; nfirst = 0
    for cell in cells:
        stack = [(cell, 0)]
        while stack:
            box, dep = stack.pop()
            I = fn(box, Ma, Mb, Mc)
            if I[1] - I[0] > tol and dep < maxdepth:
                w = [(box[i][1] - box[i][0]) / sd[i] for i in range(3)]
                d = w.index(max(w)); l_, h_ = box[d]; mid = 0.5 * (l_ + h_)
                for half in ((l_, mid), (mid, h_)):
                    nb = list(box); nb[d] = half; stack.append((tuple(nb), dep + 1))
            else:
                lo = dn(lo + I[0]); hi = up(hi + I[1]); nleaf += 1
    return lo, hi, nleaf

def run(k, which, tol, L=8.0, n0=8, nproc=4, maxdepth=60):
    sig, _ = setup(k)
    edges = [[-L * s[1] + 2 * L * s[1] * i / n0 for i in range(n0 + 1)] for s in sig]
    cells = [((edges[0][i], edges[0][i + 1]), (edges[1][j], edges[1][j + 1]), (edges[2][l], edges[2][l + 1]))
             for i in range(n0) for j in range(n0) for l in range(n0)]
    chunks = [cells[p::nproc] for p in range(nproc)]
    t = time.time()
    with Pool(nproc) as pool:
        res = pool.map(work, [(k, which, tol, ch, maxdepth) for ch in chunks])
    lo = hi = 0.0; n = 0
    for r in res:
        lo = dn(lo + r[0]); hi = up(hi + r[1]); n += r[2]
    s = 3 * k * k
    print('k=%g %s tol=%g leaves=%d  3k^2 E in [%.9f, %.9f] width %.3e  time %.0fs' %
          (k, which, tol, n, s * lo, s * hi, s * (hi - lo), time.time() - t)); sys.stdout.flush()
    return lo, hi, n

if __name__ == '__main__':
    k = float(sys.argv[1]); which = sys.argv[2]
    for tol in map(float, sys.argv[3:]):
        run(k, which, tol)
