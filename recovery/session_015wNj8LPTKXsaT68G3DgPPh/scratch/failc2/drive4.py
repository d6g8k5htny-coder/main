"""Adaptive 4D driver for I_k (k in [1/2, 2] dyadic, (a, beta, c) over [-L sigma, L sigma]^3)."""
import sys, time
from multiprocessing import Pool
import fcore2 as F, fcore4 as G
from ia import isqrt, dn, up

_ST = {}

def _init(elem):
    sig = [isqrt((2.0, 2.0)), isqrt((2.0, 2.0)), isqrt((6.0, 6.0))]
    _ST['M'] = (F.Moments(sig[0], (2.0, 2.0), 6), F.Moments(sig[1], (2.0, 2.0), 2), F.Moments(sig[2], (6.0, 6.0), 2))
    _ST['K'] = G.KMoments(elem)
    _ST['sd'] = (sig[0][1], sig[1][1], sig[2][1])

def work(args):
    tol, cells, sk, maxdepth = args
    Ma, Mb, Mc = _ST['M']; Km = _ST['K']
    sd = _ST['sd'] + (sk,)
    kmin = Km.width
    lo = hi = 0.0; nleaf = 0
    for cell in cells:
        stack = [(cell, 0)]
        while stack:
            box, dep = stack.pop()
            I = G.H_box4(box, Ma, Mb, Mc, Km)
            if I[1] - I[0] > tol and dep < maxdepth:
                w = [(box[i][1] - box[i][0]) / sd[i] for i in range(4)]
                if box[3][1] - box[3][0] <= kmin:
                    w[3] = -1.0
                d = w.index(max(w)); l_, h_ = box[d]; mid = 0.5 * (l_ + h_)
                for half in ((l_, mid), (mid, h_)):
                    nb = list(box); nb[d] = half; stack.append((tuple(nb), dep + 1))
            else:
                lo = dn(lo + I[0]); hi = up(hi + I[1]); nleaf += 1
    return lo, hi, nleaf

def run(tol, sk=0.1, L=8.0, n0=8, nk0=8, nproc=4, maxdepth=80, elem=None):
    sig = [1.4142135623730951, 1.4142135623730951, 2.449489742783178]
    sigu = [isqrt((2.0, 2.0))[1], isqrt((2.0, 2.0))[1], isqrt((6.0, 6.0))[1]]
    edges = [[-L * s + 2 * L * s * i / n0 for i in range(n0 + 1)] for s in sigu]
    kw = 1.5 / nk0
    kedges = [0.5 + kw * j for j in range(nk0 + 1)]
    cells = [((edges[0][i], edges[0][i + 1]), (edges[1][j], edges[1][j + 1]), (edges[2][l], edges[2][l + 1]),
              (kedges[q], kedges[q + 1])) for q in range(nk0) for i in range(n0) for j in range(n0) for l in range(n0)]
    chunks = [cells[p::nproc] for p in range(nproc)]
    t = time.time()
    with Pool(nproc, initializer=_init, initargs=(elem,)) as pool:
        res = pool.map(work, [(tol, ch, sk, maxdepth) for ch in chunks])
    lo = hi = 0.0; n = 0
    for r in res:
        lo = dn(lo + r[0]); hi = up(hi + r[1]); n += r[2]
    print('I_k box part: tol=%g sk=%g leaves=%d  in [%.9f, %.9f] width %.3e  time %.0fs' %
          (tol, sk, n, lo, hi, hi - lo, time.time() - t)); sys.stdout.flush()
    return lo, hi, n

if __name__ == '__main__':
    t = time.time(); elem = G.elementary_kmoments(); print('k-moments %.1fs' % (time.time() - t))
    for arg in sys.argv[1:]:
        tol, sk = map(float, arg.split(','))
        run(tol, sk, elem=elem)
