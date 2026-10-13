# Exploration only (not part of any packet): the m = 7 recursion (d = 8) through c = 1/168, with ('eK', c', al, g) entering a layer
# integrated as KL_dict(c + c', al, g, j); a surviving nested kind is reported, not refused.
import sys, time, resource
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtw7/frontiers/c8_window_coefficient_d7_20261001")
import window_coefficient_d7 as W
from fractions import Fraction as Fr
t0 = time.time()
m = 7
nv = m
def lin(i, j, den):
    co = [Fr(0)] * nv
    co[i] += Fr(1, den)
    if j is not None:
        co[j] -= Fr(1, den)
    return W.plin(co)
a = lin(0, 1, m)
ps = [lin(k, k + 1 if k + 1 < m else None, m - k) for k in range(1, m)]
mus, run = [W.pscale(a, Fr(-1))], a
for p in ps:
    run = W.padd(run, p)
    mus.append(W.pscale(run, Fr(-1)))
P = {(0,) * nv: Fr(1)}
for mu in mus:
    P = W.pmul(P, W.pmul(mu, mu))
print("prod mu^2 terms", len(P), "%.1fs" % (time.time() - t0), flush=True)
for i in range(m):
    for j in range(i + 1, m):
        P = W.pmul(P, W.padd(mus[i], W.pscale(mus[j], Fr(-1))))
    print("  vandermonde row", i, "terms", len(P), "%.1fs" % (time.time() - t0), "maxrss MB %.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), flush=True)
U, V = {}, {}
for mono, c in P.items():
    Bk, Ck = W.Jk_closed(mono[-1])
    base = mono[:-1]
    U[base] = U.get(base, Fr(0)) + c * Bk
    for n, x in enumerate(Ck):
        if x:
            mm = list(base); mm[-1] += n; mm = tuple(mm)
            V[mm] = V.get(mm, Fr(0)) + c * x
del P
pieces = {("1",): {k: v for k, v in U.items() if v}, ("e", Fr(1, 8)): {k: v for k, v in V.items() if v}}
del U, V

def layer_ext(pieces, nv, c):
    out = {}
    for kind, poly in pieces.items():
        for mono, coef in poly.items():
            j = mono[-1]
            if kind[0] == "1":
                lay = W.moment_dict(c, j)
            elif kind[0] == "e":
                lay = W.moment_dict(c + kind[1], j)
            elif kind[0] == "E":
                lay = W.L_dict(c, kind[1], j)
            elif kind[0] == "eE":
                lay = W.L_dict(c + kind[1], kind[2], j)
            elif kind[0] == "K":
                lay = W.KL_dict(c, kind[1], kind[2], j)
            elif kind[0] == "eK":
                lay = W.KL_dict(c + kind[1], kind[2], kind[3], j)
            else:
                raise ValueError("nested kind %r entered a layer" % (kind,))
            base = mono[:-1]
            for k2, pl in lay.items():
                acc = out.setdefault(k2, {})
                for n, x in enumerate(pl):
                    if x:
                        mm = list(base); mm[-1] += n; mm = tuple(mm)
                        acc[mm] = acc.get(mm, Fr(0)) + coef * x
    return {k: {mm: v for mm, v in p.items() if v} for k, p in out.items()}

nvc = nv - 1
for c in (Fr(1, 24), Fr(1, 48), Fr(1, 80), Fr(1, 120), Fr(1, 168)):
    pieces = layer_ext(pieces, nvc, c)
    nvc -= 1
    print("after c =", c, "zero kinds:", sorted(str(k) for k, p in pieces.items() if not p), "%.1fs" % (time.time() - t0), flush=True)
    print("   nonzero:", sorted(str(k) for k, p in pieces.items() if p), flush=True)
print("gamma_out", Fr(3, 4 * m * (m + 3)))
