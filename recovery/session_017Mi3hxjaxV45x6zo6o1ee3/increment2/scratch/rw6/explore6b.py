# Exploration only: the m = 6 recursion through the last layer c = 1/120, with Owen-type kinds integrated by parts:
#   int_0^T y^j e^{-cy^2} K_{al,g}(y) dy:  j = 0 -> new kind ('KK', c, al, g);
#   j >= 1 -> -(1/2c) T^{j-1} e^{-cT^2} K(T) [kind ('eK', c, al, g)] + ((j-1)/2c) [j-2] + (1/2c) L_{j-1}(c + al, g).
import sys, time
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtw6/frontiers/c8_window_coefficient_d6_20261001")
import window_coefficient_d6 as W
from fractions import Fraction as Fr
t0 = time.time()
KLM = {}
def KL(c, al, g, j):
    key = (c, al, g, j)
    if key in KLM:
        return KLM[key]
    res = {}
    def add(kind, poly):
        res[kind] = W.lpadd(res.get(kind, []), poly)
    if j == 0:
        add(("KK", c, al, g), [Fr(1)])
    else:
        add(("eK", c, al, g), [Fr(0)] * (j - 1) + [Fr(-1) / (2 * c)])
        if j >= 2:
            for kind, pl in KL(c, al, g, j - 2).items():
                add(kind, [x * Fr(j - 1) / (2 * c) for x in pl])
        for kind, pl in W.L_dict(c + al, g, j - 1).items():
            add(kind, [x / (2 * c) for x in pl])
    KLM[key] = res
    return res
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
                lay = KL(c, kind[1], kind[2], j)
            else:
                raise ValueError(kind)
            base = mono[:-1]
            for k2, pl in lay.items():
                acc = out.setdefault(k2, {})
                for n, x in enumerate(pl):
                    if x:
                        mm = list(base); mm[-1] += n; mm = tuple(mm)
                        acc[mm] = acc.get(mm, Fr(0)) + coef * x
    return {k: {mm: v for mm, v in p.items() if v} for k, p in out.items()}
m = 6
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
for i in range(m):
    for j in range(i + 1, m):
        P = W.pmul(P, W.padd(mus[i], W.pscale(mus[j], Fr(-1))))
U, V = {}, {}
for mono, c in P.items():
    Bk, Ck = W.Jk_closed(mono[-1])
    base = mono[:-1]
    U[base] = U.get(base, Fr(0)) + c * Bk
    for n, x in enumerate(Ck):
        if x:
            mm = list(base); mm[-1] += n; mm = tuple(mm)
            V[mm] = V.get(mm, Fr(0)) + c * x
pieces = {("1",): {k: v for k, v in U.items() if v}, ("e", Fr(1, 8)): {k: v for k, v in V.items() if v}}
nvc = nv - 1
for c in (Fr(1, 24), Fr(1, 48), Fr(1, 80), Fr(1, 120)):
    pieces = layer_ext(pieces, nvc, c)
    nvc -= 1
    print("after c =", c, "zero kinds:", sorted(str(k) for k, p in pieces.items() if not p), "%.1fs" % (time.time() - t0), flush=True)
print("final nonzero kinds:")
for k, p in sorted(pieces.items(), key=lambda kv: str(kv[0])):
    if p:
        lst = {}
        for mono, v in p.items():
            lst[mono[0]] = lst.get(mono[0], Fr(0)) + v
        print("  ", k, "degree", max(lst), "lead", lst[max(lst)])
print("gamma_out", Fr(3, 4 * m * (m + 3)))
