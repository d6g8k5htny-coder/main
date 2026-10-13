# Exploration only (not part of the packet): does the m = 6 recursion meet a surviving Owen-type kind before the last layer?
import sys, time
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtw6/frontiers/c8_window_coefficient_d6_20261001")
import window_coefficient_d6 as W
from fractions import Fraction as Fr
t0 = time.time()
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
print("P terms", len(P), "%.1fs" % (time.time() - t0), flush=True)
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
    surv = sorted(str(k) for k, p in pieces.items() if p and k[0] == "K")
    print("before layer c =", c, "surviving Owen-type kinds:", surv, flush=True)
    if surv:
        print("STOP: Owen-type kinds would enter the c =", c, "layer", flush=True)
        break
    pieces = W.layer(pieces, nvc, c)
    nvc -= 1
    print("after c =", c, "zero kinds:", sorted(str(k) for k, p in pieces.items() if not p), "%.1fs" % (time.time() - t0), flush=True)
else:
    print("final kinds", sorted(str(k) for k, p in pieces.items() if p))
