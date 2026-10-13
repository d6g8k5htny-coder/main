import sys, math
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/rw4")
from fractions import Fraction as Fr
import proto as P   # exact polynomial helpers, Jk_closed, moments, lpadd, lpmul
padd, pmul, pscale, pvar, plin = P.padd, P.pmul, P.pscale, P.pvar, P.plin

# m = 4: T = 4a + 3p1 + 2p2 + p3, w = 3p1 + 2p2 + p3, s = 2p2 + p3, q = p3
nv = 4
T, w, s, q = (pvar(i, nv) for i in range(4))
a  = plin([Fr(1, 4), Fr(-1, 4), 0, 0])
p1 = plin([0, Fr(1, 3), Fr(-1, 3), 0])
p2 = plin([0, 0, Fr(1, 2), Fr(-1, 2)])
p3 = q
mus = [pscale(a, -1)]
acc = a
for p in (p1, p2, p3):
    acc = padd(acc, p); mus.append(pscale(acc, -1))
# exponent check: |mu|^2/4 - T^2/28 == 3T^2/112 + w^2/48 + s^2/24 + q^2/8
n2 = {}
for mu in mus: n2 = padd(n2, pmul(mu, mu))
lhs = padd(pscale(n2, Fr(1, 4)), pscale(pmul(T, T), Fr(-1, 28)))
rhs = padd(padd(pscale(pmul(T, T), Fr(3, 112)), pscale(pmul(w, w), Fr(1, 48))), padd(pscale(pmul(s, s), Fr(1, 24)), pscale(pmul(q, q), Fr(1, 8))))
print("exponent decouples:", lhs == rhs)
Pp = {(0, 0, 0, 0): Fr(1)}
for mu in mus: Pp = pmul(Pp, pmul(mu, mu))
for i in range(4):
    for j in range(i + 1, 4):
        Pp = pmul(Pp, padd(mus[i], pscale(mus[j], -1)))   # mus[i] > mus[j]
print("odd q only:", all(k[3] % 2 == 1 for k in Pp), "terms", len(Pp))
# q-layer
U, V = {}, {}
for (i, j, k, l), c in Pp.items():
    Bk, Ck = P.Jk_closed(l)
    U = padd(U, {(i, j, k): c * Bk})
    V = padd(V, pmul({(i, j, k): c}, {(0, 0, n): x for n, x in enumerate(Ck) if x}))
# s-layer over (0, w): moments in s with gamma 1/24 (U) and 1/6 (V) -> pieces in (T, w): kinds ('1'), ('e', g) [e^{-g w^2}], ('E', g) [E_g(w)]
pieces = {}
def acc2(key, poly):   # poly dict in (T, w)
    pieces[key] = padd(pieces.get(key, {}), poly)
for W_, gam in ((U, Fr(1, 24)), (V, Fr(1, 6))):
    Ms = P.moments(gam, max(k for (_, _, k) in W_))
    for (i, j, k), c in W_.items():
        a_, r_, b_ = Ms[k]
        base = {(i, j): c}
        if a_: acc2(("1",), pscale(base, a_))
        r = {(0, n): x for n, x in enumerate(r_) if x}
        if r: acc2(("e", gam), pmul(base, r))
        if b_: acc2(("E", gam), pscale(base, b_))
for k, v in pieces.items(): print("s-layer piece", k, len(v), "max w-degree", max(e[1] for e in v))
# parity in w of each piece (as functions of w with the e^{-w^2/48} factor)
for k, v in pieces.items():
    par = set(e[1] % 2 for e in v)
    print(k, "w-parities", par)
