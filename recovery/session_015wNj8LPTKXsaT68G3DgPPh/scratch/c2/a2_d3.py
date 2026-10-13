import sympy as sp, itertools, pickle, time
from sympy import QQ
from sympy.polys.rings import ring
import pseries as P
t0 = time.time()
NR = 6
Rg, r, b, k, l1, l2 = ring('r b k l1 l2', QQ)
def trunc(p, n=NR):
    return Rg({m: c for m, c in p.terms() if m[0] <= n})
def ser(s, upto=NR):
    return sum((Rg(QQ(c.numerator, c.denominator)) * r**i for i, c in enumerate(s.c[:upto + 1]) if c), Rg(0))
d = 3
X = lambda be, sg: P.deriv_at(be, sg, d)
forms = {}
for nm, sg in (('M', -1), ('S', 1)):
    for (i, j) in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        be = [0, 0, 0]; be[i] += 1; be[j] += 1
        forms[(nm, i, j)] = X(tuple(be), sg)
forms['a11'] = {(0, 2, 0): P.S.const(1)}; forms['a12'] = {(0, 1, 1): P.S.const(1)}; forms['a22'] = {(0, 0, 2): P.S.const(1)}
pin = P.Pinned(d, forms)
print('pinned', time.time() - t0, 'p =', pin.p)
p = pin.p
v = [b - k * r**3 / 2, -k * r**2, Rg(0), 12 * k] + [Rg(0)] * (p - 4)
mean = {u: trunc(sum((ser(pin.G[u][j]) * v[j] for j in range(p)), Rg(0))) for u in pin.names}
al = ['a11', 'a12', 'a22']
CAA = [[pin.C[(x, y)] for y in al] for x in al]
CAAinv = P.mat_inv(CAA)
detA = P.mat_det(CAA)
hn = [u for u in pin.names if u not in al]
B = {u: [sum((pin.C[(u, al[j])] * CAAinv[j][i] for j in range(3)), P.S()) for i in range(3)] for u in hn}
cp = {(u, w): ser(pin.C[(u, w)] - sum((B[u][i] * pin.C[(al[i], w)] for i in range(3)), P.S())) for u in hn for w in hn}
zA = [l1 - mean['a11'], -mean['a12'], l2 - mean['a22']]
aff = {u: trunc(mean[u] + sum((ser(B[u][i]) * zA[i] for i in range(3)), Rg(0))) for u in hn}
print('affine means', time.time() - t0)
def matchings(lst):
    if not lst:
        yield []; return
    f, rest = lst[0], lst[1:]
    for mm in matchings(rest):
        yield mm
    for i, y in enumerate(rest):
        for mm in matchings(rest[:i] + rest[i+1:]):
            yield [(f, y)] + mm
MT6 = list(matchings(list(range(6))))
perms = list(itertools.permutations(range(3)))
def sgn(q):
    return 1 if sum(1 for i in range(3) for j in range(i + 1, 3) if q[i] > q[j]) % 2 == 0 else -1
F = Rg(0)
for pp in perms:
    for qq in perms:
        fac = [('M', min(i, pp[i]), max(i, pp[i])) for i in range(3)] + [('S', min(i, qq[i]), max(i, qq[i])) for i in range(3)]
        tot = Rg(0)
        for mt in MT6:
            used = set(i for pr in mt for i in pr)
            c = Rg(1)
            for (i, j) in mt:
                c = trunc(c * cp[(fac[i], fac[j])])
                if not c:
                    break
            if not c:
                continue
            for i in range(6):
                if i not in used:
                    c = trunc(c * aff[fac[i]])
            tot += c
        F += sgn(pp) * sgn(qq) * tot
F = trunc(F)
print('E[det det | A] built', time.time() - t0, 'terms', len(F.terms()))
# exponent pieces: pin density exponent q = v^T Sinv v, A density exponent Q = zA^T CAAinv zA
q = trunc(sum((v[i] * ser(pin.Sinv[i][j]) * v[j] for i in range(p) for j in range(p)), Rg(0)))
Q = trunc(sum((zA[i] * ser(CAAinv[i][j]) * zA[j] for i in range(3) for j in range(3)), Rg(0)))
pickle.dump({'F': F.as_expr(), 'q': q.as_expr(), 'Q': Q.as_expr(), 'detS': [str(c) for c in pin.detS.c],
             'detA': [str(c) for c in detA.c]}, open('a2_d3_raw.pkl', 'wb'))
print('saved', time.time() - t0)
