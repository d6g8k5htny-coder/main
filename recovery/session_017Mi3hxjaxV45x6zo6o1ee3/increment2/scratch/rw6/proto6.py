"""Prototype: total-trace reduction for the windowed birth integral at general m; layer-by-layer exact integration.
Reports the surviving kinds after each layer (in particular Owen-type K kinds)."""
import sys, time
from fractions import Fraction as Fr
MUT = None

def padd(u, v):
    w = dict(u)
    for k, c in v.items():
        w[k] = w.get(k, Fr(0)) + c
    return {k: c for k, c in w.items() if c != 0}
def pmul(u, v):
    w = {}
    for k1, c1 in u.items():
        for k2, c2 in v.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            w[k] = w.get(k, Fr(0)) + c1 * c2
    return {k: c for k, c in w.items() if c != 0}
def pscale(u, c):
    return {k: v * c for k, v in u.items() if v * c != 0}
def pvar(i, n):
    e = [0] * n; e[i] = 1
    return {tuple(e): Fr(1)}
def plin(coeffs):
    out = {}
    for i, c in enumerate(coeffs):
        if c:
            out = padd(out, pscale(pvar(i, len(coeffs)), Fr(c)))
    return out
def lpadd(u, v):
    n = max(len(u), len(v))
    return [(u[i] if i < len(u) else Fr(0)) + (v[i] if i < len(v) else Fr(0)) for i in range(n)]

def Jk_closed(k):
    if k % 2 != 1:
        raise ValueError("even power of q")
    B, C, j = Fr(4), [Fr(-4)], 1
    while j < k:
        C2 = [4 * (j + 1) * c for c in C] + [Fr(0)] * (j + 2 - len(C))
        C2[j + 1] += -4
        B, C, j = 4 * (j + 1) * B, C2, j + 2
    return B, C

MOM = {}
def moments(gamma, nmax):
    g = Fr(gamma)
    key = (g, nmax)
    if key in MOM: return MOM[key]
    M = [(Fr(0), [], Fr(1)), (1 / (2 * g), [-1 / (2 * g)], Fr(0))]
    for j in range(nmax - 1):
        a, r, b = M[j]
        f = Fr(j + 1)
        r2 = [x * f / (2 * g) for x in r] + [Fr(0)] * (j + 2 - len(r))
        r2[j + 1] += -1 / (2 * g)
        M.append((f * a / (2 * g), r2, f * b / (2 * g)))
    MOM[key] = M
    return M

def moment_dict(gamma, j):
    a_, r_, b_ = moments(gamma, max(j, 1))[j]
    out = {}
    if a_: out[("1",)] = [a_]
    if any(r_): out[("e", Fr(gamma))] = list(r_)
    if b_: out[("E", Fr(gamma))] = [b_]
    return out

L_MEMO = {}
def L_dict(al, g, j):
    key = (al, g, j)
    if key in L_MEMO: return L_MEMO[key]
    if j == 0:
        res = {("K", al, g): [Fr(1)]}
    else:
        res = {}
        def add(kind, poly):
            res[kind] = lpadd(res.get(kind, []), poly)
        add(("eE", al, g), [Fr(0)] * (j - 1) + [-1 / (2 * al)])
        if j >= 2:
            for kind, pl in L_dict(al, g, j - 2).items():
                add(kind, [x * Fr(j - 1) / (2 * al) for x in pl])
        for kind, pl in moment_dict(al + g, j - 1).items():
            add(kind, [x / (2 * al) for x in pl])
    L_MEMO[key] = res
    return res

def layer(pieces, nv, c):
    """pieces: kind -> poly in nv variables (T, x1, ..., x_{nv-1}); integrate the last variable y = x_{nv-1} over (0, x_{nv-2})
    against e^{-c y^2}. Returns kind -> poly in nv-1 variables."""
    out = {}
    for kind, poly in pieces.items():
        # group by y-power
        for mono, coef in poly.items():
            j = mono[-1]
            if kind[0] == "1":
                lay = moment_dict(c, j)
            elif kind[0] == "e":
                lay = moment_dict(c + kind[1], j)
            elif kind[0] == "E":
                lay = L_dict(c, kind[1], j)
            elif kind[0] == "eE":
                lay = L_dict(c + kind[1], kind[2], j)
            else:
                raise ValueError("K kind entering a layer: %r" % (kind,))
            base = mono[:-1]
            for k2, pl in lay.items():
                acc = out.setdefault(k2, {})
                for n, x in enumerate(pl):
                    if x:
                        mm = list(base); mm[-1] += n            # power of the new upper limit x_{nv-2}
                        mm = tuple(mm)
                        acc[mm] = acc.get(mm, Fr(0)) + coef * x
    return {k: {mm: v for mm, v in p.items() if v} for k, p in out.items()}

def build(m):
    nv = m
    T = pvar(0, nv)
    xs = [pvar(i, nv) for i in range(nv)]                  # x0 = T, x1..x_{m-1}; x_{m-1} = q
    # a = (T - x1)/m, p_k = (x_k - x_{k+1})/(m-k), x_m = 0
    def lin(i, j, den):
        co = [Fr(0)] * nv; co[i] += Fr(1, den)
        if j is not None: co[j] -= Fr(1, den)
        return plin(co)
    a = lin(0, 1, m)
    ps = [lin(k, k + 1 if k + 1 < m else None, m - k) for k in range(1, m)]
    mus, run = [pscale(a, Fr(-1))], a
    for p in ps:
        run = padd(run, p)
        mus.append(pscale(run, Fr(-1)))
    P = {(0,) * nv: Fr(1)}
    for mu in mus:
        P = pmul(P, pmul(mu, mu))
    for i in range(m):
        for j in range(i + 1, m):
            P = pmul(P, padd(mus[i], pscale(mus[j], Fr(-1))))
    # check oddness in q
    if any(k[-1] % 2 == 0 for k in P):
        raise ValueError("even q power")
    # q-layer via Jk_closed: kinds '1' (B_k) and ('e', 1/8) (C_k(s) e^{-s^2/8}) in nv-1 vars
    U, V = {}, {}
    for mono, c in P.items():
        Bk, Ck = Jk_closed(mono[-1])
        base = mono[:-1]
        U[base] = U.get(base, Fr(0)) + c * Bk
        for n, x in enumerate(Ck):
            if x:
                mm = list(base); mm[-1] += n; mm = tuple(mm)
                V[mm] = V.get(mm, Fr(0)) + c * x
    pieces = {("1",): {k: v for k, v in U.items() if v}, ("e", Fr(1, 8)): {k: v for k, v in V.items() if v}}
    nvc = nv - 1
    tri = [Fr(1, 24), Fr(1, 48), Fr(1, 80), Fr(1, 120)]
    li = 0
    while nvc > 1:
        c = tri[li]
        t0 = time.time()
        pieces = layer(pieces, nvc, c)
        nvc -= 1; li += 1
        zero = [k for k, p in pieces.items() if not p]
        nz = {k: len(p) for k, p in pieces.items() if p}
        print("layer c=%s -> vars %d; nonzero kinds: %s; zero kinds: %s (%.1fs)" % (c, nvc, nz, zero, time.time() - t0), flush=True)
    return pieces

if __name__ == "__main__":
    m = int(sys.argv[1])
    pieces = build(m)
    for k, p in sorted(pieces.items(), key=lambda kv: str(kv[0])):
        print(k, "zero" if not p else "deg %d" % max(mm[0] for mm in p))
