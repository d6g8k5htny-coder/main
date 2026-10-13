"""Exact closed form of the SIDE24 cone moments D_1, D_2, D_3 by exact integration (exploratory)."""
import math, json, sys
from fractions import Fraction as Fr
from functools import lru_cache

# ---------- surds: (c, r) means c * sqrt(r), r squarefree positive
def sqf(n):
    """n positive int -> (k, r) with sqrt(n) = k sqrt(r), r squarefree."""
    k, r, p = 1, 1, 2
    m = n
    while p * p <= m:
        while m % (p * p) == 0:
            m //= p * p
            k *= p
        if m % p == 0:
            m //= p
            r *= p
        p += 1
    r *= m
    return k, r

def surd_mul(u, v):
    k, r = sqf(u[1] * v[1])
    return (u[0] * v[0] * k, r)

def surd_sqrt_rational(q):
    q = Fr(q)
    assert q > 0
    k, r = sqf(q.numerator * q.denominator)
    return (Fr(k, q.denominator), r)

def surd_inv(u):
    c, r = u
    return (1 / (c * r), r)

def surd_sq_value(u):
    return u[0] * u[0] * u[1]

# ---------- algebra: dict {(r, twok, atom): Fraction}: c sqrt(r) pi^(twok/2) ATOM
ATOMS = [None]           # index 0: the number 1; others: (c, r) with atom = arctan(c sqrt r), 0 < c sqrt r <= 1

def E(c=1, r=1, twok=0, atom=0):
    return {(r, twok, atom): Fr(c)} if c != 0 else {}

def add(u, v):
    w = dict(u)
    for k, c in v.items():
        w[k] = w.get(k, Fr(0)) + c
    return {k: c for k, c in w.items() if c != 0}

def scale(u, c):
    c = Fr(c)
    return {k: v * c for k, v in u.items() if v * c != 0}

def mul_surd(u, s):
    out = {}
    for (r, twok, atom), c in u.items():
        k, r2 = sqf(r * s[1])
        key = (r2, twok, atom)
        out[key] = out.get(key, Fr(0)) + c * s[0] * k
    return {k: c for k, c in out.items() if c != 0}

def mul_pi_half(u, j):
    return {(r, twok + j, atom): c for (r, twok, atom), c in u.items()}

def mul(u, v):
    out = {}
    for (r1, k1, a1), c1 in u.items():
        for (r2, k2, a2), c2 in v.items():
            if a1 and a2:
                raise ValueError("product of two atoms")
            k, r = sqf(r1 * r2)
            key = (r, k1 + k2, a1 or a2)
            out[key] = out.get(key, Fr(0)) + c1 * c2 * k
    return {k: c for k, c in out.items() if c != 0}

def inv_monomial(u):
    (r, twok, atom), c = next(iter(u.items()))
    assert len(u) == 1 and atom == 0
    return E(1 / (c * r), r, -twok)

def atan_elem(t):
    """arctan(c sqrt r) as an element; rational multiples of pi recognised, else an atom with argument <= 1."""
    c, r = t
    if c == 0:
        return {}
    if c < 0:
        return scale(atan_elem((-c, r)), -1)
    if surd_sq_value(t) > 1:                       # arctan t = pi/2 - arctan(1/t)
        return add(E(Fr(1, 2), 1, 2), scale(atan_elem(surd_inv(t)), -1))
    if (c, r) == (1, 1):
        return E(Fr(1, 4), 1, 2)
    if r == 3 and c == Fr(1, 3):
        return E(Fr(1, 6), 1, 2)
    for i, a in enumerate(ATOMS):
        if a == (c, r):
            return E(1, 1, 0, i)
    ATOMS.append((c, r))
    return E(1, 1, 0, len(ATOMS) - 1)

# ---------- moments  M(n, lam, slopes) = int_0^inf s^n e^{-lam s^2} prod erf(u s) ds
@lru_cache(maxsize=None)
def M(n, lam, slopes):
    lam = Fr(lam)
    if n == 0:
        inv_sqrt_lam = surd_sqrt_rational(1 / lam)
        if not slopes:
            return mul_pi_half(mul_surd(E(Fr(1, 2)), inv_sqrt_lam), 1)
        if len(slopes) == 1:
            t = surd_mul(slopes[0], inv_sqrt_lam)
            return mul_pi_half(mul_surd(atan_elem(t), inv_sqrt_lam), -1)
        if len(slopes) == 2:
            u, v = slopes
            num = surd_mul(u, v)
            den = surd_sqrt_rational(lam * (lam + surd_sq_value(u) + surd_sq_value(v)))
            t = surd_mul(num, surd_inv(den))
            return mul_pi_half(mul_surd(atan_elem(t), inv_sqrt_lam), -1)
        raise ValueError("at most two erf factors")
    if n == 1 and not slopes:
        return E(1 / (2 * lam))
    out = {}
    if n >= 2:
        out = add(out, scale(M(n - 2, lam, slopes), n - 1))
    for i, u in enumerate(slopes):
        rest = slopes[:i] + slopes[i + 1:]
        term = M(n - 1, lam + surd_sq_value(u), rest)
        out = add(out, mul_pi_half(mul_surd(term, (2 * u[0], u[1])), -1))
    return scale(out, 1 / (2 * lam))

# ---------- polynomial helpers (copied from cone_moment_d4.py)
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
def pconst(c, nvars):
    return {(0,) * nvars: Fr(c)} if c != 0 else {}
def pvar(i, nvars):
    e = [0] * nvars; e[i] = 1
    return {tuple(e): Fr(1)}
def ppow(u, n, nvars):
    r = pconst(1, nvars)
    for _ in range(n):
        r = pmul(r, u)
    return r
def psubst(u, images, nvars_out):
    out = {}
    for k, c in u.items():
        term = pconst(c, nvars_out)
        for i, e in enumerate(k):
            if e:
                term = pmul(term, ppow(images[i], e, nvars_out))
        out = padd(out, term)
    return out
def univariate(u):
    if not u:
        return [Fr(0)]
    deg = max(k[0] for k in u)
    return [u.get((n,), Fr(0)) for n in range(deg + 1)]

def reduce_sector(m):
    nv = m
    a = pvar(0, nv)
    mus = []
    acc = a
    for i in range(m):
        if i > 0:
            acc = padd(acc, pvar(i, nv))
        mus.append(pscale(acc, Fr(-1)))
    poly = pconst(1, nv)
    for mu in mus:
        poly = pmul(poly, pmul(mu, mu))
    for i in range(m):
        for j in range(i + 1, m):
            poly = pmul(poly, padd(mus[i], pscale(mus[j], Fr(-1))))
    S = {}; norm2 = {}
    for mu in mus:
        S = padd(S, mu); norm2 = padd(norm2, pmul(mu, mu))
    quad = padd(pscale(norm2, Fr(1, 4)), pscale(pmul(S, S), -Fr(1, 4 * (m + 3))))
    return poly, quad

def a_integrated_terms(m):
    poly, quad = reduce_sector(m)
    nv = m
    alpha = quad.get((2,) + (0,) * (nv - 1), Fr(0))
    beta = {k[1:]: c for k, c in quad.items() if k[0] == 1}
    rest = {k[1:]: c for k, c in quad.items() if k[0] == 0}
    nvr = nv - 1
    P = [pconst(1, nvr), pscale(beta, Fr(-1, 2) / alpha)]
    R = [{}, pconst(Fr(1, 2) / alpha, nvr)]
    deg_a = max(k[0] for k in poly)
    for n in range(1, deg_a):
        P.append(pscale(padd(pscale(P[n - 1], Fr(n)), pscale(pmul(beta, P[n]), Fr(-1))), Fr(1, 2) / alpha))
        R.append(pscale(padd(pscale(R[n - 1], Fr(n)), pscale(pmul(beta, R[n]), Fr(-1))), Fr(1, 2) / alpha))
    PE, PR = {}, {}
    for k, c in poly.items():
        n = k[0]
        w = {k[1:]: c}
        PE = padd(PE, pmul(w, P[n])); PR = padd(PR, pmul(w, R[n]))
    expE = padd(rest, pscale(pmul(beta, beta), Fr(-1, 4) / alpha))
    return alpha, beta, PE, PR, expE, rest

# ---------- exact constants
def gamma_half(twice):
    """Gamma(twice/2) as an element."""
    if twice % 2 == 0:
        return E(math.factorial(twice // 2 - 1))
    k = (twice - 1) // 2
    return E(Fr(math.factorial(2 * k), 4 ** k * math.factorial(k)), 1, 1)

def mehta_Z(m):
    prod = E(1)
    for j in range(1, m + 1):
        prod = mul(prod, gamma_half(j + 2))
    pw = Fr(m, 2) + Fr(m * (m - 1), 4)
    two_pow = E(2 ** int(pw), 2 if pw.denominator == 2 else 1)
    twopi_pow = mul_pi_half(E(2 ** (m // 2), 2 if m % 2 else 1), m)   # (2 pi)^{m/2}
    g32 = gamma_half(3)
    g32m = E(1)
    for _ in range(m):
        g32m = mul(g32m, g32)
    return mul(mul(mul(two_pow, twopi_pow), prod), inv_monomial(g32m))

def pref(m, jac):
    """sqrt(3/(m+3)) m!/Z_m * jac as an element."""
    s = surd_sqrt_rational(Fr(3, m + 3))
    return mul_surd(scale(inv_monomial(mehta_Z(m)), math.factorial(m) * jac), s)

def jk_exact(kmax):
    """J_k(s) = A_k sqrt(2 pi) erf(s/(2 sqrt2)) + B_k + C_k(s) e^{-s^2/8}; returns list of (A_k, B_k, C_k list by power)."""
    J = [(Fr(1), Fr(0), []), (Fr(0), Fr(4), [Fr(-4)])]
    for k in range(kmax - 1):
        A, B, C = J[k]
        Cn = [4 * (k + 1) * c for c in C] + [Fr(0)] * max(0, k + 2 - len(C))
        Cn[k + 1] += Fr(-4)
        J.append((4 * (k + 1) * A, 4 * (k + 1) * B, Cn))
    return J

# ---------- D_1, D_2, D_3
def D1():
    val = M(2, Fr(3, 16), ())
    return mul_pi_half(mul_surd(scale(val, Fr(1, 4)), (Fr(1), 3)), -1)

def D2():
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(2)
    gE, gR, b1 = expE[(2,)], rest[(2,)], beta[(1,)]
    kappa = surd_mul((b1 / 2, 1), surd_sqrt_rational(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt_rational(1 / alpha)), 1)
    pe, pr = univariate(PE), univariate(PR)
    eb = {}
    for n, c in enumerate(pe):
        eb = add(eb, scale(add(M(n, gE, ()), scale(M(n, gE, (kappa,)), -1)), c))
    rb = {}
    for n, c in enumerate(pr):
        rb = add(rb, scale(M(n, gR, ()), c))
    return mul(pref(2, 1), add(mul(constE, eb), rb)), kappa, gE, gR

def D3():
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(3)
    s, q = pvar(0, 2), pvar(1, 2)
    p_img = pscale(padd(s, pscale(q, Fr(-1))), Fr(1, 2))
    images = [p_img, q]
    PEs, PRs = psubst(PE, images, 2), psubst(PR, images, 2)
    eE, eR, bs = psubst(expE, images, 2), psubst(rest, images, 2), psubst(beta, images, 2)
    assert set(eE) == {(2, 0), (0, 2)} and set(eR) == {(2, 0), (0, 2)} and eE[(0, 2)] == Fr(1, 8) == eR[(0, 2)] and set(bs) == {(1, 0)}
    gE, gR = eE[(2, 0)], eR[(2, 0)]
    kappa = surd_mul((bs[(1, 0)] / 2, 1), surd_sqrt_rational(1 / alpha))
    b = (Fr(1, 4), 2)                                    # 1/(2 sqrt 2)
    assert surd_sq_value(b) == Fr(1, 8)
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt_rational(1 / alpha)), 1)
    def by_q_power(Pxy):
        out = {}
        for (i, j), c in Pxy.items():
            out.setdefault(j, {})[(i,)] = c
        return out
    kmax = max(max(by_q_power(PEs)), max(by_q_power(PRs)))
    JK = jk_exact(kmax)
    sqrt2pi = mul_pi_half(E(1, 2), 1)
    def branch(Pxy, lam, erfc):
        tot = {}
        for k, coef_poly in sorted(by_q_power(Pxy).items()):
            A, B, C = JK[k]
            for j, c in enumerate(univariate(coef_poly)):
                if c == 0:
                    continue
                term = {}
                if A:
                    x = M(j, lam, (b,))
                    if erfc:
                        x = add(x, scale(M(j, lam, (kappa, b)), -1))
                    term = add(term, scale(mul(sqrt2pi, x), A))
                if B:
                    x = M(j, lam, ())
                    if erfc:
                        x = add(x, scale(M(j, lam, (kappa,)), -1))
                    term = add(term, scale(x, B))
                for i, ci in enumerate(C):
                    if ci == 0:
                        continue
                    x = M(j + i, lam + Fr(1, 8), ())
                    if erfc:
                        x = add(x, scale(M(j + i, lam + Fr(1, 8), (kappa,)), -1))
                    term = add(term, scale(x, ci))
                tot = add(tot, scale(term, c))
        return tot
    eb = branch(PEs, gE, True)
    rb = branch(PRs, gR, False)
    return mul(pref(3, Fr(1, 2)), add(mul(constE, eb), rb)), kappa, gE, gR, kmax

def show(u):
    def key(k):
        return (k[2], k[1], k[0])
    parts = []
    for (r, twok, atom) in sorted(u, key=key):
        c = u[(r, twok, atom)]
        s = str(c)
        if r != 1: s += "*sqrt(%d)" % r
        if twok: s += "*pi^(%s)" % Fr(twok, 2)
        if atom: s += "*arctan(%s*sqrt(%d))" % ATOMS[atom]
        parts.append(s)
    return "  +  ".join(parts)

def fval(u):
    v = 0.0
    for (r, twok, atom), c in u.items():
        t = float(c) * math.sqrt(r) * math.pi ** (twok / 2)
        if atom:
            a = ATOMS[atom]; t *= math.atan(float(a[0]) * math.sqrt(a[1]))
        v += t
    return v

d1 = D1(); print("D_1 =", show(d1), "=", fval(d1))
d2, k2, gE2, gR2 = D2(); print("D_2 =", show(d2), "=", fval(d2), " (29/6 - sqrt6 =", 29/6 - math.sqrt(6), ")")
print("   m=2: kappa", k2, "gE", gE2, "gR", gR2)
d3, k3, gE3, gR3, kmax = D3(); print("D_3 =", show(d3)); print("     =", repr(fval(d3)))
print("   m=3: kappa", k3, "gE", gE3, "gR", gR3, "kmax", kmax)
print("ATOMS", ATOMS)
print("certified D_3 = 5.323180268889896894923198739687260635784388529...")
