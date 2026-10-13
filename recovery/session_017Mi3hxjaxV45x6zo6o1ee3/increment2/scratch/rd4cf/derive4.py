"""Exploratory: exact closed form of D_4 (m = 4) by one cumulative layer on top of the D_3 machinery."""
import sys, math, time
sys.path.insert(0, "<SCRATCH>/wtd3cf/coefficients/side24_d4_closed_form_20260930")
import closed_form_d3 as X
from fractions import Fraction as Fr
E, add, scale, mul, mul_surd, mul_pi_half, surd_sqrt, M = X.E, X.add, X.scale, X.mul, X.mul_surd, X.mul_pi_half, X.surd_sqrt, X.M
pvar, psubst, pscale, padd, pmul, pconst = X.pvar, X.psubst, X.pscale, X.padd, X.pmul, X.pconst

# ---- T-functions: {(kind, mu): {n: elem}} ; kind 'c' const poly, 'e' poly e^{-mu T^2}, 'f' poly erf(sqrt(mu) T)
def tf_add(A, B):
    out = {k: dict(v) for k, v in A.items()}
    for key, poly in B.items():
        d = out.setdefault(key, {})
        for n, c in poly.items():
            d[n] = add(d.get(n, {}), c)
    return {k: {n: c for n, c in v.items() if c} for k, v in out.items()}

def tf_scale(A, c):        # c a Fraction
    return {k: {n: scale(e, c) for n, e in v.items()} for k, v in A.items()}

def tf_shift(A, a):        # multiply by T^a
    return {k: {n + a: e for n, e in v.items()} for k, v in A.items()}

_G = {}
def G(n, mu):
    """int_0^T v^n e^{-mu v^2} dv as a T-function."""
    key = (n, mu)
    if key in _G:
        return _G[key]
    if n == 0:
        val = {('f', mu): {0: mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / mu)), 1)}}
    elif n == 1:
        val = {('c', Fr(0)): {0: E(1 / (2 * mu))}, ('e', mu): {0: E(-1 / (2 * mu))}}
    else:
        val = tf_add(tf_scale(G(n - 2, mu), Fr(n - 1)), {('e', mu): {n - 1: E(-1)}})
        val = tf_scale(val, 1 / (2 * mu))
    _G[key] = val
    return val

def outer_moment(n, g, kind, mu, erfc, kappa):
    """int_0^inf T^n e^{-g T^2} [erfc(kappa T)] {1, e^{-mu T^2}, erf(sqrt mu T)} dT."""
    if kind == 'c':
        lam, slopes = g, ()
    elif kind == 'e':
        lam, slopes = g + mu, ()
    else:
        lam, slopes = g, (surd_sqrt(mu),)
    val = M(n, lam, slopes)
    if erfc:
        val = add(val, scale(M(n, lam, tuple(sorted(slopes + (kappa,)))), -1))
    return val

def organize_tvp(P):
    out = {}
    for (i, j, k), c in P.items():
        out.setdefault(i, {}).setdefault(k, {})[j] = c
    return {a: {k: [d.get(j, Fr(0)) for j in range(max(d) + 1)] for k, d in ks.items()} for a, ks in out.items()}

def integrate_branch(inner, g, erfc, kappa, JK):
    total = {}
    for a, ks in inner.items():
        La = {}
        for k, ek in ks.items():
            A, B, C = JK[k]
            assert A == 0, "even k would need erf inside J_k"
            for j, c in enumerate(ek):
                if c == 0:
                    continue
                part = tf_scale(G(j, Fr(1, 24)), B) if B else {}
                for i, ci in enumerate(C):
                    if ci:
                        part = tf_add(part, tf_scale(G(j + i, Fr(1, 6)), ci))
                La = tf_add(La, tf_scale(part, c))
        for (kind, mu), poly in La.items():
            for n, coef in poly.items():
                total = add(total, mul(coef, outer_moment(n + a, g, kind, mu, erfc, kappa)))
    return total

def m4_exact():
    m = 4
    alpha, beta, PE, PR, expE, rest = X.a_integrated_terms(m)
    T, p2, p3 = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, padd(pscale(p2, Fr(-2)), pscale(p3, Fr(-1)))), Fr(1, 3))
    img1 = [p1, p2, p3]
    PE1, PR1 = psubst(PE, img1, 3), psubst(PR, img1, 3)
    eE1, eR1, b1 = psubst(expE, img1, 3), psubst(rest, img1, 3), psubst(beta, img1, 3)
    T2, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    img2 = [T2, pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2)), q]
    PE2, PR2 = psubst(PE1, img2, 3), psubst(PR1, img2, 3)
    eE2, eR2, b2 = psubst(eE1, img2, 3), psubst(eR1, img2, 3), psubst(b1, img2, 3)
    diag = {(2, 0, 0), (0, 2, 0), (0, 0, 2)}
    for e in (eE2, eR2):
        assert set(e) == diag and e[(0, 2, 0)] == Fr(1, 24) and e[(0, 0, 2)] == Fr(1, 8), e
    assert set(b2) == {(1, 0, 0)}
    gE, gR = eE2[(2, 0, 0)], eR2[(2, 0, 0)]
    bT = b2[(1, 0, 0)]
    kappa = X.surd_mul((bT / 2, 1), surd_sqrt(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)
    pref = X.prefactor(4, Fr(1, 6))
    innerE, innerR = organize_tvp(PE2), organize_tvp(PR2)
    kmax = max(max(max(ks) for ks in innerE.values()), max(max(ks) for ks in innerR.values()))
    JK = X.jk_exact(kmax)
    print("alpha", alpha, "kappa", kappa, "kappa^2", X.surd_sq(kappa), "gE", gE, "gR", gR, "kmax", kmax, "pref", X.show(pref), "constE", X.show(constE))
    t0 = time.time()
    eb = mul(mul(pref, constE), integrate_branch(innerE, gE, True, kappa, JK))
    print("erfc branch done", round(time.time() - t0, 1), "s")
    rb = mul(pref, integrate_branch(innerR, gR, False, kappa, JK))
    print("poly branch done", round(time.time() - t0, 1), "s")
    return add(eb, rb), eb, rb

def z4_exact():
    T, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, pscale(v, Fr(-1))), Fr(1, 3))
    p2 = pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2))
    p3 = q
    V = pconst(1, 3)
    for f in (p1, p2, p3, padd(p1, p2), padd(p2, p3), padd(padd(p1, p2), p3)):
        V = pmul(V, f)
    inner = organize_tvp(V)
    kmax = max(max(ks) for ks in inner.values())
    JK = X.jk_exact(kmax)
    const = mul_pi_half(E(Fr(24, 6)), 1)
    return mul(const, integrate_branch(inner, Fr(1, 48), False, None, JK))

z4 = z4_exact()
print("Z_4 exact:", X.show(z4), " (1536 pi expected)", "match" if z4 == X.mehta_Z(4) else "MISMATCH")
d4, eb, rb = m4_exact()
print("erfc branch:", X.show(eb))
print("poly branch:", X.show(rb))
print("D_4 =", X.show(d4))
print("float", repr(X.fl_eval(d4)))
iv = X.iv_eval(d4)
print("interval", iv.pair())
print("certified 14.21876345897735760130646567787407831734060 (Math-#200)")
print("ATOMS", X.ATOMS)
