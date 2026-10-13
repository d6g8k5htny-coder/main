"""Exploratory: structure of D_5 (m = 5) after two exact layers; float estimate of the certified remainder layer."""
import sys, time, math, itertools
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtd4cf/coefficients/side24_d5_closed_form_20260930")
import closed_form_d4 as X
from fractions import Fraction as Fr
E, add, scale, mul, mul_surd, mul_pi_half, surd_sqrt, M, G = X.E, X.add, X.scale, X.mul, X.mul_surd, X.mul_pi_half, X.surd_sqrt, X.M, X.G
tf_add, tf_scale = X.tf_add, X.tf_scale
t0 = time.time()

def organize_twvq(P):
    out = {}
    for (a, b, j, k), c in P.items():
        out.setdefault(a, {}).setdefault(b, {}).setdefault(k, {})[j] = c
    return {a: {b: {k: [d.get(j, Fr(0)) for j in range(max(d) + 1)] for k, d in ks.items()} for b, ks in bs.items()} for a, bs in out.items()}

def m5_reduction():
    m = 5
    alpha, beta, PE, PR, expE, rest = X.a_integrated_terms(m)
    T, w, v, q = (X.pvar(i, 4) for i in range(4))
    p3 = X.pscale(X.padd(v, X.pscale(q, Fr(-1))), Fr(1, 2))
    p2 = X.pscale(X.padd(w, X.pscale(v, Fr(-1))), Fr(1, 3))
    p1 = X.pscale(X.padd(T, X.pscale(w, Fr(-1))), Fr(1, 4))
    img = [p1, p2, p3, q]
    eE, eR, b = X.psubst(expE, img, 4), X.psubst(rest, img, 4), X.psubst(beta, img, 4)
    diag = {(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)}
    for e in (eE, eR):
        assert set(e) == diag and e[(0, 2, 0, 0)] == Fr(1, 48) and e[(0, 0, 2, 0)] == Fr(1, 24) and e[(0, 0, 0, 2)] == Fr(1, 8), e
    assert set(b) == {(1, 0, 0, 0)}
    gE, gR = eE[(2, 0, 0, 0)], eR[(2, 0, 0, 0)]
    kappa = X.surd_mul((b[(1, 0, 0, 0)] / 2, 1), surd_sqrt(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)
    pref = X.prefactor(5, Fr(1, 24))
    PE2, PR2 = X.psubst(PE, img, 4), X.psubst(PR, img, 4)
    return dict(alpha=alpha, kappa=kappa, gE=gE, gR=gR, constE=constE, pref=pref, innerE=organize_twvq(PE2), innerR=organize_twvq(PR2))

R = m5_reduction()
print("reduction", round(time.time() - t0, 1), "s; kappa", R["kappa"], "kappa^2", X.surd_sq(R["kappa"]), "gE", R["gE"], "gR", R["gR"], "pref", X.show(R["pref"]), "constE", X.show(R["constE"]))
kmax = max(k for inner in (R["innerE"], R["innerR"]) for bs in inner.values() for ks in bs.values() for k in ks)
JK = X.jk_exact(kmax)
assert all(A == 0 for k, (A, B, C) in enumerate(JK) if k % 2 == 1)

def layer1(ks):
    """L_{a,b}(w) as a T-function in w from {k: [coeffs in v]}."""
    L = {}
    for k, ek in ks.items():
        A, B, C = JK[k]
        assert A == 0
        for j, c in enumerate(ek):
            if c == 0:
                continue
            part = tf_scale(G(j, Fr(1, 24)), B) if B else {}
            for i, ci in enumerate(C):
                if ci:
                    part = tf_add(part, tf_scale(G(j + i, Fr(1, 6)), ci))
            L = tf_add(L, tf_scale(part, c))
    return L

def layer2(inner):
    """For each a: (exact T-function in T of Lambda_a, certified polynomials Q_{a,mu}(w) as {mu: {n: elem}})."""
    out = {}
    for a, bs in inner.items():
        exact, cert = {}, {}
        for b, ks in bs.items():
            L = layer1(ks)
            for (kind, mu), poly in L.items():
                for n, coef in poly.items():
                    if kind == "c":
                        exact = tf_add(exact, {k2: {n2: mul(coef, e2) for n2, e2 in p2.items()} for k2, p2 in G(b + n, Fr(1, 48)).items()})
                    elif kind == "e":
                        exact = tf_add(exact, {k2: {n2: mul(coef, e2) for n2, e2 in p2.items()} for k2, p2 in G(b + n, Fr(1, 48) + mu).items()})
                    else:
                        d = cert.setdefault(mu, {})
                        d[b + n] = add(d.get(b + n, {}), coef)
        out[a] = (exact, cert)
    return out

t1 = time.time()
lam2E, lam2R = layer2(R["innerE"]), layer2(R["innerR"])
print("layers 1-2 done", round(time.time() - t1, 1), "s")
for name, lam in (("E", lam2E), ("R", lam2R)):
    nQ = sum(len(cert) for _, cert in lam.values())
    degs = [max(d) for _, cert in lam.values() for d in cert.values()]
    kinds = {k for ex, _ in lam.values() for k in ex}
    print(name, "a-values", len(lam), "certified (a,mu) functions", nQ, "max w-degree", max(degs), "exact kinds", sorted(kinds))

# layer 3 exact part
def outer_exact(lam, g, erfc, kappa):
    tot = {}
    for a, (exact, _) in lam.items():
        for (kind, mu), poly in exact.items():
            for n, coef in poly.items():
                tot = add(tot, mul(coef, X.outer_moment(n + a, g, kind, mu, erfc, kappa)))
    return tot
t2 = time.time()
exE = outer_exact(lam2E, R["gE"], True, R["kappa"])
exR = outer_exact(lam2R, R["gR"], False, R["kappa"])
print("layer 3 exact done", round(time.time() - t2, 1), "s; atoms", X.ATOMS[1:])
print("exact part E:", X.show(exE)[:300]); print("exact part R:", X.show(exR)[:300])
fe, fr_ = X.fl_eval(exE), X.fl_eval(exR)
prefE = X.fl_eval(mul(R["pref"], R["constE"])); prefR = X.fl_eval(R["pref"])
print("float exact parts (with prefactors):", prefE * fe, prefR * fr_)

# float estimate of the certified parts: Omega_{a,mu}(T) = int_0^T Q(w) e^{-w^2/48} erf(sqrt mu w) dw, outer int T^a e^{-gT^2} [erfc(kT)] Omega dT
def float_cert(lam, g, erfc, kap, Tmax=70.0, n=3500):
    h = Tmax / n
    grid = [i * h for i in range(n + 1)]
    total = 0.0
    for a, (_, cert) in lam.items():
        for mu, Q in cert.items():
            qc = {nn: X.fl_eval(e) for nn, e in Q.items()}
            smu = math.sqrt(mu)
            f = [sum(c * w ** nn for nn, c in qc.items()) * math.exp(-w * w / 48) * math.erf(smu * w) for w in grid]
            Om = [0.0]
            for i in range(1, n + 1):
                Om.append(Om[-1] + 0.5 * h * (f[i - 1] + f[i]))
            out = [T ** a * math.exp(-g * T * T) * (math.erfc(kap * T) if erfc else 1.0) * Om[i] for i, T in enumerate(grid)]
            total += h * (sum(out) - 0.5 * (out[0] + out[-1]))
    return total
t3 = time.time()
gE, gR = float(R["gE"]), float(R["gR"])
kap = float(R["kappa"][0]) * math.sqrt(R["kappa"][1])
cE = float_cert(lam2E, gE, True, kap); cR = float_cert(lam2R, gR, False, kap)
print("float certified parts (with prefactors):", prefE * cE, prefR * cR, round(time.time() - t3, 1), "s")
D5 = prefE * (fe + cE) + prefR * (fr_ + cR)
print("D_5 float estimate:", D5)
print("total", round(time.time() - t0, 1), "s")
