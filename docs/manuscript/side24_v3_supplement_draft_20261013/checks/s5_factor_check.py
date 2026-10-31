"""Exact checks for the kernel transfer in supplement S5 (checkpoint draft 4).

It checks the following.

1. Singular-near residual limits (Lemma "singular-near corrected value-gradient
   frame").  A polynomial field of degree five at x = 0 with free coefficients
   is subjected to the exact finite-r pair pins: grad f = 0 at M, S = x -+ (r/2) t,
   f(M) = b and f(S) = b - kappa r^3/6.  With s = r/eta and eta fixed, the
   contact limits of V0 and V1 are computed and compared with eq:selection-L0
   (including its eta^2 term) and eq:selection-L1s, together with the
   deterministic terms -c kappa eta^2/6 (V0) and kappa c^2 - kappa eta^2/4 (V1).

2. Planar covariances.  For K0(z) = exp(-|z|^2/2), Cov(d^a f, d^b f) =
   (-1)^{|b|} d^{a+b} K0(0).  It checks eq:selection-coldet, the conditional
   covariance eq:selection-C01 with eta, eq:selection-det01 (Q_eta),
   eq:selection-det4, Q_0 = 3 (c^2+rho^2)(3c^2+rho^2), and the conditional
   variances quoted in the proofs: Var(U1|P) = 2, Var(U2|P) = 1/2,
   Var(W|P) = 6, the off-diagonal entries of Sigma = Cov(U1,U2,W|P) vanish,
   and det Cov(f_vv, f_vw | P) = 2.

3. Generic even kernels.  The covariance of a symmetric discrete spectral
   measure with random rational atoms (an anisotropic even kernel) replaces K0.
   It checks the exact factorizations det C_col = rho^6 D (D a quadratic form
   in (X, rho)) and det C_sing = rho^10 Q_eta(K) det Cov(f_vv,f_vw|P) / 144,
   with Q_eta(K) given by the Cauchy-Binet sum, and positivity at sample points.

4. The numbers of Lemma "deep images": 6 B0 q + 9289 B0 q^2 < 10^-113 and
   9289 B0 q^2 < 10^-230, in exact rational arithmetic with q < 10^-125.

Pins P = (f, f_111, f_1, f_11, f_2, f_12, f_3, f_13) at x; direction 1 is the
pair axis t.  A transverse unit vector is v = (v2, v3); U1 = d_t d_t d_v f,
U2 = (1/2) d_t d_v d_v f, W = d_v^3 f.

Requires SymPy.  Run: python3 s5_factor_check.py   (a few minutes)
"""
import itertools
import random
from fractions import Fraction

import sympy as sp

X, Y, Z, c, rho, eta = sp.symbols('X Y Z c rho eta', real=True)
v2, v3 = sp.symbols('v2 v3', real=True)
phi = sp.Symbol('phi', real=True)
unit_v = {v2: sp.cos(phi), v3: sp.sin(phi)}

results = []


def report(name, ok):
    results.append(bool(ok))
    print(f"{name} -> {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- 1. residual limits
r, kap, b = sp.symbols('r kappa b', real=True)
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
xs = (x1, x2, x3)
coef = {}
f = 0
for deg in range(0, 6):
    for m in itertools.combinations_with_replacement(range(3), deg):
        mi = tuple(m.count(i) for i in range(3))
        coef[mi] = sp.Symbol('a_' + ''.join(str(k) for k in mi), real=True)
        term = coef[mi]
        for i in range(3):
            term *= xs[i] ** mi[i] / sp.factorial(mi[i])
        f += term
grad = [sp.diff(f, v) for v in xs]
Mpt = {x1: -r / 2, x2: 0, x3: 0}
Spt = {x1: r / 2, x2: 0, x3: 0}
eqs = [f.subs(Mpt) - b, f.subs(Spt) - (b - kap * r**3 / 6)]
eqs += [g.subs(Mpt) for g in grad] + [g.subs(Spt) for g in grad]
unk = [coef[(0, 0, 0)], coef[(1, 0, 0)], coef[(2, 0, 0)], coef[(3, 0, 0)],
       coef[(0, 1, 0)], coef[(1, 1, 0)], coef[(0, 0, 1)], coef[(1, 0, 1)]]
sol = sp.solve(eqs, unk, dict=True)
assert len(sol) == 1
sol = sol[0]
report("pins force f_111 = 2 kappa + O(r^2)", sp.limit(sol[coef[(3, 0, 0)]], r, 0) == 2 * kap)
fp = f.subs(sol)
s_ = r / eta
ypt = {x1: s_ * c, x2: s_ * Y, x3: s_ * Z}
g1, g2, g3 = (sp.diff(fp, v) for v in xs)
V0 = (fp.subs(ypt) - fp.subs({x1: 0, x2: 0, x3: 0}) - s_ / 2 * (Y * g2.subs(ypt) + Z * g3.subs(ypt))
      - s_ * c / 3 * g1.subs(ypt)) / s_**3
V1 = g1.subs(ypt) / s_**2
V0lim = sp.expand(sp.limit(sp.expand(V0), r, 0))
V1lim = sp.expand(sp.limit(sp.expand(V1), r, 0))
w = (c, Y, Z)


def T(i, j, k):
    mi = [0, 0, 0]
    for q in (i, j, k):
        mi[q] += 1
    return coef[tuple(mi)]


L0_text = (-sp.Rational(1, 12) * sum(w[kk] * T(kk, i, j) * w[i] * w[j] for kk in (1, 2) for i in range(3) for j in range(3))
           - eta**2 / 16 * (Y * T(0, 0, 1) + Z * T(0, 0, 2)))
L1_text = sp.Rational(1, 2) * sum(T(0, i, j) * w[i] * w[j] for i in range(3) for j in range(3))
report("V0 contact limit = L0 (eq:selection-L0, with the eta^2 term) - c kappa eta^2/6",
       sp.simplify(V0lim - (L0_text - c * kap * eta**2 / 6)) == 0)
report("V1 contact limit = L1 (eq:selection-L1s) - kappa eta^2/4, with f_111 = 2 kappa",
       sp.simplify(V1lim - (L1_text.subs(coef[(3, 0, 0)], 2 * kap) - kap * eta**2 / 4)) == 0)


# ---------------------------------------------------------------- covariance machinery
def dK0(g):
    out = sp.Integer(1)
    for k in g:
        if k % 2:
            return sp.Integer(0)
        out *= (-1) ** (k // 2) * sp.factorial2(k - 1)
    return out


def cov0(a, bb):
    gg = tuple(x + y for x, y in zip(a, bb))
    return (-1) ** sum(bb) * dK0(gg)


def make_generic(seed, natoms=40):
    rnd = random.Random(seed)
    atoms = [(tuple(sp.Rational(rnd.randint(-9, 9), rnd.randint(1, 5)) for _ in range(3)),
              sp.Rational(rnd.randint(1, 9), rnd.randint(1, 9))) for _ in range(natoms)]
    cache = {}

    def cov(a, bb):
        gg = tuple(x + y for x, y in zip(a, bb))
        if gg not in cache:
            n = sum(gg)
            if n % 2:
                cache[gg] = sp.Integer(0)
            else:
                cache[gg] = (-1) ** (n // 2) * sum(wt * xi[0] ** gg[0] * xi[1] ** gg[1] * xi[2] ** gg[2] for xi, wt in atoms)
        return (-1) ** sum(bb) * cache[gg]
    return cov


def mono(*idx):
    m = [0, 0, 0]
    for i in idx:
        m[i - 1] += 1
    return {tuple(m): sp.Integer(1)}


def lin(*terms):
    out = {}
    for cf, var in terms:
        for k, val in var.items():
            out[k] = sp.expand(out.get(k, 0) + cf * val)
    return out


def covmat(vs, cov):
    n = len(vs)
    Mx = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            Mx[i, j] = sp.expand(sum(ca * cb * cov(a, bb) for a, ca in vs[i].items() for bb, cb in vs[j].items()))
    return Mx


def cond_cov(targets, pins, cov):
    full = covmat(pins + targets, cov)
    p = len(pins)
    S11, S12, S22 = full[:p, :p], full[:p, p:], full[p:, p:]
    return sp.simplify(S22 - S12.T * S11.inv() * S12)


def cond_det(targets, pins, cov):
    full = covmat(pins + targets, cov)
    p = len(pins)
    return sp.simplify(full.det() / full[:p, :p].det())


pins = [mono(), mono(1, 1, 1), mono(1), mono(1, 1), mono(2), mono(1, 2), mono(3), mono(1, 3)]
U1 = lin((v2, mono(1, 1, 2)), (v3, mono(1, 1, 3)))
U2 = lin((v2**2 / 2, mono(1, 2, 2)), (v2 * v3, mono(1, 2, 3)), (v3**2 / 2, mono(1, 3, 3)))
W = lin((v2**3, mono(2, 2, 2)), (3 * v2**2 * v3, mono(2, 2, 3)), (3 * v2 * v3**2, mono(2, 3, 3)), (v3**3, mono(3, 3, 3)))
Qv1 = lin((v2, mono(2, 2)), (v3, mono(2, 3)))
Qv2 = lin((v2, mono(2, 3)), (v3, mono(3, 3)))
sub = {Y: rho * v2, Z: rho * v3}


def same(d1, d2):
    return all(sp.simplify(d1.get(k, 0) - d2.get(k, 0)) == 0 for k in set(d1) | set(d2))


# ---------------------------------------------------------------- 2. planar covariances
Lax = lin((X * Y, mono(1, 1, 2)), (X * Z, mono(1, 1, 3)), (Y**2 / 2, mono(1, 2, 2)), (Y * Z, mono(1, 2, 3)), (Z**2 / 2, mono(1, 3, 3)))
Lp1 = lin((Y, mono(2, 2)), (Z, mono(2, 3)))
Lp2 = lin((Y, mono(2, 3)), (Z, mono(3, 3)))
report("collar, K0: det Cov(G_col|P) = rho^6 (4X^2+rho^2)  [eq:selection-coldet]",
       sp.simplify(cond_det([Lax, Lp1, Lp2], pins, cov0) - (Y**2 + Z**2)**3 * (4 * X**2 + Y**2 + Z**2)) == 0)
report("collar: axial random part = rho (X U1 + rho U2)",
       same({k: sp.expand(val.subs(sub)) for k, val in Lax.items()}, lin((rho * X, U1), (rho**2, U2))))
D0 = cond_det([lin((X, U1), (rho, U2)), Qv1, Qv2], pins, cov0)
report("collar, K0: D = 4X^2 + rho^2 on |v| = 1  [eq:selection-Acol0]", sp.simplify(sp.simplify(D0.subs(unit_v)) - (4 * X**2 + rho**2)) == 0)
Sig0 = cond_cov([U1, U2, W], pins, cov0)
report("K0: Sigma = Cov(U1,U2,W|P) = diag(2, 1/2, 6) on |v| = 1",
       sp.simplify(Sig0.subs(unit_v) - sp.diag(2, sp.Rational(1, 2), 6)) == sp.zeros(3, 3))
report("K0: det Cov(f_vv, f_vw | P) = 2 on |v| = 1", sp.simplify(cond_det([Qv1, Qv2], pins, cov0).subs(unit_v) - 2) == 0)

L0 = {}
for kk, wk in ((2, Y), (3, Z)):
    for i in range(3):
        for j in range(3):
            L0 = lin((1, L0), (-sp.Rational(1, 12) * wk * w[i] * w[j], mono(kk, i + 1, j + 1)))
L0 = lin((1, L0), (-eta**2 / 16 * Y, mono(1, 1, 2)), (-eta**2 / 16 * Z, mono(1, 1, 3)))
L1 = {}
for i in range(3):
    for j in range(3):
        L1 = lin((1, L1), (sp.Rational(1, 2) * w[i] * w[j], mono(1, i + 1, j + 1)))
L1 = {k: val for k, val in L1.items() if k != (3, 0, 0)}
Cs = cond_cov([L0, L1], pins, cov0)
rr = Y**2 + Z**2
C00 = rr * ((c**2 + sp.Rational(3, 4) * eta**2)**2 + 4 * c**2 * rr + 3 * rr**2) / 72
C01 = -c * rr * (c**2 + rr + sp.Rational(3, 4) * eta**2) / 6
C11 = rr * (4 * c**2 + rr) / 2
report("sing, K0: conditional covariance of (L0, L1)  [eq:selection-C01]",
       all(sp.simplify(e) == 0 for e in (Cs[0, 0] - C00, Cs[0, 1] - C01, Cs[1, 1] - C11)))
Qeta = (3 * c**2 - sp.Rational(3, 4) * eta**2)**2 + 12 * c**2 * rr + 3 * rr**2
report("sing, K0: det C_(L0,L1) = rho^6 Q_eta / 144  [eq:selection-det01]", sp.simplify(Cs.det() - rr**3 * Qeta / 144) == 0)
report("sing, K0: det Cov(L0..L3|P) = rho^10 Q_eta / 72  [eq:selection-det4]",
       sp.simplify(cond_det([L0, L1, Lp1, Lp2], pins, cov0) - rr**5 * Qeta / 72) == 0)
report("Q_0 = 3 (c^2+rho^2)(3c^2+rho^2)", sp.simplify(Qeta.subs(eta, 0) - 3 * (c**2 + rr) * (3 * c**2 + rr)) == 0)
report("sing: L0 = -(rho/12)((c^2 + 3 eta^2/4) U1 + 4 c rho U2 + rho^2 W)  [eq:selection-singlead]",
       same({k: sp.expand(val.subs(sub)) for k, val in L0.items()},
            lin((-rho * (c**2 + sp.Rational(3, 4) * eta**2) / 12, U1), (-rho**2 * c / 3, U2), (-rho**3 / 12, W))))
report("sing: L1 = rho (c U1 + rho U2)", same({k: sp.expand(val.subs(sub)) for k, val in L1.items()}, lin((rho * c, U1), (rho**2, U2))))

# ---------------------------------------------------------------- 3. generic even kernels
Amat = sp.Matrix([[c**2 + sp.Rational(3, 4) * eta**2, 4 * c * rho, rho**2], [c, rho, 0]])
mt = sp.Matrix([sp.Rational(3, 4) * eta**2 - 3 * c**2, -c * rho, -rho**2])
cols = [(0, 1), (0, 2), (1, 2)]
report("sing: the 2x2 minors of A are rho * (3eta^2/4 - 3c^2, -c rho, -rho^2)",
       all(sp.simplify(Amat[:, list(cc)].det() - rho * mt[k]) == 0 for k, cc in enumerate(cols)))
for seed in (1, 2):
    cov = make_generic(seed)
    detg = cond_det([Lax, Lp1, Lp2], pins, cov)
    Dg = cond_det([lin((X, U1), (rho, U2)), Qv1, Qv2], pins, cov)
    report(f"collar, generic kernel {seed}: det C_col = rho^6 D", sp.simplify(detg.subs(sub) - rho**6 * Dg) == 0)
    Dp = sp.Poly(sp.expand(Dg), X, rho)
    report(f"collar, generic kernel {seed}: D is a quadratic form in (X, rho)", all(sum(m) == 2 for m in Dp.monoms()))
    for (a2, a3) in ((1, 0), (0, 1), (sp.Rational(3, 5), sp.Rational(4, 5))):
        Dv = sp.expand(Dg.subs({v2: a2, v3: a3}))
        Am = sp.Matrix([[Dv.coeff(X, 2).subs(rho, 0), Dv.coeff(X, 1).coeff(rho, 1) / 2],
                        [Dv.coeff(X, 1).coeff(rho, 1) / 2, Dv.coeff(rho, 2).subs(X, 0)]])
        report(f"collar, generic kernel {seed}: D positive definite at v = ({a2}, {a3})", Am[0, 0] > 0 and Am.det() > 0)
    dets = cond_det([L0, L1, Lp1, Lp2], pins, cov)
    Sg = cond_cov([U1, U2, W], pins, cov)
    Qg = sum(mt[i] * mt[j] * Sg[list(cols[i]), list(cols[j])].det() for i in range(3) for j in range(3))
    dperp = cond_det([Qv1, Qv2], pins, cov)
    report(f"sing, generic kernel {seed}: det C_sing = rho^10 Q_eta(K) det Cov(f_vv,f_vw|P) / 144",
           sp.simplify(dets.subs(sub) - rho**10 * Qg * dperp / 144) == 0)
    for (a2, a3) in ((1, 0), (sp.Rational(3, 5), sp.Rational(4, 5))):
        for ang in (0, sp.pi / 7, sp.pi / 3, sp.pi / 2):
            for et in (0, sp.Rational(1, 2), sp.Rational(9, 10)):
                val = sp.N((Qg * dperp).subs({c: sp.cos(ang), rho: sp.sin(ang), eta: et, v2: a2, v3: a3}), 30)
                report(f"sing, generic kernel {seed}: Q_eta det_perp > 0 at v=({a2},{a3}), angle {ang}, eta {et}", val > 0)

# ---------------------------------------------------------------- 4. numbers of Lemma deep images
B0 = 76 * 24**6 + 15
q_up = Fraction(1, 10**125)
lhs = 6 * B0 * q_up + 9289 * B0 * q_up**2
report(f"B0 = {B0}; 6 B0 q + 9289 B0 q^2 < 10^-113 for q < 10^-125", lhs < Fraction(1, 10**113))
report("6 B0 < 8.72e10 and 9289 B0 < 1.35e14", 6 * B0 < Fraction(872, 100) * 10**10 and 9289 * B0 < Fraction(135, 100) * 10**14)
report("9289 B0 q^2 < 10^-230 for q < 10^-125", 9289 * B0 * q_up**2 < Fraction(1, 10**230))

print("ALL PASS" if all(results) else "SOME FAIL")
