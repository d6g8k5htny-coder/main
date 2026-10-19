"""Exact checks for METHOD V2 section 2 (M1-M4) against the ORIGINAL formulas.

Claim M193-V2S2-XFAM-INPUT-017-20261019. Input only; scratch use; nothing is
written to any repository. Moments of the coordinatewise law are computed
directly from E[prod Q_i^n_i] = prod m(n_i) for independent symmetric
coordinates (m0=1, m2=a, m4=b, m6=c, odd=0), not from the formulas checked.
"""
import itertools
import random
import sympy as sp

a, b, c = sp.symbols('a b c', positive=True)
k4 = b - 3 * a**2
k6 = c - 15 * a * b + 30 * a**3
MOM = {0: sp.Integer(1), 2: a, 4: b, 6: c}
results = []


def record(name, ok):
    results.append((name, bool(ok)))
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def expect(poly_expr, qs):
    """E of a polynomial in independent symmetric coordinates qs."""
    P = sp.Poly(sp.expand(poly_expr), *qs)
    tot = sp.Integer(0)
    for mon, coef in P.terms():
        val = coef
        for n in mon:
            if n % 2:
                val = 0
                break
            val = val * MOM[n]
        tot += val
    return sp.expand(tot)


def cayley(d, rng):
    """Exact rational orthogonal matrix from a random rational skew matrix."""
    S = sp.zeros(d, d)
    for i in range(d):
        for j in range(i + 1, d):
            v = sp.Rational(rng.randint(-5, 5), rng.randint(1, 4))
            S[i, j] = v
            S[j, i] = -v
    I = sp.eye(d)
    O = (I - S) * (I + S).inv()
    assert sp.simplify(O.T * O - I) == sp.zeros(d, d)
    return O


# ---- A1: (5) M_ijkl, (6)/M2 B, (7) v, (9) sixth moment, (8)/M3 V, general symbolic e
for d in range(2, 6):
    qs = sp.symbols(f'q1:{d+1}')
    es = sp.symbols(f'e1:{d+1}')
    S2 = sum(x**2 for x in es)
    S4 = sum(x**4 for x in es)
    S6 = sum(x**6 for x in es)
    Qe = sum(q * e for q, e in zip(qs, es))
    # (5)
    ok5 = True
    for i, j, k, l in itertools.product(range(d), repeat=4):
        lhs = expect(qs[i] * qs[j] * qs[k] * qs[l], qs)
        rhs = a**2 * (int(i == j and k == l) + int(i == k and j == l) + int(i == l and j == k)) \
            + k4 * int(i == j == k == l)
        if sp.simplify(lhs - rhs) != 0:
            ok5 = False
    record(f'A1 d={d}: ORIGINAL (5) M_ijkl = E[QiQjQkQl] for all index tuples', ok5)
    # (6) = M2 (homogeneous form; equals M2 on |e|=1)
    okB = True
    for i in range(d):
        for j in range(d):
            lhs = expect(qs[i] * qs[j] * Qe**2, qs)
            rhs = a**2 * (int(i == j) * S2 + 2 * es[i] * es[j]) + k4 * int(i == j) * es[i]**2
            if sp.expand(lhs - rhs) != 0:
                okB = False
    record(f'A1 d={d}: Cov(He)=E[QiQj(Q.e)^2] equals M2 = ORIGINAL (6) (|e|=1 form)', okB)
    # (7)
    okv = all(sp.expand(expect(qs[i] * Qe**3, qs) - (3 * a**2 * es[i] * S2 + k4 * es[i]**3)) == 0
              for i in range(d))
    record(f'A1 d={d}: v_i=E[Qi(Q.e)^3] = 3a^2 e_i + k4 e_i^3 (ORIGINAL (7))', okv)
    # (9)
    m6 = expect(Qe**6, qs)
    ok9 = sp.expand(m6 - (15 * a**3 * S2**3 + 15 * a * k4 * S4 * S2 + k6 * S6)) == 0
    record(f'A1 d={d}: E(Q.e)^6 = ORIGINAL (9)', ok9)
    # (8) = M3
    vsq = sum(expect(qs[i] * Qe**3, qs)**2 for i in range(d))
    V = sp.together(m6 - vsq / a)
    M3h = 6 * a**3 * S2**3 + 9 * a * k4 * S4 * S2 + (k6 - k4**2 / a) * S6
    record(f'A1 d={d}: E(Q.e)^6 - |v|^2/a equals M3 = ORIGINAL (8)', sp.simplify(V - M3h) == 0)

# ---- A2: (11) blocks and Gamma_e with exact rational orthonormal frames; L=inf GOE form (30)
rng = random.Random(20261019)
for d in range(2, 6):
    nfr = 3 if d <= 4 else 2
    okR = okC = okG = okGOE = True
    for _ in range(nfr):
        O = cayley(d, rng)
        e = [O[i, 0] for i in range(d)]
        U = [[O[i, j] for j in range(1, d)] for i in range(d)]
        n = d - 1

        def M(i, j, k, l):  # direct moment, not (5)
            if i == j == k == l:
                return b
            if (i == j and k == l) or (i == k and j == l) or (i == l and j == k):
                return a**2
            return sp.Integer(0)

        Bd = sp.Matrix(d, d, lambda i, j: sum(e[k] * e[l] * M(i, k, j, l)
                                              for k in range(d) for l in range(d)))
        B6 = a**2 * (sp.eye(d) + 2 * sp.Matrix(e) * sp.Matrix(e).T) + k4 * sp.diag(*[x**2 for x in e])
        if sp.simplify(Bd - B6) != sp.zeros(d, d):
            okR = False
        pairs = [(p, q) for p in range(n) for q in range(p, n)]
        Rd = sp.Matrix(len(pairs), d, lambda r, i: sum(
            U[j][pairs[r][0]] * U[k][pairs[r][1]] * e[l] * M(j, k, i, l)
            for j in range(d) for k in range(d) for l in range(d)))
        Ro = sp.Matrix(len(pairs), d, lambda r, i:
                       a**2 * int(pairs[r][0] == pairs[r][1]) * e[i]
                       + k4 * U[i][pairs[r][0]] * U[i][pairs[r][1]] * e[i])
        if sp.simplify(Rd - Ro) != sp.zeros(len(pairs), d):
            okR = False
        Cd = sp.Matrix(len(pairs), len(pairs), lambda r, s: sum(
            U[i][pairs[r][0]] * U[j][pairs[r][1]] * U[k][pairs[s][0]] * U[l][pairs[s][1]] * M(i, j, k, l)
            for i in range(d) for j in range(d) for k in range(d) for l in range(d)))

        def dl(x, y):
            return int(x == y)
        Co = sp.Matrix(len(pairs), len(pairs), lambda r, s: (
            a**2 * (dl(pairs[r][0], pairs[r][1]) * dl(pairs[s][0], pairs[s][1])
                    + dl(pairs[r][0], pairs[s][0]) * dl(pairs[r][1], pairs[s][1])
                    + dl(pairs[r][0], pairs[s][1]) * dl(pairs[r][1], pairs[s][0]))
            + k4 * sum(U[i][pairs[r][0]] * U[i][pairs[r][1]] * U[i][pairs[s][0]] * U[i][pairs[s][1]]
                       for i in range(d))))
        if sp.simplify(Cd - Co) != sp.zeros(len(pairs), len(pairs)):
            okC = False
        Gd = Cd - Rd * Bd.inv() * Rd.T
        Go = Co - Ro * B6.inv() * Ro.T
        if sp.simplify(Gd - Go) != sp.zeros(len(pairs), len(pairs)):
            okG = False
        # L = infinity: a=1, b=3 (k4=0): Gamma_e = GOE_{d-1} + (2/3) delta_ab delta_cd
        Ginf = Go.subs({a: 1, b: 3})
        GOE = sp.Matrix(len(pairs), len(pairs), lambda r, s: (
            dl(pairs[r][0], pairs[s][0]) * dl(pairs[r][1], pairs[s][1])
            + dl(pairs[r][0], pairs[s][1]) * dl(pairs[r][1], pairs[s][0])
            + sp.Rational(2, 3) * dl(pairs[r][0], pairs[r][1]) * dl(pairs[s][0], pairs[s][1])))
        if sp.simplify(Ginf - GOE) != sp.zeros(len(pairs), len(pairs)):
            okGOE = False
    record(f'A2 d={d}: R_ab,i and Cov(U^T H U) blocks of ORIGINAL (11) equal direct tensor contractions ({nfr} exact frames)', okR and okC)
    record(f'A2 d={d}: Gamma_e = Schur complement equals ORIGINAL (11) closed form', okG)
    record(f'A2 d={d}: at a=1,k4=0 Gamma_e = GOE_(d-1) + sqrt(2/3) Z I covariance (ORIGINAL (30))', okGOE)

# ---- A3: d=2, F = Var(A_e)/2 equals ORIGINAL (12); exact Pythagorean directions
okF = True
for (p, q, r) in [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (1, 0, 1), (20, 21, 29)]:
    for sgn in (1, -1):
        e = [sp.Rational(p, r), sp.Rational(sgn * q, r)]
        v = [-e[1], e[0]]

        def M(i, j, k, l):
            if i == j == k == l:
                return b
            if (i == j and k == l) or (i == k and j == l) or (i == l and j == k):
                return a**2
            return sp.Integer(0)
        varH = sum(v[i] * v[j] * v[k] * v[l] * M(i, j, k, l)
                   for i in range(2) for j in range(2) for k in range(2) for l in range(2))
        Rv = sp.Matrix(1, 2, lambda _, i: sum(v[j] * v[k] * e[l] * M(j, k, i, l)
                                              for j in range(2) for k in range(2) for l in range(2)))
        Bm = a**2 * (sp.eye(2) + 2 * sp.Matrix(e) * sp.Matrix(e).T) + k4 * sp.diag(e[0]**2, e[1]**2)
        Var = varH - (Rv * Bm.inv() * Rv.T)[0, 0]
        D = sp.Matrix([e[i] * (a**2 + k4 * v[i]**2) for i in range(2)])
        F12 = sp.Rational(1, 2) * (3 * a**2 + k4 * (v[0]**4 + v[1]**4) - (D.T * Bm.inv() * D)[0, 0])
        if sp.simplify(Var / 2 - F12) != 0:
            okF = False
record('A3 d=2: Var(A_e)/2 equals ORIGINAL (12) at 12 exact directions', okF)

# ---- A4: constant chain and L=infinity reduction, symbolic d
d_, V_, ell, u, r, t = sp.symbols('d V ell u r t', positive=True)
KM1 = sp.gamma(sp.Rational(7, 6)) / (sp.root(24, 3) * sp.sqrt(sp.pi) * (2 * sp.pi)**d_ * a**(d_ / 2))
pG = (2 * sp.pi)**(-d_ / 2) * a**(-d_ / 2)
pHe_noDet = (2 * sp.pi)**(-d_ / 2)
mom23 = 2**sp.Rational(-1, 3) * sp.gamma(sp.Rational(7, 6)) / sp.sqrt(sp.pi)  # times V^(2/3)
chain = 12**sp.Rational(-1, 3) * pG * pHe_noDet * mom23
record('A4: 12^(-1/3) p_G(0) p_He(0) * (23) prefactor = K_d(a) of M1 (symbolic d)',
       sp.simplify(sp.powsimp(sp.expand_power_base(chain / KM1, force=True), force=True)) == 1)
lhs23 = sp.integrate(u**sp.Rational(4, 3) * sp.exp(-u**2 / (2 * V_)) / sp.sqrt(2 * sp.pi * V_), (u, 0, sp.oo))
record('A4: E[(u_+)^(4/3)] for N(0,V) equals ORIGINAL (23)',
       sp.simplify(lhs23 - mom23 * V_**sp.Rational(2, 3)) == 0)
rstar = (12 * ell / u)**sp.Rational(1, 3)
lhs21 = (u**2 / 4) * rstar / sp.diff(u * r**3 / 12, r).subs(r, rstar)
record('A4: radial delta integral equals ORIGINAL (21)',
       sp.simplify(lhs21 - 12**sp.Rational(-1, 3) * u**sp.Rational(4, 3) * ell**sp.Rational(-1, 3)) == 0)
# L=inf: a=1, B=I+2ee^T (det 3), V=6: M1 -> ORIGINAL (31) for every d
Sd, Fs = sp.symbols('S_d F_star', positive=True)
M1inf = KM1.subs(a, 1) * Sd * Fs * 6**sp.Rational(2, 3) / sp.sqrt(3)
O31 = Sd / ((2 * sp.pi)**d_ * sp.sqrt(3)) * sp.Rational(3, 2)**sp.Rational(1, 3) * sp.gamma(sp.Rational(7, 6)) / sp.sqrt(sp.pi) * Fs
record('A4: M1 at a=1,k=0 equals ORIGINAL (31) for symbolic d', sp.simplify(M1inf / O31) == 1)
# d=2 continuum (32): |S^1|=2pi, F*_1 = (8/3)/2
c2 = (M1inf.subs({d_: 2, Sd: 2 * sp.pi, Fs: sp.Rational(4, 3)}))
O32 = 2 / (3 * sp.pi * sp.sqrt(3)) * sp.Rational(3, 2)**sp.Rational(1, 3) * sp.gamma(sp.Rational(7, 6)) / sp.sqrt(sp.pi)
record('A4: d=2 continuum value equals ORIGINAL (32)', sp.simplify(c2 / O32) == 1)
print('N(c_2,inf) =', sp.N(O32, 40))
print('SUMMARY', sum(ok for _, ok in results), 'of', len(results), 'PASS')
