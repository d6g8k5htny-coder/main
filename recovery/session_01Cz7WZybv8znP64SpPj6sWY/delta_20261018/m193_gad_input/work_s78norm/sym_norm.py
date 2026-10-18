"""Exact/symbolic checks of the normalization chain in proof.md sections 1,7,8.

Independent first-principles derivation of the near-fold Kac-Rice constant,
compared with formula (1) of the proof (C_alg) and with (17)-(20).
Run: venv/bin/python -I sym_norm.py
"""
import sympy as sp

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


# ---------------------------------------------------------------- 1. constants
c1 = sp.Integer(2) ** sp.Rational(2, 3) / (2 * sp.Integer(12) ** sp.Rational(1, 3))
c2 = sp.Integer(24) ** sp.Rational(-1, 3)
check("2^(2/3)/(2*12^(1/3)) == 24^(-1/3) (cube of ratio == 1 exactly)",
      sp.nsimplify(sp.simplify((c1 / c2) ** 3)) == 1 and sp.simplify(c1 - c2) == 0)

# ---------------------------------------------------------------- 2. absolute moment
t, V, ell, r, w = sp.symbols('t V ell r w', positive=True)
phi = sp.exp(-t ** 2 / (2 * V)) / sp.sqrt(2 * sp.pi * V)
half = sp.simplify(sp.integrate(t ** sp.Rational(4, 3) * phi, (t, 0, sp.oo)))
absmom = 2 * half
claim19 = 2 ** sp.Rational(2, 3) * sp.gamma(sp.Rational(7, 6)) * V ** sp.Rational(2, 3) / sp.sqrt(sp.pi)
check("E|N(0,V)|^(4/3) == 2^(2/3) Gamma(7/6) V^(2/3)/sqrt(pi)  [eq. 19]",
      sp.simplify(absmom / claim19) == 1)
check("one-sided moment E[(T)_+^(4/3)] == half of (19)", sp.simplify(half / claim19 - sp.Rational(1, 2)) == 0)

# ---------------------------------------------------------------- 3. Jacobian
for d in (2, 3, 4):
    DM = sp.symbols('m0:%d' % d)
    DS = sp.symbols('s0:%d' % d)
    Dl = sp.Symbol('Delta')
    e = sp.symbols('e0:%d' % d)
    P = [(DM[i] + DS[i]) / 2 for i in range(d)]
    Dr = [(DS[i] - DM[i]) / r for i in range(d)]
    W = 12 / r ** 3 * (Dl + r * sum(e[i] * P[i] for i in range(d)))
    new = P + Dr + [W]
    old = list(DM) + list(DS) + [Dl]
    J = sp.Matrix(new).jacobian(sp.Matrix(old))
    detJ = sp.simplify(J.det())
    check("d=%d: |det d(P,D_r,W)/d(D_M,D_S,Delta)| == 12 r^-(d+3)" % d,
          sp.simplify(detJ - 12 * r ** (-(d + 3))) == 0)

# ---------------------------------------------------------------- 4. Taylor of P, D_r, W, and det H_M det H_S
# generic polynomial field of degree 6 in d=2 and d=3 (coordinates rotated so e = first axis;
# all quantities here are rotation-covariant scalars/vectors, so WLOG).
for d in (2, 3):
    X = sp.symbols('x0:%d' % d)
    # monomial coefficients
    import itertools
    coeffs = {}
    f = 0
    for deg in range(0, 7):
        for mono in itertools.combinations_with_replacement(range(d), deg):
            cs = sp.Symbol('a_' + ''.join(map(str, mono)) if mono else 'a_')
            coeffs[mono] = cs
            term = cs
            for i in mono:
                term *= X[i]
            f += term
    grad = [sp.diff(f, X[i]) for i in range(d)]
    hess = [[sp.diff(f, X[i], X[j]) for j in range(d)] for i in range(d)]
    at = lambda expr, s: expr.subs({X[0]: s, **{X[i]: 0 for i in range(1, d)}})
    M, S = -r / 2, r / 2
    P = [(at(grad[i], M) + at(grad[i], S)) / 2 for i in range(d)]
    Dr = [(at(grad[i], S) - at(grad[i], M)) / r for i in range(d)]
    Delta = at(f, M) - at(f, S)
    W = sp.expand(12 / r ** 3 * (Delta + r * P[0]))
    D0 = [at(grad[i], 0) for i in range(d)]
    He0 = [at(hess[i][0], 0) for i in range(d)]
    Te = at(sp.diff(f, X[0], 3), 0)
    check("d=%d: W - T_e = O(r^2)" % d, sp.simplify(sp.limit((W - Te) / r, r, 0)) == 0 and
          sp.expand(W - Te).subs(r, 0) == 0)
    check("d=%d: D_r - He = O(r^2) and P - D = O(r^2)" % d,
          all(sp.expand((Dr[i] - He0[i])).subs(r, 0) == 0 and sp.expand((Dr[i] - He0[i]) / r).subs(r, 0) == 0
              for i in range(d)) and
          all(sp.expand(P[i] - D0[i]).subs(r, 0) == 0 and sp.expand((P[i] - D0[i]) / r).subs(r, 0) == 0
              for i in range(d)))
    # determinant: impose the leading pin He=0 at the midpoint (exact pin D_r=0 differs by O(r^2))
    pin = {coeffs[tuple(sorted((0, i)))]: 0 for i in range(d)}
    HM = sp.Matrix(d, d, lambda i, j: at(hess[i][j], M)).subs(pin)
    HS = sp.Matrix(d, d, lambda i, j: at(hess[i][j], S)).subs(pin)
    A = sp.Matrix(d - 1, d - 1, lambda i, j: at(hess[i + 1][j + 1], 0))
    dM = sp.expand(HM.det())
    dS = sp.expand(HS.det())
    # leading orders
    lead_M = sp.expand(dM.diff(r).subs(r, 0))
    lead_S = sp.expand(dS.diff(r).subs(r, 0))
    check("d=%d: det H_M = -(r/2) T_e det A + O(r^2), det H_S = +(r/2) T_e det A + O(r^2)" % d,
          dM.subs(r, 0) == 0 and dS.subs(r, 0) == 0 and
          sp.simplify(lead_M + Te * A.det() / 2) == 0 and sp.simplify(lead_S - Te * A.det() / 2) == 0)

# ---------------------------------------------------------------- 5. r-integral (any d: r^{d-1} r^{-(d+3)} r^2 = r^-2)
Vp = sp.Symbol('V', positive=True)
for d in (2, 3, 5):
    wr = 12 * ell / r ** 3
    phiw = sp.exp(-wr ** 2 / (2 * Vp)) / sp.sqrt(2 * sp.pi * Vp)
    integrand = r ** (d - 1) * 12 * r ** (-(d + 3)) * (r ** 2 / 4) * wr ** 2 * phiw
    I_r = sp.simplify(sp.integrate(integrand, (r, 0, sp.oo)))
    target = 12 ** sp.Rational(-1, 3) * ell ** sp.Rational(-1, 3) * half.subs(V, Vp)
    check("d=%d: int_0^inf r^(d-1) 12 r^-(d+3) (r^2/4) w^2 phi_V(w) dr == 12^(-1/3) ell^(-1/3) E[T_+^(4/3)]" % d,
          sp.simplify(I_r / target) == 1)

# ---------------------------------------------------------------- 6. assembly vs (1)
for d in (2, 3, 4):
    a, detB, F = sp.symbols('a detB F', positive=True)
    pD = (2 * sp.pi) ** sp.Rational(-d, 2) * a ** sp.Rational(-d, 2)       # N(0, a I_d) at 0
    pHe = (2 * sp.pi) ** sp.Rational(-d, 2) / sp.sqrt(detB)                  # N(0, B) at 0
    first_principles = 12 ** sp.Rational(-1, 3) * pD * pHe * F * half.subs(V, Vp)
    eq17_integrand = 1 / (2 * 12 ** sp.Rational(1, 3)) * pD * pHe * F * absmom.subs(V, Vp)
    eq1_integrand = sp.gamma(sp.Rational(7, 6)) / (24 ** sp.Rational(1, 3) * sp.sqrt(sp.pi) * (2 * sp.pi) ** d *
                                                    a ** sp.Rational(d, 2)) * F * Vp ** sp.Rational(2, 3) / sp.sqrt(detB)
    check("d=%d: first-principles integrand == (17) integrand (per unit x-volume)" % d,
          sp.simplify(first_principles / eq17_integrand) == 1)
    check("d=%d: (17) integrand == (1) integrand exactly" % d, sp.simplify(eq17_integrand / eq1_integrand) == 1)

# ---------------------------------------------------------------- 7. sphere measures
for d in (2, 3, 4):
    area = 2 * sp.pi ** sp.Rational(d, 2) / sp.gamma(sp.Rational(d, 2))
    print("     |S^%d| (area measure total) = %s  -> a normalized measure would change C by this factor" % (d - 1, sp.simplify(area)))

# ---------------------------------------------------------------- 8. scaling bookkeeping
# x -> lam x (spectrum Q -> Q/lam), amplitude f -> sig f:
lam, sig = sp.symbols('lam sig', positive=True)
for d in (2, 3, 4):
    # exponents of lam and sig in each factor of (1)
    a_s = sig ** 2 / lam ** 2                        # a = Var(D_i)
    detB_s = (sig ** 2 / lam ** 4) ** d              # det B
    F_s = (sig ** 2 / lam ** 4) ** (d - 1)           # (det A)^2, A (d-1)x(d-1) Hessian block
    V_s = sig ** 2 / lam ** 6                         # Var(T_e|D=0)
    C_s = a_s ** sp.Rational(-d, 2) * detB_s ** sp.Rational(-1, 2) * F_s * V_s ** sp.Rational(2, 3)
    # nu(ell) is per unit volume per unit ell: nu_new(ell) = lam^-d sig^-1 nu(ell/sig) => C_new = lam^-d sig^-(2/3) C
    check("d=%d: C_alg scales as lam^-d sig^(-2/3) (units [length^-d][ell]^(-2/3))" % d,
          sp.simplify(C_s - lam ** (-d) * sig ** sp.Rational(-2, 3)) == 0)
# cumulative
tt = sp.Symbol('t', positive=True)
check("int_0^t ell^(-1/3) d ell == (3/2) t^(2/3)",
      sp.simplify(sp.integrate(ell ** sp.Rational(-1, 3), (ell, 0, tt)) - sp.Rational(3, 2) * tt ** sp.Rational(2, 3)) == 0)

print("ALL PASS" if ok else "SOME FAIL")
