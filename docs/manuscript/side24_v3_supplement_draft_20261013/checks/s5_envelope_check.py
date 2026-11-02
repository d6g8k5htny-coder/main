"""Symbolic check of the singular-near envelope (supplement Lemma S5, "anisotropic
triple-determinant envelope"): the cancellations behind

    |det H_M|, |det H_S| <= C rho^-2 r (r + s^2),      |det H_y| <= C rho^-2 (r^2 + s^3),

for polynomial fields of degree seven, uniformly in the angle c (no factor 1/c).

Setting.  The plane of the check contains the pair axis t = (1, 0) and the unit
vector omega = (c, rho), c^2 + rho^2 = 1; the hard direction is z3.  The pair
is M, S = M + r t, with midpoint x; the third critical point is y = x + s omega.
Constraints: grad f = 0 at M, S and y; f(M) - f(S) = k r^3 (k = kappa/6); and
the window f(S) <= f(y) <= f(M), written f(y) = f(M) - theta k r^3 with
theta in [0, 1].

Method.  Expand f about the point whose Hessian is examined (y, or the endpoint
M) as an exact polynomial of degree seven with free coefficients.  The
constraints are linear in the coefficients.  Solve them for the in-plane
Hessian and three cubic coefficients (the remaining coefficients are free and
bounded), and separately for the two mixed hard/plane Hessian entries.  Then
write each determinant as a Laurent polynomial in (s, r) and check every
monomial s^a r^b against the envelope for 0 < r <= s <= 1:

    r^2 + s^3   (third point):  b >= 2 and a + b >= 2, or b == 1 and a >= 3/2, or b == 0 and a >= 3;
    r (r + s^2) (endpoint):     b >= 2 and a + b >= 2, or b == 1 and a >= 2.

The only denominators are powers of rho and of s, and no factor of c appears.
The solved cubic coefficients are also checked to be bounded for r <= s.

Requires SymPy.  Run: python3 s5_envelope_check.py   (a few minutes)
"""
import sys
import sympy as sp

s, r, k, th = sp.symbols('s r k theta', positive=True)
c, rho = sp.symbols('c rho', real=True)
z1, z2 = sp.symbols('z1 z2')


def form(co, n):
    """Symmetric n-linear form on (z1, z2); co[j] multiplies binom(n, j) z1^(n-j) z2^j."""
    return sum(sp.binomial(n, j) * co[j] * z1**(n - j) * z2**j for j in range(n + 1))


def exponents(expr):
    num, den = sp.fraction(sp.together(expr))
    num = sp.expand(num)
    ds, dr = sp.degree(den, s), sp.degree(den, r)
    exps = sorted({(sp.degree(t, s) - ds, sp.degree(t, r) - dr) for t in sp.Add.make_args(num)})
    return sp.factor(den), exps


def within_third(a, b):
    return (b >= 2 and a + b >= 2) or (b == 1 and 2 * a >= 3) or (b == 0 and a >= 3)


def within_end(a, b):
    return (b >= 2 and a + b >= 2) or (b == 1 and a >= 2)


def setup(points, values):
    """Planar polynomial of degree 7 about the origin, critical at `points`, with
    the values `values` there (relative to the origin's value)."""
    A11, A12, A22 = sp.symbols('A11 A12 A22')
    T = sp.symbols('T0:4')
    Q = sp.symbols('Q0:5')
    P = sp.symbols('P0:6')
    S6 = sp.symbols('S0:7')
    S7 = sp.symbols('U0:8')
    g = (sp.Rational(1, 2) * form([A11, A12, A22], 2) + sp.Rational(1, 6) * form(T, 3)
         + sp.Rational(1, 24) * form(Q, 4) + sp.Rational(1, 120) * form(P, 5)
         + sp.Rational(1, 720) * form(S6, 6) + sp.Rational(1, 5040) * form(S7, 7))
    gx, gy = sp.diff(g, z1), sp.diff(g, z2)
    eqs = []
    for (px, py), val in zip(points, values):
        sub = {z1: px, z2: py}
        eqs += [gx.subs(sub), gy.subs(sub), g.subs(sub) - val]
    eqs = [sp.expand(e) for e in eqs]
    unknowns = [A11, A12, A22, T[0], T[2], T[3]]
    sol = sp.solve(eqs, unknowns, dict=True)
    assert len(sol) == 1, 'constraint system not uniquely solvable'
    sol = sol[0]
    # hard direction: d/dz3 f restricted to the plane, with one z3 factor
    A13, A23 = sp.symbols('A13 A23')
    Th = sp.symbols('Th0:3'); Qh = sp.symbols('Qh0:4'); Ph = sp.symbols('Ph0:5')
    Sh = sp.symbols('Sh0:6'); Uh = sp.symbols('Uh0:7')
    g3 = (A13 * z1 + A23 * z2 + sp.Rational(1, 2) * form(Th, 2) + sp.Rational(1, 6) * form(Qh, 3)
          + sp.Rational(1, 24) * form(Ph, 4) + sp.Rational(1, 120) * form(Sh, 5)
          + sp.Rational(1, 720) * form(Uh, 6))
    eqh = [sp.expand(g3.subs({z1: px, z2: py})) for (px, py) in points]
    solh = sp.solve(eqh, [A13, A23], dict=True)
    assert len(solh) == 1
    solh = solh[0]
    planar = sol[A11] * sol[A22] - sol[A12]**2
    cof = sol[A11] * solh[A23]**2 - 2 * sol[A12] * solh[A13] * solh[A23] + sol[A22] * solh[A13]**2
    cubics = [sol[T[0]], sol[T[2]], sol[T[3]]]
    return planar, cof, cubics


def report(name, expr, test):
    den, exps = exponents(expr)
    bad = [e for e in exps if not test(*e)]
    has_c = den.has(c)
    ok = not bad and not has_c
    print(f'{name}: denominator {den}; {len(exps)} monomial classes; '
          f'outside envelope: {bad}; denominator involves c: {has_c} -> {"PASS" if ok else "FAIL"}')
    return ok


def main():
    results = []
    # Third point y at the origin: M = -s omega - (r/2) t, S = -s omega + (r/2) t.
    a = (-s * c - r / 2, -s * rho)
    b = (-s * c + r / 2, -s * rho)
    planar, cof, cubics = setup([a, b], [th * k * r**3, (th - 1) * k * r**3])
    results.append(report('third point, in-plane 2x2 determinant', planar, within_third))
    results.append(report('third point, hard-direction cofactor', cof, within_third))
    for i, cu in enumerate(cubics):
        den, exps = exponents(cu)
        ok = all(e[1] >= 0 and e[0] + e[1] >= 0 for e in exps) and not den.has(c)
        print(f'third point, solved cubic coefficient {i}: bounded for r <= s -> {"PASS" if ok else "FAIL"}')
        results.append(ok)
    # Endpoint M at the origin: S = r t, y = (r/2) t + s omega.
    Spt = (r, 0)
    ypt = (r / 2 + s * c, s * rho)
    planar, cof, cubics = setup([Spt, ypt], [-k * r**3, -th * k * r**3])
    results.append(report('endpoint M, in-plane 2x2 determinant', planar, within_end))
    results.append(report('endpoint M, hard-direction cofactor', cof, within_end))
    for i, cu in enumerate(cubics):
        den, exps = exponents(cu)
        ok = all(e[1] >= 0 and e[0] + e[1] >= 0 for e in exps) and not den.has(c)
        print(f'endpoint M, solved cubic coefficient {i}: bounded for r <= s -> {"PASS" if ok else "FAIL"}')
        results.append(ok)
    print('ALL PASS' if all(results) else 'SOME CHECK FAILED')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
