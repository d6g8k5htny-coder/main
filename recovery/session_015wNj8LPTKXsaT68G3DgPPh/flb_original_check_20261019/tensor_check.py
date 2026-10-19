"""Exact checks of the tensor and regression identities that ORIGINAL and the finite-L binding share.

Q has independent symmetric coordinates with E Q_i^2 = a, E Q_i^4 = b, E Q_i^6 = c and zero odd moments.
Checked in d = 3 and d = 4 for a rational unit vector e:
  ORIGINAL (5) = binding (8):  E[Q_i Q_j Q_k Q_l] = a^2 (three pairings) + kappa_4 delta_ijkl
  ORIGINAL (6) = binding (11): B(e) = Cov(He)
  ORIGINAL (7) = binding (9):  v_i(e) = E[Q_i (Q.e)^3]
  ORIGINAL (9) and (8) = binding (10): E (Q.e)^6 and V(e)
  ORIGINAL (11) = binding (12)-(13): R and the unconditioned transverse covariance C, for an orthonormal U
  ORIGINAL (30): at a = 1, kappa_4 = 0 the conditioned transverse covariance is GOE + sqrt(2/3) Z I
  ORIGINAL (31) from (1): 6^(2/3) / 24^(1/3) = (3/2)^(1/3)
Needs sympy (run with 1.14.0). Usage: python3 -I -B tensor_check.py
"""
import itertools

import sympy as sp

a, b, c = sp.symbols("a b c", positive=True)
k4, k6 = b - 3 * a**2, c - 15 * a * b + 30 * a**3


def moment(counts):
    m = sp.Integer(1)
    for n in counts:
        if n % 2:
            return sp.Integer(0)
        m *= {0: 1, 2: a, 4: b, 6: c}[n]
    return m


def E_prod(idx, d):
    return moment([idx.count(i) for i in range(d)])


def E_poly(expr, q, d):
    """Expectation of a polynomial in q under the product law."""
    p = sp.Poly(sp.expand(expr), *q)
    return sp.expand(sum(coef * moment(list(mon)) for mon, coef in p.terms()))


def frame(d):
    """A rational unit vector e and an orthonormal basis U of e-perp (Gram-Schmidt, exact)."""
    e = {3: [sp.Rational(2, 3), sp.Rational(2, 3), sp.Rational(1, 3)],
         4: [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]}[d]
    basis, cols = [sp.Matrix(e)], []
    for k in range(d):
        v = sp.Matrix([1 if i == k else 0 for i in range(d)])
        for w in basis:
            v = v - (v.dot(w) / w.dot(w)) * w
        if v.norm() != 0 and len(basis) < d:
            v = v / v.norm()
            basis.append(v)
            cols.append(v)
    return sp.Matrix(e), sp.Matrix.hstack(*cols[: d - 1])


def check(d):
    dl = lambda i, j: 1 if i == j else 0
    M = {}
    for idx in itertools.product(range(d), repeat=4):
        i, j, k, l = idx
        target = a**2 * (dl(i, j) * dl(k, l) + dl(i, k) * dl(j, l) + dl(i, l) * dl(j, k)) + k4 * (
            1 if i == j == k == l else 0)
        M[idx] = E_prod(idx, d)
        assert sp.simplify(M[idx] - target) == 0, ("(5)", idx)
    e, U = frame(d)
    assert sp.simplify(e.dot(e) - 1) == 0 and sp.simplify(U.T * U - sp.eye(d - 1)) == sp.zeros(d - 1)
    assert sp.simplify(U.T * e) == sp.zeros(d - 1, 1)
    B = sp.Matrix(d, d, lambda i, k: sum(e[j] * e[l] * M[(i, j, k, l)] for j in range(d) for l in range(d)))
    B6 = a**2 * (sp.eye(d) + 2 * e * e.T) + k4 * sp.diag(*[x**2 for x in e])
    assert sp.simplify(B - B6) == sp.zeros(d), "(6)"
    q = sp.symbols("q0:%d" % d)
    Qe = sum(e[i] * q[i] for i in range(d))
    v = [E_poly(q[i] * Qe**3, q, d) for i in range(d)]
    assert all(sp.simplify(v[i] - (3 * a**2 * e[i] + k4 * e[i] ** 3)) == 0 for i in range(d)), "(7)"
    s4, s6 = sum(x**4 for x in e), sum(x**6 for x in e)
    m6 = E_poly(Qe**6, q, d)
    assert sp.simplify(m6 - (15 * a**3 + 15 * a * k4 * s4 + k6 * s6)) == 0, "(9)"
    V = m6 - sum(x**2 for x in v) / a
    assert sp.simplify(V - (6 * a**3 + 9 * a * k4 * s4 + (k6 - k4**2 / a) * s6)) == 0, "(8)"
    pairs = [(x, y) for x in range(d - 1) for y in range(x, d - 1)]
    R = sp.Matrix(len(pairs), d, lambda p, i: sum(
        U[j, pairs[p][0]] * U[k, pairs[p][1]] * e[l] * M[(j, k, i, l)]
        for j in range(d) for k in range(d) for l in range(d)))
    R11 = sp.Matrix(len(pairs), d, lambda p, i: a**2 * dl(*pairs[p]) * e[i] + k4 * U[i, pairs[p][0]] * U[i, pairs[p][1]] * e[i])
    assert sp.simplify(R - R11) == sp.zeros(len(pairs), d), "(11) R"
    C = sp.Matrix(len(pairs), len(pairs), lambda p, r: sum(
        U[i, pairs[p][0]] * U[j, pairs[p][1]] * U[k, pairs[r][0]] * U[l, pairs[r][1]] * M[(i, j, k, l)]
        for i in range(d) for j in range(d) for k in range(d) for l in range(d)))
    C11 =sp.Matrix(len(pairs), len(pairs), lambda p, r: a**2 * (
        dl(pairs[p][0], pairs[p][1]) * dl(pairs[r][0], pairs[r][1]) + dl(pairs[p][0], pairs[r][0]) * dl(pairs[p][1], pairs[r][1])
        + dl(pairs[p][0], pairs[r][1]) * dl(pairs[p][1], pairs[r][0])) + k4 * sum(
        U[i, pairs[p][0]] * U[i, pairs[p][1]] * U[i, pairs[r][0]] * U[i, pairs[r][1]] for i in range(d)))
    assert sp.simplify(C - C11) == sp.zeros(len(pairs)), "(11) C"
    Gam = (C - R * B.inv() * R.T).subs(b, 3 * a**2).subs(a, 1)
    GOE = sp.Matrix(len(pairs), len(pairs), lambda p, r: sp.Rational(2, 3) * dl(pairs[p][0], pairs[p][1]) * dl(pairs[r][0], pairs[r][1])
                    + dl(pairs[p][0], pairs[r][0]) * dl(pairs[p][1], pairs[r][1]) + dl(pairs[p][0], pairs[r][1]) * dl(pairs[p][1], pairs[r][0]))
    assert sp.simplify(Gam - GOE) == sp.zeros(len(pairs)), "(30)"
    print(f"d={d}: (5) (6) (7) (8) (9) (11) R and C, and the (30) continuum limit all hold exactly")


for d in (3, 4):
    check(d)
assert sp.simplify(sp.Integer(6) ** sp.Rational(2, 3) / sp.Integer(24) ** sp.Rational(1, 3) - sp.Rational(3, 2) ** sp.Rational(1, 3)) == 0
assert sp.simplify(sp.Integer(12) ** sp.Rational(-1, 3) * 2 ** sp.Rational(-1, 3) - sp.Integer(24) ** sp.Rational(-1, 3)) == 0
print("6^(2/3)/24^(1/3) = (3/2)^(1/3) and 12^(-1/3) 2^(-1/3) = 24^(-1/3) exactly")
c2 = 2 / (3 * sp.pi * sp.sqrt(3)) * sp.Rational(3, 2) ** sp.Rational(1, 3) * sp.gamma(sp.Rational(7, 6)) / sp.sqrt(sp.pi)
print("ORIGINAL (32): c_{2,inf} =", sp.N(c2, 20))
print("014K's C(2,24) (route B, 6277920922): 0.0734069193060351; relative difference",
      sp.N(sp.Float("0.0734069193060351", 20) / c2 - 1, 3))
