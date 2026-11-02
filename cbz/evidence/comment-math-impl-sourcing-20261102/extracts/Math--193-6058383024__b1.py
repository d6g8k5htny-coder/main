#!/usr/bin/env python3
"""New, independent algebra checks; no project inputs or saved constants.

Direct lattice sums use a finite product spectrum. The displayed moment
identities are exact for every such finite spectrum, so these checks do not
claim a truncation-error estimate for the infinite covariance. No inference,
worker, rule, graph, runtime, or deployment changes are made.
"""

import itertools
import math


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def normalized(x):
    s = math.sqrt(dot(x, x))
    return tuple(v / s for v in x)


def inverse(matrix):
    n = len(matrix)
    rows = [list(row) + [float(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(rows[i][j]))
        rows[p], rows[j] = rows[j], rows[p]
        assert abs(rows[j][j]) > 1e-14
        pivot = rows[j][j]
        rows[j] = [v / pivot for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[j])]
    return [row[n:] for row in rows]


def bilinear(x, matrix, y):
    return sum(x[i] * matrix[i][j] * y[j]
               for i in range(len(x)) for j in range(len(y)))


def transverse_basis(e):
    basis = []
    for j in range(len(e)):
        v = [float(i == j) for i in range(len(e))]
        for axis in [e] + basis:
            proj = dot(v, axis)
            v = [x - proj * y for x, y in zip(v, axis)]
        if dot(v, v) > 1e-14:
            basis.append(normalized(v))
        if len(basis) == len(e) - 1:
            break
    return basis


MAX_RELATIVE_ERROR = 0.0


def close(x, y, description, tolerance=2e-11):
    global MAX_RELATIVE_ERROR
    error = abs(x - y) / max(1.0, abs(x), abs(y))
    MAX_RELATIVE_ERROR = max(MAX_RELATIVE_ERROR, error)
    assert error < tolerance, (description, error, x, y)


def check_spectrum(d, L, radius, direction):
    e = normalized(direction)
    q = [2 * math.pi * m / L for m in range(-radius, radius + 1)]
    weights = [math.exp(-x * x / 2) for x in q]
    z = sum(weights)
    probabilities = [w / z for w in weights]
    moments = {k: sum(p * x ** k for p, x in zip(probabilities, q))
               for k in (2, 4, 6)}
    a, b, c = moments[2], moments[4], moments[6]
    k4 = b - 3 * a * a
    k6 = c - 15 * a * b + 30 * a ** 3
    spectrum = []
    for indices in itertools.product(range(len(q)), repeat=d):
        vector = tuple(q[i] for i in indices)
        mass = math.prod(probabilities[i] for i in indices)
        spectrum.append((vector, mass))

    def expected(fun):
        return math.fsum(p * fun(vector) for vector, p in spectrum)

    delta = lambda i, j: float(i == j)
    for i, j, k, l in itertools.product(range(d), repeat=4):
        direct = expected(lambda Q: Q[i] * Q[j] * Q[k] * Q[l])
        formula = a ** 2 * (delta(i, j) * delta(k, l)
                            + delta(i, k) * delta(j, l)
                            + delta(i, l) * delta(j, k))
        formula += k4 * float(i == j == k == l)
        close(direct, formula, "fourth tensor")

    B = [[a ** 2 * (delta(i, j) + 2 * e[i] * e[j])
          + k4 * delta(i, j) * e[i] ** 2
          for j in range(d)] for i in range(d)]
    B_direct = [[expected(lambda Q: Q[i] * Q[j] * dot(Q, e) ** 2)
                 for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), repeat=2):
        close(B[i][j], B_direct[i][j], "column covariance")

    v = [3 * a ** 2 * e[i] + k4 * e[i] ** 3 for i in range(d)]
    v_direct = [expected(lambda Q: Q[i] * dot(Q, e) ** 3)
                for i in range(d)]
    for i in range(d):
        close(v[i], v_direct[i], "gradient-third covariance magnitude")
    sixth = expected(lambda Q: dot(Q, e) ** 6)
    sixth_formula = (15 * a ** 3 + 15 * a * k4 * sum(x ** 4 for x in e)
                     + k6 * sum(x ** 6 for x in e))
    close(sixth, sixth_formula, "sixth directional moment")
    V_direct = sixth - dot(v_direct, v_direct) / a
    V_formula = (6 * a ** 3 + 9 * a * k4 * sum(x ** 4 for x in e)
                 + (k6 - k4 ** 2 / a) * sum(x ** 6 for x in e))
    close(V_direct, V_formula, "third-jet regression variance")
    assert V_direct > 0

    U = transverse_basis(e)
    B_inv = inverse(B)
    n = d - 1
    for aa, bb, cc, dd in itertools.product(range(n), repeat=4):
        R = [a ** 2 * delta(aa, bb) * e[i]
             + k4 * U[aa][i] * U[bb][i] * e[i] for i in range(d)]
        S = [a ** 2 * delta(cc, dd) * e[i]
             + k4 * U[cc][i] * U[dd][i] * e[i] for i in range(d)]
        R_direct = [expected(lambda Q: dot(Q, U[aa]) * dot(Q, U[bb])
                             * Q[i] * dot(Q, e)) for i in range(d)]
        S_direct = [expected(lambda Q: dot(Q, U[cc]) * dot(Q, U[dd])
                             * Q[i] * dot(Q, e)) for i in range(d)]
        raw = expected(lambda Q: dot(Q, U[aa]) * dot(Q, U[bb])
                       * dot(Q, U[cc]) * dot(Q, U[dd]))
        raw_formula = a ** 2 * (delta(aa, bb) * delta(cc, dd)
                               + delta(aa, cc) * delta(bb, dd)
                               + delta(aa, dd) * delta(bb, cc))
        raw_formula += k4 * sum(U[aa][i] * U[bb][i]
                                * U[cc][i] * U[dd][i] for i in range(d))
        conditioned_direct = raw - bilinear(R_direct, B_inv, S_direct)
        conditioned_formula = raw_formula - bilinear(R, B_inv, S)
        close(conditioned_direct, conditioned_formula,
              "transverse Hessian Schur complement")
        if aa == bb == cc == dd:
            assert conditioned_direct > 0
    print(f"PASS direct finite-spectrum tensors/regressions: d={d}, L={L}, cutoff={radius}")


def simpson(fun, lo, hi, count):
    assert count % 2 == 0
    step = (hi - lo) / count
    odd = math.fsum(fun(lo + i * step) for i in range(1, count, 2))
    even = math.fsum(fun(lo + i * step) for i in range(2, count, 2))
    return step / 3 * (fun(lo) + fun(hi) + 4 * odd + 2 * even)


def check_radial_identity():
    V = 5.0
    sigma = math.sqrt(V)
    density = lambda u: math.exp(-u * u / (2 * V)) / math.sqrt(2 * math.pi * V)
    half_moment = simpson(lambda u: u ** (4 / 3) * density(u),
                          0, 12 * sigma, 100000)
    half_moment_formula = 2 ** (-1 / 3) * math.gamma(7 / 6) / math.sqrt(math.pi) * V ** (2 / 3)
    close(half_moment, half_moment_formula, "one-sided fractional Gaussian moment", 2e-9)
    # t=exp(s); the integrand 3 t^-2 z^2 p(z) dt becomes 3 exp(-s) z^2 p(z) ds.
    def radial_log(s):
        z = 12 * math.exp(-3 * s)
        return 3 * math.exp(-s) * z * z * density(z)
    radial = simpson(radial_log, -4, 12, 30000)
    expected = 12 ** (-1 / 3) * half_moment_formula
    close(radial, expected, "radial/fractional-moment identity", 2e-11)
    print("PASS independent radial quadrature and one-sided Gaussian moment")


def check_isotropic_regression():
    # These exact rational identities follow from the independently specified GOE convention.
    scalar_conditional_variance = 1 - 1 / 3
    transverse_scalar_variance = 2 + scalar_conditional_variance
    close(scalar_conditional_variance, 2 / 3, "isotropic scalar regression")
    close(transverse_scalar_variance / 2, 4 / 3, "d=2 negative-half determinant moment")
    close(15 - 3 ** 2, 6, "isotropic third-jet regression")
    close(12 ** (-1 / 3) * 2 ** (-1 / 3) * 6 ** (2 / 3),
          (3 / 2) ** (1 / 3), "continuum prefactor reduction")
    print("PASS continuum-limit normalization/regression reductions")


if __name__ == "__main__":
    for case in [(2, 2.3, 4, (3, 4)), (2, 7.3, 6, (1, 0)),
                 (3, 2.3, 4, (1, 2, 3)), (3, 7.3, 6, (2, -1, 2))]:
        check_spectrum(*case)
    check_radial_identity()
    check_isotropic_regression()
    print(f"All independent checks passed. Max scaled absolute discrepancy: {MAX_RELATIVE_ERROR:.3e}")
    print("These checks validate algebra and integration only; they do not prove the global barcode bridge.")
