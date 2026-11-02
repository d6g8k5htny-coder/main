#!/usr/bin/env python3
"""Exact-rational support for the RTW finite-spectrum propagation note.

Standard library only; no project code or certificate is imported or executed.
All predicates remain active under Python -O. This is finite support for the
written general proof, not proof by sampling or a production-runtime replay.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from itertools import combinations, product
import json
import random
import sys
from typing import TypeAlias

Matrix: TypeAlias = list[list[F]]
COUNTS: Counter[str] = Counter()
OBS: dict[str, object] = {}


def ck(value: bool, category: str, detail: str = "") -> None:
    COUNTS[category] += 1
    if not value:
        raise AssertionError(f"{category}: {detail}")


def mat(rows) -> Matrix:
    return [[F(x) for x in row] for row in rows]


def shape(a: Matrix) -> tuple[int, int]:
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("nonempty rectangular matrix required")
    return len(a), len(a[0])


def eye(n: int) -> Matrix:
    return [[F(i == j) for j in range(n)] for i in range(n)]


def tr(a: Matrix) -> Matrix:
    shape(a)
    return [list(row) for row in zip(*a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    if shape(a) != shape(b):
        raise ValueError("shape mismatch")
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(c: F, a: Matrix) -> Matrix:
    return [[c * x for x in row] for row in a]


def sub(a: Matrix, b: Matrix) -> Matrix:
    return add(a, scale(F(-1), b))


def mm(a: Matrix, b: Matrix) -> Matrix:
    m, n = shape(a)
    p, q = shape(b)
    if n != p:
        raise ValueError("product shape mismatch")
    return [[sum((a[i][k] * b[k][j] for k in range(n)), F(0))
             for j in range(q)] for i in range(m)]


def inv(a: Matrix) -> Matrix:
    n, m = shape(a)
    if n != m:
        raise ValueError("square matrix required")
    z = [a[i][:] + eye(n)[i] for i in range(n)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if z[i][j]), None)
        if pivot is None:
            raise ValueError("singular matrix")
        z[j], z[pivot] = z[pivot], z[j]
        d = z[j][j]
        z[j] = [x / d for x in z[j]]
        for i in range(n):
            if i != j:
                c = z[i][j]
                z[i] = [x - c * y for x, y in zip(z[i], z[j])]
    return [row[n:] for row in z]


def det(a: Matrix) -> F:
    """Independent elimination determinant; supports singular principal minors."""
    n, m = shape(a)
    if n != m:
        raise ValueError("square matrix required")
    z, ans = [row[:] for row in a], F(1)
    for j in range(n):
        p = next((i for i in range(j, n) if z[i][j]), None)
        if p is None:
            return F(0)
        if p != j:
            z[p], z[j] = z[j], z[p]
            ans = -ans
        d = z[j][j]
        ans *= d
        for i in range(j + 1, n):
            c = z[i][j] / d
            for k in range(j + 1, n):
                z[i][k] -= c * z[j][k]
            z[i][j] = F(0)
    return ans


def psd(a: Matrix) -> bool:
    """Exact PSD oracle by ALL principal minors, not just leading minors."""
    n, m = shape(a)
    if n != m or a != tr(a):
        return False
    for k in range(1, n + 1):
        for ids in combinations(range(n), k):
            if det([[a[i][j] for j in ids] for i in ids]) < 0:
                return False
    return True


def blocks(joint: Matrix, p: int) -> tuple[Matrix, Matrix, Matrix]:
    n, m = shape(joint)
    if n != m or not 0 < p < n:
        raise ValueError("invalid block dimensions")
    return ([row[:p] for row in joint[:p]],
            [row[:p] for row in joint[p:]],
            [row[p:] for row in joint[p:]])


def join(g: Matrix, a: Matrix, c: Matrix) -> Matrix:
    at = tr(a)
    return [row + at[i] for i, row in enumerate(g)] + [row + c[i] for i, row in enumerate(a)]


def schur(joint: Matrix, p: int) -> tuple[Matrix, Matrix]:
    g, a, c = blocks(joint, p)
    b = mm(a, inv(g))
    return b, sub(c, mm(b, tr(a)))


def trace(a: Matrix) -> F:
    return sum((a[i][i] for i in range(len(a))), F(0))


def f2(a: Matrix) -> F:
    return sum((x*x for row in a for x in row), F(0))


def quad(a: Matrix, x: Matrix) -> F:
    return mm(mm(tr(x), a), x)[0][0]


def joint_from_regression(g: Matrix, b: Matrix, s: Matrix) -> Matrix:
    a = mm(b, g)
    return join(g, a, add(s, mm(a, tr(b))))


def check_case(gamma0: Matrix, tail: Matrix, alpha: F, p: int,
               z: Matrix, g_floor: F, category: str) -> None:
    q = len(gamma0) - p
    g0, a0, c0 = blocks(gamma0, p)
    b0, s0 = schur(gamma0, p)
    joint = add(scale(alpha, gamma0), tail)
    g, a, c = blocks(joint, p)
    b, s = schur(joint, p)
    u, v, w = blocks(tail, p)
    r = [row + eye(q)[i] for i, row in enumerate(scale(F(-1), b0))]
    d = sub(v, mm(b0, u))
    rtr = mm(mm(r, tail), tr(r))
    remainder = sub(rtr, mm(mm(d, inv(g)), tr(d)))

    ck(mm(g0, inv(g0)) == eye(p), category + "/inverse")
    ck(mm(g, inv(g)) == eye(p), category + "/inverse")
    ck(sub(a, mm(b0, g)) == d, category + "/residual_identity")
    ck(sub(b, b0) == mm(d, inv(g)), category + "/regression_identity")
    ck(s == add(scale(alpha, s0), remainder), category + "/schur_identity")
    ck(psd(remainder), category + "/lower_PSD")
    ck(psd(sub(rtr, remainder)), category + "/upper_PSD")
    ck(psd(sub(g, scale(alpha * g_floor, eye(p)))), category + "/pin_floor")
    ck(psd(sub(g0, scale(g_floor, eye(p)))), category + "/pin_floor")

    tau, beta2 = trace(tail), f2(b0)
    eps = tau * (1 + beta2)
    ck(psd(sub(scale(eps, eye(q)), remainder)), category + "/trace_covariance_bound")
    dm = mm(sub(b, b0), z)
    mean_sq_bound = tau*tau*(1 + beta2)*f2(z)/(alpha*alpha*g_floor*g_floor)
    ck(f2(dm) <= mean_sq_bound, category + "/trace_mean_bound")

    # Independent completed-square minimizer check at several vectors.
    for k in (-2, 0, 3):
        y = [[F(k+i, i+1)] for i in range(q)]
        xp = scale(F(-1), mm(tr(b), y))
        x0 = scale(F(-1), mm(tr(b0), y))
        ck(quad(joint, xp+y) == quad(s, y), category + "/variational_identity")
        ck(quad(joint, x0+y) == quad(add(scale(alpha, s0), rtr), y),
           category + "/trial_upper_identity")
        ck(quad(s, y) >= alpha*quad(s0, y), category + "/quadratic_lower")


def random_matrix(rng: random.Random, n: int, m: int) -> Matrix:
    return [[F(rng.randint(-3, 3), rng.randint(1, 5)) for _ in range(m)] for _ in range(n)]


def matrix_campaign() -> None:
    rng = random.Random(44720261022)
    for i in range(48):
        p, q = (2, 2) if i % 2 == 0 else (3, 3)
        x = random_matrix(rng, p, p)
        g0 = add(eye(p), mm(x, tr(x)))
        b0 = random_matrix(rng, q, p)
        y = random_matrix(rng, q, q)
        s0 = mm(y, tr(y))
        if i % 3:
            s0 = add(s0, eye(q))
        gamma0 = joint_from_regression(g0, b0, s0)
        n = random_matrix(rng, p+q, (i % 3)+1)
        tail = scale(F(1, i+2), mm(n, tr(n)))
        alpha = F((i % 7)+1, (i % 7)+3)
        z = random_matrix(rng, p, 1)
        check_case(gamma0, tail, alpha, p, z, F(1), "matrix")
    OBS["constructed_matrix_cases"] = 48


def spectral_campaign() -> None:
    """Finite rational spectrum with real derivative covariance at one point.

    This tests product-normalization algebra, not the project's Gaussian
    weights, pi lattice, finite-r pin transform, or certified numeric bounds.
    """
    deriv = [(0,0), (1,0), (0,1), (2,0), (1,1), (3,0)]
    cases = []
    for outer in range(2, 6):
        weights = {j: F(1, 1+j*j) for j in range(-outer, outer+1)}
        zfull = sum(weights.values(), F(0))
        def features(k: int, l: int):
            out = []
            for a, b in deriv:
                v = F(k**a * l**b)
                parity = (a+b) % 4
                out.append((v, F(0)) if parity == 0 else
                           (F(0), v) if parity == 1 else
                           (-v, F(0)) if parity == 2 else (F(0), -v))
            return out
        def unnormalized(which) -> Matrix:
            ans = scale(F(0), eye(6))
            for k, l in which:
                feat = features(k,l)
                w = weights[k]*weights[l]
                for a in range(6):
                    for b in range(6):
                        ans[a][b] += w*(feat[a][0]*feat[b][0]+feat[a][1]*feat[b][1])
            return ans
        all_modes = list(product(range(-outer, outer+1), repeat=2))
        full = scale(1/zfull**2, unnormalized(all_modes))
        for cut in range(1, outer):
            zj = sum((weights[j] for j in range(-cut, cut+1)), F(0))
            modes = list(product(range(-cut, cut+1), repeat=2))
            outside = [(k,l) for k,l in all_modes if abs(k)>cut or abs(l)>cut]
            finite = scale(1/zj**2, unnormalized(modes))
            alpha = (zj/zfull)**2
            tail = scale(1/zfull**2, unnormalized(outside))
            ck(full == add(scale(alpha, finite), tail), "spectral/product_identity")
            ck(psd(tail), "spectral/tail_PSD")
            ck(full != add(scale(zj/zfull, finite), tail), "negative/wrong_dimension_power")
            g0 = blocks(finite, 3)[0]
            # At one point, this pin covariance is diagonal and positive.
            ck(all(g0[i][j] == 0 for i in range(3) for j in range(3) if i!=j),
               "spectral/pin_diagonal")
            gf = min(g0[i][i] for i in range(3))
            check_case(finite, tail, alpha, 3, mat([[1],[2],[-1]]), gf, "spectral")
            cases.append({"outer":outer,"cut":cut,"alpha":str(alpha),
                          "conditional_singular":det(schur(finite,3)[1])==0})
    OBS["finite_rational_spectral_cases"] = cases


def density_campaign() -> None:
    # The Gaussian-density theorem uses eigenvalue and norm bounds. Here
    # determinant and exponent comparisons are checked separately, exactly.
    rng = random.Random(20261022447)
    for i in range(32):
        p, q = 2, 3
        x = random_matrix(rng,p,p)
        g0 = add(eye(p),mm(x,tr(x)))
        b0 = random_matrix(rng,q,p)
        gamma0 = joint_from_regression(g0,b0,eye(q))
        n = random_matrix(rng,p+q,2)
        t = scale(F(1,i+3),mm(n,tr(n)))
        alpha = F((i%5)+2,(i%5)+3)
        joint = add(scale(alpha,gamma0),t)
        b,s = schur(joint,p)
        zpin = random_matrix(rng,p,1)
        delta = mm(sub(b,b0),zpin)
        eps = trace(t)*(1+f2(b0))
        upper = alpha+eps
        ck(psd(sub(s,scale(alpha,eye(q)))),"density/lower_metric")
        ck(psd(sub(scale(upper,eye(q)),s)),"density/upper_metric")
        ck(det(s) <= upper**q,"density/determinant_bound")
        ck(psd(sub(scale(1/alpha,eye(q)),inv(s))),"density/inverse_order")
        for j in range(7):
            z = [[F(j-3+k,3+k)] for k in range(q)]
            # l1 norms are exact conservative l2 bounds; no sqrt rounded.
            radius = sum((abs(row[0]) for row in z),F(0))
            shift = sum((abs(row[0]) for row in delta),F(0))
            logpart = (f2(z)-quad(inv(s),sub(z,delta)))/2
            lower = -((1/alpha-1)*radius*radius+2*radius*shift/alpha+shift*shift/alpha)/2
            ck(logpart >= lower,"density/exponent_lower_bound")
    OBS["density_models"] = 32
    OBS["density_exponent_points"] = 224


def alpha_bounds() -> None:
    for dim in (1,2,3,6):
        for delta in (F(0),F(1,1000),F(1,10),F(1,2),F(5)):
            for zj in (F(1),F(2),F(13,4)):
                eta = delta/zj+F(1,10000)
                alpha = (zj/(zj+delta))**dim
                lower = 1/(1+eta)**dim
                ck(0 < lower <= alpha <= 1,"alpha/lower_bound")
                ck(1-lower <= dim*eta,"alpha/bernoulli_loss_bound")
    # tau and regression norm may depend strongly on r. Bounds are pointwise
    # unless such dependencies are uniformly bounded separately.


def sharpness_and_negative_controls() -> None:
    alpha = F(1,2)
    gamma0 = eye(2)
    pin_tail = mat([[1,0],[0,0]])
    b,s = schur(add(scale(alpha,gamma0),pin_tail),1)
    ck(s == [[alpha]],"sharp/lower_equality")
    ck(s[0][0] < 1,"negative/unscaled_covariance_monotonicity")
    y_tail = mat([[0,0],[0,3]])
    b2,s2 = schur(add(scale(alpha,gamma0),y_tail),1)
    ck(s2 == [[alpha+3]],"sharp/upper_equality")
    means=[]
    for sign in (-1,1):
        tail=mat([[1,sign],[sign,1]])
        b3,s3=schur(add(scale(alpha,gamma0),tail),1)
        ck(b3 == [[F(2*sign,3)]],"negative/mean_not_invariant")
        ck(s3 == [[F(5,6)]],"negative/omit_regression_correction")
        ck(s3 != [[F(3,2)]],"negative/upper_bound_not_exact")
        means.append(str(b3[0][0]))
    correlated=mat([[1,1],[1,2]])
    b0,s0=schur(correlated,1)
    joint=add(scale(alpha,correlated),pin_tail)
    bf,sf=schur(joint,1)
    ck(b0 == [[F(1)]] and bf == [[F(1,3)]],"negative/residual_sign")
    ck(sub(bf,b0) != [[F(2,3)]],"negative/residual_sign")
    ck(sub(bf,b0) != [[F(-2)]],"negative/ignore_tail_pin_inverse")
    # Generic positive spectra: missing probability mass cannot alone bound
    # derivative energy without a frequency/weighted-moment bound.
    mass_rows=[]
    for freq in (10,100,1000,10000):
        delta=F(1,freq*freq)
        derivative_variance=delta*freq*freq/(1+delta)
        ck(delta <= F(1,100),"mass/small_mass")
        ck(derivative_variance > F(1,2),"negative/mass_only_derivative_bound")
        ck(derivative_variance > 10*delta,"negative/mass_only_derivative_bound")
        mass_rows.append({"frequency":freq,"missing_mass":str(delta),
                          "full_derivative_variance":str(derivative_variance)})
    OBS["mean_sign_witnesses"] = means
    OBS["conditional_variance_decrease"] = {"finite":"1","full":"1/2"}
    OBS["mass_without_derivative_control"] = mass_rows
    OBS["negative_control_families"] = [
        "use 1D retained fraction for 2D product law",
        "assert unscaled conditional covariance increases",
        "assert conditional mean unchanged",
        "omit negative regression correction in Schur identity",
        "reverse residual-cross-covariance sign",
        "ignore tail contribution to pin inverse",
        "use missing mass without derivative-weight control",
    ]


def empty_tail() -> None:
    g=mat([[2,1],[1,3]])
    b=mat([[1,2],[-1,1]])
    s=mat([[2,0],[0,1]])
    baseline=joint_from_regression(g,b,s)
    for alpha in (F(1),F(1,2),F(7,9)):
        bf,sf=schur(scale(alpha,baseline),2)
        ck(bf == b,"pure_scale/mean_invariant")
        ck(sf == scale(alpha,s),"pure_scale/covariance_scaled")
    # alpha<1,T=0 is a general algebra control, not a normalized full-field
    # covariance with positive omitted mass and all field-value observations.


def main() -> None:
    matrix_campaign()
    spectral_campaign()
    density_campaign()
    alpha_bounds()
    sharpness_and_negative_controls()
    empty_tail()
    print(json.dumps({
        "status":"PASS_SUPPORT_ONLY",
        "python":sys.version.split()[0],
        "test_kind":"exact-rational author support; no project/runtime replay",
        "checks":sum(COUNTS.values()),
        "counts":dict(sorted(COUNTS.items())),
        "observations":OBS,
        "scientific_effect":"NONE",
    },sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
