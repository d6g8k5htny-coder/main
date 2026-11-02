"""Exact finite polynomial/covariance bookkeeping; no upstream code or proof review."""
import argparse
from fractions import Fraction as F
import json
from math import factorial
from pathlib import Path


def encoded(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(item) for item in value]
    return value


def monomials(degree):
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def derivative(monomial, point, dx=0, dy=0):
    i, j = monomial
    if i < dx or j < dy:
        return F(0)
    x, y = point
    coefficient = F(factorial(i), factorial(i - dx)) * F(factorial(j), factorial(j - dy))
    return coefficient * x ** (i - dx) * y ** (j - dy)


def reduced(matrix):
    a = [[F(value) for value in row] for row in matrix]
    if not a:
        return a, []
    row, pivots = 0, []
    for column in range(len(a[0])):
        pivot = next((r for r in range(row, len(a)) if a[r][column]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        divisor = a[row][column]
        a[row] = [value / divisor for value in a[row]]
        for r in range(len(a)):
            if r != row and a[r][column]:
                multiple = a[r][column]
                a[r] = [value - multiple * leading for value, leading in zip(a[r], a[row])]
        pivots.append(column)
        row += 1
        if row == len(a):
            break
    return a, pivots


def rank(matrix):
    return len(reduced(matrix)[1])


def solve(matrix, rhs):
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix) or len(rhs) != n:
        raise ValueError("expected square exact solve")
    augmented, pivots = reduced([list(row) + [value] for row, value in zip(matrix, rhs)])
    if pivots != list(range(n)):
        raise ValueError("singular or inconsistent exact solve")
    answer = [augmented[i][-1] for i in range(n)]
    if any(sum(a * b for a, b in zip(row, answer)) != value
           for row, value in zip(matrix, rhs)):
        raise ValueError("exact solve failed direct substitution")
    return answer


def six_pin_rows(basis, q):
    """Jets of cubic Hermite H and linear p_y interpolant L, at x=0.

    q>0: H matches p,p_x at (+/-q/2,0); L matches p_y at those
    endpoints. q=0: the exact confluent limit H^(0..3), L^(0..1).
    These are normalized functionals, not six duplicate raw rows at q=0.
    """
    rows = [[] for _ in range(6)]
    origin = (F(0), F(0))
    if q == 0:
        for item in basis:
            values = [derivative(item, origin, dx=k) for k in range(4)]
            values += [derivative(item, origin, dy=1), derivative(item, origin, dx=1, dy=1)]
            for row, value in zip(rows, values):
                row.append(value)
        return rows
    ends = ((-q / 2, F(0)), (q / 2, F(0)))
    univariate = [(k, 0) for k in range(4)]
    interpolation = []
    for end in ends:
        interpolation += [[derivative(item, end) for item in univariate],
                          [derivative(item, end, dx=1) for item in univariate]]
    for item in basis:
        rhs = []
        for end in ends:
            rhs += [derivative(item, end), derivative(item, end, dx=1)]
        coefficients = solve(interpolation, rhs)
        left, right = (derivative(item, end, dy=1) for end in ends)
        values = [F(factorial(k)) * coefficients[k] for k in range(4)]
        values += [(left + right) / 2, (right - left) / q]
        for row, value in zip(rows, values):
            row.append(value)
    return rows


def contact_rows(basis, q, eta, z, w):
    rows = six_pin_rows(basis, q)
    rows += [[derivative(item, z, dx=1) for item in basis],
             [derivative(item, z, dy=1) for item in basis]]
    if eta:
        shifted = tuple(x + eta * direction for x, direction in zip(z, w))
        rows += [[(derivative(item, shifted, dx=1) - derivative(item, z, dx=1)) / eta
                  for item in basis],
                 [(derivative(item, shifted, dy=1) - derivative(item, z, dy=1)) / eta
                  for item in basis]]
    else:
        rows += [[w[0] * derivative(item, z, dx=2) + w[1] * derivative(item, z, dx=1, dy=1)
                  for item in basis],
                 [w[0] * derivative(item, z, dx=1, dy=1) + w[1] * derivative(item, z, dy=2)
                  for item in basis]]
    return rows


def run():
    records = []

    def record(name, kind, valid, **evidence):
        if not valid:
            raise RuntimeError("EXACT_CONTROL_FAILED: " + name)
        records.append({"name": name, "kind": kind, "passed": True, **evidence})

    cubic, quartic = monomials(3), monomials(4)
    z, w = (F(0), F(1)), (F(3, 5), F(4, 5))
    height3 = [derivative(item, z) for item in cubic]
    height4 = [derivative(item, z) for item in quartic]
    for q in (F(0), F(1, 16), F(1, 8)):
        for eta in (F(0), F(1, 64), F(1, 32)):
            a3 = contact_rows(cubic, q, eta, z, w)
            a4 = contact_rows(quartic, q, eta, z, w)
            weights = solve(list(map(list, zip(*a3))), height3)
            reproduction = [sum(weight * row[j] for weight, row in zip(weights, a3))
                            for j in range(len(cubic))]
            r3, r4, appended = rank(a3), rank(a4), rank(a4 + [height4])
            record("generic_q=" + str(q) + "_eta=" + str(eta), "positive_fixture",
                   r3 == 10 and r4 == 10 and appended == 11 and reproduction == height3,
                   q=q, eta=eta, z=z, w=w, cubic_rank=r3, quartic_contact_rank=r4,
                   quartic_appended_height_rank=appended,
                   row_order=["H(0)", "H'(0)", "H''(0)", "H'''(0)", "L(0)", "L'(0)",
                              "p_x(z)", "p_y(z)", "divided_p_x", "divided_p_y"],
                   height_reproduction_weights=weights, reproduced_monomials=len(cubic),
                   reproduction_error=[a - b for a, b in zip(reproduction, height3)])

    origin3 = contact_rows(cubic, F(0), F(0), z, w)
    origin4 = contact_rows(quartic, F(0), F(0), z, w)
    # Here t=z_y=1. h(v)=6(v/t)^2-8(v/t)^3+3(v/t)^4, v=y.
    h = [F(6) if item == (0, 2) else F(-8) if item == (0, 3)
         else F(3) if item == (0, 4) else F(0) for item in quartic]
    contacts = [sum(a * b for a, b in zip(row, h)) for row in origin4]
    height = sum(a * b for a, b in zip(height4, h))
    record("quartic_annihilator_at_q0_eta0", "positive_certificate",
           contacts == [F(0)] * 10 and height == 1,
           t=z[1], coefficients={"y^2": F(6), "y^3": F(-8), "y^4": F(3)},
           ten_contact_values=contacts, height_at_z=height)
    record("reject_quartic_witness_as_cubic", "rejecting_counterexample", h[quartic.index((0, 4))] != 0,
           correct_degree=4, false_degree_cap=3, nonzero_quartic_coefficient=F(3))
    appended3 = rank(origin3 + [height3])
    record("reject_degree3_appended_rank11", "rejecting_counterexample", appended3 == 10 and appended3 != 11,
           actual_rank=appended3, false_claimed_rank=11, cubic_columns=len(cubic),
           explanation="The exact reproduction weights already place height in the ten cubic contact rows.")
    for label, bad_w in (("pin_parallel", (F(1), F(0))), ("radial", (F(0), F(1)))):
        bad_rank = rank(contact_rows(cubic, F(0), F(0), z, bad_w))
        record("reject_generic_fullrank_" + label, "rejecting_counterexample", bad_rank == 9,
               z=z, w=bad_w, q=F(0), eta=F(0), actual_rank=bad_rank, false_claimed_rank=10,
               b_edge=bad_w[1], bc_minus_at=z[1] * bad_w[0] - z[0] * bad_w[1])

    # Exact scale path: s=u^4,r=u^5, so a=r^2*s+s^4=u^14(1+u^2).
    # C=1 suffices to display the proxy; arbitrary fixed C multiplies it.
    sequence = []
    for denominator in (2, 4, 8, 16, 32, 64):
        u = F(1, denominator)
        s, r = u ** 4, u ** 5
        a = r ** 2 * s + s ** 4
        ratio = s ** 8 / (a ** 2)
        if ratio != u ** 4 / (1 + u ** 2) ** 2 or not (0 < ratio < u ** 4):
            raise RuntimeError("exact scaling identity failed")
        sequence.append({"u": u, "s": s, "r": r, "a": a, "variance_upper_proxy": s ** 8,
                         "proxy_over_a_squared": ratio, "simple_upper_bound": u ** 4})
    record("naive_width_exact_scale_path", "positive_certificate",
           all(sequence[i + 1]["proxy_over_a_squared"] < sequence[i]["proxy_over_a_squared"]
               for i in range(len(sequence) - 1)), C=F(1), sequence=sequence,
           exact_identity="C*s^8/a^2 = C*u^4/(1+u^2)^2 < C*u^4; hence tends to zero as u tends to zero",
           limit_scope="Exact rational proxy identity; not an actual computed Gaussian variance")
    floor = F(1, 1000000)
    last = sequence[-1]["proxy_over_a_squared"]
    record("reject_uniform_positive_floor_for_this_proxy", "rejecting_counterexample", last < floor,
           counterexample_u=sequence[-1]["u"], proxy_ratio=last, false_lower_floor=floor,
           explanation="This finite counterexample rejects only this stated proxy floor; the algebraic identity supplies the general scale limit.")

    # Abstract finite Gaussian coefficient example: U,V,W independent N(0,1).
    # Observed C=(U+V,V), target T=U+2V+W. Cov(C1,C2)=1 is retained.
    observation_covariance = [[F(2), F(1)], [F(1), F(1)]]
    target_cross = [F(3), F(2)]
    regression = solve(observation_covariance, target_cross)
    prior_variance = F(6)
    conditional_variance = prior_variance - sum(a * b for a, b in zip(target_cross, regression))
    record("finite_gaussian_full_conditioning_schur", "positive_certificate",
           regression == [F(1), F(1)] and conditional_variance == 1,
           latent_coefficients={"C1": [1, 1, 0], "observed_cubic_coefficient_C2": [0, 1, 0], "T": [1, 2, 1]},
           observation_covariance=observation_covariance, target_cross_covariance=target_cross,
           regression_weights=regression, conditional_variance=conditional_variance)
    targets = ([F(0), F(0)], [F(1), F(2)])
    means = [sum(a * b for a, b in zip(regression, c)) for c in targets]
    record("conditioning_target_shift_changes_mean_not_variance", "positive_certificate",
           means == [F(0), F(3)] and conditional_variance == 1,
           conditioning_targets=targets, means=means, conditional_variances=[conditional_variance] * 2,
           conditional_second_moments=[conditional_variance + mean ** 2 for mean in means])
    dropped = prior_variance - target_cross[0] ** 2 / observation_covariance[0][0]
    residual_cross = target_cross[1] - target_cross[0] * observation_covariance[0][1] / observation_covariance[0][0]
    record("dropping_observed_cubic_coefficient_increases_variance", "rejecting_counterexample",
           dropped == F(3, 2) and dropped > conditional_variance and residual_cross == F(1, 2),
           full_conditional_variance=conditional_variance, dropped_condition_variance=dropped,
           retained_nonzero_conditional_cross_covariance=residual_cross)
    diagonal_false = prior_variance - target_cross[0] ** 2 / 2 - target_cross[1] ** 2
    record("reject_dropping_nonzero_observation_cross_covariance", "rejecting_counterexample",
           observation_covariance[0][1] == 1 and diagonal_false == F(-5, 2)
           and diagonal_false != conditional_variance,
           actual_cross_covariance=observation_covariance[0][1], correct_variance=conditional_variance,
           false_diagonalized_variance=diagonal_false)

    return encoded({
        "scope": "Standalone exact finite polynomial and abstract Gaussian-coefficient covariance controls only; not an analytic Gaussian/Kac-Rice theorem proof or candidate review",
        "actor": "OpenAI/Codex /root/c302_architecture_ownership",
        "claim": "C322 WE2906 / 512e8945-cbb8-4e83-8600-6215e54320c7; main605/6401565747",
        "source_exposure": "Prior C317-C322 source/scout/transport/finite-control exposure; current formulas supplied by Root; no proof candidate reviewed",
        "organizational_independence_credit": 0,
        "arithmetic": "Python stdlib Fraction, exact rational row reduction and direct substitution",
        "cubic_monomials": cubic, "quartic_monomials": quartic,
        "controls_count": len(records), "all_passed": True, "controls": records,
        "upstream_code_read_imported_or_executed": False,
        "limits": ["No covariance-kernel or Gaussian integration program; no random sampling",
                   "The finite Gaussian coefficient example invokes the stated abstract independent-normal model only",
                   "Rational fixtures do not prove uniform rank or bounds on an open/compact continuum",
                   "The scale proxy is not a computed actual conditioned-field variance",
                   "No analytic proof, source review, theorem acceptance, Lean, hosted CI or integration claim"],
        "scientific_acceptance": "NONE",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "controls_count": result["controls_count"],
                      "scientific_acceptance": result["scientific_acceptance"]}))


if __name__ == "__main__":
    main()
