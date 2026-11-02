"""Finite exact bookkeeping controls, not a Gaussian/Kac-Rice proof.

Uses only stdlib and self-contained rational examples. Imports no project code.
"""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path


def rational(value):
    value = Q(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def run_controls():
    records = []

    def equal(name, actual, expected, explanation):
        actual, expected = Q(actual), Q(expected)
        if actual != expected:
            raise RuntimeError("positive bookkeeping control failed: " + name)
        records.append({"name": name, "kind": "positive", "passed": True,
                        "actual": rational(actual), "expected": rational(expected),
                        "explanation": explanation})

    def reject(name, correct, false_value, explanation):
        correct, false_value = Q(correct), Q(false_value)
        if correct == false_value:
            raise RuntimeError("counterexample did not reject substitution: " + name)
        records.append({"name": name, "kind": "rejecting_counterexample", "passed": True,
                        "correct": rational(correct), "false_value": rational(false_value),
                        "difference": rational(correct - false_value),
                        "explanation": explanation})

    # A symmetric pair kernel has two orientations. An anchor on the FIRST
    # witness breaks that symmetry; arbitrary unordered orientation is invalid.
    sites = ("a", "b", "c")
    ordered = [(x, y) for x in sites for y in sites if x != y]
    unordered = [(sites[i], sites[j]) for i in range(3) for j in range(i + 1, 3)]
    anchored = sum(x == "a" for x, y in ordered)
    forward = sum(x == "a" for x, y in unordered)
    reverse = sum(y == "a" for x, y in unordered)
    equal("symmetric_ordered_equals_twice_unordered", len(ordered), 2 * len(unordered),
          "Three sites give six ordered distinct pairs and three unordered pairs.")
    equal("asymmetric_first_anchor_count", anchored, 2,
          "Only (a,b) and (a,c) have their first witness anchored at a.")
    reject("asymmetric_anchor_does_not_allow_halving", forward, Q(anchored, 2),
           "Canonical unordered representatives with a first count two, not half the anchored ordered count.")
    reject("asymmetric_unordered_orientation_is_not_intrinsic", forward, reverse,
           "Reversing the representative of every same unordered pair changes the first-anchor count from two to zero.")

    # A finite distribution on configurations: N=1 and N=3, each probability1/2.
    configurations = [(Q(1, 2), 1), (Q(1, 2), 3)]
    factorial = sum(p * n * (n - 1) for p, n in configurations)
    diagonal = sum(p * n * n for p, n in configurations)
    mean = sum(p * n for p, n in configurations)
    explicit_distinct = sum(p * sum(i != j for i in range(n) for j in range(n))
                            for p, n in configurations)
    equal("factorial_matches_distinct_pair_enumeration", factorial, explicit_distinct,
          "Expected ordered DISTINCT pair count is three in this two-configuration distribution.")
    equal("diagonal_excess_equals_expected_N", diagonal - factorial, mean,
          "N^2 includes exactly N diagonal pairs in each configuration.")
    reject("diagonal_square_cannot_replace_factorial", factorial, diagonal,
           "E[N(N-1)]=3, while E[N^2]=5; already N=1 has zero versus one pair.")
    reject("square_of_mean_cannot_replace_factorial", factorial, mean * mean,
           "E[N(N-1)]=3 differs from (E[N])^2=4 without an applicable additional law.")

    # Joint endpoint-weight example. U,V are correlated factors on the SAME
    # three-point probability space; W=U*V occurs ONCE and Z=E[W] is full.
    probabilities = (Q(1, 2), Q(1, 3), Q(1, 6))
    u, v, g = (Q(1), Q(2), Q(1)), (Q(1), Q(1), Q(4)), (Q(0), Q(1), Q(3))
    w = tuple(a * b for a, b in zip(u, v))

    def expectation(values):
        return sum(p * value for p, value in zip(probabilities, values))

    z = expectation(w)
    numerator = expectation(tuple(a * b for a, b in zip(w, g)))
    tilted = tuple(p * weight / z for p, weight in zip(probabilities, w))
    value = numerator / z
    equal("one_weight_same_full_normalizer", value, Q(16, 11),
          "E[Wg]/E[W]=(8/3)/(11/6)=16/11, with one joint endpoint weight W.")
    equal("tilted_probability_law_is_normalized", sum(tilted), 1,
          "The same full Z normalizes all three weighted atoms, including the g=0 atom.")
    equal("direct_tilted_sum_matches_single_weight_ratio",
          sum(p * item for p, item in zip(tilted, g)), value,
          "Direct expectation under the normalized weighted law equals the single-weight ratio.")
    reject("false_weight_witness_factorization", numerator, z * expectation(g),
           "E[Wg]=8/3 is not E[W]E[g]=55/36 on this joint space.")
    reject("false_endpoint_factorization", z, expectation(u) * expectation(v),
           "E[UV]=11/6 is not E[U]E[V]=2; no endpoint independence is assumed.")
    doubled_numerator = expectation(tuple(a * a * b for a, b in zip(w, g)))
    reject("doubled_weight_same_Z", value, doubled_numerator / z,
           "Multiplying W again but retaining full Z gives56/11 rather than16/11.")
    reject("doubled_weight_renormalized_is_a_different_law", value,
           doubled_numerator / expectation(tuple(a * a for a in w)),
           "Even renormalizing W^2 gives56/27, a different law, not the original weighted law.")
    regional_z = expectation(tuple(weight if item != 0 else Q(0) for weight, item in zip(w, g)))
    reject("regional_denominator_cannot_replace_full_Z", value, numerator / regional_z,
           "Restricting Z to g's support changes11/6 to4/3 and changes the ratio to2.")

    # Illustrative four-dimensional physical-coordinate scaling ONLY:
    # (x1,x2,y1,y2)=(r*u1,r*u2,r*v1,r*epsilon*v2).
    # This finite rectangle check is not an asserted map for an upstream proof.
    r, epsilon = Q(1, 2), Q(1, 3)
    scales = (r, r, r, r * epsilon)
    widths = (Q(2), Q(3), Q(4), Q(5))
    jacobian, parameter_volume, physical_volume = Q(1), Q(1), Q(1)
    for scale, width in zip(scales, widths):
        jacobian *= scale
        parameter_volume *= width
        physical_volume *= scale * width
    equal("physical_pair_Jacobian_r4_epsilon", jacobian, r ** 4 * epsilon,
          "Product of the four physical coordinate scales is exactly1/48.")
    equal("physical_volume_direct_equals_one_Jacobian", physical_volume,
          parameter_volume * jacobian,
          "Multiplying physical side lengths gives5/2; parameter volume120 times1/48 agrees.")
    reject("do_not_reuse_full_physical_Jacobian", physical_volume, physical_volume * jacobian,
           "Applying r^4*epsilon a second time changes5/2 to5/96.")
    reject("do_not_reuse_epsilon_factor", physical_volume, physical_volume * epsilon,
           "An extra epsilon after the physical-coordinate change changes5/2 to5/6.")

    return {
        "scope": "Finite exact bookkeeping controls only; not a Kac-Rice/Gaussian theorem proof or upstream candidate review",
        "actor": "OpenAI/Codex /root/c302_architecture_ownership",
        "source_exposure": "Prior C317-C321 source/scout/transport exposure; these examples are independently authored for this finite task; no candidate was reviewed",
        "organizational_independence_credit": 0,
        "arithmetic": "Python stdlib fractions.Fraction; all comparisons exact",
        "positive_controls": sum(row["kind"] == "positive" for row in records),
        "rejecting_counterexamples": sum(row["kind"] == "rejecting_counterexample" for row in records),
        "all_passed": True, "controls": records,
        "upstream_imports_or_execution": False,
        "scientific_acceptance": "NONE",
        "limits": ["Finite atomic/rectangle examples only; no numerical Gaussian integration or asymptotic inference",
                   "No verification of upstream analytic assumptions, source identity, domains, covariance or Kac-Rice applicability",
                   "The illustrative Jacobian map is declared here, not attributed to an upstream candidate"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_controls()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("all_passed", "positive_controls", "rejecting_counterexamples",
                       "organizational_independence_credit", "scientific_acceptance")}))


if __name__ == "__main__":
    main()
