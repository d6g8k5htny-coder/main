"""Exact finite algebra controls for the critical-count diagonal route.

No Gaussian samples, approximate roots, gcd oracle, Rice calculation, or process test.
The all-K factor and genericity arguments are mathematical proofs in the report.
"""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path


def clean(p):
    return {e: v for e, v in p.items() if v}


def add(*ps):
    out = {}
    for p in ps:
        for e, v in p.items():
            out[e] = out.get(e, Q(0)) + v
    return clean(out)


def scale(a, p):
    return clean({e: a*v for e, v in p.items()})


def mul(p, q):
    out = {}
    for e, a in p.items():
        for f, b in q.items():
            ef = tuple(x+y for x, y in zip(e, f))
            out[ef] = out.get(ef, Q(0)) + a*b
    return clean(out)


def homogenize(p, d):
    assert all(sum(e) <= d for e in p)
    return {e+(d-sum(e),): c for e, c in p.items()}


def witness(k, c):
    # P=W(Z^2-1)+c(Z^2 W^2-1), Q=Z(W^2-1)+c(Z^2 W^2-1).
    p = {(2*k, k): Q(1), (0, k): Q(-1),
         (2*k, 2*k): c, (0, 0): -c}
    q = {(k, 2*k): Q(1), (k, 0): Q(-1),
         (2*k, 2*k): c, (0, 0): -c}
    return clean(p), clean(q)


def run():
    checks = {}
    for k in range(1, 9):
        for c in (Q(1), Q(-2), Q(3, 5)):
            p, q = witness(k, c)
            rhs = mul({(k, 0): Q(1), (0, k): Q(-1)},
                      {(k, k): Q(1), (0, 0): Q(1)})
            assert add(p, scale(-1, q)) == rhs
            assert max(map(sum, p)) == max(map(sum, q)) == 4*k
            for poly in (p, q):
                ph = homogenize(poly, 4*k)
                infinity = {e: a for e, a in ph.items() if e[2] == 0}
                assert infinity == {(2*k, 2*k, 0): c}
            # Recover P,Q from the actual three cosine modes' Laurent derivatives.
            f = {(k, 0): Q(1, 2), (-k, 0): Q(1, 2),
                 (0, k): Q(1, 2), (0, -k): Q(1, 2),
                 (k, k): c/2, (-k, -k): c/2}
            for axis, expected in ((0, p), (1, q)):
                actual = clean({(e[0]+k, e[1]+k): Q(2*e[axis], k)*a
                                for e, a in f.items()})
                assert actual == expected
    checks["lifted_witness_difference_and_actual_gradient_normalization_K1_to_K8"] = "PASS"
    checks["full_degree_4K_and_no_homogenizing_variable_factor"] = "PASS"

    # All line factors z-zeta*w with zeta^K=1 give Z=W.
    for c in (Q(1), Q(-2), Q(3, 5)):
        p, _ = witness(1, c)
        on_line = {}
        for (i, j), a in p.items():
            on_line[(i+j,)] = on_line.get((i+j,), Q(0)) + a
        assert clean(on_line) == {(3,): Q(1), (1,): Q(-1),
                                  (4,): c, (0,): -c}
        # All hyperbola factors zw-rho with rho^K=-1 give Z=-W^-1.
        on_hyperbola = {}
        for (i, j), a in p.items():
            e = (j-i,)
            on_hyperbola[e] = on_hyperbola.get(e, Q(0)) + a*(-1)**i
        assert clean(on_hyperbola) == {(-1,): Q(1), (1,): Q(-1)}
    checks["nonzero_line_and_hyperbola_restrictions"] = "PASS"

    # The c=0 witness is affine-coprime by factor inspection, but homogenizes badly.
    for k in range(1, 9):
        p, q = witness(k, Q(0))
        assert max(map(sum, p)) == max(map(sum, q)) == 3*k
        assert min(e[2] for e in homogenize(p, 4*k)) == k
        assert min(e[2] for e in homogenize(q, 4*k)) == k
    checks["false_degree_3K_projective_witness_has_common_T_power"] = "PASS"

    # Small radii relative to an upper cutoff do not control the actual-center ratio.
    eta, rho, actual_s, r = Q(1, 10), Q(1, 4), Q(1, 10000), Q(1, 1000)
    assert r <= rho*eta and r > rho*actual_s
    checks["actual_center_ratio_gap_preserved"] = "PASS"

    # Gaussian tail exponent after logarithmic cutoff c0=cstar/64.
    cstar = Q(5, 17)
    c0 = cstar/64
    exponent = cstar/(16*c0)
    assert exponent == 4 and exponent-Q(4, 3) == Q(8, 3)
    checks["log_cutoff_normalized_exponent"] = "PASS"

    frozen = {
        "QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md":
            "827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30",
        "fold_isolation_controls.py":
            "13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5",
        "FOLD_ISOLATION_CONTROLS.json":
            "ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf",
    }
    root = Path(__file__).resolve().parent
    for name, expected in frozen.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    checks["three_previous_isolation_files_unchanged"] = "PASS"
    return {
        "classification": "EXACT_FINITE_ALGEBRA_CONTROLS_ONLY",
        "arithmetic": "Python Fraction; no floating arithmetic",
        "sampling": "NONE", "joint_Rice_density": "NOT_USED",
        "genericity_proof": "CONVENTIONAL_REPORT_ARGUMENT_NOT_MACHINE_VERIFIED",
        "process_claim": "NONE", "scientific_effect": "NONE",
        "checks": checks,
        "normalization": {"chosen_tail_exponent": str(exponent),
                          "after_t_minus_4_over_3": str(exponent-Q(4, 3))},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
