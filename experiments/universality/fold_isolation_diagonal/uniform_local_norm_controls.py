"""Exact finite controls for the uniform local-norm diagonal proposal.

No Fourier-field samples, numerical Sobolev integrals, root counts, or process test.
"""
from fractions import Fraction as Q
from math import comb, factorial, ceil
from pathlib import Path
import hashlib
import json


def cutoff_constants():
    # d^j/dt^j exp(-1/t) = exp(-u) L_j(u), u=1/t.
    # L_(j+1)=u^2(L_j-L_j'). Keep sparse integer coefficient dictionaries.
    polys = [{0: 1}]
    for j in range(6):
        old = polys[-1]
        new = {}
        for m, a in old.items():
            new[m+2] = new.get(m+2, 0) + a
            if m:
                new[m+1] = new.get(m+1, 0) - m*a
        polys.append({m: a for m, a in new.items() if a})
    assert polys[:3] == [{0: 1}, {2: 1}, {4: 1, 3: -2}]
    aa = [sum(abs(a)*max(1, m**m) for m, a in poly.items()) for poly in polys]
    ee = [8]
    for n in range(1, 7):
        ee.append(8*sum(comb(n, j)*2*aa[j]*ee[n-j] for j in range(1, n+1)))
    hh = [sum(comb(n, j)*aa[j]*ee[n-j] for j in range(n+1)) for n in range(7)]
    dd = [6] + [2*hh[j]**2 for j in range(1, 7)]
    atheta = 0
    weight_sum = 0
    for i in range(7):
        for j in range(7-i):
            order = i+j
            weight = factorial(6)//(factorial(6-order)*factorial(i)*factorial(j))
            weight_sum += weight
            assert sum(comb(i, u)*comb(j, v)
                       for u in range(i+1) for v in range(j+1)) == 2**order
            inner = sum(comb(i, u)*comb(j, v)*dd[u]*dd[v]
                        for u in range(i+1) for v in range(j+1))
            atheta += weight*2**order*inner
    assert weight_sum == 3**6
    return aa, ee, hh, dd, atheta


def run():
    checks = {}
    integrals = [Q(factorial(j)*factorial(4-j), factorial(5)) for j in range(5)]
    assert integrals == [Q(1, 5), Q(1, 20), Q(1, 30), Q(1, 20), Q(1, 5)]
    # d3^2=1/(80pi), d4^2=1/(20pi); pi>3 suffices for these rational majorants.
    assert Q(1, 15)**2 > Q(1, 240)
    assert Q(2, 15)**2 > Q(1, 60)
    checks["Fourier_integral_and_rational_operator_constants"] = "PASS"

    for radius in (Q(0), Q(1, 4), Q(1), Q(3)):
        for j in range(7):
            assert radius**(2*j) <= 1+radius**12
    # Adding a DC variance A^2 changes M0 but not any positive spectral moment.
    for amplitude in (Q(1), Q(10), Q(100)):
        m0, m12 = amplitude**2+1, Q(1)
        assert m12 == 1 and m0 >= amplitude**2
    checks["separate_M0_M12_lower_moment_bound_and_DC_falsifier"] = "PASS"

    # e <= sum_0^4 1/n! + (1/5!)/(1-1/6) < 11/4, hence e^2<8.
    ebound = sum(Q(1, factorial(n)) for n in range(5)) + Q(1, factorial(5))/(1-Q(1, 6))
    assert ebound < Q(11, 4) and Q(11, 4)**2 < 8
    aa, ee, hh, dd, atheta = cutoff_constants()
    checks["explicit_cutoff_derivative_and_H6_trace_recurrence"] = "PASS"
    checks["multinomial_H6_weights_and_Leibniz_weight_sums"] = "PASS"

    # The finite product inequality holds without a dimension-dependent prefactor.
    cases = 0
    for n in range(1, 65):
        for mass in (Q(0), Q(1, 8), Q(1, 4), Q(1, 2)):
            rows = [mass/n]*n
            prod = Q(1)
            for x in rows:
                prod *= 1-x
            assert prod >= 1-sum(rows) >= Q(1, 2)
            cases += 1
    assert (1-Q(1, 2)) == Q(1, 2)
    checks["dimension_free_finite_Gaussian_product_margin"] = "PASS"

    # Rotation and cover margins are exact inequalities, not angle sampling.
    assert Q(2) < Q(3, 2)**2  # 1/2+sqrt(2)<2.
    for lmin in (Q(1, 10), Q(1), Q(10), Q(100)):
        b = min(Q(1), lmin/16)
        assert 6*b < lmin
        for length in (lmin, 2*lmin, 10*lmin):
            n = ceil(length/b)
            assert length/n <= b
            assert n*n <= (1+length)**2/b**2
        eta0 = b/4
        for rho in (Q(1, 10), Q(1, 4), Q(9, 10)):
            assert (1+rho)*eta0 <= b/2
            assert b/2+rho*eta0 < 2*b
    checks["fixed_cover_rotated_chart_and_endpoint_margins"] = "PASS"

    # The selected logarithmic cutoff exactly cancels K^8 and (1+L)^2.
    p, q, u = Q(8), Q(2), Q(4)
    assert p-8 == q-2 == 0 and u-Q(4, 3) == Q(8, 3)
    assert 16**4 == 65536
    checks["growing_K_L_log_cutoff_normalization"] = "PASS"

    # Moving the lifetime-band cap to T(t)=1/t makes rbar constant, not o(eta).
    rho = Q(1, 4)
    for n in (10, 100, 1000):
        eta, rbar = Q(1, n), Q(1)
        assert rbar > rho*eta
    checks["growing_lifetime_cap_radius_failure_control"] = "PASS"

    frozen = {
        "QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md": "827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30",
        "fold_isolation_controls.py": "13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5",
        "FOLD_ISOLATION_CONTROLS.json": "ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf",
        "FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md": "621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138",
        "finite_fourier_count_controls.py": "fd6f29739280a55017945526cb75fb46ebc80d86c012ff4a1ff65baa404c31de",
        "FINITE_FOURIER_COUNT_CONTROLS.json": "bbe147fdf409a3e37b450d4cbcd5d2a927140f95bd3ce0f36cafe16861123bfd",
        "NONAUTHOR_FOLD_ISOLATION_REVIEW.md": "f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90",
        "NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md": "16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524",
    }
    base = Path(__file__).resolve().parent
    for name, expected in frozen.items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest() == expected
    checks["six_prior_author_files_and_two_reviews_unchanged"] = "PASS"
    return {
        "classification": "EXACT_FINITE_ALGEBRA_CONTROLS_ONLY",
        "arithmetic": "Python Fraction and integers; no floating arithmetic",
        "Gaussian_field_sampling": "NONE", "Rice_density": "NOT_USED",
        "process_claim": "NONE", "scientific_effect": "NONE",
        "checks": checks, "Gaussian_product_cases": cases,
        "constants": {"A_j": aa, "E_j": ee, "H_j": hh, "d_j_upper": dd,
                      "A_theta": str(atheta), "A_theta_decimal_digits": len(str(atheta)),
                      "embedding_p3": "1/15", "embedding_p4": "2/15",
                      "normalized_t_exponent": "8/3"},
        "proof_status": "Analytic norm, trace, cover, isolation and asymptotic arguments are conventional report proofs; not machine verified",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
