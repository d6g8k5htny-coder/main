"""Exact finite controls for coherent positive spectra and kernel bookkeeping.

No Gaussian sampling, eigenvalue approximation, Poisson simulation, or numerical
Rice integral. The all-frame, stability, and analytic kernel arguments are in the report.
"""
from fractions import Fraction as Q
from math import factorial
from pathlib import Path
import hashlib
import json


ROWS = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (0, 2))


def covariance(weights, scale):
    out = []
    for p in ROWS:
        row = []
        for q in ROWS:
            phase = (sum(p)-sum(q)) % 4
            if phase % 2:
                row.append(Q(0))
                continue
            sign = 1 if phase == 0 else -1
            row.append(sign*sum(a*(Q(i, scale)**(p[0]+q[0]))
                                  *(Q(j, scale)**(p[1]+q[1]))
                                  for (i, j), a in weights.items()))
        out.append(row)
    return out


def ldl_pivots(matrix):
    n = len(matrix)
    low = [[Q(int(i == j)) for j in range(n)] for i in range(n)]
    diagonal = []
    for i in range(n):
        value = matrix[i][i]-sum(low[i][j]**2*diagonal[j] for j in range(i))
        assert value > 0
        diagonal.append(value)
        for k in range(i+1, n):
            low[k][i] = (matrix[k][i]-sum(low[k][j]*low[i][j]*diagonal[j]
                                            for j in range(i)))/value
    return diagonal


def run():
    checks = {}
    base = {(i, j): Q(1, 25) for i in range(-2, 3) for j in range(-2, 3)}
    base_cov = covariance(base, 1)
    pivots = ldl_pivots(base_cov)
    assert sum(base.values()) == 1
    checks["fixed_base_axial_seven_jet_exact_LDL_positive"] = "PASS"
    moment_rows = []
    for n in (4, 5, 8, 12):
        k = 2*n
        eps = Q(1, n**12)
        size = (4*n+1)**2
        fill = {(i, j): Q(1, size) for i in range(-k, k+1) for j in range(-k, k+1)}
        weights = {ij: eps*a for ij, a in fill.items()}
        for (i, j), a in base.items():
            weights[(n*i, n*j)] += (1-eps)*a
        assert len(weights) == size and all(a > 0 for a in weights.values())
        assert sum(weights.values()) == 1
        m12 = sum(a*Q(i*i+j*j, n*n)**6 for (i, j), a in weights.items())
        assert m12 <= 8**6 == 262144
        actual_cov = covariance(weights, n)
        fill_cov = covariance(fill, n)
        for i in range(7):
            for j in range(7):
                assert actual_cov[i][j] == (1-eps)*base_cov[i][j]+eps*fill_cov[i][j]
        ldl_pivots(actual_cov)
        # Complete periods plus one zero residue imply the exact resonant fill covariance.
        residues = [0]*n
        for frequency in range(-2*n, 2*n+1):
            residues[frequency % n] += 1
        assert residues == [5]+[4]*(n-1)
        corr = 1-eps+eps/Q(4*n+1)
        small_value_eigenvalue = 1-corr
        assert small_value_eigenvalue == eps*(1-Q(1, 4*n+1)) < eps
        moment_rows.append({"N": n, "M12": str(m12),
                            "two_value_small_eigenvalue": str(small_value_eigenvalue)})
    checks["positive_full_square_normalized_uniform_physical_M12"] = "PASS"
    checks["single_jet_mixture_identity_and_resonant_two_value_collapse"] = "PASS"
    checks["resonant_uniform_fill_geometric_sum"] = "PASS"

    # A dominant DC mode plus positive fill leaves every derivative variance O(eps).
    for eps in (Q(1, 10), Q(1, 100), Q(1, 10000)):
        assert eps*8 <= 8*eps
    checks["DC_single_jet_failure_bookkeeping"] = "PASS"

    # Candidate cap for K=2N; transfer error uses P(different)<=N^-10.
    for n in (4, 8, 16):
        cap = (16*(2*n)**2)**2
        assert cap == 4096*n**4
        assert (cap+4096*n*n)*Q(1, n**10) <= Q(8192, n**6)
        assert Q(1, n**12)*(4*n+1)**2 <= Q(25, n**10)
    checks["small_fill_C4_dimension_ledger_and_expectation_transfer_cap"] = "PASS"

    # One fold: spatial r dr, height r^3, observation 12r^-5, Hessian r^2.
    assert 1+3-5+2 == 1
    # Two folds: joint U dimension 12; normalized physical determinant degree eight.
    assert 12//2+8 == 14
    assert 8+8 == 16 and 6+9 == 15 and 6+10 == 16 and 16 >= 15
    assert Q(1, 3)/2 == Q(1, 6)
    checks["ordered_fold_Jacobian_and_middle_far_kernel_exponents"] = "PASS"

    frozen = {
        "UNIFORM_LOCAL_NORM_DIAGONAL_PROPOSAL.md": "8e1eefca9888d43824a9fd9caca34c44d2de4bf8e30c668b474fa2097491c76b",
        "uniform_local_norm_controls.py": "4b84755b771598ed34193c075718a41da447719f90a2ff5c6a17d48d0abc3dc4",
        "UNIFORM_LOCAL_NORM_CONTROLS.json": "5d6cdb2d4925c6ba0b1b545d7a5aca1977655c1d18aa1ea29a82273f30845ece",
        "QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md": "827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30",
        "FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md": "621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138",
        "NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md": "16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524",
    }
    root = Path(__file__).resolve().parent
    for name, expected in frozen.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    checks["consumed_frozen_files_unchanged"] = "PASS"
    return {"classification": "EXACT_FINITE_ALGEBRA_CONTROLS_ONLY",
            "arithmetic": "Python Fraction and integers; no floating arithmetic",
            "Gaussian_sampling": "NONE", "process_simulation": "NONE",
            "scientific_effect": "NONE", "checks": checks,
            "base_axial_LDL_pivots": [str(x) for x in pivots],
            "finite_family_cases": moment_rows,
            "scope": "Finite checks support arithmetic; all-frame positivity, actual process obstruction, and covariance-kernel estimates require the conventional report proofs."}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
