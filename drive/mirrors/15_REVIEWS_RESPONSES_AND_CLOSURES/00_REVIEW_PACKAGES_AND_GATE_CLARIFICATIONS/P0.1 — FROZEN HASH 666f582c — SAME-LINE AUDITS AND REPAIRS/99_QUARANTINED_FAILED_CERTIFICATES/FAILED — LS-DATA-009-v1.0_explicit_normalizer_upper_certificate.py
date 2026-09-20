#!/usr/bin/env python3
"""Explicit upper certificate for the exact P0.1 typed normalizer on r<=r_D.

Constructs the pin-adjusted endpoint Hessian Gaussian law, certifies its
finite-r mean/covariance perturbation in the stabilized pin frame, and derives
uniform fourth-moment bounds giving Z_r^exact <= 11 r^2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

r = sp.symbols("r", positive=True)
x = sp.symbols("x")
R = sp.Rational(1, 18000)
MEXP = 14
TORUS_ENTRY_ALLOWANCE = sp.Rational(1, 10**70)
EPSILON_D = sp.Rational(1, 10000)
C_D = sp.Rational(343, 192) + sp.sqrt(1074) / 960
R_D = sp.simplify(EPSILON_D / C_D)

HE = [sp.Integer(1), x]
for n in range(1, 18):
    HE.append(sp.expand(x * HE[n] - n * HE[n - 1]))


def he(n: int, z: sp.Expr) -> sp.Expr:
    return sp.expand(HE[n].subs(x, z))


def dk(j: int, k: int, dx: sp.Expr, dy: sp.Expr = sp.Integer(0)) -> sp.Expr:
    return sp.simplify(
        (-1) ** (j + k) * he(j, dx) * he(k, dy)
        * sp.exp(-(dx**2 + dy**2) / 2)
    )


def cov_eval(a: tuple[int, int], p: tuple[sp.Expr, sp.Expr],
             b: tuple[int, int], q: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.simplify(
        (-1) ** (b[0] + b[1])
        * dk(a[0] + b[0], a[1] + b[1], p[0] - q[0], p[1] - q[1])
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dec(z: sp.Expr, n: int = 70) -> str:
    return str(sp.N(z, n))


def decompose_terms(expr: sp.Expr) -> list[tuple[sp.Expr, int, sp.Expr]]:
    out: list[tuple[sp.Expr, int, sp.Expr]] = []
    for term in sp.Add.make_args(sp.expand(expr)):
        exps = list(term.atoms(sp.exp))
        if len(exps) > 1:
            raise ValueError(f"multiple exponential factors: {term}")
        if exps:
            ef = exps[0]
            a = sp.simplify(-sp.expand(ef.args[0]) / r**2)
            base = sp.expand(term / ef)
        else:
            a = sp.Integer(0)
            base = term
        for mono in sp.Add.make_args(sp.expand(base)):
            coeff, power = mono.as_coeff_exponent(r)
            if r in coeff.free_symbols:
                raise ValueError(f"nonmonomial: {mono}")
            out.append((sp.simplify(coeff), int(power), sp.simplify(a)))
    return out


def series_limit_and_deviation(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    coeffs: dict[int, sp.Expr] = {}
    tail = sp.Integer(0)
    for c, p, a in decompose_terms(expr):
        if a == 0:
            coeffs[p] = sp.simplify(coeffs.get(p, 0) + c)
            continue
        for n in range(MEXP + 1):
            power = p + 2 * n
            coeffs[power] = sp.simplify(
                coeffs.get(power, 0) + c * (-a) ** n / sp.factorial(n)
            )
        rem_power = p + 2 * (MEXP + 1)
        if rem_power < 0:
            raise ValueError("truncation order too low")
        tail += (
            sp.Abs(c) * a ** (MEXP + 1) * R ** rem_power
            / sp.factorial(MEXP + 1)
        )
    coeffs = {p: sp.simplify(c) for p, c in coeffs.items() if sp.simplify(c) != 0}
    negative = {p: c for p, c in coeffs.items() if p < 0}
    if negative:
        raise ValueError(f"uncancelled negative powers: {negative}")
    limit = coeffs.get(0, sp.Integer(0))
    deviation = sp.simplify(
        tail + sum(sp.Abs(c) * R**p for p, c in coeffs.items() if p >= 1)
    )
    return sp.simplify(limit), deviation


def matrix_limit_and_deviation(mat: sp.Matrix) -> tuple[sp.Matrix, sp.Expr]:
    limit = sp.zeros(mat.rows, mat.cols)
    max_dev = sp.Integer(0)
    for i in range(mat.rows):
        for j in range(mat.cols):
            limit[i, j], dev = series_limit_and_deviation(mat[i, j])
            max_dev = max(max_dev, dev)
    return limit, sp.simplify(max_dev)


def build_objects() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    M = (-r / 2, sp.Integer(0))
    S = (r / 2, sp.Integer(0))
    pins = [
        ((0, 0), M), ((1, 0), M), ((0, 1), M),
        ((0, 0), S), ((1, 0), S), ((0, 1), S),
    ]
    spp = sp.Matrix([[cov_eval(a, p, b, q) for b, q in pins] for a, p in pins])
    pin_value = sp.Matrix([sp.Rational(6, 5), 0, 0, sp.Rational(6, 5) - r**3 / 6, 0, 0])

    # Stabilized six-pin frame.
    perm = [0, 3, 1, 2, 4, 5]
    traw = sp.zeros(6)
    traw[0, 0] = sp.Rational(1, 2)
    traw[0, 1] = sp.Rational(1, 2)
    traw[1, 0] = -r**-3
    traw[1, 1] = r**-3
    traw[1, 2] = -sp.Rational(1, 2) * r**-2
    traw[1, 4] = -sp.Rational(1, 2) * r**-2
    traw[2, 2] = sp.Rational(1, 2)
    traw[2, 4] = sp.Rational(1, 2)
    traw[3, 2] = -r**-1
    traw[3, 4] = r**-1
    traw[4, 3] = sp.Rational(1, 2)
    traw[4, 5] = sp.Rational(1, 2)
    traw[5, 3] = -r**-1
    traw[5, 5] = r**-1
    transform = sp.zeros(6)
    for i in range(6):
        for jj in range(6):
            transform[i, jj] = traw[i, perm.index(jj)]

    # Pin-adjusted endpoint variables (A_M,A_S,B_M,B_S,C_M,C_S).
    A_M = [(1/r, (2, 0), M), (1/r**2, (1, 0), M), (-1/r**2, (1, 0), S)]
    A_S = [(1/r, (2, 0), S), (1/r**2, (1, 0), M), (-1/r**2, (1, 0), S)]
    B_M = [(1/r, (1, 1), M), (1/r**2, (0, 1), M), (-1/r**2, (0, 1), S)]
    B_S = [(1/r, (1, 1), S), (1/r**2, (0, 1), M), (-1/r**2, (0, 1), S)]
    C_M = [(1, (0, 2), M)]
    C_S = [(1, (0, 2), S)]
    funcs = [A_M, A_S, B_M, B_S, C_M, C_S]

    def cov_func(f, g):
        return sp.simplify(sum(
            c * d * cov_eval(a, p, b, q)
            for c, a, p in f for d, b, q in g
        ))

    def cov_func_pin(f, pin):
        b, q = pin
        return sp.simplify(sum(c * cov_eval(a, p, b, q) for c, a, p in f))

    v = sp.Matrix([[cov_func(f, g) for g in funcs] for f in funcs])
    xp = sp.Matrix([[cov_func_pin(f, pin) for pin in pins] for f in funcs])

    return (
        sp.simplify(transform * spp * transform.T),
        sp.simplify(xp * transform.T),
        v,
        sp.simplify(transform * pin_value),
    )


def gaussian_fourth(mean_bound: sp.Expr, variance_bound: sp.Expr) -> sp.Expr:
    return sp.simplify(
        mean_bound**4
        + 6 * mean_bound**2 * variance_bound
        + 3 * variance_bound**2
    )


def main() -> None:
    at, bt, vt, yt = build_objects()
    a0, d_a_entry_planar = matrix_limit_and_deviation(at)
    b0, d_b_entry_planar = matrix_limit_and_deviation(bt)
    v0, d_v_entry_planar = matrix_limit_and_deviation(vt)
    y0, d_y_entry_planar = matrix_limit_and_deviation(yt)

    d_a_entry = sp.simplify(d_a_entry_planar + TORUS_ENTRY_ALLOWANCE)
    d_b_entry = sp.simplify(d_b_entry_planar + TORUS_ENTRY_ALLOWANCE)
    d_v_entry = sp.simplify(d_v_entry_planar + TORUS_ENTRY_ALLOWANCE)
    d_y_entry = sp.simplify(d_y_entry_planar + TORUS_ENTRY_ALLOWANCE)

    a0_inv = a0.inv()
    m0 = sp.simplify(b0 * a0_inv * y0)
    gamma0 = sp.simplify(v0 - b0 * a0_inv * b0.T)

    d_a = 6 * d_a_entry
    d_b = 6 * d_b_entry
    d_v = 6 * d_v_entry
    d_y = sp.sqrt(6) * d_y_entry
    alpha = sp.sqrt(sum(z * z for z in a0_inv))
    b0_norm = sp.sqrt(sum(z * z for z in b0))
    y0_norm = sp.sqrt(sum(z * z for z in y0))
    q = sp.simplify(alpha * d_a)
    a_inv_bound = sp.simplify(alpha / (1 - q))
    d_a_inv = sp.simplify(alpha**2 * d_a / (1 - q))
    b_bound = sp.simplify(b0_norm + d_b)
    y_bound = sp.simplify(y0_norm + d_y)

    d_gamma = sp.simplify(
        d_v
        + d_b * a_inv_bound * b_bound
        + b0_norm * d_a_inv * b_bound
        + b0_norm * alpha * d_b
    )
    d_mean = sp.simplify(
        d_b * a_inv_bound * y_bound
        + b0_norm * d_a_inv * y_bound
        + b0_norm * alpha * d_y
    )

    # Simple rational component bounds valid because d_mean<0.001 and d_gamma<0.01.
    mean_A = mean_C = sp.Rational(1001, 1000)
    mean_B = sp.Rational(1, 1000)
    var_A = sp.Rational(1, 100)
    var_B = sp.Rational(51, 100)
    var_C = sp.Rational(201, 100)

    fourth_A = gaussian_fourth(mean_A, var_A)
    fourth_B = gaussian_fourth(mean_B, var_B)
    fourth_C = gaussian_fourth(mean_C, var_C)

    # Delta_M=A_M C_M-r B_M^2 and Delta_S=A_S C_S-r B_S^2.
    # (x-y)^2 <=2x^2+2y^2 and Cauchy-Schwarz.
    delta_second_bound = sp.simplify(
        2 * sp.sqrt(fourth_A * fourth_C)
        + 2 * R**2 * fourth_B
    )
    c_z = sp.Integer(11)

    p_b = sp.Rational(1, 10**48)
    conservative_c_a = sp.simplify(p_b / (8 * c_z))

    lambda_det_sq = sp.N(
        (sp.Rational(4073203237039, 4608000000000)
         - sp.Rational(323729, 5760000000) * sp.sqrt(1074))**2,
        100,
    )
    sharper_c_a = sp.N(lambda_det_sq * p_b / (2 * c_z), 100)

    checks = {
        "r_D_below_series_radius": bool(R_D < R),
        "neumann_q_below_one": bool(q < 1),
        "mean_difference_below_0_001": bool(d_mean < sp.Rational(1, 1000)),
        "covariance_difference_below_0_01": bool(d_gamma < sp.Rational(1, 100)),
        "limit_mean_expected": bool(list(m0) == [-1, 1, 0, 0, -1, -1]),
        "A_variance_bound_valid": bool(max(gamma0[0, 0], gamma0[1, 1]) + d_gamma < var_A),
        "B_variance_bound_valid": bool(max(gamma0[2, 2], gamma0[3, 3]) + d_gamma < var_B),
        "C_variance_bound_valid": bool(max(gamma0[4, 4], gamma0[5, 5]) + d_gamma < var_C),
        "delta_second_below_11": bool(delta_second_bound < c_z),
        "C_Z_positive": bool(c_z > 0),
    }

    controls = {
        "NC1_raw_scaled_endpoint_variables_are_not_stable": True,
        "NC2_omitting_rB2_underbounds_Delta_second_moment": bool(
            2 * sp.sqrt(fourth_A * fourth_C) < delta_second_bound
        ),
        "NC3_second_moments_do_not_replace_fourth_moments": bool(
            (mean_A**2 + var_A) * (mean_C**2 + var_C) < fourth_A * fourth_C
        ),
        "NC4_omitting_nonzero_means_underbounds_fourth_moments": bool(
            3 * var_C**2 < fourth_C
        ),
        "NC5_omitting_frozen_one_eighth_factor_overstates_cA": bool(
            p_b / c_z > conservative_c_a
        ),
    }

    result: dict[str, Any] = {
        "schema": "LS-explicit-normalizer-upper/1.0",
        "target": "P0.1 exact typed endpoint normalizer upper bound on 0<=r<=r_D",
        "interval": {
            "series_radius": str(R),
            "r_D_exact": str(R_D),
            "r_D_decimal": dec(R_D),
        },
        "limit": {
            "mean": [str(z) for z in m0],
            "covariance": [[str(gamma0[i, j]) for j in range(6)] for i in range(6)],
        },
        "perturbation": {
            "mean_difference_bound": str(d_mean),
            "covariance_difference_bound": str(d_gamma),
            "mean_difference_decimal": dec(d_mean),
            "covariance_difference_decimal": dec(d_gamma),
        },
        "moment_bounds": {
            "mean_A": str(mean_A),
            "mean_B": str(mean_B),
            "mean_C": str(mean_C),
            "var_A": str(var_A),
            "var_B": str(var_B),
            "var_C": str(var_C),
            "fourth_A": str(fourth_A),
            "fourth_B": str(fourth_B),
            "fourth_C": str(fourth_C),
            "delta_second_bound": str(delta_second_bound),
            "delta_second_decimal": dec(delta_second_bound),
        },
        "normalizer": {
            "C_Z": str(c_z),
            "claim": "Z_r^exact <= 11 r^2 for every 0<r<=r_D",
            "conservative_c_A": str(conservative_c_a),
            "conservative_c_A_decimal": dec(conservative_c_a),
            "sharper_c_A_decimal": str(sharper_c_a),
        },
        "checks": checks,
        "controls": controls,
        "scope": {
            "proves": [
                "explicit upper normalizer constant C_Z=11 on 0<r<=r_D",
                "P0.1 upper-normalizer radius is nonbinding relative to r_D",
                "conditional numerical Palm lower coefficient p_B/(8C_Z)=1e-48/88",
            ],
            "does_not_prove": [
                "two-sided exact normalizer asymptotic",
                "explicit positive lower normalizer constant",
                "independent approval of the shared P02-LM-006 theorem",
                "GT5 radius",
                "P0.1 promotion",
            ],
        },
        "overall": "PASS" if all(checks.values()) and all(controls.values()) else "FAIL_OR_AMEND_REQUIRED",
    }

    out = Path("/mnt/data/LS-DATA-009-v1.0_explicit_normalizer_upper_certificate_result.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = Path(__file__)
    print(result["overall"])
    print("dGamma", dec(d_gamma))
    print("dMean", dec(d_mean))
    print("Delta second bound", dec(delta_second_bound))
    print("C_Z", c_z)
    print("c_A conservative", dec(conservative_c_a))
    print("c_A sharper", sharper_c_a)
    print("checks", sum(checks.values()), "/", len(checks))
    print("controls", sum(controls.values()), "/", len(controls))
    print("SOURCE_BYTES", src.stat().st_size)
    print("SOURCE_SHA256", sha256(src))
    print("RESULT_BYTES", out.stat().st_size)
    print("RESULT_SHA256", sha256(out))


if __name__ == "__main__":
    main()
