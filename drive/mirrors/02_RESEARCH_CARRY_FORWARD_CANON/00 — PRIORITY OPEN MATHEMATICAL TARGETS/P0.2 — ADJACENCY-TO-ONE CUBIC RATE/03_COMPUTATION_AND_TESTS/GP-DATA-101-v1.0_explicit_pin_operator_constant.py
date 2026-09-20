#!/usr/bin/env python3
"""Exact conservative C_D bound for the GP-DER-045 pin-corrected degree-four transfer.

Scope: the single normalized chart D={|X|<=3/4, |Y|<=1}; H5 is the maximum
absolute physical fifth partial derivative on rD. The C1 norm uses Euclidean
vector norm for the gradient and operator norm for the symmetric Jacobian.
This derives a sufficient deterministic constant only. It does not provide H5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

getcontext().prec = 90
D = Decimal
F = Fraction


def dec(q: Fraction) -> Decimal:
    return D(q.numerator) / D(q.denominator)


def sqrt_q(q: Fraction) -> Decimal:
    return dec(q).sqrt()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive() -> dict[str, Any]:
    # Chart D: |X|<=3/4, |Y|<=1. For multivariate Taylor remainder,
    # |X|+|Y| <= 7/4.
    L = F(7, 4)

    # Ordinary degree-four Taylor contribution to grad (C0) and Hessian (C0).
    ordinary_grad_component = L**4 / F(24, 1)
    ordinary_vector_euclidean = sqrt_q(2 * ordinary_grad_component**2)
    ordinary_hessian_entry = L**3 / F(6, 1)
    ordinary_jacobian_operator_upper = dec(2 * ordinary_hessian_entry)  # Frobenius

    # Endpoint Taylor residuals at X=±1/2,Y=0.
    value_residual = F(1, 2) ** 5 / F(120, 1)      # 1/3840
    gradient_residual = F(1, 2) ** 4 / F(24, 1)   # 1/384

    # Cubic Hermite basis on t=X+1/2, with t∈[-1/4,5/4] on the full chart.
    # Exact suprema of derivatives on that interval:
    basis = {
        "sum_abs_value_cardinal_first_derivatives": F(15, 4),  # h00',h01'
        "sum_abs_xder_cardinal_first_derivatives": F(35, 8),   # h10',h11'
        "sum_abs_ylinear_basis_derivatives": F(2, 1),
        "sum_abs_ylinear_basis_values": F(3, 2),
        "sum_abs_value_cardinal_second_derivatives": F(18, 1),
        "sum_abs_xder_cardinal_second_derivatives": F(11, 1),
    }

    pin_px = (
        value_residual * basis["sum_abs_value_cardinal_first_derivatives"]
        + gradient_residual * basis["sum_abs_xder_cardinal_first_derivatives"]
        + gradient_residual * basis["sum_abs_ylinear_basis_derivatives"]
    )
    pin_py = gradient_residual * basis["sum_abs_ylinear_basis_values"]
    pin_vector_euclidean = sqrt_q(pin_px**2 + pin_py**2)

    pin_pxx = (
        value_residual * basis["sum_abs_value_cardinal_second_derivatives"]
        + gradient_residual * basis["sum_abs_xder_cardinal_second_derivatives"]
    )
    pin_pxy = gradient_residual * basis["sum_abs_ylinear_basis_derivatives"]
    pin_jacobian_operator_upper = sqrt_q(pin_pxx**2 + 2 * pin_pxy**2)  # Frobenius

    vector_C = ordinary_vector_euclidean + pin_vector_euclidean
    jacobian_C = ordinary_jacobian_operator_upper + pin_jacobian_operator_upper
    C_D = max(vector_C, jacobian_C)

    # Imported fixture-specific budget from GP-DATA-096.
    delta_accept = D("0.004641091949159561998550871359023117272392841003373501115178370844477885")
    r = D("0.05")
    H5_threshold = delta_accept / (C_D * r * r)

    # Mutations that must be detected as underestimates / wrong objects.
    wrong_no_pin = ordinary_jacobian_operator_upper

    # Incorrectly restrict Hermite basis to between pins t∈[0,1].
    pin_px_between = value_residual * F(3, 1) + gradient_residual * F(2, 1) + gradient_residual * F(2, 1)
    pin_py_between = gradient_residual * F(1, 1)
    pin_pxx_between = value_residual * F(12, 1) + gradient_residual * F(8, 1)
    pin_pxy_between = gradient_residual * F(2, 1)
    wrong_between = ordinary_jacobian_operator_upper + sqrt_q(pin_pxx_between**2 + 2 * pin_pxy_between**2)

    wrong_max_entry = dec(ordinary_hessian_entry) + max(dec(pin_pxx), dec(pin_pxy))
    wrong_omit_value_residual = ordinary_jacobian_operator_upper + sqrt_q((gradient_residual * F(11,1))**2 + 2*pin_pxy**2)
    wrong_C0_only = vector_C

    controls = [
        {
            "id": "NC1_OMIT_PIN_CORRECTION",
            "observed": "UNDERBOUNDS_C_D",
            "pass": wrong_no_pin < C_D,
            "wrong_constant": str(wrong_no_pin),
        },
        {
            "id": "NC2_USE_ONLY_BETWEEN_PIN_DOMAIN",
            "observed": "UNDERBOUNDS_FULL_CHART_OPERATOR",
            "pass": wrong_between < C_D,
            "wrong_constant": str(wrong_between),
        },
        {
            "id": "NC3_USE_MAX_ENTRY_NOT_OPERATOR_NORM",
            "observed": "WRONG_JACOBIAN_NORM",
            "pass": wrong_max_entry < C_D,
            "wrong_constant": str(wrong_max_entry),
        },
        {
            "id": "NC4_OMIT_VALUE_PIN_RESIDUALS",
            "observed": "INCOMPLETE_PIN_OPERATOR",
            "pass": wrong_omit_value_residual < C_D,
            "wrong_constant": str(wrong_omit_value_residual),
        },
        {
            "id": "NC5_USE_C0_ONLY_FOR_C1_TRANSFER",
            "observed": "CANNOT_VERIFY_C1",
            "pass": wrong_C0_only < C_D,
            "wrong_constant": str(wrong_C0_only),
        },
    ]

    return {
        "schema": "explicit-pin-operator-CD/1.0",
        "chart": {"X_abs_max": "3/4", "Y_abs_max": "1", "L1_radius": "7/4"},
        "norm": "max(Euclidean norm of gradient error, operator norm of symmetric Jacobian error)",
        "ordinary_taylor": {
            "gradient_component_fraction": str(ordinary_grad_component),
            "gradient_vector_constant": str(ordinary_vector_euclidean),
            "hessian_entry_fraction": str(ordinary_hessian_entry),
            "jacobian_operator_upper": str(ordinary_jacobian_operator_upper),
        },
        "endpoint_residual_coefficients": {
            "value": str(value_residual),
            "gradient": str(gradient_residual),
        },
        "hermite_full_chart_suprema": {k: str(v) for k, v in basis.items()},
        "pin_correction": {
            "px_fraction": str(pin_px),
            "py_fraction": str(pin_py),
            "vector_constant": str(pin_vector_euclidean),
            "pxx_fraction": str(pin_pxx),
            "pxy_fraction": str(pin_pxy),
            "jacobian_operator_upper": str(pin_jacobian_operator_upper),
        },
        "combined": {
            "vector_C": str(vector_C),
            "jacobian_C": str(jacobian_C),
            "C_D_conservative": str(C_D),
            "binding_part": "jacobian_C",
        },
        "fixture_translation": {
            "fixture_id": "PRCP-INTERIOR-001",
            "delta_accept": str(delta_accept),
            "r": str(r),
            "H5_threshold": str(H5_threshold),
            "criterion": "H5(D) <= H5_threshold implies C_D*r^2*H5(D) <= delta_accept",
        },
        "negative_controls": controls,
        "overall": "PASS_EXPLICIT_C_D_BOUND_ONLY_H5_MEMBERSHIP_OPEN" if all(c["pass"] for c in controls) else "FAIL_CONTROL",
        "explicit_exclusions": [
            "NO claim the exact field satisfies the H5 threshold",
            "NO probability, occupancy, exponent, P0.2, or Theorem B claim",
            "NO replacement of GP-DER-045 or CW-AUD-018",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="GP-DATA-101-v1.0_receipt.json")
    args = ap.parse_args()
    result = derive()
    src = Path(__file__)
    result["source_filename"] = src.name
    result["source_bytes"] = len(src.read_bytes())
    result["source_sha256"] = source_sha256(src)
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["overall"])
    print("C_D", result["combined"]["C_D_conservative"])
    print("H5_THRESHOLD", result["fixture_translation"]["H5_threshold"])
    print("CONTROLS", sum(bool(c["pass"]) for c in result["negative_controls"]), "/", len(result["negative_controls"]))
    print("SOURCE_SHA256", result["source_sha256"])
    print("RECEIPT_SHA256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
