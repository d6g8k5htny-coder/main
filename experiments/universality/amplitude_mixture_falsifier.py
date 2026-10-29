#!/usr/bin/env python3
"""Exact conditional covariance-only falsifier from one sourced amplitude law.

Engineering/metamorphic control, not a mathematical review or status authority.
The caller supplies an unverified expected base short-bar count lower bound.
All displayed scalar equalities are exact rational/symbolic identities.
No field sample, barcode algorithm, source code, or remote URL is executed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import re
import sys

MAX_INPUT = 32 * 1024
RATIONAL = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:/[1-9][0-9]{0,8})?\Z")
TOP_KEYS = frozenset(("schema_version", "power", "bar_interval",
                      "base_expected_bar_count_lower_bound", "grid"))
SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "9f3fb6918b28e6378c034ebc19e76d432f5391cb",
    "path": ("frontiers/gaussian_short_bar_theory_20261017/"
             "smooth_gaussian_manifold_admission/PROOF.md"),
    "git_blob": "d61fd11a826856baecb4cf4cd8a76ad5c25350dd",
    "locator": "section 11, covariance-only amplitude-mixture boundary",
}


class SpecRefused(ValueError):
    """Do not emit caller data or a parsed path in refusal diagnostics."""


def _refuse():
    raise SpecRefused("AMP_TWIN_REFUSED: invalid-input")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _refuse()
        result[key] = value
    return result


def _bad_number(_token):
    _refuse()


def load_spec(raw: bytes) -> dict:
    """Parse inert, size-limited JSON; reject duplicate keys and JSON floats."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_INPUT:
        _refuse()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"),
                           object_pairs_hook=_pairs,
                           parse_constant=_bad_number,
                           parse_float=_bad_number,
                           parse_int=_int_token)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        _refuse()
    if type(value) is not dict or value.keys() != TOP_KEYS:
        _refuse()
    # The source loader is an acceptance boundary: reject type-correct JSON
    # that violates the finite witness contract before returning it to callers.
    witness(value)
    return value


def _int_token(token: str):
    if len(token) > 8:
        _refuse()
    return int(token)


def _fraction(value) -> Fraction:
    if type(value) is not str or RATIONAL.fullmatch(value) is None:
        _refuse()
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        _refuse()


def _show(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def witness(spec: dict) -> dict:
    """Compute only an exact conditional counterexample to covariance transfer.

    R = sqrt(2k+1) U**k, where independent U is uniform on (0,1).
    An external premise E N_H([a,b]) >= p > 0 is explicitly unverified.
    At t_m = sqrt(2k+1)*b/m**k, E N_(RH)((0,t_m]) >= p/m.
    """
    if type(spec) is not dict or spec.keys() != TOP_KEYS:
        _refuse()
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        _refuse()
    k = spec["power"]
    if type(k) is not int or not 2 <= k <= 16:
        _refuse()
    interval = spec["bar_interval"]
    if type(interval) is not dict or interval.keys() != {"lower", "upper"}:
        _refuse()
    a, b = _fraction(interval["lower"]), _fraction(interval["upper"])
    p = _fraction(spec["base_expected_bar_count_lower_bound"])
    if not (0 < a < b and p > 0):
        _refuse()
    grid = spec["grid"]
    if (type(grid) is not list or not 1 <= len(grid) <= 20
            or any(type(m) is not int or not 2 <= m <= 100_000 for m in grid)
            or any(x >= y for x, y in zip(grid, grid[1:]))):
        _refuse()

    fourth = Fraction((2 * k + 1)**2, 4 * k + 1)
    excess = fourth - 1
    growth = Fraction(2 * k, 3) - 1
    if not (fourth > 1 and growth > 0):
        _refuse()
    ratio_expression = f"{_show(p)} * m^({_show(growth)})"
    examples = [
        {
            "m": m,
            "t_over_sqrt_2kplus1_times_upper_bar": _show(Fraction(1, m**k)),
            "probability_R_at_m_threshold": _show(Fraction(1, m)),
            "expected_short_bar_count_lower_bound": _show(p / m),
        }
        for m in grid
    ]
    return {
        "schema_version": 1,
        "kind": "CONDITIONAL_COVARIANCE_ONLY_FALSIFIER",
        "scientific_effect": "NONE",
        "scientific_status_authority": False,
        "source_identity": dict(SOURCE),
        "source_identity_runtime_verified": False,
        "source_pin_meaning": "declared immutable source reference, not runtime Git authentication",
        "premise_status": "CALLER_SUPPLIED_NOT_VERIFIED",
        "premises": {
            "bar_interval_lower": _show(a),
            "bar_interval_upper": _show(b),
            "expected_base_bar_count_lower_bound": _show(p),
            "underlying_field": "centered Gaussian H independent of U; nonzero finite bars required",
        },
        "scale_model": {
            "definition": "R=sqrt(2*k+1)*U**k; U uniform on (0,1) independent of H",
            "integer_power": k,
            "small_amplitude_cdf_exponent": _show(Fraction(1, k)),
            "second_scale_moment": "1",
            "fourth_scale_moment": _show(fourth),
            "fourth_scale_moment_minus_one": _show(excess),
            "normalized_gaussian_fourth_moment": "3",
            "normalized_mixture_fourth_moment": _show(3 * fourth),
            "inverse_two_thirds_scale_moment": "DIVERGES",
            "same_covariance_for_independent_centered_H": True,
            "positive_scale_preserves_each_realization_bar_pairing": True,
        },
        "conditional_short_bar_witness": {
            "threshold_definition": "t_m=sqrt(2*k+1)*b/m**k",
            "comparison_exponent": "2/3",
            "relative_lower_bound_ratio_to_threshold_power": ratio_expression,
            "growth_exponent": _show(growth),
            "positive_divergence_exponent": True,
            "grid": examples,
        },
        "covariance_only_two_thirds_transfer": "REFUSED_BY_COUNTEREXAMPLE",
        "scope_limits": [
            "No actual Gaussian field, physical persistence barcode, or finite-radius count is computed",
            "The positive base-bar expectation is an unverified external premise",
            "The lower bound does not establish a full non-Gaussian density asymptotic",
            "The existing Gaussian theorem and its reviewed hypotheses are not revised",
            "An engineering veto is not a theorem-status or independent-review decision",
        ],
    }


EXAMPLE_SPEC = {
    "schema_version": 1,
    "power": 2,
    "bar_interval": {"lower": "1/2", "upper": "3/2"},
    "base_expected_bar_count_lower_bound": "1/4",
    "grid": [2, 3, 5, 9],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="inert JSON input file, or '-' for stdin")
    source.add_argument("--example", action="store_true", help="print a synthetic example input")
    parser.add_argument("--require-covariance-transfer", action="store_true",
                        help="fail on the invalid covariance-only 2/3 inference")
    args = parser.parse_args(argv)
    try:
        if args.example:
            if args.require_covariance_transfer:
                _refuse()
            result = EXAMPLE_SPEC
        else:
            if args.spec == "-":
                raw = sys.stdin.buffer.read(MAX_INPUT + 1)
            else:
                with open(args.spec, "rb") as f:
                    raw = f.read(MAX_INPUT + 1)
            result = witness(load_spec(raw))
    except (SpecRefused, OSError, ValueError, OverflowError, RecursionError):
        print("AMP_TWIN_REFUSED: invalid-input", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")
    # Veto mode makes an impossible covariance-only transfer a machine failure.
    return 1 if args.require_covariance_transfer else 0


if __name__ == "__main__":
    raise SystemExit(main())
