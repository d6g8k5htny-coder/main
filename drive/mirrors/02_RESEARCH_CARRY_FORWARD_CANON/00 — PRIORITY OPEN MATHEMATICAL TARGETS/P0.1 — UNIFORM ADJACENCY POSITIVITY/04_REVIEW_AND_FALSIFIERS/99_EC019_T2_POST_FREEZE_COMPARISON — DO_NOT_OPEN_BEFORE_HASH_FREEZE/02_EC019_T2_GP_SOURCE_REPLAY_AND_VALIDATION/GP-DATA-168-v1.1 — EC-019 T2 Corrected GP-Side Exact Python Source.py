#!/usr/bin/env python3
"""
GP-DATA-168-v1.1 — EC-019 T2 corrected GP-side interval preflight.

Purpose
-------
Repair the certificate-level defects identified in GP-AUD-176-v1.0:
1. execute the zero-width/full-dimensionality control instead of asserting it;
2. widen Decimal square roots to guaranteed outward endpoints;
3. keep endpoint-derived inequalities inside directed interval arithmetic;
4. add regression and multi-precision stability tests.

Scientific status
-----------------
This is GP/OpenAI-authored same-family work. It earns no independent T2 credit.
A qualifying T2 verdict still requires an organizationally distinct, independently
authored implementation frozen before result comparison.

Normalization: UTF-8, LF newlines, no BOM, exactly one trailing LF.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    getcontext,
    localcontext,
)
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Callable, Dict, Iterable, Mapping, Tuple

D = Decimal
PREC = 80
getcontext().prec = PREC

PARAMETER_NAMES = ("a", "w", "z", "s", "c40", "c31", "c22", "c13", "c04")


def dec(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def calc(fn: Callable[[], Decimal], rounding: str) -> Decimal:
    """Evaluate one Decimal operation at the active precision and direction."""
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return fn()


def next_minus(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PREC
        return ctx.next_minus(x)


def next_plus(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PREC
        return ctx.next_plus(x)


class I:
    """Closed Decimal interval with outward-rounded elementary operations."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = dec(lo)
        self.hi = dec(lo if hi is None else hi)
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    def __repr__(self) -> str:
        return f"I({self.lo!s}, {self.hi!s})"

    def __add__(self, other):
        other = as_interval(other)
        return I(
            calc(lambda: self.lo + other.lo, ROUND_FLOOR),
            calc(lambda: self.hi + other.hi, ROUND_CEILING),
        )

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_interval(other))

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        lower_products = []
        upper_products = []
        for a in (self.lo, self.hi):
            for b in (other.lo, other.hi):
                lower_products.append(
                    calc(lambda a=a, b=b: a * b, ROUND_FLOOR)
                )
                upper_products.append(
                    calc(lambda a=a, b=b: a * b, ROUND_CEILING)
                )
        return I(min(lower_products), max(upper_products))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("interval contains zero")
        lower_values = [
            calc(lambda: D(1) / self.lo, ROUND_FLOOR),
            calc(lambda: D(1) / self.hi, ROUND_FLOOR),
        ]
        upper_values = [
            calc(lambda: D(1) / self.lo, ROUND_CEILING),
            calc(lambda: D(1) / self.hi, ROUND_CEILING),
        ]
        return I(min(lower_values), max(upper_values))

    def __truediv__(self, other):
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other):
        return as_interval(other) / self

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("only nonnegative integer powers")
        if exponent == 0:
            return I(1)
        if exponent % 2 == 1:
            return I(
                calc(lambda: self.lo ** exponent, ROUND_FLOOR),
                calc(lambda: self.hi ** exponent, ROUND_CEILING),
            )

        upper_candidates = [
            calc(lambda: self.lo ** exponent, ROUND_CEILING),
            calc(lambda: self.hi ** exponent, ROUND_CEILING),
        ]
        if self.lo <= 0 <= self.hi:
            return I(0, max(upper_candidates))

        lower_candidates = [
            calc(lambda: self.lo ** exponent, ROUND_FLOOR),
            calc(lambda: self.hi ** exponent, ROUND_FLOOR),
        ]
        return I(min(lower_candidates), max(upper_candidates))

    def absolute(self):
        if self.lo >= 0:
            return self
        if self.hi <= 0:
            return -self
        return I(0, max(-self.lo, self.hi))

    def sqrt(self):
        """Guaranteed enclosure despite Decimal.sqrt using nearest rounding."""
        if self.lo < 0:
            raise ValueError("sqrt of negative interval")
        with localcontext() as ctx:
            ctx.prec = PREC
            lo_nearest = self.lo.sqrt(context=ctx)
            hi_nearest = self.hi.sqrt(context=ctx)
        return I(next_minus(lo_nearest), next_plus(hi_nearest))

    def as_list(self):
        return [str(self.lo), str(self.hi)]


def as_interval(x):
    return x if isinstance(x, I) else I(x)


def supabs_upper(x) -> Decimal:
    """Exact maximum of already-outward interval endpoint magnitudes."""
    x = as_interval(x)
    return max(abs(x.lo), abs(x.hi))


@dataclass(frozen=True)
class ParameterBox:
    centers: Mapping[str, Decimal]
    radii: Mapping[str, Decimal]

    def intervals(self) -> Dict[str, I]:
        return {
            name: I(
                calc(
                    lambda name=name: self.centers[name] - self.radii[name],
                    ROUND_FLOOR,
                ),
                calc(
                    lambda name=name: self.centers[name] + self.radii[name],
                    ROUND_CEILING,
                ),
            )
            for name in PARAMETER_NAMES
        }

    def altered(self, key: str, center, radius) -> "ParameterBox":
        if key not in PARAMETER_NAMES:
            raise KeyError(key)
        centers = dict(self.centers)
        radii = dict(self.radii)
        centers[key] = dec(center)
        radii[key] = dec(radius)
        return ParameterBox(centers=centers, radii=radii)


class FullDimensionError(ValueError):
    pass


def validate_full_dimension(box: ParameterBox) -> None:
    missing = [name for name in PARAMETER_NAMES if name not in box.radii]
    if missing:
        raise FullDimensionError(f"FULL_DIMENSIONAL_BOX_FAIL missing={missing}")
    nonpositive = [
        name for name in PARAMETER_NAMES if dec(box.radii[name]) <= 0
    ]
    if nonpositive:
        raise FullDimensionError(
            "FULL_DIMENSIONAL_BOX_FAIL nonpositive_radii="
            + ",".join(nonpositive)
        )


def primary_box() -> ParameterBox:
    centers = {
        "a": D("0.0025"),
        "w": D("0"),
        "z": D("1"),
        "s": D("-1"),
        "c40": D("0"),
        "c31": D("0"),
        "c22": D("0"),
        "c13": D("0"),
        "c04": D("0"),
    }
    radii = {
        "a": D("0.0001"),
        "w": D("0.001"),
        "z": D("0.01"),
        "s": D("0.02"),
        "c40": D("0.01"),
        "c31": D("0.01"),
        "c22": D("0.01"),
        "c13": D("0.01"),
        "c04": D("0.01"),
    }
    return ParameterBox(centers=centers, radii=radii)


def constants() -> Tuple[I, I, I]:
    r = I(1) / 20
    sigma = I(1) / 10
    sqrt2 = I(2).sqrt()
    mu = I(1) / (I(40) * sqrt2)
    return r, sigma, mu


def flow_x(X, Y, p, R):
    a, w, c40, c31, c22, c13 = [
        p[k] for k in ("a", "w", "c40", "c31", "c22", "c13")
    ]
    return (
        X**2
        - I(1) / 4
        + a * X * Y
        + (w / 2) * (Y**2)
        + R
        * (
            c40 * (X**3 - X / 4) / 6
            + c31 * (3 * (X**2) - I(1) / 4) * Y / 6
            + c22 * X * (Y**2) / 2
            + c13 * (Y**3) / 6
        )
    )


def flow_y(X, Y, p, R):
    a, w, z, s, c31, c22, c13, c04 = [
        p[k] for k in ("a", "w", "z", "s", "c31", "c22", "c13", "c04")
    ]
    return (
        a * (X**2 - I(1) / 4) / 2
        + s * Y
        + w * X * Y
        + (z / 2) * (Y**2)
        + R
        * (
            c31 * (X**3 - X / 4) / 6
            + c22 * (X**2) * Y / 2
            + c13 * X * (Y**2) / 2
            + c04 * (Y**3) / 6
        )
    )


def jac11(X, Y, p, R):
    a, c40, c31, c22 = [p[k] for k in ("a", "c40", "c31", "c22")]
    return 2 * X + a * Y + R * (
        c40 * (3 * (X**2) - I(1) / 4) / 6
        + c31 * X * Y
        + c22 * (Y**2) / 2
    )


def jac12(X, Y, p, R):
    a, w, c31, c22, c13 = [
        p[k] for k in ("a", "w", "c31", "c22", "c13")
    ]
    return a * X + w * Y + R * (
        c31 * (3 * (X**2) - I(1) / 4) / 6
        + c22 * X * Y
        + c13 * (Y**2) / 2
    )


def jac22(X, Y, p, R):
    s, w, z, c22, c13, c04 = [
        p[k] for k in ("s", "w", "z", "c22", "c13", "c04")
    ]
    return s + w * X + z * Y + R * (
        c22 * (X**2) / 2 + c13 * X * Y + c04 * (Y**2) / 2
    )


def compute(box: ParameterBox, kappa_value="0.01"):
    validate_full_dimension(box)
    p = box.intervals()
    R, SIGMA, MU = constants()
    kappa = I(kappa_value)
    a, w, z, s, c40, c31, c22, c13, c04 = [
        p[k] for k in PARAMETER_NAMES
    ]

    witness_f = a.absolute() / 8 + R * c31.absolute() / 48
    witness_a = (z.absolute() + R * c13.absolute() / 2) * witness_f
    witness_b = R * c04.absolute() * (witness_f**2) / 2
    threshold_c = (
        w.absolute() / 2
        + MU
        + R * c22.absolute() / 8
        + witness_a / MU
        + witness_b / (MU**2)
    )
    width = witness_f / MU

    margins = {}
    margins["m_threshold"] = (-s - threshold_c).lo

    a_s = I(1) + R * c40 / 12
    b_s = a / 2 + R * c31 / 12
    d_s = s + w / 2 + R * c22 / 8
    det_s = a_s * d_s - b_s**2

    a_m = -I(1) + R * c40 / 12
    b_m = -a / 2 + R * c31 / 12
    d_m = s - w / 2 + R * c22 / 8
    det_m = a_m * d_m - b_m**2

    gap = a_s - d_s
    if gap.lo <= 0:
        tau_upper = D("Infinity")
    else:
        tau_interval = I(supabs_upper(b_s)) / I(gap.lo)
        tau_upper = tau_interval.hi

    margins["m_saddle_det"] = -det_s.hi
    margins["m_max_A"] = -a_m.hi
    margins["m_max_D"] = -d_m.hi
    margins["m_max_det"] = det_m.lo
    margins["m_axial"] = gap.lo
    margins["m_cone_slope"] = (kappa - 2 * I(tau_upper)).lo

    xi = I(0, SIGMA.hi)

    def qx(xi, t):
        return (
            12 * a * t * xi
            - 6 * a * t
            - 2 * c13 * R * (t**3) * (xi**2)
            + 6 * c22 * R * (t**2) * (xi**2)
            - 3 * c22 * R * (t**2) * xi
            - 6 * c31 * R * t * (xi**2)
            + 6 * c31 * R * t * xi
            - c31 * R * t
            + 2 * c40 * R * (xi**2)
            - 3 * c40 * R * xi
            + c40 * R
            - 6 * (t**2) * w * xi
            - 12 * xi
            + 12
        ) / 12

    def qy(xi, t):
        return (
            12 * a * xi
            - 12 * a
            + 4 * c04 * R * (t**3) * (xi**2)
            - 12 * c13 * R * (t**2) * (xi**2)
            + 6 * c13 * R * (t**2) * xi
            + 12 * c22 * R * t * (xi**2)
            - 12 * c22 * R * t * xi
            + 3 * c22 * R * t
            - 4 * c31 * R * (xi**2)
            + 6 * c31 * R * xi
            - 2 * c31 * R
            + 24 * s * t
            + 12 * (t**2) * xi * z
            - 24 * t * w * xi
            + 12 * t * w
        ) / 24

    t_full = I(-kappa.hi, kappa.hi)
    t_plus = I(kappa.lo)
    t_minus = I(-kappa.lo)
    margins["m_cone_axis"] = qx(xi, t_full).lo
    margins["m_cone_upper"] = (kappa * qx(xi, t_plus) - qy(xi, t_plus)).lo
    margins["m_cone_lower"] = (kappa * qx(xi, t_minus) + qy(xi, t_minus)).lo
    margins["m_handoff"] = (width / 2 - kappa * SIGMA).lo

    x_central = I(
        (-I(1) / 2 + SIGMA).lo,
        (I(1) / 2 - SIGMA).hi,
    )
    y_top = width
    y_bottom = -width
    y_strip = I(-width.hi, width.hi)

    margins["m_strip_top"] = (-flow_y(x_central, y_top, p, R)).lo
    margins["m_strip_bottom"] = flow_y(x_central, y_bottom, p, R).lo
    margins["m_central_drift"] = (-flow_x(x_central, y_strip, p, R)).lo
    margins["m_transverse_contraction"] = -jac22(
        x_central, y_strip, p, R
    ).hi

    rho = 2 * ((SIGMA**2 + width**2).sqrt())
    rho_upper = rho.hi
    center_x = -I(1) / 2
    x_capture = center_x + I(-rho_upper, rho_upper)
    y_capture = I(-rho_upper, rho_upper)

    j11 = jac11(x_capture, y_capture, p, R)
    j12 = jac12(x_capture, y_capture, p, R)
    j22 = jac22(x_capture, y_capture, p, R)
    offdiag = I(supabs_upper(j12))

    margins["m_chart_X"] = (I(3) / 4 - (I(1) / 2 + I(rho_upper))).lo
    margins["m_chart_Y"] = (I(1) - I(rho_upper)).lo
    margins["m_gersh_1"] = (-j11 - offdiag).lo
    margins["m_gersh_2"] = (-j22 - offdiag).lo

    extras = {
        "F": witness_f.as_list(),
        "A": witness_a.as_list(),
        "B": witness_b.as_list(),
        "C": threshold_c.as_list(),
        "W": width.as_list(),
        "rho": rho.as_list(),
        "tau_upper": str(tau_upper),
    }
    return margins, extras


def execute_zero_width_control(base: ParameterBox):
    cases = {}
    for name in PARAMETER_NAMES:
        trial = base.altered(name, base.centers[name], 0)
        try:
            compute(trial)
            cases[name] = {
                "pass": False,
                "result": "ZERO_RADIUS_REACHED_DYNAMICS",
            }
        except FullDimensionError as exc:
            cases[name] = {
                "pass": True,
                "result": str(exc),
            }
    return {
        "pass": all(case["pass"] for case in cases.values()),
        "cases": cases,
    }


def execute_controls(base: ParameterBox):
    nc1 = execute_zero_width_control(base)
    nc2_margins, _ = compute(base.altered("c40", 0, 500))
    nc3_margins, _ = compute(base.altered("s", -1, "1.1"))
    nc4_margins, _ = compute(base, kappa_value="0.0005")
    nc5_margins, _ = compute(base.altered("a", "0.0025", "0.5"))

    return {
        "NC1_zero_width": nc1,
        "NC2_endpoint_typing": {
            "pass": nc2_margins["m_saddle_det"] <= 0
            or nc2_margins["m_max_det"] <= 0,
            "m_saddle_det": str(nc2_margins["m_saddle_det"]),
            "m_max_det": str(nc2_margins["m_max_det"]),
        },
        "NC3_threshold": {
            "pass": nc3_margins["m_threshold"] <= 0,
            "m_threshold": str(nc3_margins["m_threshold"]),
        },
        "NC4_cone": {
            "pass": nc4_margins["m_cone_slope"] <= 0,
            "m_cone_slope": str(nc4_margins["m_cone_slope"]),
        },
        "NC5_capture_chart": {
            "pass": nc5_margins["m_chart_X"] <= 0
            or nc5_margins["m_gersh_1"] <= 0,
            "m_chart_X": str(nc5_margins["m_chart_X"]),
            "m_gersh_1": str(nc5_margins["m_gersh_1"]),
        },
    }


def run_at_precision(precision: int):
    global PREC
    old_prec = PREC
    old_context_prec = getcontext().prec
    try:
        PREC = precision
        getcontext().prec = precision
        base = primary_box()
        validate_full_dimension(base)
        margins, extras = compute(base)
        controls = execute_controls(base)
        return {
            "precision": precision,
            "primary_pass": all(value > 0 for value in margins.values()),
            "minimum_margin": min(
                [
                    {"name": key, "value": str(value)}
                    for key, value in margins.items()
                ],
                key=lambda item: D(item["value"]),
            ),
            "margins": {key: str(value) for key, value in margins.items()},
            "extras": extras,
            "negative_controls": controls,
            "all_controls_pass": all(
                item["pass"] for item in controls.values()
            ),
        }
    finally:
        PREC = old_prec
        getcontext().prec = old_context_prec


def regression_tests():
    failures = []

    # 1. Primary box passes the full-dimensionality gate.
    try:
        validate_full_dimension(primary_box())
    except Exception as exc:  # pragma: no cover - audit output needs exact failure
        failures.append(f"primary_full_dimension: {exc}")

    # 2. All nine zero-radius mutations are machine-rejected.
    nc1 = execute_zero_width_control(primary_box())
    if not nc1["pass"]:
        failures.append("zero_width_mutations_not_all_rejected")

    # 3. Square-root enclosure is genuinely outward at deliberately low precision.
    global PREC
    old_prec = PREC
    old_context_prec = getcontext().prec
    try:
        PREC = 10
        getcontext().prec = 10
        root = I(2).sqrt()
        if calc(lambda: root.lo * root.lo, ROUND_FLOOR) > D(2):
            failures.append("sqrt_lower_endpoint_above_exact_root")
        if calc(lambda: root.hi * root.hi, ROUND_CEILING) < D(2):
            failures.append("sqrt_upper_endpoint_below_exact_root")
        if root.lo == root.hi:
            failures.append("sqrt_endpoints_not_widened")
    finally:
        PREC = old_prec
        getcontext().prec = old_context_prec

    # 4. Even power crossing zero has exact zero lower bound and contains 9.
    squared = I(-2, 3) ** 2
    if squared.lo != 0 or squared.hi < 9:
        failures.append(f"even_power_bad_enclosure={squared!r}")

    # 5. Simple interval arithmetic contains endpoint truths.
    q = I(1, 2) / I(3, 4)
    if q.lo > D(1) / D(4) or q.hi < D(2) / D(3):
        failures.append(f"division_bad_enclosure={q!r}")

    # 6. All five controls execute and fire.
    controls = execute_controls(primary_box())
    if set(controls) != {
        "NC1_zero_width",
        "NC2_endpoint_typing",
        "NC3_threshold",
        "NC4_cone",
        "NC5_capture_chart",
    }:
        failures.append("control_set_mismatch")
    if not all(control["pass"] for control in controls.values()):
        failures.append("one_or_more_negative_controls_failed")

    return {
        "pass": not failures,
        "failures": failures,
        "tests": [
            "primary_full_dimension",
            "nine_zero_width_mutations",
            "outward_sqrt_low_precision",
            "even_power_crossing_zero",
            "division_containment",
            "five_negative_controls_execute",
        ],
    }


def source_identity():
    try:
        raw = Path(__file__).read_bytes()
        return {
            "path": str(Path(__file__).resolve()),
            "bytes": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }
    except Exception as exc:
        return {"path": None, "bytes": None, "sha256": None, "error": str(exc)}


def canonical_hash(payload) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def main():
    regression = regression_tests()
    precision_runs = [run_at_precision(p) for p in (40, 80, 120)]
    main_run = precision_runs[1]

    sign_stability = all(
        run["primary_pass"] and run["all_controls_pass"]
        for run in precision_runs
    )
    margin_names = set(precision_runs[0]["margins"])
    if not all(set(run["margins"]) == margin_names for run in precision_runs):
        sign_stability = False

    output = {
        "artifact": "GP-DATA-168-v1.1",
        "object": "EC-019 T2 corrected GP-side interval preflight",
        "status": "SAME-LINE PREFLIGHT; T2 OPEN",
        "independence_credit": "NONE — OpenAI/GP-authored",
        "repairs": [
            "NC1 executes all nine zero-radius mutations",
            "sqrt endpoints widened with next_minus/next_plus",
            "endpoint-derived inequalities retained in interval arithmetic",
            "regression tests added",
            "40/80/120-digit stability runs added",
        ],
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "decimal_default_rounding": getcontext().rounding,
        },
        "source": source_identity(),
        "regression": regression,
        "precision_runs": precision_runs,
        "sign_stability_40_80_120": sign_stability,
        "primary_80_digit": main_run,
        "scientific_adjudication": (
            "AMENDMENT EXECUTED FOR SAME-LINE PREFLIGHT. "
            "This does not complete source-independent T2."
        ),
    }
    output["canonical_result_sha256"] = canonical_hash(output)
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
