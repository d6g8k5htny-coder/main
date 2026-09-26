#!/usr/bin/env python3
"""Negative control for the SARD-G chart predicate.

The predecessor predicate (exactly one index-one point in the inner box, spectral
gap, gradient bound only on the outer-minus-inner annulus) accepts a torus field
with a hidden degenerate critical point and rejects an arbitrarily small C^2
perturbation that splits that point into a new index-one saddle.

The repaired predicate requires the designated saddle to be the unique critical
point of the closed outer box and a quantitative gradient lower bound on the
compact complement of its continuation neighborhood. That predicate excludes the
same field on the same outer box.

This is a predicate control. It does not sample the Bargmann-Fock measure and it
does not move any corollary.
"""

from __future__ import annotations

import math
import sys

TWOPI = 2.0 * math.pi
# Rational margins used by both predicates on this chart.
DELTA = 1.0
ETA = 1.0
EPS = 1.0e-3
# Cover rectangles (xa, xb) x (ya, yb) with 0 < length < 1, possibly wrapping mod 1.
OLD_INNER = (0.30, 1.08, 0.82, 1.18)
OLD_OUTER = (0.22, 1.16, 0.74, 1.26)
# Continuation neighborhood of the designated saddle (1/2, 0), inside the old inner box.
CONTINUATION = (0.45, 0.55, 0.92, 1.08)
# Tight chart that isolates the same saddle. Both f_0 and f_eps satisfy the repair here.
TIGHT_OUTER = (0.42, 0.58, 0.88, 1.12)
TIGHT_N = (0.47, 0.53, 0.94, 1.06)


def in_interval(t: float, a: float, b: float) -> bool:
    length = b - a
    if not 0.0 < length < 1.0:
        raise ValueError(f"interval length must lie in (0, 1), got {length}")
    return 0.0 < ((t % 1.0) - (a % 1.0)) % 1.0 < length


def in_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    return in_interval(x, rect[0], rect[1]) and in_interval(y, rect[2], rect[3])


def derivatives(x: float, y: float, eps: float) -> tuple[float, float, float, float]:
    """Gradient and diagonal Hessian of the explicit torus field.

    f_eps = (-5+eps) cos(2 pi x) - cos(4 pi x) + cos(6 pi x) + cos(2 pi y).
    """
    theta = TWOPI * (x % 1.0)
    phi = TWOPI * (y % 1.0)
    a = -5.0 + eps
    b = -1.0
    c = 1.0
    fx = (
        -TWOPI * a * math.sin(theta)
        - 2.0 * TWOPI * b * math.sin(2.0 * theta)
        - 3.0 * TWOPI * c * math.sin(3.0 * theta)
    )
    fy = -TWOPI * math.sin(phi)
    fxx = (
        -(TWOPI**2) * a * math.cos(theta)
        - (2.0 * TWOPI) ** 2 * b * math.cos(2.0 * theta)
        - (3.0 * TWOPI) ** 2 * c * math.cos(3.0 * theta)
    )
    fyy = -(TWOPI**2) * math.cos(phi)
    return fx, fy, fxx, fyy


def x_zeros(eps: float) -> list[float]:
    grid_n = 200_000
    step = 1.0 / grid_n
    zeros: list[float] = []
    previous_x = 0.0
    previous = derivatives(0.0, 0.0, eps)[0]
    for i in range(1, grid_n + 1):
        x = i * step
        value = derivatives(x % 1.0, 0.0, eps)[0]
        if previous == 0.0 or previous * value < 0.0:
            left, right = previous_x, x
            left_value = previous
            for _ in range(60):
                mid = 0.5 * (left + right)
                mid_value = derivatives(mid % 1.0, 0.0, eps)[0]
                if left_value * mid_value <= 0.0:
                    right = mid
                else:
                    left = mid
                    left_value = mid_value
            zero = 0.5 * (left + right) % 1.0
            if not zeros or min(abs(zero - z) for z in zeros) > 1.0e-6:
                zeros.append(zero)
        previous_x, previous = x, value
    if abs(derivatives(0.0, 0.0, eps)[0]) < 1.0e-10 and all(min(z, 1.0 - z) > 1.0e-5 for z in zeros):
        zeros.insert(0, 0.0)
    return zeros


def critical_points(eps: float) -> list[dict[str, float | str]]:
    points = []
    for x in x_zeros(eps):
        fxx = derivatives(x, 0.0, eps)[2]
        for y, fyy in ((0.0, derivatives(x, 0.0, eps)[3]), (0.5, derivatives(x, 0.5, eps)[3])):
            gap = min(abs(fxx), abs(fyy))
            if gap < 1.0e-6:
                kind = "degenerate"
                unstable = -1
            else:
                unstable = int(fxx > 0.0) + int(fyy > 0.0)
                kind = f"unstable-dim-{unstable}"
            points.append(
                {
                    "x": x,
                    "y": y,
                    "kind": kind,
                    "unstable": unstable,
                    "gap": gap,
                    "fxx": fxx,
                    "fyy": fyy,
                }
            )
    return points


def grad_min_on_complement(
    eps: float,
    outer: tuple[float, float, float, float],
    removed: tuple[float, float, float, float],
    n: int = 700,
) -> float:
    minimum = math.inf
    xs = [outer[0] + (outer[1] - outer[0]) * i / (n - 1) for i in range(n)]
    ys = [outer[2] + (outer[3] - outer[2]) * i / (n - 1) for i in range(n)]
    for x in xs:
        for y in ys:
            if in_rect(x, y, outer) and not in_rect(x, y, removed):
                fx, fy, _, _ = derivatives(x, y, eps)
                minimum = min(minimum, math.hypot(fx, fy))
    return minimum


def c2_size(eps: float) -> float:
    """Max sup-norm of derivatives through order 2 of eps cos(2 pi x)."""
    return abs(eps) * max(1.0, TWOPI, TWOPI**2)


def classify(points: list[dict[str, float | str]], rect: tuple[float, float, float, float]) -> list[dict[str, float | str]]:
    return [p for p in points if in_rect(float(p["x"]), float(p["y"]), rect)]


def old_accepts(points: list[dict[str, float | str]], eps: float) -> tuple[bool, str]:
    inner = classify(points, OLD_INNER)
    index_one = [p for p in inner if p["kind"] == "unstable-dim-1"]
    if len(index_one) != 1:
        return False, f"inner index-one count is {len(index_one)}"
    gap = float(index_one[0]["gap"])
    if gap <= DELTA:
        return False, f"spectral gap {gap} does not exceed {DELTA}"
    annulus = grad_min_on_complement(eps, OLD_OUTER, OLD_INNER)
    if annulus < ETA:
        return False, f"annulus gradient minimum {annulus} is below {ETA}"
    return True, f"one index-one point, gap {gap:.6f}, annulus min |grad| {annulus:.6f}"


def repaired_accepts(
    points: list[dict[str, float | str]],
    eps: float,
    outer: tuple[float, float, float, float],
    neighborhood: tuple[float, float, float, float],
) -> tuple[bool, str]:
    owned = classify(points, outer)
    if len(owned) != 1:
        return False, f"outer critical-point count is {len(owned)}"
    point = owned[0]
    if not in_rect(float(point["x"]), float(point["y"]), neighborhood):
        return False, "unique critical point lies outside the continuation neighborhood"
    if point["kind"] != "unstable-dim-1":
        return False, f"unique critical point has kind {point['kind']}"
    gap = float(point["gap"])
    if gap <= DELTA:
        return False, f"spectral gap {gap} does not exceed {DELTA}"
    outside = [p for p in owned if not in_rect(float(p["x"]), float(p["y"]), neighborhood)]
    if outside:
        return False, "a critical point lies outside the continuation neighborhood"
    margin = grad_min_on_complement(eps, outer, neighborhood)
    if margin < ETA:
        return False, f"continuation-complement gradient minimum {margin} is below {ETA}"
    return True, f"unique index-one point in the continuation neighborhood, gap {gap:.6f}, complement min |grad| {margin:.6f}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"PASS: {message}")


def main() -> int:
    base = critical_points(0.0)
    moved = critical_points(EPS)
    print(f"C^2 size of the perturbation eps={EPS} is {c2_size(EPS):.6f}")
    print("critical points at eps=0:")
    for point in base:
        print(
            f"  ({float(point['x']):.6f}, {float(point['y']):.1f}) "
            f"{point['kind']} gap={float(point['gap']):.6f}"
        )
    print(f"critical points at eps={EPS}:")
    for point in moved:
        print(
            f"  ({float(point['x']):.6f}, {float(point['y']):.1f}) "
            f"{point['kind']} gap={float(point['gap']):.6f}"
        )

    old_base, old_base_detail = old_accepts(base, 0.0)
    old_moved, old_moved_detail = old_accepts(moved, EPS)
    require(old_base, f"old predicate accepts f_0 ({old_base_detail})")
    require(not old_moved, f"old predicate rejects f_eps ({old_moved_detail})")

    repaired_base, repaired_base_detail = repaired_accepts(base, 0.0, OLD_OUTER, CONTINUATION)
    require(
        not repaired_base,
        f"repaired predicate excludes f_0 on the old outer box ({repaired_base_detail})",
    )
    extra = [
        p
        for p in classify(base, OLD_OUTER)
        if not in_rect(float(p["x"]), float(p["y"]), CONTINUATION)
    ]
    require(
        any(p["kind"] == "degenerate" for p in extra),
        "exclusion witness: a degenerate critical point lies in the old outer box and outside the continuation neighborhood",
    )

    tight_base, tight_base_detail = repaired_accepts(base, 0.0, TIGHT_OUTER, TIGHT_N)
    tight_moved, tight_moved_detail = repaired_accepts(moved, EPS, TIGHT_OUTER, TIGHT_N)
    require(tight_base, f"repaired predicate accepts the isolated saddle of f_0 ({tight_base_detail})")
    require(
        tight_moved,
        f"repaired predicate still accepts that saddle after the perturbation ({tight_moved_detail})",
    )
    print("negative control passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
