#!/usr/bin/env python3
"""
C104 coupled multi-resolution continuum-persistence validation.

Implements the frozen C104 protocol:
- exact-periodized BF Fourier weights, truncated below the coarse Nyquist;
- the same Fourier realization evaluated on nested coarse/fine grids;
- periodic Freudenthal triangulation;
- superlevel H0 persistence;
- coarse/fine diagram matching with diagonal options;
- stability-certified bar filtering;
- left-truncated power-law likelihood for the density exponent.

The experiment is diagnostic. Its result is adjudicated separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import linear_sum_assignment
from numba import njit

BASE = Path(__file__).resolve().parent
FREEZE = BASE / "C104_FREEZE.json"
OUTPUT = BASE / "C104_COUPLED_PERSISTENCE.json"

L = 24.0
N_BOOT = 1000
ALPHA_BOUNDS = (-0.95, 2.0)


def spectral_coefficients(
    n_coarse: int,
    rng: np.random.Generator,
) -> dict[tuple[int, int], complex]:
    """Real Gaussian Fourier coefficients with exact BF spectral weights."""
    cutoff = n_coarse // 2 - 1
    modes = [
        (kx, ky)
        for kx in range(-cutoff, cutoff + 1)
        for ky in range(-cutoff, cutoff + 1)
    ]
    weights = {
        mode: math.exp(
            -2.0
            * math.pi**2
            * (mode[0] ** 2 + mode[1] ** 2)
            / (L * L)
        )
        for mode in modes
    }
    total = sum(weights.values())
    weights = {mode: value / total for mode, value in weights.items()}

    coefficients: dict[tuple[int, int], complex] = {}
    seen: set[tuple[int, int]] = set()
    for mode in modes:
        if mode in seen:
            continue
        opposite = (-mode[0], -mode[1])
        if mode == (0, 0):
            coefficients[mode] = rng.normal() * math.sqrt(weights[mode])
            seen.add(mode)
            continue
        # Use one representative of each ± pair.
        if opposite not in weights:
            raise RuntimeError("mode support is not symmetric")
        real = rng.normal()
        imag = rng.normal()
        coefficient = (
            (real + 1j * imag)
            / math.sqrt(2.0)
            * math.sqrt(weights[mode])
        )
        coefficients[mode] = coefficient
        coefficients[opposite] = np.conjugate(coefficient)
        seen.add(mode)
        seen.add(opposite)
    return coefficients


def evaluate_field(
    n: int,
    coefficients: dict[tuple[int, int], complex],
) -> np.ndarray:
    spectrum = np.zeros((n, n), dtype=np.complex128)
    for (kx, ky), coefficient in coefficients.items():
        spectrum[kx % n, ky % n] = coefficient * (n * n)
    field = np.fft.ifft2(spectrum).real
    return field


@njit(cache=True)
def _find_root(parent: np.ndarray, a: int) -> int:
    root = a
    while parent[root] != root:
        root = parent[root]
    while parent[a] != a:
        nxt = parent[a]
        parent[a] = root
        a = nxt
    return root


@njit(cache=True)
def _freudenthal_pairs(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = field.shape[0]
    flat = field.ravel()
    order = np.argsort(flat)[::-1]
    total = n * n

    parent = np.full(total, -1, dtype=np.int32)
    birth = np.empty(total, dtype=np.float64)
    active = np.zeros(total, dtype=np.uint8)
    output_birth = np.empty(total, dtype=np.float64)
    output_death = np.empty(total, dtype=np.float64)
    output_count = 0
    roots = np.empty(6, dtype=np.int32)

    for position in range(order.size):
        index = int(order[position])
        value = float(flat[index])
        active[index] = 1
        parent[index] = index
        birth[index] = value

        i = index // n
        j = index - i * n
        im = (i - 1) % n
        ip = (i + 1) % n
        jm = (j - 1) % n
        jp = (j + 1) % n
        neighbors = (
            im * n + j,
            ip * n + j,
            i * n + jm,
            i * n + jp,
            im * n + jm,
            ip * n + jp,
        )

        root_count = 0
        for neighbor in neighbors:
            if active[neighbor] == 0:
                continue
            root = _find_root(parent, int(neighbor))
            duplicate = False
            for q in range(root_count):
                if roots[q] == root:
                    duplicate = True
                    break
            if not duplicate:
                roots[root_count] = root
                root_count += 1

        if root_count == 0:
            continue

        oldest = roots[0]
        for q in range(1, root_count):
            candidate = roots[q]
            if birth[candidate] > birth[oldest]:
                oldest = candidate
        parent[index] = oldest

        for q in range(root_count):
            root = _find_root(parent, int(roots[q]))
            oldest = _find_root(parent, int(oldest))
            if root == oldest:
                continue
            # oldest has maximal birth among the pre-merge roots.
            if birth[root] > birth[oldest]:
                temp = root
                root = oldest
                oldest = temp
            if birth[root] > value:
                output_birth[output_count] = birth[root]
                output_death[output_count] = value
                output_count += 1
            parent[root] = oldest

    return output_birth[:output_count], output_death[:output_count]


def freudenthal_lifetimes(field: np.ndarray) -> np.ndarray:
    """Periodic superlevel H0 pairs on the fixed-diagonal triangulation."""
    births, deaths = _freudenthal_pairs(np.ascontiguousarray(field))
    if births.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((births, deaths))


def interpolate_coarse_to_fine(coarse: np.ndarray) -> np.ndarray:
    """Factor-two PL interpolation for the fixed Freudenthal diagonal."""
    n = coarse.shape[0]
    fine = np.empty((2 * n, 2 * n), dtype=float)
    forward_i = np.roll(coarse, -1, axis=0)
    forward_j = np.roll(coarse, -1, axis=1)
    forward_diag = np.roll(forward_i, -1, axis=1)

    fine[0::2, 0::2] = coarse
    fine[1::2, 0::2] = 0.5 * (coarse + forward_i)
    fine[0::2, 1::2] = 0.5 * (coarse + forward_j)
    fine[1::2, 1::2] = 0.5 * (coarse + forward_diag)
    return fine


def match_diagrams(
    coarse: np.ndarray,
    fine: np.ndarray,
    epsilon: float,
) -> dict:
    """Minimum-total-cost matching with separate diagonal copies."""
    n_fine = fine.shape[0]
    n_coarse = coarse.shape[0]
    if n_fine == 0:
        return {
            "stable_fine": np.empty((0, 2)),
            "matched_pairs": [],
            "fine_count": 0,
            "coarse_count": n_coarse,
        }

    fine_persistence = fine[:, 0] - fine[:, 1]
    coarse_persistence = (
        coarse[:, 0] - coarse[:, 1]
        if n_coarse
        else np.empty(0)
    )

    size = n_fine + n_coarse
    large = 1e6
    cost = np.full((size, size), large, dtype=float)

    if n_coarse:
        distances = np.maximum(
            np.abs(fine[:, None, 0] - coarse[None, :, 0]),
            np.abs(fine[:, None, 1] - coarse[None, :, 1]),
        )
        cost[:n_fine, :n_coarse] = distances

    # Fine bars to their own diagonal copies.
    fine_rows = np.arange(n_fine)
    cost[fine_rows, n_coarse + fine_rows] = fine_persistence / 2.0

    # Coarse bars to their own diagonal copies.
    if n_coarse:
        coarse_rows = n_fine + np.arange(n_coarse)
        coarse_cols = np.arange(n_coarse)
        cost[coarse_rows, coarse_cols] = coarse_persistence / 2.0

    # Unused diagonal copies match at zero cost.
    if n_coarse:
        cost[n_fine:, n_coarse:] = 0.0

    rows, columns = linear_sum_assignment(cost)

    stable_indices: list[int] = []
    matched_pairs: list[dict] = []
    for row, column in zip(rows, columns):
        if row >= n_fine or column >= n_coarse:
            continue
        distance = float(cost[row, column])
        persistence = float(fine_persistence[row])
        stable = distance <= 2.0 * epsilon and persistence > 4.0 * epsilon
        if stable:
            stable_indices.append(int(row))
        matched_pairs.append(
            {
                "fine_index": int(row),
                "coarse_index": int(column),
                "distance": distance,
                "fine_persistence": persistence,
                "stable": stable,
            }
        )

    stable_fine = (
        fine[np.asarray(stable_indices, dtype=int)]
        if stable_indices
        else np.empty((0, 2), dtype=float)
    )
    return {
        "stable_fine": stable_fine,
        "matched_pairs": matched_pairs,
        "fine_count": n_fine,
        "coarse_count": n_coarse,
    }


def prepare_likelihood_stats(
    observations: list[tuple[np.ndarray, float]],
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(len(observations), dtype=np.int32)
    sum_logs = np.zeros(len(observations), dtype=np.float64)
    truncations = np.empty(len(observations), dtype=np.float64)
    for index, (lifetimes, truncation) in enumerate(observations):
        truncations[index] = truncation
        if truncation >= upper:
            continue
        selected = lifetimes[
            (lifetimes >= truncation) & (lifetimes <= upper)
        ]
        counts[index] = selected.size
        if selected.size:
            sum_logs[index] = float(np.log(selected).sum())
    return counts, sum_logs, truncations


def fit_alpha_stats(
    counts: np.ndarray,
    sum_logs: np.ndarray,
    truncations: np.ndarray,
    upper: float,
    multiplicities: np.ndarray | None = None,
) -> tuple[float, int]:
    if multiplicities is None:
        multiplicities = np.ones(counts.size, dtype=np.int32)
    weighted_counts = multiplicities * counts
    used = int(weighted_counts.sum())
    if used < 5:
        return float("nan"), used

    valid = weighted_counts > 0
    n = weighted_counts[valid].astype(np.float64)
    sums = sum_logs[valid]
    tau = truncations[valid]
    log_upper = math.log(upper)
    log_tau_ratio = np.log(tau) - log_upper
    total_sum_logs = float(np.dot(multiplicities[valid], sums))

    def negative_log_likelihood(alpha: float) -> float:
        beta = alpha + 1.0
        if beta <= 0:
            return math.inf
        ratio = np.exp(beta * log_tau_ratio)
        if np.any(ratio >= 1.0):
            return math.inf
        log_normalizer = beta * log_upper + np.log1p(-ratio)
        likelihood = (
            alpha * total_sum_logs
            + used * math.log(beta)
            - float(np.dot(n, log_normalizer))
        )
        return -likelihood

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=ALPHA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(result.x), used


def bootstrap_alpha(
    observations: list[tuple[np.ndarray, float]],
    upper: float,
    rng: np.random.Generator,
) -> dict:
    counts, sum_logs, truncations = prepare_likelihood_stats(
        observations, upper
    )
    estimate, used = fit_alpha_stats(
        counts, sum_logs, truncations, upper
    )
    n_fields = len(observations)
    boot = np.empty(N_BOOT, dtype=float)
    failures = 0
    for index in range(N_BOOT):
        chosen = rng.integers(0, n_fields, size=n_fields)
        multiplicities = np.bincount(
            chosen, minlength=n_fields
        ).astype(np.int32)
        value, _ = fit_alpha_stats(
            counts,
            sum_logs,
            truncations,
            upper,
            multiplicities,
        )
        if not np.isfinite(value):
            failures += 1
            boot[index] = np.nan
        else:
            boot[index] = value
    finite = boot[np.isfinite(boot)]
    return {
        "alpha_hat": estimate,
        "stable_bars_used": used,
        "bootstrap_valid": int(finite.size),
        "bootstrap_failures": failures,
        "bootstrap_failure_fraction": failures / N_BOOT,
        "ci_95": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size
            else [None, None]
        ),
        "bootstrap_median": (
            float(np.median(finite)) if finite.size else None
        ),
        "contains_minus_one_third": (
            bool(
                np.quantile(finite, 0.025)
                <= -1.0 / 3.0
                <= np.quantile(finite, 0.975)
            )
            if finite.size
            else False
        ),
    }


def run_pair(
    coarse_n: int,
    fine_n: int,
    n_fields: int,
    seed: int,
    upper_cutoffs: list[float],
) -> dict:
    if fine_n != 2 * coarse_n:
        raise ValueError("this implementation requires factor-two nesting")

    rng = np.random.default_rng(seed)
    field_records: list[dict] = []
    observations: list[tuple[np.ndarray, float]] = []

    for field_index in range(n_fields):
        coefficients = spectral_coefficients(coarse_n, rng)
        coarse_field = evaluate_field(coarse_n, coefficients)
        fine_field = evaluate_field(fine_n, coefficients)
        interpolated = interpolate_coarse_to_fine(coarse_field)
        epsilon = float(np.max(np.abs(fine_field - interpolated)))

        coarse_diagram = freudenthal_lifetimes(coarse_field)
        fine_diagram = freudenthal_lifetimes(fine_field)
        matching = match_diagrams(
            coarse_diagram, fine_diagram, epsilon
        )
        stable_diagram = matching.pop("stable_fine")
        stable_lifetimes = (
            stable_diagram[:, 0] - stable_diagram[:, 1]
            if stable_diagram.size
            else np.empty(0)
        )
        truncation = 4.0 * epsilon
        observations.append((stable_lifetimes, truncation))

        field_records.append(
            {
                "field_index": field_index,
                "epsilon": epsilon,
                "truncation": truncation,
                "coarse_bars": int(coarse_diagram.shape[0]),
                "fine_bars": int(fine_diagram.shape[0]),
                "matched_pairs": len(matching["matched_pairs"]),
                "stable_bars": int(stable_lifetimes.size),
                "stable_lifetimes": stable_lifetimes.tolist(),
                "matching": matching["matched_pairs"],
            }
        )

    fit_rng = np.random.default_rng(seed + 800000)
    fits = {
        str(upper): bootstrap_alpha(
            observations, float(upper), fit_rng
        )
        for upper in upper_cutoffs
    }

    epsilons = np.asarray(
        [record["epsilon"] for record in field_records]
    )
    total_fine = sum(record["fine_bars"] for record in field_records)
    total_stable = sum(record["stable_bars"] for record in field_records)

    return {
        "coarse_n": coarse_n,
        "fine_n": fine_n,
        "fields": n_fields,
        "epsilon_summary": {
            "mean": float(epsilons.mean()),
            "std": float(epsilons.std(ddof=1)),
            "quantiles": {
                str(q): float(np.quantile(epsilons, q))
                for q in [0.05, 0.25, 0.5, 0.75, 0.95]
            },
        },
        "total_fine_bars": int(total_fine),
        "total_stable_bars": int(total_stable),
        "stable_fraction": (
            float(total_stable / total_fine) if total_fine else 0.0
        ),
        "fits": fits,
        "field_records": field_records,
    }


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    pairs_report: dict[str, dict] = {}
    for pair_index, (coarse, fine) in enumerate(
        freeze["model"]["coarse_fine_pairs"]
    ):
        key = f"{coarse}_{fine}"
        pairs_report[key] = run_pair(
            int(coarse),
            int(fine),
            int(freeze["model"]["fields_per_pair"]),
            int(freeze["model"]["fresh_seed_base"]) + pair_index,
            [float(value) for value in freeze["exponent_estimator"]["upper_cutoffs"]],
        )

    primary_pair = "256_512"
    primary_upper = str(
        freeze["exponent_estimator"]["primary_upper_cutoff"]
    )
    primary = pairs_report[primary_pair]["fits"][primary_upper]
    enough = (
        primary["stable_bars_used"]
        >= freeze["primary_gate"]["minimum_stable_bars"]
    )
    bootstrap_ok = (
        primary["bootstrap_failure_fraction"] <= 0.05
    )
    if not enough or not bootstrap_ok:
        verdict = "INCONCLUSIVE"
    elif primary["contains_minus_one_third"]:
        verdict = "SURVIVES"
    else:
        verdict = "FAILS_PRIMARY_GATE"

    valid_intervals: list[tuple[float, float, float]] = []
    for pair in pairs_report.values():
        for fit in pair["fits"].values():
            low, high = fit["ci_95"]
            if low is None:
                continue
            valid_intervals.append((low, high, fit["alpha_hat"]))

    directions: list[int] = []
    all_exclude = bool(valid_intervals)
    for low, high, _ in valid_intervals:
        if low <= -1 / 3 <= high:
            all_exclude = False
            break
        directions.append(-1 if high < -1 / 3 else 1)

    same_direction = (
        bool(directions)
        and all(direction == directions[0] for direction in directions)
    )
    coarse_to_fine_primary = [
        pairs_report[key]["fits"][primary_upper]["alpha_hat"]
        for key in ("128_256", "192_384", "256_512")
    ]
    target = -1 / 3
    finest_moves_toward = (
        abs(coarse_to_fine_primary[-1] - target)
        < abs(coarse_to_fine_primary[-2] - target)
    )
    kill_signal = (
        all_exclude and same_direction and not finest_moves_toward
    )

    report = {
        "cycle": "C104",
        "result_id": "COUPLED_CONTINUUM_PERSISTENCE_VALIDATION",
        "grade": "PRE-REGISTERED-MEASURED-DIAGNOSTIC",
        "freeze_sha256": hashlib.sha256(
            FREEZE.read_bytes()
        ).hexdigest(),
        "pairs": pairs_report,
        "primary_gate": {
            "pair": primary_pair,
            "upper_cutoff": float(primary_upper),
            "result": primary,
            "enough_stable_bars": enough,
            "bootstrap_ok": bootstrap_ok,
            "verdict": verdict,
        },
        "kill_signal": {
            "triggered": bool(kill_signal),
            "all_valid_intervals_exclude_target": all_exclude,
            "same_direction": same_direction,
            "primary_estimates_coarse_to_fine": coarse_to_fine_primary,
            "finest_moves_toward_target": finest_moves_toward,
        },
        "target_alpha": -1 / 3,
        "limitations": [
            "band-limited Fourier truncation below coarse Nyquist",
            "PL Freudenthal approximation, not direct continuum Morse pairing",
            "diagram matching minimizes total cost rather than bottleneck cost",
            "empirical validation is not a theorem dependency",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    compact = {
        "primary_gate": report["primary_gate"],
        "kill_signal": report["kill_signal"],
        "pairs": {
            key: {
                "epsilon_summary": value["epsilon_summary"],
                "total_fine_bars": value["total_fine_bars"],
                "total_stable_bars": value["total_stable_bars"],
                "stable_fraction": value["stable_fraction"],
                "fits": value["fits"],
            }
            for key, value in pairs_report.items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
