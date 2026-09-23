#!/usr/bin/env python3
"""Independent stress tests for q0_llm_verifier_v2.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q0_llm_verifier_v2 as q0


def brute_force_widest(
    start: str,
    evidence: Set[str],
    adjacency: Dict[str, List[tuple[str, float]]],
) -> float:
    best = 1.0 if start in evidence else 0.0

    def dfs(node: str, capacity: float, visited: Set[str]) -> None:
        nonlocal best
        if node in evidence:
            best = max(best, capacity)
        for nxt, strength in adjacency.get(node, []):
            if nxt in visited:
                continue
            dfs(nxt, min(capacity, strength), visited | {nxt})

    dfs(start, 1.0, {start})
    return best


def graph_stress_test(seed: int = 44) -> dict:
    rng = np.random.default_rng(seed)
    nodes = [f"v{i}" for i in range(10)]
    evidence = {"v8", "v9"}
    edges = []
    seen = set()

    # Guarantee at least one route from every non-evidence node.
    for i in range(8):
        dst = f"v{min(i + 1, 9)}"
        strength = float(rng.uniform(0.2, 0.99))
        edges.append(q0.SupportEdge(f"v{i}", dst, strength))
        seen.add((f"v{i}", dst))

    while len(edges) < 32:
        src, dst = rng.choice(nodes, size=2, replace=False)
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        edges.append(q0.SupportEdge(src, dst, float(rng.uniform(0.05, 0.99))))

    cert = q0.widest_path_grounding(nodes, edges, evidence)
    adjacency: Dict[str, List[tuple[str, float]]] = {node: [] for node in nodes}
    for edge in edges:
        adjacency[edge.src].append((edge.dst, edge.strength))

    mismatches = {}
    for node in nodes:
        brute = brute_force_widest(node, evidence, adjacency)
        fast = cert.bottleneck_to_evidence[node]
        if abs(brute - fast) > 1e-12:
            mismatches[node] = {"brute": brute, "fast": fast}

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "evidence_nodes": len(evidence),
        "retained_edges": len(cert.retained_edges),
        "compression_limit": len(nodes) - len(evidence),
        "mismatches": mismatches,
    }


def qsard_slope(
    dimension: int,
    *,
    singular_values: np.ndarray | None = None,
    samples: int = 1_000_000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    singular_values = (
        np.ones(dimension)
        if singular_values is None
        else np.asarray(singular_values, dtype=float)
    )
    x = rng.normal(size=(samples, dimension))
    radii = np.linalg.norm(x * singular_values, axis=1)

    if dimension == 1:
        eps = np.geomspace(0.02, 0.25, 8)
    elif dimension == 2:
        eps = np.geomspace(0.02, 0.25, 8)
    else:
        eps = np.geomspace(0.08, 0.4, 8)

    if singular_values.min() < 1e-2:
        eps = np.geomspace(0.01, 0.15, 8)

    probabilities = np.array([(radii <= value).mean() for value in eps])
    mask = probabilities > 0
    return float(
        np.polyfit(np.log(eps[mask]), np.log(probabilities[mask]), 1)[0]
    )


def main() -> None:
    graph = graph_stress_test()
    slopes = {
        "k1": qsard_slope(1, seed=101),
        "k2": qsard_slope(2, seed=102),
        "k3": qsard_slope(3, seed=103),
        "nominal_k3_one_singular_value_1e-3": qsard_slope(
            3, singular_values=np.array([1.0, 1.0, 1e-3]), seed=104
        ),
    }

    z = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    weights = np.array([1.0, -0.4, -0.6])
    anchored = q0.anchored_rkhs_mismatch(
        z,
        weights,
        anchor_weight=0.1,
        learned_features=np.zeros((3, 4)),
    )

    report = {
        "graph_stress_test": graph,
        "qsard_loglog_slopes": slopes,
        "anchored_kernel_score_squared_with_zero_learned_component": anchored,
        "adjudication": {
            "h0_exact": not graph["mismatches"],
            "h0_compression_within_bound": (
                graph["retained_edges"] <= graph["compression_limit"]
            ),
            "rank_collapse_detected": (
                slopes["nominal_k3_one_singular_value_1e-3"] < 2.2
            ),
            "anchor_floor_positive": anchored > 0.0,
        },
    }

    output = Path("/mnt/data/q0_llm_verifier_v2_validation.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
