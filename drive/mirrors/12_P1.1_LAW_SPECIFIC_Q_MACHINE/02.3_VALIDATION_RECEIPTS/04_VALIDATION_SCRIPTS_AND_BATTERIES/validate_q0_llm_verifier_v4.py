#!/usr/bin/env python3
"""Independent validation for q0_llm_verifier_v4.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np

import q0_llm_verifier_v4 as v4
from q0_llm_verifier_v3 import (
    SupportEdge,
    widest_path_grounding,
    anchored_rkhs_mismatch,
    anisotropic_sard_tube_bound,
    effective_transverse_rank,
    CoverageCertificate,
    SelectionAwareRiskLedger,
    coverage_limited_rank_requirement,
)


def main() -> None:
    graph, roots = v4.program_grade_q0_contract()
    report = graph.validate(roots)

    bad = v4.bad_contract_examples()
    expected_gates = {
        "missing_mark": "MARK",
        "missing_composition_witness": "COMPOSITION",
        "endpoint_failure": "ENDPOINT",
        "missing_rung_tag": "RUNG",
        "uncertainty_polarity": "POLARITY",
    }
    bad_results = {}
    for name, expected_gate in expected_gates.items():
        current = bad[name]
        gates = sorted({issue.gate for issue in current.issues})
        bad_results[name] = {
            "valid": current.valid,
            "gates": gates,
            "expected_gate_detected": expected_gate in gates,
        }

    # Exact H0 grounding regression test.
    nodes = {"c0", "c1", "c2", "e0", "e1"}
    edges = [
        SupportEdge("c0", "c1", 0.91),
        SupportEdge("c1", "e0", 0.66),
        SupportEdge("c0", "e1", 0.51),
        SupportEdge("c2", "c1", 0.72),
        SupportEdge("c2", "e1", 0.60),
    ]
    grounding = widest_path_grounding(nodes, edges, {"e0", "e1"})
    assert abs(grounding.bottleneck_to_evidence["c0"] - 0.66) < 1e-12
    assert abs(grounding.bottleneck_to_evidence["c2"] - 0.66) < 1e-12

    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=float,
    )
    signed_weights = np.array([1.0, -0.4, -0.6], dtype=float)
    rkhs = anchored_rkhs_mismatch(
        embeddings,
        signed_weights,
        anchor_weight=0.1,
        learned_features=np.zeros((3, 8)),
    )

    q_sard = anisotropic_sard_tube_bound(
        0.01,
        [1.0, 1.0, 1e-3],
        density_bound=0.1,
    )
    effective_rank = effective_transverse_rank(
        [1.0, 1.0, 1e-3],
        0.01,
    )

    coverage = CoverageCertificate(uncovered_wrong_mass=0.002)
    target_below_floor_rejected = False
    try:
        coverage_limited_rank_requirement(
            0.001,
            coverage.uncovered_wrong_mass,
            0.2,
        )
    except ValueError:
        target_below_floor_rejected = True

    required_rank = coverage_limited_rank_requirement(
        0.01,
        coverage.uncovered_wrong_mass,
        0.2,
    )

    ledger = SelectionAwareRiskLedger(
        coverage_failure=0.002,
        baseline_hazard_rate=0.001,
        expected_selected_measure=10.0,
        selection_amplification=3.0,
        q_sard_tube_risk=0.0002,
        decoder_boundary_risk=0.0001,
    )

    output = {
        "verifier": "q0_llm_verifier_v4",
        "core": report.as_dict(),
        "bad_contract_regressions": bad_results,
        "numerical_regressions": {
            "grounding_c0": grounding.bottleneck_to_evidence["c0"],
            "grounding_c2": grounding.bottleneck_to_evidence["c2"],
            "retained_grounding_edges": len(grounding.retained_edges),
            "anchored_rkhs_score_squared": rkhs,
            "q_sard_rank_deficient_bound": q_sard,
            "effective_transverse_rank": effective_rank,
            "coverage_floor": coverage.irreducible_false_acceptance_floor(),
            "target_below_coverage_floor_rejected": target_below_floor_rejected,
            "rank_for_target_0p01_after_coverage": required_rank,
            "selection_aware_total_risk": ledger.total_upper_bound(),
            "selection_aware_exponential_limit_floor":
                ledger.exponential_limit_floor(),
        },
        "all_expected_checks_pass": (
            report.valid
            and all(item["expected_gate_detected"] for item in bad_results.values())
            and rkhs > 0.0
            and effective_rank == 2
            and target_below_floor_rejected
        ),
    }

    path = BASE / "q0_llm_verifier_v4_validation.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
