#!/usr/bin/env python3
"""Independent regression and discrimination battery for verifier v5."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable, Dict, List, Sequence, Tuple

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import q0_llm_verifier_v4 as v4
import q0_llm_verifier_v5 as v5
from q0_llm_verifier_v3 import Grade


def domain(rmax: float = 0.025, name: str = "r") -> v4.ParameterDomain:
    return v4.ParameterDomain(0.0, rmax, True, (("parameter", name),))


def regime(
    d: v4.ParameterDomain,
    *,
    model: str = "TEST_MODEL",
    measure: v4.Measure = v4.Measure.NONE,
    region: str = "test region",
    scale: str = "dimensionless",
    conditioning: str = "none",
    marks: Sequence[str] = (),
) -> v5.Regime:
    return v5.Regime(
        model_id=model,
        measure=measure,
        parameter_domain=d,
        spatial_region=region,
        scale=scale,
        conditioning_depth=conditioning,
        marks=frozenset(marks),
        deployment_mode="test",
    )


def claim(
    claim_id: str,
    statement: str,
    *,
    direction: v4.Direction = v4.Direction.STRUCTURAL,
    quantifier: v4.Quantifier = v4.Quantifier.CONDITIONAL,
    dependencies: Tuple[str, ...] = (),
    d: v4.ParameterDomain | None = None,
    measure: v4.Measure = v4.Measure.NONE,
    status: v4.Status = v4.Status.CONDITIONAL,
    grade: Grade = Grade.DERIVED,
    bound: float | None = None,
    measured: float | None = None,
    scale_tag: str | None = None,
    uncertainty_side: v4.UncertaintySide = v4.UncertaintySide.NONE,
    model: str = "TEST_MODEL",
    source: str = "C095-TEST",
    program_core: bool = False,
) -> v4.SemanticClaim:
    return v4.SemanticClaim(
        claim_id,
        statement,
        status,
        direction=direction,
        quantifier=quantifier,
        dependencies=dependencies,
        domain=d,
        measure=measure,
        grade=grade,
        uncertainty_side=uncertainty_side,
        measured_value=measured,
        scale_tag=scale_tag,
        bound_coefficient=bound,
        model_id=model,
        source_ids=(source,),
        program_core=program_core,
    )


def object_for(
    base_claim: v4.SemanticClaim,
    *,
    warrant: v5.Warrant = v5.Warrant.DERIVED,
    est: v5.Regime | None = None,
    dep: v5.Regime | None = None,
    **kwargs,
) -> v5.ProofObjectV5:
    d = base_claim.domain or domain()
    default = regime(
        d,
        model=base_claim.model_id or "TEST_MODEL",
        measure=base_claim.measure,
        marks=base_claim.provided_marks,
    )
    return v5.ProofObjectV5(
        base_claim,
        warrant=warrant,
        establishment_regime=est or default,
        deployment_regime=dep or default,
        **kwargs,
    )


def gate_sets(report: v5.V5ValidationReport) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {
        "FAIL": set(),
        "PROVISIONAL": set(),
        "PASS": set(),
        "NOT_APPLICABLE": set(),
    }
    for result in report.gate_results:
        out[result.verdict.value].add(result.gate)
    return out


def run_single(
    name: str,
    verifier: v5.GateVerifierV5,
    roots: Sequence[str],
    *,
    expected_valid: bool,
    expected_promotable: bool,
    required_fail_gates: Sequence[str] = (),
    required_provisional_gates: Sequence[str] = (),
) -> dict:
    report = verifier.validate(roots)
    gates = gate_sets(report)
    checks = {
        "valid": report.valid == expected_valid,
        "promotable": report.promotable == expected_promotable,
        "fail_gates": set(required_fail_gates).issubset(gates["FAIL"]),
        "provisional_gates": set(required_provisional_gates).issubset(
            gates["PROVISIONAL"]
        ),
    }
    return {
        "name": name,
        "expected": {
            "valid": expected_valid,
            "promotable": expected_promotable,
            "required_fail_gates": list(required_fail_gates),
            "required_provisional_gates": list(required_provisional_gates),
        },
        "observed": {
            "valid": report.valid,
            "promotable": report.promotable,
            "fail_gates": sorted(gates["FAIL"]),
            "provisional_gates": sorted(gates["PROVISIONAL"]),
            "root_hashes": report.root_hashes,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "report": report.as_dict(),
    }


def assembly_cases() -> List[dict]:
    d = domain()
    r = regime(
        d,
        model="BF_TORUS_EXACT_L24",
        measure=v4.Measure.PAIR_PALM,
        scale="r^3 coefficient",
        conditioning="typed pair-Palm",
        marks=("position", "height", "critical_type"),
    )
    cases = []

    bad = claim(
        "UBG_BAD_43",
        "Displayed UB-G upper coefficient 4.3.",
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        d=d,
        measure=v4.Measure.PAIR_PALM,
        bound=4.3,
        model="BF_TORUS_EXACT_L24",
    )
    bad_obj = object_for(
        bad,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        assembly_spec=v5.AssemblySpec(
            expression="(near+exterior)/typing",
            inputs=(("near", "0.66"), ("exterior", "2.82"), ("typing", "0.80")),
            required_terms=("near", "exterior", "typing"),
            displayed_value="4.3",
            claim_direction=v4.Direction.UPPER,
            rounding=v5.AssemblyRounding.UP,
            source_hashes=("C094-UBG",),
        ),
    )
    cases.append(
        run_single(
            "assembly_ubg_down_round",
            v5.GateVerifierV5([bad_obj]),
            ["UBG_BAD_43"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("ASSEMBLY",),
        )
    )

    good = replace(bad, claim_id="UBG_GOOD_435", bound_coefficient=4.35)
    good_obj = object_for(
        good,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        assembly_spec=replace(
            bad_obj.assembly_spec,
            displayed_value="4.35",
        ),
    )
    cases.append(
        run_single(
            "assembly_ubg_round_up",
            v5.GateVerifierV5([good_obj]),
            ["UBG_GOOD_435"],
            expected_valid=True,
            expected_promotable=True,
        )
    )

    lower = claim(
        "LOWER_BAD",
        "Displayed finite lower coefficient 0.8411.",
        direction=v4.Direction.LOWER,
        quantifier=v4.Quantifier.FOR_ALL,
        d=d,
        measure=v4.Measure.PAIR_PALM,
        bound=0.8411,
        model="BF_TORUS_EXACT_L24",
    )
    lower_obj = object_for(
        lower,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        assembly_spec=v5.AssemblySpec(
            expression="cstar*ao*bonf",
            inputs=(
                ("cstar", "0.8411"),
                ("ao", "0.999965"),
                ("bonf", "0.99999"),
            ),
            required_terms=("cstar", "ao", "bonf"),
            displayed_value="0.8411",
            claim_direction=v4.Direction.LOWER,
            rounding=v5.AssemblyRounding.DOWN,
            source_hashes=("C094-LOWER",),
        ),
    )
    cases.append(
        run_single(
            "assembly_lower_losses_omitted",
            v5.GateVerifierV5([lower_obj]),
            ["LOWER_BAD"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("ASSEMBLY",),
        )
    )

    residual = claim(
        "UPPER_RESIDUAL_MISSING",
        "Upper assembly names Gamma and collar but omits them.",
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        d=d,
        measure=v4.Measure.PAIR_PALM,
        bound=4.35,
        model="BF_TORUS_EXACT_L24",
    )
    residual_obj = object_for(
        residual,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        assembly_spec=v5.AssemblySpec(
            expression="(near+exterior)/typing",
            inputs=(("near", "0.657"), ("exterior", "2.8185310984873743"), ("typing", "0.8")),
            required_terms=("near", "exterior", "typing"),
            residual_terms=("gamma", "collar"),
            displayed_value="4.35",
            claim_direction=v4.Direction.UPPER,
            rounding=v5.AssemblyRounding.UP,
            source_hashes=("C095-RESIDUAL",),
        ),
    )
    cases.append(
        run_single(
            "assembly_named_residuals_missing",
            v5.GateVerifierV5([residual_obj]),
            ["UPPER_RESIDUAL_MISSING"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("ASSEMBLY",),
        )
    )

    return cases


def domain_cases() -> List[dict]:
    d = domain(0.05)
    r = regime(d)
    cases = []

    sample_claim = claim(
        "RUNG_ONLY_80",
        "C_rep C_mark <= 80 uniformly.",
        direction=v4.Direction.LOWER,
        quantifier=v4.Quantifier.FOR_ALL,
        d=d,
        bound=80.0,
    )
    sample_obj = object_for(
        sample_claim,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        requires_domain_extremum=True,
        domain_extremum_certificate=v5.DomainExtremumCertificate(
            "RUNG-SAMPLE",
            v5.ExtremumCertificateType.RUNG_SAMPLE,
            d,
            v5.ExtremumKind.INFIMUM,
            "80.390",
            "C091-RUNGS",
            "Three registered rungs only.",
            sampled_points=((0.05, 85.388), (0.025, 81.445), (0.0125, 80.390)),
        ),
    )
    cases.append(
        run_single(
            "domain_infimum_rung_sample",
            v5.GateVerifierV5([sample_obj]),
            ["RUNG_ONLY_80"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("DOMAIN-INFIMUM",),
        )
    )

    false_full = replace(sample_claim, claim_id="FULL_INFIMUM_FALSE_80")
    false_obj = object_for(
        false_full,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        requires_domain_extremum=True,
        domain_extremum_certificate=v5.DomainExtremumCertificate(
            "FULL-INFIMUM",
            v5.ExtremumCertificateType.ASYMPTOTIC_MONOTONICITY,
            d,
            v5.ExtremumKind.INFIMUM,
            "79.9889203915211757",
            "C094-DOMAIN-INFIMUM",
            "Analytic r->0 limit plus monotonicity.",
            monotonicity_direction="increasing",
        ),
    )
    cases.append(
        run_single(
            "domain_infimum_true_value_below_80",
            v5.GateVerifierV5([false_obj]),
            ["FULL_INFIMUM_FALSE_80"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("DOMAIN-INFIMUM",),
        )
    )

    true_claim = replace(
        sample_claim,
        claim_id="FULL_INFIMUM_TRUE_79P988",
        statement="C_rep C_mark <= 79.988 uniformly.",
        bound_coefficient=79.988,
    )
    true_obj = object_for(
        true_claim,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        requires_domain_extremum=True,
        domain_extremum_certificate=false_obj.domain_extremum_certificate,
    )
    cases.append(
        run_single(
            "domain_infimum_conservative_target",
            v5.GateVerifierV5([true_obj]),
            ["FULL_INFIMUM_TRUE_79P988"],
            expected_valid=True,
            expected_promotable=True,
        )
    )
    return cases


def band_cases() -> List[dict]:
    d = domain()
    r = regime(d)
    cases = []

    missing = claim(
        "BAND_MISSING",
        "Upper coefficient with +0.02 allowance.",
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        d=d,
        bound=1.01,
        uncertainty_side=v4.UncertaintySide.UPPER,
    )
    missing_obj = object_for(
        missing,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
    )
    cases.append(
        run_single(
            "band_missing_provenance",
            v5.GateVerifierV5([missing_obj]),
            ["BAND_MISSING"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("SCHEMA", "BAND-PROVENANCE"),
        )
    )

    complete = replace(missing, claim_id="BAND_COMPLETE")
    complete_obj = object_for(
        complete,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        uncertainty_provenance=v5.UncertaintyProvenance(
            "BAND-CERT",
            v5.BandKind.STATISTICAL,
            v4.UncertaintySide.UPPER,
            "Simultaneous one-sided bootstrap upper band.",
            "BAND-SOURCE-HASH",
            coverage_domain=d,
            confidence_level=0.99,
            sample_size=200000,
            multiplicity_scope="all registered stations and rungs",
            seed_or_algorithm_state="seed=95017",
            consumed_as=v4.UncertaintySide.UPPER,
        ),
    )
    cases.append(
        run_single(
            "band_complete",
            v5.GateVerifierV5([complete_obj]),
            ["BAND_COMPLETE"],
            expected_valid=True,
            expected_promotable=True,
        )
    )

    random_claim = claim(
        "RANDOM_DIAGNOSTIC",
        "Randomized orthant diagnostic with tolerance.",
        quantifier=v4.Quantifier.MEASURED_AT_RUNG,
        d=d,
        measured=0.01,
        scale_tag="r=0.025",
        uncertainty_side=v4.UncertaintySide.UPPER,
        status=v4.Status.MEASURED,
        grade=Grade.MEASURED,
    )
    random_obj = object_for(
        random_claim,
        warrant=v5.Warrant.MEASURED,
        est=r,
        dep=r,
        uncertainty_provenance=v5.UncertaintyProvenance(
            "RANDOM-NO-SEED",
            v5.BandKind.RANDOMIZED_NUMERICAL,
            v4.UncertaintySide.UPPER,
            "SciPy MVN diagnostic.",
            "RANDOM-SOURCE",
            coverage_domain=d,
            deterministic_tolerance="1e-6",
            consumed_as=v4.UncertaintySide.UPPER,
        ),
    )
    cases.append(
        run_single(
            "band_randomized_no_state",
            v5.GateVerifierV5([random_obj]),
            ["RANDOM_DIAGNOSTIC"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("BAND-PROVENANCE",),
        )
    )
    return cases


def common_mode_cases() -> List[dict]:
    d = domain()
    r = regime(d)
    cases = []

    c = claim(
        "AGREEMENT_SHARED",
        "Two instruments agree on a numerical coefficient.",
        d=d,
        grade=Grade.CERTIFIED,
    )
    shared = object_for(
        c,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        verification_mode=v5.VerificationMode.AGREEMENT,
        common_mode_certificate=v5.CommonModeCertificate(
            "CM-SHARED",
            v5.InstrumentRelation.SHARED_COMPONENTS,
            ("instrument-A", "instrument-B"),
            "coordinate convention",
            shared_components=("shared preprocessing",),
            source_hash="CM-SOURCE",
        ),
    )
    cases.append(
        run_single(
            "common_mode_shared_certified",
            v5.GateVerifierV5([shared]),
            ["AGREEMENT_SHARED"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("GRADE-CAP",),
            required_provisional_gates=("COMMON-MODE",),
        )
    )

    independent_claim = replace(c, claim_id="AGREEMENT_INDEPENDENT")
    independent = object_for(
        independent_claim,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        verification_mode=v5.VerificationMode.AGREEMENT,
        common_mode_certificate=v5.CommonModeCertificate(
            "CM-INDEPENDENT",
            v5.InstrumentRelation.INDEPENDENT,
            ("raw-cov implementation", "Hermite-Schur implementation"),
            "conditioning algebra",
            disjoint_error_argument=(
                "No shared covariance assembly, factorization, or estimator code."
            ),
            source_hash="CM-INDEPENDENT-SOURCE",
        ),
    )
    cases.append(
        run_single(
            "common_mode_independent",
            v5.GateVerifierV5([independent]),
            ["AGREEMENT_INDEPENDENT"],
            expected_valid=True,
            expected_promotable=True,
        )
    )
    return cases


def heuristic_cases() -> List[dict]:
    d = domain()
    r = regime(d)
    cases = []

    bad_claim = claim(
        "HEURISTIC_DERIVED",
        "An analogy predicts a quantitative exponent.",
        d=d,
        grade=Grade.DERIVED,
    )
    bad = object_for(
        bad_claim,
        warrant=v5.Warrant.DERIVED,
        est=r,
        dep=r,
        reasoning_mode=v5.ReasoningMode.ANALOGICAL,
    )
    cases.append(
        run_single(
            "heuristic_unbridged_derived",
            v5.GateVerifierV5([bad]),
            ["HEURISTIC_DERIVED"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("HEURISTIC-BRIDGE",),
        )
    )

    plausible_claim = replace(
        bad_claim,
        claim_id="HEURISTIC_PLAUSIBLE",
        grade=Grade.HYPOTHESIS,
    )
    plausible = object_for(
        plausible_claim,
        warrant=v5.Warrant.PLAUSIBLE,
        est=r,
        dep=r,
        reasoning_mode=v5.ReasoningMode.ANALOGICAL,
    )
    cases.append(
        run_single(
            "heuristic_unbridged_plausible",
            v5.GateVerifierV5([plausible]),
            ["HEURISTIC_PLAUSIBLE"],
            expected_valid=True,
            expected_promotable=False,
            required_provisional_gates=("HEURISTIC-BRIDGE",),
        )
    )

    bridge_claim = claim(
        "ANALOGY_BRIDGE",
        "A derivation transports the source-domain scaling to the target model.",
        d=d,
        status=v4.Status.ESTABLISHED,
        grade=Grade.ESTABLISHED,
    )
    target_claim = replace(bad_claim, claim_id="HEURISTIC_BRIDGED")
    bridge_obj = object_for(
        bridge_claim,
        warrant=v5.Warrant.PROVEN,
        est=r,
        dep=r,
        load_bearing=False,
    )
    target = object_for(
        target_claim,
        warrant=v5.Warrant.DERIVED,
        est=r,
        dep=r,
        reasoning_mode=v5.ReasoningMode.ANALOGICAL,
        heuristic_bridge=v5.HeuristicBridge(
            "ANALOGY_BRIDGE",
            "source theory",
            "target theory",
            "Exact derivation of the exponent transformation.",
            "BRIDGE-HASH",
            "Mismatch on a held-out parameter family.",
        ),
    )
    cases.append(
        run_single(
            "heuristic_bridged",
            v5.GateVerifierV5([bridge_obj, target]),
            ["HEURISTIC_BRIDGED"],
            expected_valid=True,
            expected_promotable=True,
        )
    )
    return cases


def regime_cases() -> List[dict]:
    d = domain()
    est = regime(
        d,
        model="MODEL-A",
        measure=v4.Measure.GAUSSIAN_PINNED,
        marks=("position",),
    )
    dep = regime(
        d,
        model="MODEL-A",
        measure=v4.Measure.PAIR_PALM,
        marks=("position", "height"),
    )
    c = claim(
        "REGIME_DROP",
        "A pinned estimate is deployed under pair-Palm with an extra height mark.",
        d=d,
        measure=v4.Measure.PAIR_PALM,
        model="MODEL-A",
    )
    obj = object_for(
        c,
        warrant=v5.Warrant.DERIVED,
        est=est,
        dep=dep,
    )
    bad = run_single(
        "regime_measure_mark_drop",
        v5.GateVerifierV5([obj]),
        ["REGIME_DROP"],
        expected_valid=False,
        expected_promotable=False,
        required_fail_gates=("MEASURE", "MARK"),
    )

    bridge = v5.TransferBridge(
        "PALM-MARK-BRIDGE",
        frozenset({v5.BridgeKind.MEASURE, v5.BridgeKind.MARK}),
        est,
        dep,
        v5.Warrant.CERTIFIED,
        "Exact determinant-weighted change of measure and marked density.",
        "BRIDGE-SOURCE",
    )
    good_claim = replace(c, claim_id="REGIME_BRIDGED")
    good_obj = object_for(
        good_claim,
        warrant=v5.Warrant.DERIVED,
        est=est,
        dep=dep,
        transfer_bridges=(bridge,),
    )
    good = run_single(
        "regime_measure_mark_bridged",
        v5.GateVerifierV5([good_obj]),
        ["REGIME_BRIDGED"],
        expected_valid=True,
        expected_promotable=True,
    )
    return [bad, good]


def coverage_and_active_gate_cases() -> List[dict]:
    d = domain()
    r = regime(d)
    cases = []

    c = claim(
        "COVERAGE_BAD",
        "Total false acceptance risk is at most 0.001.",
        direction=v4.Direction.UPPER,
        d=d,
        bound=0.001,
    )
    obj = object_for(
        c,
        warrant=v5.Warrant.DERIVED,
        est=r,
        dep=r,
        coverage_spec=v5.CoverageSpec(
            "wrong-output universe",
            ("chart-A", "chart-B"),
            "coverage-cert",
            uncovered_mass=0.002,
            target_total_risk=0.001,
        ),
        selection_risk_spec=v5.SelectionRiskSpec(
            coverage_failure=0.002,
            baseline_hazard_rate=0.001,
            expected_selected_measure=10,
            selection_amplification=3,
            q_sard_tube_risk=0.0002,
        ),
    )
    cases.append(
        run_single(
            "coverage_target_below_floor",
            v5.GateVerifierV5([obj]),
            ["COVERAGE_BAD"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("COVERAGE",),
        )
    )

    active_claim = claim(
        "ACTIVE_GATE_OPEN",
        "A sharpened upper theorem.",
        d=d,
        program_core=True,
    )
    active_obj = object_for(
        active_claim,
        warrant=v5.Warrant.CERTIFIED,
        est=r,
        dep=r,
        active_gates=(
            v5.ActiveGate(
                "H4-PATH",
                v5.ActiveGateStatus.OPEN,
                "H4-PATH-OBLIGATION",
            ),
        ),
    )
    cases.append(
        run_single(
            "active_q0_gate_open",
            v5.GateVerifierV5([active_obj]),
            ["ACTIVE_GATE_OPEN"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("H4-PATH",),
        )
    )
    return cases


def dependency_warrant_cases() -> List[dict]:
    d = domain()
    r = regime(d)
    measured_claim = claim(
        "MEASURED_DEP",
        "Measured input.",
        d=d,
        status=v4.Status.MEASURED,
        grade=Grade.MEASURED,
        measured=1.2,
        scale_tag="test scale",
    )
    parent_claim = claim(
        "PROVEN_PARENT",
        "A purported proven theorem.",
        d=d,
        status=v4.Status.ESTABLISHED,
        grade=Grade.ESTABLISHED,
        dependencies=("MEASURED_DEP",),
    )
    measured_obj = object_for(
        measured_claim,
        warrant=v5.Warrant.MEASURED,
        est=r,
        dep=r,
        load_bearing=False,
    )
    parent_obj = object_for(
        parent_claim,
        warrant=v5.Warrant.PROVEN,
        est=r,
        dep=r,
    )
    return [
        run_single(
            "warrant_parent_exceeds_measured_dependency",
            v5.GateVerifierV5([measured_obj, parent_obj]),
            ["PROVEN_PARENT"],
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("DEPENDENCY-WARRANT",),
        )
    ]


def main() -> None:
    cases: List[dict] = []

    current_q0, current_roots = v5.current_q0_upper_contract()
    cases.append(
        run_single(
            "current_q0_residual_gap",
            current_q0,
            current_roots,
            expected_valid=False,
            expected_promotable=False,
            required_fail_gates=("ASSEMBLY", "UB_G_RESIDUAL_UNIFORM"),
        )
    )

    second, second_roots = v5.matrix_perturbation_second_domain()
    second_case = run_single(
        "second_domain_matrix_perturbation",
        second,
        second_roots,
        expected_valid=True,
        expected_promotable=True,
    )
    cases.append(second_case)

    cases.extend(assembly_cases())
    cases.extend(domain_cases())
    cases.extend(band_cases())
    cases.extend(common_mode_cases())
    cases.extend(heuristic_cases())
    cases.extend(regime_cases())
    cases.extend(coverage_and_active_gate_cases())
    cases.extend(dependency_warrant_cases())

    passed = sum(case["passed"] for case in cases)
    result = {
        "validator": "q0_llm_verifier_v5",
        "cases": cases,
        "case_count": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "all_expected_checks_pass": passed == len(cases),
        "gate_discrimination": {
            gate: sum(
                gate in case["observed"]["fail_gates"]
                or gate in case["observed"]["provisional_gates"]
                for case in cases
            )
            for gate in [
                "ASSEMBLY",
                "DOMAIN-INFIMUM",
                "BAND-PROVENANCE",
                "COMMON-MODE",
                "HEURISTIC-BRIDGE",
                "MEASURE",
                "MARK",
                "COVERAGE",
                "DEPENDENCY-WARRANT",
                "H4-PATH",
                "UB_G_RESIDUAL_UNIFORM",
            ]
        },
        "second_domain_root_hashes": second_case["observed"]["root_hashes"],
    }
    path = BASE / "q0_llm_verifier_v5_validation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_expected_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
