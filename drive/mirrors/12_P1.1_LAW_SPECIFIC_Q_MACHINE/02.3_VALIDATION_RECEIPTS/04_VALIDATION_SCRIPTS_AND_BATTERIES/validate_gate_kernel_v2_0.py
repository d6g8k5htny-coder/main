#!/usr/bin/env python3
"""Independent validation battery for Gate Kernel 2.0."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Sequence

BASE = Path(__file__).resolve().parent
KERNEL_PATH = BASE / "gate_kernel_v2_0.py"


def load_kernel():
    spec = importlib.util.spec_from_file_location("gate_kernel_v2_0_validation_target", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Gate Kernel 2.0")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gk = load_kernel()


def d_r_b():
    return gk.domain(
        r=gk.interval("0", "1", lower_open=True),
        b=gk.interval("0", "3"),
    )


def d_r_only():
    return gk.domain(r=gk.interval("0", "1", lower_open=True))


def regime(
    domain=None,
    *,
    model="MODEL-A",
    measure="PAIR-PALM",
    scale="coefficient of r^3",
    marks=("b",),
    conditioning="typed pair",
    region="test region",
):
    return gk.Regime(
        model_id=model,
        measure_id=measure,
        domain=domain or d_r_b(),
        spatial_region=region,
        scale_id=scale,
        conditioning_depth=conditioning,
        marks=frozenset(marks),
        deployment_mode="validation",
    )


def statement(
    claim_id: str,
    *,
    warrant=gk.Warrant.DERIVED,
    status=gk.ClaimStatus.LIVE,
    evidence=True,
    quantifier=gk.Quantifier.CONDITIONAL,
    direction=gk.Direction.NONE,
    bound=None,
    est=None,
    dep=None,
    required_marks=(),
    supplied_marks=(),
    operation=gk.Operation.ATOMIC,
    verification=gk.VerificationMode.NONE,
    closed=False,
    conditional_on=(),
    active_gates=(),
    edges=(),
    assembly=None,
    endpoints=(),
    finite=False,
    uncertainty=gk.UncertaintySide.NONE,
    measured=(),
    reasoning=gk.ReasoningMode.DEDUCTIVE,
    load_bearing=True,
    kind=gk.ClaimKind.STATEMENT,
    bridge=None,
    witness=None,
    extremum=None,
    band=None,
    common=None,
    coverage=None,
):
    est = est or regime()
    dep = dep or est
    return gk.Claim(
        claim_id=claim_id,
        statement=f"Validation claim {claim_id}.",
        kind=kind,
        status=status,
        warrant=warrant,
        source_precedence_id="C096-VALIDATION",
        evidence_ids=(f"EVIDENCE-{claim_id}",) if evidence else (),
        reasoning_mode=reasoning,
        load_bearing=load_bearing,
        quantifier=quantifier,
        direction=direction,
        bound_coefficient=bound,
        uncertainty_side=uncertainty,
        establishment_regime=est,
        deployment_regime=dep,
        required_marks=frozenset(required_marks),
        supplied_marks=frozenset(supplied_marks),
        operation=operation,
        measured_values=tuple(measured),
        finite_range_bound=finite,
        registered_endpoints=tuple(endpoints),
        bridge_spec=bridge,
        witness_spec=witness,
        assembly_spec=assembly,
        extremum_spec=extremum,
        band_spec=band,
        common_mode_spec=common,
        coverage_spec=coverage,
        verification_mode=verification,
        closed_core=closed,
        conditional_on=tuple(conditional_on),
        active_gates=tuple(active_gates),
        edges=tuple(edges),
    )


def seal(*claims):
    return gk.seal_graph_unpinned(claims)


def fail_gates(report, claim_id):
    return {
        item.gate
        for item in report.claim_reports[claim_id].gate_results
        if item.verdict == gk.Verdict.FAIL
    }


def provisional_gates(report, claim_id):
    return {
        item.gate
        for item in report.claim_reports[claim_id].gate_results
        if item.verdict == gk.Verdict.PROVISIONAL
    }


def case(
    name,
    observed,
    *,
    expected_shell,
    expected_conditional,
    expected_unconditional,
    required_fail=(),
    required_provisional=(),
    claim_id=None,
):
    if isinstance(observed, gk.GraphReport):
        report = observed
        target = claim_id or report.roots[0]
        failures = fail_gates(report, target) if report.claim_reports else set()
        provisionals = (
            provisional_gates(report, target) if report.claim_reports else set()
        )
        checks = {
            "shell_valid": report.shell_valid == expected_shell,
            "conditional_promotable": (
                report.conditional_promotable == expected_conditional
            ),
            "unconditional_promotable": (
                report.unconditional_promotable == expected_unconditional
            ),
            "required_fail_gates": set(required_fail).issubset(failures),
            "required_provisional_gates": set(required_provisional).issubset(
                provisionals
            ),
        }
        payload = report.as_dict()
    else:
        checks = dict(observed["checks"])
        payload = observed["payload"]
    return {
        "name": name,
        "checks": checks,
        "passed": all(checks.values()),
        "observed": payload,
    }


def valid_second_domain():
    graph, roots = gk.demo_second_domain()
    return case(
        "valid_second_domain",
        gk.evaluate_graph(graph, roots),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def domain_empty_dependency():
    parent_regime = regime()
    dep_regime = regime(gk.ParameterDomain())
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=dep_regime, dep=dep_regime)
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        quantifier=gk.Quantifier.FOR_ALL,
        est=parent_regime,
        dep=parent_regime,
        edges=(gk.Edge("dep", gk.EdgeKind.SUPPORT),),
    )
    graph = seal(dep, parent)
    return case(
        "domain_empty_dependency",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("DOMAIN",),
    )


def domain_missing_parameter():
    dep_regime = regime(d_r_only())
    parent_regime = regime(d_r_b())
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=dep_regime, dep=dep_regime)
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        quantifier=gk.Quantifier.FOR_ALL,
        est=parent_regime,
        dep=parent_regime,
        edges=(gk.Edge("dep", gk.EdgeKind.SUPPORT),),
    )
    graph = seal(dep, parent)
    return case(
        "domain_missing_parameter",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("DOMAIN",),
    )


def mark_self_attestation():
    reg = regime()
    dep = statement(
        "unmarked", warrant=gk.Warrant.PROVEN, est=reg, dep=reg, supplied_marks=()
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=reg,
        dep=reg,
        required_marks=("height", "critical_type"),
        supplied_marks=("height", "critical_type"),
        edges=(gk.Edge("unmarked", gk.EdgeKind.SUPPORT),),
    )
    graph = seal(dep, parent)
    return case(
        "mark_self_attestation",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("MARK",),
    )


def untyped_measure_bridge():
    pinned_regime = regime(measure="GAUSSIAN-PINNED")
    palm_regime = regime(measure="PAIR-PALM")
    pinned = statement(
        "pinned", warrant=gk.Warrant.PROVEN, est=pinned_regime, dep=pinned_regime
    )
    unrelated = statement("unrelated", warrant=gk.Warrant.PROVEN, load_bearing=False)
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=palm_regime,
        dep=palm_regime,
        edges=(
            gk.Edge("pinned", gk.EdgeKind.SUPPORT),
            gk.Edge("unrelated", gk.EdgeKind.CHANGE_OF_MEASURE),
        ),
    )
    graph = seal(pinned, unrelated, parent)
    return case(
        "untyped_measure_bridge",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("MEASURE",),
    )


def typed_measure_bridge():
    pinned_regime = regime(measure="GAUSSIAN-PINNED")
    palm_regime = regime(measure="PAIR-PALM")
    pinned = statement(
        "pinned", warrant=gk.Warrant.PROVEN, est=pinned_regime, dep=pinned_regime
    )
    bridge = statement(
        "bridge",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.BRIDGE,
        load_bearing=False,
        bridge=gk.BridgeSpec(
            gk.BridgeKind.MEASURE,
            pinned_regime,
            palm_regime,
            palm_regime.domain,
            scope_statement="Exact Radon-Nikodym bridge.",
        ),
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=palm_regime,
        dep=palm_regime,
        edges=(
            gk.Edge("pinned", gk.EdgeKind.SUPPORT),
            gk.Edge("bridge", gk.EdgeKind.CHANGE_OF_MEASURE),
        ),
    )
    graph = seal(pinned, bridge, parent)
    return case(
        "typed_measure_bridge",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def untyped_model_bridge():
    a = regime(model="MODEL-A")
    b = regime(model="MODEL-B")
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=a, dep=a)
    unrelated = statement("cert", warrant=gk.Warrant.PROVEN, load_bearing=False)
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=b,
        dep=b,
        edges=(
            gk.Edge("dep", gk.EdgeKind.SUPPORT),
            gk.Edge("cert", gk.EdgeKind.MODEL_TRANSFER),
        ),
    )
    graph = seal(dep, unrelated, parent)
    return case(
        "untyped_model_bridge",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("MODEL",),
    )


def typed_model_bridge():
    a = regime(model="MODEL-A")
    b = regime(model="MODEL-B")
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=a, dep=a)
    bridge = statement(
        "bridge",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.BRIDGE,
        load_bearing=False,
        bridge=gk.BridgeSpec(
            gk.BridgeKind.MODEL,
            a,
            b,
            b.domain,
            scope_statement="Validated perturbation transfer.",
        ),
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=b,
        dep=b,
        edges=(
            gk.Edge("dep", gk.EdgeKind.SUPPORT),
            gk.Edge("bridge", gk.EdgeKind.MODEL_TRANSFER),
        ),
    )
    graph = seal(dep, bridge, parent)
    return case(
        "typed_model_bridge",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def wrong_operation_witness():
    reg = regime()
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=reg, dep=reg)
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.UNION,
            gk.WitnessType.UNION_BOUND,
            ("dep",),
            "Union bound.",
        ),
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=reg,
        dep=reg,
        operation=gk.Operation.PRODUCT,
        edges=(
            gk.Edge("dep", gk.EdgeKind.SUPPORT),
            gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),
        ),
    )
    graph = seal(dep, witness, parent)
    return case(
        "wrong_operation_witness",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("COMPOSITION",),
    )


def correct_product_witness():
    reg = regime()
    dep = statement("dep", warrant=gk.Warrant.PROVEN, est=reg, dep=reg)
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.PRODUCT,
            gk.WitnessType.INDEPENDENCE,
            ("dep",),
            "Independence theorem.",
        ),
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        est=reg,
        dep=reg,
        operation=gk.Operation.PRODUCT,
        edges=(
            gk.Edge("dep", gk.EdgeKind.SUPPORT),
            gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),
        ),
    )
    graph = seal(dep, witness, parent)
    return case(
        "correct_product_witness",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def common_mode_missing_certificate():
    c = statement(
        "agreement",
        warrant=gk.Warrant.PROVEN,
        verification=gk.VerificationMode.AGREEMENT,
    )
    graph = seal(c)
    return case(
        "common_mode_missing_certificate",
        gk.evaluate_graph(graph, ("agreement",)),
        expected_shell=True,
        expected_conditional=False,
        expected_unconditional=False,
        required_provisional=("COMMON-MODE",),
    )


def common_mode_independent_certificate():
    reg = regime()
    cert = statement(
        "cm_cert",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        common=gk.CommonModeSpec(
            gk.InstrumentRelation.INDEPENDENT,
            ("instrument-A", "instrument-B"),
            "covariance assembly",
            disjoint_error_argument="No shared code, data, or factorization.",
        ),
    )
    c = statement(
        "agreement",
        warrant=gk.Warrant.PROVEN,
        est=reg,
        dep=reg,
        verification=gk.VerificationMode.AGREEMENT,
        edges=(
            gk.Edge(
                "cm_cert",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.COMMON_MODE.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "common_mode_independent_certificate",
        gk.evaluate_graph(graph, ("agreement",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def provisional_dependency_blocks_parent():
    dep = statement(
        "dep",
        warrant=gk.Warrant.PROVEN,
        verification=gk.VerificationMode.AGREEMENT,
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        edges=(gk.Edge("dep", gk.EdgeKind.SUPPORT),),
    )
    graph = seal(dep, parent)
    return case(
        "provisional_dependency_blocks_parent",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("DEPENDENCY-WARRANT",),
    )


def provisional_dependency_as_condition():
    dep = statement(
        "dep",
        warrant=gk.Warrant.DERIVED,
        status=gk.ClaimStatus.OPEN,
        verification=gk.VerificationMode.AGREEMENT,
        load_bearing=False,
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN_MODULO,
        quantifier=gk.Quantifier.CONDITIONAL,
        conditional_on=("dep",),
        edges=(gk.Edge("dep", gk.EdgeKind.CONDITION, conditional=True),),
    )
    graph = seal(dep, parent)
    return case(
        "provisional_dependency_as_condition",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=False,
    )


def hash_mutation_rejected():
    graph, roots = gk.demo_second_domain()
    data = gk.graph_to_dict(graph, roots=roots)
    for item in data["claims"]:
        if item["claim_id"] == "matrix_perturbation":
            item["statement"] += " MUTATED"
            break
    try:
        gk.graph_from_dict(data)
        payload = {"raised": False}
        passed = False
    except ValueError as exc:
        payload = {"raised": True, "exception": str(exc)}
        passed = "content hash mismatch" in str(exc)
    return case(
        "hash_mutation_rejected",
        {
            "checks": {"hash_mismatch_detected": passed},
            "payload": payload,
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def dead_root_rejected():
    c = statement(
        "dead",
        warrant=gk.Warrant.PROVEN,
        status=gk.ClaimStatus.KILLED,
        load_bearing=False,
    ).sealed()
    report = gk.evaluate_graph({"dead": c}, ("dead",))
    checks = {
        "graph_or_precedence_rejects": (
            bool(report.graph_errors)
            or "PRECEDENCE" in fail_gates(report, "dead")
        )
    }
    return case(
        "dead_root_rejected",
        {"checks": checks, "payload": report.as_dict()},
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def closed_core_conjecture_rejected():
    cond = statement(
        "cond",
        warrant=gk.Warrant.CONJECTURE,
        status=gk.ClaimStatus.OPEN,
        load_bearing=False,
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN_MODULO,
        closed=True,
        conditional_on=("cond",),
        edges=(gk.Edge("cond", gk.EdgeKind.CONDITION, conditional=True),),
    )
    graph = seal(cond, parent)
    return case(
        "closed_core_conjecture_rejected",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("CORE-CLOSURE",),
    )


def malformed_registry_load_raises():
    good = statement("good", warrant=gk.Warrant.PROVEN).sealed()
    bad = statement(
        "bad",
        warrant=gk.Warrant.PROVEN,
        edges=(gk.Edge("ghost", gk.EdgeKind.SUPPORT, "ghost"),),
    ).sealed()
    payload = {
        "schema": gk.SCHEMA_VERSION,
        "roots": ["bad"],
        "claims": [good.as_dict(), bad.as_dict()],
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            gk.Registry.load(path)
            raised = False
            text = ""
        except ValueError as exc:
            raised = True
            text = str(exc)
    return case(
        "malformed_registry_load_raises",
        {
            "checks": {
                "raised": raised,
                "missing_target_reported": "missing-target" in text,
            },
            "payload": {"raised": raised, "exception": text},
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def schema_version_rejected():
    try:
        gk.graph_from_dict({"schema": "gate-kernel/999", "claims": []})
        raised = False
        text = ""
    except ValueError as exc:
        raised = True
        text = str(exc)
    return case(
        "schema_version_rejected",
        {
            "checks": {
                "raised": raised,
                "unsupported_reported": "unsupported schema" in text,
            },
            "payload": {"raised": raised, "exception": text},
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def cycle_rejected():
    a = statement("A", edges=(gk.Edge("B", pinned_hash="x"),)).sealed()
    b = statement("B", edges=(gk.Edge("A", pinned_hash=a.content_hash),)).sealed()
    graph = {"A": a, "B": b}
    try:
        gk.topological_order(graph)
        raised = False
        text = ""
    except ValueError as exc:
        raised = True
        text = str(exc)
    return case(
        "cycle_rejected",
        {
            "checks": {"raised": raised, "cycle_reported": "cycle" in text},
            "payload": {"raised": raised, "exception": text},
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def portable_demo_path():
    with tempfile.TemporaryDirectory() as td:
        output = Path(td) / "demo.json"
        completed = subprocess.run(
            [sys.executable, str(KERNEL_PATH), "--write-demo", str(output)],
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=180,
        )
        exists = output.exists()
        load_ok = False
        if exists:
            reg = gk.Registry.load(output)
            load_ok = reg.admit(gk.AdmissionMode.PROMOTED)[0]
    return case(
        "portable_demo_path",
        {
            "checks": {
                "returncode_zero": completed.returncode == 0,
                "output_exists": exists,
                "roundtrip_promoted": load_ok,
            },
            "payload": {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            },
        },
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def legacy_schema_rejected():
    try:
        gk.Registry.load(BASE / "q0 registry.json")
        raised = False
        text = ""
    except ValueError as exc:
        raised = True
        text = str(exc)
    return case(
        "legacy_schema_rejected",
        {
            "checks": {
                "raised": raised,
                "unsupported_reported": "unsupported schema" in text,
            },
            "payload": {"raised": raised, "exception": text},
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def migrated_registry_classified():
    registry = gk.Registry.load(BASE / "q0_registry_v2_0.json")
    archive = registry.admit(gk.AdmissionMode.ARCHIVE)
    candidate = registry.admit(gk.AdmissionMode.CANDIDATE)
    promoted = registry.admit(gk.AdmissionMode.PROMOTED)
    root = registry.report.claim_reports["T_MAIN_DEMO"]
    return case(
        "migrated_registry_classified",
        {
            "checks": {
                "archive_true": archive[0],
                "candidate_false": not candidate[0],
                "promoted_false": not promoted[0],
                "name_collision_removed": "PAIRING_REMAINDER_CONDITION" in registry.graph,
                "five_conditions": len(root.conditions) == 5,
            },
            "payload": {
                "archive": archive,
                "candidate": candidate,
                "promoted": promoted,
                "root": root.as_dict(),
            },
        },
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
    )


def assembly_down_rounding():
    reg = regime()
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.AFFINE_ASSEMBLY,
            gk.WitnessType.MONOTONE_ASSEMBLY,
            statement="Monotone upper assembly.",
        ),
    )
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        quantifier=gk.Quantifier.CONDITIONAL,
        direction=gk.Direction.UPPER,
        est=reg,
        dep=reg,
        operation=gk.Operation.AFFINE_ASSEMBLY,
        assembly=gk.AssemblySpec(
            "(near+exterior)/typing",
            (
                gk.AssemblyInput("near", "0.66", "near-source", 1),
                gk.AssemblyInput("exterior", "2.82", "ext-source", 1),
                gk.AssemblyInput("typing", "0.80", "typing-source", -1),
            ),
            ("near", "exterior", "typing"),
            (),
            "4.3",
            gk.Direction.UPPER,
            gk.RoundingRule.UP,
        ),
        edges=(gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),),
    )
    graph = seal(witness, c)
    return case(
        "assembly_down_rounding",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("ASSEMBLY",),
    )


def assembly_round_up():
    reg = regime()
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.AFFINE_ASSEMBLY,
            gk.WitnessType.MONOTONE_ASSEMBLY,
            statement="Monotone upper assembly.",
        ),
    )
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.UPPER,
        est=reg,
        dep=reg,
        operation=gk.Operation.AFFINE_ASSEMBLY,
        assembly=gk.AssemblySpec(
            "(near+exterior)/typing",
            (
                gk.AssemblyInput("near", "0.66", "near-source", 1),
                gk.AssemblyInput("exterior", "2.82", "ext-source", 1),
                gk.AssemblyInput("typing", "0.80", "typing-source", -1),
            ),
            ("near", "exterior", "typing"),
            (),
            "4.35",
            gk.Direction.UPPER,
            gk.RoundingRule.UP,
        ),
        edges=(gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),),
    )
    graph = seal(witness, c)
    return case(
        "assembly_round_up",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def assembly_missing_residual():
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.AFFINE_ASSEMBLY,
            gk.WitnessType.MONOTONE_ASSEMBLY,
        ),
    )
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.UPPER,
        operation=gk.Operation.AFFINE_ASSEMBLY,
        assembly=gk.AssemblySpec(
            "(near+exterior)/typing",
            (
                gk.AssemblyInput("near", "0.657", "near", 1),
                gk.AssemblyInput("exterior", "2.8185", "ext", 1),
                gk.AssemblyInput("typing", "0.8", "typing", -1),
            ),
            ("near", "exterior", "typing"),
            ("gamma", "collar"),
            "4.35",
            gk.Direction.UPPER,
            gk.RoundingRule.UP,
        ),
        edges=(gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),),
    )
    graph = seal(witness, c)
    return case(
        "assembly_missing_residual",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("ASSEMBLY",),
    )


def assembly_lower_losses():
    witness = statement(
        "witness",
        warrant=gk.Warrant.PROVEN,
        kind=gk.ClaimKind.WITNESS,
        load_bearing=False,
        witness=gk.WitnessSpec(
            gk.Operation.AFFINE_ASSEMBLY,
            gk.WitnessType.MONOTONE_ASSEMBLY,
        ),
    )
    c = statement(
        "lower",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.LOWER,
        operation=gk.Operation.AFFINE_ASSEMBLY,
        assembly=gk.AssemblySpec(
            "cstar*ao*bonf",
            (
                gk.AssemblyInput("cstar", "0.8411", "cstar", 1),
                gk.AssemblyInput("ao", "0.999965", "ao", 1),
                gk.AssemblyInput("bonf", "0.99999", "bonf", 1),
            ),
            ("cstar", "ao", "bonf"),
            (),
            "0.8411",
            gk.Direction.LOWER,
            gk.RoundingRule.DOWN,
        ),
        edges=(gk.Edge("witness", gk.EdgeKind.COMPOSITION_WITNESS),),
    )
    graph = seal(witness, c)
    return case(
        "assembly_lower_losses",
        gk.evaluate_graph(graph, ("lower",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("ASSEMBLY",),
    )


def domain_infimum_rung_sample():
    reg = regime(domain=gk.domain(r=gk.interval("0", "0.05", lower_open=True)))
    cert = statement(
        "cert",
        warrant=gk.Warrant.CERTIFIED,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        extremum=gk.ExtremumSpec(
            gk.ExtremumCertificateType.RUNG_SAMPLE,
            gk.ExtremumKind.INFIMUM,
            reg.domain,
            "80.390",
            "Three rungs.",
        ),
    )
    c = statement(
        "lower",
        warrant=gk.Warrant.CERTIFIED,
        quantifier=gk.Quantifier.FOR_ALL,
        direction=gk.Direction.LOWER,
        bound="80",
        est=reg,
        dep=reg,
        edges=(
            gk.Edge(
                "cert",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.DOMAIN_EXTREMUM.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "domain_infimum_rung_sample",
        gk.evaluate_graph(graph, ("lower",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("DOMAIN-INFIMUM",),
    )


def domain_infimum_conservative():
    reg = regime(domain=gk.domain(r=gk.interval("0", "0.05", lower_open=True)))
    cert = statement(
        "cert",
        warrant=gk.Warrant.CERTIFIED,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        extremum=gk.ExtremumSpec(
            gk.ExtremumCertificateType.ASYMPTOTIC_MONOTONICITY,
            gk.ExtremumKind.INFIMUM,
            reg.domain,
            "79.9889203915211757",
            "Analytic limit plus monotonicity.",
        ),
    )
    c = statement(
        "lower",
        warrant=gk.Warrant.CERTIFIED,
        quantifier=gk.Quantifier.FOR_ALL,
        direction=gk.Direction.LOWER,
        bound="79.988",
        est=reg,
        dep=reg,
        edges=(
            gk.Edge(
                "cert",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.DOMAIN_EXTREMUM.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "domain_infimum_conservative",
        gk.evaluate_graph(graph, ("lower",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def band_missing():
    reg = regime()
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.UPPER,
        uncertainty=gk.UncertaintySide.UPPER,
        est=reg,
        dep=reg,
    )
    graph = seal(c)
    return case(
        "band_missing",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("BAND-PROVENANCE",),
    )


def randomized_band_no_state():
    reg = regime()
    cert = statement(
        "band",
        warrant=gk.Warrant.CERTIFIED,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        band=gk.BandSpec(
            gk.BandKind.RANDOMIZED_NUMERICAL,
            gk.UncertaintySide.UPPER,
            "Randomized MVN integration.",
            reg.domain,
            deterministic_tolerance="1e-6",
            consumed_as=gk.UncertaintySide.UPPER,
        ),
    )
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.UPPER,
        uncertainty=gk.UncertaintySide.UPPER,
        est=reg,
        dep=reg,
        edges=(
            gk.Edge(
                "band",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.BAND_PROVENANCE.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "randomized_band_no_state",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("BAND-PROVENANCE",),
    )


def complete_band():
    reg = regime()
    cert = statement(
        "band",
        warrant=gk.Warrant.CERTIFIED,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        band=gk.BandSpec(
            gk.BandKind.STATISTICAL,
            gk.UncertaintySide.UPPER,
            "Simultaneous one-sided bootstrap.",
            reg.domain,
            confidence_level="0.99",
            sample_size=200000,
            multiplicity_scope="all stations and rungs",
            seed_or_algorithm_state="seed=96017",
            consumed_as=gk.UncertaintySide.UPPER,
        ),
    )
    c = statement(
        "upper",
        warrant=gk.Warrant.CERTIFIED,
        direction=gk.Direction.UPPER,
        uncertainty=gk.UncertaintySide.UPPER,
        est=reg,
        dep=reg,
        edges=(
            gk.Edge(
                "band",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.BAND_PROVENANCE.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "complete_band",
        gk.evaluate_graph(graph, ("upper",)),
        expected_shell=True,
        expected_conditional=True,
        expected_unconditional=True,
    )


def coverage_floor():
    reg = regime()
    cert = statement(
        "coverage",
        warrant=gk.Warrant.CERTIFIED,
        kind=gk.ClaimKind.CERTIFICATE,
        load_bearing=False,
        coverage=gk.CoverageSpec(
            "wrong-output universe",
            (("region-A", ("chart-A",)),),
            uncovered_mass="0.002",
            target_total_risk="0.001",
        ),
    )
    c = statement(
        "risk",
        warrant=gk.Warrant.CERTIFIED,
        est=reg,
        dep=reg,
        edges=(
            gk.Edge(
                "coverage",
                gk.EdgeKind.CERTIFICATE,
                role=gk.CertificateRole.COVERAGE.value,
            ),
        ),
    )
    graph = seal(cert, c)
    return case(
        "coverage_floor",
        gk.evaluate_graph(graph, ("risk",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("COVERAGE",),
    )


def active_gate_open():
    c = statement(
        "sharp",
        warrant=gk.Warrant.CERTIFIED,
        active_gates=(
            gk.ActiveGate(
                "H4-PATH",
                gk.ActiveGateStatus.OPEN,
                "H4-OBLIGATION",
            ),
        ),
    )
    graph = seal(c)
    return case(
        "active_gate_open",
        gk.evaluate_graph(graph, ("sharp",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("H4-PATH",),
    )


def measured_rung_missing_evidence():
    c = statement(
        "measurement",
        warrant=gk.Warrant.MEASURED,
        kind=gk.ClaimKind.MEASUREMENT,
        quantifier=gk.Quantifier.MEASURED_AT_RUNG,
        measured=(
            gk.MeasuredValue("x", "1.2", "r=0.025", evidence_id=""),
        ),
    )
    graph = seal(c)
    return case(
        "measured_rung_missing_evidence",
        gk.evaluate_graph(graph, ("measurement",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("RUNG",),
    )


def parent_inherits_proven_modulo_conditions():
    cond = statement(
        "cond",
        warrant=gk.Warrant.CONJECTURE,
        status=gk.ClaimStatus.OPEN,
        load_bearing=False,
    )
    lemma = statement(
        "lemma",
        warrant=gk.Warrant.PROVEN_MODULO,
        conditional_on=("cond",),
        edges=(gk.Edge("cond", gk.EdgeKind.CONDITION, conditional=True),),
    )
    parent = statement(
        "parent",
        warrant=gk.Warrant.PROVEN,
        edges=(gk.Edge("lemma", gk.EdgeKind.SUPPORT),),
    )
    graph = seal(cond, lemma, parent)
    return case(
        "parent_inherits_proven_modulo_conditions",
        gk.evaluate_graph(graph, ("parent",)),
        expected_shell=False,
        expected_conditional=False,
        expected_unconditional=False,
        required_fail=("DEPENDENCY-WARRANT",),
    )


TESTS: list[Callable[[], dict]] = [
    valid_second_domain,
    domain_empty_dependency,
    domain_missing_parameter,
    mark_self_attestation,
    untyped_measure_bridge,
    typed_measure_bridge,
    untyped_model_bridge,
    typed_model_bridge,
    wrong_operation_witness,
    correct_product_witness,
    common_mode_missing_certificate,
    common_mode_independent_certificate,
    provisional_dependency_blocks_parent,
    provisional_dependency_as_condition,
    hash_mutation_rejected,
    dead_root_rejected,
    closed_core_conjecture_rejected,
    malformed_registry_load_raises,
    schema_version_rejected,
    cycle_rejected,
    portable_demo_path,
    legacy_schema_rejected,
    migrated_registry_classified,
    assembly_down_rounding,
    assembly_round_up,
    assembly_missing_residual,
    assembly_lower_losses,
    domain_infimum_rung_sample,
    domain_infimum_conservative,
    band_missing,
    randomized_band_no_state,
    complete_band,
    coverage_floor,
    active_gate_open,
    measured_rung_missing_evidence,
    parent_inherits_proven_modulo_conditions,
]


def main() -> None:
    rows = []
    for test in TESTS:
        try:
            rows.append(test())
        except Exception as exc:
            rows.append(
                {
                    "name": test.__name__,
                    "checks": {"harness_exception": False},
                    "passed": False,
                    "observed": {"exception": repr(exc)},
                }
            )

    passed = sum(row["passed"] for row in rows)
    result = {
        "validator": "Gate Kernel 2.0",
        "case_count": len(rows),
        "passed_cases": passed,
        "failed_cases": len(rows) - passed,
        "all_expected_checks_pass": passed == len(rows),
        "cases": rows,
        "coverage": {
            "strict_domain": True,
            "evidence_cone_marks": True,
            "typed_bridges": True,
            "operation_specific_witnesses": True,
            "canonical_hashes": True,
            "strict_registry_load": True,
            "provisional_propagation": True,
            "conditional_debt_propagation": True,
            "assembly": True,
            "domain_extremum": True,
            "band_provenance": True,
            "second_domain": True,
        },
    }
    output = BASE / "gate_kernel_v2_0_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "validator": result["validator"],
        "case_count": result["case_count"],
        "passed_cases": result["passed_cases"],
        "failed_cases": result["failed_cases"],
        "all_expected_checks_pass": result["all_expected_checks_pass"],
        "output": str(output),
    }, indent=2))
    if not result["all_expected_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
