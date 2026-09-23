#!/usr/bin/env python3
"""
q0_llm_verifier_v5.py

Executable implementation of Gate Framework Master v1.1.

v5 is a strict extension of q0_llm_verifier_v4:
  * v4 still supplies DOMAIN, ENDPOINT, POLARITY, MEASURE, MARK,
    COMPOSITION, RUNG, PRECEDENCE, CORE CLOSURE, MODEL and the inherited
    graph/Merkle machinery.
  * v3 remains the numerical and risk-ledger dependency of v4.
  * v5 adds:
      SCHEMA preflight
      establishment/deployment regime checks (F-1)
      ASSEMBLY
      DOMAIN-INFIMUM
      BAND-PROVENANCE
      COMMON-MODE
      HEURISTIC-BRIDGE
      constitutional warrant propagation
      active Q0 gate states
      first-class coverage/selection risk fields
      canonical v5 hashes

The verifier validates proof contracts. It does not turn a measured or
program-grade mathematical input into an external referee-grade theorem.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from decimal import Decimal, getcontext, InvalidOperation
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import q0_llm_verifier_v4 as v4
from q0_llm_verifier_v3 import (
    Grade,
    CoverageCertificate,
    SelectionAwareRiskLedger,
    SupportEdge,
    widest_path_grounding,
    anchored_rkhs_mismatch,
    anisotropic_sard_tube_bound,
    effective_transverse_rank,
    coverage_limited_rank_requirement,
)

getcontext().prec = 80


class GateVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PROVISIONAL = "PROVISIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Warrant(str, Enum):
    PROVEN = "PROVEN"
    PROVEN_MODULO = "PROVEN_MODULO"
    CERTIFIED = "CERTIFIED"
    DERIVED = "DERIVED"
    LITERATURE_SUPPORTED = "LITERATURE_SUPPORTED"
    MEASURED = "MEASURED"
    PLAUSIBLE = "PLAUSIBLE"
    CONJECTURE = "CONJECTURE"


WARRANT_RANK: Mapping[Warrant, int] = {
    Warrant.CONJECTURE: 1,
    Warrant.PLAUSIBLE: 2,
    Warrant.LITERATURE_SUPPORTED: 3,
    Warrant.MEASURED: 3,
    Warrant.DERIVED: 4,
    Warrant.CERTIFIED: 5,
    Warrant.PROVEN_MODULO: 6,
    Warrant.PROVEN: 7,
}


class AssumptionRole(str, Enum):
    NONE = "NONE"
    DEFINITION = "DEFINITION"
    PRIMITIVE_HYPOTHESIS = "PRIMITIVE_HYPOTHESIS"
    POLICY_THRESHOLD = "POLICY_THRESHOLD"


class ReasoningMode(str, Enum):
    DEDUCTIVE = "DEDUCTIVE"
    EMPIRICAL = "EMPIRICAL"
    ANALOGICAL = "ANALOGICAL"
    HEURISTIC = "HEURISTIC"
    NORMATIVE = "NORMATIVE"


class VerificationMode(str, Enum):
    EXACT_SINGLE = "EXACT_SINGLE"
    AGREEMENT = "AGREEMENT"
    ADVERSARIAL_REIMPLEMENTATION = "ADVERSARIAL_REIMPLEMENTATION"
    EXTERNAL_CONTRADICTION = "EXTERNAL_CONTRADICTION"
    NONE = "NONE"


class InstrumentRelation(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    EXTERNAL_CROSSCHECK = "EXTERNAL_CROSSCHECK"
    SHARED_COMPONENTS = "SHARED_COMPONENTS"
    UNKNOWN = "UNKNOWN"


class AssemblyRounding(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    EXACT = "EXACT"
    OUTWARD_INTERVAL = "OUTWARD_INTERVAL"


class ExtremumKind(str, Enum):
    INFIMUM = "INFIMUM"
    SUPREMUM = "SUPREMUM"


class ExtremumCertificateType(str, Enum):
    ANALYTIC = "ANALYTIC"
    MONOTONICITY = "MONOTONICITY"
    INTERVAL = "INTERVAL"
    TAYLOR_MODEL = "TAYLOR_MODEL"
    GRID_WITH_MODULUS = "GRID_WITH_MODULUS"
    EXACT_ENUMERATION = "EXACT_ENUMERATION"
    ASYMPTOTIC_MONOTONICITY = "ASYMPTOTIC_MONOTONICITY"
    ANALYTIC_UNIFORM_BOUND = "ANALYTIC_UNIFORM_BOUND"
    RUNG_SAMPLE = "RUNG_SAMPLE"
    NONE = "NONE"


class BandKind(str, Enum):
    NONE = "NONE"
    DETERMINISTIC = "DETERMINISTIC"
    INTERVAL = "INTERVAL"
    STATISTICAL = "STATISTICAL"
    BOOTSTRAP = "BOOTSTRAP"
    MONTE_CARLO = "MONTE_CARLO"
    RANDOMIZED_NUMERICAL = "RANDOMIZED_NUMERICAL"
    FORECAST = "FORECAST"


class ActiveGateStatus(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    KILLED = "KILLED"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    NOT_CLAIMED = "NOT_CLAIMED"


class BridgeKind(str, Enum):
    MODEL = "MODEL"
    MEASURE = "MEASURE"
    DOMAIN = "DOMAIN"
    MARK = "MARK"
    SCALE = "SCALE"
    CONDITIONING = "CONDITIONING"
    HEURISTIC = "HEURISTIC"


@dataclass(frozen=True)
class Regime:
    model_id: Optional[str] = None
    measure: v4.Measure = v4.Measure.NONE
    parameter_domain: Optional[v4.ParameterDomain] = None
    spatial_region: Optional[str] = None
    scale: Optional[str] = None
    conditioning_depth: Optional[str] = None
    marks: frozenset[str] = frozenset()
    deployment_mode: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "measure": self.measure.value,
            "parameter_domain": (
                None if self.parameter_domain is None
                else self.parameter_domain.as_dict()
            ),
            "spatial_region": self.spatial_region,
            "scale": self.scale,
            "conditioning_depth": self.conditioning_depth,
            "marks": sorted(self.marks),
            "deployment_mode": self.deployment_mode,
        }


@dataclass(frozen=True)
class TransferBridge:
    bridge_id: str
    kinds: frozenset[BridgeKind]
    source_regime: Regime
    target_regime: Regime
    warrant: Warrant
    scope_statement: str
    source_hash: str

    def as_dict(self) -> dict:
        return {
            "bridge_id": self.bridge_id,
            "kinds": sorted(k.value for k in self.kinds),
            "source_regime": self.source_regime.as_dict(),
            "target_regime": self.target_regime.as_dict(),
            "warrant": self.warrant.value,
            "scope_statement": self.scope_statement,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class AssemblySpec:
    expression: str
    inputs: Tuple[Tuple[str, str], ...]
    required_terms: Tuple[str, ...]
    displayed_value: str
    claim_direction: v4.Direction
    rounding: AssemblyRounding
    precision_digits: int = 50
    source_hashes: Tuple[str, ...] = ()
    monotonicity: Tuple[Tuple[str, int], ...] = ()
    residual_terms: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "expression": self.expression,
            "inputs": dict(self.inputs),
            "required_terms": list(self.required_terms),
            "displayed_value": self.displayed_value,
            "claim_direction": self.claim_direction.value,
            "rounding": self.rounding.value,
            "precision_digits": self.precision_digits,
            "source_hashes": list(self.source_hashes),
            "monotonicity": dict(self.monotonicity),
            "residual_terms": list(self.residual_terms),
        }


@dataclass(frozen=True)
class DomainExtremumCertificate:
    certificate_id: str
    certificate_type: ExtremumCertificateType
    domain: v4.ParameterDomain
    extremum_kind: ExtremumKind
    extremum_value: str
    proof_hash: str
    description: str
    sampled_points: Tuple[Tuple[float, float], ...] = ()
    monotonicity_direction: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type.value,
            "domain": self.domain.as_dict(),
            "extremum_kind": self.extremum_kind.value,
            "extremum_value": self.extremum_value,
            "proof_hash": self.proof_hash,
            "description": self.description,
            "sampled_points": list(self.sampled_points),
            "monotonicity_direction": self.monotonicity_direction,
        }


@dataclass(frozen=True)
class UncertaintyProvenance:
    provenance_id: str
    kind: BandKind
    side: v4.UncertaintySide
    construction: str
    source_hash: str
    coverage_domain: Optional[v4.ParameterDomain] = None
    confidence_level: Optional[float] = None
    sample_size: Optional[int] = None
    multiplicity_scope: Optional[str] = None
    seed_or_algorithm_state: Optional[str] = None
    deterministic_tolerance: Optional[str] = None
    consumed_as: Optional[v4.UncertaintySide] = None
    forecast_only: bool = False

    def as_dict(self) -> dict:
        return {
            "provenance_id": self.provenance_id,
            "kind": self.kind.value,
            "side": self.side.value,
            "construction": self.construction,
            "source_hash": self.source_hash,
            "coverage_domain": (
                None if self.coverage_domain is None
                else self.coverage_domain.as_dict()
            ),
            "confidence_level": self.confidence_level,
            "sample_size": self.sample_size,
            "multiplicity_scope": self.multiplicity_scope,
            "seed_or_algorithm_state": self.seed_or_algorithm_state,
            "deterministic_tolerance": self.deterministic_tolerance,
            "consumed_as": (
                None if self.consumed_as is None else self.consumed_as.value
            ),
            "forecast_only": self.forecast_only,
        }


@dataclass(frozen=True)
class CommonModeCertificate:
    certificate_id: str
    relation: InstrumentRelation
    instrument_ids: Tuple[str, ...]
    error_channel_under_test: str
    shared_components: Tuple[str, ...] = ()
    disjoint_error_argument: Optional[str] = None
    external_crosscheck_id: Optional[str] = None
    source_hash: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "certificate_id": self.certificate_id,
            "relation": self.relation.value,
            "instrument_ids": list(self.instrument_ids),
            "error_channel_under_test": self.error_channel_under_test,
            "shared_components": list(self.shared_components),
            "disjoint_error_argument": self.disjoint_error_argument,
            "external_crosscheck_id": self.external_crosscheck_id,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class HeuristicBridge:
    bridge_id: str
    source_domain: str
    target_domain: str
    scope_statement: str
    proof_hash: str
    falsifier: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CoverageSpec:
    universe_id: str
    chart_ids: Tuple[str, ...]
    coverage_certificate_id: str
    uncovered_mass: float
    target_total_risk: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRiskSpec:
    coverage_failure: float
    baseline_hazard_rate: float
    expected_selected_measure: float
    selection_amplification: float
    q_sard_tube_risk: float
    decoder_boundary_risk: float = 0.0
    approximation_risk: float = 0.0
    calibration_shift_risk: float = 0.0

    def ledger(self) -> SelectionAwareRiskLedger:
        return SelectionAwareRiskLedger(**asdict(self))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActiveGate:
    gate_id: str
    status: ActiveGateStatus
    evidence_id: Optional[str] = None
    blocks_promotion: bool = True

    def as_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "evidence_id": self.evidence_id,
            "blocks_promotion": self.blocks_promotion,
        }


@dataclass(frozen=True)
class ProofObjectV5:
    base: v4.SemanticClaim
    warrant: Warrant = Warrant.PLAUSIBLE
    assumption_role: AssumptionRole = AssumptionRole.NONE
    reasoning_mode: ReasoningMode = ReasoningMode.DEDUCTIVE
    load_bearing: bool = True
    establishment_regime: Optional[Regime] = None
    deployment_regime: Optional[Regime] = None
    transfer_bridges: Tuple[TransferBridge, ...] = ()
    assembly_spec: Optional[AssemblySpec] = None
    requires_domain_extremum: bool = False
    domain_extremum_certificate: Optional[DomainExtremumCertificate] = None
    uncertainty_provenance: Optional[UncertaintyProvenance] = None
    verification_mode: VerificationMode = VerificationMode.NONE
    common_mode_certificate: Optional[CommonModeCertificate] = None
    heuristic_bridge: Optional[HeuristicBridge] = None
    coverage_spec: Optional[CoverageSpec] = None
    selection_risk_spec: Optional[SelectionRiskSpec] = None
    active_gates: Tuple[ActiveGate, ...] = ()
    conditional_on: Tuple[str, ...] = ()
    weakest_link_acknowledged: bool = False
    schema_version: str = "GF-1.1"

    def canonical_payload(self) -> dict:
        return {
            "base_digest": self.base.digest(),
            "warrant": self.warrant.value,
            "assumption_role": self.assumption_role.value,
            "reasoning_mode": self.reasoning_mode.value,
            "load_bearing": self.load_bearing,
            "establishment_regime": (
                None if self.establishment_regime is None
                else self.establishment_regime.as_dict()
            ),
            "deployment_regime": (
                None if self.deployment_regime is None
                else self.deployment_regime.as_dict()
            ),
            "transfer_bridges": [b.as_dict() for b in self.transfer_bridges],
            "assembly_spec": (
                None if self.assembly_spec is None
                else self.assembly_spec.as_dict()
            ),
            "requires_domain_extremum": self.requires_domain_extremum,
            "domain_extremum_certificate": (
                None if self.domain_extremum_certificate is None
                else self.domain_extremum_certificate.as_dict()
            ),
            "uncertainty_provenance": (
                None if self.uncertainty_provenance is None
                else self.uncertainty_provenance.as_dict()
            ),
            "verification_mode": self.verification_mode.value,
            "common_mode_certificate": (
                None if self.common_mode_certificate is None
                else self.common_mode_certificate.as_dict()
            ),
            "heuristic_bridge": (
                None if self.heuristic_bridge is None
                else self.heuristic_bridge.as_dict()
            ),
            "coverage_spec": (
                None if self.coverage_spec is None
                else self.coverage_spec.as_dict()
            ),
            "selection_risk_spec": (
                None if self.selection_risk_spec is None
                else self.selection_risk_spec.as_dict()
            ),
            "active_gates": [g.as_dict() for g in self.active_gates],
            "conditional_on": list(self.conditional_on),
            "weakest_link_acknowledged": self.weakest_link_acknowledged,
            "schema_version": self.schema_version,
        }

    def local_hash(self) -> str:
        return sha256(
            json.dumps(
                self.canonical_payload(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class GateResult:
    gate: str
    claim_id: str
    verdict: GateVerdict
    message: str
    evidence_id: Optional[str] = None
    grade_cap: Optional[Warrant] = None

    def as_dict(self) -> dict:
        return {
            "gate": self.gate,
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "message": self.message,
            "evidence_id": self.evidence_id,
            "grade_cap": None if self.grade_cap is None else self.grade_cap.value,
        }


@dataclass
class V5ValidationReport:
    roots: List[str]
    reachable: List[str]
    base_report: dict
    gate_results: List[GateResult]
    root_hashes: Dict[str, str]
    valid: bool
    promotable: bool
    gate_counts: Dict[str, Dict[str, int]]

    def as_dict(self) -> dict:
        return {
            "roots": self.roots,
            "reachable": self.reachable,
            "base_report": self.base_report,
            "gate_results": [r.as_dict() for r in self.gate_results],
            "root_hashes": self.root_hashes,
            "valid": self.valid,
            "promotable": self.promotable,
            "gate_counts": self.gate_counts,
        }


# ---------------------------------------------------------------------------
# Safe Decimal expression evaluator for ASSEMBLY
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** int(b),
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _eval_decimal_ast(node: ast.AST, values: Mapping[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_decimal_ast(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise KeyError(f"unknown assembly input {node.id!r}")
        return values[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, str)):
            return Decimal(str(node.value))
        if isinstance(node.value, float):
            # Convert source representation rather than binary payload.
            return Decimal(repr(node.value))
        raise TypeError("unsupported constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_decimal_ast(node.left, values)
        right = _eval_decimal_ast(node.right, values)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](
            _eval_decimal_ast(node.operand, values)
        )
    raise TypeError(f"unsupported assembly expression node: {ast.dump(node)}")


def evaluate_assembly(spec: AssemblySpec) -> Decimal:
    values = {name: Decimal(value) for name, value in spec.inputs}
    tree = ast.parse(spec.expression, mode="eval")
    return _eval_decimal_ast(tree, values)


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------

class GateVerifierV5:
    def __init__(self, objects: Iterable[ProofObjectV5] = ()) -> None:
        self.objects: Dict[str, ProofObjectV5] = {}
        for obj in objects:
            self.add(obj)

    def add(self, obj: ProofObjectV5) -> None:
        claim_id = obj.base.claim_id
        if claim_id in self.objects:
            raise ValueError(f"duplicate proof object {claim_id}")
        self.objects[claim_id] = obj

    def _extra_refs(self, obj: ProofObjectV5) -> List[str]:
        refs: List[str] = []
        if obj.heuristic_bridge is not None:
            refs.append(obj.heuristic_bridge.bridge_id)
        refs.extend(obj.conditional_on)
        return refs

    def reachable(self, roots: Sequence[str]) -> Set[str]:
        reached: Set[str] = set()
        stack = list(roots)
        while stack:
            claim_id = stack.pop()
            if claim_id in reached:
                continue
            reached.add(claim_id)
            obj = self.objects.get(claim_id)
            if obj is not None:
                stack.extend(obj.base.dependencies)
                stack.extend(self._extra_refs(obj))
        return reached

    def _v4_graph(self) -> v4.SemanticProofGraph:
        return v4.SemanticProofGraph(obj.base for obj in self.objects.values())

    def _find_bridge(
        self,
        obj: ProofObjectV5,
        kind: BridgeKind,
    ) -> Optional[TransferBridge]:
        for bridge in obj.transfer_bridges:
            if kind in bridge.kinds:
                return bridge
        return None

    def _schema_results(self, obj: ProofObjectV5) -> List[GateResult]:
        results: List[GateResult] = []
        claim = obj.base
        missing: List[str] = []
        if not claim.claim_id.strip():
            missing.append("claim_id")
        if not claim.statement.strip():
            missing.append("atomic_statement")
        if obj.load_bearing and not claim.source_ids:
            missing.append("source_ids")
        if obj.load_bearing and obj.establishment_regime is None:
            missing.append("establishment_regime")
        if obj.load_bearing and obj.deployment_regime is None:
            missing.append("deployment_regime")
        if claim.measured_value is not None and not claim.scale_tag:
            missing.append("scale_tag")
        if obj.assembly_spec is not None:
            if not obj.assembly_spec.expression:
                missing.append("assembly.expression")
            if not obj.assembly_spec.required_terms:
                missing.append("assembly.required_terms")
        if obj.requires_domain_extremum and obj.domain_extremum_certificate is None:
            missing.append("domain_extremum_certificate")
        if (
            claim.uncertainty_side != v4.UncertaintySide.NONE
            and obj.load_bearing
            and obj.uncertainty_provenance is None
        ):
            missing.append("uncertainty_provenance")
        if missing:
            results.append(
                GateResult(
                    "SCHEMA",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "missing required fields: " + ", ".join(missing),
                )
            )
        else:
            results.append(
                GateResult(
                    "SCHEMA",
                    claim.claim_id,
                    GateVerdict.PASS,
                    "required proof-object fields are present",
                )
            )
        return results

    def _regime_results(self, obj: ProofObjectV5) -> List[GateResult]:
        results: List[GateResult] = []
        est = obj.establishment_regime
        dep = obj.deployment_regime
        if est is None or dep is None:
            return results

        # MODEL
        if est.model_id != dep.model_id:
            bridge = self._find_bridge(obj, BridgeKind.MODEL)
            results.append(
                GateResult(
                    "MODEL",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "model mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        f"established model {est.model_id!r} != deployment "
                        f"model {dep.model_id!r}"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )

        # MEASURE
        if est.measure != dep.measure:
            bridge = self._find_bridge(obj, BridgeKind.MEASURE)
            results.append(
                GateResult(
                    "MEASURE",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "measure mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        f"established measure {est.measure.value} != "
                        f"deployment measure {dep.measure.value}"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )

        # DOMAIN
        if (
            est.parameter_domain is not None
            and dep.parameter_domain is not None
            and not est.parameter_domain.contains(dep.parameter_domain)
        ):
            bridge = self._find_bridge(obj, BridgeKind.DOMAIN)
            results.append(
                GateResult(
                    "DOMAIN",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "domain mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        "establishment parameter domain does not contain "
                        "deployment domain"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )

        # MARK
        missing_marks = set(dep.marks) - set(est.marks)
        if missing_marks:
            bridge = self._find_bridge(obj, BridgeKind.MARK)
            results.append(
                GateResult(
                    "MARK",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "mark mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        "deployment requires marks absent from establishment: "
                        + ", ".join(sorted(missing_marks))
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )

        # Scale and conditioning mismatches are reported under RUNG/MEASURE.
        if est.scale != dep.scale:
            bridge = self._find_bridge(obj, BridgeKind.SCALE)
            results.append(
                GateResult(
                    "RUNG",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "scale mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        f"established scale {est.scale!r} != deployment "
                        f"scale {dep.scale!r}"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )
        if est.conditioning_depth != dep.conditioning_depth:
            bridge = self._find_bridge(obj, BridgeKind.CONDITIONING)
            results.append(
                GateResult(
                    "MEASURE",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "conditioning-depth mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        "conditioning depth changed without a bridge"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )
        if (
            est.spatial_region is not None
            and dep.spatial_region is not None
            and est.spatial_region != dep.spatial_region
        ):
            bridge = self._find_bridge(obj, BridgeKind.DOMAIN)
            results.append(
                GateResult(
                    "DOMAIN",
                    obj.base.claim_id,
                    GateVerdict.PASS if bridge else GateVerdict.FAIL,
                    (
                        "spatial-region mismatch bridged by " + bridge.bridge_id
                        if bridge else
                        f"established region {est.spatial_region!r} != "
                        f"deployment region {dep.spatial_region!r}"
                    ),
                    None if bridge is None else bridge.bridge_id,
                )
            )
        return results

    def _assembly_result(self, obj: ProofObjectV5) -> GateResult:
        spec = obj.assembly_spec
        if spec is None:
            return GateResult(
                "ASSEMBLY",
                obj.base.claim_id,
                GateVerdict.NOT_APPLICABLE,
                "claim has no numerical assembly",
            )
        try:
            names = {name for name, _ in spec.inputs}
            missing_terms = set(spec.required_terms) - names
            if missing_terms:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "required assembly terms missing: "
                    + ", ".join(sorted(missing_terms)),
                )
            missing_residuals = set(spec.residual_terms) - names
            if missing_residuals:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "named residual terms are absent from the arithmetic "
                    "expression inputs: "
                    + ", ".join(sorted(missing_residuals)),
                )
            expression_names = {
                node.id
                for node in ast.walk(ast.parse(spec.expression, mode="eval"))
                if isinstance(node, ast.Name)
            }
            unused_required = (
                set(spec.required_terms) | set(spec.residual_terms)
            ) - expression_names
            if unused_required:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "required/residual inputs are not consumed by the "
                    "assembly expression: "
                    + ", ".join(sorted(unused_required)),
                )
            computed = evaluate_assembly(spec)
            displayed = Decimal(spec.displayed_value)
        except (ValueError, TypeError, KeyError, InvalidOperation, ZeroDivisionError) as exc:
            return GateResult(
                "ASSEMBLY",
                obj.base.claim_id,
                GateVerdict.FAIL,
                f"assembly evaluation failed: {exc}",
            )

        direction = spec.claim_direction
        if direction == v4.Direction.UPPER:
            if spec.rounding not in {
                AssemblyRounding.UP,
                AssemblyRounding.OUTWARD_INTERVAL,
                AssemblyRounding.EXACT,
            }:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "upper assembly does not declare upward/outward rounding",
                )
            if displayed < computed:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    f"displayed upper {displayed} < recomputed {computed}",
                )
        elif direction == v4.Direction.LOWER:
            if spec.rounding not in {
                AssemblyRounding.DOWN,
                AssemblyRounding.OUTWARD_INTERVAL,
                AssemblyRounding.EXACT,
            }:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "lower assembly does not declare downward/outward rounding",
                )
            if displayed > computed:
                return GateResult(
                    "ASSEMBLY",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    f"displayed lower {displayed} > recomputed {computed}",
                )
        elif direction == v4.Direction.EQUALITY and displayed != computed:
            return GateResult(
                "ASSEMBLY",
                obj.base.claim_id,
                GateVerdict.FAIL,
                f"displayed equality {displayed} != recomputed {computed}",
            )

        return GateResult(
            "ASSEMBLY",
            obj.base.claim_id,
            GateVerdict.PASS,
            f"displayed {displayed} conservatively encloses recomputed {computed}",
        )

    def _domain_infimum_result(self, obj: ProofObjectV5) -> GateResult:
        if not obj.requires_domain_extremum:
            return GateResult(
                "DOMAIN-INFIMUM",
                obj.base.claim_id,
                GateVerdict.NOT_APPLICABLE,
                "claim does not require a numerical full-domain extremum",
            )
        cert = obj.domain_extremum_certificate
        claim = obj.base
        if cert is None:
            return GateResult(
                "DOMAIN-INFIMUM",
                claim.claim_id,
                GateVerdict.FAIL,
                "missing full-domain extremum certificate",
            )
        if cert.certificate_type in {
            ExtremumCertificateType.RUNG_SAMPLE,
            ExtremumCertificateType.NONE,
        }:
            return GateResult(
                "DOMAIN-INFIMUM",
                claim.claim_id,
                GateVerdict.FAIL,
                f"{cert.certificate_type.value} is not a full-domain certificate",
                cert.certificate_id,
            )
        if claim.domain is not None and not cert.domain.contains(claim.domain):
            return GateResult(
                "DOMAIN-INFIMUM",
                claim.claim_id,
                GateVerdict.FAIL,
                "extremum certificate does not cover the claim domain",
                cert.certificate_id,
            )
        if claim.bound_coefficient is None:
            return GateResult(
                "DOMAIN-INFIMUM",
                claim.claim_id,
                GateVerdict.FAIL,
                "numeric uniform claim has no bound coefficient",
                cert.certificate_id,
            )
        extremum = Decimal(cert.extremum_value)
        bound = Decimal(str(claim.bound_coefficient))
        if claim.direction == v4.Direction.UPPER:
            if cert.extremum_kind != ExtremumKind.SUPREMUM:
                return GateResult(
                    "DOMAIN-INFIMUM",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "upper claim requires a supremum certificate",
                    cert.certificate_id,
                )
            if extremum > bound:
                return GateResult(
                    "DOMAIN-INFIMUM",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    f"certified supremum {extremum} exceeds bound {bound}",
                    cert.certificate_id,
                )
        elif claim.direction == v4.Direction.LOWER:
            if cert.extremum_kind != ExtremumKind.INFIMUM:
                return GateResult(
                    "DOMAIN-INFIMUM",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "lower claim requires an infimum certificate",
                    cert.certificate_id,
                )
            if extremum < bound:
                return GateResult(
                    "DOMAIN-INFIMUM",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    f"certified infimum {extremum} lies below bound {bound}",
                    cert.certificate_id,
                )
        return GateResult(
            "DOMAIN-INFIMUM",
            claim.claim_id,
            GateVerdict.PASS,
            f"{cert.certificate_type.value} certificate covers full domain",
            cert.certificate_id,
        )

    def _band_result(self, obj: ProofObjectV5) -> GateResult:
        claim = obj.base
        provenance = obj.uncertainty_provenance
        if (
            claim.uncertainty_side == v4.UncertaintySide.NONE
            and provenance is None
        ):
            return GateResult(
                "BAND-PROVENANCE",
                claim.claim_id,
                GateVerdict.NOT_APPLICABLE,
                "claim carries no load-bearing uncertainty object",
            )
        if provenance is None:
            return GateResult(
                "BAND-PROVENANCE",
                claim.claim_id,
                GateVerdict.FAIL,
                "missing uncertainty provenance",
            )
        if not provenance.construction or not provenance.source_hash:
            return GateResult(
                "BAND-PROVENANCE",
                claim.claim_id,
                GateVerdict.FAIL,
                "uncertainty construction or source hash missing",
                provenance.provenance_id,
            )
        if (
            claim.domain is not None
            and provenance.coverage_domain is not None
            and not provenance.coverage_domain.contains(claim.domain)
        ):
            return GateResult(
                "BAND-PROVENANCE",
                claim.claim_id,
                GateVerdict.FAIL,
                "uncertainty provenance does not cover claim domain",
                provenance.provenance_id,
            )
        if provenance.kind in {
            BandKind.STATISTICAL,
            BandKind.BOOTSTRAP,
            BandKind.MONTE_CARLO,
        }:
            missing = []
            if provenance.confidence_level is None:
                missing.append("confidence_level")
            if provenance.sample_size is None:
                missing.append("sample_size")
            if not provenance.multiplicity_scope:
                missing.append("multiplicity_scope")
            if missing:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "missing statistical metadata: " + ", ".join(missing),
                    provenance.provenance_id,
                )
        if provenance.kind == BandKind.RANDOMIZED_NUMERICAL:
            if not provenance.seed_or_algorithm_state:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "randomized numerical output lacks seed/algorithm state",
                    provenance.provenance_id,
                )
            if provenance.deterministic_tolerance is None:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "randomized numerical output lacks declared tolerance",
                    provenance.provenance_id,
                )
        if provenance.kind in {BandKind.DETERMINISTIC, BandKind.INTERVAL}:
            if provenance.deterministic_tolerance is None:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "deterministic/interval band lacks tolerance",
                    provenance.provenance_id,
                )
        if provenance.kind == BandKind.FORECAST and obj.warrant not in {
            Warrant.PLAUSIBLE,
            Warrant.CONJECTURE,
        }:
            return GateResult(
                "BAND-PROVENANCE",
                claim.claim_id,
                GateVerdict.FAIL,
                "forecast band cannot support the declared warrant grade",
                provenance.provenance_id,
            )
        if claim.direction == v4.Direction.UPPER:
            consumed = provenance.consumed_as or provenance.side
            if consumed not in {
                v4.UncertaintySide.UPPER,
                v4.UncertaintySide.NONE,
            }:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "upper claim does not consume the upper side of uncertainty",
                    provenance.provenance_id,
                )
        if claim.direction == v4.Direction.LOWER:
            consumed = provenance.consumed_as or provenance.side
            if consumed not in {
                v4.UncertaintySide.LOWER,
                v4.UncertaintySide.NONE,
            }:
                return GateResult(
                    "BAND-PROVENANCE",
                    claim.claim_id,
                    GateVerdict.FAIL,
                    "lower claim does not consume the lower side of uncertainty",
                    provenance.provenance_id,
                )
        return GateResult(
            "BAND-PROVENANCE",
            claim.claim_id,
            GateVerdict.PASS,
            f"{provenance.kind.value} uncertainty metadata is complete",
            provenance.provenance_id,
        )

    def _common_mode_result(self, obj: ProofObjectV5) -> GateResult:
        if obj.verification_mode != VerificationMode.AGREEMENT:
            return GateResult(
                "COMMON-MODE",
                obj.base.claim_id,
                GateVerdict.NOT_APPLICABLE,
                "claim is not promoted from instrument agreement",
            )
        cert = obj.common_mode_certificate
        if cert is None:
            return GateResult(
                "COMMON-MODE",
                obj.base.claim_id,
                GateVerdict.PROVISIONAL,
                "agreement has no common-mode certificate",
                grade_cap=Warrant.DERIVED,
            )
        if cert.relation == InstrumentRelation.INDEPENDENT:
            if not cert.disjoint_error_argument or not cert.source_hash:
                return GateResult(
                    "COMMON-MODE",
                    obj.base.claim_id,
                    GateVerdict.PROVISIONAL,
                    "independence is asserted without an error-channel argument/hash",
                    cert.certificate_id,
                    Warrant.DERIVED,
                )
            return GateResult(
                "COMMON-MODE",
                obj.base.claim_id,
                GateVerdict.PASS,
                "instruments have a documented disjoint error channel",
                cert.certificate_id,
            )
        if cert.relation == InstrumentRelation.EXTERNAL_CROSSCHECK:
            if not cert.external_crosscheck_id:
                return GateResult(
                    "COMMON-MODE",
                    obj.base.claim_id,
                    GateVerdict.PROVISIONAL,
                    "external-crosscheck relation lacks crosscheck ID",
                    cert.certificate_id,
                    Warrant.DERIVED,
                )
            return GateResult(
                "COMMON-MODE",
                obj.base.claim_id,
                GateVerdict.PASS,
                "agreement is supported by an external contradiction test",
                cert.external_crosscheck_id,
            )
        return GateResult(
            "COMMON-MODE",
            obj.base.claim_id,
            GateVerdict.PROVISIONAL,
            "instrument agreement retains shared or unknown error channels",
            cert.certificate_id,
            Warrant.DERIVED,
        )

    def _heuristic_result(self, obj: ProofObjectV5) -> GateResult:
        if (
            obj.reasoning_mode
            not in {ReasoningMode.ANALOGICAL, ReasoningMode.HEURISTIC}
            or not obj.load_bearing
        ):
            return GateResult(
                "HEURISTIC-BRIDGE",
                obj.base.claim_id,
                GateVerdict.NOT_APPLICABLE,
                "claim is not a load-bearing heuristic/analogical transition",
            )
        bridge = obj.heuristic_bridge
        if bridge is None:
            if WARRANT_RANK[obj.warrant] > WARRANT_RANK[Warrant.PLAUSIBLE]:
                return GateResult(
                    "HEURISTIC-BRIDGE",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    "unbridged heuristic is load bearing above Plausible",
                    grade_cap=Warrant.PLAUSIBLE,
                )
            return GateResult(
                "HEURISTIC-BRIDGE",
                obj.base.claim_id,
                GateVerdict.PROVISIONAL,
                "unbridged heuristic is retained at Plausible/Conjecture grade",
                grade_cap=Warrant.PLAUSIBLE,
            )
        if not bridge.scope_statement or not bridge.proof_hash:
            return GateResult(
                "HEURISTIC-BRIDGE",
                obj.base.claim_id,
                GateVerdict.FAIL,
                "bridge lacks scope or proof hash",
                bridge.bridge_id,
            )
        if bridge.bridge_id not in self.objects:
            return GateResult(
                "HEURISTIC-BRIDGE",
                obj.base.claim_id,
                GateVerdict.FAIL,
                "bridge claim is absent from the proof graph",
                bridge.bridge_id,
            )
        return GateResult(
            "HEURISTIC-BRIDGE",
            obj.base.claim_id,
            GateVerdict.PASS,
            "load-bearing heuristic transition has an explicit bridge",
            bridge.bridge_id,
        )

    def _coverage_results(self, obj: ProofObjectV5) -> List[GateResult]:
        results: List[GateResult] = []
        if obj.coverage_spec is not None:
            spec = obj.coverage_spec
            if not spec.universe_id or not spec.chart_ids or not spec.coverage_certificate_id:
                results.append(
                    GateResult(
                        "COVERAGE",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        "coverage universe, charts, or certificate is missing",
                    )
                )
            elif not 0 <= spec.uncovered_mass <= 1:
                results.append(
                    GateResult(
                        "COVERAGE",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        "uncovered mass is outside [0,1]",
                        spec.coverage_certificate_id,
                    )
                )
            elif (
                spec.target_total_risk is not None
                and spec.target_total_risk < spec.uncovered_mass
            ):
                results.append(
                    GateResult(
                        "COVERAGE",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        "target risk lies below the uncovered-mass floor",
                        spec.coverage_certificate_id,
                    )
                )
            else:
                results.append(
                    GateResult(
                        "COVERAGE",
                        obj.base.claim_id,
                        GateVerdict.PASS,
                        "coverage universe and uncovered-mass certificate are explicit",
                        spec.coverage_certificate_id,
                    )
                )
        if obj.selection_risk_spec is not None:
            try:
                ledger = obj.selection_risk_spec.ledger()
                total = ledger.total_upper_bound()
                floor = ledger.exponential_limit_floor()
            except ValueError as exc:
                results.append(
                    GateResult(
                        "COVERAGE",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        f"selection-aware risk ledger invalid: {exc}",
                    )
                )
            else:
                if (
                    obj.coverage_spec is not None
                    and obj.coverage_spec.target_total_risk is not None
                    and obj.coverage_spec.target_total_risk < floor
                ):
                    results.append(
                        GateResult(
                            "COVERAGE",
                            obj.base.claim_id,
                            GateVerdict.FAIL,
                            f"target risk is below exponential-limit floor {floor}",
                        )
                    )
                else:
                    results.append(
                        GateResult(
                            "COVERAGE",
                            obj.base.claim_id,
                            GateVerdict.PASS,
                            f"selection-aware total={total}, rank-limit floor={floor}",
                        )
                    )
        if not results:
            results.append(
                GateResult(
                    "COVERAGE",
                    obj.base.claim_id,
                    GateVerdict.NOT_APPLICABLE,
                    "claim carries no coverage/risk assertion",
                )
            )
        return results

    def _active_gate_results(self, obj: ProofObjectV5) -> List[GateResult]:
        results: List[GateResult] = []
        for gate in obj.active_gates:
            if gate.status == ActiveGateStatus.CLOSED:
                results.append(
                    GateResult(
                        gate.gate_id,
                        obj.base.claim_id,
                        GateVerdict.PASS,
                        "active domain-specific gate is closed",
                        gate.evidence_id,
                    )
                )
            elif gate.status in {
                ActiveGateStatus.NOT_CLAIMED,
            }:
                results.append(
                    GateResult(
                        gate.gate_id,
                        obj.base.claim_id,
                        GateVerdict.NOT_APPLICABLE,
                        "claim explicitly does not consume this gate",
                        gate.evidence_id,
                    )
                )
            elif gate.blocks_promotion and obj.load_bearing:
                results.append(
                    GateResult(
                        gate.gate_id,
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        f"active gate status is {gate.status.value}",
                        gate.evidence_id,
                    )
                )
            else:
                results.append(
                    GateResult(
                        gate.gate_id,
                        obj.base.claim_id,
                        GateVerdict.PROVISIONAL,
                        f"active gate status is {gate.status.value}",
                        gate.evidence_id,
                        Warrant.PLAUSIBLE,
                    )
                )
        return results

    def _dependency_warrant_results(
        self,
        obj: ProofObjectV5,
    ) -> List[GateResult]:
        results: List[GateResult] = []
        parent_rank = WARRANT_RANK[obj.warrant]
        for dep_id in obj.base.dependencies:
            dep = self.objects.get(dep_id)
            if dep is None:
                continue
            if dep.base.status in {
                v4.Status.KILLED,
                v4.Status.SUPERSEDED,
            }:
                # v4 also catches this; keep explicit v5 result.
                results.append(
                    GateResult(
                        "PRECEDENCE",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        f"dependency {dep_id} is {dep.base.status.value}",
                        dep_id,
                    )
                )
                continue
            if dep.assumption_role == AssumptionRole.PRIMITIVE_HYPOTHESIS:
                if obj.warrant == Warrant.PROVEN:
                    results.append(
                        GateResult(
                            "DEPENDENCY-WARRANT",
                            obj.base.claim_id,
                            GateVerdict.FAIL,
                            "Proven claim depends on a primitive hypothesis; "
                            "use Proven-Modulo",
                            dep_id,
                        )
                    )
                elif dep_id not in obj.conditional_on:
                    results.append(
                        GateResult(
                            "DEPENDENCY-WARRANT",
                            obj.base.claim_id,
                            GateVerdict.FAIL,
                            "primitive hypothesis is not named in conditional_on",
                            dep_id,
                        )
                    )
                continue
            dep_rank = WARRANT_RANK[dep.warrant]
            if parent_rank > dep_rank:
                results.append(
                    GateResult(
                        "DEPENDENCY-WARRANT",
                        obj.base.claim_id,
                        GateVerdict.FAIL,
                        f"parent warrant {obj.warrant.value} exceeds dependency "
                        f"{dep_id} warrant {dep.warrant.value}",
                        dep_id,
                        dep.warrant,
                    )
                )
        if not results:
            results.append(
                GateResult(
                    "DEPENDENCY-WARRANT",
                    obj.base.claim_id,
                    GateVerdict.PASS,
                    "dependency warrants are admissible",
                )
            )
        return results

    def _grade_cap_results(
        self,
        obj: ProofObjectV5,
        claim_results: Sequence[GateResult],
    ) -> List[GateResult]:
        caps = [
            result.grade_cap
            for result in claim_results
            if result.verdict == GateVerdict.PROVISIONAL
            and result.grade_cap is not None
        ]
        if not caps:
            return []
        cap = min(caps, key=lambda x: WARRANT_RANK[x])
        if WARRANT_RANK[obj.warrant] > WARRANT_RANK[cap]:
            return [
                GateResult(
                    "GRADE-CAP",
                    obj.base.claim_id,
                    GateVerdict.FAIL,
                    f"declared warrant {obj.warrant.value} exceeds provisional "
                    f"cap {cap.value}",
                    grade_cap=cap,
                )
            ]
        return [
            GateResult(
                "GRADE-CAP",
                obj.base.claim_id,
                GateVerdict.PASS,
                f"declared warrant respects provisional cap {cap.value}",
                grade_cap=cap,
            )
        ]

    def _base_results(
        self,
        base_report: v4.SemanticValidationReport,
    ) -> List[GateResult]:
        results: List[GateResult] = []
        for issue in base_report.issues:
            results.append(
                GateResult(
                    issue.gate,
                    issue.claim_id,
                    GateVerdict.FAIL,
                    issue.message,
                    issue.dependency_id,
                )
            )
        return results

    def merkle_hash(self, root: str) -> str:
        memo: Dict[str, str] = {}
        active: Set[str] = set()

        def visit(claim_id: str) -> str:
            if claim_id in memo:
                return memo[claim_id]
            if claim_id in active:
                raise ValueError(f"cycle at {claim_id}")
            obj = self.objects[claim_id]
            active.add(claim_id)
            child_ids = list(obj.base.dependencies) + self._extra_refs(obj)
            child_hashes = sorted(visit(child) for child in child_ids)
            active.remove(claim_id)
            payload = obj.local_hash() + "|" + "|".join(child_hashes)
            memo[claim_id] = sha256(payload.encode()).hexdigest()
            return memo[claim_id]

        return visit(root)

    def validate(self, roots: Sequence[str]) -> V5ValidationReport:
        roots = list(roots)
        reached = self.reachable(roots)
        base_graph = self._v4_graph()
        base_report = base_graph.validate(roots, require_closed_core=True)
        results: List[GateResult] = self._base_results(base_report)

        for claim_id in sorted(reached):
            obj = self.objects.get(claim_id)
            if obj is None:
                results.append(
                    GateResult(
                        "SCHEMA",
                        claim_id,
                        GateVerdict.FAIL,
                        "reachable proof object is missing",
                    )
                )
                continue
            claim_results: List[GateResult] = []
            claim_results.extend(self._schema_results(obj))
            claim_results.extend(self._regime_results(obj))
            claim_results.append(self._assembly_result(obj))
            claim_results.append(self._domain_infimum_result(obj))
            claim_results.append(self._band_result(obj))
            claim_results.append(self._common_mode_result(obj))
            claim_results.append(self._heuristic_result(obj))
            claim_results.extend(self._coverage_results(obj))
            claim_results.extend(self._active_gate_results(obj))
            claim_results.extend(self._dependency_warrant_results(obj))
            claim_results.extend(self._grade_cap_results(obj, claim_results))
            results.extend(claim_results)

        fail = any(r.verdict == GateVerdict.FAIL for r in results)
        provisional = any(
            r.verdict == GateVerdict.PROVISIONAL for r in results
        )
        root_hashes: Dict[str, str] = {}
        if not fail:
            for root in roots:
                root_hashes[root] = self.merkle_hash(root)

        counts: Dict[str, Dict[str, int]] = {}
        for result in results:
            bucket = counts.setdefault(result.gate, {})
            bucket[result.verdict.value] = bucket.get(result.verdict.value, 0) + 1

        return V5ValidationReport(
            roots=roots,
            reachable=sorted(reached),
            base_report=base_report.as_dict(),
            gate_results=results,
            root_hashes=root_hashes,
            valid=not fail,
            promotable=not fail and not provisional,
            gate_counts=counts,
        )


# ---------------------------------------------------------------------------
# Contract constructors used by the independent validation battery
# ---------------------------------------------------------------------------

def _domain(
    r_min: float = 0.0,
    r_max: float = 0.025,
    *,
    name: str = "r",
) -> v4.ParameterDomain:
    return v4.ParameterDomain(
        r_min,
        r_max,
        True,
        (("parameter", name),),
    )


def _regime(
    domain: v4.ParameterDomain,
    *,
    model: str = "BF_TORUS_EXACT_L24",
    measure: v4.Measure = v4.Measure.PAIR_PALM,
    region: str = "global torus",
    scale: str = "r^3 coefficient",
    conditioning: str = "typed maximum-saddle pair",
    marks: Iterable[str] = ("position", "height", "critical_type"),
) -> Regime:
    return Regime(
        model_id=model,
        measure=measure,
        parameter_domain=domain,
        spatial_region=region,
        scale=scale,
        conditioning_depth=conditioning,
        marks=frozenset(marks),
        deployment_mode="arbitrary-precision contract",
    )


def current_q0_upper_contract() -> Tuple[GateVerifierV5, List[str]]:
    domain = _domain()
    regime = _regime(domain)
    model_claim = v4.SemanticClaim(
        "MODEL_EXACT",
        "Exact normalized periodized Bargmann-Fock field on T_24^2.",
        v4.Status.DERIVED_EXACT,
        direction=v4.Direction.EQUALITY,
        quantifier=v4.Quantifier.EXACT_IDENTITY,
        domain=domain,
        measure=v4.Measure.NONE,
        grade=Grade.CERTIFIED,
        model_id="BF_TORUS_EXACT_L24",
        source_ids=("C090-PERIODIZED",),
        program_core=True,
    )
    ub_claim = v4.SemanticClaim(
        "UB_G_435",
        "0 <= 1-q(r,1.2) <= 4.35 r^3 for 0<r<=0.025, L=24.",
        v4.Status.PROGRAM_GRADE_CLOSED,
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        dependencies=("MODEL_EXACT",),
        domain=domain,
        measure=v4.Measure.PAIR_PALM,
        grade=Grade.CERTIFIED,
        bound_coefficient=4.35,
        model_id="BF_TORUS_EXACT_L24",
        source_ids=("C094-UBG-CORRECTION",),
        program_core=True,
    )
    q0_claim = v4.SemanticClaim(
        "Q0_LIMIT_C094",
        "q(r,1.2) tends to one as r tends to zero at fixed L=24.",
        v4.Status.PROGRAM_GRADE_CLOSED,
        direction=v4.Direction.EQUALITY,
        quantifier=v4.Quantifier.ASYMPTOTIC,
        dependencies=("UB_G_435",),
        domain=domain,
        measure=v4.Measure.PAIR_PALM,
        grade=Grade.CERTIFIED,
        model_id="BF_TORUS_EXACT_L24",
        source_ids=("C094-UBG-CORRECTION",),
        program_core=True,
    )
    objects = [
        ProofObjectV5(
            model_claim,
            warrant=Warrant.CERTIFIED,
            establishment_regime=Regime(
                model_id="BF_TORUS_EXACT_L24",
                measure=v4.Measure.NONE,
                parameter_domain=domain,
                spatial_region="T_24^2",
                scale="exact covariance",
                conditioning_depth="none",
                marks=frozenset(),
                deployment_mode="exact",
            ),
            deployment_regime=Regime(
                model_id="BF_TORUS_EXACT_L24",
                measure=v4.Measure.NONE,
                parameter_domain=domain,
                spatial_region="T_24^2",
                scale="exact covariance",
                conditioning_depth="none",
                marks=frozenset(),
                deployment_mode="exact",
            ),
        ),
        ProofObjectV5(
            ub_claim,
            warrant=Warrant.CERTIFIED,
            establishment_regime=regime,
            deployment_regime=regime,
            assembly_spec=AssemblySpec(
                expression="(near + exterior) / typing",
                inputs=(
                    ("near", "0.657"),
                    (
                        "exterior",
                        "2.8185310984873743106223995278378565957688812716057680797457",
                    ),
                    ("typing", "0.80"),
                ),
                required_terms=("near", "exterior", "typing"),
                displayed_value="4.35",
                claim_direction=v4.Direction.UPPER,
                rounding=AssemblyRounding.UP,
                source_hashes=("C094-UBG-CORRECTION",),
                monotonicity=(
                    ("near", 1),
                    ("exterior", 1),
                    ("typing", -1),
                ),
                residual_terms=("gamma", "collar"),
            ),
            requires_domain_extremum=True,
            domain_extremum_certificate=DomainExtremumCertificate(
                "UB_G_UNIFORM",
                ExtremumCertificateType.ANALYTIC_UNIFORM_BOUND,
                domain,
                ExtremumKind.SUPREMUM,
                "4.344413873109217888277999409797320744711101589507210099682126",
                "C094-UBG-CORRECTION",
                "Uniform global count architecture on the certified radius box.",
            ),
            active_gates=(
                ActiveGate(
                    "TORUS-GLOBAL-PARTITION",
                    ActiveGateStatus.CLOSED,
                    "C091-MATRIX-TRANSFER",
                ),
                ActiveGate(
                    "UB_G_RESIDUAL_UNIFORM",
                    ActiveGateStatus.OPEN,
                    "C095-RESIDUAL-SOURCE-SEARCH",
                ),
            ),
        ),
        ProofObjectV5(
            q0_claim,
            warrant=Warrant.CERTIFIED,
            establishment_regime=regime,
            deployment_regime=regime,
        ),
    ]
    return GateVerifierV5(objects), ["UB_G_435", "Q0_LIMIT_C094"]


def matrix_perturbation_second_domain() -> Tuple[GateVerifierV5, List[str]]:
    domain = _domain(0.0, 0.8, name="rho")
    regime = Regime(
        model_id="FINITE_DIMENSIONAL_MATRIX_2_NORM",
        measure=v4.Measure.NONE,
        parameter_domain=domain,
        spatial_region="finite-dimensional complex matrices",
        scale="operator norm",
        conditioning_depth="none",
        marks=frozenset({"invertible_A", "perturbation_E"}),
        deployment_mode="exact theorem",
    )
    lemma = v4.SemanticClaim(
        "NEUMANN_SERIES_LEMMA",
        "If ||B||_2<1 then I+B is invertible and "
        "||(I+B)^(-1)||_2 <= 1/(1-||B||_2).",
        v4.Status.ESTABLISHED,
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        domain=domain,
        measure=v4.Measure.NONE,
        grade=Grade.ESTABLISHED,
        bound_coefficient=5.0,
        model_id="FINITE_DIMENSIONAL_MATRIX_2_NORM",
        source_ids=("NEUMANN-SERIES-EXACT",),
    )
    theorem = v4.SemanticClaim(
        "MATRIX_PERTURBATION_THEOREM",
        "If A is invertible and ||A^(-1)E||_2<=rho<=0.8 then "
        "||(A+E)^(-1)||_2 <= 5||A^(-1)||_2.",
        v4.Status.ESTABLISHED,
        direction=v4.Direction.UPPER,
        quantifier=v4.Quantifier.FOR_ALL,
        dependencies=("NEUMANN_SERIES_LEMMA",),
        domain=domain,
        measure=v4.Measure.NONE,
        grade=Grade.ESTABLISHED,
        bound_coefficient=5.0,
        model_id="FINITE_DIMENSIONAL_MATRIX_2_NORM",
        source_ids=("MATRIX-PERTURBATION-EXACT",),
    )
    lemma_obj = ProofObjectV5(
        lemma,
        warrant=Warrant.PROVEN,
        establishment_regime=regime,
        deployment_regime=regime,
        assembly_spec=AssemblySpec(
            expression="1 / (1-rho)",
            inputs=(("rho", "0.8"),),
            required_terms=("rho",),
            displayed_value="5",
            claim_direction=v4.Direction.UPPER,
            rounding=AssemblyRounding.EXACT,
            source_hashes=("NEUMANN-SERIES-EXACT",),
            monotonicity=(("rho", 1),),
        ),
        requires_domain_extremum=True,
        domain_extremum_certificate=DomainExtremumCertificate(
            "NEUMANN-RHO-MONOTONE",
            ExtremumCertificateType.MONOTONICITY,
            domain,
            ExtremumKind.SUPREMUM,
            "5",
            "NEUMANN-SERIES-EXACT",
            "1/(1-rho) is increasing on [0,0.8].",
            monotonicity_direction="increasing",
        ),
    )
    theorem_obj = ProofObjectV5(
        theorem,
        warrant=Warrant.PROVEN,
        establishment_regime=regime,
        deployment_regime=regime,
        assembly_spec=AssemblySpec(
            expression="ainv * factor",
            inputs=(("ainv", "1"), ("factor", "5")),
            required_terms=("ainv", "factor"),
            displayed_value="5",
            claim_direction=v4.Direction.UPPER,
            rounding=AssemblyRounding.EXACT,
            source_hashes=("MATRIX-PERTURBATION-EXACT",),
            monotonicity=(("ainv", 1), ("factor", 1)),
        ),
        requires_domain_extremum=True,
        domain_extremum_certificate=DomainExtremumCertificate(
            "MATRIX-PERTURBATION-RHO-MONOTONE",
            ExtremumCertificateType.MONOTONICITY,
            domain,
            ExtremumKind.SUPREMUM,
            "5",
            "MATRIX-PERTURBATION-EXACT",
            "The Neumann factor is maximal at rho=0.8.",
            monotonicity_direction="increasing",
        ),
    )
    return GateVerifierV5([lemma_obj, theorem_obj]), ["MATRIX_PERTURBATION_THEOREM"]


def _demo() -> None:
    q0, q0_roots = current_q0_upper_contract()
    q0_report = q0.validate(q0_roots)
    q0_fail_gates = {
        result.gate
        for result in q0_report.gate_results
        if result.verdict == GateVerdict.FAIL
    }
    assert "ASSEMBLY" in q0_fail_gates
    assert "UB_G_RESIDUAL_UNIFORM" in q0_fail_gates

    second, second_roots = matrix_perturbation_second_domain()
    second_report = second.validate(second_roots)
    assert second_report.valid and second_report.promotable, second_report.as_dict()

    print(
        json.dumps(
            {
                "current_q0_contract_valid": q0_report.valid,
                "current_q0_contract_promotable": q0_report.promotable,
                "current_q0_blocking_gates": sorted(q0_fail_gates),
                "second_domain_valid": second_report.valid,
                "second_domain_promotable": second_report.promotable,
                "second_domain_root_hashes": second_report.root_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _demo()
