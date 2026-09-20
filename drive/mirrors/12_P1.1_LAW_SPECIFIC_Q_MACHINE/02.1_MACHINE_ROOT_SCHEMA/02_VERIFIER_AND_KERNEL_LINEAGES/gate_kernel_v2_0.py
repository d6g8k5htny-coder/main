#!/usr/bin/env python3
"""
GATE KERNEL 2.0

Standalone executable shell for Gate Framework Master v1.2 (reconciled).

BREAKING CHANGES FROM 1.x
-------------------------
* Claim status, warrant, assumption role, and reasoning mode are separate.
* Content hashes are recomputed from canonical claim content.
* Every dependency edge pins the target's canonical SHA-256.
* Parameter domains are strict: every parent parameter must be present and
  covered by every load-bearing dependency or a typed domain bridge.
* Change-of-measure, mark-transfer, model-transfer, scale-transfer, and
  heuristic bridges are typed first-class nodes with source/target regimes.
* Composition witnesses are first-class nodes and operation-specific.
* PROVISIONAL ceilings propagate through non-conditional dependencies.
* Conditional debts propagate as explicit condition sets.
* Registry loading is all-or-nothing; unsupported schemas, cycles, missing
  nodes, hash drift, and admission refusals raise.
* ASSEMBLY, DOMAIN-INFIMUM, and BAND-PROVENANCE are implemented.
* No hard-coded output path is used.

SCOPE
-----
This kernel checks the mechanically decidable SHELL: schema completeness,
tag compatibility, canonical hash integrity, graph structure, numerical
assembly arithmetic, certificate coverage, and declared grade/status
propagation.

It does not prove the mathematical CORE asserted by a tag or certificate.
A bridge node may be well-typed and still mathematically false. Tag-to-content
fidelity remains the job of proofs, adversarial reimplementation, separated
adjudication, and external source review.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

getcontext().prec = 90

SCHEMA_VERSION = "gate-kernel/2.0"


# ============================================================================
# Controlled lexicons
# ============================================================================

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PROVISIONAL = "PROVISIONAL"
    NA = "NOT-APPLICABLE"


class ClaimStatus(str, Enum):
    LIVE = "LIVE"
    OPEN = "OPEN"
    KILLED = "KILLED"
    SUPERSEDED = "SUPERSEDED"
    RETIRED = "RETIRED"


class Warrant(str, Enum):
    PROVEN = "PROVEN"
    PROVEN_MODULO = "PROVEN-MODULO"
    CERTIFIED = "CERTIFIED"
    DERIVED = "DERIVED"
    LITERATURE_SUPPORTED = "LITERATURE-SUPPORTED"
    MEASURED = "MEASURED"
    PLAUSIBLE = "PLAUSIBLE"
    CONJECTURE = "CONJECTURE"


class AssumptionRole(str, Enum):
    NONE = "NONE"
    DEFINITION = "DEFINITION"
    PRIMITIVE_HYPOTHESIS = "PRIMITIVE-HYPOTHESIS"
    POLICY_THRESHOLD = "POLICY-THRESHOLD"


class ReasoningMode(str, Enum):
    DEDUCTIVE = "DEDUCTIVE"
    EMPIRICAL = "EMPIRICAL"
    ANALOGICAL = "ANALOGICAL"
    HEURISTIC = "HEURISTIC"
    NORMATIVE = "NORMATIVE"


class ClaimKind(str, Enum):
    STATEMENT = "STATEMENT"
    ASSUMPTION = "ASSUMPTION"
    DEFINITION = "DEFINITION"
    MEASUREMENT = "MEASUREMENT"
    BRIDGE = "BRIDGE"
    WITNESS = "WITNESS"
    CERTIFICATE = "CERTIFICATE"


class Direction(str, Enum):
    NONE = "NONE"
    UPPER = "UPPER"
    LOWER = "LOWER"
    EQUALITY = "EQUALITY"
    STRUCTURAL = "STRUCTURAL"


class Quantifier(str, Enum):
    EXACT = "EXACT"
    FOR_ALL = "FOR-ALL"
    EXISTS = "EXISTS"
    ASYMPTOTIC = "ASYMPTOTIC"
    TYPICAL = "TYPICAL"
    MEASURED_AT_RUNG = "MEASURED-AT-RUNG"
    CONDITIONAL = "CONDITIONAL"


class UncertaintySide(str, Enum):
    NONE = "NONE"
    UPPER = "UPPER"
    LOWER = "LOWER"
    SYMMETRIC = "SYMMETRIC"


class Operation(str, Enum):
    ATOMIC = "ATOMIC"
    PRODUCT = "PRODUCT"
    UNION = "UNION"
    AFFINE_ASSEMBLY = "AFFINE-ASSEMBLY"
    MONOTONE_TRANSFORM = "MONOTONE-TRANSFORM"
    INTERVAL_ENCLOSURE = "INTERVAL-ENCLOSURE"
    EXACT_IDENTITY = "EXACT-IDENTITY"


class WitnessType(str, Enum):
    INDEPENDENCE = "INDEPENDENCE"
    CONDITIONAL_INDEPENDENCE = "CONDITIONAL-INDEPENDENCE"
    MARKOV_PROPERTY = "MARKOV-PROPERTY"
    NEGATIVE_DEPENDENCE = "NEGATIVE-DEPENDENCE"
    COMPARISON_THEOREM = "COMPARISON-THEOREM"
    JOINT_CERTIFICATE = "JOINT-CERTIFICATE"
    UNION_BOUND = "UNION-BOUND"
    MONOTONE_ASSEMBLY = "MONOTONE-ASSEMBLY"
    INTERVAL_ARITHMETIC = "INTERVAL-ARITHMETIC"
    EXACT_ALGEBRA = "EXACT-ALGEBRA"


_ALLOWED_WITNESSES: Mapping[Operation, frozenset[WitnessType]] = {
    Operation.ATOMIC: frozenset(),
    Operation.PRODUCT: frozenset({
        WitnessType.INDEPENDENCE,
        WitnessType.CONDITIONAL_INDEPENDENCE,
        WitnessType.MARKOV_PROPERTY,
        WitnessType.NEGATIVE_DEPENDENCE,
        WitnessType.COMPARISON_THEOREM,
        WitnessType.JOINT_CERTIFICATE,
        WitnessType.EXACT_ALGEBRA,
    }),
    Operation.UNION: frozenset({
        WitnessType.UNION_BOUND,
        WitnessType.JOINT_CERTIFICATE,
        WitnessType.EXACT_ALGEBRA,
    }),
    Operation.AFFINE_ASSEMBLY: frozenset({
        WitnessType.MONOTONE_ASSEMBLY,
        WitnessType.INTERVAL_ARITHMETIC,
        WitnessType.EXACT_ALGEBRA,
    }),
    Operation.MONOTONE_TRANSFORM: frozenset({
        WitnessType.MONOTONE_ASSEMBLY,
        WitnessType.COMPARISON_THEOREM,
        WitnessType.EXACT_ALGEBRA,
    }),
    Operation.INTERVAL_ENCLOSURE: frozenset({
        WitnessType.INTERVAL_ARITHMETIC,
        WitnessType.JOINT_CERTIFICATE,
    }),
    Operation.EXACT_IDENTITY: frozenset({WitnessType.EXACT_ALGEBRA}),
}


class EdgeKind(str, Enum):
    SUPPORT = "SUPPORT"
    CONDITION = "CONDITION"
    CHANGE_OF_MEASURE = "CHANGE-OF-MEASURE"
    MARK_TRANSFER = "MARK-TRANSFER"
    MODEL_TRANSFER = "MODEL-TRANSFER"
    DOMAIN_TRANSFER = "DOMAIN-TRANSFER"
    SCALE_TRANSFER = "SCALE-TRANSFER"
    CONDITIONING_TRANSFER = "CONDITIONING-TRANSFER"
    HEURISTIC_BRIDGE = "HEURISTIC-BRIDGE"
    COMPOSITION_WITNESS = "COMPOSITION-WITNESS"
    CERTIFICATE = "CERTIFICATE"


class BridgeKind(str, Enum):
    MEASURE = "MEASURE"
    MARK = "MARK"
    MODEL = "MODEL"
    DOMAIN = "DOMAIN"
    SCALE = "SCALE"
    CONDITIONING = "CONDITIONING"
    HEURISTIC = "HEURISTIC"


class VerificationMode(str, Enum):
    NONE = "NONE"
    EXACT_SINGLE = "EXACT-SINGLE"
    AGREEMENT = "AGREEMENT"
    ADVERSARIAL_REIMPLEMENTATION = "ADVERSARIAL-REIMPLEMENTATION"
    EXTERNAL_CONTRADICTION = "EXTERNAL-CONTRADICTION"


class InstrumentRelation(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    EXTERNAL_CROSSCHECK = "EXTERNAL-CROSSCHECK"
    SHARED_COMPONENTS = "SHARED-COMPONENTS"
    UNKNOWN = "UNKNOWN"


class RoundingRule(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    EXACT = "EXACT"
    OUTWARD_INTERVAL = "OUTWARD-INTERVAL"


class ExtremumKind(str, Enum):
    INFIMUM = "INFIMUM"
    SUPREMUM = "SUPREMUM"


class ExtremumCertificateType(str, Enum):
    ANALYTIC = "ANALYTIC"
    MONOTONICITY = "MONOTONICITY"
    INTERVAL = "INTERVAL"
    TAYLOR_MODEL = "TAYLOR-MODEL"
    GRID_WITH_MODULUS = "GRID-WITH-MODULUS"
    EXACT_ENUMERATION = "EXACT-ENUMERATION"
    ASYMPTOTIC_MONOTONICITY = "ASYMPTOTIC-MONOTONICITY"
    ANALYTIC_UNIFORM_BOUND = "ANALYTIC-UNIFORM-BOUND"
    RUNG_SAMPLE = "RUNG-SAMPLE"


class BandKind(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    INTERVAL = "INTERVAL"
    STATISTICAL = "STATISTICAL"
    BOOTSTRAP = "BOOTSTRAP"
    MONTE_CARLO = "MONTE-CARLO"
    RANDOMIZED_NUMERICAL = "RANDOMIZED-NUMERICAL"
    FORECAST = "FORECAST"


class CertificateRole(str, Enum):
    DOMAIN_EXTREMUM = "DOMAIN-EXTREMUM"
    BAND_PROVENANCE = "BAND-PROVENANCE"
    COMMON_MODE = "COMMON-MODE"
    COVERAGE = "COVERAGE"
    ENDPOINT = "ENDPOINT"


class ActiveGateStatus(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    KILLED = "KILLED"
    BLOCKED_EXTERNAL = "BLOCKED-EXTERNAL"
    NOT_CLAIMED = "NOT-CLAIMED"


class AdmissionMode(str, Enum):
    ARCHIVE = "ARCHIVE"
    CANDIDATE = "CANDIDATE"
    PROMOTED = "PROMOTED"


# A conservative total order used only for explicit weakest-link ceilings.
# Conditional theorem logic is handled separately.
_WARRANT_RANK: Mapping[Warrant, int] = {
    Warrant.CONJECTURE: 0,
    Warrant.PLAUSIBLE: 1,
    Warrant.MEASURED: 2,
    Warrant.DERIVED: 3,
    Warrant.CERTIFIED: 4,
    Warrant.LITERATURE_SUPPORTED: 5,
    Warrant.PROVEN_MODULO: 6,
    Warrant.PROVEN: 7,
}


# ============================================================================
# Typed schema
# ============================================================================

@dataclass(frozen=True)
class Interval:
    lower: str
    upper: str
    lower_open: bool = False
    upper_open: bool = False

    def _d(self, value: str) -> Decimal:
        return Decimal(value)

    def contains(self, other: "Interval") -> bool:
        lo, hi = self._d(self.lower), self._d(self.upper)
        olo, ohi = other._d(other.lower), other._d(other.upper)
        if lo > olo or hi < ohi:
            return False
        if lo == olo and self.lower_open and not other.lower_open:
            return False
        if hi == ohi and self.upper_open and not other.upper_open:
            return False
        return True

    def as_dict(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "lower_open": self.lower_open,
            "upper_open": self.upper_open,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Interval":
        return Interval(
            str(data["lower"]),
            str(data["upper"]),
            bool(data.get("lower_open", False)),
            bool(data.get("upper_open", False)),
        )


@dataclass(frozen=True)
class ParameterDomain:
    parameters: tuple[tuple[str, Interval], ...] = ()

    def as_map(self) -> dict[str, Interval]:
        return dict(self.parameters)

    def contains(self, other: "ParameterDomain") -> bool:
        mine = self.as_map()
        theirs = other.as_map()
        for name, interval in theirs.items():
            if name not in mine or not mine[name].contains(interval):
                return False
        return True

    def missing_or_narrow(self, other: "ParameterDomain") -> list[str]:
        mine = self.as_map()
        issues: list[str] = []
        for name, interval in other.as_map().items():
            if name not in mine:
                issues.append(f"missing parameter {name}")
            elif not mine[name].contains(interval):
                issues.append(
                    f"{name}: {mine[name].as_dict()} does not contain "
                    f"{interval.as_dict()}"
                )
        return issues

    def as_dict(self) -> dict:
        return {name: interval.as_dict() for name, interval in self.parameters}

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> "ParameterDomain":
        if not data:
            return ParameterDomain()
        return ParameterDomain(
            tuple(
                (name, Interval.from_dict(spec))
                for name, spec in sorted(data.items())
            )
        )


@dataclass(frozen=True)
class Regime:
    model_id: str = ""
    measure_id: str = ""
    domain: ParameterDomain = ParameterDomain()
    spatial_region: str = ""
    scale_id: str = ""
    conditioning_depth: str = ""
    marks: frozenset[str] = frozenset()
    deployment_mode: str = ""

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "measure_id": self.measure_id,
            "domain": self.domain.as_dict(),
            "spatial_region": self.spatial_region,
            "scale_id": self.scale_id,
            "conditioning_depth": self.conditioning_depth,
            "marks": sorted(self.marks),
            "deployment_mode": self.deployment_mode,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> "Regime":
        data = data or {}
        return Regime(
            model_id=str(data.get("model_id", "")),
            measure_id=str(data.get("measure_id", "")),
            domain=ParameterDomain.from_dict(data.get("domain")),
            spatial_region=str(data.get("spatial_region", "")),
            scale_id=str(data.get("scale_id", "")),
            conditioning_depth=str(data.get("conditioning_depth", "")),
            marks=frozenset(data.get("marks", [])),
            deployment_mode=str(data.get("deployment_mode", "")),
        )


@dataclass(frozen=True)
class MeasuredValue:
    name: str
    value: str
    scale_id: str
    parameter_tags: tuple[tuple[str, str], ...] = ()
    evidence_id: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "scale_id": self.scale_id,
            "parameter_tags": dict(self.parameter_tags),
            "evidence_id": self.evidence_id,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "MeasuredValue":
        return MeasuredValue(
            name=str(data["name"]),
            value=str(data["value"]),
            scale_id=str(data.get("scale_id", "")),
            parameter_tags=tuple(sorted(
                (str(k), str(v))
                for k, v in data.get("parameter_tags", {}).items()
            )),
            evidence_id=str(data.get("evidence_id", "")),
        )


@dataclass(frozen=True)
class BridgeSpec:
    bridge_kind: BridgeKind
    source_regime: Regime
    target_regime: Regime
    scope_domain: ParameterDomain
    transferred_marks: frozenset[str] = frozenset()
    source_domain_name: str = ""
    target_domain_name: str = ""
    scope_statement: str = ""
    falsifier: str = ""

    def as_dict(self) -> dict:
        return {
            "bridge_kind": self.bridge_kind.value,
            "source_regime": self.source_regime.as_dict(),
            "target_regime": self.target_regime.as_dict(),
            "scope_domain": self.scope_domain.as_dict(),
            "transferred_marks": sorted(self.transferred_marks),
            "source_domain_name": self.source_domain_name,
            "target_domain_name": self.target_domain_name,
            "scope_statement": self.scope_statement,
            "falsifier": self.falsifier,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BridgeSpec":
        return BridgeSpec(
            bridge_kind=BridgeKind(data["bridge_kind"]),
            source_regime=Regime.from_dict(data.get("source_regime")),
            target_regime=Regime.from_dict(data.get("target_regime")),
            scope_domain=ParameterDomain.from_dict(data.get("scope_domain")),
            transferred_marks=frozenset(data.get("transferred_marks", [])),
            source_domain_name=str(data.get("source_domain_name", "")),
            target_domain_name=str(data.get("target_domain_name", "")),
            scope_statement=str(data.get("scope_statement", "")),
            falsifier=str(data.get("falsifier", "")),
        )


@dataclass(frozen=True)
class WitnessSpec:
    operation: Operation
    witness_type: WitnessType
    covered_claim_ids: tuple[str, ...] = ()
    statement: str = ""

    def as_dict(self) -> dict:
        return {
            "operation": self.operation.value,
            "witness_type": self.witness_type.value,
            "covered_claim_ids": list(self.covered_claim_ids),
            "statement": self.statement,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "WitnessSpec":
        return WitnessSpec(
            operation=Operation(data["operation"]),
            witness_type=WitnessType(data["witness_type"]),
            covered_claim_ids=tuple(data.get("covered_claim_ids", [])),
            statement=str(data.get("statement", "")),
        )


@dataclass(frozen=True)
class AssemblyInput:
    name: str
    value: str
    source_id: str
    monotonicity: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "source_id": self.source_id,
            "monotonicity": self.monotonicity,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AssemblyInput":
        return AssemblyInput(
            str(data["name"]),
            str(data["value"]),
            str(data.get("source_id", "")),
            int(data.get("monotonicity", 0)),
        )


@dataclass(frozen=True)
class AssemblySpec:
    expression: str
    inputs: tuple[AssemblyInput, ...]
    required_terms: tuple[str, ...]
    residual_terms: tuple[str, ...]
    displayed_value: str
    direction: Direction
    rounding: RoundingRule
    precision_digits: int = 50

    def as_dict(self) -> dict:
        return {
            "expression": self.expression,
            "inputs": [item.as_dict() for item in self.inputs],
            "required_terms": list(self.required_terms),
            "residual_terms": list(self.residual_terms),
            "displayed_value": self.displayed_value,
            "direction": self.direction.value,
            "rounding": self.rounding.value,
            "precision_digits": self.precision_digits,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AssemblySpec":
        return AssemblySpec(
            expression=str(data["expression"]),
            inputs=tuple(AssemblyInput.from_dict(x) for x in data.get("inputs", [])),
            required_terms=tuple(data.get("required_terms", [])),
            residual_terms=tuple(data.get("residual_terms", [])),
            displayed_value=str(data["displayed_value"]),
            direction=Direction(data["direction"]),
            rounding=RoundingRule(data["rounding"]),
            precision_digits=int(data.get("precision_digits", 50)),
        )


@dataclass(frozen=True)
class ExtremumSpec:
    certificate_type: ExtremumCertificateType
    extremum_kind: ExtremumKind
    domain: ParameterDomain
    extremum_value: str
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "certificate_type": self.certificate_type.value,
            "extremum_kind": self.extremum_kind.value,
            "domain": self.domain.as_dict(),
            "extremum_value": self.extremum_value,
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExtremumSpec":
        return ExtremumSpec(
            certificate_type=ExtremumCertificateType(data["certificate_type"]),
            extremum_kind=ExtremumKind(data["extremum_kind"]),
            domain=ParameterDomain.from_dict(data.get("domain")),
            extremum_value=str(data["extremum_value"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class BandSpec:
    kind: BandKind
    side: UncertaintySide
    construction: str
    coverage_domain: ParameterDomain
    confidence_level: Optional[str] = None
    sample_size: Optional[int] = None
    multiplicity_scope: str = ""
    seed_or_algorithm_state: str = ""
    deterministic_tolerance: str = ""
    consumed_as: UncertaintySide = UncertaintySide.NONE
    forecast_only: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "side": self.side.value,
            "construction": self.construction,
            "coverage_domain": self.coverage_domain.as_dict(),
            "confidence_level": self.confidence_level,
            "sample_size": self.sample_size,
            "multiplicity_scope": self.multiplicity_scope,
            "seed_or_algorithm_state": self.seed_or_algorithm_state,
            "deterministic_tolerance": self.deterministic_tolerance,
            "consumed_as": self.consumed_as.value,
            "forecast_only": self.forecast_only,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BandSpec":
        return BandSpec(
            kind=BandKind(data["kind"]),
            side=UncertaintySide(data["side"]),
            construction=str(data.get("construction", "")),
            coverage_domain=ParameterDomain.from_dict(data.get("coverage_domain")),
            confidence_level=(
                None if data.get("confidence_level") is None
                else str(data["confidence_level"])
            ),
            sample_size=(
                None if data.get("sample_size") is None
                else int(data["sample_size"])
            ),
            multiplicity_scope=str(data.get("multiplicity_scope", "")),
            seed_or_algorithm_state=str(data.get("seed_or_algorithm_state", "")),
            deterministic_tolerance=str(data.get("deterministic_tolerance", "")),
            consumed_as=UncertaintySide(data.get("consumed_as", "NONE")),
            forecast_only=bool(data.get("forecast_only", False)),
        )


@dataclass(frozen=True)
class CommonModeSpec:
    relation: InstrumentRelation
    instrument_ids: tuple[str, ...]
    error_channel_under_test: str
    shared_components: tuple[str, ...] = ()
    disjoint_error_argument: str = ""
    external_crosscheck_id: str = ""

    def as_dict(self) -> dict:
        return {
            "relation": self.relation.value,
            "instrument_ids": list(self.instrument_ids),
            "error_channel_under_test": self.error_channel_under_test,
            "shared_components": list(self.shared_components),
            "disjoint_error_argument": self.disjoint_error_argument,
            "external_crosscheck_id": self.external_crosscheck_id,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CommonModeSpec":
        return CommonModeSpec(
            relation=InstrumentRelation(data["relation"]),
            instrument_ids=tuple(data.get("instrument_ids", [])),
            error_channel_under_test=str(data.get("error_channel_under_test", "")),
            shared_components=tuple(data.get("shared_components", [])),
            disjoint_error_argument=str(data.get("disjoint_error_argument", "")),
            external_crosscheck_id=str(data.get("external_crosscheck_id", "")),
        )


@dataclass(frozen=True)
class CoverageSpec:
    universe_id: str
    failure_regions: tuple[tuple[str, tuple[str, ...]], ...]
    uncovered_mass: str = "0"
    target_total_risk: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "universe_id": self.universe_id,
            "failure_regions": {
                region: list(charts) for region, charts in self.failure_regions
            },
            "uncovered_mass": self.uncovered_mass,
            "target_total_risk": self.target_total_risk,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CoverageSpec":
        return CoverageSpec(
            universe_id=str(data.get("universe_id", "")),
            failure_regions=tuple(
                (str(region), tuple(charts))
                for region, charts in sorted(
                    data.get("failure_regions", {}).items()
                )
            ),
            uncovered_mass=str(data.get("uncovered_mass", "0")),
            target_total_risk=(
                None if data.get("target_total_risk") is None
                else str(data["target_total_risk"])
            ),
        )


@dataclass(frozen=True)
class EndpointSpec:
    point: str
    expression: str
    satisfies: bool
    evidence_id: str

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "expression": self.expression,
            "satisfies": self.satisfies,
            "evidence_id": self.evidence_id,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EndpointSpec":
        return EndpointSpec(
            point=str(data["point"]),
            expression=str(data.get("expression", "")),
            satisfies=bool(data.get("satisfies", False)),
            evidence_id=str(data.get("evidence_id", "")),
        )


@dataclass(frozen=True)
class ActiveGate:
    gate_id: str
    status: ActiveGateStatus
    evidence_id: str = ""
    blocks_promotion: bool = True

    def as_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "evidence_id": self.evidence_id,
            "blocks_promotion": self.blocks_promotion,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ActiveGate":
        return ActiveGate(
            gate_id=str(data["gate_id"]),
            status=ActiveGateStatus(data["status"]),
            evidence_id=str(data.get("evidence_id", "")),
            blocks_promotion=bool(data.get("blocks_promotion", True)),
        )


@dataclass(frozen=True)
class Edge:
    target: str
    kind: EdgeKind = EdgeKind.SUPPORT
    pinned_hash: str = ""
    conditional: bool = False
    role: str = ""

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind.value,
            "pinned_hash": self.pinned_hash,
            "conditional": self.conditional,
            "role": self.role,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Edge":
        return Edge(
            target=str(data["target"]),
            kind=EdgeKind(data.get("kind", "SUPPORT")),
            pinned_hash=str(data.get("pinned_hash", "")),
            conditional=bool(data.get("conditional", False)),
            role=str(data.get("role", "")),
        )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    statement: str
    kind: ClaimKind
    status: ClaimStatus
    warrant: Warrant
    source_precedence_id: str
    evidence_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    alias_of: str = ""

    assumption_role: AssumptionRole = AssumptionRole.NONE
    reasoning_mode: ReasoningMode = ReasoningMode.DEDUCTIVE
    load_bearing: bool = True

    quantifier: Quantifier = Quantifier.CONDITIONAL
    quantifier_order: str = ""
    direction: Direction = Direction.NONE
    bound_coefficient: Optional[str] = None
    uncertainty_side: UncertaintySide = UncertaintySide.NONE

    establishment_regime: Regime = Regime()
    deployment_regime: Regime = Regime()
    required_marks: frozenset[str] = frozenset()
    supplied_marks: frozenset[str] = frozenset()

    operation: Operation = Operation.ATOMIC
    measured_values: tuple[MeasuredValue, ...] = ()
    finite_range_bound: bool = False
    registered_endpoints: tuple[EndpointSpec, ...] = ()

    bridge_spec: Optional[BridgeSpec] = None
    witness_spec: Optional[WitnessSpec] = None
    assembly_spec: Optional[AssemblySpec] = None
    extremum_spec: Optional[ExtremumSpec] = None
    band_spec: Optional[BandSpec] = None
    common_mode_spec: Optional[CommonModeSpec] = None
    coverage_spec: Optional[CoverageSpec] = None

    verification_mode: VerificationMode = VerificationMode.NONE
    closed_core: bool = False
    conditional_on: tuple[str, ...] = ()
    active_gates: tuple[ActiveGate, ...] = ()
    edges: tuple[Edge, ...] = ()
    content_hash: str = ""

    def canonical_payload(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "kind": self.kind.value,
            "status": self.status.value,
            "warrant": self.warrant.value,
            "source_precedence_id": self.source_precedence_id,
            "evidence_ids": list(self.evidence_ids),
            "supersedes": list(self.supersedes),
            "alias_of": self.alias_of,
            "assumption_role": self.assumption_role.value,
            "reasoning_mode": self.reasoning_mode.value,
            "load_bearing": self.load_bearing,
            "quantifier": self.quantifier.value,
            "quantifier_order": self.quantifier_order,
            "direction": self.direction.value,
            "bound_coefficient": self.bound_coefficient,
            "uncertainty_side": self.uncertainty_side.value,
            "establishment_regime": self.establishment_regime.as_dict(),
            "deployment_regime": self.deployment_regime.as_dict(),
            "required_marks": sorted(self.required_marks),
            "supplied_marks": sorted(self.supplied_marks),
            "operation": self.operation.value,
            "measured_values": [m.as_dict() for m in self.measured_values],
            "finite_range_bound": self.finite_range_bound,
            "registered_endpoints": [e.as_dict() for e in self.registered_endpoints],
            "bridge_spec": None if self.bridge_spec is None else self.bridge_spec.as_dict(),
            "witness_spec": None if self.witness_spec is None else self.witness_spec.as_dict(),
            "assembly_spec": None if self.assembly_spec is None else self.assembly_spec.as_dict(),
            "extremum_spec": None if self.extremum_spec is None else self.extremum_spec.as_dict(),
            "band_spec": None if self.band_spec is None else self.band_spec.as_dict(),
            "common_mode_spec": (
                None if self.common_mode_spec is None
                else self.common_mode_spec.as_dict()
            ),
            "coverage_spec": None if self.coverage_spec is None else self.coverage_spec.as_dict(),
            "verification_mode": self.verification_mode.value,
            "closed_core": self.closed_core,
            "conditional_on": list(self.conditional_on),
            "active_gates": [g.as_dict() for g in self.active_gates],
            "edges": [e.as_dict() for e in self.edges],
        }

    def computed_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def sealed(self) -> "Claim":
        return replace(self, content_hash=self.computed_hash())

    def with_pinned_edges(self, graph: Mapping[str, "Claim"]) -> "Claim":
        edges = []
        for edge in self.edges:
            if edge.target not in graph:
                raise KeyError(f"cannot pin missing dependency {edge.target}")
            edges.append(replace(edge, pinned_hash=graph[edge.target].content_hash))
        return replace(self, edges=tuple(edges)).sealed()

    def as_dict(self) -> dict:
        data = self.canonical_payload()
        data["content_hash"] = self.content_hash
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, verify_hash: bool = True) -> "Claim":
        claim = Claim(
            claim_id=str(data["claim_id"]),
            statement=str(data["statement"]),
            kind=ClaimKind(data["kind"]),
            status=ClaimStatus(data["status"]),
            warrant=Warrant(data["warrant"]),
            source_precedence_id=str(data["source_precedence_id"]),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            supersedes=tuple(data.get("supersedes", [])),
            alias_of=str(data.get("alias_of", "")),
            assumption_role=AssumptionRole(data.get("assumption_role", "NONE")),
            reasoning_mode=ReasoningMode(data.get("reasoning_mode", "DEDUCTIVE")),
            load_bearing=bool(data.get("load_bearing", True)),
            quantifier=Quantifier(data.get("quantifier", "CONDITIONAL")),
            quantifier_order=str(data.get("quantifier_order", "")),
            direction=Direction(data.get("direction", "NONE")),
            bound_coefficient=(
                None if data.get("bound_coefficient") is None
                else str(data["bound_coefficient"])
            ),
            uncertainty_side=UncertaintySide(data.get("uncertainty_side", "NONE")),
            establishment_regime=Regime.from_dict(data.get("establishment_regime")),
            deployment_regime=Regime.from_dict(data.get("deployment_regime")),
            required_marks=frozenset(data.get("required_marks", [])),
            supplied_marks=frozenset(data.get("supplied_marks", [])),
            operation=Operation(data.get("operation", "ATOMIC")),
            measured_values=tuple(
                MeasuredValue.from_dict(x)
                for x in data.get("measured_values", [])
            ),
            finite_range_bound=bool(data.get("finite_range_bound", False)),
            registered_endpoints=tuple(
                EndpointSpec.from_dict(x)
                for x in data.get("registered_endpoints", [])
            ),
            bridge_spec=(
                None if data.get("bridge_spec") is None
                else BridgeSpec.from_dict(data["bridge_spec"])
            ),
            witness_spec=(
                None if data.get("witness_spec") is None
                else WitnessSpec.from_dict(data["witness_spec"])
            ),
            assembly_spec=(
                None if data.get("assembly_spec") is None
                else AssemblySpec.from_dict(data["assembly_spec"])
            ),
            extremum_spec=(
                None if data.get("extremum_spec") is None
                else ExtremumSpec.from_dict(data["extremum_spec"])
            ),
            band_spec=(
                None if data.get("band_spec") is None
                else BandSpec.from_dict(data["band_spec"])
            ),
            common_mode_spec=(
                None if data.get("common_mode_spec") is None
                else CommonModeSpec.from_dict(data["common_mode_spec"])
            ),
            coverage_spec=(
                None if data.get("coverage_spec") is None
                else CoverageSpec.from_dict(data["coverage_spec"])
            ),
            verification_mode=VerificationMode(data.get("verification_mode", "NONE")),
            closed_core=bool(data.get("closed_core", False)),
            conditional_on=tuple(data.get("conditional_on", [])),
            active_gates=tuple(
                ActiveGate.from_dict(x) for x in data.get("active_gates", [])
            ),
            edges=tuple(Edge.from_dict(x) for x in data.get("edges", [])),
            content_hash=str(data.get("content_hash", "")),
        )
        if verify_hash and claim.content_hash != claim.computed_hash():
            raise ValueError(
                f"content hash mismatch for {claim.claim_id}: "
                f"declared {claim.content_hash}, computed {claim.computed_hash()}"
            )
        return claim


Graph = dict[str, Claim]


@dataclass(frozen=True)
class GateResult:
    gate: str
    claim_id: str
    verdict: Verdict
    message: str
    evidence_id: str = ""
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
class ClaimReport:
    claim_id: str
    gate_results: list[GateResult]
    effective_warrant: Warrant
    conditions: tuple[str, ...]
    shell_valid: bool
    conditional_promotable: bool
    unconditional_promotable: bool
    closable: bool

    def as_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "gate_results": [r.as_dict() for r in self.gate_results],
            "effective_warrant": self.effective_warrant.value,
            "conditions": list(self.conditions),
            "shell_valid": self.shell_valid,
            "conditional_promotable": self.conditional_promotable,
            "unconditional_promotable": self.unconditional_promotable,
            "closable": self.closable,
        }


@dataclass
class GraphReport:
    roots: tuple[str, ...]
    graph_errors: list[str]
    claim_reports: dict[str, ClaimReport]
    root_hashes: dict[str, str]
    shell_valid: bool
    conditional_promotable: bool
    unconditional_promotable: bool

    def as_dict(self) -> dict:
        return {
            "roots": list(self.roots),
            "graph_errors": self.graph_errors,
            "claim_reports": {
                key: value.as_dict()
                for key, value in sorted(self.claim_reports.items())
            },
            "root_hashes": self.root_hashes,
            "shell_valid": self.shell_valid,
            "conditional_promotable": self.conditional_promotable,
            "unconditional_promotable": self.unconditional_promotable,
        }


# ============================================================================
# Canonical expression evaluator
# ============================================================================

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** int(b),
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval_decimal(node: ast.AST, values: Mapping[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_decimal(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise KeyError(node.id)
        return values[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return Decimal(str(node.value))
        raise TypeError("unsupported constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _eval_decimal(node.left, values),
            _eval_decimal(node.right, values),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_decimal(node.operand, values))
    raise TypeError(f"unsupported expression node: {ast.dump(node)}")


def evaluate_assembly(spec: AssemblySpec) -> Decimal:
    values = {item.name: Decimal(item.value) for item in spec.inputs}
    return _eval_decimal(ast.parse(spec.expression, mode="eval"), values)


def expression_names(expression: str) -> set[str]:
    return {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    }


# ============================================================================
# Graph helpers and structural validation
# ============================================================================

def dependency_ids(claim: Claim) -> tuple[str, ...]:
    return tuple(edge.target for edge in claim.edges)


def reachable(root: str, graph: Graph) -> set[str]:
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        claim = graph.get(current)
        if claim is not None:
            stack.extend(edge.target for edge in claim.edges)
    return seen


def find_cycle(graph: Graph) -> Optional[list[str]]:
    white, grey, black = 0, 1, 2
    colour = {claim_id: white for claim_id in graph}
    path: list[str] = []

    def visit(claim_id: str) -> Optional[list[str]]:
        colour[claim_id] = grey
        path.append(claim_id)
        for edge in graph[claim_id].edges:
            target = edge.target
            if target not in graph:
                continue
            if colour[target] == grey:
                index = path.index(target)
                return path[index:] + [target]
            if colour[target] == white:
                cycle = visit(target)
                if cycle:
                    return cycle
        path.pop()
        colour[claim_id] = black
        return None

    for claim_id in graph:
        if colour[claim_id] == white:
            cycle = visit(claim_id)
            if cycle:
                return cycle
    return None


def topological_order(graph: Graph) -> list[str]:
    cycle = find_cycle(graph)
    if cycle:
        raise ValueError("cycle: " + " -> ".join(cycle))
    remaining = set(graph)
    order: list[str] = []
    while remaining:
        ready = sorted(
            claim_id
            for claim_id in remaining
            if all(edge.target in order for edge in graph[claim_id].edges)
        )
        if not ready:
            missing = {
                edge.target
                for claim_id in remaining
                for edge in graph[claim_id].edges
                if edge.target not in graph
            }
            raise ValueError(
                "topological order incomplete; missing targets: "
                + ", ".join(sorted(missing))
            )
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def _kind_schema_errors(claim: Claim) -> list[str]:
    errors: list[str] = []
    if claim.kind == ClaimKind.BRIDGE and claim.bridge_spec is None:
        errors.append("BRIDGE node lacks bridge_spec")
    if claim.kind != ClaimKind.BRIDGE and claim.bridge_spec is not None:
        errors.append("non-BRIDGE node carries bridge_spec")
    if claim.kind == ClaimKind.WITNESS and claim.witness_spec is None:
        errors.append("WITNESS node lacks witness_spec")
    if claim.kind != ClaimKind.WITNESS and claim.witness_spec is not None:
        errors.append("non-WITNESS node carries witness_spec")
    if claim.assumption_role == AssumptionRole.PRIMITIVE_HYPOTHESIS:
        if claim.kind != ClaimKind.ASSUMPTION:
            errors.append("primitive hypothesis must have kind ASSUMPTION")
    if claim.assumption_role == AssumptionRole.DEFINITION:
        if claim.kind != ClaimKind.DEFINITION:
            errors.append("definition role must have kind DEFINITION")
    return errors


def validate_graph(graph: Graph) -> list[str]:
    errors: list[str] = []
    if not graph:
        return ["graph is empty"]

    hashes: dict[str, str] = {}
    for claim_id, claim in graph.items():
        if claim_id != claim.claim_id:
            errors.append(
                f"[id-key] mapping key {claim_id!r} != claim_id {claim.claim_id!r}"
            )
        if not claim.claim_id.strip():
            errors.append("[schema] blank claim_id")
        if not claim.statement.strip():
            errors.append(f"[schema] {claim_id}: blank statement")
        if not claim.source_precedence_id.strip():
            errors.append(f"[schema] {claim_id}: blank source_precedence_id")
        computed = claim.computed_hash()
        if claim.content_hash != computed:
            errors.append(
                f"[hash] {claim_id}: declared {claim.content_hash}, computed {computed}"
            )
        if claim.content_hash in hashes and not claim.alias_of:
            errors.append(
                f"[duplicate-hash] {claim_id} and {hashes[claim.content_hash]}"
            )
        hashes[claim.content_hash] = claim_id
        for issue in _kind_schema_errors(claim):
            errors.append(f"[schema] {claim_id}: {issue}")

        if claim.status == ClaimStatus.KILLED and claim.closed_core:
            errors.append(f"[status] {claim_id}: killed claim marked closed_core")
        if claim.status == ClaimStatus.OPEN and claim.warrant in {
            Warrant.PROVEN,
            Warrant.CERTIFIED,
        }:
            errors.append(
                f"[status/warrant] {claim_id}: OPEN cannot be {claim.warrant.value}"
            )
        if claim.status in {
            ClaimStatus.KILLED,
            ClaimStatus.SUPERSEDED,
            ClaimStatus.RETIRED,
        } and claim.load_bearing:
            # Archived dead claims may remain load_bearing in historical graphs,
            # but not as standalone live objects. Record a structural warning as error.
            errors.append(
                f"[status] {claim_id}: dead claim cannot be load_bearing"
            )

        for edge in claim.edges:
            if edge.target not in graph:
                errors.append(
                    f"[missing-target] {claim_id} -> {edge.target}"
                )
            elif not edge.pinned_hash:
                errors.append(
                    f"[unpinned-edge] {claim_id} -> {edge.target}"
                )
            elif edge.pinned_hash != graph[edge.target].content_hash:
                errors.append(
                    f"[stale-edge] {claim_id} -> {edge.target}: "
                    f"{edge.pinned_hash} != {graph[edge.target].content_hash}"
                )

    cycle = find_cycle(graph)
    if cycle:
        errors.append("[cycle] " + " -> ".join(cycle))
    return errors


# ============================================================================
# Gate helpers
# ============================================================================

def _support_edges(claim: Claim) -> tuple[Edge, ...]:
    return tuple(
        edge for edge in claim.edges
        if edge.kind in {EdgeKind.SUPPORT, EdgeKind.CONDITION}
    )


def _edges_of_kind(claim: Claim, kind: EdgeKind) -> tuple[Edge, ...]:
    return tuple(edge for edge in claim.edges if edge.kind == kind)


def _certificate_nodes(
    claim: Claim,
    graph: Graph,
    role: CertificateRole,
) -> list[Claim]:
    nodes = []
    for edge in claim.edges:
        if edge.kind == EdgeKind.CERTIFICATE and edge.role == role.value:
            node = graph.get(edge.target)
            if node is not None:
                nodes.append(node)
    return nodes


def _typed_bridge_nodes(
    claim: Claim,
    graph: Graph,
    edge_kind: EdgeKind,
    bridge_kind: BridgeKind,
) -> list[Claim]:
    nodes: list[Claim] = []
    for edge in claim.edges:
        if edge.kind != edge_kind:
            continue
        node = graph.get(edge.target)
        if (
            node is not None
            and node.kind == ClaimKind.BRIDGE
            and node.bridge_spec is not None
            and node.bridge_spec.bridge_kind == bridge_kind
        ):
            nodes.append(node)
    return nodes


def _bridge_matches(
    bridge: Claim,
    *,
    source: Regime,
    target: Regime,
    parent_domain: ParameterDomain,
    missing_marks: frozenset[str] = frozenset(),
) -> bool:
    spec = bridge.bridge_spec
    if spec is None:
        return False
    if not spec.scope_domain.contains(parent_domain):
        return False
    if spec.bridge_kind == BridgeKind.MEASURE:
        return (
            spec.source_regime.measure_id == source.measure_id
            and spec.target_regime.measure_id == target.measure_id
        )
    if spec.bridge_kind == BridgeKind.MODEL:
        return (
            spec.source_regime.model_id == source.model_id
            and spec.target_regime.model_id == target.model_id
        )
    if spec.bridge_kind == BridgeKind.MARK:
        return missing_marks.issubset(spec.transferred_marks)
    if spec.bridge_kind == BridgeKind.DOMAIN:
        return spec.target_regime.domain.contains(parent_domain)
    if spec.bridge_kind == BridgeKind.SCALE:
        return (
            spec.source_regime.scale_id == source.scale_id
            and spec.target_regime.scale_id == target.scale_id
        )
    if spec.bridge_kind == BridgeKind.CONDITIONING:
        return (
            spec.source_regime.conditioning_depth == source.conditioning_depth
            and spec.target_regime.conditioning_depth == target.conditioning_depth
        )
    if spec.bridge_kind == BridgeKind.HEURISTIC:
        return bool(spec.scope_statement and spec.falsifier)
    return False


def _node_grade_admissible_for_parent(
    node: Claim,
    parent: Claim,
) -> bool:
    if node.status in {
        ClaimStatus.KILLED,
        ClaimStatus.SUPERSEDED,
        ClaimStatus.RETIRED,
    }:
        return False
    if node.warrant in {Warrant.PLAUSIBLE, Warrant.CONJECTURE}:
        return parent.warrant in {Warrant.PLAUSIBLE, Warrant.CONJECTURE}
    if parent.warrant == Warrant.PROVEN:
        return node.warrant in {
            Warrant.PROVEN,
            Warrant.LITERATURE_SUPPORTED,
        }
    if parent.warrant == Warrant.CERTIFIED:
        return node.warrant in {
            Warrant.PROVEN,
            Warrant.CERTIFIED,
            Warrant.LITERATURE_SUPPORTED,
        }
    if parent.warrant == Warrant.PROVEN_MODULO:
        return node.warrant in {
            Warrant.PROVEN,
            Warrant.PROVEN_MODULO,
            Warrant.LITERATURE_SUPPORTED,
        }
    if parent.warrant == Warrant.DERIVED:
        return node.warrant not in {
            Warrant.PLAUSIBLE,
            Warrant.CONJECTURE,
        }
    if parent.warrant == Warrant.MEASURED:
        return node.warrant in {
            Warrant.PROVEN,
            Warrant.CERTIFIED,
            Warrant.DERIVED,
            Warrant.LITERATURE_SUPPORTED,
            Warrant.MEASURED,
        }
    return True


def _minimum_warrant(a: Warrant, b: Warrant) -> Warrant:
    return a if _WARRANT_RANK[a] <= _WARRANT_RANK[b] else b


# ============================================================================
# Gates
# ============================================================================

def gate_schema(claim: Claim, graph: Graph) -> GateResult:
    issues = _kind_schema_errors(claim)
    if claim.content_hash != claim.computed_hash():
        issues.append("canonical content hash mismatch")
    if claim.load_bearing and not claim.evidence_ids and claim.kind not in {
        ClaimKind.DEFINITION,
        ClaimKind.ASSUMPTION,
    }:
        issues.append("load-bearing claim lacks evidence_ids")
    if claim.quantifier == Quantifier.FOR_ALL and not claim.deployment_regime.domain.parameters:
        issues.append("FOR-ALL claim lacks deployment parameter domain")
    if claim.operation != Operation.ATOMIC and claim.kind == ClaimKind.WITNESS:
        # Witness claims describe an operation but do not themselves perform it.
        pass
    if issues:
        return GateResult(
            "SCHEMA", claim.claim_id, Verdict.FAIL, "; ".join(issues)
        )
    return GateResult(
        "SCHEMA", claim.claim_id, Verdict.PASS,
        "required typed fields and canonical hash are present",
    )


def gate_domain(claim: Claim, graph: Graph) -> GateResult:
    parent_domain = claim.deployment_regime.domain
    if not parent_domain.parameters:
        return GateResult("DOMAIN", claim.claim_id, Verdict.NA, "no domain")
    issues: list[str] = []
    for edge in _support_edges(claim):
        dep = graph[edge.target]
        dep_domain = dep.establishment_regime.domain
        gaps = dep_domain.missing_or_narrow(parent_domain)
        if gaps:
            bridges = _typed_bridge_nodes(
                claim, graph, EdgeKind.DOMAIN_TRANSFER, BridgeKind.DOMAIN
            )
            if not any(
                _bridge_matches(
                    bridge,
                    source=dep.establishment_regime,
                    target=claim.deployment_regime,
                    parent_domain=parent_domain,
                )
                and _node_grade_admissible_for_parent(bridge, claim)
                for bridge in bridges
            ):
                issues.append(f"{dep.claim_id}: " + ", ".join(gaps))
    if issues:
        return GateResult(
            "DOMAIN", claim.claim_id, Verdict.FAIL,
            "domain not covered: " + "; ".join(issues),
        )
    return GateResult(
        "DOMAIN", claim.claim_id, Verdict.PASS,
        "every load-bearing dependency covers the full parent domain",
    )


def gate_endpoint(claim: Claim, graph: Graph) -> GateResult:
    if not claim.finite_range_bound:
        return GateResult("ENDPOINT", claim.claim_id, Verdict.NA, "not finite-range")
    if not claim.registered_endpoints:
        return GateResult(
            "ENDPOINT", claim.claim_id, Verdict.FAIL,
            "finite-range claim has no registered endpoint certificates",
        )
    bad = [
        endpoint.point
        for endpoint in claim.registered_endpoints
        if not endpoint.satisfies or not endpoint.evidence_id
    ]
    if bad:
        return GateResult(
            "ENDPOINT", claim.claim_id, Verdict.FAIL,
            "failed or unevidenced endpoint(s): " + ", ".join(bad),
        )
    return GateResult(
        "ENDPOINT", claim.claim_id, Verdict.PASS,
        "all registered endpoints satisfy the declared inequality",
    )


def gate_polarity(claim: Claim, graph: Graph) -> GateResult:
    if claim.direction not in {Direction.UPPER, Direction.LOWER}:
        return GateResult("POLARITY", claim.claim_id, Verdict.NA, "not one-sided")
    if claim.direction == Direction.UPPER and claim.uncertainty_side not in {
        UncertaintySide.NONE,
        UncertaintySide.UPPER,
    }:
        return GateResult(
            "POLARITY", claim.claim_id, Verdict.FAIL,
            "upper claim does not carry upper one-sided uncertainty",
        )
    if claim.direction == Direction.LOWER and claim.uncertainty_side not in {
        UncertaintySide.NONE,
        UncertaintySide.LOWER,
    }:
        return GateResult(
            "POLARITY", claim.claim_id, Verdict.FAIL,
            "lower claim does not carry lower one-sided uncertainty",
        )
    bad: list[str] = []
    for edge in _support_edges(claim):
        dep = graph[edge.target]
        if dep.direction in {Direction.UPPER, Direction.LOWER}:
            if dep.direction != claim.direction:
                bad.append(f"{dep.claim_id}:{dep.direction.value}")
    if bad:
        return GateResult(
            "POLARITY", claim.claim_id, Verdict.FAIL,
            "opposite-sided support: " + ", ".join(bad),
        )
    return GateResult(
        "POLARITY", claim.claim_id, Verdict.PASS,
        "sidedness and uncertainty direction are compatible",
    )


def gate_measure(claim: Claim, graph: Graph) -> GateResult:
    target = claim.deployment_regime
    if not target.measure_id:
        return GateResult("MEASURE", claim.claim_id, Verdict.NA, "no measure")
    failures: list[str] = []
    for edge in _support_edges(claim):
        dep = graph[edge.target]
        source = dep.establishment_regime
        if not source.measure_id or source.measure_id == target.measure_id:
            continue
        bridges = _typed_bridge_nodes(
            claim, graph, EdgeKind.CHANGE_OF_MEASURE, BridgeKind.MEASURE
        )
        if not any(
            _bridge_matches(
                bridge,
                source=source,
                target=target,
                parent_domain=target.domain,
            )
            and _node_grade_admissible_for_parent(bridge, claim)
            for bridge in bridges
        ):
            failures.append(
                f"{dep.claim_id}:{source.measure_id}->{target.measure_id}"
            )
    if failures:
        return GateResult(
            "MEASURE", claim.claim_id, Verdict.FAIL,
            "unbridged measure change: " + ", ".join(failures),
        )
    return GateResult(
        "MEASURE", claim.claim_id, Verdict.PASS,
        "all measure changes are absent or typed and graded",
    )


def gate_mark(claim: Claim, graph: Graph) -> GateResult:
    if not claim.required_marks:
        return GateResult("MARK", claim.claim_id, Verdict.NA, "no required marks")
    supports = _support_edges(claim)
    supplied: set[str] = set()
    if not supports:
        supplied.update(claim.supplied_marks)
    else:
        for edge in supports:
            supplied.update(graph[edge.target].supplied_marks)
    missing = frozenset(claim.required_marks - supplied)
    if not missing:
        return GateResult(
            "MARK", claim.claim_id, Verdict.PASS,
            "required marks are supplied by evidence dependencies",
        )
    bridges = _typed_bridge_nodes(
        claim, graph, EdgeKind.MARK_TRANSFER, BridgeKind.MARK
    )
    if any(
        _bridge_matches(
            bridge,
            source=bridge.bridge_spec.source_regime,  # type: ignore[union-attr]
            target=claim.deployment_regime,
            parent_domain=claim.deployment_regime.domain,
            missing_marks=missing,
        )
        and _node_grade_admissible_for_parent(bridge, claim)
        for bridge in bridges
    ):
        return GateResult(
            "MARK", claim.claim_id, Verdict.PASS,
            "missing evidence marks are supplied by a typed mark-transfer node",
        )
    return GateResult(
        "MARK", claim.claim_id, Verdict.FAIL,
        "marks absent from evidence cone: " + ", ".join(sorted(missing)),
    )


def gate_composition(claim: Claim, graph: Graph) -> GateResult:
    if claim.operation == Operation.ATOMIC:
        return GateResult(
            "COMPOSITION", claim.claim_id, Verdict.NA, "atomic claim"
        )
    allowed = _ALLOWED_WITNESSES[claim.operation]
    witness_edges = _edges_of_kind(claim, EdgeKind.COMPOSITION_WITNESS)
    if not witness_edges:
        return GateResult(
            "COMPOSITION", claim.claim_id, Verdict.FAIL,
            f"{claim.operation.value} has no witness node",
        )
    wrong: list[str] = []
    for edge in witness_edges:
        node = graph[edge.target]
        spec = node.witness_spec
        if (
            node.kind == ClaimKind.WITNESS
            and spec is not None
            and spec.operation == claim.operation
            and spec.witness_type in allowed
            and _node_grade_admissible_for_parent(node, claim)
        ):
            return GateResult(
                "COMPOSITION", claim.claim_id, Verdict.PASS,
                f"{claim.operation.value} licensed by {node.claim_id}:"
                f"{spec.witness_type.value}",
                node.claim_id,
            )
        wrong.append(node.claim_id)
    return GateResult(
        "COMPOSITION", claim.claim_id, Verdict.FAIL,
        "witness nodes are absent, weak, or operation-incompatible: "
        + ", ".join(wrong),
    )


def gate_rung(claim: Claim, graph: Graph) -> GateResult:
    if not claim.measured_values:
        return GateResult("RUNG", claim.claim_id, Verdict.NA, "no measured values")
    missing = [
        value.name
        for value in claim.measured_values
        if not value.scale_id or not value.evidence_id
    ]
    if missing:
        return GateResult(
            "RUNG", claim.claim_id, Verdict.FAIL,
            "measured value lacks scale/evidence: " + ", ".join(missing),
        )
    target_scale = claim.deployment_regime.scale_id
    mismatches: list[str] = []
    for edge in _support_edges(claim):
        dep = graph[edge.target]
        if not dep.measured_values:
            continue
        for value in dep.measured_values:
            if target_scale and value.scale_id != target_scale:
                mismatches.append(
                    f"{dep.claim_id}:{value.name}:{value.scale_id}->{target_scale}"
                )
    if mismatches:
        bridges = _typed_bridge_nodes(
            claim, graph, EdgeKind.SCALE_TRANSFER, BridgeKind.SCALE
        )
        if not bridges:
            return GateResult(
                "RUNG", claim.claim_id, Verdict.FAIL,
                "scale mismatch without typed bridge: " + ", ".join(mismatches),
            )
    return GateResult(
        "RUNG", claim.claim_id, Verdict.PASS,
        "all measured values carry evidence and scale tags",
    )


def gate_precedence(claim: Claim, graph: Graph) -> GateResult:
    dead_statuses = {
        ClaimStatus.KILLED,
        ClaimStatus.SUPERSEDED,
        ClaimStatus.RETIRED,
    }
    dead = [
        claim_id
        for claim_id in reachable(claim.claim_id, graph)
        if graph[claim_id].status in dead_statuses
    ]
    if dead:
        return GateResult(
            "PRECEDENCE", claim.claim_id, Verdict.FAIL,
            "root or dependency is dead: " + ", ".join(sorted(dead)),
        )
    return GateResult(
        "PRECEDENCE", claim.claim_id, Verdict.PASS,
        "no killed, superseded, or retired node is reachable",
    )


def gate_core_closure(claim: Claim, graph: Graph) -> GateResult:
    if not claim.closed_core:
        return GateResult(
            "CORE-CLOSURE", claim.claim_id, Verdict.NA, "not declared closed core"
        )
    unresolved: list[str] = []
    for claim_id in reachable(claim.claim_id, graph):
        node = graph[claim_id]
        if node.status == ClaimStatus.OPEN:
            unresolved.append(claim_id)
        if node.warrant in {Warrant.CONJECTURE, Warrant.PLAUSIBLE}:
            unresolved.append(claim_id)
    unresolved.extend(claim.conditional_on)
    open_active = [
        gate.gate_id
        for gate in claim.active_gates
        if gate.blocks_promotion and gate.status != ActiveGateStatus.CLOSED
    ]
    if unresolved or open_active:
        return GateResult(
            "CORE-CLOSURE", claim.claim_id, Verdict.FAIL,
            "closed core retains unresolved objects/gates: "
            + ", ".join(sorted(set(unresolved + open_active))),
        )
    return GateResult(
        "CORE-CLOSURE", claim.claim_id, Verdict.PASS,
        "closed core contains no unresolved dependency or active gate",
    )


def gate_model(claim: Claim, graph: Graph) -> GateResult:
    target = claim.deployment_regime
    if not target.model_id:
        return GateResult("MODEL", claim.claim_id, Verdict.NA, "no model")
    failures: list[str] = []
    for edge in _support_edges(claim):
        dep = graph[edge.target]
        source = dep.establishment_regime
        if not source.model_id or source.model_id == target.model_id:
            continue
        bridges = _typed_bridge_nodes(
            claim, graph, EdgeKind.MODEL_TRANSFER, BridgeKind.MODEL
        )
        if not any(
            _bridge_matches(
                bridge,
                source=source,
                target=target,
                parent_domain=target.domain,
            )
            and _node_grade_admissible_for_parent(bridge, claim)
            for bridge in bridges
        ):
            failures.append(
                f"{dep.claim_id}:{source.model_id}->{target.model_id}"
            )
    if failures:
        return GateResult(
            "MODEL", claim.claim_id, Verdict.FAIL,
            "unbridged model transfer: " + ", ".join(failures),
        )
    return GateResult(
        "MODEL", claim.claim_id, Verdict.PASS,
        "exact model is stable or every transfer is typed and scoped",
    )


def gate_coverage(claim: Claim, graph: Graph) -> GateResult:
    certificates = _certificate_nodes(
        claim, graph, CertificateRole.COVERAGE
    )
    if not certificates:
        return GateResult(
            "COVERAGE", claim.claim_id, Verdict.NA,
            "claim makes no registered global coverage assertion",
        )
    for node in certificates:
        spec = node.coverage_spec
        if spec is None:
            continue
        uncharted = [
            region for region, charts in spec.failure_regions if not charts
        ]
        try:
            uncovered = Decimal(spec.uncovered_mass)
            target = (
                None if spec.target_total_risk is None
                else Decimal(spec.target_total_risk)
            )
        except InvalidOperation:
            return GateResult(
                "COVERAGE", claim.claim_id, Verdict.FAIL,
                f"invalid coverage numeric field in {node.claim_id}",
                node.claim_id,
            )
        if not spec.universe_id or uncharted:
            return GateResult(
                "COVERAGE", claim.claim_id, Verdict.FAIL,
                "coverage universe missing or uncharted regions: "
                + ", ".join(uncharted),
                node.claim_id,
            )
        if uncovered < 0 or uncovered > 1:
            return GateResult(
                "COVERAGE", claim.claim_id, Verdict.FAIL,
                "uncovered mass outside [0,1]",
                node.claim_id,
            )
        if target is not None and target < uncovered:
            return GateResult(
                "COVERAGE", claim.claim_id, Verdict.FAIL,
                f"target risk {target} lies below coverage floor {uncovered}",
                node.claim_id,
            )
        return GateResult(
            "COVERAGE", claim.claim_id, Verdict.PASS,
            "registered failure universe is charted with explicit uncovered mass",
            node.claim_id,
        )
    return GateResult(
        "COVERAGE", claim.claim_id, Verdict.FAIL,
        "coverage certificate edge does not point to a coverage certificate",
    )


def gate_common_mode(claim: Claim, graph: Graph) -> GateResult:
    if claim.verification_mode != VerificationMode.AGREEMENT:
        return GateResult(
            "COMMON-MODE", claim.claim_id, Verdict.NA,
            "promotion does not rest on instrument agreement",
        )
    certificates = _certificate_nodes(
        claim, graph, CertificateRole.COMMON_MODE
    )
    if not certificates:
        return GateResult(
            "COMMON-MODE", claim.claim_id, Verdict.PROVISIONAL,
            "agreement has no first-class common-mode certificate",
            grade_cap=Warrant.DERIVED,
        )
    for node in certificates:
        spec = node.common_mode_spec
        if spec is None:
            continue
        if spec.relation == InstrumentRelation.INDEPENDENT:
            if (
                len(spec.instrument_ids) >= 2
                and spec.error_channel_under_test
                and spec.disjoint_error_argument
            ):
                return GateResult(
                    "COMMON-MODE", claim.claim_id, Verdict.PASS,
                    "disjoint error channels are explicitly certified",
                    node.claim_id,
                )
        if spec.relation == InstrumentRelation.EXTERNAL_CROSSCHECK:
            if spec.external_crosscheck_id:
                return GateResult(
                    "COMMON-MODE", claim.claim_id, Verdict.PASS,
                    "external contradiction/crosscheck certificate is present",
                    node.claim_id,
                )
    return GateResult(
        "COMMON-MODE", claim.claim_id, Verdict.PROVISIONAL,
        "agreement retains shared or unknown error channels",
        grade_cap=Warrant.DERIVED,
    )


def gate_heuristic_bridge(claim: Claim, graph: Graph) -> GateResult:
    if (
        claim.reasoning_mode
        not in {ReasoningMode.ANALOGICAL, ReasoningMode.HEURISTIC}
        or not claim.load_bearing
    ):
        return GateResult(
            "HEURISTIC-BRIDGE", claim.claim_id, Verdict.NA,
            "no load-bearing heuristic or analogy",
        )
    bridges = _typed_bridge_nodes(
        claim, graph, EdgeKind.HEURISTIC_BRIDGE, BridgeKind.HEURISTIC
    )
    if any(
        _bridge_matches(
            bridge,
            source=bridge.bridge_spec.source_regime,  # type: ignore[union-attr]
            target=bridge.bridge_spec.target_regime,  # type: ignore[union-attr]
            parent_domain=claim.deployment_regime.domain,
        )
        and _node_grade_admissible_for_parent(bridge, claim)
        for bridge in bridges
    ):
        return GateResult(
            "HEURISTIC-BRIDGE", claim.claim_id, Verdict.PASS,
            "load-bearing analogy has a typed, graded derivation bridge",
        )
    if claim.warrant in {Warrant.PLAUSIBLE, Warrant.CONJECTURE}:
        return GateResult(
            "HEURISTIC-BRIDGE", claim.claim_id, Verdict.PROVISIONAL,
            "unbridged heuristic retained at or below Plausible",
            grade_cap=Warrant.PLAUSIBLE,
        )
    return GateResult(
        "HEURISTIC-BRIDGE", claim.claim_id, Verdict.FAIL,
        "load-bearing heuristic supports a claim above Plausible without a bridge",
        grade_cap=Warrant.PLAUSIBLE,
    )


def gate_assembly(claim: Claim, graph: Graph) -> GateResult:
    if claim.operation != Operation.AFFINE_ASSEMBLY and claim.assembly_spec is None:
        return GateResult(
            "ASSEMBLY", claim.claim_id, Verdict.NA, "no numerical assembly"
        )
    spec = claim.assembly_spec
    if spec is None:
        return GateResult(
            "ASSEMBLY", claim.claim_id, Verdict.FAIL,
            "AFFINE-ASSEMBLY claim lacks assembly_spec",
        )
    try:
        names = {item.name for item in spec.inputs}
        required = set(spec.required_terms) | set(spec.residual_terms)
        absent = required - names
        used = expression_names(spec.expression)
        unused = required - used
        source_missing = [
            item.name for item in spec.inputs if not item.source_id
        ]
        if absent:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                "required/residual inputs absent: " + ", ".join(sorted(absent)),
            )
        if unused:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                "required/residual inputs unused: " + ", ".join(sorted(unused)),
            )
        if source_missing:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                "assembly input lacks source_id: "
                + ", ".join(sorted(source_missing)),
            )
        computed = evaluate_assembly(spec)
        displayed = Decimal(spec.displayed_value)
    except Exception as exc:
        return GateResult(
            "ASSEMBLY", claim.claim_id, Verdict.FAIL,
            f"assembly evaluation failed: {exc}",
        )

    if spec.direction == Direction.UPPER:
        if spec.rounding not in {
            RoundingRule.UP,
            RoundingRule.OUTWARD_INTERVAL,
            RoundingRule.EXACT,
        }:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                "upper assembly lacks upward/outward rounding",
            )
        if displayed < computed:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                f"displayed upper {displayed} < recomputed {computed}",
            )
    elif spec.direction == Direction.LOWER:
        if spec.rounding not in {
            RoundingRule.DOWN,
            RoundingRule.OUTWARD_INTERVAL,
            RoundingRule.EXACT,
        }:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                "lower assembly lacks downward/outward rounding",
            )
        if displayed > computed:
            return GateResult(
                "ASSEMBLY", claim.claim_id, Verdict.FAIL,
                f"displayed lower {displayed} > recomputed {computed}",
            )
    elif spec.direction == Direction.EQUALITY and displayed != computed:
        return GateResult(
            "ASSEMBLY", claim.claim_id, Verdict.FAIL,
            f"displayed equality {displayed} != recomputed {computed}",
        )
    return GateResult(
        "ASSEMBLY", claim.claim_id, Verdict.PASS,
        f"displayed {displayed} conservatively encloses recomputed {computed}",
    )


def gate_domain_infimum(claim: Claim, graph: Graph) -> GateResult:
    requires = (
        claim.quantifier == Quantifier.FOR_ALL
        and claim.direction in {Direction.UPPER, Direction.LOWER}
        and claim.bound_coefficient is not None
        and bool(claim.deployment_regime.domain.parameters)
    )
    if not requires:
        return GateResult(
            "DOMAIN-INFIMUM", claim.claim_id, Verdict.NA,
            "claim does not require a numerical full-domain extremum",
        )
    certificates = _certificate_nodes(
        claim, graph, CertificateRole.DOMAIN_EXTREMUM
    )
    if not certificates:
        return GateResult(
            "DOMAIN-INFIMUM", claim.claim_id, Verdict.FAIL,
            "uniform numerical claim lacks an extremum certificate node",
        )
    bound = Decimal(claim.bound_coefficient)
    for node in certificates:
        spec = node.extremum_spec
        if spec is None:
            continue
        if spec.certificate_type == ExtremumCertificateType.RUNG_SAMPLE:
            continue
        if not spec.domain.contains(claim.deployment_regime.domain):
            continue
        value = Decimal(spec.extremum_value)
        if (
            claim.direction == Direction.UPPER
            and spec.extremum_kind == ExtremumKind.SUPREMUM
            and value <= bound
        ):
            return GateResult(
                "DOMAIN-INFIMUM", claim.claim_id, Verdict.PASS,
                "full-domain supremum certificate supports the upper coefficient",
                node.claim_id,
            )
        if (
            claim.direction == Direction.LOWER
            and spec.extremum_kind == ExtremumKind.INFIMUM
            and value >= bound
        ):
            return GateResult(
                "DOMAIN-INFIMUM", claim.claim_id, Verdict.PASS,
                "full-domain infimum certificate supports the lower coefficient",
                node.claim_id,
            )
    return GateResult(
        "DOMAIN-INFIMUM", claim.claim_id, Verdict.FAIL,
        "no certificate covers the full domain in the required direction",
    )


def gate_band_provenance(claim: Claim, graph: Graph) -> GateResult:
    if claim.uncertainty_side == UncertaintySide.NONE:
        return GateResult(
            "BAND-PROVENANCE", claim.claim_id, Verdict.NA,
            "no load-bearing uncertainty band"
        )
    certificates = _certificate_nodes(
        claim, graph, CertificateRole.BAND_PROVENANCE
    )
    if not certificates:
        return GateResult(
            "BAND-PROVENANCE", claim.claim_id, Verdict.FAIL,
            "uncertainty band lacks a provenance certificate node",
        )
    for node in certificates:
        spec = node.band_spec
        if spec is None:
            continue
        if not spec.construction:
            continue
        if not spec.coverage_domain.contains(claim.deployment_regime.domain):
            continue
        if spec.kind in {
            BandKind.STATISTICAL,
            BandKind.BOOTSTRAP,
            BandKind.MONTE_CARLO,
        } and (
            spec.confidence_level is None
            or spec.sample_size is None
            or not spec.multiplicity_scope
        ):
            continue
        if spec.kind == BandKind.RANDOMIZED_NUMERICAL and (
            not spec.seed_or_algorithm_state
            or not spec.deterministic_tolerance
        ):
            continue
        if spec.kind in {BandKind.DETERMINISTIC, BandKind.INTERVAL} and (
            not spec.deterministic_tolerance
        ):
            continue
        if spec.kind == BandKind.FORECAST and claim.warrant not in {
            Warrant.PLAUSIBLE,
            Warrant.CONJECTURE,
        }:
            continue
        consumed = (
            spec.consumed_as
            if spec.consumed_as != UncertaintySide.NONE
            else spec.side
        )
        if claim.direction == Direction.UPPER and consumed != UncertaintySide.UPPER:
            continue
        if claim.direction == Direction.LOWER and consumed != UncertaintySide.LOWER:
            continue
        return GateResult(
            "BAND-PROVENANCE", claim.claim_id, Verdict.PASS,
            f"{spec.kind.value} provenance is complete and domain-covering",
            node.claim_id,
        )
    return GateResult(
        "BAND-PROVENANCE", claim.claim_id, Verdict.FAIL,
        "no complete band provenance certificate supports the claim",
    )


def gate_active(claim: Claim) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in claim.active_gates:
        if gate.status == ActiveGateStatus.CLOSED:
            results.append(
                GateResult(
                    gate.gate_id, claim.claim_id, Verdict.PASS,
                    "domain-specific gate is closed", gate.evidence_id,
                )
            )
        elif gate.status == ActiveGateStatus.NOT_CLAIMED:
            results.append(
                GateResult(
                    gate.gate_id, claim.claim_id, Verdict.NA,
                    "claim explicitly does not consume this gate",
                    gate.evidence_id,
                )
            )
        elif gate.blocks_promotion and claim.load_bearing:
            results.append(
                GateResult(
                    gate.gate_id, claim.claim_id, Verdict.FAIL,
                    f"active gate status is {gate.status.value}",
                    gate.evidence_id,
                )
            )
        else:
            results.append(
                GateResult(
                    gate.gate_id, claim.claim_id, Verdict.PROVISIONAL,
                    f"active gate status is {gate.status.value}",
                    gate.evidence_id,
                    Warrant.PLAUSIBLE,
                )
            )
    return results


# ============================================================================
# Dependency/condition propagation and graph evaluation
# ============================================================================

def _conditional_debts(
    claim: Claim,
    graph: Graph,
    prior_reports: Mapping[str, ClaimReport],
) -> tuple[str, ...]:
    conditions: set[str] = set(claim.conditional_on)
    for edge in claim.edges:
        dep = graph[edge.target]
        report = prior_reports.get(dep.claim_id)
        if edge.conditional or edge.kind == EdgeKind.CONDITION:
            conditions.add(dep.claim_id)
        if dep.status == ClaimStatus.OPEN:
            conditions.add(dep.claim_id)
        if dep.warrant in {Warrant.CONJECTURE, Warrant.PLAUSIBLE}:
            conditions.add(dep.claim_id)
        if report is not None:
            conditions.update(report.conditions)
    return tuple(sorted(conditions))


def _dependency_results(
    claim: Claim,
    graph: Graph,
    prior_reports: Mapping[str, ClaimReport],
) -> list[GateResult]:
    results: list[GateResult] = []
    declared_conditions = set(claim.conditional_on)
    for edge in claim.edges:
        dep = graph[edge.target]
        dep_report = prior_reports.get(dep.claim_id)
        conditional = edge.conditional or edge.kind == EdgeKind.CONDITION

        if conditional:
            if claim.warrant != Warrant.PROVEN_MODULO:
                results.append(
                    GateResult(
                        "DEPENDENCY-WARRANT", claim.claim_id, Verdict.FAIL,
                        f"conditional dependency {dep.claim_id} requires "
                        "parent warrant PROVEN-MODULO",
                        dep.claim_id,
                    )
                )
            if dep.claim_id not in declared_conditions:
                results.append(
                    GateResult(
                        "DEPENDENCY-WARRANT", claim.claim_id, Verdict.FAIL,
                        f"conditional dependency {dep.claim_id} is not named "
                        "in conditional_on",
                        dep.claim_id,
                    )
                )
            continue

        if not _node_grade_admissible_for_parent(dep, claim):
            results.append(
                GateResult(
                    "DEPENDENCY-WARRANT", claim.claim_id, Verdict.FAIL,
                    f"{dep.claim_id}:{dep.warrant.value} cannot support "
                    f"{claim.warrant.value}",
                    dep.claim_id,
                )
            )
        if dep_report is not None:
            if dep_report.conditions and claim.warrant != Warrant.PROVEN_MODULO:
                results.append(
                    GateResult(
                        "DEPENDENCY-WARRANT", claim.claim_id, Verdict.FAIL,
                        f"unconditional parent inherits unresolved conditions "
                        f"from {dep.claim_id}",
                        dep.claim_id,
                    )
                )
            if (
                _WARRANT_RANK[dep_report.effective_warrant]
                < _WARRANT_RANK[claim.warrant]
                and claim.warrant not in {
                    Warrant.PLAUSIBLE,
                    Warrant.CONJECTURE,
                }
            ):
                results.append(
                    GateResult(
                        "DEPENDENCY-WARRANT", claim.claim_id, Verdict.FAIL,
                        f"parent warrant exceeds effective dependency ceiling "
                        f"{dep.claim_id}:{dep_report.effective_warrant.value}",
                        dep.claim_id,
                        dep_report.effective_warrant,
                    )
                )
    if not results:
        results.append(
            GateResult(
                "DEPENDENCY-WARRANT", claim.claim_id, Verdict.PASS,
                "dependency warrants and explicit condition edges are admissible",
            )
        )
    return results


def _effective_warrant(
    claim: Claim,
    results: Sequence[GateResult],
    graph: Graph,
    prior_reports: Mapping[str, ClaimReport],
) -> Warrant:
    effective = claim.warrant
    for result in results:
        if (
            result.verdict == Verdict.PROVISIONAL
            and result.grade_cap is not None
        ):
            effective = _minimum_warrant(effective, result.grade_cap)
    for edge in claim.edges:
        if edge.conditional or edge.kind == EdgeKind.CONDITION:
            continue
        dep_report = prior_reports.get(edge.target)
        if dep_report is not None:
            effective = _minimum_warrant(
                effective, dep_report.effective_warrant
            )
    return effective


NUMBERED_GATES = (
    "DOMAIN",
    "ENDPOINT",
    "POLARITY",
    "MEASURE",
    "MARK",
    "COMPOSITION",
    "RUNG",
    "PRECEDENCE",
    "CORE-CLOSURE",
    "MODEL",
    "COVERAGE",
    "COMMON-MODE",
    "HEURISTIC-BRIDGE",
    "ASSEMBLY",
    "DOMAIN-INFIMUM",
    "BAND-PROVENANCE",
)


def evaluate_graph(graph: Graph, roots: Sequence[str]) -> GraphReport:
    errors = validate_graph(graph)
    if errors:
        return GraphReport(
            roots=tuple(roots),
            graph_errors=errors,
            claim_reports={},
            root_hashes={},
            shell_valid=False,
            conditional_promotable=False,
            unconditional_promotable=False,
        )
    for root in roots:
        if root not in graph:
            errors.append(f"missing root {root}")
    if errors:
        return GraphReport(
            roots=tuple(roots),
            graph_errors=errors,
            claim_reports={},
            root_hashes={},
            shell_valid=False,
            conditional_promotable=False,
            unconditional_promotable=False,
        )

    reports: dict[str, ClaimReport] = {}
    for claim_id in topological_order(graph):
        claim = graph[claim_id]
        results = [
            gate_schema(claim, graph),
            gate_domain(claim, graph),
            gate_endpoint(claim, graph),
            gate_polarity(claim, graph),
            gate_measure(claim, graph),
            gate_mark(claim, graph),
            gate_composition(claim, graph),
            gate_rung(claim, graph),
            gate_precedence(claim, graph),
            gate_core_closure(claim, graph),
            gate_model(claim, graph),
            gate_coverage(claim, graph),
            gate_common_mode(claim, graph),
            gate_heuristic_bridge(claim, graph),
            gate_assembly(claim, graph),
            gate_domain_infimum(claim, graph),
            gate_band_provenance(claim, graph),
        ]
        results.extend(gate_active(claim))
        results.extend(_dependency_results(claim, graph, reports))
        effective = _effective_warrant(claim, results, graph, reports)
        conditions = _conditional_debts(claim, graph, reports)
        fail = any(result.verdict == Verdict.FAIL for result in results)
        provisional = any(
            result.verdict == Verdict.PROVISIONAL for result in results
        )
        shell_valid = not fail
        conditional_promotable = (
            shell_valid
            and not provisional
            and (
                not conditions
                or (
                    claim.warrant == Warrant.PROVEN_MODULO
                    and set(conditions).issubset(set(claim.conditional_on))
                )
            )
            and claim.status == ClaimStatus.LIVE
        )
        unconditional_promotable = (
            conditional_promotable and not conditions
        )
        closable = (
            shell_valid
            and not provisional
            and not conditions
            and claim.status == ClaimStatus.LIVE
        )
        reports[claim_id] = ClaimReport(
            claim_id=claim_id,
            gate_results=results,
            effective_warrant=effective,
            conditions=conditions,
            shell_valid=shell_valid,
            conditional_promotable=conditional_promotable,
            unconditional_promotable=unconditional_promotable,
            closable=closable,
        )

    root_reports = [reports[root] for root in roots]
    shell_valid = all(report.shell_valid for report in root_reports)
    conditional_promotable = all(
        report.conditional_promotable for report in root_reports
    )
    unconditional_promotable = all(
        report.unconditional_promotable for report in root_reports
    )
    root_hashes = {
        root: merkle_hash(root, graph) for root in roots
    } if shell_valid else {}

    return GraphReport(
        roots=tuple(roots),
        graph_errors=[],
        claim_reports=reports,
        root_hashes=root_hashes,
        shell_valid=shell_valid,
        conditional_promotable=conditional_promotable,
        unconditional_promotable=unconditional_promotable,
    )


def merkle_hash(root: str, graph: Graph) -> str:
    memo: dict[str, str] = {}

    def visit(claim_id: str) -> str:
        if claim_id in memo:
            return memo[claim_id]
        child_hashes = sorted(
            visit(edge.target) for edge in graph[claim_id].edges
        )
        payload = graph[claim_id].content_hash + "|" + "|".join(child_hashes)
        memo[claim_id] = sha256(payload.encode("utf-8")).hexdigest()
        return memo[claim_id]

    return visit(root)


# ============================================================================
# Serialization and strict registry
# ============================================================================

def graph_to_dict(
    graph: Graph,
    *,
    roots: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "roots": list(roots),
        "metadata": dict(metadata or {}),
        "claims": [
            graph[claim_id].as_dict()
            for claim_id in topological_order(graph)
        ],
    }


def graph_to_json(
    graph: Graph,
    *,
    roots: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
    indent: int = 2,
) -> str:
    return json.dumps(
        graph_to_dict(graph, roots=roots, metadata=metadata),
        indent=indent,
        ensure_ascii=False,
    )


def graph_from_dict(data: Mapping[str, Any]) -> tuple[Graph, tuple[str, ...], dict]:
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema {data.get('schema')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    claims = [
        Claim.from_dict(item, verify_hash=True)
        for item in data.get("claims", [])
    ]
    graph = {claim.claim_id: claim for claim in claims}
    if len(graph) != len(claims):
        raise ValueError("duplicate claim IDs")
    errors = validate_graph(graph)
    if errors:
        raise ValueError("invalid graph:\n" + "\n".join(errors))
    roots = tuple(data.get("roots", []))
    for root in roots:
        if root not in graph:
            raise ValueError(f"missing root {root}")
    return graph, roots, dict(data.get("metadata", {}))


def graph_from_json(text: str) -> tuple[Graph, tuple[str, ...], dict]:
    return graph_from_dict(json.loads(text))


def save_graph(
    graph: Graph,
    path: str | Path,
    *,
    roots: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    Path(path).write_text(
        graph_to_json(graph, roots=roots, metadata=metadata),
        encoding="utf-8",
    )


def load_graph(path: str | Path) -> tuple[Graph, tuple[str, ...], dict]:
    return graph_from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class Registry:
    graph: Graph
    roots: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    report: Optional[GraphReport] = None

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        graph, roots, metadata = load_graph(path)
        report = evaluate_graph(graph, roots)
        if report.graph_errors:
            raise ValueError(
                "registry graph errors:\n" + "\n".join(report.graph_errors)
            )
        return cls(graph, roots, metadata, report)

    def save(self, path: str | Path) -> None:
        save_graph(
            self.graph,
            path,
            roots=self.roots,
            metadata=self.metadata,
        )

    def admit(self, mode: AdmissionMode) -> tuple[bool, list[str]]:
        report = self.report or evaluate_graph(self.graph, self.roots)
        if report.graph_errors:
            return False, report.graph_errors
        if mode == AdmissionMode.ARCHIVE:
            return True, ["structurally valid archive"]
        if mode == AdmissionMode.CANDIDATE:
            if report.shell_valid:
                return True, ["shell-valid candidate"]
            return False, ["candidate shell failed"]
        if mode == AdmissionMode.PROMOTED:
            if report.conditional_promotable:
                if report.unconditional_promotable:
                    return True, ["unconditionally promotable"]
                return True, ["promotable only as an explicitly conditional theorem"]
            return False, ["root is not promotable at its stated warrant"]
        raise ValueError(mode)


# ============================================================================
# Constructors and demo
# ============================================================================

def interval(
    lower: str | float,
    upper: str | float,
    *,
    lower_open: bool = False,
    upper_open: bool = False,
) -> Interval:
    return Interval(
        str(lower), str(upper), lower_open=lower_open, upper_open=upper_open
    )


def domain(**parameters: Interval) -> ParameterDomain:
    return ParameterDomain(tuple(sorted(parameters.items())))


def seal_graph_unpinned(claims: Iterable[Claim]) -> Graph:
    """
    Seal a DAG whose edges name dependencies but have blank pinned hashes.

    Claims must be supplied in dependency-before-parent order. Each claim is
    sealed after its edges are pinned to already sealed dependencies.
    """
    graph: Graph = {}
    for claim in claims:
        if claim.edges:
            claim = claim.with_pinned_edges(graph)
        else:
            claim = claim.sealed()
        graph[claim.claim_id] = claim
    errors = validate_graph(graph)
    if errors:
        raise ValueError("cannot seal graph:\n" + "\n".join(errors))
    return graph


def demo_second_domain() -> tuple[Graph, tuple[str, ...]]:
    d = domain(rho=interval("0", "0.8"))
    regime = Regime(
        model_id="FINITE-DIMENSIONAL-MATRIX-2-NORM",
        measure_id="DETERMINISTIC",
        domain=d,
        spatial_region="finite-dimensional matrices",
        scale_id="operator norm",
        conditioning_depth="none",
        marks=frozenset({"invertible-A", "perturbation-E"}),
        deployment_mode="exact",
    )
    witness = Claim(
        "exact_algebra",
        "Neumann-series algebra licenses the exact factorization.",
        ClaimKind.WITNESS,
        ClaimStatus.LIVE,
        Warrant.PROVEN,
        "GK2-DEMO",
        evidence_ids=("NEUMANN-SERIES-PROOF",),
        load_bearing=False,
        witness_spec=WitnessSpec(
            Operation.EXACT_IDENTITY,
            WitnessType.EXACT_ALGEBRA,
            statement="(A+E)^-1=(I+A^-1E)^-1 A^-1",
        ),
    )
    extremum = Claim(
        "rho_extremum",
        "The function 1/(1-rho) is increasing on [0,0.8].",
        ClaimKind.CERTIFICATE,
        ClaimStatus.LIVE,
        Warrant.PROVEN,
        "GK2-DEMO",
        evidence_ids=("MONOTONICITY-DERIVATION",),
        load_bearing=False,
        extremum_spec=ExtremumSpec(
            ExtremumCertificateType.MONOTONICITY,
            ExtremumKind.SUPREMUM,
            d,
            "5",
            "Derivative is positive for rho<1.",
        ),
    )
    theorem = Claim(
        "matrix_perturbation",
        "If ||A^-1 E||_2 <= rho <= 0.8, then "
        "||(A+E)^-1||_2 <= 5 ||A^-1||_2.",
        ClaimKind.STATEMENT,
        ClaimStatus.LIVE,
        Warrant.PROVEN,
        "GK2-DEMO",
        evidence_ids=("NEUMANN-SERIES-PROOF",),
        quantifier=Quantifier.FOR_ALL,
        direction=Direction.UPPER,
        bound_coefficient="5",
        establishment_regime=regime,
        deployment_regime=regime,
        required_marks=frozenset({"invertible-A", "perturbation-E"}),
        supplied_marks=frozenset({"invertible-A", "perturbation-E"}),
        operation=Operation.EXACT_IDENTITY,
        finite_range_bound=True,
        registered_endpoints=(
            EndpointSpec("rho=0", "1", True, "ENDPOINT-RHO0"),
            EndpointSpec("rho=0.8", "5", True, "ENDPOINT-RHO08"),
        ),
        assembly_spec=AssemblySpec(
            "one/(one-rho)",
            (
                AssemblyInput("one", "1", "NEUMANN-SERIES-PROOF", 1),
                AssemblyInput("rho", "0.8", "RHO-ENDPOINT", 1),
            ),
            ("one", "rho"),
            (),
            "5",
            Direction.UPPER,
            RoundingRule.EXACT,
        ),
        closed_core=True,
        edges=(
            Edge(
                "exact_algebra",
                EdgeKind.COMPOSITION_WITNESS,
            ),
            Edge(
                "rho_extremum",
                EdgeKind.CERTIFICATE,
                role=CertificateRole.DOMAIN_EXTREMUM.value,
            ),
        ),
    )
    graph = seal_graph_unpinned([witness, extremum, theorem])
    return graph, ("matrix_perturbation",)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-demo",
        type=Path,
        help="write a valid second-domain registry to this path",
    )
    args = parser.parse_args()

    graph, roots = demo_second_domain()
    report = evaluate_graph(graph, roots)
    print(json.dumps(report.as_dict(), indent=2))
    if not (
        report.shell_valid
        and report.conditional_promotable
        and report.unconditional_promotable
    ):
        raise SystemExit(1)
    if args.write_demo is not None:
        save_graph(
            graph,
            args.write_demo,
            roots=roots,
            metadata={"purpose": "Gate Kernel 2.0 second-domain demo"},
        )
        print(f"wrote {args.write_demo}")


if __name__ == "__main__":
    _main()
