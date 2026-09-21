#!/usr/bin/env python3
"""
q0_llm_verifier_v4.py

Final proof-carrying verifier prototype for the q0 Rate Program and its
LLM application.

v4 preserves the numerical/graph primitives from q0_llm_verifier_v3 and adds
semantic theorem-contract gates discovered by the C089--C091 audits:

  DOMAIN       A dependency must cover the parent claim's parameter domain.
  ENDPOINT     Every registered endpoint must satisfy a claimed bound.
  POLARITY     Lower/upper propagation and one-sided uncertainty must agree.
  MEASURE      Gaussian pinning, typed pinning, and pair-Palm laws may not be
               silently interchanged.
  MARK         An unmarked spatial point-process estimate cannot discharge a
               marked height/type estimate.
  COMPOSITION  Products of probabilities require a named dependence witness.
  RUNG         Measured values require explicit scale/parameter tags.
  PRECEDENCE   A superseded or killed claim may not be reachable from a live
               theorem root.
  CLOSURE      A declared finished core may not reach OPEN extension nodes.
  MODEL        Every covariance-dependent claim identifies its exact model or
               an explicit transfer certificate.
  COVERAGE     Exponential verifier rank cannot reduce uncovered wrong mass.

The file is a research prototype. It validates proof contracts; it does not
turn measured mathematical inputs or LLM detector scores into theorems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Reuse the already tested numerical and graph primitives.
from q0_llm_verifier_v3 import (
    Grade,
    ClaimNode,
    ProofGraph,
    SupportEdge,
    widest_path_grounding,
    anchored_rkhs_mismatch,
    anisotropic_sard_tube_bound,
    effective_transverse_rank,
    CoverageCertificate,
    SelectionAwareRiskLedger,
    coverage_limited_rank_requirement,
)


class Status(str, Enum):
    ESTABLISHED = "ESTABLISHED"
    DERIVED_EXACT = "DERIVED_EXACT"
    PROGRAM_GRADE_CLOSED = "PROGRAM_GRADE_CLOSED"
    EXTERNAL_REVIEW_PENDING = "EXTERNAL_REVIEW_PENDING"
    CONDITIONAL = "CONDITIONAL"
    MEASURED = "MEASURED"
    OPEN = "OPEN"
    KILLED = "KILLED"
    SUPERSEDED = "SUPERSEDED"


class Direction(str, Enum):
    LOWER = "LOWER"
    UPPER = "UPPER"
    EQUALITY = "EQUALITY"
    STRUCTURAL = "STRUCTURAL"


class Quantifier(str, Enum):
    EXACT_IDENTITY = "EXACT_IDENTITY"
    FOR_ALL = "FOR_ALL"
    ASYMPTOTIC = "ASYMPTOTIC"
    REGISTERED_RUNGS = "REGISTERED_RUNGS"
    MEASURED_AT_RUNG = "MEASURED_AT_RUNG"
    CONDITIONAL = "CONDITIONAL"


class Measure(str, Enum):
    NONE = "NONE"
    UNCONDITIONED = "UNCONDITIONED"
    GAUSSIAN_PINNED = "GAUSSIAN_PINNED"
    TYPED_GAUSSIAN_PINNED = "TYPED_GAUSSIAN_PINNED"
    PAIR_PALM = "PAIR_PALM"
    SELECTED_CONDITIONED = "SELECTED_CONDITIONED"


class UncertaintySide(str, Enum):
    NONE = "NONE"
    UPPER = "UPPER"
    LOWER = "LOWER"
    TWO_SIDED_CENTRAL = "TWO_SIDED_CENTRAL"


class CompositionWitness(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NONE = "NONE"
    INDEPENDENCE = "INDEPENDENCE"
    CONDITIONAL_INDEPENDENCE = "CONDITIONAL_INDEPENDENCE"
    MARKOV_PROPERTY = "MARKOV_PROPERTY"
    NEGATIVE_DEPENDENCE = "NEGATIVE_DEPENDENCE"
    COMPARISON_THEOREM = "COMPARISON_THEOREM"
    JOINT_CERTIFICATE = "JOINT_CERTIFICATE"
    UNION_BOUND = "UNION_BOUND"
    MONOTONICITY = "MONOTONICITY"
    EXACT_ALGEBRA = "EXACT_ALGEBRA"


@dataclass(frozen=True)
class ParameterDomain:
    """Closed numerical interval plus fixed parameter tags."""

    r_min: Optional[float] = None
    r_max: Optional[float] = None
    r_min_open: bool = False
    fixed: Tuple[Tuple[str, str], ...] = ()

    def contains(self, other: "ParameterDomain", tolerance: float = 1e-15) -> bool:
        if self.r_min is not None:
            if other.r_min is None or other.r_min < self.r_min - tolerance:
                return False
            if (
                self.r_min_open
                and abs(other.r_min - self.r_min) <= tolerance
                and not other.r_min_open
            ):
                return False
        if self.r_max is not None:
            if other.r_max is None or other.r_max > self.r_max + tolerance:
                return False
        own = dict(self.fixed)
        for key, value in other.fixed:
            if key in own and own[key] != value:
                return False
        return True

    def as_dict(self) -> dict:
        return {
            "r_min": self.r_min,
            "r_max": self.r_max,
            "r_min_open": self.r_min_open,
            "fixed": dict(self.fixed),
        }


@dataclass(frozen=True)
class SemanticClaim:
    claim_id: str
    statement: str
    status: Status
    direction: Direction = Direction.STRUCTURAL
    quantifier: Quantifier = Quantifier.CONDITIONAL
    dependencies: Tuple[str, ...] = ()
    domain: Optional[ParameterDomain] = None
    measure: Measure = Measure.NONE
    grade: Grade = Grade.HYPOTHESIS
    provided_marks: frozenset[str] = frozenset()
    required_marks: frozenset[str] = frozenset()
    operation: str = "atomic"
    composition_witness: CompositionWitness = CompositionWitness.NOT_APPLICABLE
    composition_evidence: Tuple[str, ...] = ()
    uncertainty_side: UncertaintySide = UncertaintySide.NONE
    measured_value: Optional[float] = None
    scale_tag: Optional[str] = None
    rung_values: Tuple[Tuple[float, float], ...] = ()
    bound_coefficient: Optional[float] = None
    model_id: Optional[str] = None
    model_bridge_id: Optional[str] = None
    measure_bridge_id: Optional[str] = None
    domain_bridge_id: Optional[str] = None
    supersedes: Tuple[str, ...] = ()
    source_ids: Tuple[str, ...] = ()
    program_core: bool = False
    notes: Tuple[str, ...] = ()

    def digest(self) -> str:
        payload = {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "status": self.status.value,
            "direction": self.direction.value,
            "quantifier": self.quantifier.value,
            "dependencies": sorted(self.dependencies),
            "domain": None if self.domain is None else self.domain.as_dict(),
            "measure": self.measure.value,
            "grade": int(self.grade),
            "provided_marks": sorted(self.provided_marks),
            "required_marks": sorted(self.required_marks),
            "operation": self.operation,
            "composition_witness": self.composition_witness.value,
            "composition_evidence": sorted(self.composition_evidence),
            "uncertainty_side": self.uncertainty_side.value,
            "measured_value": self.measured_value,
            "scale_tag": self.scale_tag,
            "rung_values": list(self.rung_values),
            "bound_coefficient": self.bound_coefficient,
            "model_id": self.model_id,
            "model_bridge_id": self.model_bridge_id,
            "measure_bridge_id": self.measure_bridge_id,
            "domain_bridge_id": self.domain_bridge_id,
            "supersedes": sorted(self.supersedes),
            "source_ids": sorted(self.source_ids),
            "program_core": self.program_core,
            "notes": list(self.notes),
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ContractIssue:
    gate: str
    claim_id: str
    message: str
    dependency_id: Optional[str] = None
    severity: str = "ERROR"

    def as_dict(self) -> dict:
        return {
            "gate": self.gate,
            "claim_id": self.claim_id,
            "dependency_id": self.dependency_id,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class SemanticValidationReport:
    valid: bool
    roots: List[str]
    reachable: List[str]
    issues: List[ContractIssue]
    root_hashes: Dict[str, str]
    gate_counts: Dict[str, int]

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "roots": self.roots,
            "reachable": self.reachable,
            "issues": [issue.as_dict() for issue in self.issues],
            "root_hashes": self.root_hashes,
            "gate_counts": self.gate_counts,
        }


class SemanticProofGraph:
    def __init__(self, claims: Iterable[SemanticClaim] = ()) -> None:
        self.claims: Dict[str, SemanticClaim] = {}
        for claim in claims:
            self.add(claim)

    def add(self, claim: SemanticClaim) -> None:
        if claim.claim_id in self.claims:
            raise ValueError(f"duplicate claim ID: {claim.claim_id}")
        self.claims[claim.claim_id] = claim

    def reachable(self, roots: Sequence[str]) -> Set[str]:
        reached: Set[str] = set()
        stack = list(roots)
        while stack:
            claim_id = stack.pop()
            if claim_id in reached:
                continue
            reached.add(claim_id)
            claim = self.claims.get(claim_id)
            if claim is not None:
                stack.extend(claim.dependencies)
                if claim.measure_bridge_id:
                    stack.append(claim.measure_bridge_id)
                if claim.domain_bridge_id:
                    stack.append(claim.domain_bridge_id)
                if claim.model_bridge_id:
                    stack.append(claim.model_bridge_id)
                stack.extend(claim.composition_evidence)
        return reached

    def _cycle_issues(self, reached: Set[str]) -> List[ContractIssue]:
        issues: List[ContractIssue] = []
        color = {claim_id: 0 for claim_id in reached if claim_id in self.claims}
        path: List[str] = []

        def visit(claim_id: str) -> None:
            color[claim_id] = 1
            path.append(claim_id)
            claim = self.claims[claim_id]
            for dep_id in claim.dependencies:
                if dep_id not in color:
                    continue
                if color[dep_id] == 0:
                    visit(dep_id)
                elif color[dep_id] == 1:
                    start = path.index(dep_id)
                    cycle = path[start:] + [dep_id]
                    issues.append(
                        ContractIssue(
                            "ACYCLICITY",
                            claim_id,
                            "dependency cycle: " + " -> ".join(cycle),
                            dep_id,
                        )
                    )
            path.pop()
            color[claim_id] = 2

        for claim_id in sorted(color):
            if color[claim_id] == 0:
                visit(claim_id)
        return issues

    def merkle_hash(self, root: str) -> str:
        memo: Dict[str, str] = {}
        active: Set[str] = set()

        def visit(claim_id: str) -> str:
            if claim_id in memo:
                return memo[claim_id]
            if claim_id in active:
                raise ValueError(f"cycle at {claim_id}")
            claim = self.claims[claim_id]
            active.add(claim_id)
            child_ids = list(claim.dependencies)
            child_ids.extend(claim.composition_evidence)
            for optional in (
                claim.measure_bridge_id,
                claim.domain_bridge_id,
                claim.model_bridge_id,
            ):
                if optional:
                    child_ids.append(optional)
            child_hashes = sorted(visit(child) for child in child_ids)
            active.remove(claim_id)
            payload = claim.digest() + "|" + "|".join(child_hashes)
            memo[claim_id] = sha256(payload.encode()).hexdigest()
            return memo[claim_id]

        return visit(root)

    def validate(
        self,
        roots: Sequence[str],
        *,
        require_closed_core: bool = True,
        enforce_grade_floor: bool = False,
    ) -> SemanticValidationReport:
        roots = list(roots)
        reached = self.reachable(roots)
        issues: List[ContractIssue] = []

        # Dependency closure.
        for claim_id in sorted(reached):
            claim = self.claims.get(claim_id)
            if claim is None:
                issues.append(
                    ContractIssue("CLOSURE", claim_id, "missing claim object")
                )
                continue
            refs = list(claim.dependencies) + list(claim.composition_evidence)
            for optional in (
                claim.measure_bridge_id,
                claim.domain_bridge_id,
                claim.model_bridge_id,
            ):
                if optional:
                    refs.append(optional)
            for dep_id in refs:
                if dep_id not in self.claims:
                    issues.append(
                        ContractIssue(
                            "CLOSURE",
                            claim_id,
                            f"missing referenced claim {dep_id}",
                            dep_id,
                        )
                    )

        issues.extend(self._cycle_issues(reached))

        # Reachable semantic gates.
        for claim_id in sorted(reached):
            claim = self.claims.get(claim_id)
            if claim is None:
                continue

            if claim.status in {Status.KILLED, Status.SUPERSEDED}:
                issues.append(
                    ContractIssue(
                        "PRECEDENCE",
                        claim_id,
                        f"reachable claim has status {claim.status.value}",
                    )
                )

            if require_closed_core and claim.program_core and claim.status == Status.OPEN:
                issues.append(
                    ContractIssue(
                        "CORE_CLOSURE",
                        claim_id,
                        "finished core reaches an OPEN claim",
                    )
                )

            # Measured values require an explicit rung/scale tag.
            if claim.status == Status.MEASURED or claim.measured_value is not None:
                if not claim.scale_tag:
                    issues.append(
                        ContractIssue(
                            "RUNG",
                            claim_id,
                            "measured value lacks an explicit scale/rung tag",
                        )
                    )

            # Endpoint bound audit.
            if claim.bound_coefficient is not None and claim.rung_values:
                for rung, value in claim.rung_values:
                    if claim.direction == Direction.UPPER and value > claim.bound_coefficient + 1e-15:
                        issues.append(
                            ContractIssue(
                                "ENDPOINT",
                                claim_id,
                                f"rung r={rung} value {value} exceeds upper "
                                f"coefficient {claim.bound_coefficient}",
                            )
                        )
                    if claim.direction == Direction.LOWER and value < claim.bound_coefficient - 1e-15:
                        issues.append(
                            ContractIssue(
                                "ENDPOINT",
                                claim_id,
                                f"rung r={rung} value {value} lies below lower "
                                f"coefficient {claim.bound_coefficient}",
                            )
                        )

            # Probability products require a valid dependence witness.
            if claim.operation.lower() in {"product", "probability_product"}:
                if claim.composition_witness in {
                    CompositionWitness.NONE,
                    CompositionWitness.NOT_APPLICABLE,
                }:
                    issues.append(
                        ContractIssue(
                            "COMPOSITION",
                            claim_id,
                            "probability product lacks a dependence/comparison witness",
                        )
                    )
                elif not claim.composition_evidence:
                    issues.append(
                        ContractIssue(
                            "COMPOSITION",
                            claim_id,
                            "composition witness has no evidence node",
                        )
                    )

            # Exact model tag.
            covariance_words = (
                "covariance",
                "Gaussian",
                "Kac-Rice",
                "Schur",
                "Bargmann",
                "kernel",
            )
            if (
                any(word.lower() in claim.statement.lower() for word in covariance_words)
                and claim.model_id is None
                and claim.model_bridge_id is None
            ):
                issues.append(
                    ContractIssue(
                        "MODEL",
                        claim_id,
                        "covariance-dependent claim lacks exact model or transfer certificate",
                    )
                )

            provided_by_deps: Set[str] = set()
            for dep_id in claim.dependencies:
                dep = self.claims.get(dep_id)
                if dep is not None:
                    provided_by_deps.update(dep.provided_marks)

                    # Domain coverage.
                    if (
                        claim.domain is not None
                        and dep.domain is not None
                        and not dep.domain.contains(claim.domain)
                        and claim.domain_bridge_id is None
                    ):
                        issues.append(
                            ContractIssue(
                                "DOMAIN",
                                claim_id,
                                "dependency domain does not cover parent domain",
                                dep_id,
                            )
                        )

                    # Measure compatibility.
                    if (
                        claim.measure != Measure.NONE
                        and dep.measure != Measure.NONE
                        and dep.measure != claim.measure
                        and claim.measure_bridge_id is None
                    ):
                        issues.append(
                            ContractIssue(
                                "MEASURE",
                                claim_id,
                                f"parent measure {claim.measure.value} consumes "
                                f"{dep.measure.value} without a bridge",
                                dep_id,
                            )
                        )

                    if enforce_grade_floor and claim.grade > dep.grade:
                        issues.append(
                            ContractIssue(
                                "GRADE",
                                claim_id,
                                "claim grade exceeds dependency grade",
                                dep_id,
                            )
                        )

                    # One-sided uncertainty propagation.
                    if dep.status == Status.MEASURED or dep.measured_value is not None:
                        if (
                            claim.direction == Direction.UPPER
                            and dep.uncertainty_side
                            not in {UncertaintySide.UPPER, UncertaintySide.NONE}
                        ):
                            issues.append(
                                ContractIssue(
                                    "POLARITY",
                                    claim_id,
                                    "upper claim consumes a measured input without "
                                    "a one-sided upper allowance",
                                    dep_id,
                                )
                            )
                        if (
                            claim.direction == Direction.LOWER
                            and dep.uncertainty_side
                            not in {UncertaintySide.LOWER, UncertaintySide.NONE}
                        ):
                            issues.append(
                                ContractIssue(
                                    "POLARITY",
                                    claim_id,
                                    "lower claim consumes a measured input without "
                                    "a one-sided lower allowance",
                                    dep_id,
                                )
                            )

            missing_marks = set(claim.required_marks) - provided_by_deps
            if missing_marks:
                issues.append(
                    ContractIssue(
                        "MARK",
                        claim_id,
                        "required marked coordinates are absent from dependencies: "
                        + ", ".join(sorted(missing_marks)),
                    )
                )

        root_hashes: Dict[str, str] = {}
        if not any(issue.gate in {"CLOSURE", "ACYCLICITY"} for issue in issues):
            for root in roots:
                root_hashes[root] = self.merkle_hash(root)

        gate_counts: Dict[str, int] = {}
        for issue in issues:
            gate_counts[issue.gate] = gate_counts.get(issue.gate, 0) + 1

        return SemanticValidationReport(
            valid=not issues,
            roots=roots,
            reachable=sorted(reached),
            issues=issues,
            root_hashes=root_hashes,
            gate_counts=gate_counts,
        )


def program_grade_q0_contract() -> Tuple[SemanticProofGraph, List[str]]:
    """Build the closed Master-v3.2 core cone.

    This graph does not claim external referee conversion. It freezes the
    program's own verification-grade status and keeps later sharpening and
    external-audit work outside the core roots.
    """

    d = ParameterDomain(
        r_min=0.0,
        r_max=0.025,
        r_min_open=True,
        fixed=(("b", "1.2"), ("L", "24")),
    )
    exact_model = "BF_TORUS_EXACT_L24"

    claims = [
        SemanticClaim(
            "MODEL_EXACT",
            "Exact normalized periodized Bargmann-Fock covariance on T_24^2.",
            Status.DERIVED_EXACT,
            direction=Direction.EQUALITY,
            quantifier=Quantifier.EXACT_IDENTITY,
            domain=d,
            grade=Grade.CERTIFIED,
            model_id=exact_model,
            program_core=True,
            source_ids=("C090_PERIODIZED_BF",),
        ),
        SemanticClaim(
            "TORUS_TRANSFER",
            "Planar-reference local/far covariance calculations transfer to "
            "the exact periodized model in the declared matrix scopes.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.STRUCTURAL,
            Quantifier.FOR_ALL,
            dependencies=("MODEL_EXACT",),
            domain=d,
            grade=Grade.CERTIFIED,
            model_id=exact_model,
            program_core=True,
            source_ids=("C091_MATRIX_TRANSFER",),
        ),
        SemanticClaim(
            "R0_SARD_G",
            "Almost surely no saddle-saddle heteroclinic under the program's "
            "SARD-G proof.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.STRUCTURAL,
            Quantifier.FOR_ALL,
            dependencies=("TORUS_TRANSFER",),
            domain=d,
            measure=Measure.NONE,
            grade=Grade.DERIVED,
            model_id=exact_model,
            program_core=True,
            source_ids=("C041", "C042", "C043", "C046"),
            notes=("external specialist review remains pending",),
        ),
        SemanticClaim(
            "LAMBDA_SIDE",
            "Certified lower one-point qualifying intensity coefficient.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.LOWER,
            Quantifier.FOR_ALL,
            dependencies=("TORUS_TRANSFER",),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.CERTIFIED,
            provided_marks=frozenset(
                {"position", "height", "critical_type", "flow_qualification"}
            ),
            model_id=exact_model,
            program_core=True,
            source_ids=("C026", "C027", "C037", "C043"),
        ),
        SemanticClaim(
            "AO_SIDE",
            "Uniform adjacency/other-branch terminal survival factor.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.LOWER,
            Quantifier.FOR_ALL,
            dependencies=("R0_SARD_G", "TORUS_TRANSFER"),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.DERIVED,
            provided_marks=frozenset({"flow_qualification"}),
            model_id=exact_model,
            program_core=True,
            source_ids=("C032", "C036"),
        ),
        SemanticClaim(
            "ND_PRIME_INTERNAL",
            "Internal program-grade marked near-diagonal two-point certificate "
            "for the Bonferroni pair term.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.UPPER,
            Quantifier.FOR_ALL,
            dependencies=("TORUS_TRANSFER",),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.DERIVED,
            provided_marks=frozenset(
                {
                    "position_pair",
                    "height_1",
                    "height_2",
                    "critical_type_1",
                    "critical_type_2",
                }
            ),
            model_id=exact_model,
            program_core=True,
            source_ids=("ND_SYMBOLIC_ADJUDICATION", "C024"),
            notes=(
                "live at program grade",
                "external source-level marked-density audit remains pending",
            ),
        ),
        SemanticClaim(
            "BONFERRONI_INTERNAL",
            "Second factorial moment is O(r^6), so the lower first moment "
            "converts to a probability at relative loss O(r^3).",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.LOWER,
            Quantifier.FOR_ALL,
            dependencies=("ND_PRIME_INTERNAL",),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.DERIVED,
            required_marks=frozenset(
                {
                    "position_pair",
                    "height_1",
                    "height_2",
                    "critical_type_1",
                    "critical_type_2",
                }
            ),
            operation="inclusion_exclusion",
            composition_witness=CompositionWitness.EXACT_ALGEBRA,
            model_id=exact_model,
            program_core=True,
            source_ids=("C024",),
        ),
        SemanticClaim(
            "LOWER_RATE_PG",
            "0.8411 r^3 <= 1-q(r,1.2) for 0<r<=0.025 at the "
            "program's rigorized/verification grade.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.LOWER,
            Quantifier.FOR_ALL,
            dependencies=("LAMBDA_SIDE", "AO_SIDE", "BONFERRONI_INTERNAL"),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.CERTIFIED,
            bound_coefficient=0.8411,
            model_id=exact_model,
            program_core=True,
            source_ids=("MASTER_V3_2",),
        ),
        SemanticClaim(
            "UB_G",
            "Global interceptor counting gives the 4.3 r^3 upper coefficient "
            "at fixed L=24.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.UPPER,
            Quantifier.FOR_ALL,
            dependencies=("TORUS_TRANSFER", "R0_SARD_G"),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.CERTIFIED,
            model_id=exact_model,
            program_core=True,
            source_ids=("C038", "C039"),
        ),
        SemanticClaim(
            "RATE_PROGRAM_GRADE",
            "0.8411 r^3 <= 1-q(r,1.2) <= 4.3 r^3 for "
            "0<r<=0.025, L=24, at program grade.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.STRUCTURAL,
            Quantifier.FOR_ALL,
            dependencies=("LOWER_RATE_PG", "UB_G"),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.CERTIFIED,
            model_id=exact_model,
            program_core=True,
            source_ids=("MASTER_V3_2", "C092"),
        ),
        SemanticClaim(
            "Q0_LIMIT",
            "q(r,1.2) tends to one as r tends to zero at fixed L=24.",
            Status.PROGRAM_GRADE_CLOSED,
            Direction.EQUALITY,
            Quantifier.ASYMPTOTIC,
            dependencies=("UB_G",),
            domain=d,
            measure=Measure.PAIR_PALM,
            grade=Grade.CERTIFIED,
            model_id=exact_model,
            program_core=True,
            source_ids=("MASTER_V3_2", "C092"),
        ),
    ]
    return SemanticProofGraph(claims), ["RATE_PROGRAM_GRADE", "Q0_LIMIT"]


def bad_contract_examples() -> Mapping[str, SemanticValidationReport]:
    """Return deliberately invalid mini-contracts for regression testing."""

    d = ParameterDomain(0.0, 0.05, True, (("b", "1.2"),))
    model = "BF_TORUS_EXACT_L24"
    examples: Dict[str, SemanticValidationReport] = {}

    # MARK failure: unmarked spatial repulsion used for a marked window count.
    graph = SemanticProofGraph(
        [
            SemanticClaim(
                "UNMARKED",
                "Spatial saddle-saddle repulsion only.",
                Status.ESTABLISHED,
                provided_marks=frozenset({"position_pair", "critical_type_1", "critical_type_2"}),
                domain=d,
                measure=Measure.PAIR_PALM,
                model_id=model,
            ),
            SemanticClaim(
                "MARKED_PAIR",
                "Two window heights and two saddle types.",
                Status.CONDITIONAL,
                dependencies=("UNMARKED",),
                required_marks=frozenset(
                    {
                        "position_pair",
                        "height_1",
                        "height_2",
                        "critical_type_1",
                        "critical_type_2",
                    }
                ),
                domain=d,
                measure=Measure.PAIR_PALM,
                model_id=model,
            ),
        ]
    )
    examples["missing_mark"] = graph.validate(["MARKED_PAIR"], require_closed_core=False)

    # COMPOSITION failure: q_step^n without a dependence theorem.
    graph = SemanticProofGraph(
        [
            SemanticClaim(
                "Q_STEP",
                "One-step Gaussian corridor probability.",
                Status.DERIVED_EXACT,
                domain=d,
                measure=Measure.GAUSSIAN_PINNED,
                model_id=model,
            ),
            SemanticClaim(
                "BAD_PRODUCT",
                "Multiply one-step probabilities along a correlated corridor.",
                Status.CONDITIONAL,
                dependencies=("Q_STEP",),
                operation="probability_product",
                composition_witness=CompositionWitness.NONE,
                domain=d,
                measure=Measure.GAUSSIAN_PINNED,
                model_id=model,
            ),
        ]
    )
    examples["missing_composition_witness"] = graph.validate(
        ["BAD_PRODUCT"], require_closed_core=False
    )

    # ENDPOINT failure: C089's 0.97 ceiling includes a 0.985 endpoint.
    graph = SemanticProofGraph(
        [
            SemanticClaim(
                "BAD_CEILING",
                "U(r)<=0.97 through r=0.05.",
                Status.CONDITIONAL,
                Direction.UPPER,
                Quantifier.FOR_ALL,
                domain=d,
                rung_values=((0.0125, 0.9411), (0.025, 0.9605), (0.05, 0.985)),
                bound_coefficient=0.97,
                model_id=model,
            )
        ]
    )
    examples["endpoint_failure"] = graph.validate(
        ["BAD_CEILING"], require_closed_core=False
    )

    # RUNG failure: a measured 0.946 is called a scale-free truth constant.
    graph = SemanticProofGraph(
        [
            SemanticClaim(
                "TRUTH_CONST",
                "Scale-free measured truth constant.",
                Status.MEASURED,
                measured_value=0.946,
                uncertainty_side=UncertaintySide.TWO_SIDED_CENTRAL,
                model_id=model,
            )
        ]
    )
    examples["missing_rung_tag"] = graph.validate(
        ["TRUTH_CONST"], require_closed_core=False
    )

    # POLARITY failure.
    graph = SemanticProofGraph(
        [
            SemanticClaim(
                "CENTRAL",
                "Measured central upper input with symmetric uncertainty.",
                Status.MEASURED,
                Direction.STRUCTURAL,
                measured_value=0.95,
                scale_tag="r=0.025",
                uncertainty_side=UncertaintySide.TWO_SIDED_CENTRAL,
                model_id=model,
            ),
            SemanticClaim(
                "UPPER",
                "Upper theorem consumes the central value unchanged.",
                Status.CONDITIONAL,
                Direction.UPPER,
                dependencies=("CENTRAL",),
                model_id=model,
            ),
        ]
    )
    examples["uncertainty_polarity"] = graph.validate(
        ["UPPER"], require_closed_core=False
    )

    return examples


def _demo() -> None:
    graph, roots = program_grade_q0_contract()
    core = graph.validate(roots)
    assert core.valid, core.as_dict()

    bad = bad_contract_examples()
    expected = {
        "missing_mark": "MARK",
        "missing_composition_witness": "COMPOSITION",
        "endpoint_failure": "ENDPOINT",
        "missing_rung_tag": "RUNG",
        "uncertainty_polarity": "POLARITY",
    }
    for name, gate in expected.items():
        assert not bad[name].valid
        assert any(issue.gate == gate for issue in bad[name].issues)

    # Preserve earlier numerical primitives.
    support_edges = [
        SupportEdge("claim", "bridge", 0.82),
        SupportEdge("bridge", "source", 0.74),
        SupportEdge("claim", "source", 0.51),
    ]
    grounding = widest_path_grounding(
        {"claim", "bridge", "source"}, support_edges, {"source"}
    )
    assert abs(grounding.bottleneck_to_evidence["claim"] - 0.74) < 1e-12

    import numpy as np

    z = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    weights = np.array([1.0, -0.4, -0.6])
    rkhs = anchored_rkhs_mismatch(z, weights)
    assert rkhs > 0.0

    tube = anisotropic_sard_tube_bound(
        0.01, [1.0, 1.0, 1e-3], density_bound=0.1
    )
    assert effective_transverse_rank([1.0, 1.0, 1e-3], 0.01) == 2

    coverage = CoverageCertificate(uncovered_wrong_mass=0.002)
    try:
        coverage_limited_rank_requirement(0.001, 0.002, 0.2)
        raise AssertionError("coverage floor should make target unattainable")
    except ValueError:
        pass

    ledger = SelectionAwareRiskLedger(
        coverage_failure=0.002,
        baseline_hazard_rate=0.001,
        expected_selected_measure=10.0,
        selection_amplification=3.0,
        q_sard_tube_risk=0.0002,
        decoder_boundary_risk=0.0001,
    )

    output = {
        "core_valid": core.valid,
        "core_roots": roots,
        "core_reachable_nodes": len(core.reachable),
        "core_root_hashes": core.root_hashes,
        "bad_contract_gate_counts": {
            name: report.gate_counts for name, report in bad.items()
        },
        "grounding_threshold": grounding.bottleneck_to_evidence["claim"],
        "grounding_retained_edges": len(grounding.retained_edges),
        "anchored_rkhs_score_squared": rkhs,
        "rank_deficient_q_sard_bound": tube,
        "coverage_floor": coverage.irreducible_false_acceptance_floor(),
        "selection_aware_total_risk": ledger.total_upper_bound(),
        "selection_aware_exponential_limit_floor": ledger.exponential_limit_floor(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    _demo()
