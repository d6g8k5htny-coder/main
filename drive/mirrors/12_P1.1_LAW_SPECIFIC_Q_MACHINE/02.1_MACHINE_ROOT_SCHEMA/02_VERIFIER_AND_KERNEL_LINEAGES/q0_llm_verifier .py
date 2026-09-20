"""
q0_llm_verifier.py

Prototype implementation of four reusable pieces of the q0-to-LLM program:

1. Proof/dependency DAG validation with supersession and grade checks.
2. Incremental Merkle certificates and descendant invalidation.
3. H0 grounding persistence through maximum-bottleneck paths.
4. An anchored RKHS mismatch score whose fixed universal RBF component
   cannot be removed by training.
5. Quantitative SARD-style tube-bound utilities.

This is a research prototype, not a production factuality guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256
import heapq
import json
import math
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class Grade(IntEnum):
    """Ordered epistemic grades. Higher values may depend on lower/equal values."""

    HYPOTHESIS = 0
    MEASURED = 1
    DERIVED = 2
    CERTIFIED = 3
    ESTABLISHED = 4


@dataclass(frozen=True)
class ClaimNode:
    """One proof-carrying claim."""

    claim_id: str
    statement: str
    dependencies: Tuple[str, ...] = ()
    grade: Grade = Grade.HYPOTHESIS
    evidence: Tuple[str, ...] = ()
    superseded: bool = False
    version: str = "1"

    def local_digest(self) -> str:
        payload = {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "dependencies": sorted(self.dependencies),
            "grade": int(self.grade),
            "evidence": sorted(self.evidence),
            "superseded": self.superseded,
            "version": self.version,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class ValidationReport:
    valid: bool
    missing_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    reachable_superseded: List[str] = field(default_factory=list)
    grade_violations: List[Tuple[str, str]] = field(default_factory=list)
    reachable_nodes: List[str] = field(default_factory=list)


class ProofGraph:
    """Exact structural checks for a dependency graph."""

    def __init__(self, nodes: Iterable[ClaimNode] = ()) -> None:
        self.nodes: Dict[str, ClaimNode] = {}
        self.reverse: Dict[str, Set[str]] = {}
        for node in nodes:
            self.add(node)

    def add(self, node: ClaimNode) -> None:
        old = self.nodes.get(node.claim_id)
        if old is not None:
            for dep in old.dependencies:
                self.reverse.get(dep, set()).discard(node.claim_id)
        self.nodes[node.claim_id] = node
        self.reverse.setdefault(node.claim_id, set())
        for dep in node.dependencies:
            self.reverse.setdefault(dep, set()).add(node.claim_id)

    def descendants(self, changed_ids: Iterable[str]) -> Set[str]:
        """All nodes invalidated by changed_ids, including changed_ids."""
        seen: Set[str] = set(changed_ids)
        stack = list(seen)
        while stack:
            current = stack.pop()
            for child in self.reverse.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def _reachable(self, roots: Optional[Sequence[str]]) -> Set[str]:
        if roots is None:
            return set(self.nodes)
        seen: Set[str] = set()
        stack = list(roots)
        while stack:
            node_id = stack.pop()
            if node_id in seen or node_id not in self.nodes:
                continue
            seen.add(node_id)
            stack.extend(self.nodes[node_id].dependencies)
        return seen

    def validate(
        self,
        roots: Optional[Sequence[str]] = None,
        *,
        enforce_grade_monotonicity: bool = False,
    ) -> ValidationReport:
        reachable = self._reachable(roots)
        missing: Dict[str, List[str]] = {}
        superseded: List[str] = []
        grade_violations: List[Tuple[str, str]] = []

        for node_id in sorted(reachable):
            node = self.nodes[node_id]
            absent = [d for d in node.dependencies if d not in self.nodes]
            if absent:
                missing[node_id] = absent
            if node.superseded:
                superseded.append(node_id)
            if enforce_grade_monotonicity:
                # Optional policy, not a law of logic: a claim may not advertise a
                # stronger grade than any load-bearing dependency.
                for dep_id in node.dependencies:
                    dep = self.nodes.get(dep_id)
                    if dep is not None and node.grade > dep.grade:
                        grade_violations.append((node_id, dep_id))

        color: Dict[str, int] = {node_id: 0 for node_id in reachable}
        path: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node_id: str) -> None:
            color[node_id] = 1
            path.append(node_id)
            for dep in self.nodes[node_id].dependencies:
                if dep not in reachable:
                    continue
                if color[dep] == 0:
                    dfs(dep)
                elif color[dep] == 1:
                    start = path.index(dep)
                    cycles.append(path[start:] + [dep])
            path.pop()
            color[node_id] = 2

        for node_id in sorted(reachable):
            if color[node_id] == 0:
                dfs(node_id)

        valid = not missing and not cycles and not superseded and not grade_violations
        return ValidationReport(
            valid=valid,
            missing_dependencies=missing,
            cycles=cycles,
            reachable_superseded=superseded,
            grade_violations=grade_violations,
            reachable_nodes=sorted(reachable),
        )

    def merkle_hash(self, root: str) -> str:
        """
        Content-addressed proof certificate. A dependency change changes every
        downstream root hash.
        """
        memo: Dict[str, str] = {}
        active: Set[str] = set()

        def visit(node_id: str) -> str:
            if node_id in memo:
                return memo[node_id]
            if node_id in active:
                raise ValueError(f"Cycle encountered at {node_id!r}")
            node = self.nodes.get(node_id)
            if node is None:
                raise KeyError(f"Missing dependency {node_id!r}")
            active.add(node_id)
            dep_hashes = sorted(visit(dep) for dep in node.dependencies)
            active.remove(node_id)
            payload = node.local_digest() + "|" + "|".join(dep_hashes)
            memo[node_id] = sha256(payload.encode()).hexdigest()
            return memo[node_id]

        return visit(root)


@dataclass(frozen=True)
class SupportEdge:
    """
    Directed support edge: src claim is supported by dst claim/evidence with
    strength in [0,1].
    """

    src: str
    dst: str
    strength: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must lie in [0,1]")


@dataclass
class GroundingCertificate:
    bottleneck_to_evidence: Dict[str, float]
    persistence_parent: Dict[str, Optional[str]]
    retained_edges: List[SupportEdge]


def widest_path_grounding(
    nodes: Iterable[str],
    edges: Sequence[SupportEdge],
    evidence_nodes: Iterable[str],
) -> GroundingCertificate:
    """
    Compute, for every claim v,

        d(v) = max_{v -> ... -> evidence} min(edge strengths on the path).

    This is the H0 merge/grounding threshold for the edge superlevel
    filtration. The returned predecessor forest preserves every d(v), so the
    full support graph can be compressed to at most |V|-|evidence| edges.
    """
    node_set = set(nodes)
    evidence = set(evidence_nodes)
    for e in edges:
        node_set.add(e.src)
        node_set.add(e.dst)

    # Reverse edges so a max-capacity search can grow outward from evidence.
    incoming: Dict[str, List[Tuple[str, float]]] = {v: [] for v in node_set}
    edge_lookup: Dict[Tuple[str, str], float] = {}
    for e in edges:
        incoming[e.dst].append((e.src, e.strength))
        edge_lookup[(e.src, e.dst)] = max(
            e.strength, edge_lookup.get((e.src, e.dst), 0.0)
        )

    cap = {v: 0.0 for v in node_set}
    parent: Dict[str, Optional[str]] = {v: None for v in node_set}
    heap: List[Tuple[float, str]] = []

    for root in evidence:
        if root not in node_set:
            node_set.add(root)
            incoming[root] = []
            cap[root] = 1.0
            parent[root] = None
        else:
            cap[root] = 1.0
        heapq.heappush(heap, (-1.0, root))

    while heap:
        neg_value, current = heapq.heappop(heap)
        value = -neg_value
        if value + 1e-15 < cap[current]:
            continue
        for predecessor, strength in incoming.get(current, ()):
            candidate = min(value, strength)
            if candidate > cap[predecessor] + 1e-15:
                cap[predecessor] = candidate
                parent[predecessor] = current
                heapq.heappush(heap, (-candidate, predecessor))

    retained: List[SupportEdge] = []
    for src, dst in parent.items():
        if dst is not None:
            retained.append(
                SupportEdge(src, dst, edge_lookup[(src, dst)])
            )

    return GroundingCertificate(cap, parent, retained)


def pairing_defects(
    geometric_partner: Mapping[str, Optional[str]],
    certificate: GroundingCertificate,
) -> Dict[str, bool]:
    """
    Compare a local/geometric partner (e.g. highest-attention evidence) with
    the persistence partner that realizes the maximum-bottleneck grounding
    path.
    """
    return {
        node: geometric_partner.get(node) != certificate.persistence_parent.get(node)
        for node in certificate.bottleneck_to_evidence
        if certificate.persistence_parent.get(node) is not None
    }


def rbf_kernel_matrix(x: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    x = np.asarray(x, dtype=float)
    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(sq_norm + sq_norm.T - 2.0 * x @ x.T, 0.0)
    return np.exp(-dist2 / (2.0 * bandwidth * bandwidth))


def anchored_rkhs_mismatch(
    embeddings: NDArray[np.float64],
    signed_weights: NDArray[np.float64],
    *,
    bandwidth: float = 1.0,
    anchor_weight: float = 0.1,
    learned_features: Optional[NDArray[np.float64]] = None,
) -> float:
    """
    Squared RKHS norm of a signed evidence residual.

    K = anchor_weight * K_RBF + Phi_learned Phi_learned^T.

    Since anchor_weight > 0 and the Gaussian RBF is universal on compact
    subsets of R^d, a trainable PSD component cannot destroy the anchor's
    injectivity on finite signed measures.
    """
    z = np.asarray(embeddings, dtype=float)
    w = np.asarray(signed_weights, dtype=float).reshape(-1)
    if z.ndim != 2 or z.shape[0] != w.shape[0]:
        raise ValueError("embeddings and signed_weights have incompatible shapes")
    if anchor_weight <= 0:
        raise ValueError("anchor_weight must be strictly positive")

    gram = anchor_weight * rbf_kernel_matrix(z, bandwidth)
    if learned_features is not None:
        features = np.asarray(learned_features, dtype=float)
        if features.shape[0] != z.shape[0]:
            raise ValueError("learned_features has incompatible row count")
        gram = gram + features @ features.T

    score2 = float(w @ gram @ w)
    return max(score2, 0.0)


def quantitative_sard_tube_bound(
    epsilon: float,
    rank: int,
    *,
    density_bound: float,
    inverse_jacobian_bound: float,
    chart_multiplicity: int = 1,
    unit_ball_volume: Optional[float] = None,
) -> float:
    """
    A conservative coarea-style small-tube bound

        P(||D|| <= epsilon)
        <= multiplicity * density_bound * Vol(B_rank)
           * (epsilon * inverse_jacobian_bound)^rank.

    inverse_jacobian_bound is 1/sigma_min for the selected witness Jacobian.
    The bound must be supplied only when its geometric assumptions have been
    certified.
    """
    if epsilon < 0 or rank < 1:
        raise ValueError("epsilon must be nonnegative and rank >= 1")
    if density_bound < 0 or inverse_jacobian_bound < 0:
        raise ValueError("bounds must be nonnegative")
    if chart_multiplicity < 1:
        raise ValueError("chart_multiplicity must be >= 1")

    if unit_ball_volume is None:
        unit_ball_volume = math.pi ** (rank / 2.0) / math.gamma(rank / 2.0 + 1.0)

    return min(
        1.0,
        chart_multiplicity
        * density_bound
        * unit_ball_volume
        * (epsilon * inverse_jacobian_bound) ** rank,
    )


def required_mismatch_rank(
    target_failure_probability: float,
    epsilon_times_inverse_jacobian: float,
    *,
    prefactor: float = 1.0,
) -> int:
    """
    Smallest k such that prefactor * rho^k <= target, for 0 < rho < 1.
    """
    if not 0 < target_failure_probability < 1:
        raise ValueError("target_failure_probability must be in (0,1)")
    if not 0 < epsilon_times_inverse_jacobian < 1:
        raise ValueError("epsilon_times_inverse_jacobian must be in (0,1)")
    if prefactor <= 0:
        raise ValueError("prefactor must be positive")

    rhs = math.log(target_failure_probability / prefactor)
    denominator = math.log(epsilon_times_inverse_jacobian)
    return max(1, math.ceil(rhs / denominator))


def _demo() -> None:
    nodes = [
        ClaimNode("E1", "Retrieved source supports atomic fact.", grade=Grade.CERTIFIED),
        ClaimNode("L1", "Derived intermediate claim.", ("E1",), grade=Grade.DERIVED),
        ClaimNode("ROOT", "Final answer.", ("L1",), grade=Grade.DERIVED),
    ]
    graph = ProofGraph(nodes)
    report = graph.validate(["ROOT"])
    assert report.valid
    assert len(graph.merkle_hash("ROOT")) == 64

    support_nodes = {"claim", "bridge", "source"}
    support_edges = [
        SupportEdge("claim", "bridge", 0.82),
        SupportEdge("bridge", "source", 0.74),
        SupportEdge("claim", "source", 0.51),
    ]
    cert = widest_path_grounding(support_nodes, support_edges, {"source"})
    assert abs(cert.bottleneck_to_evidence["claim"] - 0.74) < 1e-12
    assert len(cert.retained_edges) <= len(support_nodes) - 1

    z = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=float)
    w = np.array([1.0, -0.4, -0.6], dtype=float)
    score2 = anchored_rkhs_mismatch(z, w)
    assert score2 > 0.0

    k = required_mismatch_rank(1e-6, 0.2)
    assert 0.2**k <= 1e-6

    print(
        json.dumps(
            {
                "proof_graph_valid": report.valid,
                "root_merkle_hash_prefix": graph.merkle_hash("ROOT")[:16],
                "claim_grounding_threshold": cert.bottleneck_to_evidence["claim"],
                "retained_support_edges": len(cert.retained_edges),
                "anchored_rkhs_score_squared": round(score2, 8),
                "rank_needed_for_1e-6_at_rho_0.2": k,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _demo()
