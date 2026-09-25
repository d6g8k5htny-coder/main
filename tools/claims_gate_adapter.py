#!/usr/bin/env python3
"""Thin hardening adapter: claims/graph.json → #90 gate semantics (D7).

Semantic reference: Math- PR13 (required REFUTED blocks; reverse impact over
UNION(old,new) edges; REQUIRED_SATISFIED = {PROVED_REVIEWED} only). This module
does **not** vendor Math- code and never grants promotion permission.

Outputs are HOLD / REVALIDATION proposals only. Scientific effect: NONE.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "claims" / "graph.json"
CROSSWALK_PATH = ROOT / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json"
AUTHORITY_PATH = ROOT / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json"

_DIGEST_SPEC = importlib.util.spec_from_file_location(
    "semantic_digest", ROOT / "tools" / "semantic_digest.py"
)
assert _DIGEST_SPEC and _DIGEST_SPEC.loader
_SD = importlib.util.module_from_spec(_DIGEST_SPEC)
_DIGEST_SPEC.loader.exec_module(_SD)

# #90 clarified: disposition vocabulary is broader than premise satisfaction.
REQUIRED_SATISFIED = frozenset({"PROVED_REVIEWED"})
# SUPERSEDED_NONBLOCKING is terminal disposition but NOT premise satisfaction
# unless the reviewed edge was actually replaced/removed (handled by absence
# from the new graph's required edges, not by treating it as satisfied here).

REFUTED_CLASSIFICATIONS = frozenset({"REFUTED", "REFUTED_AS_WRITTEN"})
BLOCKED_CLASSIFICATIONS = frozenset({"BLOCKED_ABSENT"})


class AdapterError(ValueError):
    """Fail-closed adapter refusal."""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def map_classification(record: dict[str, Any], *, bucket: str) -> str:
    """Map claims/premise vocabulary onto gate classifications.

    Never invents PROVED_REVIEWED from firewall grades alone — transcribed
    CERTIFIED_RUNG / LIVE_ROOT_THEOREM remain author-side for integrity purposes
    until an external authority independently grants PROVED_REVIEWED.
    """
    if bucket == "premises":
        status = str(
            record.get("status_frozen_v2_2")
            or record.get("status_register_note")
            or "OPEN"
        ).upper()
        if status in REFUTED_CLASSIFICATIONS or status == "REFUTED":
            return "REFUTED"
        if status in {"OPEN", "NOT_CLOSED", "RESTATED", "REFINEMENT", "NAMED_HYPOTHESIS"}:
            return "OPEN_ACTIVE"
        if status == "CLOSED":
            # Closed display is not PROVED_REVIEWED for #90 premise satisfaction.
            return "AUTHOR_SIDE_CANDIDATE"
        return "OPEN_ACTIVE"

    grade = str(record.get("grade") or "OPEN")
    if grade in REFUTED_CLASSIFICATIONS:
        return "REFUTED"
    if grade == "OPEN":
        return "OPEN_ACTIVE"
    if grade == "RETRACTED_TO_CANDIDATE":
        return "AUTHOR_SIDE_CANDIDATE"
    # All other firewall grades are non-PROVED_REVIEWED for this adapter.
    return "AUTHOR_SIDE_CANDIDATE"


def _source_snapshot(record: dict[str, Any]) -> str:
    """Complete canonical JSON of the source record (supplementary detector)."""
    if not isinstance(record, dict):
        raise AdapterError("source snapshot requires an object record")
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digests_for(nid: str, record: dict[str, Any]) -> tuple[str, str]:
    try:
        return _SD.semantic_digest(nid, record), _SD.evidence_digest(record)
    except _SD.DigestError as exc:
        raise AdapterError(str(exc)) from exc


def _fingerprint(record: dict[str, Any]) -> str:
    """Deprecated alias: full-record snapshot (not the sole change detector)."""
    return _source_snapshot(record)


def _require_strict_bool(value: Any, *, field: str) -> bool:
    """Fail closed unless `value` is a Python bool (reject 0/1/str/null)."""
    if type(value) is not bool:
        raise AdapterError(
            f"{field} must be a strict boolean, got {type(value).__name__}: {value!r}"
        )
    return value


def _edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
    return (edge["from"], edge["to"], edge["relation"], edge["required"])


def _edge(frm: str, to: str, relation: str, *, required: bool = True) -> dict[str, Any]:
    _require_strict_bool(required, field="required")
    return {
        "from": frm,
        "to": to,
        "required": required,
        "relation": relation,
    }


def claims_to_gate_graph(
    claims: dict[str, Any],
    *,
    crosswalk: dict[str, Any] | None = None,
    authority_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project claims/graph.json into a gate-shaped working graph.

    Includes depends_on and sub_obligations as required typed edges.
    Fail-closed on unknown IDs, cycles, malformed records, ambiguous owners,
    and missing as_of / schema identity.
    """
    if not isinstance(claims, dict):
        raise AdapterError("malformed claims document")
    if "as_of" not in claims:
        raise AdapterError("stale or incomplete claims document: missing as_of")
    premises = claims.get("premises")
    claim_nodes = claims.get("claims")
    if not isinstance(premises, dict) or not isinstance(claim_nodes, dict):
        raise AdapterError("malformed claims: premises/claims must be objects")

    # Ambiguous owner: crosswalk rows must cite known authorities when present.
    if crosswalk is not None and authority_map is not None:
        known = set(authority_map.get("authorities") or {})
        this_pkg = authority_map.get("this_package") or {}
        if isinstance(this_pkg, dict) and this_pkg.get("id"):
            known.add(this_pkg["id"])
        owners_for_main: dict[str, set[str]] = {}
        for row in crosswalk.get("rows") or []:
            if not isinstance(row, dict):
                raise AdapterError("malformed crosswalk row")
            auth = row.get("authority")
            if auth not in known:
                raise AdapterError(f"unknown crosswalk authority: {auth!r}")
            main_id = row.get("main_claim_or_premise_id")
            if main_id:
                owners_for_main.setdefault(main_id, set()).add(auth)
        for main_id, owners in owners_for_main.items():
            if len(owners) > 1:
                raise AdapterError(
                    f"ambiguous canonical owner for {main_id!r}: {sorted(owners)}"
                )

    nodes: dict[str, Any] = {}
    edges: list[dict[str, Any]] = []

    def add_node(nid: str, bucket: str, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise AdapterError(f"malformed record for {nid!r}")
        if nid in nodes:
            raise AdapterError(f"duplicate node id {nid!r}")
        snapshot = _source_snapshot(record)
        sem, evid = _digests_for(nid, record)
        nodes[nid] = {
            "bucket": bucket,
            "classification": map_classification(record, bucket=bucket),
            "controlling": False,
            "semantic_digest": sem,
            "evidence_digest": evid,
            "source_snapshot": snapshot,
            "fingerprint": sem,  # derived digest; never a manual sole detector
            "version": claims.get("as_of"),
        }

    for nid, record in premises.items():
        add_node(nid, "premises", record)
    for nid, record in claim_nodes.items():
        add_node(nid, "claims", record)

    def add_deps(nid: str, record: dict[str, Any]) -> None:
        for dep in record.get("depends_on") or []:
            if dep not in nodes:
                raise AdapterError(f"unknown dependency id {dep!r} from {nid!r}")
            edges.append(_edge(nid, dep, "depends_on"))
        for dep in record.get("sub_obligations") or []:
            if dep not in nodes:
                raise AdapterError(f"unknown sub_obligation id {dep!r} from {nid!r}")
            edges.append(_edge(nid, dep, "sub_obligation"))

    for nid, record in premises.items():
        add_deps(nid, record)
    for nid, record in claim_nodes.items():
        add_deps(nid, record)

    graph = {
        "schema_version": 1,
        "object": "CLAIMS-GATE-ADAPTER-20260925-v1",
        "source": "claims/graph.json",
        "as_of": claims.get("as_of"),
        "nodes": nodes,
        "edges": edges,
        "scientific_effect": "NONE",
        "promotion_permission": False,
        "meaning": "hardening adapter projection; never promotion permission",
    }
    validate_graph_fail_closed(graph)
    return graph


def required_dependencies(graph: dict[str, Any], node_id: str) -> list[str]:
    if node_id not in graph["nodes"]:
        raise AdapterError(f"unknown node: {node_id}")
    deps: list[str] = []
    for edge in graph["edges"]:
        if edge["from"] != node_id:
            continue
        if _require_strict_bool(edge["required"], field="required"):
            deps.append(edge["to"])
    return deps


def transitive_required(graph: dict[str, Any], node_id: str) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []

    def walk(nid: str) -> None:
        for dep in required_dependencies(graph, nid):
            if dep in seen:
                continue
            seen.add(dep)
            order.append(dep)
            walk(dep)

    walk(node_id)
    return order


def _required_cycle(graph: dict[str, Any]) -> list[str] | None:
    visiting: set[str] = set()
    done: set[str] = set()
    stack: list[str] = []

    def visit(nid: str) -> list[str] | None:
        if nid in visiting:
            i = stack.index(nid)
            return stack[i:] + [nid]
        if nid in done:
            return None
        visiting.add(nid)
        stack.append(nid)
        for dep in required_dependencies(graph, nid):
            cycle = visit(dep)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(nid)
        done.add(nid)
        return None

    for nid in sorted(graph["nodes"]):
        cycle = visit(nid)
        if cycle:
            return cycle
    return None


def validate_graph_fail_closed(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        raise AdapterError("malformed graph")
    seen_edges: set[tuple[Any, ...]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or not {"from", "to", "required", "relation"} <= set(edge):
            raise AdapterError("malformed edge record")
        _require_strict_bool(edge["required"], field="required")
        if edge["from"] not in nodes or edge["to"] not in nodes:
            raise AdapterError("edge references missing node")
        # Duplicate identity: same from/to/relation regardless of required flag.
        identity = (edge["from"], edge["to"], edge["relation"])
        if identity in seen_edges:
            raise AdapterError(
                "duplicate edge record: "
                f"{edge['from']!r} -> {edge['to']!r} ({edge['relation']})"
            )
        seen_edges.add(identity)
    cycle = _required_cycle(graph)
    if cycle:
        raise AdapterError("required dependency cycle: " + " -> ".join(cycle))


def required_holds(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """HOLD proposals from still-required REFUTED / BLOCKED_ABSENT premises.

    SUPERSEDED_NONBLOCKING does not satisfy a required premise here.
    Never returns promotion permission.
    """
    if node_id not in graph["nodes"]:
        raise AdapterError(f"unknown node: {node_id}")
    required = transitive_required(graph, node_id)
    refuted = []
    blocked = []
    unsatisfied = []
    for dep in required:
        cls = graph["nodes"][dep]["classification"]
        if cls in REFUTED_CLASSIFICATIONS:
            refuted.append(dep)
        elif cls in BLOCKED_CLASSIFICATIONS:
            blocked.append(dep)
        elif cls not in REQUIRED_SATISFIED:
            # Includes SUPERSEDED_NONBLOCKING, OPEN_*, AUTHOR_SIDE_*, etc.
            unsatisfied.append({"id": dep, "classification": cls})

    proposals: list[str] = []
    reasons: list[str] = []
    if refuted:
        proposals.append("HOLD")
        reasons.append("required REFUTED dependency forces HOLD")
    if blocked:
        proposals.append("HOLD")
        reasons.append("required BLOCKED_ABSENT dependency forces HOLD")
    if unsatisfied:
        proposals.append("HOLD")
        reasons.append("required transitive dependency is not satisfied")

    return {
        "node": node_id,
        "proposals": sorted(set(proposals)) or ["HOLD"],
        "promotion_permission": False,
        "required_dependencies": required,
        "refuted_required": refuted,
        "blocked_absent": blocked,
        "unsatisfied_required": unsatisfied,
        "reasons": reasons,
        "scientific_effect": "NONE",
        "meaning": "HOLD proposal only; never promotion permission",
    }


def _node_identity_changed(old_node: dict[str, Any], new_node: dict[str, Any]) -> bool:
    """True when semantic digest / source snapshot / classification / version diverge.

    Manual fingerprint alone is never the sole detector when digests exist.
    """
    old_sem = old_node.get("semantic_digest")
    new_sem = new_node.get("semantic_digest")
    if old_sem is not None or new_sem is not None:
        if old_sem != new_sem:
            return True
    old_snap = old_node.get("source_snapshot")
    new_snap = new_node.get("source_snapshot")
    if old_snap is not None or new_snap is not None:
        if old_snap != new_snap:
            return True
    elif old_node.get("fingerprint") != new_node.get("fingerprint"):
        # Legacy fixtures without digests/snapshots — fallback only.
        return True
    return (
        old_node.get("classification") != new_node.get("classification")
        or old_node.get("version") != new_node.get("version")
    )


def reverse_impact_between(old_graph: dict[str, Any], new_graph: dict[str, Any]) -> dict[str, Any]:
    """Reverse impact over UNION(old,new) edges; deleted edges cannot erase impact.

    Seeds include:
      - nodes whose canonical source snapshot / classification / version change;
      - endpoints of edge-only changes (add/remove/mutate) even when fingerprints
        and classifications are unchanged;
      - the changed controlling node itself (self-hold / self-revalidation).

    Marks impacted nodes with REVALIDATION_REQUIRED proposals on a copy of
    new_graph. Never sets controlling=True. Never grants promotion permission.
    """
    validate_graph_fail_closed(old_graph)
    validate_graph_fail_closed(new_graph)
    old_nodes, new_nodes = old_graph["nodes"], new_graph["nodes"]
    all_ids = set(old_nodes) | set(new_nodes)
    changed: set[str] = set()
    for nid in all_ids:
        if nid not in old_nodes or nid not in new_nodes:
            changed.add(nid)
            continue
        if _node_identity_changed(old_nodes[nid], new_nodes[nid]):
            changed.add(nid)

    # Edge-only seeds: compare complete edge records; endpoints are impact seeds
    # even when node fingerprints/status are unchanged.
    old_edge_keys = {_edge_key(e) for e in old_graph["edges"]}
    new_edge_keys = {_edge_key(e) for e in new_graph["edges"]}
    edge_only_seeds: set[str] = set()
    for key in old_edge_keys.symmetric_difference(new_edge_keys):
        frm, to, _relation, _required = key
        edge_only_seeds.add(frm)
        edge_only_seeds.add(to)
        changed.add(frm)
        changed.add(to)

    union_edges = {
        (e["from"], e["to"]) for g in (old_graph, new_graph) for e in g["edges"]
    }
    reverse: dict[str, set[str]] = {}
    for child, dep in union_edges:
        reverse.setdefault(dep, set()).add(child)

    # Self-hold: the changed node itself is included when present in new_graph.
    impacted: set[str] = {nid for nid in changed if nid in new_nodes}
    queue = list(changed)
    seen = set(queue)
    while queue:
        dep = queue.pop()
        for child in reverse.get(dep, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
            if child in new_nodes:
                impacted.add(child)

    clone = copy.deepcopy(new_graph)
    proposals: list[dict[str, Any]] = []
    for nid in sorted(impacted):
        node = clone["nodes"][nid]
        node["classification"] = "REVALIDATION_REQUIRED"
        node["controlling"] = False
        proposals.append(
            {
                "node": nid,
                "proposal": "REVALIDATION_REQUIRED",
                "promotion_permission": False,
            }
        )

    return {
        "changed_nodes": sorted(changed),
        "edge_only_seeds": sorted(edge_only_seeds),
        "impacted": sorted(impacted),
        "proposals": proposals,
        "graph": clone,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "meaning": "union-edge reverse-impact HOLD/REVALIDATION proposal; never promotion permission",
    }


def compare_claims_files(
    old_claims: dict[str, Any],
    new_claims: dict[str, Any],
    *,
    crosswalk: dict[str, Any] | None = None,
    authority_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    old_g = claims_to_gate_graph(old_claims, crosswalk=crosswalk, authority_map=authority_map)
    new_g = claims_to_gate_graph(new_claims, crosswalk=crosswalk, authority_map=authority_map)
    impact = reverse_impact_between(old_g, new_g)
    holds = {
        nid: required_holds(new_g, nid)
        for nid in sorted(new_g["nodes"])
        if required_holds(new_g, nid)["refuted_required"]
        or required_holds(new_g, nid)["blocked_absent"]
    }
    return {
        "reverse_impact": impact,
        "hold_proposals": holds,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "semantic_reference": "Math- PR13 tip baca69c394ab42130c61771bee74e808703f1ce7 / main #90 clarification",
        "meaning": "before/after claims adapter report; never promotion permission",
    }


def audit_tip(root: Path | None = None) -> dict[str, Any]:
    """Project the tip claims graph and summarize HOLD proposals (no mutation)."""
    root = root or ROOT
    claims = load_json(root / "claims" / "graph.json")
    crosswalk = load_json(root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json")
    authority = load_json(root / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json")
    graph = claims_to_gate_graph(claims, crosswalk=crosswalk, authority_map=authority)
    # Self-compare: identity before/after should yield empty impact.
    impact = reverse_impact_between(graph, copy.deepcopy(graph))
    return {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "sub_obligation_edges": sum(1 for e in graph["edges"] if e["relation"] == "sub_obligation"),
        "depends_on_edges": sum(1 for e in graph["edges"] if e["relation"] == "depends_on"),
        "identity_impacted": impact["impacted"],
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "problems": [],
        "meaning": "tip projection health; not mathematical acceptance",
    }


def main(argv: list[str] | None = None) -> int:
    report = audit_tip()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("problems") else 0


if __name__ == "__main__":
    sys.exit(main())
