#!/usr/bin/env python3
"""Thin hardening adapter: claims/graph.json → #90 gate semantics (D7).

Semantic reference: Math- PR13 (required REFUTED blocks; reverse impact over
UNION(old,new) edges; REQUIRED_SATISFIED = {PROVED_REVIEWED} only). This module
does **not** vendor Math- code and never grants promotion permission.

Outputs are HOLD / REVALIDATION proposals only. Scientific effect: NONE.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
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


def _require_as_of(claims: dict[str, Any]) -> str:
    """Fail closed unless as_of is a nonempty string (reject null/''/False/0)."""
    if "as_of" not in claims:
        raise AdapterError("stale or incomplete claims document: missing as_of")
    as_of = claims["as_of"]
    if type(as_of) is not str or not as_of.strip():
        raise AdapterError(
            f"as_of must be a nonempty string, got {type(as_of).__name__}: {as_of!r}"
        )
    return as_of


def _dependency_container(
    record: dict[str, Any], field: str, *, node_id: str
) -> list[Any]:
    """Return depends_on / sub_obligations list; reject False/0/''/{}/null.

    Valid omission of the key → empty list. Genuine empty array → empty list.
    """
    if field not in record:
        return []
    value = record[field]
    if type(value) is not list:
        raise AdapterError(
            f"{node_id}.{field} must be a list, got {type(value).__name__}: {value!r}"
        )
    return value


def claims_to_gate_graph(
    claims: dict[str, Any],
    *,
    crosswalk: dict[str, Any] | None = None,
    authority_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project claims/graph.json into a gate-shaped working graph.

    Includes depends_on and sub_obligations as required typed edges.
    Fail-closed on unknown IDs, cycles, malformed records, ambiguous owners,
    and missing/invalid as_of / schema identity.
    """
    if not isinstance(claims, dict):
        raise AdapterError("malformed claims document")
    as_of = _require_as_of(claims)
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
        # Validate dep containers before digest so False/0/''/{}/null fail closed.
        _dependency_container(record, "depends_on", node_id=nid)
        _dependency_container(record, "sub_obligations", node_id=nid)
        snapshot = _source_snapshot(record)
        sem, evid = _digests_for(nid, record)
        # Surface source grade/status; never invent controlling=True.
        source_controlling = record.get("controlling")
        if source_controlling is not None:
            source_controlling = _require_strict_bool(
                source_controlling, field=f"{nid}.controlling"
            )
        nodes[nid] = {
            "bucket": bucket,
            "classification": map_classification(record, bucket=bucket),
            "controlling": False,  # projection never grants controlling
            "source_controlling": source_controlling,
            "source_grade": record.get("grade"),
            "source_status": record.get("status_frozen_v2_2")
            or record.get("status_register_note"),
            # Keep every source-status channel: `or` must not hide a claim
            # of controlling use behind another, noncontrolling field.
            "source_statuses": [
                record.get(key)
                for key in (
                    "status_frozen_v2_2",
                    "status_register_note",
                    "scientific_status",
                )
            ],
            "source_reference": record.get("source") or record.get("canon_source"),
            "semantic_digest": sem,
            "evidence_digest": evid,
            "source_snapshot": snapshot,
            "fingerprint": sem,  # derived digest; never a manual sole detector
            "version": as_of,
        }

    for nid, record in premises.items():
        add_node(nid, "premises", record)
    for nid, record in claim_nodes.items():
        add_node(nid, "claims", record)

    def add_deps(nid: str, record: dict[str, Any]) -> None:
        for dep in _dependency_container(record, "depends_on", node_id=nid):
            if not isinstance(dep, str) or not dep:
                raise AdapterError(f"malformed depends_on entry on {nid!r}: {dep!r}")
            if dep not in nodes:
                raise AdapterError(f"unknown dependency id {dep!r} from {nid!r}")
            edges.append(_edge(nid, dep, "depends_on"))
        for dep in _dependency_container(record, "sub_obligations", node_id=nid):
            if not isinstance(dep, str) or not dep:
                raise AdapterError(f"malformed sub_obligation entry on {nid!r}: {dep!r}")
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
        "as_of": as_of,
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


def reverse_impact_between(
    old_graph: dict[str, Any],
    new_graph: dict[str, Any],
    *,
    old_sources: dict[str, Any] | None = None,
    new_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reverse impact over UNION(old,new) edges; deleted edges cannot erase impact.

    Seeds include:
      - nodes whose canonical source snapshot / classification / version change;
      - endpoints of edge-only changes (add/remove/mutate) even when fingerprints
        and classifications are unchanged;
      - nodes whose bound repository source-file bytes change (PR15 contract);
      - the changed node itself (self-hold / self-revalidation).

    Attaches REVALIDATION_REQUIRED proposals without erasing REFUTED.
    Never sets controlling=True. Never grants promotion permission.
    """
    validate_graph_fail_closed(old_graph)
    validate_graph_fail_closed(new_graph)
    old_nodes, new_nodes = old_graph["nodes"], new_graph["nodes"]
    all_ids = set(old_nodes) | set(new_nodes)
    changed: set[str] = set()
    source_byte_seeds: set[str] = set()
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

    # Source FILE byte drift (distinct from claims-record JSON snapshot).
    if old_sources is not None and new_sources is not None:
        for nid in set(old_sources) | set(new_sources):
            old_s = old_sources.get(nid) or {"kind": "absent", "coverage_sha256": ""}
            new_s = new_sources.get(nid) or {"kind": "absent", "coverage_sha256": ""}
            old_cov = old_s.get("coverage_sha256") or old_s.get("sha256")
            new_cov = new_s.get("coverage_sha256") or new_s.get("sha256")
            if old_cov != new_cov or old_s.get("kind") != new_s.get("kind"):
                if (
                    old_s.get("kind") in {"blob", "tree", "multi"}
                    or new_s.get("kind") in {"blob", "tree", "multi"}
                    or old_cov
                    or new_cov
                ):
                    source_byte_seeds.add(nid)
                    changed.add(nid)
                elif old_s.get("kind") != new_s.get("kind"):
                    source_byte_seeds.add(nid)
                    changed.add(nid)

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
        # Preserve REFUTED (and other terminal dispositions) while attaching proposal.
        if node.get("classification") not in REFUTED_CLASSIFICATIONS:
            node["classification"] = "REVALIDATION_REQUIRED"
        node["revalidation_proposal"] = "REVALIDATION_REQUIRED"
        node["controlling"] = False
        proposals.append(
            {
                "node": nid,
                "proposal": "REVALIDATION_REQUIRED",
                "source_grade": node.get("source_grade"),
                "source_status": node.get("source_status"),
                "source_controlling": node.get("source_controlling"),
                "preserved_classification": node.get("classification"),
                "promotion_permission": False,
            }
        )

    return {
        "changed_nodes": sorted(changed),
        "edge_only_seeds": sorted(edge_only_seeds),
        "source_byte_seeds": sorted(source_byte_seeds),
        "impacted": sorted(impacted),
        "proposals": proposals,
        "graph": clone,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "meaning": "union-edge reverse-impact HOLD/REVALIDATION proposal; never promotion permission",
    }


def node_has_hold(hold: dict[str, Any]) -> bool:
    """True when required_holds found unresolved still-required premises.

    Includes unsatisfied_required (AUTHOR_SIDE_CANDIDATE / OPEN_ACTIVE /
    SUPERSEDED_NONBLOCKING / etc.), not only REFUTED / BLOCKED_ABSENT.
    Does not treat the default proposals=["HOLD"] padding as a real hold.
    """
    return bool(
        hold.get("refuted_required")
        or hold.get("blocked_absent")
        or hold.get("unsatisfied_required")
    )


def aggregate_hold_proposals(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fail-closed HOLD map over every node with unresolved required premises."""
    holds: dict[str, dict[str, Any]] = {}
    for nid in sorted(graph["nodes"]):
        hold = required_holds(graph, nid)
        if node_has_hold(hold):
            holds[nid] = hold
    return holds


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
    holds = aggregate_hold_proposals(new_g)
    enforcement = evaluate_transition_enforcement(
        old_graph=old_g, new_graph=new_g, impact=impact, holds=holds
    )
    return {
        **enforcement,
        "reverse_impact": impact,
        "hold_proposals": holds,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "semantic_reference": "Math- PR13 tip baca69c394ab42130c61771bee74e808703f1ce7 / main #90 clarification",
        "meaning": "before/after claims adapter report; never promotion permission",
    }


def audit_tip(root: Path | None = None) -> dict[str, Any]:
    """Tip-health only: identity self-compare + HOLD inventory. Not base→head evidence."""
    root = root or ROOT
    claims = load_json(root / "claims" / "graph.json")
    crosswalk = load_json(root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json")
    authority = load_json(root / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json")
    graph = claims_to_gate_graph(claims, crosswalk=crosswalk, authority_map=authority)
    # Self-compare: identity before/after should yield empty impact.
    impact = reverse_impact_between(graph, copy.deepcopy(graph))
    holds = aggregate_hold_proposals(graph)
    hold_nodes = sorted(holds)
    return {
        "mode": "tip_health",
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "sub_obligation_edges": sum(1 for e in graph["edges"] if e["relation"] == "sub_obligation"),
        "depends_on_edges": sum(1 for e in graph["edges"] if e["relation"] == "depends_on"),
        "identity_impacted": impact["impacted"],
        "hold_node_count": len(hold_nodes),
        "hold_nodes": hold_nodes,
        "hold_proposals": holds,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "problems": [],
        "meaning": (
            "tip-health only (identity self-compare + HOLD inventory); "
            "NOT evidence that a PR/push base→head transition is clean"
        ),
    }


ZERO_SHA = "0" * 40
CLAIMS_REL = "claims/graph.json"
CROSSWALK_REL = "architecture/scientific_state/v1/ID_CROSSWALK.json"
AUTHORITY_REL = "architecture/scientific_state/v1/AUTHORITY_MAP.json"

# Grades/statuses that indicate the *source* record is treated as load-bearing.
# Projection never sets controlling=True; these only flag report attention /
# loss-only transition refusal (never positive acceptance).
SOURCE_CONTROLLING_HINTS = frozenset(
    {
        "LIVE_ROOT_THEOREM",
        "CERTIFIED_RUNG",
        "CONTROLLING",
        "PROVED_REVIEWED",
    }
)


def _source_is_controlling(node: dict[str, Any]) -> bool:
    """Interpret all observed source channels without awarding acceptance.

    Explicit false does not override a simultaneously controlling grade/status.
    A demotion must consistently remove the source's claims of controlling use.
    Ported from OpenAI trial PR128 F1 candidate (ACCEPT'd nonauthor review).
    """
    if node.get("source_controlling") is True:
        return True
    statuses = [node.get("source_grade"), node.get("source_status")]
    statuses.extend(node.get("source_statuses") or [])
    return any(
        isinstance(value, str) and value.strip().upper() in SOURCE_CONTROLLING_HINTS
        for value in statuses
    )


def evaluate_transition_enforcement(
    *,
    old_graph: dict[str, Any],
    new_graph: dict[str, Any],
    impact: dict[str, Any],
    holds: dict[str, Any],
) -> dict[str, Any]:
    """Loss-only refusal, not a positive mathematical admission mechanism.

    No newly controlling source, or impacted source still used as controlling,
    can be admitted here merely by attaching a proposed HOLD. Unchanged legacy
    statuses remain visible; they are not retroactively accepted. Corrective
    edits with consistent demotion are allowed. F2–F5 (source bind / crosswalk /
    refs / strict JSON) remain separate layers.
    """
    old_controls = {
        nid for nid, node in old_graph["nodes"].items() if _source_is_controlling(node)
    }
    new_controls = {
        nid for nid, node in new_graph["nodes"].items() if _source_is_controlling(node)
    }
    newly_controlling = new_controls - old_controls
    retained_impacted = new_controls.intersection(impact["impacted"])
    errors: list[dict[str, Any]] = []
    for nid in sorted(newly_controlling | retained_impacted):
        reasons: list[str] = []
        if nid in newly_controlling:
            reasons.append("UNSUPPORTED_CONTROLLING_PROMOTION")
        if nid in retained_impacted:
            reasons.append("CONTROLLING_SOURCE_REQUIRES_REVALIDATION")
        if nid in holds:
            reasons.append("UNSATISFIED_REQUIRED_PREMISE")
        errors.append(
            {
                "node": nid,
                "reasons": reasons,
                "required_hold": holds.get(nid),
                "promotion_permission": False,
            }
        )
    return {
        "transition_ok": not errors,
        "transition_errors": errors,
        # Compat alias for earlier Cursor tip reports / FiveBoundary tests.
        "illegal_controlling_transitions": errors,
        "source_controlling_after": sorted(new_controls),
        "unchanged_controlling_holds": sorted(
            (new_controls - retained_impacted).intersection(holds)
        ),
        "promotion_permission": False,
        "enforcement_scope": (
            "loss-only source transition over the detected impact graph; "
            "not positive admission, complete source coverage, or legacy acceptance"
        ),
        "meaning": (
            "transition_ok is false for unsupported new controlling status or "
            "impacted consumers left controlling; consistent demotion remains OK"
        ),
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json_strict(raw: str | bytes, *, where: str = "json") -> Any:
    """Fail closed on duplicate keys and nonfinite constants (NaN/Infinity)."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = raw

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AdapterError(f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise AdapterError(f"{where}: nonfinite JSON value {value!r}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{where}: malformed JSON: {exc}") from exc


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return load_json_strict(handle.read(), where=str(path))


def relative_repo_path(value: str) -> str:
    """PR15 contract: repository-relative path only; reject absolute/.. /URL-ish."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or "\\" in value
    ):
        raise AdapterError(f"repository-relative source path required: {value!r}")
    return path.as_posix()


def _candidate_source_strings(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Yield (kind_hint, reference, field) triples from a claims record.

    kind_hint: 'path' | 'external' | 'prose'
    field: provenance of the declaration (mirror_path / source.path / source_bindings / …)
    """
    out: list[tuple[str, str, str]] = []
    for key in ("mirror_path",):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            out.append(("path", val.strip(), key))
    # Canonical source_bindings used by semantic_digest — must also bind bytes.
    raw_bindings = record.get("source_bindings")
    if isinstance(raw_bindings, dict):
        raw_bindings = [raw_bindings]
    if isinstance(raw_bindings, list):
        for index, item in enumerate(raw_bindings):
            if isinstance(item, str) and item.strip():
                out.append(("path", item.strip(), f"source_bindings[{index}]"))
            elif isinstance(item, dict):
                path = item.get("path") or item.get("repo_path") or item.get("file")
                if isinstance(path, str) and path.strip():
                    out.append(("path", path.strip(), f"source_bindings[{index}].path"))
                url = item.get("url") or item.get("uri")
                if isinstance(url, str) and url.strip():
                    out.append(("external", url.strip(), f"source_bindings[{index}].url"))
    src = record.get("source")
    if isinstance(src, dict):
        path = src.get("path") or src.get("repo_path") or src.get("file")
        if isinstance(path, str) and path.strip():
            out.append(("path", path.strip(), "source.path"))
        url = src.get("url") or src.get("uri")
        if isinstance(url, str) and url.strip():
            out.append(("external", url.strip(), "source.url"))
    elif isinstance(src, str) and src.strip():
        text = src.strip()
        if text.startswith(("https://", "http://", "external:")):
            out.append(("external", text, "source"))
        else:
            first = re.split(r"[;]", text, maxsplit=1)[0].strip()
            first = re.split(r"\s+§", first, maxsplit=1)[0].strip()
            if (
                "/" in first
                or first.endswith((".md", ".json", ".py", ".lean", ".txt"))
            ) and " " not in first:
                out.append(("path", first, "source"))
            else:
                out.append(("prose", text, "source"))
    canon = record.get("canon_source")
    if isinstance(canon, str) and canon.strip():
        if canon.startswith(("https://", "http://", "external:")):
            out.append(("external", canon.strip(), "canon_source"))
        else:
            out.append(("prose", canon.strip(), "canon_source"))
    return out


def _bind_one_path(
    root: Path, revision: str, reference: str, *, field: str
) -> dict[str, Any]:
    try:
        path = relative_repo_path(reference)
    except AdapterError:
        return {
            "kind": "unresolved_prose",
            "reference": reference,
            "field": field,
        }
    object_spec = f"{revision}:{path.rstrip('/')}"
    kind = _git_bytes(root, "cat-file", "-t", object_spec, missing_ok=True)
    if kind is None:
        return {
            "kind": "missing",
            "reference": reference,
            "path": path,
            "field": field,
        }
    kind_s = kind.decode().strip()
    if kind_s not in {"blob", "tree"}:
        raise AdapterError(f"source object must be blob or tree: {object_spec}")
    body = _git_bytes(root, "cat-file", "-p", object_spec)
    assert body is not None
    return {
        "kind": kind_s,
        "reference": reference,
        "path": path,
        "field": field,
        "bytes": len(body),
        "sha256": _sha256_bytes(body),
    }


def bind_source_at_revision(
    root: Path, revision: str, record: dict[str, Any]
) -> dict[str, Any]:
    """Bind ALL declared load-bearing repository sources at a revision.

    Does not let mirror_path silently shadow source.path / source_bindings.
    External refs → external_unresolved (never fetched). Missing → missing.
    Prose-only → unresolved_prose. No invention of absent objects.
    """
    candidates = _candidate_source_strings(record)
    if not candidates:
        return {
            "kind": "record_only",
            "bindings": [],
            "coverage_sha256": _sha256_bytes(b"record_only"),
        }
    bindings: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for kind_hint, reference, field in candidates:
        if kind_hint == "path":
            # Deduplicate identical path strings but keep distinct field origins
            # when paths differ (mirror vs source.path).
            key = reference
            if key in seen_paths:
                # Same path from another field — still record the field alias.
                bindings.append(
                    {
                        "kind": "path_alias",
                        "reference": reference,
                        "field": field,
                        "alias_of": reference,
                    }
                )
                continue
            seen_paths.add(key)
            bindings.append(_bind_one_path(root, revision, reference, field=field))
        elif kind_hint == "external":
            bindings.append(
                {
                    "kind": "external_unresolved",
                    "reference": reference,
                    "field": field,
                }
            )
        else:
            bindings.append(
                {
                    "kind": "unresolved_prose",
                    "reference": reference,
                    "field": field,
                }
            )
    # coverage hash over all binding identities so any drift seeds impact.
    coverage_payload = [
        {
            "field": b.get("field"),
            "kind": b.get("kind"),
            "path": b.get("path"),
            "reference": b.get("reference"),
            "sha256": b.get("sha256"),
        }
        for b in bindings
    ]
    blob_bindings = [b for b in bindings if b.get("kind") in {"blob", "tree"}]
    if blob_bindings:
        primary_kind = "multi" if len(blob_bindings) > 1 else blob_bindings[0]["kind"]
    elif any(b.get("kind") == "external_unresolved" for b in bindings):
        primary_kind = "external_unresolved"
    elif any(b.get("kind") == "missing" for b in bindings):
        primary_kind = "missing"
    elif any(b.get("kind") == "unresolved_prose" for b in bindings):
        primary_kind = "unresolved_prose"
    else:
        primary_kind = "record_only"
    return {
        "kind": primary_kind,
        "bindings": bindings,
        "coverage_sha256": _sha256_canonical_payload(coverage_payload),
        # Backward-compatible single-binding view: first real blob/tree if any.
        **(
            {
                "reference": blob_bindings[0].get("reference"),
                "path": blob_bindings[0].get("path"),
                "bytes": blob_bindings[0].get("bytes"),
                "sha256": blob_bindings[0].get("sha256"),
                "field": blob_bindings[0].get("field"),
            }
            if blob_bindings
            else {}
        ),
    }


def _sha256_canonical_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(blob.encode("utf-8"))


def bind_claims_sources_at_ref(
    root: Path, revision: str, claims: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for bucket in ("premises", "claims"):
        for nid, record in (claims.get(bucket) or {}).items():
            if not isinstance(record, dict):
                raise AdapterError(f"malformed record for {nid!r}")
            sources[nid] = bind_source_at_revision(root, revision, record)
    return sources


def _git_bytes(
    root: Path, *args: str, missing_ok: bool = False
) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        if missing_ok:
            return None
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise AdapterError(f"git {' '.join(args)} failed: {err or 'unknown error'}")
    return result.stdout


def _require_usable_ref(
    ref: str,
    *,
    role: str,
    root: Path | None = None,
    resolve: bool = True,
) -> str:
    """Reject empty/all-zero refs; optionally resolve once to a full 40-hex commit."""
    if not isinstance(ref, str) or not ref.strip():
        raise AdapterError(f"{role} ref missing or empty")
    cleaned = ref.strip()
    if cleaned == ZERO_SHA or set(cleaned) == {"0"}:
        raise AdapterError(f"{role} ref unavailable or all-zero: {cleaned!r}")
    if not resolve:
        return cleaned
    if root is None:
        # Event extraction without a compare root keeps the raw token; compare_claims_refs
        # always resolves against the actual repository being audited.
        return cleaned
    resolved = _git_bytes(
        root, "rev-parse", "--verify", f"{cleaned}^{{commit}}", missing_ok=True
    )
    if resolved is None:
        raise AdapterError(f"{role} ref is not a resolvable commit: {cleaned!r}")
    full = resolved.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", full):
        raise AdapterError(f"{role} ref did not resolve to a 40-hex commit id: {full!r}")
    kind = _git_bytes(root, "cat-file", "-t", full, missing_ok=True)
    if kind is None or kind.decode().strip() != "commit":
        raise AdapterError(f"{role} ref is not a commit object: {full!r}")
    return full


def git_show_bytes(root: Path, ref: str, relpath: str) -> bytes:
    data = _git_bytes(root, "show", f"{ref}:{relpath}")
    assert data is not None
    return data


def load_claims_at_ref(root: Path, ref: str) -> tuple[dict[str, Any], dict[str, str]]:
    raw = git_show_bytes(root, ref, CLAIMS_REL)
    claims = load_json_strict(raw, where=f"{ref}:{CLAIMS_REL}")
    identity = {
        "ref": ref,
        "path": CLAIMS_REL,
        "blob_sha256": _sha256_bytes(raw),
        "bytes": str(len(raw)),
    }
    return claims, identity


def load_json_at_ref(root: Path, ref: str, relpath: str) -> tuple[Any, dict[str, str]]:
    raw = git_show_bytes(root, ref, relpath)
    return load_json_strict(raw, where=f"{ref}:{relpath}"), {
        "ref": ref,
        "path": relpath,
        "blob_sha256": _sha256_bytes(raw),
        "bytes": str(len(raw)),
    }


def _owner_map(crosswalk: dict[str, Any] | None) -> dict[str, str]:
    """main_id → authority from crosswalk rows (absent schema → empty)."""
    owners: dict[str, str] = {}
    if not isinstance(crosswalk, dict):
        return owners
    for row in crosswalk.get("rows") or []:
        if not isinstance(row, dict):
            continue
        main_id = row.get("main_claim_or_premise_id")
        auth = row.get("authority")
        if isinstance(main_id, str) and main_id and isinstance(auth, str) and auth:
            owners[main_id] = auth
    return owners


def compare_claims_paths(
    before_path: Path,
    after_path: Path,
    *,
    crosswalk_path: Path | None = None,
    authority_path: Path | None = None,
) -> dict[str, Any]:
    before_raw = before_path.read_bytes()
    after_raw = after_path.read_bytes()
    before_claims = load_json_strict(before_raw, where=str(before_path))
    after_claims = load_json_strict(after_raw, where=str(after_path))
    crosswalk = load_json(crosswalk_path) if crosswalk_path else None
    authority = load_json(authority_path) if authority_path else None
    old_g = claims_to_gate_graph(
        before_claims, crosswalk=crosswalk, authority_map=authority
    )
    new_g = claims_to_gate_graph(
        after_claims, crosswalk=crosswalk, authority_map=authority
    )
    impact = reverse_impact_between(old_g, new_g)
    holds = aggregate_hold_proposals(new_g)
    enforcement = evaluate_transition_enforcement(
        old_graph=old_g, new_graph=new_g, impact=impact, holds=holds
    )
    return {
        "mode": "path_compare",
        "before_identity": {
            "path": str(before_path),
            "blob_sha256": _sha256_bytes(before_raw),
            "bytes": len(before_raw),
        },
        "after_identity": {
            "path": str(after_path),
            "blob_sha256": _sha256_bytes(after_raw),
            "bytes": len(after_raw),
        },
        "reverse_impact": impact,
        "hold_proposals": holds,
        **enforcement,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "semantic_reference": "Math- PR13/PR15 / main #90 clarification",
        "meaning": (
            "path compare; unsupported new/retained controlling fails; "
            "consistent demotion and noncontrolling impact remain OK"
        ),
    }


def compare_claims_refs(
    before_ref: str,
    after_ref: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or ROOT
    before_ref = _require_usable_ref(before_ref, role="before", root=root)
    after_ref = _require_usable_ref(after_ref, role="after", root=root)
    before_claims, before_id = load_claims_at_ref(root, before_ref)
    after_claims, after_id = load_claims_at_ref(root, after_ref)
    try:
        old_crosswalk, old_crosswalk_id = load_json_at_ref(
            root, before_ref, CROSSWALK_REL
        )
    except AdapterError:
        old_crosswalk, old_crosswalk_id = {"rows": []}, {
            "ref": before_ref,
            "path": CROSSWALK_REL,
            "blob_sha256": "",
            "bytes": "0",
            "absent_old_schema": True,
        }
    try:
        old_authority, old_authority_id = load_json_at_ref(
            root, before_ref, AUTHORITY_REL
        )
    except AdapterError:
        old_authority, old_authority_id = {"authorities": {}}, {
            "ref": before_ref,
            "path": AUTHORITY_REL,
            "blob_sha256": "",
            "bytes": "0",
            "absent_old_schema": True,
        }
    crosswalk, crosswalk_id = load_json_at_ref(root, after_ref, CROSSWALK_REL)
    authority, authority_id = load_json_at_ref(root, after_ref, AUTHORITY_REL)
    old_g = claims_to_gate_graph(
        before_claims, crosswalk=old_crosswalk, authority_map=old_authority
    )
    new_g = claims_to_gate_graph(
        after_claims, crosswalk=crosswalk, authority_map=authority
    )
    old_sources = bind_claims_sources_at_ref(root, before_ref, before_claims)
    new_sources = bind_claims_sources_at_ref(root, after_ref, after_claims)
    impact = reverse_impact_between(
        old_g, new_g, old_sources=old_sources, new_sources=new_sources
    )
    old_owners = _owner_map(old_crosswalk)
    new_owners = _owner_map(crosswalk)
    authority_seeds: list[str] = []
    for main_id in set(old_owners) | set(new_owners):
        if old_owners.get(main_id) != new_owners.get(main_id):
            authority_seeds.append(main_id)
            if main_id in new_g["nodes"] and main_id not in impact["impacted"]:
                impact["impacted"] = sorted(set(impact["impacted"]) | {main_id})
                impact["changed_nodes"] = sorted(
                    set(impact["changed_nodes"]) | {main_id}
                )
                node = impact["graph"]["nodes"][main_id]
                if node.get("classification") not in REFUTED_CLASSIFICATIONS:
                    node["classification"] = "REVALIDATION_REQUIRED"
                node["revalidation_proposal"] = "REVALIDATION_REQUIRED"
                node["controlling"] = False
                impact["proposals"].append(
                    {
                        "node": main_id,
                        "proposal": "REVALIDATION_REQUIRED",
                        "source_grade": node.get("source_grade"),
                        "source_status": node.get("source_status"),
                        "source_controlling": node.get("source_controlling"),
                        "preserved_classification": node.get("classification"),
                        "promotion_permission": False,
                        "reason": "crosswalk_authority_owner_change",
                    }
                )
    holds = aggregate_hold_proposals(new_g)
    controlling_impacted = []
    unresolved_controlling = []
    for nid in impact["impacted"]:
        if nid not in new_g["nodes"]:
            continue
        node = new_g["nodes"][nid]
        if _source_is_controlling(node):
            controlling_impacted.append(nid)
            src = new_sources.get(nid) or {}
            if src.get("kind") not in {"blob", "tree", "multi"}:
                unresolved_controlling.append(nid)
    enforcement = evaluate_transition_enforcement(
        old_graph=old_g, new_graph=new_g, impact=impact, holds=holds
    )
    return {
        "mode": "ref_compare",
        "base_commit": before_ref,
        "head_commit": after_ref,
        "before_ref": before_ref,
        "after_ref": after_ref,
        "before_identity": before_id,
        "after_identity": after_id,
        "crosswalk_identity": crosswalk_id,
        "authority_identity": authority_id,
        "old_crosswalk_identity": old_crosswalk_id,
        "old_authority_identity": old_authority_id,
        "old_sources": old_sources,
        "new_sources": new_sources,
        "authority_owner_seeds": sorted(authority_seeds),
        "reverse_impact": impact,
        "hold_proposals": holds,
        "controlling_impacted": controlling_impacted,
        "unresolved_controlling_sources": unresolved_controlling,
        **enforcement,
        "promotion_permission": False,
        "scientific_effect": "NONE",
        "semantic_reference": (
            "Math- PR13 tip baca69c… / PR15 git_transition_audit contract 8c4c946… "
            "/ OpenAI trial PR128 F1 / main #90 clarification"
        ),
        "meaning": (
            "immutable base→head claims+source-file compare (PR15 + F1); "
            "unsupported new/retained controlling fails transition_ok; "
            "F2–F5 bind/crosswalk/ref/JSON layers retained; never promotion permission"
        ),
    }



def resolve_event_refs(
    *,
    environ: dict[str, str] | None = None,
    event: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[str, str, str]:
    """Return (before_ref, after_ref, event_name). Fail closed if unavailable.

    When ``root`` is provided, refs resolve to full immutable commit IDs in that
    repository. Without ``root``, only empty/all-zero rejection runs (unit harnesses);
    ``compare_claims_refs`` always resolves against the audited checkout.
    """
    import os

    env = environ if environ is not None else os.environ
    resolve = root is not None
    # Explicit overrides for sentinels / local harnesses.
    override_before = env.get("CLAIMS_GATE_BEFORE_REF")
    override_after = env.get("CLAIMS_GATE_AFTER_REF")
    if override_before or override_after:
        if not override_before or not override_after:
            raise AdapterError(
                "CLAIMS_GATE_BEFORE_REF and CLAIMS_GATE_AFTER_REF must both be set"
            )
        return (
            _require_usable_ref(
                override_before, role="before", root=root, resolve=resolve
            ),
            _require_usable_ref(
                override_after, role="after", root=root, resolve=resolve
            ),
            "env_override",
        )

    event_name = (env.get("GITHUB_EVENT_NAME") or "").strip()
    if event is None:
        event_path = env.get("GITHUB_EVENT_PATH")
        if not event_path:
            raise AdapterError(
                "event-compare requires GITHUB_EVENT_PATH or CLAIMS_GATE_BEFORE_REF/"
                "CLAIMS_GATE_AFTER_REF (no tip self-compare fallback)"
            )
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise AdapterError("malformed GitHub event payload")

    if event_name == "pull_request" or "pull_request" in event:
        pr = event.get("pull_request") or {}
        base = (pr.get("base") or {}).get("sha")
        head = (pr.get("head") or {}).get("sha")
        return (
            _require_usable_ref(
                str(base or ""), role="before", root=root, resolve=resolve
            ),
            _require_usable_ref(
                str(head or ""), role="after", root=root, resolve=resolve
            ),
            "pull_request",
        )
    if event_name == "push" or ("before" in event and "after" in event):
        return (
            _require_usable_ref(
                str(event.get("before") or ""), role="before", root=root, resolve=resolve
            ),
            _require_usable_ref(
                str(event.get("after") or ""), role="after", root=root, resolve=resolve
            ),
            "push",
        )
    raise AdapterError(
        f"unsupported or missing GitHub event for claims-gate compare: {event_name!r}"
    )


def event_compare(
    *,
    root: Path | None = None,
    environ: dict[str, str] | None = None,
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compare_root = root or ROOT
    before_ref, after_ref, event_name = resolve_event_refs(
        environ=environ, event=event, root=compare_root
    )
    report = compare_claims_refs(before_ref, after_ref, root=compare_root)
    report["mode"] = "event_compare"
    report["event_name"] = event_name
    # Keep enforcement transition_ok from compare_claims_refs. Impact/HOLD alone
    # remain admissible; unsupported controlling-over-unresolved fails closed.
    report["meaning"] = (
        "event-derived immutable base→head compare; REVALIDATION/HOLD proposals "
        "do not block corrective edits; unsupported controlling over unresolved "
        "required premises fails transition_ok; never promotion permission"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="claims_gate_adapter",
        description=(
            "Thin #90 claims→gate adapter. tip-health is identity-only; "
            "compare/event-compare exercise real before/after inputs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "tip-health",
        help="Identity self-compare + HOLD inventory (NOT base→head evidence)",
    )

    compare_p = sub.add_parser(
        "compare",
        help="Compare two claims JSON files (path mode)",
    )
    compare_p.add_argument("--before", type=Path, required=True)
    compare_p.add_argument("--after", type=Path, required=True)
    compare_p.add_argument("--crosswalk", type=Path, default=None)
    compare_p.add_argument("--authority", type=Path, default=None)
    compare_p.add_argument("--write-report", type=Path, default=None)

    refs_p = sub.add_parser(
        "compare-refs",
        help="Compare claims at two immutable git refs",
    )
    refs_p.add_argument("--before-ref", required=True)
    refs_p.add_argument("--after-ref", required=True)
    refs_p.add_argument("--repo-root", type=Path, default=None)
    refs_p.add_argument("--write-report", type=Path, default=None)

    event_p = sub.add_parser(
        "event-compare",
        help="Compare using GitHub event base/head (or CLAIMS_GATE_*_REF overrides)",
    )
    event_p.add_argument("--repo-root", type=Path, default=None)
    event_p.add_argument("--write-report", type=Path, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "tip-health":
            report = audit_tip()
        elif args.command == "compare":
            if not args.before.is_file() or not args.after.is_file():
                raise AdapterError("compare --before/--after must be existing files")
            report = compare_claims_paths(
                args.before,
                args.after,
                crosswalk_path=args.crosswalk,
                authority_path=args.authority,
            )
        elif args.command == "compare-refs":
            report = compare_claims_refs(
                args.before_ref,
                args.after_ref,
                root=args.repo_root or ROOT,
            )
        elif args.command == "event-compare":
            report = event_compare(root=args.repo_root or ROOT)
        else:
            raise AdapterError(f"unknown command {args.command!r}")
    except AdapterError as exc:
        print(json.dumps({"error": str(exc), "promotion_permission": False,
                          "scientific_effect": "NONE"}, indent=2, sort_keys=True))
        return 1

    write_report = getattr(args, "write_report", None)
    if write_report is not None:
        out = write_report.resolve()
        repo_root = (getattr(args, "repo_root", None) or ROOT).resolve()
        # PR15 contract: report must be new and outside the repository checkout.
        try:
            out.relative_to(repo_root)
            inside = True
        except ValueError:
            inside = False
        if inside:
            print(
                json.dumps(
                    {
                        "error": (
                            f"write-report must be outside the repository "
                            f"(refused in-tree path {out})"
                        ),
                        "promotion_permission": False,
                        "scientific_effect": "NONE",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        if out.exists():
            print(
                json.dumps(
                    {
                        "error": f"write-report must be a new file: {out}",
                        "promotion_permission": False,
                        "scientific_effect": "NONE",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    # Diagnostics may propose HOLD without failing. Transition modes must also
    # refuse unsafe actual source status; missing enforcement fails closed.
    if args.command != "tip-health" and report.get("transition_ok") is not True:
        return 1
    if report.get("promotion_permission") is True:
        return 1
    if report.get("problems"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
