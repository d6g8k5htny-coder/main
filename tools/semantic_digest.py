#!/usr/bin/env python3
"""Canonical semantic / evidence digests for scientific-state nodes (#95 v1.1).

Four orthogonal axes (digest match never promotes mathematics):
  semantic_digest     = H(canonical scientific content + typed deps)
  evidence_digest     = H(evidence identities)
  verification_level  = L0–L5 evidence metadata (separate)
  scientific_status   = owned by external authorities only (never derived here)

Manual fingerprints may remain provenance; they are not the sole change detector.
Scientific effect: NONE.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


class DigestError(ValueError):
    """Fail-closed digest / typed-edge refusal."""


def _require_strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise DigestError(
            f"{field} must be a strict boolean, got {type(value).__name__}: {value!r}"
        )
    return value


def _sha256_canonical(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalize_typed_edge(edge: dict[str, Any] | str, *, default_relation: str) -> dict[str, Any]:
    """Normalize a depends_on / sub_obligation entry into a typed edge record."""
    if isinstance(edge, str):
        return {
            "target_id": edge,
            "relation": default_relation,
            "required": True,
        }
    if not isinstance(edge, dict):
        raise DigestError("typed edge must be a string id or object")
    target = edge.get("target_id") or edge.get("to") or edge.get("id")
    if not isinstance(target, str) or not target:
        raise DigestError("typed edge missing target_id")
    relation = edge.get("relation") or edge.get("edge_type") or default_relation
    required = edge["required"] if "required" in edge else True
    _require_strict_bool(required, field="required")
    return {
        "target_id": target,
        "relation": relation,
        "required": required,
    }


def validate_typed_edge(edge: dict[str, Any]) -> None:
    if not isinstance(edge, dict):
        raise DigestError("typed edge must be an object")
    if not isinstance(edge.get("target_id"), str) or not edge["target_id"]:
        raise DigestError("typed edge missing target_id")
    if not isinstance(edge.get("relation"), str) or not edge["relation"]:
        raise DigestError("typed edge missing relation")
    _require_strict_bool(edge.get("required"), field="required")


def validate_source_binding(binding: dict[str, Any]) -> None:
    """Canonical source binding: owner/repo/path + immutable commit/blob/hash when present."""
    if not isinstance(binding, dict):
        raise DigestError("source binding must be an object")
    path = binding.get("path")
    if not isinstance(path, str) or not path:
        raise DigestError("source binding missing path")
    # Prefer structured owner/repo; accept repo string that already includes owner/name.
    has_owner_repo = (
        (isinstance(binding.get("owner"), str) and isinstance(binding.get("repo"), str))
        or isinstance(binding.get("repo"), str)
    )
    if not has_owner_repo:
        raise DigestError("source binding missing owner/repo")
    immutable = (
        binding.get("commit")
        or binding.get("blob")
        or binding.get("hash")
        or binding.get("sha256")
    )
    if immutable is not None and not isinstance(immutable, str):
        raise DigestError("source binding immutable identity must be a string when present")


def _typed_edges_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def container(field: str) -> list[Any]:
        if field not in record:
            return []
        value = record[field]
        if type(value) is not list:
            raise DigestError(
                f"{field} must be a list, got {type(value).__name__}: {value!r}"
            )
        return value

    for raw in container("depends_on"):
        edge = normalize_typed_edge(raw, default_relation="depends_on")
        validate_typed_edge(edge)
        key = (edge["target_id"], edge["relation"])
        if key in seen:
            raise DigestError(
                f"duplicate typed edge {edge['target_id']!r} ({edge['relation']})"
            )
        seen.add(key)
        edges.append(edge)
    for raw in container("sub_obligations"):
        edge = normalize_typed_edge(raw, default_relation="sub_obligation")
        validate_typed_edge(edge)
        key = (edge["target_id"], edge["relation"])
        if key in seen:
            raise DigestError(
                f"duplicate typed edge {edge['target_id']!r} ({edge['relation']})"
            )
        seen.add(key)
        edges.append(edge)
    return sorted(edges, key=lambda e: (e["relation"], e["target_id"], e["required"]))


def _source_bindings_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    # Prefer explicit source_bindings. Do not fold `evidence` into the semantic
    # axis — evidence identities belong exclusively to evidence_digest.
    raw = record.get("source_bindings") or []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise DigestError("source_bindings must be a list or object")
    for item in raw:
        if isinstance(item, str):
            bindings.append({"path": item, "legacy_string": True})
            continue
        if not isinstance(item, dict):
            raise DigestError("source binding entries must be objects or strings")
        if "path" in item:
            bindings.append(
                {
                    "owner": item.get("owner"),
                    "repo": item.get("repo"),
                    "path": item.get("path"),
                    "commit": item.get("commit"),
                    "blob": item.get("blob"),
                    "hash": item.get("hash") or item.get("sha256"),
                }
            )
        else:
            bindings.append({"legacy": item})
    # Scalar source / canon_source pointers remain part of scientific content.
    for key in ("source", "canon_source"):
        if key in record and record[key] is not None:
            bindings.append({"scalar_key": key, "value": record[key]})
    return bindings


def canonical_semantic_payload(claim_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Normalized scientific content used for semantic_digest.

    claim_id + statement/version + scope/domain + canonical source bindings +
    typed dependency/sub_obligation records. Excludes grades/statuses/controlling.
    """
    if not isinstance(claim_id, str) or not claim_id:
        raise DigestError("claim_id required")
    if not isinstance(record, dict):
        raise DigestError("record must be an object")
    return {
        "claim_id": claim_id,
        "statement": record.get("statement") or record.get("note"),
        "version": record.get("version") or record.get("as_of"),
        "scope": record.get("scope"),
        "domain": record.get("domain"),
        "source_bindings": _source_bindings_from_record(record),
        "typed_edges": _typed_edges_from_record(record),
    }


def semantic_digest(claim_id: str, record: dict[str, Any]) -> str:
    """H(canonical scientific content). Digest match ≠ mathematical acceptance."""
    return _sha256_canonical(canonical_semantic_payload(claim_id, record))


def evidence_digest(record: dict[str, Any]) -> str:
    """H(evidence identities only). Separate from semantic_digest."""
    if not isinstance(record, dict):
        raise DigestError("record must be an object")
    evidence = record.get("evidence") or record.get("source_bindings") or []
    if isinstance(evidence, dict):
        evidence = [evidence]
    identities: list[Any] = []
    for item in evidence:
        if isinstance(item, str):
            identities.append({"path": item})
        elif isinstance(item, dict):
            identities.append(
                {
                    "owner": item.get("owner"),
                    "repo": item.get("repo"),
                    "path": item.get("path"),
                    "commit": item.get("commit"),
                    "blob": item.get("blob"),
                    "hash": item.get("hash") or item.get("sha256"),
                }
            )
        else:
            raise DigestError("evidence entry must be string or object")
    return _sha256_canonical({"evidence_identities": identities})


def migrate_legacy_fingerprint(
    claim_id: str,
    record: dict[str, Any],
    *,
    legacy_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Crosswalk a legacy fingerprint-bearing node without rewriting the source record.

    Returns derived digests + provenance of the old fingerprint. Does not mutate
    `record` and does not treat fingerprint equality as scientific acceptance.
    """
    derived = semantic_digest(claim_id, record)
    return {
        "claim_id": claim_id,
        "semantic_digest": derived,
        "evidence_digest": evidence_digest(record),
        "legacy_fingerprint": legacy_fingerprint,
        "legacy_fingerprint_is_sole_detector": False,
        "scientific_effect": "NONE",
        "meaning": "migration projection; digest match is not acceptance",
    }
