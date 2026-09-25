#!/usr/bin/env python3
"""Fail-closed checker for architecture/scientific_state (main #95 pilot).

Validates schema shape, authority ownership, ID crosswalk referential integrity
against local claims/graph.json, and refuses smuggled status/grade/classification/
controlling payloads in this package.

Scientific effect: NONE. This tool never promotes mathematics, flips lemma_closed,
or writes authoritative statuses. It can only REFUSE.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "architecture" / "scientific_state" / "v1"
CLAIMS_GRAPH = ROOT / "claims" / "graph.json"

REQUIRED_FILES = (
    "SCHEMA.json",
    "AUTHORITY_MAP.json",
    "ID_CROSSWALK.json",
    "VERIFICATION_LEVELS.json",
)

FORBIDDEN_OWNED = frozenset(
    {
        "status",
        "grade",
        "classification",
        "controlling",
        "lemma_closed",
        "prizes_solved",
        "independence_credit",
    }
)

REQUIRED_SCHEMA_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "object",
        "scientific_effect",
        "node_fields",
        "forbidden_owned_fields",
    }
)

REQUIRED_LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def walk_forbidden(obj: Any, path: str, problems: list[str], *, allow_schema_meta: bool) -> None:
    """Refuse forbidden owned field names appearing as object keys.

    SCHEMA.json may *list* the forbidden names under forbidden_owned_fields and
    may document them inside node_fields notes — those meta locations are
    allowed. AUTHORITY_MAP may name them under owns/never_writes lists.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else key
            if key in FORBIDDEN_OWNED:
                if allow_schema_meta and path in {
                    "",
                    "node_fields",
                    "forbidden_owned_fields",
                }:
                    # SCHEMA top-level listing / documentation only.
                    pass
                elif allow_schema_meta and path.startswith("node_fields."):
                    pass
                else:
                    problems.append(
                        f"{path or '<root>'}: forbidden owned field key {key!r} "
                        f"(statuses/grades/classifications belong to external authorities)"
                    )
            walk_forbidden(value, child, problems, allow_schema_meta=allow_schema_meta)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            walk_forbidden(value, f"{path}[{index}]", problems, allow_schema_meta=allow_schema_meta)


def check_schema(schema: dict[str, Any], problems: list[str]) -> None:
    missing = sorted(REQUIRED_SCHEMA_KEYS - set(schema))
    for key in missing:
        problems.append(f"SCHEMA.json missing required key {key!r}")
    if schema.get("scientific_effect") != "NONE":
        problems.append(
            f"SCHEMA.json scientific_effect must be 'NONE', got {schema.get('scientific_effect')!r}"
        )
    forbidden = schema.get("forbidden_owned_fields")
    if not isinstance(forbidden, list) or not FORBIDDEN_OWNED.issubset(set(forbidden)):
        problems.append(
            "SCHEMA.json forbidden_owned_fields must include "
            + ", ".join(sorted(FORBIDDEN_OWNED))
        )
    node_fields = schema.get("node_fields")
    if not isinstance(node_fields, dict) or "claim_id" not in node_fields:
        problems.append("SCHEMA.json node_fields.claim_id is required")
    elif not node_fields["claim_id"].get("required"):
        problems.append("SCHEMA.json node_fields.claim_id.required must be true")
    if "source_authority" not in (node_fields or {}):
        problems.append("SCHEMA.json node_fields.source_authority is required")
    # Refuse using verification_level as an acceptance field in the contract.
    vl = (node_fields or {}).get("verification_level", {})
    role = vl.get("role") if isinstance(vl, dict) else None
    if role and role != "evidence_metadata":
        problems.append(
            "SCHEMA.json verification_level.role must be 'evidence_metadata' "
            "(level is not acceptance)"
        )


def check_authority_map(auth: dict[str, Any], problems: list[str]) -> None:
    if auth.get("scientific_effect") != "NONE":
        problems.append("AUTHORITY_MAP.json scientific_effect must be 'NONE'")
    authorities = auth.get("authorities")
    if not isinstance(authorities, dict):
        problems.append("AUTHORITY_MAP.json authorities must be an object")
        authorities = {}
    this_pkg = auth.get("this_package")
    if not isinstance(this_pkg, dict) or this_pkg.get("id") != "scientific_state_architecture":
        problems.append("AUTHORITY_MAP.json this_package.id must be scientific_state_architecture")
    never = set(this_pkg.get("never_writes", [])) if isinstance(this_pkg, dict) else set()
    if not FORBIDDEN_OWNED.issubset(never):
        problems.append(
            "AUTHORITY_MAP.json this_package.never_writes must cover forbidden owned fields"
        )
    required = auth.get("required_authority_ids")
    if not isinstance(required, list):
        problems.append("AUTHORITY_MAP.json required_authority_ids must be a list")
        return
    known = set(authorities) | (
        {this_pkg["id"]} if isinstance(this_pkg, dict) and "id" in this_pkg else set()
    )
    for auth_id in required:
        if auth_id not in known:
            problems.append(f"AUTHORITY_MAP.json missing required authority {auth_id!r}")
    for needed in (
        "claims_firewall",
        "math_downstream_gate",
        "human_d0_crosswalk",
        "math_status_packet",
    ):
        if needed not in authorities:
            problems.append(f"AUTHORITY_MAP.json authorities missing {needed!r}")


def check_verification_levels(levels_doc: dict[str, Any], problems: list[str]) -> None:
    if levels_doc.get("scientific_effect") != "NONE":
        problems.append("VERIFICATION_LEVELS.json scientific_effect must be 'NONE'")
    levels = levels_doc.get("levels")
    if not isinstance(levels, dict):
        problems.append("VERIFICATION_LEVELS.json levels must be an object")
        return
    for level_id in REQUIRED_LEVELS:
        if level_id not in levels:
            problems.append(f"VERIFICATION_LEVELS.json missing {level_id}")
    rule = str(levels_doc.get("rule", "")).lower()
    if "acceptance" not in rule and "controlling" not in rule:
        problems.append(
            "VERIFICATION_LEVELS.json rule must state that level is not acceptance/CONTROLLING"
        )


def check_crosswalk(
    crosswalk: dict[str, Any],
    auth: dict[str, Any],
    claims: dict[str, Any],
    problems: list[str],
) -> None:
    if crosswalk.get("scientific_effect") != "NONE":
        problems.append("ID_CROSSWALK.json scientific_effect must be 'NONE'")
    rows = crosswalk.get("rows")
    if not isinstance(rows, list) or not rows:
        problems.append("ID_CROSSWALK.json rows must be a non-empty list")
        return
    authorities = set((auth.get("authorities") or {})) | {"scientific_state_architecture"}
    this_pkg = auth.get("this_package")
    if isinstance(this_pkg, dict) and "id" in this_pkg:
        authorities.add(this_pkg["id"])
    premises = set((claims.get("premises") or {}))
    claim_ids = set((claims.get("claims") or {}))
    seen_row_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"ID_CROSSWALK.json rows[{index}] must be an object")
            continue
        # Smuggled status payloads
        for banned in FORBIDDEN_OWNED:
            if banned in row:
                problems.append(
                    f"ID_CROSSWALK.json rows[{index}] smuggles owned field {banned!r}"
                )
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            problems.append(f"ID_CROSSWALK.json rows[{index}] missing row_id")
        elif row_id in seen_row_ids:
            problems.append(f"ID_CROSSWALK.json duplicate row_id {row_id!r}")
        else:
            seen_row_ids.add(row_id)
        authority = row.get("authority")
        if authority not in authorities:
            problems.append(
                f"ID_CROSSWALK.json rows[{index}] unknown authority {authority!r}"
            )
        main_id = row.get("main_claim_or_premise_id")
        bucket = row.get("main_bucket")
        math_id = row.get("math_gate_id")
        if main_id is None and math_id is None:
            problems.append(
                f"ID_CROSSWALK.json rows[{index}] needs main_claim_or_premise_id or math_gate_id"
            )
        if main_id is not None:
            if bucket == "premises":
                if main_id not in premises:
                    problems.append(
                        f"ID_CROSSWALK.json rows[{index}] unknown premise {main_id!r}"
                    )
            elif bucket == "claims":
                if main_id not in claim_ids:
                    problems.append(
                        f"ID_CROSSWALK.json rows[{index}] unknown claim {main_id!r}"
                    )
            else:
                problems.append(
                    f"ID_CROSSWALK.json rows[{index}] main_bucket must be "
                    f"'premises' or 'claims' when main id is set"
                )
        if math_id is not None and not isinstance(math_id, str):
            problems.append(f"ID_CROSSWALK.json rows[{index}] math_gate_id must be a string")


def audit(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    pkg = root / "architecture" / "scientific_state" / "v1"
    problems: list[str] = []
    for name in REQUIRED_FILES:
        if not (pkg / name).is_file():
            problems.append(f"missing {pkg.relative_to(root) / name}")
    if problems:
        return {"problems": problems, "files_checked": 0, "scientific_effect": "NONE"}

    schema = load_json(pkg / "SCHEMA.json")
    auth = load_json(pkg / "AUTHORITY_MAP.json")
    crosswalk = load_json(pkg / "ID_CROSSWALK.json")
    levels = load_json(pkg / "VERIFICATION_LEVELS.json")
    claims_path = root / "claims" / "graph.json"
    if not claims_path.is_file():
        problems.append("claims/graph.json missing (required for crosswalk integrity)")
        claims: dict[str, Any] = {}
    else:
        claims = load_json(claims_path)

    check_schema(schema, problems)
    check_authority_map(auth, problems)
    check_verification_levels(levels, problems)
    check_crosswalk(crosswalk, auth, claims, problems)

    # Package-wide forbidden key scan (crosswalk + authority this_package already covered).
    walk_forbidden(crosswalk, "ID_CROSSWALK", problems, allow_schema_meta=False)
    # SCHEMA may document forbidden names.
    walk_forbidden(schema, "", problems, allow_schema_meta=True)

    return {
        "problems": problems,
        "files_checked": len(REQUIRED_FILES),
        "crosswalk_rows": len(crosswalk.get("rows") or []),
        "scientific_effect": "NONE",
        "scope": "architecture/scientific_state pointers only; not mathematical acceptance",
    }


def main(argv: list[str] | None = None) -> int:
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
