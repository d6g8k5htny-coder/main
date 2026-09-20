#!/usr/bin/env python3
"""Build a noncanonical q0 composite migration candidate.

The output preserves the law-specific successor unchanged at its core while
carrying forward every legacy consolidated-machine payload needed by the
active verifier. Missing legacy roots are mapped to explicit nonterminal
provenance claims, so no historical result is silently narrowed or promoted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ACTIVE_PATH = Path("q0_machine_active.json")
SUCCESSOR_PATH = Path("q0_machine_successor_1_2.json")
OUTPUT_PATH = Path("q0_machine_composite_candidate.json")

ACTIVE_DRIVE_ID = "1PXa-ZqCrICicUUDIy37PafHgdjCOb3cj"
ACTIVE_VERIFIER_DRIVE_ID = "1FxdPkPmK-WhTm-9hQwyJsuJiscCJoy-7"
SUCCESSOR_DRIVE_ID = "1KWNv1Yz95o42tMocUA_G7qocQrioS52o"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strictify(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[str]]:
    """Replace the active file's non-standard nonfinite JSON constants by null."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        changed: list[str] = []
        for key, item in value.items():
            out[key], item_changed = strictify(item, path + (str(key),))
            changed.extend(item_changed)
        return out, changed
    if isinstance(value, list):
        out_list: list[Any] = []
        changed = []
        for index, item in enumerate(value):
            new_item, item_changed = strictify(item, path + (str(index),))
            out_list.append(new_item)
            changed.extend(item_changed)
        return out_list, changed
    if isinstance(value, float) and not math.isfinite(value):
        return None, ["/".join(path)]
    return value, []


def archival_claim(root_id: str, root_hash: str, disposition: str) -> dict[str, Any]:
    return {
        "bridge_ids": [],
        "claim_grade": "LEGACY-PROVENANCE-NONTERMINAL",
        "dependency_ids": [],
        "dependency_routes": [],
        "falsifiers": [
            "legacy root hash does not reproduce from the exact active machine",
            "a migration target is treated as mathematical proof or status promotion",
        ],
        "law_object": "LEGACY_Q0_ROOT_RECORD",
        "measure_tag": "LEGACY_CONSOLIDATED_ARCHIVE",
        "required_marks": [],
        "source_files": [
            f"q0_machine.json Drive ID {ACTIVE_DRIVE_ID}",
            f"legacy root hash {root_hash}",
        ],
        "statement": (
            f"Preserve legacy root {root_id} at hash {root_hash}; "
            f"migration disposition: {disposition}."
        ),
        "status": "SUPERSEDED",
        "verification_state": "PRESERVED-NOT-PROMOTED",
    }


def main() -> None:
    active_raw = ACTIVE_PATH.read_bytes()
    successor_raw = SUCCESSOR_PATH.read_bytes()
    active = json.loads(active_raw)
    successor = json.loads(successor_raw)
    active_strict, normalized_paths = strictify(active)

    active_roots = {
        **active["live_roots"],
        **{
            key: value
            for key, value in active["historical_roots"].items()
            if key != "note"
        },
    }
    existing_map = successor["migration_map"]
    preexisting_mapping_count = len(existing_map)
    missing_roots = sorted(set(active_roots) - set(existing_map))

    successor["objects"]["LEGACY_Q0_ROOT_RECORD"] = {
        "definition": (
            "Content-addressed provenance record for a root in the active "
            "q0-consolidated/1.0 archive; it carries no theorem authority."
        ),
        "measure_tag": "LEGACY_CONSOLIDATED_ARCHIVE",
        "object_type": "provenance_record",
        "required_marks": [],
    }

    for root_id in missing_roots:
        claim_id = f"LEGACY_ROOT_{root_id}"
        root_hash = active_roots[root_id]
        disposition = (
            "retained as an unresolved or governance-only historical root; "
            "no law-specific theorem target is asserted"
        )
        successor["claims"][claim_id] = archival_claim(
            root_id, root_hash, disposition
        )
        successor["migration_map"][root_id] = [claim_id]

    # Preserve the legacy payloads consumed by q0_verify.py. Extra fields are
    # ignored by the law-specific verifier but keep old extraction/selftests
    # operational.
    for key in (
        "consolidation",
        "live_roots",
        "historical_roots",
        "json_artifacts",
        "csv_artifacts",
        "png_artifacts_base64",
        "text_artifacts",
        "manifest",
        "fixtures",
    ):
        successor[key] = active_strict[key]

    successor["legacy_provenance"] = {
        "active_machine": {
            "drive_id": ACTIVE_DRIVE_ID,
            "bytes": len(active_raw),
            "sha256": hashlib.sha256(active_raw).hexdigest(),
            "schema": active["schema"],
        },
        "active_verifier": {
            "drive_id": ACTIVE_VERIFIER_DRIVE_ID,
            "bytes": Path("q0_verify_active.py").stat().st_size,
            "sha256": sha256(Path("q0_verify_active.py")),
        },
        "law_specific_predecessor": {
            "drive_id": SUCCESSOR_DRIVE_ID,
            "bytes": len(successor_raw),
            "sha256": hashlib.sha256(successor_raw).hexdigest(),
            "schema": "q0-law-specific/1.2",
        },
        "strict_json_normalization": {
            "count": len(normalized_paths),
            "paths": normalized_paths,
            "replacement": "null",
            "reason": "RFC-8259-compatible JSON; original byte identity retained above",
        },
        "inventory": {
            "live_roots": len(active["live_roots"]),
            "historical_named_roots": len(active_roots) - len(active["live_roots"]),
            "json_artifacts": len(active["json_artifacts"]),
            "csv_artifacts": len(active["csv_artifacts"]),
            "png_artifacts_base64": len(active["png_artifacts_base64"]),
            "text_artifacts": len(active["text_artifacts"]),
            "manifest_entries": len(active["manifest"]),
            "fixtures": len(active["fixtures"]),
        },
        "migration_coverage": {
            "named_roots_total": len(active_roots),
            "named_roots_mapped": len(successor["migration_map"]),
            "preexisting_mappings": preexisting_mapping_count,
            "added_fail_closed_mappings": len(missing_roots),
            "missing_after_build": sorted(
                set(active_roots) - set(successor["migration_map"])
            ),
        },
    }
    successor["status"] = (
        "NONCANONICAL-COMPOSITE-MIGRATION-CANDIDATE-NO-INSTALL"
    )
    successor["transition"] = "Q0-COMPOSITE-MIGRATION-20260730"
    successor["notes"].extend(
        [
            "The complete legacy consolidated payload is preserved so the active verifier's extraction and selftest surfaces remain available.",
            "Every named active or historical root has an explicit migration entry; unmatched roots map only to nonterminal provenance claims.",
            "The one non-standard NaN in the active archive is represented as null in the strict-JSON composite. The exact original active hash is retained.",
            "This composite does not synchronize later theorem verdicts, install a verifier, or replace either active file.",
        ]
    )

    OUTPUT_PATH.write_text(
        json.dumps(successor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "bytes": OUTPUT_PATH.stat().st_size,
                "sha256": sha256(OUTPUT_PATH),
                "active_sha256": hashlib.sha256(active_raw).hexdigest(),
                "successor_sha256": hashlib.sha256(successor_raw).hexdigest(),
                "normalized_paths": normalized_paths,
                "root_coverage": successor["legacy_provenance"][
                    "migration_coverage"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
