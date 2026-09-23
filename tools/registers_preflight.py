#!/usr/bin/env python3
"""Preview a declared register successor without activating it or editing a checkout.

Output is a diagnostic, not a governing register or mathematical verdict. The
normal import selector, all source exports, and live JSON/CSV remain untouched.
The preview checks transcription, register observations, review form, and frozen
source-map compatibility; it does not execute every mathematical CI command.

Usage: python3 tools/registers_preflight.py --source-name FILE.xlsx > /tmp/preview.json
Exit 0: covered interfaces compatible; 1: explicit blockers; 2: source/read error.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import registers_import as RI
import registers_check as RC
import reviews_check as RV
import frozen_check as FZ


def identity(path: str | Path) -> dict:
    p = Path(path)
    data = p.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def preview(source_name: str) -> dict:
    """Evaluate one provenance-declared export; diagnostics are never suppressed."""
    source = RI.resolve_declared_source(export_name=source_name)
    current = RI.resolve_current_source()
    protected = [Path(RI.SOURCE_MANIFEST), Path(current), Path(source)]
    protected += sorted(Path(RI.OUT_JSON).glob("*.json"))
    protected += sorted(Path(RI.OUT_CSV).glob("*.csv"))
    before = {str(p): identity(p) for p in protected}
    old, proposed = RI.parse(current), RI.parse(source)
    delta = RI.export_diff(old, proposed)
    with tempfile.TemporaryDirectory(prefix="register-preflight-") as tmp:
        j, c = Path(tmp) / "json", Path(tmp) / "csv"
        RI.write(proposed, str(j), str(c))
        drift = RI.check(proposed, str(j), str(c))
        structural, tab_count = RC.check(str(j))
        known = RC.load_known(RC.KNOWN_PATH)
        new_structural = [p for p in structural if p not in known]
        dead = sorted(set(known) - set(structural))
        obs_problems, bindings = RC.check_observations(
            RC.load_observations(RC.KNOWN_PATH), str(j),
            RC.INVENTORY_PATH, RC.PATH_CHANGES_PATH)
        review_problems = RV.check_all(RV.RECORDS_DIR, RV.SCHEMA_PATH,
                                       str(j / "review_queue.json"))
        review_problems += RV.check_companion_prose(RV.RECORDS_DIR)
        frozen_rows, frozen_problems = FZ.check(
            str(j / "frozen_objects.json"), FZ.DEFAULT_INVENTORY,
            FZ.DEFAULT_PAYLOADS, FZ.DEFAULT_MEMBERS)
        # Remove only machine-dependent temp-directory prefixes from diagnostics.
        # No source text, status, classification, or problem is changed.
        import os
        temp_relative = os.path.relpath(tmp, RI.ROOT)
        review_problems = [p.replace(tmp, "<stage>").replace(temp_relative, "<stage>")
                           for p in review_problems]
    after = {str(p): identity(p) for p in protected}
    if after != before:
        raise RI.SourceError("preflight input changed during inspection")
    blockers = {"transcription": drift, "register_structure": new_structural,
                "stale_known_findings": dead, "bound_observations": obs_problems,
                "review_interface": review_problems, "frozen_interface": frozen_problems}
    compatible = not any(blockers.values())
    return {
        "schema": "q0.register-preflight/v1",
        "current_source": Path(current).name,
        "proposed_source": Path(source).name,
        "current_identity": identity(current), "proposed_identity": identity(source),
        "snapshot_only": True, "tabs": tab_count, "generated_files_checked": 2 * tab_count,
        "row_counts": {t["tab"]: len(t["rows"]) for t in proposed},
        "comparison": delta,
        "bound_observations_checked": bindings,
        "frozen_status_counts": dict(sorted(Counter(r["status"] for r in frozen_rows).items())),
        "blockers": blockers, "compatible_for_covered_interfaces": compatible,
        "protected_files_checked": len(before), "protected_files_unchanged": True,
        "canonical_import_completed": False, "scientific_status_changed": False,
        "does_not_establish": "No activation, source repair, review filing, full repository CI, "
                              "mathematical correctness, or independence credit. A compatible "
                              "preview covers only the named metadata/form interfaces; R1 is a "
                              "dated snapshot, not current live Drive state.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-name", required=True,
                    help="an XLSX basename declared in registers/source/SOURCES.json")
    args = ap.parse_args(argv)
    try:
        report = preview(args.source_name)
    except (RI.SourceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "canonical_import_completed": False,
                          "scientific_status_changed": False}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["compatible_for_covered_interfaces"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
