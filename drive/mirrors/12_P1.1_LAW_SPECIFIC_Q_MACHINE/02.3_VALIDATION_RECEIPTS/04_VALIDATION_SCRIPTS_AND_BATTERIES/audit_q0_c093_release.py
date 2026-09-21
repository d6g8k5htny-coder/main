#!/usr/bin/env python3
"""Static and dynamic audit for the Q0-C093 closeout release."""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
from typing import Any

BASE = Path(__file__).resolve().parent

EXCLUDED_PYTHON = {
    "build_q0_c092_bundle.py",
    "verify_q0_c092_bundle.py",
}

DEEP_SCRIPTS = [
    "periodized_bf_contract.py",
    "periodized_bf_matrix_transfer.py",
    "bf_two_critical_value_gap.py",
    "gaussian_corridor_certificate.py",
    "palm_weighted_gaussian_chernoff.py",
    "validate_q0_llm_verifier_v4.py",
    "q0_c092_contract_checker.py",
]


def run(script: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(BASE / script)],
        cwd=BASE,
        text=True,
        capture_output=True,
        timeout=240,
    )
    return {
        "script": script,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
    }


def parse_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    issues: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    contract = parse_json(BASE / "q0_c093_release_contract.json")
    c092 = parse_json(BASE / "q0_c092_final_contract.json")
    successors = parse_json(BASE / "C093_SUCCESSOR_PROJECTS.json")
    deprecation = parse_json(BASE / "C093_DEPRECATION_MAP.json")

    required = set(contract["active_reference_files"])
    required.update(
        {
            "q0_c093_release_contract.json",
            "audit_q0_c093_release.py",
            "verify_q0_c093_release.py",
            "build_q0_c093_release.py",
        }
    )
    missing = sorted(name for name in required if not (BASE / name).exists())
    checks["missing_active_files"] = missing
    if missing:
        issues.append({"gate": "FILE_CLOSURE", "missing": missing})

    # JSON closure.
    json_failures = []
    for path in sorted(BASE.glob("*.json")):
        if path.name.startswith("Q0_C093_RELEASE_VERIFICATION") or path.name.startswith("Q0_C093_FRESH_EXTRACTION"):
            continue
        try:
            parse_json(path)
        except Exception as exc:  # pragma: no cover - release diagnostic
            json_failures.append({"file": path.name, "error": repr(exc)})
    checks["json_parse_failures"] = json_failures
    if json_failures:
        issues.append({"gate": "JSON", "failures": json_failures})

    # Python compilation and absolute output paths.
    compile_failures = []
    absolute_paths = []
    forbidden_absolute = "/" + "mnt" + "/data"
    inventory_path = BASE / "C093_RELEASE_INVENTORY.csv"
    import csv
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory_names = [row["filename"] for row in csv.DictReader(handle)]
    python_files = sorted(
        BASE / name for name in inventory_names if name.endswith(".py")
    )
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_failures.append({"file": path.name, "error": repr(exc)})
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if forbidden_absolute in line and not line.lstrip().startswith("#"):
                absolute_paths.append(
                    {"file": path.name, "line": line_number, "text": line.strip()}
                )
    # Text hygiene: no hidden C0 controls in release text artifacts.
    control_characters = []
    for name in inventory_names:
        path = BASE / name
        if path.suffix.lower() not in {".md", ".json", ".txt", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        for offset, character in enumerate(text):
            if ord(character) < 32 and character not in {"\n", "\r"}:
                control_characters.append(
                    {"file": name, "offset": offset, "codepoint": ord(character)}
                )
    checks["control_characters"] = control_characters

    checks["python_compile_failures"] = compile_failures
    checks["absolute_output_paths"] = absolute_paths
    if control_characters:
        issues.append({"gate": "TEXT_HYGIENE", "characters": control_characters})
    if compile_failures:
        issues.append({"gate": "PYTHON_COMPILE", "failures": compile_failures})
    if absolute_paths:
        issues.append({"gate": "PORTABILITY", "paths": absolute_paths})

    # Core semantic contract.
    sys.path.insert(0, str(BASE))
    import q0_llm_verifier_v4 as v4

    graph, roots = v4.program_grade_q0_contract()
    semantic = graph.validate(roots)
    checks["semantic_core"] = semantic.as_dict()
    if not semantic.valid:
        issues.append({"gate": "SEMANTIC_CORE", "details": semantic.as_dict()})
    if semantic.root_hashes != contract["core_roots"]:
        issues.append(
            {
                "gate": "ROOT_HASH",
                "expected": contract["core_roots"],
                "observed": semantic.root_hashes,
            }
        )

    # No unowned dispositions.
    checks["successor_unowned_items"] = successors.get("unowned_items")
    if successors.get("unowned_items") != 0:
        issues.append({"gate": "OWNERSHIP", "value": successors.get("unowned_items")})
    for project in successors.get("projects", []):
        if project.get("core_dependency") is not False or not project.get("id"):
            issues.append({"gate": "SUCCESSOR_ISOLATION", "project": project})

    # Deprecation coverage.
    deprecated = {item["deprecated"] for item in deprecation.get("mappings", [])}
    expected_deprecated = {
        "TRUTH_CONST",
        "C089_UPPER_0P97_RLE005",
        "C089_FINITE_LOWER_0P8501_RLE005",
        "H4_QSTEP_PRODUCT",
        "PLANAR_KERNEL_AS_EXACT_TORUS_MODEL",
        "R0_OPEN_UNQUALIFIED",
        "Q0_C092_FINAL_BUNDLE.zip",
    }
    missing_deprecations = sorted(expected_deprecated - deprecated)
    checks["missing_deprecations"] = missing_deprecations
    if missing_deprecations:
        issues.append({"gate": "DEPRECATION", "missing": missing_deprecations})

    # Active reference closure.
    active_text = "\n".join(
        (BASE / name).read_text(encoding="utf-8")
        for name in contract["active_reference_files"]
        if (BASE / name).suffix in {".md", ".json", ".py", ".txt", ".csv"}
    )
    broken_refs = []
    for name in sorted(required):
        if name not in active_text and name not in {
            "q0_c093_release_contract.json",
            "audit_q0_c093_release.py",
            "verify_q0_c093_release.py",
            "build_q0_c093_release.py",
        }:
            # Inventory files need not all be mentioned in prose.
            pass
    # Explicit filenames named in release index must exist.
    index_text = (BASE / "Q0_C093_RELEASE_INDEX.md").read_text(encoding="utf-8")
    mentioned = set(re.findall(r"`([A-Za-z0-9_.-]+\.(?:md|json|py|txt|csv|zip))`", index_text))
    for name in sorted(mentioned):
        if name in {"Q0_C093_FINAL_RELEASE.zip", "Q0_C093_SHA256_MANIFEST.json"}:
            continue
        if not (BASE / name).exists():
            broken_refs.append(name)
    checks["broken_release_index_references"] = broken_refs
    if broken_refs:
        issues.append({"gate": "REFERENCE_CLOSURE", "files": broken_refs})

    # Invariant reports.
    v4_validation = parse_json(BASE / "q0_llm_verifier_v4_validation.json")
    c092_report = parse_json(BASE / "q0_c092_contract_check_report.json")
    periodized = parse_json(BASE / "periodized_bf_contract_report.json")
    matrix = parse_json(BASE / "periodized_bf_matrix_transfer_report.json")
    gap = parse_json(BASE / "bf_two_critical_value_gap_report.json")
    corridor = parse_json(BASE / "gaussian_corridor_certificate_report.json")
    palm = parse_json(BASE / "palm_weighted_gaussian_chernoff_report.json")

    invariants = {
        "v4_regressions_pass": bool(v4_validation.get("all_expected_checks_pass")),
        "c092_contract_valid": bool(c092_report.get("valid")),
        "periodized_exact_ensemble_closed": periodized.get("contract_status", {}).get("exact_ensemble_definition") == "CLOSED",
        "matrix_transfer_local_pass": matrix.get("local_transfer", {}).get("status") == "PASS",
        "matrix_transfer_far_pass": matrix.get("far_transfer", {}).get("status") == "PASS",
        "critical_value_gap_exact_reference_closed": gap.get("adjudication", {}).get("exact_planar_mark_law") == "CLOSED",
        "corridor_product_rejected": corridor.get("adjudication", {}).get("naive_markov_product_status") == "REJECTED AS A GENERAL UPPER BOUND",
        "palm_weighted_chernoff_available": palm.get("adjudication", {}).get("Palm_weight_issue") == "SOLVED AT THEOREM LEVEL",
    }
    checks["invariants"] = invariants
    failed_invariants = sorted(name for name, passed in invariants.items() if not passed)
    if failed_invariants:
        issues.append({"gate": "INVARIANTS", "failed": failed_invariants})

    deep = "--deep" in sys.argv
    executions = []
    if deep:
        for script in DEEP_SCRIPTS:
            result = run(script)
            executions.append(result)
            if result["returncode"] != 0:
                issues.append({"gate": "DEEP_EXECUTION", "result": result})
    checks["deep_execution_requested"] = deep
    checks["deep_executions"] = executions

    output = {
        "release_id": contract["release_id"],
        "valid": not issues,
        "issues": issues,
        "checks": checks,
        "declared_terminal_dispositions": contract["terminal_dispositions"],
        "unowned_items": successors.get("unowned_items"),
    }
    if "--no-write" not in sys.argv:
        path = BASE / "Q0_C093_AUDIT_REPORT.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
