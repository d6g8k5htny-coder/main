#!/usr/bin/env python3
"""
GP-DATA-188-v1.0 — fail-closed verifier for q0-law-specific/1.2.
Noncanonical successor verifier; does not modify active q0_verify.py.
"""
from __future__ import annotations
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "q0-law-specific/1.2"
TERMINAL_STATUSES = {"INDEPENDENTLY-TERMINAL", "OPERATOR-TERMINAL"}
ALLOWED_STATUSES = TERMINAL_STATUSES | {
    "OPEN", "CANDIDATE", "CONDITIONAL-PROVED", "PROVED-MODULO",
    "ARCHIVE-CLAIMED", "SAME-LINE-PROVED-PENDING-INDEPENDENT",
    "SAME-LINE-EXACT-PROVED", "BLOCKED", "RETRACTED", "SUPERSEDED",
    "KILLED", "RETIRED"
}
REQUIRED_FIELDS = {
    "law_object", "measure_tag", "status", "claim_grade", "statement",
    "dependency_ids", "dependency_routes", "bridge_ids", "required_marks",
    "source_files", "falsifiers", "verification_state"
}
B0_REQUIRED = {
    "EXACT_GAP_JACOBIAN",
    "MARKED_PROCESS_MULTIPLICITY",
    "COMPACT_MARK_CONTACT_INTENSITY",
    "COMPACT_MARK_UNIFORM_SELECTION",
    "COMPACT_MARK_DOMINATION",
}
FULL_B_REQUIRED = {
    "THEOREM_B0_COMPACT_MARK",
    "FULL_KAPPA_SMALL_TAIL",
    "FULL_KAPPA_LARGE_TAIL",
    "OFF_FOLD_LOWER_ORDER",
}

def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def is_q_claim(claim: dict[str, Any]) -> bool:
    return claim.get("law_object", "").startswith("Q_") or "Q_" in claim.get("statement", "")

def verify(machine: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    objects = machine.get("objects", {})
    claims = machine.get("claims", {})
    roots = machine.get("proposed_roots", [])
    migration = machine.get("migration_map", {})

    if machine.get("schema") != SCHEMA:
        errors.append("SCHEMA")
    if not machine.get("active_machine_untouched"):
        errors.append("ACTIVE_MACHINE_OVERWRITE")
    if not isinstance(claims, dict) or not claims:
        errors.append("CLAIMS_EMPTY")
    if not isinstance(objects, dict) or not objects:
        errors.append("OBJECTS_EMPTY")

    for cid, claim in claims.items():
        missing = sorted(REQUIRED_FIELDS - set(claim))
        for field in missing:
            errors.append(f"{cid}:MISSING:{field}")
        status = claim.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{cid}:UNKNOWN_STATUS:{status}")
        obj = claim.get("law_object")
        if obj not in objects:
            errors.append(f"{cid}:UNKNOWN_OBJECT:{obj}")
        if obj in {"Q", "q", "Q0", ""} and is_q_claim(claim):
            errors.append(f"{cid}:BARE_Q")
        if re.search(r"(?<![A-Za-z0-9_])q\s*\(", claim.get("statement", "")):
            errors.append(f"{cid}:BARE_Q_STATEMENT")
        for dep in claim.get("dependency_ids", []):
            if dep in objects:
                errors.append(f"{cid}:OBJECT_DEPENDENCY_SMUGGLING:{dep}")
            elif dep not in claims:
                errors.append(f"{cid}:UNKNOWN_DEP:{dep}")
        for route in claim.get("dependency_routes", []):
            if not route.get("route_id"):
                errors.append(f"{cid}:ROUTE_WITHOUT_ID")
            for dep in route.get("dependencies", []):
                if dep in objects:
                    errors.append(f"{cid}:ROUTE_OBJECT_DEPENDENCY_SMUGGLING:{dep}")
                elif dep not in claims:
                    errors.append(f"{cid}:UNKNOWN_ROUTE_DEP:{dep}")

    # Law/mark invariants.
    for cid, claim in claims.items():
        if claim.get("measure_tag") == "P_ADJ":
            if "ADJ_CONDITIONING_TRANSFER" not in claim.get("bridge_ids", []):
                errors.append(f"{cid}:MISSING_ADJ_BRIDGE")
            if "GRADIENT_ADJACENCY_MS" not in claim.get("required_marks", []):
                errors.append(f"{cid}:MISSING_ADJ_MARK")

    # Theorem B dependency completeness.
    b0 = claims.get("THEOREM_B0_COMPACT_MARK")
    if not b0:
        errors.append("THEOREM_B0:MISSING_CLAIM")
    else:
        missing = sorted(B0_REQUIRED - set(b0.get("dependency_ids", [])))
        for dep in missing:
            errors.append(f"THEOREM_B0:MISSING:{dep}")
    full_b = claims.get("THEOREM_B_FULL_KAPPA_CANDIDATE")
    if not full_b:
        errors.append("THEOREM_B_FULL:MISSING_CLAIM")
    else:
        missing = sorted(FULL_B_REQUIRED - set(full_b.get("dependency_ids", [])))
        for dep in missing:
            errors.append(f"THEOREM_B_FULL:MISSING:{dep}")

    # Historical migration cannot point directly to a partial factor.
    tb_targets = migration.get("THEOREM_B_FULL_KAPPA_C104", [])
    if tb_targets != ["THEOREM_B_FULL_KAPPA_CANDIDATE"]:
        errors.append("MIGRATION:THEOREM_B_INCOMPLETE")
    for old, targets in migration.items():
        if not isinstance(targets, list) or not targets:
            errors.append(f"MIGRATION:{old}:EMPTY")
        for target in targets:
            if target not in claims:
                errors.append(f"MIGRATION:{old}:UNKNOWN_TARGET:{target}")

    def claim_blockers(cid: str, seen: frozenset[str] = frozenset()) -> tuple[set[str], dict[str, Any]]:
        if cid in seen:
            return {f"CYCLE:{cid}"}, {}
        if cid not in claims:
            return {f"UNKNOWN:{cid}"}, {}
        seen2 = seen | {cid}
        claim = claims[cid]
        blockers: set[str] = set()
        if claim.get("status") not in TERMINAL_STATUSES:
            blockers.add(cid)

        for dep in claim.get("dependency_ids", []):
            b, _ = claim_blockers(dep, seen2)
            blockers |= b

        route_reports: dict[str, Any] = {}
        routes = claim.get("dependency_routes", [])
        if routes:
            route_options: list[tuple[int, str, set[str]]] = []
            for route in routes:
                rb: set[str] = set()
                for dep in route.get("dependencies", []):
                    b, _ = claim_blockers(dep, seen2)
                    rb |= b
                rid = route.get("route_id", "UNNAMED")
                route_reports[rid] = sorted(rb)
                route_options.append((len(rb), rid, rb))
            # An alternate route is sufficient; use the least-blocked route.
            _, _, best = min(route_options, key=lambda x: (x[0], x[1]))
            blockers |= best

        return blockers, route_reports

    root_reports: dict[str, Any] = {}
    for root in roots:
        if root not in claims:
            errors.append(f"UNKNOWN_ROOT:{root}")
            continue
        blockers, route_reports = claim_blockers(root)
        root_reports[root] = {
            "blockers": sorted(blockers),
            "route_reports": route_reports,
            "unconditional_promotable": not blockers,
        }

    return {"valid": not errors, "errors": errors, "root_reports": root_reports}

def negative_tests(base: dict[str, Any]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    def run_error(name, mutator, needle):
        m = copy.deepcopy(base)
        mutator(m)
        report = verify(m)
        passed = (not report["valid"]) and any(needle in e for e in report["errors"])
        tests.append({"name": name, "passed": passed, "errors": report["errors"]})

    def run_blocker(name, mutator, root, blocker):
        m = copy.deepcopy(base)
        mutator(m)
        report = verify(m)
        blockers = report.get("root_reports", {}).get(root, {}).get("blockers", [])
        tests.append({
            "name": name,
            "passed": report["valid"] and blocker in blockers
                      and not report["root_reports"][root]["unconditional_promotable"],
            "blockers": blockers,
            "errors": report["errors"],
        })

    run_error("schema_mismatch", lambda m: m.update(schema="q0-law-specific/1.1"), "SCHEMA")
    run_error("active_overwrite", lambda m: m.update(active_machine_untouched=False), "ACTIVE_MACHINE_OVERWRITE")
    run_error("bare_q", lambda m: m["claims"]["Q_TYPED_MS_RATE"].update(law_object="q"), "BARE_Q")
    run_error("object_dependency_smuggling",
              lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"].append("FULL_LIFETIME_DENSITY"),
              "OBJECT_DEPENDENCY_SMUGGLING")
    run_error("b0_drops_multiplicity",
              lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"].update(
                  dependency_ids=[d for d in m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"]
                                  if d != "MARKED_PROCESS_MULTIPLICITY"]),
              "THEOREM_B0:MISSING:MARKED_PROCESS_MULTIPLICITY")
    run_error("b0_drops_uniform_selection",
              lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"].update(
                  dependency_ids=[d for d in m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"]
                                  if d != "COMPACT_MARK_UNIFORM_SELECTION"]),
              "THEOREM_B0:MISSING:COMPACT_MARK_UNIFORM_SELECTION")
    run_error("full_b_drops_small_tail",
              lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
                  dependency_ids=[d for d in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                                  if d != "FULL_KAPPA_SMALL_TAIL"]),
              "THEOREM_B_FULL:MISSING:FULL_KAPPA_SMALL_TAIL")
    run_error("full_b_drops_large_tail",
              lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
                  dependency_ids=[d for d in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                                  if d != "FULL_KAPPA_LARGE_TAIL"]),
              "THEOREM_B_FULL:MISSING:FULL_KAPPA_LARGE_TAIL")
    run_error("full_b_drops_off_fold",
              lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
                  dependency_ids=[d for d in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                                  if d != "OFF_FOLD_LOWER_ORDER"]),
              "THEOREM_B_FULL:MISSING:OFF_FOLD_LOWER_ORDER")
    run_error("historical_migration_to_partial_factor",
              lambda m: m["migration_map"].update(
                  THEOREM_B_FULL_KAPPA_C104=["THEOREM_B_ADJ_CONTACT_FACTOR"]),
              "MIGRATION:THEOREM_B_INCOMPLETE")
    run_blocker("archive_claimed_is_not_terminal",
                lambda m: m["claims"]["COMPACT_MARK_CONTACT_INTENSITY"].update(status="ARCHIVE-CLAIMED"),
                "THEOREM_B0_COMPACT_MARK", "COMPACT_MARK_CONTACT_INTENSITY")
    run_blocker("same_line_exact_is_not_terminal",
                lambda m: None,
                "EXACT_GAP_JACOBIAN", "EXACT_GAP_JACOBIAN")

    return tests

def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "q0_law_specific_successor_v1_2.json"
    machine = load(path)
    report = verify(machine)
    tests = negative_tests(machine)
    report["negative_tests"] = tests
    report["negative_tests_passed"] = sum(t["passed"] for t in tests)
    report["negative_tests_total"] = len(tests)
    report["all_roots_fail_closed"] = all(
        not rr["unconditional_promotable"] for rr in report["root_reports"].values()
    )
    out = Path("q0_law_specific_verify_v1_2_report.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    ok = report["valid"] and all(t["passed"] for t in tests) and report["all_roots_fail_closed"]
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
