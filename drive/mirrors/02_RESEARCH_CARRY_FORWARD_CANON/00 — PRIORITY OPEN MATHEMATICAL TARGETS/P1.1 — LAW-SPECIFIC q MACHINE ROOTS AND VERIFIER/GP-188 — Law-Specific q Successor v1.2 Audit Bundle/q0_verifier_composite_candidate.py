#!/usr/bin/env python3
"""
Noncanonical fail-closed composite verifier candidate
for machine schema q0-law-specific/1.2.

Additive, noncanonical successor to GP-DATA-188-v1.0.
It does not modify the active q0_machine.json, active q0_verify.py,
GP-REG-188-v1.0, or GP-DATA-188-v1.0.

Preserved 1.2.1 hardening:
1. reject duplicate JSON keys before semantic parsing;
2. reject alternate routes with empty dependency lists;
3. classify every dependency cycle as a structural verification error.

Additional composite hardening:
4. reject NaN and Infinity at the JSON byte boundary;
5. return a deterministic invalid report for an incompatible schema or shape
   instead of entering schema-specific mutation tests and crashing;
6. bind execution to the exact composite migration candidate that preserves
   every active legacy payload and maps all thirteen named legacy roots.

All predecessor invariants and twelve predecessor negative tests are preserved,
and the nine CL-AUD-218 adversarial mutations remain a separate battery.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import platform
from pathlib import Path
from typing import Any, Iterable

MACHINE_SCHEMA = "q0-law-specific/1.2"
VERIFIER_RELEASE = "q0-law-specific-verifier/composite-candidate-20260730"
PREDECESSOR_MACHINE_SHA256 = "3e66e41b316244d291a0b660c610f61f43d3f3a6c4f55e452a413bb27932a427"
PREDECESSOR_VERIFIER_SHA256 = "10ac7b82923c79d491923a9c4a840797004056326b02cd48aa2e40c51ed4d2a3"
LAW_SPECIFIC_PREDECESSOR_MACHINE_SHA256 = "65af52096aa40a4ab49f0a7b0fca6d29876aaafa153139da8de351ab3b632219"
ACTIVE_CONSOLIDATED_MACHINE_SHA256 = "c3a93bd250c49e256467939bf714c18488fa506e87ae10636608fae99fb9cd38"

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


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains a repeated key."""


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateJSONKeyError(f"DUPLICATE_JSON_KEY:{key}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> dict[str, Any]:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"NONSTANDARD_JSON_CONSTANT:{value}")

    obj = json.loads(
        text,
        object_pairs_hook=_no_duplicate_pairs,
        parse_constant=reject_nonstandard_constant,
    )
    if not isinstance(obj, dict):
        raise ValueError("TOP_LEVEL_JSON_NOT_OBJECT")
    return obj


def load(path: str | Path) -> dict[str, Any]:
    # UTF-8 without BOM is intentional. A BOM or malformed byte sequence fails closed.
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def is_q_claim(claim: dict[str, Any]) -> bool:
    return claim.get("law_object", "").startswith("Q_") or "Q_" in claim.get("statement", "")


def _claim_neighbors(cid: str, claims: dict[str, Any]) -> list[str]:
    """All declared claim-to-claim edges, including every alternate route."""
    claim = claims[cid]
    neighbors: list[str] = []
    for dep in claim.get("dependency_ids", []):
        if dep in claims:
            neighbors.append(dep)
    for route in claim.get("dependency_routes", []):
        for dep in route.get("dependencies", []):
            if dep in claims:
                neighbors.append(dep)
    return neighbors


def _canonical_cycle(nodes: list[str]) -> tuple[str, ...]:
    """
    Canonicalize a cycle represented without a repeated final node.
    Direction is preserved because dependency edges are directed.
    """
    if not nodes:
        return tuple()
    rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
    return min(rotations)


def dependency_cycles(claims: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return unique directed dependency cycles across base and route edges."""
    state: dict[str, int] = {}
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    found: set[tuple[str, ...]] = set()

    def visit(cid: str) -> None:
        state[cid] = 1
        stack_index[cid] = len(stack)
        stack.append(cid)
        for dep in _claim_neighbors(cid, claims):
            dep_state = state.get(dep, 0)
            if dep_state == 0:
                visit(dep)
            elif dep_state == 1:
                start = stack_index[dep]
                cycle = stack[start:].copy()
                found.add(_canonical_cycle(cycle))
        stack.pop()
        stack_index.pop(cid, None)
        state[cid] = 2

    for cid in sorted(claims):
        if state.get(cid, 0) == 0:
            visit(cid)
    return sorted(found)


def verify(machine: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    objects = machine.get("objects", {})
    claims = machine.get("claims", {})
    roots = machine.get("proposed_roots", [])
    migration = machine.get("migration_map", {})

    if machine.get("schema") != MACHINE_SCHEMA:
        errors.append("SCHEMA")
    if not machine.get("active_machine_untouched"):
        errors.append("ACTIVE_MACHINE_OVERWRITE")
    if not isinstance(claims, dict) or not claims:
        errors.append("CLAIMS_EMPTY")
    if not isinstance(objects, dict) or not objects:
        errors.append("OBJECTS_EMPTY")
    if not isinstance(roots, list):
        errors.append("ROOTS_NOT_LIST")
        roots = []
    if not isinstance(migration, dict):
        errors.append("MIGRATION_NOT_OBJECT")
        migration = {}

    if isinstance(claims, dict):
        for cid, claim in claims.items():
            if not isinstance(claim, dict):
                errors.append(f"{cid}:CLAIM_NOT_OBJECT")
                continue
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

            deps = claim.get("dependency_ids", [])
            if not isinstance(deps, list):
                errors.append(f"{cid}:DEPENDENCIES_NOT_LIST")
                deps = []
            for dep in deps:
                if dep in objects:
                    errors.append(f"{cid}:OBJECT_DEPENDENCY_SMUGGLING:{dep}")
                elif dep not in claims:
                    errors.append(f"{cid}:UNKNOWN_DEP:{dep}")

            routes = claim.get("dependency_routes", [])
            if not isinstance(routes, list):
                errors.append(f"{cid}:ROUTES_NOT_LIST")
                routes = []
            route_ids: set[str] = set()
            for route in routes:
                if not isinstance(route, dict):
                    errors.append(f"{cid}:ROUTE_NOT_OBJECT")
                    continue
                route_id = route.get("route_id")
                if not route_id:
                    errors.append(f"{cid}:ROUTE_WITHOUT_ID")
                elif route_id in route_ids:
                    errors.append(f"{cid}:DUPLICATE_ROUTE_ID:{route_id}")
                else:
                    route_ids.add(route_id)

                route_deps = route.get("dependencies")
                if not isinstance(route_deps, list):
                    errors.append(f"{cid}:ROUTE_DEPENDENCIES_NOT_LIST:{route_id}")
                    continue
                if not route_deps:
                    errors.append(f"{cid}:ROUTE_EMPTY_DEPENDENCIES:{route_id}")
                for dep in route_deps:
                    if dep in objects:
                        errors.append(f"{cid}:ROUTE_OBJECT_DEPENDENCY_SMUGGLING:{dep}")
                    elif dep not in claims:
                        errors.append(f"{cid}:UNKNOWN_ROUTE_DEP:{dep}")

    # A cycle is malformed machine structure, not merely a root blocker.
    if isinstance(claims, dict):
        for cycle in dependency_cycles(claims):
            path = "->".join(cycle + (cycle[0],))
            errors.append(f"CYCLE:{path}")

    # Law/mark invariants.
    if isinstance(claims, dict):
        for cid, claim in claims.items():
            if not isinstance(claim, dict):
                continue
            if claim.get("measure_tag") == "P_ADJ":
                if "ADJ_CONDITIONING_TRANSFER" not in claim.get("bridge_ids", []):
                    errors.append(f"{cid}:MISSING_ADJ_BRIDGE")
                if "GRADIENT_ADJACENCY_MS" not in claim.get("required_marks", []):
                    errors.append(f"{cid}:MISSING_ADJ_MARK")

    # Theorem B dependency completeness.
    b0 = claims.get("THEOREM_B0_COMPACT_MARK") if isinstance(claims, dict) else None
    if not b0:
        errors.append("THEOREM_B0:MISSING_CLAIM")
    else:
        missing = sorted(B0_REQUIRED - set(b0.get("dependency_ids", [])))
        for dep in missing:
            errors.append(f"THEOREM_B0:MISSING:{dep}")

    full_b = claims.get("THEOREM_B_FULL_KAPPA_CANDIDATE") if isinstance(claims, dict) else None
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
            continue
        for target in targets:
            if target not in claims:
                errors.append(f"MIGRATION:{old}:UNKNOWN_TARGET:{target}")

    def claim_blockers(
        cid: str, seen: frozenset[str] = frozenset()
    ) -> tuple[set[str], dict[str, Any]]:
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
            if route_options:
                # An alternate route is sufficient; use the least-blocked valid route.
                _, _, best = min(route_options, key=lambda item: (item[0], item[1]))
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

    return {
        "verifier_release": VERIFIER_RELEASE,
        "machine_schema": MACHINE_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "root_reports": root_reports,
    }


def predecessor_negative_tests(base: dict[str, Any]) -> list[dict[str, Any]]:
    """The twelve GP-DATA-188 tests, unchanged in scientific intent."""
    tests: list[dict[str, Any]] = []

    def run_error(name: str, mutator, needle: str) -> None:
        m = copy.deepcopy(base)
        mutator(m)
        report = verify(m)
        passed = (not report["valid"]) and any(needle in error for error in report["errors"])
        tests.append({"name": name, "passed": passed, "errors": report["errors"]})

    def run_blocker(name: str, mutator, root: str, blocker: str) -> None:
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
    run_error(
        "object_dependency_smuggling",
        lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"].append("FULL_LIFETIME_DENSITY"),
        "OBJECT_DEPENDENCY_SMUGGLING",
    )
    run_error(
        "b0_drops_multiplicity",
        lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"].update(
            dependency_ids=[
                dep for dep in m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"]
                if dep != "MARKED_PROCESS_MULTIPLICITY"
            ]
        ),
        "THEOREM_B0:MISSING:MARKED_PROCESS_MULTIPLICITY",
    )
    run_error(
        "b0_drops_uniform_selection",
        lambda m: m["claims"]["THEOREM_B0_COMPACT_MARK"].update(
            dependency_ids=[
                dep for dep in m["claims"]["THEOREM_B0_COMPACT_MARK"]["dependency_ids"]
                if dep != "COMPACT_MARK_UNIFORM_SELECTION"
            ]
        ),
        "THEOREM_B0:MISSING:COMPACT_MARK_UNIFORM_SELECTION",
    )
    run_error(
        "full_b_drops_small_tail",
        lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
            dependency_ids=[
                dep for dep in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                if dep != "FULL_KAPPA_SMALL_TAIL"
            ]
        ),
        "THEOREM_B_FULL:MISSING:FULL_KAPPA_SMALL_TAIL",
    )
    run_error(
        "full_b_drops_large_tail",
        lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
            dependency_ids=[
                dep for dep in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                if dep != "FULL_KAPPA_LARGE_TAIL"
            ]
        ),
        "THEOREM_B_FULL:MISSING:FULL_KAPPA_LARGE_TAIL",
    )
    run_error(
        "full_b_drops_off_fold",
        lambda m: m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"].update(
            dependency_ids=[
                dep for dep in m["claims"]["THEOREM_B_FULL_KAPPA_CANDIDATE"]["dependency_ids"]
                if dep != "OFF_FOLD_LOWER_ORDER"
            ]
        ),
        "THEOREM_B_FULL:MISSING:OFF_FOLD_LOWER_ORDER",
    )
    run_error(
        "historical_migration_to_partial_factor",
        lambda m: m["migration_map"].update(
            THEOREM_B_FULL_KAPPA_C104=["THEOREM_B_ADJ_CONTACT_FACTOR"]
        ),
        "MIGRATION:THEOREM_B_INCOMPLETE",
    )
    run_blocker(
        "archive_claimed_is_not_terminal",
        lambda m: m["claims"]["COMPACT_MARK_CONTACT_INTENSITY"].update(status="ARCHIVE-CLAIMED"),
        "THEOREM_B0_COMPACT_MARK",
        "COMPACT_MARK_CONTACT_INTENSITY",
    )
    run_blocker(
        "same_line_exact_is_not_terminal",
        lambda m: None,
        "EXACT_GAP_JACOBIAN",
        "EXACT_GAP_JACOBIAN",
    )
    return tests


def cl_adversarial_tests(base: dict[str, Any]) -> list[dict[str, Any]]:
    """The nine undisclosed CL-AUD-218 mutations, now frozen as regressions."""
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, **details: Any) -> None:
        tests.append({"name": name, "passed": bool(passed), **details})

    # 1. Lowercase terminal status is not terminal or allowed.
    m = copy.deepcopy(base)
    m["claims"]["EXACT_GAP_JACOBIAN"]["status"] = "independently-terminal"
    report = verify(m)
    add(
        "cl_lowercase_terminal_status_rejected",
        (not report["valid"]) and any("UNKNOWN_STATUS:independently-terminal" in e for e in report["errors"]),
        errors=report["errors"],
    )

    # 2. Whitespace-padded terminal status is not terminal or allowed.
    m = copy.deepcopy(base)
    m["claims"]["EXACT_GAP_JACOBIAN"]["status"] = " INDEPENDENTLY-TERMINAL "
    report = verify(m)
    add(
        "cl_whitespace_terminal_status_rejected",
        (not report["valid"]) and any("UNKNOWN_STATUS: INDEPENDENTLY-TERMINAL " in e for e in report["errors"]),
        errors=report["errors"],
    )

    # 3. A terminal root with open dependencies remains nonpromotable.
    m = copy.deepcopy(base)
    m["claims"]["Q_TYPED_MS_RATE"]["status"] = "INDEPENDENTLY-TERMINAL"
    report = verify(m)
    rr = report["root_reports"]["Q_TYPED_MS_RATE"]
    add(
        "cl_terminal_root_with_open_dependencies_nonpromotable",
        report["valid"] and (not rr["unconditional_promotable"]) and "MORSE_SMALE_R0" in rr["blockers"],
        blockers=rr["blockers"],
        errors=report["errors"],
    )

    # 4. A self-cycle makes the machine structurally invalid.
    m = copy.deepcopy(base)
    m["claims"]["EXACT_GAP_JACOBIAN"]["dependency_ids"].append("EXACT_GAP_JACOBIAN")
    report = verify(m)
    add(
        "cl_self_dependency_cycle_structurally_invalid",
        (not report["valid"]) and any(
            error.startswith("CYCLE:EXACT_GAP_JACOBIAN->EXACT_GAP_JACOBIAN")
            for error in report["errors"]
        ),
        errors=report["errors"],
    )

    # 5. Unknown route dependency is rejected.
    m = copy.deepcopy(base)
    m["claims"]["Q_TYPED_MS_RATE"]["dependency_routes"][0]["dependencies"].append("UNKNOWN_CL_ROUTE_DEP")
    report = verify(m)
    add(
        "cl_unknown_route_dependency_rejected",
        (not report["valid"]) and any("UNKNOWN_ROUTE_DEP:UNKNOWN_CL_ROUTE_DEP" in e for e in report["errors"]),
        errors=report["errors"],
    )

    # 6. An object cannot be proposed as a root.
    m = copy.deepcopy(base)
    m["proposed_roots"].append("FULL_LIFETIME_DENSITY")
    report = verify(m)
    add(
        "cl_object_as_root_rejected",
        (not report["valid"]) and "UNKNOWN_ROOT:FULL_LIFETIME_DENSITY" in report["errors"],
        errors=report["errors"],
    )

    # 7. Empty migration target is rejected.
    m = copy.deepcopy(base)
    m["migration_map"]["Q0_LIMIT_C101"] = []
    report = verify(m)
    add(
        "cl_empty_migration_target_rejected",
        (not report["valid"]) and "MIGRATION:Q0_LIMIT_C101:EMPTY" in report["errors"],
        errors=report["errors"],
    )

    # 8. Empty alternate route is rejected before route minimization.
    m = copy.deepcopy(base)
    m["claims"]["Q_TYPED_MS_RATE"]["dependency_routes"].append(
        {"route_id": "AAA_EMPTY", "dependencies": []}
    )
    report = verify(m)
    add(
        "cl_empty_alternate_route_rejected",
        (not report["valid"]) and any("ROUTE_EMPTY_DEPENDENCIES:AAA_EMPTY" in e for e in report["errors"]),
        errors=report["errors"],
    )

    # 9. Duplicate keys are rejected at the byte boundary, before verify().
    duplicate_text = (
        '{"claims":{"MORSE_SMALE_R0":{"status":"OPEN"},'
        '"MORSE_SMALE_R0":{"status":"INDEPENDENTLY-TERMINAL"}}}'
    )
    try:
        strict_json_loads(duplicate_text)
    except DuplicateJSONKeyError as exc:
        add(
            "cl_duplicate_json_key_rejected_before_verification",
            str(exc) == "DUPLICATE_JSON_KEY:MORSE_SMALE_R0",
            exception=str(exc),
        )
    else:
        add("cl_duplicate_json_key_rejected_before_verification", False)

    return tests


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "q0_machine_composite_candidate.json"
    output = (
        Path(argv[2])
        if len(argv) > 2
        else Path("q0_composite_candidate_verification_report.json")
    )

    machine_path = Path(path)
    machine_bytes = machine_path.read_bytes()
    try:
        machine = load(machine_path)
    except (DuplicateJSONKeyError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        failure_report = {
            "verifier_release": VERIFIER_RELEASE,
            "machine_schema": MACHINE_SCHEMA,
            "valid": False,
            "errors": [f"INPUT_REJECTED:{exc}"],
            "root_reports": {},
            "successor_verifier": {
                "path": Path(__file__).name,
                "bytes": Path(__file__).stat().st_size,
                "sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
                "release": VERIFIER_RELEASE,
            },
            "execution": {
                "python": sys.version,
                "platform": platform.platform(),
                "argv": [str(item) for item in argv],
                "output": str(output),
            },
            "predecessor_machine": {
                "path": machine_path.name,
                "bytes": len(machine_bytes),
                "sha256_expected": PREDECESSOR_MACHINE_SHA256,
                "sha256_actual": __import__("hashlib").sha256(machine_bytes).hexdigest(),
                "identity_pass": False,
            },
            "cycle_policy": "ANY_BASE_OR_ROUTE_DEPENDENCY_CYCLE_IS_STRUCTURALLY_INVALID",
            "duplicate_key_policy": "REJECT_BEFORE_SEMANTIC_VERIFICATION",
            "empty_route_policy": "REJECT",
            "predecessor_negative_tests": [],
            "predecessor_negative_tests_passed": 0,
            "predecessor_negative_tests_total": 0,
            "cl_adversarial_tests": [],
            "cl_adversarial_tests_passed": 0,
            "cl_adversarial_tests_total": 0,
            "all_roots_fail_closed": True,
            "active_machine_modified": False,
            "active_verifier_modified": False,
            "noncanonical_successor_only": True,
        }
        output.write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure_report, indent=2, sort_keys=True))
        return 1

    report = verify(machine)
    # Mutation batteries assume the declared law-specific schema and complete
    # shape. A malformed or incompatible object is already rejected by
    # verify(); do not enter schema-specific mutations on it.
    if report["valid"]:
        predecessor_tests = predecessor_negative_tests(machine)
        cl_tests = cl_adversarial_tests(machine)
    else:
        predecessor_tests = []
        cl_tests = []

    successor_path = Path(__file__)
    successor_bytes = successor_path.read_bytes()
    successor_sha256 = __import__("hashlib").sha256(successor_bytes).hexdigest()

    report.update({
        "successor_verifier": {
            "path": successor_path.name,
            "bytes": len(successor_bytes),
            "sha256": successor_sha256,
            "release": VERIFIER_RELEASE,
        },
        "execution": {
            "python": sys.version,
            "platform": platform.platform(),
            "argv": [str(item) for item in argv],
            "output": str(output),
        },
        "predecessor_machine": {
            "path": machine_path.name,
            "bytes": len(machine_bytes),
            "sha256_expected": PREDECESSOR_MACHINE_SHA256,
            "sha256_actual": __import__("hashlib").sha256(machine_bytes).hexdigest(),
            "identity_pass": __import__("hashlib").sha256(machine_bytes).hexdigest()
                             == PREDECESSOR_MACHINE_SHA256,
        },
        "predecessor_verifier_sha256": PREDECESSOR_VERIFIER_SHA256,
        "law_specific_predecessor_machine_sha256": LAW_SPECIFIC_PREDECESSOR_MACHINE_SHA256,
        "active_consolidated_machine_sha256": ACTIVE_CONSOLIDATED_MACHINE_SHA256,
        "cycle_policy": "ANY_BASE_OR_ROUTE_DEPENDENCY_CYCLE_IS_STRUCTURALLY_INVALID",
        "duplicate_key_policy": "REJECT_BEFORE_SEMANTIC_VERIFICATION",
        "empty_route_policy": "REJECT",
        "predecessor_negative_tests": predecessor_tests,
        "predecessor_negative_tests_passed": sum(t["passed"] for t in predecessor_tests),
        "predecessor_negative_tests_total": len(predecessor_tests),
        "cl_adversarial_tests": cl_tests,
        "cl_adversarial_tests_passed": sum(t["passed"] for t in cl_tests),
        "cl_adversarial_tests_total": len(cl_tests),
    })
    report["all_roots_fail_closed"] = all(
        not root_report["unconditional_promotable"]
        for root_report in report["root_reports"].values()
    )
    report["active_machine_modified"] = False
    report["active_verifier_modified"] = False
    report["noncanonical_successor_only"] = True

    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    ok = (
        report["valid"]
        and report["predecessor_machine"]["identity_pass"]
        and report["predecessor_negative_tests_passed"]
            == report["predecessor_negative_tests_total"] == 12
        and report["cl_adversarial_tests_passed"]
            == report["cl_adversarial_tests_total"] == 9
        and report["all_roots_fail_closed"]
        and not report["active_machine_modified"]
        and not report["active_verifier_modified"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
