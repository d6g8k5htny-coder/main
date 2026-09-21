#!/usr/bin/env python3
"""
Adversarial regression suite for uploaded `gate kernel v1_2.py`.

This file does not modify the uploaded kernel. It constructs proof objects
that should be rejected under the uploaded Master v1.1's own shell/core,
bridge-node, hash-drift, and dependability language, then records whether the
kernel rejects or accepts them.

Every test is frozen by name before execution. A "vulnerability reproduced"
result means the kernel accepted a structurally malformed or over-promoted
object, or failed to reject a bad registry load.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

BASE = Path(__file__).resolve().parent
KERNEL_PATH = BASE / "gate kernel v1_2.py"


def load_kernel():
    spec = importlib.util.spec_from_file_location("gate_kernel_v1_2_uploaded", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import uploaded gate kernel v1.2")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


k = load_kernel()


def claim(
    cid: str,
    *,
    grade=None,
    statement: str | None = None,
    content_hash: str | None = None,
    **kwargs,
):
    return k.Claim(
        id=cid,
        statement=statement or cid,
        grade=grade or k.Grade.DERIVED,
        content_hash=content_hash or f"{cid}#1",
        **kwargs,
    )


def run_claim_verdict(c, g):
    return k.run_claim(c, g)


def result(name: str, vulnerability_reproduced: bool, observed, expected_secure_behavior: str):
    return {
        "id": name,
        "vulnerability_reproduced": vulnerability_reproduced,
        "observed": observed,
        "expected_secure_behavior": expected_secure_behavior,
    }


def t_domain_missing_all():
    dep = claim("dep", grade=k.Grade.PROVEN)
    parent = claim(
        "parent",
        grade=k.Grade.PROVEN,
        parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
        edges=[k.Edge("dep", k.EdgeKind.SUPPORT, dep.content_hash)],
    )
    g = {"dep": dep, "parent": parent}
    r = run_claim_verdict(parent, g)
    accepted = "DOMAIN" not in r["fails"]
    return result(
        "V12-DOMAIN-EMPTY-DEPENDENCY",
        accepted,
        r,
        "A dependency with no domain must not discharge a parent quantified over r and b.",
    )


def t_domain_missing_parameter():
    dep = claim(
        "dep",
        grade=k.Grade.PROVEN,
        parameter_domain={"r": (0.0, 1.0)},
    )
    parent = claim(
        "parent",
        grade=k.Grade.PROVEN,
        parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
        edges=[k.Edge("dep", k.EdgeKind.SUPPORT, dep.content_hash)],
    )
    g = {"dep": dep, "parent": parent}
    r = run_claim_verdict(parent, g)
    accepted = "DOMAIN" not in r["fails"]
    return result(
        "V12-DOMAIN-MISSING-PARAMETER",
        accepted,
        r,
        "The dependency must explicitly cover every free parent parameter.",
    )


def t_mark_self_attestation():
    dep = claim("unmarked", grade=k.Grade.PROVEN, supplied_marks=set())
    parent = claim(
        "marked_parent",
        grade=k.Grade.PROVEN,
        required_marks={"height", "critical_type"},
        supplied_marks={"height", "critical_type"},
        edges=[k.Edge("unmarked", k.EdgeKind.SUPPORT, dep.content_hash)],
    )
    g = {dep.id: dep, parent.id: parent}
    r = run_claim_verdict(parent, g)
    accepted = "MARK" not in r["fails"]
    return result(
        "V12-MARK-PARENT-SELF-ATTESTS",
        accepted,
        r,
        "Marks must be supplied by evidence/dependencies or a typed mark-transfer node, not merely asserted on the parent.",
    )


def t_untyped_measure_bridge():
    pinned = claim(
        "pinned",
        grade=k.Grade.PROVEN,
        measure_tag="Gaussian-pinned",
    )
    unrelated = claim(
        "bridge",
        grade=k.Grade.PROVEN,
        statement="A theorem unrelated to change of measure.",
    )
    parent = claim(
        "parent",
        grade=k.Grade.PROVEN,
        measure_tag="pair-Palm",
        edges=[
            k.Edge("pinned", k.EdgeKind.SUPPORT, pinned.content_hash),
            k.Edge("bridge", k.EdgeKind.CHANGE_OF_MEASURE, unrelated.content_hash),
        ],
    )
    g = {x.id: x for x in (pinned, unrelated, parent)}
    r = run_claim_verdict(parent, g)
    accepted = "MEASURE" not in r["fails"]
    return result(
        "V12-BRIDGE-NO-SOURCE-TARGET-TYPE",
        accepted,
        r,
        "A bridge must declare and cover the exact source and target measures.",
    )


def t_untyped_model_bridge():
    dep = claim("model_a", grade=k.Grade.PROVEN, model_tag="MODEL-A")
    unrelated = claim("cert", grade=k.Grade.PROVEN, statement="Unrelated certificate.")
    parent = claim(
        "model_b_parent",
        grade=k.Grade.PROVEN,
        model_tag="MODEL-B",
        edges=[
            k.Edge("model_a", k.EdgeKind.SUPPORT, dep.content_hash),
            k.Edge(
                "cert",
                k.EdgeKind.TRANSFER_CERTIFICATE,
                unrelated.content_hash,
                scope_covers=True,
            ),
        ],
    )
    g = {x.id: x for x in (dep, unrelated, parent)}
    r = run_claim_verdict(parent, g)
    accepted = "MODEL" not in r["fails"]
    return result(
        "V12-TRANSFER-CERT-NO-MODEL-PAIR",
        accepted,
        r,
        "A transfer certificate must bind MODEL-A to MODEL-B and carry an exact scope.",
    )


def t_wrong_operation_witness():
    dep = claim("dep", grade=k.Grade.PROVEN)
    witness = claim(
        "union_witness",
        grade=k.Grade.PROVEN,
        statement="Union bound witness.",
    )
    parent = claim(
        "product_parent",
        grade=k.Grade.PROVEN,
        statement="A product of correlated probabilities.",
        is_combination=True,
        edges=[
            k.Edge("dep", k.EdgeKind.SUPPORT, dep.content_hash),
            k.Edge(
                "union_witness",
                k.EdgeKind.COMPOSITION_WITNESS,
                witness.content_hash,
            ),
        ],
    )
    g = {x.id: x for x in (dep, witness, parent)}
    r = run_claim_verdict(parent, g)
    accepted = "COMPOSITION" not in r["fails"]
    return result(
        "V12-COMPOSITION-WITNESS-WRONG-OPERATION",
        accepted,
        r,
        "The witness type must be compatible with the operation; UNION_BOUND cannot license a probability product.",
    )


def t_common_mode_boolean_attestation():
    c = claim(
        "agreement",
        grade=k.Grade.PROVEN,
        verification="agreement_based",
        error_independence_arg=True,
    )
    r = run_claim_verdict(c, {c.id: c})
    accepted = r["results"]["COMMON-MODE"][0] == k.V.PASS
    return result(
        "V12-COMMON-MODE-BOOLEAN-ATTESTATION",
        accepted,
        r,
        "Agreement promotion requires a first-class certificate with instruments, shared components, error channel, and hash.",
    )


def t_provisional_dependency_laundering():
    dep = claim(
        "provisional_dep",
        grade=k.Grade.PROVEN,
        verification="agreement_based",
        error_independence_arg=False,
        external_crosscheck=False,
    )
    parent = claim(
        "proven_parent",
        grade=k.Grade.PROVEN,
        edges=[k.Edge("provisional_dep", k.EdgeKind.SUPPORT, dep.content_hash)],
    )
    g = {dep.id: dep, parent.id: parent}
    dep_result = run_claim_verdict(dep, g)
    parent_result = run_claim_verdict(parent, g)
    accepted = (
        dep_result["provs"]
        and not parent_result["fails"]
        and parent_result["ceiling"] == k.Grade.PROVEN
    )
    return result(
        "V12-PROVISIONAL-DEPENDENCY-LAUNDERING",
        bool(accepted),
        {"dependency": dep_result, "parent": parent_result},
        "A provisional dependency must cap or block every non-conditional parent until the certificate is discharged.",
    )


def t_arbitrary_hash_not_content_bound():
    dep = claim(
        "dep",
        grade=k.Grade.PROVEN,
        statement="Original statement.",
        content_hash="fixed#hash",
    )
    parent = claim(
        "parent",
        grade=k.Grade.PROVEN,
        edges=[k.Edge("dep", k.EdgeKind.SUPPORT, "fixed#hash")],
    )
    g = {dep.id: dep, parent.id: parent}
    before = run_claim_verdict(parent, g)
    dep.statement = "Materially changed statement without changing the declared hash."
    after = run_claim_verdict(parent, g)
    accepted = before["drift"][0] == k.V.PASS and after["drift"][0] == k.V.PASS
    return result(
        "V12-HASH-NOT-CONTENT-BOUND",
        accepted,
        {"before": before["drift"], "after": after["drift"]},
        "The kernel must recompute a canonical SHA-256 from load-bearing content and detect silent mutation.",
    )


def t_dead_root_precedence():
    c = claim(
        "dead_root",
        grade=k.Grade.PROVEN,
        status="killed",
    )
    r = run_claim_verdict(c, {c.id: c})
    accepted = r["results"]["PRECEDENCE"][0] == k.V.PASS
    return result(
        "V12-PRECEDENCE-IGNORES-ROOT-STATUS",
        accepted,
        r,
        "A killed/superseded/retired root must fail PRECEDENCE even without descendants.",
    )


def t_closed_core_conjecture():
    conjecture = claim("conj", grade=k.Grade.CONJECTURE)
    parent = claim(
        "closed",
        grade=k.Grade.PROVEN_MODULO,
        closed_core=True,
        edges=[
            k.Edge(
                "conj",
                k.EdgeKind.SUPPORT,
                conjecture.content_hash,
                conditional=True,
            )
        ],
    )
    g = {conjecture.id: conjecture, parent.id: parent}
    r = run_claim_verdict(parent, g)
    accepted = "CORE CLOSURE" not in r["fails"]
    return result(
        "V12-CORE-CLOSURE-ONLY-CHECKS-OPEN",
        accepted,
        r,
        "A declared closed core must not retain Conjecture/Open conditional debts; it may be Proven-Modulo but not closed.",
    )


def t_registry_admits_live_killed():
    reg = k.Registry()
    c = claim("bad", grade=k.Grade.KILLED, status="live")
    ok, msg = reg.admit(c)
    return result(
        "V12-REGISTRY-SKIPS-GRAPH-VALIDATION",
        ok,
        {"ok": ok, "message": msg, "stored": list(reg.by_id)},
        "Registry.admit must invoke structural validation and reject status/grade inconsistency.",
    )


def t_registry_load_silent_partial():
    good = claim("good", grade=k.Grade.PROVEN)
    missing_parent = claim(
        "bad_parent",
        grade=k.Grade.PROVEN,
        edges=[k.Edge("ghost", k.EdgeKind.SUPPORT, "ghost#1")],
    )
    payload = {
        "schema": "gate-kernel/1.2",
        "claims": [good.to_dict(), missing_parent.to_dict()],
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            reg = k.Registry.load(str(path))
            observed = {
                "raised": False,
                "stored": sorted(reg.by_id),
            }
            vulnerability = True
        except Exception as exc:
            observed = {"raised": True, "exception": repr(exc)}
            vulnerability = False
    return result(
        "V12-REGISTRY-LOAD-IGNORES-REFUSALS",
        vulnerability,
        observed,
        "Loading a malformed registry must raise and report every rejected claim, not silently return a partial registry.",
    )


def t_schema_version_ignored():
    c = claim("x", grade=k.Grade.PROVEN)
    payload = {"schema": "unrelated/999", "claims": [c.to_dict()]}
    try:
        g = k.graph_from_json(json.dumps(payload))
        observed = {"raised": False, "claims": sorted(g)}
        vulnerability = True
    except Exception as exc:
        observed = {"raised": True, "exception": repr(exc)}
        vulnerability = False
    return result(
        "V12-SCHEMA-VERSION-IGNORED",
        vulnerability,
        observed,
        "Deserializer must reject unsupported schema versions.",
    )


def t_cycle_topo_silent_skip():
    a = claim("A", edges=[k.Edge("B")])
    b = claim("B", edges=[k.Edge("A")])
    order = k._topo_order({"A": a, "B": b})
    vulnerability = order == []
    return result(
        "V12-TOPO-SILENTLY-SKIPS-CYCLE",
        vulnerability,
        {"order": order},
        "Topological ordering must raise on a cycle or incomplete ordering.",
    )


def t_nonportable_demo_path():
    completed = subprocess.run(
        [sys.executable, str(KERNEL_PATH)],
        cwd=BASE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    vulnerability = completed.returncode != 0 and "/mnt/user-data/outputs" in completed.stderr
    return result(
        "V12-HARDCODED-OUTPUT-PATH",
        vulnerability,
        {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1500:],
            "stderr_tail": completed.stderr[-1500:],
        },
        "The executable demo must use a CLI argument or a path relative to the script/current directory.",
    )


def t_registry_demo_semantic_status():
    registry_path = BASE / "q0 registry.json"
    reg = k.Registry.load(str(registry_path))
    g = reg.graph()
    r = run_claim_verdict(g["T_main"], g)
    vulnerability = r["verdict"].startswith("PASS") and (
        "R0" in r["open_deps"] or "C_const" in r["modulo"]
    )
    return result(
        "V12-DEMO-PASS-WITH-OPEN-AND-PROVISIONAL-DEBTS",
        vulnerability,
        {
            "verdict": r["verdict"],
            "open_deps": r["open_deps"],
            "modulo": r["modulo"],
            "recorded_ceilings": {k_: v.value for k_, v in reg.ceilings.items()},
        },
        "The registry must distinguish storage/well-typedness from promotability and display outstanding conditions in the primary verdict.",
    )


TESTS: list[Callable[[], dict]] = [
    t_domain_missing_all,
    t_domain_missing_parameter,
    t_mark_self_attestation,
    t_untyped_measure_bridge,
    t_untyped_model_bridge,
    t_wrong_operation_witness,
    t_common_mode_boolean_attestation,
    t_provisional_dependency_laundering,
    t_arbitrary_hash_not_content_bound,
    t_dead_root_precedence,
    t_closed_core_conjecture,
    t_registry_admits_live_killed,
    t_registry_load_silent_partial,
    t_schema_version_ignored,
    t_cycle_topo_silent_skip,
    t_nonportable_demo_path,
    t_registry_demo_semantic_status,
]


def main() -> None:
    rows = []
    for test in TESTS:
        try:
            rows.append(test())
        except Exception as exc:
            rows.append(
                {
                    "id": test.__name__,
                    "vulnerability_reproduced": None,
                    "observed": {"test_exception": repr(exc)},
                    "expected_secure_behavior": "Test harness should execute.",
                }
            )
    reproduced = sum(row["vulnerability_reproduced"] is True for row in rows)
    not_reproduced = sum(row["vulnerability_reproduced"] is False for row in rows)
    harness_errors = sum(row["vulnerability_reproduced"] is None for row in rows)
    report = {
        "audit_id": "C096-GATE-KERNEL-V1.2-REDTEAM",
        "kernel": {
            "path": str(KERNEL_PATH),
            "schema": "gate-kernel/1.2",
        },
        "tests": rows,
        "summary": {
            "test_count": len(rows),
            "vulnerabilities_reproduced": reproduced,
            "secure_rejections_observed": not_reproduced,
            "harness_errors": harness_errors,
        },
        "interpretation": (
            "A reproduced vulnerability is a shell/registry implementation gap. "
            "It does not assert that any mathematical bridge is false; it shows "
            "that the kernel cannot enforce the uploaded Master v1.1 contract as written."
        ),
    }
    output = BASE / "C096_GATE_KERNEL_V1_2_REDTEAM.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if harness_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
