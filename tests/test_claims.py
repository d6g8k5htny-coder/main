"""Claim-graph firewalls, with negative controls.

A checker that passes on the current data proves nothing on its own. Each test
below mutates the graph in exactly one way that a careless future edit could
introduce, and asserts that `tools/claims_check.py` rejects it. This mirrors the
program's own requirement that a verifier ship with declared negative controls
(OP-PROT-012 §4(c), OP-CNS-001 §7).
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import claims_check  # noqa: E402


def graph():
    return claims_check.load()


def run_on(mutated: dict, manifest: dict | str | None = None) -> int:
    """Run the checker against a mutated copy of the graph.

    The checker is always invoked with an explicit `--graph`. It is worth saying
    why: a default-argument bug (`def load(path=GRAPH)`, bound at import time)
    once made every mutation test below silently re-check the committed graph
    and pass no matter what was mutated. Never call the checker here without
    pointing it at the file this function wrote.

    `manifest` may be a dict (written out and passed with `--manifest`) or a
    path string, which is passed through unchanged so that a deliberately
    missing path can be tested.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mutated, f)
        argv = [sys.executable, os.path.join(ROOT, "tools", "claims_check.py"),
                "--graph", path]
        if isinstance(manifest, dict):
            mpath = os.path.join(d, "MANIFEST.json")
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            argv += ["--manifest", mpath]
        elif isinstance(manifest, str):
            argv += ["--manifest", manifest]
        return subprocess.run(argv, capture_output=True, text=True).returncode


LM_PREREQUISITES = [
    "RV-LM003-MAIN", "RV-LM004-MAIN", "RV-LM006-MAIN", "RV-LM009-MAIN",
    "RV-LM010-MAIN", "RV-LM012-MAIN", "RV-LM013-BOREL-GLOBALIZATION",
    "LM013-JOINT-STACK",
]


def all_prerequisites_satisfied(g: dict) -> dict:
    """Mutate every RV-LM011 prerequisite into a satisfied one.

    Used only as the *background* for the independence control below: it is a
    hypothetical, and nothing in the repository records any of these as
    satisfied. The register records seven open component verdicts and an open
    joint stack.
    """
    for name in LM_PREREQUISITES:
        if name in g["premises"]:
            g["premises"][name]["status_frozen_v2_2"] = "CLOSED"
            g["premises"][name]["status_register_note"] = "CLOSED"
            continue
        node = g["claims"][name]
        node["grade"] = "PASS_TECHNICAL"
        node["technical_status"] = "PASS_TECHNICAL"
        node["independent_review_state"] = "SATISFIED"
        node["independence_credit"] = 1
    g["claims"]["RV-LM011-MAIN"]["synthesis_route_satisfiable"] = True
    return g


def test_current_graph_passes():
    assert run_on(graph()) == 0


def test_rejects_unconditional_claim_resting_on_open_premise():
    g = graph()
    # D1-v2.2(2) rests on five open premises; relabelling it a live root theorem
    # is exactly the promotion the HOLD checklist forbids.
    g["claims"]["D1-v2.2(2)"]["grade"] = "LIVE_ROOT_THEOREM"
    assert run_on(g) == 1


def test_rejects_2d_3d_composition():
    g = graph()
    # The ERRATA firewall: never compose the ratified 3D upper with a 2D claim.
    g["claims"]["D1-v2.2(2)"]["depends_on"].append("SIDE24-3D-AO48-OPR-045")
    assert run_on(g) == 1


def test_rejects_prize_track_leaking_into_q0():
    g = graph()
    g["claims"]["D1-v2.2(2)"]["depends_on"].append("P15-A..D")
    assert run_on(g) == 1


def test_rejects_prize_claim_without_open_prize_flag():
    g = graph()
    g["claims"]["PR-TAL-003..008"]["original_prize_closed"] = True
    assert run_on(g) == 1


def test_rejects_dropping_the_decimal_kill():
    g = graph()
    g["claims"]["Q0-C101-QUALITATIVE-RATE"]["forbidden_extrapolations"] = []
    assert run_on(g) == 1


def test_rejects_unknown_dependency():
    g = graph()
    g["claims"]["D1-v2.2(2)"]["depends_on"].append("OBL-DOES-NOT-EXIST")
    assert run_on(g) == 1


def test_rejects_dependency_cycle():
    g = graph()
    g["premises"]["D3-LEMMA-RN-UNIF"]["depends_on"] = ["OBL-H5-REMOTE-THRESHOLD"]
    assert run_on(g) == 1


def test_rejects_conditional_claim_with_no_premises():
    g = graph()
    g["claims"]["D1-v2.2(2)"]["depends_on"] = []
    assert run_on(g) == 1


def test_five_open_premises_are_still_recorded_as_open_in_frozen_v2_2():
    """The HOLD checklist lists all five as blocking; the frozen layer must say so."""
    g = graph()
    five = ["OBL-D1-PROMOTE", "D3-LEMMA-RN-UNIF", "PERC-DECAY",
            "OBL-B1-BRANCH(loop|B1)", "B4.loc-damline"]
    assert g["claims"]["D1-v2.2(2)"]["depends_on"] == five
    for name in five:
        assert g["premises"][name]["status_frozen_v2_2"] in {"OPEN", "NOT_CLOSED"}, name


def test_rn3_is_not_marked_independent():
    """Same-provider author-side work earns zero organizational independence."""
    g = graph()
    assert g["claims"]["RN3-FAR"]["independence_credit"] == 0
    assert g["claims"]["RN5-NEAR-POINT-CERTS"]["independence_credit"] == 0


# --------------------------------------------------------------------------
# FW-LM011-PRECONDITION — the LM lemma stack
#
# docs/OPEN_PROBLEMS.md §D records the precondition in prose: RV-LM011 "needs
# LM003, LM004-v1.1, LM006, LM009, LM010-v1.1, LM012-v1.1 and LM013
# Carrier-B-v1.1 plus the joint stack FIRST." These tests are what make that a
# computed fact rather than a remembered one.
# --------------------------------------------------------------------------


def test_lm011_depends_on_all_eight_named_prerequisites():
    g = graph()
    lm011 = g["claims"]["RV-LM011-MAIN"]
    assert lm011["depends_on"] == LM_PREREQUISITES
    assert lm011["precondition_routes"] == LM_PREREQUISITES
    assert lm011["synthesis_route_satisfiable"] is False


def test_lm_route_statuses_are_the_register_statuses():
    """Transcribed, not decided: seven NEEDS_RECONCILIATION and one PASS_TECHNICAL."""
    g = graph()
    expected = {
        "RV-LM003-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM004-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM006-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM009-MAIN": "PASS_TECHNICAL",
        "RV-LM010-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM011-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM012-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM013-BOREL-GLOBALIZATION": "NEEDS_RECONCILIATION",
    }
    for name, status in expected.items():
        assert g["claims"][name]["technical_status"] == status, name
        assert g["claims"][name]["grade"] == status, name
    assert g["premises"]["LM013-JOINT-STACK"]["status_frozen_v2_2"] == "OPEN"


def test_lm009_technical_pass_carries_zero_independence_credit():
    """PASS_TECHNICAL with an OpenAI author and an OpenAI reviewer: same provider."""
    g = graph()
    lm009 = g["claims"]["RV-LM009-MAIN"]
    assert lm009["technical_status"] == "PASS_TECHNICAL"
    assert lm009["independence_credit"] == 0
    assert lm009["same_provider"] is True
    assert lm009["independent_review_state"] == "OPEN"


def test_rejects_marking_the_synthesis_route_satisfiable_while_prerequisites_are_open():
    g = graph()
    g["claims"]["RV-LM011-MAIN"]["synthesis_route_satisfiable"] = True
    assert run_on(g) == 1


def test_rejects_dropping_a_named_prerequisite_from_the_dependency_list():
    g = graph()
    g["claims"]["RV-LM011-MAIN"]["depends_on"].remove("RV-LM006-MAIN")
    assert run_on(g) == 1


def test_rejects_dropping_the_joint_stack_from_the_dependency_list():
    """'plus the joint stack' is half the precondition, not a footnote."""
    g = graph()
    g["claims"]["RV-LM011-MAIN"]["depends_on"].remove("LM013-JOINT-STACK")
    assert run_on(g) == 1


def test_rejects_unwiring_the_precondition_entirely():
    g = graph()
    del g["claims"]["RV-LM011-MAIN"]["precondition_routes"]
    assert run_on(g) == 1


def test_a_zero_independence_pass_does_not_discharge_the_lm011_gate():
    """The sign-flip control: one field decides it, and it decides it correctly.

    With every prerequisite hypothetically satisfied, the route still fails if
    RV-LM009-MAIN's organizational independence credit is zero — a same-provider
    technical pass does not discharge a gate that requires an organizationally
    distinct verdict. Granting that one credit is the only difference between
    the rejected graph and the accepted one.
    """
    g = all_prerequisites_satisfied(graph())
    g["claims"]["RV-LM009-MAIN"]["independence_credit"] = 0
    assert run_on(g) == 1

    g["claims"]["RV-LM009-MAIN"]["independence_credit"] = 1
    assert run_on(g) == 0


def test_a_satisfied_independent_review_state_alone_does_not_discharge_the_gate():
    """SATISFIED on the review state with no credit behind it is still zero."""
    g = all_prerequisites_satisfied(graph())
    g["claims"]["RV-LM009-MAIN"]["independence_credit"] = 0
    g["claims"]["RV-LM009-MAIN"]["independent_review_state"] = "SATISFIED"
    assert run_on(g) == 1


def test_rejects_technical_status_drifting_from_grade():
    """Two status fields that can disagree are two chances to promote one."""
    g = graph()
    g["claims"]["RV-LM009-MAIN"]["grade"] = "ACCEPTED_AT_REVIEW_SCOPE"
    assert run_on(g) == 1


# --------------------------------------------------------------------------
# FW-NO-RECEIPT-PROMOTION and FW-FLOAT-NOT-CERTIFIED
# --------------------------------------------------------------------------


def test_rejects_a_claim_promoted_on_receipt_only_evidence():
    g = graph()
    g["claims"]["RN3-FAR"]["evidence"] = [{
        "kind": "receipt",
        "ref": "a green run of the far-zone driver",
        "arithmetic": "exact_rational",
        "certifying": False,
    }]
    # Grade is left at AUTHOR_SIDE_CERTIFIED and the arithmetic is exact, so the
    # only thing wrong with this graph is that a receipt is carrying the grade.
    assert run_on(g) == 1


def test_rejects_a_premise_discharged_on_carrier_binding_evidence():
    """A byte-exact carrier recovery is provenance. It closes nothing."""
    g = graph()
    g["premises"]["D3-LEMMA-RN-UNIF"]["status_frozen_v2_2"] = "CLOSED"
    assert run_on(g) == 1

    # Same status change, but now carried by a proof body rather than a carrier
    # binding: the receipt firewall is what was rejecting it, nothing else.
    # (Hypothetical. This repository holds no proof body for either piece.)
    g["premises"]["D3-LEMMA-RN-UNIF"]["evidence"] = [{
        "kind": "proof_body",
        "ref": "hypothetical",
        "arithmetic": "exact_rational",
        "certifying": True,
    }]
    assert run_on(g) == 0


def test_rejects_a_register_note_discharge_on_receipt_only_evidence():
    """Both status columns are checked; neither may be moved by a receipt."""
    g = graph()
    g["premises"]["D3-LEMMA-RN-UNIF"]["status_register_note"] = "DISCHARGED"
    assert run_on(g) == 1


def test_rejects_evidence_that_calls_floating_point_certifying():
    g = graph()
    g["claims"]["RN3-FAR"]["evidence"][0]["arithmetic"] = "mpmath_float"
    assert run_on(g) == 1


def test_rejects_a_certified_claim_resting_only_on_floating_point():
    g = graph()
    for ev in g["claims"]["RN3-FAR"]["evidence"]:
        ev["arithmetic"] = "mpmath_float"
        ev["certifying"] = False
    assert run_on(g) == 1


def test_rejects_an_unknown_evidence_kind():
    g = graph()
    g["claims"]["RN3-FAR"]["evidence"][0]["kind"] = "vibes"
    assert run_on(g) == 1


def test_carrier_manifest_overrides_the_graphs_own_evidence_record():
    """When the manifest exists, the carrier's own arithmetic is what counts.

    The graph records RNENG-01 as mpmath float and NON-CERTIFYING, which is what
    engine/rn_engine/BINDING.json says. A manifest that called the same carrier
    certifying would be claiming certification for a float computation, and the
    checker must refuse it even though the graph's own copy looks clean.
    """
    manifest = {"carriers": [{
        "carrier_id": "RNENG-01",
        "arithmetic": "mpmath binary floating point at mp.dps = 100",
        "certifying": True,
    }]}
    assert run_on(graph(), manifest=manifest) == 1


def test_carrier_manifest_is_skipped_cleanly_when_it_does_not_exist():
    assert run_on(graph(), manifest=os.path.join(ROOT, "no", "such", "MANIFEST.json")) == 0


def test_rejects_restoring_an_unconditional_grade_on_a_retracted_claim():
    """Theorem B's unconditional PROVEN-HERE label was retracted by GP-AUD-187;
    the register says not to cite the historical label as current proof. Putting
    LIVE_ROOT_THEOREM back on the node — which is exactly what the first port
    did — must be refused."""
    g = graph()
    g["claims"]["Q0-C104-THEOREM-B"]["grade"] = "LIVE_ROOT_THEOREM"
    g["claims"]["Q0-C104-THEOREM-B"]["technical_status"] = "LIVE_ROOT_THEOREM"
    assert run_on(g) == 1
    # and the firewall is the one that fires, not a side effect
    problems = claims_check.check(g, None) if hasattr(claims_check, "check") else None
    if problems is not None:
        assert any("FW-RETRACTED-NOT-UNCONDITIONAL" in p for p in problems)


def test_a_retraction_record_without_a_register_status_is_refused():
    g = graph()
    del g["claims"]["Q0-C104-THEOREM-B"]["register_status"]
    assert run_on(g) == 1


def test_theorem_b_register_status_is_the_register_row_verbatim():
    """The graph node transcribes the register; it may not paraphrase it."""
    import json as _json
    ac = _json.load(open(os.path.join(ROOT, "registers", "json", "automation_config.json"), encoding="utf-8"))
    row = [r for r in ac["rows"] if r[0] == "THEOREM_B_CURRENT_STATUS"][0]
    node = graph()["claims"]["Q0-C104-THEOREM-B"]
    assert node["register_status"] == row[1]
    assert "RETRACTED" in node["register_status"]
    assert node["grade"] == "RETRACTED_TO_CANDIDATE"
    gates = [r for r in ac["rows"] if r[0] == "THEOREM_B_OPEN_GATES"][0][1].split("__")
    assert node["open_gates"] == gates and len(gates) == 7


def test_k3_assembly_is_filed_with_the_lower_campaign_and_names_its_carriers():
    """K3-THM-001 is a liminf (lower-bound) assembly consuming the LOWER2D
    premise H-B3; until 2026-09-18 it was filed under UPPER2D with no source
    identity. Both carriers are named by Drive id and digest, and the digests
    are the inventory's."""
    import json as _json
    node = graph()["claims"]["K3-THM-001"]
    assert node["track"] == "LOWER2D" and node["grade"] == "REFUTED_AS_WRITTEN"
    assert "H-B3" in node["depends_on"]
    inv = {}
    with open(os.path.join(ROOT, "drive", "inventory.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = _json.loads(line); inv[r["id"]] = r
    for fid in ("1lfH7g57LcshqpLfNbrx-H7gbJckpzPqr", "1VCDrxE5sp983c1uMTFCpmH4hJdDUw6Ut"):
        assert fid in node["source"] and inv[fid]["sha256"] in node["source"]


def test_the_five_original_firewalls_are_still_declared():
    ids = [f["id"] for f in graph()["firewalls"]]
    for original in ("FW-2D-3D-COMPOSITION", "FW-PRIZE-ISOLATION", "FW-UNCONDITIONAL",
                     "FW-NO-PRIZE-CLOSURE", "FW-DECIMAL-KILL"):
        assert original in ids
    for added in ("FW-LM011-PRECONDITION", "FW-NO-RECEIPT-PROMOTION",
                  "FW-FLOAT-NOT-CERTIFIED", "FW-RETRACTED-NOT-UNCONDITIONAL"):
        assert added in ids
