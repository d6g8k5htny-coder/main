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
    """Transcribed, not decided: six NEEDS_RECONCILIATION and two PASS_TECHNICAL
    (RV-LM004-MAIN moved to PASS_TECHNICAL in the 2026-09-18 register export)."""
    g = graph()
    expected = {
        "RV-LM003-MAIN": "NEEDS_RECONCILIATION",
        "RV-LM004-MAIN": "PASS_TECHNICAL",
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


def test_lm004_technical_pass_carries_zero_independence_credit():
    """The 2026-09-18 register moved RV-LM004-MAIN to PASS_TECHNICAL with the
    reviewer line 'Prior OpenAI technical reconciliation; ROUND5 author-line
    erratum; zero org credit'. The register's own words are transcribed; the
    independence gate is untouched."""
    g = graph()
    lm004 = g["claims"]["RV-LM004-MAIN"]
    assert lm004["technical_status"] == "PASS_TECHNICAL" and lm004["grade"] == "PASS_TECHNICAL"
    assert lm004["independence_credit"] == 0
    assert lm004["independent_review_state"] == "OPEN"
    assert lm004["requires_independent_verdict"] is True
    assert "zero org credit" in lm004["note"]
    assert g["claims"]["RV-LM011-MAIN"]["synthesis_route_satisfiable"] is False


# Every field of a review-route node that transcribes a register column, with
# the register tab and column it transcribes. `age_days` is an int in the graph
# and a string in the register. The list is the control for the 2026-09-18
# labelling residue: a refresh that moved technical_status/grade but left
# reviewer_claim, aging_action and age_days at the previous export's words.
ROUTE_FIELDS_FROM_REVIEW_QUEUE = [
    ("technical_status", "Technical status", str),
    ("grade", "Technical status", str),
    ("independence_status", "Independence status", str),
    ("reviewer_claim", "Reviewer / claim", str),
    ("aging_action", "Aging action", str),
    ("age_days", "Age days", int),
    ("body_bytes", "Body bytes", int),
    ("body_sha256", "Body SHA-256", str),
    ("author_provider", "Author / provider", str),
]
ROUTE_FIELDS_FROM_EASY_CLOSURE_QUEUE = [
    ("queue_state", "Queue state", str),
    ("remaining_decisive_work", "Remaining decisive work", str),
    ("independent_review_verbatim", "Independent review needed", str),
]


def route_transcription_mismatches(g: dict) -> list[tuple[str, str, object, object]]:
    """(route, field, graph value, register value) for every review-route node
    field that does not equal the cell of the register row the node's `source`
    cites. Empty when the graph transcribes the register."""
    import json as _json
    import re as _re
    rq = _json.load(open(os.path.join(ROOT, "registers", "json", "review_queue.json"), encoding="utf-8"))
    ecq = _json.load(open(os.path.join(ROOT, "registers", "json", "easy_closure_queue.json"), encoding="utf-8"))
    rh, eh = rq["header"], ecq["header"]
    rrows = {r[rh.index("Review key")]: r for r in rq["rows"]}
    erows = {r[eh.index("Candidate ID")]: r for r in ecq["rows"]}
    out = []
    routes = [k for k, v in g["claims"].items() if k.startswith("RV-") and "technical_status" in v]
    assert routes, "the graph carries review-route nodes"
    for name in routes:
        node = g["claims"][name]
        assert name in rrows, name
        assert f"review_queue.json row {name}" in node["source"], name
        for field, col, conv in ROUTE_FIELDS_FROM_REVIEW_QUEUE:
            reg = conv(rrows[name][rh.index(col)])
            if node.get(field) != reg:
                out.append((name, field, node.get(field), reg))
        m = _re.search(r"easy_closure_queue\.json row (\S+)", node["source"])
        if m is None:
            # A route with no easy_closure_queue row must say so explicitly
            # (easy_closure_queue_row: null); silence is a missing citation.
            if "easy_closure_queue_row" not in node or node["easy_closure_queue_row"] is not None:
                out.append((name, "easy_closure_queue_row", node.get("easy_closure_queue_row", "<absent>"), None))
            continue
        assert m.group(1) in erows, (name, m.group(1))
        for field, col, conv in ROUTE_FIELDS_FROM_EASY_CLOSURE_QUEUE:
            reg = conv(erows[m.group(1)][eh.index(col)])
            if node.get(field) != reg:
                out.append((name, field, node.get(field), reg))
    return out


def test_a_route_without_an_easy_closure_row_declares_it():
    """RV-RN5-MOMENT-REPAIR (created 2026-09-17, first seen in the 2026-09-18
    export) has a review_queue row and no easy_closure_queue row. The node
    transcribes the row it has and declares the row it lacks; the checker
    accepts READY as a strength-1 word (work recorded, nothing discharged)."""
    g = graph()
    node = g["claims"]["RV-RN5-MOMENT-REPAIR"]
    assert node["technical_status"] == node["grade"] == "READY"
    assert node["independence_credit"] == 0 and node["easy_closure_queue_row"] is None
    assert claims_check.GRADE_STRENGTH["READY"] == 1 and claims_check.GRADE_STRENGTH["AMEND"] == 1
    assert run_on(g) == 0


def test_control_a_route_silent_about_its_missing_easy_closure_row_is_a_mismatch():
    g = graph()
    del g["claims"]["RV-RN5-MOMENT-REPAIR"]["easy_closure_queue_row"]
    found = {(r, f) for r, f, _, _ in route_transcription_mismatches(g)}
    assert found == {("RV-RN5-MOMENT-REPAIR", "easy_closure_queue_row")}


def test_control_ready_cannot_be_read_as_stronger_than_conditional():
    """READY and AMEND are review-queue words at strength 1: a route carrying
    either can never make a claim resting on it unconditional."""
    g = graph()
    g["claims"]["RV-RN5-MOMENT-REPAIR"]["grade"] = "LIVE_ROOT_THEOREM"
    g["claims"]["RV-RN5-MOMENT-REPAIR"]["technical_status"] = "LIVE_ROOT_THEOREM"
    assert route_transcription_mismatches(g), "the register's word is READY"


def test_review_route_nodes_transcribe_the_register():
    """Every review-route node in the graph carries, field for field, the words
    of the review_queue row and the easy_closure_queue row its source cites —
    not only Technical status but Independence status, Reviewer / claim, Aging
    action, Age days, the body identity and the queue state — and no route
    carries independence credit. A row is a transcription; a PASS_TECHNICAL is
    a same-line pass at zero credit; an aging action approves nothing."""
    g = graph()
    assert route_transcription_mismatches(g) == []
    for name, node in g["claims"].items():
        if name.startswith("RV-") and "technical_status" in node:
            assert node["independence_credit"] == 0, name


def test_control_a_stale_reviewer_line_or_age_is_a_mismatch():
    """The residue the 2026-09-18 adversary found, replayed on a copy: the
    register's words on technical_status/grade with the previous export's words
    left on reviewer_claim, aging_action and age_days. The comparison must name
    all three, and only those three."""
    g = graph()
    node = g["claims"]["RV-LM004-MAIN"]
    assert node["technical_status"] == "PASS_TECHNICAL"
    node["reviewer_claim"] = "UNASSIGNED"
    node["aging_action"] = "ESCALATE"
    node["age_days"] = 54
    found = {(r, f) for r, f, _, _ in route_transcription_mismatches(g)}
    assert found == {("RV-LM004-MAIN", "reviewer_claim"), ("RV-LM004-MAIN", "aging_action"),
                     ("RV-LM004-MAIN", "age_days")}


def test_control_a_route_status_stronger_than_the_register_is_a_mismatch():
    g = graph()
    g["claims"]["RV-LM003-MAIN"]["technical_status"] = "PASS_TECHNICAL"
    g["claims"]["RV-LM003-MAIN"]["grade"] = "PASS_TECHNICAL"
    found = {(r, f) for r, f, _, _ in route_transcription_mismatches(g)}
    assert found == {("RV-LM003-MAIN", "technical_status"), ("RV-LM003-MAIN", "grade")}


def test_a_zero_credit_lm004_pass_does_not_discharge_the_lm011_gate():
    """Same control as for LM009, on the route the refresh moved: with every
    other prerequisite hypothetically satisfied, LM004's zero credit alone keeps
    the synthesis route unsatisfiable."""
    g = all_prerequisites_satisfied(graph())
    g["claims"]["RV-LM004-MAIN"]["independence_credit"] = 0
    assert run_on(g) == 1
    g["claims"]["RV-LM004-MAIN"]["independence_credit"] = 1
    assert run_on(g) == 0


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


def _side24_mismatches(g):
    """Field-for-field binding of the SIDE24 3D node to the operator_decisions
    register row, the inventory row and the mirrored record bytes."""
    import hashlib as _h
    import json as _json
    out = []
    node = g["claims"]["SIDE24-3D-AO48-OPR-045"]
    od = _json.load(open(os.path.join(ROOT, "registers", "json", "operator_decisions.json"), encoding="utf-8"))
    row = [r for r in od["rows"] if r[0] == "AO48-OPR-045"][0]
    if node.get("register_status") != row[5]:
        out.append(("register_status", node.get("register_status"), row[5]))
    if node.get("drive_id") != row[3]:
        out.append(("drive_id", node.get("drive_id"), row[3]))
    if row[7] != "Dylan Roy" or "Dylan Roy" not in node.get("operator_verbatim", ""):
        out.append(("operator", node.get("operator_verbatim"), row[7]))
    inv = {}
    with open(os.path.join(ROOT, "drive", "inventory.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = _json.loads(line); inv[r["id"]] = r
    r = inv[row[3]]
    if node.get("body_sha256") != r["sha256"] or node.get("body_bytes") != r["bytes"]:
        out.append(("body", (node.get("body_sha256"), node.get("body_bytes")), (r["sha256"], r["bytes"])))
    if row[3] not in node["source"] or r["sha256"] not in node["source"]:
        out.append(("source", node["source"], (row[3], r["sha256"])))
    mp = os.path.join(ROOT, node.get("mirror_path", ""))
    if not os.path.isfile(mp):
        out.append(("mirror_path", node.get("mirror_path"), "missing"))
    else:
        b = open(mp, "rb").read()
        if _h.sha256(b).hexdigest() != node.get("body_sha256") or len(b) != node.get("body_bytes"):
            out.append(("mirror bytes", _h.sha256(b).hexdigest(), node.get("body_sha256")))
        text = b.decode("utf-8")
        for key in ("scope_verbatim", "firewall_verbatim", "record_author_verbatim", "ratification_text_verbatim"):
            # the record wraps lines; compare with whitespace collapsed
            if " ".join(node.get(key, "").split()) not in " ".join(text.split()):
                out.append((key, node.get(key), "not in the mirrored record"))
        for dep in node.get("carried_dependencies_verbatim", []) + node.get("reopening_conditions_verbatim", []):
            if " ".join(dep.split()) not in " ".join(text.split()):
                out.append(("verbatim", dep, "not in the mirrored record"))
    if node.get("track") != "LIFETIME3D" or node.get("depends_on") != []:
        out.append(("track/depends_on", (node.get("track"), node.get("depends_on")), ("LIFETIME3D", [])))
    if node.get("independence_credit") != 0:
        out.append(("independence_credit", node.get("independence_credit"), 0))
    for e in node.get("evidence_trail", []):
        r = inv.get(e["drive_id"])
        if r is None or r["sha256"] != e["sha256"] or r["bytes"] != e["bytes"]:
            out.append(("evidence_trail", e, r and (r["sha256"], r["bytes"])))
    return out


def test_side24_3d_node_transcribes_the_register_row_and_the_mirrored_record():
    """The node's status is the operator_decisions row's word, its bytes are the
    inventory's, and every quoted sentence is in the byte-exact mirror."""
    assert _side24_mismatches(graph()) == []
    node = graph()["claims"]["SIDE24-3D-AO48-OPR-045"]
    assert node["register_status"] == "RATIFIED-AT-STATED-SCOPE"
    assert node["grade"] == "RATIFIED_3D_ONLY"
    assert any("2D" in x for x in node["forbidden_extrapolations"])
    assert len(node["carried_dependencies_verbatim"]) == 3 and len(node["reopening_conditions_verbatim"]) == 3


def test_control_a_paraphrased_3d_status_is_a_mismatch():
    g = graph()
    g["claims"]["SIDE24-3D-AO48-OPR-045"]["register_status"] = "RATIFIED"  # the scope qualifier dropped
    assert [m[0] for m in _side24_mismatches(g)] == ["register_status"]


def test_control_a_sentence_the_record_does_not_contain_is_a_mismatch():
    g = graph()
    g["claims"]["SIDE24-3D-AO48-OPR-045"]["firewall_verbatim"] = "this ratification also closes P0.1"
    assert ("firewall_verbatim" in [m[0] for m in _side24_mismatches(g)])


def test_control_a_dropped_carried_dependency_is_a_mismatch():
    g = graph()
    g["claims"]["SIDE24-3D-AO48-OPR-045"]["carried_dependencies_verbatim"].append("nothing is carried")
    assert ("verbatim" in [m[0] for m in _side24_mismatches(g)])


def test_the_3d_node_still_cannot_be_composed_with_2d_after_enrichment():
    g = graph()
    g["claims"]["SIDE24-3D-AO48-OPR-045"]["depends_on"] = ["D1-v2.2(2)"]
    assert run_on(g) == 1


def test_the_five_original_firewalls_are_still_declared():
    ids = [f["id"] for f in graph()["firewalls"]]
    for original in ("FW-2D-3D-COMPOSITION", "FW-PRIZE-ISOLATION", "FW-UNCONDITIONAL",
                     "FW-NO-PRIZE-CLOSURE", "FW-DECIMAL-KILL"):
        assert original in ids
    for added in ("FW-LM011-PRECONDITION", "FW-NO-RECEIPT-PROMOTION",
                  "FW-FLOAT-NOT-CERTIFIED", "FW-RETRACTED-NOT-UNCONDITIONAL"):
        assert added in ids


# ---------------------------------------------------------------------------
# FW-PROPOSED-LAYER-NOT-A-STATUS
#
# The corpus transcribes some PROPOSED-tier promotions verbatim so a reader can
# see what a source proposed without leaving this repository. `OBL-H5-ZBAND`
# carries "OBL-H5-ZBAND: OPEN -> DISCHARGED (consumption grade)" beside two
# status fields that both say OPEN. That is correct -- and until 2026-09-22 it
# was correct by FIELD NAMING AND PROSE ALONE; `tools/claims_check.py` did not
# know the field existed. The pull request's owner flagged exactly that
# skim-trap: "do not promote from word search. Green CI != discharge."
# ---------------------------------------------------------------------------

PROPOSED_PREMISE = "OBL-H5-ZBAND"


def test_the_proposed_layer_premise_is_still_shaped_the_way_the_firewall_expects():
    """Not vacuous: the graph really does carry a proposed-layer transcription."""
    p = graph()["premises"][PROPOSED_PREMISE]
    assert "DISCHARGED" in p["proposed_layer_verbatim"]
    assert "→" in p["proposed_layer_verbatim"] or "->" in p["proposed_layer_verbatim"]
    assert p["proposed_layer_source"]
    assert p["status_frozen_v2_2"] == "OPEN"
    assert p["status_register_note"] == "OPEN"


def test_the_firewall_is_declared_in_the_graph():
    ids = {f["id"] for f in graph()["firewalls"]}
    assert "FW-PROPOSED-LAYER-NOT-A-STATUS" in ids


def test_rejects_the_proposed_value_appearing_in_a_status_field():
    """The skim-trap itself: DISCHARGED pasted where OPEN belongs."""
    g = graph()
    g["premises"][PROPOSED_PREMISE]["status_frozen_v2_2"] = "DISCHARGED"
    assert run_on(g) != 0


def test_rejects_the_proposed_value_in_the_other_status_field():
    g = graph()
    g["premises"][PROPOSED_PREMISE]["status_register_note"] = "DISCHARGED (consumption grade)"
    assert run_on(g) != 0


def test_rejects_a_status_field_holding_a_transition():
    """A status is a value, not an arrow."""
    g = graph()
    g["premises"][PROPOSED_PREMISE]["status_frozen_v2_2"] = "OPEN → DISCHARGED"
    assert run_on(g) != 0


def test_rejects_an_ascii_transition_in_a_status_field():
    g = graph()
    g["premises"][PROPOSED_PREMISE]["status_register_note"] = "OPEN -> DISCHARGED"
    assert run_on(g) != 0


def test_rejects_a_transcription_with_no_source():
    """An unsourced proposal is indistinguishable from an assertion."""
    g = graph()
    del g["premises"][PROPOSED_PREMISE]["proposed_layer_source"]
    assert run_on(g) != 0


def test_rejects_a_source_that_does_not_record_the_tier():
    g = graph()
    g["premises"][PROPOSED_PREMISE]["proposed_layer_source"] = \
        "H5_ZBAND_CONSUMPTION_2026-09-15.md, AUTHORITY: none"
    assert run_on(g) != 0


def test_rejects_a_source_that_does_not_record_the_authority():
    g = graph()
    g["premises"][PROPOSED_PREMISE]["proposed_layer_source"] = \
        "H5_ZBAND_CONSUMPTION_2026-09-15.md, STATUS: PROPOSED"
    assert run_on(g) != 0


def test_rejects_a_status_moved_off_the_pre_promotion_value():
    """The transition says OPEN -> ...; a status that is no longer OPEN needs its own source."""
    g = graph()
    g["premises"][PROPOSED_PREMISE]["status_frozen_v2_2"] = "CLOSED"
    assert run_on(g) != 0


def test_rejects_a_verbatim_field_that_names_no_transition():
    g = graph()
    g["premises"][PROPOSED_PREMISE]["proposed_layer_verbatim"] = "OBL-H5-ZBAND is discharged"
    assert run_on(g) != 0


def test_a_second_proposed_layer_entry_is_guarded_too():
    """The rule is general: a new transcription cannot land unguarded."""
    g = graph()
    victim = "OBL-H5-JETMOD"
    assert victim in g["premises"], "pick a premise that exists"
    g["premises"][victim]["proposed_layer_verbatim"] = "OBL-H5-JETMOD: OPEN → DISCHARGED"
    assert run_on(g) != 0, "a transcription with no source must be refused"
    g["premises"][victim]["proposed_layer_source"] = "somewhere: STATUS: PROPOSED, AUTHORITY: none"
    assert run_on(g) == 0, "sourced, with statuses still OPEN, is the legal shape"
    g["premises"][victim]["status_frozen_v2_2"] = "DISCHARGED"
    assert run_on(g) != 0, "and the proposed value must not reach a status field"
