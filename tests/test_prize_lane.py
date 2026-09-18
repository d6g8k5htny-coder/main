"""The prize lane's status layer is bound to the claim graph, with negative controls.

`drive/mirrors/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK/`
holds the lane's grading files byte-exact (their manifests are verified by
tools/verify_manifests.py). These tests bind what `claims/graph.json` says about
P14, P15 and PR-TAL-003…008 to those bytes, so that a later edit of the graph
cannot drift from the source it names, and check that the prize firewalls still
bite on the new node. Nothing here grades anything: the grades compared are the
authors' own labels, transcribed.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANE = os.path.join(ROOT, "drive", "mirrors",
                    "2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK")
STATUS = os.path.join(LANE, "00_CURRENT_STATE_AND_ROUTING")
GRAPH = os.path.join(ROOT, "claims", "graph.json")


def graph():
    with open(GRAPH, encoding="utf-8") as f:
        return json.load(f)


def inventory():
    inv = {}
    with open(os.path.join(ROOT, "drive", "inventory.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            inv[r["id"]] = r
    return inv


def run_on(mutated: dict) -> int:
    """Invoke tools/claims_check.py through its CLI on a written copy (never on
    the module's default path: see tests/test_claims.py for why)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mutated, f)
        argv = [sys.executable, os.path.join(ROOT, "tools", "claims_check.py"),
                "--graph", path]
        return subprocess.run(argv, capture_output=True, text=True).returncode


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_the_mirrored_status_files_are_the_inventory_bytes():
    inv = inventory()
    rows = []
    for d in (LANE, STATUS, os.path.join(LANE, "05_ERDOS_142 — r_k(N) ASYMPTOTIC — HIGH-PRIZE PROBE")):
        with open(os.path.join(d, "_MANIFEST.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows.append((d, row))
    assert len(rows) == 8
    for d, row in rows:
        assert row["exact"] is True and row["stored"] is True
        assert row["sha256"] == inv[row["id"]]["sha256"] == _sha(os.path.join(d, row["dest"]))
        assert row["bytes"] == int(inv[row["id"]]["bytes"])


def test_p14_node_is_the_registry_transcribed():
    with open(os.path.join(STATUS, "P14_CLAIM_REGISTRY.json"), encoding="utf-8") as f:
        reg = json.load(f)
    node = graph()["claims"]["P14-A..E"]
    assert node["track"] == "NUMBER_THEORY"
    assert node["grade"] == reg["grade"] == "AUTHOR_SIDE_COMPLETE"
    assert node["external_independent_reviews"] == reg["external_independent_reviews"] == 0
    assert node["prizes_solved"] == reg["prizes_solved"] == 0
    assert node["historical_novelty"] == reg["historical_novelty"] == "UNESTABLISHED"
    assert node["original_prize_closed"] is False
    assert [c["id"] for c in reg["claims"]] == ["P14-A", "P14-B", "P14-C", "P14-D/E"]
    for c in reg["claims"]:
        assert c["sha256"][:8] in node["source"]["proof_bodies"]
    inv = inventory()
    assert inv["1iMJsX4j1DRsBy6CnPun4e65w2u1xK2qx"]["sha256"] in node["source"]["registry"]


def test_p15_depends_on_p14_as_its_current_state_says():
    with open(os.path.join(STATUS, "P15_CURRENT_STATE_AUTHOR_CANDIDATE.md"), encoding="utf-8") as f:
        text = f.read()
    assert "through the preceding P14 theorem" in text
    with open(os.path.join(STATUS, "P15_CLAIM_REGISTRY.json"), encoding="utf-8") as f:
        reg = json.load(f)
    node = graph()["claims"]["P15-A..D"]
    assert "P14-A..E" in node["depends_on"]
    assert reg["automatic_scientific_promotion"] is False
    assert reg["external_independence_credit"] == 0 and reg["q0_changes"] == 0
    for c in reg["claims"]:
        assert c["grade"] == "AUTHOR_SIDE_COMPLETE" and c["prize_closed"] is False
        assert c["external_reviews"] == 0
    assert node["original_prize_closed"] is False
    inv = inventory()
    assert inv["1r8weyHObFfHAp-_PfI8mlSXPZQC8Zoh1"]["sha256"] in node["source"]["current_state"]


def test_pr_tal_node_matches_the_intake_registry():
    with open(os.path.join(STATUS, "CLAIM_REGISTRY_VERIFIED_INTAKE.json"), encoding="utf-8") as f:
        reg = json.load(f)
    by_id = {c["claim_id"]: c for c in reg["claims"]}
    assert len(by_id) == 28
    node = graph()["claims"]["PR-TAL-003..008"]
    for k in ("PR-TAL-003", "PR-TAL-004", "PR-TAL-005", "PR-TAL-006", "PR-TAL-007", "PR-TAL-008"):
        assert by_id[k]["declared_proof_grade"] == node["grade"] == "AUTHOR_SIDE_PROOF_PRESENT"
        assert by_id[k]["original_prize_closed"] is False
    # the two weaker Talagrand grades the map must not fold into the node's grade
    assert by_id["PR-TAL-009"]["declared_proof_grade"] == "SOURCE_DECLARED_AUTHOR_ARGUMENT"
    assert by_id["PR-TAL-010"]["declared_proof_grade"] == "SOURCE_DECLARED_FINITE_CERTIFICATE_NOTE"
    assert all(c["original_prize_closed"] is False for c in reg["claims"])
    assert "122 members" in node["source"]


def test_negative_control_p14_cannot_be_filed_under_a_q0_track():
    g = copy.deepcopy(graph())
    assert run_on(g) == 0
    g["claims"]["P14-A..E"]["track"] = "UPPER2D"
    # P15 (NUMBER_THEORY) now reaches a q0 track through its dependency
    assert run_on(g) != 0


def test_negative_control_p14_must_carry_original_prize_closed_false():
    g = copy.deepcopy(graph())
    del g["claims"]["P14-A..E"]["original_prize_closed"]
    assert run_on(g) != 0
    g = copy.deepcopy(graph())
    g["claims"]["P14-A..E"]["original_prize_closed"] = True
    assert run_on(g) != 0


def test_negative_control_a_q0_claim_cannot_depend_on_p14():
    g = copy.deepcopy(graph())
    g["claims"]["Q0-C101-QUALITATIVE-RATE"]["depends_on"].append("P14-A..E")
    assert run_on(g) != 0
