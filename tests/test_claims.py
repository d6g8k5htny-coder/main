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


def run_on(mutated: dict) -> int:
    """Run the checker against a mutated copy of the graph."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mutated, f)
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "claims_check.py"),
             "--graph", path],
            capture_output=True, text=True).returncode


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
