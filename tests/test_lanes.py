"""The work-ledger invariants, with negative controls.

`tools/lanes_check.py` passing on the current lane files proves nothing on its
own: a checker that cannot fail is decoration. Each test below breaks the ledger
in exactly one way a careless future edit could break it — most importantly, by
promoting a lane above the status `claims/graph.json` records for the same
object — and asserts that the checker rejects it.

The controls run against a mutated *copy* of the ledger in a temporary
directory. Nothing under `engine/lanes/`, `claims/` or `registers/` is modified.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "lanes_check.py")
DISPATCHER = os.path.join(ROOT, "engine", "next_action.py")
LANES = os.path.join(ROOT, "engine", "lanes")
DOC = os.path.join(ROOT, "docs", "OPEN_PROBLEMS.md")
GRAPH = os.path.join(ROOT, "claims", "graph.json")
REGISTERS = os.path.join(ROOT, "registers", "json")
REVIEW_QUEUE = os.path.join(REGISTERS, "review_queue.json")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")
BINDING = os.path.join(ROOT, "engine", "rn_engine", "BINDING.json")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class Workspace:
    """A writable copy of the ledger's inputs."""

    def __init__(self, tmp_path):
        self.root = str(tmp_path)
        self.lanes = os.path.join(self.root, "lanes")
        shutil.copytree(LANES, self.lanes)
        self.doc = os.path.join(self.root, "OPEN_PROBLEMS.md")
        shutil.copyfile(DOC, self.doc)
        self.graph = os.path.join(self.root, "graph.json")
        shutil.copyfile(GRAPH, self.graph)
        # The two carrier indexes are ledger inputs too: the real lanes bind
        # carrier ids from both, so a faithful copy carries both. A test that
        # wants an index absent removes the copy; nothing real is touched.
        self.manifest = os.path.join(self.root, "MANIFEST.json")
        shutil.copyfile(MANIFEST, self.manifest)
        self.binding = os.path.join(self.root, "BINDING.json")
        shutil.copyfile(BINDING, self.binding)
        self.repo_root = os.path.join(self.root, "repo")
        os.makedirs(self.repo_root, exist_ok=True)

    def _append(self, path, carrier_id):
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        index["carriers"].append({"carrier_id": carrier_id})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def add_carrier(self, carrier_id):
        """Add a Drive-file record to the workspace's copy of MANIFEST.json."""
        self._append(self.manifest, carrier_id)

    def add_member(self, carrier_id):
        """Add an archive-member record to the workspace's copy of BINDING.json."""
        self._append(self.binding, carrier_id)

    def lane(self, key):
        with open(os.path.join(self.lanes, f"{key}.json"), encoding="utf-8") as f:
            return json.load(f)

    def write_lane(self, key, data):
        with open(os.path.join(self.lanes, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def edit(self, key, **fields):
        lane = self.lane(key)
        lane.update(fields)
        self.write_lane(key, lane)

    def run(self):
        return subprocess.run(
            [sys.executable, CHECKER, "--lanes", self.lanes, "--doc", self.doc,
             "--graph", self.graph, "--registers", REGISTERS,
             "--review-queue", REVIEW_QUEUE, "--manifest", self.manifest,
             "--binding", self.binding,          # never the real one: a workspace resolves only what it wrote
             "--repo-root", self.repo_root],
            capture_output=True, text=True)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)


def dispatcher(*args):
    out = subprocess.run([sys.executable, DISPATCHER, *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


# --------------------------------------------------------------------------
# the ledger as it stands
# --------------------------------------------------------------------------

def test_checker_passes_on_the_repository():
    out = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_unmutated_copy_passes(ws):
    assert ws.run().returncode == 0


def test_every_lane_says_what_it_does_not_establish():
    for name in sorted(os.listdir(LANES)):
        with open(os.path.join(LANES, name), encoding="utf-8") as f:
            lane = json.load(f)
        assert lane["does_not_establish"].strip(), name


def test_lane_d_heading_counts_are_the_register_recount():
    """'Review queue — N routes, M unassigned' is transcribed from
    registers/json/review_queue.json: N rows, M with Reviewer / claim = UNASSIGNED
    (25 and 22 at the 2026-09-18 export). Recounted here from the register so a
    stale heading, in the lane or in docs/OPEN_PROBLEMS.md, is caught."""
    import re
    with open(os.path.join(LANES, "D.json"), encoding="utf-8") as f:
        lane = json.load(f)
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        rq = json.load(f)
    h = rq["header"]
    n = len(rq["rows"])
    m = sum(1 for r in rq["rows"] if r[h.index("Reviewer / claim")] == "UNASSIGNED")
    assert lane["title"] == f"Review queue — {n} routes, {m} unassigned"
    assert lane["source_section"]["heading"] == f"## D. {lane['title']}"
    with open(DOC, encoding="utf-8") as f:
        assert lane["source_section"]["heading"] in f.read()
    assert len(lane["sub_items"]) == n
    assert re.search(r"\bREADY ×4\b", lane["status_source"]["note"])
    assert re.search(r"\bPASS_TECHNICAL ×2\b", lane["status_source"]["note"])
    assert re.search(r"\bAMEND ×2\b", lane["status_source"]["note"])


def test_lane_d_sub_items_carry_the_refreshed_register_words():
    """The three routes the 2026-09-18 refresh moved, verbatim from the register."""
    with open(os.path.join(LANES, "D.json"), encoding="utf-8") as f:
        items = {i["review_key"]: i for i in json.load(f)["sub_items"]}
    assert items["RV-LM004-MAIN"]["technical_status"] == "PASS_TECHNICAL"
    assert items["RV-LM004-MAIN"]["independence_status"] == "EXTERNAL_REVIEW_OPEN"
    assert items["RV-LM004-MAIN"]["reviewer_claim"].endswith("zero org credit")
    assert items["RV-RN-ALIGN"]["technical_status"] == "AMEND"
    assert items["RV-RN-ALIGN"]["independence_status"] == "AUTHOR_SIDE / ZERO ORG CREDIT"
    assert items["RV-RN5-MOMENT-REPAIR"]["technical_status"] == "READY"
    assert items["RV-RN5-MOMENT-REPAIR"]["reviewer_claim"] == "UNASSIGNED"
    assert items["RV-RN5-MOMENT-REPAIR"]["blocks"] == []


def test_no_lane_carries_a_certified_or_closed_status():
    """Transcription check: the open-problems document has no discharged lane."""
    for name in sorted(os.listdir(LANES)):
        with open(os.path.join(LANES, name), encoding="utf-8") as f:
            lane = json.load(f)
        assert lane["status"] not in {"CLOSED", "DISCHARGED", "PROMOTED"}, name


# --------------------------------------------------------------------------
# negative controls — each must FAIL the checker
# --------------------------------------------------------------------------

def test_negative_control_promotion_above_the_claim_graph(ws):
    """A8's premise is OPEN in the frozen layer; a lane may not read CLOSED."""
    ws.edit("A8", status="CLOSED")
    out = ws.run()
    assert out.returncode != 0
    assert "PROMOTION REFUSED" in out.stdout


def test_negative_control_promotion_of_the_rn_lemma(ws):
    ws.edit("A5", status="DISCHARGED")
    out = ws.run()
    assert out.returncode != 0
    assert "PROMOTION REFUSED" in out.stdout


def test_negative_control_unknown_blocks_id(ws):
    ws.edit("A1", blocks=["OBL-H5-JETMOD", "OBL-NO-SUCH-PREMISE"])
    out = ws.run()
    assert out.returncode != 0
    assert "OBL-NO-SUCH-PREMISE" in out.stdout


def test_negative_control_missing_lane_for_a_documented_section(ws):
    os.remove(os.path.join(ws.lanes, "A1.json"))
    out = ws.run()
    assert out.returncode != 0
    assert "has no lane file A1.json" in out.stdout


def test_negative_control_lane_with_no_section(ws):
    lane = ws.lane("A1")
    lane["key"] = "Z9"
    ws.write_lane("Z9", lane)
    out = ws.run()
    assert out.returncode != 0
    assert "Z9.json has no section" in out.stdout


def test_negative_control_section_list_is_not_hardcoded(ws):
    """Adding a section to the document demands a lane for it. The section
    letter is the first one no lane file uses, so the control keeps working as
    real sections are added."""
    used = {name[:-5] for name in os.listdir(ws.lanes) if name.endswith(".json")}
    letter = next(c for c in "ZYXWVUTSRQPONMLKJIH" if c not in used)
    with open(ws.doc, "a", encoding="utf-8") as f:
        f.write(f"\n## {letter}. A newly documented open problem\n\nBody.\n")
    out = ws.run()
    assert out.returncode != 0
    assert f"section {letter}" in out.stdout


def test_negative_control_status_outside_the_register_vocabulary(ws):
    ws.edit("A1", status="LOOKS_FINE_TO_ME")
    out = ws.run()
    assert out.returncode != 0
    assert "LOOKS_FINE_TO_ME" in out.stdout


def test_negative_control_null_status_without_a_reason(ws):
    lane = ws.lane("A2")
    lane["status"] = None
    lane.pop("status_absent_reason", None)
    ws.write_lane("A2", lane)
    out = ws.run()
    assert out.returncode != 0
    assert "status_absent_reason" in out.stdout


def test_negative_control_review_route_status_edited(ws):
    lane = ws.lane("D")
    lane["sub_items"][0]["technical_status"] = "PASS_TECHNICAL"
    ws.write_lane("D", lane)
    out = ws.run()
    assert out.returncode != 0
    assert "the register says" in out.stdout


def test_negative_control_review_route_dropped(ws):
    lane = ws.lane("D")
    dropped = lane["sub_items"].pop()["review_key"]
    ws.write_lane("D", lane)
    out = ws.run()
    assert out.returncode != 0
    assert dropped in out.stdout


def test_negative_control_unbound_carrier_input(ws):
    ws.edit("A5", inputs=["CARRIER-THAT-IS-NOT-THERE"])
    out = ws.run()
    assert out.returncode != 0
    assert "CARRIER-THAT-IS-NOT-THERE" in out.stdout
    # and the same lane with a carrier id the manifest lists passes
    ws.add_carrier("CARRIER-REAL-001")
    ws.edit("A5", inputs=["CARRIER-REAL-001"])
    assert ws.run().returncode == 0


def test_input_may_resolve_against_the_archive_member_index(ws):
    """Two indexes on purpose: MANIFEST.json for Drive files, BINDING.json for
    files recovered from inside ZIP carriers. An input naming an id in neither
    must fail, and the message must name both indexes; the same input passes
    once the binding index lists it, without the manifest changing."""
    ws.edit("A5", inputs=["RNENG-TEST-99"])
    out = ws.run()
    assert out.returncode != 0
    assert "neither index" in out.stdout
    assert "MANIFEST.json" in out.stdout and "BINDING.json" in out.stdout
    ws.add_member("RNENG-TEST-99")
    assert ws.run().returncode == 0


def test_the_real_lanes_depend_on_the_archive_member_index(ws):
    """A5 binds the eight recovered RN-engine files. Remove the binding index and
    the unmutated lanes must fail on exactly those ids: the second index is
    load-bearing, not decorative."""
    os.remove(ws.binding)
    out = ws.run()
    assert out.returncode != 0
    assert "RNENG-01" in out.stdout
    assert "neither index" in out.stdout


def test_dispatcher_and_checker_agree_on_the_second_index(ws):
    """The checker and the dispatcher must read the same lane the same way.
    On the unmutated copy A5 is bound, and the dispatcher's note names the
    index that resolved the engine members; remove that index and the
    dispatcher reads A5 as unbound while the checker fails on the same ids."""
    def dispatch():
        out = subprocess.run(
            [sys.executable, DISPATCHER, "--lanes", ws.lanes, "--graph", ws.graph,
             "--manifest", ws.manifest, "--binding", ws.binding, "--json"],
            capture_output=True, text=True, check=True).stdout
        return {r["key"]: r for r in json.loads(out)["lanes"]}["A5"]

    bound = dispatch()
    assert bound["inputs_bound_here"] is True
    assert "BINDING.json" in bound["inputs_note"]
    assert ws.run().returncode == 0

    os.remove(ws.binding)
    unbound = dispatch()
    assert unbound["inputs_bound_here"] is False
    assert "RNENG-01" in unbound["inputs_note"]
    out = ws.run()
    assert out.returncode != 0 and "RNENG-01" in out.stdout


def test_negative_control_inputs_declared_without_a_manifest(ws):
    os.remove(ws.manifest)
    out = ws.run()
    assert out.returncode != 0
    assert "does not exist" in out.stdout


def test_negative_control_empty_does_not_establish(ws):
    ws.edit("B", does_not_establish="")
    out = ws.run()
    assert out.returncode != 0
    assert "does_not_establish" in out.stdout


def test_negative_control_repo_state_used_as_a_status(ws):
    ws.edit("A4", status="running")
    out = ws.run()
    assert out.returncode != 0
    assert "not a mathematical status" in out.stdout


def test_negative_control_repo_state_inside_an_evidentiary_structure(ws):
    """The claim graph must never acquire a code-state field."""
    os.makedirs(os.path.join(ws.repo_root, "claims"), exist_ok=True)
    with open(GRAPH, encoding="utf-8") as f:
        graph = json.load(f)
    graph["premises"]["OBL-H5-JETMOD"]["repo_state"] = "running"
    with open(os.path.join(ws.repo_root, "claims", "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f)
    out = ws.run()
    assert out.returncode != 0
    assert "must not contain repo_state" in out.stdout


def test_negative_control_repo_state_nested_in_a_lane(ws):
    lane = ws.lane("A1")
    lane["status_source"]["repo_state"] = "running"
    ws.write_lane("A1", lane)
    out = ws.run()
    assert out.returncode != 0
    assert "exactly once" in out.stdout


def test_negative_control_unknown_lane_field(ws):
    ws.edit("A1", verdict="PASS")
    out = ws.run()
    assert out.returncode != 0
    assert "unknown fields" in out.stdout


def test_negative_control_heading_drift(ws):
    lane = ws.lane("C")
    lane["source_section"]["heading"] = "## C. Something else entirely"
    ws.write_lane("C", lane)
    out = ws.run()
    assert out.returncode != 0
    assert "source_section.heading" in out.stdout


# --------------------------------------------------------------------------
# repo_state is a code state and changes no verdict
# --------------------------------------------------------------------------

def test_repo_state_running_changes_no_verdict(ws):
    """Declaring every lane 'running' must not move any status or any check."""
    before = {}
    for name in sorted(os.listdir(ws.lanes)):
        key = name[:-5]
        before[key] = ws.lane(key)["status"]
        ws.edit(key, repo_state="running")

    out = ws.run()
    assert out.returncode == 0, out.stdout

    after = {key: ws.lane(key)["status"] for key in before}
    assert after == before

    mutated = json.loads(subprocess.run(
        [sys.executable, DISPATCHER, "--lanes", ws.lanes, "--graph", ws.graph,
         "--manifest", ws.manifest, "--binding", ws.binding, "--json"],
        capture_output=True, text=True, check=True).stdout)
    original = json.loads(dispatcher("--json"))
    m = {r["key"]: r for r in mutated["lanes"]}
    o = {r["key"]: r for r in original["lanes"]}
    for key in o:
        assert m[key]["status"] == o[key]["status"]
        assert m[key]["blocked_claims"] == o[key]["blocked_claims"]
        assert m[key]["falsifier"] == o[key]["falsifier"]
    # ranks may move — that is scheduling — but nothing else may.
    assert mutated["ranking_carries_no_mathematical_authority"] is True


# --------------------------------------------------------------------------
# the dispatcher
# --------------------------------------------------------------------------

def test_dispatcher_always_disclaims_authority():
    for args in ([], ["--lane", "A5"], ["--json"], ["--lane", "A5", "--json"]):
        out = dispatcher(*args)
        assert "NO MATHEMATICAL AUTHORITY" in out.upper()
        assert "operator decision" in out


def test_dispatcher_ranking_rule_is_in_its_own_output():
    out = dispatcher()
    assert "RANKING RULE" in out
    for fragment in ("claims blocked", "falsifier", "inputs", "repo_state"):
        assert fragment in out


def test_dispatcher_blocked_counts_follow_the_claim_graph():
    data = json.loads(dispatcher("--json"))
    rows = {r["key"]: r for r in data["lanes"]}
    # A5 blocks D3-LEMMA-RN-UNIF, which the certified rung and both all-small-r
    # claims rest on, directly or through OBL-H5-REMOTE-THRESHOLD.
    assert set(rows["A5"]["blocked_claims"]) == {
        "D1-v2.2(1)", "D1-v2.2(2)", "D1-v2.3-DRAFT(2)"}
    assert rows["A2"]["blocked_claims"] == []
    assert rows["A5"]["rank"] < rows["A2"]["rank"]


def test_dispatcher_ranks_a_falsifiable_lane_above_one_that_is_not():
    data = json.loads(dispatcher("--json"))
    rows = {r["key"]: r for r in data["lanes"]}
    same_blocks = [r for r in data["lanes"]
                   if r["blocked_claim_count"] == rows["A1"]["blocked_claim_count"]]
    assert rows["A1"]["falsifier_defined"]
    for other in same_blocks:
        if not other["falsifier_defined"]:
            assert rows["A1"]["rank"] < other["rank"]


def test_dispatcher_reports_unbound_inputs_when_the_manifest_is_absent():
    data = json.loads(dispatcher("--json"))
    if os.path.exists(os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")):
        pytest.skip("carrier manifest now exists; binding is checked by lanes_check")
    for row in data["lanes"]:
        assert row["inputs"] == []
        assert row["inputs_bound_here"] is False


def test_unknown_lane_is_an_error():
    out = subprocess.run([sys.executable, DISPATCHER, "--lane", "NOPE"],
                         capture_output=True, text=True)
    assert out.returncode == 2


# --- the repo_state disclaimer --------------------------------------------
#
# `repo_state` is a code state. The sentence saying so is what stops it being
# read as a mathematical grade, and until 2026-09-21 the checker required only
# that `repo_state_note` be NON-EMPTY -- a note could be rewritten without the
# sentence and still pass. A third-party branch rewrote two lane notes in one
# change; the A1 rewrite kept the idea in new words, the A5 rewrite dropped it,
# and nothing here could tell. The required phrase is deliberately short so an
# honest rewording keeps it and only dropping the idea trips the check.

def test_every_lane_note_says_repo_state_is_not_a_status():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import lanes_check as L

    for name in sorted(os.listdir(LANES)):
        with open(os.path.join(LANES, name), encoding="utf-8") as f:
            lane = json.load(f)
        note = " ".join(str(lane.get("repo_state_note", "")).lower().split())
        assert L.NOT_A_STATUS in note, name


def test_negative_control_note_without_the_disclaimer(ws):
    """Drop the sentence from one lane; the checker must refuse."""
    path = os.path.join(ws.lanes, "A5.json")
    with open(path, encoding="utf-8") as f:
        lane = json.load(f)
    lane["repo_state_note"] = (
        "The generic partition ledger and local spatial adapter now exist. "
        "Both RN pieces remain OPEN."
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lane, f, ensure_ascii=False, indent=2)
    out = ws.run()
    assert out.returncode != 0, out.stdout
    assert "not a mathematical status" in out.stdout


def test_an_honest_rewording_still_passes(ws):
    """The control must not force one fixed sentence, only the idea."""
    path = os.path.join(ws.lanes, "A5.json")
    with open(path, encoding="utf-8") as f:
        lane = json.load(f)
    lane["repo_state_note"] = (
        "Some code exists. repo_state stays partial for the actual obligation, "
        "not a mathematical status. Both RN pieces remain OPEN."
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lane, f, ensure_ascii=False, indent=2)
    out = ws.run()
    assert out.returncode == 0, out.stdout
