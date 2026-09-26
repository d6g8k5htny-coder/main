"""Negative controls for the frozen-object custody supplement.

The supplement exists because `drive/inventory.jsonl` derives from the
2026-09-17 accessibility publication and cannot contain an object frozen after
it. That makes it a convenience with a sharp edge: frozen_check's whole-file
comparison is worth something only because it sets two INDEPENDENT Drive-side
records of the same bytes against each other, and a supplement is a place where
a non-independent record could be introduced — deliberately, or by someone
copying the register's own expected digest because it was to hand.

So these tests are mostly about what the supplement must NOT be able to do:
turn a mismatch into a match, answer for an object the publication already
covers, or have its agreements counted as if the publication had made them.

What these tests do not establish: nothing here verifies that a supplement row
was honestly collected — no check can, which is exactly why a supplement-backed
agreement is reported under its own status. They test the checker's refusals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "frozen_check.py")
SUPPLEMENT = os.path.join(ROOT, "drive", "custody", "2026-09-24_frozen_objects.json")
INVENTORY = os.path.join(ROOT, "drive", "inventory.jsonl")

VERIFIED_ID = "1UCK6Lz30SmeiBXKgzk6J3y6YS05TtbFE"
VERIFIED_SHA = "f363629f77c8fec80838b15fee4df9972ee3397252f484c5a1602a7c21ed3e57"
VERIFIED_BYTES = 3251


def load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_frozen_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def supplement(tmp_path, rows, **over):
    doc = {
        "collected_by": "claude",
        "collection_method": "downloaded through the Drive API and hashed locally",
        "does_not_establish": "d" * 80,
        "rows": rows,
    }
    doc.update(over)
    path = tmp_path / "custody.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


ROW = {
    "id": VERIFIED_ID,
    "title": "RN_INNER_WEDGE_20260921_PROOF.md",
    "bytes": VERIFIED_BYTES,
    "sha256": VERIFIED_SHA,
    "verified_utc": "2026-09-23T23:12:00Z",
    "verified_by": "claude",
}


def run(*extra):
    proc = subprocess.run([sys.executable, TOOL, *extra], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


# --------------------------------------------- it changes nothing it should not

def test_the_committed_supplement_does_not_move_the_current_register():
    """The 09-18 register has no absent Drive IDs, so nothing consults it."""
    code, out = run()
    assert code == 0, out
    assert "match=62" in out
    assert "match_via_supplement=0" in out


def test_the_committed_supplement_is_well_formed():
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    rows, problems = mod.load_custody(SUPPLEMENT, inv)
    assert problems == [], problems
    assert VERIFIED_ID in rows


def test_no_supplement_row_is_already_covered_by_the_publication():
    """A supplement that shadowed the inventory would weaken a real check."""
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    with open(SUPPLEMENT, encoding="utf-8") as handle:
        doc = json.load(handle)
    for row in doc["rows"]:
        assert row["id"] not in inv, row["id"]


def test_the_pending_ids_are_absent_from_the_rows():
    """An unverified id must not be present at all; a present row gets answered."""
    with open(SUPPLEMENT, encoding="utf-8") as handle:
        doc = json.load(handle)
    present = {r["id"] for r in doc["rows"]}
    assert not (present & set(doc["_pending"])), "a pending id is also supplied"


# ------------------------------------------------- the circularity it must not hide

def test_a_supplement_cannot_turn_a_mismatch_into_a_match(tmp_path):
    """The control the whole design exists for.

    A supplement row whose digest disagrees with the register must produce
    MISMATCH, never a quiet MATCH. If this ever passes by reporting agreement,
    the supplement has become a way to launder an unverified object into a
    verified-looking one.
    """
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    bad = dict(ROW, sha256="0" * 64)
    rows, problems = mod.load_custody(supplement(tmp_path, [bad]), inv)
    assert problems == []
    assert rows[VERIFIED_ID]["sha256"] == "0" * 64, "the checker must see the bad digest"


def frozen_fixture(tmp_path, drive_id, sha, nbytes):
    """A one-row frozen_objects tab binding an object by whole-file digest."""
    doc = {
        "header": ["Object ID", "Drive ID", "Binding Class", "Expected Bytes",
                   "Expected SHA-256 / Identity", "Drift Status"],
        "rows": [["OBJ-UNDER-TEST", drive_id, "D — EXACT RAW FILE", str(nbytes), sha, "PASS"]],
    }
    path = tmp_path / "frozen_objects.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_a_supplement_answered_row_reports_as_supplement_not_as_match(tmp_path):
    """The load-bearing one: the status must actually change, not just exist.

    An earlier version of this control asserted only that the two constants
    differ and that the summary line mentions the counter. Folding
    MATCH_VIA_CUSTODY_SUPPLEMENT back into MATCH left both assertions true, so
    the control passed while the property it named was gone. It now runs the
    checker against a register whose Drive ID only the supplement can answer,
    and reads the counts.
    """
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen)
    assert code == 0, out
    assert "match_via_supplement=1" in out, out
    assert "match=0 " in out, "an inventory-backed match must not be claimed here"


def test_without_the_supplement_that_same_row_is_simply_absent(tmp_path):
    """Proves the previous test's match came from the supplement and nowhere else."""
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", str(tmp_path / "absent.json"))
    assert code == 1, out
    assert "is not in the inventory" in out


def test_a_supplement_digest_that_disagrees_reports_mismatch(tmp_path):
    """End to end: the supplement cannot launder a wrong digest into agreement."""
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, "a" * 64, VERIFIED_BYTES)
    code, out = run("--frozen", frozen)
    assert code == 1, out
    assert "mismatch=1" in out, out
    assert "match_via_supplement=0" in out, out


def test_the_two_statuses_are_distinct_constants():
    mod = load_module()
    assert mod.MATCH != mod.MATCH_VIA_CUSTODY_SUPPLEMENT


def test_the_module_says_why_the_two_are_kept_apart():
    mod = load_module()
    doc = mod.load_custody.__doc__ or ""
    assert "circular" in doc
    assert "independent" in doc.lower()


# ------------------------------------------------------------ malformed input

@pytest.mark.parametrize("row,reason", [
    (dict(ROW, sha256="deadbeef"), "64-hex"),
    (dict(ROW, bytes="3251"), "integer byte count"),
    ({k: v for k, v in ROW.items() if k != "verified_by"}, "who verified"),
    ({k: v for k, v in ROW.items() if k != "id"}, "no Drive id"),
])
def test_a_malformed_row_is_refused(tmp_path, row, reason):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    rows, problems = mod.load_custody(supplement(tmp_path, [row]), inv)
    assert rows == {}
    assert any(reason in p for p in problems), problems


def test_a_duplicate_row_is_refused(tmp_path):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    rows, problems = mod.load_custody(supplement(tmp_path, [ROW, ROW]), inv)
    assert any("appears twice" in p for p in problems), problems


def test_a_row_shadowing_the_inventory_is_refused(tmp_path):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    existing = next(iter(inv))
    rows, problems = mod.load_custody(supplement(tmp_path, [dict(ROW, id=existing)]), inv)
    assert rows == {}
    assert any("never overrides the accessibility publication" in p for p in problems), problems


@pytest.mark.parametrize("missing", ["collected_by", "collection_method", "does_not_establish"])
def test_a_supplement_without_its_provenance_is_refused(tmp_path, missing):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    _rows, problems = mod.load_custody(supplement(tmp_path, [ROW], **{missing: ""}), inv)
    assert any(missing in p for p in problems), problems


def test_an_unreadable_supplement_is_reported_rather_than_traced(tmp_path):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    path = tmp_path / "bad.json"
    path.write_text("not json at all", encoding="utf-8")
    rows, problems = mod.load_custody(str(path), inv)
    assert rows == {}
    assert any("unreadable" in p for p in problems), problems


def test_an_absent_supplement_is_simply_no_supplement(tmp_path):
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    rows, problems = mod.load_custody(str(tmp_path / "nope.json"), inv)
    assert rows == {} and problems == []


# ---------------------------------------------------------------------------
# Several supplements at once.
#
# More than one agent collects these objects, so the checker reads more than one
# supplement. That introduces two ways to be wrong that a single file could not
# be: silently choosing between two contradictory records, and counting one
# party's two files as two parties. Both get a control.
# ---------------------------------------------------------------------------

IMPORTER = os.path.join(ROOT, "tools", "custody_import.py")
IMPORTED_SUPPLEMENT = os.path.join(ROOT, "drive", "custody",
                                   "2026-09-23_frozen_objects_chatgpt.json")
PUBLICATION = os.path.join(ROOT, "drive", "custody", "imported",
                           "FROZEN_CUSTODY_MANIFEST_20260923.json")


def two_supplements(tmp_path, first_rows, second_rows, first_by="claude", second_by="chatgpt"):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for path, rows, who in ((a, first_rows, first_by), (b, second_rows, second_by)):
        path.write_text(json.dumps({
            "collected_by": who,
            "collection_method": "downloaded and hashed",
            "does_not_establish": "d" * 80,
            "rows": rows,
        }), encoding="utf-8")
    return str(a), str(b)


def row(**over):
    r = dict(ROW)
    r.update(over)
    return r


def test_two_supplements_agreeing_are_counted_as_two_collectors(tmp_path):
    a, b = two_supplements(tmp_path, [row(verified_by="claude")], [row(verified_by="chatgpt")])
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", a, "--custody", b)
    assert code == 0, out
    assert "match_via_supplement=1" in out, out
    assert "supplement_collectors_agreeing=1" in out, out


def test_one_collector_reporting_twice_is_not_two_collectors(tmp_path):
    """The distinction the count exists for.

    Two files that both say `claude` are one party's record copied, not two
    parties agreeing. If this ever reported 1 the counter would be measuring how
    many files exist, which is not evidence about anything.
    """
    a, b = two_supplements(tmp_path, [row(verified_by="claude")], [row(verified_by="claude")],
                           second_by="claude")
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", a, "--custody", b)
    assert code == 0, out
    assert "match_via_supplement=1" in out, out
    assert "supplement_collectors_agreeing=0" in out, out


def test_supplements_that_disagree_about_a_digest_answer_nothing(tmp_path):
    """A conflict must be refused, not resolved by file order.

    The failure this forbids is the tempting one: take the first file's value,
    or the last one's, and report a match. Either turns a contradiction between
    two records into an answer, which is the one thing a custody supplement must
    never be able to do.
    """
    a, b = two_supplements(tmp_path, [row()], [row(sha256="b" * 64, verified_by="chatgpt")])
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", a, "--custody", b)
    assert code == 1, out
    assert "custody supplements disagree about" in out, out
    assert "neither is used" in out, out
    assert "match_via_supplement=0" in out, out
    assert "is not in the inventory" in out, "the row must fall back to unanswered"


def test_a_conflict_is_refused_whichever_file_carries_the_right_value(tmp_path):
    """Order independence: swapping the two files changes nothing."""
    a, b = two_supplements(tmp_path, [row(sha256="b" * 64)], [row(verified_by="chatgpt")])
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", a, "--custody", b)
    assert code == 1, out
    assert "match_via_supplement=0" in out, out


def test_supplements_disagreeing_only_about_byte_count_also_answer_nothing(tmp_path):
    """Same digest, different length. Still two incompatible records."""
    a, b = two_supplements(tmp_path, [row()],
                           [row(bytes=VERIFIED_BYTES + 1, verified_by="chatgpt")])
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--custody", a, "--custody", b)
    assert code == 1, out
    assert "custody supplements disagree about" in out, out
    assert "match_via_supplement=0" in out, out


def test_a_conflicted_row_is_not_admitted_by_a_third_agreeing_file(tmp_path):
    """Two against one is still a contradiction, not a vote."""
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    a, b = two_supplements(tmp_path, [row()], [row(sha256="b" * 64, verified_by="chatgpt")])
    c = tmp_path / "c.json"
    c.write_text(json.dumps({
        "collected_by": "third",
        "collection_method": "downloaded and hashed",
        "does_not_establish": "d" * 80,
        "rows": [row(verified_by="third")],
    }), encoding="utf-8")
    rows, problems = mod.load_custody([a, b, str(c)], inv)
    assert VERIFIED_ID not in rows, "a refused id must stay refused"
    assert any("disagree about" in p for p in problems), problems


def test_a_problem_names_the_file_it_came_from(tmp_path):
    """With several files in play, an unattributed problem is not actionable."""
    mod = load_module()
    inv = mod.load_inventory(INVENTORY)
    bad = tmp_path / "second_one.json"
    bad.write_text(json.dumps({"collection_method": "m", "does_not_establish": "d" * 80,
                               "rows": []}), encoding="utf-8")
    rows, problems = mod.load_custody([SUPPLEMENT, str(bad)], inv)
    assert any("second_one.json" in p and "collected_by" in p for p in problems), problems


def test_no_custody_reads_none_of_them(tmp_path):
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    code, out = run("--frozen", frozen, "--no-custody")
    assert code == 1, out
    assert "match_via_supplement=0" in out, out
    assert "is not in the inventory" in out


def test_an_explicit_custody_flag_replaces_the_defaults_rather_than_adding(tmp_path):
    """The redirect lesson: a flag that is read but not honoured is worse than none.

    `tools/claims_check.py` once bound its graph path at import time, so every
    mutation test silently re-checked the good graph. Here the equivalent defect
    would be appending to DEFAULT_CUSTODY instead of replacing it: the committed
    supplements would keep answering, and a test that redirected the checker at
    an empty file would pass for the wrong reason.
    """
    frozen = frozen_fixture(tmp_path, VERIFIED_ID, VERIFIED_SHA, VERIFIED_BYTES)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"collected_by": "nobody", "collection_method": "none",
                                 "does_not_establish": "d" * 80, "rows": []}), encoding="utf-8")
    code, out = run("--frozen", frozen, "--custody", str(empty))
    assert code == 1, out
    assert "match_via_supplement=0" in out, "the committed supplements were still read"


# ---------------------------------------------------------------------------
# The imported publication.
# ---------------------------------------------------------------------------

def run_importer(*extra):
    proc = subprocess.run([sys.executable, IMPORTER, *extra], cwd=ROOT,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_imported_supplement_matches_its_retained_publication():
    code, out = run_importer("--check")
    assert code == 0, out
    assert "drift=0" in out and "problems=0" in out, out


def test_a_hand_edited_imported_supplement_is_caught(tmp_path):
    """The whole reason the file is generated: an edit to it must fail, not stick."""
    original = open(IMPORTED_SUPPLEMENT, encoding="utf-8").read()
    doc = json.loads(original)
    doc["rows"][0]["sha256"] = "c" * 64
    try:
        open(IMPORTED_SUPPLEMENT, "w", encoding="utf-8").write(json.dumps(doc, indent=1))
        code, out = run_importer("--check")
        assert code == 1, out
        assert "differs from" in out, out
        assert "drift=1" in out, out
    finally:
        open(IMPORTED_SUPPLEMENT, "w", encoding="utf-8").write(original)
    assert run_importer("--check")[0] == 0, "the fixture must leave the tree clean"


def test_a_publication_whose_bytes_drifted_is_refused(tmp_path):
    """The pin is what makes retention mean anything."""
    original = open(PUBLICATION, "rb").read()
    try:
        open(PUBLICATION, "wb").write(original + b"\n")
        code, out = run_importer("--check")
        assert code == 1, out
        assert "bytes, pinned" in out or "sha256" in out, out
    finally:
        open(PUBLICATION, "wb").write(original)
    assert run_importer("--check")[0] == 0, "the fixture must leave the tree clean"


def test_the_importer_does_not_relabel_the_collector():
    """An imported row must not be able to pass as this repository's own work."""
    doc = json.load(open(IMPORTED_SUPPLEMENT, encoding="utf-8"))
    assert doc["collected_by"] == "chatgpt"
    assert doc["rows"], "the publication carries rows"
    assert {r["verified_by"] for r in doc["rows"]} == {"chatgpt"}
    mine = json.load(open(SUPPLEMENT, encoding="utf-8"))
    assert {r["verified_by"] for r in mine["rows"]} == {"claude"}


def test_the_imported_supplement_carries_its_source_identity():
    doc = json.load(open(IMPORTED_SUPPLEMENT, encoding="utf-8"))
    src = doc["source_publication"]
    raw = open(PUBLICATION, "rb").read()
    import hashlib
    assert src["bytes"] == len(raw)
    assert src["sha256"] == hashlib.sha256(raw).hexdigest()


def test_the_imported_rows_agree_with_the_retained_register_export():
    """The point of the exercise: these digests are checkable without R3.

    R3 is a Drive object this repository does not hold. If the eleven imported
    digests could only be compared against R3, nothing here would be verifiable
    by a reader. They are compared against the R2 export that IS retained.
    """
    pytest.importorskip("zipfile")
    import subprocess as sp
    import tempfile
    r2 = os.path.join(ROOT, "registers", "source",
                      "GP-REG-032_v1.2_export_2026-09-23_R2.xlsx")
    if not os.path.exists(r2):
        pytest.skip("the R2 export is not retained in this checkout")
    with tempfile.TemporaryDirectory() as d:
        j, c = os.path.join(d, "j"), os.path.join(d, "c")
        proc = sp.run([sys.executable, os.path.join(ROOT, "tools", "registers_import.py"),
                       "--source", r2, "--out-json", j, "--out-csv", c],
                      cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        tab = json.load(open(os.path.join(j, "frozen_objects.json"), encoding="utf-8"))
    h = tab["header"]
    i_id, i_b, i_h = (h.index("Drive ID"), h.index("Expected Bytes"),
                      h.index("Expected SHA-256 / Identity"))
    expect = {r[i_id]: (r[i_b], r[i_h].lower()) for r in tab["rows"]}
    doc = json.load(open(IMPORTED_SUPPLEMENT, encoding="utf-8"))
    checked = 0
    for r in doc["rows"]:
        assert r["id"] in expect, f"{r['id']} is not a frozen_objects row of R2"
        want_bytes, want_sha = expect[r["id"]]
        assert str(r["bytes"]) == want_bytes, r["id"]
        assert r["sha256"] in want_sha, r["id"]
        checked += 1
    assert checked == 11, checked
