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
