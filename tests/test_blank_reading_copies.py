"""Negative controls for the blank-reading-copy rule in tools/verify_manifests.py.

A reading copy is a rendering of a document.  A blank one -- empty, whitespace
only, or nothing but a UTF-8 byte-order mark -- is either a real property of
the source or a failed fetch stored as though it were a rendering.  The checker
tells those apart from OUTSIDE the manifest: drive/inventory.jsonl, which is
exported source data this repository never edits, marks eight objects
EMPTY_NATIVE_BODY.  A blank export of one of those is the export the inventory
predicts.  A blank export of anything else is corroborated by nothing.

Every control runs the checker through its CLI against a synthetic root, so an
inventory path bound at import time cannot silently re-check the real
repository -- the defect CLAUDE.md records in tools/claims_check.py.  The first
control here is exactly that: it asserts a synthetic tree with NO inventory
refuses its blank file, which fails if the lookup reaches this repository's
inventory instead of the tree under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "verify_manifests.py")

BOM = b"\xef\xbb\xbf"
BLANK_SHA = "f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5"


def run(root):
    return subprocess.run([sys.executable, CHECKER, root],
                          capture_output=True, text=True, cwd=ROOT)


def build(tmp_path, *, payload, inventory_status, row=None):
    """A one-file mirror whose single row calls `payload` a stored reading copy."""
    mirror = tmp_path / "drive" / "mirrors" / "LANE"
    mirror.mkdir(parents=True)
    dest = mirror / "doc.export.txt"
    dest.write_bytes(payload)
    entry = {
        "id": "1SyntheticDriveIdAAAAAAAAAAAAAAAAAAAAAAAA",
        "title": "A synthetic native Doc",
        "mimeType": "application/vnd.google-apps.document",
        "drive_path": "LANE/A synthetic native Doc",
        "dest": "doc.export.txt",
        "bytes": len(payload),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "exact": False,
        "stored": True,
        "inventory_sha256": None,
    }
    entry.update(row or {})
    (mirror / "_MANIFEST.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    if inventory_status is not None:
        inv = tmp_path / "drive" / "inventory.jsonl"
        inv.write_text(json.dumps({
            "id": entry["id"],
            "title": entry["title"],
            "path": entry["drive_path"],
            "mimeType": entry["mimeType"],
            "bytes": 1024,
            "sha256": None,
            "access_status": inventory_status,
        }) + "\n", encoding="utf-8")
    return str(tmp_path)


def test_blank_with_no_inventory_in_the_tree_is_refused(tmp_path):
    """The control that catches an inventory bound at import time.

    There is no inventory in this tree at all, so nothing corroborates the
    blank file and the checker must refuse it.  If the lookup were resolved
    against the real repository, this synthetic id would be absent from its
    EMPTY_NATIVE_BODY set anyway -- so make the id one that IS in this
    repository's set, and the two outcomes separate.
    """
    root = build(tmp_path, payload=BOM, inventory_status=None,
                 row={"id": "1q7coeCo2lbh11MUu6dF4VJYVoQnIZix1tiY1kKY1rT4"})
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNCORROBORATED BLANK READING COPY" in out.stdout


def test_bom_only_export_is_not_treated_as_text(tmp_path):
    """bytes.strip() leaves a BOM standing; three bytes is still blank."""
    root = build(tmp_path, payload=BOM, inventory_status="DIRECT_NATIVE")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNCORROBORATED BLANK READING COPY" in out.stdout


def test_whitespace_only_export_is_refused(tmp_path):
    root = build(tmp_path, payload=b"\r\n   \t\n", inventory_status="DIRECT_NATIVE")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNCORROBORATED BLANK READING COPY" in out.stdout


def test_zero_byte_export_is_refused(tmp_path):
    root = build(tmp_path, payload=b"", inventory_status="DIRECT_NATIVE")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNCORROBORATED BLANK READING COPY" in out.stdout


def test_blank_export_the_inventory_predicts_is_accepted(tmp_path):
    """The positive case, so the control cannot pass by refusing everything."""
    root = build(tmp_path, payload=BOM, inventory_status="EMPTY_NATIVE_BODY")
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "BLANK READING COPY" not in out.stdout
    assert "verified=1" in out.stdout


def test_a_byte_exact_row_is_not_subject_to_the_rule(tmp_path):
    """The rule is about renderings.  A byte-exact row stands on its digest."""
    root = build(tmp_path, payload=BOM, inventory_status="DIRECT_NATIVE",
                 row={"exact": True})
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "BLANK READING COPY" not in out.stdout


def test_a_nonblank_export_passes(tmp_path):
    root = build(tmp_path, payload=BOM + b"TB-G2 ALGEBRA CAPSULE\n",
                 inventory_status="DIRECT_NATIVE")
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "verified=1" in out.stdout


def test_every_manifested_blank_reading_copy_is_corroborated_and_disclosed(tmp_path):
    """The blank reading copies this repository carries are the disclosed ones.

    Scoped to files a manifest actually calls a stored reading copy, which is
    what the checker governs -- a blank export sitting in the tree with no row
    is an unfinished port, caught before it is committed, not a claim this
    repository makes.  Each carried one must be marked EMPTY_NATIVE_BODY by the
    inventory AND say in its own note that the export is a byte-order mark and
    holds no payload.

    The bound is the inventory's eight EMPTY_NATIVE_BODY ids, not a count of
    today's files: porting another lane legitimately adds blank copies of those
    same eight ids, and a control that pinned the number would fail on correct
    work while catching nothing a weaker check would miss.  What must never
    grow is the set they are drawn from.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import verify_manifests as V

    declared = V.empty_native_body_ids(os.path.join(ROOT, "drive"))
    assert len(declared) == 8, len(declared)

    carried = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "drive")):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        if "_MANIFEST.jsonl" not in filenames:
            continue
        with open(os.path.join(dirpath, "_MANIFEST.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                dest = row.get("dest")
                if not dest or not row.get("stored") or row.get("exact") is not False:
                    continue
                path = os.path.normpath(os.path.join(dirpath, dest))
                if not os.path.exists(path) or not V.is_blank(path):
                    continue
                carried.append(row["id"])
                assert row["id"] in declared, row["id"]
                assert row.get("access_status") == "EMPTY_NATIVE_BODY", row
                assert "byte-order mark" in row.get("note", ""), row["id"]
    assert carried, "no blank reading copy found -- this control has gone vacuous"
    assert set(carried) <= declared, set(carried) - declared


# --- the not-stored marker convention -------------------------------------
#
# Found by a third party's checker, not by this repository's own.  A row that
# stores nothing must open its note with one of NOT_STORED_MARKERS; 1,012 rows
# did and one did not, and the one that did not was invisible here because it
# also had a null `dest` and fell through the branch that skips rows with no
# destination.  A structured `not_stored_reason` field does not help: nothing
# in this checker reads it.

def test_an_unmarked_not_stored_row_is_refused(tmp_path):
    root = build(tmp_path, payload=b"text", inventory_status="DIRECT_NATIVE",
                 row={"stored": False, "dest": None, "bytes": None,
                      "sha256": None, "not_stored_reason": "SOME_REASON",
                      "note": "Not held, and the refusal is the finding."})
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNMARKED NOT-STORED ROW" in out.stdout


def test_a_structured_reason_alone_does_not_satisfy_it(tmp_path):
    """The field nothing reads must not be able to stand in for the note."""
    root = build(tmp_path, payload=b"text", inventory_status="DIRECT_NATIVE",
                 row={"stored": False, "dest": None, "bytes": None,
                      "sha256": None, "not_stored_reason": "AMBIGUOUS_RENDERING_REFUSED",
                      "note": "no marker here at all"})
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "UNMARKED NOT-STORED ROW" in out.stdout


def test_each_established_marker_is_accepted(tmp_path):
    """The positive case, so the control cannot pass by refusing everything."""
    for i, marker in enumerate(("skipped", "failed", "tree-only")):
        d = tmp_path / str(i)
        d.mkdir()
        root = build(d, payload=b"text", inventory_status="DIRECT_NATIVE",
                     row={"stored": False, "dest": None, "bytes": None,
                          "sha256": None,
                          "note": f"{marker}: a reason in the established form"})
        out = run(root)
        assert out.returncode == 0, (marker, out.stdout)
        assert "UNMARKED NOT-STORED ROW" not in out.stdout


def test_the_repository_has_no_unmarked_not_stored_row():
    """The standing invariant, over the tree as it is."""
    import json as _json
    import glob as _glob
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import verify_manifests as V

    unmarked = []
    for m in _glob.glob(os.path.join(ROOT, "drive", "**", "_MANIFEST.jsonl"),
                        recursive=True):
        with open(m, encoding="utf-8") as handle:
            for n, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                if row.get("stored") is False and not str(
                        row.get("note", "")).startswith(V.NOT_STORED_MARKERS):
                    unmarked.append((m, n))
    assert not unmarked, unmarked
