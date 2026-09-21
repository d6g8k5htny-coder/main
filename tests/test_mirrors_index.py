"""Negative controls for tools/mirrors_index_check.py.

Every control runs the checker through its CLI against a synthetic root, so no
path bound at import time can silently re-check the good repository -- the
defect CLAUDE.md records, found once already in tools/claims_check.py.

The index exists because prose counts rot: drive/README.md said "Mirrored so
far" and named six lanes while the tree held twenty-one, and three lane
verifications found the same shape inside lane READMEs.  These controls make a
rotted count fail the build instead of a reader.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "mirrors_index_check.py")
INDEX = os.path.join(ROOT, "drive", "MIRRORS.md")
MIRRORS = os.path.join(ROOT, "drive", "mirrors")


def run(*args):
    return subprocess.run([sys.executable, CHECKER, *args],
                          capture_output=True, text=True, cwd=ROOT)


def rows(root, lane, records):
    d = os.path.join(str(root), "drive", "mirrors", lane)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "_MANIFEST.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.fixture
def tree(tmp_path):
    """Two lanes: one with a stored exact row, a stored reading copy and a
    tree-only row; one with a single tree-only row and no bytes at all."""
    root = tmp_path / "repo"
    rows(root, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True, "stored": True},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False, "stored": True},
        {"id": "a3", "title": "c.bin", "stored": False, "note": "tree-only index row"},
    ])
    rows(root, "BETA_LANE", [
        {"id": "b1", "title": "d.md", "stored": False, "note": "tree-only index row"},
    ])
    # Drive paths on the manifest rows, and an inventory to measure coverage
    # against: five items in one Drive lane, of which two are held, one is
    # indexed tree-only and two are in neither.
    rows(root, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False, "note": "tree-only index row",
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
    ])
    rows(root, "BETA_LANE", [
        {"id": "b1", "title": "d.md", "stored": False, "note": "tree-only index row",
         "drive_path": "02_OTHER_LANE/d.md"},
    ])
    # The inventory declares a digest for a1 and a3, which is what lets a manifest
    # row claim exact: true against it. a2 is a native-Doc export: no digest exists
    # for it anywhere, so its row must be exact: false.
    inv = os.path.join(str(root), "drive", "inventory.jsonl")
    with open(inv, "w", encoding="utf-8") as handle:
        for rid, path, digest in (
                ("a1", "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md", "0" * 64),
                ("a2", "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt", None),
                ("a3", "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin", "3" * 64),
                ("z1", "01_ACTIVE_RESEARCH_PACKAGES/L1/unheld_one.md", "5" * 64),
                ("z2", "01_ACTIVE_RESEARCH_PACKAGES/L1/unheld_two.md", "6" * 64),
                ("b1", "02_OTHER_LANE/d.md", None)):
            handle.write(json.dumps({"id": rid, "path": path, "sha256": digest}) + "\n")
    out = run("--root", str(root), "--write")
    assert out.returncode == 0, out.stdout + out.stderr
    return root


def check(root):
    return run("--root", str(root))


def index_of(root):
    return os.path.join(str(root), "drive", "MIRRORS.md")


def read(root):
    with open(index_of(root), encoding="utf-8") as handle:
        return handle.read()


def overwrite(root, text):
    with open(index_of(root), "w", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# the generated index is right, and the check is not vacuous
# ---------------------------------------------------------------------------

def test_control_the_generated_index_passes(tree):
    out = check(tree)
    assert out.returncode == 0 and "problems=0" in out.stdout


def test_the_counts_are_derived_from_the_manifest_rows(tree):
    text = read(tree)
    assert "| `ALPHA_LANE` | 1 | 2 | 1 | 1 | 350 |" in text
    assert "| `BETA_LANE` | 1 | 0 | 0 | 1 | 0 |" in text
    assert "| **2 lanes** | **2** | **2** | **1** | **2** | **350** |" in text


# ---------------------------------------------------------------------------
# negative controls: every drift must be refused through the CLI
# ---------------------------------------------------------------------------

def test_control_a_stored_count_typed_by_hand_is_refused(tree):
    overwrite(tree, read(tree).replace("| `ALPHA_LANE` | 1 | 2 | 1 | 1 | 350 |",
                                       "| `ALPHA_LANE` | 1 | 9 | 1 | 1 | 350 |"))
    out = check(tree)
    assert out.returncode == 1 and "the index does not" in out.stdout


def test_control_a_new_lane_not_in_the_index_is_refused(tree):
    """The exact defect: a port adds a lane and the index still names the old set."""
    rows(tree, "GAMMA_LANE", [{"id": "g1", "title": "e.md", "sha256": "2" * 64,
                               "bytes": 7, "exact": True, "stored": True}])
    out = check(tree)
    assert out.returncode == 1 and "GAMMA_LANE" in out.stdout and "the index does not" in out.stdout


def test_control_a_lane_removed_from_the_tree_is_refused(tree):
    os.remove(os.path.join(str(tree), "drive", "mirrors", "BETA_LANE", "_MANIFEST.jsonl"))
    os.rmdir(os.path.join(str(tree), "drive", "mirrors", "BETA_LANE"))
    out = check(tree)
    assert out.returncode == 1 and "BETA_LANE" in out.stdout and "the tree does not" in out.stdout


def test_control_a_file_added_to_a_lane_is_refused(tree):
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True, "stored": True},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False, "stored": True},
        {"id": "a3", "title": "c.bin", "stored": False},
        {"id": "a4", "title": "new.md", "sha256": "3" * 64, "bytes": 11, "exact": True, "stored": True},
    ])
    out = check(tree)
    assert out.returncode == 1 and "problems=" in out.stdout and "problems=0" not in out.stdout


def test_control_a_reading_copy_relabelled_exact_is_refused(tree):
    """exact:false is a reading copy of a native Doc. Calling it exact changes
    what the index claims about the digest, so the index must not drift."""
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True, "stored": True},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": True, "stored": True},
        {"id": "a3", "title": "c.bin", "stored": False},
    ])
    out = check(tree)
    assert out.returncode == 1


def test_control_a_tree_only_row_turned_stored_is_refused(tree):
    rows(tree, "BETA_LANE", [{"id": "b1", "title": "d.md", "sha256": "4" * 64,
                              "bytes": 9, "exact": True, "stored": True}])
    out = check(tree)
    assert out.returncode == 1


def test_control_a_missing_index_is_refused(tree):
    os.remove(index_of(tree))
    out = check(tree)
    assert out.returncode == 1 and "is missing; run --write" in out.stdout


def test_control_hand_edited_prose_is_refused(tree):
    """The index is generated whole. Editing its prose is refused with the
    remedy named, so nobody quietly maintains two versions of the caveats."""
    overwrite(tree, read(tree).replace("A mirror is a copy.", "A mirror is evidence."))
    out = check(tree)
    assert out.returncode == 1 and "regenerate it with --write" in out.stdout


def test_control_write_repairs_every_drift(tree):
    overwrite(tree, "garbage\n")
    assert check(tree).returncode == 1
    assert run("--root", str(tree), "--write").returncode == 0
    assert check(tree).returncode == 0


# ---------------------------------------------------------------------------
# the real repository
# ---------------------------------------------------------------------------

def test_the_committed_index_is_current():
    out = run()
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


def test_the_real_index_covers_every_lane_directory():
    with open(INDEX, encoding="utf-8") as handle:
        text = handle.read()
    lanes = sorted(n for n in os.listdir(MIRRORS) if os.path.isdir(os.path.join(MIRRORS, n)))
    assert lanes, MIRRORS
    for lane in lanes:
        assert f"| `{lane}` |" in text, lane
    assert f"| **{len(lanes)} lanes** |" in text


def test_every_lane_has_a_root_readme():
    """A lane's root README is where a reader learns what the lane holds and what
    holding it does not establish. 14_COORDINATION_AUTOMATION_SPINE was the one
    lane without one until 2026-09-20, with its README a level down."""
    missing = [name for name in sorted(os.listdir(MIRRORS))
               if os.path.isdir(os.path.join(MIRRORS, name))
               and not os.path.isfile(os.path.join(MIRRORS, name, "README.md"))]
    assert missing == [], missing


def test_every_lane_readme_says_what_it_does_not_establish():
    """The most load-bearing field in this repository, per CLAUDE.md rule 2."""
    thin = []
    for name in sorted(os.listdir(MIRRORS)):
        path = os.path.join(MIRRORS, name, "README.md")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read().lower()
        if "does not establish" not in text and "not review" not in text:
            thin.append(name)
    assert thin == [], thin


def test_the_index_states_that_a_mirror_is_not_evidence():
    with open(INDEX, encoding="utf-8") as handle:
        text = handle.read()
    assert "not review, replay, endorsement or promotion" in text
    assert "moves no status" in text
    assert "may be cited as evidence" in text


# ---------------------------------------------------------------------------
# the coverage table: how much of the Drive this repository actually holds
# ---------------------------------------------------------------------------

def test_the_coverage_table_is_derived_from_the_manifests_drive_paths(tree):
    text = read(tree)
    # L1 has five inventory items: a1 and a2 held, a3 indexed, z1 and z2 neither.
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 2 | 1 | 2 |" in text
    assert "| `02_OTHER_LANE` | 1 | 0 | 1 | 0 |" in text
    assert "| **2 Drive lanes** | **6** | **2** | **2** | **2** |" in text


def test_control_a_held_count_typed_by_hand_is_refused(tree):
    overwrite(tree, read(tree).replace("| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 2 | 1 | 2 |",
                                       "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 5 | 0 | 0 |"))
    out = check(tree)
    assert out.returncode == 1 and "the index does not" in out.stdout


def test_control_storing_a_file_moves_it_from_neither_to_held(tree):
    """The table has to react to a port, or it measures nothing."""
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
        {"id": "z1", "title": "unheld_one.md", "sha256": "5" * 64, "bytes": 5, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/unheld_one.md"},
    ])
    assert check(tree).returncode == 1
    assert run("--root", str(tree), "--write").returncode == 0
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 3 | 1 | 1 |" in read(tree)


def test_a_post_snapshot_object_counts_in_no_coverage_column(tree):
    """An id with no inventory row is stored, and is not Drive coverage."""
    rows(tree, "BETA_LANE", [
        {"id": "b1", "title": "d.md", "stored": False, "drive_path": "02_OTHER_LANE/d.md"},
        {"id": "new1", "title": "after_the_snapshot.json", "sha256": "6" * 64, "bytes": 12,
         "exact": False, "stored": True, "drive_path": "02_OTHER_LANE/after_the_snapshot.json"},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    text = read(tree)
    assert "| `02_OTHER_LANE` | 1 | 0 | 1 | 0 |" in text          # coverage unchanged
    assert "| `BETA_LANE` | 1 | 1 | 0 | 1 | 12 |" in text          # lane table sees the bytes


def test_a_repeated_drive_id_in_one_manifest_is_reported(tree):
    """A lane that recorded an object tree-only under an earlier store-size limit,
    and later added a stored row beside it rather than rewriting the first, has
    two rows for one id. That is the repository not editing a record in place, and
    it is reported rather than failed so a second, unexplained case shows up."""
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
        {"id": "a3", "title": "c.bin", "sha256": "3" * 64, "bytes": 42, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin",
         "note": "Added later; the tree-only row above stands as written."},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    out = check(tree)
    assert out.returncode == 0, out.stdout
    assert "repeated_ids=1" in out.stdout
    assert "is carried by 2 rows" in out.stdout


def test_no_repeated_id_reports_zero(tree):
    out = check(tree)
    assert out.returncode == 0 and "repeated_ids=0" in out.stdout


def test_a_repeated_id_is_counted_once_in_coverage_not_twice(tree):
    """Two rows for one id must not inflate how much of the Drive is held."""
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    text = read(tree)
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 2 | 1 | 2 |" in text, text


# ---------------------------------------------------------------------------
# what `exact: true` is claiming, which differs between the two roots
# ---------------------------------------------------------------------------

def delta_rows(root, folder, records):
    d = os.path.join(str(root), "drive", "deltas", folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "_MANIFEST.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_exactness_claims_are_counted_by_what_backs_them(tree):
    out = check(tree)
    assert out.returncode == 0, out.stdout
    # a1 and a3 carry exact:true in the fixture's mirror lane and both have an
    # inventory row; nothing is post-snapshot yet.
    assert "exact_backed_by_inventory=1" in out.stdout, out.stdout
    assert "exact_post_snapshot=0" in out.stdout, out.stdout


def test_control_a_mirror_row_exact_against_nothing_is_refused(tree):
    """The claim `exact: true` means the bytes hash to the digest the inventory
    declares. An id the inventory has no digest for cannot back that claim."""
    rows(tree, "GAMMA_LANE", [
        {"id": "g1", "title": "e.md", "sha256": "2" * 64, "bytes": 7, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/e.md"},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    out = check(tree)
    assert out.returncode == 1
    assert "the inventory declares no digest for that id to be exact against" in out.stdout


def test_control_a_mirror_row_exact_against_a_different_digest_is_refused(tree):
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "9" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
    ])
    out = check(tree)
    assert out.returncode == 1 and "is not the inventory's" in out.stdout


def test_a_delta_row_exact_after_the_snapshot_passes_and_is_counted(tree):
    """A delta object was created after the snapshot the inventory is of, so no
    corpus digest for it exists. There `exact: true` means raw bytes rather than a
    native-Doc export, and the count says how many rows are in that position."""
    delta_rows(tree, "2026-09-18", [
        {"id": "new1", "title": "handoff.json", "sha256": "8" * 64, "bytes": 31,
         "exact": True, "stored": True,
         "drive_path": "07_MODEL_ACCESSIBILITY/handoff.json"},
    ])
    out = check(tree)
    assert out.returncode == 0, out.stdout
    assert "exact_post_snapshot=1" in out.stdout


def test_the_real_tree_has_every_mirror_exactness_claim_backed():
    """554 of them at the time this landed, and not one disagreeing.

    This asserts on the exactness lines specifically rather than on the exit
    code, because the index also fails while a lane is mid-port and that drift
    says nothing about whether an exactness claim is honest.
    """
    out = run()
    backed = int(out.stdout.split("exact_backed_by_inventory=")[1].split()[0])
    assert backed >= 500, out.stdout
    assert "the inventory declares no digest" not in out.stdout
    assert "is not the inventory's" not in out.stdout


def test_the_real_coverage_table_states_the_gap_rather_than_hiding_it():
    with open(INDEX, encoding="utf-8") as handle:
        text = handle.read()
    assert "| Drive lane | inventory items | held | indexed | neither |" in text
    assert "**4456**" in text, "the inventory total must be stated in full"
    # Lanes with no coverage at all must appear with their item count, not be omitted.
    for lane in ("01_ACTIVE_RESEARCH_PACKAGES/12_P1.1_LAW_SPECIFIC_Q_MACHINE",
                 "01_ACTIVE_RESEARCH_PACKAGES/10_AXIOMATIC_CORE_SPINE",
                 "01_ACTIVE_RESEARCH_PACKAGES/11_P0.2_ADJACENCY_TRANSIT_TREE"):
        assert f"| `{lane}` |" in text, lane


def test_the_summary_line_reports_the_coverage_numbers():
    out = run()
    assert "inventory=4456" in out.stdout and "held=" in out.stdout, out.stdout


# ---------------------------------------------------------------------------
# the remainder: what the gap is made of
#
# Until 2026-09-20 the index printed one number for every kind of gap at once.
# 2,678 items read as 2,678 pieces of undone work, when 472 of them are folders
# that no manifest row can ever hold and 1,186 are native Google Docs the corpus
# declares no digest for, so the most this repository can hold of one is a
# reading copy.  These controls keep the three kinds apart, and keep the one
# that does measure undone work honest.
# ---------------------------------------------------------------------------

def inventory_of(root):
    return os.path.join(str(root), "drive", "inventory.jsonl")


def add_inventory(root, records):
    with open(inventory_of(root), "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


FOLDER = "application/vnd.google-apps.folder"


def test_the_remainder_is_split_by_each_records_own_fields(tree):
    """A folder, a digest-less id and a digest-bearing id are three gaps, not one."""
    add_inventory(tree, [
        {"id": "f1", "path": "01_ACTIVE_RESEARCH_PACKAGES/L1/sub", "mimeType": FOLDER},
        {"id": "n1", "path": "01_ACTIVE_RESEARCH_PACKAGES/L1/native.doc",
         "mimeType": "application/vnd.google-apps.document", "sha256": None},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    text = read(tree)
    # seven items now: two held, one indexed, four in neither -- one folder, one
    # with no declared digest, and the two digest-bearing ones from the fixture.
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 7 | 2 | 1 | 4 |" in text
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 4 | 1 | 1 | 2 |" in text


def test_the_folder_test_is_the_mime_type_and_not_the_path(tree):
    """A file whose title looks like a folder is still a file."""
    add_inventory(tree, [
        {"id": "f2", "path": "01_ACTIVE_RESEARCH_PACKAGES/L1/looks_like_a_folder",
         "mimeType": "text/markdown", "sha256": "7" * 64},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 3 | 0 | 0 | 3 |" in read(tree)


def test_control_a_remainder_count_typed_by_hand_is_refused(tree):
    overwrite(tree, read(tree).replace(
        "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 2 | 0 | 0 | 2 |",
        "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 2 | 2 | 0 | 0 |"))
    out = check(tree)
    assert out.returncode == 1 and "the index does not" in out.stdout


def test_control_storing_a_digest_bearing_gap_empties_that_column(tree):
    """The column that measures undone work has to react to the work."""
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 2 | 0 | 0 | 2 |" in read(tree)
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
        {"id": "z1", "title": "unheld_one.md", "sha256": "5" * 64, "bytes": 5, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/unheld_one.md"},
        {"id": "z2", "title": "unheld_two.md", "sha256": "6" * 64, "bytes": 5, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/unheld_two.md"},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    text = read(tree)
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 5 | 4 | 1 | 0 |" in text
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` |" not in text.split("### What the remainder is")[1]


def test_a_reading_copy_takes_a_native_doc_out_of_the_no_digest_column(tree):
    """A reading copy is all this repository can ever hold of a native Doc, and
    holding one is still coverage.  It is counted as held, never as exact."""
    add_inventory(tree, [
        {"id": "n1", "path": "01_ACTIVE_RESEARCH_PACKAGES/L1/native.doc",
         "mimeType": "application/vnd.google-apps.document", "sha256": None},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 3 | 0 | 1 | 2 |" in read(tree)
    rows(tree, "ALPHA_LANE", [
        {"id": "a1", "title": "a.md", "sha256": "0" * 64, "bytes": 100, "exact": True,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md"},
        {"id": "a2", "title": "b.export.txt", "sha256": "1" * 64, "bytes": 250, "exact": False,
         "stored": True, "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/b.txt"},
        {"id": "a3", "title": "c.bin", "stored": False,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/c.bin"},
        {"id": "n1", "title": "native.export.txt", "sha256": "9" * 64, "bytes": 30,
         "exact": False, "stored": True,
         "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/L1/native.doc"},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    assert "| `01_ACTIVE_RESEARCH_PACKAGES/L1` | 2 | 0 | 0 | 2 |" in read(tree)


def test_a_lane_with_no_gap_is_left_out_of_the_remainder_table(tree):
    remainder = read(tree).split("### What the remainder is")[1]
    assert "02_OTHER_LANE" not in remainder, "a lane at neither=0 has nothing to report"
    assert "| **1 lanes with a gap** | **2** | **0** | **0** | **2** |" in remainder


def test_the_split_is_refused_when_an_id_is_in_the_inventory_twice(tree):
    """held/indexed are sets and items is a row count, so a duplicated id makes
    the two disagree.  The checker says so instead of printing a wrong column."""
    add_inventory(tree, [{"id": "a1", "path": "01_ACTIVE_RESEARCH_PACKAGES/L1/a.md",
                          "sha256": "0" * 64}])
    assert run("--root", str(tree), "--write").returncode == 0
    out = check(tree)
    assert out.returncode == 1
    assert "an id is in the inventory twice" in out.stdout, out.stdout


def test_control_a_row_two_tables_share_may_not_be_deleted_from_one(tree):
    """The index carries three tables and two can render the same row for one
    lane.  Deleting one of the pair was refused before this change too, but as a
    prose difference naming no row: the set of lines was unchanged, so only the
    whole-file compare noticed.  This pins the row being named and counted."""
    # A lane holding nothing, whose whole gap is digest-bearing, renders the
    # same five cells in both: "| X | n | 0 | 0 | n |".
    add_inventory(tree, [
        {"id": "q1", "path": "03_UNTOUCHED_LANE/q.md", "sha256": "a" * 64},
        {"id": "q2", "path": "03_UNTOUCHED_LANE/r.md", "sha256": "b" * 64},
    ])
    assert run("--root", str(tree), "--write").returncode == 0
    text = read(tree)
    twice = [l for l in set(text.splitlines()) if l.startswith("| `") and text.count(l + "\n") == 2]
    assert twice, "expected one row string rendered by both tables"
    overwrite(tree, text.replace(twice[0] + "\n", "", 1))
    out = check(tree)
    assert out.returncode == 1, out.stdout
    assert twice[0].strip() in out.stdout, "the deleted row must be named"
    assert "2 times and the index says it 1" in out.stdout, out.stdout
    assert "prose differs" not in out.stdout, "the prose is not what changed"


# --- the same invariants, on the tree this repository actually has -----------

def test_the_real_remainder_columns_sum_to_the_gap():
    out = run()
    assert "problems=0" in out.stdout, out.stdout
    numbers = dict(part.split("=") for part in out.stdout.strip().splitlines()[-1].split()
                   if "=" in part)
    assert (int(numbers["gap_folders"]) + int(numbers["gap_no_digest"])
            + int(numbers["gap_portable"])
            == int(numbers["inventory"]) - int(numbers["held"])
            - real_indexed()), out.stdout


def real_indexed():
    """Inventory ids a manifest indexes tree-only, counted from the tree."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import mirrors_index_check as tool
    rows_ = tool.coverage(ROOT, os.path.join("drive", "mirrors"),
                          os.path.join("drive", "deltas"),
                          os.path.join("drive", "inventory.jsonl"))
    return sum(counts["indexed"] for _lane, counts in rows_)


def test_every_real_no_digest_item_really_has_no_declared_digest():
    """Re-derived from the inventory rather than from the tool's own answer."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import mirrors_index_check as tool
    folders = no_digest = digest = 0
    with open(os.path.join(ROOT, "drive", "inventory.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = tool.remainder_kind(record)
            if kind == "folders":
                assert record["mimeType"] == FOLDER
                folders += 1
            elif kind == "no_digest":
                assert not record.get("sha256"), record["id"]
                no_digest += 1
            else:
                assert record["sha256"], record["id"]
                digest += 1
    assert folders and no_digest and digest


def test_the_summary_line_reports_the_three_kinds_of_gap():
    out = run()
    for key in ("gap_folders=", "gap_no_digest=", "gap_portable="):
        assert key in out.stdout, out.stdout


def test_the_index_says_a_native_doc_can_only_be_a_reading_copy():
    with open(INDEX, encoding="utf-8") as handle:
        text = handle.read()
    assert "### What the remainder is" in text
    assert "a text export, not the object" in text
    assert "| Drive lane | neither | folders | no digest declared | digest-bearing |" in text
