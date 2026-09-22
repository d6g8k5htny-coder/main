"""The live-path overlay over the inventory export, with negative controls.

`drive/inventory.jsonl` is an export and is never edited; Drive moves made after
the snapshot are recorded under `drive/deltas/<date>/PATH_CHANGES.jsonl` and
overlaid by `tools/drive_index.py`. These tests pin that the overlay is bound to
the export (every row's snapshot path is the inventory's path for that id), that
it fails closed on a row it cannot tie to the export, and that the 2026-09-18
delta is exactly the 238 rows the Drive session's own handoff counted.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import drive_index as DI  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "drive_index.py")
DELTA = os.path.join(ROOT, "drive", "deltas", "2026-09-18", "PATH_CHANGES.jsonl")


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def test_the_2026_09_18_delta_is_238_rows_bound_to_the_export():
    snap = {e["id"]: e for e in DI.load(overlay=False)}
    rows = [json.loads(l) for l in open(DELTA, encoding="utf-8") if l.strip()]
    assert len(rows) == 238
    for r in rows:
        assert r["id"] in snap
        assert r["path_2026_09_17"] == snap[r["id"]]["path"]
        assert r["path_2026_09_18"] != r["path_2026_09_17"]
        # identity is untouched by a move
        assert r["sha256"] == snap[r["id"]]["sha256"]


def test_overlay_changes_paths_only():
    snap = {e["id"]: e for e in DI.load(overlay=False)}
    live = DI.load(overlay=True)
    moved = [e for e in live if e.get("moved")]
    assert len(moved) == 238
    for e in live:
        s = snap[e["id"]]
        assert (e["sha256"], e["bytes"], e["title"], e["mimeType"]) == (s["sha256"], s["bytes"], s["title"], s["mimeType"])
        if e.get("moved"):
            assert e["path_snapshot"] == s["path"] and e["path"] != s["path"]
        else:
            assert e["path"] == s["path"]


def test_cli_shows_both_paths_for_a_moved_folder_and_the_export_with_snapshot():
    fid = "1cohBT2r5pMnlzQXWxkqWp89c8RezwYQ7"   # 2026-07 -> 2026/07_JULY
    out = run("id", fid)
    assert out.returncode == 0
    obj = json.loads(out.stdout)
    assert obj["moved"] == "2026-09-18" and obj["path"].endswith("/2026/07_JULY")
    assert obj["path_snapshot"].endswith("/2026-07")
    out = run("id", fid, "--snapshot")
    obj = json.loads(out.stdout)
    assert "moved" not in obj and obj["path"].endswith("/2026-07")
    # the new container name is findable live and absent from the export
    assert "10_ORIGIN_PRESERVED_REVIEW_COLLECTIONS" in run("find", "05_FROM_THEOREM_TRACKS").stdout
    assert "10_ORIGIN_PRESERVED" not in run("find", "05_FROM_THEOREM_TRACKS", "--snapshot").stdout


def test_stats_reports_the_overlay():
    assert "238 items moved or renamed" in run("stats").stdout


def test_control_a_row_naming_an_unknown_id_is_refused(tmp_path):
    d = tmp_path / "2026-12-31"; d.mkdir()
    f = d / "PATH_CHANGES.jsonl"
    f.write_text(json.dumps({"id": "NOT-A-DRIVE-ID", "path_2026_09_17": "a/b", "path_2026_12_31": "c/b"}) + "\n")
    with pytest.raises(ValueError):
        DI.load_path_changes([str(f)], {"some-other-id"})


def test_control_a_row_without_a_live_path_is_refused(tmp_path):
    d = tmp_path / "2026-12-31"; d.mkdir()
    f = d / "PATH_CHANGES.jsonl"
    f.write_text(json.dumps({"id": "x", "path_2026_09_17": "a/b"}) + "\n")
    with pytest.raises(ValueError):
        DI.load_path_changes([str(f)], {"x"})


def test_control_a_snapshot_path_that_disagrees_with_the_export_is_refused(tmp_path, monkeypatch):
    d = tmp_path / "2026-12-31"; d.mkdir()
    f = d / "PATH_CHANGES.jsonl"
    f.write_text(json.dumps({"id": "1cohBT2r5pMnlzQXWxkqWp89c8RezwYQ7", "path_2026_09_17": "wrong/old/path",
                             "path_2026_12_31": "new/path"}) + "\n")
    monkeypatch.setattr(DI, "path_change_files", lambda deltas_dir=None: [str(f)])
    with pytest.raises(ValueError):
        DI.load(overlay=True)


# --------------------------------------------------------------------------
# The deltas directory is an argument, not a module global bound at the call
# site, and the CLI has a failing exit code that something asserts.
#
# `load()` used to call `path_change_files()` with no argument, so `DELTAS` --
# resolved against the real repository at import time -- was fixed for every
# caller. CLAUDE.md records that exact shape as the bug that silently neutered
# every mutation test in `tools/claims_check.py`. Separately, the only
# returncode assertion in this file was `== 0`: nothing pinned that the tool
# can fail at all.
# --------------------------------------------------------------------------

def _tiny_inventory(tmp_path):
    rows = [{"id": "AAA", "path": "x/old.txt", "name": "old.txt", "mimeType": "text/plain"},
            {"id": "BBB", "path": "x/keep.txt", "name": "keep.txt", "mimeType": "text/plain"}]
    inv = tmp_path / "inventory.jsonl"
    inv.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(inv)


def _deltas(tmp_path, rows, date="2026-09-30"):
    d = tmp_path / "deltas" / date
    d.mkdir(parents=True)
    (d / "PATH_CHANGES.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(tmp_path / "deltas")


def test_load_takes_the_deltas_directory_as_an_argument(tmp_path):
    inv = _tiny_inventory(tmp_path)
    deltas = _deltas(tmp_path, [{"id": "AAA", "path_snapshot": "x/old.txt",
                                 "path_live": "y/new.txt"}])
    moved = {e["id"]: e for e in DI.load(inv, overlay=True, deltas_dir=deltas)}
    assert moved["AAA"]["path"] == "y/new.txt"
    assert moved["AAA"]["path_snapshot"] == "x/old.txt"
    assert moved["AAA"]["moved"] == "2026-09-30"
    assert "moved" not in moved["BBB"]


def test_the_argument_is_what_decides_and_not_the_module_global(tmp_path):
    """Pointing it at an empty directory must yield no overlay at all.

    If `DELTAS` were still bound at the call site this would silently apply the
    real repository's 2026-09-18 delta instead, and the assertion below would
    still pass by accident only because these ids are not in it. So the test
    also checks the positive direction above.
    """
    inv = _tiny_inventory(tmp_path)
    empty = tmp_path / "no_deltas"
    empty.mkdir()
    rows = DI.load(inv, overlay=True, deltas_dir=str(empty))
    assert all("moved" not in e for e in rows)
    assert {e["path"] for e in rows} == {"x/old.txt", "x/keep.txt"}


def test_a_delta_naming_an_unknown_id_still_fails_closed_through_the_argument(tmp_path):
    inv = _tiny_inventory(tmp_path)
    deltas = _deltas(tmp_path, [{"id": "ZZZ", "path_snapshot": "q", "path_live": "r"}])
    with pytest.raises(ValueError):
        DI.load(inv, overlay=True, deltas_dir=deltas)


def test_the_cli_threads_both_paths_through(tmp_path):
    inv = _tiny_inventory(tmp_path)
    deltas = _deltas(tmp_path, [{"id": "AAA", "path_snapshot": "x/old.txt",
                                 "path_live": "y/new.txt"}])
    out = run("id", "AAA", "--inventory", inv, "--deltas", deltas)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "y/new.txt" in out.stdout


def test_the_cli_snapshot_flag_refuses_the_overlay(tmp_path):
    inv = _tiny_inventory(tmp_path)
    deltas = _deltas(tmp_path, [{"id": "AAA", "path_snapshot": "x/old.txt",
                                 "path_live": "y/new.txt"}])
    out = run("id", "AAA", "--inventory", inv, "--deltas", deltas, "--snapshot")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "x/old.txt" in out.stdout and "y/new.txt" not in out.stdout


def test_the_cli_exits_nonzero_on_an_unknown_id():
    out = run("id", "NOT-AN-ID")
    assert out.returncode != 0, out.stdout


def test_the_cli_exits_nonzero_when_a_delta_cannot_be_tied_to_the_export(tmp_path):
    """The fail-closed path, asserted at the CLI rather than only in-process."""
    inv = _tiny_inventory(tmp_path)
    deltas = _deltas(tmp_path, [{"id": "ZZZ", "path_snapshot": "q", "path_live": "r"}])
    out = run("stats", "--inventory", inv, "--deltas", deltas)
    assert out.returncode != 0, out.stdout
    assert "not in the inventory" in (out.stdout + out.stderr)


def test_a_find_with_no_hits_is_not_an_error():
    """Pinned because it is a choice, not an oversight: an empty result is a result."""
    out = run("find", "zzzz-no-such-path-zzzz")
    assert out.returncode == 0, out.stdout
