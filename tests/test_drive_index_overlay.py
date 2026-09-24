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
    with open(DELTA, encoding="utf-8") as handle:
        rows = [json.loads(l) for l in handle if l.strip()]
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
