"""Inventable JETMOD probes: honest REFUSED/EMPTY/ABSENT only; no discharge."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "docs", "math_status_probes")
RUNNER = os.path.join(PROBES, "inventable_jetmod_probes.py")
CHECKER = os.path.join(ROOT, "tools", "math_status_check.py")

EXPECTED = {
    "inventable_interval_schur_ainv_REFUSED_receipt.json": "REFUSED_IA_STRADDLES",
    "inventable_eval_F_G12box_REFUSED_receipt.json": "REFUSED",
    "inventable_joint_ry_cancel_EMPTY_receipt.json": "EMPTY",
    "inventable_phi_bridge_ABSENT_receipt.json": "ABSENT",
    "inventable_24jet_roster_without_Drive_list_REFUSED_NOT_24JET_receipt.json": "REFUSED_NOT_24JET",
    "inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json": "REFUSED",
    "inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json": "REFUSED",
}

SHORTCUTS = {
    "inventable_24jet_roster_without_Drive_list_REFUSED_NOT_24JET_receipt.json",
    "inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json",
    "inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json",
}


def _probe_snapshot():
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in Path(PROBES).glob("*.json")
    }


def test_runner_writes_refused_receipts_only(tmp_path):
    before = _probe_snapshot()
    probes = tmp_path / "repo" / "docs" / "math_status_probes"
    probes.mkdir(parents=True)
    runner = probes / Path(RUNNER).name
    shutil.copy2(RUNNER, runner)
    # No old output is copied: success requires fresh receipts from the runner.
    assert not list(probes.glob("*.json"))
    try:
        result = subprocess.run(
            [sys.executable, str(runner)], cwd=probes.parent.parent,
            capture_output=True, text=True, check=False, timeout=60,
        )
    finally:
        assert _probe_snapshot() == before, "probe runner modified source receipts"
    assert result.returncode == 0, result.stdout + result.stderr
    assert "discharges=false" in result.stdout
    for name, status in EXPECTED.items():
        path = probes / name
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert obj["status"] == status
        assert obj["inventable_attempt_accepted"] is False
        assert obj["discharges_OBL_H5_JETMOD"] is False
        assert obj["lemma_closed"] is False
        assert obj["freeze"] is False
        if name in SHORTCUTS:
            assert obj["discharges_lemma"] is False
            assert obj["certified_C_H"] is False
            assert obj["works"] is False
        if "24jet" in name:
            assert obj["status"] == "REFUSED_NOT_24JET"
            assert obj["refused_not_24jet"] is True
            assert obj["partial_roster"] is False
            assert obj["roster_invented"] is False
            assert "jet_roster" not in obj
            assert "roster" not in obj
    index = json.loads((probes / "INVENTABLE_PROBES_INDEX.json").read_text(encoding="utf-8"))
    assert index["discharges_OBL_H5_JETMOD"] is False
    assert index["discharges_lemma"] is False
    assert index["lemma_closed"] is False
    assert index["certified_C_H"] is False
    assert index["freeze"] is False
    assert index["inventable_attempt_accepted"] is False
    assert index["OBL_H5_JETMOD"] == "OPEN"
    assert index["disposition"] == "OPEN_HOLD"
    assert index["pr12_action"] == "NONE_left_unmerged"
    walls = index["named_walls_only"]
    assert any("24-jet roster" in wall and "REFUSED_NOT_24JET" in wall for wall in walls)
    assert any("display residual" in wall for wall in walls)
    assert any("PR #12" in wall and "RUNG2/3" in wall for wall in walls)
    files = {row["file"] for row in index["receipts"]}
    assert SHORTCUTS <= files


def test_math_status_check_validates_inventable_probes():
    result = subprocess.run(
        [sys.executable, CHECKER], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "problems=0" in result.stdout


def test_negative_accepting_inventable_attempt_is_refused(tmp_path):
    # Mutate only a copied receipt; the published source is never rewritten.
    dest = tmp_path / "repo"
    shutil.copytree(os.path.join(ROOT, "docs"), dest / "docs")
    shutil.copytree(os.path.join(ROOT, "tools"), dest / "tools")
    path = dest / "docs" / "math_status_probes" / "inventable_phi_bridge_ABSENT_receipt.json"
    with open(path, encoding="utf-8") as handle:
        obj = json.load(handle)
    obj["inventable_attempt_accepted"] = True
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    result = subprocess.run(
        [sys.executable, str(dest / "tools" / "math_status_check.py")],
        cwd=dest,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "inventable_attempt_accepted must be false" in result.stdout
