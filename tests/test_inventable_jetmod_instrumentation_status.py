"""Instrumentation STATUS vocabulary: PARTIAL / REFUSED_NOT_24JET; no discharge."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "docs", "math_status_probes")
RUNNER = os.path.join(PROBES, "inventable_jetmod_instrumentation_status.py")
CHECKER = os.path.join(ROOT, "tools", "math_status_check.py")

EXPECTED = {
    "inventable_first_band_proto_PARTIAL_receipt.json": "PARTIAL",
    "inventable_first_band_smoke_PARTIAL_receipt.json": "PARTIAL",
    "inventable_first_band_multi_gram_PARTIAL_receipt.json": "PARTIAL",
    "inventable_multi_jet_band_REFUSED_NOT_24JET_receipt.json": "REFUSED_NOT_24JET",
    "inventable_g12_ext_named_REFUSED_NOT_24JET_receipt.json": "REFUSED_NOT_24JET",
}


def _probe_snapshot():
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in Path(PROBES).glob("*.json")
    }


def test_runner_writes_partial_and_refused_not_24jet_only(tmp_path):
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
        assert obj["certified_C_H"] is False
        assert obj["prizes_solved"] == 0
    index = json.loads(
        (probes / "INVENTABLE_INSTRUMENTATION_STATUS_INDEX.json").read_text(encoding="utf-8")
    )
    assert index["discharges_OBL_H5_JETMOD"] is False
    assert index["lemma_closed"] is False
    assert index["certified_C_H"] is False
    assert index["prizes_solved"] == 0
    assert index["OBL_H5_JETMOD"] == "OPEN"


def test_math_status_check_validates_instrumentation_status():
    result = subprocess.run(
        [sys.executable, CHECKER], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "problems=0" in result.stdout


def test_negative_accepting_instrumentation_attempt_is_refused(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(os.path.join(ROOT, "docs"), dest / "docs")
    shutil.copytree(os.path.join(ROOT, "tools"), dest / "tools")
    path = (
        dest
        / "docs"
        / "math_status_probes"
        / "inventable_g12_ext_named_REFUSED_NOT_24JET_receipt.json"
    )
    obj = json.load(open(path, encoding="utf-8"))
    obj["inventable_attempt_accepted"] = True
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)
    result = subprocess.run(
        [sys.executable, str(dest / "tools" / "math_status_check.py")],
        cwd=dest,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "inventable_attempt_accepted must be false" in result.stdout


def test_negative_24jet_status_promotion_is_refused(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(os.path.join(ROOT, "docs"), dest / "docs")
    shutil.copytree(os.path.join(ROOT, "tools"), dest / "tools")
    path = (
        dest
        / "docs"
        / "math_status_probes"
        / "inventable_multi_jet_band_REFUSED_NOT_24JET_receipt.json"
    )
    obj = json.load(open(path, encoding="utf-8"))
    obj["status"] = "CERTIFIED_24JET"  # inventable promotion
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)
    result = subprocess.run(
        [sys.executable, str(dest / "tools" / "math_status_check.py")],
        cwd=dest,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "status must be REFUSED_NOT_24JET" in result.stdout
