"""Inventable JETMOD probes: honest REFUSED/EMPTY/ABSENT only; no discharge."""
from __future__ import annotations

import json
import os
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
}


def test_runner_writes_refused_receipts_only():
    before = {
        name: open(os.path.join(PROBES, name), "rb").read()
        for name in EXPECTED
        if os.path.isfile(os.path.join(PROBES, name))
    }
    result = subprocess.run(
        [sys.executable, RUNNER], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "discharges=false" in result.stdout
    for name, status in EXPECTED.items():
        path = os.path.join(PROBES, name)
        obj = json.load(open(path, encoding="utf-8"))
        assert obj["status"] == status
        assert obj["inventable_attempt_accepted"] is False
        assert obj["discharges_OBL_H5_JETMOD"] is False
        assert obj["lemma_closed"] is False
    # re-run is allowed; flags must stay false
    index = json.load(open(os.path.join(PROBES, "INVENTABLE_PROBES_INDEX.json"), encoding="utf-8"))
    assert index["discharges_OBL_H5_JETMOD"] is False
    assert index["lemma_closed"] is False
    assert index["OBL_H5_JETMOD"] == "OPEN"


def test_math_status_check_validates_inventable_probes():
    result = subprocess.run(
        [sys.executable, CHECKER], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "problems=0" in result.stdout


def test_negative_accepting_inventable_attempt_is_refused(tmp_path):
    # mutate a receipt to accept inventable attempt → checker must fail
    import shutil

    dest = tmp_path / "repo"
    # minimal: copy probes + docs/math_status + tools checker into temp layout
    # Use in-place mutation of a copy of probes next to a fake root is hard;
    # instead mutate real receipt temporarily is bad. Copy whole needed tree.
    shutil.copytree(os.path.join(ROOT, "docs"), dest / "docs")
    shutil.copytree(os.path.join(ROOT, "tools"), dest / "tools")
    path = dest / "docs" / "math_status_probes" / "inventable_phi_bridge_ABSENT_receipt.json"
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
