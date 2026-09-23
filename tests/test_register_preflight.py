"""Source-bound preflight: observable blockers, no activation or scientific verdict."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import registers_preflight as PF
import registers_import as RI

R1 = "GP-REG-032_v1.2_export_2026-09-23_R1.xlsx"


def test_current_snapshot_passes_only_covered_interfaces():
    report = PF.preview(Path(RI.resolve_current_source()).name)
    assert report["compatible_for_covered_interfaces"] is True
    assert not any(report["blockers"].values())
    assert report["comparison"]["changed_cells"] == []
    assert report["protected_files_unchanged"] is True
    assert report["canonical_import_completed"] is False
    assert report["scientific_status_changed"] is False


def test_r1_has_exact_unresolved_interfaces_and_never_activates():
    before = Path(RI.SOURCE_MANIFEST).read_bytes()
    report = PF.preview(R1)
    assert report["proposed_identity"] == {"bytes": 1979318, "sha256":
        "6cec54cefa3ee8f9885dcfd763407a2e0274d2887d73bdc23ccc7793cd721469"}
    assert report["tabs"] == 44 and report["generated_files_checked"] == 88
    assert report["row_counts"]["review_queue"] == 40
    assert {k: len(v) for k, v in report["blockers"].items()} == {
        "transcription": 0, "register_structure": 3, "stale_known_findings": 0,
        "bound_observations": 0, "review_interface": 3, "frozen_interface": 11}
    assert report["frozen_status_counts"]["MATCH"] == 62
    assert report["frozen_status_counts"]["DRIVE_ID_NOT_IN_INVENTORY"] == 9
    assert report["frozen_status_counts"]["UNKNOWN_CLASS"] == 2
    assert report["frozen_status_counts"].get("MISMATCH", 0) == 0
    assert report["compatible_for_covered_interfaces"] is False
    assert report["canonical_import_completed"] is False
    assert report["scientific_status_changed"] is False
    assert report["protected_files_unchanged"] is True
    assert Path(RI.SOURCE_MANIFEST).read_bytes() == before
    assert Path(RI.resolve_current_source()).name.endswith("2026-09-18.xlsx")


@pytest.mark.parametrize("name", ["../outside.xlsx", "not-declared.xlsx", "SOURCES.json"])
def test_undeclared_or_unsafe_candidate_is_refused_before_preview(name):
    with pytest.raises(RI.SourceError):
        PF.preview(name)


def test_cli_exit_one_for_blockers_not_success(capsys):
    assert PF.main(["--source-name", R1]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["compatible_for_covered_interfaces"] is False


def test_cli_exit_two_for_source_failure(capsys):
    assert PF.main(["--source-name", "not-declared.xlsx"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["canonical_import_completed"] is False


def test_record_form_error_is_not_hidden(monkeypatch):
    monkeypatch.setattr(PF.RV, "check_all", lambda *args: ["deliberate invalid review form"])
    report = PF.preview(Path(RI.resolve_current_source()).name)
    assert report["blockers"]["review_interface"] == ["deliberate invalid review form"]
    assert report["compatible_for_covered_interfaces"] is False


def test_new_digest_mismatch_is_not_hidden(monkeypatch):
    original = PF.FZ.check
    def changed(*args):
        rows, problems = original(*args)
        rows.append({"status": "MISMATCH"})
        return rows, problems + ["deliberate whole-file digest mismatch"]
    monkeypatch.setattr(PF.FZ, "check", changed)
    report = PF.preview(Path(RI.resolve_current_source()).name)
    assert report["blockers"]["frozen_interface"] == ["deliberate whole-file digest mismatch"]
    assert report["frozen_status_counts"]["MISMATCH"] == 1
    assert report["compatible_for_covered_interfaces"] is False


def test_observation_drift_is_not_hidden(monkeypatch):
    monkeypatch.setattr(PF.RC, "check_observations", lambda *args: (["deliberate observation drift"], 45))
    report = PF.preview(Path(RI.resolve_current_source()).name)
    assert report["blockers"]["bound_observations"] == ["deliberate observation drift"]
    assert report["compatible_for_covered_interfaces"] is False
