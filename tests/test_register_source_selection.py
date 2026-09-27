"""Manifest-bound import, immutable historical replay, and exact R1 source controls.

These tests establish export identity and transcription, never mathematics or
review acceptance. Explicit --source remains a standalone inspection override.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import registers_import as RI
import registers_check as RC

R1_NAME = "GP-REG-032_v1.2_export_2026-09-23_R1.xlsx"
R1 = ROOT / "registers/source" / R1_NAME
R1_SHA256 = "6cec54cefa3ee8f9885dcfd763407a2e0274d2887d73bdc23ccc7793cd721469"
ADDITIONS = {"start_here": 3, "review_queue": 15, "work_events": 131,
             "artifact_index": 11, "evidence_lineage": 12, "work_orders": 1,
             "frozen_objects": 11, "identity_drift_watch": 11}
EXPECTED_PHRASES = {
    34: ("RV-WITHDRAWAL-20260921-01", "OPEN / NONAUTHOR TECHNICAL REVIEW REQUIRED"),
    35: ("RV-ROLLOUT-R2-20260921-01", "OPEN / NONAUTHOR TECHNICAL REVIEW REQUIRED"),
    36: ("RV-H3-SOLVER-20260921-01", "OPEN / NONAUTHOR MATHEMATICAL AND SOFTWARE REVIEW REQUIRED"),
}


@pytest.fixture
def declared(tmp_path):
    data = b"identity-only fixture: not a workbook"
    source = tmp_path / "fixture.xlsx"
    source.write_bytes(data)
    manifest = {"drive_object": {"id": RI.DRIVE_OBJECT_ID},
                "current_import_source": source.name,
                "exports": [{"file": source.name, "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest(), "exact": False}]}
    path = tmp_path / "SOURCES.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, source, manifest


def test_manifest_chooses_exactly_its_declared_source(declared):
    path, source, _ = declared
    before = source.read_bytes(), path.read_bytes()
    assert RI.resolve_current_source(str(path)) == str(source)
    assert before == (source.read_bytes(), path.read_bytes())


@pytest.mark.parametrize("mutation", [
    "missing_selector", "path_traversal", "absolute_path", "windows_path", "nul_path",
    "wrong_suffix", "no_exports", "exports_not_list", "record_not_object", "absent_record",
    "duplicate_record", "wrong_object", "missing_object", "wrong_hash", "uppercase_hash",
    "wrong_size", "boolean_size", "string_size", "zero_size", "negative_size",
    "exact_true", "exact_missing", "exact_zero",
])
def test_invalid_manifest_is_refused_without_writes(declared, mutation):
    path, source, original = declared
    m = copy.deepcopy(original)
    e = m["exports"][0]
    if mutation == "missing_selector": m.pop("current_import_source")
    elif mutation == "path_traversal": m["current_import_source"] = "../fixture.xlsx"
    elif mutation == "absolute_path": m["current_import_source"] = str(source)
    elif mutation == "windows_path": m["current_import_source"] = r"C:\fixture.xlsx"
    elif mutation == "nul_path": m["current_import_source"] = "fixture\0.xlsx"
    elif mutation == "wrong_suffix": m["current_import_source"] = "fixture.json"
    elif mutation == "no_exports": m.pop("exports")
    elif mutation == "exports_not_list": m["exports"] = {}
    elif mutation == "record_not_object": m["exports"] = [1]
    elif mutation == "absent_record": e["file"] = "other.xlsx"
    elif mutation == "duplicate_record": m["exports"].append(dict(e))
    elif mutation == "wrong_object": m["drive_object"]["id"] = "different-native-sheet"
    elif mutation == "missing_object": m.pop("drive_object")
    elif mutation == "wrong_hash": e["sha256"] = "0" * 64
    elif mutation == "uppercase_hash": e["sha256"] = e["sha256"].upper()
    elif mutation == "wrong_size": e["bytes"] += 1
    elif mutation == "boolean_size": e["bytes"] = True
    elif mutation == "string_size": e["bytes"] = str(e["bytes"])
    elif mutation == "zero_size": e["bytes"] = 0
    elif mutation == "negative_size": e["bytes"] = -1
    elif mutation == "exact_true": e["exact"] = True
    elif mutation == "exact_missing": e.pop("exact")
    elif mutation == "exact_zero": e["exact"] = 0
    else: raise AssertionError(mutation)
    path.write_text(json.dumps(m), encoding="utf-8")
    before = source.read_bytes(), path.read_bytes()
    with pytest.raises(RI.SourceError):
        RI.resolve_current_source(str(path))
    assert before == (source.read_bytes(), path.read_bytes())


@pytest.mark.parametrize("text", ["[]", "null", "{", '{"x":1,"x":2}',
                                  '{"outer":{"sha256":"a","sha256":"b"}}'])
def test_malformed_and_duplicate_key_json_is_refused(tmp_path, text):
    path = tmp_path / "SOURCES.json"
    path.write_text(text)
    with pytest.raises(RI.SourceError):
        RI.resolve_current_source(str(path))


@pytest.mark.parametrize("mutation", ["missing_manifest", "missing_export", "symlink_export", "corrupt_export"])
def test_missing_or_changed_source_is_refused(declared, mutation):
    path, source, _ = declared
    if mutation == "missing_manifest": path.unlink()
    elif mutation == "missing_export": source.unlink()
    elif mutation == "symlink_export":
        original = source.with_name("original.xlsx")
        source.rename(original)
        source.symlink_to(original)
    else: source.write_bytes(b"corrupt")
    with pytest.raises(RI.SourceError):
        RI.resolve_current_source(str(path))


def test_default_resolution_happens_at_call_time(declared, monkeypatch):
    path, source, _ = declared
    monkeypatch.setattr(RI, "SOURCE_MANIFEST", str(path))
    assert RI.resolve_current_source() == str(source)
    path.unlink()
    with pytest.raises(RI.SourceError):
        RI.resolve_current_source()


def test_bad_manifest_prevents_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "SOURCE_MANIFEST", str(tmp_path / "missing.json"))
    j, c = tmp_path / "json", tmp_path / "csv"
    assert RI.main(["--out-json", str(j), "--out-csv", str(c)]) == 2
    assert not j.exists() and not c.exists()


def test_explicit_source_is_standalone_and_not_false_manifest_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "SOURCE_MANIFEST", str(tmp_path / "missing.json"))
    j, c = tmp_path / "json", tmp_path / "csv"
    args = ["--source", str(R1), "--out-json", str(j), "--out-csv", str(c)]
    assert RI.main(args) == 0
    assert RI.main(args + ["--check"]) == 0


def test_historical_diff_default_does_not_follow_current_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "SOURCE_MANIFEST", str(tmp_path / "missing.json"))
    out = tmp_path / "historical.json"
    assert RI.main(["--diff-exports", str(out)]) == 0
    assert out.read_bytes() == Path(RI.EXPORT_DIFF).read_bytes()


def test_r1_identity_preview_outputs_and_immutable_baseline(tmp_path):
    assert RI.resolve_current_source() == RI.SOURCE
    assert RI.resolve_declared_source(export_name=R1_NAME) == str(R1)
    assert len(R1.read_bytes()) == 1979318
    assert hashlib.sha256(R1.read_bytes()).hexdigest() == R1_SHA256
    old, current = RI.parse(RI.SOURCE), RI.parse(str(R1))
    assert len(current) == 44
    j, c = str(tmp_path / "json"), str(tmp_path / "csv")
    RI.write(current, j, c)
    assert RI.check(current, j, c) == []
    d = {"comparison": RI.export_diff(old, current)}
    assert d["comparison"]["summary"]["rows_added"] == ADDITIONS
    assert len(d["comparison"]["changed_cells"]) == 27
    assert d["comparison"]["rows_removed"] == {}
    assert d["comparison"]["header_changes"] == []
    assert d["comparison"]["status_word_changes"] == []
    before = next(t for t in old if t["tab"] == "work_events")
    after = next(t for t in current if t["tab"] == "work_events")
    assert after["rows"][:len(before["rows"])] == before["rows"]
    assert len(after["rows"]) == 166
    rq = next(t for t in current if t["tab"] == "review_queue")
    assert len(rq["rows"]) == 40 and len({r[0] for r in rq["rows"]}) == 40


def test_new_findings_remain_unresolved_not_new_allowed_statuses(tmp_path):
    current = RI.parse(str(R1))
    rq = next(t for t in current if t["tab"] == "review_queue")
    expected = set()
    for i, (key, phrase) in EXPECTED_PHRASES.items():
        assert rq["rows"][i][0] == key
        assert rq["rows"][i][rq["header"].index("Technical status")] == phrase
        assert phrase not in RC.R17_TECH_STATUS
        expected.add(f"review_queue: row {i} status {phrase!r} not in R17 set")
    j, c = str(tmp_path / "json"), str(tmp_path / "csv")
    RI.write(current, j, c)
    problems, _ = RC.check(j)
    known = RC.load_known(str(ROOT / "registers/KNOWN_FINDINGS.json"))
    assert set(problems) - set(known) == expected
    assert not (expected & set(known)), "preview must not silently allowlist its own blockers"
