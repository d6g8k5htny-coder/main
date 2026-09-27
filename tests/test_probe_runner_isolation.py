"""Regression for probe-test isolation; no mathematical certification or status change."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "test_inventable_jetmod_probes.py",
        "inventable_jetmod_probes.py",
        "test_runner_writes_refused_receipts_only",
        8,
    ),
    (
        "test_inventable_jetmod_instrumentation_status.py",
        "inventable_jetmod_instrumentation_status.py",
        "test_runner_writes_partial_and_refused_not_24jet_only",
        6,
    ),
]


def _load_case(tmp_path, monkeypatch, case, regress=False):
    test_name, runner_name, function_name, output_count = case
    source = tmp_path / "source" / "docs" / "math_status_probes"
    source.mkdir(parents=True)
    shutil.copy2(ROOT / "docs" / "math_status_probes" / runner_name, source / runner_name)
    (source / "sentinel.json").write_text('{"published": true}\n', encoding="utf-8")
    text = (ROOT / "tests" / test_name).read_text(encoding="utf-8")
    if regress:
        # Reintroduce the actual bug in a disposable copy, never in the checkout.
        needle = "[sys.executable, str(runner)]"
        assert text.count(needle) == 1
        text = text.replace(needle, "[sys.executable, RUNNER]", 1)
    path = tmp_path / test_name
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_probe_isolation_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROBES", str(source))
    monkeypatch.setattr(module, "RUNNER", str(source / runner_name))
    return module, source, function_name, output_count


def _snapshot(directory):
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.iterdir() if p.is_file()
    }


@pytest.mark.parametrize("case", CASES, ids=["probes", "instrumentation"])
def test_real_runner_test_preserves_source(tmp_path, monkeypatch, case):
    module, source, function_name, output_count = _load_case(tmp_path, monkeypatch, case)
    before = _snapshot(source)
    work = tmp_path / "work"
    getattr(module, function_name)(work)
    assert _snapshot(source) == before
    generated = work / "repo" / "docs" / "math_status_probes"
    assert len(list(generated.glob("*.json"))) == output_count


@pytest.mark.parametrize("case", CASES, ids=["probes", "instrumentation"])
def test_source_path_regression_is_detected(tmp_path, monkeypatch, case):
    module, source, function_name, _ = _load_case(tmp_path, monkeypatch, case, regress=True)
    before = _snapshot(source)
    with pytest.raises(AssertionError, match="probe runner modified source receipts"):
        getattr(module, function_name)(tmp_path / "work")
    assert _snapshot(source) != before  # the control really exercised a source write
