"""Register invariants, run in CI.

* the JSON/CSV registers must be exactly what tools/registers_import.py produces
  from the committed source export (no hand edits that drift from the source);
* structural checks in tools/registers_check.py must pass modulo the documented
  findings allowlist (registers/KNOWN_FINDINGS.json);
* work_events is append-only: rows present in the previous commit must still be
  present, in the same order, at the same positions.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_registers_match_source_export():
    r = run("tools/registers_import.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_structural_checks_pass_modulo_known_findings():
    r = run("tools/registers_check.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_work_events_append_only():
    cur = json.load(open(os.path.join(ROOT, "registers", "json", "work_events.json"), encoding="utf-8"))
    prev = subprocess.run(["git", "show", "HEAD:registers/json/work_events.json"], cwd=ROOT,
                          capture_output=True, text=True)
    if prev.returncode != 0:
        return  # first commit of the register: nothing to compare against
    old = json.loads(prev.stdout)
    assert old["header"] == cur["header"], "work_events header changed"
    assert len(cur["rows"]) >= len(old["rows"]), "work_events lost rows"
    for i, row in enumerate(old["rows"]):
        assert cur["rows"][i] == row, f"work_events row {i} was rewritten (append-only violation)"
