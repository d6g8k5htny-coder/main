"""Negative controls for tools/consumers_check.py.

Every control runs the checker through its CLI against a synthetic root, so a
path bound at import time cannot silently re-check the good repository — the
defect these controls exist to catch, recorded in CLAUDE.md and found once
already in tools/claims_check.py.

The last group pins the committed map of the real repository: the classes are
re-derived from the live scan, so a consumer added or removed anywhere makes
the map stale rather than quietly wrong.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "consumers_check.py")
MAP = os.path.join(ROOT, "registers", "CONSUMERS.json")
JSON_DIR = os.path.join(ROOT, "registers", "json")


def run(*args):
    return subprocess.run([sys.executable, CHECKER, *args],
                          capture_output=True, text=True, cwd=ROOT)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def sandbox(tmp_path):
    """A five-tab root: one read by code, one quoted in prose, one unread,
    and two named only by a file that names four tabs (generic machinery)."""
    root = tmp_path / "repo"
    for tab in ("alpha", "beta", "gamma", "delta", "epsilon"):
        write(str(root / "registers" / "json" / f"{tab}.json"),
              json.dumps({"tab": tab, "header": ["x"], "rows": []}))
    write(str(root / "tools" / "reader.py"), "PATH = 'alpha.json'\n")
    write(str(root / "docs" / "note.md"), "It quotes `beta` and mentions gamma as a plain word.\n")
    write(str(root / "tools" / "importer.py"),
          "TABS = ['alpha', 'beta', 'delta', 'epsilon']\n")
    return root


def generate(root, threshold=4):
    out = run("--root", str(root), "--threshold", str(threshold), "--write")
    assert out.returncode == 0, out.stderr + out.stdout
    with open(root / "registers" / "CONSUMERS.json", encoding="utf-8") as handle:
        return json.load(handle)


def verify(root):
    return run("--root", str(root))


def rewrite(root, doc):
    with open(root / "registers" / "CONSUMERS.json", "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# the classes are derived from the scan, and the scan means what it says
# ---------------------------------------------------------------------------

def test_the_three_classes_are_derived_from_the_scan(sandbox):
    doc = generate(sandbox)
    assert doc["tabs"]["alpha"]["class"] == "MACHINE"
    assert doc["tabs"]["beta"]["class"] == "PROSE_ONLY"
    assert doc["tabs"]["gamma"]["class"] == "UNREAD"
    assert verify(sandbox).returncode == 0


def test_a_bare_word_in_prose_is_not_a_consumer(sandbox):
    """'gamma' appears in the note as an English word and must not count;
    otherwise 'definitions' and 'relations' would invent consumers."""
    doc = generate(sandbox)
    assert doc["tabs"]["gamma"]["prose"] == []
    assert doc["tabs"]["beta"]["prose"] == ["docs/note.md"]


def test_a_file_naming_most_tabs_is_generic_and_attributed_to_none(sandbox):
    doc = generate(sandbox)
    assert doc["generic_files"] == ["tools/importer.py"]
    assert doc["tabs"]["delta"]["class"] == "UNREAD"
    assert doc["tabs"]["epsilon"]["class"] == "UNREAD"
    assert "tools/importer.py" not in doc["tabs"]["alpha"]["machine"]


def test_the_map_is_excluded_from_its_own_scan(sandbox):
    """The map names every tab; counting it would make it generic machinery
    and hide every consumer behind it."""
    doc = generate(sandbox)
    assert "registers/CONSUMERS.json" not in doc["generic_files"]
    for entry in doc["tabs"].values():
        assert "registers/CONSUMERS.json" not in entry["machine"]
    assert verify(sandbox).returncode == 0


# ---------------------------------------------------------------------------
# negative controls: each mutation must be refused through the CLI
# ---------------------------------------------------------------------------

def test_control_the_unmutated_sandbox_passes(sandbox):
    generate(sandbox)
    out = verify(sandbox)
    assert out.returncode == 0 and "problems=0" in out.stdout


def test_control_a_dropped_tab_entry_is_refused(sandbox):
    doc = generate(sandbox)
    del doc["tabs"]["gamma"]
    doc["counts"]["tabs"] -= 1
    doc["counts"]["UNREAD"] -= 1
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "gamma: exported tab missing from the map" in out.stdout


def test_control_a_phantom_tab_entry_is_refused(sandbox):
    doc = generate(sandbox)
    doc["tabs"]["not_a_tab"] = {"class": "UNREAD", "machine": [], "prose": []}
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "not_a_tab: recorded in the map but not an exported tab" in out.stdout


def test_control_a_hand_typed_class_is_refused(sandbox):
    doc = generate(sandbox)
    doc["tabs"]["alpha"]["class"] = "UNREAD"
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1
    assert "alpha: class recorded 'UNREAD' but the scan derives 'MACHINE'" in out.stdout


def test_control_a_dropped_consumer_is_refused(sandbox):
    doc = generate(sandbox)
    doc["tabs"]["alpha"]["machine"] = []
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "alpha: machine consumers differ" in out.stdout


def test_control_an_invented_consumer_is_refused(sandbox):
    doc = generate(sandbox)
    doc["tabs"]["gamma"]["machine"] = ["tools/nonexistent.py"]
    doc["tabs"]["gamma"]["class"] = "MACHINE"
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1
    assert "gamma: recorded consumer 'tools/nonexistent.py' does not exist" in out.stdout


def test_control_wrong_counts_are_refused(sandbox):
    doc = generate(sandbox)
    doc["counts"]["UNREAD"] = 0
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "counts differ" in out.stdout


def test_control_a_hidden_generic_file_is_refused(sandbox):
    doc = generate(sandbox)
    doc["generic_files"] = []
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "generic_files differ" in out.stdout


def test_control_a_new_exported_tab_makes_the_map_stale(sandbox):
    generate(sandbox)
    write(str(sandbox / "registers" / "json" / "zeta.json"),
          json.dumps({"tab": "zeta", "header": ["x"], "rows": []}))
    out = verify(sandbox)
    assert out.returncode == 1 and "zeta: exported tab missing from the map" in out.stdout


def test_control_a_new_consumer_makes_the_map_stale(sandbox):
    generate(sandbox)
    write(str(sandbox / "tools" / "late.py"), "P = 'gamma.json'\n")
    out = verify(sandbox)
    assert out.returncode == 1
    assert "gamma: machine consumers differ" in out.stdout and "gamma: class recorded 'UNREAD'" in out.stdout


def test_control_an_empty_does_not_establish_is_refused(sandbox):
    doc = generate(sandbox)
    doc["does_not_establish"] = "   "
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "does_not_establish is empty" in out.stdout


@pytest.mark.parametrize("mutation", [
    {"schema": "wrong"},
    {"generic_threshold": "30"},
    {"tabs": {}},
])
def test_control_a_malformed_map_is_refused(sandbox, mutation):
    doc = generate(sandbox)
    doc.update(mutation)
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "consumers_check:" in out.stdout


def test_control_a_missing_key_is_refused(sandbox):
    doc = generate(sandbox)
    del doc["counts"]
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "missing 'counts'" in out.stdout


# ---------------------------------------------------------------------------
# the committed map of this repository
# ---------------------------------------------------------------------------

def test_the_committed_map_is_current():
    out = run()
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


def test_the_committed_map_states_what_it_does_not_establish():
    with open(MAP, encoding="utf-8") as handle:
        doc = json.load(handle)
    text = doc["does_not_establish"]
    for phrase in ("reads, checks or grades", "No class is a status", "not defective"):
        assert phrase in text, phrase
    assert "coverage of the export" in text


def test_every_exported_tab_is_covered_exactly_once():
    with open(MAP, encoding="utf-8") as handle:
        doc = json.load(handle)
    exported = sorted(f[:-5] for f in os.listdir(JSON_DIR) if f.endswith(".json"))
    assert sorted(doc["tabs"]) == exported
    assert doc["counts"]["tabs"] == len(exported) == 44


def test_the_recorded_classes_agree_with_a_fresh_scan():
    sys.path.insert(0, ROOT)
    try:
        from tools import consumers_check as cc
    finally:
        sys.path.pop(0)
    with open(MAP, encoding="utf-8") as handle:
        doc = json.load(handle)
    live = cc.scan(ROOT, JSON_DIR, doc["generic_threshold"],
                   os.path.relpath(MAP, ROOT))
    assert live["counts"] == doc["counts"]
    assert live["generic_files"] == doc["generic_files"]
    assert {t: e["class"] for t, e in live["tabs"].items()} == \
           {t: e["class"] for t, e in doc["tabs"].items()}


def test_some_tabs_are_read_by_nothing_and_the_map_says_so():
    """The point of the map. If this ever reaches zero the repository reads
    every tab, and the assertion should be updated deliberately, not silently."""
    with open(MAP, encoding="utf-8") as handle:
        doc = json.load(handle)
    unread = sorted(t for t, e in doc["tabs"].items() if e["class"] == "UNREAD")
    assert unread and doc["counts"]["UNREAD"] == len(unread)
    for tab in unread:
        assert doc["tabs"][tab]["machine"] == [] and doc["tabs"][tab]["prose"] == []


def test_no_recorded_consumer_is_an_export_or_missing():
    with open(MAP, encoding="utf-8") as handle:
        doc = json.load(handle)
    for tab, entry in doc["tabs"].items():
        for rel in entry["machine"] + entry["prose"]:
            assert os.path.isfile(os.path.join(ROOT, rel)), (tab, rel)
            assert not rel.startswith(("registers/source/", "registers/json/", "registers/csv/"))


@pytest.mark.parametrize("excluded", ["drive", "legacy", "quarantine", "recovery"])
def test_excluded_source_roots_are_not_read(sandbox, monkeypatch, excluded):
    import importlib.util
    spec = importlib.util.spec_from_file_location("consumer_policy_control", CHECKER)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    write(str(sandbox / excluded / "nested" / "body.md"), "`gamma` gamma.json")
    original = cc._read
    def guarded(root, rel):
        assert rel.split(os.sep)[0] not in cc.EXCLUDED_ROOTS, rel
        return original(root, rel)
    monkeypatch.setattr(cc, "_read", guarded)
    live = cc.scan(str(sandbox), str(sandbox / "registers" / "json"), 4)
    assert live["tabs"]["gamma"] == {"class": "UNREAD", "machine": [], "prose": []}
    doc = generate(sandbox)
    assert doc["scan_scope"]["excluded_roots"] == list(cc.EXCLUDED_ROOTS)
    assert verify(sandbox).returncode == 0


def test_an_allowed_path_cannot_alias_an_excluded_body(sandbox, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("consumer_alias_control", CHECKER)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    target = sandbox / "legacy" / "body.md"
    write(str(target), "`gamma` gamma.json")
    (sandbox / "docs" / "alias.md").symlink_to(target)
    (sandbox / "docs" / "alias_dir").symlink_to(target.parent, target_is_directory=True)
    original = cc._read
    def guarded(root, rel):
        assert "alias" not in rel, rel
        return original(root, rel)
    monkeypatch.setattr(cc, "_read", guarded)
    live = cc.scan(str(sandbox), str(sandbox / "registers" / "json"), 4)
    assert live["tabs"]["gamma"]["class"] == "UNREAD"


def test_a_changed_scope_is_refused(sandbox):
    doc = generate(sandbox)
    doc["scan_scope"]["excluded_roots"] = []
    rewrite(sandbox, doc)
    out = verify(sandbox)
    assert out.returncode == 1 and "scan_scope differs" in out.stdout
