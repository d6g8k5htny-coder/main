"""Negative controls for tools/disclosure_check.py.

Each control builds its own git history in a temporary directory and drives the
checker through its CLI, so no path or revision bound at import time can make a
control re-check the good repository -- the defect CLAUDE.md records.

Building real history is the point: the checker's whole job is to compare the
working tree against a committed version, and a control that faked that
comparison would prove nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "disclosure_check.py")

ORIGINAL = """# Mirror of a synthetic lane

- `CL-TEST-001.md` carries *"CANONICAL IMPACT: NONE — no theorem is weakened by any item."*
- A second bullet quotes *"AUTHORITY: none, recorded by the synthetic operator"*.
"""

# The corrected file: the first quote is repaired and the old wording disclosed.
CORRECTED = """# Mirror of a synthetic lane

- `CL-TEST-001.md` carries *"CANONICAL IMPACT: NONE — no theorem is weakened by any item;
  every item is a label defect."* Until 2026-09-20 this bullet read
  *"CANONICAL IMPACT: NONE — no theorem is weakened by any item."*
- A second bullet quotes *"AUTHORITY: none, recorded by the synthetic operator"*.
"""


def git(root, *args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(root),
                          check=True)


def run(root, *args):
    return subprocess.run([sys.executable, CHECKER, "--root", str(root), *args],
                          capture_output=True, text=True, cwd=ROOT)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def repo(tmp_path):
    """A git repository whose committed README carries the original wording."""
    root = tmp_path / "repo"
    readme = root / "drive" / "mirrors" / "SYNTHETIC" / "README.md"
    write(str(readme), ORIGINAL)
    write(str(root / "drive" / "mirrors" / "SYNTHETIC" / "_MANIFEST.jsonl"),
          json.dumps({"id": "x1", "title": "CL-TEST-001.md", "stored": False}) + "\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "controls@example.invalid")
    git(root, "config", "user.name", "controls")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "original")
    return root


def readme_path(root):
    return str(root / "drive" / "mirrors" / "SYNTHETIC" / "README.md")


def test_control_a_faithful_disclosure_passes(repo):
    write(readme_path(repo), CORRECTED)
    out = run(repo)
    assert out.returncode == 0, out.stdout
    assert "disclosures=1 checked=1" in out.stdout and "problems=0" in out.stdout


def test_control_a_misquoted_disclosure_is_refused(repo):
    """The defect this tool exists for: the note invents what the file said."""
    write(readme_path(repo), CORRECTED.replace(
        "no theorem is weakened by any item.\"*\n",
        "every theorem is weakened by several items.\"*\n"))
    out = run(repo)
    assert out.returncode == 1
    assert "not verbatim in HEAD" in out.stdout


def test_control_a_disclosure_off_by_one_word_is_refused(repo):
    write(readme_path(repo), CORRECTED.replace(
        "Until 2026-09-20 this bullet read\n  *\"CANONICAL IMPACT: NONE — no theorem is weakened by any item.\"*",
        "Until 2026-09-20 this bullet read\n  *\"CANONICAL IMPACT: NONE — no theorem was weakened by any item.\"*"))
    out = run(repo)
    assert out.returncode == 1 and "not verbatim in HEAD" in out.stdout


def test_control_a_disclosure_in_a_brand_new_file_is_refused(repo):
    """A file with no committed version has no former wording to disclose."""
    write(str(repo / "drive" / "mirrors" / "BRAND_NEW" / "README.md"),
          "# New lane\n\nUntil 2026-09-20 this read \"a wording that never existed here\".\n")
    out = run(repo)
    assert out.returncode == 1
    assert "the file is new at HEAD" in out.stdout


def test_control_the_unchanged_tree_has_nothing_to_check(repo):
    out = run(repo)
    assert out.returncode == 0 and "disclosures=0 checked=0" in out.stdout


def test_this_tool_does_not_decide_whether_a_quotation_matches_a_stored_byte(repo):
    """The division of labour. Rewriting a published quotation is caught here as
    a silent rewording; whether the NEW wording is verbatim in a stored byte is
    tools/mirror_quotes_check.py's question, and this tool answers it for no
    quotation at all -- note disclosures=0 even as it refuses the change."""
    write(readme_path(repo), ORIGINAL.replace(
        "AUTHORITY: none, recorded by the synthetic operator",
        "AUTHORITY: the operator signed this off in full"))
    out = run(repo)
    assert out.returncode == 1
    assert "disclosures=0" in out.stdout and "records no correction at all" in out.stdout
    # and with the reverse invariant off, it has nothing to say about it
    assert run(repo, "--skip-removals").returncode == 0


def test_control_an_earlier_revision_can_be_named(repo):
    """--rev is a real knob: a disclosure resolves against the revision that
    actually preceded it, and a wrong one is caught there."""
    write(readme_path(repo), CORRECTED)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "corrected")
    assert run(repo, "--rev", "HEAD~1").returncode == 0
    write(readme_path(repo), CORRECTED.replace(
        "no theorem is weakened by any item.\"*\n", "a fabricated former wording.\"*\n"))
    out = run(repo, "--rev", "HEAD~1")
    assert out.returncode == 1 and "not verbatim in HEAD~1" in out.stdout


def test_once_committed_the_check_becomes_self_satisfying(repo):
    """The limit that makes this a pre-commit gate and not a CI step.

    A disclosure at commit N describes the text at commit N-1. Once the
    correction is committed, the named revision carries the note itself, so the
    quoted former wording is trivially found inside it and the check can no
    longer fail. Pinning that here so nobody later mistakes a green post-commit
    run for a standing guarantee, or promotes this tool into CI on the strength
    of one."""
    write(readme_path(repo), CORRECTED.replace(
        "no theorem is weakened by any item.\"*\n", "a fabricated former wording.\"*\n"))
    assert run(repo).returncode == 1          # caught while it is uncommitted
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "the fabrication, committed")
    assert run(repo).returncode == 0          # and invisible once it is not


# ---------------------------------------------------------------------------
# the reverse invariant: a correction must not erase a quotation silently
# ---------------------------------------------------------------------------

def test_control_a_silent_rewording_is_refused(repo):
    """Published text rewritten, and the file says nothing about any correction."""
    write(readme_path(repo), ORIGINAL.replace(
        "no theorem is weakened by any item.", "every theorem is weakened by several items."))
    out = run(repo)
    assert out.returncode == 1
    assert "records no correction at all" in out.stdout


def test_a_rewording_in_a_file_that_records_corrections_is_reported_not_refused(repo):
    """Some corrections remove quotation marks on purpose, because the object was
    read and never stored. Such a correction cannot disclose itself by quoting the
    old wording -- that is the thing it is fixing -- so it is printed for a reader
    rather than failed."""
    write(readme_path(repo), ORIGINAL.replace(
        '*"CANONICAL IMPACT: NONE \u2014 no theorem is weakened by any item."*',
        "its canonical impact as recorded at the time, which no stored byte carries") +
        "\nUntil 2026-09-20 that clause quoted the header, although nothing here holds it.\n")
    out = run(repo)
    assert out.returncode == 0, out.stdout
    assert "reworded=1" in out.stdout
    assert "read them and confirm this one is covered" in out.stdout


def test_a_quotation_merely_extended_is_not_a_rewording(repo):
    """Restoring a dropped clause leaves the old text as a substring of the new,
    which is a repair, not an erasure."""
    write(readme_path(repo), CORRECTED)
    out = run(repo)
    assert out.returncode == 0 and "reworded=0" in out.stdout, out.stdout


def test_skip_removals_turns_the_reverse_invariant_off(repo):
    write(readme_path(repo), ORIGINAL.replace(
        "no theorem is weakened by any item.", "every theorem is weakened by several items."))
    assert run(repo).returncode == 1
    out = run(repo, "--skip-removals")
    assert out.returncode == 0 and "reworded=0" in out.stdout


def test_the_tool_runs_clean_against_the_real_tree():
    """A smoke test, and honest about being one: by the time this runs in CI the
    corrections are committed, so per the control above the comparison is
    self-satisfying. The value of this tool is the pre-commit run recorded in
    CLAUDE.md, not this line."""
    out = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


def test_the_real_scan_is_not_vacuous():
    """If the repository carried no disclosures the control above would be
    empty. It carries plenty; they are how every correction here is recorded."""
    out = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True, cwd=ROOT)
    disclosures = int(out.stdout.split("disclosures=")[1].split()[0])
    assert disclosures >= 10, out.stdout
