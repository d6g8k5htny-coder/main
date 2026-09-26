"""Negative controls for ``tools/noncertifying_check.py``.

The checker exists because `CLAUDE.md` rule 3 -- "Where you compute in floats,
label the path NON-CERTIFYING in the code and in any output" -- was enforced by
discipline only. No tool in `tools/` contained the string. Twenty-two
repository-authored files hold a float literal or a `float(` call and eight
carried no label; six spellings of the word were in use.

Each control runs the checker through its CLI against synthetic files, so a
path or a pattern bound at import time cannot silently re-check the real tree.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "noncertifying_check.py")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import noncertifying_check as NCC  # noqa: E402


def run(root, *files):
    args = [sys.executable, CHECKER, "--root", root]
    for f in files:
        args += ["--file", f]
    return subprocess.run(args, capture_output=True, text=True)


def write(tmp_path, name, body):
    (tmp_path / name).write_text(body, encoding="utf-8")
    return str(tmp_path), name


# ---------------------------------------------------------------------------
# 1. The rule
# ---------------------------------------------------------------------------

def test_an_unlabelled_float_literal_is_refused(tmp_path):
    root, name = write(tmp_path, "m.py", "def f():\n    return 0.5\n")
    out = run(root, name)
    assert out.returncode == 1, out.stdout
    assert "float site" in out.stdout and "NON-CERTIFYING" in out.stdout


def test_an_unlabelled_float_call_is_refused(tmp_path):
    root, name = write(tmp_path, "m.py", "def f(x):\n    return float(x)\n")
    out = run(root, name)
    assert out.returncode == 1, out.stdout


def test_a_labelled_float_path_passes(tmp_path):
    root, name = write(tmp_path, "m.py",
                       'def f(x):\n    """NON-CERTIFYING display."""\n'
                       "    return float(x)\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "with_float_sites=1" in out.stdout


def test_a_file_with_no_float_site_needs_no_label(tmp_path):
    root, name = write(tmp_path, "m.py",
                       "from fractions import Fraction\n\n\n"
                       "def f():\n    return Fraction(1, 2)\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "with_float_sites=0" in out.stdout


def test_a_float_inside_a_docstring_is_text_not_a_float(tmp_path):
    """The numbers quoted in prose throughout this repository are strings."""
    root, name = write(tmp_path, "m.py",
                       'def f():\n    """The value is 6.239e-44 exactly."""\n'
                       "    return 1\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "with_float_sites=0" in out.stdout


# ---------------------------------------------------------------------------
# 2. Spelling
# ---------------------------------------------------------------------------

def test_a_prose_misspelling_without_the_canonical_label_is_refused(tmp_path):
    root, name = write(tmp_path, "m.py",
                       'def f():\n    """This path is non_certifying."""\n'
                       "    return 1\n")
    out = run(root, name)
    assert out.returncode == 1, out.stdout
    assert "spelled" in out.stdout


def test_an_identifier_spelling_beside_the_canonical_label_is_allowed(tmp_path):
    """A hyphen cannot appear in a Python name, so both must coexist."""
    root, name = write(tmp_path, "m.py",
                       'NON_CERTIFYING = True\n\n\ndef f():\n'
                       '    """NON-CERTIFYING float display."""\n'
                       "    return float(1)\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout


def test_a_longer_identifier_is_not_the_label_misspelled(tmp_path):
    """`float_noncertifying` is a receipt FIELD NAME. This was a false positive."""
    root, name = write(tmp_path, "m.py",
                       'def f():\n    """Counts float_noncertifying rows."""\n'
                       "    return 1\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "identifier_uses=0" in out.stdout


# ---------------------------------------------------------------------------
# 3. Declarations
# ---------------------------------------------------------------------------

def test_the_declared_files_all_exist_and_still_need_their_declaration():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert NCC.DECLARED, "an empty exemption list would make the rule vacuous"
    for rel, reason in NCC.DECLARED.items():
        assert os.path.isfile(os.path.join(ROOT, rel)), rel
        assert len(reason) > 40, f"{rel}: the reason must say something"


def test_a_declaration_for_a_file_with_no_float_is_refused(tmp_path, monkeypatch):
    """A stale exemption hides the next real one."""
    root, name = write(tmp_path, "m.py", "def f():\n    return 1\n")
    scanned, floats, ident, problems = NCC.audit(root, [name])
    assert not problems
    monkeypatch.setitem(NCC.DECLARED, name, "x" * 50)
    scanned, floats, ident, problems = NCC.audit(root, [name])
    assert any("has none" in p for p in problems), problems


def test_a_declaration_for_a_file_that_is_labelled_is_refused(tmp_path, monkeypatch):
    root, name = write(tmp_path, "m.py",
                       'def f():\n    """NON-CERTIFYING."""\n    return 0.5\n')
    monkeypatch.setitem(NCC.DECLARED, name, "x" * 50)
    _, _, _, problems = NCC.audit(root, [name])
    assert any("carries one" in p for p in problems), problems


def test_a_declaration_for_a_missing_file_is_refused(tmp_path, monkeypatch):
    """Checked on a default-roots run: an explicit --file list scopes it away."""
    monkeypatch.setitem(NCC.DECLARED, "not/here.py", "x" * 50)
    _, _, _, problems = NCC.audit(str(tmp_path))
    assert any("not/here.py" in p and "not in the tree" in p for p in problems), problems


def test_an_empty_tree_is_a_vacuous_run_and_is_refused(tmp_path):
    """The same lesson `verify_manifests.py` learned: scanning nothing is not a pass."""
    out = run(str(tmp_path))
    assert out.returncode == 1, out.stdout
    assert "VACUOUS RUN" in out.stdout
    assert "files=0" in out.stdout


# ---------------------------------------------------------------------------
# 4. Scope, stated rather than assumed
# ---------------------------------------------------------------------------

def test_ported_bytes_are_out_of_scope():
    """Blobs and frozen bodies are byte-exact copies and are never edited."""
    for rel in ("engine/carriers/blobs", "engine/rn_engine/frozen"):
        assert NCC.is_excluded(rel + "/x.py")
    assert not NCC.is_excluded("research/cover/driver.py")


def test_the_scanner_is_not_vacuous():
    files = NCC.python_files(ROOT)
    assert len(files) > 60
    assert "tools/noncertifying_check.py" in files
    assert not any("blobs" in f for f in files)


def test_the_repository_passes_and_actually_found_float_sites():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout
    n = int(out.stdout.split("with_float_sites=")[1].split()[0])
    assert n >= 20, f"only {n} files with float sites; the tree had 22"


def test_the_known_float_bearing_modules_are_labelled_or_declared():
    """The eight that were unlabelled when the checker landed.

    ``research/bands/ladder.py`` was labelled in-file until the pin
    reconciliation below moved it into ``DECLARED``; see that test for why.
    """
    for rel in ("research/rn/moment_envelope.py",
                "tests/test_rn_moment_envelope.py"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            assert NCC.LABEL in f.read(), rel
    for rel in ("tools/registers_import.py", "tools/drive_index.py",
                "tests/test_bridge.py", "tests/test_hermite_envelope.py",
                "tests/test_lpw_headline.py", "research/bands/ladder.py"):
        assert rel in NCC.DECLARED, rel


#: The bytes two certificates on the ``chatgpt/drive-github-hardening-20260919``
#: lane bind as a source identity.  Recorded here so an edit to the file fails
#: loudly in this repository rather than silently in theirs.
LADDER_PINNED_SHA256 = \
    "9ea576708e146aa54fdc3c6859274d2c35a81218438bcdb34283afe9f79ac0c5"
LADDER_PINNED_BYTES = 26286


def test_ladder_py_still_matches_the_digest_two_certificates_pin():
    """A cross-lane byte binding, made visible from inside this repository.

    ``research/parallel/h3/candidate.json`` and
    ``research/rn/candidates/inner_wedge_20260920_v1.json`` on the hardening
    lane record this file's SHA-256 as a source identity of their certificates,
    and ``tools/twelve_project_check.py`` there fails closed on a mismatch.
    Nothing in *this* repository said so, so commit ``4cc0f7a`` added a
    NON-CERTIFYING docstring and two ``format_report`` lines, changed the bytes,
    and broke a binding no checker here could see -- 19 failures in a merged
    tree, none of them reproducible on either branch alone.

    The label now lives in ``NCC.DECLARED`` instead and the bytes are back.
    This control is the part that keeps it that way: it is a repository-side
    record of somebody else's dependency.

    If this fails, the file was edited.  That is not forbidden -- but it
    invalidates two certificates' provenance, so re-pin both JSON files on the
    hardening lane in the same change, or revert the edit.  Do not simply
    update the constant here; the constant is not the authority, their pins are.
    """
    path = os.path.join(ROOT, "research", "bands", "ladder.py")
    raw = open(path, "rb").read()
    assert len(raw) == LADDER_PINNED_BYTES, (
        f"{len(raw)} bytes, pinned at {LADDER_PINNED_BYTES}")
    assert hashlib.sha256(raw).hexdigest() == LADDER_PINNED_SHA256


def test_negative_control_a_changed_ladder_is_caught():
    """Flipping one byte must fail the pin, or the control above is decoration."""
    raw = open(os.path.join(ROOT, "research", "bands", "ladder.py"), "rb").read()
    mutated = raw + b"\n"
    assert hashlib.sha256(mutated).hexdigest() != LADDER_PINNED_SHA256
    assert len(mutated) != LADDER_PINNED_BYTES
