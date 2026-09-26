"""Negative controls for ``tools/noncertifying_check.py``.

The checker exists because `CLAUDE.md` rule 3 -- "Where you compute in floats,
label the path NON-CERTIFYING in the code and in any output" -- was enforced by
discipline only. No tool in `tools/` contained the string. On this line
forty-one repository-authored files hold a float literal or a `float(` call and
twenty-four carried no label; six spellings of the word were in use.

Reading all twenty-four changed the shape of the fix, and group 4 below is the
consequence: sixteen of them use a float ONLY as a REJECTION PROBE, so the
banner would be a FALSE statement about the file. Those are declared, and a
control asserts that a declaration is not interchangeable with a label.

Each control runs the checker through its CLI against synthetic files, so a
path or a pattern bound at import time cannot silently re-check the real tree.
"""
from __future__ import annotations

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
    assert n >= 38, f"only {n} files with float sites; the tree had 41"


def test_the_one_file_that_could_be_labelled_carries_the_label():
    """Two files compute in floats where it matters; only one is editable.

    `research/rn/moment_envelope.py` and `research/bands/ladder.py` are both
    PINNED, so neither can carry the banner in its bytes. Both are declared as
    UNMET obligations by the test below rather than as absent ones.
    """
    rel = "tests/test_rn_moment_envelope.py"
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        assert NCC.LABEL in f.read(), rel


def test_both_pinned_display_paths_are_declared_as_unmet_obligations():
    """The honest cases, and the ones a silent exclusion would have hidden: each
    owes the label under rule 3 and cannot carry it."""
    for rel in ("research/bands/ladder.py", "research/rn/moment_envelope.py"):
        reason = NCC.DECLARED[rel]
        assert "PINNED" in reason, rel
        assert "cannot be added in place" in reason, rel
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            assert NCC.LABEL not in f.read(), rel   # the bytes are untouched


def test_the_pinned_moment_envelope_bytes_match_the_archive_member_pin():
    """The control that would have stopped this branch adding a label to pinned
    bytes, and did not exist because `research/PINNED_SOURCES.md` does not cover
    pins declared inside an ARCHIVE MEMBER. `tools/rn_bernstein_sharp_check.py`
    caught it on the CI replay instead, with "repository dependency mismatch".

    Asserted here against the archive itself, so a future edit to this file fails
    in a test named after the reason rather than inside a campaign replay.
    """
    import hashlib
    import json
    import zipfile
    archive = os.path.join(
        ROOT, "research", "campaigns",
        "rn_bernstein_sharp_variance_20260921_v1.zip")
    with zipfile.ZipFile(archive) as z:
        deps = json.loads(z.read("bernstein/DEPENDENCIES.json"))
    # `files` is a path -> {bytes, sha256} MAPPING in this member; the sibling
    # `sharp_variance/DEPENDENCIES.json` uses the same shape. Both forms are
    # handled so this control does not break on the other one.
    files = deps["files"]
    rows = (files if isinstance(files, dict)
            else {r["path"]: r for r in files})
    rel = "research/rn/moment_envelope.py"
    assert rel in rows, sorted(rows)[:5]
    raw = open(os.path.join(ROOT, rel), "rb").read()
    assert len(raw) == rows[rel]["bytes"], (len(raw), rows[rel]["bytes"])
    assert hashlib.sha256(raw).hexdigest() == rows[rel]["sha256"]
    # and the index that a contributor would consult does NOT list it, which is
    # the gap this test exists to make visible rather than to paper over
    with open(os.path.join(ROOT, "research", "PINNED_SOURCES.md"),
              encoding="utf-8") as f:
        assert "moment_envelope" not in f.read()


def test_the_rejection_probe_files_are_declared_and_not_labelled():
    """The finding that shaped this port. A float fed to a checker so the
    checker REFUSES it is not a float path: pasting NON-CERTIFYING onto such a
    file states the opposite of what the file demonstrates. Declared, every one,
    and none of them carries the banner."""
    probes = ("tests/test_c2_band.py", "tests/test_gaussian_moments.py",
              "tests/test_hermite_gaussian.py", "tests/test_rn_certificate.py",
              "tests/test_rn_conditioning.py", "tests/test_rn_side24.py",
              "tests/test_rn_side24_cell.py", "tests/test_rn_side24_density.py",
              "tests/test_rn_side24_wedge.py", "tests/test_rn_spatial_cover.py",
              "tests/test_rn_density_majorant.py", "tests/test_gaussian_families.py",
              "tests/test_manifest_integrity_hardening.py", "tests/test_bridge.py",
              "tests/test_lpw_headline.py")
    for rel in probes:
        assert rel in NCC.DECLARED, rel
        assert "refus" in NCC.DECLARED[rel].lower() or "reject" in NCC.DECLARED[rel].lower(), rel
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            assert NCC.LABEL not in f.read(), rel


def test_a_declaration_and_a_label_are_not_interchangeable():
    """A file cannot hold both: a declaration says "no label is owed here", so
    carrying one means the declaration is stale. The checker already refuses
    that, and this pins it, because the whole rejection-probe argument rests on
    the two being different states rather than two spellings of the same one."""
    for rel, reason in sorted(NCC.DECLARED.items()):
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            has_label = NCC.LABEL in f.read()
        assert not has_label, f"{rel} is DECLARED and carries the label"


def test_the_wall_clock_declarations_say_so():
    for rel in ("tools/run_checks.py", "tools/twelve_project_check.py",
                "tests/test_twelve_project_check.py", "tests/test_run_checks.py"):
        reason = NCC.DECLARED[rel].lower()
        assert ("wall-clock" in reason or "duration" in reason
                or "timeout" in reason), rel


def test_the_recovered_custody_tree_is_excluded_not_declared():
    """Recovered Drive bytes, pinned. Same ground as the carrier blobs: the one
    edit that would break them is adding a banner. Excluded, and the exclusion
    is stated in the module docstring rather than only in a tuple."""
    assert "research/side24/source_recovery/custody" in NCC.EXCLUDED
    assert NCC.is_excluded(
        "research/side24/source_recovery/custody/arithmetic_ledgers/"
        "40ad76a1973248d9/arithmetic_ledgers.py")
    assert "source_recovery/custody" in NCC.__doc__
    # and it really does hold float sites, so the exclusion is load-bearing
    scanned = [r for r in NCC.python_files(ROOT) if "source_recovery/custody" in r]
    assert scanned == [], scanned


def test_the_spelling_declaration_is_a_stated_reason_not_a_silencer():
    """The checker's SECOND false positive of the same family as its first.
    `research/cover/audit.py` has ZERO float sites and uses "noncertifying" as an
    English adjective about a total's status; rule 2 fired on it anyway. The rule
    is not narrowed -- a file that labels an mpmath path in prose and writes the
    word badly must still be caught -- so the exemption is a declaration with the
    reason, and the file's bytes are untouched."""
    reason = NCC.SPELLING_DECLARED["research/cover/audit.py"]
    assert "adjective" in reason and "ZERO float sites" in reason
    src = open(os.path.join(ROOT, "research", "cover", "audit.py"),
               encoding="utf-8").read()
    assert NCC.LABEL not in src            # unedited
    assert "noncertifying" in src
    assert NCC.float_sites(__import__("ast").parse(src)) == 0


def test_the_spelling_rule_still_fires_for_an_undeclared_file(tmp_path):
    """SPELLING_DECLARED must not have disabled rule 2. An undeclared file that
    writes the label badly is still refused."""
    root, name = write(tmp_path, "m.py",
                       '"""A noncertifying estimate."""\nX = 1.5\n')
    os.makedirs(os.path.join(root, "tools"), exist_ok=True)
    out = run(root, name)
    assert out.returncode == 1, out.stdout
    assert "spelled 'noncertifying' in prose" in out.stdout, out.stdout


def test_the_summary_reports_both_declaration_counts():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert f"declared={len(NCC.DECLARED)}" in out.stdout, out.stdout
    assert f"spelling_declared={len(NCC.SPELLING_DECLARED)}" in out.stdout, out.stdout
