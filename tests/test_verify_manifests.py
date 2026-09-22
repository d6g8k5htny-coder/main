"""Controls for ``tools/verify_manifests.py``: the floor, and the sha256sum reader.

TWO HOLES THIS FILE EXISTS FOR, both found on 2026-09-22.

1. A VACUOUS PASS. The tool reported ``manifests=0 verified=0 problems=0`` and
   exited 0 against a tree with no manifests in it. The exit code would have
   been identical if every manifest in the repository had been deleted, and
   nothing in the suite pinned a floor. CI invokes it bare.
2. A SILENT SKIP. ``check_sha256sum`` walked past any line that was not
   ``<64 chars> <name>`` without counting it ok or bad, so a manifest could be
   truncated mid-line, or have its digest column mangled, and still report
   ``problems=0``. The branch was wholly unexercised: no test in the tree
   mentioned ``check_sha256sum``, ``MANIFEST.sha256`` or ``sha256sum`` at all.
   The length test also did not check that the 64 characters were hex.

Every control here drives the tool through its CLI.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "verify_manifests.py")


def run(*args):
    return subprocess.run([sys.executable, CHECKER, *args],
                          capture_output=True, text=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def make_sha_manifest(tmp_path, lines, payload=b"hello\n"):
    d = tmp_path / "lane"
    d.mkdir()
    (d / "body.txt").write_bytes(payload)
    digest = sha256_of(str(d / "body.txt"))
    text = "\n".join(l.replace("<DIGEST>", digest) for l in lines) + "\n"
    (d / "MANIFEST.sha256").write_text(text, encoding="utf-8")
    return str(tmp_path), digest


# ---------------------------------------------------------------------------
# 1. The floor
# ---------------------------------------------------------------------------

def test_an_empty_tree_is_refused(tmp_path):
    out = run(str(tmp_path))
    assert out.returncode == 1, out.stdout
    assert "VACUOUS RUN" in out.stdout
    assert "manifests=0" in out.stdout


def test_the_escape_hatch_allows_an_empty_scan(tmp_path):
    out = run(str(tmp_path), "--min-manifests", "0")
    assert out.returncode == 0, out.stdout
    assert "manifests=0 verified=0 problems=0" in out.stdout


def test_the_equals_form_of_the_flag_works(tmp_path):
    out = run(str(tmp_path), "--min-manifests=0")
    assert out.returncode == 0, out.stdout


def test_a_floor_above_the_real_count_is_refused(tmp_path):
    make_sha_manifest(tmp_path, ["<DIGEST>  body.txt"])
    assert run(str(tmp_path)).returncode == 0
    out = run(str(tmp_path), "--min-manifests", "2")
    assert out.returncode == 1, out.stdout
    assert "fewer than the required 2" in out.stdout


def test_the_repository_is_well_above_the_floor():
    """Pin the real count, so a collapse shows up as a failure and not a pass."""
    out = run()
    assert out.returncode == 0, out.stdout
    n = int(out.stdout.split("manifests=")[1].split()[0])
    assert n >= 400, f"only {n} manifests found; the tree carried 411 on 2026-09-22"


def test_a_missing_number_after_the_flag_is_refused(tmp_path):
    out = run(str(tmp_path), "--min-manifests")
    assert out.returncode != 0
    assert "needs a number" in (out.stdout + out.stderr)


# ---------------------------------------------------------------------------
# 2. The sha256sum reader
# ---------------------------------------------------------------------------

def test_a_correct_sha_manifest_passes(tmp_path):
    root, _ = make_sha_manifest(tmp_path, ["<DIGEST>  body.txt"])
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "verified=1" in out.stdout


def test_a_wrong_digest_is_a_mismatch(tmp_path):
    root, _ = make_sha_manifest(tmp_path, ["0" * 64 + "  body.txt"])
    out = run(root)
    assert out.returncode == 1, out.stdout
    assert "SHA MISMATCH" in out.stdout


def test_a_missing_file_is_reported(tmp_path):
    root, _ = make_sha_manifest(tmp_path, ["<DIGEST>  gone.txt"])
    out = run(root)
    assert out.returncode == 1, out.stdout
    assert "MISSING" in out.stdout


@pytest.mark.parametrize("line", [
    "<DIGEST>",                       # truncated: digest with no name
    "body.txt",                       # name with no digest
    "abc123  body.txt",               # digest too short
    "<DIGEST>0  body.txt",            # digest too long
    "zz" + "0" * 62 + "  body.txt",   # right length, not hex
])
def test_an_unparsable_line_is_a_problem_not_silence(tmp_path, line):
    root, _ = make_sha_manifest(tmp_path, [line])
    out = run(root)
    assert out.returncode == 1, (line, out.stdout)
    assert "UNPARSABLE LINE" in out.stdout


def test_a_non_hex_token_is_diagnosed_as_unparsable_not_as_a_mismatch(tmp_path):
    """The length check alone gave the wrong diagnosis for a mangled column."""
    root, _ = make_sha_manifest(tmp_path, ["z" * 64 + "  body.txt"])
    out = run(root)
    assert "UNPARSABLE LINE" in out.stdout
    assert "SHA MISMATCH" not in out.stdout


def test_comments_and_blank_lines_are_still_skipped(tmp_path):
    root, _ = make_sha_manifest(
        tmp_path, ["# a comment", "", "   ", "<DIGEST>  body.txt"])
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "verified=1" in out.stdout


def test_a_truncated_manifest_does_not_report_zero_problems(tmp_path):
    """The shape of the hole: a good line, then a half-written one."""
    root, _ = make_sha_manifest(
        tmp_path, ["<DIGEST>  body.txt", "a1b2c3"])
    out = run(root)
    assert out.returncode == 1, out.stdout
    assert "verified=1" in out.stdout and "problems=1" in out.stdout


def test_a_star_prefixed_name_is_still_accepted(tmp_path):
    """`sha256sum --binary` writes `*name`; that is not a malformed line."""
    root, _ = make_sha_manifest(tmp_path, ["<DIGEST> *body.txt"])
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "verified=1" in out.stdout
