"""Negative controls for the ported-artifact identity check.

The defect these guard against was real and was found in this repository's own
migration by its own review harness: ``OP-PROT-019-v1.1_R17.md`` carries the same
byte count as the object the register names (14,073) and a different SHA-256, so
a byte-count check confirms the wrong bytes. Two other ported files carried a
provenance header reading "Ported verbatim" — and that header was itself the only
thing making them non-verbatim.

Every test below constructs the violation and asserts the checker rejects it. The
last one asserts the opposite: that the corpus's own discussion of byte-exactness
as subject matter is NOT flagged, because a checker that cries wolf on
"the byte-exact TB-G2 capsules" would be turned off within a day.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "provenance_check.py")
RECORD = os.path.join(ROOT, "governance", "PROVENANCE.json")


def run(record_path: str, root: str = ROOT) -> int:
    """Invoke the checker through its CLI, which is the only supported entry.

    Deliberately not by import: a default argument once bound a checker's data
    path at import time in this repository, so every mutation test silently
    re-checked the good data and all eight passed against a broken checker.
    """
    proc = subprocess.run(
        [sys.executable, CHECKER, "--record", record_path, "--root", root],
        capture_output=True,
        text=True,
    )
    return proc.returncode


def record() -> dict:
    with open(RECORD, encoding="utf-8") as f:
        return json.load(f)


def write_tmp(data: dict, tmpdir: str) -> str:
    p = os.path.join(tmpdir, "PROVENANCE.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return p


# --------------------------------------------------------------- the real state

def test_the_committed_record_passes():
    assert run(RECORD) == 0


def test_nothing_currently_claims_byte_exactness():
    # If a future port really is byte-exact this test should be updated with the
    # matching digest — not deleted.
    assert all(not a["byte_exact"] for a in record()["artifacts"])


def test_the_byte_count_trap_is_recorded_not_hidden():
    arts = {a["path"]: a for a in record()["artifacts"]}
    prot = arts["governance/protocols/OP-PROT-019-v1.1_R17.md"]
    assert prot["byte_count_matches"] is True
    assert prot["byte_exact"] is False
    assert prot["source_declared_sha256"] != prot["repo_sha256"]


# ------------------------------------------------------------ negative controls

def test_rejects_a_record_that_has_drifted_from_disk(tmp_path):
    d = record()
    d["artifacts"][0]["repo_sha256"] = "0" * 64
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_a_record_with_a_wrong_byte_count(tmp_path):
    d = record()
    d["artifacts"][0]["repo_bytes"] = 1
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_byte_exact_claimed_without_a_full_digest(tmp_path):
    d = record()
    for a in d["artifacts"]:
        if a["path"].startswith("docs/"):
            a["byte_exact"] = True          # declared digest is truncated
            break
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_byte_exact_claimed_against_a_mismatching_digest(tmp_path):
    d = record()
    for a in d["artifacts"]:
        if a["path"].endswith("OP-PROT-019-v1.1_R17.md"):
            a["byte_exact"] = True          # full digest present, and it does not match
            break
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_a_missing_artifact(tmp_path):
    d = record()
    d["artifacts"][0]["path"] = "governance/protocols/does-not-exist.md"
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_duplicate_records_for_one_path(tmp_path):
    d = record()
    d["artifacts"].append(dict(d["artifacts"][0]))
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_a_reintroduced_verbatim_claim(tmp_path):
    """The exact defect: someone re-adds "Ported verbatim" to a non-exact copy."""
    fake_root = str(tmp_path / "repo")
    shutil.copytree(
        ROOT,
        fake_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "drive", "registers"),
    )
    os.makedirs(os.path.join(fake_root, "registers", "source"), exist_ok=True)
    target = os.path.join(fake_root, "docs", "README.md")
    with open(target, encoding="utf-8") as f:
        text = f.read()
    text += "\n| `FULL_DOCS_MATH_READ.md` | ported verbatim from Drive |\n"
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)

    d = record()
    # Only keep records whose files survived the partial copy.
    d["artifacts"] = [
        a for a in d["artifacts"] if os.path.isfile(os.path.join(fake_root, a["path"]))
    ]
    assert run(write_tmp(d, str(tmp_path)), root=fake_root) == 1


def test_subject_matter_mentions_of_byte_exactness_are_not_flagged(tmp_path):
    """False-positive control.

    The corpus legitimately discusses byte-exact Drive capsules — LS-DATA-015's
    whole purpose is to carry exact bytes. Flagging that would make the checker
    noise, and a noisy checker gets disabled. This asserts the narrowing holds.
    """
    fake_root = str(tmp_path / "repo")
    shutil.copytree(
        ROOT,
        fake_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "drive", "registers"),
    )
    os.makedirs(os.path.join(fake_root, "registers", "source"), exist_ok=True)
    target = os.path.join(fake_root, "docs", "README.md")
    with open(target, "a", encoding="utf-8") as f:
        f.write(
            "\nThe eight EMPTY_NATIVE_BODY carriers include the byte-exact and "
            "hex-gzip TB-G2 algebra result capsules, whose entire purpose is to "
            "carry exact bytes and which are byte-identical to one another.\n"
        )

    d = record()
    d["artifacts"] = [
        a for a in d["artifacts"] if os.path.isfile(os.path.join(fake_root, a["path"]))
    ]
    assert run(write_tmp(d, str(tmp_path)), root=fake_root) == 0
