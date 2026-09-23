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


def test_every_byte_exact_claim_rests_on_a_full_matching_digest():
    """The property, not the state.

    An earlier version asserted that NOTHING was byte-exact. That recorded the
    repository's condition on the morning of 2026-09-18, not a property, and it
    failed the moment the port-fidelity repair restored three objects from their
    declared digests. What must hold: a record may say byte_exact only when a
    full 64-hex declared digest exists and equals the on-disk digest exactly.
    """
    for a in record()["artifacts"]:
        if a["byte_exact"]:
            declared = a["source_declared_sha256"]
            assert declared and len(declared) == 64, a["path"]
            assert declared == a["repo_sha256"], a["path"]
            assert a["source_declared_bytes"] == a["repo_bytes"], a["path"]


def test_the_sharp_case_is_now_the_object():
    """OP-PROT-019 was the file with the object's byte count and the wrong
    digest. After the repair it must hash to the digest the register declares."""
    arts = {a["path"]: a for a in record()["artifacts"]}
    prot = arts["governance/protocols/OP-PROT-019-v1.1_R17.md"]
    assert prot["byte_exact"] is True
    assert prot["repo_sha256"] == prot["source_declared_sha256"]
    assert prot["repo_sha256"].startswith("04987ba47b58be62")


def test_the_byte_count_trap_is_still_named_when_it_occurs(tmp_path):
    """The checker must still call out a count-match-digest-mismatch record by
    name. The real data no longer contains one, so construct it."""
    d = record()
    for a in d["artifacts"]:
        if a["path"].endswith("OP-PROT-019-v1.1_R17.md"):
            a["byte_exact"] = False
            a["source_declared_sha256"] = "0" * 64      # wrong, full-length
            a["byte_count_matches"] = True
            break
    proc = subprocess.run(
        [sys.executable, CHECKER, "--record", write_tmp(d, str(tmp_path)), "--root", ROOT],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0                          # a trap is named, not failed
    assert "BYTE COUNT MATCHES, DIGEST DOES NOT" in proc.stdout
    assert "count_only_traps=1" in proc.stdout


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
    """Set both fields explicitly: the docs records now carry full digests, so a
    bare byte_exact=True on one of them would be legitimate, not a violation."""
    d = record()
    a = d["artifacts"][0]
    a["byte_exact"] = True
    a["source_declared_sha256"] = a["repo_sha256"][:16]     # truncated on purpose
    assert run(write_tmp(d, str(tmp_path))) == 1


def test_rejects_byte_exact_claimed_against_a_mismatching_digest(tmp_path):
    """OP-PROT-019 now genuinely matches, so force a full-length wrong digest."""
    d = record()
    for a in d["artifacts"]:
        if a["path"].endswith("OP-PROT-019-v1.1_R17.md"):
            a["byte_exact"] = True
            a["source_declared_sha256"] = "f" * 64            # full, and wrong
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
    # OP-PROT-012 is a native-Doc export with no declared digest anywhere in the
    # corpus, so it can never legitimately be called verbatim. (FULL_DOCS_MATH_READ
    # was the earlier target; it is byte-exact since the repair, so a verbatim
    # claim about it would now be TRUE and correctly allowed.)
    text += "\n| `OP-PROT-012.md` | ported verbatim from Drive |\n"
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)

    d = record()
    # Only keep records whose files survived the partial copy.
    d["artifacts"] = [
        a for a in d["artifacts"] if os.path.isfile(os.path.join(fake_root, a["path"]))
    ]
    assert any(a["path"].endswith("OP-PROT-012.md") and not a["byte_exact"] for a in d["artifacts"])
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


# --------------------------------------------------------------------------
# README.md's provenance tally is computed from the register, not typed
#
# The sentence said "the seven artifacts under `governance/` and `docs/`". The
# seven span three directories: four under `governance/`, two under `docs/`,
# and `registers/source/GP-REG-032_v1.2_export_2026-09-17.md`. Naming two of
# the three left one of the four undeclared-digest artifacts out of the account
# it belongs to, and nothing compared the sentence against the register.
# --------------------------------------------------------------------------

def _provenance_artifacts():
    with open(os.path.join(ROOT, "governance", "PROVENANCE.json"), encoding="utf-8") as f:
        return json.load(f)["artifacts"]


def _readme_paragraph():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        text = f.read()
    marker = "that mirror a Drive"
    assert marker in text, "README.md has no provenance tally paragraph"
    start = text.rindex("**Of the", 0, text.index(marker))
    return text[start:text.index("\n\n", start)]


def test_the_readme_names_every_directory_the_seven_artifacts_live_in():
    arts = _provenance_artifacts()
    roots = sorted({a["path"].split("/")[0] + "/" for a in arts})
    para = _readme_paragraph()
    missing = [r for r in roots if f"`{r}" not in para]
    assert not missing, (
        f"PROVENANCE.json puts artifacts under {roots} and README.md's tally "
        f"paragraph does not name {missing}")


def test_the_readme_tally_matches_the_register():
    arts = _provenance_artifacts()
    exact = sum(1 for a in arts if a.get("outcome") == "REPLACED_BYTE_EXACT")
    para = _readme_paragraph()
    words = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    assert words[len(arts)] in para, f"{len(arts)} artifacts; the paragraph must say so"
    assert words[exact] in para, f"{exact} byte-identical; the paragraph must say so"
    assert words[len(arts) - exact] in para


def test_negative_control_a_dropped_directory_is_refused():
    arts = _provenance_artifacts()
    roots = sorted({a["path"].split("/")[0] + "/" for a in arts})
    assert len(roots) >= 3, roots
    # The paragraph may name a deeper path (`registers/source/`) than the root
    # the register groups by (`registers/`), so drop the backtick-prefixed
    # root rather than an exact `root/` token.
    para = _readme_paragraph().replace(f"`{roots[-1]}", "")
    assert [r for r in roots if f"`{r}" not in para]


def test_the_artifact_paths_all_exist():
    for a in _provenance_artifacts():
        assert os.path.isfile(os.path.join(ROOT, a["path"])), a["path"]
