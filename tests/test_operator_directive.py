"""Negative controls for tools/operator_directive_check.py.

The checker exists because a third-party branch asserted, in
``governance/GIT_ADAPTATION.md``, that "Dylan subsequently directed that the
repository remain private until publication is explicitly authorized" -- with
no source anywhere in the tree, deleting a sourced sentence that said the
opposite, and alongside a present-tense status claim that was false. Every
checker in the repository passed.

Each control runs the checker through its CLI against a synthetic document, so
a path or a pattern bound at import time cannot silently re-check the real
tree. The last two controls run it against the real offending text and against
the real repository.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "operator_directive_check.py")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import operator_directive_check as ODC  # noqa: E402

# The sentence as the branch actually wrote it, with the sentence that followed.
REAL_OFFENDER = (
    "Dylan subsequently directed that the repository remain private until "
    "publication is explicitly authorized. The authenticated GitHub API "
    "confirmed private visibility on 2026-09-20 UTC. Private draft work may "
    "continue; this does not authorize public publication or mathematical "
    "promotion.\n"
)


def run(root, *docs):
    args = [sys.executable, CHECKER, "--root", root]
    for d in docs:
        args += ["--doc", d]
    return subprocess.run(args, capture_output=True, text=True)


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(tmp_path), name


def test_the_real_offending_sentence_is_refused(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md", REAL_OFFENDER)
    out = run(root, name)
    assert out.returncode != 0, out.stdout
    assert "asserts an operator directive with no source" in out.stdout
    # It must name the directive, not some neighbouring run-on chunk.
    assert "Dylan subsequently directed" in out.stdout


def test_the_noun_form_is_refused(tmp_path):
    """"by Dylan's subsequent explicit instruction" asserts just as plainly."""
    root, name = write(tmp_path, "GOVERNED.md",
                       "This repository is now private by Dylan's subsequent "
                       "explicit instruction; public publication requires his "
                       "explicit authorization.\n")
    out = run(root, name)
    assert out.returncode != 0, out.stdout


def test_a_bare_date_is_not_a_citation(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "The operator approved the change on 2026-09-20 UTC, "
                       "confirmed at 17:46 the same day.\n")
    out = run(root, name)
    assert out.returncode != 0, out.stdout


def test_a_drive_id_in_the_same_sentence_is_accepted(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "The operator approved it (Drive "
                       "`10o4YRYOr8a2fB6rtnFnzMn7HkQMfv9-FZ5Mh0L-KF_o`).\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout


def test_a_citation_in_the_following_sentence_is_accepted(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "The operator approved the routing. See OP-PROT-012 for "
                       "the decision as recorded.\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout


def test_a_cross_reference_to_a_governed_file_is_accepted(tmp_path):
    """docs/RESEARCH_MAP.md cites by pointing at the file holding the id."""
    root, name = write(tmp_path, "GOVERNED.md",
                       "The Board's decision of 2026-07-24 is the one operator "
                       "sentence about a Git repository. (The visibility "
                       "decision is the owner's -- see "
                       "`governance/GIT_ADAPTATION.md`.)\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout


def test_the_explicit_disclaimer_opts_a_paragraph_out(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "A reader might think the operator approved this; no "
                       "operator directive is asserted here.\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout


def test_prose_with_no_directive_is_untouched(tmp_path):
    """The control cannot pass by refusing everything."""
    root, name = write(tmp_path, "GOVERNED.md",
                       "The lane holds 449 native Google Docs. None of them is "
                       "certified, and the premises remain OPEN.\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "directives=0" in out.stdout


def test_authority_does_not_rub_off_from_a_neighbouring_citation(tmp_path):
    """The defect the first version of this checker missed.

    An uncited directive appended to a long, heavily cited paragraph about a
    different decision. Paragraph-granularity passed it; sentences do not.
    """
    root, name = write(tmp_path, "GOVERNED.md",
                       "The Board's OPERATOR PACKAGE DECISION of 2026-07-24 "
                       "(Drive `10o4YRYOr8a2fB6rtnFnzMn7HkQMfv9-FZ5Mh0L-KF_o`) "
                       "approved private repository creation. " + REAL_OFFENDER)
    out = run(root, name)
    assert out.returncode != 0, out.stdout
    assert "Dylan subsequently directed" in out.stdout


def test_the_repository_passes():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


# ---------------------------------------------------------------------------
# Fenced code blocks are not prose
#
# README.md's "What CI enforces" block lists `tools/operator_directive_check.py`.
# That filename contains both "operator" and "directive", so the first version
# of this checker counted it as an asserted operator directive -- and PASSED
# it, because the same block names `registers/`, which counts as a citation. A
# false positive that passes is worse than one that fails: it inflates the
# count and teaches a reader to ignore it.
# ---------------------------------------------------------------------------

def test_a_filename_in_a_bash_block_is_not_a_directive(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "## What CI enforces\n\n```bash\n"
                       "python3 tools/operator_directive_check.py   # cites its source\n"
                       "python3 tools/registers_check.py\n```\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "directives=0" in out.stdout


def test_a_tilde_fence_is_stripped_too(tmp_path):
    root, name = write(tmp_path, "GOVERNED.md",
                       "~~~\nthe operator directed everything here\n~~~\n")
    out = run(root, name)
    assert out.returncode == 0, out.stdout
    assert "directives=0" in out.stdout


def test_prose_after_a_fence_is_still_read(tmp_path):
    """Stripping must not swallow the rest of the document."""
    root, name = write(tmp_path, "GOVERNED.md",
                       "```bash\npython3 tools/operator_directive_check.py\n```\n\n"
                       "Dylan subsequently directed that the repository remain private.\n")
    out = run(root, name)
    assert out.returncode != 0, out.stdout
    assert "Dylan subsequently directed" in out.stdout


def test_line_numbers_survive_the_stripping(tmp_path):
    """Blanked fences keep their newlines, so a report still points at the line."""
    body = ("```bash\n" + "echo hello\n" * 8 + "```\n\n"
            "The operator approved the routing with no source at all.\n")
    root, name = write(tmp_path, "GOVERNED.md", body)
    out = run(root, name)
    assert out.returncode != 0, out.stdout
    assert f"{name}:12:" in out.stdout, out.stdout


def test_the_real_readme_block_no_longer_counts():
    """The live tree: three directives, all in prose, none from a code block."""
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert "directives=3" in out.stdout


# ---------------------------------------------------------------------------
# A checker whose coverage can fall to nothing is not enforcing anything
#
# This tool used to skip a governed document that was absent. With the whole
# set gone it printed `docs=0 directives=0 problems=0` and exited 0; deleting
# any one of the six dropped the count by one and still passed. CI invokes it
# bare, so nothing else would have noticed.
# ---------------------------------------------------------------------------

def test_an_empty_root_is_refused(tmp_path):
    out = run(str(tmp_path))
    assert out.returncode != 0, out.stdout
    assert "docs=0" in out.stdout
    assert out.stdout.count("governed document is missing") == 6


def test_one_missing_governed_document_is_refused(tmp_path):
    """Copy the real governed set, drop one, and the checker must object."""
    import shutil
    for rel in ODC.GOVERNED_DOCS:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(str(tmp_path), rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    out = run(str(tmp_path))
    assert out.returncode == 0, out.stdout
    assert "docs=6" in out.stdout

    os.remove(os.path.join(str(tmp_path), ODC.GOVERNED_DOCS[0]))
    out = run(str(tmp_path))
    assert out.returncode != 0, out.stdout
    assert ODC.GOVERNED_DOCS[0] in out.stdout
    assert "docs=5" in out.stdout


def test_the_governed_set_is_explicit_and_all_of_it_is_in_the_tree():
    for rel in ODC.GOVERNED_DOCS:
        assert os.path.isfile(os.path.join(ROOT, rel)), rel
    assert len(ODC.GOVERNED_DOCS) == 6
