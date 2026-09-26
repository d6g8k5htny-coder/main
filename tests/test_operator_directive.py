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


# ------------- the pair of contradictory directives this branch withdrew -------
# The tree carried BOTH of these, unsourced, on the same date and in opposite
# directions: governance/GIT_ADAPTATION.md said the operator directed the
# repository stay private, docs/RESEARCH_MAP.md said he authorized it public. The
# controls above already pin the first shape, because it is the edit this checker
# was written for. These pin the second, and pin the mechanism the withdrawal note
# relies on.

WITHDRAWN_PUBLIC_DIRECTIVE = (
    "Dylan subsequently authorized public visibility on 2026-09-20; "
    "unauthenticated access was verified that day.\n")


def test_the_opposite_unsourced_directive_is_also_refused(tmp_path):
    """Refusing only the "remain private" wording would leave the mirror image of
    the same defect live, which is exactly what happened."""
    doc = tmp_path / "docs" / "RESEARCH_MAP.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Map\n\n" + WITHDRAWN_PUBLIC_DIRECTIVE, encoding="utf-8")
    out = run(str(tmp_path), os.path.join("docs", "RESEARCH_MAP.md"))
    assert out.returncode == 1, out.stdout
    assert "no source" in out.stdout, out.stdout


def test_a_verbatim_quotation_inside_a_fence_is_not_an_assertion(tmp_path):
    """The mechanism governance/GIT_ADAPTATION.md's withdrawal note depends on.

    The note quotes both withdrawn sentences so the withdrawal is checkable. In
    prose that reads as asserting them -- the disclaimer only covers a sentence
    and the one after it, and the quotation is longer than that -- so the
    transcription sits in a fenced block, which this checker strips because a
    fence holds text, not a claim about anybody. If fences ever stopped being
    stripped, the note would fail and the real prose would be edited to appease
    it, which is the wrong repair.
    """
    doc = tmp_path / "README.md"
    doc.write_text("# R\n\nWithdrawn, quoted verbatim below.\n\n```text\n"
                   + WITHDRAWN_PUBLIC_DIRECTIVE + "```\n", encoding="utf-8")
    out = run(str(tmp_path), "README.md")
    assert out.returncode == 0, out.stdout
    assert "directives=0" in out.stdout, out.stdout


def test_the_same_quotation_outside_a_fence_is_still_refused(tmp_path):
    """So the control above is not licensing a way to assert a directive by
    calling it a quotation."""
    doc = tmp_path / "README.md"
    doc.write_text("# R\n\nWithdrawn, quoted verbatim below.\n\n"
                   + WITHDRAWN_PUBLIC_DIRECTIVE, encoding="utf-8")
    out = run(str(tmp_path), "README.md")
    assert out.returncode == 1, out.stdout


def test_the_real_tree_has_no_unsourced_operator_directive():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout, out.stdout


def test_the_withdrawal_note_cites_the_record_that_contradicts_it():
    """The withdrawn "remain private" directive is not merely unsourced: a sourced
    record in the tree says the opposite. The note must cite it, or the reader has
    only my word for the contradiction."""
    with open(os.path.join(ROOT, "governance", "GIT_ADAPTATION.md"),
              encoding="utf-8") as handle:
        text = handle.read()
    cited = ("drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/"
             "DRIVE_GITHUB_EXECUTION_HANDOFF.md")
    assert cited in text, "the withdrawal note does not cite the contradicting record"
    with open(os.path.join(ROOT, cited), encoding="utf-8") as handle:
        record = handle.read()
    assert "Preserve that visibility" in record, "the cited record does not say that"
    assert "restrict who works in it" in record
