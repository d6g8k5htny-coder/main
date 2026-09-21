"""Negative controls for tools/mirror_quotes_check.py.

Every control runs the checker through its CLI against a synthetic root, so a
path or a threshold bound at import time cannot silently re-check the good
repository -- the defect CLAUDE.md records, found once already in
tools/claims_check.py.

Each control mutates a README the way a real misquote does -- a dropped clause,
an added terminal period, a period for the source's semicolon, straight quotes
for the source's curly ones, an invented separator standing in for a line break,
a transliterated formula -- and asserts the checker refuses it.  If any of these
starts passing, the checker has stopped checking.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "mirror_quotes_check.py")


def run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, CHECKER, *args],
                          capture_output=True, text=True, cwd=cwd)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# The stored object, exactly as a Drive export would land: a banner whose
# fields sit on separate lines, a semicolon mid-sentence, curly quotation
# marks, LaTeX, and no terminal period on the last field.
SOURCE = (
    "CL-TEST-001 — A SYNTHETIC CARRIER\n"
    "\n"
    "STATUS: PROPOSED\n"
    "AUTHORITY: none\n"
    "PROVENANCE: recorded by the synthetic operator, session of 2026-09-20\n"
    "CANONICAL IMPACT: NONE — no theorem in the tree is weakened by any item;\n"
    "every item is a label or provenance defect with conservative polarity.\n"
    "\n"
    "The operator returns HOLD / NOT YET ELIGIBLE, not “ask the author.”\n"
    "There exists a finite constant \\(C<\\infty\\) with \\(0\\le 1-q(r)\\le Cr^3\\).\n"
)

GOOD_README = """# Mirror of a synthetic lane

## The status banners, verbatim

- `CL-TEST-001.md` — *"STATUS: PROPOSED"*, *"AUTHORITY: none"*,
  *"PROVENANCE: recorded by the synthetic operator, session of 2026-09-20"* and
  *"CANONICAL IMPACT: NONE — no theorem in the tree is weakened by any item;
  every item is a label or provenance defect with conservative polarity."*
- The same file: *"The operator returns HOLD / NOT YET ELIGIBLE, not “ask the author.”"*
- Its displayed bound: *"There exists a finite constant \\(C<\\infty\\) with \\(0\\le 1-q(r)\\le Cr^3\\)."*

## What this does not establish

Mirroring is not review. No claim, premise or obligation moves here.
"""


@pytest.fixture
def lane(tmp_path):
    """A one-object synthetic mirror lane whose README quotes it correctly."""
    root = tmp_path / "repo"
    base = root / "drive" / "mirrors" / "SYNTHETIC_LANE"
    write(str(base / "CL-TEST-001.md"), SOURCE)
    write(str(base / "_MANIFEST.jsonl"),
          json.dumps({"id": "1synthetic", "title": "CL-TEST-001.md",
                      "sha256": "0" * 64, "bytes": len(SOURCE.encode()),
                      "exact": True, "stored": True}) + "\n")
    write(str(base / "README.md"), GOOD_README)
    write(str(root / "registers" / "json" / "alpha.json"),
          json.dumps({"tab": "alpha", "header": ["Cell"],
                      "rows": [["A register cell quoted by a mirror README."]]}))
    write(str(root / "drive" / "inventory.jsonl"),
          json.dumps({"id": "1synthetic", "title": "CL-TEST-001.md",
                      "drive_path": "SYNTHETIC_LANE/CL-TEST-001.md"}) + "\n")
    write(str(root / "CLAUDE.md"), "Nothing under legacy/ may be cited as evidence.\n")
    return root


def readme(root):
    return str(root / "drive" / "mirrors" / "SYNTHETIC_LANE" / "README.md")


def mutate(root, old, new):
    path = readme(root)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert text.count(old) == 1, (old, text.count(old))
    write(path, text.replace(old, new))


def check(root):
    return run("--root", str(root))


# ---------------------------------------------------------------------------
# the unmutated lane passes, and the passing is not vacuous
# ---------------------------------------------------------------------------

def test_control_the_unmutated_lane_passes(lane):
    out = check(lane)
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


def test_the_lane_really_is_being_checked(lane):
    """A green run on zero fragments would pass every control below vacuously."""
    out = run("--root", str(lane), "--list")
    assert out.returncode == 0
    assert "readmes=1" in out.stdout
    fragments = int(out.stdout.split("fragments=")[1].split()[0])
    assert fragments >= 4, out.stdout


# ---------------------------------------------------------------------------
# one control per shape of misquote this repository has actually committed
# ---------------------------------------------------------------------------

def test_a_dropped_clause_is_refused(lane):
    """The real defect: a quote ends at the source's semicolon, substitutes a
    period, and drops the continuation with no ellipsis."""
    mutate(lane,
           "weakened by any item;\n  every item is a label or provenance defect with conservative polarity.",
           "weakened by any item.")
    out = check(lane)
    assert out.returncode == 1
    assert "quoted text is in no stored byte" in out.stdout


def test_an_added_terminal_period_is_refused(lane):
    """The source's PROVENANCE line ends without a period; adding one is not verbatim."""
    mutate(lane, "session of 2026-09-20\"*", "session of 2026-09-20.\"*")
    out = check(lane)
    assert out.returncode == 1 and "session of 2026-09-20." in out.stdout


def test_a_short_banner_field_is_now_checked(lane):
    """`"AUTHORITY: none"` is 15 characters. Under the old 24-character floor,
    corrupting it passed. The floor is 12 and it does not."""
    mutate(lane, '*"AUTHORITY: none"*', '*"AUTHORITY: none."*')
    out = check(lane)
    assert out.returncode == 1 and "AUTHORITY: none." in out.stdout


def test_a_fragment_under_the_floor_is_still_not_checked(lane):
    """A documented limit, pinned so nobody mistakes silence for a guarantee.

    Below 12 characters a quoted run is short enough to appear in the corpus by
    coincidence, which would weaken the check rather than strengthen it. The
    floor is measured: on this repository the only fragments it forgives are the
    repository's own shorthand, and a control below re-derives that.
    """
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nA short invention: *"NOT SO"*.\n')
    assert check(lane).returncode == 0
    out = run("--root", str(lane), "--min", "4")
    assert out.returncode == 1 and "NOT SO" in out.stdout


def test_an_invented_separator_for_a_line_break_is_refused(lane):
    """The source has STATUS and AUTHORITY on separate lines; joining them with
    a middot invents a rendering the Drive object does not carry."""
    mutate(lane, '*"STATUS: PROPOSED"*, *"AUTHORITY: none"*',
           '*"STATUS: PROPOSED · AUTHORITY: none"*')
    out = check(lane)
    assert out.returncode == 1 and "STATUS: PROPOSED · AUTHORITY: none" in out.stdout


def test_straight_quotes_for_the_sources_curly_ones_are_refused(lane):
    mutate(lane, "not “ask the author.”", 'not \\"ask the author.\\"')
    out = check(lane)
    assert out.returncode == 1


def test_a_transliterated_formula_is_refused(lane):
    """LaTeX rewritten as plain text reads better and is not what the file says."""
    mutate(lane, "There exists a finite constant \\(C<\\infty\\) with \\(0\\le 1-q(r)\\le Cr^3\\).",
           "There exists a finite constant C < ∞ with 0 ≤ 1 − q(r) ≤ Cr³.")
    out = check(lane)
    assert out.returncode == 1 and "C < ∞" in out.stdout


def test_a_wholly_invented_quotation_is_refused(lane):
    mutate(lane, '*"AUTHORITY: none"*',
           '*"AUTHORITY: the operator has signed this off in full"*')
    out = check(lane)
    assert out.returncode == 1 and "the operator has signed this off" in out.stdout


def test_a_quote_of_a_read_but_unstored_object_is_refused(lane):
    """The case that has no allowlist: quoting bytes the repository does not hold."""
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nRead as an export, not stored: *"TERMINAL STATUS: COMPLETE '
                     'AND RATIFIED BY THE OPERATOR"*.\n')
    out = check(lane)
    assert out.returncode == 1 and "TERMINAL STATUS: COMPLETE" in out.stdout


# ---------------------------------------------------------------------------
# and one per thing the rule deliberately forgives, so nobody tightens it by
# accident and nobody thinks it proves more than it does
# ---------------------------------------------------------------------------

def test_markdown_soft_wrapping_is_forgiven(lane):
    mutate(lane, '*"STATUS: PROPOSED"*', '*"STATUS:\n  PROPOSED"*')
    assert check(lane).returncode == 0


def test_an_ellipsis_elides_and_each_side_is_still_checked(lane):
    mutate(lane,
           '*"CANONICAL IMPACT: NONE — no theorem in the tree is weakened by any item;\n'
           '  every item is a label or provenance defect with conservative polarity."*',
           '*"CANONICAL IMPACT: NONE — no theorem in the tree is weakened […] '
           'with conservative polarity."*')
    assert check(lane).returncode == 0
    mutate(lane, "with conservative polarity.", "with aggressive polarity.")
    out = check(lane)
    assert out.returncode == 1 and "aggressive polarity" in out.stdout


def test_an_unquoted_blockquote_banner_is_checked(lane):
    """The other way this repository presents a verbatim banner. Quotation-mark
    extraction cannot see it, so the whole block is one fragment."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write("\n> CANONICAL IMPACT: NONE \u2014 no theorem in the tree is weakened by any item;\n"
                     "> every item is a label or provenance defect with conservative polarity.\n")
    assert check(lane).returncode == 0
    mutate(lane, "> every item is a label or provenance defect with conservative polarity.",
           "> every item is a label or provenance defect with aggressive polarity.")
    out = check(lane)
    assert out.returncode == 1 and "aggressive polarity" in out.stdout


def test_a_blockquote_carrying_quotation_marks_is_left_to_them(lane):
    """There the marks say which part is transcription and which is framing, so
    the block as a whole must not be required to match."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write("\n> The operator wrote, in a memo this repository does not hold:\n"
                     '> *"CANONICAL IMPACT: NONE \u2014 no theorem in the tree is weakened by any item;\n'
                     '> every item is a label or provenance defect with conservative polarity."*\n')
    assert check(lane).returncode == 0, check(lane).stdout
    mutate(lane, "> every item is a label or provenance defect with conservative polarity.\"*",
           "> every item is a label or provenance defect with aggressive polarity.\"*")
    out = check(lane)
    assert out.returncode == 1 and "aggressive polarity" in out.stdout


def test_an_unquoted_blockquote_elides_with_an_ellipsis_too(lane):
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write("\n> CANONICAL IMPACT: NONE \u2014 no theorem in the tree is weakened [\u2026]\n"
                     "> with conservative polarity.\n")
    assert check(lane).returncode == 0, check(lane).stdout


def test_a_table_row_is_not_scanned(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\n| object | note |\n|---|---|\n| x | "a tabulated phrase of ample length" |\n')
    assert check(lane).returncode == 0


def test_a_fenced_block_is_not_scanned(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\n```json\n{"key": "a fenced value of quite sufficient length"}\n```\n')
    assert check(lane).returncode == 0


def test_a_short_quoted_word_is_below_the_threshold(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nThe word *"PROPOSED"* is a status.\n')
    assert check(lane).returncode == 0


def test_the_threshold_is_a_real_knob(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nA short invention: *"NOPE"*.\n')
    assert check(lane).returncode == 0
    out = run("--root", str(lane), "--min", "4")
    assert out.returncode == 1 and "NOPE" in out.stdout


def test_a_register_cell_is_a_legitimate_source(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nThe register says *"A register cell quoted by a mirror README."*\n')
    assert check(lane).returncode == 0


def test_an_inventory_title_is_a_legitimate_source(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nThe inventory path *"SYNTHETIC_LANE/CL-TEST-001.md"* is indexed.\n')
    assert check(lane).returncode == 0


def test_claude_md_is_a_legitimate_source(lane):
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nRule 10: *"Nothing under legacy/ may be cited as evidence."*\n')
    assert check(lane).returncode == 0


def test_dropping_claude_md_from_the_corpus_makes_that_quote_fail(lane):
    """Confirms the previous control passes because of the corpus entry and not
    because the fragment was never checked."""
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('\nRule 10: *"Nothing under legacy/ may be cited as evidence."*\n')
    out = run("--root", str(lane), "--governance", "NO_SUCH_FILE.md")
    assert out.returncode == 1 and "Nothing under legacy/" in out.stdout


def test_the_readme_itself_is_not_its_own_source(lane):
    """A README that quotes itself would make every invention self-justifying."""
    path = readme(lane)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\nThis sentence is long enough to be a fragment on its own.\n"
                     'And here it is quoted: *"This sentence is long enough to be a '
                     'fragment on its own."*\n')
    out = check(lane)
    assert out.returncode == 1 and "long enough to be a fragment" in out.stdout


def test_a_manifest_note_is_a_legitimate_source_and_is_counted_as_circular(lane):
    """Manifests are repository-authored metadata, and READMEs legitimately quote
    their notes -- "the manifest records X" is a true, checkable statement. But a
    quotation whose ONLY source is a manifest is the repository quoting itself one
    step removed, not a transcription of any Drive byte. It passes and is counted,
    so the circularity is in the summary line rather than hidden inside a pass."""
    base = lane / "drive" / "mirrors" / "SYNTHETIC_LANE"
    with open(base / "_MANIFEST.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "1other", "stored": False,
                                 "note": "Tree-only index row; the payload was never fetched."}) + "\n")
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nThe manifest records *"Tree-only index row; the payload was never fetched."*\n')
    out = check(lane)
    assert out.returncode == 0
    assert "manifest_only=1" in out.stdout, out.stdout


def test_a_quote_carried_by_a_stored_payload_is_not_counted_as_circular(lane):
    """Confirms the count above means what it says, and is not just counting
    every quote that a manifest happens to mention."""
    out = check(lane)
    assert out.returncode == 0 and "manifest_only=0" in out.stdout, out.stdout


# ---------------------------------------------------------------------------
# the disclosure convention: a correction says what it used to say, in quotes
# ---------------------------------------------------------------------------
#
# "Until 2026-09-20 this read ..." deliberately quotes text that is in no stored
# byte -- the old, wrong wording.  The checker exempts a fragment after such a
# marker in its own paragraph and counts it, so the exemption shows in the
# summary line.  These controls pin how far it reaches.

PRIOR = ('\nUntil 2026-09-20 this sentence read "AUTHORITY: none, and the operator '
         'has already signed it off in full".\n')


def test_a_quote_of_prior_wording_is_exempt_and_counted(lane):
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write(PRIOR)
    out = check(lane)
    assert out.returncode == 0, out.stdout
    assert "prior_wording=1" in out.stdout, out.stdout


def test_the_same_quote_without_the_marker_is_refused(lane):
    """Confirms the control above passes because of the marker, not by accident."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write(PRIOR.replace("Until 2026-09-20 this sentence read", "The source says"))
    out = check(lane)
    assert out.returncode == 1 and "signed it off in full" in out.stdout


def test_a_marker_does_not_exempt_a_quote_earlier_in_its_paragraph(lane):
    """A disclosure at the end of a paragraph must not licence an invention at
    the start of it."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nThe file says "AUTHORITY: the operator has signed this off in full", '
                     'and it does not. Until 2026-09-20 this read "something else entirely '
                     'and at sufficient length".\n')
    out = check(lane)
    assert out.returncode == 1 and "signed this off in full" in out.stdout


def test_a_marker_does_not_reach_into_the_next_paragraph(lane):
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nUntil 2026-09-20 this read "an older wording of ample length here".\n'
                     '\nThe file says "AUTHORITY: the operator approved it, in writing".\n')
    out = check(lane)
    assert out.returncode == 1 and "the operator approved it" in out.stdout
    assert "an older wording" not in out.stdout


def test_the_exemption_is_reported_in_the_listing(lane):
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write(PRIOR)
    out = run("--root", str(lane), "--list")
    assert "[prior wording]" in out.stdout, out.stdout


def test_the_marker_needs_a_date(lane):
    """A bare "until" is ordinary English and must not exempt anything."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nUntil recently this read "AUTHORITY: none, and the operator '
                     'has already signed it off in full".\n')
    out = check(lane)
    assert out.returncode == 1 and "signed it off in full" in out.stdout


def test_a_bare_three_dot_ellipsis_splits_too(lane):
    """A seam quote marks its elision with three dots as often as with a real
    ellipsis character, and both mean the same thing. Each side is still checked."""
    mutate(lane,
           '*"CANONICAL IMPACT: NONE \u2014 no theorem in the tree is weakened by any item;\n'
           '  every item is a label or provenance defect with conservative polarity."*',
           '*"CANONICAL IMPACT: NONE \u2014 no theorem in the tree is weakened...\n'
           '  every item is a label or provenance defect with conservative polarity."*')
    assert check(lane).returncode == 0, check(lane).stdout
    mutate(lane, "with conservative polarity.", "with aggressive polarity.")
    out = check(lane)
    assert out.returncode == 1 and "aggressive polarity" in out.stdout


def test_the_claim_graph_is_a_legitimate_source(lane):
    """Lane READMEs quote claims/graph.json by name where a mirrored banner and
    the claim graph disagree about a lemma's status."""
    write(str(lane / "claims" / "graph.json"),
          json.dumps({"nodes": [{"id": "D3-LEMMA", "note":
                                 "used at the rung only and is precisely stated, NOT closed"}]}))
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\n`claims/graph.json` adds: *"used at the rung only and is '
                     'precisely stated, NOT closed"*.\n')
    assert check(lane).returncode == 0
    out = run("--root", str(lane), "--governance", "CLAUDE.md")
    assert out.returncode == 1 and "precisely stated" in out.stdout


def test_a_named_governance_document_is_a_legitimate_source(lane):
    """A lane README that says governance/GIT_ADAPTATION.md records X, and quotes
    X, is making a checkable statement about a file here. Nothing else under
    governance/ is in the corpus."""
    write(str(lane / "governance" / "GIT_ADAPTATION.md"),
          "| Fresh Start control plane | **No repository equivalent.** The lane is indexed only. |\n")
    write(str(lane / "governance" / "README.md"),
          "A different governance file, carrying a sentence of its own about the lane.\n")
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\n`governance/GIT_ADAPTATION.md` records *"No repository equivalent. '
                     'The lane is indexed only."* for the control plane.\n')
    assert check(lane).returncode == 0, check(lane).stdout
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\n`governance/README.md` says *"A different governance file, carrying '
                     'a sentence of its own about the lane."*\n')
    out = check(lane)
    assert out.returncode == 1 and "A different governance file" in out.stdout


def test_a_present_tense_sentence_about_the_world_is_not_a_marker(lane):
    """"The hold runs until 2026-10-01 and the file says X" is not a disclosure,
    and must not exempt the quotation after it."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\nThe hold runs until 2026-10-01 and the file says '
                     '"AUTHORITY: the operator approved it, in writing".\n')
    out = check(lane)
    assert out.returncode == 1 and "the operator approved it" in out.stdout


def test_a_marker_inside_a_quotation_is_the_sources_words(lane):
    """The real defect this caught: a mirrored banner reading "Kimi is dark until
    2026-09-30" was read as a disclosure marker and exempted every quotation in
    the nine bullets after it."""
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\n- The file says *"the snapshot is frozen until 2026-09-30, and final"*.\n'
                     '- The next bullet says *"AUTHORITY: the operator approved it, in writing"*.\n')
    out = check(lane)
    assert out.returncode == 1 and "the operator approved it" in out.stdout


def test_a_marker_does_not_reach_past_its_own_bullet(lane):
    with open(readme(lane), "a", encoding="utf-8") as handle:
        handle.write('\n- Until 2026-09-20 this bullet read "an older wording of ample length".\n'
                     '- The next bullet says *"AUTHORITY: the operator approved it, in writing"*.\n')
    out = check(lane)
    assert out.returncode == 1 and "the operator approved it" in out.stdout
    assert "an older wording" not in out.stdout


# ---------------------------------------------------------------------------
# the real repository
# ---------------------------------------------------------------------------


def test_every_disclosure_marker_in_the_tree_matches_the_verb_list():
    """The verb list claims to be derived from the markers actually written
    here. This fails if a real marker stops being recognised, which would
    silently turn its disclosure back into a reported defect."""
    sys.path.insert(0, ROOT)
    try:
        from tools import mirror_quotes_check as mq
    finally:
        sys.path.pop(0)
    rejected = []
    for path in mq.readmes(ROOT, mq.SCAN_RELS):
        text = mq.readable(path) or ""
        quoted = [(m.start(), m.end())
                  for pattern in (mq._STRAIGHT, mq._CURLY)
                  for m in pattern.finditer(text)]
        for found in mq._PRIOR_WORDING.finditer(text):
            if any(a <= found.start() < b for a, b in quoted):
                continue
            window = text[found.end():found.end() + mq._DISCLOSURE_WINDOW]
            if not mq._DISCLOSURE_VERB.search(window):
                rejected.append((os.path.relpath(path, ROOT),
                                 " ".join(window.split())[:90]))
    # Markers that are ordinary prose about a date are expected to be rejected;
    # a marker followed by a verb of saying is not. Assert on the shape, not a
    # count, so a new disclosure phrasing shows up here rather than in the
    # problem list.
    for rel, window in rejected:
        assert not window.lower().startswith(("this ", "the ")) or "not the" in window.lower(), \
            f"unrecognised disclosure phrasing in {rel}: {window}"

def test_every_mirror_readme_quotation_is_verbatim():
    out = run()
    assert out.returncode == 0, out.stdout
    assert "problems=0" in out.stdout


def test_the_real_scan_is_not_vacuous():
    out = run()
    readmes = int(out.stdout.split("readmes=")[1].split()[0])
    fragments = int(out.stdout.split("fragments=")[1].split()[0])
    assert readmes >= 20 and fragments >= 700, out.stdout


def test_the_floor_is_where_the_measurement_puts_it():
    """The floor is 12 because that is the lowest value at which this repository
    is clean. If a future change makes 12 fail, the fix is the quotation, not the
    floor -- and if 10 is also clean the floor can come down, deliberately."""
    assert run("--min", "12").returncode == 0, run("--min", "12").stdout
    lower = run("--min", "8")
    if lower.returncode == 0:
        pytest.skip("the tree is clean at 8 too; lower the floor deliberately")
    assert "quoted text is in no stored byte" in lower.stdout


def test_the_disclosure_exemption_stays_a_small_minority():
    """If most of the corpus became "prior wording" the checker would be
    checking almost nothing. The bound is the enforced statement; the exact
    pair moves with every correction and is deliberately not written down."""
    out = run()
    fragments = int(out.stdout.split("fragments=")[1].split()[0])
    prior = int(out.stdout.split("prior_wording=")[1].split()[0])
    assert prior < fragments // 4, out.stdout
