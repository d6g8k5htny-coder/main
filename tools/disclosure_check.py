#!/usr/bin/env python3
"""A correction that says what it used to say must quote that correctly.

``tools/mirror_quotes_check.py`` requires every quotation in a Drive-mirror
README to be verbatim in a stored byte, with one exemption: this repository's
disclosure convention, where a corrected passage records its own former wording.

    Until 2026-09-20 this sentence read "there exists a finite constant
    `C_Q0 < ∞` such that …".

Those quotation marks are deliberately around text that is in no stored byte --
that is the point of them -- so the quote checker steps over them and counts
them.  This checker closes the gap the exemption opens, in both directions.  It takes
each of those fragments and requires it to be verbatim in the **previous
committed version of the same file**; and it takes every quotation the committed
version carried whose wording no longer appears in the working tree, and requires
the file to record a correction.  The first stops a note from inventing a
history.  The second stops a correction from erasing one silently.  A disclosure note that misquotes the thing it is disclosing is
worse than the defect it records, because it fabricates a history the reader
cannot check.

Why this is a pre-commit check and not a CI step
------------------------------------------------
It compares the working tree against ``git show <rev>:<path>``, so it is correct
exactly while the correction is uncommitted.  Once committed, the named revision
carries the corrected text and the disclosure necessarily no longer matches it.
Worse than merely unhelpful: once committed, the named revision carries the
disclosure note itself, so the quoted former wording is trivially found inside it
and the check can no longer fail.  A control in ``tests/test_disclosure.py`` pins
that, by committing a fabricated disclosure and showing this tool stops seeing
it.  CI also checks out two commits deep, which is not enough history to resolve
an older disclosure either.  So this runs in the pre-commit block in CLAUDE.md,
against HEAD, before the correction lands -- and its negative controls, which
build their own git history in a temporary directory, do run in CI.

What this does NOT establish: that the correction itself is right, that the new
quotation is verbatim (``tools/mirror_quotes_check.py`` decides that), or that
the disclosure is complete.  A passage can be corrected wrongly, and disclose
that wrong correction perfectly.  Nothing here reads, grades or moves any claim,
premise or obligation.

Run:  python3 tools/disclosure_check.py [--root DIR] [--rev HEAD]
Every path is resolved when ``main()`` runs, never at import time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_RELS = (os.path.join("drive", "mirrors"), os.path.join("drive", "deltas"))

# Governed prose that is not a Drive-mirror README.
#
# ``mirror_quotes_check.readmes()`` walks a tree and takes files named
# README.md, which is right for the mirrors and wrong for everything else:
# governance/GIT_ADAPTATION.md and docs/RESEARCH_MAP.md are not READMEs, so
# neither the scan nor the removal scan could ever see them.  Between them
# these five files carry 22 "Until <date>" notes and roughly 150 quoted
# fragments, and none of it was checked -- the disclosure discipline was
# enforced under drive/ and nowhere else.
#
# That is the gap a third-party branch walked through on 2026-09-21: it
# replaced a sourced sentence about this repository's visibility with an
# uncited operator directive, deleted the sourced reading, and recorded no
# disclosure note.  Nothing failed, because nothing was looking.
#
# The list is explicit rather than "every .md outside drive/" so that adding a
# file to it is a deliberate act a reviewer can see.
GOVERNED_DOCS = (
    os.path.join("governance", "GIT_ADAPTATION.md"),
    os.path.join("docs", "RESEARCH_MAP.md"),
    os.path.join("docs", "OPEN_PROBLEMS.md"),
    os.path.join("docs", "FINDINGS_2026-09-18.md"),
    "README.md",
)


def documents(quotes, root, scan_rels):
    """Every file this checker governs: the mirror READMEs, plus GOVERNED_DOCS.

    Resolved when this runs, never at import time.
    """
    found = list(quotes.readmes(root, scan_rels))
    for rel in GOVERNED_DOCS:
        path = os.path.join(root, rel)
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return sorted(found)


def _quotes_module():
    """Import the quote checker by path, so the two tools share one definition
    of what a fragment is and what counts as a disclosure marker."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mirror_quotes_check.py")
    spec = importlib.util.spec_from_file_location("mirror_quotes_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def committed_text(root, rel, rev):
    """The file's content at `rev`, or None when git has no such path there."""
    done = subprocess.run(["git", "show", f"{rev}:{rel}"],
                          capture_output=True, text=True, cwd=root)
    return done.stdout if done.returncode == 0 else None


def removed_quotations(quotes, old_text, new_text, min_fragment):
    """Quotations the committed version carried and the working tree no longer does.

    A quotation that was merely extended -- the dropped clause restored -- is
    still a substring of the new one, so it is not "removed"; only a quotation
    whose wording actually changed counts.  Emphasis and whitespace are already
    normalised away by the shared extractor, so re-bolding a phrase is invisible
    here and has to be recorded by a human note instead.
    """
    was = {frag for _line, frag, prior in quotes.fragments(old_text, min_fragment) if not prior}
    now = [frag for _line, frag, prior in quotes.fragments(new_text, min_fragment) if not prior]
    disclosed = {frag for _line, frag, prior in quotes.fragments(new_text, min_fragment) if prior}
    gone = []
    for frag in sorted(was):
        if any(frag in current for current in now):
            continue
        if any(frag in note for note in disclosed):
            continue
        gone.append(frag)
    return gone


def scan(root, scan_rels, rev, min_fragment=None):
    quotes = _quotes_module()
    if min_fragment is None:
        min_fragment = quotes.MIN_FRAGMENT
    total = checked = new_file = 0
    problems = []
    for path in documents(quotes, root, scan_rels):
        text = quotes.readable(path)
        if text is None:
            continue
        rel = os.path.relpath(path, root)
        prior = [(line, frag) for line, frag, is_prior
                 in quotes.fragments(text, min_fragment) if is_prior]
        if not prior:
            continue
        total += len(prior)
        old = committed_text(root, rel, rev)
        if old is None:
            # A README added in this change cannot be disclosing its own past.
            new_file += len(prior)
            for line, frag in prior:
                problems.append((rel, line, "the file is new at "
                                 f"{rev}, so it has no former wording to disclose", frag))
            continue
        haystack = quotes.normalise(quotes.strip_emphasis(old))
        for line, frag in prior:
            checked += 1
            if frag not in haystack:
                problems.append((rel, line, f"not verbatim in {rev}", frag))
    return total, checked, new_file, problems


def scan_removals(root, scan_rels, rev, min_fragment=None):
    """Every tracked README: a quotation whose wording changed must be recorded.

    Two cases have to be told apart, and only one of them can be decided
    mechanically.

    Some corrections *remove* quotation marks on purpose -- the object was read
    and never stored, so no byte here carries its words and quoting them at all
    was the defect.  Such a correction cannot disclose itself by quoting the old
    wording, because putting those words back in quotation marks is exactly what
    it is fixing.  Demanding a quoted note there would push an author back into
    the dishonest form, so a changed quotation in a file that *does* record
    corrections is counted and printed, not failed: a reader judges whether the
    prose says enough.

    A changed quotation in a file that records no correction at all is a
    different thing: published text was rewritten and the file says nothing about
    it.  That fails.
    """
    quotes = _quotes_module()
    if min_fragment is None:
        min_fragment = quotes.MIN_FRAGMENT
    files = reworded = 0
    problems, noted = [], []
    for path in documents(quotes, root, scan_rels):
        new_text = quotes.readable(path)
        if new_text is None:
            continue
        rel = os.path.relpath(path, root)
        old = committed_text(root, rel, rev)
        if old is None:
            continue
        files += 1
        gone = removed_quotations(quotes, old, new_text, min_fragment)
        if not gone:
            continue
        records_corrections = any(
            prior for _line, _frag, prior in quotes.fragments(new_text, min_fragment)
        ) or bool(quotes._PRIOR_WORDING.search(new_text))
        for frag in gone:
            reworded += 1
            if records_corrections:
                noted.append((rel, frag))
            else:
                problems.append((rel, 0, f"a quotation in {rev} was reworded and this file "
                                 "records no correction at all", frag))
    return files, reworded, problems, noted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=REPO)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--scan", action="append", default=None)
    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--skip-removals", action="store_true",
                        help="check only that disclosures quote the revision correctly, "
                             "not that every changed quotation carries one")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    scan_rels = tuple(args.scan) if args.scan else SCAN_RELS
    total, checked, new_file, problems = scan(root, scan_rels, args.rev, args.min)
    tracked, reworded, removal_problems, noted = (0, 0, [], [])
    if not args.skip_removals:
        tracked, reworded, removal_problems, noted = scan_removals(
            root, scan_rels, args.rev, args.min)
    problems = problems + removal_problems

    for rel, frag in noted:
        print(f"disclosure_check: {rel}: reworded, and the file records corrections -- "
              f"read them and confirm this one is covered: {frag}")

    for rel, line, why, frag in problems:
        where = f"{rel}:{line}" if line else rel
        print(f"disclosure_check: {where}: {why}: {frag}")

    print(f"disclosure_check: disclosures={total} checked={checked} "
          f"in_a_new_file={new_file} tracked_readmes={tracked} "
          f"reworded={reworded} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
