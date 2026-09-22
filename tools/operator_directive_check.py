#!/usr/bin/env python3
"""An operator directive asserted in this repository must cite its source.

Dylan Roy is the single final authority for canonical promotion, external
release, permanent deletion and machine-root replacement.  This repository
records his decisions; it does not make them, and it must never be the only
place one of them exists.  The convention the corpus already follows is that
every operator decision is quoted with its source: a Drive id, an artifact id
(``OP-PROT-012``, ``GP-AUD-187``), or a path under ``registers/`` or
``drive/``.

Nothing enforced that.  On 2026-09-21 a third-party branch rewrote the
repository-visibility paragraph of ``governance/GIT_ADAPTATION.md`` to read
"Dylan subsequently directed that the repository remain private until
publication is explicitly authorized", deleting a sourced sentence that said
the opposite.  No source for the directive existed anywhere in the tree, and
the accompanying present-tense status claim was false: the repository is
public.  Every checker passed, because none of them was looking at governance
prose for invented authority.

WHAT THIS CHECKS.  In each governed document, a SENTENCE asserting that the
operator directed / instructed / decided / authorised / approved / ordered /
ruled something must carry a citation in that sentence or the one after it.

Sentence granularity is the point, and it was learned the hard way: the first
version of this checker worked per paragraph and did not catch the very edit
it was written for.  The invented directive had been appended to a long,
heavily cited paragraph about a different, real decision, so the paragraph
contained citations and passed.  Authority does not rub off on the sentence
next to it.

A bare date is NOT a citation: "confirmed on 2026-09-20 UTC" is an observation
someone made, not a record anyone else can look up.

Fenced code blocks are removed before sentences are cut. A filename is not an
assertion about anybody, and `tools/operator_directive_check.py` listed in a
bash block contains both "operator" and "directive".

WHAT THIS DOES NOT ESTABLISH.  That a cited directive is real, that it says
what the paragraph says it says, that the citation resolves, or that an
uncited paragraph is false.  This is a presence check on sourcing, not on
truth.  It reads no claim, grades nothing, moves no gate, and a pass here
authorizes nothing.  Only the operator decides what the operator decided.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Prose in which an operator directive carries weight.  Explicit, so that
#: adding a file is a deliberate act a reviewer can see in the diff.
GOVERNED_DOCS = (
    os.path.join("governance", "GIT_ADAPTATION.md"),
    os.path.join("governance", "README.md"),
    os.path.join("docs", "RESEARCH_MAP.md"),
    os.path.join("docs", "OPEN_PROBLEMS.md"),
    os.path.join("docs", "FINDINGS_2026-09-18.md"),
    "README.md",
)

#: "the operator did X" -- the subject and the verb in one sentence.
DIRECTIVE = re.compile(
    r"\b(?:Dylan(?:\s+Roy)?(?:'s)?|the\s+operator(?:'s)?|operator(?:'s)?)\b"
    r"[^.!?]{0,140}?"
    r"\b(?:directed|instructed|decided|authorised|authorized|approved|ordered|ruled"
    r"|instruction|directive|direction)\b",
    re.IGNORECASE)

#: What counts as a source someone else can go and read.  Deliberately NOT a
#: date: a date is when a thing was observed, not a record of it.
CITATION = re.compile(
    r"\b[01][A-Za-z0-9_-]{25,}\b"                     # a Drive file id
    r"|\b(?:OP-PROT|OP-GDN|OP-CNS|OD-OP)-\d"          # operator protocol / decision
    r"|\b(?:GP|CL|LS|S2|AO48|TRC|EC|PKG|DQ)-[A-Z]{2,4}-\d"   # artifact id
    r"|registers/|drive/mirrors/|drive/deltas/|claims/"
    # A cross-reference to the governed file where the id lives is a
    # citation too: docs/RESEARCH_MAP.md quotes the Board decision verbatim
    # and points at governance/GIT_ADAPTATION.md for its Drive id, which is
    # sourcing, not hand-waving. Requiring the id inline everywhere would
    # push noise into prose to satisfy a regex.
    r"|governance/[A-Za-z0-9_.-]+\.md|docs/[A-Za-z0-9_.-]+\.md")

#: A paragraph may opt out by saying, in the same paragraph, that it is NOT
#: reporting a directive.  The phrase is fixed so the exemption is greppable.
DISCLAIMER = "no operator directive is asserted here"


#: End of sentence: . ! or ? then whitespace then a capital or a quote.  Not
#: perfect English, and it does not need to be -- a split that occasionally
#: joins two sentences only makes the check more permissive, never less.
SENTENCE_END = re.compile(r"(?<=[.!?])[)\]\"'\u201d`*]*\s+(?=[\"“(*`A-Z])")


#: A fenced code block. Its contents are commands and filenames, not prose, and
#: a filename is not an assertion about anybody. `README.md`'s "What CI
#: enforces" block lists `tools/operator_directive_check.py`, which contains
#: both "operator" and "directive" and was read as an operator directive by
#: this checker's first version -- passing only because the same block names
#: `registers/`, which counts as a citation. A false positive that passes is
#: worse than one that fails: it inflates the directive count and teaches a
#: reader to ignore it. Blocks are removed before sentences are cut, with their
#: newlines kept so reported line numbers stay right.
CODE_FENCE = re.compile(r"^(?P<fence>```+|~~~+).*?^(?P=fence)[ \t]*$",
                        re.S | re.M)


def strip_code_fences(text: str) -> str:
    """Blank out fenced blocks, preserving line count."""
    def blank(m):
        return "\n" * m.group(0).count("\n")
    return CODE_FENCE.sub(blank, text)


def sentences(text: str):
    """(1-based line of the sentence, sentence text) over the whole document.

    Line numbers are approximate to the line the sentence starts on, which is
    all a reader needs to find it.
    """
    out = []
    line_no = 1
    buf, start = [], 1
    for raw in text.splitlines():
        if not buf and not raw.strip():
            # A blank line before any content of the next sentence. Skipping it
            # is what lets `start` advance past a paragraph break -- and past a
            # blanked-out code fence, whose lines are all blank. Without this,
            # every sentence in a document that opens with a fence is reported
            # at line 1.
            line_no += 1
            continue
        if not buf:
            start = line_no
        buf.append(raw)
        joined = " ".join(" ".join(buf).split())
        parts = SENTENCE_END.split(joined)
        if len(parts) > 1:
            for part in parts[:-1]:
                out.append((start, part))
            buf, start = [parts[-1]], line_no
        line_no += 1
    tail = " ".join(" ".join(buf).split())
    if tail:
        out.append((start, tail))
    return out


def audit(root: str, docs=None):
    """Return (documents scanned, directive sentences, problems)."""
    docs = GOVERNED_DOCS if docs is None else tuple(docs)
    scanned = directives = 0
    problems = []
    for rel in docs:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        scanned += 1
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        items = sentences(strip_code_fences(text))
        for index, (start, sentence) in enumerate(items):
            if not DIRECTIVE.search(sentence):
                continue
            window = sentence
            if index + 1 < len(items):
                # A citation that trails into the next sentence still counts.
                window += " " + items[index + 1][1]
            if DISCLAIMER in " ".join(window.lower().split()):
                continue
            directives += 1
            if not CITATION.search(window):
                hit = DIRECTIVE.search(sentence)
                lo = max(0, hit.start() - 30)
                excerpt = ("..." if lo else "") + sentence[lo:hit.end() + 90]
                problems.append(
                    f"{rel}:{start}: asserts an operator directive with no source in the "
                    f"sentence or the one after it. Quote it with its Drive id or artifact "
                    f"id, or say {DISCLAIMER!r}: {excerpt}")
    return scanned, directives, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--doc", action="append", default=[],
                        help="check this document instead of the governed set")
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else ROOT
    scanned, directives, problems = audit(root, args.doc or None)
    for item in problems:
        print(item)
    print(f"operator_directive_check: docs={scanned} directives={directives} "
          f"problems={len(problems)}")
    print("A pass here means a directive names a source, not that the directive is real, "
          "says what the paragraph says, or authorizes anything. Only the operator decides "
          "what the operator decided.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
