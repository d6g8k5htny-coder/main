#!/usr/bin/env python3
"""Every quotation in a Drive-mirror README must be verbatim in a stored byte.

The mirror READMEs under ``drive/mirrors/`` and ``drive/deltas/`` carry sections
headed "the status banners, verbatim".  Those quotations are the whole reason
the mirrors exist: they are how a reader learns, without opening the Drive, what
status each mirrored object claims for itself.  A quotation that silently drops
a clause, adds a terminal period, substitutes a period for the source's
semicolon, replaces the source's curly quotation marks with straight ones,
transliterates the source's LaTeX, adds bold the source does not carry, or
invents a separator such as " / " or " · " where the source has a line break is
a misrepresentation of the Drive -- under a heading that promises it is not.

Those are not hypothetical.  Every one of them was found in this repository, by
hand, twice; this checker exists so the next one is found by the build instead.

The rule, deliberately mechanical
---------------------------------
A **fragment** is a run of 12 or more characters inside a pair of quotation
marks -- ``"..."`` or the curly ``“...”`` -- in a README under a scanned root,
split on ellipsis (``[…]``, ``…``, ``[...]``) so an elided quote is checked
piecewise.  Markdown blockquote ``>`` prefixes are stripped first; table rows and
fenced code blocks are skipped.  A fragment **matches** when, after Unicode NFC
normalisation and collapsing every run of whitespace to one space, it is a
substring of some corpus entry.  Markdown emphasis (``**``, backticks) is
stripped from both sides, so *adding* bold is not caught here -- only a human or
a lane verifier catches that.

The corpus is the bytes this repository actually holds:

* every non-README file under the scanned roots -- the stored Drive payloads and
  the manifests that describe them.  A fragment matching ONLY a manifest is the
  repository quoting its own metadata rather than the Drive; that is true but is
  not a transcription, so it passes and is counted as ``manifest_only``;
* every cell of ``registers/json/*.json``;
* every ``title`` / ``name`` / ``drive_path`` / ``path`` in ``drive/inventory.jsonl``;
* three repository documents the lane READMEs quote *by name*: ``CLAUDE.md`` for
  the rules they operate under, ``claims/graph.json`` where a mirrored banner and
  the claim graph disagree about a lemma's status, and
  ``governance/GIT_ADAPTATION.md`` where a lane records that a Drive construct has
  no repository equivalent.  Each is named in the sentence that quotes it, so the
  quotation is a checkable statement about a file here rather than a transcription
  of the Drive; nothing else under ``governance/`` is in the corpus.

There is deliberately **no allowlist**.  If a README quotes an object that was
read but not stored, the honest repair is to say so in prose without quotation
marks, or to store the reading copy -- not to register an exception.  A checker
with an exemption list for the very thing it checks is a checker that stops
working the first time somebody is in a hurry.

What this does NOT establish
----------------------------
Matching proves a string is *present somewhere* in the stored bytes.  It does
not prove the quotation is attributed to the right file, that the surrounding
sentence describes it correctly, that a generalisation over several files
("each notice ends with...") holds of all of them, or that the quoted claim is
true.  Whitespace and emphasis differences are forgiven by construction, so a
reflowed block and an added bold both pass.  Nothing here reads, grades or moves
any claim, premise or obligation: a green run says the quotation marks in these
READMEs are honest about the bytes, and says nothing whatever about the
mathematics the bytes discuss.

Run:  python3 tools/mirror_quotes_check.py [--root DIR] [--scan DIR ...]
      python3 tools/mirror_quotes_check.py --list      # print every fragment
Every path is resolved when ``main()`` runs, never at import time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_RELS = (os.path.join("drive", "mirrors"), os.path.join("drive", "deltas"))
REGISTER_JSON_REL = os.path.join("registers", "json")
INVENTORY_REL = os.path.join("drive", "inventory.jsonl")
# Files outside the scanned roots that a mirror README legitimately quotes: the
# rules it operates under, and the claim graph, which several lane READMEs quote
# by name when a mirrored banner and the graph disagree.
GOVERNANCE_RELS = ("CLAUDE.md",
                   os.path.join("claims", "graph.json"),
                   os.path.join("governance", "GIT_ADAPTATION.md"))
INVENTORY_KEYS = ("title", "name", "drive_path", "path")
MIN_FRAGMENT = 12

# An ellipsis marks elision, bracketed or bare, curly or three dots, so each
# side of it is checked as its own fragment.
_SPLIT = re.compile(r"\[…\]|…|\[\.\.\.\]|\.\.\.")
_STRAIGHT = re.compile(r'"([^"\n]*(?:\n(?!\s*\n)[^"\n]*)*?)"')
_CURLY = re.compile(r"“([^”]*?)”")
_FENCE = re.compile(r"^\s*(```|~~~)")
_BLOCKQUOTE = re.compile(r"^(\s*)>\s?")
# The repository's disclosure convention: a corrected passage says what it used
# to say, and quotes the old wording.  "Until 2026-09-20 this read ..." puts text
# in quotation marks that is deliberately NOT in any stored byte -- that is the
# whole point of it.  A fragment after such a marker, in the same paragraph, is
# exempt from the stored-bytes rule and counted separately, so the exemption is
# visible in the summary line rather than silent.  Its fidelity to the previous
# text is checked by tools/disclosure_check.py, which reads it against
# `git show HEAD:<path>` before the correction is committed -- a pre-commit gate,
# because after the commit the named revision carries the note itself.
_PRIOR_WORDING = re.compile(r"\b[Uu]ntil\s+\d{4}-\d{2}-\d{2}\b")
# A marker only counts when it is this repository speaking about its own former
# text, so it must be followed by a verb of saying.  Without this, the source
# sentence "Kimi is dark until 2026-09-30" -- quoted, inside a bullet list -- read
# as a marker and exempted every quotation in the nine bullets after it.  A
# marker inside quotation marks is the source's words and never a marker at all,
# and the exemption stops at the end of the marker's own bullet.
# Past-tense verbs of saying only: a disclosure is always about what the text
# used to do.  Present-tense forms ("the register says X until 2026-10-01") are
# deliberately absent, because those sentences are about the world, not about
# this document's history.  The list is derived from the markers actually written
# here, and a control fails if any marker in the tree stops matching it.
# "closed" is deliberately absent although it reads as a verb of saying: it is the
# most status-loaded word in this repository, and "the gate stayed open until
# 2026-09-20 and then closed" is prose about the world, not about this document's
# history.  Write "ended" instead; a control refuses an unrecognised phrasing
# rather than letting one silently exempt the quotations after it.
_DISCLOSURE_VERB = re.compile(
    r"\b(read|said|ended|opened|quoted|named|listed|reported|described|carried"
    r"|gave|showed|attributed|departed|transliterated|spliced|dropped|cut|turned"
    r"|omitted|spelled|rendered|stood|used|counted|cited|paraphrased|began|called"
    r"|treated|wrote|put|set|contained|included|lacked|misquoted|understated"
    r"|overstated|was|were|had|did)\b")
_DISCLOSURE_WINDOW = 160
_BULLET = re.compile(r"\n[ \t]*(?:[*+-]|\d+\.)[ \t]")
# An inline code span is a command or a literal, not a quotation: a `"` inside
# one opens no quote.  Backticks are still stripped from BOTH sides when
# matching, so a quotation that merely CONTAINS a backticked token still lines up
# with the plain text of its source -- only a quotation that BEGINS inside a code
# span is skipped.
_CODE_SPAN = re.compile(r"`+[^`\n]*`+")


def normalise(text):
    """NFC, then every run of whitespace becomes one space.

    Markdown soft-wrapping is invisible to a reader, so a quote wrapped across
    README lines is not a departure.  A source *line break* rendered as " · "
    or " / " is a departure, and survives this normalisation to be caught.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def strip_emphasis(text):
    return text.replace("**", "").replace("`", "")


def readable(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def fragments(text, min_fragment):
    """Yield (line_number, fragment, prior_wording) for every quoted run.

    Blockquote markers are stripped, table rows and fenced code blocks are
    blanked out.  Blanking rather than deleting keeps the line numbering exact,
    so a reported line number points at the README line a reader must edit.

    ``prior_wording`` is True when the fragment sits after an "Until <date>"
    marker that is outside quotation marks and followed by a past-tense verb of
    saying, and before the end of that marker's own bullet or paragraph.  That is
    this repository's way of recording what a corrected passage used to say.
    """
    lines = text.split("\n")
    kept, quoted_line, fenced = [], [], False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
            kept.append("")
            quoted_line.append(False)
            continue
        if fenced or line.lstrip().startswith("|"):
            kept.append("")
            quoted_line.append(False)
            continue
        kept.append(_BLOCKQUOTE.sub(r"\1", line))
        quoted_line.append(bool(_BLOCKQUOTE.match(line)))

    joined = "\n".join(kept)
    starts, offset = [], 0
    for line in kept:
        starts.append(offset)
        offset += len(line) + 1

    def line_of(pos):
        low, high = 0, len(starts) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if starts[mid] <= pos:
                low = mid
            else:
                high = mid - 1
        return low + 1

    code_spans = [(m.start(), m.end()) for m in _CODE_SPAN.finditer(joined)]

    def inside_code(pos):
        return any(start <= pos < end for start, end in code_spans)

    # Spans that are inside quotation marks: a marker there is the source
    # speaking, not this repository disclosing its own former text.
    quoted_spans = [(m.start(), m.end()) for pattern in (_STRAIGHT, _CURLY)
                    for m in pattern.finditer(joined) if not inside_code(m.start())]

    def inside_a_quote(pos):
        return any(start <= pos < end for start, end in quoted_spans)

    # Each disclosure marker's reach: from the marker to the end of its own
    # bullet, or of its paragraph when it is not in a list.
    marks = []
    for para in re.finditer(r"(?:\A|\n)[^\n]*(?:\n(?!\s*\n)[^\n]*)*", joined):
        for found in _PRIOR_WORDING.finditer(joined, para.start(), para.end()):
            if inside_a_quote(found.start()):
                continue
            window = joined[found.end():min(found.end() + _DISCLOSURE_WINDOW, para.end())]
            if not _DISCLOSURE_VERB.search(window):
                continue
            nxt = _BULLET.search(joined, found.end(), para.end())
            marks.append((found.end(), nxt.start() if nxt else para.end()))

    # A blockquote carrying no quotation marks is the other way this repository
    # presents a verbatim banner, and quotation-mark extraction cannot see it.
    # Such a block is one fragment.  A block that DOES carry quotation marks is
    # left to the patterns below, because there the marks say which part is the
    # transcription and which is this repository's framing.
    index = 0
    while index < len(kept):
        if not quoted_line[index]:
            index += 1
            continue
        start = index
        while index < len(kept) and quoted_line[index]:
            index += 1
        body = "\n".join(kept[start:index])
        if '"' not in body and "\u201c" not in body:
            for piece in _SPLIT.split(body):
                fragment = normalise(strip_emphasis(piece))
                if len(fragment) >= min_fragment:
                    yield start + 1, fragment, False

    for pattern in (_STRAIGHT, _CURLY):
        for match in pattern.finditer(joined):
            if inside_code(match.start()):
                continue
            line = line_of(match.start())
            after_marker = any(start <= match.start() < end for start, end in marks)
            for piece in _SPLIT.split(match.group(1)):
                fragment = normalise(strip_emphasis(piece))
                if len(fragment) >= min_fragment:
                    yield line, fragment, after_marker


def build_corpus(root, scan_rels, register_rel, inventory_rel, governance_rels):
    """The stored bytes a README may quote, normalised for matching.

    Returns (corpus, manifests).  ``manifests`` is the subset written by this
    repository about the objects it holds, rather than by the Drive: a fragment
    that matches only there is the repository quoting its own metadata, which is
    true but is not a transcription of anything on the Drive.  Keeping them apart
    lets the summary line count that case instead of hiding it inside a pass.
    """
    corpus, manifests = [], []
    for scan_rel in scan_rels:
        base = os.path.join(root, scan_rel)
        for dirpath, _dirs, files in os.walk(base):
            for name in sorted(files):
                if name == "README.md":
                    continue
                text = readable(os.path.join(dirpath, name))
                if text is None:
                    continue
                entry = normalise(strip_emphasis(text))
                corpus.append(entry)
                if name.endswith("_MANIFEST.jsonl") or name == "MANIFEST.jsonl":
                    manifests.append(entry)

    json_dir = os.path.join(root, register_rel)
    if os.path.isdir(json_dir):
        for name in sorted(os.listdir(json_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(json_dir, name), encoding="utf-8") as handle:
                    doc = json.load(handle)
            except (OSError, ValueError):
                continue
            for row in doc.get("rows", []):
                cells = row.values() if isinstance(row, dict) else row
                for cell in cells:
                    if isinstance(cell, str) and cell.strip():
                        corpus.append(normalise(strip_emphasis(cell)))

    inventory = os.path.join(root, inventory_rel)
    if os.path.isfile(inventory):
        with open(inventory, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                for key in INVENTORY_KEYS:
                    if isinstance(record.get(key), str):
                        corpus.append(normalise(strip_emphasis(record[key])))

    for governance_rel in governance_rels:
        text = readable(os.path.join(root, governance_rel))
        if text is not None:
            corpus.append(normalise(strip_emphasis(text)))

    return corpus, manifests


def readmes(root, scan_rels):
    found = []
    for scan_rel in scan_rels:
        base = os.path.join(root, scan_rel)
        for dirpath, _dirs, files in os.walk(base):
            if "README.md" in files:
                found.append(os.path.join(dirpath, "README.md"))
    return sorted(found)


def scan(root, scan_rels, register_rel, inventory_rel, governance_rels,
         min_fragment=None):
    """Return (checked, problems, listing).

    ``min_fragment`` resolves to the module constant *when this runs*, never at
    import time.  A default argument evaluated at import once neutered every
    mutation test in ``tools/claims_check.py``; CLAUDE.md records that bug, and
    this checker does not repeat it.
    """
    if min_fragment is None:
        min_fragment = MIN_FRAGMENT
    corpus, manifests = build_corpus(
        root, scan_rels, register_rel, inventory_rel, governance_rels)
    elsewhere = [entry for entry in corpus if entry not in manifests]
    checked, prior, manifest_only, problems, listing = 0, 0, 0, [], []
    for path in readmes(root, scan_rels):
        text = readable(path)
        if text is None:
            problems.append((path, 0, "README.md is not readable as UTF-8"))
            continue
        rel = os.path.relpath(path, root)
        for line, fragment, prior_wording in fragments(text, min_fragment):
            checked += 1
            listing.append((rel, line, fragment, prior_wording))
            if prior_wording:
                prior += 1
                continue
            if not any(fragment in entry for entry in corpus):
                problems.append((rel, line, fragment))
            elif not any(fragment in entry for entry in elsewhere):
                manifest_only += 1
    return checked, prior, manifest_only, problems, listing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=REPO)
    parser.add_argument("--scan", action="append", default=None,
                        help="a root to scan, relative to --root; repeatable")
    parser.add_argument("--register-json", default=REGISTER_JSON_REL)
    parser.add_argument("--inventory", default=INVENTORY_REL)
    parser.add_argument("--governance", action="append", default=None)
    parser.add_argument("--min", type=int, default=MIN_FRAGMENT)
    parser.add_argument("--list", action="store_true",
                        help="print every fragment checked, then the summary")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    scan_rels = tuple(args.scan) if args.scan else SCAN_RELS
    governance = tuple(args.governance) if args.governance is not None else GOVERNANCE_RELS
    checked, prior, manifest_only, problems, listing = scan(
        root, scan_rels, args.register_json, args.inventory, governance, args.min)

    if args.list:
        for rel, line, fragment, prior_wording in listing:
            tag = "  [prior wording]" if prior_wording else ""
            print(f"  {rel}:{line}  {fragment}{tag}")

    for rel, line, fragment in problems:
        print(f"mirror_quotes_check: {rel}:{line}: quoted text is in no stored "
              f"byte, register cell or inventory title: {fragment}")

    readme_count = len(readmes(root, scan_rels))
    print(f"mirror_quotes_check: readmes={readme_count} fragments={checked} "
          f"prior_wording={prior} manifest_only={manifest_only} "
          f"problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
