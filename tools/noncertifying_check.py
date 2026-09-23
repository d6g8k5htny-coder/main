#!/usr/bin/env python3
"""Every float path in repository-authored code is labelled, or declared.

`CLAUDE.md` rule 3 says, of float arithmetic: "Where you compute in floats,
label the path NON-CERTIFYING in the code and in any output." Until this
checker existed nothing enforced it -- no tool in `tools/` contained the string
at all -- so the rule was honoured by discipline, and a float could reach a
printed number with nothing saying so. Twenty-two repository-authored files
hold a float literal or a `float(` call; eight of them carried no label.

WHAT THIS CHECKS.

  1. Every repository-authored Python file containing a FLOAT SITE -- a float
     literal, or a call to the builtin `float` -- either contains the canonical
     label `NON-CERTIFYING`, or is named in `DECLARED` below with a reason. A
     declaration is a claim a reviewer can check; silence is not.

  2. Where the label appears in PROSE -- a string, a comment or a docstring --
     it is spelled `NON-CERTIFYING`. Identifier spellings (`NON_CERTIFYING`,
     `non_certifying`, `noncertifying`) are allowed wherever Python needs a
     name, because a hyphen cannot appear in one; the tool counts them and
     says so rather than pretending they are not there.

WHAT IS OUT OF SCOPE, stated so the pass is not read as more than it is.
Ported Drive source under `engine/carriers/blobs/` and frozen bodies under
`engine/rn_engine/frozen/` are excluded: they are byte-exact copies and are
never edited, so a labelling rule cannot apply to them. `mpmath` and `numpy`
are floats too, and a file using them without a Python float literal is not
caught here -- that is a real limit of a syntactic check and is not hidden.

WHAT THIS DOES NOT ESTABLISH. That a labelled path is correct, that an
unlabelled file computes nothing in floats by some other route, or that a
declared file's reason is true. It checks that a float site is accompanied by
a label or a stated reason. It reads no claim, grades nothing, moves no gate,
and a pass here certifies nothing.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The canonical label. One spelling, so it is greppable.
LABEL = "NON-CERTIFYING"

#: Identifier spellings Python forces, allowed wherever a name is needed.
#: Matched as WHOLE TOKENS: `float_noncertifying` is a receipt field name, not
#: this label misspelled, and a docstring naming that field is not a defect.
IDENTIFIER_SPELLINGS = ("NON_CERTIFYING", "non_certifying", "noncertifying",
                        "NONCERTIFYING")

#: Roots of repository-authored code.
CODE_ROOTS = ("tools", "tests", "research", "engine", "claims")

#: Ported bytes. Never edited, so never labelled.
EXCLUDED = ("engine/carriers/blobs", "engine/rn_engine/frozen",
            "__pycache__", "sandbox", "legacy", "quarantine")

#: Files with a float site and no label, each with the reason. Adding a line
#: here is a deliberate, visible edit; a file quietly acquiring a float is not.
DECLARED: dict[str, str] = {
    "tools/registers_import.py":
        "one `float(new)`, converting an Excel serial date. Not a mathematical "
        "quantity and never part of a bound.",
    "tools/drive_index.py":
        "one `n /= 1024.0`, rendering a byte count as KiB/MiB for a human. No "
        "quantity of the program passes through it.",
    "tests/test_bridge.py":
        "one `1.0` inside a parametrize list of values the bridge schema must "
        "REFUSE. The float is the test input, not a computed result.",
    "tests/test_hermite_envelope.py":
        "three `0.5` literals passed to assert that `he`, `he_abs` and "
        "`kernel_exponent` raise TypeError on a float. The float is refused, "
        "not computed with.",
    "tests/test_lpw_headline.py":
        "one `parse_decimal(6.239e-44)`, a negative control asserting a float "
        "literal is refused where an exact decimal string is required.",
    "research/bands/ladder.py":
        "`AdjacentBand.label` renders the exact endpoints as decimals for a "
        "human reading a report, and `format_report` prints them; the endpoints "
        "themselves are exact rationals and `name` prints them as such, so no "
        "bound passes through either. DECLARED here rather than labelled in the "
        "file because these bytes are content-pinned: "
        "`research/parallel/h3/candidate.json` and "
        "`research/rn/candidates/inner_wedge_20260920_v1.json` on the "
        "`chatgpt/drive-github-hardening-20260919` lane bind this file by "
        "SHA-256 (`9ea57670…`, 26286 bytes) as a source identity of their "
        "certificates. An in-file label changes those bytes and breaks the "
        "binding, and re-pinning is a judgement about their certificates' "
        "provenance rather than ours to make. The label therefore lives here, "
        "where it costs them nothing.",
}


def is_excluded(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(part in rel for part in EXCLUDED)


def python_files(root: str) -> list[str]:
    out = []
    for code_root in CODE_ROOTS:
        base = os.path.join(root, code_root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not is_excluded(d)]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if not is_excluded(rel):
                    out.append(rel)
    return sorted(out)


def float_sites(tree: ast.AST) -> int:
    """Float literals plus calls to the builtin ``float``.

    A float literal inside a docstring is a string, not a float, so it does not
    count -- the numbers quoted in prose throughout this repository are text.
    """
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            n += 1
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "float"):
            n += 1
    return n


def prose_spans(tree: ast.AST, source: str) -> list[str]:
    """Every string constant, plus every comment line.

    Comments are recovered by line scan rather than by tokenize: a `#` inside a
    string would be a false positive, and a false positive here only makes the
    check stricter about spelling, never more permissive about labelling.
    """
    out = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for line in source.splitlines():
        if "#" in line:
            out.append(line[line.index("#"):])
    return out


def audit(root: str, files=None) -> tuple[int, int, int, list[str]]:
    """(files scanned, files with float sites, identifier uses, problems)."""
    rels = python_files(root) if files is None else list(files)
    scanned = with_floats = identifier_uses = 0
    problems: list[str] = []
    for rel in rels:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            problems.append(f"{rel}: declared or requested and not in the tree")
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"{rel}: will not parse ({exc})")
            continue
        scanned += 1
        prose = prose_spans(tree, source)
        blob = "\n".join(prose)
        for spelling in IDENTIFIER_SPELLINGS:
            # Whole tokens only. `float_noncertifying` is a RECEIPT FIELD NAME,
            # and a docstring naming that field is not the label written badly.
            # Counting it was this checker's first false positive.
            token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(spelling)}(?![A-Za-z0-9_])")
            identifier_uses += len(token.findall(source))
            if token.search(blob) and LABEL not in blob:
                problems.append(
                    f"{rel}: the label is spelled {spelling!r} in prose and the "
                    f"canonical {LABEL!r} appears nowhere in the file. A reader "
                    f"greps for one spelling; give them that one")
        sites = float_sites(tree)
        if not sites:
            continue
        with_floats += 1
        if LABEL in source:
            continue
        if rel in DECLARED:
            continue
        problems.append(
            f"{rel}: {sites} float site(s) and no {LABEL!r} label. Label the "
            f"path, or add the file to DECLARED in {os.path.basename(__file__)} "
            f"with the reason it needs none")
    # Declarations are audited against whatever this run actually covers. With
    # an explicit --file list the caller has scoped the run, and reporting the
    # whole tree's declarations as "not in the tree" would drown the result --
    # which is what it did the first time a control ran against a temp root.
    in_scope = set(DECLARED) if files is None else (set(DECLARED) & set(rels))
    if files is None and scanned == 0:
        problems.append(
            f"VACUOUS RUN: no Python file found under {list(CODE_ROOTS)} in "
            f"{root!r}. A run that scanned nothing is not a pass; the exit code "
            f"would be the same if every module had been deleted")
    for rel in sorted(in_scope):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            problems.append(f"{rel}: DECLARED and not in the tree; drop the entry")
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        try:
            sites = float_sites(ast.parse(source))
        except SyntaxError:
            continue
        if not sites:
            problems.append(
                f"{rel}: DECLARED as having an unlabelled float site and it has "
                f"none. A stale exemption hides the next real one; drop it")
        elif LABEL in source:
            problems.append(
                f"{rel}: DECLARED as needing no label and it carries one. Drop "
                f"the declaration")
    return scanned, with_floats, identifier_uses, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--file", action="append", default=[],
                        help="check these files instead of walking the roots")
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else ROOT
    scanned, with_floats, ident, problems = audit(root, args.file or None)
    for item in problems:
        print(item)
    print(f"noncertifying_check: files={scanned} with_float_sites={with_floats} "
          f"declared={len(DECLARED)} identifier_uses={ident} "
          f"problems={len(problems)}")
    print("A pass here means a float site carries the label or a stated reason. It does not "
          "mean the labelled path is correct, and it certifies nothing.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
