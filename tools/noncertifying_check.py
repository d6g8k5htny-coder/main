#!/usr/bin/env python3
"""Every float path in repository-authored code is labelled, or declared.

`CLAUDE.md` rule 3 says, of float arithmetic: "Where you compute in floats,
label the path NON-CERTIFYING in the code and in any output." Until this
checker existed nothing enforced it -- no tool in `tools/` contained the string
at all -- so the rule was honoured by discipline, and a float could reach a
printed number with nothing saying so. Forty-one repository-authored files hold
a float literal or a `float(` call; on this line, twenty-four of them carried no
label.

WHAT READING ALL OF THEM CHANGED, and the reason `DECLARED` below is long.
Sixteen of those files use a float ONLY as a REJECTION PROBE: the float is the
input a checker must REFUSE, and the test asserts the refusal. Labelling one of
them `NON-CERTIFYING` would be a FALSE label -- the file's whole content is that
floats are rejected -- so a checker that a contributor can satisfy by pasting the
banner is a checker that invites a false statement into the tree. They are
declared, with the line each one refuses, and the declaration is the deliverable
rather than the label. Five more are wall-clock (poll intervals, timeouts,
elapsed seconds) and one is a display path inside a PINNED file, whose label
cannot be written into the bytes at all; that one is declared as an UNMET
obligation rather than an absent one, because a reader needs to know it exists.

Two files were computing in floats where it mattered. Exactly ONE could be
labelled: `tests/test_rn_moment_envelope.py`, which pins the digits
`research/rn/moment_envelope.py` prints, using a float fourth root against a
float literal inside a float tolerance.

The other could not, and finding that out cost a broken pin.
`research/rn/moment_envelope.py`'s `__main__` block prints float fourth roots of
exact rationals, so rule 3 plainly applies -- and the file is pinned by an
ARCHIVE MEMBER, `research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip::
bernstein/DEPENDENCIES.json`, which lists 30 Python files by digest.
`research/PINNED_SOURCES.md` does not cover pins of that shape, so the index
reported the file as unpinned; the label was added, and
`tools/rn_bernstein_sharp_check.py` rejected the tree with "repository dependency
mismatch: research/rn/moment_envelope.py" on the CI replay. The bytes are restored
and the obligation is DECLARED as unmet, alongside `research/bands/ladder.py`.
Two of the twenty-four therefore end as recorded unmet obligations rather than as
labels, which is the honest count.

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
The SIDE24 custody tree under `research/side24/source_recovery/custody/` is
excluded on the same ground as the carrier blobs: recovered Drive source, held
byte-exact and pinned, so adding a banner is the one edit that would break it.

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

#: Ported bytes. Never edited, so never labelled. The SIDE24 custody tree is
#: here for the same reason as the carrier blobs: recovered Drive source, held
#: byte-exact and pinned by a certificate, so a labelling rule cannot apply to
#: it -- editing it to add a banner is the one thing that would break it.
EXCLUDED = ("engine/carriers/blobs", "engine/rn_engine/frozen",
            "research/side24/source_recovery/custody",
            "__pycache__", "sandbox", "legacy", "quarantine")

#: Files with a float site and no label, each with the reason. Adding a line
#: here is a deliberate, visible edit; a file quietly acquiring a float is not.
DECLARED: dict[str, str] = {
    # --- incidental conversions ---------------------------------------------
    "tools/registers_import.py":
        "one `float(new)`, converting an Excel serial date. Not a mathematical "
        "quantity and never part of a bound.",
    "tools/drive_index.py":
        "one `n /= 1024.0`, rendering a byte count as KiB/MiB for a human. No "
        "quantity of the program passes through it.",

    # --- wall-clock: poll intervals, timeouts, elapsed seconds ---------------
    # A float that measures TIME. None of these is a quantity of the
    # mathematics, and none reaches a bound or a printed result.
    "tools/run_checks.py":
        "one `selector.select(timeout=0.05)`, the poll interval of the output "
        "reader. Wall-clock, not arithmetic.",
    "tools/twelve_project_check.py":
        "`selector.select(min(0.05, timeout_seconds))` and "
        "`process.wait(timeout=max(0.001, ...))`. Both are wall-clock bounds on "
        "a subprocess, never quantities of the campaign.",
    "tests/test_twelve_project_check.py":
        "a `.15` run_bounded timeout, a `time.sleep(.5)` and a "
        "`budget_seconds=0.000001` exercising the budget path. Wall-clock only.",
    "tests/test_run_checks.py":
        "one `assert report[...]['timeout_seconds'] < 0.4`, checking that a "
        "wall-clock budget was recorded. The float is a duration.",

    # --- REJECTION PROBES ----------------------------------------------------
    # The dominant category in this tree, and the reason this dict is long. A
    # float here is the INPUT a checker must REFUSE; the test asserts the
    # refusal. Labelling one of these NON-CERTIFYING would be a FALSE label:
    # the file's content is that floats are rejected. See the module docstring.
    "tests/test_bridge.py":
        "one `1.0` in a parametrize list of values the bridge schema must "
        "REFUSE. The float is the test input, not a computed result.",
    "tests/test_lpw_headline.py":
        "one `parse_decimal(6.239e-44)`, a negative control asserting a float "
        "literal is refused where an exact decimal string is required.",
    "tests/test_c2_band.py":
        "one `0.05`, id'd `float`, in a parametrize list of rejected radii.",
    "tests/test_gaussian_families.py":
        "two `1.0` entries in parametrize lists of values the family "
        "constructors must refuse.",
    "tests/test_gaussian_moments.py":
        "one `MomentEngine((0.0,0,0), ...)` inside `pytest.raises(TypeError)`. "
        "The float exists to be refused.",
    "tests/test_hermite_gaussian.py":
        "one `1.0` in a parametrize list of rejected arguments.",
    "tests/test_manifest_integrity_hardening.py":
        "one `1.5` in a parametrize list of byte sizes the manifest reader must "
        "refuse.",
    "tests/test_rn_certificate.py":
        "`for value in (0.5, True)` -- two rejected replacements for an exact "
        "field.",
    "tests/test_rn_conditioning.py":
        "one `((1.0,),)` matrix in a parametrize list of refused shapes.",
    "tests/test_rn_side24.py":
        "a `0.5` and a `1.0` in parametrize lists of refused arguments and "
        "refused points.",
    "tests/test_rn_side24_cell.py":
        "one `192.0` in a parametrize list of refused precisions -- a float "
        "where an int is required.",
    "tests/test_rn_side24_density.py":
        "a `0.0` centre and a `1.0` direction component, both refused.",
    "tests/test_rn_side24_wedge.py":
        "`6.0`, `256.0` and `12.0` in parametrize lists of refused order, bits "
        "and piece counts.",
    "tests/test_rn_spatial_cover.py":
        "a `1.0` and a `192.0` in parametrize lists of refused arguments.",
    "tests/test_rn_density_majorant.py":
        "a `1.0` mahalanobis_lower and a `192.0` bits, both refused.",

    # --- one file in both categories -----------------------------------------
    "tests/test_rn_bernstein_sharp_check.py":
        "`float('nan')` and `float('inf')` fed to the budget and timing "
        "validators to assert they are refused, plus two "
        "`elapsed_seconds=0.01` fixtures. Rejection probes and wall-clock; no "
        "float reaches a bound.",
    "tests/test_h3_rn_n6_check.py":
        "a `25.0` replacing a boolean field, to assert the refusal, and a "
        "`budget_seconds=0.000001`. A probe and a duration.",

    # --- display paths in PINNED files, which cannot carry the label ---------
    # The two entries that record an UNMET obligation rather than an absent one.
    # Rule 3 says to label the path in the code and in any output; every float
    # site in both files is a display conversion, so the rule applies -- and both
    # are pinned, so the label cannot be written into the bytes without breaking
    # the certificates that bind them. Declared rather than silently excluded,
    # because the reader needs to know the obligation exists and why the bytes do
    # not meet it.
    #
    # `moment_envelope.py` is here because attempting the label BROKE A PIN and
    # the CI replay caught it: `tools/rn_bernstein_sharp_check.py` rejected the
    # tree with "repository dependency mismatch". Its pin is declared inside an
    # ARCHIVE MEMBER --
    # `research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip`,
    # member `bernstein/DEPENDENCIES.json`, 30 Python files by digest -- and
    # `research/PINNED_SOURCES.md` does not cover that shape, so the index said
    # the file was free to edit. Recorded here because a reader checking the index
    # would reach the same wrong conclusion.
    "research/rn/moment_envelope.py":
        "six float sites, all in the `__main__` display block: `float(...)` "
        "conversions and two `** 0.25` fourth roots, printed for a reader to "
        "compare by eye. The block's verdict line is NOT on that path -- it "
        "compares `defective_pow4 < typed_expectation_lower ** 4` in exact "
        "rationals -- so the module's result is exact and only its digits are "
        "float. The file is PINNED by "
        "`research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip::"
        "bernstein/DEPENDENCIES.json` (bytes 6778, sha256 107c873f...), so the "
        "label cannot be added in place. The obligation is recorded here and "
        "unmet in the bytes; a successor edition should carry it.",
    "research/bands/ladder.py":
        "eleven float sites, every one a `float(...):.Nf` conversion inside a "
        "report f-string (lines 262, 555, 564-567, 576-577, 585) -- display "
        "only; no bound and no compared quantity passes through a float here. "
        "The file is PINNED (see research/PINNED_SOURCES.md), so the label "
        "cannot be added in place. The obligation is recorded here and unmet in "
        "the bytes; a successor edition of this module should carry the label.",
}

#: The spelling rule (2) fires on any identifier spelling in prose, whether or
#: not the file has anything to label. That is right for a file that owes a
#: label and wrote it badly, and WRONG for a file using "noncertifying" as an
#: ordinary English adjective -- which is this checker's SECOND false positive
#: of the same family as the first (a receipt field name, recorded in the
#: spelling loop below). A declaration keeps the finding visible and the reason
#: checkable, which is the point of DECLARED above; narrowing the rule silently
#: would lose the real case of a file that labels its mpmath path in prose
#: without ever writing a Python float literal.
SPELLING_DECLARED: dict[str, str] = {
    "research/cover/audit.py":
        "'noncertifying' is an adjective here, not a label: \"Well-formed "
        "noncertifying or incomplete totals remain noncertifying/incomplete, "
        "rather than being promoted\" describes a TOTAL's status. The file has "
        "ZERO float sites, so it owes no label at all, and rewriting the "
        "sentence to say NON-CERTIFYING would make it say something else.",
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
            if token.search(blob) and LABEL not in blob and rel not in SPELLING_DECLARED:
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
          f"declared={len(DECLARED)} spelling_declared={len(SPELLING_DECLARED)} "
          f"identifier_uses={ident} "
          f"problems={len(problems)}")
    print("A pass here means a float site carries the label or a stated reason. It does not "
          "mean the labelled path is correct, and it certifies nothing.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
