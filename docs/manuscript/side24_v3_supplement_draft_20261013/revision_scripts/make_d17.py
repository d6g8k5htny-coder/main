"""Supplement draft 17 (build procedure): reproducible build and the distributed-PDF copy step.

The same repair as main-text r18 (Copilot finding on main#389, discussion_r4365148684; verified in
389/6379998598): build.sh now copies supp_main.pdf to SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf, pins
SOURCE_DATE_EPOCH (13 October 2026 00:00Z) with FORCE_SOURCE_DATE=1 so the PDF is byte-reproducible, and
verifies SHA256SUMS (or rewrites it with --update-sums). build.sh was rewritten directly; this script makes the
record edits. Draft 17 also records that main#486 (the reviewed dependency-checker guard, OpenAI) was merged
onto the draft-16 branch with Dylan's approval. No statement, number or label changes; the TeX sources are
unchanged.

Usage: python3 make_d17.py SUPPDIR   (applied to de21bfb6, after build.sh was rewritten)
"""
import sys

d = sys.argv[1]
date = '26 October 2026'


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~16 of 21 October 2026,""", r"""Checkpoint draft~17 of """ + date + ","),
    (r"""\item \emph{Draft~16}""",
     r"""\item \emph{Draft~17} changes the build procedure only: \path{build.sh} now copies the compiled PDF to the distributed file name, pins the PDF's dates so that a rebuild reproduces the committed bytes, and verifies \path{SHA256SUMS}. The sources, statements and numbers are those of draft~16.
\item \emph{Draft~16}"""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 16 (21 October 2026)",
     "# SIDE24 V3 supplement: checkpoint draft 17 (" + date + ")"),
    ("`build.sh` rebuilds the PDF.",
     "`build.sh` rebuilds the PDF, copies it to `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf` and verifies `SHA256SUMS`; since draft 17 the build is byte-reproducible (`SOURCE_DATE_EPOCH` is pinned), so the PDF's checksum line is a real check. `sh build.sh --update-sums` rewrites the manifest after a source change."),
])
readme = open(d + '/README.md', encoding='utf-8').read()
readme = readme.rstrip('\n') + (
    "\n\nDraft 17 (" + date + ") changes the build procedure only, after a Copilot finding on main#389 verified by an "
    "OpenAI/Codex lane (389/6379998598): `build.sh` now copies the compiled PDF to the distributed file name, pins "
    "`SOURCE_DATE_EPOCH` so the build reproduces the committed PDF byte for byte, and verifies `SHA256SUMS`. "
    "Between drafts 16 and 17 the branch also received main#486 (`de21bfb6`), the reviewed dependency-checker guard "
    "proposed by an OpenAI lane and merged with Dylan's approval; it changed `checks/dep_graph_check.py`, its test "
    "and output, and `SHA256SUMS` only. No statement, number or label changes.\n")
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

sec = ("## Checkpoint draft 17 (" + date + "): reproducible build; main#486 recorded\n\n"
       "Author-side, by Claude01N, on top of draft 16 (`3d434b42`) and the merge of main#486 (`de21bfb6`).\n\n"
       "- **main#486** (OpenAI, `chatgpt/supp16-reviewed-checker-20261021-6328868960`), merged onto the draft-16 branch on 23 October "
       "with Dylan's approval (\"Merge it now\", 475/6357614048; clearance review by Grok Bot agent 3). It replaced the edge-count "
       "snapshot of `checks/dep_graph_check.py` by exact edge sets (the author's own uncommitted attempt at the same change was "
       "discarded in its favour), extended `checks/test_dep_graph_check.py` to 14 tests, and reconciled the report and `SHA256SUMS`. "
       "At draft 17: 14 tests pass; the report reproduces; 403 edges; `NO_UNEXPLAINED_COMPONENT`.\n"
       "- **`build.sh`.** After the final `pdflatex` it copies `supp_main.pdf` to `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`. It pins "
       "`SOURCE_DATE_EPOCH=1791849600` (13 October 2026 00:00Z) with `FORCE_SOURCE_DATE=1`, so pdfTeX writes fixed dates and a fixed "
       "trailer `/ID` and the build is byte-reproducible (two builds here gave the same SHA-256). Without an argument it then runs "
       "`sha256sum -c SHA256SUMS`; with `--update-sums` it rewrites the manifest. The same repair is main-text r18.\n"
       "- **The PDF** is rebuilt by the new script: 162 pages, the content of draft 16, new dates and `/ID`. All 943 labels keep their numbers.\n")
conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
anchor = "## Checkpoint draft 16 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d17 edits applied')
