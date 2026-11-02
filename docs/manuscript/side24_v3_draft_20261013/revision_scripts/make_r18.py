"""Main-text r18 (documentary, build procedure): reproducible build and the distributed-PDF copy step.

Copilot review discussion_r4365148684 on main#389, verified by an OpenAI/Codex lane (389/6379998598): build.sh
wrote main.pdf and never copied it to the distributed file SIDE24_V3_DRAFT_20261013.pdf, so a future revision
that regenerated SHA256SUMS without the manual copy could ship a stale PDF with a passing manifest (every
revision so far copied by hand; 14/14 manifest entries of r17 were verified correct). r18 makes the copy part of
build.sh, pins SOURCE_DATE_EPOCH (13 October 2026 00:00Z) with FORCE_SOURCE_DATE=1 so the PDF is byte-reproducible,
and has build.sh verify SHA256SUMS (or rewrite it with --update-sums). build.sh was rewritten directly; this
script makes the record edits. r18 also records the adoption of #485 (G6 qualification) onto the branch.
No theorem, display or constant changes; the mathematical TeX sources are unchanged (s0_status.tex changes only
its revision/date line; this sentence was corrected in r19 after the slice-A AMEND A-F1, main#389 6421761943).

Usage: python3 make_r18.py DRAFTDIR   (applied to 29ca667d, after build.sh was rewritten)
"""
import sys

d = sys.argv[1]


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


edit('s0_status.tex', [
    ("revised through r17 on 21 October 2026.", "revised through r18 on 26 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r17 of 21 October)", "(13 October 2026; revision r18 of 26 October)"),
    ("the compiled draft, revision r17, 23 pages.", "the compiled draft, revision r18, 23 pages."),
    ("Earlier revisions are kept at their commits: r16 at `fcc83ab5`,",
     "Earlier revisions are kept at their commits: r17 (23 pages, SHA-256 `8ac5e278…`) at `10e91203`, r16 at `fcc83ab5`,"),
    ("`build.sh` rebuilds the PDF with `pdflatex` and `bibtex` (amsart, amsplain).",
     "`build.sh` rebuilds the PDF with `pdflatex` and `bibtex` (amsart, amsplain), copies it to `SIDE24_V3_DRAFT_20261013.pdf` and verifies `SHA256SUMS`; since r18 the build is byte-reproducible (`SOURCE_DATE_EPOCH` is pinned), so the PDF's checksum line is a real check. `sh build.sh --update-sums` rewrites the manifest after a source change."),
    ("- **Requested:** a nonauthor read of the r16→r17 delta (Appendix A [R1] now cites supplement draft 16; §0; this README; the change list).",
     "- **Reads of r17:** Cursor PASS (main#389 6326776936).\n- **Requested:** a nonauthor read of the r17→r18 delta (`build.sh`: the copy step, the pinned epoch and the manifest check; the rebuilt PDF; §0; this README; the change list; `SUBMISSION_READINESS.md`)."),
])

edit('SUBMISSION_READINESS.md', [
    ("Status as of 21 October 2026, about 21:20Z (r17 made; supplement draft 16 read).",
     "Status as of 26 October 2026, about 16:30Z (r18 made: reproducible build; #485 adopted)."),
    ("| G12 | **Journal style pass**",
     "| G15 | **Build procedure.** A Copilot review of main#389 (discussion_r4365148684), verified by an OpenAI/Codex lane (389/6379998598), found that `build.sh` never copied `main.pdf` to the distributed `SIDE24_V3_DRAFT_20261013.pdf`; every revision so far did the copy by hand and r17's 14 manifest entries were verified correct, but a future revision could have shipped a stale PDF with a passing manifest. | **[DONE]** in r18: `build.sh` copies the PDF, pins `SOURCE_DATE_EPOCH` so the build is byte-reproducible, and verifies (or, with `--update-sums`, rewrites) `SHA256SUMS`. The supplement's `build.sh` gets the same change in draft 17. | Resolve the Copilot thread once the r18 read passes. |\n| G12 | **Journal style pass**"),
    ("| Author agent **[AI]** | r13 (G5, G7) is done and read (main#389 6325193355, 6325266500). After the venue choice: G11, G12 and G14. |",
     "| Author agent **[AI]** | r13–r18 done and read (r18's read requested). #485 (G6 qualification, OpenAI proposal, documentary PASS_SCOPED 485/6328282240) adopted onto the branch as merge `29ca667d`. After the venue choice: G11, G12 and G14. |"),
])
print('r18 edits applied')
