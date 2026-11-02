"""Main-text r19 (documentary): cite supplement checkpoint draft 17 as [R1]; record the r18 and draft-17 reads.

Supplement draft 17 (commit 71ee360c of claude/side24-v3-supplement-draft17-20261026) keeps every statement
number of drafts 12-16 and changes only the build procedure (the same repair as main-text r18) and its records;
its build-procedure read by an OpenAI/Codex lane passed (main#389 6421820710). The r18 build-procedure read by
another OpenAI/Codex lane passed on the build and asked one record amendment, A-F1 (main#389 6421761943),
applied as commit e9583f09. r19 also corrects the same statement in the docstring of make_r18.py. No theorem,
display or constant changes; of the TeX sources only s0_status.tex (the revision line) and appA_records.tex
(the [R1] record) change.

Usage: python3 make_r19.py DRAFTDIR            (applied to e9583f09)
       python3 make_r19.py DRAFTDIR --pdf-stats  (after `sh build.sh --update-sums`: fills the PDF line of the
                                                change list; then run `sh build.sh --update-sums` once more)
"""
import hashlib
import os
import sys

d = sys.argv[1]
phase2 = len(sys.argv) > 2 and sys.argv[2] == '--pdf-stats'


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


if phase2:
    pdf = d + '/SIDE24_V3_DRAFT_20261013.pdf'
    size = os.path.getsize(pdf)
    sha = hashlib.sha256(open(pdf, 'rb').read()).hexdigest()
    edit('V3_CHANGE_LIST.md', [
        ("r19 is this PDF: 23 pages, PDFSIZE B, SHA-256 `PDFSHA`.",
         "r19 is this PDF: 23 pages, %d B, SHA-256 `%s`." % (size, sha)),
    ])
    print('pdf stats filled: %d %s' % (size, sha))
    raise SystemExit(0)

edit('appA_records.tex', [
    (r"""checkpoint draft~16 (21 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{3d434b42} of the branch \path{claude/side24-v3-supplement-draft16-20261021}""",
     r"""checkpoint draft~17 (26 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{71ee360c} of the branch \path{claude/side24-v3-supplement-draft17-20261026}"""),
])

edit('s0_status.tex', [
    ("revised through r18 on 26 October 2026.", "revised through r19 on 26 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r18 of 26 October)", "(13 October 2026; revision r19 of 26 October)"),
    ("the compiled draft, revision r18, 23 pages.", "the compiled draft, revision r19, 23 pages."),
    ("Earlier revisions are kept at their commits: r17 (23 pages, SHA-256 `8ac5e278…`) at `10e91203`,",
     "Earlier revisions are kept at their commits: r18 (23 pages, SHA-256 `905681ec…`) at `c47c0391`, with its record amendment at `e9583f09`; r17 (23 pages, SHA-256 `8ac5e278…`) at `10e91203`,"),
    ("The companion supplement, which r17 cites as [R1] at `3d434b42` (draft 16, branch `claude/side24-v3-supplement-draft16-20261021`; it keeps every statement number of drafts 12–15;",
     "The companion supplement, which r19 cites as [R1] at `71ee360c` (draft 17, branch `claude/side24-v3-supplement-draft17-20261026`; it keeps every statement number of drafts 12–16, and changes only the build procedure of draft 16, with an OpenAI build-procedure read, PASS_SCOPED_BUILD_PROCEDURE, main#389 6421820710;"),
    ("- **Requested:** a nonauthor read of the r17→r18 delta (`build.sh`: the copy step, the pinned epoch and the manifest check; the rebuilt PDF; §0; this README; the change list; `SUBMISSION_READINESS.md`).",
     "- **Reads of r18:** OpenAI/Codex Architecture Task Watch (main#389 6421761943): build procedure PASS (clean rebuild byte-identical to the committed PDF; 14/14 manifest lines; `make_r18.py` replays the record edits), with one record amendment, A-F1 (the r18 heading of the change list called all TeX sources unchanged while `s0_status.tex` changed its revision line), applied as `e9583f09`.\n- **Requested:** a nonauthor readback of the A-F1 commit `c47c0391..e9583f09` and a read of the r18→r19 delta (Appendix A [R1] now cites supplement draft 17; §0; this README; the change list; `SUBMISSION_READINESS.md`; the `make_r18.py` docstring)."),
])

edit('SUBMISSION_READINESS.md', [
    ("Status as of 26 October 2026, about 16:30Z (r18 made: reproducible build; #485 adopted).",
     "Status as of 26 October 2026, about 17:05Z (r19 made: supplement draft 17 cited; the r18 and draft-17 build reads passed)."),
    ("| Supplement | draft 16, cited as [R1] at `3d434b42` (from r17), 162 pages; statement numbers unchanged since draft 12 |",
     "| Supplement | draft 17, cited as [R1] at `71ee360c` (from r19), 162 pages; statement numbers unchanged since draft 12 |"),
    ("draft 16 (agent 9's AMEND; G5 closed): Cursor commit 1 PASS, commit 2 AMEND applied, readback PASS 6326705347 |",
     "draft 16 (agent 9's AMEND; G5 closed): Cursor commit 1 PASS, commit 2 AMEND applied, readback PASS 6326705347; draft 17 (build procedure only): OpenAI PASS_SCOPED_BUILD_PROCEDURE 6421820710 |"),
    ("| Resolve the Copilot thread once the r18 read passes. |",
     "| Read passed: OpenAI slice A (389/6421761943; build PASS, record AMEND A-F1 applied as `e9583f09`) and slice B for the supplement's draft 17 (389/6421820710). The Copilot thread is resolved. |"),
    ("| Author agent **[AI]** | r13–r18 done and read (r18's read requested).",
     "| Author agent **[AI]** | r13–r19 done; r13–r18 read (r19's read and the A-F1 readback requested)."),
])

edit('revision_scripts/make_r18.py', [
    ("No theorem, display or constant changes; the TeX sources are unchanged.",
     "No theorem, display or constant changes; the mathematical TeX sources are unchanged (s0_status.tex changes only\nits revision/date line; this sentence was corrected in r19 after the slice-A AMEND A-F1, main#389 6421761943)."),
])

entry = """
## Revision r19 (26 October, about 17:05Z)

r18 is commit `c47c0391`, with its record amendment A-F1 at `e9583f09`. Its build-procedure read by the OpenAI/Codex Architecture Task Watch passed on the build (main#389 6421761943: a clean rebuild gives the committed PDF byte for byte; 14/14 manifest lines; `make_r18.py` replays the record edits) and asked one record amendment, A-F1: the r18 heading below said "the TeX sources are unchanged" although `s0_status.tex` changed its revision line. The heading was amended as requested in `e9583f09`. The supplement's checkpoint draft 17 (commit `71ee360c` of `claude/side24-v3-supplement-draft17-20261026`), the same build-procedure repair for the supplement, had its read by the OpenAI/Codex Research Task Watch: PASS_SCOPED_BUILD_PROCEDURE (main#389 6421820710; clean rebuild byte-identical to the committed PDF, 46/46 manifest lines, 943 labels unchanged). r19 is this PDF: 23 pages, PDFSIZE B, SHA-256 `PDFSHA`.

**Why.** Supplement draft 17 is read, so the main text cites it; and the reads of r18 and draft 17 are recorded.

**Changes (records only; of the TeX sources, `appA_records.tex` changes the [R1] record and `s0_status.tex` its revision line).**
- **Appendix A, [R1]:** now draft 17, commit `71ee360c`, dated 26 October 2026. Draft 17 keeps every statement number of drafts 12–16; every "Supplement, Sx.y" reference in the main text is unchanged.
- **§0:** "Revised through r19 on 26 October 2026".
- **README:** revision r19; r18 and its amendment are added to the earlier revisions; [R1] is draft 17, with its read; the r18 read is recorded; the A-F1 readback and a read of r18→r19 are requested.
- **`SUBMISSION_READINESS.md`:** status line; the Supplement row (draft 17 and its read); G15 (read passed, Copilot thread resolved); the author-agent row.
- **`revision_scripts/make_r18.py`:** its docstring made the same statement as the amended heading; corrected, with a pointer to A-F1.

**Build.** `sh build.sh --update-sums` compiles with no LaTeX error and the same 2 pt overfull line as before; 23 pages; no BibTeX warnings; the manifest is rewritten (PDF, `V3_CHANGE_LIST.md`, `appA_records.tex`, `s0_status.tex`) and `sha256sum -c` passes 14/14. Record edits by `revision_scripts/make_r19.py`.
"""
p = d + '/V3_CHANGE_LIST.md'
s = open(p, encoding='utf-8').read()
assert '## Revision r19' not in s
open(p, 'w', encoding='utf-8').write(s.rstrip('\n') + '\n' + entry)
print('r19 edits applied')
