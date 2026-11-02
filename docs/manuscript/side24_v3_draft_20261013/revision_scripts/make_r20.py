"""Main-text r20 (documentary): cite supplement checkpoint draft 18 as [R1]; record the r19 and draft-18 reads.

Supplement draft 18 (commit 8253410a of claude/side24-v3-supplement-draft18-20261026: e3ed9a19 plus the
amendment C327-N1-01) keeps every statement number of drafts 12-17 and repairs one notation clash, N1 of the
whole-document reads: the coefficient expression c_d[K] of Definition S4.2 at d = 2 is no longer spelled
c_2[K_24] in the proof of Corollary S9.16, and a sentence after Definition S4.2 reserves c_2[K] for
Definition S11.2 (the fixed-cone expression, the c_2 of Theorem 2.3(iii) here). Reads: OpenAI/Codex root
01a0bbb5, S9 PASS_SCOPED and one AMEND on the new S4 sentence (main#389 6424476174), applied as 8253410a;
OpenAI/Codex GTP readback PASS_SCOPED_SOURCE_AND_RECORD_READBACK (main#389 6424972265). The r19 read by the
OpenAI/Codex Research Task Watch passed with no findings (main#389 review 5656459000). No theorem, display or
constant changes; of the TeX sources only appA_records.tex (the [R1] record) and s0_status.tex (the revision
line) change.

Usage: python3 make_r20.py DRAFTDIR            (applied to f29efae4 = r19 c00fc6c2 + a base-only merge of main)
       python3 make_r20.py DRAFTDIR --pdf-stats  (after `sh build.sh --update-sums`; then `sh build.sh --update-sums` again)
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
        ("r20 is this PDF: 23 pages, PDFSIZE B, SHA-256 `PDFSHA`.",
         "r20 is this PDF: 23 pages, %d B, SHA-256 `%s`." % (size, sha)),
    ])
    print('pdf stats filled: %d %s' % (size, sha))
    raise SystemExit(0)

edit('appA_records.tex', [
    (r"""checkpoint draft~17 (26 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{71ee360c} of the branch \path{claude/side24-v3-supplement-draft17-20261026}""",
     r"""checkpoint draft~18 (26 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{8253410a} of the branch \path{claude/side24-v3-supplement-draft18-20261026}"""),
])

edit('s0_status.tex', [
    ("revised through r19 on 26 October 2026.", "revised through r20 on 26 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r19 of 26 October)", "(13 October 2026; revision r20 of 26 October)"),
    ("the compiled draft, revision r19, 23 pages.", "the compiled draft, revision r20, 23 pages."),
    ("Earlier revisions are kept at their commits: r18 (23 pages, SHA-256 `905681ec…`) at `c47c0391`, with its record amendment at `e9583f09`;",
     "Earlier revisions are kept at their commits: r19 (23 pages, SHA-256 `aacaddaf…`) at `c00fc6c2`; r18 (23 pages, SHA-256 `905681ec…`) at `c47c0391`, with its record amendment at `e9583f09`;"),
    ("The companion supplement, which r19 cites as [R1] at `71ee360c` (draft 17, branch `claude/side24-v3-supplement-draft17-20261026`; it keeps every statement number of drafts 12–16, and changes only the build procedure of draft 16, with an OpenAI build-procedure read, PASS_SCOPED_BUILD_PROCEDURE, main#389 6421820710;",
     "The companion supplement, which r20 cites as [R1] at `8253410a` (draft 18, branch `claude/side24-v3-supplement-draft18-20261026`; it keeps every statement number of drafts 12–17 and repairs one notation clash, N1: the §S4 coefficient expression at d = 2 is no longer spelled `c_2[K_24]` in the proof of Corollary S9.16, and `c_2[K]` is reserved for Definition S11.2; OpenAI reads: root 01a0bbb5, S9 PASS_SCOPED with one AMEND on the new S4 sentence, main#389 6424476174, applied as `8253410a`, and the GTP readback PASS_SCOPED_SOURCE_AND_RECORD_READBACK, 6424972265; draft 17 was the build-procedure repair, with an OpenAI build-procedure read, PASS_SCOPED_BUILD_PROCEDURE, main#389 6421820710;"),
    ("- **Requested:** a nonauthor readback of the A-F1 commit `c47c0391..e9583f09` and a read of the r18→r19 delta (Appendix A [R1] now cites supplement draft 17; §0; this README; the change list; `SUBMISSION_READINESS.md`; the `make_r18.py` docstring).",
     "- **Reads of r19 and of the A-F1 readback:** OpenAI/Codex Research Task Watch, PASS_SCOPED_DOCUMENTARY_AND_RECORD_READBACK, no findings (main#389 review 5656459000, 6423194618).\n- **Requested:** a nonauthor read of the r19→r20 delta (Appendix A [R1] now cites supplement draft 18; §0; this README; the change list; `SUBMISSION_READINESS.md`)."),
])

edit('SUBMISSION_READINESS.md', [
    ("Status as of 26 October 2026, about 17:05Z (r19 made: supplement draft 17 cited; the r18 and draft-17 build reads passed).",
     "Status as of 26 October 2026, about 19:20Z (r20 made: supplement draft 18 cited; the r19 read passed; the N1 notation clash is fixed)."),
    ("| Supplement | draft 17, cited as [R1] at `71ee360c` (from r19), 162 pages; statement numbers unchanged since draft 12 |",
     "| Supplement | draft 18, cited as [R1] at `8253410a` (from r20), 163 pages; statement numbers unchanged since draft 12 |"),
    ("draft 17 (build procedure only): OpenAI PASS_SCOPED_BUILD_PROCEDURE 6421820710 |",
     "draft 17 (build procedure only): OpenAI PASS_SCOPED_BUILD_PROCEDURE 6421820710; draft 18 (notation N1): OpenAI root 01a0bbb5 S9 PASS_SCOPED + AMEND C327-N1-01 (6424476174) applied as `8253410a`, GTP readback PASS 6424972265 |"),
    ("**Still open (style pass):** the notation clashes (c_2[K] in §S4 vs §S11, and others) and the abstract's wording for the computer-assisted theorems.",
     "**Still open (style pass):** the remaining notation overloads (N2–N6: `J_0`, `𝒦`, `r_0`, `r_1`, `r_*`, `Ψ`, `ν_cand`; N1, the `c_2[K]` clash between §S4 and §S11, is fixed in draft 18) and the abstract's wording for the computer-assisted theorems."),
    ("| Author agent **[AI]** | r13–r19 done; r13–r18 read (r19's read and the A-F1 readback requested).",
     "| Author agent **[AI]** | r13–r20 done; r13–r19 read (r20's read requested)."),
])

entry = """
## Revision r20 (26 October, about 19:20Z)

r19 is commit `c00fc6c2`; the branch then received a base-only merge of `main` (`f29efae4`; no manuscript file changed). The r19 read and the A-F1 readback by the OpenAI/Codex Research Task Watch passed with no findings (main#389 review 5656459000, 6423194618). Supplement checkpoint draft 18 (commit `8253410a` of `claude/side24-v3-supplement-draft18-20261026`) repairs the one notation clash that reached across the two documents, N1 of the whole-document reads: the coefficient expression `c_d[K]` of Supplement Definition S4.2 at d = 2 (`c_{2,24} ≈ 0.0734`) was spelled `c_2[K_24]` in the proof of Corollary S9.16, the same symbol as the fixed-cone expression `c_2[K]` of Definition S11.2 (`c_2[K_24] ≈ 0.1612` at d = 3, the `c_2` of Theorem 2.3(iii) here). Draft 18 names the §S4 values without that spelling and reserves `c_2[·]` for Definition S11.2; its reads: OpenAI/Codex root 01a0bbb5, S9 PASS_SCOPED and one AMEND on the new S4 sentence (the d = 2 fixed-cone value is 0.2215, not the d = 3 value; main#389 6424476174), applied as `8253410a`; OpenAI/Codex GTP readback of `e3ed9a19..8253410a`, PASS_SCOPED_SOURCE_AND_RECORD_READBACK (6424972265). Every statement number of drafts 12–17 is unchanged, so every "Supplement, Sx.y" reference here is unchanged. r20 is this PDF: 23 pages, PDFSIZE B, SHA-256 `PDFSHA`.

**Why.** Supplement draft 18 is read, so the main text cites it; and the reads of r19 and draft 18 are recorded. The main text's own uses of `c_2[K_ref]` and `c_2[K_24]` (§2, Theorem 2.3(iii), §0) are in the §S11 sense and need no change.

**Changes (records only; of the TeX sources, `appA_records.tex` changes the [R1] record and `s0_status.tex` its revision line).**
- **Appendix A, [R1]:** now draft 18, commit `8253410a`, dated 26 October 2026.
- **§0:** "Revised through r20 on 26 October 2026".
- **README:** revision r20; r19 added to the earlier revisions; [R1] is draft 18, with its reads; the r19 read recorded; a read of r19→r20 requested.
- **`SUBMISSION_READINESS.md`:** status line; the Supplement row (draft 18, 163 pages, its reads); G2's open list (N1 fixed; N2–N6 remain for the style pass); the author-agent row.

**Build.** `sh build.sh --update-sums` compiles with no LaTeX error and the same 2 pt overfull line as before; 23 pages; no BibTeX warnings; the manifest is rewritten (PDF, `V3_CHANGE_LIST.md`, `appA_records.tex`, `s0_status.tex`) and `sha256sum -c` passes 14/14. Record edits by `revision_scripts/make_r20.py`.
"""
p = d + '/V3_CHANGE_LIST.md'
s = open(p, encoding='utf-8').read()
assert '## Revision r20' not in s
open(p, 'w', encoding='utf-8').write(s.rstrip('\n') + '\n' + entry)
print('r20 edits applied')
