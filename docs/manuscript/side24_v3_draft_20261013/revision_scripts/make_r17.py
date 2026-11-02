"""Main-text r17 (documentary): cite supplement checkpoint draft 16 as [R1].

Draft 16 (commit 3d434b42 of claude/side24-v3-supplement-draft16-20261021) keeps every statement number of
drafts 12-15 (all 943 labels identical). It applies the AMEND of Grok Bot agent 9's provider-distinct
whole-document read of draft 15 and closes gap G5 (the source of the d = 2 enclosure in Corollary S9.16,
certified by checks/s9_c2_d2_check.py). Cursor read: commit 1 PASS, commit 2 AMEND applied, readback PASS
(main#389 6326651921, 6326705347). No theorem, display or constant changes.

Usage: python3 make_r17.py DRAFTDIR   (applied to fcc83ab5)
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


edit('appA_records.tex', [
    (r"""checkpoint draft~15 (21 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{ddd60647} of the branch \path{claude/side24-v3-supplement-draft15-20261021}""",
     r"""checkpoint draft~16 (21 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{3d434b42} of the branch \path{claude/side24-v3-supplement-draft16-20261021}"""),
])

edit('s0_status.tex', [
    ("revised through r16 on 21 October 2026.", "revised through r17 on 21 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r16 of 21 October)", "(13 October 2026; revision r17 of 21 October)"),
    ("the compiled draft, revision r16, 23 pages.", "the compiled draft, revision r17, 23 pages."),
    ("Earlier revisions are kept at their commits: r15 at `18543a20`,",
     "Earlier revisions are kept at their commits: r16 at `fcc83ab5`, r15 at `18543a20`,"),
    ("The companion supplement, which r16 cites as [R1] at `ddd60647` (draft 15, branch `claude/side24-v3-supplement-draft15-20261021`; it keeps every statement number of drafts 12–14;",
     "The companion supplement, which r17 cites as [R1] at `3d434b42` (draft 16, branch `claude/side24-v3-supplement-draft16-20261021`; it keeps every statement number of drafts 12–15; draft 15 had a provider-distinct whole-document read by Grok Bot agent 9, main#389 6326419744;"),
    ("- **Requested:** a nonauthor read of the r15→r16 delta (Appendix A [R1] now cites supplement draft 15; §0; this README; the change list).",
     "- **Reads of r16:** Cursor PASS (main#389 6326401627).\n- **Requested:** a nonauthor read of the r16→r17 delta (Appendix A [R1] now cites supplement draft 16; §0; this README; the change list)."),
])
print('r17 edits applied')
