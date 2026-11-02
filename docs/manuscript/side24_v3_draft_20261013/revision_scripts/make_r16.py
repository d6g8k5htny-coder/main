"""Main-text r16 (documentary): cite supplement checkpoint draft 15 as [R1].

Draft 15 (commit ddd60647 of claude/side24-v3-supplement-draft15-20261021) keeps every statement number of
drafts 12-14 (all 943 labels identical); draft 14 removed provenance wording (readback PASS, main#389
6325795760) and draft 15 made radius/parameter matching explicit and hardened the dependency check (Cursor read
6326091969, AMEND applied, readback PASS 6326245246). So every Supplement number the main text cites names the
same statement. No theorem, display or constant changes.

Usage: python3 make_r16.py DRAFTDIR   (applied to 18543a20)
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
    (r"""checkpoint draft~13 (20 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{d28e6ab4} of the branch \path{claude/side24-v3-supplement-draft13-20261020}""",
     r"""checkpoint draft~15 (21 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{ddd60647} of the branch \path{claude/side24-v3-supplement-draft15-20261021}"""),
])

edit('s0_status.tex', [
    ("revised through r15 on 21 October 2026.", "revised through r16 on 21 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r15 of 21 October)", "(13 October 2026; revision r16 of 21 October)"),
    ("the compiled draft, revision r15, 23 pages.", "the compiled draft, revision r16, 23 pages."),
    ("Earlier revisions are kept at their commits: r14 at `ba1b904d`,",
     "Earlier revisions are kept at their commits: r15 at `18543a20`, r14 at `ba1b904d`,"),
    ("The companion supplement, which r11 cites as [R1] at `d28e6ab4` (draft 13, branch `claude/side24-v3-supplement-draft13-20261020`; draft 12 is `59d20df5`, which every provider slice read binds)",
     "The companion supplement, which r16 cites as [R1] at `ddd60647` (draft 15, branch `claude/side24-v3-supplement-draft15-20261021`; it keeps every statement number of drafts 12–14; draft 12 is `59d20df5`, which every provider slice read binds)"),
    ("- **Requested:** a nonauthor read of the r14→r15 delta (`refs.bib` completed from Crossref and zbMATH Open data, main#389 6325580607; §0; this README; the change list).",
     "- **Reads of r15:** Cursor PASS (main#389 6326222268).\n- **Requested:** a nonauthor read of the r15→r16 delta (Appendix A [R1] now cites supplement draft 15; §0; this README; the change list)."),
])
print('r16 edits applied')
