"""Main-text r14 (documentary): bibliography corrections and subject classification.

Checklist items G6/G12 (part) and G8 (part) of SUBMISSION_READINESS.md.
- refs.bib: BobrowskiBorman pages 49--70 (J. Topol. Anal. 4 (2012), no. 1; per the publication record at
  cris.iucc.ac.il); ChenavierHirsch title corrected to the published "Extremal lifetimes of persistent loops
  and holes" (per N. Chenavier's publication list; the arXiv version 2009.13202 is "Extremal life times of
  persistent loops and holes"); LadghamLR article number 104221 and DOI 10.1016/j.spa.2023.09.008 (Crossref).
- main.tex: MSC 2020 codes and keywords (amsart \\subjclass, \\keywords). Identity details stay with the author.
- s0_status.tex date line; README.
No theorem, display or constant changes.

Usage: python3 make_r14.py DRAFTDIR   (applied to 4994f890)
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


edit('refs.bib', [
    ("""  volume  = {4},
  number  = {1},
  year    = {2012},
  doi     = {10.1142/S1793525312500057},""",
     """  volume  = {4},
  number  = {1},
  pages   = {49--70},
  year    = {2012},
  doi     = {10.1142/S1793525312500057},"""),
    ("title   = {Extremal lifetimes of persistent cycles},",
     "title   = {Extremal lifetimes of persistent loops and holes},"),
    ("""  journal = {Stochastic Process. Appl.},
  volume  = {166},
  year    = {2023},
  note    = {arXiv:2209.04150}""",
     """  journal = {Stochastic Process. Appl.},
  volume  = {166},
  pages   = {Paper No. 104221},
  year    = {2023},
  doi     = {10.1016/j.spa.2023.09.008},
  note    = {arXiv:2209.04150}"""),
])

edit('main.tex', [
    (r"""\date{V3 draft, 13 October 2026}""",
     r"""\date{V3 draft, 13 October 2026}
\subjclass[2020]{Primary 60G60; Secondary 60G15, 55N31, 60D05}
\keywords{Gaussian random fields, persistent homology, persistence diagrams, Kac--Rice formula, critical points, elder rule}"""),
])

edit('s0_status.tex', [
    ("revised through r13 on 21 October 2026.", "revised through r14 on 21 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r13 of 21 October)", "(13 October 2026; revision r14 of 21 October)"),
    ("the compiled draft, revision r13, 23 pages.", "the compiled draft, revision r14, 23 pages."),
    ("Earlier revisions are kept at their commits: r12 (23 pages, SHA-256 `72f6b056…`) at `d550d238`,",
     "Earlier revisions are kept at their commits: r13 (23 pages, SHA-256 `b3852c53…`) at `04a3ee03`, r12 (23 pages, SHA-256 `72f6b056…`) at `d550d238`,"),
    ("- **Requested:** a nonauthor read of the r12→r13 delta (the Baryshnikov citation in §6 and `refs.bib`; the §7 sentence; §0; this README; the change list).",
     "- **Reads of r13:** Cursor AMEND (main#389 6325193355), applied; readback AMEND (6325247357), applied; readback PASS (6325266500).\n- **Requested:** a nonauthor read of the r13→r14 delta (three `refs.bib` corrections; MSC 2020 codes and keywords in `main.tex`; §0; this README; the change list)."),
])
print('r14 edits applied')
