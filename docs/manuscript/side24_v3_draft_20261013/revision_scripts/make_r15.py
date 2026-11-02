"""Main-text r15 (documentary): bibliography completed from Cursor's data record B (main#389 6325580607).

Checklist items G6 and G12 (part) of SUBMISSION_READINESS.md. Sources: Crossref and zbMATH Open, as reported in
record B (Cursor cloud agent, Grok; data only).
- ChenavierHirsch: Extremes 25 (2022), no. 2, 299--330.
- HLO: Extremes 29 (2026), no. 3, 495--559.
- Perez: published, J. Appl. Comput. Topol. 7 (2023), no. 4, 879--906, DOI 10.1007/s41468-023-00132-x (zbMATH 1527.60062).
- HirschLR: published, Ann. Appl. Probab. 36 (2026), no. 5, 4430--4464, DOI 10.1214/26-AAP2333.
- Pranav (arXiv:2109.08721): no published version found; stays a preprint.
Also: s0_status.tex date line; README. No theorem, display or constant changes.

Usage: python3 make_r15.py DRAFTDIR   (applied to ba1b904d)
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
    ("""  journal = {Extremes},
  volume  = {25},
  year    = {2022},
  doi     = {10.1007/s10687-021-00430-6}""",
     """  journal = {Extremes},
  volume  = {25},
  number  = {2},
  pages   = {299--330},
  year    = {2022},
  doi     = {10.1007/s10687-021-00430-6}"""),
    ("""  journal = {Extremes},
  year    = {2026},
  doi     = {10.1007/s10687-026-00526-x},""",
     """  journal = {Extremes},
  volume  = {29},
  number  = {3},
  pages   = {495--559},
  year    = {2026},
  doi     = {10.1007/s10687-026-00526-x},"""),
    ("""@misc{Perez,
  author = {Perez, Daniel},
  title  = {On the persistent homology of almost surely {$C^0$} stochastic processes},
  year   = {2022},
  note   = {arXiv:2012.09459}
}""",
     """@article{Perez,
  author  = {Perez, Daniel},
  title   = {On the persistent homology of almost surely {$C^0$} stochastic processes},
  journal = {J. Appl. Comput. Topol.},
  volume  = {7},
  number  = {4},
  pages   = {879--906},
  year    = {2023},
  doi     = {10.1007/s41468-023-00132-x},
  note    = {arXiv:2012.09459}
}"""),
    ("""@misc{HirschLR,
  author = {Hirsch, Christian and Lachi{\\`e}ze-Rey, Rapha{\\"e}l},
  title  = {Functional central limit theorem for topological functionals of {G}aussian critical points},
  year   = {2024},
  note   = {arXiv:2411.11429}
}""",
     """@article{HirschLR,
  author  = {Hirsch, Christian and Lachi{\\`e}ze-Rey, Rapha{\\"e}l},
  title   = {Functional central limit theorem for topological functionals of {G}aussian critical points},
  journal = {Ann. Appl. Probab.},
  volume  = {36},
  number  = {5},
  pages   = {4430--4464},
  year    = {2026},
  doi     = {10.1214/26-AAP2333},
  note    = {arXiv:2411.11429}
}"""),
])

edit('s0_status.tex', [
    ("revised through r14 on 21 October 2026.", "revised through r15 on 21 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r14 of 21 October)", "(13 October 2026; revision r15 of 21 October)"),
    ("the compiled draft, revision r14, 23 pages.", "the compiled draft, revision r15, 23 pages."),
    ("Earlier revisions are kept at their commits: r13 (23 pages, SHA-256 `b3852c53…`) at `04a3ee03`,",
     "Earlier revisions are kept at their commits: r14 at `ba1b904d`, r13 (23 pages, SHA-256 `b3852c53…`) at `04a3ee03`,"),
    ("- **Requested:** a nonauthor read of the r13→r14 delta (three `refs.bib` corrections; MSC 2020 codes and keywords in `main.tex`; §0; this README; the change list).",
     "- **Reads of r14:** Cursor AMEND (main#389 6325580614), applied; readback AMEND (6325782917), applied; readback PASS (6325821191).\n- **Requested:** a nonauthor read of the r14→r15 delta (`refs.bib` completed from Crossref and zbMATH Open data, main#389 6325580607; §0; this README; the change list)."),
])
print('r15 edits applied')
