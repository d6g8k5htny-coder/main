"""Main-text r13 (documentary): checklist items G5 and G7 of SUBMISSION_READINESS.md.

G5: cite Baryshnikov (J. Appl. Comput. Topol. 9 (2025) 28), the one-dimensional near-diagonal H_0 law for
    Brownian motion with drift, in the Section 6 paragraph on limit theorems and short features.
G7: Section 7, "Remaining before submission": the parent-record imports are settled by decision 4 (r12).
Also: Section 0 date line; README (r13, the r12 read recorded); no theorem, display or constant changes.

Usage: python3 make_r13.py DRAFTDIR   (applied to r12 plus its documentary additions, 8130d3a1)
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
    ("@misc{Heo2026,",
     """@article{Baryshnikov2025,
  author  = {Baryshnikov, Yuliy},
  title   = {Brownian motions, persistent homology and chirality},
  journal = {J. Appl. Comput. Topol.},
  volume  = {9},
  pages   = {Paper No. 28},
  year    = {2025},
  doi     = {10.1007/s41468-025-00224-w}
}

@misc{Heo2026,"""),
])

edit('s6_literature.tex', [
    (r"""that concerns rough paths and counts above a cutoff, a different regularity class.""",
     r"""that concerns rough paths and counts above a cutoff, a different regularity class. Baryshnikov \cite{Baryshnikov2025} computes the intensity of the $H_0$ persistence diagram of Brownian motion with drift $m$ in closed form; near the diagonal it grows like $1/(m\Delta^3)$ in the lifetime $\Delta$. This is the one-dimensional, rough-path counterpart of the near-diagonal question studied here. That paper says generalizing to higher-dimensional domains seems difficult but not entirely out of reach, at least for some classes of Gaussian processes, and will be pursued elsewhere."""),
])

edit('s7_disclosure.tex', [
    (r"""including its nonauthor reading, the removal of provenance labels, and the treatment of its imports from the parent record (\S\ref{sec:literature}); and""",
     r"""including a reading of it as a whole (each of its sections has had a nonauthor reading) and the removal of provenance labels; and"""),
])

edit('s0_status.tex', [
    ("revised through r12 on 21 October 2026.", "revised through r13 on 21 October 2026."),
])

edit('README.md', [
    ("(13 October 2026; revision r12 of 21 October)", "(13 October 2026; revision r13 of 21 October)"),
    ("the compiled draft, revision r12, 23 pages.", "the compiled draft, revision r13, 23 pages."),
    ("Earlier revisions are kept at their commits: r11 (23 pages, SHA-256 `cbfe9db6…`) at `b033a118`,",
     "Earlier revisions are kept at their commits: r12 (23 pages, SHA-256 `72f6b056…`) at `d550d238`, r11 (23 pages, SHA-256 `cbfe9db6…`) at `b033a118`,"),
    ("- **Requested:** a nonauthor read of the r11→r12 delta (decision 4: the citations in §3 and the last sentence of the §6 paragraph; §0; this README; the change list).",
     "- **Reads of r12:** the Cursor read of the r11→r12 delta passed (main#389 6324514156).\n- **Requested:** a nonauthor read of the r12→r13 delta (the Baryshnikov citation in §6 and `refs.bib`; the §7 sentence; §0; this README; the change list)."),
])
print('r13 edits applied')
