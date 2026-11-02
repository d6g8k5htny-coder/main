"""Main-text r12: apply Dylan's decision 4 ("Supplement only", main#389 6324313740).

Section 3 cites Supplement Theorem S12.3 (Theorem A of [R2], proved in Supplement Part III) for (3.13)
and Supplement Proposition S6.2 for Proposition 3.10, in place of [R2]. Section 0 records the four
decisions as decided. No theorem, display or constant changes.

Usage: python3 make_r12.py DRAFTDIR
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


edit('s3_proof.tex', [
    (r"""Corollary~\ref{cor:selection} is proved after Theorem~\ref{thm:escape}; its upper bound \eqref{eq:selection} is Theorem~A of \rec{parent}.""",
     r"""Corollary~\ref{cor:selection} is proved after Theorem~\ref{thm:escape}; its upper bound \eqref{eq:selection} is Supplement Theorem~S12.3, which states Theorem~A of \rec{parent} and proves it in Supplement Part~III."""),
    (r"""The bound \eqref{eq:selection} is Theorem~A of \rec{parent} at $d=3$ and $L=24$. That theorem is proved for every fixed $d\ge2$ and $L>0$,""",
     r"""The bound \eqref{eq:selection} is Supplement Theorem~S12.3 at $d=3$ and $L=24$; it is Theorem~A of \rec{parent}, with its proof in Supplement Part~III. That theorem is proved for every fixed $d\ge2$ and $L>0$,"""),
    (r"""This is the bound (14.1) of \rec{parent}. There it is stated for the pairs at distance at least a small radius $r_0$; its proof, a nonsingular two-site covariance on a compact set of pairs, applies at every separation $\varrho$ (Supplement Lemma S12.29 and Proposition S6.2).""",
     r"""This is Supplement Proposition~S6.2. Its proof bounds the elder mark by one and applies the marked two-point Kac--Rice formula (Supplement Proposition~S12.22) on the compact set of pairs at distance at least $\varrho$, where the value--gradient covariance is nonsingular; it cites no result of \rec{parent} that the Supplement does not itself prove. For $0<\ell\le1$ and pairs at distance at least the small radius $r_0$ of \rec{parent}, the same bound is its (14.1) (Supplement Lemma~S12.29)."""),
])

edit('s6_literature.tex', [
    (r"""Supplement Proposition~S6.2 proves Proposition~\ref{prop:far} without the parent record, and Supplement Theorem~S5.36 does the same for \eqref{eq:selection}.""",
     r"""Section~\ref{sec:proof} cites these Supplement statements and not the record: Supplement Theorem~S12.3 for \eqref{eq:selection}, and Supplement Proposition~S6.2 for Proposition~\ref{prop:far}, whose proof uses from Part~III only the marked Kac--Rice formula (Supplement Proposition~S12.22). Supplement Theorem~S5.36 gives a second, three-dimensional proof of \eqref{eq:selection}."""),
])

edit('s0_status.tex', [
    ("revised through r11 on 20 October 2026.", "revised through r12 on 21 October 2026."),
    (r"""\noindent\textbf{Author decisions, as drafted (defaults; each is a local switch).}""",
     r"""\noindent\textbf{Author decisions (decided).} The author kept the drafted defaults of decisions~1--3 and chose the Supplement-only form of decision~4 (main\#389, comment 6324313740)."""),
    (r""" Alternatives: put the three-term law into Theorem~\ref{thm:main}, or keep only the floor and move everything stronger to the Discussion.""",
     ""),
    (r"""Default: \S\ref{sec:literature} states which results are cases of the parent record \rec{parent} and which are specific to the side-24 field, and Supplement Part~III (Sections S12--S13) states and proves the parent record's results that the Supplement uses. The proofs of Corollary~\ref{cor:selection} and Proposition~\ref{prop:far} cite \rec{parent}. Alternatives: cite Supplement Theorem~S5.36 and Proposition~S6.2 there instead, or Supplement Theorem~S12.3 (Theorem~A of \rec{parent}, with its proof in Part~III) and Proposition~S6.2, so that the leading term rests on the Supplement alone; or publish the parent record as a companion preprint and cite it in place of Part~III.""",
     r"""Decided: the leading term rests on the Supplement alone. The proofs of Corollary~\ref{cor:selection} and Proposition~\ref{prop:far} cite Supplement Theorem~S12.3 (Theorem~A of \rec{parent}, with its proof in Supplement Part~III) and Supplement Proposition~S6.2. \S\ref{sec:literature} states which results are cases of the parent record and which are specific to the side-24 field, and Supplement Part~III (Sections S12--S13) states and proves the parent record's results that the Supplement uses."""),
    (r"""have the Supplement \rec{supp} read as a whole; settle decision~4; run""",
     r"""have the Supplement \rec{supp} read as a whole; run"""),
])

edit('README.md', [
    ("(13 October 2026; revision r11 of 20 October)", "(13 October 2026; revision r12 of 21 October)"),
    ("the compiled draft, revision r11, 23 pages.", "the compiled draft, revision r12, 23 pages."),
    ("Earlier revisions are kept at their commits: r10 (23 pages, SHA-256 `edf1bc77…`) at `d9f01444`,",
     "Earlier revisions are kept at their commits: r11 (23 pages, SHA-256 `cbfe9db6…`) at `b033a118`, r10 (23 pages, SHA-256 `edf1bc77…`) at `d9f01444`,"),
    ("""## Decisions still with the author

The draft uses defaults, and each is a local switch:
1. Theorem 2.1 states the two-provider form `c ℓ^{-1/3}(1 + O(ℓ^{1/3}))`. The merged three-term law is Theorem 2.3. The later refinements are Remark 2.4.
2. Human review is stated as a disclosure.
3. Planar (`d = 2`) material appears only in the Discussion.
4. The relation to the parent record [R2]: a §6 paragraph (added in r4) and, since r5, the parent record written out in Supplement Part III. The alternatives are to publish the parent record as a companion preprint, or to cite Supplement Theorem S5.36, or Supplement Theorem S12.3 (the parent's Theorem A with its proof in Part III), with Proposition S6.2, in §3, so that the leading term rests on the Supplement alone.""",
     """## Author decisions (decided; main#389 6324313740)

1. Theorem 2.1 states the two-provider form `c ℓ^{-1/3}(1 + O(ℓ^{1/3}))`. The merged three-term law is Theorem 2.3. The later refinements are Remark 2.4. (Default kept.)
2. Human review is stated as a disclosure. (Default kept.)
3. Planar (`d = 2`) material appears only in the Discussion. (Default kept.)
4. The relation to the parent record [R2]: **Supplement only** (applied in r12). §3 cites Supplement Theorem S12.3 (the parent's Theorem A with its proof in Part III) for (3.13) and Supplement Proposition S6.2 for Proposition 3.10. The §6 paragraph "Relation to the parent record" stays, and Supplement Part III writes the parent record out."""),
    ("- **Requested:** a Cursor read of the r10→r11 delta (documentary: Appendix A record [R1], the §0 date line, this README, the change list).",
     "- **Reads of r11:** the Cursor read of the r10→r11 delta passed (main#389 6302472048).\n- **Requested:** a nonauthor read of the r11→r12 delta (decision 4: the citations in §3 and the last sentence of the §6 paragraph; §0; this README; the change list)."),
    ("The draft is not for merge before the author's decisions.",
     "The author's decisions are made; submission still waits on the items in §0, \"Before submission\"."),
])
print('r12 edits applied')
