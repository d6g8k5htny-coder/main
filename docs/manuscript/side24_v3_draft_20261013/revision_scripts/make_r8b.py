"""Main-text revision r8 (version b).

Cites supplement draft 9, records the repair of the S5 finite-scale gap with its read status,
and carries five citation/wording fixes from a same-provider audit of Supplement S2-S7 and S12-S13.

Usage: python3 make_r8b.py DIR READS_SENTENCE_TEX READS_SENTENCE_MD
READS_* describe the reads of the repair, for example
  'the repair has one nonauthor read so far, by a Grok Bot support agent (PASS)'.
"""
import sys

d, reads_tex, reads_md = sys.argv[1], sys.argv[2], sys.argv[3]


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:80]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


# ---------------------------------------------------------------- Appendix A
edit('appA_records.tex', [(
    r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~8 (14 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{a39aeadd} of the branch \path{claude/side24-v3-supplement-draft-20261013} in""",
    r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~9 (15 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{4db2c57a} of the branch \path{claude/side24-v3-supplement-draft9-20261015} in""")])

# ---------------------------------------------------------------- Section 0
edit('s0_status.tex', [
    (r"""It was assembled on 13 October 2026 and revised through r7 on 15 October 2026.""",
     r"""It was assembled on 13 October 2026 and revised through r8 on 15 October 2026."""),
    (r"""Alternatives: once the open gap in Supplement~\S S5 is closed (\S\ref{sec:proof}), cite Supplement Theorem~S5.36 and Proposition~S6.2 there instead, so that the leading term rests on the Supplement alone; or publish the parent record as a companion preprint and cite it in place of Part~III.""",
     r"""Alternatives: cite Supplement Theorem~S5.36 and Proposition~S6.2 there instead, or Supplement Theorem~S12.3 (Theorem~A of \rec{parent}, with its proof in Part~III) and Proposition~S6.2, so that the leading term rests on the Supplement alone; or publish the parent record as a companion preprint and cite it in place of Part~III."""),
    (r"""The second route in Supplement~\S S5: one nonauthor read (+GB), and a provider-distinct read (OpenAI) that found an open gap in its finite-scale witness densities & \rec{parent}, \rec{supp} \\""",
     r"""The second route in Supplement~\S S5: a provider-distinct read (OpenAI) found a gap in its finite-scale witness densities, which Supplement draft~9 repairs (\S S5.8); """ + reads_tex + r""" & \rec{parent}, \rec{supp} \\"""),
])

# ---------------------------------------------------------------- Section 1 (audit note S26-3)
edit('s1_field.tex', [(
    r"""In the proof below, $\nu_{3,24}$ denotes the pointwise Kac--Rice representative obtained from the pair coordinates and the lifetime pushforward (Convention~\ref{conv:density}).""",
    r"""In the proof below, $\nu_{3,24}$ denotes the pointwise Kac--Rice representative of Convention~\ref{conv:density}.""")])

# ---------------------------------------------------------------- Section 2 (audit note S26-3)
edit('s2_results.tex', [(
    r"""refer to their canonical marked Kac--Rice density versions, the representatives obtained from the pair coordinates of \S\ref{ssec:pairs}; a change of density on a null set need not preserve a pointwise asymptotic.""",
    r"""refer to their canonical marked Kac--Rice density versions: the representatives given by the marked two-point formula, written in the pair coordinates of \S\ref{ssec:pairs} for separations below a fixed $\varrho\in(0,12)$ and in the coordinates $(M,S)$ for the larger separations (Supplement (S6.7) and (S6.10) for $\nu_{3,24}$, and the same formulas without the elder mark for the candidate density). The substitution $\kappa=6\ell/r^3$ shows that their sum does not depend on $\varrho$. A change of density on a null set need not preserve a pointwise asymptotic.""")])

# ---------------------------------------------------------------- Section 3
edit('s3_proof.tex', [
    # audit note S26-6
    (r"""with uniformly bounded conditional Gaussian moments; consequently the two endpoint determinant weights contribute $r^2$.""",
     r"""with conditional Gaussian moments bounded by a polynomial in $|b|+\kappa$, uniformly in $t$ and in $r$ below any fixed $\varrho<12$, and hence uniformly bounded on compact mark sets (Supplement Lemma S2.9 and the proof of Proposition S6.4); consequently the two endpoint determinant weights contribute $r^2$."""),
    # the S5 repair (section 3.4)
    (r"""A provider-distinct read of that section found a gap in it, which is open. The collar and singular-near counts (Supplement Lemmas S5.19, S5.20 and S5.23) pass from contact-limit densities to the densities at fixed positive scale without a proof that is uniform near the pair axis. The proof of the corollary does not use this route.""",
     r"""A provider-distinct read of the eighth draft of that section found a gap in it: the collar and singular-near counts (Supplement Lemmas S5.19, S5.20 and S5.23) passed from contact-limit densities to the densities at a fixed positive scale without a proof that is uniform near the pair axis. Supplement draft~9 repairs it in \S S5.8 (Lemmas S5.39--S5.41), by a radial blow-up of the parameters at the axis with explicit limit functionals, and proves the two counts again from it (\S\ref{sec:disclosure} records the reads of this repair). The proof of the corollary does not use this route."""),
    (r"""these are to give Palm expectations $O(r^3)$ for collar and singular-near witnesses (Supplement Lemmas S5.19 and S5.23), subject to the open gap just described.""",
     r"""these give Palm expectations $O(r^3)$ for collar and singular-near witnesses (Supplement Lemmas S5.19 and S5.23), with the finite-scale densities of Supplement \S S5.8."""),
    (r"""The witness route above outlines a second, three-dimensional argument (Supplement Theorem~S5.36), complete except for the open gap noted at the start of this subsection:""",
     r"""The witness route above outlines a second, three-dimensional argument (Supplement Theorem~S5.36), with the repair noted at the start of this subsection:"""),
    # audit notes S26-1 / S12-N2
    (r"""This is the bound (14.1) of \rec{parent}.""",
     r"""This is the bound (14.1) of \rec{parent}. There it is stated for the pairs at distance at least a small radius $r_0$; its proof, a nonsingular two-site covariance on a compact set of pairs, applies at every separation $\varrho$ (Supplement Lemma S12.29 and Proposition S6.2)."""),
    # audit note S34-5
    (r"""In the jet coordinates $a$, $m_4$, $\chi$, $\Omega$ of the technical appendix (Module~H; Supplement Remark S4.26),""",
     r"""In the jet coordinates $a$, $m_4$, $\chi$, $\Omega$ of the technical appendix (Module~H; Supplement Lemma S4.20 and Corollary S4.25),"""),
    (r"""and the scale check $d\log c/d\log\lambda=3/2$ for the kernel $e^{-\lambda|z|^2/2}$;""",
     r"""and the scale check $d\log c/d\log\lambda=3/2$ for the kernel $e^{-\lambda|z|^2/2}$ (Supplement Remark S4.26);"""),
    # audit note A7-1
    (r"""This is Theorem~R of \rec{thmR} at $d=3$ and $L=24$; Theorem~R holds for every fixed $d\ge2$ and $L>0$, for all marks and separations, with existential constants.""",
     r"""This is Theorem~R of \rec{thmR} at $d=3$ and $L=24$, proved in Supplement Theorem~S7.9; Supplement Remark~S7.22 identifies its density and its constant with those of Theorem~\ref{thm:main}. Theorem~R holds for every fixed $d\ge2$ and $L>0$, for all marks and separations, with existential constants."""),
])

# ---------------------------------------------------------------- Section 6
edit('s6_literature.tex', [
    # audit note S26-1
    (r"""the selection bound \eqref{eq:selection} is its Theorem~A, and Proposition~\ref{prop:far} is its (14.1).""",
     r"""the selection bound \eqref{eq:selection} is its Theorem~A, and Proposition~\ref{prop:far} is its (14.1), whose proof applies at every separation."""),
    (r"""and the second, three-dimensional route to \eqref{eq:selection} in Supplement~\S S5, which has an open gap in its finite-scale witness densities (\S\ref{sec:proof}).""",
     r"""and the second, three-dimensional route to \eqref{eq:selection} in Supplement~\S S5, whose finite-scale witness densities Supplement~\S S5.8 supplies (\S\ref{sec:proof})."""),
    (r"""Supplement Proposition~S6.2 proves Proposition~\ref{prop:far} without the parent record, and Supplement Theorem~S5.36 would do the same for \eqref{eq:selection} once that gap is closed.""",
     r"""Supplement Proposition~S6.2 proves Proposition~\ref{prop:far} without the parent record, and Supplement Theorem~S5.36 does the same for \eqref{eq:selection}."""),
])

# ---------------------------------------------------------------- Section 7
edit('s7_disclosure.tex', [(
    r"""(its second route in Supplement~\S S5 has an open gap in its finite-scale witness densities; that route, with Lemma~S4.29 and Proposition~S8.13 of the Supplement, has one nonauthor read by a Grok Bot support agent)""",
    r"""(its second route in Supplement~\S S5, with Lemma~S4.29 and Proposition~S8.13 of the Supplement, has one nonauthor read by a Grok Bot support agent; a provider-distinct read by OpenAI found a gap in that route's finite-scale witness densities, Supplement draft~9 repairs it, and """ + reads_tex + r""")""")])

# ---------------------------------------------------------------- README
edit('README.md', [
    ("""# SIDE24 manuscript — V3 draft (13 October 2026; revision r7 of 15 October)""",
     """# SIDE24 manuscript — V3 draft (13 October 2026; revision r8 of 15 October)"""),
    ("""- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r7, 22 pages. Its first section, "Status of this draft", is for the author and is removed before submission. Earlier revisions are kept at their commits: r6 (22 pages, SHA-256 `602762a2…`) at `398cf6fb`,""",
     """- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r8, PAGES_R8 pages. Its first section, "Status of this draft", is for the author and is removed before submission. Earlier revisions are kept at their commits: r7 (22 pages, SHA-256 `af944708…`) at `5543bd20`, r6 (22 pages, SHA-256 `602762a2…`) at `398cf6fb`,"""),
    ("""The alternatives are to publish the parent record as a companion preprint, or, once the open gap in Supplement §S5 is closed, to cite Supplement Theorem S5.36 and Proposition S6.2 in §3, so that the leading term rests on the Supplement alone.""",
     """The alternatives are to publish the parent record as a companion preprint, or to cite Supplement Theorem S5.36, or Supplement Theorem S12.3 (the parent's Theorem A with its proof in Part III), with Proposition S6.2, in §3, so that the leading term rests on the Supplement alone."""),
    ("""- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3, the changed passages of r4, and each AMEND's readback. The companion supplement (`claude/side24-v3-supplement-draft-20261013`), which r6 cites as [R1] at `a39aeadd`, has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts.""",
     """- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r7, and each AMEND's readback. The companion supplement, which r8 cites as [R1] at `4db2c57a` (draft 9, branch `claude/side24-v3-supplement-draft9-20261015`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved (its "Checkpoint draft 9"); """ + reads_md + """."""),
    ("""- **Requested:** a read of r7.""",
     """- **Requested:** a read of r8."""),
])
print('r8b edits applied')
