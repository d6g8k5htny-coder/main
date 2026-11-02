"""Main-text revision r8: cite supplement draft 9 and record the repair of the S5 finite-scale gap.
Usage: python3 make_r8.py DIR READS_SENTENCE_TEX READS_SENTENCE_MD
READS_* describe the reads of the repair, e.g.
  'the repair has nonauthor reads by OpenAI, by xAI and by a Grok Bot support agent (all PASS)'.
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

edit('appA_records.tex', [(
"""\\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~8 (14 October 2026): \\path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \\texttt{a39aeadd} of the branch \\path{claude/side24-v3-supplement-draft-20261013} in""",
"""\\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~9 (15 October 2026): \\path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \\texttt{4db2c57a} of the branch \\path{claude/side24-v3-supplement-draft9-20261015} in""")])

edit('s0_status.tex', [
("""Alternatives: once the open gap in Supplement~\\S S5 is closed (\\S\\ref{sec:proof}), cite Supplement Theorem~S5.36 and Proposition~S6.2 there instead, so that the leading term rests on the Supplement alone; or publish the parent record as a companion preprint and cite it in place of Part~III.""",
 """Alternatives: cite Supplement Theorem~S5.36 and Proposition~S6.2 there instead, or Supplement Theorem~S12.3 (Theorem~A of \\rec{parent}, with its proof in Part~III) and Proposition~S6.2, so that the leading term rests on the Supplement alone; or publish the parent record as a companion preprint and cite it in place of Part~III."""),
("""The second route in Supplement~\\S S5: one nonauthor read (+GB), and a provider-distinct read (OpenAI) that found an open gap in its finite-scale witness densities & \\rec{parent}, \\rec{supp} \\\\""",
 """The second route in Supplement~\\S S5: a provider-distinct read (OpenAI) found a gap in its finite-scale witness densities, which Supplement draft~9 repairs (\\S S5.8); """ + reads_tex + """ & \\rec{parent}, \\rec{supp} \\\\"""),
])

edit('s3_proof.tex', [
("""A provider-distinct read of that section found a gap in it, which is open. The collar and singular-near counts (Supplement Lemmas S5.19, S5.20 and S5.23) pass from contact-limit densities to the densities at fixed positive scale without a proof that is uniform near the pair axis. The proof of the corollary does not use this route.""",
 """A provider-distinct read of the eighth draft of that section found a gap in it: the collar and singular-near counts (Supplement Lemmas S5.19, S5.20 and S5.23) passed from contact-limit densities to the densities at a fixed positive scale without a proof that is uniform near the pair axis. Supplement \\S S5.8 (Lemmas S5.39--S5.41) closes it by a radial blow-up of the parameters at the axis, with explicit limit functionals, and the two counts are proved again from it; """ + reads_tex + """. The proof of the corollary does not use this route."""),
("""these are to give Palm expectations $O(r^3)$ for collar and singular-near witnesses (Supplement Lemmas S5.19 and S5.23), subject to the open gap just described.""",
 """these give Palm expectations $O(r^3)$ for collar and singular-near witnesses (Supplement Lemmas S5.19 and S5.23), with the finite-scale densities of Supplement \\S S5.8."""),
("""The witness route above outlines a second, three-dimensional argument (Supplement Theorem~S5.36), complete except for the open gap noted at the start of this subsection:""",
 """The witness route above outlines a second, three-dimensional argument (Supplement Theorem~S5.36), with the repair noted at the start of this subsection:"""),
])

edit('s6_literature.tex', [
("""and the second, three-dimensional route to \\eqref{eq:selection} in Supplement~\\S S5, which has an open gap in its finite-scale witness densities (\\S\\ref{sec:proof}).""",
 """and the second, three-dimensional route to \\eqref{eq:selection} in Supplement~\\S S5, whose finite-scale witness densities Supplement~\\S S5.8 supplies (\\S\\ref{sec:proof})."""),
("""Supplement Proposition~S6.2 proves Proposition~\\ref{prop:far} without the parent record, and Supplement Theorem~S5.36 would do the same for \\eqref{eq:selection} once that gap is closed.""",
 """Supplement Proposition~S6.2 proves Proposition~\\ref{prop:far} without the parent record, and Supplement Theorem~S5.36 does the same for \\eqref{eq:selection}."""),
])

edit('s7_disclosure.tex', [
("""(its second route in Supplement~\\S S5 has an open gap in its finite-scale witness densities; that route, with Lemma~S4.29 and Proposition~S8.13 of the Supplement, has one nonauthor read by a Grok Bot support agent)""",
 """(its second route in Supplement~\\S S5, with Lemma~S4.29 and Proposition~S8.13 of the Supplement, has one nonauthor read by a Grok Bot support agent; a provider-distinct read by OpenAI found a gap in that route's finite-scale witness densities, Supplement draft~9 repairs it, and """ + reads_tex + """)"""),
])

edit('README.md', [
("""The alternatives are to publish the parent record as a companion preprint, or, once the open gap in Supplement §S5 is closed, to cite Supplement Theorem S5.36 and Proposition S6.2 in §3, so that the leading term rests on the Supplement alone.""",
 """The alternatives are to publish the parent record as a companion preprint, or to cite Supplement Theorem S5.36, or Supplement Theorem S12.3 (the parent's Theorem A with its proof in Part III), with Proposition S6.2, in §3, so that the leading term rests on the Supplement alone."""),
("""The companion supplement (`claude/side24-v3-supplement-draft-20261013`), which r6 cites as [R1] at `a39aeadd`, has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts.""",
 """The companion supplement, which r8 cites as [R1] at `4db2c57a` (draft 9, branch `claude/side24-v3-supplement-draft9-20261015`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved (its "Checkpoint draft 9"); """ + reads_md + """."""),
("""- **Requested:** a read of r7.""",
 """- **Requested:** a read of r8."""),
])
print('r8 edits applied')
