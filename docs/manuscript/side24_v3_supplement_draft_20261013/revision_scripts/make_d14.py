"""Supplement draft 14 (documentary): remove provenance wording from the mathematical sections.

The supplement writers' rule (COMMON rule 3, quoted in CONCORDANCE.md: no provenance labels, status words or
historical comparisons in the LaTeX) and checklist item G11 of the main text's SUBMISSION_READINESS.md.
Outside the status section S0, eleven clauses named earlier drafts ("drafts 1--8 ...") and one sentence named a
repository path. Each is removed or restated without the history; the mathematical content is kept; the history
is recorded verbatim in CONCORDANCE.md, "Checkpoint draft 14". No statement, number or label changes.

Usage: python3 make_d14.py SUPPDIR   (applied to draft 13 as at f8b71cab; TeX identical to d28e6ab4)
"""
import sys

d = sys.argv[1]
date = '21 October 2026'


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


# The removed or restated wording, kept for the concordance (old, new).
S5 = [
    (r"""That script subtracts $f(x)$, as drafts~1--8 did, but the pins do not fix $f(x)$.""",
     r"""That script subtracts $f(x)$, but the pins do not fix $f(x)$."""),
    (r"""Drafts~1--8 stated in its place an integral of $\rho^{-m}$ times the density over $S^2$, bounded by $C_ms^{-7}$; at a fixed positive scale that integral diverges for $m\ge2$, because the density is positive on the axis.""",
     r"""An integral of $\rho^{-m}$ times the density over $S^2$ would not serve in its place: at a fixed positive scale that integral diverges for $m\ge2$, because the density is positive on the axis."""),
    (r"""Drafts~1--8 bounded these moments by a power of $1+\rho^{-1}$, through a lower bound $\rho^{10}$ for the third-point block that holds only in the contact limit.""",
     r"""A bound of these moments by a power of $1+\rho^{-1}$, through a lower bound $\rho^{10}$ for the third-point block, would not serve here: that lower bound holds only in the contact limit."""),
    (r"""So a bound by $\rho^{-m}$ times a function of the scale alone cannot be integrated over the directions, and drafts~1--8 of this supplement asserted such a bound in \eqref{eq:selection-singdens} (\path{CONCORDANCE.md}, ``Checkpoint draft~9'').""",
     r"""So a bound by $\rho^{-m}$ times a function of the scale alone cannot be integrated over the directions."""),
]
S7 = [
    (r"""Theorem~\ref{thm:rate-bounded} is the statement of Theorem~R (\texttt{frontiers/three\_fronts\_20260924/LIFETIME\_REMAINDER.md}, blob \texttt{247b3ecf}) at every fixed $d\ge 2$ and $L>0$.""",
     r"""Theorem~\ref{thm:rate-bounded} is the statement of Theorem~R of record [R3] of the main text at every fixed $d\ge 2$ and $L>0$."""),
]
S8 = [
    (r""" and $r_Z:=r_K$ (drafts~1--11 took $r_Z$ to be the radius of the near-diagonal analysis of Theorem~Z of record [R7] of the main text, which the proof of Proposition~\ref{prop:remainder-Z} in \S\ref{ssec:remainder-Z} does not use).""",
     r""" and $r_Z:=r_K$."""),
    (r"""Together the two bounds are Theorem~K of record [R7] of the main text, which drafts~1--10 cited for them.""",
     r"""Together the two bounds are Theorem~K of record [R7] of the main text."""),
    (r"""Items (i) and (ii) are Theorem~Z of record [R7] of the main text, which drafts~1--11 cited for them; they are proved""",
     r"""Items (i) and (ii) are Theorem~Z of record [R7] of the main text; they are proved"""),
    (r"""This subsection proves the second bound of Proposition~\ref{prop:remainder-K}, which drafts~1--10 imported from Theorem~K of record [R7] of the main text.""",
     r"""This subsection proves the second bound of Proposition~\ref{prop:remainder-K}, which is also part of Theorem~K of record [R7] of the main text."""),
    (r"""This subsection proves items (i) and (ii) of Proposition~\ref{prop:remainder-Z}, which drafts~1--11 imported from Theorem~Z of record [R7] of the main text.""",
     r"""This subsection proves items (i) and (ii) of Proposition~\ref{prop:remainder-Z}, which are also Theorem~Z of record [R7] of the main text."""),
]
S11 = [
    (r"""Drafts~1--9 stated them as a hypothesis; both follow from Section~\ref{sec:third}.""",
     r"""Both follow from Section~\ref{sec:third}."""),
]
edit('supp_s5_selection.tex', S5)
edit('supp_s7_rate.tex', S7)
edit('supp_s8_remainder.tex', S8)
edit('supp_s11_ctwo.tex', S11)

edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~13 of 20 October 2026,""", r"""Checkpoint draft~14 of """ + date + ","),
    (r"""and draft~13 adds a proof of Proposition~\ref{prop:parent-push}, makes two item-level dependencies explicit, and ships a proof-dependency check (below).""",
     r"""draft~13 adds a proof of Proposition~\ref{prop:parent-push}, makes two item-level dependencies explicit, and ships a proof-dependency check, and draft~14 removes the remaining references to earlier drafts and to repository paths from Sections~\ref{sec:setting}--\ref{sec:cap} (below)."""),
    (r"""\item \emph{Draft~13}""",
     r"""\item \emph{Draft~14} is documentary. Outside this section, eleven clauses that named earlier drafts and one sentence that named a repository path are removed or restated without the history; the mathematics they carried is kept. The concordance records the old wording. No statement, number or label changes.
\item \emph{Draft~13}"""),
    (r"""and draft~13 changes no number (a proof carries none).""",
     r"""draft~13 changes no number (a proof carries none), and neither does draft~14."""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 13 (20 October 2026)",
     "# SIDE24 V3 supplement: checkpoint draft 14 (" + date + ")"),
    ("  - in \"Checkpoint draft 13\", a proof of Proposition S12.24, two item-level dependency sentences, and the proof-dependency check.",
     "  - in \"Checkpoint draft 13\", a proof of Proposition S12.24, two item-level dependency sentences, and the proof-dependency check;\n"
     "  - in \"Checkpoint draft 14\", the removal of references to earlier drafts and to repository paths from the mathematical sections, with the old wording."),
])
readme = open(d + '/README.md', encoding='utf-8').read()
readme = readme.rstrip('\n') + (
    "\n\nDraft 14 (" + date + ") is documentary. Outside the status section, eleven clauses that named earlier drafts "
    "(\"drafts 1–8 …\") and one sentence that named a repository path are removed or restated without the history; "
    "the mathematics they carried is kept, and `CONCORDANCE.md`, \"Checkpoint draft 14\", records the old wording. "
    "No statement, number or label changes.\n")
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

rows = []
for fn, lst in (('supp_s5_selection.tex', S5), ('supp_s7_rate.tex', S7),
                ('supp_s8_remainder.tex', S8), ('supp_s11_ctwo.tex', S11)):
    for old, new in lst:
        rows.append("  - `%s`\n    - was: %s\n    - now: %s" % (fn, old, new if new.strip() else '(removed)'))
sec = ("## Checkpoint draft 14 (" + date + "): provenance wording removed from the mathematical sections\n\n"
       "Author-side, by Claude01N, on top of draft 13 (`d28e6ab4`). Documentary. The supplement writers' rule (COMMON rule 3, quoted further down this file: "
       "no provenance labels, status words or historical comparisons in the LaTeX) is applied to the clauses that later "
       "drafts added. Outside §S0, eleven clauses named earlier drafts and one sentence named a repository path. "
       "Each is removed or restated without the history; any mathematical content is kept. No statement, number or label changes. "
       "`\\path{checks/...}` references to the scripts shipped with this supplement are kept: they are reproducibility pointers, not provenance. "
       "Citations of the companion record [R2] and of records [R3] and [R7] of the main text are kept.\n\n"
       "- **The changed wording (old → new):**\n" + "\n".join(rows) + "\n"
       "- **Build.** See README and `SHA256SUMS`.\n")
conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
anchor = "## Checkpoint draft 13 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d14 edits applied')
