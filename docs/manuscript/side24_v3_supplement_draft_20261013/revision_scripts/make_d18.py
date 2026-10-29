"""Supplement draft 18 (notation, N1 only): the symbol c_2[K] is reserved for Definition S11.2.

The whole-document reads of drafts 14 and 15 flagged N1: the coefficient expression of Definition S4.2,
c_d[K], specialised to d = 2 was written c_2[K_24] and c_2[phi] in the proof of Corollary S9.16, while
c_2[K] in section S11 (Definition S11.2) and in the main text is the fixed-cone coefficient expression,
a different number. Draft 18 removes the specialised spelling at its only occurrence (S9) and states the
convention in one sentence after Definition S4.2. No number, statement or label changes.

Usage: python3 make_d18.py SUPPDIR   (applied to 71ee360c; then sh build.sh --update-sums)
"""
import sys

d = sys.argv[1]
date = '26 October 2026'


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


edit('supp_s9_cusp.tex', [
    (r"By Definition~\ref{def:correction-coef}, $c_{2,24}=c_2[K_{24}]$ and $c_{2,\mathrm{ref}}=c_2[\varphi]$, and Theorem~\ref{thm:correction-main} for $d=2$ gives",
     r"By Definition~\ref{def:correction-coef} at $d=2$, $c_{2,24}$ and $c_{2,\mathrm{ref}}$ are the values of the coefficient expression $c_d[\,\cdot\,]$ at $K_{24}$ and at $\varphi$ (the symbol $c_2[\,\cdot\,]$ is reserved for Definition~\ref{def:ctwo-c2}), and Theorem~\ref{thm:correction-main} for $d=2$ gives"),
])

edit('supp_s4_correction.tex', [
    (r"""The number $c_{d,\mathrm{ref}}$ is the value of the coefficient expression at the Euclidean covariance; its definition involves no persistence statement for a field on $\R^d$.""",
     r"""The number $c_{d,\mathrm{ref}}$ is the value of the coefficient expression at the Euclidean covariance; its definition involves no persistence statement for a field on $\R^d$. \emph{Notation.} At $d=2$ we write $c_{2,24}$ and $c_{2,\mathrm{ref}}$, never $c_2[K_{24}]$ or $c_2[\varphi]$: the symbol $c_2[K]$ is reserved for the fixed-cone coefficient expression of Definition~\ref{def:ctwo-c2} in Section~\ref{sec:ctwo}, a different quantity (for $K_{24}$, $c_{2,24}\approx0.0734$ while $c_2[K_{24}]\approx0.1612$)."""),
])

edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~17 of 26 October 2026,""", r"""Checkpoint draft~18 of """ + date + ","),
    (r"""\item \emph{Draft~17}""",
     r"""\item \emph{Draft~18} is a notation repair only (N1 of the whole-document reads): the coefficient expression of Definition~\ref{def:correction-coef} at $d=2$ is no longer written $c_2[\,\cdot\,]$ in the proof of Corollary~\ref{cor:cusp-torus}, and a sentence after Definition~\ref{def:correction-coef} reserves that symbol for Definition~\ref{def:ctwo-c2}. No number, statement or label changes.
\item \emph{Draft~17}"""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 17 (26 October 2026)",
     "# SIDE24 V3 supplement: checkpoint draft 18 (" + date + ")"),
    ("the compiled draft, 162 pages;", "the compiled draft, 163 pages;"),
])
readme = open(d + '/README.md', encoding='utf-8').read()
readme = readme.rstrip('\n') + (
    "\n\nDraft 18 (" + date + ") repairs one notation clash, N1 of the whole-document reads of drafts 14 and 15: the "
    "coefficient expression $c_d[K]$ of Definition S4.2 at $d=2$ was written $c_2[K_{24}]$, $c_2[\\varphi]$ in the proof of "
    "Corollary S9.16, the same symbol as the fixed-cone coefficient expression $c_2[K]$ of Definition S11.2 (and of the main "
    "text), a different number. The S9 sentence now names the values of $c_d[\\cdot]$ at $d=2$ without that spelling, and one "
    "sentence after Definition S4.2 reserves $c_2[\\cdot]$ for Definition S11.2. No number, statement or label changes; the "
    "other overloaded symbols (N2--N6) wait for the journal style pass.\n")
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

sec = ("## Checkpoint draft 18 (" + date + "): notation N1, `c_2[K]` reserved for Definition S11.2\n\n"
       "Author-side, by Claude01N, on top of draft 17 (`71ee360c`). Two TeX sentences change; no number, statement or label.\n\n"
       "- **Why.** The whole-document reads (`reads/WHOLE_READ_D14_ANTHROPIC_20261021.md`, N1; Grok Bot agent 9 on draft 15) found "
       "that `c_2[K]` has two meanings: the coefficient expression of Definition S4.2 at d = 2 (the 2-D leading coefficient; "
       "`c_{2,24} ≈ 0.0734`) and the fixed-cone coefficient expression of Definition S11.2 (`c_2[K_24] ≈ 0.1612`, the `c_2` of the "
       "main text's Theorem 2.3(iii)). The only place the first was spelled `c_2[·]` is the proof of Corollary S9.16, so a reader "
       "comparing S9.16 with Theorem 2.3(iii) met one symbol with two values.\n"
       "- **`supp_s9_cusp.tex`, proof of Corollary S9.16:** \"By Definition S4.2 at d = 2, c_{2,24} and c_{2,ref} are the values of "
       "the coefficient expression c_d[·] at K_24 and at φ (the symbol c_2[·] is reserved for Definition S11.2)\". The computation that "
       "follows is unchanged.\n"
       "- **`supp_s4_correction.tex`, after Definition S4.2:** one \"Notation\" sentence reserving `c_2[K]` for Definition S11.2, with "
       "the two approximate values so the distinction is visible.\n"
       "- **Build.** `sh build.sh --update-sums`; 163 pages (the added sentence after Definition S4.2 moves one page break; draft 17 had 162); all 943 labels keep their numbers; the manifest is rewritten for the PDF, "
       "the two TeX files, §S0, README and this file.\n")
conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
anchor = "## Checkpoint draft 17 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d18 edits applied')
