"""Supplement draft 15: radius and parameter matching made explicit (whole-document consistency read).

A fresh-context whole-document consistency read of draft 14 (same provider, Anthropic; no independence credit)
found no error and five places where a radius or parametrisation proved in one section is used in another without
the matching sentence (G1-G5), plus notation clashes. The author verified G1-G4 against the sources; this draft adds
the matching sentences. G5 (the source of the d=2 enclosure of c_{2,24}) and the notation clashes are recorded as
open in CONCORDANCE, not changed. No statement is strengthened; no number or label changes.

G1  S5: the normalizer bounds hold for 0<r<=r_0 of Lemma S2.9(iii); the radius of Theorem S5.36 is the least of the
    radii of the results its proof combines.
G2  S6 (proof of Proposition S6.4): Theorem S12.3 is applied with k = kappa/6 and a frame R with R e_1 = t.
G3  S7: the radius r_0 of (rate-r0) is also taken no larger than the r_0 of S12 (ssec:parent-majorant).
G4  S8 (Proposition S8.2): r_R <= r_0 of (rate-r0); S10 cites r_R in (0,1/2] of Proposition S8.2.
N-c2 Proposition S8.2(iv) for k >= 0 (it is used at k = 0 in Lemma S8.14, Lemma S9.9 and (third-W1)).
N-c3 Convention S8.1(b) and S10: the unrestricted densities cite (parent-135) and Lemma S12.29, not (parent-112).

Usage: python3 make_d15.py SUPPDIR   (applied to draft 14 at 860d82b4)
The checks/ changes (dep_graph_check.py hardening, test_dep_graph_check.py) are written directly.
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


edit('supp_s5_selection.tex', [
    (r"""By Section~\ref{sec:contact}, $c_Zr^2\le Z_r\le C_Zr^2$ on compact positive mark sets; in particular $Z_r>0$.""",
     r"""By Lemma~\ref{lem:contact-normalizer}, $Z_r>0$ for $0<r<12$, and $c_Zr^2\le Z_r\le C_Zr^2$ on a compact positive mark set for $0<r\le r_0$ of Lemma~\ref{lem:contact-normalizer}(iii)."""),
    (r"""Recall from Section~\ref{sec:contact} that, on compact positive mark sets,""",
     r"""Recall from Lemma~\ref{lem:contact-normalizer}(iii) that, on a compact positive mark set and for $0<r\le r_0$ of that lemma,"""),
    (r"""with all constants uniform in $t\in S^2$ and on the compact mark set.
\end{proof}""",
     r"""with all constants uniform in $t\in S^2$ and on the compact mark set. Here $r_0$ is the least of the radii for which Lemma~\ref{lem:contact-normalizer}(iii) on $I_b\times I_\kappa$ and the results just cited hold, so that each of them applies for $0<r\le r_0$.
\end{proof}"""),
])

edit('supp_s6_limit.tex', [
    (r"""By Theorem~\ref{thm:parent-A} at $d=3$ and $L=24$, whose selection probability is $p_r$ (Remark~\ref{rem:selection-parent}),""",
     r"""By Theorem~\ref{thm:parent-A} at $d=3$ and $L=24$, applied with gap mark $k=\kappa/6$, gap window $\{\kappa/6:\kappa\in I_\kappa\}$ for a compact $I_\kappa\subset(0,\infty)$ and any orthonormal frame $R$ with $Re_1=t$ (the conversion of the preceding paragraph), and whose selection probability is $p_r$ (Remark~\ref{rem:selection-parent}),"""),
])

edit('supp_s7_rate.tex', [
    (r"""and a pair at torus distance $r\le r_0$ has a unique displacement $S-M=rt$.""",
     r"""and a pair at torus distance $r\le r_0$ has a unique displacement $S-M=rt$. We also take $r_0$ no larger than the radius $r_0$ fixed in \S\ref{ssec:parent-majorant}, so that the results of Section~\ref{sec:parent} quoted in Hypotheses~\ref{hyp:rate-generic}--\ref{hyp:rate-far} hold for $0<r\le r_0$."""),
])

edit('supp_s8_remainder.tex', [
    (r"""There is $r_R\in(0,\frac12]$ such that, for $0<r\le r_R$, $b\in\R$ and every frame:""",
     r"""There is $r_R\in(0,\frac12]$, with $r_R\le r_0$ of \eqref{eq:rate-r0}, such that, for $0<r\le r_R$, $b\in\R$ and every frame:"""),
    (r"""\item For $k>0$, with $f=F_r$ and $A_0:=D_y^2F_0(0)$:""",
     r"""\item For $k\ge0$, with $f=F_r$ and $A_0:=D_y^2F_0(0)$:"""),
    (r"""Proposition~\ref{prop:parent-KR}(b), \eqref{eq:parent-112} and \eqref{eq:parent-135})""",
     r"""Proposition~\ref{prop:parent-KR}(b), \eqref{eq:parent-135} and Lemma~\ref{lem:parent-far})"""),
])

edit('supp_s10_third.tex', [
    (r"""There is $r_R\in(0,1]$ such that for $0<r\le r_R$:""",
     r"""With $r_R\in(0,\frac12]$ of Proposition~\ref{prop:remainder-R}, for $0<r\le r_R$:"""),
    (r"""Proposition~\ref{prop:parent-KR}(b), \eqref{eq:parent-112} and \eqref{eq:parent-135})""",
     r"""Proposition~\ref{prop:parent-KR}(b), \eqref{eq:parent-135} and Lemma~\ref{lem:parent-far})"""),
])

edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~14 of """ + date + ",", r"""Checkpoint draft~15 of """ + date + ","),
    (r"""\item \emph{Draft~14}""",
     r"""\item \emph{Draft~15} makes radius and parameter matching explicit at six places that a whole-document consistency read identified (normalizer radius and the radius of Theorem~\ref{thm:selection-main}; Theorem~\ref{thm:parent-A} in the variables of \S\ref{sec:limit}; the radii of \S\ref{sec:rate} against \S\ref{sec:parent}; $r_R$ in \S\S\ref{sec:remainder} and~\ref{sec:third}; Proposition~\ref{prop:remainder-R}(iv) at $k=0$; the density citation for the unrestricted densities). No statement is strengthened and no number changes.
\item \emph{Draft~14}"""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 14 (" + date + ")",
     "# SIDE24 V3 supplement: checkpoint draft 15 (" + date + ")"),
])
readme = open(d + '/README.md', encoding='utf-8').read()
readme = readme.rstrip('\n') + (
    "\n\nDraft 15 (" + date + ") adds the sentences that match radii and parameters across sections at six places "
    "found by a whole-document consistency read (CONCORDANCE, \"Checkpoint draft 15\"). No statement is strengthened "
    "and no number changes. Two findings of that read stay open there: the source of the d = 2 enclosure of c_{2,24} "
    "in Corollary S9.16, and the notation clashes (notably c_2[K] in §S4 and §S11).\n")
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

sec = ("## Checkpoint draft 15 (" + date + "): radius and parameter matching made explicit\n\n"
       "Author-side, by Claude01N, on top of draft 14. Source: a fresh-context whole-document consistency read of "
       "draft 14 (Anthropic, same provider as the author: no independence credit), which checked 46 cross-section "
       "citations against the cited statements and items and found no error. The author verified each finding below "
       "against the sources before changing anything.\n\n"
       "- **Applied (one sentence or clause each).**\n"
       "  - G1 §S5: the normalizer bounds hold for `0<r≤r_0` of Lemma S2.9(iii) (Definition S5.1 and (selection-normalizer) said only \"on compact positive mark sets\"); the proof of Theorem S5.36 now says its `r_0` is the least of the radii of the results it combines.\n"
       "  - G2 §S6, proof of Proposition S6.4: Theorem S12.3 is applied with `k=κ/6`, the window `{κ/6}` and a frame with `Re_1=t` (the conversion was written out only for the kernels).\n"
       "  - G3 §S7: the `r_0` of (rate-r0) is also taken no larger than the `r_0` of §S12 (ssec:parent-majorant), under which Hypotheses S7.4–S7.7's sources are proved.\n"
       "  - G4 §S8: Proposition S8.2 takes `r_R ≤ r_0` of (rate-r0), since its items come from §S7; §S10 now cites `r_R ∈ (0,1/2]` of Proposition S8.2 (it said `(0,1]`).\n"
       "  - N-c2: Proposition S8.2(iv) is stated for `k ≥ 0`; it is used at `k=0` in (remainder-W4), Lemma S9.9 and (third-W1). **For the reader to check:** that the bounds of (iv) hold at `k=0` as they do for `k>0` (item (i) is already stated for `k ≥ 0`).\n"
       "  - N-c3: Convention S8.1(b) and §S10 cite (parent-135) and Lemma S12.29 for the unrestricted densities, in place of (parent-112), which is the compact-window density of Theorem S12.4.\n"
       "- **Proof-dependency check hardened** after an OpenAI engineering probe (main#389 6325826874, findings DG-COV-01 to 04; "
       "applied by the author, written directly in `checks/`). The script exits 2, before building the graph, if the directory "
       "or a section file is missing or a `\\label` is defined twice (a duplicate could overwrite a statement and hide a cycle); "
       "the DOT export is sorted; each KNOWN component carries its internal edge count (`KNOWN_EDGES`), and a different count "
       "fails with CHANGED, so an explanation cannot outlive a change of the edges it explains. New `checks/test_dep_graph_check.py` "
       "(7 standard-library tests on synthetic inputs, including the probe's duplicate-label reproducer). On this draft: "
       "194 statements, 396 edges, 102 forward uses, seven explained components with unchanged internal edges, `NO_UNEXPLAINED_COMPONENT`.\n"
       "- **Open, not changed.**\n"
       "  - G5: Corollary S9.16 states `c_{2,24} ∈ [0.07340691930603427103, …104]` without a source. The ingredients (Theorem S4.4 at d = 2, Theorem S11.5, and `c_{2,ref} = c[K_ref] = c_{2,∞}`) are in the supplement, but the identification is stated only for d = 3. A sentence needs a nonauthor check of the d = 2 identification.\n"
       "  - Notation: `c_2[K]` is the 2-D leading coefficient in §S4 and the third-order expression in §S11; `J_0`, `𝒦`, `r_0`, `r_1`, `r_*`, `Ψ`, `ν_cand` are overloaded across sections (each locally defined). A renaming pass is left for the journal style pass.\n"
       "  - Presentation: no Part has an opening paragraph; the abstract's \"with their proofs\" for Part II should say that Theorems S9.13 and S11.5 are computer-assisted.\n")
conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
anchor = "## Checkpoint draft 14 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d15 edits applied')
