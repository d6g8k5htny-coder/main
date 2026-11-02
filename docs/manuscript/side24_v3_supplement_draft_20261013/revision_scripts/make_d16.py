"""Supplement draft 16: the source of the d = 2 enclosure of c_{2,24} in Corollary S9.16 (gap G5).

G5 of the whole-document read of draft 14 (reads/WHOLE_READ_D14_ANTHROPIC_20261021.md), confirmed as a GAP by
Grok Bot agent 3 (main#389 6326409144): Corollary S9.16 quoted c_{2,24} in [0.07340691930603427103, ...104]
without a source; the d = 2 value of c_{2,ref} was not computed anywhere, and the certified table of Theorem
S11.5 has only 16 digits. Draft 16 adds one passage to the proof of Corollary S9.16 and a check script
(checks/s9_c2_d2_check.py: the d = 2 moments of phi symbolically, and a 300-digit interval enclosure through
Theorem S4.4). No statement changes; no number or label changes.

Usage: python3 make_d16.py SUPPDIR   (applied after the A1-A4 commit c41e84e8 on top of draft 15 ddd60647)
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


edit('supp_s9_cusp.tex', [
    (r"""the factor $[1-\eta,1+\eta]$ and the quoted enclosures of $c_{3,24}$ and $c_{2,24}$.
\end{proof}""",
     r"""the factor $[1-\eta,1+\eta]$ and the quoted enclosures of $c_{3,24}$ and $c_{2,24}$. The enclosure of $c_{2,24}$ is obtained as follows. By Definition~\ref{def:correction-coef}, $c_{2,24}=c_2[K_{24}]$ and $c_{2,\mathrm{ref}}=c_2[\varphi]$, and Theorem~\ref{thm:correction-main} for $d=2$ gives $c_{2,24}=c_{2,\mathrm{ref}}\bigl(1+P_2(24)q+\varepsilon\bigr)$ with $|P_2(24)q|<3\cdot10^{-118}$ and $|\varepsilon|<10^{-216}$. For $d=2$ and $K=\varphi$, the quantities of Definition~\ref{def:correction-coef} do not depend on $t$: $\Cov g_t=I_2$, $\Cov h_t=\mathrm{diag}(3,1)$, $\tau_t^2=15-9=6$, and $A_t=\partial_n^2f(0)$ has conditional variance $3-\frac13=\frac83$ given $h_t=0$, so $D_t=\frac43$. Hence \eqref{eq:correction-coef} gives $c_{2,\mathrm{ref}}=\Gamma(\frac76)(\frac32)^{1/3}\frac43/(2\sqrt3\,\pi^{3/2})=\Gamma(\frac16)\,12^{-1/6}/(9\pi^{3/2})$, the value of $c[K_{\mathrm{ref}}]$ for $d=2$ in Theorem~\ref{thm:ctwo-closed}. Outward-rounded interval arithmetic at $300$ digits (\path{checks/s9_c2_d2_check.py}, which also derives these moments symbolically) encloses this number and the factor, and gives the stated enclosure of $c_{2,24}$.
\end{proof}"""),
])

edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~15 of """ + date + ",", r"""Checkpoint draft~16 of """ + date + ","),
    (r"""\item \emph{Draft~15}""",
     r"""\item \emph{Draft~16} gives the source of the $d=2$ enclosure of $c_{2,24}$ in Corollary~\ref{cor:cusp-torus}: one passage in its proof computes $c_{2,\mathrm{ref}}$ from Definition~\ref{def:correction-coef} and transfers a certified enclosure through Theorem~\ref{thm:correction-main}, with a check script. No statement or number changes.
\item \emph{Draft~15}"""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 15 (" + date + ")",
     "# SIDE24 V3 supplement: checkpoint draft 16 (" + date + ")"),
])
readme = open(d + '/README.md', encoding='utf-8').read()
readme = readme.rstrip('\n') + (
    "\n\nDraft 16 (" + date + ") closes the open item G5 of draft 15: the proof of Corollary S9.16 now derives the "
    "d = 2 value c_{2,ref} = Γ(1/6)·12^{-1/6}/(9π^{3/2}) from Definition S4.2 and transfers a certified 300-digit "
    "interval enclosure to c_{2,24} through Theorem S4.4 (`checks/s9_c2_d2_check.py`). No statement or number changes.\n")
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

sec = ("## Checkpoint draft 16 (" + date + "): the source of the d = 2 enclosure in Corollary S9.16\n\n"
       "Author-side, by Claude01N, on top of draft 15 (`ddd60647`). It closes item G5 of the whole-document read, "
       "which Grok Bot agent 3 confirmed as a GAP (main#389 6326409144). That agent's chain and proposed sentence "
       "were the starting point; its second point, that the 16-digit table of Theorem S11.5 cannot certify a "
       "20-digit interval, is met by a direct interval evaluation.\n\n"
       "- **Proof of Corollary S9.16, one added passage.** `c_{2,24} = c_2[K_24]` and `c_{2,ref} = c_2[φ]` (Definition S4.2). "
       "Theorem S4.4 at d = 2 gives `c_{2,24} = c_{2,ref}(1 + P_2(24)q + ε)` with `|P_2(24)q| < 3·10^{-118}`. At d = 2 for `φ`: "
       "`Cov g_t = I_2`, `Cov h_t = diag(3,1)`, `τ_t^2 = 6`, `Var(A_t | h_t = 0) = 8/3`, so `D_t = 4/3`. Hence "
       "`c_{2,ref} = Γ(1/6)·12^{-1/6}/(9π^{3/2})`, the d = 2 value of `c[K_ref]` in Theorem S11.5 (written with `c[·]`, not `c_2[·]`, "
       "to avoid the §S4/§S11 notation clash).\n"
       "- **`checks/s9_c2_d2_check.py`** (new; SymPy and mpmath). It derives the four moments symbolically and checks the "
       "closed form. It then encloses `c_{2,ref}` and `c_{2,24}` in `mpmath.iv` at 300 digits, and asserts that the "
       "enclosure lies in `[0.07340691930603427103, 0.07340691930603427104]` and that `c_{2,24} < c_{2,ref}`. "
       "Output: `checks/s9_c2_d2_check.out.txt`, `CERTIFIED`.\n"
       "- **Also in draft 16 (its first commit):** the AMEND of the provider-distinct whole-document read of draft 15 "
       "(Grok Bot agent 9, main#389 6326419744): A1, the §S6 pointer to the `k=κ/6` conversion; A2, the source of Proposition "
       "S8.2(iv) at `k=0` in its proof (no division by `k` in the §S7 proofs; target-independent `Σ_r`; the endpoint identities "
       "(parent-52) hold for every gap mark; the determinant comparison (iii) allows singular blocks); A3, the records now say "
       "Proposition S8.2(iv) is the one strengthened statement; A4, `r_R ≤ 1/2` in §S10. The new citations add four edges "
       "to the dependency graph; the seven components and their internal edges are unchanged.\n"
       "- **Still open from draft 15:** the notation clashes and the abstract's wording for the computer-assisted theorems.\n")
conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
anchor = "## Checkpoint draft 15 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d16 edits applied')
