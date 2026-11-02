"""Supplement draft 13, documentary: a proof for Proposition S12.24; the item-level dependency between
Propositions S12.22 and S12.23 and between Lemmas S5.20 and S5.40 made explicit; a proof-dependency check
script shipped in checks/. No number of draft 12 changes (a proof environment carries no number).

Usage: python3 make_d13.py SUPPDIR
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


edit('supp_s12_parent.tex', [
    # Proposition S12.24: a proof (drafts 1-12 stated it without one)
    (r"""and $\nu_\eld$ has the additional factor $p_{(\ell/k)^{1/3}}$ in the integrand.
\end{proposition}

\begin{proof}[Proof of Theorem~\ref{thm:parent-B}]""",
     r"""and $\nu_\eld$ has the additional factor $p_{(\ell/k)^{1/3}}$ in the integrand.
\end{proposition}

\begin{proof}
For $k\in\mathcal K$, $k\ge k_-$, so $\ell<k_-r_*^3$ gives $r^3=\ell/k<r_*^3$. At fixed $k$, $dr/d\ell=\frac13k^{-1/3}\ell^{-2/3}$, and multiplying by $r=k^{-1/3}\ell^{1/3}$ gives \eqref{eq:parent-111}. The candidate intensity \eqref{eq:parent-102} is $r\,\mathcal A_r\,dr\,db\,dk\,d\sigma(t)$ per unit midpoint volume; its pushforward under $\ell=kr^3$ at fixed $(b,k,t)$ replaces $r\,dr$ by $\frac13k^{-2/3}\ell^{-1/3}\,d\ell$, by \eqref{eq:parent-111}, and integrating over $\mathcal B\times\mathcal K\times S^{d-1}$ gives \eqref{eq:parent-112}. The selected intensity is $r\,\mathcal A_r\,p_r$ with the same measures (Proposition~\ref{prop:parent-ledger}(b)), whence the factor $p_{(\ell/k)^{1/3}}$.
\end{proof}

\begin{proof}[Proof of Theorem~\ref{thm:parent-B}]"""),
    # S12.23's proof uses only (a) and (b) of S12.22; S12.22(c) uses S12.23: acyclic item by item
    (r"""There is no factor $1/2$: maximum--saddle pairs are ordered by their different roles. With (a) and Proposition~\ref{prop:parent-KR}, the powers of $r$ are""",
     r"""There is no factor $1/2$: maximum--saddle pairs are ordered by their different roles. With (a) and Proposition~\ref{prop:parent-KR}(a),(b) (this proof does not use its part~(c), whose proof uses the present proposition), the powers of $r$ are"""),
    (r"""\emph{Exhaustion.} Finally, exhaust $\Omega$ towards the diagonal. The bound of Proposition~\ref{prop:parent-ledger} is locally $r\,dr$, hence integrable at zero,""",
     r"""\emph{Exhaustion.} Finally, exhaust $\Omega$ towards the diagonal. The bound of Proposition~\ref{prop:parent-ledger}, whose proof uses only parts~(a) and~(b) above, is locally $r\,dr$, hence integrable at zero,"""),
])

edit('supp_s5_selection.tex', [
    (r"""These are statements about the contact limit. The finite-scale bound \eqref{eq:selection-singdens} is Lemma~\ref{lem:selection-finsing}(d),(e). Drafts~1--8 stated""",
     r"""These are statements about the contact limit. The finite-scale bound \eqref{eq:selection-singdens} is Lemma~\ref{lem:selection-finsing}(d),(e); the proof of that lemma uses from the present one only the window \eqref{eq:selection-window}, the chart \eqref{eq:selection-V0}--\eqref{eq:selection-V123} with its Jacobian, the residual limits \eqref{eq:selection-L0}--\eqref{eq:selection-L23} and, through Lemma~\ref{lem:selection-singfactor}, the covariances \eqref{eq:selection-C01}--\eqref{eq:selection-det4}, all proved above, so there is no circularity. Drafts~1--8 stated"""),
])
print('d13 TeX edits applied')
