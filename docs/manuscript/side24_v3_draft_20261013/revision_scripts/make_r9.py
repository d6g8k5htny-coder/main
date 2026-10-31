"""Main-text revision r9: status of the reads since r8, supplement draft 10 as [R1], and (optionally)
the unconditional identification of c_2 (Supplement Corollary S11.11 of draft 10).

Usage: python3 make_r9.py DIR DATE D10_COMMIT D10_BRANCH D10_PAGES S1111(yes|no) S1111_READ_TEX S1111_READ_MD DRAFTNO
  DRAFTNO  the supplement draft that r9 cites as [R1] (10 or 11)
  S1111_READ_TEX / _MD  the read status of Supplement Corollary S11.11 at r9, e.g.
                        'a Grok Bot read (PASS)' (LaTeX-safe) and the same in Markdown with the comment id
"""
import sys

d, date, d10c, d10b, d10p, s1111, rd_tex, rd_md, dno = sys.argv[1:10]
s1111 = s1111 == 'yes'


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


reads_tex = (r"""The second route in Supplement~\S S5 has provider-distinct reads: by xAI of \S\S S5.1--S5.3 and S5.5--S5.7 (PASS\_SCOPED), and by OpenAI of \S S5.4, which found a gap in its finite-scale witness densities; Supplement draft~9 repairs it (\S S5.8), and the repair has nonauthor reads by OpenAI (PASS\_SCOPED, bound to the \TeX{} source) and by a Grok Bot support agent (PASS)""")

edit('appA_records.tex', [(
    r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~9 (15 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{4db2c57a} of the branch \path{claude/side24-v3-supplement-draft9-20261015} in""",
    r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~""" + dno + r""" (""" + date + r"""): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{""" + d10c[:8] + r"""} of the branch \path{""" + d10b + r"""} in""")])

edit('s0_status.tex', [
    (r"""It was assembled on 13 October 2026 and revised through r8 on 15 October 2026.""",
     r"""It was assembled on 13 October 2026 and revised through r9 on """ + date + "."),
    (r"""The second route in Supplement~\S S5: a provider-distinct read (OpenAI) found a gap in its finite-scale witness densities, which Supplement draft~9 repairs (\S S5.8); the repair has one nonauthor read so far, by a Grok Bot support agent (PASS), while an OpenAI re-read awaits publication of its verdict and an xAI read is open & \rec{parent}, \rec{supp} \\""",
     reads_tex + r""" & \rec{parent}, \rec{supp} \\"""),
])

edit('s7_disclosure.tex', [(
    r"""a provider-distinct read by OpenAI found a gap in that route's finite-scale witness densities, Supplement draft~9 repairs it, and the repair has one nonauthor read so far, by a Grok Bot support agent (PASS), while an OpenAI re-read awaits publication of its verdict and an xAI read is open)""",
    r"""a provider-distinct read by OpenAI found a gap in that route's finite-scale witness densities, Supplement draft~9 repairs it, and the repair has nonauthor reads by OpenAI (PASS\_SCOPED, bound to the \TeX{} source) and by a Grok Bot support agent (PASS); the other slices of that section have provider-distinct reads by xAI (PASS\_SCOPED))""")])

# the c_1 reading uses Theorem CU (Supplement Theorem S9.2), which the Supplement proves; "conditional" was the record's word
edit('appA_records.tex', [(
    r"""the enclosure is of the ratio of the integral \eqref{eq:c1} to $c_{3,24}$, and its reading as a coefficient ratio is conditional, as Theorem~\ref{thm:three-term}(iii) states;""",
    r"""the enclosure is of the ratio of the integral \eqref{eq:c1} to $c_{3,24}$, and its reading as a coefficient ratio uses Theorem~CU, as Theorem~\ref{thm:three-term}(iii) states;""")])

# audit S10S11-4: the order of the periodization change of the coefficients
edit('s5_discussion.tex', [(
    r"""at such sides the periodization changes the coefficients by terms of order $L^6e^{-L^2/2}$, as in \eqref{eq:correction})""",
    r"""at such sides the periodization changes the coefficients by terms of order at most $L^8e^{-L^2/2}$, as in \eqref{eq:correction} and Supplement Corollary S11.10)""")])

if s1111:
    # audit S10S11-1 and -2: c_2[K_ref] notation, the side-24 interval, and the unconditional identification
    edit('s2_results.tex', [(
        r"""Second, for the Gaussian kernel in $d=3$,
\[
c_2\in[0.1612340491269447,\ 0.1612340491269810],
\]
in closed form with a certified enclosure. The transfer of this coefficient expression to the torus of side $L\ge24$ changes it by less than $10^{-60}$ (Supplement Corollary S11.10); its identification with the constant of \eqref{eq:three-term} uses the two interfaces of the transfer record \rec{consts} (Supplement Hypothesis S11.11). Third, with the enclosure of $c_{3,24}$ in \rec{v1coef}, under the same interfaces, at $L=24$
\[
c_2/c_{3,24}\in[3.859496174547456,\ 3.859496174548331].
\]""",
        r"""Second, the fixed-cone coefficient expression $c_2[K_{\rm ref}]$ of the Gaussian kernel $K_{\rm ref}(z)=e^{-|z|^2/2}$ on $\R^3$ (Supplement Definition S11.2; \rec{consts}) has a closed form with a certified enclosure,
\[
c_2[K_{\rm ref}]\in[0.1612340491269447,\ 0.1612340491269810]
\]
(Supplement Theorem S11.5). The same expression for the side-24 covariance, $c_2[K_{24}]$, differs from it by less than $10^{-60}$ (Supplement Corollary S11.10), and $c_2=c_2[K_{24}]$: Supplement Corollary S11.11 derives from the fold expansion of Supplement~\S S10 the two identifications that \rec{consts} states as interfaces. So $c_2\in[0.1612340491269446,\ 0.1612340491269811]$. Third, with the enclosure of $c_{3,24}$ in \rec{v1coef},
\[
c_2/c_{3,24}\in[3.859496174547456,\ 3.859496174548331]
\]
(Supplement Corollary S11.12).""")])
    # the status row of Theorem 2.3
    edit('s0_status.tex', [(
        r"""the torus transfer of $c_2$ has an OpenAI \texttt{ACCEPT\_SCOPED} of slices A--C and an Anthropic acceptance of slices A and C, and no Grok Bot read & \rec{chain}, \rec{consts} \\""",
        r"""the torus transfer of $c_2$ has an OpenAI \texttt{ACCEPT\_SCOPED} of slices A--C and an Anthropic acceptance of slices A and C, and no Grok Bot read; the identification $c_2=c_2[K_{24}]$ is Supplement Corollary S11.11 (draft~""" + dno + r"""), with """ + rd_tex + r""" & \rec{chain}, \rec{consts} \\""")])

edit('README.md', [
    ("""# SIDE24 manuscript — V3 draft (13 October 2026; revision r8 of 15 October)""",
     """# SIDE24 manuscript — V3 draft (13 October 2026; revision r9 of """ + date.replace(' 2026', '') + """)"""),
    ("""- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r8, 22 pages.""",
     """- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r9, PAGES_R9 pages."""),
    ("""Earlier revisions are kept at their commits: r7 (22 pages, SHA-256 `af944708…`) at `5543bd20`,""",
     """Earlier revisions are kept at their commits: r8 (22 pages, SHA-256 `9e051654…`) at `2f176488`, r7 (22 pages, SHA-256 `af944708…`) at `5543bd20`,"""),
    ("""- **Requested:** a read of r8.""",
     """- **Requested:** a read of r9."""),
    ("""- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r7, and each AMEND's readback. The companion supplement, which r8 cites as [R1] at `4db2c57a` (draft 9, branch `claude/side24-v3-supplement-draft9-20261015`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved (its "Checkpoint draft 9"); the repair has one nonauthor read so far, by a Grok Bot support agent (PASS, main#389 6182910843), while an OpenAI re-read awaits publication of its verdict and an xAI read is open.""",
     """- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r8, and each AMEND's readback. **Provider reads of the main text (xAI):** slice M3 (§§4–7 and Appendix A) of r8, PASS_SCOPED (main#389 6191908445); the pickups of slices M1 and M2 were never delivered, and their reservations were ended by owner authority on 19 October (main#389 6278216270), so those slices are open for a fresh pickup at r9. The companion supplement, which r9 cites as [R1] at `""" + d10c[:8] + """` (draft """ + dno + """, branch `""" + d10b + """`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved; the repair has nonauthor reads by OpenAI (PASS_SCOPED, bound to the TeX source, main#389 6193704852) and by a Grok Bot support agent (PASS, 6182910843), and the other slices of §S5 have xAI reads (PASS_SCOPED). Draft 10 makes the repairs of a same-provider audit and proves the identification of c₂ (Corollary S11.11), with a Cursor read (AMEND applied, readback PASS, main#389 6276344629)""" + ("; draft 11 proves the cap-region bound of Proposition S8.3 in Lemma S8.16" if dno == '11' else "") + """."""),
])
if dno == '11' or dno == '12':
    edit('appA_records.tex', [(
        r"""Theorem Z: \path{frontiers/c7_zero_gap_limit_20260929/PROOF.md} (\texttt{5b6328ea}; PR~155, \texttt{7a1cb09a}).""",
        r"""Theorem Z: \path{frontiers/c7_zero_gap_limit_20260929/PROOF.md} (\texttt{5b6328ea}; PR~155, \texttt{7a1cb09a}). The Supplement proves the bound (K2) of Theorem~K as its Lemma S8.16""" + (r""" and Theorem~Z as its Proposition S8.4""" if dno == '12' else '') + r""" (draft~""" + dno + r""").""")])
print('r9 edits applied')
