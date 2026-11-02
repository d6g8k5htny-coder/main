"""Main-text revision r10: supplement draft 12 as [R1]; Theorem Z of [R7] proved in the Supplement; the three
provider reads of 20 October (M1, M2 at r9; D10-C1111 at draft 12) in the status rows and the README.
Documentary only: no theorem, display, constant or proof step changes.

Usage: python3 make_r10.py DIR DATE D12_COMMIT D12_BRANCH
  DATE  e.g. '20 October 2026'
"""
import sys

d, date, d12c, d12b = sys.argv[1:5]
d12c = d12c[:8]


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


edit('appA_records.tex', [
    (r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~11 (19 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{c1d92e94} of the branch \path{claude/side24-v3-supplement-draft11-20261019} in""",
     r"""\emph{Supplement.} SIDE24 V3 supplement, checkpoint draft~12 (19 October 2026): \path{docs/manuscript/side24_v3_supplement_draft_20261013/} at commit \texttt{""" + d12c + r"""} of the branch \path{""" + d12b + r"""} in"""),
    (r"""The Supplement proves the bound (K2) of Theorem~K as its Lemma S8.16 (draft~11).""",
     r"""The Supplement proves the bound (K2) of Theorem~K as its Lemma S8.16 (draft~11) and Theorem~Z as its Proposition S8.4(i)--(ii), in its \S S8.6 (draft~12); with these, Part~II of the Supplement imports no analytic statement from a repository record, and its only computer-assisted inputs are the certified enclosures of $c_1$ and $c_2$ (Supplement \S\S S9, S11)."""),
])

edit('s0_status.tex', [
    (r"""It was assembled on 13 October 2026 and revised through r9 on 19 October 2026.""",
     r"""It was assembled on 13 October 2026 and revised through r10 on """ + date + r"""."""),
    (r"""the identification $c_2=c_2[K_{24}]$ is Supplement Corollary S11.11 (draft~11), with a Grok Bot read of its proof in draft~10 (PASS after one AMEND) & \rec{chain}, \rec{consts} \\""",
     r"""the identification $c_2=c_2[K_{24}]$ is Supplement Corollary S11.11, with a Grok Bot read of its proof in draft~10 (PASS after one AMEND) and an OpenAI read of its proof at draft~12 (\texttt{PASS\_SCOPED}, 20 October 2026) & \rec{chain}, \rec{consts} \\"""),
    (r"""Rem.~\ref{rem:cand} & Theorem Z R$\cdot$2P; Theorems U, K, P R$\cdot$1P; the $c_3\ell^{2/3}$ term and $\delta=1/50$ READ;""",
     r"""Rem.~\ref{rem:cand} & Theorem Z R$\cdot$2P, and proved in the Supplement (\S S8.6, draft~12); Theorems U, K, P R$\cdot$1P, with the bound (K2) of Theorem~K proved in the Supplement (Lemma S8.16, draft~11); the $c_3\ell^{2/3}$ term and $\delta=1/50$ READ;"""),
])

edit('README.md', [
    ("""# SIDE24 manuscript — V3 draft (13 October 2026; revision r9 of 19 October)""",
     """# SIDE24 manuscript — V3 draft (13 October 2026; revision r10 of """ + date.replace(' 2026', '') + """)"""),
    ("""- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r9, 23 pages.""",
     """- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r10, PAGES_R10 pages."""),
    ("""Earlier revisions are kept at their commits: r8 (22 pages, SHA-256 `9e051654…`) at `2f176488`,""",
     """Earlier revisions are kept at their commits: r9 (23 pages, SHA-256 `2bea32be…`) at `3022daae`, r8 (22 pages, SHA-256 `9e051654…`) at `2f176488`,"""),
    ("""- **Requested:** a read of r9.""",
     """- **Requested:** a Cursor read of the r9→r10 delta (documentary: Appendix A records [R1] and [R7], two §0 status rows, this README, the change list)."""),
    ("""- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r8, and each AMEND's readback. **Provider reads of the main text (xAI):** slice M3 (§§4–7 and Appendix A) of r8, PASS_SCOPED (main#389 6191908445); the pickups of slices M1 and M2 were never delivered, and their reservations were ended by owner authority on 19 October (main#389 6278216270), so those slices are open for a fresh pickup at r9. The companion supplement, which r9 cites as [R1] at `c1d92e94` (draft 11, branch `claude/side24-v3-supplement-draft11-20261019`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved; the repair has nonauthor reads by OpenAI (PASS_SCOPED, bound to the TeX source, main#389 6193704852) and by a Grok Bot support agent (PASS, 6182910843), and the other slices of §S5 have xAI reads (PASS_SCOPED). Draft 10 makes the repairs of a same-provider audit and proves the identification of c₂ (Corollary S11.11), with a Cursor read (AMEND applied, readback PASS, main#389 6276344629); draft 11 proves the cap-region bound of Proposition S8.3 in Lemma S8.16.""",
     """- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r9, and each AMEND's readback. **Provider reads of the main text, all three slices:** M3 (§§4–7 and Appendix A) of r8 by xAI, PASS_SCOPED (main#389 6191908445), with the r8→r9 delta covered by the Cursor read of r9; M1 (§§1–2) and M2 (§3) of r9 at `3022daae` by OpenAI Codex on 20 October, both PASS_SCOPED (main#389 6289211527, 6289166739). The 15 October pickups of M1 and M2 were never delivered, and their reservations were ended by owner authority on 19 October (main#389 6278216270). The companion supplement, which r10 cites as [R1] at `""" + d12c + """` (draft 12, branch `""" + d12b + """`), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved; the repair has nonauthor reads by OpenAI (PASS_SCOPED, bound to the TeX source, main#389 6193704852) and by a Grok Bot support agent (PASS, 6182910843), and the other slices of §S5 have xAI reads (PASS_SCOPED). Draft 10 makes the repairs of a same-provider audit and proves the identification of c₂ (Corollary S11.11), with a Cursor read (AMEND applied, readback PASS, main#389 6276344629) and an OpenAI read of the proof at draft 12, whose §S11 is byte-identical to drafts 10 and 11 (PASS_SCOPED, 6289117999); draft 11 proves the cap-region bound of Proposition S8.3 in Lemma S8.16 (Cursor read, readback PASS, 6278180423); draft 12 proves Proposition S8.4(i)–(ii), Theorem Z of [R7], in §S8.6 (Cursor read, readback PASS, 6278603420). With drafts 11 and 12, Part II of the supplement imports no analytic statement from a repository record."""),
])
print('r10 edits applied')
