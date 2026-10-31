"""Supplement draft 13, documents: status section, README, concordance section.

Usage: python3 make_d13_docs.py SUPPDIR CONFIG.json
CONFIG keys: date, pages, d12 (draft-12 commit, short), reads_md.
"""
import json
import sys

d = sys.argv[1]
cfg = json.load(open(sys.argv[2], encoding='utf-8'))
date, pages, d12, reads_md = cfg['date'], cfg['pages'], cfg['d12'], cfg['reads_md']


def edit(fn, pairs):
    p = d + '/' + fn
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            raise SystemExit('target count %d in %s: %s' % (n, fn, a[:90]))
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)


edit('supp_s0_status.tex', [
    (r"""Checkpoint draft~12 of 19 October 2026, an author-side assembly""",
     r"""Checkpoint draft~13 of """ + date + r""", an author-side assembly"""),
    (r"""draft~11 proves the cap-region bound that drafts~1--10 imported, and draft~12 proves the equal-height statements that drafts~1--11 imported (below).""",
     r"""draft~11 proves the cap-region bound that drafts~1--10 imported, draft~12 proves the equal-height statements that drafts~1--11 imported, and draft~13 adds a proof of Proposition~\ref{prop:parent-push}, makes two item-level dependencies explicit, and ships a proof-dependency check (below)."""),
    (r"""Its sections ``Checkpoint draft~2'' to ``Checkpoint draft~12'' give the disposition of each issue.""",
     r"""Its sections ``Checkpoint draft~2'' to ``Checkpoint draft~13'' give the disposition of each issue."""),
    (r"""\item \emph{Part~III, added in draft~5.}""",
     r"""\item \emph{Draft~13} is documentary. Proposition~\ref{prop:parent-push}, which drafts~1--12 stated without a proof, gets its two-line proof from Proposition~\ref{prop:parent-ledger}(b) and the change of variables $\ell=kr^3$. The proofs of Propositions~\ref{prop:parent-KR} and~\ref{prop:parent-ledger} and of Lemma~\ref{lem:selection-singchart} now say which parts of each other they use, so that the dependency is visibly acyclic item by item. The check script \path{checks/dep_graph_check.py} builds the proof-dependency graph of the 194 numbered statements from the sources (395 references from proofs and statements to other statements), finds seven statement-level strongly connected components, each explained in the script and in the concordance as a pointer, a definition used from another statement or an item-level acyclic dependency, and exits with an error if a later draft adds one that is not explained. It judges no mathematics.
\item \emph{Part~III, added in draft~5.}"""),
    (r"""and draft~12 adds only the unnumbered \S\ref{ssec:remainder-Z}.""",
     r"""draft~12 adds only the unnumbered \S\ref{ssec:remainder-Z}, and draft~13 changes no number (a proof carries none)."""),
])

edit('README.md', [
    ("# SIDE24 V3 supplement: checkpoint draft 12 (19 October 2026)",
     "# SIDE24 V3 supplement: checkpoint draft 13 (" + date + ")"),
    ("- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, 161 pages;",
     "- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, " + pages + " pages;"),
    ("draft 10 (`5aa2fe29`) 158, draft 11 (`c1d92e94`) 160.",
     "draft 10 (`5aa2fe29`) 158, draft 11 (`c1d92e94`) 160, draft 12 (`" + d12 + "`) 161."),
    ("  - in \"Checkpoint draft 12\", the proof of Proposition S8.4(i)–(ii) (the equal-height mass and the fixed-separation coupling), which drafts 1–11 imported from Theorem Z of the same record.",
     "  - in \"Checkpoint draft 12\", the proof of Proposition S8.4(i)–(ii) (the equal-height mass and the fixed-separation coupling), which drafts 1–11 imported from Theorem Z of the same record;\n"
     "  - in \"Checkpoint draft 13\", a proof of Proposition S12.24, two item-level dependency sentences, and the proof-dependency check."),
    ("- `writer_originals/`:",
     "- `checks/dep_graph_check.py` and its output (new in draft 13): the proof-dependency graph of the 194 numbered statements, built from the TeX sources (numbering recomputed from the sources and cross-checked against the build's `.aux`); seven statement-level strongly connected components, each explained in the script; exit 1 only if an unexplained one appears. Run from this directory: `python3 -I checks/dep_graph_check.py .`\n"
     "- `writer_originals/`:"),
])
readme = open(d + '/README.md', encoding='utf-8').read()
para = ("\n\nDraft 13 (" + date + ") is documentary. Proposition S12.24 (the lifetime pushforward), which drafts 1–12 stated without a proof, gets its two-line proof; "
        "the proofs of Propositions S12.22–S12.23 and of Lemma S5.20 say which parts of each other they use (S12.23 uses S12.22(a),(b) and S12.22(c) uses S12.23; "
        "Lemma S5.40 uses only the chart, Jacobian, residual limits and covariances of S5.20), so the dependency is visibly acyclic item by item; and "
        "`checks/dep_graph_check.py` ships the proof-dependency graph with its seven explained statement-level components. No number of draft 12 changes.\n")
readme = readme.rstrip('\n') + para
open(d + '/README.md', 'w', encoding='utf-8').write(readme)

conc = open(d + '/CONCORDANCE.md', encoding='utf-8').read()
sec = ("## Checkpoint draft 13 (" + date + "): the proof-dependency graph, and three documentary repairs\n\n"
       "Author-side, by Claude01N, on top of draft 12 (`" + d12 + "`). No number of draft 12 changes; one proof environment is added (a proof carries no number).\n\n"
       "- **Method.** `checks/dep_graph_check.py` parses the TeX sources, attaches every proof to its statement (the next `proof` environment; joint proofs "
       "`Proof of Theorems X, Y and Corollary Z`; the subsection `Proof of Theorem S9.2`), maps every `\\ref`/`\\eqref` in a statement or its proof to the statement that "
       "owns the target, and computes the strongly connected components of the graph \"A uses B\". Numbering is recomputed from the sources (one shared counter per "
       "section) and agrees with the build's `.aux` for all 257 statements. 194 numbered statements, 190 proofs, 395 edges, 101 forward uses.\n"
       "- **Result.** Seven statement-level components, none a circularity:\n"
       "  - S12.22 ↔ S12.23: S12.23 uses S12.22(a),(b); S12.22(c) (the exhaustion to the diagonal) uses S12.23. **Repair (documentary):** both proofs now say so.\n"
       "  - S12.3, S12.4, S12.5, S12.19, S12.24, S12.28: closed by two pointers (S12.24 names the density of Theorem B; the proof of S12.19(b) remarks that the proof of Theorem C uses it).\n"
       "  - S5.26, S5.29, S5.30: S5.30 states margins for the inequalities of S5.26–S5.29 and is proved from them; the closing sentence of the proof of S5.26 points forward to S5.30 for the perturbed field.\n"
       "  - S5.16, S5.18, S5.20, S5.21, S5.40, S5.41: the finite-scale display (S5.56) of Lemma S5.20 is Lemma S5.40(d),(e), whose proof uses only the chart, Jacobian, residual limits and covariances of S5.16–S5.21. **Repair (documentary):** the proof of S5.20 now says so.\n"
       "  - S3.19 ↔ S3.20: a pointer in the proof of S3.19 to the comparison made in S3.20.\n"
       "  - S4.24 ↔ S4.4: the statement of S4.24 uses the polynomial `P_d` defined in a display of the statement of S4.4.\n"
       "  - S8.4, S8.6, S8.9, S8.14: the item-level order stated in §S8.6 of draft 12 (S8.4(ii),(iii), `B < ∞` → S8.9 → the limit in S8.4(i) → the elder paragraph of S8.6); S8.14 reuses a decomposition display of the proof of S8.6.\n"
       "- **Statements without a proof of their own.** Lemmas S7.1–S7.3 restate Part III lemmas named in their headings. **Proposition S12.24** had none in drafts 1–12 (the record [R2] states it without one). **Repair:** a proof from Proposition S12.23(b) and `ℓ = k r³` at fixed `k`: `r dr/dℓ = (1/3) k^{-2/3} ℓ^{-1/3}`, so the pushforward of the intensity `r 𝒜_r dr db dk dσ` is (S12.40); `r < r_*` because `ℓ < k_- r_*³ ≤ k r_*³`.\n"
       "- **Reads.** " + reads_md + "\n"
       "- **Build.** " + pages + " pages, no LaTeX error, no undefined reference. All 948 labels of draft 12 keep their numbers; no label is added. The one overfull line, 29 pt in S10, is unchanged.\n")
anchor = "## Checkpoint draft 12 ("
assert conc.count(anchor) == 1
conc = conc.replace(anchor, sec + "\n" + anchor)
open(d + '/CONCORDANCE.md', 'w', encoding='utf-8').write(conc)
print('d13 docs edits applied')
