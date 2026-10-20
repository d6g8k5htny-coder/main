# SIDE24 V3 supplement: checkpoint draft 13 (20 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 draft (main#389). This is an **author-side checkpoint** of the supplementary material for the V3 manuscript draft. It is pushed so that the work is durable. No pull request is open for it. Every section has had a bounded nonauthor read, in slices, in the thread of the main text's pull request, main#389, which records each verdict; "Checkpoint draft 8" in `CONCORDANCE.md` tabulates them. A second provider-distinct read of §S5.4 (OpenAI, on draft 8) found that its finite-scale witness densities near the pair axis were not proved. Draft 9 repairs them; the repair has a nonauthor read by a Grok Bot support agent (PASS, main#389 6182910843) and a provider-distinct read by OpenAI (PASS_SCOPED, bound to the TeX source, main#389 6193704852); the other slices of §S5 have provider-distinct reads by xAI (PASS_SCOPED, 6184275398 and 6188554663). Draft 10 makes the repairs of a same-provider audit of S2–S4 and S6–S13, which found no mathematical error; it also proves the identification of c₂ (Corollary S11.11) that drafts 1–9 stated as a hypothesis. Scientific effect NONE: no register, status or proof file changes.

## Contents

- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, 162 pages; its size and SHA-256 are in `SHA256SUMS`. Its first section, "Status of this draft", is for the author. Earlier drafts: draft 1 (`4f9e6180`) 123 pages, draft 2 (`238ca036`) 126, draft 3 (`e0d08f0c`) 127, draft 4 (`c6aa790d`, amended in `5cd17651`) 129, draft 5 (`ffd749e5`, amended in `a336aaf9`) 149, draft 6 (`a0889f36`) 150, draft 7 (`9d8fc85b`, amended in `a06e0107` and `3eb1d594`) 150, draft 8 (`a39aeadd`) 150, draft 9 (`4db2c57a`) 156, draft 10 (`5aa2fe29`) 158, draft 11 (`c1d92e94`) 160, draft 12 (`59d20df5`) 161.
  - Part I (S1–S7): the setting, the contact measure, the finite-side coefficient, the side-24 correction with |eps_24| < 10^-216, elder selection, the limit, and the bounded remainder.
  - Part II (S8–S11): the remainder's vanishing and rate, the cusp term, the third-order terms, and the constant c_2.
  - Part III (S12–S13), new in draft 5: the parent theorem (Theorems A, B and C of the companion record, in every dimension d ≥ 2 and for every side L > 0) and the deterministic marked-cylinder cap on which its elder selection rests.
- `supp_main.tex`, `supp_s0_status.tex` … `supp_s13_cap.tex`, `refs.bib`, `supp_refs_s8s9.bib`, `supp_refs_s12.bib`: the LaTeX sources. `build.sh` rebuilds the PDF.
- `CONCORDANCE.md`: the source of every statement and display, the corrections applied, the added steps and the issues found. Its head summarizes:
  - the author correction made at integration (the elder mark);
  - the findings that bear on the main text;
  - in "Checkpoint draft 2", the disposition of every issue of S1–S6, and the repairs that make S5 a complete second proof of the selection bound (3.13);
  - in "Checkpoint draft 3", the far elder rate (Proposition S8.13) and two pointer and citation fixes in S10–S11;
  - in "Checkpoint draft 4", a correction to the singular-near chart of S5 (a random `η²` term in the residual of `V_0`), the proof of the kernel transfer in S5 by exact factorizations, and the jet bounds (3.29) of the main text in Lemma S4.29;
  - in "Checkpoint draft 5", Part III, the re-pointing of every citation of the companion record to its counterpart in S12–S13, the changes made to S12 at integration, and the slice SF AMEND;
  - in "Checkpoint draft 6", the steps of S12 that the record states without derivation and that are now written out;
  - in "Checkpoint draft 7", the reads of drafts 5 and 6, the derivation of the Hermite identity in S13, and the landing of the c₁ erratum;
  - in "Checkpoint draft 8", the last two AMENDs and a table of every slice read;
  - in "Checkpoint draft 9", the findings of the OpenAI read of §S5.4 (O2-F1 to F3) and their repair;
  - in "Checkpoint draft 10", the dispositions of a same-provider audit against the main text, the proof of Corollary S11.11, and three shipped check scripts;
  - in "Checkpoint draft 11", the proof of the cap-region bound (Lemma S8.16), which drafts 1–10 imported from Theorem K of record [R7] of the main text;
  - in "Checkpoint draft 12", the proof of Proposition S8.4(i)–(ii) (the equal-height mass and the fixed-separation coupling), which drafts 1–11 imported from Theorem Z of the same record;
  - in "Checkpoint draft 13", a proof of Proposition S12.24, two item-level dependency sentences, and the proof-dependency check.
- `checks/s5_envelope_check.py` and its output `checks/s5_envelope_check.out.txt`: a SymPy check of the cancellations behind the singular-near envelope of Lemma S5.22.
- `checks/s5_factor_check.py` and its output `checks/s5_factor_check.out.txt`: an exact SymPy check of the singular-near residual limits, the planar covariances of Lemmas S5.16–S5.21, the factorizations behind Lemmas S5.18 and S5.21 for random anisotropic even kernels, and the numbers of Lemma S4.29.
- `checks/s12_parent_check.py` and its output `checks/s12_parent_check.out.txt`: 59 exact SymPy checks of the finite algebra of S12–S13 (determinants, congruences, the cap constants, the embedding radius). They do not check any analytic estimate.
- `checks/s12_derivations_check.py` and its output `checks/s12_derivations_check.out.txt`: 22 exact SymPy checks of the identities written out in draft 6.
- `checks/s5_finite_scale_check.py` and its output `checks/s5_finite_scale_check.out.txt`: 60 checks of draft 9. Exact SymPy checks of the blow-up expansions of §S5.8 for a polynomial field under the exact pins, of the corrected value of `V_0` and of the degree statement of Lemma S5.15; and a 60-digit numerical comparison, for the planar kernel, of the finite-scale determinant with its blow-up limit.
- `checks/s1s2_verify_check.py` and its output: 56 exact checks of the displays of S1 and S2 (the concordance's `chk_s1s2/verify_s1s2.py`, shipped in draft 10).
- `checks/s6_limit_check.py` and its output: 19 checks of S6, including the separation-independence of the canonical density version (new in draft 10).
- `checks/s10_s11_fold_check.py` and its output: 54 exact SymPy checks, for pinned polynomial fields in `d = 2, 3`, of the fold identities (S10.16) and (S10.41) of Lemma S10.12 and of the identity of its `r²` coefficient with `X₂` of Theorem S11.7 (S11.23) (new in draft 10).
- `checks/s8_capK_check.py` and its output: 11 checks of Lemma S8.16 (new in draft 11): the envelope (S8.39) on exact rational samples in `d = 2, 3, 4`, with equality cases, the layer integral and the bound of each of its terms, and the arithmetic of the far branch.
- `checks/dep_graph_check.py` and its output (new in draft 13): the proof-dependency graph of the 194 numbered statements, built from the TeX sources (numbering recomputed from the sources and cross-checked against the build's `.aux`); seven statement-level strongly connected components, each explained in the script; exit 1 only if an unexplained one appears. Run from this directory: `python3 -I checks/dep_graph_check.py .`
- `writer_originals/`: the S5 writer's faithful transcription, before the elder-mark correction, and the S12 writer's file, before the changes listed in "Checkpoint draft 5".
- `SHA256SUMS`: hashes of every file above except this README.

## How it was made

Nine fresh subagents of the author's session each wrote one or two sections from the exact technical appendix (30 July 2026), the gap-fill supplement (1 August 2026) and the cited Math- records at `origin/main` `9844f785`. All wrote under the same rules: transcribe faithfully, apply only the listed corrections, and report anything else as an issue. The author integrated the result; the build has 0 LaTeX errors and no undefined references. The supplement has not been read as a whole, and same-provider work earns no independence.

Draft 2 is the author's repair pass over the concordance issues of S1–S6; the concordance records each disposition. In S5:
- the trichotomy uses same-maximum self-attachment, with a new merge-level lemma;
- the regions are measured from the midpoint;
- the fixed-distance count holds on compact marks;
- the singular-near envelope is proved by Taylor's formula for every angle, which closes the `c = 0` case;
- the capture-escape lemmas carry the five local repairs.

The main text (main#389) cites Theorem A of the companion paper for (3.13); S5 is a second proof.

Draft 3 adds the proof of the far elder rate `O(ℓ^{2/3})` (Proposition S8.13, transcribed from Math- record [R9] in Part II's notation). Part II previously cited that record for the rate. It also fixes two pointers and adds a DLMF citation.

Draft 4 makes four repairs, none of which changes a conclusion.
- It corrects the singular-near residual of `V_0`, which has a random `η²` term that the sources omitted. The planar covariances now depend on `η`, and the axis suppression used is `exp(−c/ρ²)`.
- It proves the positivity of `A_col` and `A_sing` for the side-24 field by exact factorizations and the derivative nondegeneracy of S1. This replaces a derivative bound `10^40` that S4 does not contain.
- It adds the main text's jet bounds (3.29) to Lemma S4.29.
- It displays the level terms of Lemma S5.28 and identifies the near kernels of S6 and Part II.

Draft 5 adds Part III. A tenth subagent transcribed the companion record (Math- `imports/lifetime_parent_20260925/`, blobs `dfed3b8d` and `0633aca3`), read as reconciled in `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md` (blob `75da2597`): with the congruence erratum, with the record's §9 replaced by the repaired Kac–Rice section, and with the explicit embedding radius. Every statement that Parts I and II quote from the record now names its counterpart in S12 or S13. At integration the author added the genericity of the unpinned field (Lemma S12.19(b)), which the record's proof of Theorem C uses without proof, and made two wordings precise. No statement or display number of S1–S11 changes.

Draft 6 writes out four steps that the record states without derivation: the determinant of the contact transformation (Lemma S12.10), the averaging kernels of the contact rows (Lemma S12.11(a)), the endpoint bounds (Lemma S12.14) and the measurability of the elder mark (Lemma S12.20(a)). In S8 it makes the absolute-Jacobian convention explicit in two sentences and adds a completing-square proof of the sign-window bound. No number changes.

Draft 7 derives the Hermite identity of Lemma S13.3 and records the reads of drafts 5 and 6: slices SB, SG1 and SG2 passed, and the SF AMEND was applied and read back. No number changes.

Draft 8 is documentary. It records the last two AMENDs, from slices SD and SE: the integral of the kernel in the proof of Lemma S13.3, and an exponent in Remark S11.6. Both were applied exactly and read back. It also tabulates every slice read: each section of the supplement has now had one nonauthor read.

Draft 9 repairs §S5.4 after a provider-distinct read by an OpenAI agent (main#389 6181596945, BLOCK on draft 8, accepted). Draft 8 asserted in (S5.56) a bound on the singular-near density that is false at a fixed positive scale, because the density is positive on the pair axis; the collar count made the same passage from contact limits to finite scale. The new subsection §S5.8 proves the finite-scale densities near the axis by a radial blow-up of the parameters, with explicit limit functionals and a Gaussian-conditioning lemma, and Lemmas S5.19, S5.20, S5.22 and S5.23 now use it. `V_0` now subtracts `(f(M) + f(S))/2`, which the pins fix, in place of `f(x)`, and a degree statement in Lemma S5.15 is corrected. No number of draft 8 changes.

Draft 10 (19 October 2026) makes the repairs of a same-provider audit of S2–S4 and S6–S13 against the main text. The audit found no mathematical error; "Checkpoint draft 10" in `CONCORDANCE.md` lists each point. Hypothesis S11.11 of drafts 1–9 is now Corollary S11.11, proved from the fold expansion of S10, so Corollary S11.12, which gives c₂ at side 24, is unconditional. The audit of S8–S9 found that the bound of Theorem K of record [R7] of the main text, which Proposition S8.3 imports, is the one input of the c₁ term that the supplement did not prove; draft 11 proves it (below). The draft ships three check scripts. No number of draft 9 changes.

Draft 11 (19 October 2026) proves the second bound of Proposition S8.3, E_Q[(W_r/r²)1_{G_r^c}] ≤ C(r³/k)P^N for r ≤ k, in a new last subsection S8.5 (Lemma S8.16). Drafts 1–10 imported it from Theorem K of record [R7] of the main text, and the c₁ term rests on it. No number of draft 10 changes; the new subsection, lemma and displays come after every existing number of S8.

Draft 12 (19 October 2026) proves items (i) and (ii) of Proposition S8.4, the equal-height mass 0 < B_{d,L} < ∞ with the limit ρ_rej(ℓ) → B_{d,L}, and the fixed-separation coupling, in a new unnumbered last subsection S8.6. Drafts 1–11 imported them from Theorem Z of record [R7] of the main text. The limit is derived from (S8.16) and (S8.20), whose proofs do not use it. With Lemma S8.16, Part II now imports no analytic statement from a repository record; what remains computer-assisted are the enclosures of c₁ and c₂, whose certificates S9 and S11 describe. No number of draft 11 changes.

Draft 13 (20 October 2026) is documentary. Proposition S12.24 (the lifetime pushforward), which drafts 1–12 stated without a proof, gets its two-line proof; the proofs of Propositions S12.22–S12.23 and of Lemma S5.20 say which parts of each other they use (S12.23 uses S12.22(a),(b) and S12.22(c) uses S12.23; Lemma S5.40 uses from S5.20 only the window, the chart with its Jacobian, the residual limits and, through S5.21, the covariances), so the dependency is visibly acyclic item by item; and `checks/dep_graph_check.py` ships the proof-dependency graph with its seven explained statement-level components. No number of draft 12 changes.
