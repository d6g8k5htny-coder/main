# Statement and source crosswalk for the planar exposition

This is the audit companion to [MANUSCRIPT.md](MANUSCRIPT.md), not a scientific-status register. The primary source cut is Math- `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1`; the principal identities are in [SOURCES.json](SOURCES.json). All sixteen Math- entries in that manifest were read from the pinned Git commit and checked against their Git blob, byte count and SHA256 during this assembly. The main-repository STATUS entry is outside that sixteen-file check. No source body, review verdict, register or theorem status was changed.

## Required reading rule

| Symbol | Exact source at the cut | Required use |
|---|---|---|
| P | [Parent](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md), blob `dfed3b8d318a3ab1950957f393307733a4bef3f2` | Read with every amendment below; historical candidate header remains intact. |
| CAP | [Deterministic cap](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md), blob `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1` | Consume §§1–5's deterministic implication. The separate fixed-axis numerical probability corollary and its §6 imports are not used. |
| E1 | [Congruence erratum](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md), blob `213594d6ca6a86fb938110f4d166d9ce275a02d0` | P §5 uses `diag(r^(−1/2),I) H diag(r^(−1/2),I)`. D2's inverse of `diag(√r,I)` is the same operation. |
| E2 | [Borel replacement v1.1](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_section9_borel_repair_20260925/REPAIR.md), blob `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a` | Replaces P §9, including the whole-field regular conditional kernel, finite measures and generating-algebra closure. |
| REC | [Reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md), blob `75da2597971510f843f8d90c743950cb8c177342` | §5 W1 uses Slutsky/joint convergence in law and the explicit embedded-cylinder bound `r<L/(4√2)`; §7 defines the consumption contract. |
| R | [D2 remainder](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md), blob `247b3ecf80bfbe896948d5d489b2d5842a81c481` | Theorem R: existential unrestricted bounded remainder. |
| S24 | [SIDE24 coefficient](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/coefficients/side24_v1/PROOF.md), blob `44b66f04f89fcd87383b3603fa69f1feb64cdddd` | Evaluates exactly P (15.2), with ordinary sphere area and ordered maximum/saddle roles. |

REC §7 explicitly reconciles D2's consumed parent interfaces: the §8 good-cap/global-selector comparison, E2, §10, §14 and (15.2). D2 does not require a globally uniform form of P Theorem A. The same contract reconciles S24's interpretation as the coefficient of P Theorem C. S24's own arithmetic review does not independently certify that interpretation.

## Paragraph-level crosswalk

| Draft paragraphs | Source sections and statements | Preserved hypotheses and review scope | Assembly disposition |
|---|---|---|---|
| M1 | P §§1–3, positive Fourier expansion | Exact variance-one torus; fixed `d=2,L=24`; all Fourier modes; no arbitrary rotational invariance | Direct specialization. |
| M2–M3 | P §1 Theorem C; §8; §14 final paragraph | Finite ordinary superlevel H₀; negative-eigenvalue index; global essential class excluded; expected measure per area | Measure definition makes the source convention explicit. No per-realization normalization. |
| M4 | P Theorem C; R Theorem R (R1); S24 scope display; REC §§2,7 | Unrestricted marks/separations, density versions, existential constants | Composition of existing statements, not a new acceptance. |
| M5 | R §8 (R19)–(R20); P §12 | `q>−2/3`; per-area counts; no subtraction of divergent individual moments | Cumulative and moment consequences use the unrestricted coefficient. |
| M6 | P §1 and §§3–5 | Six observations; continuous Gaussian regression; full typed `Z_r`; actual global selector | Exact planar notation. |
| M7 | CAP §§1–5; P §8; REC §5 | Partial-block derivative norms; strict depth bound; fourth-derivative bound; embedded chart; Morse distinct values | Local cap controls all global exit paths. CAP's historical numerical corollary is not imported. |
| M8 | P §§3–7, especially (5.4)–(5.5), (6.2), (7.5)–(7.8); E1; REC W1 | Compact `B,K`, `inf K>0`, all orientations; scalar far branch retained; independent full-field residual, not independent eigenvalues | Outline only; full moment/integration proof remains in P. |
| M9 | E2 §2 replacement §§9.1–9.4; P §8 | Finite separated pair domains, whole `C²` field, Borel elder mark, height disintegration | Uses the repaired measure argument, not continuity of the elder indicator. |
| M10 | P §§3,10–11; REC §6 | Ordinary `dσ`, no half-factor; compact intervals have positive length; small radius; `ℓ<k_-r_*³` | `c_(B,K)` is explicitly separated from `c₂,₂₄`. |
| M11 | P §§13–14 | Unnormalized polynomial-Gaussian majorant; fixed radius band independent of marks; exact lower `k` cutoff; fixed far region | Dominated convergence supplies leading term only. |
| M12 | R §§2–7, (R4)–(R18) | Quadratic filtered determinant estimate; target-growth bounds; `r/k≤1` for cap loss; retained `a` and `η` | Source-derived remainder outline; no evaluated constants. |
| M13 | P §15 (15.1)–(15.2) | Odd/even independence from parity; actual periodized jet law; unnormalized angular measure | Same coefficient as R and S24. The prefactor's 24 is not a side-length variable. |
| M14 | S24 §§1–5 | Reference expression only; full image bound through order six; all frames; Schur-complement and cone comparison; outward arithmetic | Numerical enclosure's scope remains separate from persistence interpretation. |
| M15 | M2 definition; R (R19); experiment protocol | Independent realization replicates; exact area; integrated bins; essential exclusion | Experimental guidance and algebra, not a theorem about discretization. The separate executed pilot supports only the reported grid sensitivity and inconclusive comparison; its RUN/observations are linked from M15. |
| M16 | Planar elder proof §§1–2,8–10; unrestricted difference Theorem U §§1,3–5 | Refined planar fixed/compact parameter scope; rejected candidates differ from witness counts and actual replacement bars | Context only. Neither refined proof is required to establish M4. |

The abstract summarizes M1–M5 and M13–M15 and adds no larger theorem scope.

## Review depth and exact supplemental review identities

REC records nonauthor model review of each parent interface, with its actual provider and exposure. It records its own reviewer/reconciler overlap; those roles are not independent votes. Its consumption contract is the basis of the assembled reading. The later records below add depth at the same source cut, so REC's older invitation to review those slices should not be repeated as an unfilled mathematical gap.

| Supplemental record | Pinned identity | What it establishes |
|---|---|---|
| [D1 C/D/E full-depth review](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_cde_full_depth_claude_20260929/REVIEW.md) | Blob `f05700e24e108e9ae5f563e15e33a22a12710f93`; 13279 bytes; SHA256 `8f137401f1f27b76f0202f7d4c271f47899f786cf3bb7064249b46c1c268d09c` | Anthropic full-depth ACCEPT of P §10, §§13–14, §15 at the repaired existential scope; upgrades that provider's earlier partial depth, not a third provider. |
| [D2 full-depth review](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d2_remainder_full_depth_claude_20260930/REVIEW.md) | Blob `529c5264ed790ab1df36c14f567f460e2a5ed974`; full identity in SOURCES.json | R1–R6 ACCEPT at Theorem R's `O(1)` scope; prior xAI review and Anthropic author's overlap with adjacent work disclosed. |
| [SIDE24 arithmetic review](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/side24_v1_coefficient_claude_20260929/REVIEW.md) | Blob `665d1683f33a7180957923c3cb2abce8ade235d1`; 17903 bytes; SHA256 `258c414c46be93142ee552cfa4c32c03dd89b6bf7b0b470334f152e1e0a57eef` | Arithmetic-enclosure ACCEPT of exact P (15.2); documents prior xAI review and independent arithmetic reproduction, not independent parent acceptance. |

The two supplemental identities not listed in SOURCES.json were verified directly from the pinned Git objects. Their scope is stated here rather than silently broadening the manifest. This assembly was performed by OpenAI, the same provider as the mathematical author, with source exposure and the same shared account. It contributes editorial/source-alignment work and zero organizational-independence credit. Finite checks establish only their stated identity, arithmetic or software coverage; analytic arguments remain in the written proofs and review records.

## Specific remaining assembly questions

1. **Complete appendices.** The exposition's cap, weighted failure integral and Borel Kac–Rice passages are outlines. A submission must include complete arguments or stable appendices with the repaired text assembled in reading order. The present source links support inspection but do not make this draft a self-contained proof. No contradictory composition was found in the bounded read of D1/R/S24; that is not a new full-depth review of every proof interface.
2. **Primary-source theorem comparison.** E2 identifies Armentano–Azaïs–León v3 Theorems 2.2 and 7.1 and supplies its own Borel extension. This assembly read E2 and the source reviews; it did not newly inspect the entire external article or conduct a theorem-level novelty comparison with the literature in README. That remains a bibliographic/assumption audit, not a missing blanket elder-pairing premise.
3. **Older manuscript alignment.** Live Math- #187/#188 descriptions refer to a V3 manuscript, a referee map and an older Proposition A.3.2 `O(ℓ)` claim for far elder density, identified as unsupported in finding F-02. Their full manuscript text was not inspected. This draft consumes only the reviewed P (14.1) `O(1)` far bound and must not be represented as repairing those unseen files. If the V3 text is reused, compare its exact version paragraph by paragraph before importing its stronger claims.
4. **Numerical regime.** No value of `C` or `ℓ_*` in M4 is supplied by S24's tiny coefficient interval. An explicit error certificate is needed before calling an experimental finite-lifetime range asymptotic. Spectral truncation, interpolation and persistence discretization require quantified errors for this observable; empirical refinement alone is insufficient.
5. **Literature and human assessment.** A full theorem-level related-work section and an actual human reader's documented assessment remain absent. Neither a provider-distinct model read nor a shared-account PR disposition is human review.

## Bounded live-update check

A read-only recent-PR query on 30 September 2026 found the following relevant open work. It is recorded to avoid duplicate claims, not incorporated into the theorem above. PR bodies can lag their current heads; the head identities below came from live metadata, and no current-body claim was transferred to proof bytes without review.

| PR | Observed head | Relevance and boundary |
|---|---|---|
| [Math #175](https://github.com/d6g8k5htny-coder/Math-/pull/175) | `b15fe5a8edf3d59c84128de7df69982faf7002d3` | Fixed-dimensional actual elder partner through hard negative fibres; separate scope from this planar leading-law assembly. |
| [Math #187](https://github.com/d6g8k5htny-coder/Math-/pull/187) | `ecd47e7b1d447eefec963b1fa8395777db09a190` | Candidate fixed-far elder bound `O(ℓ^(2/3))`; not consumed. |
| [Math #188](https://github.com/d6g8k5htny-coder/Math-/pull/188) | `85f0586a468c2b2f73af0e340d251e33a52caade` | Candidate fixed-far elder bound of every polynomial order; not consumed and does not itself improve D2's total remainder. |
| [Math #178](https://github.com/d6g8k5htny-coder/Math-/pull/178), [#190](https://github.com/d6g8k5htny-coder/Math-/pull/190) | `48407d43d2603b8f04f8ba3824b53fac903ed0be`, `30de5e327a61228479238679896ef4701024e67f` | Existing microscopic radius-tail numerical work and proposed certified enclosures; different estimands from the unrestricted lifetime histogram or S24 coefficient. |

The recent-PR endpoint returned no open main-repository PRs in this bounded query. That does not establish absence of peer work in closed PRs, comments, Drive or private manuscripts. No Drive crawl, peer-branch change or scientific-register mutation was performed for this audit.
