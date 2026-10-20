# SIDE24 V3 supplement, checkpoint draft 1: concordance (13 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 draft. Nine fresh subagents of this session wrote the sections, each under the same written rules (`COMMON.md`): transcribe faithfully, apply only the listed corrections, and report anything else as an issue instead of fixing it. The author then integrated the sections. Credit 0. Scientific effect NONE.

**Sources.**
- The exact technical appendix of 30 July 2026, Modules A–L (TA).
- The gap-fill supplement of 1 August 2026, Parts A–E (GF).
- The Math- records the main text cites, at `origin/main` `9844f785`. The cited folders are unchanged from the main text's pin `b68d88b3`.

Each section's concordance below maps every statement and numbered display to its source. It also lists notation conversions, corrections applied, added steps and issues.

## Author correction made at integration (after the writers' reports)

- **S5, the elder mark.** GF Definition A.1.2 sets `e = 1{m_f(u+,u-) < h and f(M) = min(b_f(u+,h), b_f(u-,h))}`. As written, this mark is identically zero. The straight segment from `u+` through `S` to `u-` has minimum `h`, so `m_f(u+,u-) >= h` always. GF's proof sentence "equality would force a degenerate or shared critical value" is false. The birth level with `1{m >= a}` at `a = h` also merges the two sectors. (Writer W4, issues 1–2.)
  - Definition S5.2 now uses the maximin connection level, `e = 1{d_f(M) = f(S)}`. This is the formalization of the companion paper (its §8) and of the blind-reconstruction packet.
  - Lemma S5.3 proves that on the Morse locus this is exactly the elder-pair condition (i)–(iii), and that the mark is Borel (lower semicontinuity of `g -> d_g(x)`).
  - The same-component event `L_same` is restated without `m_f`.
  - The main text's verbal Definition 3.1 already says this; only the formula changes.
  - The writer's faithful transcription is kept as `supp_s5_selection.w4orig.tex`.

## Findings that bear on the main text (for the next revision; none is applied to the main text here)

1. **The selection chain of main §3.4.** The supplement-level proof that main Lemma 3.4, Corollary 3.5 (3.13) and Theorem 3.6 cite has gaps (W4, issues 4–7 and 10–11):
   - the regions of the trichotomy are measured from different centres;
   - same-component and same-maximum self-attachment are not the same event;
   - the fixed-distance count uses a joint density where a conditional one is needed, so the Gaussian factor in (3.17) is shown only on compact marks;
   - the singular-near envelope (3.16), read with display (B.1), needs the value-window constraint and a separate argument near `c = 0`;
   - there are details in GF Lemmas A.4.2 and A.4.4.

   The bound (3.13) is also Theorem A of the companion paper (`[R2]`), for all `d >= 2` and `L > 0`, accepted at existential scope (Math- `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md`). Suggested fix: the proof of Corollary 3.5 cites `[R2]` Theorem A for (3.13) and keeps the d = 3 witness chain as a second argument; (3.17) is stated on compact marks.
2. **Main §3.8 displays the 10^-180 chain** ((3.26)–(3.29), (3.32)–(3.33)), while (2.10) states 10^-216 (W3, issue 1). Suggested fix: the proof cites Theorem S4.4/Corollary S4.5 (the 10^-216 record) and drops the 10^-180 displays, or labels them as the weaker historical chain.
3. **Theorem 2.3(ii)** calls `c_2` "the fold-scale finite part of the two-point contact kernel". In the records, `c_2` is the finite part of the `r^2` coefficient of the two-point kernel. The certified ratio interval `c_2/c_{3,24} in [3.859496174547456, 3.859496174548331]` is available for (2.14) (W9, issues 1–2).
4. **TA Module K's general-d formula.** It is wrong by a factor `beta^n` when `beta != 1`. The Bargmann–Fock case (`beta = 1`) is unaffected (W2, issue I5).
5. **Notation.** `kappa` is the cubic coefficient in main (3.5) and the cusp variable in §5 (W7, W8, W9).

## Checkpoint draft 2 (14 October 2026)

Author-side, by Claude01N (session `session_01NMeKEismAyeqgdB4sy2NJU`). It applies the repairs listed below to checkpoint draft 1 (`4f9e6180`). The main text's findings 1, 2, 3 and 5 above were applied in main#389 revision r3 (`13c1868d`, with the read AMEND applied in `7aff36d0` and the readback PASS 6176065794). Finding 4 is repaired here, in S3.

**Numbering.** In S5 the new Lemma S5.6 (merge level) moves every later statement up by one. Old S5.6–S5.35 are now S5.7–S5.36, and two remarks are added, S5.37 (the parent theorem) and S5.38 (the Morse event under the Palm law). Equation numbers change from Lemma S5.22 on: new displays are added in Lemmas S5.22 and S5.29, and the display `‖L_r‖² ≥ ½(b² + κ²)` of draft 1 is removed. The tables below keep the draft-1 numbers; the text cites labels where a number could mislead.

### S5, issue by issue (the CONC_s5 list below)

- **1–3 (GF A.1.2: the mark, the birth level, the sector offset).** Already resolved at integration, because the mark is the maximin connection level (Definition S5.2, Lemma S5.3). No sector offset or two-sector formula remains.
- **4 (`L_same`).** Repaired.
  - `L_same,r` is now same-maximum self-attachment (both unstable half-branches converge to `M`), in Definition S5.5(c).
  - The trichotomy proof (Lemma S5.7) has a new Case 0, in which the two sectors lie in one component. If the other half-branch ends at a critical point `z ≠ M`, then either `f(z) < f(M)`, and `z` is a window witness, or the component contains a maximum higher than `M`. In the second case the new Lemma S5.6 (merge level: mountain-pass level, deformation argument) gives a critical point with value in `(h, f(M))`.
  - Case 1 uses the same lemma, in place of the merge-tree sentence.
- **5 (regions).** Repaired: all three regions are measured from `x`, with `C_1 > 1` and `r ≤ δ/C_1`. Region (iii) is `dist(y, x) > δ`, so `dist(y, {M, S}) ≥ δ/2`.
- **6 (fixed-distance count).** Repaired. Lemma S5.24 (old S5.23) states the conditional density of `(f(y), ∇f(y))` given the pins, `p_A(a_r)/p_{P^corr}(L_r)`, bounded on compact marks. Lemma S5.25 (old S5.24) states `E^MS N_far ≤ C r^3` on compact marks, with no Gaussian factor. The `‖L_r‖² ≥ ½(b² + κ²)` claim (issue 16) is no longer used and is removed.
- **7 (singular-near envelope).** Repaired by a new proof, which also closes the `c = 0` case.
  - Lemma S5.22 now proves `eq:selection-endenv` and `eq:selection-thirdenv` ((S5.61)–(S5.62) in draft 2) from Taylor's formula with derivatives to order five and the value window, for every angle, with an explicit factor `ρ^-2` per determinant. The bounds are `|det H_M|, |det H_S| ≤ Cρ^-2 r(r + s²)Ψ` and `|det H_y| ≤ Cρ^-2 (r² + s³)Ψ`.
  - At the endpoint the key identity is `A[w,w] = 6F_M(w) + Θ[w,w,w,w]/12 + O(s⁵)` with `F_M(w) = f(y) − f(M)`.
  - At the third point, the pins give `T[ω,ω,t] = (2/3)sΘ[ω,ω,ω,t] + O(r²/s² + s²)` and the window gives `T[ω,ω,ω] = (1/2)sΘ[ω,ω,ω,ω] + O(r²/s² + s²)`. With these, the in-plane minor is `O(r² + s³)`.
  - The elimination (B.1) is not used there.
  - The conditional moments of `Ψ` grow at most polynomially in `1/ρ`. They are absorbed by the weighted angular bound `eq:selection-singdens` ((S5.56) in draft 2), which now holds for every weight `ρ^-m` because of the `exp(−c_*/ρ⁴)` axis suppression.
  - The symbolic check `checks/s5_envelope_check.py` (SymPy; output `checks/s5_envelope_check.out.txt`, `ALL PASS`) confirms the cancellations for exact polynomials of degree seven. It checks every monomial in `(r, s)` of the in-plane and hard-direction parts at an endpoint and at the third point. The denominators are powers of `ρ` and `s` only, with no factor of `c`, and the solved cubic coefficients are bounded for `r ≤ s`.
  - The collar count (Lemma S5.19, old S5.18) now uses (B.1) only where `|c| ≥ c_0`. For `|c| < c_0` it uses Hessian collision zeros at each vertex of the triangle `M, S, y`: every angle has sine at least `c′ρ`, so each determinant is at most `Cρ^-1 r²Ψ`. The factor `ρ^-3` is absorbed by the `exp(−c/ρ⁴)` suppression near `X = 0`.
- **8 (GF B.1(b) display slips).** As in draft 1. The displays are the transcript forms, which agree with main Remark 3.8.
- **9 (`(2π)^-3`).** Repaired to `(2π)^-3/2` in `eq:selection-coldens`, (S5.33) in draft 2.
- **10 (GF A.4.2, `|rQ_X|`).** Repaired in Lemma S5.27 (old S5.26): `|Q_X| ≤ R(ξ + |Y|) ≤ R(1 + 1/K)ξ`, so `|rQ_X| ≤ 2rRξ ≤ ξ/16`.
- **11 (GF A.4.4).** All four parts are repaired in Lemma S5.29 (old S5.28):
  - (a) The inward half-branch is tangent to `−v`, with `v` the positive eigenvector; its slope is at most `√(8/3)R/τ < m`.
  - (b) `eq:selection-Qtube` ((S5.90) in draft 2): `|∇Q| ≤ R(2 + m)(¼ − X²)` on the tube, from `∇Q(M) = ∇Q(S) = 0`. The lateral flux and `F_X < 0` (`eq:selection-FXtube`, (S5.91) in draft 2) then hold on the whole tube, near `S` included, and the `δ_0` restriction is gone. The ledger `eq:selection-tubeledger`, (S5.93) in draft 2, is unchanged, because `3rR ≤ 3ε_0 < 1/8192`.
  - (c) The capture section is at `X = −½ + 2δ_0`.
  - (d) The Lyapunov remainder is split. Its quadratic part is at most `3R|z − M|²`, and its `rQ` part `r[∇Q(z) − D²Q(M)(z − M)]` is at most `2rR|z − M|`. So `V̇ ≤ −¾V + 6RV^{3/2} + 4rRV ≤ (−3/8 + 4ε_0)V` on `∂B_{ρ_B}(M)`.

  The paragraph after Theorem S5.31 lists the five repairs, matching main §3.4.
- **12, 13 (provenance displays).** Open. The level terms of Lemma S5.28 and an explicit bound for `dA_col`, `dA_sing` are not yet displayed.
- **14 (Morse event under the Palm law).** Repaired by Remark S5.38. For almost every `(b, κ)` it follows by disintegration. For every `(b, κ)` it follows from nondegenerate conditional laws of `H_M`, `H_S` and a Bulinskaya-type lemma away from `{M, S}`, for degenerate critical points and for equal critical values.
- **15 (joint continuity of GF's functional).** Moot: that functional is not used.
- **16 (`‖L_r‖²` normalization).** Moot: see 6.
- **17 (collision strata).** Repaired in Lemma S5.15 (old S5.14): the denominators vanish only at `c = 0`, `ρ = 0` (and `η = 0` for `det H_y`). On `2c = ±η` the `κ²` coefficient vanishes.
- **18 (labels).** Unchanged.

With these, S5 gives a complete second proof of (3.13) in `d = 3` (Theorem S5.36). Remark S5.37 records Theorem A of `[R2]`, which the main text cites.

### Other sections

- **S1–S2.**
  - I2 is repaired in Convention S1.6: a pointwise asymptotic is a property of the Kac–Rice representative, and the convention refers to the asymptotics of Theorem 2.1.
  - I3, I5, I6 and I7 were already handled in draft 1; their fixes are suggestions for the sources.
  - I8 is a main-text item: cite Lemma S2.9(i) for `Z_r > 0` in main Definition 3.1. It waits for a later main#389 head.
  - I1, I9 and I10 are not errors.
- **S3.**
  - I5 is repaired. `D_n` is the moment of `M_n = β^{-1/2}Q_n`; without the normalization the formulas are off by `β^n` when `β ≠ 1`. Bargmann–Fock has `β = 1`. This is main finding 4.
  - I2 and I3 were already handled in draft 1.
  - I1 (the scope `L = 24`), I4, I6, I7, I8 and I9 are unchanged and recorded below.
- **S4.** Issues 1 and 2 are main-text items. Item 1 was applied in r3; item 2 waits for the supplement's adoption.
- **S6.** Items 1–5 are covered by the added steps listed there; item 6 is notation and item 7 is for the integrator. No change.
- **S7–S11.** Not worked in draft 2; their issue lists below stand.

**Build.** `build.sh` (TeX Live 2023): 126 pages, no LaTeX error and no undefined reference. There is one overfull line, 29 pt, in S10 at lines 439–440, unchanged from draft 1.

## Checkpoint draft 3 (14 October 2026)

Author-side, by Claude01N, on top of draft 2 (`238ca036`).

- **S8: the far elder rate is now proved in the supplement.** This resolves S10 issue I3, "The far elder rate has no home in the supplement". New Proposition S8.13 (`prop:remainder-farrate`) states `ν_eld^{far,ρ}(ℓ) ≤ C(d, L, ρ) ℓ^{2/3}` for `0 < ℓ ≤ 1` and every `ρ ∈ (0, L/2]`.
  - **Source.** Math- `frontiers/far_elder_rate_20260930/PROOF.md`, blob `07260114`, Theorem F and §§2–3; merged in PR 187, record [R9] of the main text. Its erratum `ERRATUM_20261004.md` (blob `93afac3b`) is provenance only. The second-provider read (Grok Bot agent 6, main#229 5975813301) found the mathematics PASS.
  - **Notation.** The transcription uses Part II's notation, Convention S8.1(f). The barrier step is the existing Lemma S8.10 (`lem:remainder-barrier`), which is the record's Lemma 1.
  - **Proof structure.** As in the record:
    - uniform eigenvalue bounds for `V_y = (O_y, H_0, H_y)` on `D_ρ`;
    - the joint density bound, regression and conditional `K`-moments;
    - the pathwise barrier bound, Markov's inequality and the spectral integration over `H_0`;
    - the split at `u = (Aℓ)^{1/3}` with `q = 2`.
  - **Not transcribed.** The record's §0 context (the 30 July manuscript and the audit finding), as its erratum asks.
  - **What uses it.** Lemma S8.12, the crude `ℓ^{1/3}` bound, stays. Its closing remark now points to S8.13, and S10's `eq:third-farrate` cites S8.13 instead of "record [R9] of the main text".
- **S11 issue 7.** Hypothesis S11.11(a) now points to the precise place of the typed-kernel/fixed-cone comparison: Step 2 of the proof of Lemma S10.12 (`lem:third-fold`), display `eq:third-F3`.
- **S11 issue 8.** A DLMF entry is added to `refs.bib`. It is cited for Euler's integral and `Γ(11/6) = (5/6)Γ(5/6)` (§§5.2, 5.5), and for the Stirling remainder bound in Step 8 (§5.11(ii)).
- **Status section.** Updated to draft 3.

**Build.** 127 pages, no LaTeX error and no undefined reference. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 4 (14 October 2026)

Author-side, by Claude01N, on top of draft 3 (`e0d08f0c`). The statement and equation numbers of S5 are unchanged; in particular (S5.56), (S5.64)–(S5.69) and (S5.90)–(S5.91) keep their numbers.

- **S5, Lemma S5.20 (`lem:selection-singchart`): a correction.** At fixed `η = r/s`, the contact limit of `V_0` has the random term `−(η²/16)(Y f_112 + Z f_113)`. Drafts 1–3, following the sources, said that the `η`-dependent corrections were multiples of the pinned `f_111`. That is true for `V_1` (the term `−κη²/4`) but not for `V_0`. The pins force `f_k(x) = −(r²/8) f_{k11}(x) + O(r⁴)` for `k = 2, 3`, and `V_0` multiplies `f_2(x)`, `f_3(x)` by `Y/(2s²)`, `Z/(2s²)`.
  - `eq:selection-L0` now has this term. `eq:selection-C01`, `eq:selection-det01` (with `Q_η = (3c² − ¾η²)² + 12c²ρ² + 3ρ⁴`) and `eq:selection-det4` are the exact planar values for every `η`. At `η = 0` they reduce to the old displays, and `eq:selection-det4sphere` is now stated at `η = 0`.
  - The axis suppression `eq:selection-pcorr` is now `exp(−c_*/ρ²)`. It comes from the `V_1` mismatch alone: the penalty is at least the squared `V_1` mismatch over `C_11 ≤ 2ρ²`. The old `exp(−c_*/ρ⁴)` came from the near-null direction of the `η = 0` covariance. With the correct `V_0` residual and the window term, that direction's mismatch can vanish for some `η < 1`, so the stronger factor is not claimed.
  - **Effect.** None on the conclusions. `exp(−c_*/ρ²)` still beats every power of `ρ`, so `eq:selection-singdens` and the count are unchanged, and `Q_η > 0` keeps the `ρ^10` lower bound used by the envelope's conditional moments.
- **S5, Lemmas S5.18 and S5.21 (`lem:selection-collarfactor`, `lem:selection-singfactor`): the kernel transfer is proved.** Drafts 1–3 obtained `A_col ≥ 1 − 10^-70` and `A_sing ≥ 1/24 − 10^-70` on the jet ball from a derivative bound `‖dA‖ ≤ 10^40`, which Remark S5.12 attributed to S4. S4 contains no such bound (issue 13 in section (d) "Issues" of the S5 concordance). Draft 4 replaces this argument.
  - **Collar.** In the frame `(t, v, w)` the contact limit of the scaled gradient is `(½(X² − ¼) f_ttt + ρ(X U_1 + ρ U_2), ρ f_vv, ρ f_vw)` for every kernel, with `U_1 = f_ttv`, `U_2 = ½ f_tvv`. So `det C_col = ρ⁶ D` exactly, with `D = det Cov(X U_1 + ρ U_2, f_vv, f_vw | P)` a quadratic form in `(X, ρ)`.
  - **Singular-near.** `(L_0, L_1) = (−(ρ/12) A_1 (U_1, U_2, W), ρ A_2 (U_1, U_2, W))`, where `A_1`, `A_2` are the two rows of `A`. Every 2×2 minor of `A` is divisible by `ρ` (Cauchy–Binet), and odd and even blocks are conditionally independent for an even kernel. So `det C_sing = ρ^10 Q_η(𝒦) det Cov(f_vv, f_vw | P)/144` exactly.
  - **Positivity.** All symbols involved are distinct monomials, so by Theorem S1.3 (`thm:setting-gram`; for `K_0`, the positive spectral density) the conditional covariances are positive definite. With `Q_η ≥ λ_min(Σ)²|m̃|²` and compactness, this gives `A_col ≥ a_col(𝒦) > 0` and `A_sing ≥ a_sing(𝒦) > 0`. The constants are existential and depend on the kernel; the main text (3.14)–(3.15) asserts only positivity.
  - **Means.** For an even kernel, odd random parts regress only on the odd pins, and the only odd pin with a nonzero target is `f_ttt`. So the mean changes are `O(ρκ)`, and the suppression statements persist for small `ρ`. This replaces "the perturbation of its regression coefficient is below 10^-70".
  - **Other changes.** The jet ball now enters only through Lemma S5.14 (`lem:selection-pinstab`). The statements "uniform for every kernel in the jet ball" in S5.19, S5.22 and S5.23 now read "for every kernel satisfying Hypothesis S5.11, with constants depending on the kernel".
- **S5, Remark S5.12 (`rem:selection-side24`).** The jet bound `‖J_6(K_24) − J_0‖_max < 10^-113` is now cited to Lemma S4.29 (see the next item) instead of "Section S4". The `10^40` envelope is removed.
- **S4, Lemma S4.29 (`lem:correction-deep`): the bounds (3.29) of the main text.** S4 proved `|J_24 − J_0| < E ≈ 2.12·10^-112` (Lemma S4.16, `lem:correction-jet`), which is weaker than the `10^-113` that the main text (3.29) and Remark S5.12 state. The lemma now also states `|(J^(1)_24)_k| ≤ 6B_0q`, `|(J_24 − J_0)_k| < 6B_0q + 9289B_0q² < 10^-113` and `|(R_jet)_k| < 10^-230`. Both follow from Lemma S4.27 (`lem:correction-normalization`) and the bound on `γ_1` already shown in the proof. No statement is added, so S4's numbering is unchanged.
- **S5, Lemma S5.28 (`lem:selection-strip`): item 12.** The adverse level terms at `X = 5/4` are now displayed with their bounds: `21/(32K)`, `1/K²`, `3ε_0`, and the exact-field transfer `≤ η_R ≤ 1/8192` (ledger `3/8192`).
- **S6, after Proposition S6.1: S7 issue I2.** An unnumbered paragraph identifies the kernels: `G^all_r(t, b, κ) = 𝒜_r(b, κ/6, t)` and `G^sel_r = p_r 𝒜_r = 𝒜^eld_r`, through `p_Π(z_r) = 12 r^-6 π_r(v_r)` and `Z_r = E_Q[W_r]`. The near-density version and the near-pair formulas of S7–S10 are then visibly the same function.
- **New check.** `checks/s5_factor_check.py` (SymPy, exact) tests:
  - the contact limits of `V_0` and `V_1` from a degree-5 polynomial field under the exact finite-`r` pins;
  - all planar displays of S5.16, S5.18, S5.20 and S5.21;
  - the factorizations for two random anisotropic even kernels (discrete spectral measures), with positivity at sample points;
  - the numbers of Lemma S4.29.

  Its output `checks/s5_factor_check.out.txt` ends `ALL PASS`.
- **Status section.** Updated to draft 4. The open item "two provenance displays" is closed: the level terms are written, and the `dA` bound is no longer needed.

**Build.** 129 pages, no LaTeX error and no undefined reference. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 5 (14 October 2026)

Author-side, by Claude01N, on top of draft 4 as amended (`5cd17651`). The statement and display numbers of S1–S11 are unchanged: all 794 labels of draft 4 keep their numbers, and Part III adds 133 labels.

- **Part III: the parent theorem (S12) and the marked-cylinder cap (S13).** A fresh subagent of the author's session transcribed the companion record P and its cap import CAP, read as reconciled in REC. The reconciled object is:
  - P: Math- `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`, blob `dfed3b8d`;
  - CAP: `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md`, blob `0633aca3`;
  - E1, the congruence erratum: `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md`, blob `213594d6`;
  - E2, which replaces P §9: `reviews/d1_section9_borel_repair_20260925/REPAIR.md`, blob `fe9b9ce4`;
  - REC, with the wording W1 and the embedding radius `r < L/(4√2)`: `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md`, blob `75da2597`.

  The writer's concordance is the section "CONC_s12" at the end of this file. The writer's S12 file is kept unchanged as `writer_originals/supp_s12_parent.rev2orig.tex` (SHA-256 `10f33167…`). `supp_s13_cap.tex` and `supp_refs_s12.bib` are the writer's files, byte for byte.
- **Citations re-pointed (CONC_s12, section (e)).** The 50 citations of P in S0 and S5–S10 were mapped.
  - 49 now name their counterpart in S12 or S13 next to the citation.
  - The 50th, in Remark S5.37 ("a second, three-dimensional proof that does not use [parentP]"), is a statement about S5 and is unchanged.
  - Lemmas S7.1–S7.3, Hypotheses S7.4–S7.7 and Definition S7.8 name their counterparts in their headings. The paragraph before them now says that S12 states and proves them.
- **Changes to S12 at integration.** Each is a wording change or an addition; no formula of P changes.
  1. The opening paragraph states REC's reading rule. It cites a new bibliography entry, `parentREC` in `refs.bib`, which names REC, E1, E2 and CAP with their blobs.
  2. **Issue I-3 is resolved by an addition.** Lemma S12.19 (`lem:parent-generic`) has a new part (b): the unpinned field is almost surely Morse with pairwise distinct critical values. Its proof is the overdetermined-zero argument of part (a) without pins, applied to `(∇f(x), D²f(x)v)` on `T^d_L × S^{d−1}` (dimensions `2d − 1 → 2d`) and to `(∇f(x), ∇f(y), f(x) − f(y))` on `{dist(x, y) ≥ δ}` (dimensions `2d → 2d + 1`), with covariance positivity from Lemma S12.8. P's proof of Theorem C uses this fact without proof. For `d = 3`, `L = 24` it is Proposition 1.1 of the main text. The proof of Theorem S12.5 now cites part (b); the proof of Theorem S12.3, S7's Hypothesis S7.4 and S8's cap paragraph cite part (a).
  3. **Issue I-5 is resolved.** "Every `b`" now reads "every `b ∈ R`" in Lemma S12.27(c), twice, and in the proof of Proposition S12.28.
  4. **Issue I-8 is resolved.** Lemma S12.8 now reads "at finitely many distinct sites and with several functionals allowed at each site", which is how every use reads it.
  5. The header comment points to this file.
- **Issue I-14 is resolved by citation.** Convention S8.1(b) and §S10.1 cite Proposition S12.22(b), (S12.40) and (S12.51) for the density versions in general `d`, `L`. The word "canonical" there refers to Convention 2.2 of the main text and its construction.
- **Issue I-11** (Theorem C does not say "admit versions"). S12 is unchanged; the citations of (S12.40) and (S12.51) name the versions used.
- **Open S12/S13 issues.** I-6, I-7, I-9, I-10, I-12 and I-13 remain, all low: steps that the record states without derivation or only sketches, transcribed as there, and record-internal notation. None is fixed here.
- **S7 layout.** The tuple in Lemma S7.2(a) is now displayed, because the longer heading caused a 30.7 pt overfull line. The content is unchanged.
- **Status section, abstract, README.** Updated to draft 5. Two open items of the status section are closed by main#389 r4: citing Lemma S2.9(i) for `Z_r > 0`, and using this supplement's numbering.
- **New check.** `checks/s12_parent_check.py` is the writer's `verify_s12.py`; only the "Run:" comment line differs. Its output `checks/s12_parent_check.out.txt` is byte-identical to the writer's `verify_s12.out.txt` and ends `ALL PASS (59/59)`.

**Build.** 149 pages, no LaTeX error and no undefined reference. The one overfull line, 29 pt in S10, is unchanged.

**Slice SF AMEND (main#389 6180073112), applied exactly in `a336aaf9`.** Hypothesis S7.6's heading adds Lemma S12.27(c), which is where `0 < Z_r < ∞` for every `b ∈ ℝ` and `k > 0` is stated; §S10.1's pin rows cite Definition S12.9 and Lemma S12.10 (the Jacobian and the target); and the status sentence on re-pointing covers the citations that use a statement of the record, with Remark S5.37's non-use sentence unchanged. Slice SB (§§S3–S4) is PASS (main#389 6180217523).

## Checkpoint draft 6 (14 October 2026)

Author-side, by Claude01N, on top of draft 5 with the slice SF AMEND applied (`a336aaf9`). No statement or numbered display is added, removed or renumbered: every display added below is unnumbered.

- **S12: steps that the record states without derivation are written out (issues I-6 and I-7 of CONC_s12).**
  - **Lemma S12.10.** The proof displays the axial block of `T_r`. In the order `(f(M), f(S), f_x(M), f_x(S))` it is block lower triangular with diagonal determinants `1/r` and `−12/r³`; each transverse pair has determinant `1/r`; so `|det T_r| = 12 r^{−4} r^{−m}`. It evaluates the four axial rows at the pins.
  - **Lemma S12.11(a).** The proof gives the averaging kernels of the rows. In particular `U^(3) = ∫_{−r/2}^{r/2} (6/r³)(r²/4 − s²) φ'''(s) ds`, an average of the third axial derivative against a probability density on the pin interval.
  - **Lemma S12.14.** The proof derives the bounds `eq:parent-51` and `eq:parent-52` (the record's (5.1)–(5.2)). From the gradient pins, `f_xx(a) = −(1/r)∫_a^c (c − x) f_xxx` and `f_xx(c) = (1/r)∫_a^c (x − a) f_xxx`, and `f_xx` and `∇_y f_x` have mean zero on the pin interval. The Hermite identity makes `12k` the `(x − a)(c − x)`-weighted mean of `f_xxx`, so `|f_xxx − 12k| ≤ rM_4` on the interval. `A_S − A_M = ∫ ∂_x D_y² f` gives `‖A_S − A_M‖ ≤ rM_3`.
  - **Lemma S12.20(a).** A complete measurability argument is added: the lower-semicontinuity proof of Lemma S5.3, which uses neither `d = 3` nor `L = 24`. The record's polygonal-path sketch is kept.
  - The opening paragraph of §S12 names these places.
- **S8 (optional points from two support notes by OpenAI RLOOP, both found the checked algebra correct).**
  - From the normalization note (main#389 6180051224): before `eq:remainder-lead` the substitution `k = ℓ/r³` is stated with `|dk/dr| = 3ℓ r^{−4}` and the order of the endpoints reversed, and before `eq:remainder-far` the height map has "absolute Jacobian one".
  - From the sign-window note (main#389 6180226921): after `eq:remainder-window`, one sentence gives the completing-square identity `det K_M det K_S = (w + (R_M + R_S)/2)² − (6kΔ + (R_S − R_M)/2)²`. On the support of `W_r` it bounds `W_r/r²` by `(6k|Δ| + (|R_M| + |R_S|)/2)²`, which gives the same order as the root-window argument with a smaller constant.
  - No numbered formula changes.
- **New check.** `checks/s12_derivations_check.py` (SymPy, exact) verifies every identity added to S12 above: the axial and transverse determinants for `d = 2, …, 6`, the target, the averaging kernels for a generic degree-7 polynomial, the endpoint identities and the Hermite identity under the gradient pins, and the cubic case `α_M = −6k`, `α_S = 6k`. It also checks the completing-square identity of S8. Its output `checks/s12_derivations_check.out.txt` ends `ALL PASS (22/22)`.

**Build.** 150 pages, no LaTeX error and no undefined reference; all 927 labels keep their numbers. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 7 (14 October 2026)

Author-side, by Claude01N, on top of draft 6 (`a0889f36`). No statement or numbered display is added, removed or renumbered.

- **Reads of drafts 5 and 6** (Cursor, Grok; provider-distinct, same account; recorded on main#389):
  - slice SB (§§S3–S4): PASS, 6180217523;
  - the SF readback (`ffd749e5..a336aaf9`): PASS, 6180283717;
  - slice SG1 (§§S12.1–S12.8 and §S13 against the reconciled record, with draft 6): PASS, 6180392775;
  - slice SG2 (§§S12.9–S12.16 against the record and the §9 repair): PASS, 6180432992.

  Neither SG read found a gap beyond the issues CONC_s12 (d) lists. SG1 found the four steps written out in draft 6 correct, and also the three S8 changes.
- **S13, Lemma S13.3: the Hermite identity is derived** (issue I-7, noted again by SG1). With `a = −r/2`, `c = r/2` and `ψ = f_x(·, 0)`, the critical pins give `ψ(a) = ψ(c) = 0`. The kernel `K = (x − a)(c − x)/2` vanishes at both ends and has `K'' = −1`, so two integrations by parts give `∫ψ = −∫Kψ''`, which is `eq:cap-hermite`. The identity was already checked by `checks/s12_derivations_check.py`. The opening paragraph of §S12 and the status section list this place with the other written-out steps.
- **The c₁ erratum has landed.** The S8/S9 concordance below (source row [454] and item 9 of its corrections) records Math- PR 454 as open. It landed on 13 October at 21:47Z (merge `a6116d8a`; `ERRATUM_20261013.md`, blob `a9c3f95a`). Remark S9.17 already states the reading that the erratum requires, so the LaTeX is unchanged. The writer's text below is left as written.
- **Open issues of CONC_s12** after drafts 5–7: I-7 in part (the remaining unproved steps are listed there), and I-9 to I-13. All are low severity, and the SG reads confirmed the list.

**Build.** 150 pages, no LaTeX error and no undefined reference; all 927 labels keep their numbers. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 8 (14 October 2026): the reading is complete

Author-side, by Claude01N, on top of `3eb1d594`. Documentary only: no statement, display or number changes.

**Two AMENDs from the last slices, each applied exactly and read back.**
- Slice SD (§§S8–S9 with the draft-7 delta), main#389 6180560525. The kernel defined in draft 7's proof of Lemma S13.3, `K = (x − a)(c − x)/2`, has integral `r³/12`, not `r³/6`; with the height drop `r³/6` the weighted average of `f_xxx` is 2. Applied in `a06e0107`.
- Slice SE (§§S10–S11), main#389 6180651247. In Remark S11.6 the first `d = 2` error now reads `1.9·10⁻⁹`, as in the source note. Applied in `3eb1d594`.
- Readback of `9d8fc85b..3eb1d594`: PASS, 6180761684.

Otherwise SD found §§S8–S9 matching their sources, including every printed enclosure, and Remark S9.17 matching the landed c₁ erratum. SE found §§S10–S11 matching theirs, including the closed forms and intervals of `c₂`.

**Reading record.** Every section of the supplement has had a bounded nonauthor read. All reads were by Cursor cloud agents (Grok): provider-distinct from the author, but posting from the same account, so no organizational independence is claimed. Comment numbers are on main#389.

| Slice | Sections | Commit read | Verdict | AMEND applied, readback |
|---|---|---|---|---|
| S5 read | §S5, Lemma S4.29, Proposition S8.13 | `c6aa790d` | PASS, 6178514413 | — |
| SA | §§S1–S2 | `c6aa790d` | PASS, 6178898935 | — |
| SC | §§S6–S7 | `c6aa790d` | AMEND, 6179041392 | `5cd17651`; PASS 6179473089 |
| SB | §§S3–S4 | `ffd749e5` | PASS, 6180217523 | — |
| SF | the draft-5 delta (Part III citations, Lemma S12.19(b), wording) | `ffd749e5` | AMEND, 6180073112 | `a336aaf9`; PASS 6180283717 |
| SG1 | §§S12.1–S12.8, §S13, draft-6 steps | `a0889f36` | PASS, 6180392775 | — |
| SG2 | §§S12.9–S12.16 | `a0889f36` | PASS, 6180432992 | — |
| SD | §§S8–S9, draft-7 delta | `9d8fc85b` | AMEND, 6180560525 | `a06e0107`; PASS 6180761684 |
| SE | §§S10–S11 | `9d8fc85b` | AMEND, 6180651247 | `3eb1d594`; PASS 6180761684 |

Two OpenAI RLOOP support notes checked the S8 opening normalizations and the sign-window algebra: 6180051224 and 6180226921, both correct. The draft has not been read as a whole by anyone, and it has had no human read.

**Build.** 150 pages, no LaTeX error and no undefined reference. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 9 (15 October 2026): the finite-scale witness densities

Author-side, by Claude01N, on top of draft 8 (`a39aeadd`). No statement or numbered display of draft 8 is renumbered: all 927 labels keep their numbers, and 10 labels are added at the end of S5 (Lemmas S5.39–S5.41, displays (S5.107)–(S5.112), subsection S5.8).

**The O2 read.** A provider-distinct nonauthor read of §S5.4 by an OpenAI agent (main#389: pickup 6181542001, verdict BLOCK 6181596945, on `a39aeadd`) found three things. The author accepted all three (6181660569), and main#389 r7 (`570f9c6f`, with its read AMEND applied in `5543bd20`) qualified the main text.
- **O2-F1.** (S5.56) bounded `∫_{S²} ρ^{−m} p_{(f(y),∇f(y))|pins}(u,0) dω` by `C_m s^{−7}`. At a fixed positive scale this is false for `m ≥ 2`: the density is positive at points of the pair axis other than `M` and `S`, so the integral diverges. The proof had transferred the contact-limit suppression `exp(−c_*/ρ²)` to fixed scale. Lemma S5.23 used the false bound, and the collar count of Lemma S5.19 made the same transfer.
- **O2-F2.** `V_0` subtracted `f(x)`, which the pins do not fix, so `(f(y), ∇f(y)) → V` was not an affine change of the conditioned variables.
- **O2-F3.** Lemma S5.15 said that the product of the three determinants has "exact degree six in κ". At `c = ½`, `ρ = √3/2`, `η = 1`, `q_M = q_S = 0` all three vanish.

On these points the BLOCK overrides the earlier Cursor read of §S5 at `c6aa790d` (6178514413, PASS), which did not raise them; that read's PASS of the rest of §S5 stands.

**Dispositions.**
- **F1, a new subsection S5.8 and new proofs of Lemmas S5.19 and S5.23.**
  - Lemma S5.39 (`lem:selection-gauss`) collects four Gaussian facts: the one-coordinate penalty; the conditional field `f = f̄ + g`, whose mean derivatives are bounded by the Mahalanobis norm of the conditioning values and whose centred part has moments independent of them; the joint density bound with `exp(−𝓜²/4)`; and a bound for a density at a point at distance `|m̃|` from the mean of one coordinate, which needs no lower bound on that coordinate's variance.
  - Lemma S5.40 (`lem:selection-finsing`) treats the singular-near chart. With `ε² = ρ² + s²` and `(ρ, s) = ε(cos θ, sin θ)`, the variables `Ṽ_0 = ε^{−2}(V_0 + ς(4 + 3η²)V_1/48)` and `Ṽ_i = ε^{−1}V_i` have explicit limits modulo the pins (S5.110), which are jointly nondegenerate for every `θ ∈ [0, π/2]`, `η ∈ [0, η_0]` and both caps. The coefficients of `V` are polynomials in `s`, `ρ`, `c` and `η`, with no negative power of `s` (the proof gives the reason: `s³V_0 = f(y) − U^(0) − ½(y − x)·∇f(y) + (sc/6)∂_t f(y)`, and modulo the pins `f(x)`, `f_t(x)` are `O(r⁴)` and the other contact coordinates `O(r²)`), so the expansion is uniform in `θ`, including `s ≪ ρ`. Hence `det Cov(V | pins) ≥ c ε^{10}`, `Var(V_1 | pins) ≤ Cε²`, and the `V_1` mean is at least `κ/2` near the axis. The finite-scale density, weighted by any power of `1 + 𝓜`, is at most `C s^{−7} ε^{−5} exp(−c_*/ε²)` near the axis and `C s^{−7}` elsewhere.
  - Lemma S5.41 (`lem:selection-fincol`) treats the collar. Away from the endpoints the blow-up is in `(ρ, r)`; near `M` or `S` it is centred at the endpoint, in `(sin α, r)`, for the variable `G̃ = ((r²τ)^{−1}∂_t f(y), (rτ)^{−1}∂_v f(y), (rτ)^{−1}∂_w f(y))`, uniformly in `τ = |y − p|/r ∈ [0, ¼]`; its coefficients have no negative power of `r` or `τ`, so it is jointly continuous up to `(r, τ) = (0, 0)`. Only the transverse pair needs a determinant bound; the axial component is handled by Lemma S5.39(d). This avoids a second blow-up at the midpoint `X = 0`, where the axial leading term vanishes.
  - Lemma S5.19's proof is new. Its deterministic envelopes come from the Hessian collision zero (Lemma S5.17) and the angle bounds of the triangle `MSy`; near an endpoint each Hessian has two soft directions, using `‖H_S ŷ‖ ≤ ‖H_M ŷ‖ + CrΨ`. It no longer uses the elimination (B.1) or "uniformly bounded moments" of a random factor. Three regions, each `O(r⁵)` before the normalizer.
  - Lemma S5.22 adds the crude envelopes `|det H_M| + |det H_S| ≤ CrΨ`, `|det H_y| ≤ CsΨ`, valid on the axis, and bounds the conditional moments by `C(1 + 𝓜(u))^q`. Drafts 1–8 used a `ρ^{10}` lower bound for the third-point block, which holds only in the contact limit.
  - Lemma S5.23's proof is new. Where `ρ ≥ s` the refined envelope and the suppression `exp(−c_*/(2ρ²))` give the six-term integral (S5.71)–(S5.72) as before. In the boundary layer `ρ < s` the crude envelope and `exp(−c_*/(2s²))` give `O(r⁵)`. The fixed-scale part `s ≥ s_0` is `O(r⁵)` directly.
  - (S5.56) now states the pointwise finite-scale bound; (S5.55) is labelled as the contact limit. Lemma S5.21's remark on persistence is labelled likewise.
- **F2.** `V_0` subtracts `U^(0) = ½(f(M) + f(S))`, which equals `b − κr³/12` under the pins. Under the pins `f(x) − U^(0) = (r⁴/384) f_tttt(x) + O(r⁶)`, which moves `V_0` by `O(η³r)`, so the contact limits (S5.47)–(S5.53) are unchanged. `checks/s5_factor_check.py` still subtracts `f(x)`; the text says so.
- **F3.** Lemma S5.15 now says: degree at most six in `κ`, with `κ⁶` coefficient `(4c² − η²)⁶/(64c²ρ²)³`, so exactly six off the collision strata `2c = ±η`; the reviewer's example is quoted.
- The status section, the opening paragraphs of §S5 and §S5.4, and Lemma S5.16 (a pointer) are updated.

**Superseded statements in the sections above.** The claims in "Checkpoint draft 2" (S5 item 7: the moments "are absorbed by the weighted angular bound"; the closing sentence that S5 gives a complete second proof) and in "Checkpoint draft 4" ("`eq:selection-singdens` and the count are unchanged"; "`Q_η > 0` keeps the `ρ^10` lower bound used by the envelope's conditional moments") were wrong at fixed positive scale, for the reason in O2-F1. They are left as written, as history.

**New check.** `checks/s5_finite_scale_check.py` (SymPy and mpmath). For a polynomial field of degree six under the exact finite-`r` pins, it verifies, for both caps and both endpoints: that the random parts vanish at `ε = 0`; the `β` that cancels the order-`ε` terms; the limits (S5.110)–(S5.112) exactly; the pin parts of the means; the corrected value of F2; and the degree statement of F3 with the reviewer's example. It also compares, at four sample angles, the exact finite-scale `det Cov(V | pins)/ε^{10}` for the planar kernel at `ε = 0.004` (60-digit arithmetic, exact kernel derivatives) with the determinant of the limit functionals; the relative errors are below `5·10⁻⁵`. It also checks that the coefficients of `V`, `G` and `G̃` have no negative power of `s`, `r` or `τ`, and the value of `G̃` at `r = 0`. Its output `checks/s5_finite_scale_check.out.txt` ends `ALL PASS` (60 checks).

**Pre-delivery check (same provider, no independence credit).** A fresh subagent of the author's session, which had not seen the repair being written, read §S5 of draft 9 as a hostile referee: PASS WITH FIXES, no blocker. It reproduced the expansions with its own SymPy code for a degree-7 field and the determinant limits with 160-digit planar numerics, at θ = 0.5°, 5°, 45°, 89.5° and 90°, for η = 0.05 and 0.9 and both caps. Its fixes are all applied: the remainder bound and the reason for the absence of negative powers of `s` in the proof of Lemma S5.40(a); the joint continuity of `G̃` at `(r, τ) = (0, 0)`; a stale sentence after Lemma S5.15 (the collar count no longer uses the elimination); the contact-limit qualifier in Lemma S5.16; a gap in the compactness argument of Lemma S5.40(e) (now run on `[min(s_0, s̄(ρ_0)), δ]`); notation clashes (`f̄` for the conditional mean, `c_V` for constants, `Ṽ` and `G̃` for the normalized variables, `γ_1` and `χ`); the explicit choices of `ε_1 ≤ 1/8`, `ρ_0`, `s_0` and `r_0`; and the pointers in the remark after Lemma S5.22.

**Not yet done.** The repair has not been read. Its read is requested on main#389.

**Build.** 156 pages, no LaTeX error and no undefined reference; all 927 labels of draft 8 keep their numbers. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 13 (20 October 2026): the proof-dependency graph, and three documentary repairs

Author-side, by Claude01N, on top of draft 12 (`59d20df5`). No number of draft 12 changes; one proof environment is added (a proof carries no number).

- **Method.** `checks/dep_graph_check.py` parses the TeX sources, attaches every proof to its statement (the next `proof` environment; joint proofs `Proof of Theorems X, Y and Corollary Z`; the subsection `Proof of Theorem S9.2`), maps every `\ref`/`\eqref` in a statement or its proof to the statement that owns the target, and computes the strongly connected components of the graph "A uses B". Numbering is recomputed from the sources (one shared counter per section) and agrees with the build's `.aux` for all 257 statements. 194 numbered statements, 190 proofs, 395 edges, 101 forward uses.
- **Result.** Seven statement-level components, none a circularity:
  - S12.22 ↔ S12.23: S12.23 uses S12.22(a),(b); S12.22(c) (the exhaustion to the diagonal) uses S12.23. **Repair (documentary):** both proofs now say so.
  - S12.3, S12.4, S12.5, S12.19, S12.24, S12.28: closed by two pointers (S12.24 names the density of Theorem B; the proof of S12.19(b) remarks that the proof of Theorem C uses it).
  - S5.26, S5.29, S5.30: S5.30 states margins for the inequalities of S5.26–S5.29 and is proved from them; the closing sentence of the proof of S5.26 points forward to S5.30 for the perturbed field.
  - S5.16, S5.18, S5.20, S5.21, S5.40, S5.41: the finite-scale display (S5.56) of Lemma S5.20 is Lemma S5.40(d),(e), whose proof uses from S5.20 only the window, the chart with its Jacobian, the residual limits and, through S5.21, the covariances, and from S5.16–S5.18 the collar limits. **Repair (documentary):** the proof of S5.20 now says so.
  - S3.19 ↔ S3.20: a pointer in the proof of S3.19 to the comparison made in S3.20.
  - S4.24 ↔ S4.4: the statement of S4.24 uses the polynomial `P_d` defined in a display of the statement of S4.4.
  - S8.4, S8.6, S8.9, S8.14: the item-level order stated in §S8.6 of draft 12 (S8.4(ii),(iii), `B < ∞` → S8.9 → the limit in S8.4(i) → the elder paragraph of S8.6); S8.14 reuses a decomposition display of the proof of S8.6.
- **Statements without a proof of their own.** Lemmas S7.1–S7.3 restate Part III lemmas named in their headings. **Proposition S12.24** had none in drafts 1–12 (the record [R2] states it without one). **Repair:** a proof from Proposition S12.23(b) and `ℓ = k r³` at fixed `k`: `r dr/dℓ = (1/3) k^{-2/3} ℓ^{-1/3}`, so the pushforward of the intensity `r 𝒜_r dr db dk dσ` is (S12.40); `r < r_*` because `ℓ < k_- r_*³ ≤ k r_*³`.
- **Reads.** Draft 12 at `59d20df5` has a Cursor read (AMEND on one majorant constant and two parent hashes [main#389 6278559198], applied; readback PASS [6278603420]) and an OpenAI read of Corollary S11.11 (PASS_SCOPED [6289117999]); five provider-distinct slice reads of draft 12 (SUPP-A to SUPP-E) are offered at main#419 6290107952 and bind the draft-12 bytes. Main text r10 at `d9f01444` cites draft 12 as [R1] (Cursor AMEND [6290059705] applied; readback PASS [6290117154]). A Cursor read of this draft's delta is requested in main#389.
- **Build.** 162 pages (the status section grows by one item), no LaTeX error, no undefined reference. All 948 labels of draft 12 keep their numbers; no label is added. The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 12 (19 October 2026): the equal-height statements are proved

Author-side, by Claude01N, on top of draft 11 (`c1d92e94`). One unnumbered subsection is added at the end of S8; no number of draft 11 changes.

- **S8, new §S8.6: Proposition S8.4(i)–(ii).** Drafts 1–11 cited Theorem Z of record [R7] of the main text for the equal-height mass `0 < B_{d,L} < ∞`, the limit `ρ_rej(ℓ) → B_{d,L}`, and the fixed-separation coupling `f_ℓ = f_0 + ℓh_y`. Now:
  - (ii) is Gaussian regression on the two-site pins `O_y`: `h_y = −C_yΓ_y⁻¹e` is deterministic and smooth with `O_y(h_y) = −e`; the Hessian pair has a nondegenerate Gaussian law by Lemma S12.8 (Schur complement), so `{det H_0 det H_y = 0}` is null; `Ψ^cand_ℓ → Ψ_0` by the local Lipschitz bound of Lemma S7.14 with a polynomial majorant.
  - `B_{d,L} < ∞` from (S8.33) near the diagonal and (ii)–(iii) on the far compact set; `B_{d,L} > 0` because the Hessian pair's density is positive and the typed set is open and nonempty.
  - `ρ_rej → B_{d,L}` from (S8.16) and (S8.20). The proof of (S8.16) uses S8.4 only through (ii), (iii) and the finiteness of `B_{d,L}`; the proof of (S8.20) does not use it (Remark S8.15). No circularity.
  - The proof paragraph of S8.4 points to §S8.6; the elder paragraph of Theorem S8.6's proof says where the limit comes from; the radius `r_Z` of Convention S8.1(h) is set to `r_K`, since the old meaning (the radius of Theorem Z's near-diagonal analysis) is no longer used.
- **Audit.** A same-provider referee (a fresh subagent of the author's session, zero independence credit) read §S8.6 against its inputs, traced the dependency graph of (S8.16) and (S8.20) down to S7 and S12 (no cycle), checked the regression coupling numerically in a `d = 2` torus model, and returned AMEND, wording only: the proof paragraph of S8.4 now says that (S8.16) and Theorem S8.9, not Theorem S8.6 as a whole, are what the limit rests on; the positivity argument names the open set exactly (the typed set less the null set, since an index changes only where an eigenvalue vanishes); the closing sentence names Sections S1–S7; the finite-expectation step through the coupling is written out; and two smaller wordings. All applied.
- **Reads.** Draft 11 at `c1d92e94` has a Cursor read (the proof of Lemma S8.16 complete; AMEND on four documentary lines [main#389 6278149205], applied; readback PASS [6278180423]). Main text r9 at `3022daae` cites draft 11 as [R1] (Cursor AMEND [6278373147] applied; readback PASS [6278394928]). A Cursor read of this draft is requested in main#389.
- **Build.** 161 pages, no LaTeX error, no undefined reference. Every label of draft 11 keeps its number; the one new label is `ssec:remainder-Z` (§S8.6). The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 11 (19 October 2026): the cap-region bound is proved

Author-side, by Claude01N, on top of draft 10 (`5aa2fe29`). One subsection, one lemma and three displays are added at the end of S8. No number of draft 10 changes.

- **S8, new §S8.5, Lemma S8.16 (audit S89-4 of "Checkpoint draft 10").**
  - The second bound of Proposition S8.3, `E_Q[(W_r/r²)1_{G_r^c}] ≤ C(r³/k)P^N` for `r ≤ k`, is Theorem K of record [R7] of the main text. Drafts 1–10 imported it.
  - The c₁ term rests on it through Lemma S8.14, the `I_1` domination in the proof of Theorem S9.2, and (S10.12). Lemma S7.18 alone gives only `(r/k)³`.
  - Lemma S8.16 proves it. The proof is that of Lemma S7.18 with two changes. First, `|α_i| ≤ a = 6k + rM_4/2`, from (S7.14), replaces `M_3/2` in the envelope (S8.39), which keeps the factor `k²`. Second, the density of `A_M` under `Q` is bounded as a Gaussian with a shifted centre, which removes the pin density. For `r ≤ k`, `a ≤ C_*kΥ`, and each term of the depth-failure layer integral is at most `C(r³/k)PΥ⁸`; the case `m = 1` and the fourth-derivative exception are as in Lemma S7.18, with the weight bound (S8.38).
  - The proof of Proposition S8.3 now cites the lemma and fixes `r_K = min{r_R, r_0}`.
- **Check script.** `checks/s8_capK_check.py`, 11 checks:
  - the three endpoint bounds and the envelope (S8.39) on 120 exact rational samples in `d = 2, 3, 4`;
  - equality cases, so that none of the terms of (S8.39) can be dropped;
  - the layer integral, and the quotient of each of its terms by `(r³/k)PΥ⁸`;
  - the arithmetic of the far branch.
- **Audit.** A same-provider referee (a fresh subagent of the author's session, zero independence credit) read the lemma against every statement it cites and against its uses in S8, S9 and S10, and returned AMEND: one required change, the citation of (S8.5), which is stated only for `r ≤ r_R`, replaced by (S7.28) of Lemma S7.12, which holds for `r ≤ r₀`; and four recommended ones (the local meaning of `T` is declared; the envelope is bounded by the `λ₁`-free `ā` before the layer integral; the closing sentence and the remark are made exact). All are applied. The referee also checked the envelope adversarially (Nelder–Mead with 50-digit re-verification, `d = 2, 3, 4`; every maximum ratio is at most 1, so the bounds are tight, and 18 mutants are refuted), the shifted-Gaussian density step, the layer integral and (S7.44) exactly, the absence of circularity, and the three downstream uses; a stratified Monte Carlo in a `d = 2` model shows the `r³/k` scaling, with `X/δ³` falling like `k²`.
- **Reads.** Draft 10 at `5aa2fe29` has a Cursor read (AMEND [main#389 6276296631], applied; readback PASS [6276344629]). A Cursor read of this draft is requested in main#389.
- **Build.** 160 pages, with no LaTeX error and no undefined reference. Every label of draft 10 keeps its number; the five new labels are `ssec:remainder-capK` (§S8.5), `lem:remainder-capK` (Lemma S8.16) and `eq:remainder-capK`, `eq:remainder-capW`, `eq:remainder-capenv` ((S8.37)–(S8.39)). The one overfull line, 29 pt in S10, is unchanged.

## Checkpoint draft 10 (19 October 2026): repairs from a same-provider audit; Corollary S11.11

Author-side, by Claude01N, on top of draft 9 (`4db2c57a`). No numbered display or label is added or removed, and every label of draft 9 keeps its number. Hypothesis S11.11 becomes Corollary S11.11, with the same number and label, proved from S10; see the item on S11.11 below.

**The audit.** Fresh subagents of the author's session read the supplement as hostile referees, against the main text r8, in six scopes: §§S2 and S6 with the parts of S1 they import; §§S3 and S4; §S7; §§S8 and S9; §§S10 and S11; and §§S12 and S13. §S5 has provider-distinct reads instead ("Checkpoint draft 9"). The subagents are the same provider as the author and earn no independence credit. Every scope returned PASS, with no BLOCK and no AMEND. Each subagent recomputed its scope independently, for example:
- (2.9) at `L = 24`, to about 330 digits. This gives `ε₂₄ ≈ −7.5·10⁻²³⁷`, inside the stated bound `|ε₂₄| < 10⁻²¹⁶` by about twenty orders.
- `D_t = 29/6 − √6`, by four routes.
- `P₃(24)` and `B₃`.
- Stress tests of the determinant bounds of Lemmas S7.14, S7.15 and S12.16.
- 117 random fields that satisfy the hypotheses of the cap theorem S13.2.
- `c₂[K_ref]` from its definition by a separate pipeline (the surrogate at finite `r`, exact Gaussian conditioning, Wick moments, extrapolation in `r²`). It agrees with the closed form of Theorem S11.5 to `3.5·10⁻²²` for `d = 3`, and to `9·10⁻²⁴` and `3·10⁻²²` for `d = 1, 2`. Theorem S11.7, implemented literally for `d = 3`, reproduces `(a, p₀, p₂, p₄)` to about `10⁻²⁰`.
- The constants of Theorem S11.9 and Corollary S11.10, the ratio interval of Corollary S11.12(iii), and the exponent bookkeeping of S10 (the splits `ℓ^{2/7}`, `ℓ^{2/9}` and `ℓ^{1/9}`, by an exact max–min).
- `c₁` for `d = 3` by two routes of the audit's own (a direct route with closed-form noncentral moments, and the Mellin route with a separate quadrature): `c₁ = −0.2118483462111935`, inside the certified enclosure with margins `7.9·10⁻¹⁰` and `2.1·10⁻¹⁰`. A Monte Carlo of the unreduced formula (`8·10⁷` samples) gives `−0.21178 ± 0.00019`. Also `c₁` for `d = 1, 2`, `T₃`, every entry of the table of Theorem S9.13, the constants of Lemma S9.15, the algebra of S8–S9 on pinned polynomial fields (122 checks), and the maximin level of Theorem S9.5(c) on a grid (73 random models, no mismatch).
All findings are at NOTE level. The dispositions follow.

- **S2: Definition S2.1 and Theorem S2.12.** The weight `W_r` tests only the index. So the all-typed measures count every pair of a maximum and an index-`(d − 1)` critical point, whether or not the point merges two components; only the elder mark selects merging saddles (Proposition S1.11). Reworded.
- **S2, after Lemma S2.9, and main-text (3.9).** The phrase "uniformly bounded conditional Gaussian moments" holds only on compact mark sets.
  - In general the moments are bounded by a polynomial in `|b| + κ`, uniformly in `t` and in `r` below any fixed `ϱ < 12` (the proof of Proposition S6.4).
  - Reworded. The main text is reworded in r8, as amended by its read (main#389 6187682210).
- **S2, after Theorem S2.12, and the proofs of Propositions S6.1 and S6.2.** The marked two-point Kac–Rice formula with the global Borel mark `e` is Proposition S12.22, which these places did not cite. They now cite it, and say that the conditional expectation is the Gaussian-regression version, defined for every target and jointly measurable.
- **S6 after (S6.5), and main-text Convention 2.2.** The canonical version of `ν_{3,24}` is built with a fixed separation `ϱ`, and its independence of `ϱ` was not stated.
  - The added sentence proves it with the substitution `κ = 6ℓ/r³`. Item 10 of `checks/s6_limit_check.py` checks it.
  - Main text: r8.
- **S6, the proof of Proposition S6.4.** (H1) cited "the selection bound of Section S5". It now cites Theorem S12.3 with Remark S5.37, and Theorem S5.36 as the second route.
- **S6 after Proposition S6.2, and Lemma S12.29.** (14.1) is stated at the small radius `r₀` of §S12. Its proof applies at every separation, and a sentence after Lemma S12.29 now says so. Main text: r8, in the proof of Proposition 3.10.
- **S3, "The limiting contact density".** The limit (S3.27) cited only Lemma S3.3(ii). It now also cites Lemma S2.3(iii), Proposition S2.8 with Lemma S2.9(ii), and (S6.19).
- **S3, Theorem S3.10.** The identity of (S3.28) with `c_{3,L}` holds for every `L ∈ (0, ∞]` without hypothesis. The hypothesis is needed only to call (S3.28) the short-lifetime coefficient. The theorem is restated accordingly.
- **S3, Proposition S3.19.** (S3.58) and (S3.59) were asserted without derivation, and `c_d^iso` was not defined for `d ≠ 3`. The derivation from Lemma S4.20(b) is added, and `c_d^iso` is `c_d[K]` of Definition S4.2.
- **S3, Remark S3.21.** The remainder `O(L¹² e^{−L²})` is now said to be stated for orientation and not proved for general `L`.
- **S7, Lemma S7.1(b).** It states several sites in a rotated frame, which Lemma S12.8 states only at one site. A sentence derives the general case from the proof of S12.8. S7 uses only part (a).
- **S7, Remark S7.22.** It attributed to the main text's proof of Proposition 3.12 a statement that the main text makes in §§5.1 and 6. Corrected. In r8 that proof cites Theorem S7.9 and Remark S7.22.
- **S9, Theorem S9.13 and Corollary S9.16 (audit S89-1).** The printed ratio intervals are right, but dividing the printed intervals gives slightly wider ones, so a reader could not reproduce them. Theorem S9.13 now states the unrounded enclosures of `c₁` from which the ratios are formed, `[−0.2693988256731888, −0.2693988256725268]` for `d = 2` and `[−0.2118483463910089, −0.2118483460317414]` for `d = 3` (those of the certificate in [R12] of the main text), and the values of `c_{d,∞}` used; the proof of Corollary S9.16 names its inputs. Exact rational division confirms all four ratio intervals, at `L = ∞` and at `L = 24`.
- **S9, the method of Theorem S9.13 (audit S89-2).** The sentence that bounds from different representations on different boxes may be combined was false as a general principle: a principal branch can evaluate without exception on a box that a cut separates from the real interval. It now says what the certificate does: one representation on all boxes of an ellipse, each principal branch off its cut, and `Re Φ` continued as `(Φ + Φ*)/2`. The proof now also states the error decomposition of the tensor rule and names the certificate ([R12]). Not changed: the floating cross-check `−0.211848346210999` is inside the enclosure, as stated; the audit's recomputation to about `10⁻¹⁵` gives `−0.2118483462111935`.
- **S8, Propositions S8.2 and S8.3, Lemma S8.14, and the S10 inputs (audit S89-3).** With the dominating variable `T` of Convention S8.1, Taylor's formula gives `‖A_i − A_0‖ ≤ (r/2 + r²)T`, which is at most `rT` only for `r ≤ 1/2`. The radius `r_R` is now taken in `(0, 1/2]`, and the proof shows the bound. The inclusion `1 − e ≤ 1_{G_r^c}` needs the cap cylinder to be embedded, `r < L/(4√2)`. Proposition S8.3 now says so, as do the proof of Lemma S8.14 and the S10 inputs. Every use is at `r ≤ r₀^* < L/(4√2)`, so no conclusion changes.
- **S8, the proof of Theorem S8.6 (audit S89-5; OpenAI support note main#389 6193989322).** The pointwise limit used the Gaussian-nullity of the type boundaries. It now uses that the filtered determinants are locally Lipschitz, Lemma S7.14, which is the lemma of the OpenAI note. The audit found the earlier argument correct too.
- **Not taken in draft 10 (audit S89-4).** The bound `E_Q[(W_r/r²)1_{G_r^c}] ≤ C(r³/k)P^N` of Proposition S8.3 is imported from Theorem K of record [R7] of the main text, and the `c₁ℓ^{1/4}` term rests on it; Lemma S7.18 alone does not give it. The audit read the record, which is stated for every `d ≥ 2` and `L > 0`, and checked its layer arithmetic. A proof in S8, by the argument of Lemma S7.18 with `|α_M| ≤ 6k + rM₄/2` in the envelope, is planned for a later draft.
- **S11, Definition S11.1 and the proof of Theorem S11.5 (audit S10S11-3).** Definition S11.1 was stated for `k > 0`, while (S11.12) uses `Ā₂[K](0,u)`. The same formulas now define the kernels at `k = 0`, and `Ā₂[K](·,u)` is continuous on `[0,∞)` by (S11.16). In the proof of Theorem S11.5, `q₀ = (3/2)b² + 24k²` is the pin quadratic form: the pin density is a constant times `e^{−q₀/2}`, as the next sentence there uses.
- **S11, Hypothesis S11.11 → Corollary S11.11 (audit S10S11-1, checked independently).** Both parts follow from S10.
  - Part (a). Since `det H_i = r det K_i`, the surrogate (S11.8) is `12π_r E_Q[−det K_M det K_S 1{A≺0}]`, and the typed kernel is `12π_r E_Q[F_d(K_M) F_{d−1}(K_S)]` (S10.3). By (S10.40), `−det K_M det K_S = Ω + ϱ` with `|ϱ| ≤ Cr⁴T^{n₀}`, and (S10.42) compares the typed kernel with the same Ω-expectation. So the two kernels differ by `O(r³(1+k⁻¹)P^N e^{−c(b²+k²)})`, and by (S10.38) they have the same `r²` coefficient `𝒜₂` at every `k > 0`. Integrating in `b`, continuity in `k ≥ 0` ((S11.16) and (S10.39)) and Fubini's theorem (Remark S11.3(i)) give `c₂ = c₂[K_L]`.
  - Part (b) is Theorems S10.4 and S10.5, whose `c₂` is defined by (S10.22), which is (S11.11).
  - A paragraph after the proof records that at `k = 0` the value is the continuous extension: the typed kernel is `O(r⁵ log(1/r))` there (Proposition S10.7), while for the surrogate the two agree.
  - The S11 introduction, the pointer of the fold-expansion paragraph (now also to Lemma S10.12), Corollary S11.12(ii)–(iv), its proof, Remark S11.13, the status section and the two label rows of `hyp:ctwo-ident` below are updated. The statement now holds for every `d ≥ 2` and `L > 0`.
  - The new `checks/s10_s11_fold_check.py` checks the polynomial identities that the proof rests on, for pinned polynomial fields in `d = 2, 3`.
  - The S10S11 audit's Monte Carlo test (`d = 2`, Gaussian kernel, exact conditional law, 4·10⁶ samples per point) shows the difference of the two kernels scaling as `r³`, and a second, independent same-provider check of the proof found it complete.
- **S12, Theorem S12.5.** "Admit versions" is added, naming the versions.
- **S12, Lemma S12.21.** "Let r, b, k>0" read literally restricts to `b > 0`. It now takes the ranges of Definition S12.2, which include every `b ∈ ℝ`.
- **S12, Corollary S12.25 and the paragraph after it.**
  - The unrestricted statements that hold are now named exactly.
  - The compact-window bounds on the nonselected pairs fail for the unrestricted populations, whose nonselected density does not vanish (main-text Remark 2.5). The text now says so.
  - The integrability thresholds are attached to the displays that give them.
- **Main text only, applied in r8.** The §3.8 pointers for the jet coordinates (Lemma S4.20, Corollary S4.25) and for the scale check (Remark S4.26).
- **Main text only, for its next revision.** Theorem 2.3(iii) uses `c₂` both for the side-24 constant and for the Gaussian-kernel expression on `ℝ³`. It should write `c₂[K_ref]` for the latter, quote the interval at `L = 24`, and cite Theorem S11.5 and Corollaries S11.10 and S11.12(iii) (audit S10S11-2). §5.1 says that the periodization changes the coefficients by terms of order `L⁶e^{−L²/2}`. For `c₂`, Corollary S11.10 gives order `L⁸e^{−L²/2}` (audit S10S11-4).
- **Not taken.** The audit suggested proving the remainder of Remark S3.21 for `L ≥ 8` by rerunning the proof of Theorem S4.4. The remark is marked as orientation instead.

**Check scripts now shipped.** All three end `ALL_CHECKS_PASS`.
- `checks/s1s2_verify_check.py`: 56 exact checks. It is this concordance's `chk_s1s2/verify_s1s2.py`, unchanged except for its name.
- `checks/s6_limit_check.py`: 19 checks. It contains the draft-2 checks of S6, now written as assertions, and the `ϱ`-independence.
- `checks/s10_s11_fold_check.py`: 54 exact checks, for pinned polynomial fields of degree 6 to 8 in `d = 2, 3`: (S10.16), the identities (S10.41) of Lemma S10.12, Step 1, and the identity of the `r²` coefficient there with `X₂` of Theorem S11.7, Step 2 (S11.23). It is new in draft 10.
**Reads of the draft-9 repair at the time of this draft.** Cursor PASS (main#389 6182910843); OpenAI DOT-O2-D9 PASS_SCOPED, bound to the TeX source (6193704852). Two xAI pickups (6182815146, 6184901584) had no delivery or release. §S5's other slices: O1 and O3 PASS_SCOPED (xAI). The main text's slice M3 PASS_SCOPED (xAI, 6191908445).

**Build.** 158 pages, with no LaTeX error and no undefined reference. Every label of draft 9 keeps its number. The one overfull line, 29 pt in S10, is unchanged.

---

# Concordance — supplement sections S1 and S2

Files: `supp_s1_setting.tex` (Section S1, `\label{sec:setting}`), `supp_s2_contact.tex` (Section S2, `\label{sec:contact}`).
Writer: W1. Numbers below are those produced by the compile check in `chk_s1s2/` (statements and equations numbered within section).

## Sources

| key | document | identity |
|---|---|---|
| TA | `04_TECHNICAL_APPENDIX.pdf` (project doc), "Exact technical appendix for the side-24 manuscript", 30 July 2026 | per-module SHA-256 as printed in the TA: A `835d9ce4…`, F `9d227108…`, H `8dafdd7b…`, I `b958c21d…`, K `82e1d2f5…`, L `bcfe2f86…` |
| GF | `/home/claude/okc/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` (1 Aug 2026), Parts C.1, C.4 (and A.1.1, A.1.4 for the definitions shared with S5/S6) | 48,972 B, SHA-256 `b2269c72…` |
| MT | V3 main text `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/` | `s1_field.tex` `587e8302…`, `s2_results.tex` `35d04369…`, `s3_proof.tex` `1019605d…` (SHA-256) |

No Math- repository record is used in S1–S2.
Exact symbolic re-checks of the transcribed displays: `chk_s1s2/verify_s1s2.py` (56 checks, transcript `verify_s1s2.out.txt`, ends `ALL_CHECKS_PASS`).

## (a) Statements and displays

### Section S1 (`supp_s1_setting.tex`)

| label | no. | content | source |
|---|---|---|---|
| `eq:setting-K24` | (S1.1) | definition of K_24, Z_24 | MT (1.1); K_24(0)=1 added (B-list S1-1) |
| `lem:setting-spectral` | Lemma S1.1 | Poisson-summation spectral representation, all weights positive | GF C.1 eq. (C.1); TA F §2 (1)–(2) **corrected** (correction 1); MT (1.2) |
| `eq:setting-spectral` | (S1.2) | K_24 = Z_24^{-1} Σ a_m e_m, a_m = (2π)^{3/2}24^{-3}e^{-2π²|m|²/24²} | GF (C.1); TA F (1)–(2) corrected |
| `eq:setting-T` | (S1.3) | finite derivative evaluation T | TA F §2 (3) |
| `eq:setting-dist` | (S1.4) | associated distribution 𝒯; convention T̂(m)=T(e_m) | TA F §2 (5); convention made explicit (added) |
| `lem:setting-gram` | Lemma S1.2 | spectral Gram identity and zero-variance criterion | TA F §2 (4), (6) (corrected prefactor, GF C.1) |
| `eq:setting-var` | (S1.5) | Var T(f) = Z_24^{-1} Σ a_m |T̂(m)|² | TA F (4) corrected per GF C.1; = MT (1.3) |
| `eq:setting-var-zero` | (S1.6) | Var = 0 ⇔ T̂ ≡ 0 ⇔ 𝒯 = 0 | TA F (6) |
| `thm:setting-gram` | Theorem S1.3 | full-spectrum Gram positivity, with proof | TA F §3, Theorem 3.1 and its proof |
| `cor:setting-eigenfloor` | Corollary S1.4 | compact uniform eigenfloors | TA F §3, Corollary 3.2 (formalized; proof added) |
| `prop:setting-frame` | Proposition S1.5 | independence of the corrected pair frame: list (7) positive definite, det 12 at e^{-|z|²/2}; remote block (8) | TA F §4 (7)–(8); det 12: TA F §4 ("planar contact covariance determinant is 12"), cf. TA B (6), GF B.1(a) |
| `eq:setting-list` | (S1.7) | contact-limit derivative list | TA F (7) |
| `eq:setting-remote` | (S1.8) | remote family f(y), ∇f(y), Hess f(y) | TA F (8) |
| (paragraph after Prop. S1.5) | — | corrected frame = invertible image of raw pins, contact limit = (7) up to order and −1/12; kernel-independent transformation | TA F §4, first paragraph and "same algebraic coordinate transformation" sentence |
| `conv:setting-density` | Convention S1.6 | density convention (verbatim content) | GF C.4 (correction 3) |
| `def:setting-pairspace` | Definition S1.7 | typed ordered local pair space 𝒫_ρ(f) | TA L §2 (1) |
| `eq:setting-pairspace` | (S1.9) | 𝒫_ρ(f) | TA L (1) |
| `lem:setting-coordinates` | Lemma S1.8 | Log_M(S) unique; (x,r,t) bijection | TA L §2 (2)–(4); proof added |
| `eq:setting-coords` | (S1.10) | u = Log_M(S), x, r, t | TA L (2)–(3) |
| `eq:setting-inverse` | (S1.11) | M = x−(r/2)t, S = x+(r/2)t | TA L (4); MT (3.1), (3.4) |
| `prop:setting-jacobian` | Proposition S1.9 | dM dS = dx r^{d-1}dr dσ; antipodal remark; no factor 1/2 | TA L §3 (5)–(10) and the antipodal paragraph |
| `eq:setting-dMdS` | (S1.12) | dM dS = dx r^{d-1} dr dσ_{d-1}(t) | TA L (10) |
| `eq:setting-linear` | (S1.13) | (x,u) ↦ (M,S) | TA L (5); MT (3.2) |
| `eq:setting-Theta` | (S1.14) | block matrix Θ | TA L (6)–(7) (symbol A → Θ) |
| `eq:setting-detone` | (S1.15) | |det Θ| = 1, dM dS = dx du | TA L (8) |
| `eq:setting-polar` | (S1.16) | du = r^{d-1} dr dσ_{d-1}(t) | TA L (9) |
| `cor:setting-d3` | Corollary S1.10 | d = 3, L = 24 specialization | TA L §6 (13)–(14); MT (3.1)–(3.3) and the antipodal sentence |
| `eq:setting-dMdS3` | (S1.17) | dM dS = dx du = dx r² dr dσ_2(t) | TA L (14); MT (3.3) |
| `prop:setting-multiplicity` | Proposition S1.11 | H_0 unit multiplicity, with proof | TA L §4 (11) and proof; MT §3.1 second paragraph |
| `eq:setting-bijections` | (S1.18) | the three bijections | TA L (11) |
| (paragraph after Prop. S1.11) | — | hypotheses used; application to f_24 | TA L §4 last paragraph; MT Prop. 1.1 |
| `rem:setting-branch` | Remark S1.12 | branch count; adjacency as an existential Boolean mark | TA L §5 (12) |
| `eq:setting-adjacency` | (S1.19) | A(M,S) = 1 iff some half-branch converges to M | TA L (12) |

### Section S2 (`supp_s2_contact.tex`)

| label | no. | content | source |
|---|---|---|---|
| `def:contact-pins` | Definition S2.1 | contact pair, pins, intended types, raw pins, corrected frame | TA A §2 (2)–(3); MT (3.4)–(3.5), (3.7)–(3.8); Π_{r,x,t}, z_r from MT Def. 3.1 / GF A.1.1 |
| `eq:contact-MS` | (S2.1) | M, S | TA A §2; MT (3.4) |
| `eq:contact-pins` | (S2.2) | pins | TA A (2); MT (3.5) |
| `eq:contact-raw` | (S2.3) | Π_r and target z_r(b,κ) | TA A §2 (X_r → Π_r); MT Def. 3.1 |
| `eq:contact-Vframe` | (S2.4) | V^+, V^-_corr | TA A (3); MT (3.7) |
| `eq:contact-Gframe` | (S2.5) | G^±_e | TA A (3); MT (3.8) |
| (paragraph after Def. S2.1) | — | frame invariance; finite frame atlas | TA A §2 last two sentences |
| `lem:contact-pin-jacobian` | Lemma S2.2 | pin Jacobian, corrected target, density identity, p^corr_r → p^corr_0 under convergence of Σ_r | TA A §3 (4)–(6) and last paragraph; MT "determinant … is r^{-6}" after (3.8) |
| `eq:contact-detDelta` | (S2.6) | |det Δ_r| = r^{-(d+3)} | TA A (4) (T_r → Δ_r) |
| `eq:contact-target` | (S2.7) | y_r(b,κ) = (b−κr³/12, −κ/6, 0,…) | TA A (5) |
| `eq:contact-density` | (S2.8) | p_Π = r^{-(d+3)} p^corr_r | TA A (6) (p_X, p_Y → p_Π, p^corr_r) |
| `lem:contact-corrected` | Lemma S2.3 | side-24 corrected pins: nondegeneracy for 0<r<12, segment averages, contact limit Y_0, uniform eigenfloors, remote block | TA F §4 (contact limit P_0^corr, nondegeneracy) + TA F §3 (Thm 3.1, Cor 3.2) + TA A §3 (assumption verified); uniform floors as in TA H §5 "G1"/§6 (26) (assembled; added steps) |
| `eq:contact-segment` | (S2.9) | segment-average form of V^-_corr, G^-_e | added (integration by parts) |
| `eq:contact-limitframe` | (S2.10) | Y_0 | TA F §4 ("contact limit … distinct derivative coordinates", list (7)); explicit form added |
| `def:contact-weight` | Definition S2.4 | E^0, W_r, Z_r, K^all_r | MT Def. 3.1 (3.3a), Lemma 3.2; GF A.1.1, A.1.4; TA A (10) (Z_r) |
| `eq:contact-weight` | (S2.11) | W_r, Z_r, K^all_r = p_Π Z_r | MT (3.3a); GF A.1.4 |
| `lem:contact-endpoint` | Lemma S2.5 | exact endpoint factor r; det H_p = r D_p | TA I §2 (1), (3)–(5); TA A §4 (7) (general d) |
| `eq:contact-Hblock` | (S2.12) | H_p block form | TA I (1) |
| `eq:contact-taylor` | (S2.13) | Taylor integral identities | TA I (3) |
| `eq:contact-scaled` | (S2.14) | a_p = rα_p, b_p = rβ_p | TA I (4); TA A (7) (A^r_p, B^r_p → α_p, β_p) |
| `eq:contact-detrD` | (S2.15) | det H_M = r D_M, det H_S = r D_S; explicit D_p | TA I (5); MT (3.9); explicit D_p formula added |
| `eq:contact-Wr` | (S2.16) | W_r = r²|D_M D_S| 1{types} ≤ r²|D_M D_S| | TA I §2 ("Type indicators can only decrease absolute values") |
| `lem:contact-moments` | Lemma S2.6 | Gaussian polynomial moments; degree six (two Hessians) and nine (three Hessians) | TA H §2.1 (1) (correction 6) |
| `eq:contact-polymoment` | (S2.17) | E|P(G)| ≤ C_P(1+‖a‖^k) | TA H (1) (degree ≤ 9 there; general k, see added steps) |
| `hyp:contact-regular` | Hypothesis S2.7 | corrected-pin nondegeneracy, uniform Gaussian moment control, null boundary {det Q = 0} | TA A §3 (assumption), §4 ("uniform Gaussian moment control", "probability-zero boundary det Q = 0"), §5 ("endpoint regression/moment hypotheses"); made precise (see Issues I4) |
| (sentence after Hyp. S2.7) | — | (a)–(b) ⇒ p^corr_r → p^corr_0 | TA A §3 last sentence |
| `prop:contact-soft-limits` | Proposition S2.8 | contact limits of the endpoint factors | TA A §4 (8)–(10); proof added |
| `eq:contact-alpha-limits` | (S2.18) | α_M → −κ, α_S → +κ | TA A (8) |
| `eq:contact-det-limits` | (S2.19) | r^{-1}det H_M → −κ det Q, r^{-1}det H_S → +κ det Q | TA A (9) |
| `eq:contact-zeta` | (S2.20) | Z_r = r²ζ_r, ζ_r → ζ_0 | TA A (10) (z_r, z_0 → ζ_r, ζ_0) |
| (sentence after Prop. S2.8) | — | r² independent of d: one soft eigenvalue per endpoint | TA A §4 last sentence; MT before (3.9) |
| `lem:contact-normalizer` | Lemma S2.9 | f_24: Z_r > 0 (all 0<r<12); Hypothesis S2.7 holds; c_Z r² ≤ Z_r ≤ C_Z r², E^0[W_r²] ≤ C_W r⁴ | TA H §5 item "G2" (22) with TA I §2 (5) and its last sentence (correction 8); (i) and (ii) added steps |
| `eq:contact-normalizer` | (S2.21) | normalizer bounds | TA H (22) |
| (paragraph after Lemma S2.9) | — | same soft-mode factor; MT (3.9) restated | TA I §2 last sentence; MT (3.9) and following sentence |
| `lem:contact-gap` | Lemma S2.10 | cubic-gap change of variables | TA K §2 (1)–(3); MT (3.6) |
| `eq:contact-h` | (S2.22) | h = f(S) = b − κr³/6 | TA K (1) |
| `eq:contact-gapjac` | (S2.23) | dh/dκ = −r³/6, |dh| = (r³/6)dκ | TA K (2); MT (3.6) |
| `rem:contact-gap` | Remark S2.11 | 1/6 is not an orientation/branch/index factor; contact-density factor; universal radial measure | TA K §2 (3)–(5) and the paragraph "The factor 1/6 is not…" |
| `eq:contact-factor` | (S2.24) | (r³/6) r^{-(d+3)} r² = (1/6) r^{2−d} | TA K (4) |
| `eq:contact-universal` | (S2.25) | (1/6) r dr | TA K (5) |
| `thm:contact-cancellation` | Theorem S2.12 | generic-fold dimension cancellation (i)–(iv) and conclusion | TA A §5 (11)–(15) and Theorem 5.1 |
| `eq:contact-KR` | (S2.26) | Kac–Rice density (r³/6) K^all_r = (r³/6) p_Π Z_r | TA A (11) (K_r^{(A)} = (r³/6)K^all_r; see Issues I5) |
| `eq:contact-J` | (S2.27) | J_r := r^{d−2}(r³/6)K^all_r | TA A (12) |
| `eq:contact-Jexact` | (S2.28) | J_r = (1/6) p^corr_r ζ_r | TA A (13) |
| `eq:contact-J0` | (S2.29) | J_0 | TA A (14); TA K (12) |
| `eq:contact-radial` | (S2.30) | (r³/6)K^all_r r^{d−1}dr = J_r r dr | TA A (15) |
| (paragraph after Thm. S2.12) | — | stationarity, per-unit-volume normalization; hypotheses hold for f_24 | GF A.3.1 proof (24^{-3}∫dx = 1); MT Prop. 1.2 proof (Kac–Rice validity); added assembly |
| `cor:contact-powercount` | Corollary S2.13 | d = 3 power count; J_r = G^all_r/6; no extra multiplier | TA L §6 (15) and last paragraph; TA K (4)–(5); MT (3.10) and the sentence after it |
| `eq:contact-powercount` | (S2.31) | (r³/6) r^{-6} r² r² dr dσ_2 = (1/6) r dr dσ_2 | TA L (15); MT (3.10) |
| `lem:contact-lifetime` | Lemma S2.14 | lifetime pushforward; lifetime factor 6^{2/3}/3; exponent −1/3 | TA K §3 (6)–(11); TA A (16)–(17); MT (3.11)–(3.12) |
| `eq:contact-ell` | (S2.32) | ℓ = b − h = κr³/6 | TA K (6); TA A (16); MT (3.11) |
| `eq:contact-rell` | (S2.33) | r = (6ℓ/κ)^{1/3} | TA K (7) |
| `eq:contact-dr` | (S2.34) | dr = (6/κ)^{1/3}(1/3)ℓ^{-2/3}dℓ | TA K (8) |
| `eq:contact-rdr` | (S2.35) | r dr = (6^{2/3}/3)κ^{-2/3}ℓ^{-1/3}dℓ | TA K (9); TA A (17); MT (3.12) |
| `eq:contact-inverse` | (S2.36) | dℓ/dr = κr²/2, r dr = (2/(κr))dℓ | TA K (10) |
| `prop:contact-pushforward` | Proposition S2.15 | lifetime density of the contact measure; DCT; coefficient formula | TA A §6 (18)–(19) and the DCT sentence; exact representation (S2.37) added |
| `eq:contact-nu-exact` | (S2.37) | exact pushforward representation | added (abstract form of GF (A.8) / MT (3.12b)) |
| `eq:contact-c` | (S2.38) | c = (6^{2/3}/3)∫ J_0 s_0 κ^{-2/3} | TA A (19) |
| `thm:contact-exponent` | Theorem S2.16 | dimension-independent generic-fold exponent | TA A §6 Theorem 6.1 with (18) |
| `eq:contact-asymp` | (S2.39) | ν = cℓ^{-1/3}(1+o(1)) | TA A (18) (placed under Theorem 6.1's hypothesis; see Issues I3) |
| (sentence after Thm. S2.16) | — | identification for ν_{3,24} is in Section S6 | pointer only |
| `rem:contact-nodouble` | Remark S2.17 | no double counting of 1/6 | TA K §4 (12)–(13); MT (3.12b) |
| `eq:contact-scalar` | (S2.40) | (1/6)(6^{2/3}/3) = 6^{2/3}/18 | TA K (13) |

## Notation conversions (source symbol → supplement symbol)

| source | supplement | reason |
|---|---|---|
| TA A X_r (raw pins) | Π_r = Π_{r,x,t} | MT Def. 3.1 notation |
| TA A Y_r (corrected pins); TA F/H/I P_r^corr | Y_r | TA A notation; matches S6 draft |
| TA A T_r (raw-to-corrected map) | Δ_r | MT uses T_r for the transverse matrix (3.18) |
| TA A target (5) "Y_r = …" | y_r(b,κ); limit y_0(b,κ) | matches S6 draft |
| TA A p_X, p_Y, p_0 | p_Π, p^corr_r, p^corr_0 | matches S6 draft (GF writes p_Π(z_r(b,κ))) |
| TA A A_M^r, A_S^r, B_M^r, B_S^r | α_M, α_S, β_M, β_S | TA I notation |
| TA A Q_M^r, Q_S^r | Q_M, Q_S | TA I notation |
| TA A z_r, z_0 (= Z_r/r², limit) | ζ_r, ζ_0 | MT uses z_r(b,κ) for the pin target; matches S6 draft |
| TA A K_r = (r³/6)p_X Z_r | (r³/6) K^all_r, with K^all_r = p_Π Z_r | MT Lemma 3.2 / (3.12a) and GF A.1.4 define K^all_r without the gap Jacobian |
| TA A/K "Q<0", "Q negative definite" | Q ≺ 0 | MT notation |
| TA L A (block matrix (6)) | Θ | A(M,S) is kept for adjacency (TA L (12)) |
| TA L P_ρ(f) | 𝒫_ρ(f) | typography |
| TA F Λ_i^*δ | 𝒯_i = P_i(−∂)δ_{x_i} | definition made explicit |
| TA H §5 "planar contact weight" | contact-limit weight κ²(det Q)²1{Q≺0} | V3 reserves "planar" for d = 2 |
| TA I e_i (coordinate direction) | frame direction e ∈ {t, n_1, …} | same object in the frame of TA A §2 |

## (b) Corrections applied (PLAN.md numbering)

1. **Correction 1 (spectral weight).** Lemma S1.1 and Lemma S1.2 use a_m = (2π)^{3/2} 24^{-3} exp(−2π²|m|²/24²) with the normalization Z_24^{-1} kept as a separate factor, so that K̂_24(m) = Z_24^{-1} a_m agrees with MT (1.2), and Var T(f) = Z_24^{-1} Σ a_m |T̂(m)|² (GF C.1, eq. (C.1)). In the TA extract, Module F (2) lacks the factor (2π)^{3/2}24^{-3}; Module F (1) and (4) already carry Z_24^{-1} (see Issues I1). Positivity conclusions are unchanged.
2. **Correction 3 (density convention).** Convention S1.6 is GF C.4 verbatim in content (only "Theorem 2.1" and "Proposition 1.2" are written "of the main text"). The GF C.3 part of PLAN correction 3 (Module K §5 symbols) concerns Module K §§5–6, which are not in S1–S2.
3. **Correction 6 (determinant degrees).** Lemma S2.6: det H_M det H_S (and D_M D_S) has degree six; a product of three 3×3 Hessian determinants has degree nine (TA H §2.1). Lemma S2.9(iii) uses degree six for ζ_r and degree twelve for the second moment of D_M D_S.
4. **Correction 8 (normalizer).** TA A §4's "Under uniform Gaussian moment control, Z_r = r²z_r" is not left as an assumption for f_24: Lemma S2.9 proves c_Z r² ≤ Z_r ≤ C_Z r² and E^0[W_r²] ≤ C_W r⁴ by the argument of TA H §5 item "G2" together with the endpoint factor of TA I §2 (Lemma S2.5). The assignment's item (b) labels this "correction 3"; it is correction 8 in PLAN.md.

## (c) Added steps (each fully justified in the text; numerically or symbolically re-checked where marked ✓ in `verify_s1s2.py`)

S1
- S1-1. K_24(0) = 1 from |24n|²/2 = 288|n|² (✓).
- S1-2. Proof of Lemma S1.1 by Poisson summation with ĝ(k) = (2π)^{3/2}e^{−|k|²/2} (GF C.1 states the identity); summability Σ K̂(m)(1+|m|)^N < ∞ (✓ 1-D factor to 50 digits).
- S1-3. Explicit Fourier convention T̂(m) = T(e_m) = ⟨𝒯, e_m⟩ (TA F: "up to the fixed Fourier convention").
- S1-4. Proof of the variance identity (S1.5) by term-wise differentiation of (S1.2) (TA F states (4) without proof).
- S1-5. Theorem S1.3: functionals written as Λ_i = (P_i(∂)f)(x_i) with 𝒯_i = P_i(−∂)δ_{x_i}; the second part of the proof (vanishing of the partial sum at each support point; symbol criterion) written out.
- S1-6. Corollary S1.4: the "compact parameter family" formalized (coefficients and support points continuous in the parameter); proof (continuity of entries, Weyl's inequality, compactness) added.
- S1-7. Proposition S1.5(i): symbols are distinct monomials in rotated orthonormal coordinates; explicit block computation det = 2·6·1 = 12 for e^{−|z|²/2} (✓ det = 12; corrected-limit determinant 1/12 ✓).
- S1-8. Proposition S1.5(ii): uniform lower bound in (x, y, frame) via Corollary S1.4 (TA F §4 states nondegeneracy only) (positivity at a sample remote point ✓ for e^{−|z|²/2}).
- S1-9. Index convention stated (number of negative Hessian eigenvalues).
- S1-10. Proof of the bijection in Lemma S1.8 (TA L asserts it).
- S1-11. Proposition S1.11: the handle-index statement (crossing an index-k point attaches a (d−k)-handle) and the third bullet (index ≤ d−2 does not change β_0), used implicitly by TA L §4.
- S1-12. Application of Proposition S1.11 to f_24 via MT Proposition 1.1.
- S1-13. |det Θ| = 1 re-checked for d = 1,…,5 (✓).

S2
- S2-1. Stationarity: no dependence on x for f_24 (intro); per-unit-volume normalization 24^{-3}∫dx = 1 (paragraph after Theorem S2.12; GF A.3.1 proof).
- S2-2. Lemma S2.2: explicit 2×2 blocks of Δ_r (✓ |det Δ_r| = r^{-(d+3)} and target for d = 1,…,5); explicit Gaussian density formula for the last assertion; qualifier "whenever Cov(Π_r) is nondegenerate"; "continuous in t" added to the convergence assumption of TA A §3.
- S2-3. Lemma S2.3 (assembled): (i) from Theorem S1.3; (ii) segment-average identities (S2.9) by integration by parts (✓ trapezoid identity, limit −∂_t³f/12, weight mass r³/6) and L²-convergence with error O(r); (iii) uniform eigenfloor on (0, r_1] × S² for every r_1 < 12 (TA B §2 / E §3 / H "G1" state compact eigenfloors "for sufficiently small r"; the extension to (0, r_1] combines this with Corollary S1.4); (iv) uniform eigenfloor of Y_r joined with the remote block for 0 < r ≤ r_0 < δ (cf. TA H §6 (26)); the statement that the limiting target pins ∂_t³f(x) = 2κ (✓).
- S2-4. Definition S2.4 placed in S2 because Z_r is needed here; Palm law, elder mark and K^sel_r are left to Section S5.
- S2-5. Lemma S2.5 stated and proved for every d (TA I §2 is for d = 3; TA A (7) asserts the same structure for every d); explicit D_p = α_p det Q_p − r β_p^T adj(Q_p) β_p (✓ det H_p = r D_p and deg D_p = d for d = 2, 3, 4; Taylor identities ✓).
- S2-6. Lemma S2.6 for every degree k (TA H §2.1 states degree ≤ 9 and "or any larger integer used elsewhere"); proof written out.
- S2-7. Hypothesis S2.7 (see Issues I4).
- S2-8. Proof of Proposition S2.8: Gaussian regression bounds (uniform Gaussian moment control); identification α_{M,0} = 6·(second coordinate of Y_0) ⇒ α_M → −κ, α_S → +κ (✓); almost-sure representation U = m + Γ^{1/2}N; Sylvester's law of inertia for the type indicator (✓ congruence identity); uniform integrability via Lemma S2.6 (degree 4d); uniform convergence on compacts and continuity of ζ_0.
- S2-9. Lemma S2.9(i): Z_r > 0 for every 0 < r < 12 and all marks, by Gram positivity of (Π_r, H_M, H_S).
- S2-10. Lemma S2.9(ii): verification of Hypothesis S2.7 for f_24 (segment averages of U_r with densities 2(r/2 ∓ s)/r² (✓ masses r²/2, limits ∓½·g''(0)); nondegenerate limiting conditional law of Q; ζ_0 > 0).
- S2-11. Lemma S2.9(iii): |D_M D_S| ≤ C|U_r|⁶ for r ≤ 1, and the degree-6/degree-12 moment bounds.
- S2-12. Theorem S2.12: explicit assumption that the two-point Kac–Rice formula holds; proof of (i) from the Kac–Rice formula and the two Jacobians; d = 1 remark (σ_0 counting measure on S⁰); paragraph showing the hypotheses hold for f_24 (Kac–Rice on {δ ≤ r < ρ} as in MT Prop. 1.2, then δ ↓ 0).
- S2-13. Corollary S2.13: J_r = G^all_r/6 = (1/6) p^corr_r ζ_r (conversion to MT's G-notation; ✓).
- S2-14. Lemma S2.14: substitution in the inverse-derivative check displayed (✓ all of (S2.32)–(S2.36)).
- S2-15. Proposition S2.15: definition of ν_{s,ϱ} (separation cutoff ϱ), exact representation (S2.37) (✓ cutoff κ > 6ℓ/ϱ³), DCT conclusion stated as ℓ^{1/3}ν_{s,ϱ}(ℓ) → c.
- S2-16. Theorem S2.16: one-line proof from Proposition S2.15.
- S2-17. Remark S2.17: d = 3 form c = (6^{2/3}/18)∫κ^{-2/3}G^all_0 s_0 (✓ scalar identities).
- S2-18. Power counts (S2.24), (S2.25), (S2.28), (S2.30), (S2.31) re-checked symbolically for general d (✓).

## (d) Issues (not fixed silently; source location, problem, suggested fix)

- **I1. Assignment wording of correction 1.** The assignment says TA Module F (1)–(2) and (4) omitted "the positive prefactor (2π)^{3/2}24^{-3} and the normalization Z_24^{-1}". In the TA extract, (1) and (4) do carry Z_24^{-1}; only the weight (2) lacks (2π)^{3/2}24^{-3} (as GF C.1 says). The supplement applies GF (C.1) exactly. No source error beyond GF C.1's.
- **I2. GF C.4 (Convention S1.6), last sentence.** "…i.e. for Lebesgue-a.e. ℓ in the sequential sense selected by that representative" is not a precise statement: a pointwise asymptotic as ℓ↓0 is a property of one fixed representative, and it is not preserved by changes on null sets (MT Convention 2.2 says this correctly). C.4 also quotes the law as (1+o(1)), whereas MT Theorem 2.1 now states (1+O(ℓ^{1/3})). Kept verbatim as instructed. Suggested fix: replace the "i.e." clause by "a change of density on a null set need not preserve a pointwise asymptotic" and refer to "the asymptotics of Theorem 2.1".
- **I3. TA A §6, (18).** (18) is asserted as the output of dominated convergence before Theorem 6.1 imposes "finite and strictly positive"; DCT gives ℓ^{1/3}ν(ℓ) → c, and the form c ℓ^{-1/3}(1+o(1)) needs 0 < c < ∞. Also ν(ℓ) is not defined in §6 (no separation cutoff). The supplement states the limit in Proposition S2.15 (with ν_{s,ϱ} the contact density at separation below ϱ, as in MT (3.12b)) and displays (18) in Theorem S2.16 under Theorem 6.1's hypothesis. Suggested fix: the same in the source.
- **I4. TA A §§3–5, hypotheses of the general-d statements.** "Uniform Gaussian moment control" and "the endpoint regression/moment hypotheses" are not defined; the validity of the two-point Kac–Rice formula is implicit in (11); and the "probability-zero boundary det Q = 0" is asserted for an arbitrary C⁴ field, although it needs a nondegenerate (or at least non-concentrated) limiting conditional law of Q. The supplement collects them as Hypothesis S2.7 (with (b) made precise as convergence of the joint covariance of (U_r, Y_r) to that of the explicit limit (U_0, Y_0), the form in which TA I §2 and TA H §2.1 use it) and as an explicit assumption of Theorem S2.12. For f_24 every one of them is proved (Lemma S2.3, Lemma S2.9(ii), paragraph after Theorem S2.12). Suggested fix: adopt this wording in the source.
- **I5. Clash of K_r.** TA A (11) defines K_r = (r³/6)p_X Z_r (gap Jacobian included), whereas MT Lemma 3.2 / (3.12a) and GF A.1.4 use K^all_r = p_Π Z_r (gap Jacobian written separately). The supplement writes (r³/6)K^all_r in (S2.26). Related clashes resolved by renaming: TA A's z_r (normalized normalizer) vs MT's pin target z_r(b,κ) (→ ζ_r), TA A's pin map T_r vs MT's transverse matrix T_r of (3.18) (→ Δ_r). Suggested fix: the integrator and the S5 writer should use ζ_r, Δ_r and K^all_r = p_Π Z_r consistently (the S6 draft already uses ζ_r, Y_r, y_r, p^corr_r).
- **I6. TA H §5, item "G2".** "planar contact weight" must mean the contact-limit weight κ²(det Q)²1{Q≺0} of the side-24 field (V3 reserves "planar" for d = 2), and "the odd Schur variance is positive" is not by itself why the expectation is positive: positivity follows from the nondegeneracy of the limiting conditional law of Q (Gram positivity at one point), which also contains the positivity of the odd Schur variance. Lemma S2.9(ii) argues this way. Suggested fix: reword G2 accordingly.
- **I7. TA F §4, wording.** "Equivalently, the exact planar contact covariance determinant is 12": the value 12 is for the R³ kernel e^{−|z|²/2}, not for K_24, so it confirms nondegeneracy at that kernel rather than being equivalent to it; and in d = 3 the contact limit of the corrected frame is the full list (7), not a proper sublist. The supplement states "for the Bargmann–Fock kernel e^{−|z|²/2} on R³ … equals 12". Wording only.
- **I8. Z_r > 0 in MT Definition 3.1.** MT states Z_r = E^0[W_r] > 0 without a range; GF A.1.1 derives it from G2, which covers only small r on compact marks. Lemma S2.9(i) proves Z_r > 0 for all 0 < r < 12 and all marks. Suggested fix: cite Lemma S2.9(i) in MT Definition 3.1.
- **I9. Scope of TA I §2.** TA I §2 is written for d = 3 while TA A (7) uses the same structure in every dimension; the supplement proves Lemma S2.5 for every d (same proof). Not an error.
- **I10. Assignment numbering.** Item (b) of the assignment calls the normalizer "correction 3"; per PLAN.md it is correction 8 (applied as such).

## Compile check

`chk_s1s2/test_s1s2.tex` = preamble of `supp_main.tex` + `\input{../supp_s1_setting}` + `\input{../supp_s2_contact}`; two `pdflatex -interaction=nonstopmode` runs. No LaTeX errors, no overfull or underfull boxes. Undefined references: only `sec:selection` (3×), `sec:coefficient` (1×) and `sec:limit` (1×), which belong to other sections. No `\cite` is used, so no BibTeX run is needed. Output: 14 pages.

---

# Concordance — Section S3, "The finite-side coefficient" (`supp_s3_coefficient.tex`)

Writer W2, 13 October 2026. Section label `\label{sec:coefficient}`. Numbers below are the
printed numbers in the integrated supplement (section S3); the isolated compile check
(`chk_s3/`) prints the same numbers with prefix S1.

## Sources

| Key | Document | Identity |
|---|---|---|
| TA-A | Technical appendix (`04_TECHNICAL_APPENDIX.pdf`, project doc, 30 July 2026), Module A (corrected pins, dimension cancellation, pushforward) | extract SHA-256 `835d9ce4…` (as printed in TA) |
| TA-C | same, Module C (exact finite-side coefficient reduction), §§2–10, eqs. (2)–(39) | extract SHA-256 `bbcbc0c4…` (as printed in TA) |
| TA-K | same, Module K (absolute-normalization factorization), §§5–6, eqs. (14)–(22) | extract SHA-256 `82e1d2f5…` (as printed in TA) |
| GF | `/home/claude/okc/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` (1 Aug 2026), Part B.2 (lines 562–646) and Part C.3 (lines 684–699) | SHA-256 `b2269c72…` |
| GF-s | GF scripts `scripts/d2_closed_form.py` (`c22f32ad…`), `scripts/goe_cone_coefficient.py` (`82822afd…`) and their `.out.txt` transcripts | — |
| MT | V3 main text `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/`: `s1_field.tex` (`587e8302…`), `s2_results.tex` (`35d04369…`), `s3_proof.tex` (`1019605d…`) | — |

No Math- repository record is used in this section.

## (a) Statements

| Label | No. | Content | Source |
|---|---|---|---|
| `conv:coefficient-kernels` | Conv. S3.1 | K_L (0<L<∞), K_∞=e^{-\|z\|²/2}, field f, covariance formula, jets J_L, J_0, surface measure dσ₂ | TA-C uses K_L without display; K_L written as MT (1.1) with 24→L; J_0, J_24 as MT (3.25); covariance formula as GF B.1(a) |
| `def:coefficient-frame` | Def. S3.2 | directional contact frame u_t, g_t, h_t, q_t, Q_t; limiting corrected pin vector Y_t and target | TA-C §2 (2)–(7); MT (2.2)–(2.4) |
| `lem:coefficient-parity` | Lemma S3.3 | (i) even/odd independence; (ii) the corrected pin frame converges to Y_t, target → (b,−κ/6,0,0) | (i) TA-C §2 last paragraph; (ii) TA-C (6)–(7) with TA-A §2 (3), §3 (5) and MT (3.5), (3.7)–(3.8). Both proofs are added steps |
| `def:coefficient-cov` | Def. S3.4 | Σ_g, s_u, c_ug, σ_t², Σ_h, Σ_q, C_qh, Γ_t, F, D_t | TA-C §3 (8)–(13), §4 (15), (18); MT (2.5)–(2.8), (3.20) |
| `lem:coefficient-posdef` | Lemma S3.5 | entries are derivatives of K_L at 0 (lattice sums); positive definiteness; q_t\|{h_t=0} ~ N(0,Γ_t) | TA-C §3 (14) and last paragraph (see Issue I2); positivity proof added via MT (1.2)–(1.3) |
| `lem:coefficient-birth` | Lemma S3.6 | birth-integration collapse | TA-C §4 (16), (17), (19); MT (3.20) |
| `lem:coefficient-gap` | Lemma S3.7 | normalized-gap integral | TA-C §5 (20)–(23); MT (3.21) |
| `lem:coefficient-frame` | Lemma S3.8 | frame invariance; I(t) well defined, continuous, positive | TA-C Theorem 6.1, second paragraph (see Issue I3); continuity and positivity added |
| `def:coefficient-cL` | Def. S3.9 | c_{3,L} for L∈(0,∞], B_3; 0<c_{3,L}<∞; c_{3,∞}; c_{3,L}=Φ(J_L) | TA-C §6 (24)–(25); MT (2.9), (2.11), (3.25) |
| `thm:coefficient-main` | Thm. S3.10 | the selected coefficient (6^{2/3}/18)∫κ^{-2/3}G^all_0 equals c_{3,L}; frame independence; inputs explicit | TA-C Theorem 6.1 and §6 text ("Kac–Rice contact factor 1/6, lifetime Jacobian 6^{2/3}/3"); hypothesis wording: Issue I1; uses MT Theorem 3.9 |
| `prop:coefficient-cone` | Prop. S3.11 | deterministic cone–sphere reduction, constant 15/(4π), nested formula | TA-C §7 (26)–(30); MT (3.22) |
| `prop:coefficient-cubic` | Prop. S3.12 | invariance under signed permutations, 48-fold chamber reduction | TA-C §8 (31)–(32) |
| `rem:coefficient-quadrature` | Rem. S3.13 | one forty-eighth of the outer sphere; inner cone quadrature with exact cone tests | TA-C §8 last two sentences |
| `lem:coefficient-planar-jet` | Lemma S3.14 | at the Bargmann–Fock jet: Σ_g=I_3, Σ_h=diag(3,1,1), σ²=1/24, Γ_t=Γ^{(∞)} (eigenvalues 1,2,10/3), GOE-plus-scalar law; D_t=D_2 | GF B.2 "Derivation displayed" paragraph (lines 564–573) and GF-s `goe_cone_coefficient.py`; values also TA-C (33)–(36); MT (3.23)–(3.24) |
| `prop:coefficient-planar-limit` | Prop. S3.15 | L→∞ limits of all objects; D_t→D_2; c_{3,L}→c_{3,∞} | TA-C §9 (33)–(37); proof added (TA-C gives none) |
| `thm:coefficient-D2` | Thm. S3.16 | D_2 = 29/6 − √6 (closed-form proof, Steps 1–7) | GF B.2 (B.2), (B.2′), I_1, I_3, I_5, boxed value (lines 575–601); GF-s `d2_closed_form.py`; TA-C (37); integration-by-parts steps displayed and checked (added steps) |
| `rem:coefficient-D2check` | Rem. S3.17 | Step 2 = the reduction stated with MT (3.24); numerical value of D_2 | MT §3.7 text after (3.24); decimal computed here (mpmath) |
| `cor:coefficient-cinf` | Cor. S3.18 | c_{3,∞} two closed forms and decimal | TA-C (38); GF B.2 "Coefficient recombination" (lines 610–615); TA-K (21); MT (2.11) |
| `prop:coefficient-iso` | Prop. S3.19 | isotropic recombination c_d^iso = A_d^jet Ω^{2/3} m_4^{(d−2)/2}/a^{(3d+4)/6} | TA-K §5 (14)–(19); symbols per GF C.3 (Correction 3); see Issues I5–I6 |
| `cor:coefficient-BF` | Cor. S3.20 | Bargmann–Fock specialization; c_3^BF = c_{3,∞}; factor check; c_{3,24}=c_3^BF[1+δ_24]; normalizations cancel | TA-K §6 (20)–(22); factor check added |
| `rem:coefficient-first-image` | Rem. S3.21 | first-image expansion of the relative correction; pointer to Section S4 | TA-C §10 (39); MT (3.31) for P_3; see Issue I7 |

Subsection labels: `ssec:coefficient-frame` (S3.1), `ssec:coefficient-cov` (S3.2),
`ssec:coefficient-birth` (S3.3), `ssec:coefficient-main` (S3.4), `ssec:coefficient-reductions` (S3.5),
`ssec:coefficient-planar` (S3.6), `ssec:coefficient-D2` (S3.7), `ssec:coefficient-isotropic` (S3.8).

## (a) Numbered displays

| Label | No. | Display | Source |
|---|---|---|---|
| `eq:coefficient-KL` | (S3.1) | K_L, Z_L | MT (1.1) with 24→L; GF C.1 (C.1) left form |
| `eq:coefficient-covformula` | (S3.2) | Cov(∂^αf,∂^βf)=(−1)^{\|β\|}∂^{α+β}K_L(0) | GF B.1(a) (line 509) |
| `eq:coefficient-odd` | (S3.3) | u_t, g_t | TA-C (2); MT (2.2) |
| `eq:coefficient-h` | (S3.4) | h_t | TA-C (3); MT (2.3) |
| `eq:coefficient-q` | (S3.5) | q_t, Q_t(q) | TA-C (4)–(5); MT (2.4) |
| `eq:coefficient-Y` | (S3.6) | Y_t=(f,u_t,g_t,h_t) | TA-C (6) |
| `eq:coefficient-target` | (S3.7) | target (b,−κ/6,0,0) | TA-C (7) |
| `eq:coefficient-Sigmag` | (S3.8) | Σ_g | TA-C (8); MT (2.5) |
| `eq:coefficient-su` | (S3.9) | s_u, c_ug, c_gu | TA-C (9) |
| `eq:coefficient-sigma` | (S3.10) | σ_t² | TA-C (10); MT (2.6) |
| `eq:coefficient-Sigmah` | (S3.11) | Σ_h | TA-C (11); MT (2.5) |
| `eq:coefficient-Sigmaq` | (S3.12) | Σ_q, C_qh, C_hq | TA-C (12) |
| `eq:coefficient-Gamma` | (S3.13) | Γ_t | TA-C (13); MT (2.7) |
| `eq:coefficient-F` | (S3.14) | F(q) | TA-C (15); MT (3.20) |
| `eq:coefficient-Dt` | (S3.15) | D_t | TA-C (18); MT (2.8) |
| `eq:coefficient-qcond` | (S3.16) | q_t\|{h_t=0} ~ N(0,Γ_t) | TA-C (14) |
| `eq:coefficient-even-factor` | (S3.17) | even contact factor | TA-C (16) |
| `eq:coefficient-collapse` | (S3.18) | birth-integration collapse | TA-C (17); MT (3.20) |
| `eq:coefficient-ph` | (S3.19) | p_{h_t}(0)D_t | TA-C (19) |
| `eq:coefficient-ucond` | (S3.20) | u_t\|{g_t=0} ~ N(0,σ_t²) | TA-C (20) |
| `eq:coefficient-podd` | (S3.21) | odd pin density | TA-C (21) |
| `eq:coefficient-pg` | (S3.22) | p_{g_t}(0) | TA-C (22) |
| `eq:coefficient-gap` | (S3.23) | normalized-gap integral | TA-C (23); MT (3.21) |
| `eq:coefficient-I` | (S3.24) | outer integrand I(t) | integrand of TA-C (24); named I(t) as in TA-C (31) |
| `eq:coefficient-cL` | (S3.25) | c_{3,L} | TA-C (24); MT (2.9) |
| `eq:coefficient-B3` | (S3.26) | B_3 | TA-C (25); MT (2.9) |
| `eq:coefficient-G0` | (S3.27) | G^all_0 = p_{Y_t}·κ²·E[F\|pins] | TA-A (6), (8)–(10), (14) with G_0=6J_0 (conversion); MT Theorem 3.9 notation |
| `eq:coefficient-coefint` | (S3.28) | (6^{2/3}/18)∫κ^{-2/3}G^all_0 | MT Theorem 3.9 display; TA-A (19) with s_0=1 |
| `eq:coefficient-At` | (S3.29) | A_tA_t^⊤=Γ_t | TA-C (26) |
| `eq:coefficient-calQ` | (S3.30) | q=A_tv, 𝒬_t(v) | TA-C (27) |
| `eq:coefficient-cone` | (S3.31) | cone–sphere formula for D_t | TA-C (28); MT (3.22) |
| `eq:coefficient-1534` | (S3.32) | (2π)^{-3/2}∫ϱ⁶e^{-ϱ²/2}=15/(4π) | TA-C (29) |
| `eq:coefficient-nested` | (S3.33) | nested S²×S² formula, constant 45Γ(7/6)/(2^{10/3}π^{9/2}) | TA-C (30) |
| `eq:coefficient-48` | (S3.34) | ∫_{S²}I = 48∫_𝓕 I | TA-C (31) |
| `eq:coefficient-chamber` | (S3.35) | 𝓕={t_1≥t_2≥t_3≥0} | TA-C (32) |
| `eq:coefficient-planar-values` | (S3.36) | Σ_g=I_3, Σ_h=diag(3,1,1), σ²=1/24 | GF B.2 line 573; MT (3.23); TA-C (33)–(35) (as limits) |
| `eq:coefficient-Gamma-inf` | (S3.37) | Γ^{(∞)} | GF B.2 lines 568–571 |
| `eq:coefficient-GOE` | (S3.38) | Q_t(q_t)\|{h_t=0} =ᵈ G_2+√(2/3)ZI_2 | TA-C (36); GF B.2 line 564–565; MT (3.24) |
| `eq:coefficient-D2def` | (S3.39) | definition of D_2 | GF B.2 lines 564–566; TA-C (37) |
| `eq:coefficient-lim-g` | (S3.40) | Σ_g→I_3, det→1 | TA-C (33) |
| `eq:coefficient-lim-h` | (S3.41) | Σ_h→diag(3,1,1), det→3 | TA-C (34) |
| `eq:coefficient-lim-sigma` | (S3.42) | σ²→1/24 | TA-C (35) |
| `eq:coefficient-lim-Q` | (S3.43) | limit law G_2+√(2/3)ZI_2 | TA-C (36) |
| `eq:coefficient-lim-D` | (S3.44) | D_t→D_2=29/6−√6 | TA-C (37) |
| `eq:coefficient-mu` | (S3.45) | truncated normal moments μ_k | GF B.2 line 591 ("M_k") |
| `eq:coefficient-mu-values` | (S3.46) | μ_0, μ_2, μ_4 | GF B.2 line 591 |
| `eq:coefficient-D2` | (S3.47) | D_2=29/6−√6 | GF B.2 boxed (line 601); TA-C (37); MT (3.24) |
| `eq:coefficient-B2` | (S3.48) | eigenvalue form of D_2 | GF (B.2) (line 579) |
| `eq:coefficient-WD` | (S3.49) | det Q=(W²−D²)/4, Q≺0 ⇔ W<−D | GF B.2 line 589; MT text with (3.24) |
| `eq:coefficient-B2prime` | (S3.50) | one-dimensional D-integral | GF (B.2′) (line 593) |
| `eq:coefficient-Ik` | (S3.51) | I_k | GF B.2 lines 596–597 |
| `eq:coefficient-D2split` | (S3.52) | Φ/φ split of D_2 | added step (GF "term-by-term recombination"; coefficients match GF-s `d2_closed_form.py`: −5/24·I_3, 1/64·I_5, 25/12·I_1, √30 terms) |
| `eq:coefficient-I135` | (S3.53) | I_1, I_3, I_5 | GF B.2 lines 598–599 |
| `eq:coefficient-cinf` | (S3.54) | c_{3,∞} closed forms | TA-C (38); GF B.2 line 612–613; MT (2.11) |
| `eq:coefficient-cinf-decimal` | (S3.55) | decimal 0.041775931840598343342936665428575556466681519661… | GF B.2 line 614 (48 digits as printed there; MT (2.11) prints the first 46) |
| `eq:coefficient-symbols` | (S3.56) | a, m_4, χ, Ω | GF C.3 (C.2) — Correction 3 |
| `eq:coefficient-Dn` | (S3.57) | D_n | TA-K (14); wording GF C.3 — Correction 3 |
| `eq:coefficient-Bd` | (S3.58) | 𝓑_d | TA-K (15) |
| `eq:coefficient-Kd` | (S3.59) | 𝒦_d | TA-K (16) |
| `eq:coefficient-gamma76` | (S3.60) | Γ(7/6)=Γ(1/6)/6 | TA-K (17) |
| `eq:coefficient-ciso` | (S3.61) | c_d^iso | TA-K (18) |
| `eq:coefficient-Ajet` | (S3.62) | A_d^jet | TA-K (19) |
| `eq:coefficient-BFparams` | (S3.63) | D_2, a=1, m_4=3, Ω=6 | TA-K (20) |
| `eq:coefficient-c3BF` | (S3.64) | c_3^BF closed form | TA-K (21) |
| `eq:coefficient-delta24` | (S3.65) | c_{3,24}=c_3^BF[1+δ_24] | TA-K (22) |
| `eq:coefficient-first-image` | (S3.66) | first-image expansion | TA-C (39); MT (3.31) |

Unnumbered displays inside proofs are either transcriptions of the cited source steps or added steps (listed below).

## Notation conversions

- TA-C "dt" (and "dv") on S² → dσ₂(t), dσ₂(v); TA-C (29) radial variable r → ϱ (r is the pair separation).
- TA-C §8 "signed permutation group B_3" → 𝓗_3 (B_3 is the constant (S3.26)).
- TA-C (36) "Q | h_t=0 = G_2+√(2/3)ZI_2" → equality in law, written =ᵈ; "negative definite"/"<0" → ≺0.
- Planar values: Γ_t at the Bargmann–Fock jet named Γ^{(∞)}; the Bargmann–Fock kernel is K_∞ and its jet J_0 (as MT (3.25)); objects of K_L carry a superscript (L) only in Prop. S3.15.
- TA-A J_0 (density w.r.t. dκ, including 1/6; prefactor 6^{2/3}/3) → MT's G^all_0 = 6J_0 (rescaled density r⁴K^all_r w.r.t. dh; prefactor 6^{2/3}/18).
- GF B.2: truncated moments M_0, M_2, M_4 → μ_0, μ_2, μ_4 (M_2 would clash with the matrix M_n of TA-K at n=2 and with the maximum M); GF "c" (=√(3/20)) → θ; GF "a"=1/8 written as the number; T, D, W, s kept (D, T, W as in MT §3.7).
- TA-K: B_d → 𝓑_d (B_3 would clash with the constant B_3); K_d → 𝒦_d (clash with the kernel K_L).
- TA-C Theorem 6.1 "every admissible fixed L in the LS-DER-053 regime" → "a fixed side L for which the contact limit and the compact elder selection hold (… the case for L = 24)" (Issue I1).
- Provenance phrases dropped: "recovers LS-DER-055", "reproducing LS-DER-047", "LS-DATA-056A/B verify …", "in LS-DER-050", "externally reviewed relative correction".

## (b) Corrections applied

- **Correction 3** (symbols of TA Module K defined per GF C.3): before Prop. S3.19 the text defines n=d−1, the sectional covariance K^{(1)}, a=−(K^{(1)})''(0), m_4=(K^{(1)})^{(4)}(0), χ=−(K^{(1)})^{(6)}(0), Ω=aχ−m_4² (display (S3.56), = GF (C.2)), their variance interpretations (a=Var ∂_tf, m_4=Var ∂_ttf, χ=Var ∂_tttf, Ω/a=Var(∂_tttf | ∂_tf)), β=m_4/3, D_n as "the centered negative-definite determinant moment of the conditional transverse Hessian M_n in dimension n, as in (S3.15)" (display (S3.57)), and the Bargmann–Fock values a=1, m_4=3, χ=15, Ω=6 with Ω/(2a)=3=72σ_t². (The density-convention half of PLAN item 3, GF C.4, does not concern S3.)
- No other corrections were applied.

## (c) Added steps

All checked by exact/symbolic or high-precision computation in `chk_s3/verify/` (scripts `v1`–`v7`, transcripts `*.out.txt`, environment `ENV.txt`: Python 3.11.15, SymPy 1.14.0, mpmath 1.3.0, SciPy 1.17.1).

1. Lemma S3.3(i): proof of even/odd independence (odd derivatives of the even kernel vanish at 0; jointly Gaussian ⇒ independent). [v2: max |Cov(even,odd)| = 0 at the planar jet]
2. Lemma S3.3(ii): Taylor proof that V^+, V^-_corr, G^±_e converge to f, u_t, g_t, h_t, and that the pins give V^+=b−κr³/12, V^-_corr=−κ/6, G^±=0.
3. Lemma S3.5: positivity justification (TA-C cites the planar nondegeneracy only): full-spectrum Gram positivity, MT (1.2)–(1.3) with 24→L, positive spectral density for L=∞, distinct monomial symbols, Schur complements; explicit statement that σ_t² and Γ_t are rational functions of the derivative entries (Issue I2).
4. Lemma S3.6: proof details (factorization p_{(f,h)}=p_h p_{f|h}, mixture/tower property, integrability 0≤F≤|q|⁴ from |det Q|≤|q|²).
5. Lemma S3.7: the substitution κ=(72σ²y)^{1/2} giving (S3.23). [v1]
6. Lemma S3.8: transformation of g_t (omitted in TA-C), invariance of det Σ_g, det Σ_h and σ_t², regression equivariance Γ'=ρΓρ^⊤, F∘ρ=F; continuity of I(t) (local frames, continuity of Γ↦D(Γ)); strict positivity; hence 0<c_{3,L}<∞ (Def. S3.9). [v6: numerical frame-invariance check at L=4]
7. Paragraph before Thm. S3.10 and display (S3.27): recollection of the contact limit (TA-A (6), (8)–(10), (14)) in the main-text normalization G^all_r=r⁴K^all_r → p_{Y_t}·κ²E[F|pins].
8. Thm. S3.10 proof: factorization of G^all_0 by independence, Tonelli, and the constant identity (6^{2/3}/18)(2π)^{-3}Γ(7/6)72^{7/6}/(2√(2π)) = B_3 with 6^{2/3}72^{7/6}=2^{25/6}3³, 18·2·(2π)^{7/2}=2^{11/2}3²π^{7/2}. [v1]
9. Prop. S3.11 proof: polar integration, ∫ϱ⁶e^{-ϱ²/2}=15√(π/2), B_3·15/(4π)=45Γ(7/6)/(2^{10/3}π^{9/2}). [v1; v4: cone–sphere quadrature of D_2 = 2.3838435905…, A_t symmetric and Cholesky]
10. Prop. S3.12 proof: K_L(Pz)=K_L(z), transfer to the covariance objects, a.e. tiling of S² by the 48 chambers. [v6: I(Pt)=I(t) numerically at L=4; v7: group order 48, chamber area fraction 1/48]
11. Lemma S3.14 proof: explicit planar computation (Σ_q, C_qh, eigenvectors of Γ^{(∞)}, s_u=5/48, c_ug=(1/4,0,0), σ²=1/24, covariance match with (g_1+sZ,g_2,g_3+sZ)). [v2]
12. Prop. S3.15 proof: J_L→J_0 with O(L⁶e^{-L²/2}) image terms and Z_L=1+O(e^{-L²/2}); uniform convergence over t and frames (compactness of O(3)); continuity of D; c_{3,L}→c_{3,∞}.
13. Thm. S3.16 proof, Steps 1–7 (GF summarizes them): eigenvalue form; the trace/difference change of variables with the alternative entries-based derivation (T=g_1+g_3, D=((g_1−g_3)²+4g_2²)^{1/2}, independent N(0,4) components); the W-integration; the explicit Φ/φ split (S3.52) with φ-coefficient θD(400/3−20D²/3); the φ-terms = 625√6/1536; the by-parts formula for I_{2j+1} with R_j(D)=½e^{-D²/8}Σ_i (j!/i!)8^{j−i+1}D^{2i}, the coefficient sums 4, 42, 747; the recombination 928/3 − (2161/24)√6 and 29/6 − 2161√6/1536 + 625√6/1536 = 29/6 − √6. [v3: every intermediate asserted symbolically; I_1, I_3, I_5 also by 25-digit quadrature; (B.2′) by 30-digit quadrature; v4: Monte Carlo]
14. Rem. S3.17: decimal D_2 = 2.38384359055015523513… [v3]
15. Cor. S3.18 proof: B_3=Γ(1/6)/(2^{7/3}π^{7/2}); intermediate form 2^{-7/3}3^{-7/6}D_2Γ(1/6)π^{-5/2}; 29/6−√6 = 2^{-3/2}3^{-3/2}(29√6−36); 432=2⁴3³; decimal recomputed to 60 digits (0.041775931840598343342936665428575556466681519661044737884977…), agreeing with GF B.2 (48 digits) and MT (2.11) (46 digits). [v5, v7]
16. Prop. S3.19 proof: explicit exponent bookkeeping for general d. [v5: symbolic identity and d=1,…,5 numeric]
17. Cor. S3.20 proof: d=3 substitution; factor-by-factor check 𝓑_3=(2π)^{-1}p_{h_t}(0)D_t and 𝒦_3=(2π)^{-1/2}p_{g_t}(0)^{-1}∫p_{(u,g)}(−κ/6,0)κ^{4/3}dκ at the Bargmann–Fock jet; "normalizations cancel" justified through the common constant B_3. [v5, v7]

## (d) Issues (source mathematics kept; nothing silently fixed)

- **I1 — undefined scope in TA-C Theorem 6.1.** "For every admissible fixed L in the LS-DER-053 regime" is not defined in TA; the selection and transfer results in the sources (TA Modules E, H §5 "Only the fixed side-24 field is asserted here", I) are proved only for L = 24. The supplement states Thm. S3.10 for "a fixed side L for which the contact limit and the compact elder selection hold … this is the case for L = 24". Suggested fix: state the theorem for L = 24 only, or define admissibility explicitly.
- **I2 — TA-C §3 last paragraph.** "All entries in (8)–(13) are exact finite lattice sums of directional derivatives of K_L at zero." (a) For K_L the sums over n ∈ Z³ are infinite (absolutely convergent), not finite; (b) σ_t² (10) and Γ_t (13) are Schur complements, i.e. rational functions of such entries, not lattice sums themselves. The supplement writes: entries of Σ_g, s_u, c_ug, Σ_h, Σ_q, C_qh are constant multiples of derivatives of K_L at 0 (exact lattice sums), and σ_t², Γ_t are rational functions of them. Suggested fix: adopt this wording.
- **I3 — TA-C §2, first paragraph.** "All quantities below are independent of this transverse-frame choice" is imprecise: Σ_g, Σ_h, c_ug, Γ_t, h_t, q_t change with the frame (by congruence, as TA-C Theorem 6.1 itself says); only det Σ_g, det Σ_h, σ_t², D_t and the integrand are frame-independent. Also TA-C Theorem 6.1 omits the transformation of g_t (needed for det Σ_g and σ_t²). The supplement says "the quantities entering the coefficient formula … do not depend on the choice" and proves the invariance including g_t (added step 6).
- **I4 — TA-C §9 gives no proof** of the L → ∞ limits (33)–(37); GF B.2 derives the values at the planar jet only. Supplied as added step 12 (no change to the statement).
- **I5 — normalization of D_n in TA-K (14) / GF C.3.** GF C.3 defines D_n as the moment of "the conditional transverse Hessian in dimension n, exactly as in Module C (18)". For an isotropic kernel with β ≠ 1 the conditional transverse Hessian (given h_t = 0) is √β times the β = 1 law, so the Module C moment is β^n D_n^{(β=1)}. TA-K (15) and (18)–(19) agree with TA-C (19), (24) only if D_n is the moment of the β-normalized matrix M_n = G_n + √(2/3) Z I_n (G_n standard GOE_n): with the literal reading, (18) exceeds the TA-C value by the factor β^n (verified symbolically at d = 3: with D_n set equal to the TA-C moment D_t, (TA-C value)/(TA-K value) = 9/m_4² = β^{-2}; with D_n the β-normalized moment the ratio is 1; `v5_cinf_K.out.txt`), and the scale family K_λ(z)=e^{-λ|z|²/2} would give c ∝ λ^{3/2+2n} instead of the exact λ^{3/2} (GF B.2 scale check). No effect for Bargmann–Fock (β = 1), hence none on any statement used by the main text. The supplement transcribes GF C.3 as written. Suggested fix: "D_n = E[(det M_n)² 1{M_n ≺ 0}] with M_n = β^{-1/2}× the conditional transverse Hessian given h_t = 0; for an isotropic kernel M_n = G_n + √(2/3) Z I_n".
- **I6 — TA-K (15)–(16) are imported, not derived.** TA-K states 𝓑_d and 𝒦_d "in LS-DER-050" without derivation, and GF does not supply one. The supplement states them as the source does, proves only the multiplication (18)–(19), and checks (15)–(16) against TA-C (19), (23) at d = 3 for Bargmann–Fock (Cor. S3.20). A general-d derivation needs the d-dimensional analogues of (S3.19) and (S3.23) and the normalization of I5.
- **I7 — TA-C (39) remainder for general L.** The first-image expansion with remainder O(L¹²e^{-L²}) as L → ∞ is stated in TA-C §10 by reference to an earlier record; no proof is in the provided sources. Rem. S3.21 states it as a remark and points to Section S4 for the proved side-24 statement with explicit remainder bound.
- **I8 — integration overlap (not a source error).** `supp_s6_limit.tex` (Corollary `cor:limit-coefficient`) repeats the evaluation of (6^{2/3}/18)∫κ^{-2/3}G^all_0 into (2.9) (same computation as Thm. S3.10), and its (H1) paragraph proves the limit (S3.27) (`eq:limit-Gall`, ζ_0). The integrator may let one cite the other, and may replace the main-text pointers in the paragraph before Thm. S3.10 by references to the contact-measure section (S2) and to `eq:limit-Gall`.
- **I9 — symbol clashes in the sources** (resolved by notation conversion, listed above): TA-C's group "B_3" vs the constant B_3; TA-K's B_3 (even block at d=3) vs the constant B_3; GF's M_2 (truncated moment) vs TA-K's matrix M_2.

## Compile check

`chk_s3/test_s3.tex` = preamble of `supp_main.tex` + `\input{../supp_s3_coefficient}`; `pdflatex -interaction=nonstopmode` run three times: no errors, no overfull or underfull boxes; the only undefined reference is `sec:correction` (used twice; defined by Section S4). No citations (no BibTeX run needed). 12 pages in isolation. 21 statement labels, 66 equation labels, 8 subsection labels, no duplicates; `\tag` not used.

---

# CONC_s4 — concordance for `supp_s4_correction.tex` (section "The side-24 correction", `\label{sec:correction}`)

Writer: W3. Date: 13 October 2026. Math- `origin/main` at `9844f785` when read.

Statement numbers below are those of the stand-alone check build `chk_s4/test_s4.pdf` (prefix S1); in the assembled
supplement the prefix is that of this section (S4 if it is the fourth numbered section). Labels are authoritative.

## Sources (abbreviations used only in this file)

| Key | Document | Location used | Blob (first 8 hex) |
|---|---|---|---|
| [R] | Math- `coefficients/side24_remainder_20260929/PROOF.md` (primary) | §§1–7, (R1)–(R17) | `0bcf0a6f` |
| [R-res] | Math- `coefficients/side24_remainder_20260929/RESULTS.json` | `relative_remainder_rational`, `taylor_majorant`, `correction_intervals` | `14fd5a39` |
| [R-chk] | Math- `coefficients/side24_remainder_20260929/check.py` | `exp_bounds` (tail form), `q_interval` (enclosure method) | `dbcd70b8` |
| [PV] | Math- `frontiers/side24_periodization_20260929/PROOF.md` | preamble, §§1–4 (first variation) | `1a84f215` |
| [PR] | Math- `frontiers/side24_periodization_remainder_20260929/PROOF.md` | Lemma 1, Lemma 2, Lemma 3, §5 fact (iv) only | `03291568` |
| [V1] | Math- `coefficients/side24_v1/PROOF.md` (relied on by [R] §4: "We use the elementary derivative and reference-matrix bounds of side24_v1 Sections 2-3") | §1 (1), §2 (2), §3 (3) | `44b66f04` |
| [P15] | Math- `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (the coefficient expression that [R] §1 defines c_(d,L) by) | §15, (15.1)–(15.2) | `dfed3b8d` |
| [GF] | `/home/claude/okc/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md`, Part B.2 "First variation, derived from first principles"; `scripts/first_variation_derivation.py` (+ `.out.txt`); Part C.3 (symbols a, m4, χ, Ω) | B.2, (B.3), C.3 (C.2) | — |
| [TA-C], [TA-D], [TA-G], [TA-H] | `04_TECHNICAL_APPENDIX.pdf` (project), Modules C, D, G, H, local equation numbers | as cited | — |
| [MT] | V3 main text `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/` `s1_field.tex`, `s2_results.tex`, `s3_proof.tex` | (1.1), (2.2)–(2.11), (3.25)–(3.33) | — |

## (a) Map of statements and numbered displays

| Label | Kind (chk no.) | Content | Source |
|---|---|---|---|
| `sec:correction` | Section | proves (2.10) of the main text; d = 2, 3 | [R] §1 Theorem R; [MT] (2.10) and §3 "Side-24 correction" |
| (intro paragraph) | text | three-part outline | summary of [R] §§2–7 |
| `conv:correction-notation` | Convention S1.1 | φ, q_L, Z_L, K_L, q = e^{-288}, frames, svec, M_{4/3} | K_L: [R] §1; q: [MT] (3.28) ("Put q=e^{-288}"), [R] z; svec: [R] §2; M_{4/3} = 2^{2/3}Γ(7/6)/√π: [R] (R4), [PR] Lemma 1 proof (E\|N(0,τ²)\|^{4/3}) |
| `def:correction-coef` | Definition S1.2 | coefficient expression c_d[K]; c_{d,ref}, c_{d,24}; "admissible" | [P15] §15 (15.1)–(15.2), quoted in [PR] "Scope and exact sources"; [R] §1 (c_(d,L), c_(d,ref) definitions), §2 (G, t, H, V = Hu, A). "Admissible" names the nondegeneracy of [P15] ("Finite-jet rank proves tau_u>0, positive densities ... D_u>0") |
| `eq:correction-jetvars` | (S1.1) | g_t, ζ_t, h_t, A_t | [R] §2; [P15] §15; for d = 3 [MT] (2.2)–(2.4) |
| `eq:correction-tauD` | (S1.2) | τ_t², D_t | [P15] (15.1); [R] text after (R4) |
| `eq:correction-coef` | (S1.3) | (15.2) | [P15] (15.2); [PR] "Scope" (identical quotation) |
| (text after Def.) | text | c_{d,ref} is a reference expression; no isotropy; not three scalars | [R] §1 ("local reference expression, not an infinite-volume persistence theorem"); [R] §2 ("uses the FULL covariance ... need not be rotationally invariant ... NOT claimed to depend only on three isotropic moment scalars") |
| `lem:correction-d3` | Lemma S1.3 | d = 3: c_3[K] is (2.9); c_{3,ref} = c_{3,∞} | [MT] (2.2)–(2.9), (2.11); [TA-C] §§2–6 eqs (2)–(25); [PR] Lemma 1 (prefactor and reproduction of [V1] (1)); [V1] §1 (1) |
| `eq:correction-d3` | (S1.4) | (2.9) with B_3 | [MT] (2.9); [TA-C] (24)–(25) |
| `thm:correction-main` | Theorem S1.4 | \|c_{d,24}/c_{d,ref} − 1 − P_d(24)q\| < 10^{-216}, d = 2, 3; c_{d,24} < c_{d,ref} | [R] §1 Theorem R, (R1), (R2), "In particular c_(d,24) < c_(d,ref) for both dimensions"; proof [R] §7 |
| `eq:correction-main` | (S1.5) | (R1) | [R] (R1) |
| `eq:correction-Pd` | (S1.6) | P_d(L), P_2(24) = −26045568, P_3(24) = −620813376/35 | [R] (R2) |
| `cor:correction-main` | Corollary S1.5 | (2.10) of the main text; c_{3,24} < c_{3,∞} | [MT] (2.10), (3.31); [R] (R1)–(R2) at d = 3; [PV] Theorem; [GF] (B.3) |
| `eq:correction-main3` | (S1.7) | (2.10) | [MT] (2.10) |
| `def:correction-subspace` | Definition S1.6 | pinned subspace 𝒫, weight ψ, Ψ(C) | [R] §2 (E, h, p, ν, (R3)); [PR] §1 (section, w, F, n_free) |
| `eq:correction-weight` | (S1.8) | ψ, p = 4/3 + 2m, ν = w + p | [R] §2 |
| `eq:correction-Psi` | (S1.9) | Ψ(C) | [R] (R3) |
| `eq:correction-dims` | (S1.10) | (N, w, p, ν) for d = 2, 3 | [R] §3 ("For d=2: (n,k,p,nu)=(6,2,10/3,16/3). For d=3: (10,4,16/3,28/3)"); ν − N = −2/3 for all d: added step A3 |
| `lem:correction-subspace` | Lemma S1.7 | fixed-subspace representation; c_d = 2^{m/2}96^{-1/3}∫Ψ | [R] (R4) and the paragraph after it; [PR] Lemma 1 (parity, factorization, 96^{-1/3}) |
| `eq:correction-subspace-rep` | (S1.11) | (R4) | [R] (R4) |
| `eq:correction-coef-Psi` | (S1.12) | c_d[K] = 2^{m/2}96^{-1/3}∫Ψ(C_t)dσ | [PR] (1.1) (raw coordinates, constant 96^{-1/3}) combined with the factor 2^{-m/2} of [R] (R4) (svec coordinates) |
| `lem:correction-frame` | Lemma S1.8 | transverse-frame invariance; continuity in t | [R] §2 last paragraph; [PR] Lemma 2 proof (F(SCSᵀ) = F(C)); [P15] after (15.1); [TA-C] Theorem 6.1, last paragraph |
| `lem:correction-score` | Lemma S1.9 | score identities | [R] (R5) ("The negative sign in the second quadratic term matters"); [PR] Lemma 3; [TA-H] §3 (4)–(5) |
| `eq:correction-score` | (S1.13) | (R5) | [R] (R5) |
| `lem:correction-radial` | Lemma S1.10 | χ²_ν radial law, E R² = ν, E R⁴ = ν(ν+2) | [R] §3, paragraph after (R6) |
| `eq:correction-chi` | (S1.14) | moments | [R] §3 |
| `prop:correction-taylor` | Proposition S1.11 | (a) sandwich, (b) (R6), (c) (R7), (d) ratio, Taylor bound 2000θ² | [R] §3 (R6)–(R8) ("covariance sandwich", "uniform Gaussian majorant times a polynomial", K(n,ν), K = 1012/9, (4/3)^5(5/4)^{14/3} ≤ (5/3)^5 = 3125/243, (1−e)^{-2} ≤ 16/9, "The latter bounds the former term by term"); dominating function: [PR] §5 fact (iv); two-sided Loewner bound also [V1] §3 (3) |
| `eq:correction-pointwise` | (S1.15) | (R6) | [R] (R6) |
| `eq:correction-second` | (S1.16) | (R7) | [R] (R7) |
| `eq:correction-ratio` | (S1.17) | Ψ(C_s) ≤ (1−θ)^{-N/2}(1+θ)^{ν/2}Ψ(C_0) | [R] §3, display before (R8) |
| `eq:correction-taylor` | (S1.18) | (R8) | [R] (R8); constant 25300000/19683 = [R-res] `taylor_majorant` |
| `lem:correction-first` | Lemma S1.12 | (R9), constants 29/3 (d=3), 17/3 (d=2) | [R] (R9) |
| `eq:correction-first` | (S1.19) | (R9) | [R] (R9) |
| `rem:correction-homogeneity` | Remark S1.13 | Ψ(aC) = a^{-1/3}Ψ(C); derivatives −1/3, 4/9; amplitude check | [R] §3 "As an independent normalization check ..." (incl. "Confusing the ambient dimension n with the free dimension k fails this check"); amplitude: [P15] §15 last paragraph, [PV] §4 |
| `eq:correction-homog` | (S1.20) | scaling | [R] §3; [PR] Lemma 4 proof (s^{-1/3}) |
| `lem:correction-hermite` | Lemma S1.14 | Wick formula; 76\|x\|^6e^{-\|x\|²/2}, 15; directional Hermite identity | [R] §4 first paragraph; [V1] §2 ("product-rule pairings ... sum_j q!\|x\|^(q-2j)/[2^j j!(q-2j)!] ... 1+15+45+15=76 ... at most 15"); Hermite: [PV] §2, [R] §5 ("Apply the Hermite identities") |
| `eq:correction-wick` | (S1.21) | partial-matching expansion | added step A8 (explicit form of [V1] §2 "product-rule pairings") |
| `eq:correction-derivbound` | (S1.22) | derivative bounds | [R] §4; [V1] §2 |
| `lem:correction-lattice` | Lemma S1.15 | 10^{-126} < q < 10^{-125}; lattice sums ≤ 1458q | [R] §4 (27j³, 27j⁶, ratio 512z³ < 1/2, 1458z, exp(288/125) > 10 by degree-20 sum); [R] §7 (exp(16/7) < 10, degree 40 + geometric tail); [V1] §2; tail form: [R-chk] `exp_bounds` |
| `eq:correction-lattice` | (S1.23) | 1458q bounds | [R] §4 |
| `lem:correction-jet` | Lemma S1.16 | (R10) | [R] (R10); [V1] (2) and the text before it (normalizer subtraction, infinite sum, all directions, exact variance-one normalization) |
| `eq:correction-E` | (S1.24) | B_0 = 14523826191, E = 21175738586478·10^{-125} | [R] §4/(R10); [V1] (2) |
| `eq:correction-jetbound` | (S1.25) | 1458B_0q < E | [R] (R10) |
| `lem:correction-cov` | Lemma S1.17 | C_0 ≥ I/3 (block structure, eigenvalues d+2, 2; 2/3, 7/9); entries < 2E; ‖C_24 − C_0‖ ≤ 20E; θ ≤ 60E | [R] (R11) and the paragraph before it; [V1] §1 (reference covariances), §3 (eigenvalues 8 ± √58, C_ref ≥ I/3); [PR] §5 fact (ii) |
| `eq:correction-theta` | (S1.26) | θ_0 = 60E = 87480B_0·10^{-125} < 1/4 | [R] (R11); 87480B_0 as in [R] (R17) |
| `def:correction-jets` | Definition S1.18 | jets 𝒥_d, action of O(d), frame matrix C(J;𝔣), 𝒰_d, Φ_d, C_0 | [TA-D] §2 (1) (coefficient map on the even jet through order six); [MT] (3.25); [PV] preamble ("Φ(J) ... O(3)-invariant coefficient functional on the full even covariance-jet tensors through order six"); covariance rule Cov(∂^αf, ∂^βf) = (−1)^{\|β\|}∂^{α+β}K(0): [PV] §4, [PR] Lemma 1 proof, [GF] B.1(a); formalization: added step A11 |
| `eq:correction-CJ` | (S1.27) | C(J;𝔣) | covariance rule as above; matrix form added (A11) |
| `eq:correction-Phi` | (S1.28) | Φ_d(J) | [PR] (1.1); [TA-D] (1) |
| `lem:correction-rot` | Lemma S1.19 | linearity; C(g·J; g𝔣) = C(J;𝔣); Φ_d O(d)-invariant | [PV] §1 ("O(3)-invariant because rotating a covariance jet rotates the angular integration variable"); [R] §5 ("O(d)-invariant"); [PR] Lemma 2 ("transforms covariantly") |
| `lem:correction-iso` | Lemma S1.20 | isotropic jets J^iso(a, m4, χ); covariances; conditional covariance; Φ ∝ a^{-d/2-2/3}m4^{d/2-1}Ω^{2/3}; d log Ψ | [R] §5 (isotropic family, Cov(V), conditional covariance, D_u ∝ m4^{d-1}, (R13)); [PV] §1 (a = −⟨∂²_v⟩, m4 = ⟨∂⁴_v⟩, χ = −⟨∂⁶_v⟩), §4 items 1–4 and "Exponents in every d"; [PR] Lemma 2 last sentence ("determined by one scalar per order"); [GF] C.3 (C.2) |
| `eq:correction-Jiso` | (S1.29) | J^iso | [PV] §4 ("∂_i∂_jK(0) = −aδ_ij, ∂_i∂_j∂_k∂_lK(0) = (m4/3)(δδ+δδ+δδ), axial sixth derivative −χ") |
| `eq:correction-isocond` | (S1.30) | Cov(g), Cov(h), τ², conditional Cov(A \| h = 0) | [R] §5; [PV] §4 items 1–4 |
| `eq:correction-isoPsi` | (S1.31) | Φ ∝ a^{-d/2-2/3}m4^{d/2-1}Ω^{2/3} | [R] (R13) first line; [PV] §4 |
| `eq:correction-isodiff` | (S1.32) | (1 − d/2)da + (d/6 − 1)dm4 + dχ/9 | [R] (R13) second line |
| `lem:correction-haar` | Lemma S1.21 | Haar projection of the first variation; differentiability of Φ_d at J_0 | [R] §5 first paragraph ("invariant LINEAR functional and sees precisely the Haar projection"); [PV] §1; [PR] Lemma 2 (∫dF[C_ref](Δ_u)dσ = 4π dF[C_ref](Δ̄)) and §5 fact (iv) last sentence; [TA-H] §4 text after (13); part (d): added step A13 |
| `eq:correction-haarvar` | (S1.33) | normalized derivative | [R] (R13) with §5 first paragraph |
| (text after Lemma S1.21) | text | first-derivative statement only | [R] §5 ("only a first-derivative assertion; it is not a replacement ... at second order"); [PV] §1 ("no such reduction is asserted for Φ at a general anisotropic jet") |
| `def:correction-nearest` | Definition S1.22 | J^(1)_L with normalizer share; equals jet of 2q_L e^{-\|x\|²/2}Σ(cosh(Lx_i) − 1); = J^(1)_near of (3.28); R_jet | [R] §5 ("Apply the Hermite identities to the 2d nearest images ±Le_i and subtract their normalizer contribution 2dz times the reference jet"), §6 ("J_24^(1) includes the nearest-image normalizer subtraction; it is not just the raw six-image numerator", b_1 = a_1 − s_1a_0); [GF] B.2 (K_p); [PV] preamble; [TA-H] §4 (11); [TA-G] §4; [MT] (3.28) |
| `eq:correction-J1` | (S1.34) | J^(1)_L | [R] §5, §6 |
| `lem:correction-images` | Lemma S1.23 | per-direction identity; sphere moments; (R12) | [R] §5 (sphere moments via Gaussian factorization; (R12)); [PV] §§1–3 (d = 3); [GF] B.2 (δm_{2k} = 2(−1)^k(He_{2k}(L) − He_{2k}(0)), spherical average); [TA-H] (14) |
| `eq:correction-images` | (S1.35) | (J^(1)_L)_{2j}(v,…,v) = 2q_LΣ[He_{2j}(Lv_i) − He_{2j}(0)] | added step A14 (explicit form of [R] §5 "Apply the Hermite identities") |
| `eq:correction-variations` | (S1.36) | (R12) | [R] (R12) |
| `prop:correction-firstvar` | Proposition S1.24 | first normalized nearest-image variation = P_d(L)q_L, every d ≥ 2, L > 0 | [R] §1 Proposition V; §5 ("Substitution of (R12) gives exactly (R2) ... for every fixed integer d>=2"); [PV] Theorem (d = 3); [PR] Lemma 2 |
| `eq:correction-firstvar` | (S1.37) | DΦ_d(J_0)[J^(1)_L]/Φ_d(J_0) = P_d(L)q_L | [R] §5; [MT] (3.30) |
| `cor:correction-firstvar3` | Corollary S1.25 | d = 3: P_3, DF(J_0)[J^(1)_24] = P_3(24)e^{-288}; δa, δm4, δχ; pure partials; regroupings; P_2(24) | [GF] B.2 (all items) and `first_variation_derivation.py` steps 2–4; [TA-H] §4 (13)–(17); [MT] (3.30)–(3.31) and the paragraph after (3.31); [PV] Theorem, §3, §4; [R] (R2) |
| `eq:correction-P3` | (S1.38) | P_3(L) two forms, P_3(24) | [MT] (3.31); [TA-H] (16)–(17); [GF] (B.3); [PV] Theorem |
| `eq:correction-var3` | (S1.39) | δa = −2L², δm4 = 6L⁴/5 − 12L², δχ = −6L⁶/7 + 18L⁴ − 90L² | [GF] B.2; [TA-H] (14); [PV] §3; [MT] paragraph after (3.31) |
| `eq:correction-partials` | (S1.40) | ∂_a = ∂_{m4} = −1/2, ∂_χ = 1/9 | [GF] B.2 ("including the homogeneity-of-degree-2 term of the cone moment D(Γ), which contributes +2/m4"); script step 3; [MT] paragraph after (3.31) |
| `eq:correction-regroup` | (S1.41) | −δa/2 − δm4/2 + δχ/9 = δΩ/9 + δm4/6 − (13/6)δa = P_3(L) | [GF] B.2; [TA-H] (15); [PV] §4; [MT] paragraph after (3.31) |
| `rem:correction-scale` | Remark S1.26 | scale check d log c/d log λ = 3/2 | [GF] B.2 and script step 5 (K_λ = e^{-λ\|z\|²/2}, (a, m4, χ) ∝ (λ, λ², λ³)); [MT] paragraph after (3.31) |
| `lem:correction-normalization` | Lemma S1.27 | (R14) | [R] §6 (a_0, a_1, a_2, s_1 = 2dz, s_2, b_1, (R14)) |
| `eq:correction-normid` | (S1.42) | (R14) | [R] (R14) |
| `rem:correction-product` | Remark S1.28 | the normalization product must be retained; 1-D example | [R] §6 ("The last product MUST be retained ... −1+2L²z−4L²z²+O(z³)"); exact identity: [PV] §2 ((h_0 + 2h_Lq)/(1+2q) − h_0 − 2(h_L − h_0)q = −4(h_L − h_0)q²/(1+2q)); cf. [TA-G] §3 (8) |
| `lem:correction-deep` | Lemma S1.29 | T_6 < 541q²; (R15); (R16) | [R] §6 |
| `eq:correction-deepjet` | (S1.43) | (R15) | [R] (R15) |
| `eq:correction-rho` | (S1.44) | (R16), ρ_0 | [R] (R16) |
| (proof of Thm S1.4) | proof | assembly and sign | [R] §7 |
| `eq:correction-proof1` | (S1.45) | Taylor split, \|r_t\| ≤ 2000θ_0²Ψ(C_0) | [R] §7 with (R8), (R11) |
| `eq:correction-proof2` | (S1.46) | \|DΨ(C_0)[C(R_jet)]\| < (29/3)ρ_0Ψ(C_0) | [R] §7 with (R9), (R16) |
| `eq:correction-proof3` | (S1.47) | integrated first variation at L = 24 | [R] §7 with (R2) / §5 |
| `eq:correction-Rrat` | (S1.48) | (R17) | [R] (R17) |
| `eq:correction-Rrat-value` | (S1.49) | R_rat = [2000(87480B_0)² + 580·9289B_0]·10^{-250} | [R] (R17); exact value 3228565713716543733736856747955420·10^{-250} equals [R-res] `relative_remainder_rational` (verified) |
| (sign paragraph) | proof | q > 10^{-126}, \|P_d(24)\|10^{-126} > R_rat ⇒ negative | [R] §7 |
| `rem:correction-numerics` | Remark S1.30 | rational enclosure method; decimals of q, P_3(24)q, P_2(24)q; relative precision < 10^{-98} | [R] §7 last paragraph; [R-chk] `q_interval`; [R-res] `correction_intervals`; [MT] sentence after Theorem 2.1 |

### Notation conversions

| Source symbol | Here | Note |
|---|---|---|
| [R] z = exp(−288) | q (and q_L = e^{-L²/2}) | [MT] (3.28) uses q; [PR] uses q |
| [R] u (axis direction) | t ∈ S^{d−1} | [MT] notation |
| [R] t = ∂_u³f | ζ_t | avoids clash with the direction t; [MT] u_t = −ζ_t/12 |
| [R] G, V = Hu, A | g_t, h_t, A_t | [MT] (2.2)–(2.4) for d = 3 (A_t = Q_t(q_t)) |
| [R] n, k (ambient, free dimension) | N, w | n is used for lattice points |
| [R] E (subspace), dx_E | 𝒫, dξ_𝒫 | E is kept for the jet constant (as in [R], [V1]) |
| [R] h(t, A), F_u(C) | ψ, Ψ(C) | [PR] w, F (raw coordinates) |
| [R] e, e_0 | θ, θ_0 | |
| [R] K(n, ν) | 𝒦(N, ν) | |
| [R] S (normalizer) | Z_24 | [MT] (1.1) |
| [R] a_0, a_1, a_2, s_1, s_2, b_1 | α_0, α_1, α_2, β_1, β_2, γ_1 | |
| [R] J_ref, J_24^(1), C_0 | J_0, J^(1)_24, C_0 | [MT] J_0, J^(1)_near |
| [PV] J_∞ | J_0 | |
| [PV]/[GF] (a, m4, χ, Ω), [GF] c | same; c(a, m4, χ) = Φ_3(J^iso(a, m4, χ)) | |
| [V1]/[R] q (derivative order) | k | |
| [P15] p_Vu(0) | p_{h_t}(0) | |

## (b) Corrections applied

- **Correction 7.** The theorem is the 10^{-216} bound of [R] (R1)/(R17), for d = 2 and d = 3 exactly as [R] states it, with
  c_{3,ref} = c_{3,∞} (Lemma S1.3). The 10^{-180} chain of [TA-D] §§3–7, [TA-G] §§2–6 and [TA-H] §4 ((3.26)–(3.29) and
  (3.32)–(3.33) of the main text: jet ball 10^{-110}, sup‖DF‖ < 10^{16}, sup‖D²F‖ < 10^{42}, ‖J_24 − J_0‖_max < 10^{-113},
  ‖R_jet‖_max < 10^{-230}, \|DF(J_0)[R_jet]\| < 10^{-210}, \|R_quad\| < 5·10^{-185}, \|ε_24\| < 10^{-180}) is neither stated nor
  used: the 10^{-216} proof does not need any lemma of that chain. From [TA-H] only §3 (fixed-cone differentiation, cited for
  Lemma S1.9) and §4 (13)–(17) (the first variation, cited for Corollary S1.25) are used; from [TA-D] only §2 (1) (the coefficient
  map Φ, Definition S1.18). The sharper 10^{-218} bound of [PR] (its Theorem R, Lemma 4, Proposition 5, Lemma 6, §§6–7) is not
  stated either; from [PR] only Lemma 1 (section identity and prefactor), Lemma 2 (Haar averaging), Lemma 3 (score identities)
  and §5 fact (iv) (dominating function) are used, where the 10^{-216} proof uses the corresponding steps.
- **Correction 9** (no status words): none in the LaTeX.
- Corrections 1–6 and 8 do not concern this section.

## (c) Added steps

- A1. Lemma S1.3: the identification σ_t² = τ_t²/144 and the prefactor algebra 144^{2/3}/24^{1/3} = 6·2^{2/3}, giving
  Γ(7/6)144^{2/3}/(24^{1/3}√π(2π)³) = B_3. ([PR] Lemma 1 performs the equivalent check only at the reference covariance;
  [TA-C] derives (24)–(25) from the contact density.)
- A2. Lemma S1.7: stated for any positive definite C with vanishing odd–even cross block (so that it applies to C(J;𝔣) for
  jets J that are not a priori kernel jets); the svec constant 2^{m/2}96^{-1/3} is assembled from [R] (R4) (factor 2^{-m/2})
  and [PR] Lemma 1 (96^{-1/3}, raw coordinates).
- A3. ν − N = −2/3 for every d ≥ 2 (text after (S1.10)); [R] and [PR] state the exponent −1/3 for d = 2, 3.
- A4. Lemma S1.8: continuity of t ↦ Ψ(C_t) (dominated convergence and locally continuous frames).
- A5. Lemma S1.10: finiteness and positivity of the angular factor.
- A6. Proposition S1.11: explicit dominating function for differentiation under the integral (expanding [R] §3's sentence
  and [PR] §5 (iv)); statements (a)–(d) for general N, ν and θ < 1 (as in [R] §3, whose (R5)–(R7) are general); explicit d = 2
  constants 𝒦(6, 16/3) = 388/9 and the factor comparison for "the latter bounds the former term by term".
- A7. Lemma S1.12: the d = 2 value (N+ν)/2 = 17/3.
- A8. Lemma S1.14: the partial-matching (Wick) expansion (S1.21) with its inductive proof, and the directional Hermite
  identity via the generating function.
- A9. Lemma S1.15: explicit shell count (2j+1)³ − (2j−1)³ = 24j² + 2 ≤ 27j³; the tail bound written as
  (x^{41}/41!)/(1 − x/42), which is the form used in [R-chk] `exp_bounds`.
- A10. Lemma S1.16: the explicit normalized-difference display.
- A11. Definition S1.18: formal definition of 𝒥_d, the O(d) action, the frame matrix C(J;𝔣), the open set 𝒰_d and Φ_d on
  jets; C_0 defined in every dimension.
- A12. Lemma S1.20: polarization proof that O(d)-invariant even tensors are multiples of S_{2j}; the explicit parametrization
  J^iso(a, m4, χ) = (−aS_2, (m4/3)S_4, −(χ/15)S_6); the explicit constant κ_d.
- A13. Lemma S1.21 (d): differentiability of Φ_d at J_0 along a jet H, via the uniform O(ε²) Taylor bound of
  Proposition S1.11 (justifies equating the angular integral of DΨ with DΦ_d, which [R] §5 and [PV] §1 use).
- A14. Lemma S1.23: the per-direction formula (S1.35) and the three averaged Hermite sums (explicit form of "Apply the
  Hermite identities ... and subtract their normalizer contribution").
- A15. Proposition S1.24: explicit collection of the L², L⁴, L⁶ coefficients ([R]: "Substitution of (R12) gives exactly (R2)").
- A16. Corollary S1.25: det Σ_h = m4³/9 written explicitly ([GF] script: "∝ m4^3").
- A17. Remark S1.26: exact homogeneity c_3[K_λ] = λ^{3/2}c_3[φ] from (S1.31), and K_λ identified as the covariance of f(√λ·).
- A18. Remark S1.28: the one-dimensional example written as the exact identity of [PV] §2 rather than [R] §6's O(z³) expansion
  (same q_L² coefficient −4L²).
- A19. Proof of Theorem S1.4: the exact integer value R_rat = 3228565713716543733736856747955420·10^{-250} (equal to [R-res]).
- A20. Remark S1.30: decimal values of q, P_3(24)q, P_2(24)q (independently computed; they lie in the [R-res] intervals).
- A21. Remark S1.13, last two sentences: the link a = A² between the covariance scaling and the amplitude scaling.

## (d) Issues

1. **Main text §3 "Side-24 correction" vs. this section** ([MT] `s3_proof.tex`, (3.25)–(3.33), lines ~242–285). The main text's
   displayed proof of (2.10) is the 10^{-180} chain ((3.26)–(3.29), (3.32)–(3.33)); the 10^{-216} bound that (2.10) actually states
   appears only as alternative (ii) "derived in \rec{eps}". Per correction 7 this section proves (2.10) through the 10^{-216} chain and
   does not prove (3.26), (3.27), (3.29), (3.32) or (3.33). Suggested fix: in the main text, replace (3.26)–(3.29) and (3.32)–(3.33)
   by a citation of Theorem `thm:correction-main` / Corollary `cor:correction-main` (and, for (3.30)–(3.31), Proposition
   `prop:correction-firstvar` and Corollary `cor:correction-firstvar3`), keeping (3.25), (3.28) [as Definition `def:correction-nearest`],
   (3.30), (3.31); or move the 10^{-180} chain out of the proof. No mathematical error is involved.
2. **Cross-references to update** ([MT] paragraph after (3.31)): "derived from first principles in Supplement B.2" and "the jet
   coordinates a, m4, χ, Ω of the technical appendix (Module H)" should point to Corollary `cor:correction-firstvar3` and Remark
   `rem:correction-scale` of this section.
3. No discrepancy was found in the arithmetic or the mathematics of [R], [PV], [PR] (the parts used), [GF] B.2 or [V1] §§1–3.
   ([R] (R9)'s constant 29/3 and [PR] Lemma 6's 71.1 are different valid bounds; [R]'s θ ≤ 60E (svec coordinates) and [PR]'s
   δ ≤ 10E, ε' = 3δ (raw coordinates) are consistent.)

## (e) Independent checks (run outside the repository, in `scratchpad/w3_checks/`)

- `arith.py` (written from scratch; exact `Fraction` and 200-digit `mpmath`; 304 assertions, all pass): P_3(24) = −620813376/35;
  P_2(24) = −26045568; factored form of P_3 for L = 1..39; (R12)–(R13) composition equals P_d(L) for d = 2..20 at several L;
  d = 3 regroupings and pure-partial forms; factor accounting gives (−1/2, −1/2, 1/9) and the scale value 3/2; (R13) coefficients
  for d = 2..20; isotropic scaling exponent λ^{d/2}; dimensions (6, 2, 10/3, 16/3), (10, 4, 16/3, 28/3); ν − N = −2/3 for d = 2..29;
  𝒦(10, 28/3) = 1012/9, 𝒦(6, 16/3) = 388/9; score moments (−1/3, 4/9); (5/3)^5 = 3125/243 and (4/3)^5(5/4)^{14/3} ≤ (5/3)^5;
  25300000/19683 < 2000; (N+ν)/2 = 29/3, 17/3; matching counts (1+15+45+15 = 76, others ≤ 76, ≤ 15 at 0); B_0 = 14523826191;
  shell counts ≤ 27j³; e^{288/125} > 10 (degree 20); e^{16/7} < 10 (degree 40 + tail); 512q³ < 1/2; (3/2)^9q^5 < 1/2;
  746496 = 729·2·2^9, 746496q⁴ < q²; 1458B_0 = 21175738586478; θ_0 = 87480B_0·10^{-125} < 1/4; 541 + 6·1458 = 9289;
  R_rat = 3228565713716543733736856747955420·10^{-250} ≈ 3.2286·10^{-217} < 10^{-216} (equal to [R-res]); sign margins;
  minors 2/3 and 7/9; (23/3)² > 58; the one-dimensional expansion −1 + 2L²q − 4L²q²; q = 8.37894253381936897136…·10^{-126} ∈
  (10^{-126}, 10^{-125}); **P_3(24)e^{-288} = −1.486217029065827607228927850931666803294…·10^{-118}** (matches the "about
  1.486e−118" magnitude and the main-text digits); P_2(24)e^{-288} = −2.182343175326846742607…·10^{-118}; 10^{-216}/\|P_3(24)q\|
  = 6.73·10^{-99} < 10^{-98}; both [R-res] intervals contain the computed values (widths 6.457·10^{-217} ≈ 2R_rat);
  c_{3,ref} by three closed forms ([V1] (1), [MT] (2.11), B_3·4π·24^{-2/3}D_2/√3) agree to 50 digits and match the main-text decimal;
  (15.2) prefactor × (2π)^{-3}144^{2/3} = B_3; M_{4/3} = 2^{2/3}Γ(7/6)/√π by quadrature; prefactor/M_{4/3} = 96^{-1/3}; the
  parent's κ-integral 144∫κ^{4/3}φ_τ(12κ)dκ = Γ(7/6)τ^{4/3}/(24^{1/3}√π) by quadrature.
- `firstvar_sym.py` (SymPy): (R12) from the Hermite identities and sphere moments, and P_d, for d = 2..11; the d = 3 projection with
  the product-structure weights of [PV] §1 gives the same variations; the isotropic conditional covariance (S1.30) and τ² for
  d = 2, 3, 4; svec Hessian covariance eigenvalues {d+2, 2} for d = 2, 3, 4. All pass.
- `haar_exact.py` (NumPy, deterministic quadrature exact for the degrees involved): the Haar-averaged frame matrix of
  J^(1)_L/q_L equals C(J^iso(δa, δm4, δχ)) with the (R12) values, to ≤ 9·10^{-14}, for d = 2, 3 and L ∈ {0.7, 1, 2, 3}
  (independent check of Lemma S1.21 and Lemma S1.23 and of the covariance sign rule (S1.27)).
- `firstvar_mc.py`, `firstvar_mc_exact.py` (Monte Carlo score estimator on the ψ-weighted reference law): the isotropic partials
  (1 − d/2, d/6 − 1, 1/9) are reproduced for d = 2, 3 within Monte Carlo error; along the exactly averaged matrix the normalized first
  variation is −0.6470 ± 0.0051 vs P_2(1) = −0.6389; 7.1086 ± 0.0066 vs P_2(2) = 7.1111; −1.686 ± 0.035 vs P_3(1) = −1.6952;
  4.34 ± 0.12 vs P_3(2) = 4.3048 (all within 1.6 standard errors). (An earlier run with a sampled 2·10⁵-frame average showed
  d = 2 deviations of 1.7–3.8 reported standard errors; that standard error omitted the frame-sampling noise, and the
  deterministic average removes it.)

## Compile check

`chk_s4/test_s4.tex` = preamble of `supp_main.tex` + `\input{../supp_s4_correction}`; `pdflatex -interaction=nonstopmode` run
three times: no errors, no warnings, no undefined references, no overfull boxes (14 pages). No citations, so no `refs.bib` or
`bibtex` run was needed.

---

# CONC_s5 — concordance for Section S5 (Elder selection)

File: `supp_s5_selection.tex` (section label `sec:selection`). Statements S5.1–S5.35, numbered displays (S5.1)–(S5.100). Numbers are as they will appear when S5 is the fifth section; the standalone check `chk_s5/` prints them as S1.x.

## Sources and their identity

| Key | Document | Identity |
|---|---|---|
| GF | `/home/claude/okc/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` (1 Aug 2026), Parts A.1, A.2, A.4, B.1, C.2 | sha256 `b2269c72…` |
| GF-T1 | `…/scripts/triple_contact_elimination.out.txt` (transcript of the elimination in GF B.1(b)) | sha256 `b7385cfe…` (script `.py` `c5aabd63…`) |
| GF-T2 | `…/scripts/contact_covariance_verification.out.txt` (GF B.1(a)) | sha256 `b1431b34…` |
| GF-T3 | `…/scripts/arithmetic_ledgers.out.txt` (GF A.4 margins) | sha256 `630a8cad…` (script `40ad76a1…`) |
| TA | Project file `04_TECHNICAL_APPENDIX.pdf` (30 Jul 2026), local equation numbers per module | module text sha256 as printed in TA: B `7fdee97a…`, D `a85f8c2c…`, E `da35b526…`, F `9d227108…`, H `8dafdd7b…`, I `b958c21d…`, J `d2649750…` |
| Main | `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/s3_proof.tex` (what is cited) | — |

No Math- repository record is used in this section. TA Module J was consulted only for consistency with GF A.4 (see the note at the end).

## Notation conversions (source symbol → supplement symbol)

- TA B §§5–6 distance `t` (third point to midpoint) → `s` (main text (3.16), TA E §5); `η = r/t` → `η = r/s`.
- GF B.1(b) angular sine `s` (with `c²+s²=1`) → `ρ`; GF third critical point `P`, `det H_P` → `y`, `det H_y`.
- TA I §2 mixed Hessian block `b_p` → `m_p` (clash with the birth height `b`); dummy variable `s` in TA I (3) → `ϑ`; dummy `s` in TA I (17) → `λ`.
- TA I `E_Q`, `P_Q` (pinned, unnormalized) → `E^0`, `P^0`; `P_r^MS` → `Prob^MS_r`.
- Generic kernel `K` (TA E, GF C.2) → `𝒦` (clash with `K = 4096`); planar kernel `K_0(z)=e^{-|z|²/2}`.
- GF A.1.2 sector offset `δ` → `δ_sec`; GF A.4 `δ = 1/(512R³)` → `δ_R` (both clash with the region constant `δ`).
- GF A.2 sector components `C_1, C_2` → `𝒰_1, 𝒰_2` (clash with the region constant `C_1`).
- Fixed polynomials `P(U)` (TA B (39)–(40), TA I (15)–(16)) and `P` in TA H (1) → `Ψ` (clash with the pin vector `P` and the potential `P(X,Y)`).
- Constant `c` in `exp(-c/ρ⁴)` (TA B (36)) → `c_*` (clash with the cosine `c`).
- Lower limit `cr` in TA B (42) → `C_1 r` (the region constant of GF A.2(ii)).
- `index-(d-1)` (GF A.2.1 proof) → `index-two` (d = 3).
- `N_add(C_r)` (TA B (23)) → `N_col`; ball `B(M,ρ_B)` → `B_{ρ_B}(M)`; compact mark sets `B, K` (TA B Thm 7.1) → `I_b, I_κ`; event "1_(15)" (TA I (17)) → `𝒮_r`.
- Interface names (G5, G6, G6′, G7) → descriptions ("collar count", "singular-near witness bound", "fixed-distance count").
- Frame: `(e_1,e_2,e_3) = (t, v, w)` (TA B uses `e_i`, TA E/I use `(t,v,w)`).

## (a) Statements

| No. | Label | Content | Source |
|---|---|---|---|
| S5.1 | `def:selection-palm` | pins, `W_r`, `Z_r`, Palm law | GF A.1 Def. A.1.1 (+ GF "Conventions" for `W_r`); Module F/H(G2) citations → Sections `sec:setting`/`sec:contact`. Main Def. 3.1, (3.3a) |
| S5.2 | `def:selection-mark` | connection level, birth level, elder mark | GF A.1 Def. A.1.2 |
| S5.3 | `lem:selection-mark` | mark correct and Borel (+proof) | GF A.1 Lemma A.1.3 and proof; "dossier §6" → main Section 1 (`S = D_f(M)`). Main Lemma 3.2 |
| S5.4 | `def:selection-prob` | `p_r`, `K^sel = p_r K^all` | GF A.1 Def. A.1.4; last sentence from GF A.1 status paragraph. Main (3.3b) |
| S5.5 | `def:selection-events` | events `A_r, E_r, L_same,r, B_r`, regions (i)–(iii) | GF A.2 "Setup"; Module L §5 citation → `sec:setting` |
| S5.6 | `lem:selection-trichotomy` | `A_r ∩ E_r^c ⊂ B_r ∪ L_same,r` (+proof) | GF A.2 Lemma A.2.1 and proof. Main Lemma 3.4 |
| S5.7 | `cor:selection-reduction` | union bound and Markov | GF A.2 Cor. A.2.2 (first part; the O(r³) part is in S5.35); TA I §5 (23)–(24). Counts `N_col` (no value restriction, as TA I §5 (21)(i)), `N_sing`, `N_far` |
| S5.8 | `lem:selection-endpoint` | exact endpoint factor `r²` (+proof) | TA I §2 (1)–(5) |
| S5.9 | `lem:selection-moments` | degree-nine conditional moments (+proof) | TA H §2.1 (1) and its proof; correction 6 |
| S5.10 | `hyp:selection-jet` | jet ball and standing hypotheses | GF C.2 items (1)–(2); correction 2 |
| S5.11 | `rem:selection-side24` | `K_24` satisfies S5.10; provenance of the `10^40` envelope | GF C.2 (envelope provenance; Fourier support); jet bound `<10^-113` from TA D §7 (21) / TA H §4 (12) (cited as `sec:correction`); correction 2 |
| S5.12 | `lem:selection-pairlaw` | contact-limit pin vector, `det Cov(P)=12`, `Q=-bI+G_2` | TA B §2 (4)–(8); GF B.1(a) (`det Cov(P)=12` from first principles), GF-T2 line 1 |
| S5.13 | `lem:selection-pinstab` | pair-pin stability on the jet ball (+proof) | TA E §3 (3)–(4) |
| S5.14 | `lem:selection-elimination` | exact triple-contact elimination (+proof) | GF B.1(b) (B.1) and GF-T1 lines 1–3; correction 4. See Issue 8 |
| S5.15 | `lem:selection-collarchart` | collar scaled-gradient chart (+proof) | TA B §3 (9)–(20); GF B.1(a), GF-T2 lines 2–3 |
| S5.16 | `lem:selection-collision` | Hessian collision zero (+proof) | TA B §4 Lemma 4.1 (21) and proof |
| S5.17 | `lem:selection-collarfactor` | `det C_col = ρ⁶(X²+ρ²)A_col` (+proof) | TA E §4 Lemma 4.1 (5)–(7) and proof, (8)–(9); GF C.2 (envelope). Main (3.14) |
| S5.18 | `lem:selection-collar` | `E^MS N_col ≤ C_col r³` (+proof) | TA B §4 text after Lemma 4.1, (22)–(23) (LS-DER-024 citation replaced by S5.14, correction 4); TA E §4 last paragraph, §6 (16); TA I §5 (22a) |
| S5.19 | `lem:selection-singchart` | singular-near corrected frame, `s^-7` density (+proof) | TA B §5 (24)–(37); GF B.1(a), GF-T2 lines 4–5 |
| S5.20 | `lem:selection-singfactor` | `det C_sing = ρ¹⁰(3c²+ρ²)A_sing` (+proof) | TA E §5 Lemma 5.1 (10)–(11) and proof, (12)–(14); GF C.2. Main (3.15) |
| S5.21 | `lem:selection-envelope` | anisotropic envelope `[r(r+s²)]²(r²+s³)` (+proof) | TA B §6 (38)–(41), with (38) replaced by S5.14 (correction 4); TA E §6 (15); TA H §5 (24). Main (3.16) |
| S5.22 | `lem:selection-sing` | `E^MS N_sing ≤ C_sing r³` (+proof) | TA B §6 (42)–(44); TA E §6 (16); TA I §5 (22b) |
| S5.23 | `lem:selection-farblock` | joint density block (+proof) | TA H §6 (25)–(30); proof's first sentence from TA F §4 (7)–(8) (cited as `sec:setting`) |
| S5.24 | `lem:selection-far` | three-determinant bound and `E^MS N_far` (+proof) | TA I §2 (6), §3 (7)–(11); TA I §5 (22c). Main (3.17) |
| S5.25 | `lem:selection-slope` | endpoint energy slope (+proof) | GF A.4 Lemma A.4.1 and proof |
| S5.26 | `lem:selection-cone` | outward energy cone (+proof) | GF A.4 Lemma A.4.2 and proof; exact sum from GF-T3 line 1 |
| S5.27 | `lem:selection-strip` | outward energy strip, level bound (+proof) | GF A.4 Lemma A.4.3 and proof; exact sums from GF-T3 lines 2–4 |
| S5.28 | `lem:selection-capture` | inward capture (+proof) | GF A.4 Lemma A.4.4 (i)–(iii) (justifications of (i) moved into the proof); GF-T3 lines 15–16 |
| S5.29 | `lem:selection-robust` | exact-field robustness (+proof) | GF A.4 Lemma A.4.5 ("computation identical to Module J §8" dropped, its content kept) |
| S5.30 | `thm:selection-escape` | energy-adapted capture and escape | GF A.4 Theorem A.4.6. Main Thm 3.6 |
| S5.31 | `lem:selection-shallow` | shallow layer (+proof) | TA I §4 (15)–(18); proof uses TA H §5 G4 (matrix-boundary regularity) |
| S5.32 | `lem:selection-fifth` | fifth-derivative control (+proof) | TA H §5 G3 (23) |
| S5.33 | `lem:selection-failure` | regular-domain and fifth-derivative failures (+proof) | TA I §4 (19) |
| S5.34 | `prop:selection-adjacency` | `P(A_r^c), P(L_same,r) ≤ Cr³` (+proof) | TA I §4 (20) and the text before (14) ("the deterministic matrix capture/escape lemma of LS-DER-051" replaced by S5.30 = GF A.4.6, as GF A.4 states it supersedes) |
| S5.35 | `thm:selection-main` | `1 - p_r ≤ C_sel r³` on compact marks (+proof) | GF A.2 (A.5); TA I §5 (21)–(24); TA B §7 Theorem 7.1. Main Cor. 3.5, (3.13) |

Unnumbered prose: the paragraph after S5.7 is GF A.2.2 (second half); the sentence after S5.8 is TA I §2 (last sentence); the paragraph after S5.14 is GF B.1(b) (chart declaration and "generically O(κ²) as η→0" from the disposition) plus added step A4; the sentences after S5.24 are TA I §3 (last two sentences); "Normalized potential" is GF A.4 (opening, (A.10)–(A.12)); "Transverse normalization" and the paragraph before S5.31 are TA I §4 (12)–(14) and the sentences after (14).

## (a) Numbered displays

| No. | Label | Source |
|---|---|---|
| (S5.1) | `eq:selection-palm` | GF (A.1) |
| (S5.2) | `eq:selection-connection` | GF A.1.2, display of `𝔪_g` |
| (S5.3) | `eq:selection-birth` | GF A.1.2, display of `𝔟_g` |
| (S5.4) | `eq:selection-mark` | GF (A.2) |
| (S5.5) | `eq:selection-Ksel` | GF (A.3); main (3.3b) |
| (S5.6) | `eq:selection-trichotomy` | GF (A.4); TA I (21) |
| (S5.7) | `eq:selection-reduction` | GF A.2.2 first display; TA I (24) |
| (S5.8) | `eq:selection-markov` | GF A.2.2 (inline); TA I (23) |
| (S5.9) | `eq:selection-normalizer` | recalled from `sec:contact`; source TA H §5 (22), TA I (10) |
| (S5.10) | `eq:selection-frame` | TA I (1) |
| (S5.11) | `eq:selection-scaled` | TA I (4) |
| (S5.12) | `eq:selection-softdet` | TA I (5) |
| (S5.13) | `eq:selection-taylor` | TA I (3) |
| (S5.14) | `eq:selection-moment9` | TA H (1) |
| (S5.15) | `eq:selection-jetball` | GF C.2 (1) (= TA D (2)) |
| (S5.16) | `eq:selection-P` | TA B (4) |
| (S5.17) | `eq:selection-Ptarget` | TA B (5) |
| (S5.18) | `eq:selection-detP` | TA B (6); GF B.1(a); GF-T2 line 1 |
| (S5.19) | `eq:selection-Qlaw` | TA B (7)–(8) |
| (S5.20) | `eq:selection-pinfloor` | TA E (3) |
| (S5.21) | `eq:selection-pinpert` | TA E (4) |
| (S5.22) | `eq:selection-B1` | GF (B.1); GF-T1 line 1 (identical) |
| (S5.23) | `eq:selection-B1S` | GF-T1 line 2 (GF B.1(b) second display differs in sign: Issue 8) |
| (S5.24) | `eq:selection-B1y` | GF-T1 line 3 (denominator `64c²ηs²`) |
| (S5.25) | `eq:selection-Ny` | GF-T1 line 3 numerator, regrouped (added step A3; GF B.1(b) third display is partial and differs in one sign: Issue 8) |
| (S5.26) | `eq:selection-collarpt` | TA B (9)–(10) |
| (S5.27) | `eq:selection-Gcol` | TA B (11) |
| (S5.28) | `eq:selection-L1` | TA B (12) |
| (S5.29) | `eq:selection-Lperp` | TA B (13) |
| (S5.30) | `eq:selection-colmeans` | TA B (14) |
| (S5.31) | `eq:selection-colvar` | TA B (15); GF B.1(a) |
| (S5.32) | `eq:selection-Cperp` | TA B (16)–(17); GF B.1(a) |
| (S5.33) | `eq:selection-coldet` | TA B (18); GF B.1(a) |
| (S5.34) | `eq:selection-coldens` | TA B (19) (Issue 9) |
| (S5.35) | `eq:selection-colscale` | TA B (20) |
| (S5.36) | `eq:selection-collision` | TA B (21) |
| (S5.37) | `eq:selection-collarpt2` | TA E (5) |
| (S5.38) | `eq:selection-colgrad` | TA E (6) |
| (S5.39) | `eq:selection-colfactor` | TA E (7); main (3.14) |
| (S5.40) | `eq:selection-Acol0` | TA E (8) |
| (S5.41) | `eq:selection-Acol` | TA E (9); GF C.2 |
| (S5.42) | `eq:selection-colcount` | TA B (23); TA E (16) first line; TA I (22a) |
| (S5.43) | `eq:selection-colprod` | TA B (22) |
| (S5.44) | `eq:selection-singpt` | TA B (24); TA E §5 |
| (S5.45) | `eq:selection-window` | TA B (25) |
| (S5.46) | `eq:selection-V0` | TA B (26) |
| (S5.47) | `eq:selection-V123` | TA B (27) |
| (S5.48) | `eq:selection-L0` | TA B (28) |
| (S5.49) | `eq:selection-L1s` | TA B (29) |
| (S5.50) | `eq:selection-L23` | TA B (30) |
| (S5.51) | `eq:selection-C01` | TA B (31); GF B.1(a) |
| (S5.52) | `eq:selection-det01` | TA B (32); GF B.1(a) |
| (S5.53) | `eq:selection-det4` | TA B (33); GF B.1(a) |
| (S5.54) | `eq:selection-det4sphere` | TA B (34) |
| (S5.55) | `eq:selection-mismatch` | TA B (35); TA E (14) |
| (S5.56) | `eq:selection-pcorr` | TA B (36) |
| (S5.57) | `eq:selection-singdens` | TA B (37) |
| (S5.58) | `eq:selection-singfactor` | TA E (10); main (3.15) |
| (S5.59) | `eq:selection-Asing0` | TA E (12) |
| (S5.60) | `eq:selection-Asing` | TA E (13); GF C.2 |
| (S5.61) | `eq:selection-singlead` | TA E (11) |
| (S5.62) | `eq:selection-endenv` | TA B (39) |
| (S5.63) | `eq:selection-thirdenv` | TA B (40) |
| (S5.64) | `eq:selection-envelope` | TA B (41); TA E (15); TA H (24); main (3.16) |
| (S5.65) | `eq:selection-singcount` | TA B (44); TA E (16) second line; TA I (22b) |
| (S5.66) | `eq:selection-singint` | TA B (42) (the first of the two displays numbered (42) in TA B) |
| (S5.67) | `eq:selection-sixterms` | TA B (43) and the bounds after it |
| (S5.68) | `eq:selection-farblock` | TA H (25) (notation `𝒜_r` from TA I §1) |
| (S5.69) | `eq:selection-farfloor` | TA H (26) |
| (S5.70) | `eq:selection-farwindow` | TA H (27) |
| (S5.71) | `eq:selection-fartarget` | TA H (28); TA I (7) |
| (S5.72) | `eq:selection-Lr` | TA H (29) (Issue 16) |
| (S5.73) | `eq:selection-fardens` | TA H (30); TA I (8) |
| (S5.74) | `eq:selection-threedet` | TA I (6) |
| (S5.75) | `eq:selection-farcount` | TA I (11); main (3.17) (Issue 6) |
| (S5.76) | `eq:selection-farnum` | TA I (9) |
| (S5.77) | `eq:selection-potential` | GF (A.10); main (3.18a) |
| (S5.78) | `eq:selection-R` | GF A.4 display after (A.10) |
| (S5.79) | `eq:selection-assume` | GF (A.11) |
| (S5.80) | `eq:selection-gradient` | GF (A.12) |
| (S5.81) | `eq:selection-slope` | GF (A.13) |
| (S5.82) | `eq:selection-coneflux` | GF (A.14) |
| (S5.83) | `eq:selection-coneledger` | GF A.4.2 proof, aligned display; sum `8625610753/34359738368` from GF-T3 line 1 |
| (S5.84) | `eq:selection-stripflux` | GF (A.15) |
| (S5.85) | `eq:selection-level` | GF A.4.3 statement; GF-T3 lines 3–4 |
| (S5.86) | `eq:selection-tubeterms` | GF A.4.4(ii), aligned display |
| (S5.87) | `eq:selection-tubeledger` | GF A.4.4(ii), collected-ratio display |
| (S5.88) | `eq:selection-rmat` | GF A.4.5 (`r_mat`) (= TA J (34)) |
| (S5.89) | `eq:selection-Tr` | TA I (12); main (3.18) |
| (S5.90) | `eq:selection-RU` | TA I (13) |
| (S5.91) | `eq:selection-conditions` | TA I (14) (= GF (A.11) with `R = R(U)`, `τ = τ_r`) |
| (S5.92) | `eq:selection-shallow` | TA I (15) |
| (S5.93) | `eq:selection-Wshallow` | TA I (16) |
| (S5.94) | `eq:selection-shallowmass` | TA I (17) |
| (S5.95) | `eq:selection-shallowP` | TA I (18) |
| (S5.96) | `eq:selection-borell` | TA H §5 (23) |
| (S5.97) | `eq:selection-failure` | TA I (19) |
| (S5.98) | `eq:selection-adjacency` | TA I (20) |
| (S5.99) | `eq:selection-main` | GF (A.5); TA I (24); TA B §7 Theorem 7.1 (its display, also numbered (42) in TA B) |
| (S5.100) | `eq:selection-regional` | TA I (22a)–(22c) |

## (b) Corrections applied

- **Correction 2 (TA Module E hypotheses restored, GF C.2).** Hypothesis S5.10 states the jet ball `B = {J : ‖J − J_0‖_max ≤ 10^-110}` and the standing hypotheses on the kernel exactly as GF C.2 (1)–(2). Remark S5.11 records the provenance of the `10^40` envelope as GF C.2 gives it (the coarse coefficient-map bound applied to the analytic dependence of the conditional covariances on `J_6`). Lemmas S5.13, S5.17 and S5.20 and the side-24 parts of S5.18, S5.21 and S5.22 are stated under Hypothesis S5.10 instead of the unstated "(1)"/"(2)" of TA E §§3–6.
- **Correction 4 (no per-Hessian `κ²η²` normalization; use (B.1) and the anisotropic envelope).** The historical normalization `det H = B κ² η²/(64c²s²)` does not appear. TA B (22) ("the universal two-dimensional elimination used in the retained regional part of LS-DER-024 gives an r² factor in each …") is replaced in the proof of S5.18 by the exact elimination S5.14 read in physical units. TA B (38) (`∏ det[H_p|_Π] = κ⁶η⁶R_η`) is replaced in the proof of S5.21 by S5.14, keeping TA B (39)–(44), i.e. the anisotropic envelope `[r(r+s²)]²(r²+s³)` and its six radial terms. The covariance side (S5.18), (S5.31)–(S5.33), (S5.51)–(S5.53) is as verified from first principles in GF B.1(a).
- **Correction 6 (three Hessian determinants give degree nine, two give degree six).** Lemma S5.9 states the moment bound for polynomials of degree at most nine (TA H §2.1 (1)) and records that `det H_M det H_S det H_y` has total degree nine and `det H_M det H_S` degree six; S5.24 uses degree nine (TA I (6)). No degree-six claim for three determinants appears.

## (c) Added steps

- A1 (S5.13 proof): `λ_min ≥ det/λ_max⁷` for a positive definite 8×8 matrix with `λ_max ≤ 120` (row sums), and Weyl's inequality plus analytic dependence of Gaussian regression for the final "Hence".
- A2 (S5.14 statement and proof): the hypothesis `c, ρ ≠ 0` and the uniqueness argument: the 10×10 linear system has determinant `−24cρ³/η⁴` (computed with SymPy in `scratchpad/w4_checks/linsys_check.py`).
- A3 ((S5.25)): `N_y` is a regrouping of the expanded numerator printed in GF-T1 line 3; the equality with the transcript and with a fresh elimination was checked symbolically (`scratchpad/w4_checks/elim_check.py`).
- A4 (paragraph after S5.14): in the witness charts lengths are in units of `r` and values in units of `r³`, so physical two-dimensional Hessian determinants are `r²` times the chart determinants (deduced from the declared chart: pair at `(∓1/2, 0)`, value gap `κ/6` versus `κr³/6`).
- A5 (S5.17 proof): the explicit mean-value step `A_col(𝒦) ≥ A_col(K_0) − 10^40·10^-110`.
- A6 (S5.8): "In particular `W_r ≤ r²|D_M||D_S|`", made explicit from "Type indicators can only decrease absolute values".
- A7 (S5.19 proof): the transformation (S5.46)–(S5.47) is upper triangular with diagonal `s^-3, s^-2, s^-1, s^-1`, which explains the stated `s^-7`.
- A8 (S5.32): `H_5` is defined as the supremum over the torus of the norm of the fifth-derivative field, and `{H_5 ≤ r^-1}` is named the fifth-derivative good event (TA H G3 uses `H_5` and "taking u = r^-1"; GF A.4.5 uses "the fifth-derivative good event"; neither defines them).
- A9 (S5.33): `F_r` is defined as `{rR(U)⁵ > ε_0} ∪ {H_5 > r^-1}` (TA I (19): "the regular-domain and fifth-derivative failures F_r").
- A10 (S5.4): `p_Π` is named as the Gaussian density of `Π_{r,x,t}`.
- A11 (§S5.5 opening): "the half-branches are trajectories of the gradient-ascent flow `(Ẋ,Ẏ) = (F_X,F_Y)`" and "`P(M) = p_0(−1/2) = 0`".
- A12 (S5.30 proof): the assembly sentence (i) ← S5.28, (ii) ← S5.25–S5.27, (iii) ← S5.29.
- A13 (S5.7): the definitions of `N_col`, `N_sing`, `N_far` (the collar count without value restriction, as in TA I §5 (21)(i) and TA B (23)).
- A14 (S5.23 proof): first sentence (distinct derivative coordinates at `x` and at `y` ⇒ linearly independent distributions ⇒ Gram positivity), assembled from TA F §4 (7)–(8).
- A15 (S5.11, first sentence): `K_24` satisfies S5.10, assembled from GF C.2 (Fourier support) and TA D (21) / TA H (12) (jet bound).
- A16 (S5.15 proof): "(S5.33) is the product of (S5.31) and `det C_⊥`"; "`r^-4` is the Jacobian of (S5.27)" (the latter is TA B's "r^-4 from the third-gradient density").
- A17 (S5.21 proof): "Multiplying the three bounds and taking conditional expectations gives (S5.64)" (TA B: "Consequently").
- A18 (S5.34 proof): written out from TA I §4 "Consequently": outside `𝒮_r ∪ F_r`, S5.30(i) gives `A_r` and S5.30(ii) excludes same-maximum self-attachment.
- Editorial: GF A.2.1 Case 2 says "(ii) of (A.2)"; written as "condition (ii) of Lemma S5.3" (the only part (ii) that exists). GF A.4.3's duplicated "`= (4δ²/(9/16))E_*`" is written once. GF A.4.2's "`F_X ≥ ξ − 3ξ/16 − (smaller)`" is written "`F_X ≥ ξ − 3ξ/16`" (the three adverse terms are the only ones).

## (d) Issues (not fixed in the LaTeX unless stated; most severe first)

1. **GF A.1.2 (A.2), Lemma A.1.3 proof → (S5.4), S5.3.** For `u_± = S ± δ v(S)` the straight segment through `S` stays in `{f ≥ h}` with minimum `h`, so `𝔪_f(u_+,u_-) ≥ h` always. The condition `𝔪_f(u_+,u_-) < h` in (A.2) never holds, and `e ≡ 0` as written. The proof's sentence "equality `𝔪 = h` would force a degenerate or shared critical value" is false: equality is exactly the separated case. *Fix:* replace `< h` by `≤ h` (equivalently `= h`) in (A.2) and write in the proof "(i) is `{𝔪 = h}`, since the axial segment `[u_-,u_+]` gives `𝔪 ≥ h`". The main text's Definition 3.1 (verbal description) is correct.
2. **GF A.1.2 birth level → (S5.3).** (a) At `a = h`, `{𝔪_f(u_+,·) ≥ h}` contains `u_-` (through `S`), hence both sector components, so `𝔟_f(u_+,h) = 𝔟_f(u_-,h) = max(b_1,b_2)` and (A.2)'s `min` would select the older maximum. (b) `g(z)1{…}` is `0` off the set, so the supremum is `≥ 0` and is wrong when the birth value is negative. (c) "`{𝔪_g(u,·) ≥ a}` is the closed superlevel component of `u`" is inaccurate at critical levels. *Fix:* `𝔟_g(u,a) = sup{g(z) : z ∈ D, 𝔪_g(u,z) > a}` (supremum of the empty set `= −∞`), the maximum of `g` on the open component of `u` in `{g > a}`; it is Borel as a countable supremum of Borel extended-real functions.
3. **GF A.1.2 sector offset (`δ_sec` in S5.2).** "Fix once and for all a small δ below the injectivity scale of the Morse local normal form" is not possible uniformly: the scale is random and, in the contact regime, at most of order `r` (the axial eigenvalue at `S` is about `κr`, and `M` is a critical point at distance `r`). For `r < δ_sec` the points `u_±` need not lie in the upper sectors. *Fix:* choose `δ_sec = δ_sec(f,S) > 0` Borel-measurably below the normal-form scale, or define `u_±` on the two unstable half-branches at a measurable small height above `h`.
4. **`L_same,r` mismatch (GF A.2 vs TA I §4; S5.5(c), S5.6, S5.34).** GF defines `L_same,r` as "the two upper sectors lie in the same component" (`𝔪 > h`). TA I (20) and S5.34 bound only same-maximum self-attachment (S5.30(ii): the outward half-branch cannot end at `M`). The case "same component, outward half-branch ascends to a different maximum `M'`" is not covered by (20) as derived. It is covered by `B_r`: `M` and `M'` lie in one component of `{f > h}`, so their merge saddle `S''` has `h < f(S'') < f(M)`. *Fix:* add this case to the proof of S5.6, so that `A_r ∩ E_r^c ⊂ B_r ∪ {both half-branches ascend to M}`, and define `L_same,r` as the same-maximum event (or add the case to the proof of S5.34).
5. **GF A.2 region partition (S5.5(d)).** Region (i) uses `dist(y,{M,S})`, (ii) uses `dist(y,x)` and (iii) uses `dist(y,{M,S})`. The regions are not exhaustive: `y = x + C_1 r n` with `n ⊥ t` lies in none, and neither do points with `dist(y,x) ∈ (δ, δ + r/2)` and `dist(y,{M,S}) < δ`. The claim "exactly one" fails, and the last sentence of the proof of S5.6 relies on it. *Fix:* measure all three from `x`: (i) `dist(y,x) ≤ C_1 r` (the collar chart `y = x + r(Xt + ρv)` with `|X|`, `ρ` bounded), (ii) `C_1 r < dist(y,x) ≤ δ`, (iii) `dist(y,x) > δ` (then `dist(y,{M,S}) ≥ δ − r/2`; use `δ/2` in S5.23–S5.24). Also require `C_1 > 1` for `η ≤ 1/C_1 < 1`.
6. **Fixed-distance count (TA I (8)–(9); (S5.76), (S5.75), main (3.17)).** `E^0` (TA I's `E_Q`, with `Z_r = E_Q[W_r]`) is conditional on the pins. The Kac–Rice numerator therefore involves the conditional density `p_𝒜(a_r)/p_{P^corr}(L_r)`, not the joint density `p_𝒜(a_r)`. On compact marks `p_{P^corr}(L_r)` is bounded below, so `E^MS N_far ≤ C r³` holds there, which is all S5.35 needs. The Gaussian factor `exp[−c(b²+κ²)]` in (S5.75) is not established for unbounded marks by this argument. *Fix:* state (S5.75) uniformly on compact positive mark sets, or bound the conditional density separately.
7. **Correction 4 and the singular-near envelope (S5.14, S5.21).** (B.1) treats `q_M, q_S` as free chart parameters. Its `det H_y` contains the κ-free term `−16cρ⁴(q_M² − q_S²)/(64c²ηρ²) = O(1/η)`, and all three chart determinants have a `1/c²` pole. TA B's sentence "the exact two-dimensional determinant is O(r²)" (used for (S5.62)–(S5.63)) therefore needs one more step once (38) is replaced. Symbolic check (this writer, `scratchpad/w4_checks/window_check.py`): imposing the value window `p(y) = −ακ/6` gives `q_M + q_S = η(4c²κ + 4cηκ + η²κ − 8αcηκ + 4q_Mρ²)/(2ρ²(2c+η))`, and the three chart determinants acquire the common denominator `16ρ²(2c+η)²`, so they are `O(1)` on charts where `ρ` and `2c+η` are bounded away from 0 (for example `|c| ≥ c_0 > η_0/2`). Near `c = 0`, which contains the stratum `2c+η = 0` for small `η`, a separate argument using the boundedness of the third derivatives is needed. *Fix:* add this step, or derive (S5.62)–(S5.63) directly from the Taylor structure of the plane/hard decomposition, which does not need (B.1). Note that even `|det H_y| ≤ C(rs + s³)Ψ` would still give `O(r³)` (terms `r³s⁻⁴, r²s⁻², r, s²` after the prefactor `r³`).
8. **GF B.1(b) display slips (deviation from the GF markdown).** The GF markdown's second display (`−det H_S`) has `+4κs²[…]`. The exact elimination (GF-T1 line 2, which GF says is printed verbatim, and an independent recomputation) has `−4κs²[(4c²−η²)q_M + (12c²+η²)q_S]`. The GF description "the M↔S mirror … q_M, q_S interchanged" omits the sign reversal. The markdown's `det H_P` shows the κ-free term `+16c(q_M²−q_S²)s⁴`; the transcript has `−16c(q_M²−q_S²)s⁴ + 4η(q_M−q_S)²s⁴`. GF's "κ² coefficient `(4c²∓η²)²/(64c²s²)`" should read `(4c²−η²)²` for both endpoints (the script asserts this for both). (S5.23) and (S5.24)–(S5.25) display the transcript forms, which agree with main-text Remark 3.8. After integration, Remark 3.8 (which corrects "Supplement (B.1)") can be simplified. If strict transcription of the GF markdown is preferred, revert (S5.23) and (S5.25) and keep the Issue.
9. **TA B (19) → (S5.34).** The prefactor `(2π)^-3` should be `(2π)^-3/2` for a three-dimensional Gaussian density at 0 with determinant (S5.33). The exponent matches `−½ μᵀC⁻¹μ`. This is a constant only and has no effect on any conclusion.
10. **GF A.4.2 proof (S5.26).** "`|rQ_X| ≤ rR ≤ ε_0R⁻⁴ ≤ ξ/16`" is justified only by `rR < 3δ_R/16`, which does not give `≤ ξ/16` for small `ξ`. *Fix:* `∇Q(S) = 0` and the `C²` bound give `|Q_X| ≤ R(ξ + |Y|) ≤ R(1 + 1/K)ξ` on `𝒞_E`, so `|rQ_X| ≤ 2rRξ ≤ ξ/16`. The proof already uses this device for `∇_Y Q`.
11. **GF A.4.4 (S5.28).**
    - (a) Part (i) says "stable axial eigenvector … applied to the stable eigenvalue". The inward half-branch is the other half of the one-dimensional unstable manifold, tangent to the positive eigenvector of S5.25. Its slope bound `|v_Y|/|v_X| ≤ √(8/3)R/τ < m` follows from (S5.81) and `|v_Y|² ≤ E(v_Y)/τ`; "stable" should read "unstable (positive)".
    - (b) The lateral flux and `F_X < 0` are proved only where `¼ − X² ≥ δ_0`. On the initial arc near `S`, containment rests on the tangent slope alone. Using `∇Q(S) = 0` makes the `r`-terms scale with `¼ − X²` and removes the restriction near `S`; alternatively, add a local unstable-manifold estimate.
    - (c) In (iii), at `X = −½ + δ_0` one has `¼ − X² = δ_0 − δ_0² < δ_0`, so "holds with room to spare" is false. Take the section at `X = −½ + 2δ_0`, which still lies inside `B_{ρ_B}(M)`.
    - (d) `V̇ ≤ −2min(3/8, τ/2)V + O(RV^{3/2})` uses third derivatives of `P`. `R` controls these except for `rQ`, where only `‖Q‖_{C²}` is assumed. Replace the `rQ` part of the remainder by `r|∇Q(z) − D²Q(M)(z − M)| ≤ 2rR|z − M|`.
12. **GF A.4.3 (S5.27).** The "remaining adverse level terms" `21/(32K) + 1/K² + 3ε_0 + 3/8192` and the strip bound `F_X ≥ 3δ_R/4` are stated without derivation, although GF calls A.4 self-contained. Both check out: for example, at `X = 5/4`, `|½(X²−¼)a·Y| ≤ 63/(256K)`; `F_X ≥ 3δ_R/4` holds. The term `3/8192` (exact-field path reserve) is not tied to `η_R` in the text. Suggest displaying these.
13. **Envelope provenance (GF C.2; S5.11).** `‖dA_col‖, ‖dA_sing‖ ≤ 10^40` is attributed to "the coarse coefficient-map bound … applied to the analytic dependence of conditional covariances on `J_6`". The displayed ledger (TA D §§4–6) bounds `D²Φ` for the coefficient functional, not `dA_col`, `dA_sing`, which involve the radial blow-up. Since the ball radius is `10^-110`, any explicit finite uniform bound suffices. Suggest displaying one.
14. **Morse property under the Palm law (S5.6–S5.7).** S5.6 holds on `Ω_Mo`. S5.7 applies it under `Prob^MS_r`, which needs `Prob^MS_r(Ω_Mo) = 1`; no source states this. *Suggest:* note that it holds for a.e. `(b,κ)` by disintegration (enough for the intensities), or prove it for every pin value by a Bulinskaya-type argument for the conditioned field.
15. **Minor (S5.2).** The Lipschitz bound in `g` gives continuity in `g`. Joint continuity in `(u,v)` needs the path-concatenation argument with short segments at `u` and `v`. Measurability is unaffected.
16. **Minor ((S5.72), TA H (29)).** `‖L_r‖² ≥ ½(b²+κ²)` depends on the coordinates of the pin target. It holds for the target `(b, 2κ, 0, …)` of (S5.17); in the corrected frame of the contact section (`V⁻_corr → −κ/6`) the constant is `1/36`. Any positive constant suffices for (S5.73).
17. **Minor (S5.14, GF B.1(b)).** "The only angular singularities are … and the collision strata `2c = ±η`". The linear system has determinant `−24cρ³/η⁴`, and the displayed denominators vanish only at `c = 0`, `ρ = 0` (and `η = 0` for `det H_y`). At `2c = ±η` the `κ²` coefficient `(4c²−η²)²` vanishes; this is a zero, not a pole. GF's "removable after the ρ-blow-up" for `c = 0, s = 0` is printed by the script but not asserted. Kept as in GF.
18. **Labels.** Besides `sec:setting` and `sec:contact` (given in the assignment), the LaTeX uses `\ref{sec:correction}` (Remark S5.11, for the side-24 jet bound and the coefficient-map ledger). The label is defined in `supp_main.tex`'s abstract, but it is not among the labels the assignment lists; please confirm.

## Note: consistency of GF A.4 with TA Module J

GF A.4 and TA J agree on the potential (J (1)), `R`, `τ`, `T = −Q_⊥/(κr)`, `K = 4096`, `ε_0 = 1/(1024K)`, the assumptions (J (5)), `δ = 1/(512R³)`, `X_* = 5/4`, `η_R = 1/(8192R³)`, the cone ledger (J (18): same four brackets, `< 1/3`), the strip ledger (J (25): same four, `< 9/10`), the level bracket (J (28)–(30): `9869/16777216 < 1/768`, `P(5/4,Y) > 65/256`), and `r_mat` (J (34)). One difference: J (13) gives the slope constant `7/2`, while GF A.4.1 gives `8/3`; both are `< 4`. J §7 defers the inward tube to an earlier version; GF A.4.4 re-derives it. No inconsistency was found.

## Not included

- The lower bound (3.13a) of the main text (cited to records there), as instructed.
- TA B §7's sentence listing the remaining interfaces for planar Bargmann–Fock (provenance only); its content is covered by S5.12–S5.24 and S5.31–S5.34.

---

# CONC_s6 — concordance for `supp_s6_limit.tex` (`\label{sec:limit}`)

Section: "The near/far decomposition and the limit". Writer W5, 13 October 2026.
Numbers below assume the section is the sixth numbered section (S6) of `supp_main.tex`; the stand-alone check build in `chk_s6/` prints them as S1.x.

## Sources and their identities

| Tag | Source | Identity |
|---|---|---|
| GF | `/home/claude/okc/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` (1 August 2026) | sha256 `b2269c726e768387…`; Part A.1 lines 36–105, Cor. A.2.2 lines 161–169, Part A.3 lines 177–261 |
| TA-A | `04_TECHNICAL_APPENDIX.pdf` (project file, 30 July 2026), Module A | extract SHA-256 `835d9ce4…` |
| TA-C | same, Module C | `bbcbc0c4…` |
| TA-F | same, Module F | `9d227108…` |
| TA-H | same, Module H | `8dafdd7b…` |
| TA-I | same, Module I | `b958c21d…` |
| TA-K | same, Module K | `82e1d2f5…` |
| TA-L | same, Module L | `bcfe2f86…` |
| MT | V3 main text `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/`: `s1_field.tex`, `s2_results.tex` (sha256 `35d04369…`), `s3_proof.tex` (sha256 `1019605d…`) | read-only |
| P | Math- `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (`\cite{parentP}`) | blob `dfed3b8d` (`origin/main`); §14, (14.1) — cross-reference only |

## (a) Statements and numbered displays

### Statements

| Label | No. | Content | Source |
|---|---|---|---|
| `sec:limit` | S6 | section | GF Part A.3; TA-I §6; MT §3.3 (end) and §3.5 |
| `prop:limit-near` | Prop. S6.1 | exact near-density pushforward: (S6.6), (S6.7) | GF Prop. A.3.1, (A.7)–(A.8), proof (lines 184–205); MT Prop. 3.3, (3.12a)–(3.12b) (s3_proof.tex 89–98). Proof ingredients: GF Def. A.1.1, (A.1), Lemma A.1.3, Def. A.1.4, (A.3); TA-L §3 (10), §4 (11), §6 (14); TA-K §2 (2), §3 (9), §4 (13) |
| `prop:limit-far` | Prop. S6.2 | far elder pairs: `sup_{0<ℓ≤ℓ0} ν^far_ϱ(ℓ) < ∞` for every ℓ0, hence O(1) = o(ℓ^{-1/3}) | GF Prop. A.3.2, **bounded part only** of (A.9) and its "Hence" sentence (lines 207–211); proof adapted from GF lines 213–234 (**correction 5**). Same bound: MT Prop. 3.10 (s3_proof.tex 202–208); P (14.1) |
| `thm:limit-interchange` | Thm. S6.3 | limit interchange under (H1)–(H3): `ℓ^{1/3}ν^near_ϱ → c`, `ℓ^{1/3}ν_{3,24} → c`, c of (S6.15); (1+o(1)) form when c > 0 | GF Thm. A.3.3, statement and first sentence of its proof (lines 236–247); MT Thm. 3.9 (s3_proof.tex 194–200); TA-A §6 (18)–(19) |
| `prop:limit-hypotheses` | Prop. S6.4 | (H1)–(H3) hold for the side-24 field; the all-mark majorant (S6.17), integrable including κ↓0; 0 < c_{3,24} < ∞ | GF Thm. A.3.3 proof, verification part (lines 247–256); TA-I §6 (25); TA-H §6 (29)–(31) (method: target injectivity, density envelope, regression moments); TA-A §3 (5)–(6), §4 (10), §5 (13)–(14); TA-I §2 (3)–(5); TA-H §5 G2 (22); GF (A.5); MT §3.5 first paragraph and (3.19) |
| `cor:limit-coefficient` | Cor. S6.5 | the selected and all-typed contact integrals have the same limit, equal to the coefficient (2.9) of MT; hence ν_{3,24}(ℓ) = c_{3,24} ℓ^{-1/3}(1+o(1)) | TA-I §6 (25)–(27); MT §3.5 first paragraph (last sentence). Evaluation of the constant: MT (2.2)–(2.4), (2.8), (2.9), (3.20), (3.21); TA-C §2 (even/odd independence), §4 (17), (19), §5 (21)–(23), §6 (24)–(25); TA-A (14); TA-K §4 (13) |

### Numbered displays

| Label | No. | Display | Source |
|---|---|---|---|
| `eq:limit-jacobian` | (S6.1) | dM dS = dx r² dr dσ₂(t) | TA-L §3 (10), §6 (14); MT (3.3) |
| `eq:limit-K` | (S6.2) | K^all_r = p_Π(z_r)Z_r, K^sel_r = p_r K^all_r, G^all_r = r⁴K^all_r, G^sel_r = r⁴K^sel_r | GF Def. A.1.4 and (A.3) (K^all_r = p_Π(z_r(b,κ)) E⁰[W_r]); GF (A.8) ("G^sel_r = r⁴K^sel_r (dimension three)"); G^all_r = r⁴K^all_r is the all-typed counterpart used in GF Thm. A.3.3 (H3) |
| `eq:limit-pushforward` | (S6.3) | r dr = (6^{2/3}/3) κ^{-2/3} ℓ^{-1/3} dℓ | TA-K §3 (9); TA-A §6 (17); MT (3.12) |
| `eq:limit-Lambda` | (S6.4) | Λ^near_ϱ(A) = 24⁻³ E Σ_{M≠M*} 1{f(M)−f(D_f(M)) ∈ A, dist(M,D_f(M)) < ϱ} | added (explicit form of GF's split "of the expected lifetime measure of Proposition 1.2", line 180); MT (1.9) |
| `eq:limit-split` | (S6.5) | ν_{3,24} = ν^near_ϱ + ν^far_ϱ | GF (A.6) |
| `eq:limit-near-measure` | (S6.6) | ∫φ ν^near_ϱ dℓ = ∫_{S²}∫_ℝ∫_0^∞∫_0^ϱ φ(κr³/6) K^sel_r (r³/6) r² dr dκ db dσ₂ | GF (A.7); MT (3.12a) |
| `eq:limit-near-density` | (S6.7) | ν^near_ϱ(ℓ) = (6^{2/3}/18) ℓ^{-1/3} ∫∫∫_{6ℓ/ϱ³}^∞ κ^{-2/3} G^sel_{r_ℓ(κ)} dκ db dσ₂ | GF (A.8); MT (3.12b) |
| `eq:limit-near-kr` | (S6.8) | marked two-point Kac–Rice integral over {0 < dist(M,S) < ϱ} | added (the GF proof names the ingredients "Marked two-point Kac–Rice … with the bounded Borel mark e" without the display) |
| `eq:limit-far-sup` | (S6.9) | sup_{0<ℓ≤ℓ0} ν^far_ϱ(ℓ) < ∞ | GF (A.9), first clause only (correction 5) |
| `eq:limit-far-version` | (S6.10) | marked Kac–Rice version of ν^far_ϱ (with the mark e) | added (GF starts from the inequality); version as in MT Convention 2.2 |
| `eq:limit-far-alltyped` | (S6.11) | ν^far_ϱ(ℓ) ≤ all-typed ordered pair density at height gap ℓ on 𝒯_ϱ | GF A.3.2 proof, first display (lines 216–217) |
| `eq:limit-far-density` | (S6.12) | p_Ξ(ξ_ℓ(b)) ≤ C_ϱ e^{−c_ϱ b²}, uniformly on 𝒯_ϱ, 0<ℓ≤ℓ0 | GF A.3.2 proof (lines 219–225) |
| `eq:limit-far-moment` | (S6.13) | E[\|det H_x det H_y\| \| Ξ = ξ_ℓ(b)] ≤ C_ϱ(1+(\|b\|+ℓ)⁶) | GF A.3.2 proof display (line 228) |
| `eq:limit-far-final` | (S6.14) | ν^far_ϱ(ℓ) ≤ C_ϱ ∫(1+(\|b\|+ℓ0)⁶)e^{−c_ϱ b²} db · 24⁻³\|𝒯_ϱ\| | GF A.3.2 proof last display (line 233) **with the factor "· ℓ ≤ C′_ρ ℓ" removed** and the finite-volume factor written out (correction 5) |
| `eq:limit-const` | (S6.15) | c_{3,24} = (6^{2/3}/18)∫κ^{-2/3}G^all_0 dκ db dσ₂ | GF Thm. A.3.3 (line 243); MT Thm. 3.9; TA-A (19) with s₀ = 1 and J₀ = G^all_0/6 |
| `eq:limit-asymp` | (S6.16) | ℓ^{1/3}ν^near_ϱ(ℓ) → c_{3,24}, ℓ^{1/3}ν_{3,24}(ℓ) → c_{3,24} | GF Thm. A.3.3 conclusion (ν_{3,24}) and MT Thm. 3.9 conclusion (ν^near_ϱ), written as limits (see Added steps A4) |
| `eq:limit-majorant` | (S6.17) | 0 ≤ κ^{-2/3}G^sel_r ≤ κ^{-2/3}G^all_r ≤ C[1+(\|b\|+κ)^N]e^{−c(b²+κ²)}κ^{-2/3}, 0<r<ϱ | TA-I §6 (25); MT (3.19); GF Thm. A.3.3 proof (line 250) |
| `eq:limit-target` | (S6.18) | Y_r = y_r(b,κ) = (b − κr³/12, −κ/6, 0, …, 0) | TA-A §3 (5) (d = 3) |
| `eq:limit-Gall` | (S6.19) | G^all_r = p^corr_r ζ_r | TA-A (6), (10), (13): J_r = (1/6)p_Y z_r and G^all_r = 6J_r (see conversions) |
| `eq:limit-ident` | (S6.20) | lim ∫p_rψ_r = lim ∫ψ_r = c_{3,24} | TA-I §6 (26); MT §3.5 ("the selected and all-typed contact integrals have the same limit") |
| `eq:limit-law` | (S6.21) | ν_{3,24}(ℓ) = c_{3,24}ℓ^{-1/3}(1+o(1)) | TA-I §6 (27); GF Thm. A.3.3 conclusion |

### Unnumbered displays and claims

| Place | Content | Source |
|---|---|---|
| Notation | M = x − (r/2)t, S = x + (r/2)t; bijection onto pairs with 0 < dist < ϱ < 12 | TA-L §2 (1)–(4); MT (3.1), (3.4) |
| Notation | W_r, Z_r = E⁰[W_r] > 0, Palm law, mark e, p_r | GF Def. A.1.1, (A.1), (A.2), Lemma A.1.3, Def. A.1.4; MT Def. 3.1, Lemma 3.2, (3.3a)–(3.3b) |
| Notation | unit multiplicity: M ↦ (M, D_f(M)) bijection | TA-L §4 (11); MT §3.1 |
| Notation | \|dh\| = (r³/6)dκ | TA-K §2 (2); MT (3.6) |
| Prop. S6.1 proof | a.s. identity of the two sums; Palm identity p_Π Z_r p_r = K^sel_r; Tonelli display | added (from TA-L §4 and GF Lemma A.1.3; GF (A.1), (A.3)); GF A.3.1 proof ("K r⁵/6 = (r⁴K)·r/6 = G^sel_r r/6", "(1/6)(6^{2/3}/3) = 6^{2/3}/18", "r < ρ becomes κ > 6ℓ/ρ³") |
| Prop. S6.2 proof | Gram positivity on 𝒯_ϱ; regression sentence; "marked two-point Kac–Rice on a compact separated configuration set" | GF A.3.2 proof (lines 218–228); TA-F Thm. 3.1, Cor. 3.2 |
| Sentence after Prop. S6.2 | the argument bounds all typed far pairs; no rate; same bound is P (14.1) | added (scope sentence, correction 5); P §14 (14.1), blob `dfed3b8d` |
| Thm. S6.3 proof | ℓ^{1/3}ν^near_ϱ as an integral with indicator; DCT | GF A.3.3 proof ("(A.8) + dominated convergence using (H1)–(H2), with the lower κ-limit 6ℓ/ρ³ ↓ 0; (H3) identifies the constant; Proposition A.3.2 removes ν^far") |
| Prop. S6.4 proof | ζ_r → ζ₀ = κ²E[(det Q)²1{Q≺0} \| limiting pins]; Z_r ≥ c_Z r² | TA-A §4 (9)–(10); TA-H §5 G2 (22); GF Def. A.1.1 |
| Prop. S6.4 proof | 1 − p_r ≤ Cr³ on compact positive mark sets | GF (A.5); TA-I §5 (24); MT (3.13) |
| Prop. S6.4 proof | "The corrected pair target is uniformly injective in (b,κ)" and the bound \|y_r\|² ≥ c_ϱ(b²+κ²), c_ϱ = (1/36)/(1+1/36+ϱ⁶/144) | MT §3.5 (sentence); constant added (A5) |
| Prop. S6.4 proof | density envelope (2π)⁻⁴λ₀⁻⁴exp(−\|y\|²/(2λ₁)) | added (A5); method of TA-H §6 (30) |
| Prop. S6.4 proof | det H_M = rD_M, det H_S = rD_S; scaled variables are linear functionals of the third-derivative process on the segment | TA-I §2 (3)–(5); MT (3.9) |
| Prop. S6.4 proof | integrability product bound, including κ↓0; "degenerate (quartic) contacts have κ = 0, a null set of marks" | GF Thm. A.3.3 proof (lines 250–255); TA-I §6 ("integrable on S²×ℝ×(0,∞)"); TA-H §6 last sentences |
| Cor. S6.5 proof | p^corr_0 = p_{(f,h_t)}(b,0)p_{(u_t,g_t)}(−κ/6,0); ζ₀ = κ²E[F(q_t) \| f=b, h_t=0]; the b- and κ-integrals; constant identity (6^{2/3}/18)·72^{7/6}/(2√(2π))·(2π)⁻³ = 3/(2^{4/3}π^{7/2}) | TA-C §2 (6)–(7) and parity sentence, §4 (17), (19), §5 (21)–(23), §6 (24)–(25); MT §3.6 (3.20), (3.21), (2.9); constant identity added (A6), verified with SymPy |

### Notation conversions

| Source symbol | Here | Reason |
|---|---|---|
| GF ρ (cutoff) | ϱ | MT uses ϱ for the cutoff (ρ is the collar variable) |
| GF D_ρ = {dist(x,y) ≥ ρ} | 𝒯_ϱ | MT (1.4) notation 𝒯_δ for the same compact configuration set; avoids a clash with D_f(M), D_M, D_t |
| GF (b, b−ℓ, 0, 0) | ξ_ℓ(b) | shorthand |
| TA-A p_Y, p_0 | p^corr_r, p^corr_0 | density of the corrected pin vector at its target |
| TA-A z_r (= r⁻²Z_r), z₀ | ζ_r, ζ₀ | MT uses z_r(b,κ) for the pin target |
| TA-A Y_r on the target, (5) | y_r(b,κ) | value of the corrected pin vector on the target |
| TA-A J_r = r^{d−2}K_r^{(A)}, with K_r^{(A)} = (r³/6)p_X Z_r | G^all_r/6 | TA-A's K_r includes the gap factor r³/6; GF's (and this section's) K^all_r does not |
| TA-I g_r (rescaled integrand) | ψ_r | MT uses g_t for the gradient pins (2.2), which appear in the same proof |
| GF/MT (H2) majorant (unnamed) | Ḡ | named for the proof |
| TA-H L_r(b,κ) | y_r(b,κ) | the corrected pair target in MT coordinates (3.7)–(3.8) |
| GF Ω_Mo | "the Morse locus" | prose |
| GF (H3) "lim G^sel_0 = lim G^all_0" | "G^all_r → G^all_0 and G^sel_0 = G^all_0" | MT Thm. 3.9 form; see Issue 6 |

## (b) Corrections applied

**Correction 5 (assignment): GF Proposition A.3.2.** GF (A.9) asserts, besides the bounded statement, "in fact ν^far_ρ(ℓ) ≤ C_ρ ℓ for ℓ ≤ ℓ0". That O(ℓ) claim is withdrawn. Its proof's step "Integrating, the ℓ-window contributes its exact width ℓ (the map h ↦ ℓ = b − h is a translation; for the density at fixed ℓ the window is the single slice — the O(ℓ) form follows by integrating the test function against a window of width Δℓ and dividing)" is not valid for a density at fixed ℓ: the density at height gap ℓ is the slice ∫ db of the Kac–Rice integrand, with no window factor, and the majorant used (all typed far pairs, elder mark bounded by one) is the all-typed far density at height gap exactly ℓ, which tends to a positive limit as ℓ → 0; so no bound C ℓ can follow from it. Applied as follows:
- Statement (Prop. S6.2): only the bounded part, `sup_{0<ℓ≤ℓ0} ν^far_ϱ(ℓ) < ∞` for every ℓ0 ∈ (0,∞) (as GF's "for each finite ℓ0"), and `ν^far_ϱ(ℓ) = O(1) = o(ℓ^{-1/3})`.
- Proof: the source's slice bound, uniformly in 0 < ℓ ≤ ℓ0: (S6.11)–(S6.13), then (S6.14) = C_ϱ ∫(1+(|b|+ℓ0)⁶)e^{−c_ϱ b²} db times 24⁻³ times the finite volume |𝒯_ϱ| ≤ 24⁶ of the separated pair domain. The source's phrase "with lifetime window of width ℓ" became "at height gap ℓ"; the window sentence quoted above was removed; the factor "· ℓ ≤ C′_ρ ℓ" was removed from the last display.
- No rate for the far elder density beyond boundedness is stated anywhere in the section. The sentence after the proof says that the argument bounds all typed far pairs and gives boundedness and no rate. Theorem S6.3 uses only the bounded part (as GF Thm. A.3.3 does: "Proposition A.3.2 removes ν^far").
- Consistent with MT Prop. 3.10 and the paragraph after it (s3_proof.tex 202–210) and with P (14.1) (`0 ≤ ν^far_eld ≤ ν^far_cand ≤ C`, a bounded, not vanishing, candidate bound).

No other correction was applied; corrections 1–4 and 6–9 of the plan concern other sections (the degree-six two-Hessian bound used in (S6.13) and (S6.17) is the source's and agrees with correction 6).

## (c) Added steps

All are routine and fully justified in the text; none changes a statement of the sources.

- **A1 (the split).** Explicit measures (S6.4); absolute continuity because both are bounded by Λ_{3,24} (MT Prop. 1.2); the canonical versions (S6.7), (S6.10) and their sum, so that (S6.5) holds for every ℓ > 0 (MT Convention 2.2).
- **A2 (Prop. S6.1 proof).** The almost-sure identity of the two sums (unit multiplicity, TA-L §4, and the mark identification, GF Lemma A.1.3); the explicit marked Kac–Rice integral (S6.8); h < b ⇔ κ > 0 because a pair with e = 1 has f(S) < f(M); the Palm identity p_Π(z_r)E⁰[W_r e] = p_r K^all_r = K^sel_r; Tonelli and the identification of the bracket as the density version obtained from the pair coordinates.
- **A3 (Prop. S6.2 proof).** The marked Kac–Rice version (S6.10) before bounding e ≤ 1; the largest eigenvalue of Cov Ξ is bounded (entries are derivatives of K_24 of order ≤ 2), needed for the Gaussian exponent; |ξ_ℓ(b)|² = b² + (b−ℓ)² ≥ b²/2 (true for every ℓ); |𝒯_ϱ| ≤ 24⁶; ℓ^{1/3}ν^far_ϱ → 0.
- **A4 (Thm. S6.3).** (H2) formalized as one L¹ function Ḡ dominating κ^{-2/3}G^sel_r for all 0 < r < ϱ (the range of r_ℓ(κ) in (S6.7)); (H3) includes existence of G^all_0 = lim G^all_r; c_{3,24} < ∞ from (H1)–(H3); the conclusion written as the limits (S6.16), with the source's (1+o(1)) form stated under c_{3,24} > 0 (equivalent when c > 0; see Issue 1); the near-part conclusion (the form cited by MT Thm. 3.9) made explicit; DCT details (indicator eventually 1, r_ℓ(κ) ↓ 0, r_ℓ(κ) < ϱ on the support, sequences ℓ_n ↓ 0).
- **A5 (Prop. S6.4).** G^all_r = p^corr_r ζ_r from p_Π = r⁻⁶p^corr_r and Z_r = r²ζ_r; uniform eigenvalue bounds λ₀ ≤ Σ_r(t) ≤ λ₁ on (0,ϱ] × S² by positivity at each r (Gram positivity, invertible corrected map), the positive-definite contact limit, and compactness of [0,ϱ] × S² (eigenvalues are frame-independent); the explicit injectivity constant c_ϱ (2×2 form of determinant 1/36; checked symbolically); the Gaussian density envelope; regression bound with means linear in y_r and coefficients bounded for r ≤ ϱ, giving ζ_r ≤ C[1+(|b|+κ)⁶], hence N = 6 (D_M D_S has degree six; D_M's coefficients involve r ≤ ϱ); the integrability factorization 1+(|b|+κ)^N ≤ 2^{N−1}(1+|b|^N)(1+κ^N); c_{3,24} > 0 from p^corr_0 > 0 and ζ₀ ≥ c_Z > 0 (Z_r ≥ c_Z r² on compact positive mark sets).
- **A6 (Cor. S6.5).** The evaluation of (S6.15) as MT (2.9): even/odd factorization of the limiting pin density and of ζ₀, MT (3.20) and (3.21), and the constant identity (6^{2/3}/18)·72^{7/6}/(2√(2π))·(2π)⁻³ = 3/(2^{4/3}π^{7/2}), so that the prefactor is B₃ = 3Γ(7/6)/(2^{4/3}π^{7/2}). TA-I §6 takes this identification as given ("whose limit gives the [Module C] coefficient").
- **A7.** The scope sentence after Prop. S6.2 and its cross-reference to P (14.1).

Symbolic checks (SymPy, all True): the pushforward (S6.3); (1/6)(6^{2/3}/3) = 6^{2/3}/18; K(r³/6)r² = (r⁴K)r/6; MT (3.21); the B₃ constant identity; the injectivity form (det 1/36, λ_min ≥ det/trace); the target (S6.18); V⁻_corr → −∂_t³f/12. Script: `/tmp/claude-0/-home-claude/b41b316f-232c-5cf7-a0ee-80f0de603a6f/scratchpad/w5_checks/check_s6.py`.

## (d) Issues

None affects a statement of this section. Items 1–5 are recorded per rule 2; where an added step covers them, it is listed in (c).

1. **The (1+o(1)) form presupposes c_{3,24} > 0** (GF Thm. A.3.3 line 242; MT Thm. 3.9). Positivity is not among (H1)–(H3); for c = 0 the form is not meaningful. Not an error for the side-24 field. Handled by stating the limits (S6.16) and the (1+o(1)) form under c > 0, and by proving 0 < c_{3,24} < ∞ in Prop. S6.4 (A5). Suggested main-text fix (optional): in MT Thm. 3.9 add "(here 0 < c_{3,24} < ∞)".
2. **The r-range of the majorant is not stated** (TA-I §6 (25); GF Thm. A.3.3 proof; MT (3.19)). Dominated convergence in (A.8) at a fixed cutoff ϱ needs the majorant for every 0 < r < ϱ, since r_ℓ(κ) ranges over (0, ϱ), not only for small r. Supplied by the compactness argument of A5. Suggested main-text fix (optional): "(3.19) holds for 0 < r < ϱ".
3. **Citation of TA Module H §6 for the contact majorant** (GF Thm. A.3.3 proof, line 250). The majorant displayed in TA-H §6 is (32), the envelope of the fixed-distance witness expectation E^MS N_far, not the contact-integrand majorant; only its ingredients (29)–(31) carry over. The contact-integrand majorant itself is TA-I §6 (25). This section derives it (Prop. S6.4 proof) and cites the sections, not Module H.
4. **Normalization of TA-H §6 (29)** (relevant to the fixed-distance witness count of `sec:selection`). "‖L_r(b,κ)‖² ≥ (1/2)(b² + κ²)" holds for small r in the contact coordinates with target (b, 2κ) (the ∂_t³f normalization of TA Module B), but not in MT's corrected coordinates (3.7): there |y_r|² = (b − κr³/12)² + κ²/36, which at b = 0 is κ²/36 < κ²/2. Only positivity of the constant is ever used. Suggested fix where (29) is transcribed: "≥ c(b² + κ²) for a constant c > 0", or name the coordinates. This section uses the explicit constant c_ϱ.
5. **GF A.3.2 proof, eigenvalue sentence** (lines 222–225). The parenthetical justification of p_Ξ ≤ C_ρe^{−c_ρb²} cites only the lower eigenvalue bound; the exponent needs the (immediate) upper bound as well. The qualifier "for ℓ bounded" in "‖(b, b−ℓ, 0, 0)‖² ≥ b²/2" is unnecessary (it holds for every ℓ). Supplied in A3; no change of mathematics.
6. **GF Thm. A.3.3 (H3) notation** (line 240): "lim G^sel_0 = lim G^all_0" applies lim to the limits; rendered in MT Thm. 3.9's form, G^sel_0 = G^all_0, with G^all_0 = lim G^all_r made explicit. Notation only.
7. **For the integrator (cross-references).**
   - MT cites "Supplement Proposition A.3.1" (Prop. 3.3), "Supplement Theorem A.3.3" (Thm. 3.9), "the bounded part of Supplement Proposition A.3.2" (Prop. 3.10) and "Supplement A.3" (verification of (H1)–(H3)). Here these are Prop. S6.1, Thm. S6.3, Prop. S6.2 and Prop. S6.4; Cor. S6.5 is MT §3.5's "the selected and all-typed contact integrals have the same limit".
   - MT Prop. 3.10 says "there is ℓ0 > 0"; Prop. S6.2 proves it for every ℓ0 ∈ (0,∞) (as the source's bounded part). Consistent.
   - Cor. S6.5 cites MT (2.2)–(2.4), (2.9), (3.20) and (3.21) for the coefficient evaluation, because no label for the coefficient section was assigned to this writer. Read-only check of `supp_s3_coefficient.tex` (as of 20:00): its `thm:coefficient-main` proves exactly this evaluation, with the same G^all_0 (its `eq:coefficient-G0` equals p^corr_0 ζ₀ here) and the same constant identity; it cites "Theorem 3.9 of the main text" for the limit. The integrator may replace the evaluation paragraph of Cor. S6.5's proof by `Theorem~\ref{thm:coefficient-main}`, and point S3's citation to `Theorem~\ref{thm:limit-interchange}` and `Proposition~\ref{prop:limit-hypotheses}`. Notation: S3 writes Y_t for the limiting corrected pin vector; this section writes Y_r for the finite-r corrected pin vector and (f, u_t, g_t, h_t) for its limit.
   - Read-only check of `supp_s1_setting.tex`: it contains the facts cited here from `sec:setting` (`thm:setting-gram`, `cor:setting-eigenfloor`, `prop:setting-frame`, `prop:setting-jacobian`, `prop:setting-multiplicity`).
   - The section cites `\cite{parentP}` once (P (14.1)).

## Compile check

`chk_s6/test_s6.tex` = preamble of `supp_main.tex` + `\input{../supp_s6_limit}` + `\bibliographystyle{amsplain}\bibliography{refs}`; `refs.bib` copied. pdflatex ×3 with bibtex: 0 errors, 0 overfull or underfull boxes; the citation `parentP` resolves; the only undefined references are the other-section labels `sec:setting` (4×), `sec:contact` (5×) and `sec:selection` (2×). All 27 labels of this section are defined.

---

# Concordance for `supp_s7_rate.tex` (section S7, `\label{sec:rate}`: A bounded remainder)

Writer: W6. Date: 13 October 2026. Statement and equation numbers below are those the section receives as the
seventh numbered section of `supp_main.tex` (S7.n). The compile check in `chk_s7/` shows S1.n, because there the
section is the only numbered one.

## Sources

All Math- paths are read at `origin/main` = `9844f785` (`git -C /home/claude/math- show origin/main:<path>`).

| Key | Source | Path | Blob (8 hex) |
|---|---|---|---|
| LR | Theorem R and its proof (the transcribed record) | `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md` | `247b3ecf` |
| P | Parent (`\cite{parentP}`) | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d` |
| E1 | Parent congruence erratum | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` | `213594d6` |
| E2 | Parent Section 9 replacement v1.1 | `reviews/d1_section9_borel_repair_20260925/REPAIR.md` | `fe9b9ce4` |
| REC | Reconciliation (reading rule, Section 2 scope, Section 5 W1 and embedding radius, Section 7 consumption contract) | `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md` | `75da2597` |
| CAP | Deterministic separating-cylinder theorem (cited only through P) | `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md` | `0633aca3` |
| RV | Anthropic full-depth review of LR | `reviews/d2_remainder_full_depth_claude_20260930/REVIEW.md` | `529c5264` |
| X1 | xAI slice review R1-R4 (R5, R6 imported-open) | main issue 67, comment 5841270276 (read-only via `gh api`) | n/a |
| X2 | xAI delta review R5, R6 | main issue 67, comment 5841782206 (read-only via `gh api`) | n/a |
| MT | V3 main text | `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/`: `s1_field.tex` (1.1); `s2_results.tex` Theorem 2.1, (2.1), (2.2)-(2.9), Convention 2.2; `s3_proof.tex` (3.5), Proposition 3.12 and the paragraph after it (lines 291-301); `s5_discussion.tex` Section 5.1 (line 6) | n/a |

Line numbers "LR l. n" and "P l. n" refer to the lines of the blob as printed by `git show`.

## (a) Statements and displays

| Label | No. | Content | Source |
|---|---|---|---|
| `sec:rate` (intro paragraph) | S7 | scope; three new estimates | LR Section 1 (l. 18-28, mechanism paragraph l. 28); MT Proposition 3.12 and proof |
| `eq:rate-KL` | (S7.1) | covariance `K_L` | LR Section 1 l. 10-12; P Section 1 l. 12-15 |
| `eq:rate-pins` | (S7.2) | pins `f(M)=b`, `f(S)=b-kr^3`, zero gradients; `l=kr^3`; `k=kappa/6` | LR Section 1 l. 14; P Section 1 l. 19-23; conversion `k=kappa/6` as in MT Section 5.1 and MT (3.5) |
| `eq:rate-rows` | (S7.3) | four axial rows; transverse rows (in text) | LR Section 2 l. 32-39; P (3.1) l. 76-87 |
| `eq:rate-U0` | (S7.4) | contact jet `U_0` | LR Section 2 l. 43; P (3.4) l. 99-101 |
| `eq:rate-target` | (S7.5) | targets `v_r`, `v_0` | LR Section 2 l. 41 and l. 43; P (3.3) l. 94 |
| text after (S7.5) | - | `Sigma_r`, `C_r(z)`, `pi_r` | LR Section 2 l. 49 (`Sigma_r`, `C_r`), l. 68 (`pi_r`) |
| `eq:rate-W` | (S7.6) | `W_r`, full normalizer `Z_r`, `E^MS_r`, `p_r` | LR Section 1 l. 14; P Section 1 l. 25-30 |
| `eq:rate-z0` | (S7.7) | `z_0=(6k)^2 E[det(A_0)^2 1{A_0<0}]` | LR Section 4 l. 136-140 ("exactly as in the parent"); P (5.4) l. 165-172 (definition only) |
| `eq:rate-A` | (S7.8) | `A_r=12 pi_r(v_r) Z_r/r^2`, `A_0=12 pi_0(v_0) z_0` | LR Section 4 l. 136-140; P (10.2) l. 322, (10.3) l. 331 |
| frame-invariance sentence | - | functions of `t` | P Section 10 l. 333 (for `A_r`); added step S11 for `pi_0(v_0)`, `z_0` |
| `eq:rate-Mj` | (S7.9) | partial-block norms `M_3`, `M_4` on the cylinder | CAP Section 1 (definition of `M_j`), used in P Section 7 l. 214-218 and LR Section 5 l. 151-154 |
| `eq:rate-G` | (S7.10) | cap event `G_r` | LR Section 5 l. 151-154; P Section 7 l. 216-218 |
| Densities paragraph | - | candidate pairs, finite bars, near/far | LR Section 1 l. 24 (populations); P Theorem C l. 48-52 |
| `lem:rate-spectrum` | Lemma S7.1 | positive spectrum, summability, smooth version, moments; distinct-functional independence | P Section 2 l. 54-67 |
| `eq:rate-an` | (S7.11) | `a_n` and summability | P Section 2 l. 56-61; LR Section 2 l. 47 |
| `lem:rate-frame` | Lemma S7.2 | (a) invertible contact transform, `|det|=12 r^-(d+3)`; (b) uniform eigenvalue bounds for `Cov(U_r)` and for `U_r` extended by `D_y^2 f(M)`; (c) target bounds and `pi_r(v_r) <= C exp[-c(b^2+k^2)]` | (a) P Section 3 l. 71-96 ((3.1), (3.2)); (b) P Section 3 l. 105 and l. 109, P Section 13 l. 388; LR Section 2 l. 49 and Section 5 l. 166; (c) P (13.1)-(13.2) l. 390-397 |
| `eq:rate-r0` | (S7.12) | choice of `r_0` | LR Section 2 l. 49 ("one band") and l. 55 ("Choose r_0<=1 inside the torus injectivity scale"); embedding bound `r_0 < L/(4 sqrt 2)` from REC Section 5 l. 163-167 (correction C1) |
| text after (S7.12) | - | cylinder embedded; unique displacement | REC Section 5 l. 163-167 |
| `lem:rate-endpoint` | Lemma S7.3 | endpoint identities and envelope, deterministic | P Section 5 (5.1)-(5.3) l. 141-161; P (6.1)-(6.2) l. 186-208; P Section 13 l. 405-409 (`W_r/r^2 <= C(1+||f||_{C^3})^{2d}`) |
| `eq:rate-51` | (S7.13) | `|alpha_i|`, `||beta_i|| <= M_3/2`, `||A_S-A_M|| <= r M_3` | P (5.1) l. 147-149 |
| `eq:rate-52` | (S7.14) | `|alpha_M+6k|`, `|alpha_S-6k| <= r M_4/2` | P (5.2) l. 155-156 |
| `eq:rate-53` | (S7.15) | block determinant | P (5.3) l. 160-161 |
| `eq:rate-envelope` | (S7.16) | typed-weight envelope (LR (R14), with `h=M_3`) | P (6.2) l. 206-208 divided by `r^2`; LR (R14) l. 173-176 |
| `hyp:rate-generic` | Hypothesis S7.4 | (a) pinned genericity under `E^0` and `E^MS_r`; (b) Borel elder mark `1{d_phi(M)=phi(S)}`; finite bars = pairs with `e=1`; essential class excluded | P Section 8 l. 276-295 (genericity l. 288; Borel event l. 290-295); P Section 14 l. 445 (finite-bar correspondence) |
| `hyp:rate-cap` | Hypothesis S7.5 | separating-cylinder implication: on `G_r`, `S` is the global elder partner | P Section 7 l. 214-219 and Section 8 l. 296 (deterministic cap theorem cited there: CAP Section 1 and Section 5); embedding radius REC Section 5 (correction C1); consumed as "the Section 8 comparison" in REC Section 7 l. 211 |
| `hyp:rate-KR` | Hypothesis S7.6 | (a) Borel-marked two-point Kac-Rice identity; (b) `0<Z_r<infinity`, intensity `r A_r dr db dk dsigma`, selected `r A_r p_r`; (c) near density versions | (a) E2 Sections 9.1-9.3, identity (9.1) l. 57-92 extended to bounded Borel `h` (E2 l. 84-92), applied as in E2 Section 9.4 l. 94-110; (b) P Section 13 l. 415 (`Z_r>0`), P (10.1)-(10.2) l. 312-335, E2 Section 9.4; (c) P (13.5) l. 417-425 with P (11.1) l. 341; LR Section 6 l. 212-215 |
| `eq:rate-KR` | (S7.17) | identity `mu = nu` | E2 Section 9.2 (definitions of `mu`, `nu`) and (9.1) |
| `eq:rate-intensity` | (S7.18) | intensity `r A_r dr db dk dsigma(t)` | P (10.2) l. 320-322; LR Section 6 l. 212 |
| `eq:rate-near-cand` | (S7.19) | near candidate density version | P (13.5) l. 419-422; LR Section 6 l. 214-215 |
| `eq:rate-near-eld` | (S7.20) | near elder density version (factor `p_r`) | P l. 424 ("The selected near density has the additional factor p_r") |
| paragraph after Hypothesis S7.6 | - | how (b), (c) follow; power count `(d-1)+3-(d+3)+2=1` | P Section 10 l. 312-328; P (11.1) l. 341; E2 Section 9.4 |
| `hyp:rate-far` | Hypothesis S7.7 | far densities bounded | P Section 14 l. 435-441; LR Section 7 l. 245 |
| `eq:rate-far` | (S7.21) | `0 <= nu_eld^far <= nu_cand^far <= C`, `0<l<=1` | P (14.1) l. 441 |
| sentence after (S7.21) | - | near + far = all pairs, boundary null | P Section 14 l. 445 |
| `def:rate-coef` | Definition S7.8 | coefficient `c_{d,L}` | P (13.6) l. 426-431; LR Section 1 l. 16 ("exactly the parent's equation (15.2)"), Section 6 l. 217 |
| `eq:rate-c` | (S7.22) | `c_{d,L} = int A_0/(3k^{2/3}) = 4 int k^{-2/3} pi_0 z_0` | P (13.6); LR Section 6 l. 217 |
| `eq:rate-152` | (S7.23) | closed Gaussian-angular form; factor 24 sentence | P (15.1) l. 449-454, (15.2) l. 470-478, l. 480 |
| `thm:rate-bounded` | Theorem S7.9 | bounded remainder | LR Theorem R l. 18-24 |
| `eq:rate-R1` | (S7.24) | the three bounds | LR (R1) l. 20-22 |
| `rem:rate-scope` | Remark S7.10 | relative error; what is not proved; mechanism | LR Section 1 l. 26 and l. 28; P Theorem B (1.2) l. 38-46 (compact windows only) |
| `lem:rate-second` | Lemma S7.11 | second-order contact estimates | LR Section 2 l. 45-53 |
| `eq:rate-rowexp` | (S7.25) | expansions of the four axial rows | fourth row: LR Section 2 l. 45; rows 1-3: added step S1 (RV R1 item 1, l. 40-46) |
| `eq:rate-R2` | (S7.26) | `Sigma_r-Sigma_0`, `Sigma_r^{-1}-Sigma_0^{-1}`, `C_r-C_0` are `O(r^2)` | LR (R2) l. 49-55 |
| `lem:rate-coupling` | Lemma S7.12 | coupling and its moment bounds | LR Section 2 l. 55-66 |
| `eq:rate-R3` | (S7.27) | `F_r`, `F_0` | LR (R3) l. 57-58 |
| `eq:rate-R4` | (S7.28) | `||F_r-F_0||_{L^p(C^q)} <= C r^2 P`, moment bound | LR (R4) l. 60-64 |
| sentence after Lemma S7.12 | - | coupling of laws only | LR l. 66 |
| `lem:rate-density` | Lemma S7.13 | pin density bound and difference | LR Section 2 l. 68-74 |
| `eq:rate-R5` | (S7.29) | (R5) | LR (R5) l. 70-72 |
| `eq:rate-Phi` | (S7.30) | filtered determinant `Phi_j` (LR `F_j`) | LR Section 3 l. 78-82 |
| `lem:rate-filtered` | Lemma S7.14 | Lipschitz bound for `Phi_j` | LR Lemma R3.1 l. 84-88 |
| `eq:rate-R6` | (S7.31) | (R6) | LR (R6) l. 86 |
| `eq:rate-detdiff` | (S7.32) | determinant difference bound | LR l. 88 ("telescoping columns and Hadamard's inequality"); display added (S6) |
| `lem:rate-offdiag` | Lemma S7.15 | quadratic off-diagonal cancellation | LR Lemma R3.2 l. 90-99 |
| `eq:rate-R7` | (S7.33) | (R7) | LR (R7) l. 95-97 |
| `eq:rate-affine` | (S7.34) | `det K_s = alpha det A - s beta^T adj(A) beta` | LR l. 99 |
| paragraph after Lemma S7.15 | - | no inverse, no margin; naive bound loses a square root | LR l. 99 and l. 101 |
| `lem:rate-contact` | Lemma S7.16 | contact error of the normalizer | LR Section 4 l. 105-134 |
| `eq:rate-R10` | (S7.35) | (R10), `N=2d` | LR (R10) l. 131-136 |
| `eq:rate-Ki` | (S7.36) | rescaled Hessians `K_i` (LR `D_r` renamed `Delta_r`) | LR Section 4 l. 105-110; agrees with E1 |
| `eq:rate-R8` | (S7.37) | (R8) | LR (R8) l. 112-115 (from P (5.1)-(5.2)) |
| `eq:rate-R9` | (S7.38) | (R9) | LR (R9) l. 117-123 |
| `cor:rate-majorant` | Corollary S7.17 | intensity majorant | LR Section 4 l. 136-147 |
| `eq:rate-Psi` | (S7.39) | `Psi = C P^N exp[-c(b^2+k^2)]` (LR `H`) | LR l. 141 |
| `eq:rate-R11` | (S7.40) | (R11) | LR (R11) l. 143-145, l. 147 |
| sentence after Corollary S7.17 | - | all `b,k`; no division by `k` or `Z_r`; one common `Psi` | LR l. 147, l. 208 ("Increasing the common polynomial degree and constants") |
| `lem:rate-caploss` | Lemma S7.18 | unnormalized cap loss | LR Section 5 l. 151-208 |
| `eq:rate-B` | (S7.41) | `B_r` | LR l. 156-158 |
| `eq:rate-R12` | (S7.42) | `B_r <= delta^3 Psi` for `delta<=1` | LR (R12) l. 160-164 |
| `eq:rate-R13` | (S7.43) | joint density bound | LR (R13) l. 166-169 |
| depth-failure displays (unnumbered) | - | width `D_* delta (T+Lambda)^2`; soft integral | LR l. 178-187 |
| `eq:rate-R15` | (S7.44) | scalar split | LR (R15) l. 189-192 |
| near/far branch and exception displays (unnumbered) | - | Markov bounds | LR l. 194-208 |
| `eq:rate-Psibar` | (S7.45) | integrability of `Psi` | added step S13 (implicit in LR l. 217-222, "Integrating H in b,k") |
| `eq:rate-monomials` | (S7.46) | the two monomials; `delta = l^{1/3} k^{-4/3}` | LR Section 6 l. 217-220; LR Section 7 l. 235 |
| `lem:rate-nearcand` | Lemma S7.19 | near candidate density rate | LR Section 6 l. 212-226 |
| `eq:rate-R16` | (S7.47) | (R16) | LR (R16) l. 224 |
| remark after Lemma S7.19 | - | rate from (R11), not from dominated convergence; cutoff `a` kept | LR l. 226 and l. 243 |
| `lem:rate-loss` | Lemma S7.20 | near pairing loss | LR Section 7 l. 230-243 |
| `eq:rate-R17` | (S7.49) | (R17) | LR (R17) l. 230-233 |
| `eq:rate-R18` | (S7.48) | (R18) | LR (R18) l. 235-241 |
| sentence after Lemma S7.20 | - | no normalized floor for small `k` | LR l. 243 |
| proof of Theorem S7.9 | - | assembly | LR l. 245 |
| `cor:rate-counts` | Corollary S7.21 | counts and moments | LR Section 8 l. 249-259 |
| `eq:rate-R19` | (S7.50) | (R19) | LR (R19) l. 249-252 |
| `eq:rate-R20` | (S7.51) | (R20) | LR (R20) l. 254-257 |
| paragraph after Corollary S7.21 | - | rejected-pair moments for `q>-1`; thresholds `p<1` (sufficient), `p<2/3` | LR l. 259 and l. 261 |
| `rem:rate-side24` | Remark S7.22 | `d=3`, `L=24`: Proposition 3.12 and the rate in (2.1) of MT; `c_{3,24}` of (15.2) = (2.9) of MT; Theorem S7.9 does not give vanishing of the remainder | assignment; MT `s3_proof.tex` Proposition 3.12, its proof ("This is Theorem R ... at d=3 and L=24") and l. 301; MT `s5_discussion.tex` l. 6 (Section 5.1: (3.10) with `k=kappa/6`, leading law an instance of P Theorem C, coefficient P (15.2)); MT (1.1), (2.1), (2.2)-(2.9), Convention 2.2; term-by-term identity added step S17 |

Theorem S7.9 is LR's Theorem R. The name "Theorem R" is not used in the LaTeX (record name).

## (b) Corrections applied

| No. | Correction | Assignment basis | Where |
|---|---|---|---|
| C1 | `r_0 < L/(4 sqrt 2)` added to LR's choice "r_0 <= 1 inside the torus injectivity scale" (LR l. 55), so that the separating-cylinder implication is used only where the cap cylinder `[-2r,2r] x B(0,2r)` is embedded. Hypothesis S7.5 is stated for `0<r<=r_0`. | Assignment: read the parent with REC Section 5, embedding radius `r < L/(4 sqrt 2)` (REC reading rule l. 25-32: "Read P Section 1's 'sufficiently small r' as including the explicit embedding bound"). RV Remark 1 (l. 172-176) states the same requirement for LR ("taking r_0 < min(1, L/(4 sqrt 2)) costs nothing; a successor should say so"); RV calls it non-blocking. For `L=24` it is implied by `r_0 <= 1`. | (S7.12) and the paragraph after it; Hypothesis S7.5 |
| C2 | Congruence: no change. LR Section 4 (l. 105-110) already uses `K_i = D_r^{-1} H_i D_r^{-1}` with `D_r = diag(sqrt r, I)`, i.e. E1's corrected congruence `diag(r^{-1/2}, I) H_i diag(r^{-1/2}, I)`; RV R3 item 1 confirms. | Assignment: read P with E1. | (S7.36) |
| C3 | Wording W1 (REC Section 5 l. 153-161): not applicable. The section imports no statement of P Section 5 about convergence of the rescaled endpoint Hessians (in probability or in law); LR proves its own quantitative comparison in the coupling, and `z_0` is defined directly by (S7.7). | Assignment: read P with REC Section 5 W1. | none |
| C4 | P Section 9 is read as replaced by E2: Hypothesis S7.6(a) states the identity in E2's form (two finite measures on `Omega x C^2`, equality for every bounded Borel `h` via the functional monotone-class argument), not P's original Section 9 wording. | REC reading rule ("Read P Section 9 as replaced by E2"); REC Section 7 (LR consumes "Section 9 (read as E2)"). | Hypothesis S7.6(a), (S7.17) |
| C5 | Scope of imports: only interfaces REC accepts (REC Section 2 and Section 4 table; LR's consumption per REC Section 7 l. 211: the Section 8 comparison, Section 9 as E2, Section 10, (14.1), (15.2); not Theorem A (1.1)). The further parent facts the proof re-uses (P Sections 2-4, (5.1)-(5.4), (6.1)-(6.2), (7.2), (13.1)-(13.6), cap theorem) are those listed in RV Remark 3 (l. 194-199) and are stated as Lemmas S7.1-S7.3 and in Lemma S7.18's proof. Theorem A, the floor (5.5) and any uniform `C r^3` bound are not used. | Assignment ("at exactly the scope the reconciliation accepts"). | Lemmas S7.1-S7.3, Hypotheses S7.4-S7.7, Definition S7.8 |

Review amendments: X1 and X2 (xAI) state no correction to the text. RV found no defect; its Remark 1 is applied as C1. RV Remark 2 (at fixed marks the contact error is in fact `O(r^2)`) is context and claims nothing new; it is not used. RV Remark 3 is implemented by C5. No author-accepted amendment to LR exists on Math- `origin/main` (LR's blob `247b3ecf` is unchanged since it landed; `PROOF_INDEX.md` l. 26 lists no successor).

## Notation conversions (rule 5)

| LR / P | S7 | Reason |
|---|---|---|
| `u` (axial unit vector) | `t` | main-text direction |
| gap mark `k` with `f(S)=b-kr^3` | kept as `k`, with `k=kappa/6` stated (MT's `kappa`); `l=kr^3=kappa r^3/6` | displays and constants of LR and P kept exactly; MT Section 5.1 uses the same conversion |
| `Q`, `E_Q` | `E^0 = E^0_{r,b,k,R}`; contact law `E^0_{0,b,k,R}` | MT Palm notation |
| `Q^W`, `p` | `E^MS_r`, `p_r` | MT |
| `W`, `Z` | `W_r`, `Z_r` | MT |
| `X = R^d/(LZ^d)`, `Vol(X)` | `T^d_L`, `L^d` | |
| `c` (coefficient) | `c_{d,L}` | separates it from generic constants `c` |
| `A_r(b,k,u)`, `A_0` (intensities) | `\mathcal A_r`, `\mathcal A_0` | LR uses `A_0` also for the contact transverse Hessian |
| `C_r(z)` | `\mathcal C_r(z)` | generic constants `C` |
| `F_j(H)` | `\Phi_j(H)` | LR uses `F`, `F_r`, `F_0` also for fields |
| `H(b,k)` (majorant) | `\Psi(b,k)` | Hessians `H_M`, `H_S`, `H_i` |
| `B_r` | `\mathcal B_r` (the matrix `B=-A_M` keeps its name) | |
| `G_r` | `\mathcal G_r` | |
| `D_r = diag(sqrt r, I)` | `\Delta_r` | clash with `D_*`, `D_t`, `\mathcal D_r`, `D_y^2` |
| `h = M3` | `M_3` | MT uses `h` for the saddle level |
| `M3`, `M4` | `M_3`, `M_4` | |
| constant `K` (absorbing `sqrt m`) | `K_*` | clash with `K_L`, `K_i`, `K_s` |
| `D = 4K^2/3` | `D_* = (4/3) K_*^2` | |
| `E` (constant in the soft integral, undefined in LR) | `E_* = (3/2) K_*` | added step S12 |
| `U = T + Lambda` | `\Upsilon` | `U_r` is the observation vector |
| `K_t`, `t >= 0` (Lemma R3.2) | `K_s`, `s >= 0` | `t` is the direction |
| cap cylinder `D` (CAP, REC) | `\mathcal D_r` | |
| `T_r` (contact transform) | not named; "invertible linear image ... absolute determinant `12 r^{-(d+3)}`" | |
| `t` in (R19)-(R20) | `\varepsilon` | `t` is the direction |
| `N_cand-but-not-eld` | `N_rej` | MT Remark 2.5 calls these rejected pairs |
| `Vol(X)^{-1} E sum` | `\mathcal N_q(\varepsilon)` | |
| `d_f(M)` (maximin level) | `d_\varphi(M)` | |

## (c) Added steps

Each is routine and is justified in the text where it appears.

- S1. Expansions of the first three axial rows (`r^2/8`, `r^2/24`, `r^2/24`; LR gives only the fourth, `r^2/40`) and the explicit fourth-row computation in the proof of Lemma S7.11(a); the derivative orders (seven for the `O(r^4)` remainders, five for the `O(r^2)` bounds). From RV R1 item 1; re-verified symbolically (SymPy) for this section.
- S2. In Lemma S7.11(b): the mode bound `(2 pi |n|/L)^j`, the error `C r^2 (1+|n|)^5`, and the form of cross-covariances as sums over modes (LR l. 47 states the argument in one paragraph).
- S3. In Lemma S7.11(c): `Sigma_r - Sigma_0 = Cov(U_r-U_0, U_r) + Cov(U_0, U_r-U_0)` with Cauchy-Schwarz, and `Sigma_r^{-1} - Sigma_0^{-1} = Sigma_r^{-1}(Sigma_0 - Sigma_r)Sigma_0^{-1}` (RV R1 item 2).
- S4. In Lemma S7.12: the independence argument for the law of `F_r`, the explicit vector `v_r - v_0`, and the decomposition of `F_r - F_0` (RV R1 item 3).
- S5. In Lemma S7.13: the explicit derivative of `log pi_theta(v_theta)` (LR names its three terms; RV R1 item 4) and `|v_theta| <= CP`.
- S6. Proofs of Lemmas S7.14 and S7.15 written out: the sign argument for equal inertia, the singular point on the segment by continuity of eigenvalues, the display (S7.32), and the polynomial-identity justification of (S7.34) for singular `A` (LR gives these steps in condensed form; RV R2).
- S7. In Lemma S7.16: `||A_i - A_0|| <= (r/2)||F_r||_{C^3} + ||F_r - F_0||_{C^2}` and "T a sufficiently large constant multiple of ..." (LR: "For example T can include ..."; RV R3 item 2 notes the constant 3/2).
- S8. In Lemma S7.16: Sylvester's law of inertia giving `W_r = r^2 Phi_d(K_M) Phi_{d-1}(K_S)` (RV R3 item 1); `||adj(A)|| <= ||A||^{m-1}`; the endpoint-error arithmetic; `N = 2d`.
- S9. In Corollary S7.17: `pi_0(v_0) <= C exp[-c(b^2+k^2)]` from positivity of `Sigma_0` and `|v_0|^2 = b^2 + 144 k^2`; `z_0 <= 36 k^2 E||A_0||^{2m}`; `r^2 k^2 <= r(k+r) P`.
- S10. In Lemma S7.18 (residual step): the justification of the uniform `C^4` moments of the residual `g` (inverse joint covariance, cross-covariance bounds), which LR asserts.
- S11. Frame invariance of `pi_0(v_0)` and `z_0` (P Section 10 states it for `A_r`).
- S12. `E_* = (3/2) K_*` (LR's `E` is not defined) and the explicit constants `64 D_*^3/3`, `8 C D_*^2` of the scalar near-branch integral.
- S13. (S7.45): `sup_k int Psi db < infinity` and `int int Psi db dk < infinity`, used for every "integrating `H` in `b,k`" step of LR Sections 6-7.
- S14. In Lemma S7.20: the split of the `k >= eta` integral at `k = 1` (LR: "plus a smaller Gaussian-tail term"); explicit `ell_1^{5/6} < r_0^3` for `a < eta` (LR: "shrink ell_* so that a<eta<1").
- S15. In Lemma S7.19: `int_a^1 k^{-4/3} dk <= 3 a^{-1/3}` and `a^{7/3} = r_0^{-7} l^{7/3} <= C l^{1/3}` written out.
- S16. Corollary S7.21: proof by integration written out; the statement that the expected sum of `l_i^{-p}` over `l_i <= epsilon` is finite exactly for `p < 2/3` is justified from (S7.24) (densities at least `c_{d,L} l^{-1/3}/2` for small `l`); LR l. 261 states the threshold.
- S17. Remark S7.22: term-by-term identification of P (15.2) at `d=3` with MT (2.9): `p_G(0) = (2 pi)^{-3/2} det Sigma_g^{-1/2}`, `p_{V_t}(0) = (2 pi)^{-3/2} det Sigma_h^{-1/2}`, `tau_t^2 = 144 sigma_t^2`, `D_t` identical, and `(2 pi)^{-3} 144^{2/3} Gamma(7/6)/(24^{1/3} sqrt pi) = 3 Gamma(7/6)/(2^{4/3} pi^{7/2}) = B_3`. Verified numerically to 40 digits (mpmath); `dt` in (2.9) is surface measure, checked by recomputing `c_{3,infinity}` from (2.9) with total area `4 pi` against (2.11) (agreement to 40 digits). The identification itself is cited to MT as instructed.
- S18. The embedding argument after (S7.12) (`D_r` inside the ball of radius `2 sqrt 2 r`; injectivity on balls of radius `L/2`), taken from REC Section 5.
- S19. Remark S7.10's sentence "the latter rate is proved in [parentP, Theorem B] only on compact windows" combines LR l. 26 ("nor ... claim that the compact-window O(ell^(2/3)) selection difference holds without cutoffs") with P (1.2).

## (d) Issues

None of these changes the LaTeX beyond C1.

- I1 (reading note, parent wording). P Section 13 l. 415 reads "For every fixed positive r and every b,k>0, Z_r is positive and finite". LR and Hypothesis S7.6(b) need `Z_r > 0` for every `b in R` and `k > 0`. P's argument (positive residual Gaussian density on a product of open maximum and index-(d-1) cones) does not use the sign of `b`, and P's own Section 13 integrates `p_r` over all `b in R`; the phrase is read as "every b and every k>0". Suggested fix in the parent: "for every b in R and k>0".
- I2 (version identification with the main text). Remark S7.22 relies on MT (proof of Proposition 3.12) for the identification of MT's canonical Kac-Rice version of `nu_{3,24}` (Convention 2.2) with the near/far versions of Hypotheses S7.6-S7.7. Both are built from the continuous Gaussian regression kernel in pair coordinates; S7 does not re-derive their equality. Comparing MT (3.12a) with (S7.18) under `kappa = 6k` gives `G^sel_r(b,kappa,t) = A_r(b,kappa/6,t) p_r` and `G^all_r = A_r`, assuming S2/S6 normalize `K^all_r` as in MT (3.12a). Suggested fix: S6 (limit) state this correspondence explicitly, so that (3.12b) of MT and (S7.20) are visibly the same function.
- I3 (minor, source wording). LR (R9), first line, `||A_i - A_0|| <= rT`, holds with LR's suggested `T` only after multiplying `T` by a constant (3/2 suffices); LR l. 123 says "For example T can include ...". Transcribed as "a sufficiently large constant multiple". Not a mathematical error (RV R3 item 2 notes the same).
- I4 (minor, source wording). LR Section 5 uses a constant `E` in `int lambda(lambda + E r U) d lambda` without defining it (as does P (7.3)). Defined here as `E_* = (3/2) K_*` (added step S12).
- I5 (reading note). P (13.5) and (14.1) are stated for P's `r_0` from Section 13 ("a fixed r_0>0 ... on which [Sigma_r's] eigenvalues are bounded above and away from zero. Reduce r_0 below the torus injectivity scale and below one"). S7's `r_0` (S7.12) satisfies these defining conditions, so it is an admissible choice of P's `r_0`; LR uses the parent statements at its own `r_0` in the same way ("as in parent Section 14").
- I6 (record-internal notation). LR reuses symbols for different objects (`T` in Sections 4 and 5; `A_0` for the contact Hessian and the limiting intensity; `H` for Hessians and the majorant; `F` for fields and filtered determinants). Renamed (see Notation conversions); no mathematical effect.

## Compile check

`chk_s7/test_s7.tex` = preamble of `supp_main.tex` + `\input{../supp_s7_rate}` + `\bibliographystyle{amsplain}\bibliography{refs}`; `refs.bib` copied. `pdflatex` x3 with `bibtex`: no errors, no undefined references or citations, no overfull boxes (none above 5pt), 13 pages. The only citation key used is `parentP` (present in `refs.bib`). No new bibliography file was needed.

---

# Concordance for S8 (`supp_s8_remainder.tex`) and S9 (`supp_s9_cusp.tex`)

Writer W7, 13 October 2026. Part II of the SIDE24 V3 supplement: the first two links of Theorem 2.3 of the main text.
Supplement numbers below are those printed in the compile check `chk_s8s9/` (Part I replaced by seven stub
sections, so S8 and S9 carry their final numbers).

## Sources (Math- `origin/main` = `9844f785`, which contains the main text's pinned commit `b68d88b3`; read-only)

| Tag | Path | Blob (8 hex) | Role |
|---|---|---|---|
| [R+] | `frontiers/remainder_vanishing_20260930/PROOF.md` | `441152df` | Lemma E, Theorem R+ (S8.2–S8.3) |
| [R+-E] | `frontiers/remainder_vanishing_20260930/ERRATUM_20261004.md` | `0ffb1031` | erratum 1 of [R+] (landed with Math- PR 270, merge `ced1fca9`) |
| [R+-RM] | `frontiers/remainder_vanishing_20260930/README.md` | `4014de40` | pointer to [R+-E] |
| [R++] | `frontiers/remainder_rate_20260930/PROOF.md` | `abfb98ae` | Theorem R++, Lemmas B, F′, W (S8.4) |
| [R++-RM] | `frontiers/remainder_rate_20260930/README.md` | `4758c914` | summary (no erratum exists for this folder) |
| [R] | `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md` | `247b3ecf` | Theorem R inputs quoted in Prop. S8.2 (proved in S7, `sec:rate`) |
| [Z] | `frontiers/c7_zero_gap_limit_20260929/PROOF.md` | `5b6328ea` | Theorem Z inputs quoted in Prop. S8.4 (main-text record [R7]) |
| [K] | `frontiers/c7_total_bounded_20260929/PROOF.md` | `28748b08` | (K1), (K2), §4 `1−e ≤ 1_{G_r^c}` quoted in Prop. S8.3 (main-text record [R7]) |
| [K-E] | `frontiers/c7_total_bounded_20260929/ERRATUM_NORMALIZED_FORM.md` | `b6764ce6` | checked; concerns only [K]'s normalized remark, not used here |
| [P] | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d` | `\cite{parentP}`: §§2, 4, 7–10, (10.2), (13.6), §14 (14.1), §15 (15.2) |
| [182] | `frontiers/fixed_r_inverse_lifetime_20260930/PROOF.md` | `0d401877` | §3 (8)–(9), the barrier; re-proved in [R++] §1 and here (Lemma S8.10) |
| [F-E] | `frontiers/far_elder_rate_20260930/ERRATUM_20261004.md` | `93afac3b` | checked (PR 270); provenance only, no effect |
| [CU] | `frontiers/cusp_second_order_20261001/PROOF.md` | `f6df5a73` | Theorems CU, CU′, CU.1–CU.4, Prop. CU.3, Lemmas CU.5, L (S9.1–S9.6) |
| [CU-RM] | `frontiers/cusp_second_order_20261001/README.md` | `d41d6239` | summary |
| [C1] | `frontiers/cusp_coefficient_certified_20261001/NOTE.md` | `b2866871` | certified enclosure of c1, Lemmas R, Q, S, Corollary S (S9.7) |
| [C1-RM] | `frontiers/cusp_coefficient_certified_20261001/README.md` | `8870c5bf` | summary table |
| [C1-RES] | `frontiers/cusp_coefficient_certified_20261001/RESULTS.json` | `8bbd0b0c` | 16-digit intervals (checked against the NOTE's rounded tables) |
| [454] | Math- PR 454 (OPEN, not merged), `frontiers/cusp_coefficient_certified_20261001/ERRATUM_20261013.md` | `a9c3f95a` (head `d517287c`) | pending erratum 1 of [C1] (XA-219-01..04), read via `gh api` |
| [MT] | `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/s2_results.tex`, `appA_records.tex` | — | Theorem 2.3, (2.12)–(2.14); records [R7], [R9], [R11], [R12] |

## (a) Statements and numbered displays

### S8 `supp_s8_remainder.tex` (`sec:remainder`)

| Label | Printed | Content | Source | Proof |
|---|---|---|---|---|
| `conv:remainder-notation` | Convention S8.1 | field, densities, pins, U_r rows and target, weights, scaled Hessians, maximin and elder mark, kernels, two-site kernels, radii, T | [R] §§1–2, §4 (scaling dictionary); [R+] §0; [R++] §0; [Z] §1 (Z2)–(Z3); [P] §1, §8, (15.2); [R+-E] F1, O3 | — |
| `eq:remainder-sep` | (S8.1) | separation-variable identity for cand/eld/rej | [R] §6; [R+] §2 (2.1); [R++] §1 ("identity behind (B.2)"); [Z] (Z13)–(Z14); [P] (10.2) | — |
| `eq:remainder-lead` | (S8.2) | c ℓ^{-1/3} = ∫∫∫ r^{-2} A_0 | [R+] §2 (second display); [R] §6; [P] (13.6) | substitution step added (A5) |
| `eq:remainder-polar` | (S8.3) | r^{-2}A_r = r^{d-1}Ψ^cand_ℓ(b,ru); k=0 version | [R+] §2 ("r^{−2}A_r(b, ℓ/r³, u) = r^{d−1}Ψ_ℓ^{cand}(b, ru)"); [R++] §3 proof of (W.4); [Z] (Z14) | — |
| `eq:remainder-far` | (S8.4) | far Kac–Rice identities | [R+] §2 "far candidate density"; [R++] §2 (2.1); [CU] §7 (far identity) | — |
| `prop:remainder-R` | Prop. S8.2 | coupling, moments (R4), pin density (R5), block identity, weight bounds (R8)–(R11) | [R] §2 (R2)–(R5), §3 Lemma R3.2, §4 (R8)–(R11); k ≥ 0 extension: [R++] Lemma W ("(R2)–(R5) do not depend on the target"); T: [R+] §1 | imported (pointer to S7) |
| `eq:remainder-R4` | (S8.5) | (R4) | [R] §2 (R4) (also quoted in [R+] §1) | — |
| `eq:remainder-block` | (S8.6) | det K_t = α det A − t βᵀadj(A)β | [R] §3 Lemma R3.2 | — |
| `eq:remainder-R11` | (S8.7) | (R10) first line, (R11) | [R] §4 (R10)–(R11) | — |
| `prop:remainder-K` | Prop. S8.3 | (K1), (K2); 1−e ≤ 1_{G_r^c} | [K] §0 (K1)–(K2), §4; G_r: [R] §5; [P] §§7–8 | imported |
| `eq:remainder-K` | (S8.8) | (K1), (K2) | [K] §0 | — |
| `prop:remainder-Z` | Prop. S8.4 | (i) 0<B<∞, ρ_rej → B; (ii) coupling (Z10), a.s. nonsingular Hessians, Ψ^cand_ℓ → Ψ_0; (iii) far kernel bound | [Z] §1 (Z3)–(Z4), §2 (Z10), §5 remark; [R+] §2; [P] §14 (14.1) (bound C(1+|b|)^{2d}e^{−cb²}) | imported |
| `lem:remainder-even` | Lemma S8.5 | even contact expansion | [R+] §0 Lemma E (E.1) | condensed (see below) |
| `eq:remainder-E` | (S8.9) | (E.1) | [R+] §0 (E.1) | — |
| `eq:remainder-alpha` | (S8.10) | α_M, α_S to second order | [R+] §1 (1.1) | — |
| `eq:remainder-alphaprod` | (S8.11) | α_Mα_S | [R+] §1 (1.2) | — |
| `eq:remainder-A` | (S8.12) | A_M, A_S | [R+] §1 (1.3) | — |
| `eq:remainder-beta` | (S8.13) | β_i, q_i | [R+] §1 (1.5) | — |
| `eq:remainder-prod` | (S8.14) | product bound | [R+] §1 (1.6) | — |
| `eq:remainder-Y` | (S8.15) | common Y | [R+] §1, display after (1.6) | — |
| `thm:remainder-vanish` | Thm. S8.6 (Theorem R+) | (R+.1)–(R+.3) with the density-version qualifier | [R+] §0 Theorem R+; [R+-E] F1 | full |
| `eq:remainder-Rp1` | (S8.16) | (R+.1) | [R+] §0 (R+.1) | — |
| `eq:remainder-Rp2` | (S8.17) | (R+.2) | [R+] §0 (R+.2) | — |
| `eq:remainder-Rp3` | (S8.18) | (R+.3) | [R+] §0 (R+.3) | — |
| `eq:remainder-diff` | (S8.19) | (2.1) of [R+] | [R+] §2 (2.1) | — |
| (unlabelled scope paragraph after Thm. S8.6) | — | constant terms; not claimed; versions; c1 is a different term | [R+] §0 "What is not claimed"; [R+-E] F1 (versions), F4 (c1) | — |
| (sentence after proof of Thm. S8.6) | — | C in (S8.18) depends on d, L only | [R++] §4 (i) (read off [R+] §2) | — |
| `rem:remainder-signed` | Rem. S8.7 | signed near bound | [R+] §2 "Signed near bound" | full |
| `rem:remainder-where` | Rem. S8.8 | where Theorem R's O(1) goes | [R+] §3 Remark 1 with [R+-E] O1 | condensed (the "O(1) per unit separation" clause dropped; [R++] Remark 2 supersedes it near the diagonal) |
| `thm:remainder-rate` | Thm. S8.9 (Theorem R++) | (0.1)–(0.2) | [R++] §0 Theorem R++ | full |
| `eq:remainder-Rpp` | (S8.20) | (0.1) | [R++] §0 (0.1) | — |
| (paragraph after Thm. S8.9) | — | not claimed; plan of proof | [R++] §0 "What is not claimed" and the paragraph before Lemma B | — |
| `lem:remainder-barrier` | Lemma S8.10 | deterministic barrier | [R++] §1 (1.1)–(1.3) (re-proof of [182] §3 (8)–(9)) | full |
| `eq:remainder-barrier` | (S8.21) | (1.1) of [R++] | [R++] §1 (1.1); [182] §3 (9) | — |
| `lem:remainder-B` | Lemma S8.11 | short elder bars at near separations | [R++] §0 Lemma B, §1 | full |
| `eq:remainder-B1` | (S8.22) | (B.1) | [R++] (B.1) | — |
| `eq:remainder-B2` | (S8.23) | (B.2) | [R++] (B.2) | — |
| `lem:remainder-far` | Lemma S8.12 | far elder density, crude rate | [R++] §0 Lemma F′, §2 | full |
| `eq:remainder-F` | (S8.24) | (F′.1) | [R++] (F′.1) | — |
| `eq:remainder-Kmom` | (S8.25) | (2.2) of [R++] | [R++] §2 (2.2) | — |
| (paragraph after Lemma S8.12) | — | sharper far bound O(ℓ^{2/3}) in main-text record [R9] | [R++] §0 parenthetical after (F′.1); [F-E]; main text §3.5 | — |
| `lem:remainder-W` | Lemma S8.13 | sign-window majorant (W.1)–(W.4) | [R++] §0 Lemma W, §3 | full |
| `eq:remainder-W1` … `eq:remainder-W4` | (S8.26)–(S8.29) | (W.1)–(W.4) | [R++] (W.1)–(W.4) | — |
| `eq:remainder-detK` | (S8.30) | (3.1) of [R++] with R_M, R_S | [R++] §3 (3.1) and the R_M, R_S display | — |
| `eq:remainder-window` | (S8.31) | (3.2) of [R++] | [R++] §3 (3.2) | — |
| (paragraph after Lemma S8.13) | — | window interpretation; why (E.1) is needed | [R++] §3 closing parenthesis | — |
| `eq:remainder-split` | (S8.32) | (4.1) of [R++] | [R++] §4 (4.1) | — |
| (proof of Thm. S8.9) | — | four terms, ρ = ℓ^{1/12}, ℓ_0, eight-term ledger | [R++] §4 | full |
| `rem:remainder-exponent` | Rem. S8.14 | what fixes the exponent; attained by Theorem CU; (0.1) re-derives (Z4) | [R++] §5 Remark 1 (first part, last sentence); [CU] §0 "the exponent question left in #198 Remark 1 is settled" | condensed (formal referee computation replaced by the pointer to Thm. S9.2, which proves it) |

### S9 `supp_s9_cusp.tex` (`sec:cusp`)

| Label | Printed | Content | Source | Proof |
|---|---|---|---|---|
| `conv:cusp-notation` | Convention S9.1 | κ, s, jets, Y, φ, E_0, π_0, cusp kernels | [CU] §0 (0.1), "Cusp variables", "Jets"; §4 (w^•, 𝒜^•) | — |
| `eq:cusp-Y` | (S9.1) | Y, φ | [CU] (0.1) | — |
| `thm:cusp-CU` | Thm. S9.2 (Theorem CU) | (CU.1), (CU.2), parity form, relative form | [CU] §0 Theorem CU; main text (2.12) first two terms, (2.13) | proof in §S9.5 (full) |
| `eq:cusp-CU1` | (S9.2) | (CU.1) | [CU] (CU.1) | — |
| `eq:cusp-c1` | (S9.3) | (CU.2) = main text (2.13) | [CU] (CU.2); [MT] (2.13) | — |
| `eq:cusp-c1parity` | (S9.4) | parity form | [CU] §0 "Equivalently, with the parity factorization of [P] §15" | — |
| `thm:cusp-CUprime` | Thm. S9.3 (Theorem CU′) | (CU′.1), (CU′.2), far O(ℓ) | [CU] §0 and §7 Theorem CU′ | proof full |
| `eq:cusp-CUp1`, `eq:cusp-Icand`, `eq:cusp-CUp2` | (S9.5)–(S9.7) | (CU′.1) with I^cand formula, (CU′.2) | [CU] §7 (CU′.1)–(CU′.2) | — |
| (paragraph after Thm. S9.3) | — | dependency split; not claimed | [CU] §0 "What is not claimed", §7 paragraph after (CU′.2) | — |
| `thm:cusp-field` | Thm. S9.4 | cusp limit field | [CU] §1 Theorem CU.1 | full |
| `eq:cusp-P` | (S9.8) | 𝔓 | [CU] (0.2) | — |
| `eq:cusp-FP` | (S9.9) | (1.1) of [CU] | [CU] (1.1) | — |
| `eq:cusp-axial` | (S9.10) | pinned axial jets | [CU] §1 proof (display "ϑ₂ = …") | — |
| `thm:cusp-model` | Thm. S9.5 | fibre reduction, types, elder window, critical values | [CU] §2 Theorem CU.2 | full; (d) "direct substitution" (source: "algebra (checker C1)"); identities re-verified symbolically by W7 |
| `eq:cusp-g` | (S9.11) | g, g′ | [CU] (2.1) | — |
| `prop:cusp-stable` | Prop. S9.6 | stability of the decision | [CU] §3 Proposition CU.3 | full (two numerical asides omitted, see below) |
| `eq:cusp-decision` | (S9.12) | e = 1{|φ| < 1/3} | [CU] (3.1) | — |
| `thm:cusp-kernels` | Thm. S9.7 | kernel limits at fixed κ | [CU] §4 Theorem CU.4 | full |
| `eq:cusp-kernellim` | (S9.13) | (4.1) of [CU] | [CU] (4.1) | — |
| `lem:cusp-window` | Lemma S9.8 | typed window | [CU] §5 Lemma CU.5 | full (historical v1 note omitted) |
| `eq:cusp-window` | (S9.14) | (5.1) of [CU] | [CU] (5.1) | — |
| §S9.5 (`ssec:cusp-proof`) | — | proof of Thm. S9.2 | [CU] §6 | full |
| `eq:cusp-loss` | (S9.15) | (6.1) of [CU] | [CU] (6.1) | — |
| `eq:cusp-Bsplit` | (S9.16) | decomposition of B_{d,L} | [CU] §7 opening paragraph | — |
| `lem:cusp-lipschitz` | Lemma S9.9 | the gap enters Lipschitz | [CU] §7 Lemma L | full (checker C8 aside omitted) |
| `eq:cusp-L1`, `eq:cusp-L2` | (S9.17), (S9.18) | (7.1), (7.2) of [CU] | [CU] (7.1)–(7.2) | — |
| `eq:cusp-mininteg` | (S9.19) | (7.3) of [CU] | [CU] (7.3) | — |
| `rem:cusp-rejected` | Rem. S9.10 | rejected coefficient | [CU] §7 paragraph after the proof of CU′ | condensed (Gaussian-kernel exploration numbers omitted) |
| `rem:cusp-crossover` | Rem. S9.11 | crossover, 13/27 | [CU] §8 Remark 1 | condensed (referee numerics 0.0106, 0.48148 and the "rejected fraction ≍ (r/k)^3" clause omitted) |
| (paragraph after Prop. S9.6) | — | only window information; deterministic statements | [CU] §3 closing paragraph; §8 Remark 5 | — |
| `lem:cusp-gauss` | Lemma S9.12 | zero-gap contact law of the Gaussian kernel | [C1] §2 R.1 | full |
| `thm:cusp-enclosure` | Thm. S9.13 | enclosures of c1, I^cand, I^cand − c1 (d = 1, 2, 3); c_{d,∞}; ratios; d = 1 closed form | [C1] §0 Theorem table, c_(d,ref) table, "Consequences" (d = 1 closed form); [C1-RES] | computer-assisted; method condensed (see below) |
| `lem:cusp-quad` | Lemma S9.14 | Gauss–Legendre error on an ellipse | [C1] §3 Lemma Q | full |
| `lem:cusp-transfer` | Lemma S9.15 | transfer to the torus, L ≥ 24 | [C1] §3b Lemma S, S.1–S.3, constants table | full |
| `cor:cusp-torus` | Cor. S9.16 | periodized fields; c1^{(24)}/c_{3,24} ∈ [−5.07106215, −5.07106213] | [C1] §0 Corollary S table; c_{d,24} from `coefficients/side24_v1/ENCLOSURE.json` (`57af39a0`) as quoted in [C1] §0; main text Thm. 2.3(iii) | full |
| `rem:cusp-reading` | Rem. S9.17 | how the enclosed ratio is read | [454] XA-219-01 (and XA-219-02/03) — **pending, not merged** | — |

All numbered displays carry a label; the table lists every one.

## Proofs: full or condensed, and what was condensed

- Props. S8.2–S8.4: imported results, stated with a pointer (S7 for Theorem R; main-text record [R7] for Theorems K and Z; `\cite{parentP}` for [P]); no proof in the sources either. Since draft 11, the second bound of S8.3 (Theorem K's (K2)) is proved in Lemma S8.16. Since draft 12, S8.4(i)–(ii) (Theorem Z) are proved in §S8.6.
- Lemma S8.5: condensed. Kept every step (1–5), every expansion coefficient (1/24, 1/40, 1/12, 1/120), the good event G with 𝗎 = C₁rT(1+T/k), all three off-G cases and their bounds. Condensed: the Taylor expansions of the centred rules are stated with their results ("Taylor's formula with integral remainder"), not expanded term by term; the checker references (E1, E2) are dropped; one parenthetical of Step 4 is dropped (Issue I-2).
- Thm. S8.6, Rem. S8.7, Thm. S8.9, Lemmas S8.10–S8.13: full (all estimates, exponents and constants of the source).
- Thm. S9.4–S9.5, Prop. S9.6, Thm. S9.7, Lemmas S9.8–S9.9, proofs of Thms. S9.2–S9.3, Lemmas S9.12, S9.14, S9.15, Cor. S9.16: full. Omitted asides: in Prop. S9.6 Step 5 the "referee computation c_δ ≈ 3.6κδ" and in Case E "numerically g(X_R) ≥ 0.8κ at δ = 1/21"; in Lemma S9.8 the note on the v1 asymmetric identity and the referee's exact verification; in Lemma S9.9 the checker-C8 polynomial-model aside.
- Thm. S9.13 (computer-assisted): "method in words", condensed. Kept: the Mellin identity with Tonelli, the reduction principle for d = 1 (closed form), d = 2 (in words) and d = 3 (the T₃ triple integral with its prefactor, E(v), Φ), the cancellation-free form, Lemma Q, the analyticity certification on complex boxes (identity theorem), node/weight certification, the d = 3 box counts (247 = 76 + 171) and tolerances, the tails, the arithmetic (outward-rounded binary64, series with explicit remainders, no trusted libm), widening 1e−12, and the two floating cross-check values. Omitted: the d = 2 formulas of R.4 (β(ω), H(ω), C₀), the explicit safe-form kernels Rt, It of R.6, the d = 2 panel table, the node count 1,406,081, the Newton/sign-change grid details, the parts table, the self-test and mutants, the provenance; no code transcribed. (The d = 3 prefactor was re-evaluated by W7: P₃ = −0.004703632154880 inside [C1]'s enclosure; P₂ likewise.)

## Notation conversions

| Source | Here | Reason |
|---|---|---|
| direction u (all sources; main text (2.13)) | u, noted as the t of main-text (3.4) | (2.13) of the main text writes u; t clashes with t_u = ∂_u³f(0) of (S9.4) and with the parameter of the block lemma |
| gap mark k with f(S) = b − kr³ | k, noted as κ/6 of main-text (3.5) | keeps all source constants (6k, 12k, 36k²) |
| κ := k/r (cusp variable, [CU]) | κ (S9 only), with an explicit warning | as in main text §5.2 |
| X = R^d/(LZ^d) | T^d_L | X is also the rescaled axial coordinate of 𝔓(X, Ξ) |
| kernel A_r, A_r^{eld}, A_r^{rej}; kernel A_0 = 12π₀(v₀)z₀ | 𝒜_r, 𝒜^eld_r, 𝒜^rej_r, 𝒜_0 | [R+] §0 uses A_0 for both the matrix D_y²F₀(0) and the kernel (Issue I-1) |
| c = c_{d,L} | c_{d,L} | c is also a generic constant |
| [R]'s r_0, [K]'s r_0, [Z]'s r_0 (r_0^{[R]}, r_0^{[K]}, r_0^{[Z]}) | r_R, r_K, r_Z | |
| u := C₁rT(1+T/k) ([R+] Step 4) | 𝗎 | clash with the direction u |
| 𝒜^cand, 𝒜^eld, 𝒜^con ([CU] §4) | 𝒞^cand, 𝒞^eld, 𝒞^con | clash with the kernels 𝒜_r |
| cutoff S ≥ 1 ([CU] §§6–7) | s_* | clash with the saddle S |
| N = ‖f‖_{C⁵} ([CU] §1) | N_f | clash with the exponent N |
| τ := δ/10 ([CU] §3 Step 5) | τ_δ | clash with τ_u and τ = tr(adj(A₀)B) |
| w₀ ([CU] §4 proof) | χ₀ | clash with the cusp weights w^• |
| φ (Gaussian e^{−|x|²/2}), φ_C, F, M = N/2+1, multi-indices a, b ([C1] §3b, R.1) | K_∞, p_C, 𝖥, N̄, 𝖺, 𝖻 | clashes with φ (cusp ratio), F_j, the maximum M, the birth b |
| φ_τ(0) ([P] §15, [C1] R.1) | (2π)^{−1/2}τ^{−1} written out | clash with φ |
| K = π/(2Γ(11/4) sin(7π/8)) ([C1] R.2) | K_{7/4} | clash with the barrier constant K |
| X (perturbation matrix in det(A₀ + X), [R+] Step 2) | V | |

## (b) Corrections applied

Assignment (W7): "apply every erratum that has landed; list them". The landed errata for the S8 sources are those of
[R+-E] (Math- PR 270, merge `ced1fca9`; the only erratum file in `frontiers/remainder_vanishing_20260930`; `git log`
of the folder shows no later change). No erratum exists for `frontiers/remainder_rate_20260930`,
`frontiers/cusp_second_order_20261001`, or (on `main`) `frontiers/cusp_coefficient_certified_20261001`.

1. [R+-E] **F1** (density versions): Theorem S8.6 states the canonical marked Kac–Rice density versions, "the versions for which Proposition S8.4(i) is stated", with ν_cand, ν_eld the unrestricted densities; the scope paragraph adds "no statement about other almost-everywhere-equal versions". Also carried into Convention S8.1(b) and Theorems S8.9, S9.2.
2. [R+-E] **F4** (c₁ notation clash in Remark 4): the scope paragraph after Theorem S8.6 says that no expansion ν_eld = c ℓ^{−1/3} + C₀ + o(1) with C₀ ≠ 0 is asserted and that the c₁ of S9 is the coefficient of ℓ^{1/4}, a different term.
3. [R+-E] **O1**: Remark S8.8 reads "the rejected mass at separations below r_0".
4. [R+-E] **O2**: the domination step in the proof of Theorem S8.6 does not attribute the phrase "from (R11) alone" to Theorem Z; it states only the bound (1 + ℓ/r⁴)H that the first-order estimate gives.
5. [R+-E] **O3**: the strict inequality r_0^* < L/(4√2) is implemented in Convention S8.1(h) ("decreased if necessary"; added step A2).
6. [R+-E] **F2, F3, O4**: status labels, two source-table rows and a branch-base note; no LaTeX effect (no status words or source tables appear in the supplement). Per F3, [R]'s §§2–7 including (R8)–(R13) and the ledgers (R16)–(R18) are consumed (Remark S8.8 uses the latter through "the proof of Theorem R"), and [K]'s radius r_K enters r_0^* (Convention S8.1(h)).
7. PLAN "Known corrections" item 9: no status words in the LaTeX (the main text's §7 carries review status). Items 1–8 concern Part I sources and do not apply to these sections.
8. Checked and not applicable: [K-E] (Math- PR 289) withdraws the unrestricted reading of [K]'s *normalized* remark; S8–S9 use only the unnormalized (K1)–(K2) (Prop. S8.3), and Remark S9.11 uses only the mass scaling r³/k. [F-E] (PR 270) is provenance only.
9. **Pending erratum (not merged): Math- PR 454**, "Math-#219 erratum 1 (XA-219-01..04): the correction-ratio reading is conditional", state OPEN (`merged: false`, head `d517287c`, base `main` `9844f785`), file `ERRATUM_20261013.md` blob `a9c3f95a` (8,430 bytes). It concerns how the c₁ ratio is read: what [C1] encloses is the ratio of the integral (CU.2) to c_{3,24}; its reading as the relative correction (c₁/c)ℓ^{7/12} of (CU.1) is conditional on Theorem CU and on side24_v1's enclosure of c_{3,24} (likewise for I^cand and Math-#218). Per the assignment, the reading is stated in Remark S9.17 (without status words: "uses two inputs that the enclosure itself does not supply: Theorem S9.2 … and the enclosure of c_{3,24}"). [C1] `NOTE.md` on `main` still says "certified" in its §0 bullet; nothing else changes (no number, code or certificate change).

## (c) Added steps

- A1. Convention S8.1(b): at d = 3, L = 24 the coefficient c_{d,L} of [P] (15.2) equals c_{3,24} of main-text (2.9), because both are the leading coefficient of ν_{3,24} (uniqueness of the leading term). The sources use "c = c_{d,L}" and call ν_eld "the manuscript's ν_{3,24}" ([R++] §0) without stating this identification.
- A2. Convention S8.1(h): O3's strict inequality realized by decreasing r_0^* (each of the three radii is existential and the statements hold for smaller radii); for L = 24 nothing changes ([R+-E] O3: 24/(4√2) ≈ 4.24 > r_R).
- A3. (S8.2): the one-line substitution k = ℓ/r³, dk = 3ℓr^{−4}dr from c = ∫∫∫ 𝒜₀/(3k^{2/3}) ([R] §6, [P] (13.6)); [R+] §2 states only the result.
- A4. Theorem S8.6: "the bound holding for every ℓ > 0" — [R+]'s (R+.3) carries no restriction on ℓ and its proof (majorant H(b, ℓ/r³) ≤ Ce^{−cb²} for all k; tail ≤ Cℓ²r₀^{−7}) gives it for all ℓ > 0.
- A5. Lemma S9.8: the radius r₁ is made existential ("there are r₁ > 0, C, N"); [CU] (5.1) uses r₁ without quantifier.
- A6. §S9.5, last sentence: the parity form (S9.4) is derived explicitly from π₀ = p_{(f,V_u)}(b,0)p_G(0)(2π)^{−1/2}τ_u^{−1} and b-integration; [CU] §0 states it "with the parity factorization of [P] §15".
- A7. §S9.7 opening: for d = 1, Δ = 1 (empty A); [C1] R.3 says "A is empty and Y = f4/12" (the closed form uses |Δ|^{1/4} = 1).
- A8. Prop. S8.4(iii): the far kernel bound is stated as C_ρ(1+|b|)^{2d}e^{−c_ρb²} for every ρ ∈ (0, L/2], as [P] §14 proves it on any compact set away from the diagonal ([R+] §2 states "≤ Ce^{−c″b²} uniformly in 0 < ℓ ≤ 1" at its r₀; [R++] §2 uses r₀ ∈ (0, L/2]).

## (d) Issues

- **I-1 (notation clash, [R+] §0, low).** [R+] §0 writes "`A_0 = D_y²F_0(0)`, `z_0 = E[(6k)²(det A_0)²1{A_0 < 0}]`; `A_r = 12π_r(v_r)Z_r/r²`, `A_0 = 12π_0(v_0)z_0`": the symbol A₀ denotes both the transverse contact Hessian and the contact kernel ([R] §4 has the same double use). No mathematical error; resolved here by 𝒜_r, 𝒜₀ for the kernels. Suggested fix in the source: rename the kernel.
- **I-2 (imprecise parenthetical, [R+] §1 Step 4, low, no effect).** The second off-G case says it "also contains the pattern index A_S = m − 1, σ_S < 0, which F_d(K_M) = 0 kills". On {k ≥ C₁rT} that pattern forces λ_max(A₀) > −rT > −𝗎, so it lies in the second *or the third* case ({|λ_max(A₀)| < 𝗎}); the third case's bound |det K_i| ≤ Ck𝗎T^{m−1} covers it regardless of the pattern, so the proof is complete. The parenthetical is omitted from Lemma S8.5's proof. Suggested fix: "this pattern falls in the second or the third case".
- **I-3 (notation clash, [CU] §1/§3, low).** [CU] writes the torus as X and also uses X for the rescaled axial coordinate (e.g. "Let f ∈ C⁵(X) … 𝔉(X, Ξ)"). Resolved here by writing the torus T^d_L. Similar harmless clashes (S as saddle and as cutoff; u as direction and as layer width; A₀; φ) are listed under "Notation conversions".
- **I-4 (main text, observation only).** The main text writes the pair direction as t in §3 and as u in (2.13); S8–S9 use u throughout and say so. No change requested.
- No mathematical error was found in [R+], [R++], [CU] or [C1]. Spot checks by W7: the quintic pin expansions and α_Mα_S (Lemma S8.5, Steps 1–3) by hand; g′, g(X₃), g(X₃)+κ, g″(X₃), the φ = ±1/3 factorizations, g(−3/2), g(2), g(−3), max_{[−3,2]}|g|/κ = 279.69 ≤ 280, the loss integrals (16/7)2^{1/4}, (8/7)6^{1/4}, I^cand/c₁ = 3^{1/4}/2, the rejected integral (8/7)(2^{5/4} − 6^{1/4}) and 13/27 symbolically (sympy); R_M, R_S of (S8.30) by hand; the d = 1 closed form (−0.2276063587753, inside the enclosure); the d = 2, 3 prefactors of [C1] R.4–R.5; the constants of Lemma S9.15 (E = 3.678e−108, ε, η); and the ratio c₁/c_{3,24} from [C1-RES] (−5.0710621417, −5.0710621331).

## Not transcribed (by design)

[R+] §3 Remarks 2 (conditional rate, superseded by Theorem S8.9), 3 (consistency), 4 (manuscript wording); §§4–6 (sources, controls, review slices). [R++] §4 parenthetical (v1 ledger, rate 1/6), §5 Remarks 2–4 (consistency, manuscript, inputs by status), §§6–8. [CU] §8 Remarks 2 (history), 3 (floating exploration of c₁, superseded by Thm. S9.13), 4 (manuscript), 6 (consistency); §§9–11. [C1] §§4–9 except as summarized in the method. The separate record `frontiers/cusp_torus_transfer_20261001` (L ≥ 10 transfer, not among the assigned sources or main-text records [R11]–[R12]) is not used.

## Bibliography

`\cite{parentP}` (refs.bib) and `\cite{CarberyWright}`, whose entry is in `supp_refs_s8s9.bib` (written per COMMON rule
"new reference → supp_refs_<section>.bib"; the integrator must add it to `\bibliography{refs,supp_refs_s8s9}`).
Data: A. Carbery, J. Wright, *Distributional and L^q norm inequalities for polynomials over convex bodies in R^n*,
Math. Res. Lett. 8 (2001), 233–248 (as cited in [CU] §9; issue number and DOI omitted).

## Cross-references used (labels defined by other writers)

`sec:setting` (density convention, finite-jet nondegeneracy), `sec:contact` (marked Kac–Rice intensity),
`sec:selection` (Borel elder mark), `sec:rate` (Theorem R and its coupling, density and weight estimates). Any labels
inside those sections are not referenced; the needed statements are restated in Prop. S8.2 with a pointer.

## Compile check (`chk_s8s9/`)

`chk_s8s9.tex` = preamble of `supp_main.tex` + seven stub sections carrying the Part I labels + `\input` of both files
+ `\bibliography{refs,supp_refs_s8s9}`; `pdflatex` ×3 with `bibtex`: no errors, no undefined references or citations,
no overfull boxes (all were fixed; none remains above 0 pt). 22 pages: S8 ≈ 9 pages, S9 ≈ 12 pages (of which the
value of c₁, §S9.7, ≈ 3 pages).

---

# CONC_s10: concordance for `supp_s10_third.tex`

This file covers `\section{The third-order terms}\label{sec:third}` (Part II, writer W8).

**Mode.** The statements are exact. Proofs are condensed but complete in logic, and every condensed proof is marked in table (a).

**Compilation.** `chk_s10/` gives 17 pages: about 16.3 pages of text plus the reference list, with no errors and no overfull boxes. The only undefined references point to other sections: `sec:rate`, `sec:remainder`, `sec:cusp` and `sec:ctwo`.

**Joint build.** A separate build of S1–S11 together (outside `tex/`) gave 122 pages, with no errors, no undefined references and no multiply defined labels. In that build this section runs from the middle of p. 95 to the top of p. 112, about 16.4 pages.

## 0. Sources

The Math- repository was read at `origin/main` = `9844f78`, read-only (`git show`). Blob prefixes come from `git rev-parse origin/main:<path>`.

### Primary sources

| Short | Path | Blob |
|---|---|---|
| [CT] | `frontiers/candidate_third_order_20261001/PROOF.md` (Theorem T) | `70ca57ef` |
| [CT-R] | `frontiers/candidate_third_order_20261001/README.md` | `4761618c` |
| [ET] | `frontiers/elder_third_order_20261001/PROOF.md` (Theorem E3, v1.1 with the status update) | `c8767dde` |
| [ET-R] | `frontiers/elder_third_order_20261001/README.md` | `dfd2623a` |
| [RT] | `frontiers/third_order_rate_20261001/PROOF.md` (Theorems T⁺, E3⁺, Corollary R⁺, Proposition W⁺) | `110ed33a` |
| [RT-R] | `frontiers/third_order_rate_20261001/README.md` | `e13ae7a2` |

### Folder histories (`git log --oneline origin/main -- <folder>`)

None of the three folders contains an ERRATUM or AMENDMENT file.

- **[CT]:** `7262001`, a single commit.
- **[ET]:**
  - `6da0715` v1;
  - `827812b` v1.1: Lemmas H and S′ replace Lemma B′;
  - `b229143` wording;
  - `e481d23` rebind of #187 to `07260114`;
  - `70dcf31` status update, no mathematical change.
- **[RT]:**
  - `15cf419`;
  - `5dbe15c` rebind of [ET] to `c8767dde`, status only.

### Upstream sources

These were read to state the inputs exactly. They are cited in the LaTeX through the supplement sections that house them.

| Short | Path | Blob |
|---|---|---|
| [R] | `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md` (Theorem R) | `247b3ecf` |
| [P] | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d` |
| [P-err] | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` | `213594d6` |
| [Z] | `frontiers/c7_zero_gap_limit_20260929/PROOF.md` | `5b6328ea` |
| [C7-K] | `frontiers/c7_total_bounded_20260929/PROOF.md` | `28748b08` |
| [C7-K-err] | `frontiers/c7_total_bounded_20260929/ERRATUM_NORMALIZED_FORM.md` | `b6764ce6` |
| #191 | `frontiers/remainder_vanishing_20260930/PROOF.md` | `441152df` |
| #191-err | `frontiers/remainder_vanishing_20260930/ERRATUM_20261004.md` | `0ffb1031` |
| #198 | `frontiers/remainder_rate_20260930/PROOF.md` | `abfb98ae` |
| #207 | `frontiers/cusp_second_order_20261001/PROOF.md` | `f6df5a73` |
| #187 | `frontiers/far_elder_rate_20260930/PROOF.md` | `07260114` |
| #187-err | `frontiers/far_elder_rate_20260930/ERRATUM_20261004.md` | `93afac3b` |

### Main text

`/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/`, read-only:

- `s1_field.tex`: (1.1).
- `s2_results.tex`: Convention 2.2, Theorem 2.3 with (2.12)–(2.14), Remark 2.4, and Remark 2.5 with (2.17)–(2.18).
- `s3_proof.tex`: (3.4)–(3.5).
- `appA_records.tex`: [R7] Theorem K/Z, [R9] far elder pairs, and [R11] the three-term-law chain, which lists exactly [CT], [ET], [RT] and their inputs.

### Neighbouring supplement sections

These were read only, to align notation and the inputs:

- `supp_s7_rate.tex` (`sec:rate`);
- `supp_s8_remainder.tex` (`sec:remainder`, including its "notation for Part II" convention);
- `supp_s9_cusp.tex` (`sec:cusp`);
- `supp_s11_ctwo.tex` (`sec:ctwo`).

## 1. Notation conversions

### Conventions shared with the rest of Part II

The section follows the Part II conventions of `sec:remainder` and `sec:cusp`.

| Source | Supplement | Note |
|---|---|---|
| pinned kernels `A_r`, `A_r^{eld}`, `A_r^{rej}` ([CT], [ET], [RT]) | `\mathcal A_r`, `\mathcal A^\eld_r`, `\mathcal A^\rej_r` | |
| contact kernel `A_0`; fold coefficient `A₂` ([CT], [ET], [RT]) | `\mathcal A_0`; `\mathcal A_2` | as in `sec:remainder`, `sec:ctwo` |
| cusp kernels `𝒜^{cand}`, `𝒜^{eld}`, `𝒜^{con}` ([CT], [ET], [RT] §0) | `\mathcal C^\cand`, `\mathcal C^\eld`, `\mathcal C^{\rm con}` | as in `sec:cusp` |
| `A_0 = D_y²F_0(0)` on the coupling space, "Δ := det A_0" ([CT] Lemma C proof) | `A_0`, `\Delta_0` | |
| the coupled `Y` of #198 (3.1) | `\widehat Y` | It is distinct from the own-jet `Y_r` of [ET] §2, which is kept, and from the cusp `Y`, which is kept. |
| `r_0^{[R]}`, `r_0^{[K]}` | `r_R`, `r_K` | as in `sec:remainder` |
| `A < 0`, `A ≤ −λI`, `cI ≤ Γ ≤ CI` (matrix order) | `A\prec0`, `A\preceq-\lambda I`, `cI\preceq V\preceq CI` | main-text notation |

The main text's gap parameter is related to the gap mark as follows (stated in the LaTeX):

- κ_main of (3.5) of the main text equals `6k`;
- ℓ = `kr³`;
- in this section `κ` is the cusp variable `k/r = ℓ/r⁴`, as in `sec:cusp`.

At `d = 3`, `L = 24`:

- `u` is the `t` of (3.4) of the main text, and Θ = (n₁, n₂);
- `ν_eld = ν_{3,24}` and `ν_cand = ν^cand_{3,24}`;
- `c_{d,L} = c_{3,24}` of (2.9), `B_{d,L} = B_{3,24}` and `K_L = K_{24}` of (1.1).

### Renamings local to this section

These renamings avoid clashes with other symbols of the section.

| Source | Supplement | Reason |
|---|---|---|
| [CT] Step F1 polynomial `Φ`, `Φ_j` | `\Omega`, `\Omega_j` | `Φ` is the window map of the decision subsection |
| [CT] Step F2 good event `G`, constant `C₁`, `g(t)` | `\mathcal G`, `C_\star`, `\omega(t)` | `C₁` is Lemma CU.1′'s constant |
| covariance `Γ_r` of `J′` ([CT] Step F3, [ET] (F6)) | `V_r` | `Γ = |γ|/λ` in Lemma Q |
| [CT] Lemma D constants `C_φ`, `c_φ` | `C_D`, `c_D` | `φ` is the cusp ratio |
| [RT] Lemma L: `c` | `a` | `c` is generic |
| [CT] Lemma C: `m_i := det K_i/r`, `s := (−1)^m` | `\delta_i`, `\varsigma` | `m = d − 1`; `s` is the cusp variable |
| [ET] Lemma S (1.2): `b, b′, s, s′` | `w, w′, \varsigma, \varsigma′` | |
| [ET] Lemma Q steps Q1–Q6 | Steps 1–6 | Hypotheses (Q1)–(Q4) and margins (M1)–(M6) keep their names. |
| [ET] Lemma H: Gram matrix `Γ_r`, `m_κ` | `\mathfrak G_r`, `\mathsf m_\kappa` | |
| [ET] Lemma S′: `K` (barrier constant), `C₃` | `K_f`, `C_\varepsilon` | |
| [ET] Lemma G: `Γ, Γ̃, H(e), s, s₀, h(s)` | `V, \tilde V, h(e), \xi, \xi_0, \chi(\xi)` | |
| [ET] Step E1: `F_dF_{d−1}`, `G(J′)`, `G₀(J′)` | `\mathbf F`, `\mathsf w(J')`, `\mathsf w_0(J')` | |
| [ET] (F1) remainders `R_M, R_S` (own-jet form) | `R'_M, R'_S` | distinguishes them from the coupled (3.1) |
| [RT] (4.2) variable `s` | `\varepsilon` | |

### Decomposition

| Source | Supplement |
|---|---|
| [CT] `F`, `K`, `J₂`, `J₃`, `J₄`, `J₅` | `I_f`, `I_c`, `J_2`, `J_3`, `−J_4`, `J_5` |
| [ET] `F^{eld}`, `K^{eld}`, `I^{eld}`, `J₄` | `I^\eld_f`, `I^\eld_c`, `I^\eld_m`, `J_4` |
| [CT] `K₀ := ℓ/ρ_f³` | `k_f` |
| [CT] `σ`, `S` | `s_f`, `s_c` (the names of [ET] and [RT]) |
| error integrals of the cusp region | `\mathcal E`, `\mathcal E^\eld` (new names) |

- **Sign of `J_4`.** [CT]'s `J₄ := −∫r^{−2}A_0` is the negative of the supplement's `J_4 ≥ 0`, which is [ET]'s `J₄`.
- **Dominating variable.** `T` of eq:third-T is [CT] §1's, with `C⁹` and `C⁴` norms. It dominates the `T` of `sec:remainder`, which uses `C⁷` and `C²` norms, and the LaTeX says so.

### Statement names

| Source | Supplement |
|---|---|
| T | thm:third-T |
| T′ | the "in particular" sentence of thm:third-T |
| E3 | thm:third-E3 |
| E3′ | cor:third-E3prime |
| T⁺ | thm:third-Tplus |
| E3⁺ | thm:third-E3plus |
| R⁺ | cor:third-Rplus |
| W⁺ | prop:third-window |
| D | lem:third-D |
| (1.2) | lem:third-condmom |
| L | lem:third-L |
| F | lem:third-fold |
| C and C⁺ | lem:third-C (merged) |
| O | lem:third-O |
| CU.1′ | lem:third-CU1 |
| S | lem:third-S |
| Q | lem:third-Q |
| (F2)–(F7) | lem:third-facts (i)–(vi) |
| (F1) | eq:third-ownexp |
| G | lem:third-G |
| CE and CE⁺ | lem:third-CE (merged) |
| O′ | lem:third-Oprime |
| H | lem:third-H |
| S′ | lem:third-Sprime |

## (a) Statements and numbered displays

Line numbers refer to the source files at the blobs above.

- **Full:** every step is kept.
- **Condensed:** wording is compressed, and only the items listed are omitted or merged.

### Setting (`ssec:third-setting`)

| Label | Content | Source | Blob |
|---|---|---|---|
| (intro text) | scope; (2.12) of the main text is the case d = 3, L = 24 of eq:third-E3p1 | [RT] §0; main text Theorem 2.3; [R11] | 110ed33a |
| eq:third-pins | pins `M, S`, `f(M) = b`, `f(S) = b − kr³`, zero gradients | [CT] §0 l.33; [ET] §0 l.106–107 | 70ca57ef, c8767dde |
| eq:third-cuspvar | `κ = k/r = ℓ/r⁴`, `s = rℓ^{−1/4} = κ^{−1/4}`, `P = 1 + |b| + k` | [RT] §0 l.55; [CT] §4 l.376; [CT] §0 l.35 | 110ed33a, 70ca57ef |
| (text) | `U_r`, `v_r`, `π_r`, `Q`, `U_0`, `v_0(b,k)`, contact law, Jacobian `12r^{−(d+3)}` | [CT] §0 l.34–35; [ET] Lemma H l.552–555, l.564 (from [P] (3.1)–(3.3)) | 70ca57ef, c8767dde |
| (text) | `F_j`, `K_i = D_r^{−1}H_iD_r^{−1}`, `W_r`, elder mark `e` | [CT] §0 l.36–37; [ET] §0 l.107–109; [R] §4; [P-err] | 70ca57ef, c8767dde, 213594d6 |
| eq:third-kernels | `\mathcal A_r`, `\mathcal A^\eld_r`, `\mathcal A^\rej_r`, `\mathcal A_0`, `z_0` | [CT] §0 l.37–38; [ET] §0 l.110, §4 l.654 | 70ca57ef, c8767dde |
| eq:third-Y | cusp `Y`; `φ = Y/(6κΔ) = (f₄ − 3γᵀA^{−1}γ)/(72κ)` | [CT] §0 l.43–44; [ET] §1 l.217–218; [RT] §0 l.59 | 70ca57ef, c8767dde, 110ed33a |
| (text) | `w_κ` | [CT] Lemma C l.268; [RT] §0 l.61 | 70ca57ef, 110ed33a |
| eq:third-weights | `w^\eld_κ`, `loss_κ` | [ET] §0 (0.1) l.124–125 | c8767dde |
| eq:third-cuspdiff | `\mathcal C^\cand − \mathcal C^{con}` and `\mathcal C^\eld − \mathcal C^{con}` | [CT] §0 l.46–48; [ET] §0 l.127–128 | 70ca57ef, c8767dde |
| eq:third-c1 | `c₁ < 0` (the `s`-integral) and `I^cand = (3^{1/4}/2)c₁`; at d = 3, L = 24, `c₁` is (2.13) of the main text | [ET] (0.2) l.130; [CT] §0 l.49–50; #207 (CU.2), (7.3); added step A1 | c8767dde, 70ca57ef, f6df5a73 |
| eq:third-far | far candidate and far elder densities | [CT] §0 l.40 and §4 l.357; [ET] §4 l.696–698 (#198 (2.1)); `B_{d,L}` from [Z] (Z3) | 70ca57ef, c8767dde, abfb98ae, 5b6328ea |

### Inputs (`ssec:third-inputs`)

| Label | Content | Source | Blob |
|---|---|---|---|
| eq:third-coupling | the coupling `(F_r, F_0)` and its moments | [R] (R2)–(R4); [CT] §0 l.35, §1 l.117–123 | 247b3ecf, 70ca57ef |
| eq:third-pindens | pin density bounds | [R] (R5); [CT] Step C3 l.297 | 247b3ecf, 70ca57ef |
| (text) | the block form and determinant of `K_i` | [R] §4; [P] §5 with [P-err] | 247b3ecf, dfed3b8d, 213594d6 |
| eq:third-T | dominating `T` (`C⁹`/`C⁴`) and `‖T‖_p ≤ C_pP`; `‖A_i − A_0‖ ≤ (r/2 + r²)T`; `F_j(K_i) ≤ C(k+r)T^d`; `0 ≤ \mathcal A_0 ≤ k²H` | [CT] §1 l.117–118; [CT] Lemma C proof l.261 ([R] (R9)); [R] (R11); [CT] §4 l.388 | 70ca57ef, 247b3ecf |
| (text) | `r_0^* ≤ r_R ≤ 1`; `r_0^* < L/(4√2)` (strict: correction 2); `r_0^* ≤ r_K` | #191 §0 l.27; [ET] (F1) l.371; [ET] §4 l.657; #191-err O3 | 441152df, c8767dde, 0ffb1031 |
| eq:third-K2 | rejected mass `E_Q[(W_r/r²)(1−e)] ≤ C(r³/k)P^N` for `r ≤ min{k, r_K}` | [C7-K] (K2) with `1 − e ≤ 1_{G_r^c}` ([C7-K] §4), as used in [ET] §4 l.657–658 | 28748b08, c8767dde |
| eq:third-ident | near/far identities for `ν_cand`, `ν_eld`, and `c_{d,L}ℓ^{−1/3}` | [ET] §0 l.112–118; [CT] §0 l.39–40, §4 l.356 (#191 (2.1), #198 §1 and (2.1), [P] §10, [Z] (Z13)–(Z14)) | c8767dde, 70ca57ef |
| eq:third-Bsplit | split of `B_{d,L}` at `r_0` | [CT] §4 l.357 (#207 §7); [RT] §4 l.290 for `r^{−2}\mathcal A_r(b,0,u) = r^{d−1}Ψ_0(b,ru)` | 70ca57ef, 110ed33a |
| eq:third-exp | coupled expansions of `det K_M`, `det K_S` | #198 (3.1) as quoted in [CT] Step C1 l.263–266 | abfb98ae, 70ca57ef |
| eq:third-ownexp | own-jet form, for every `r ≤ r_0^*`, `k ≥ 0` | [ET] (F1) l.366–371 (v1.1 MINOR-1); [RT] §4 l.240–242 (`k = 0`) | c8767dde, 110ed33a |
| (text) | sign window | #198 §3; [CT] l.281; [RT] l.249–250 | abfb98ae |
| eq:third-W1 | (W.1) and (W.4) | #198 Lemma W; [ET] §4 l.678; [CT] §4 l.386 | abfb98ae |
| eq:third-barrier | deterministic barrier and its consequence | #198 Lemma B (1.2); [ET] Lemma S′ l.599 | abfb98ae, c8767dde |
| eq:third-Fprime | crude far elder bound `Cℓ^{1/3}` | #198 Lemma F′; [ET] §4 l.691–692 | abfb98ae |
| eq:third-gapLip | gap-Lipschitz bounds | #207 §7 Lemma L (7.1)–(7.2); [CT] §4 l.387, 389 | f6df5a73 |
| eq:third-farrate | far elder rate `Cℓ^{2/3}` | #187 Theorem F; [ET] (E3.3) l.146, §4 l.695–698; main text [R9] | 07260114 |
| (text) | finite-jet linear independence and parity | [P] §2, §15; [ET] Lemma H l.566–570; [CT] Step C3 l.303–310 | dfed3b8d |

### Results (`ssec:third-results`)

| Label | Content | Source | Blob |
|---|---|---|---|
| eq:third-c2 | definition of `c₂` (Hadamard finite part) | [CT] §0 (0.1) l.52–58 | 70ca57ef |
| thm:third-T, eq:third-T1 | Theorem T (remainder `ℓ^{4/11}`); absolute convergence; Corollary T′ as "in particular" | [CT] l.61–69 | 70ca57ef |
| thm:third-E3, eq:third-E30 to eq:third-E33 | Theorem E3 with (E3.0)–(E3.3) and the equivalences | [ET] l.137–153 | c8767dde |
| cor:third-E3prime, eq:third-rej0 | Corollary E3′ with items (1)–(3) | [ET] l.155–163 | c8767dde |
| thm:third-Tplus, eq:third-Tp | Theorem T⁺ | [RT] l.65–67 | 110ed33a |
| thm:third-E3plus, eq:third-E3p0, eq:third-E3p1 | Theorem E3⁺, the intermediate-separation bound and the equivalence | [RT] l.69–79 | 110ed33a |
| cor:third-Rplus, eq:third-Rp | Corollary R⁺ | [RT] l.81–83 | 110ed33a |
| prop:third-window, eq:third-Wp1 to eq:third-Wp3 | Proposition W⁺ | [RT] l.85–95 | 110ed33a |
| rem:third-main | at d = 3, L = 24: eq:third-E3p1 is (2.12), with `c₁` of (2.13); (2.14) follows with relative remainder `O(ℓ^{16/21})`; eq:third-Rp is (2.18) | main text Theorem 2.3, Remark 2.5; [RT] §0; added step A8 | 110ed33a |

### Lemmas and proofs

| Label | Content | Source | Blob | Proof |
|---|---|---|---|---|
| lem:third-D, eq:third-D | eigenvalue and determinant layers | [CT] Lemma D (1.1), l.94–115 | 70ca57ef | full |
| lem:third-condmom | `E[T^p | J₀] ≤ C_p(P + |J₀|)^p` | [CT] §1 (1.2), l.117–132 | 70ca57ef | full |
| lem:third-L, eq:third-L1 | κ-free local Lipschitz bound | [RT] Lemma L (L.1), l.121–132 | 110ed33a | full; the sharpness remark l.134–137 is omitted |
| lem:third-fold, eq:third-F1, eq:third-F2 | fold expansion | [CT] Lemma F (F.1)–(F.2), l.136–143 | 70ca57ef | condensed (see below) |
| eq:third-Omega | product polynomial | [CT] (2.1), l.166 | 70ca57ef | |
| eq:third-Omega012 | `Ω₀`, `Ω₁ = 0`, `Ω₂` at `k = 0` | [CT] (2.2), l.176 | 70ca57ef | |
| eq:third-F3 | the typed indicator replaced by `1{A ≺ 0}` | [CT] (2.3), l.213 | 70ca57ef | |
| eq:third-A2 | formula for `\mathcal A_2` | [CT] (2.4), l.230 | 70ca57ef | |
| lem:third-C, eq:third-Cp1, eq:third-C1 | candidate cusp kernel: rate `r(1+κ)`, and in particular `r(1+κ)²` | [RT] Lemma C⁺ (C⁺.1), l.141–144; [CT] Lemma C (C.1), l.254–257 | 110ed33a, 70ca57ef | condensed and merged (see below) |
| eq:third-C2 | pathwise comparison | [CT] (C.2), l.285; [RT] l.148–151 | 70ca57ef, 110ed33a | |
| lem:third-O, eq:third-O1 | overlap `O(κ^{−1})` | [CT] Lemma O (O.1), l.321–336 | 70ca57ef | full; exponent written as generic `N` (Issue I2) |
| eq:third-cuspP | cusp polynomial; `Ξ*`; ridge `g` | [ET] §1 l.208–218 (#207 (0.2), Theorem CU.2(a)) | c8767dde | |
| lem:third-CU1, eq:third-CU1 | weighted Taylor bounds | [ET] Lemma CU.1′ (1.1), l.220–235 | c8767dde | full (wording); the checker sentence is omitted |
| lem:third-S, eq:third-Schur | Schur complements | [ET] Lemma S (1.2), l.241–247 | c8767dde | full |
| lem:third-Q, eq:third-decision | elder decision with explicit margins | [ET] Lemma Q (1.3), l.249–349 | c8767dde | full (see below) |
| eq:third-trans | transverse boundary bound | [ET] (1.4), l.296 | c8767dde | |
| lem:third-facts, eq:third-f4, eq:third-Alayer | facts (i)–(vi) | [ET] §2 (F2)–(F7), (2.1), (2.2), l.372–394 | c8767dde | condensed (see below) |
| lem:third-G | two Gaussian comparisons | [ET] Lemma G, l.396–416 (v1.1 integrability) | c8767dde | full |
| lem:third-CE, eq:third-CEp1, eq:third-CE1 | elder cusp kernel with rate; `r_2 = min(r_0^*, 1/2, r_Q)` | [RT] Lemma CE⁺ (CE⁺.1), l.175–177, l.231; [ET] Lemma CE (CE.1), l.418–425 | 110ed33a, c8767dde | condensed and merged (see below) |
| eq:third-E1 | pathwise comparison | [ET] (2.3), l.437; [RT] l.181 | c8767dde, 110ed33a | |
| eq:third-B4 | the edge window charged `κr` | [RT] (3.1), l.195 | 110ed33a | |
| eq:third-B4incl | inclusion | [RT] (3.2), l.202 | 110ed33a | |
| eq:third-E5 | kernel difference via `loss_κ` | [ET] (2.4), l.490 | c8767dde | |
| lem:third-Oprime, eq:third-Oprime | elder overlap | [ET] Lemma O′ (O′.1), l.507–516 | c8767dde | full |
| lem:third-H, eq:third-H1 | rescaled Hessian | [ET] Lemma H (H.1), l.522–590 | c8767dde | condensed (see below) |
| lem:third-Sprime, eq:third-Sp, eq:third-Sp2 | elder weight through `\widetilde H` | [ET] Lemma S′ (S′.1)–(S′.2), l.592–624 | c8767dde | full (see below) |
| proof of prop:third-window; eq:third-W41, eq:third-W42 | Steps W1–W3, (W⁺.2), (W⁺.3) | [RT] §4 (4.1)–(4.2), l.236–292 | 110ed33a | full (see below) |

### Proofs of the theorems (`ssec:third-proofs`)

| Label | Content | Source | Blob | Proof |
|---|---|---|---|---|
| eq:third-decT | candidate decomposition (with the `J₄` sign convention) | [CT] §4 l.343–357; [RT] §5 l.307–311 | 70ca57ef, 110ed33a | |
| eq:third-decE | elder decomposition at `r_0 = r_0^*` | [ET] §4 l.641–652; [RT] §5 l.320–321 | c8767dde, 110ed33a | |
| (text) | fold region | [CT] §4 l.359–371; [ET] §4 l.654–660 | 70ca57ef, c8767dde | |
| eq:third-FK | cusp region | [CT] §4 l.373–383; [ET] §4 l.662–673; [RT] §5 l.313–316, 324–329, 339 | 70ca57ef, c8767dde, 110ed33a | |
| (text) | remaining candidate terms `J₂`–`J₅` | [CT] §4 l.385–389 | 70ca57ef | |
| (text) | intermediate separations | [ET] §4 l.677–685; [RT] §5 l.331–337 | c8767dde, 110ed33a | |
| (text) | far part | [ET] §4 l.691–698; [RT] §5 l.356–358 | c8767dde, 110ed33a | |
| proof of thm:third-T, thm:third-E3, cor:third-E3prime | the `4/11` ledger | [CT] §4 l.391–407; [ET] §4 l.687–705 | 70ca57ef, c8767dde | condensed (see below) |
| proof of thm:third-Tplus, thm:third-E3plus, cor:third-Rplus | the `3/7` ledger | [RT] §5 l.341–362 | 110ed33a | condensed (see below) |

### What was condensed or merged in each proof

**lem:third-fold.**
- Step F1's bullets are merged into one paragraph.
- The [CT] attributions to #191 and #198 inside Step F1 are folded into "by eq:third-ownexp".
- Step F2 keeps all its estimates: the layers, `p > N₂ + 4` and `Cr³(1+k^{−1})P^N`.
- The surrogate remark (l.239–250) is omitted; it is not part of the proof.

**lem:third-C** merges [CT] Lemma C with [RT] Lemma C⁺.
- The statement gives (C⁺.1), and (C.1) as "in particular".
- Step 1 is [CT] Step C1 together with [RT]'s remark (l.148–151) that it is linear in κ.
- Step 2 is [RT] Step C2⁺, via Lemma L. [CT]'s `12κ|det A|` estimate is kept in a parenthesis.
- Step 3 is [CT] Step C3 with [RT]'s density estimate (l.161–162).
- [CT]'s closing remark (l.315–317) is omitted, because C⁺ supersedes it.

**lem:third-Q.** All of Steps 1–6, the constants 35/8, 105/64, 2911/128, 317/4 and 80, and (M1)–(M6) are kept. Only the checker sentences (l.289, 321) and the closing remark (l.351–352) are omitted.

**lem:third-facts.** Each justification is shortened to a pointer within the section:
- (i) is (F3) with (2.1);
- (ii) is (F4);
- (iii) is (F5) with (2.2);
- (iv) is (F6);
- (v) is (F7);
- (vi) is (F2).

**lem:third-CE** merges [ET] Lemma CE with [RT] Lemma CE⁺.
- Step E1 is unchanged.
- Step E2 gives [ET]'s four events, with [RT]'s sharper `\mathfrak B_\lambda` bound (l.188–189) and Step E2⁺ (eq:third-B4, eq:third-B4incl, with both pieces).
- Step E3 is [RT] Step E3⁺; [ET]'s `12κ|Δ|` version is kept in a parenthesis.
- Steps E4–E6 are [ET]'s, with [RT]'s note (l.228–229).
- The closing sentence derives both (CEp1) and (CE1).
- [RT]'s remark l.231–234 is omitted, except for the implication, which is in the statement.

**lem:third-H.** The following are omitted:
- the relation to `D_r` (l.528–529);
- the Jensen alternative to Anderson's inequality (l.588–589);
- the control sentence (l.550).

The continuity, invertibility, mean and conditioning arguments are all kept.

**lem:third-Sprime.** The comparison paragraph (l.626–637) is omitted.

**Proof of prop:third-window.** [RT]'s comparison remark (l.294–299) is omitted.

**Proofs of thm:third-T, thm:third-E3 and cor:third-E3prime.**
- The fold, cusp, overlap, `J₂`–`J₅`, intermediate and far parts are written once, in the decomposition paragraphs, for both rates.
- The ledger table is given in one sentence with all nine exponents.
- The checker references are omitted.

**Proofs of thm:third-Tplus, thm:third-E3plus and cor:third-Rplus.** The ledger table is written as one sentence listing every term, its source and its exponent. The optimal-split sentence (l.361–362) is kept.

## (b) Corrections applied

The assignment gives no numbered corrections for this section. Item 9 of `PLAN.md`, "no status words", is applied throughout; items 1–8 of `PLAN.md` concern Part I and do not apply.

Every landed correction to the three folders and to their consumed inputs is listed below.

1. **Density versions (#191-err F1, `0ffb1031`).** The statements are pointwise in ℓ and are made for the canonical marked Kac–Rice density versions of `ν_cand`, `ν_eld` and `ρ_rej`.
   - The Setting carries this qualifier ("in their canonical marked Kac–Rice density versions (Convention 2.2 of the main text; [P])"), and all theorems and corollaries are stated for these versions.
   - This also realizes the assignment's "for the canonical density version".
2. **Strict cylinder radius (#191-err O3).** The bound `r_0^* < L/(4√2)` is strict, in the Inputs and in the far-part step.
3. **Elder note v1.1 (commits `827812b`, `b229143`, `e481d23`, `70dcf31`; transcribed from blob `c8767dde`).**
   - Lemmas H and S′ replace Lemma B′ and the exponent β, so the radius of Lemma CE no longer involves #207 Lemma CU.5.
   - (E3.3) has the rate `ℓ^{4/11}` and uses #187 only at the single separation `r_0^*`.
   - Lemma G(b) assumes `E[h(E)] < ∞` (review amendment OA-220-B-01); Step E6 records why it holds.
   - (F1) is stated for every `r ≤ r_0^*` (MINOR-1; eq:third-ownexp).
   - Lemma S′'s barrier step states `r ≤ r_0^* ≤ r_R ≤ 1` and `−\widetilde H ≻ 0`, and notes that the inequality can fail for `r > 1` or for an indefinite `\widetilde H`.
4. **Status-only commits.** The rate note's rebind (`5dbe15c`) and the elder note's #187 rebind (`e481d23`) change no mathematics; nothing was applied.
5. **#187-err (`93afac3b`).** It concerns provenance only. Theorem F, the far rate used here, is unchanged; nothing was applied.
6. **[C7-K-err] (`b6764ce6`).** It withdraws the unrestricted normalized remark of [C7-K] §3. That remark is not used here: eq:third-K2 is the unnormalized (K2) for `r ≤ min{k, r_K}`, which the erratum retains.
7. **[P-err] (`213594d6`).** The congruence `K_i = D_r^{−1}H_iD_r^{−1}` with `D_r = diag(r^{1/2}, I)` (the [R] §4 convention) is the corrected one, with off-diagonal blocks `r^{1/2}β_i` and `det K_i = det H_i/r`. It is used as stated; no change was needed.
8. **#191-err F4.** `c₁` denotes only the coefficient of `ℓ^{1/4}`, as the erratum requires. Items F2, F3, O1, O2 and O4 concern status, wording or source rows and have no effect here.

## (c) Added steps

- **A1.** `I^cand = (3^{1/4}/2)c₁` in eq:third-c1. It is the ratio of the closed forms `−12·(8/7)6^{1/4}∫∫π_0E_0[|Y|^{7/4}|Δ|^{1/4}1{A≺0}|b]` ([CT] l.50) and `−12·(16/7)2^{1/4}∫∫…` (#207 (CU.2); main text (2.13)), since `(8/7)6^{1/4}/((16/7)2^{1/4}) = 3^{1/4}/2`. The same identity appears in (2.17) of the main text and in `sec:cusp`.
- **A2.** In the proof of Theorem T, the setup requires `ρ_c ≤ r_2 = min(r_0^*, 1/2, r_Q)`.
  - [CT] l.343–344 asks only for `ρ_c < r_0` and `ℓρ_f^{−3} ≤ 1`, but Lemma C, which is applied for `r ≤ ρ_c`, needs `r ≤ min(r_0^*, 1/2)`.
  - The condition holds for ℓ small; [RT] l.304 has `ρ_c ≤ r₂` and [ET] l.642–643 has the same.
- **A3.** The unified decomposition states its conditions explicitly:
  - `ℓ^{1/3} ≤ ρ_f < ℓ^{1/4}`, which gives `k ≤ 1` on `[ρ_f, ∞)` and `k ≥ r` on `(0, ρ_f]`;
  - `ρ_c ≤ r_2`, `ρ_c < r_0` and `a < r_0^*`.

  The second and third proofs check them with exponents: `ℓρ_f^{−3} = ℓ^{2/11}` and `ℓa^{−4} = ℓ^{7/15}` for T/E3; the [RT] l.305 statements for T⁺/E3⁺.
- **A4.** Intermediate separations.
  - The (W⁺.2) alternative on `[ρ_c, a]` is stated under `ℓρ_c^{−4} ≤ 1`, so that `k ≤ r` there. [RT] l.305 asserts this for its split.
  - The use of Lemma S′ on `[a, r_0^*]` is stated under `ℓa^{−4} ≤ 1`.
- **A5.** In lem:third-C, (C.1) follows from (C⁺.1) because `1 + κ ≤ (1 + κ)²`.
- **A6.** In the cusp region of the proofs, the error integrals 𝓔 and 𝓔^eld are evaluated once for each lemma version, `∫_{ρ_f}^{ρ_c} r(1 + ℓr^{−4})^j dr` for `j = 1, 2`, and for (CEp1) as in [RT] l.326. This only reorganizes [CT] l.375, [ET] l.663, [RT] l.314 and l.326.
- **A7.** In lem:third-O, the generic exponent `N` replaces the source's explicit exponent (see Issue I2).
- **A8.** In rem:third-main:
  - the relative remainder in (2.14) is `O(ℓ^{3/7+1/3}) = O(ℓ^{16/21})`;
  - `I^cand − c₁ = (3^{1/4}/2 − 1)c₁ = (1 − 3^{1/4}/2)|c₁| > 0`, since `c₁ < 0`.

## (d) Issues

- **I1. Page budget.** The section has about 16.3 pages of text (mid p. 95 to the top of p. 112 in the joint build), against the target of about 14.
  - Shortening further would drop estimates or constants that the mode requires.
  - The Setting subsection, about 1.2 pages, repeats definitions that the Part II conventions of `sec:remainder` and `sec:cusp` already give, with the same symbols after the alignment in §1. At integration it could be replaced by a pointer to those conventions plus the local definitions (`w_κ`, `w^\eld_κ`, `loss_κ`, eq:third-cuspdiff, eq:third-c1, eq:third-far).
- **I2. Lemma O exponent ([CT] l.334).** The middle line of the display gives the factor `(1 + |b|)^{n_p + m(m−1)/2}`.
  - The weight `(1 + |b| + ‖A‖)^{n_p}` contributes `(1 + |b|)^{n_p}`, and Lemma D, with mean `O(1 + |b|)`, contributes a further `(1 + |b|)^{n_p + m(m−1)/2}`. The exponent is therefore `2n_p + m(m−1)/2`.
  - The effect is cosmetic: (O.1) has a generic `N`. The LaTeX writes `(1 + |b|)^N` in that line.
  - Suggested fix at the source: replace the exponent by `2n_p + m(m−1)/2`, or by a generic `N`.
- **I3. The far elder rate has no home in the supplement.** eq:third-farrate (#187 Theorem F, `O(ℓ^{2/3})` at fixed ρ ≤ L/2) is cited to record [R9] of the main text, as `sec:remainder` also does. [R9] is a pointer to the repository folders, not a statement. The supplement may want to state the bound, with its proof or a precise citation.
  - Only (E3.3), Corollary E3′(3), (E3⁺.1) and the second form of (R⁺.1) use it.
  - eq:third-K2 is now taken from `sec:remainder`, which states it as the gap-mark-explicit bound and cites [R7].
- **I4. Two meanings of κ.** The main text's κ of (3.5) is the gap parameter `6k`; this section and `sec:cusp` use κ for the cusp variable `k/r`. The LaTeX states the relation, but the reader must keep the two meanings apart across the main text and Part II.
- **I5. Duplicate definition of `c₂`.** eq:third-c2 and `sec:ctwo`'s definition of `c₂` are identical. `sec:ctwo` cites this section for the fold expansion, which it states for `0 < r ≤ r_*`; here the range is `0 < r ≤ min(r_0^*, 1/2)`. The integrator may keep one definition.
- **I6. Two dominating variables named `T`.** This section's `T` (`C⁹`/`C⁴`, from [CT] §1) differs from the `T` of `sec:remainder` (`C⁷`/`C²`). The LaTeX states that it dominates the latter, so the imported bounds hold with it.
- **I7. Lemma C's range (Added step A2).** [CT]'s proof of Theorem T applies Lemma C on `[ρ_f, ρ_c]` without stating `ρ_c ≤ 1/2`. This is harmless for small ℓ, and the supplement states the condition.

## Checks run

- **Source checkers.** The three checkers were run from the scratchpad with `python3 -I -B -S`. They use only the standard library and exact rationals. All passed:
  - `thmT_check.py` (controls T1–T5);
  - `e3_check.py` (controls X1–X9);
  - `r37_check.py` (controls R1–R6).

  These checkers test the sources, not the LaTeX.
- **Transcription.** The transcription was checked by hand against the sources:
  - the ledgers (4/11, 2/5, 3/7, 4/9);
  - the Lemma Q constants (35/8, 105/64, 2911/128, 317/4, 80; (M1)–(M6));
  - the Step E2 reductions (18, 60, 18^{2/3}2^{1/3}, 160/L, (96/L)², 10, 16C₁, 80C₁, 1280C₁, 144C₀, 192ε*, 288C₀);
  - (H.1), (4.2) and Lemma L.

## Not done

- The page target of about 14 pages is not met; see I1.

---

# Concordance for S11 (`supp_s11_ctwo.tex`, `\section{The constant $c_2$}\label{sec:ctwo}`)

Writer W9, 13 October 2026. Math- sources read at `origin/main` = `9844f785` (4 commits after the main-text
pin `b68d88b3`; `git diff b68d88b3 origin/main` is empty on all three assigned folders).

## Source keys

| Key | Path (Math- `origin/main`) | Blob (8 hex) | Used for |
|---|---|---|---|
| [TOC] | `frontiers/third_order_coefficient_20261001/NOTE.md` (v1.4.1) | `fa448934` | definition of c2, fold/cusp matching (§0, §1), accuracy of floating values (§2.4), status of the SIDE24 numbers (§4) |
| [TOC-R] | `frontiers/third_order_coefficient_20261001/README.md` | `ccd40968` | "What it is", common c2, values table |
| [TOC-J] | `frontiers/third_order_coefficient_20261001/RESULTS.json` | `8a22bd73` | errors of the floating c2 against the closed forms (T1, T3, T4) |
| [CX] | `frontiers/c2_exact_20261001/NOTE.md` (v1.1) | `a2798b87` | closed forms, certified enclosures, method (§§0–7) |
| [CX-R] | `frontiers/c2_exact_20261001/README.md` | `9fb11719` | closed forms, ratio formula |
| [CX-J] | `frontiers/c2_exact_20261001/RESULTS.json` | `3077d9b2` | exact strings, intervals, series orders (10, 6; stability 12, 8), cone constants |
| [FJ] | `frontiers/c2_finite_jet_transfer_20261001/PROOF.md` | `54cc4a1a` | Theorems J, Q, Corollary Q24, §§1–7 |
| [FJ-R] | `frontiers/c2_finite_jet_transfer_20261001/README.md` | `ab4b5c7b` | scope, "no positive c1 sandwich" |
| [FJ-J] | `frontiers/c2_finite_jet_transfer_20261001/RESULTS.json` | `0de6626a` | budget values (W2 norm 226469/180, final 1.952e47), torus intervals |
| [FJ-S] | `frontiers/c2_finite_jet_transfer_20261001/SOURCES.json` | `168be38f` | the interfaces ("actual typed-kernel and persistence identification retained as conditional imported interfaces") |
| [FJ-V] | `frontiers/c2_finite_jet_transfer_20261001/VALIDATION.md` | `8884e509` | read; test-suite only (not transcribed) |
| [T] | `frontiers/candidate_third_order_20261001/PROOF.md` | `70ca57ef` | (0.1), Lemma F (F.1)–(F.2), the surrogate paragraph after Lemma F (context; the interface of Hypothesis (a)) |
| [CU] | `frontiers/cusp_second_order_20261001/PROOF.md` | `f6df5a73` | (0.1) (Y), Theorem CU.4 (4.1), §6 loss(s) (context for Remark S11.4) |
| [E3] | `frontiers/elder_third_order_20261001/PROOF.md` | `c8767dde` | §5 Remark 1 (why the elder law has the candidate's c2; context) |
| [R] | `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md` | `247b3ecf` | §2 pin rows, |det| = 12 r^{-(d+3)}, target v_r; §4 kernel A_r = 12 pi_r(v_r) Z_r/r^2 |
| [ENC] | `coefficients/side24_v1/ENCLOSURE.json` | `57af39a0` | cross-check only of the 20-digit c_{3,24} enclosure (the supplement derives it from (2.10)–(2.11) of the main text) |
| MT | V3 main text `/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013/` | — | (1.1), (2.10), (2.11), (2.12)–(2.14), (2.17), (3.5), (3.7), (3.11), Theorem 2.3(iii), §5 |

## (a) Statements and displays

"Proof" column: **full** = every step and constant of the source proof is present; **condensed** = complete in
logic, with the routine algebra named in the entry omitted.

| Label (as written) | Content | Source | Proof |
|---|---|---|---|
| `sec:ctwo` (intro paragraph) | roadmap | new prose | — |
| `ssec:ctwo-def` "Notation" paragraph | notation of Part II, pins, U_r, v_r, 12 r^{-(d+3)}, U_0, v_0, pi_r, pi_0, E_0[·|b], W_r, e, P | [R] §1–§2, §4; [T] §0; [CU] §0; Section S8 convention (sec:remainder) | — |
| `eq:ctwo-pins` | pins f(M)=b, f(S)=b−kr³, gradients 0 | [R] §1; [T] §0 | — |
| `eq:ctwo-tau` | fourth axial row τ_r | [R] §2 (4th axial row); [FJ] §1 (T_r) | — |
| (text) τ_r = −12 V⁻_corr | relation to MT (3.7) | **added step** (algebra checked) | — |
| `eq:ctwo-jetcov` | Cov(D^α f, D^β f) = (−1)^{|β|} D^{α+β}K(0) | [FJ] §5 first display; [CX] §2.1 | — |
| `eq:ctwo-KL` | normalized periodization K_L | [FJ] §0 Corollary Q24; [R] §1; MT (1.1) | — |
| `eq:ctwo-Y` | A, Δ, γ, f_4, Y | [CU] (0.1); [T] §0 "Cusp objects"; MT (2.13) | — |
| (text) d = 1 conventions | empty matrix, Δ = 1, Y = f_4/12, |S^0| = 2 | [FJ] §0 ("empty-matrix convention det A=1 and sphere S^0 of measure2"); [TOC] §2.3 (Y = f₄/12) | — |
| `def:ctwo-kernel` | two-point kernel, contact kernel, elder kernel, signed fixed-cone surrogate, birth integral | [R] §4 (A_r, A_0); [T] §0 (z_0); [CU] §0 (A_r^eld); [FJ] §0 (signed fixed-cone surrogate, bar A_2 "AFTER integrating all birth heights"); [CX] §1 (Ã_r) | — |
| `eq:ctwo-kernel` | A_r = 12 π_r(v_r) E[W_r | U_r=v_r]/r² | [R] §4 (A_r = 12 pi_r(v_r) Z_r/r^2); [TOC] §0 | — |
| `eq:ctwo-contact` | A_0 = 12 π_0 · 36k² E[Δ² 1{A<0} | U_0=v_0(b,k)] | [R] §4; [T] §0 | — |
| `eq:ctwo-surrogate` | Ã_r[K] | [FJ] §0 para 2; [CX] §1 | — |
| `eq:ctwo-birthint` | bar A_r[K] = ∫ Ã_r db | [FJ] §0, §1 (1.2) | — |
| (text after def.) | |det H_M||det H_S| = −det H_M det H_S on the typed event; not a count | [CX] §1 last sentence; [FJ] §0 ("We do not identify the full finite-r signed surrogate with a count"), §1 | — |
| `eq:ctwo-fold` | fold expansion (F.1)–(F.2), no r¹ term | [T] Lemma F (F.1), (F.2), (2.2); [TOC] §0 ((0.1) and "no r¹ term"). r_* = min(r_0^*, 1/2) of [T] | stated, proved in S10 |
| `def:ctwo-c2` / `eq:ctwo-def` | c_2 (finite part) | [T] (0.1); [TOC] §0 definition | — |
| `eq:ctwo-J0` | c_2[K] (birth integral first) | [FJ] (J0) | — |
| `rem:ctwo-fp` | absolute convergence, Hadamard finite part, (J0) = same expression, d = 1 expression only | [TOC] §0 ("The integral converges: A₂(b,k) − A₂(b,0) = O(k)"; "Hadamard finite part"); [T] Theorem T ("(0.1) converges absolutely"); [FJ] §0 ("Define the same finite-part expression and spatial normalization as #218 (0.1)"; d=1 "defines an expression") | bound in (i) = **added step** |
| `rem:ctwo-mechanism` | formal fold–cusp matching; regimes (a)(b); composite approximation; neglected pieces formally O(ℓ^{1/2}); interface cancellation; elder rule only through the cusp window; same c_2 for candidate density | [TOC] §1 (whole), §0 ("Claim", "The reason for the common c₂"); [TOC-R] "What it is"; [CU] Theorem CU.4 (4.1) and §6 (loss(s)); [E3] §5 Remark 1; MT (2.17) | formal (no proof claimed; the record says "This is a sketch") |
| `eq:ctwo-pieces` | three substitution integrals | [TOC] §1 display ("∫ r^{−2}A₀ dr = cℓ^{−1/3}" etc.), written with the explicit k- and s-integrals | — |
| (unnumbered) 𝒞(b,s,u) | elder cusp correction | [CU] §6: (𝒜^{eld}−𝒜^{con})(b,s^{−4},u) = −12π₀E₀[loss(s)|b] | — |
| `ssec:ctwo-closed` first sentence | c[K_ref] := (1/3)∫∫∫ A_0 k^{−2/3} | [CX] §0 (c = (1/3)|S^{d−1}|∫db∫A_0 k^{−2/3}dk) | — |
| `thm:ctwo-closed` | closed forms d = 1,2,3; ratio; c = c_{d,∞} (d = 2,3), = MT (2.11) for d = 3; enclosure tables | [CX] §0 Theorem + certified table; [CX-R]; [CX-J] "exact", "values"; [CX] §4 item 2 | **condensed** (computer-assisted exact computation; its algebra is not reproduced, the method is described step by step) |
| `eq:ctwo-ratio` | c_2/c = ω_d Γ(5/6)/(12^{2/3}Γ(1/6)) | [CX] §0; [CX-J] "c2 / c" | — |
| proof Step 1 | isotropy, |S^{d−1}|, interchange of b and k integrals | [CX] §0–§1; interchange + "r²-coefficient of birth integral = birth integral of r²-coefficient" = **added step** | |
| proof Step 2 | exact series, conditioning, eigenvalue reduction (π(λ₁−λ₂)), Isserlis, analyticity of Ã_r | [CX] §2.1–§2.3; series orders from [CX-J] "derivation" | |
| proof Step 3 | structural facts (no r¹ term, q_0 = 3b²/2 + 24k², Q_0 k-free, even in k) | [CX] §2.4, §3.1 first sentence | |
| `eq:ctwo-kint` | k-integrals by Gamma finite parts | [CX] §3.1 (restated as two formulas; see Issue 3) | |
| proof Steps 5–6 | b by completing the square; cone: g = 3/16; (3/10, 3/10, 1/5), D_θ = 3/20, I_{0,0} = θ; θ cancels | [CX] §3.2, §3.3; [CX-J] cone exponents; θ = 2.354098… checked by quadrature | |
| proof Step 7 | exact checks | [CX] §4 items 1–5; the identity with 2^{−23/6}3^{−8/3}(29√6−36)Γ(1/6)π^{−5/2} = **added step** (checked: both equal 2^{−10/3}3^{−13/6}(29−6√6)Γ(1/6)π^{−5/2}) | |
| proof Step 8 | interval evaluation; widening 1e−14 relative, 16 digits; five seeded defects | [CX] §5, §6 | |
| `rem:ctwo-controls` | floating controls (Richardson; r-fit errors 5.7e−10, 1.9e−9/2.6e−9, 1.0–1.4e−9) | [CX] §7 item 1; [TOC] §0 item 2 "Accuracy", §2.4; [TOC-J] T1/T3/T4 | — |
| `ssec:ctwo-transfer` first paragraph | signed finite part: no positivity sandwich | [FJ-R] ("does not import the positive c1 sandwich into a signed finite-part problem"); [CX] §8 ("do not fit the homogeneity argument") | — |
| `thm:ctwo-jet` | Theorem J (finite-jet representation) | [FJ] §0 Theorem J | **condensed** (routine Taylor algebra behind (S11.21)–(S11.23) and the moment bookkeeping of (S11.25) are named and omitted; every expansion, coefficient and parity step is kept) |
| `eq:ctwo-J1` | (J1) | [FJ] (J1) | |
| `eq:ctwo-J2` | (J2) | [FJ] (J2) | |
| `eq:ctwo-V` | V_r = (G⁺_r, G⁻_r, τ_r), target (0,0,12k) | [FJ] §1 (1.1) | |
| `eq:ctwo-birth` | birth disintegration | [FJ] (1.2) | |
| `eq:ctwo-Vexp` | r-expansions of V_r | [FJ] (1.3) | |
| `eq:ctwo-endpoint` | α_ε, β_ε, A_ε | [FJ] (2.1) and the two pin identities after it | |
| `eq:ctwo-Ups` | Υ_1, Υ_2 | [FJ] (2.2) (U → Υ_1, V → Υ_2) | |
| (unnumbered) D_ε expansion | bordered determinant identity | [FJ] §2 display after (2.2); [R] (R7) identity | |
| `eq:ctwo-X2` | −D_−D_+ = 36k²Δ² + r²X_2 + O(r³) | [FJ] (2.3) (W2 → X_2) | |
| `eq:ctwo-scores` | p_o, a, q(A), scores | [FJ] (3.1) | |
| (text) density/score interpretation; "Δ is a function of A" | | **added step** (explanation of [FJ] "Differentiating the Gaussian density at r²=0") | |
| `eq:ctwo-A2bar` | bar A_2 = 12 p_o e^{−ak²} ∫ q h | [FJ] (3.2) (H → h) | |
| `eq:ctwo-hj` | h = h_0 + h_2k² + h_4k⁴, p_j | [FJ] (3.3) | |
| `eq:ctwo-extract` | coefficient extraction | [FJ] (3.4) | |
| `eq:ctwo-mellin` | three integrals | [FJ] §4 | |
| (text) absolute convergence of (J0) | | **added step** (from (J1)) | |
| "Derivative orders" paragraph | orders ≤ 8, f_5 only linear | [FJ] §5 | |
| `rem:ctwo-recover` | exact recoveries d = 1, 2; scaling t^{−4/3} | [FJ] §4 (4.1) and "Additional independent planar recovery" | — (consistency check, not part of a proof) |
| `thm:ctwo-stability` | Theorem Q | [FJ] §0 Theorem Q | **full** (every bound and constant of [FJ] §6 is reproduced) |
| `eq:ctwo-J3` | δ ≤ 1e−4 | [FJ] (J3) | |
| `eq:ctwo-J4` | ≤ 1e50 δ | [FJ] (J4) | |
| `eq:ctwo-floor` | conditioning floor | [FJ] (6.1) and the sentences around it | |
| `eq:ctwo-mean` | conditional means | [FJ] (6.2) | |
| `eq:ctwo-schur` | Schur derivative < 1e8 δ | [FJ] (6.3) | |
| `eq:ctwo-score` | score bounds | [FJ] (6.4) and the following paragraph (540090, 72, 450150) | |
| `eq:ctwo-po` | p_o, q, a bounds | [FJ] (6.5) | |
| `eq:ctwo-coeff` | coefficient norms; 226469/180 | [FJ] (6.6); exact value from [FJ-J] `budget.weight_l1` | |
| `eq:ctwo-wick` | Wick moments | [FJ] (6.7) (U0 → μ_*) | |
| `eq:ctwo-hbd` | bounds on h, h_j | [FJ] (6.8) | |
| `eq:ctwo-cone` | eighth-moment cone integral 824629750641408 | [FJ] (6.9) | |
| `eq:ctwo-pbd` | |p_j| < 1e41, |ṗ_j| < 1e44 δ | [FJ] (6.10) | |
| `eq:ctwo-cdot` | |ċ_2| ≤ 1.952e47 δ < 1e50 δ | [FJ] (6.11); value from [FJ-J] `budget.final_bound` | |
| `cor:ctwo-torus` | Corollary Q24; torus intervals | [FJ] §0 Corollary Q24, §7 table; [FJ-J] `torus_expression_enclosures` | **full** |
| `eq:ctwo-J5` | (J5) with E_d(L) | [FJ] (J5) | |
| `eq:ctwo-J6` | < 1e−60 for L ≥ 24 | [FJ] (J6) | |
| proof of `cor:ctwo-torus` | P_8, shells, ratio 2^{10}e^{−150} < 1/2, e^{−50}, e^{−288}, E_3(10), polynomial factors at L = 24 | [FJ] §7; the explicit image-sum expression Σ_{n≠0}(P_8(L|n|)+105)e^{−L²|n|²/2} spells out [FJ] "the image sum with P8+105" | |
| `ssec:ctwo-side24` first paragraph | two identifications, not proved here | [FJ] §0 (after (J6)), §7 last paragraph; [FJ-S] T entry | — |
| `hyp:ctwo-ident` (a) | typed kernel vs surrogate: same r²-coefficient, so c_2 = c_2[K_L] | [FJ] §0 para 2 ("identification of its r^2 coefficient with the actual typed kernel is an imported, separately reviewed comparison"); [FJ] after (J6) ("Conditional on their exact coefficient-identification interfaces"); [FJ-S] (T: "actual typed-kernel ... identification retained as conditional imported interfaces"); content from [T] surrogate paragraph after Lemma F and [CX] §1 (A_r = Ã_r + O(r³) at fixed admissible (b,k)) | hypothesis; since draft 10, Corollary S11.11(a), proved from Lemma S10.12 (see "Checkpoint draft 10") |
| `hyp:ctwo-ident` (b) | persistence identification | [FJ] after (J6) and §7 ("interpreting these as the actual elder-density coefficient remains conditional on the corrected parent identification"); [FJ-S] (T: "persistence identification"); [CX] §8 | hypothesis; since draft 10, Corollary S11.11(b), which is Theorems S10.4–S10.5 |
| `cor:ctwo-side24` | (i) torus value; (ii) c_2 under (a); (iii) c_2/c_{3,24} ∈ [3.859496174547456, 3.859496174548331]; (iv) under (a),(b) the constant of MT (2.12), (2.14) | (i) [CX] Theorem (d=3) + [FJ] (J6), §7 table; (iii) **added step**; MT (2.10), (2.11), Theorem 2.3(iii), (2.14) | **full** |
| `eq:ctwo-ratio24` | ratio interval | **added step**: exact rational interval division (checked with Python `fractions`): 0.1612340491269446/0.04177593184059834335 = 3.859496174547456752…, 0.1612340491269811/0.04177593184059834334 = 3.859496174548330462… | |
| (unnumbered) c_{3,24} enclosure | 0.04177593184059834334 < c_{3,24} < 0.04177593184059834335 | **added step** from MT (2.10)–(2.11) (|c_{3,24}/c_{3,∞} − 1| < 1.5e−118; c_{3,∞} = 0.041775931840598343342936…); agrees with [ENC] `dimensions.3` and MT Theorem 2.3(iii) | |
| `rem:ctwo-scope` | scope of (J6), looseness near L = 10, reference interval vs torus | [FJ] §0 after (J6), §7 last paragraph; [FJ-R] last paragraph; [TOC] §4 ("The table is the three-term truncation ... not quantified") | — |

### Notation conversions (source → S11)

- Gap: records' k (heights b, b − kr³) kept; MT (3.5)/(3.11) has ℓ = κr³/6, so **k = κ/6**, ℓ = kr³ (as in S8).
- Direction: u (records; MT (2.13)) = t of MT (3.1)–(3.4).
- Axial derivative ∂_u (records) → ∂_x; transverse ∇_Θ, D_Θ² → ∇_y, D_y² (following S8).
- Two-point kernels A_r, A_0, A_2, A_r^eld (records) → 𝒜_r, 𝒜_0, 𝒜_2, 𝒜^eld_r (calligraphic, as in S8); Ã_r → 𝒜̃_r[K]; bar A_r, bar A_2 ([FJ]) → 𝒜̄_r[K], 𝒜̄_2[K].
- Cusp variable κ = k/r = ℓ/r⁴ (records) is **not used** (it clashes with MT's κ); written through s = rℓ^{−1/4} (κ = s^{−4}). (𝒜^{eld} − 𝒜^{con})(b, s^{−4}, u) of [CU] → 𝒞(b, s, u).
- [FJ]: T_r, T3, T5 → τ_r, f_3, f_5; G_av, H_diff → G⁺_r, G⁻_r; endpoint sign σ → ε; U, V → Υ_1, Υ_2; W2 → X_2; O_r, E_r → 𝒪_r, ℰ_r; S, S2, T, T2 → Σ_o, Σ_o′, Σ_e, Σ_e′; s_o, s_e → ψ_o, ψ_e; φ_T → g_{Σ_e}; H(k,A), h_j → h(k,A), h_j; R = |A_upper|, M = 1+R → R_A, ⟨A⟩; U0 = 20000M → μ_*; P8(R) → P8(w); J = adj A kept.
- [CX]: transverse block T → A; eigenvalues x1, x2 → λ₁, λ₂; cone variables (s, t) → (ξ₁, ξ₂); exponent coefficients (al, be, ga) → (α, β, γ′); discriminant D → D_θ (D_{d−1} is reserved for the cone moments 4/3, 29/6 − √6); quadrant moments M(i, j) → I_{i,j} (M is the maximum); b-exponent L(x), R(x) → Λ₁(λ), Λ₀(λ) (L is the torus side); Isserlis polynomial F kept; κ_d → ω_d; c → c[K_ref].
- Leading constant c (records) → c_{d,L} (torus; S8), c[K_ref] (Gaussian kernel), c_{3,∞}, c_{3,24} (MT).

## (b) Corrections applied

Assignment list (PLAN "Known corrections"):
- **#9 (no status words)**: applied. No review/status vocabulary (author-side, candidate status, merged, accepted, review) in the LaTeX; "candidate" occurs only as the mathematical candidate density; "certified enclosure"/"certificate" describe the computer-assisted interval method, as in MT Theorem 2.3(iii).
- **#7 (|ε_24| < 10^{-216})**: used through MT (2.10) in the proof of `cor:ctwo-side24` (iii).
- #1–#6, #8: not applicable to S11.

Landed corrections in the three assigned folders (all checked with `git log --oneline origin/main -- <folder>`; no ERRATUM/AMENDMENT files exist in any of them):
- `c2_exact_20261001` **v1.1, `59a06b6` (OA-223-A-01)**: the cone integral is the fixed-cone surrogate Ã_r; analyticity is proved for Ã_r only; the typed kernel enters only through A_r = Ã_r + O(r³) at fixed admissible (b, k). **Applied**: Definition S11.1(ii), proof of Theorem S11.5 Step 2 ("Ã_r is real analytic"), Hypothesis S11.11(a).
- `c2_exact_20261001` `b43d36a` (ia.py docstring, Φ-series rounding bound): no mathematical effect; the Φ routine is not used in the c2 evaluation. Nothing to apply.
- `c2_exact_20261001` `f4e5312` (workflow triggers, provenance): no mathematical effect.
- `third_order_coefficient_20261001` **v1.2 (`4f0927f`)**: withdraws v1.1's claim that a rate in CU.4 suffices; lists three requirements. **Applied**: Remark S11.4 is presented as formal, makes no sufficiency claim, and refers to the expansions of S10 for the proof.
- `third_order_coefficient_20261001` **v1.4 (`1bfdf28`), item C2**: grid/r-set spreads do not bound the error; quote the errors against the closed forms. **Applied** in Remark S11.6(ii). Item D-1 (the ℓ^{1/2} reading of the Monte Carlo residual withdrawn): not applicable (no Monte Carlo in S11). Status rewording (S1–S5): not applicable (#9).
- `third_order_coefficient_20261001` **v1.4.1 (`ba10835`)**, C129-A1 (finite-r transverse covariance 2+σ_r², σ_r², 1 with σ_r² = r⁸/6144 + O(r¹⁰)): not applicable (the d = 3 floating checker's covariance is not described in S11). C129-A2 (numerical inputs vs analytic identifications): reflected (the identification (0.1) is cited to S10's fold expansion).
- `third_order_coefficient_20261001` v1.1 (`605af74`), v1.3 (`7e97018`): replay/status only; no mathematical change.
- `c2_finite_jet_transfer_20261001`: single commit `89117ea`; no later correction. [FJ-V]'s corrected test fixture (−5625/16) concerns the test suite only.

Not used (read for context only): `frontiers/c2_torus_transfer_20261001` (a separate, not-cited transfer record with sharper bounds, ε_3(24) ≈ 2.1e−111); `reviews/c2_interval_audit_20261001` (`008b0175`; an independent all-rational re-evaluation finding the nine published enclosures strictly contained — a review record, so not mentioned in the LaTeX).

## (c) Added steps

1. τ_r = −12 V⁻_corr (relation to MT (3.7)); checked algebraically.
2. Remark S11.3(i): |𝒜_2(b,k,u) − 𝒜_2(b,0,u)| ≤ C min(k,1)(1+|b|)^N e^{−cb²} from (F.2) (mean value theorem for k ≤ 1; the two Gaussian bounds for k > 1), giving absolute convergence of (S11.11). The sources assert convergence without this one-line bound.
3. Proof of Theorem S11.5, Step 1: the r²-coefficient of the birth integral equals the birth integral of the r²-coefficient (uniform Gaussian domination of Step 2), and the b- and k-integrals may be interchanged (polynomial × Gaussian, Step 3). This connects [CX]'s definition (b-integral outside) with [FJ]'s (J0) (birth integral first), which [FJ] declares "the same finite-part expression".
4. Proof of Theorem S11.5, Step 2: the eigenvalue formula ∫_{A<0} g p dA = π∫_{λ₂<λ₁<0} g p (λ₁−λ₂) is stated for conjugation-invariant g, p (implicit in [CX] §2.2). Jacobian checked (det of the 3×3 map = λ₁ − λ₂, θ ∈ [0, π)).
5. Proof of Theorem S11.5, Step 7: the d = 3 identity with MT (2.11)'s form 2^{−23/6}3^{−8/3}(29√6−36)Γ(1/6)π^{−5/2} (both equal 2^{−10/3}3^{−13/6}(29−6√6)Γ(1/6)π^{−5/2}).
6. Proof of Theorem S11.7, Step 3: the interpretation of p_o e^{−ak²} q(A) as the joint density at the targets and of ψ_o, ψ_e as r²-derivatives of the log densities; the remark that 36k²Δ² contributes only through the scores because Δ is a function of the conditioning variable A; the parity argument written as h(−k, A) = h(k, A).
7. Proof of Theorem S11.7, Step 4: absolute convergence of (J0) from (J1).
8. Proof of Corollary S11.10: the image sum written explicitly.
9. Corollary S11.12(iii): the c_{3,24} enclosure derived from MT (2.10)–(2.11), and the interval for c_2/c_{3,24} by exact interval division.
10. Verified but not written in the LaTeX: [FJ] §6.1's "sharpened" mean bounds 1000|A_upper| (even) and 10000|k| (odd) follow from the row norms ≤ √6·106 and ≤ √4·106 with ‖Σ⁻¹‖ ≤ 3 (779 ≤ 1000; 7632 ≤ 10000); |μ̇| ≤ 40000δ · 24⟨A⟩ < 2·10⁶δ⟨A⟩. All other constants of [FJ] §§6–7 were recomputed (Schur 14,409,601δ; 72·540090 = 38,886,480; 226469/180; 824629750641408; p bounds 1.08e40 and 1.308e43; 1.952e47; polynomial factors at L = 24; E_3(10) ≈ 8.90e−11; 8 − √58 ≈ 0.3842; θ = 2.354098… by quadrature) and agree.

## (d) Issues (not fixed in the LaTeX)

1. **MT Theorem 2.3(ii) wording.** "c_2 is the fold-scale finite part of the two-point contact kernel." In the records the *contact kernel* is 𝒜_0; c_2 is the finite part of (1/3)∫∫∫𝒜_2 k^{−4/3}, 𝒜_2 being the r²-coefficient of the two-point kernel 𝒜_r. Suggested: "the fold-scale finite part of the r²-coefficient of the two-point kernel (Supplement S11)".
2. **MT prints no interval for c_2/c_{3,24}**, only "≈ 3.8595" in (2.14). The records give c_2/c for the Gaussian kernel ([CX]: [3.859496174547095, 3.859496174548665]); S11 Corollary S11.12(iii) derives, for L = 24 under Hypothesis S11.11(a), c_2/c_{3,24} ∈ [3.859496174547456, 3.859496174548331]. The integrator may wish to quote this in (2.14) or Theorem 2.3(iii).
3. **[CX] §3.1 scope of the Gamma formula.** It states "For −2 < s < 0, ∫(k^l e^{−12k²} − [l = 0])k^{s−1}dk = Γ((l+s)/2)12^{−(l+s)/2}/2" and then uses "s = 1/3 for c". For s = 1/3 the subtraction is absent and the plain Euler integral (l + s > 0) applies; with the subtraction the l = 0 integral would diverge at ∞. No effect on any result. S11 (S11.15) states the two cases separately. Suggested fix in [CX]: "for s > 0 without the subtraction; for −2 < s < 0 with it".
4. **[FJ] §7 "the corrected parent identification"** is not defined in [FJ]. S11 renders it as Hypothesis S11.11(b) (c_2 is the ℓ^{1/3} coefficient of the elder and candidate densities, as stated in S10). If a specific corrected parent statement is meant, the hypothesis should name it.
5. **[FJ] §6.1 mean bounds** ("can be sharpened separately to 1000|A_upper| ... and 10000|k|") are stated without derivation; they are correct (see Added step 10). Suggest adding the row-norm justification to [FJ].
6. **MT uses κ for two objects**: the cubic coefficient (3.5)/(3.11) (ℓ = κr³/6) and the cusp variable κ = ℓ/r⁴ in §5 ("At large κ = ℓ/r^4"). S11 avoids κ. Suggest renaming the §5 variable.
7. **Coordination with S10.** S11 restates the fold expansion as (S11.10) "(Section sec:third)" and points Hypothesis S11.11(a) to "the comparison of the typed kernel with its fixed-cone surrogate that accompanies the fold expansion in Section sec:third". If S10 does not state the surrogate comparison (the paragraph after Lemma F of [T]), the pointer in (a) should be adjusted; if S10 states the fold expansion as a numbered lemma, (S11.10) can be replaced by a reference to it.
8. **No DLMF citation.** [FJ] §4 cites NIST DLMF 5.2.1 and 5.5.1, and the Stirling remainder bound (first omitted term, positive real argument) is DLMF §5.11(ii). The assignment fixed the files to write, so no `supp_refs_s11.bib` was created; the integrator may add a DLMF entry and cite it in Step 8 of the proof of Theorem S11.5 and in (S11.28).

## Compile check

`chk_s11/test_s11.tex` = preamble of `supp_main.tex` + `\input{../supp_s11_ctwo}`; `pdflatex` run twice (no bibliography needed: S11 has no `\cite`). Result: 11 pages (10.4 pages of text), no LaTeX errors, **no overfull boxes**, no undefined labels of S11; the only undefined references are the other sections' labels `sec:remainder`, `sec:cusp`, `sec:third` (expected in the standalone check).

---

> **Integration note (draft 5).** The concordance below is the S12/S13 writer's file `CONC_s12.md`, revision 2, appended unchanged. Its file names and hashes are those of the writer's outputs: `supp_s12_parent.tex` there is `writer_originals/supp_s12_parent.rev2orig.tex` here, and `chk_s12/verify_s12.py` is `checks/s12_parent_check.py`. The changes made at integration are listed in "Checkpoint draft 5" above.

# CONC_s12: concordance for `supp_s12_parent.tex` (`\label{sec:parent}`) and `supp_s13_cap.tex` (`\label{sec:cap}`)

Dylan Roy — delegated AI work. Writer: a subagent of the author's session (Anthropic Claude), 14 October 2026. Author-side drafting; no push, commit or comment. Scientific effect NONE: this is a transcription, and it accepts nothing.

The two files make the supplement self-contained for the statements that S0, S5–S10 import from the companion parent record by `\cite{parentP}`. S12 transcribes the parent record; S13 transcribes the deterministic separating-cylinder ("marked-cylinder cap") theorem on which the parent relies. Statement and equation numbers below are those printed in the compile check `chk_s12/`, where 11 stub sections carry the labels `sec:setting` … `sec:ctwo`, so S12 and S13 receive their final numbers if they are appended after S11.

**Revision 2 (14 October 2026, on the author's instruction relayed by the coordinator).** The transcription now follows the *reconciled object* of REC §1: P read with E1, with P §9 replaced by E2, with P l. 165 read with W1, and with P §1's "sufficiently small r" including the embedding bound `r < L/(4√2)`. These are corrections C1–C4 in (b). Four steps that REC §5 supplies where P only sketches are added (A11–A14 in (c)). Issues I-1, I-2 and I-4 of revision 1 are resolved by C2, C3 and C4. **No statement or equation number changed**: the 133 labels of S12/S13 print the same numbers as in revision 1 (checked by diffing the `.aux` label lists).

## Sources

| Key | Source | Path (Math- `origin/main`) | Git blob | SHA-256 (first 16) | Role |
|---|---|---|---|---|---|
| P | Parent record "Uniform matrix-cap selection and unrestricted short-lifetime asymptotics" (UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1), 494 lines, §§1–16 | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d` | `9350ad6eaba6626b` | transcribed in S12 |
| CAP | Marked-cylinder cap (MARKED-CYLINDER-CAP-20260924-v1), 230 lines | `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md` | `0633aca3` | `0bf922b9203c2908` | §§1–5 transcribed in S13 |
| CPI | Cap pairing identities (GROK-HEAVY-CAP-PAIRING-20260926-v1) | `imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md` | `96407c95` | `69ff3a12d847b9ee` | details of CAP (7), (11); P (7.3) |
| E1 | Congruence erratum (UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1-ERRATUM-1), merged | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` | `213594d6` | `bad7ef609c4ad8c4` | correction C1 |
| E2 | §9 replacement v1.1 (D1-SECTION9-BOREL-REPAIR-20260925-v1.1), 147 lines | `reviews/d1_section9_borel_repair_20260925/REPAIR.md` | `fe9b9ce4` | `845abf9f9c99d672` | replaces P §9: correction C2 |
| REC | Chain reconciliation (D1-CHAIN-RECONCILIATION-20260928-v1), 309 lines | `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md` | `75da2597` | `451b9d7ffee072a7` | §1 reading rule (l. 25–32); §5: W1 (C3), embedding radius (C4), discharges (A11–A14) |
| EP | Erratum pointer | `imports/lifetime_parent_20260925/ERRATUM_POINTER.md` | `0ba9acc4` | `e29e7733e9f34289` | provenance only |
| RM | Folder README | `imports/lifetime_parent_20260925/README.md` | `2d7a60b4` | `0cbc77d96d1b4f56` | provenance only |

The scratchpad copies (`s12/PARENT_dfed3b8d.md`, …, `s12/E2_REPAIR_fe9b9ce4.md`, `s12/RECONCILIATION_75da2597.md`) were hashed before use (`git hash-object`, `sha256sum`): all eight match the blobs above; P and CAP match the SHA-256 values in RM, and P, CAP, E1 and E2 match the full SHA-256 values in REC's identity table (REC l. 17–22). "P l. n", "CAP l. n", "E2 l. n" and "REC l. n" are line numbers of these files.

**Reading rule (REC §1, l. 25–32), applied in revision 2.** Read P §5 with E1 (congruence factor `diag(r^{-1/2}, I)`); read P §9 as replaced by E2; read P l. 165 with W1 (REC §5 l. 153–161); read P §1's "sufficiently small r" as including `r < L/(4√2)` (REC §5 l. 163–166). "No formula of P changes" (REC l. 32). The supplement sections were read at `/home/claude/wt-supp/docs/manuscript/side24_v3_supplement_draft_20261013/` (draft 4).

## (a) Statements and numbered displays

### S12 `supp_s12_parent.tex`

| Label | Printed | Content | Source |
|---|---|---|---|
| `sec:parent` (intro) | S12 | purpose, scope (all `d≥2`, `L>0`), plan, existential constants | assignment; P l. 1–7 (purpose), l. 45 and l. 52 (no numerical constants); P §16 l. 484 |
| `ssec:parent-setting` | §S12.1 | — | P §1 |
| `def:parent-model` | Def. S12.1 | field, covariance, windows, constants convention; side-24 instance | P l. 11–16 |
| `eq:parent-KL` | (S12.1) | `K_L` | P l. 13–14 |
| `def:parent-pinned` | Def. S12.2 | frame, coordinates, `0<r<L/(4√2)` and the embedding of the cylinder, pins, `P^0`, index, `W_r`, `Z_r`, `P^MS_r`, `p_r` | P l. 18–29; radius and its one-line proof REC §5 l. 163–166 (C4) |
| `eq:parent-pins` | (S12.2) | pins | P l. 20–22 |
| `eq:parent-W` | (S12.3) | `W_r`, `Z_r`, `dP^MS_r` | P l. 26–27 |
| `thm:parent-A` | Thm. S12.3 | Theorem A, with `r_* < L/(4√2)` | P l. 31–33; `r_*` below the radius: REC l. 165–166 (C4) |
| `eq:parent-A` | (S12.4) | (1.1) | P (1.1) l. 33 |
| `thm:parent-B` | Thm. S12.4 | Theorem B | P l. 37–43 |
| `eq:parent-B` | (S12.5) | (1.2) | P (1.2) l. 39–41 |
| `thm:parent-C` | Thm. S12.5 | Theorem C | P l. 47–52 |
| `eq:parent-C` | (S12.6) | (1.3) | P (1.3) l. 49–50 |
| `rem:parent-scopeABC` | Rem. S12.6 | scope of A–C | P l. 35, l. 45 (first three sentences), l. 52 |
| `ssec:parent-spectrum` | §S12.2 | — | P §2 |
| `lem:parent-spectrum` | Lem. S12.7 | Poisson summation, `a_n>0`, summability, smooth version, moments | P l. 56–61 |
| `eq:parent-an` | (S12.7) | `a_n` | P l. 58–59 |
| `lem:parent-jetrank` | Lem. S12.8 | finite-jet rank (distinct sites; rotated frame at one site); "distinct"; Hessian entries | P l. 63–65 |
| paragraph after S12.8 | — | pins at distinct points; residual jet lists; finite modes would not suffice | P l. 67 |
| `ssec:parent-frame` | §S12.3 | — | P §3 |
| `def:parent-rows` | Def. S12.9 | `O_r`, rows `U^(0..3)`, `V_j^(0,1)` | P l. 71–86 |
| `eq:parent-rows` | (S12.8) | (3.1) | P (3.1) l. 76–81 |
| `lem:parent-frame` | Lem. S12.10 | invertible `T_r`, `|det T_r|`, exact target | P l. 88–94 |
| `eq:parent-detT` | (S12.9) | (3.2) | P (3.2) l. 88–90 |
| `eq:parent-target` | (S12.10) | (3.3) | P (3.3) l. 92–94 |
| sentence after S12.10 | — | exact transformation; factor 12 kept | P l. 96 |
| `lem:parent-contact` | Lem. S12.11 | (a) contact jet, covariance convergence, `v_r→v_0`; (b) uniform positive definiteness of `Σ_r`; (c) enlarged covariance with `A_M`, density bound | (a) P l. 98–103; (b) P l. 105–107; (c) P l. 109–113; the uniform Gramian gap in the proof of (b), (c): REC §5 l. 144 (A12) |
| `eq:parent-U0` | (S12.11) | (3.4) | P (3.4) l. 98–101 |
| `eq:parent-Adens` | (S12.12) | (3.5) | P (3.5) l. 111 |
| `ssec:parent-residual` | §S12.4 | `r_1 ∈ (0, L/(4√2))` | P §4; `r_1` added step A3; its bound by `L/(4√2)`: C4 |
| `lem:parent-moments` | Lem. S12.12 | uniform conditional moments | P l. 117–119 |
| `eq:parent-41` | (S12.13) | (4.1) | P (4.1) l. 119 |
| `lem:parent-residual` | Lem. S12.13 | independent residual, `J_r`, norm bounds | P l. 121–134 |
| `eq:parent-42` | (S12.14) | (4.2) | P (4.2) l. 123 |
| `eq:parent-43` | (S12.15) | (4.3) | P (4.3) l. 131–134 |
| sentences after S12.13 | — | no inverse Hessian moment | P l. 136 |
| `ssec:parent-normalizer` | §S12.5 | — | P §5 |
| `lem:parent-endpoint` | Lem. S12.14 | `α_i`, `β_i`, `A_i`, `H_i`; (5.1)–(5.3) (deterministic) | P l. 140–161 |
| `eq:parent-51` | (S12.16) | (5.1) | P (5.1) l. 146–149; proof sketch l. 151 |
| `eq:parent-52` | (S12.17) | (5.2) | P (5.2) l. 153–156; height identity = CAP l. 55–58 |
| `eq:parent-53` | (S12.18) | (5.3) | P (5.3) l. 158–161 |
| `prop:parent-normalizer` | Prop. S12.15 | (a) moments, `sup E^0(W_r/r^2)^2 < ∞`; (b) `Z_r/r^2 → z_0` (convergence in law, Slutsky); (c) floor | (a) P l. 163, UI step REC §5 l. 140 (A11); (b) P l. 165–176 read with E1 (C1) and W1 = REC §5 l. 153–161 (C3), difference entries A14; (c) P l. 178–182 |
| `eq:parent-54` | (S12.19) | (5.4) | P (5.4) l. 171–174 |
| `eq:parent-55` | (S12.20) | (5.5) | P (5.5) l. 180 |
| `eq:parent-congr` | (S12.21) | corrected congruence and `det = det H_i / r` | **E1** l. 19–31 (replaces P l. 169, first sentence; correction C1) |
| `ssec:parent-soft` | §S12.6 | — | P §6 |
| `lem:parent-soft` | Lem. S12.16 | Schur complement, (6.1), saddle bound, (6.2) | P l. 186–208; `det H_M = rα_M det B'`, `0 ≺ −B' ⪯ B`: REC §5 l. 145 (A13) |
| `eq:parent-61` | (S12.22) | (6.1) | P (6.1) l. 194–196 |
| `eq:parent-HS` | (S12.23) | `|det H_S|` bound | P l. 200–203 (unnumbered display) |
| `eq:parent-62` | (S12.24) | (6.2) | P (6.2) l. 205–208 |
| paragraph after S12.16 | — | mixed-square term; no entry independence | P l. 210 |
| `ssec:parent-layer` | §S12.7 | cap event, role of the cap theorem | P l. 214–219 |
| `eq:parent-G` | (S12.25) | `G_r` | P l. 216–217; CAP (1) l. 23–26 |
| `lem:parent-eigen` | Lem. S12.17 | eigenvalue coordinates; (7.2) | P l. 227–232 |
| `eq:parent-72` | (S12.26) | (7.2) | P (7.2) l. 229–232 |
| `prop:parent-layer` | Prop. S12.18 | `P^MS_r(G_r^c) ≤ Cr^3` with full proof | P l. 221–274 |
| `eq:parent-78` | (S12.27) | (7.8) | P (7.8) l. 272–274 |
| `eq:parent-71` | (S12.28) | (7.1) | P (7.1) l. 221–223 |
| `eq:parent-73` | (S12.29) | (7.3) | P (7.3) l. 234–238; CPI l. 36–43 |
| `eq:parent-74` | (S12.30) | (7.4) | P (7.4) l. 240–247 |
| `eq:parent-75` | (S12.31) | (7.5) | P (7.5) l. 249–256 |
| `eq:parent-76` | (S12.32) | (7.6) | P (7.6) l. 258–263 |
| `eq:parent-77` | (S12.33) | (7.7) | P (7.7) l. 265–270 |
| `ssec:parent-generic` | §S12.8 | — | P §8 |
| `lem:parent-generic` | Lem. S12.19 | pinned genericity | P l. 278–288 |
| `lem:parent-mark` | Lem. S12.20 | (a) Borel elder event; (b) Morse-locus meaning; (c) finite bars = pairs with `e=1` | (a), (b) P l. 290–294; (c) P §14 l. 445; background P l. 494 (Curry) |
| `eq:parent-df` | (S12.34) | maximin level `d_f(M)` | P l. 292 |
| `lem:parent-capimpl` | Lem. S12.21 | on `G_r` (Morse locus) the elder partner is `S`; one branch | P l. 214, l. 296; CPI l. 47–49 (hypotheses: embedded chart, Morse/distinct values) |
| proof of Thm. S12.3 | — | Theorem A | P l. 296 ("(7.8) and the full-probability locus prove Theorem A") |
| `ssec:parent-KR` | §S12.9 | weighted pair Kac–Rice for the global mark; finite-measure route; no local cap event substituted | E2 l. 25 (title), l. 35 (last sentence), l. 102 (C2; replaces P l. 300) |
| `prop:parent-KR` | Prop. S12.22 | (a) regular conditional law `𝒫_{x,y,z}` of the whole `C^2` field, continuity on cylinder functionals, marked Kac–Rice identity for bounded Borel `h`, both sides finite measures; (b) Borel elder/type marks, disintegration, `W_r`, intensities with `E^0[W_r]` and `p_r`, no second determinant factor; (c) exhaustion | **E2 §2** (C2): statement from l. 27–35, §9.1 l. 39–55, §9.2 l. 59–74, §9.4 l. 96–110; proof: setting l. 27–35, §9.1 l. 37–55, §9.2 l. 57–74, §9.3 l. 76–92, §9.4 l. 94–110 |
| `eq:parent-KR` | (S12.35) | the identity `∫h dμ_1 = ∫h dμ_2`, written out | E2 (9.1) l. 72–74 with `μ`, `ν` of l. 59–70 (in revision 1 this display was added step A2) |
| `ssec:parent-ledger` | §S12.10 | — | P §10 |
| `prop:parent-ledger` | Prop. S12.23 | (a) pin density; (b) intensity, power count; (c) transverse invariance; (d) contact limit | P l. 312–335 |
| `eq:parent-101` | (S12.36) | (10.1) | P (10.1) l. 312–315 |
| `eq:parent-102` | (S12.37) | (10.2) | P (10.2) l. 319–322; powers l. 324–327 |
| `eq:parent-103` | (S12.38) | (10.3) | P (10.3) l. 329–331 |
| sentence after S12.23 | — | selected intensity gives the domination | P l. 335 |
| `ssec:parent-B` | §S12.11 | — | P §11 |
| `prop:parent-push` | Prop. S12.24 | (11.1), (11.2) | P l. 339–347 |
| `eq:parent-111` | (S12.39) | (11.1) | P (11.1) l. 339–341 |
| `eq:parent-112` | (S12.40) | (11.2) | P (11.2) l. 343–347 |
| proof of Thm. S12.4 | — | Theorem B | P l. 349–363 |
| `eq:parent-113` | (S12.41) | (11.3) | P (11.3) l. 351–353 |
| `eq:parent-diff` | (S12.42) | difference bound | P l. 357–361 (unnumbered display) |
| paragraph after the proof | — | identified integral; no rate for `A_r→A_0` | P l. 355, l. 363 |
| `ssec:parent-moments` | §S12.12 | — | P §12 |
| `cor:parent-moments` | Cor. S12.25 | (12.1)–(12.3), `q ≤ −2/3`, unrestricted forms | P l. 367–384 |
| `eq:parent-121` | (S12.43) | (12.1) | P (12.1) l. 369–370 |
| `eq:parent-122` | (S12.44) | (12.2) | P (12.2) l. 374–375 |
| `eq:parent-123` | (S12.45) | (12.3) | P (12.3) l. 379–380 |
| paragraph after S12.25 | — | counting measure; threshold `p<2/3`; compact rates not unrestricted | P l. 382, l. 384 |
| `ssec:parent-majorant` | §S12.13 | choice of `r_0` (below the injectivity scale, below `L/(4√2)`, below one); no global floor | P l. 388; `r_0 < L/(4√2)`: REC l. 30 with §5 l. 163–166 (C4) |
| `lem:parent-growth` | Lem. S12.26 | (13.1)–(13.3) | P l. 390–403 |
| `eq:parent-131` | (S12.46) | (13.1) | P (13.1) l. 392–393 |
| `eq:parent-132` | (S12.47) | (13.2) | P (13.2) l. 397 |
| `eq:parent-133` | (S12.48) | (13.3) | P (13.3) l. 401 |
| `lem:parent-majorant` | Lem. S12.27 | (a) determinant bounds; (b) (13.4); (c) `Z_r>0`, `p_r→1` | P l. 405–415 |
| `eq:parent-detbound` | (S12.49) | `|det H_i|/r`, `W_r/r^2` bounds | P l. 407–408 (unnumbered display) |
| `eq:parent-134` | (S12.50) | (13.4), majorant `Ψ` (P's `H`) | P (13.4) l. 412–413 |
| `prop:parent-near` | Prop. S12.28 | (13.5), domination, (13.6), monotone `c_{B,K} ↑ c_{d,L}` | P l. 417–433 |
| `eq:parent-135` | (S12.51) | (13.5) | P (13.5) l. 419–422 |
| `eq:parent-136` | (S12.52) | (13.6) | P (13.6) l. 428–431 |
| `ssec:parent-far` | §S12.14 | — | P §14 |
| `lem:parent-far` | Lem. S12.29 | far covariance, target density, moments, Jacobian one, (14.1); near ∪ far | P l. 437–445 |
| `eq:parent-141` | (S12.53) | (14.1) | P (14.1) l. 441 |
| proof of Thm. S12.5 | — | Theorem C | P l. 445 |
| `ssec:parent-coef` | §S12.15 | parity, not isotropy | P l. 449 |
| `lem:parent-parity` | Lem. S12.30 | odd/even independence, (15.1), factorization | P l. 449–461 |
| `eq:parent-151` | (S12.54) | (15.1) | P (15.1) l. 453–454 |
| `eq:parent-factor` | (S12.55) | `π_0 = p_{(f,V)} p_G φ_τ(12k)`, `z_0` | P l. 460–461 (unnumbered display) |
| `prop:parent-coef` | Prop. S12.31 | (15.2) | P l. 463–478 |
| `eq:parent-15a` | (S12.56) | `c_{d,L}` with the `k`-integral | P l. 465–467 (unnumbered) |
| `eq:parent-15b` | (S12.57) | gamma integral | P l. 471–472 (unnumbered) |
| `eq:parent-152` | (S12.58) | (15.2) | P (15.2) l. 476–478 |
| paragraph after S12.31 | — | finiteness, positivity; factor 24 ≠ `L`; amplitude `a^{−2/3}` | P l. 480 |
| `ssec:parent-scope` | §S12.16 | — | P §16 |
| `rem:parent-scope` | Rem. S12.32 | what is not proved (mathematical items) | P l. 486 (first five items) |
| `rem:parent-sources` | Rem. S12.33 | background: AAL §§2, 7; Curry §3; cap = S13 | P l. 494 |

### S13 `supp_s13_cap.tex`

| Label | Printed | Content | Source |
|---|---|---|---|
| `sec:cap` (intro) | S13 | role; deterministic; all `d`, births, gaps, frames | CAP l. 7–11, l. 47 (first sentence); P l. 7, l. 214 |
| `def:cap-norms` | Def. S13.1 | cylinder, partial-block norms, `d=2` case | CAP l. 11–19 |
| `eq:cap-D` | (S13.1) | `D_r` | CAP l. 13 |
| `eq:cap-Mj` | (S13.2) | `M_j` (written in S7's index form, see conversions) | CAP l. 17 |
| `thm:cap-main` | Thm. S13.2 | separating cylinder: (a) ridge and types, (b) cap, (c) elder pairing, (d) branches | CAP l. 15, l. 21–28; (a) l. 89–91, l. 129; (b) l. 131–155; (c) l. 159–161, l. 167; (d) l. 163 |
| `eq:cap-cond` | (S13.3) | conditions (1) | CAP (1) l. 23–26 |
| `ssec:cap-normalized` | §S13.1 | normalization `k=1/6` | CAP l. 51 |
| `eq:cap-cond6` | (S13.4) | (4) | CAP (4) l. 53 |
| `lem:cap-forced` | Lem. S13.3 | Hermite identity, weighted mean 2, `M_3≥2`, `x_0`, (5) | CAP l. 55–68 |
| `eq:cap-hermite` | (S13.5) | Hermite identity | CAP l. 57–58 |
| `eq:cap-5` | (S13.6) | (5) | CAP (5) l. 64–66 |
| `lem:cap-ridge` | Lem. S13.4 | (6)–(10), the ridge `h` | CAP l. 72–106; CPI l. 7–19 |
| `eq:cap-6` | (S13.7) | (6) | CAP (6) l. 74; CPI l. 19 |
| `eq:cap-7` | (S13.8) | (7) | CAP (7) l. 80–81; CPI l. 11–17 (sharpness at `x=2r`) |
| `eq:cap-8` | (S13.9) | (8) | CAP (8) l. 91 |
| `eq:cap-9` | (S13.10) | (9) | CAP l. 95, (9) l. 99–100 |
| `eq:cap-10` | (S13.11) | (10) | CAP (10) l. 104 |
| `lem:cap-third` | Lem. S13.5 | (11), (12), zeros of `F`, critical points, types | CAP l. 110–129; CPI l. 21–34 |
| `eq:cap-11` | (S13.12) | (11) | CAP (11) l. 112–113; CPI l. 30–31 |
| `eq:cap-12` | (S13.13) | (12) | CAP l. 117–125 (adverse part l. 117–120, (12) l. 122–125) |
| `lem:cap-boundary` | Lem. S13.6 | cap boundary: (13), (14), faces, ridge segment | CAP l. 131–155 |
| `eq:cap-13` | (S13.14) | (13) | CAP (13) l. 137–138 |
| `eq:cap-14` | (S13.15) | (14), stated for `−2r ≤ x ≤ r/2` | CAP (14) l. 144–145 (range: issue I-9) |
| `ssec:cap-proof` | §S13.2 | proof of Thm. S13.2 incl. scaling | CAP §5 l. 159–165 |
| `rem:cap-margins` | Rem. S13.7 | margins for general `k`; frames; embedding on the torus for `r < L/(4√2)`; no invariance claim for a random field | CAP l. 165 (constants), l. 167; torus embedding: REC §5 l. 163–166 (C4, pointer to Def. S12.2) |

Every numbered display has a label and appears above.

## Notation conversions (source → S12/S13)

| Source | Here | Reason |
|---|---|---|
| `u` (axial unit vector, P) | `t`; frame `R=(t,…)`, `t=Re_1` | S7's conversion (main-text direction); `u` is also CAP's `2/k+2/k^2` |
| `X = R^d/(LZ^d)` | `T^d_L` | S7 |
| `Q = Q_{r,b,k,R}`, `E_Q`, "Q-law" | `P^0 = P^0_{r,b,k,R}`, `E^0`, "`P^0`-law" | S7 (`E^0`); `P^0` added for probabilities |
| `Q^W` | `P^MS_r`, `E^MS_r` | S7 |
| windows `B=[b_-,b_+]`, `K=[k_-,k_+]`; `c_{B,K}`, `C_{B,K}` | `\mathcal B`, `\mathcal K`; `c_{\mathcal B,\mathcal K}`, `C_{\mathcal B,\mathcal K}` | P uses `B` also for `−A_M` (§6) and `K` for a constant (§7); S7 keeps `B=−A_M` |
| `A_r = D_y^2 f(M)` (P §§3–4) | `A_M` | P's own §5 name for the same matrix; `A_r` clashes with the intensity |
| intensities `A_r(b,k,u)`, `A_0`, `A^*` (P §§10–15) | `\mathcal A_r`, `\mathcal A_0`, `\mathcal A^*` | S7; P uses `A_0` also for the contact Hessian |
| `A_0` (contact transverse Hessian) | `A_0` | S7 |
| `H(b,k)` (majorant (13.4)) | `Ψ(b,k)` | S7 (Hessians `H_M`, `H_S`, `H_i`) |
| `h = M3` (P §§6–7) | `M_3` | S7 |
| `M3`, `M4` | `M_3`, `M_4` | S7 |
| `K` (absorbing `√m`, P §7) | `K_*` | S7 |
| `D = 4K^2/(3k_-)` | `D_* = 4K_*^2/(3k_-)` | S7 style |
| `E` (undefined, P (7.3)) | `E_* = (3/2)K_*` | added step A6; S7 did the same |
| `U = J_r + Λ` | `Υ` | S7 (`U_r` is the observation vector) |
| `G_r` | `\mathcal G_r` | S7 |
| `D_r = diag(r^{-1/2}, I)` (E1) | `Δ_r^{-1}` with `Δ_r = diag(√r, I_m)` | S7 notation; same matrix |
| `a=-r/2`, `c=r/2` (points on the axis, P §3) | `M`, `S` (`x=∓r/2`) | `c` is a generic constant in S12 (CAP's `a`, `c` are kept in S13) |
| rows `U0,U1,U2,U3`, `Vj0,Vj1`; jet `U0*` | `U^(0)…U^(3)`, `V_j^(0)`, `V_j^(1)`; jet `U_0` | P's `U0` (row) clashes with S7's `U_0` (jet) |
| `Σ_r(R)`, `π_r(R;v)`, `z_0(b,k,R)`, `p_r(b,k,R)` | kept | no clash |
| "density_Q(A)" (3.5) | `ρ_r(A)` | S7 |
| `μ_r`, `B_r(z)`, `g_r`, `J_r`, `K_0`, `C_0`, `c_0`, `T_r`, `O_r` | kept | no clash in S12 |
| directed separation `h = r u` (P §10) | `S − M = rt` | `h` was `M_3` |
| path parameter `t` in `min_t f(γ(t))` (P §8) | `s'` | `t` is the direction, `s=f(S)` |
| cutoff `t` in (12.1)–(12.3) | `ε` | S7 |
| `t_u = ∂_u^3 f`, `V_u`, `A_u`, `τ_u`, `D_u`, `p_{V_u}` (P §15) | `∂_t^3 f(0)`, `V_t`, `A_t`, `τ_t`, `D_t`, `p_{V_t}` | direction `t`; as S7 (S7.23) |
| substitution `t = 12k` (P §15) | `w = 12k` | `t` is the direction |
| `ν_cand`, `ν_eld` (Thm B); `ν_cand^all`, `ν_eld^all` (Thm C) | kept (`\nu_\cand`, `\nu^{\rm all}_\cand`, …) | S7's `ν_cand` is S12's `ν^all_cand` |
| CAP gap mark `κ` | `k` | main-text `κ = 6k` (S7); CAP's `κ` is P's `k` |
| CAP `m = M_3`, `n = M_4` | `M_3`, `M_4` | `m = d−1` throughout the supplement |
| CAP `k = 8m−5` | `q = 8M_3 − 5` | clash with the gap mark |
| CAP `u = 2/k + 2/k^2` | `ω = 2/q + 2/q^2` | clash with the direction `u` of S8–S10 |
| CAP cylinder `D` | `\mathcal D_r` | S7 |
| CAP cap `C` | `\mathcal C` | generic constants `C` |
| CAP exponents `a+c=j` in `M_j` | `j−i`, `i` | S7 (S7.9); `a`, `c` are the pin abscissas |
| CAP integration variable `t` in (7) | `x'` | readability |
| E2 pair domain `D`, points `t = (x,y) ∈ D` | `Ω`, `(x,y) ∈ Ω` | `D` is used for derivatives and cylinders; `t` is the direction |
| E2 `G(x,y) = (grad f(x), grad f(y))` | `𝖦(x,y)` (sans-serif) | `G = ∇f(0)` in §S12.15 and the cap event `\mathcal G_r` |
| E2 `E = C^2(X)` | `C^2(T^d_L)` written out | `E` is the expectation |
| E2 `Delta(x,y) = |det H_x det H_y|` | written out | `Δ_r = diag(√r, I_m)` (S7) |
| E2 conditional mean `m_(t,z)`, residual `R_t` | `\bar f_{x,y,z}`, `𝖱_{x,y}` | `m = d−1`; `R` is the frame |
| E2 kernel `K(t,z,A)` | `𝒫_{x,y,z}(A)` | `K_L`, `K_0`, `K_*`, `\mathcal K` |
| E2 measures `mu`, `nu` | `μ_1`, `μ_2` | `μ_r` (regression mean, §S12.4) and the densities `ν` |
| E2 class `H` of functions; dense set `Q` | `\mathcal H`; `\mathcal Q` | Hessians `H_x`; the law `Q` |
| E2 `Q`, `E_Q W_r`, `Q^W` | `P^0`, `E^0[W_r]`, `P^MS_r` | as for P |
| REC `D_r = diag(r^{-1/2}, I)` (reading rule, W1) | `Δ_r^{-1}` | S7 |
| REC cap cylinder `D` (embedding radius) | `\mathcal D_r` | S7 |
| REC `U_r ⊕ vec A_r`, measures `μ_r → δ_0`, range `[0, r_0] × O(d)` (A2 discharge) | `(U_r, A_M)`, `η_r → δ_0`, `[0, \bar r] × O(d)` | `A_r` is `A_M` here; `μ_r` is the regression mean; `r_0` is fixed in §S12.13 |
| REC `B' = A − (r/α)ββ^T` (A5 discharge) | `B' = A_M − (r/α_M)β_Mβ_M^T` | indices made explicit |

## (b) Corrections applied

| No. | Correction | Basis | Where |
|---|---|---|---|
| C1 | P l. 169, "Here the congruence uses diag(sqrt(r),I), so its off-diagonal entries are sqrt(r) beta_i", is replaced by E1: the congruence is `D_r H_i D_r` with `D_r = diag(r^{-1/2}, I)` (= `Δ_r^{-1}`, `Δ_r = diag(√r, I_m)`), giving the matrix `[[α_i, √r β_i^T],[√r β_i, A_i]]` with determinant `det(H_i)/r`. E1's limits `diag(∓6k, A_0)` (E1 l. 35–38) are read in law, as C3 requires. E1: no formula (5.3)–(5.5) changes. | Assignment (revision 1); REC reading rule l. 27. `verify_s12.py` checks the corrected congruence exactly for `m = 1, 2, 3` and that the displayed factor does not give this matrix. | proof of Prop. S12.15(b), (S12.21) |
| C2 | P §9 (l. 298–308) is replaced in full by E2 §2 (l. 23–110): the regular conditional law of the whole `C^2` field (§9.1), the two finite measures and the identity on continuous cylinder weights by AAL Thm. 7.1, with finiteness from AAL Thm. 2.2 (§9.2), generation of the Borel σ-algebra and the functional monotone-class closure (§9.3), the Borel elder and type marks, the disintegration and the exhaustion (§9.4). The statement of Prop. S12.22 is changed accordingly: (a) now asserts the kernel `𝒫_{x,y,z}`, its continuity on cylinder functionals and the finiteness of both sides; (b) follows E2 §9.4 ("no second determinant factor"); P's sentence on the canonical density version (l. 306) is not in E2 and is dropped (issue I-14). The subsection's opening paragraph is E2 l. 35 and l. 102 in place of P l. 300. | REC reading rule l. 28 ("Read P §9 as replaced by E2"); coordinator's instruction. | §S12.9: opening paragraph, Prop. S12.22 statement and proof; (S12.35) keeps its number |
| C3 | P l. 165–167 ("the congruence-scaled endpoint Hessians converge, in probability uniformly over parameters, to diag(−6k, A_0) … diag(6k, A_0)") is read with W1: the differences `Δ_r^{-1}H_MΔ_r^{-1} − diag(−6k, A_M)` and `Δ_r^{-1}H_SΔ_r^{-1} − diag(6k, A_M)` tend to 0 in probability, by (5.1), (5.2) and `‖A_S − A_M‖ ≤ rM_3`; `A_M → A_0` in law; Slutsky gives joint convergence in law; continuous mapping applies off the null set `{det A_0 = 0}`. The type indicator therefore converges in law, and P's subsequence argument (l. 176) is stated as an argument in law, as REC notes it already is. | REC reading rule l. 29; REC §5 l. 141 and l. 153–161. | proof of Prop. S12.15(b) |
| C4 | P l. 18 ("For sufficiently small r>0, use the embedded local cylinder") is read as including `r < L/(4√2)`, with REC's proof: `\mathcal D_r ⊂ B̄(0, 2√2 r)`, the exponential chart of `R^d/(LZ^d)` is injective on open balls of radius `L/2`, and `2√2 r < L/2`. Theorem A's `r_*` "may be, and is, taken below this" (REC l. 165–166). Where used: Def. S12.2 (radius and proof); Theorem S12.3 (`r_* ∈ (0, L/(4√2))`); §S12.4 (`r_1 ∈ (0, L/(4√2))`, since the band `0 < r ≤ r_1` uses the pinned law of Def. S12.2); §S12.13 (`r_0` reduced below `L/(4√2)`, so that Def. S12.2 applies for `r ≤ r_0`); Remark S13.7 (pointer: on the torus the product chart is embedded for `r < L/(4√2)`). | REC reading rule l. 30; REC §5 l. 147–148 and l. 163–166. | Def. S12.2, Thm. S12.3, §§S12.4 and S12.13, Rem. S13.7 |

No other correction is made. C1–C4 are exactly REC's reading rule (REC l. 25–32), which the author instructed should govern the transcription ("transcribe the reconciled object"). `verify_s12.py` checks the finite algebra behind C3 (the difference matrices) and C4 (`|corner| = 2√2 r`; `2√2 r = L/2` at `r = L/(4√2)`).

## (c) Added steps (formatting level)

- A1. Names for objects P uses without a symbol: the probability `P^0` (P's `Q`), `E^MS_r`; the elder mark `e(f;M,S) = 1{d_f(M)=f(S)}` (P: "the Borel elder mark"); `ρ_r` for the density of `A_M`; `Ψ` (P's `H`).
- A2. (Revision 1 only; superseded by C2.) P §9 referred to "the identity" without displaying it, and (S12.35) displayed it. In revision 2, (S12.35) is E2's displayed identity (9.1), `∫h dμ = ∫h dν`, written out with E2's definitions of `μ` and `ν` (E2 l. 59–74), so it is no longer an added step.
- A3. `r_1` in §S12.4: P's `sup_{r,b,k,R}` (4.1) leaves the range of `r` implicit ("small r"); `r_1` names it.
- A4. `μ_r = E^0 f` and "`B_r(z)` the regression coefficient of `f(z)` on the entries of `A_M`" (P (4.2) uses both without definition; `μ_r = E^0 f` is forced because `g_r` is centred).
- A5. Lemma S12.14 and Lemma S12.16 are stated for every `C^4` function satisfying the pins (P derives (5.1)–(5.3), (6.1)–(6.2) pathwise from the pins: "The two gradient pins imply…"; S7's Lemma S7.3 uses the same deterministic form).
- A6. In the proof of Prop. S12.18: `‖A_M‖_F ≤ √m Λ` (behind P's "absorb sqrt(m) into K"); `E_* = (3/2)K_*` (P's undefined `E`, from (6.2) and `M_3 ≤ K_*Υ`); the inclusion `{rM_4 > 3k/10} ⊂ {rM_4 > 3k_-/10}` (P bounds the larger event without saying so).
- A7. Proof pointers only, no new mathematics: "direct computations" for (S12.9)–(S12.10); `|α_M| ≤ M_3/2` from (5.1) in the proof of (6.1); "with (a) and Proposition S12.22" in the power count; Theorem A (for `p_r→1`) and `r^3 = ℓ/k` in the proof of Theorem B; `4·(6k)^2k^{−2/3} = 144k^{4/3}` in Prop. S12.31; the factorization (S12.55) "follows from this independence and the target"; the unpacking of P's "(7.8) and the full-probability locus prove Theorem A" into `1 − p_r ≤ P^MS_r(G_r^c)`.
- A8. Lemma S12.21 states the hypotheses of the cap implication (`C^4`, pins, Morse with distinct critical values, inequalities of `G_r`, embedded cylinder); P l. 296 and CPI l. 47–49 ("conditional on parent §1 embedded 2r chart and parent §8 Morse / distinct values").
- A9. Statement environments for P's prose: Lemma S12.17 (P §7 prose), Prop. S12.24 (P §11), Cor. S12.25 (P §12), Lemma S12.20(c) (P §14 l. 445), Lemma S12.29 (P §14). P's §§ map one-to-one to §§S12.1–S12.16.
- A10. S13: the Hermite identity is stated "for every `C^3` function with critical points `M` and `S`" (CAP states it inside the normalized proof; it does not involve `k`; S12 uses it for general `k` in (5.2)); "by the signs of `F = g'`" for `g ≤ b` on `\mathcal C` (see issue I-10); the monotonicity of `g` along the ridge for "its minimum is exactly `s`"; "`D^2 f(M)` is negative definite" in Theorem S13.2(a) (CAP: "M is a maximum", proved through the Schur complement `F'(a) < 0`); the drop/rise in Remark S13.7 written as `b − g(−2r)` and `g(2r) − s`.

**Added in revision 2 from REC §5 (steps that REC supplies where P only sketches; nothing beyond REC's text):**
- A11. Proof of Prop. S12.15(a): `sup E^0(W_r/r^2)^2 < ∞`, by (5.3), (5.1), `‖A_S‖ ≤ ‖A_M‖ + rM_3` and §S12.4; this is the uniform integrability used in (b). Source: REC §5 l. 140 ("C1 A3 item 4 … and A4").
- A12. Proof of Lemma S12.11(b),(c): each coordinate of `U_r`, and of `U_r` extended by the entries of `A_M`, is an integral `∫∂^α f(s't) η_r(ds')` of a derivative of `f` along the axis against a measure `η_r → δ_0`; so the covariance is jointly continuous on `[0, \bar r] × O(d)`, positive definite at `r = 0` by Lemma S12.8, and compactness gives a positive lower bound for its least eigenvalue (existential). Source: REC §5 l. 144 (the "uniform Gramian gap").
- A13. Proof of Lemma S12.16: `det H_M = rα_M det B'` with `B' = A_M − (r/α_M)β_Mβ_M^T`, and `0 ≺ −B' ⪯ B` on the typed support, before "determinant monotonicity … gives (6.1)". Source: REC §5 l. 145 ("A5 for m ≥ 2"). Checked symbolically for `m = 1, 2, 3` in `verify_s12.py`.
- A14. Proof of Prop. S12.15(b): the entries of the two difference matrices of W1 are named (`α_M + 6k`, `α_S − 6k`, `√r β_i`, `A_S − A_M`), read off (S12.21); this makes explicit why W1's "by (5.1), (5.2) and `‖A_S − A_M‖ ≤ rM_3`" suffices. Formatting level; checked symbolically in `verify_s12.py`.

**The other REC §5 discharges, checked against the transcription (no text change):**

| REC §5 item (line) | Where in S12 | Result |
|---|---|---|
| congruence factor (l. 139) | (S12.21) | C1 |
| UI of `W_r/r^2` (l. 140) | Prop. S12.15(a),(b) | A11 |
| type-indicator convergence under `D_r` (l. 141) | Prop. S12.15(b) | C3 |
| `z_* > 0` (l. 142): nondegenerate `A_0` charges the open cone `{A ≺ 0}`; continuity and compactness give (5.5) | Prop. S12.15(c) | already P l. 178–180, transcribed; no change |
| `A_M → A_0` uniformly (l. 143) | Prop. S12.15(b), first sentences | P's sentence kept; REC points to A1–A2, i.e. Lemmas S12.8 and S12.11 with A12; no further text |
| uniform Gramian gap (l. 144) | Lemma S12.11 | A12 |
| A5 for `m ≥ 2` (l. 145) | Lemma S12.16 | A13 |
| A7: `r^5` integrals and far branch (l. 146) | Prop. S12.18 | P (7.5) already says depth failure "is contained in the union" and keeps the far branch in the upper bound, which is REC's point; REC writes the far threshold as `λ ≥ 1/(Dr)` in its reviewer's notation, while P (7.5) has `λ > 1/(4Dr)`; S12 keeps P's; no change |
| cap pairing: embedded chart and Morse/§8 (l. 147) | Lemma S12.21, Thm. S13.2(c) | C4 and the existing hypotheses (A8) |
| embedding radius (l. 148) | Def. S12.2 etc. | C4 |
| §9 write-up gap (l. 149) | Prop. S12.22 | C2 |
| §§11–12 never reviewed (l. 150) | §§S12.11–S12.12 | REC §6 re-derives (11.1)–(11.3), the difference bound and (12.1)–(12.3); it uses Theorem A for `p_r → 1` uniformly and `r^3 = ℓ/k`, the two pointers already added as A7; no change |
| second non-Claude review request (l. 151) | — | review status; not transcribed |

## Proofs: full or as condensed in the source

- Full (every step of the source): Lemma S12.8, Lemma S12.16 (with A13), Prop. S12.18, Lemma S12.19, Lemma S12.21, proof of Thm. S12.3, Prop. S12.22 (every step of E2 §2), Prop. S12.23, Prop. S12.24 and the proof of Thm. S12.4, Lemma S12.26, Prop. S12.28, Lemma S12.29 and the proof of Thm. S12.5, Prop. S12.31; S13 throughout (Lemmas S13.3–S13.6, proof of Thm. S13.2).
- As condensed in the source (transcribed as sketched; see issue I-7): Lemma S12.7 (one line); Lemma S12.10 (no computation in P; verified in `verify_s12.py`); Lemma S12.11(a) ("Taylor formulas with integral remainders"; (b)–(c) now with REC's Gramian-gap argument, A12); Lemma S12.13 (bounds on `B_r`, `g_r`); Lemma S12.14 ((5.1)–(5.2) by one averaging sentence each; `‖A_S−A_M‖ ≤ rM_3` not derived); Prop. S12.15(b) (W1's in-law route and the subsequence argument, at REC's level of detail); Lemma S12.20 ((a) sketched, I-6; (b)–(c) asserted as the elder rule); within Prop. S12.22, E2 §9.1's continuity of the regression subtraction "in every finite `C^q` mean norm needed here" is asserted (I-7); Lemma S12.27(c); Lemma S12.30; Corollary S12.25; Lemma S13.3 (Hermite identity stated, not derived; verified on a generic quintic in `verify_s12.py`); Thm. S13.2(c) (maximin level = elder death level asserted).

## Not transcribed (by design)

COMMON rule 3 (the supplement writers' rules, `scratchpad/supp/COMMON.md`: no provenance labels, status words or historical comparisons in the LaTeX) and the assignment's scope:

- P l. 3–7 (object, author, disposition, continuation of CAP, campaign #61), except the mathematical content ("non-numerically-evaluated constants" → "All constants are existential").
- P l. 31 ff.: the word "candidate" in the theorem titles (status word; the counting term "candidate pairs" is kept).
- P l. 45: "and no claim that r_*=1/20. The new asymptotic normalizer proof does not replace or re-certify the historical H3 finite-band floor."
- P l. 182: "not the historical numerical floor 17/10" (the rest of the sentence is kept).
- P l. 382: "This is unrelated to the inverse-Hessian/Schur thresholds in issue #56."
- P l. 443: "It is not the old shrinking-annulus RN enclosure and does not discharge that numerical obligation."
- P l. 484 (list of interfaces supplied); l. 486, the non-mathematical items (H3/LPW replay, shrinking-annulus RN certificate, 24-jet displacement enclosure, formal verification, nonauthor acceptance, P15 prize closure, "No historical scientific status is changed"); l. 488 (exact-arithmetic program and mutations); l. 490 (review priorities); l. 494, Beliaev–Cammarota–Wigman (neighbouring work, "not a source of our elder-selection conclusion") and "No priority or novelty claim follows from this bounded reconnaissance."
- Capitalized emphasis (FULL, NOT, ALL, …) rendered as ordinary prose.
- CAP l. 3–5 (header); l. 19 "used by #57"; CAP l. 30–45 and §§6–8 (l. 169–230): the fixed-axis two-dimensional probabilistic corollary (2)–(3), its imported numerical premises, the exceptional layer (15)–(18), the comparison with the predecessor's constants and the remaining-work list (CPI l. 49: "Cap 2D numerical corollary is out of scope"); l. 127 "The bound is weaker than #58's F''>1 … second source of the improved depth condition"; l. 161 "This is the same separating-cap criterion established in #58 …".
- CPI l. 1–5 and l. 44–45 (status notes on the r^5 skeleton, which duplicates P (7.3)).
- EP, RM: provenance only.
- Revision 2. P §9 (l. 298–308) as a whole, replaced by E2 under C2; in particular P l. 300 ("a formula for local Hessian index alone would not suffice"), the last sentence of l. 306 ("The continuous Gaussian regression kernel defines the canonical density version; the underlying measure identity is unaffected by height-null-set choices", see I-14) and the last sentence of l. 308 ("No finite-grid approximation of the continuum field is substituted") have no counterpart in E2.
- E2 l. 1–7 (header, author lane, disposition); §1 l. 9–21 (the review finding and the theorem-number discussion; its mathematical content, AAL Theorem 2.2 for the unweighted formula and finiteness and Theorem 7.1 for continuous weights, is used in Prop. S12.22); l. 92 "This is the missing closure."; l. 110 "and does not re-prove or accept them here" (status; the dependency on §§S12.3, S12.5, S12.10 is kept); §§3–5 l. 112–147 (what the repair closes, review request, source-integrity note).
- REC l. 1–8 (header), §1 l. 34–41 (machine binding, name collision), §§2–4 and §§6–10 (reconciled statements at existential scope, components, interface ledger with review verdicts, review of §§11–12, consumption contract, independence, register transitions, checks), and the "Raised in" and status content of the §5 table. Their mathematical content is either already P's (and transcribed from P) or is applied as C1–C4 and A11–A14.

## (d) Issues, most severe first

Issues I-1, I-2 and I-4 are **resolved** by C2, C3 and C4 (revision 2). The others are not fixed in the LaTeX. "Suggested fix" is for the author.

- **I-1 — resolved by C2.** (Was medium-high.) P §9 l. 304 → Prop. S12.22(a): the passage from continuous cylinder weights to bounded Borel marks named the monotone-class theorem without constructing the regular conditional law of the whole field on `C^2(T^d_L)`, without the closure hypotheses of the functional monotone-class theorem, and without the unweighted Kac–Rice theorem behind the finiteness of the two measures. Prop. S12.22 now transcribes E2 §§9.1–9.4, which supplies all three; its statement (a) is now the same as S7's Hypothesis S7.6(a).
- **I-2 — resolved by C3.** (Was medium.) P l. 165–167 → proof of Prop. S12.15(b): "converge, in probability … to diag(∓6k, A_0)", where `A_0` lives under a different law. The proof now follows W1: differences to `diag(∓6k, A_M)` tend to 0 in probability, `A_M → A_0` in law, Slutsky, continuous mapping off `{det A_0 = 0}`.
- **I-3 (low–medium; unproved step used in Theorem C).** P §14 l. 445 → proof of Thm. S12.5 and Lemma S12.20(c). The identification of the selected pairs with the finite bars needs the unconditioned field `f` to be almost surely Morse with distinct critical values; P §8 (Lemma S12.19) proves genericity only under each pinned law `P^0_{r,b,k,R}`. Suggested fix: add the unpinned case to Lemma S12.19 (the same overdetermined-zero argument without pins), or cite a standard Bulinskaya-type statement.
- **I-4 — resolved by C4.** (Was low.) P l. 18 gave no explicit embedding radius. Def. S12.2 now requires `0 < r < L/(4√2)` with REC's one-line proof, Theorem S12.3 takes `r_* < L/(4√2)`, and §S12.13 takes `r_0 < L/(4√2)`; this matches S7's `r_0 < L/(4√2)`.
- **I-5 (low; wording).** P l. 415 "For every fixed positive r and every b,k>0, Z_r is positive and finite" and "For every fixed b,k>0, Theorem A … implies p_r->1", and P l. 426 "For each fixed b,k>0,u" → Lemma S12.27(c), proof of Prop. S12.28. Read literally `b > 0`; P's argument does not use the sign of `b`, and (13.5)–(13.6) integrate over `b ∈ R`, where both facts are needed. Transcribed as "every `b` and every `k>0`" (the source's comma list, read as its own use requires; no formula changes). S7 issue I1 is the same. Suggested fix: "for every `b ∈ R` and `k > 0`".
- **I-6 (low; sketch).** P l. 290–294 → Lemma S12.20(a). Borel measurability of `{d_f(M) > h}` in `(M, f)` is sketched ("strict clearances allow approximation by such paths"; "an initial segment depending continuously on `M`"). Suggested fix: expand, or use the lower-semicontinuity proof of S5's Lemma S5.3.
- **I-7 (low; steps stated without derivation, transcribed as stated).** P (3.2)–(3.3) (computation not shown; verified, `verify_s12.py`); P l. 103 (mean-square and covariance convergence of `U_r`); P l. 125 (bounds on `B_r`, `g_r`); P l. 151 and l. 153–154 ((5.1)–(5.2) by one averaging sentence; `‖A_S − A_M‖ ≤ rM_3` not derived; the averaging identities `f_xx(a) = −(1/r)∫_a^c (c−x) f_xxx`, `f_xx(c) = (1/r)∫_a^c (x−a) f_xxx` are verified on a generic quintic); P l. 176 (uniformity by a subsequence argument); P l. 458–461 (the factorization); CAP l. 55–57 (Hermite identity; verified); CAP l. 161 (maximin level = elder death level, the elder rule; P cites Curry §3 as background); revision 2: E2 l. 47 (continuity of the regression subtraction in `(x, y, z)` "in every finite C^q mean norm needed here", asserted from uniform nonsingularity and the smooth covariance). Suggested fix: write out the two averaging identities and the (3.2) block determinants; cite a reference for the elder rule.
- **I-8 (low; wording).** P l. 63 → Lemma S12.8: "Any finite list of distinct derivative evaluation functionals at distinct sites". The uses (the contact jet (3.4), the enlarged list `(U_r, A_M)`, two-site lists with Hessians, S10's `(J', U_r)`) have several functionals at each of finitely many sites, in one orthonormal frame; the proof (distributional argument; test functions with prescribed jets) covers that. Suggested fix: "at finitely many sites, several functionals per site allowed, in one orthonormal frame" (as S7's Lemma S7.1(b) reads it).
- **I-9 (low; range of a display, S13).** CAP (14) l. 142–147 → (S13.15). The bound `f(x,y) < b − (4400/121) r^3` on `‖y‖ = 2r` is derived from `f ≤ g ≤ b`, which holds for `x ∈ [−2r, r/2]` (the cap `\mathcal C`). For `x ∈ (r/2, 2r]`, `g` rises above `b` (`g(2r) ≥ b + (11/96) r^3`) and the displayed bound does not follow from the given argument. CAP's sentence "This covers the entire curved side of the cylinder, including its intersections with the end faces" is read as the curved side of `\mathcal C` (whose end faces `x = −2r`, `x = r/2` are the faces treated next), which is all the proof uses. Transcribed with `−2r ≤ x ≤ r/2`. Suggested fix: "the entire curved side of C".
- **I-10 (very low; attribution).** CAP l. 133 "Strong concavity gives f(x,y)<=g(x) and g(x)<=b on C": the second inequality follows from the signs of `F = g'` (CAP l. 129), not from strong concavity. The LaTeX points to Lemma S13.5 (added step A10).
- **I-11 (low; statement form).** P Theorem C l. 47–52 → Thm. S12.5. Unlike Theorem B, Theorem C does not say that the densities "admit versions" with (1.3), although its proof works with the density versions (13.5) and the Kac–Rice far versions, and S7, S8, S10 cite it for "canonical marked Kac–Rice density versions". Suggested fix: "These densities admit versions with (1.3)", naming the versions (S12.40) and (S12.51), defined through the regression-law intensities of Prop. S12.22(b) (see also I-14).
- **I-12 (low; undefined constant).** P (7.3) l. 236–238 uses `E` without definition; defined here as `E_* = (3/2)K_*` (A6). S7 issue I4 is the same.
- **I-13 (low; record-internal notation clashes).** P uses `A_r` for `D_y^2 f(M)` and for the intensity, `A_0` for the contact Hessian and the limit intensity, `B` for the birth window and `−A_M`, `h` for `M_3`, the separation vector and a level, `U0` for a row and `U0*` for the jet, `t` for the cutoff, in `t_u` and in `t = 12k`; CAP uses `κ` for the gap, `k = 8m−5`, `m = M_3`, `u = 2/k + 2/k^2`. Resolved by the conversions above; no mathematical effect. Suggested fix: rename in the sources.
- **I-14 (low; revision 2, a sentence lost with P §9).** E2 replaces P §9 in full, and E2 has no counterpart of P l. 306's last sentence, "The continuous Gaussian regression kernel defines the canonical density version; the underlying measure identity is unaffected by height-null-set choices." S8 l. 12 (`\cite[\S9]{parentP}`) and S10 l. 9 cite exactly this for "canonical marked Kac–Rice density versions". In the reconciled S12 the versions are still specified: Prop. S12.22(b) expresses the intensities through `E^0[W_r]` and `p_r`, where `E^0` is the continuous regression law of Def. S12.2 at the prescribed heights; (S12.40) is stated as "a version" of the compact-window density, and (S12.51) gives the near densities by an explicit formula. But the word "canonical" and the null-set sentence are gone. Suggested fix: S8 and S10 cite (S12.40), (S12.51) and Prop. S12.22(b); or E2 restores the sentence. Not added here (it is not in the reconciled object).

Beyond the items resolved by C2–C4, no mathematical error was found; I-3 is the only open item above low severity. Spot checks by the writer: all finite identities in `verify_s12.py` (59/59 PASS); by hand, (5.1)–(5.2), (6.1)–(6.2), (7.1)–(7.7), (13.1)–(13.4), (14.1), (15.1)–(15.2), CAP (5)–(14), the scaling to general `k`, and REC's embedding and A5 algebra.

## (e) Import map

All 50 occurrences of `\cite{parentP}` / `\cite[…]{parentP}` in `supp_s0_status.tex`, `supp_s1`–`supp_s11` (none in S1–S4 and S11; one textual "parent formula" in S3 l. 560 is not a citation). "Covered" means the cited statement is now stated, with proof, at the S12/S13 label given. Numbers are those of the compile check.

| # | Place | Citation | What is used there | Now stated in | Status |
|---|---|---|---|---|---|
| 1 | S0 l. 13 | `\cite{parentP}` | Theorem A: `1−p_r ≤ Cr^3` on compact marks, all `d`, `L` | `thm:parent-A` (Thm. S12.3) | covered |
| 2 | S5 l. 4 | `\cite{parentP}` | Theorem A for (3.13) | `thm:parent-A` | covered |
| 3 | S5 l. 902 | `\cite{parentP}` | Theorem A at `d=3`, `L=24`, uniform over compact windows and frames; `p_r` = probability that the global elder partner is `S` | `thm:parent-A`, `def:parent-pinned` (Def. S12.2) | covered |
| 4 | S5 l. 902 | `\cite{parentP}` | "a second … proof that does not use [parentP]" | — | n/a (statement of non-use) |
| 5 | S6 l. 125 | `\cite{parentP}` | (14.1): far bound for elder and all typed far pairs, `0<ℓ≤1` | `eq:parent-141` in `lem:parent-far` (Lemma S12.29) | covered |
| 6 | S7 l. 4 | `\cite{parentP}` | the statements collected in Lemmas S7.1–S7.3, Hyp. S7.4–S7.7, Def. S7.8 | `sec:parent` (rows 9–19 below) | covered |
| 7 | S7 l. 46 | `\cite[\S10]{parentP}` | `\mathcal A_r` invariant under transverse rotations | `prop:parent-ledger`(c) (Prop. S12.23) | covered |
| 8 | S7 l. 65 | `\cite{parentP}` | "the following statements" | `sec:parent` | covered |
| 9 | S7 l. 67 | `\cite[\S2]{parentP}` | Lemma S7.1: (a) Fourier expansion, `a_n>0`, summability, smooth version, moments; (b) finite-jet rank, one or several points, any frame | (a) `lem:parent-spectrum` (Lemma S12.7); (b) `lem:parent-jetrank` (Lemma S12.8) | covered (several functionals per site: by the proof; wording I-8) |
| 10 | S7 l. 77 | `\cite[\S3, (13.1) and (13.2)]{parentP}` | Lemma S7.2: (a) `|det| = 12r^{−(d+3)}`; (b) eigenvalues of `Σ_r` and of the covariance of `(U_r, A_M)` in a compact subinterval of `(0,∞)`, `0≤r≤r_1`; (c) (13.1)–(13.2) | (a) `lem:parent-frame` (Lemma S12.10); (b) `lem:parent-contact`(b),(c) (Lemma S12.11) with the `r_0` paragraph of §S12.13; (c) `lem:parent-growth` (Lemma S12.26) | covered; the upper eigenvalue bound of the enlarged covariance is implicit in S12 (its entries are bounded derivatives of `K_L`) |
| 11 | S7 l. 91 | `\cite[(5.1)--(5.3), (6.1), (6.2) and \S13]{parentP}` | Lemma S7.3: deterministic (5.1)–(5.3), envelope (6.2)/r², `W_r/r^2 ≤ C(1+‖φ‖_{C^3})^{2d}` | `lem:parent-endpoint` (Lemma S12.14), `lem:parent-soft` (Lemma S12.16, (S12.24)), `lem:parent-majorant`(a) (Lemma S12.27, (S12.49)) | covered |
| 12 | S7 l. 109 | `\cite[\S\S8 and 14]{parentP}` | Hyp. S7.4: (a) pinned genericity under `E^0`, `E^MS_r`; (b) `d_φ`, Borel mark, Morse-locus meaning, essential class, finite bars ↔ `e=1` | (a) `lem:parent-generic` (Lemma S12.19); (b) `lem:parent-mark` (Lemma S12.20) | covered (for (b)'s finite-bar part see I-3) |
| 13 | S7 l. 119 | `\cite[\S\S7 and 8]{parentP}` | Hyp. S7.5: on `G_r`, Morse with distinct values, `S` is the elder partner | `lem:parent-capimpl` (Lemma S12.21) + `thm:cap-main`(c) (Thm. S13.2) | covered; the radius `r < L/(4√2)` is now in Def. S12.2 (C4), matching S7's `r_0 < L/(4√2)` |
| 14 | S7 l. 125 | `\cite[\S\S9--11 and (13.5)]{parentP}` | Hyp. S7.6: (a) marked Kac–Rice for bounded Borel `h` (E2's form); (b) `0<Z_r<∞` for `b∈R`, `k>0`; intensity `r\mathcal A_r`, selected `r\mathcal A_r p_r`, `σ` unnormalized, no 1/2; (c) near density versions | (a) `prop:parent-KR`(a) (Prop. S12.22); (b) `lem:parent-majorant`(c), `prop:parent-ledger`(b), `prop:parent-KR`(b),(c); (c) `prop:parent-near` (Prop. S12.28, (S12.51)) | covered; (a) is now in E2's form, as S7 states it (C2); (b) `b∈R` wording (I-5) |
| 15 | S7 l. 147 | `\cite{parentP}` | how (b), (c) follow from (a): disintegration, `12r^{−(d+3)}π_r`, polar factor, height Jacobian, exhaustion, power count, `ℓ = kr^3` | `prop:parent-KR`(b),(c), `prop:parent-ledger`(a),(b), `prop:parent-push` (S12.39), `prop:parent-near` | covered |
| 16 | S7 l. 149 | `\cite[\S14 and (14.1)]{parentP}` | Hyp. S7.7: far densities bounded, versions with original pins, Jacobian one | `lem:parent-far` (Lemma S12.29) | covered |
| 17 | S7 l. 156 | `\cite[\S14]{parentP}` | near ∪ far = all pairs; boundary null | `lem:parent-far` (last sentence) | covered |
| 18 | S7 l. 158 | `\cite[(13.6), \S15 and (15.2)]{parentP}` | Def. S7.8: `c_{d,L} = ∫\mathcal A_0/(3k^{2/3}) = 4∫k^{−2/3}π_0z_0`; (S7.23) | (S12.52) in `prop:parent-near` with `\mathcal A_0 = 12π_0z_0` (S12.38); `lem:parent-parity` (S12.54); `prop:parent-coef` (S12.58) | covered |
| 19 | S7 l. 166 | `\cite[(13.6) and \S15]{parentP}` | `0<c_{d,L}<∞` and (15.2) | `prop:parent-near`, `prop:parent-coef` (Prop. S12.31) | covered |
| 20 | S7 l. 191 | `\cite[Theorem~B]{parentP}` | `O(ℓ^{2/3})` difference only on compact windows | `thm:parent-B` (Thm. S12.4) | covered |
| 21 | S7 l. 411 | `\cite[\S4]{parentP}` | regression of the field on `(U_r, A_M)`, residual independent of the whole vector | `lem:parent-residual` (Lemma S12.13) | covered (method; S7's growing-target moment bounds are S7's own added step S10) |
| 22 | S7 l. 419 | `\cite[\S7]{parentP}` | eigenvalue coordinates, Vandermonde factor, not GOE | `lem:parent-eigen` (Lemma S12.17) | covered |
| 23 | S7 l. 478 | `\cite[\S13]{parentP}` | dominated convergence gives the limit only | `prop:parent-near` | covered |
| 24 | S7 l. 534 | `\cite[(15.2)]{parentP}` | `c_{3,24}` = (15.2) at `d=3`, `L=24` | `prop:parent-coef` (S12.58) | covered |
| 25 | S7 l. 534 | `\cite[Theorem~C]{parentP}` | leading law, instance `d=3`, `L=24` | `thm:parent-C` (Thm. S12.5) | covered |
| 26 | S7 l. 534 | `\cite[(15.2)]{parentP}` | coefficient of Theorem C | `prop:parent-coef` | covered |
| 27 | S8 l. 12 | `\cite[\S9]{parentP}` | canonical marked Kac–Rice density versions for general `d`, `L` | `prop:parent-KR`(b) with (S12.40), (S12.51) | covered, except the word "canonical", which went with P §9 (I-14; also I-11) |
| 28 | S8 l. 12 | `\cite[(15.2)]{parentP}` | `c_{d,L}` | `prop:parent-coef` | covered |
| 29 | S8 l. 12 | `\cite[Theorem~C]{parentP}` | `ν_cand ∼ ν_eld ∼ c_{d,L}ℓ^{−1/3}` (unrestricted) | `thm:parent-C` | covered |
| 30 | S8 l. 23 | `\cite[\S8]{parentP}` | the mark `e` is Borel | `lem:parent-mark`(a) | covered (sketch, I-6) |
| 31 | S8 l. 40 | `\cite[\S\S9--10, (10.2)]{parentP}` | intensity `r\mathcal A_r dr db dk dσ`, elder/rejected analogues | `prop:parent-KR`, `prop:parent-ledger` (S12.37) | covered |
| 32 | S8 l. 44 | `\cite[(13.6)]{parentP}` | `c_{d,L} = ∫∫∫ \mathcal A_0 (3k^{2/3})^{−1}` | (S12.52) | covered |
| 33 | S8 l. 86 | `\cite[\S\S4,7]{parentP}` | `M_3`, `M_4` local seminorms on the cap cylinder; `G_r` | `def:cap-norms` (Def. S13.1, display (S13.2)), `eq:parent-G` (S12.25), `eq:parent-43` (S12.15) | covered |
| 34 | S8 l. 94 | `\cite[\S\S7--8]{parentP}` | deterministic cap theorem: on `G_r` and the typed support, `S` is the global elder partner | `lem:parent-capimpl`, `thm:cap-main` | covered |
| 35 | S8 l. 94 | `\cite[\S8]{parentP}` | `Q` a.s. Morse with distinct critical values | `lem:parent-generic` | covered |
| 36 | S8 l. 107 | `\cite[\S14, (14.1)]{parentP}` | Prop. S8.4(iii): `Ψ^cand_ℓ(b,y) ≤ C_ρ(1+|b|)^{2d}e^{−c_ρb^2}` for `dist ≥ ρ`, every `ρ ∈ (0, L/2]`, `0<ℓ≤1` | `lem:parent-far`: target density `≤ Ce^{−cb^2}` and moments `≤ C(1+|b|)^{2d}` on `dist ≥ r_0` | covered: for `ρ ≥ r_0` by inclusion; for `ρ < r_0` take `r_0 ≤ ρ`, since the conditions fixing `r_0` in §S12.13 survive when `r_0` is decreased (S8's added step A8 reads P the same way) |
| 37 | S8 l. 292 | `\cite[\S2]{parentP}` | finite-jet nondegeneracy at distinct sites | `lem:parent-jetrank` | covered |
| 38 | S8 l. 292 | `\cite[\S14]{parentP}` | `λ_min(Cov O_y) ≥ c_0` on `dist ≥ r_0`, used for every fixed `r_0 ∈ (0, L/2]` | `lem:parent-far` (first sentence) | covered, with the `r_0` remark of row 36 |
| 39 | S8 l. 311 | `\cite[\S2]{parentP}` | `(O_y, H_0, H_y)` positive definite | `lem:parent-jetrank` | covered |
| 40 | S9 l. 30 | `\cite[\S15]{parentP}` | odd and even jets independent | `lem:parent-parity` (Lemma S12.30) | covered |
| 41 | S9 l. 236 | `\cite[\S15]{parentP}` | `π_0(u; v_0(b,0)) = p_{(f,V_u)}(b,0)p_G(0)(2π)^{−1/2}τ_u^{−1}` and integration over `b` | (S12.55) at `k=0` (`φ_τ(0) = (2π)^{−1/2}τ^{−1}`); `b`-integration in the proof of Prop. S12.31 | covered |
| 42 | S10 l. 9 | `\cite{parentP}` | canonical marked Kac–Rice density versions | `prop:parent-KR`(b) with (S12.40), (S12.51) | covered, except the word "canonical" (I-14; also I-11) |
| 43 | S10 l. 9 | `\cite[(15.2)]{parentP}` | `c_{d,L}` | `prop:parent-coef` | covered |
| 44 | S10 l. 19 | `\cite[(3.1)]{parentP}` | pin rows, Jacobian `12r^{−(d+3)}`, target `v_r` | `def:parent-rows` (S12.8), `lem:parent-frame` (S12.9)–(S12.10) | covered |
| 45 | S10 l. 21 | `\cite{parentP}` | elder mark Borel; on the Morse locus it is the elder pairing | `lem:parent-mark` | covered |
| 46 | S10 l. 122 | `\cite{parentP}` | jet independence (distinct sites; one site, rotated frame); odd/even independence | `lem:parent-jetrank`, `lem:parent-parity` | covered |
| 47 | S10 l. 254 | `\cite{parentP}` | `(J', U_r)` at the sites `0, ±(r/2)u`; `(J', U_0)` at one point | `lem:parent-jetrank` | covered (several functionals per site, I-8) |
| 48 | S10 l. 315 | `\cite{parentP}` | parity: even/odd jets independent at contact | `lem:parent-parity` | covered (S10 derives the target-shift consequences itself) |
| 49 | S10 l. 540 | `\cite{parentP}` | `(\mathcal G_r(F), U_r)` = distinct functionals at `M ≠ S` | `lem:parent-jetrank` | covered |
| 50 | S10 l. 540 | `\cite{parentP}` | `(\mathcal G_0(F), U_0)` = distinct jets at one point in a frame | `lem:parent-jetrank` | covered |

**Summary.** 50 citations: 49 covered by S12/S13 labels, 1 not applicable (S5 l. 902, a statement that the S5 proof does not use the parent). None is "not covered". Caveats on coverage (revision 2): rows 14 and 27/42 now rest on Prop. S12.22 in E2's form (C2), the same form as S7's Hypothesis S7.6(a); rows 27/42 lose only the word "canonical" (I-14); row 14(b) carries the wording issue I-5; row 13's radius is now in S12 (C4); rows 36 and 38 use the far bound at every `ρ`, covered by choosing `r_0 ≤ ρ`; rows 9, 47 rely on the reading I-8; row 12(b) depends on I-3.

**For the integrator.** To make the supplement self-contained, each citation above can be replaced by a reference to the label in the "Now stated in" column; S7's Lemmas S7.1–S7.3, Hypotheses S7.4–S7.7 and Definition S7.8 then become restatements with pointers to S12/S13.

## Bibliography

New file `supp_refs_s12.bib` (COMMON rule: a new reference goes in `supp_refs_<section>.bib`), two entries cited in S12:
- `AAL`: D. Armentano, J.-M. Azaïs, J. R. León, *On a general Kac–Rice formula for the measure of a level set*, arXiv:2304.07424v3 (2023); P l. 494 (§§2 and 7) and, after C2, E2 l. 13–21, l. 35, l. 63, l. 72 (Theorem 2.2 for the unweighted formula and finiteness, Theorem 7.1 for continuous weights). E2 l. 21 gives the v3 date as 5 December 2023.
- `Curry`: J. Curry, *The fiber of the persistence map for functions on the interval*, arXiv:1706.06059v2 (2017); P l. 494 (§3, elder-rule background).
Only arXiv data is given (no journal data). `parentP` is in `refs.bib`.

## Integration (not done here)

`supp_main.tex` would need `\input{supp_s12_parent}` and `\input{supp_s13_cap}` after `supp_s11_ctwo` (a `\part*` heading is the integrator's choice) and `\bibliography{refs,supp_refs_s8s9,supp_refs_s12}`. The status section S0 and the README/SHA256SUMS of the draft folder are not touched.

## Cross-references used (labels defined elsewhere)

`sec:selection` and `sec:third` (opening paragraph of S12 only). S12 and S13 reference each other (`sec:cap`, `def:cap-norms`, `eq:cap-Mj`, `eq:cap-hermite`, `thm:cap-main`; `sec:parent`, `lem:parent-capimpl`).

## Compile check (`chk_s12/`)

`chk_s12/test_s12.tex` = the preamble of `supp_main.tex` (everything before `\begin{document}`, copied verbatim) + 11 stub sections carrying `sec:setting` … `sec:ctwo` + `\input{../supp_s12_parent}` + `\input{../supp_s13_cap}` + `\bibliographystyle{amsplain}\bibliography{refs,supp_refs_s12}`; `refs.bib` (copy of the supplement's) and `supp_refs_s12.bib` copied in. `pdflatex` ×3 with `bibtex` (TeX Live 2023), from a clean directory:
- LaTeX errors: 0.
- Undefined references or citations: 0 (the only external labels, `sec:selection` and `sec:third`, are stubbed); no "Rerun" warning; bibtex: no warnings (3 entries: `parentP`, `AAL`, `Curry`).
- Overfull / underfull boxes: 0. Revision 1's first build had three overfull boxes (22.4 pt in the proof of Lemma S12.16, 15.0 pt in Definition S13.1, 4.4 pt in Proposition S12.22(b)); revision 2's first build had one (59.5 pt, the regression subtraction in the new proof of Prop. S12.22). All were fixed by displaying the long formula or rewording.
- Pages: 20 in the test document (revision 1: 19) = 1 page of title and stubs + S12 on pp. 2–17 (about 15 pages) + S13 on pp. 17–20 (about 3 pages, with the bibliography on p. 20). The new proof of Prop. S12.22 accounts for the extra page.
- Numbering: the label lists of revision 1 and revision 2 (`chk_s12/labels_before.txt`, `chk_s12/labels_after.txt`, 133 labels each) are identical, so no statement, display or subsection number shifted.

`chk_s12/verify_s12.py` (SymPy, exact; output `chk_s12/verify_s12.out.txt`): 59 finite checks, ALL PASS — `|det T_r|` for `d = 2…6`, the target `v_r`, the Hermite identity and the two averaging identities on a generic quintic with critical pins, `α_{M,S} = ∓6k` for `f_xxx ≡ 12k`, the block identity (5.3) and the corrected congruence for `m = 1, 2, 3` (and that the uncorrected factor fails), (7.3), the (7.5) split, (13.1), (11.1), the exponents of (11.3) and the difference bound, (12.1), the gamma factor of (15.2), the amplitude exponent, every CAP constant (`24/121`, `48/121`, `2554128/1771561`, `2184415/7086244`, `2r^2`, `15/8`, `9/32`, `11/96`, `4400/121`, the scaling to `27/16`, `11/16`, `26400/121`, `3/2`, and the scaling of (4) to (1)), and, in revision 2, C4's embedding geometry (`|corner| = 2√2 r`, `2√2 r = L/2` at `r = L/(4√2)`), REC's A5 factorization and `B − (−B') = (r/(−α))ββ^T` for `m = 1, 2, 3`, and the entries of W1's difference matrices for `m = 1, 2, 3`. These are finite algebra checks, not checks of any analytic estimate.

### File hashes (revision 2)

SHA-256 of the output files, computed after the final build (this file's own hash is reported separately, since it cannot contain itself):

| File | Bytes | SHA-256 |
|---|---:|---|
| `supp_s12_parent.tex` | 63,987 | `10f33167b43e78c58151897eb68ed2c6d80c3da1e73267aa9c87672eda8c35f8` |
| `supp_s13_cap.tex` | 14,975 | `601c85fa71aa90cea608743e946016d55c4acb1321a6ad4181c85c4ad9c6f892` |
| `supp_refs_s12.bib` | 399 | `4ac5436d450d5857a147b42fd2a9a694b59d06ecbb0c147703a65beb48575c36` |
| `chk_s12/test_s12.tex` | 2,362 | `7f1e904c6abf13b6ecbe2666ca3251ad9ad740ec69bdefef43ac46157467111e` |
| `chk_s12/test_s12.pdf` | 493,594 | `36965aeea91cbcf67d0c733d40dfa576718721cd03cc94dccb6eb833289cdcfa` |
| `chk_s12/verify_s12.py` | 10,655 | `067557f852a530f9ed0f6d442d315f507a21189ffd1a510818e4f540ab31cfab` |
| `chk_s12/verify_s12.out.txt` | 2,478 | `cabbd30d5f3d5a588b0ca7ca3b241276c6e5e45f2d8573b70a4c1e9466156dc4` |
| `chk_s12/labels_before.txt` | 2,869 | `aebfe144dd9ee187bbaed48b8d63196970c12ee5afc60e04759cdceadd1dbec3` |
| `chk_s12/labels_after.txt` | 2,869 | `aebfe144dd9ee187bbaed48b8d63196970c12ee5afc60e04759cdceadd1dbec3` |
