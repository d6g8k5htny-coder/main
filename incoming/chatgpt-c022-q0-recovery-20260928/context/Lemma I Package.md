# Lemma I, Resolved: the Inner-Zone Count and the Completed r³ Architecture for Theorem A
**Artifact class:** lemma-resolution package + zone-architecture consolidation (C009).
**Chain:** C009 snapshot 2e363140…9dc3; observed ledger (companion, hashed). Inputs: Lemma T certified table (C007-B), FlatSaddle Tail Lemma + verification (C007-C, exponent 4.959), bi-graded general-τ laws + O2 closure (C008), window_lemma_v2_patch (Sections R, P, A′, B, I).

## 1. The lemma as posed, and its resolution

**Lemma I (window_lemma_v2_patch, Section I).** Under (H1+)–(H2′), E[#{critical points in B_{Cr²}(S) ∪ B_{Cr²}(M), ∉ {M,S}} | pair] ≤ Cr^β, some β > 0. Candidate given there: β = 4. Stated reduction: boundedness of the conditional 3-point/2-point Kac–Rice ratio, requiring nondegeneracy of the leading coefficient of the jet-covariance determinant under the two-scale degeneration |x_S − x_M| ~ r, |y − x_S| ~ r².

**Resolution.** β = **5**, strictly better than the candidate and than the β ≥ 3 needed for the r³ rate:
(i) *Nondegeneracy of the blow-up.* The two-scale jet covariance is the C007-B certified exact table (gates G1–G4); its general-τ extension is the C008 bi-graded table. The load-bearing Gram — the 6×6 conditional covariance of the pair-Hessian block given E_lin — has determinant **(1/540)·r¹⁸ + O(r¹⁹)** (exact, this cycle): strictly positive leading coefficient at finite order, which is precisely the required blow-up nondegeneracy and discharges the density-positivity part of (U3). The remaining (U3) piece (bounded density of the third-derivative functional ξ) follows from Lemma H directly.
(ii) *The count.* The pair-Palm expectation (which is what "given the pair" means: both member determinants in the weight) of the r²-ball third-point count is O(r⁵) by the Flat-Saddle Tail Lemma with every ledger input now a certified coefficient, and is **numerically verified at 4.959** (unwindowed; 4.967 windowed) over r ∈ [0.014, 0.2]. The M-side ball is symmetric. The r⁵ arises as candidate-β=4 (area × bounded ratio) *plus one power* from the M-side determinant drag |det H_M| ≍ κr² on the contributing near-monkey-saddle set — the ratio is not merely bounded but vanishing after correct weight bookkeeping.
(iii) *Assumption grade.* (U1) conditional Kac–Rice validity ← Lemma H nondegeneracy of the conditional Gaussian + standard a.s.-Morse arguments [obligation, mechanical]. (U2) L^{2+ε} uniformity of the O_P(r) corrections ← finite Gaussian moments under λ₈ < ∞ of (H1+) [obligation, mechanical]. (U3) ← discharged as above, exact constant on record. (U4) ← Palm-base compactness [obligation, bookkeeping].

## 2. Two corrections to window_lemma_v2_patch (registered, superseding)

**C-1 (Localization statement).** The claim that third critical points in B_{3r} concentrate within Cr² of {M, S} "up to exp(−c/r²) tails" is a conditional-at-fixed-w statement (the R3-class error already catalogued in the FlatSaddle kill log). The Palm-averaged law in r² ≪ τ ≲ r is polynomial: **E[N(shell τ)] ≍ r·τ² = r^{2γ+1}**, certified inputs, verified at γ = 2 and γ = 3/2 (C008). Assembly-harmless: the polynomial mass totals ≍ r³, boundary-dominated.

**C-2 (Lemma A′ scope).** A′'s t-floor argument (|t² − a| ≥ cτ² for τ ≥ 3r) fails on the vertical strips |t ∓ r/2| ≪ r, where the f_t conditional mean vanishes at every distance. Measured on the strip {|t − r/2| ≤ r/2, s ∈ [r/2, 4r]}: windowed pair-Palm count with **tail exponent 2.948** (unwindowed 3.014) — polynomial r³-class where the skeleton claimed exp-smallness, at a stable ≈ 0.24 of the C008 annulus total. A′ is scope-corrected to the off-strip annulus, where the floor argument is degeneracy-proof (the fold modulus κ is *fixed* by the Palm base, so the suppression constant cannot degenerate under the |det|-biased average — the flat direction is μ, which the strips expose and the off-strip t-floor does not involve).

## 3. The completed zone architecture (all measured or graded)

| zone | law | grade |
|---|---|---|
| inner balls B_{Cr²}(S) ∪ B_{Cr²}(M) | Θ(r⁵)-bounded, window const ≈ 0.51 | derived (certified inputs) + verified (4.959/4.967) |
| shells r² ≪ τ ≤ r/2 | r^{2γ+1} per shell; total r³, top-octave 0.86–1.00 | derived + verified at two γ; successive slopes → 2.96 |
| peak zone τ ≍ r (strip base [r/2, r]) | r³-class, ≈ 0.24 × annulus; window acceptance ≈ 0.02 | measured (tail 2.95) |
| strip decay s ∈ [r, O(1)] | geometric per-octave decay ×0.08–0.14 | measured (three rungs) |
| far floor, absolute O(1) distances | C(L)·ℓ (Lemma G) | graded per v2 patch; magnitude cross-checked: measured [8r,16r] octave 3.7e-7 at r=0.1 vs ledger estimate ~5e-7 ✓ |
| off-strip annulus [3r, δ] | exp(−c/τ²), κ-pinned floor | derived (v2 A′, scope-corrected) |
| band/loop | ≤ exp(−c/r²) + inner | v2 B1–B2 |

**Assembly.** 1 − q(r, b) ≤ C(L)ℓ + C·r³ (peak + shells) + C·r⁵ (inner) + exp-small = **O(r³)**, uniformly for b in compacts. Every polynomial component is now either verified numerically against certified-coefficient theory or measured directly; the exp-small components carry degeneracy-proof constants. Remaining formal obligations: (U1), (U2), (U4)-type discharges; W-far/Lemma-G uniformity bookkeeping; Sublemma R0 full write-out. The matching lower bound (the exact law) remains open and now owns three measured constants: the 0.51 inner window coin, the 0.83–0.85 boundary acceptance, and the 0.24 strip/annulus ratio.

## 4. Process record

The P-L2 commitment "strip exponent ≥ 4" FAILED-AS-WRITTEN (measured 2.95): root cause, regime conflation in the pre-commitment ledger — the deep-strip (s ≫ r) flatness suppression was applied to a domain dominated by its s ≍ r edge, where the peak-zone physics rules. The pre-committed adversarial branch ("≤ 3 ⟹ re-architect") executed: the peak zone is now an explicit, measured component and the rate conclusion is unchanged and strengthened. Clause (i) (ratio monotonicity) PASSED. The turnover beyond the peak and the far-floor crossover were then measured rather than asserted.
