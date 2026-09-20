# LB-3 DISCHARGE - R0 / γ-LOC architecture item (SIDE24/q0)

**Date:** 2026-08-04 · **Certificate:** `lb3_certificate.py` (sha256 `6daa149cb5a7580a4c4e16e973da2cc08486db09219fbe83963319bbd17a6f75`),
transcript `transcript_normal.txt` (sha256 `96c90d791376fdaa135bc84e6c439441c57fe9e78475d0fcc1ead82cbc91091b`),
exit 0 in normal and `python -O` modes, **byte-identical transcripts** (`cmp` clean). All numbers below are recomputed there; no PASS label is trusted.

**Verdicts.** (A) R0 / MS-Sard covering: **CLOSED** at specialist-reviewed-conditional grade (KIMI-AUD-022 binding), residual conditionality = three execution-level clarifications, none touching the AO usage. (B) γ-LOC tube-local at the UB0-stated scope: **PARTIAL** - locality/measurability and termination-at-a-maximum clauses PROVED (given (A) + Bulinskaya); the terminal-value clause (1 − O(exp) above b) is OPEN, with the exact remaining gap isolated in §2.6 (one sentence).

---

## 0. Verbatim ledger statements bound

(L1) C031_LBRATE_Integration.md §6 (sha256 `e7998ef0…f32e`):
> "1. **R0 / MS-Sard covering** (derived-architecture since C015) - inherited by all AO statements.
> 2. **γ-LOC tube-local** (architecture) - inherited by the C022 positivity chain."

(L2) "C022 Observed Update.json" (sha256 `9bc0647b…d5cb`), `lemma_FD.gamma_LOC_corollary`:
> "far-field stabilization certified: all one-point conditional quantities within 2.2% of unconditional at d >= 3, < 1e-4 by d = 5, enveloped beyond 6; tube-local part remains architecture (unchanged scope)"

and its `obligations.carried`:
> "gamma-LOC tube-local architecture (inside Lemma UB0 scope)".

(L3) Lemma UB0 §4 (sha256 `c6a90469…41c0`) - **the exact UB0 scope of γ-LOC**:
> "**Hypothesis (γ-LOC).** The qualification verdict of a window saddle y - adjacency to x_M and the other-branch terminal's value comparison against b - is measurable with respect to the field on the union of (i) the O(r) pair neighborhood and (ii) the other branch's ascent tube up to its terminal, and the terminal-value law stabilizes: for rim/arch saddles the other branch exits the conditioned zone with running height > b and terminates at an ambient maximum of value > b with probability 1 − O(exp)."

with the same section's framing:
> "γ-LOC is the step that makes E[N_qual] a *local* Palm functional evaluable by the C020 instrument. It is stated as architecture: the program's remaining route to full proof is a quantitative off-pair decorrelation bound (carried obligation OBL-FAR-DECAY)."

(L4) Lemma UB0 §0 epistemic label:
> "the elder-verdict locality step is architecture-grade (stated as an explicit hypothesis (γ-LOC), supported by measurement, not proven)".

(L5) MS Shift R0 Closure §0 (sha256 `6575e2bb…6c32`) - what R0 is and where it enters:
> "**R0.** Let f be the stationary Gaussian field on T²_L satisfying (H1+)/(H2′), conditioned on the C006 pair event … Then almost surely the gradient flow of f restricted to any compact window is Morse-Smale on the complement of the degenerate-critical set: in particular there are **no saddle-to-saddle connections**, so the ascent separatrices emanating from any index-1 critical point terminate at local maxima (or leave the window), and the elder-rule pairing D(·) is well defined for the corridor-local analysis of C017-C019."
>
> "R0 enters the program in two load-bearing places: (i) Lemma LB0 (C017, Proved) assumes its conclusion to convert the window-saddle + separatrix event into D(M) ≠ S; (ii) the C018/C019 flow instruments type their branch terminals assuming generic termination."

(L6) SARD-G manuscript (sha256 `4a77f3b7…c1fe`, byte-match to the KIMI-AUD-022 bound input confirmed), §1 proposed theorem:
> "With probability one, there is no gradient trajectory whose α- and ω-limits are two distinct index-one saddles. Equivalently, for every ordered pair of distinct saddles p, q, W^u(p) ⋔ W^s(q), and in dimension two this excludes saddle-saddle heteroclinic orbits. The theorem is only about saddle-saddle connections."

for "the exact normalized periodized Bargmann-Fock Gaussian field on a two-dimensional torus" (abstract), with §2.1: "almost surely all critical points are nondegenerate; almost surely critical values are distinct."

(L7) KIMI-AUD-022 (sha256 `8efcd937…29b8`; body self-hash `b00f0721…0f88` recomputed in certificate S5):
> "VERDICT: APPROVE WITH CLARIFICATIONS. The argument is a correct and essentially complete proof at the standard-reading level, and its own conservative statement (conditional on the Sections 8-11 charting/measurability lemmas, no saddle-saddle heteroclinic connection a.s.) is VERIFIED as written. One of the five flagged points is resolved internally; two are standard consequences; two are genuine execution-level gaps, both repairable in principle, neither substantive."

(L8) Supplement Part F.4 (sha256 `33609b70…d41f`) - the strongest proved tube machinery:
> "2. **Inward branch, no threshold.** In T_in = {−1/2 < X < 1/2, |Y| ≤ m h(X)}, h(X) = 1/4 − X² … so Ġ_in/2 ≤ −2(R²/τ)h² < 0. Since X decreases strictly … X(t) → −1/2 and |Y(t)| ≤ m h(X(t)) → 0: convergence to M. The V2 auxiliary threshold δ₀ and terminal section are removed entirely…"
> "…terminal ledger P(5/4,Y) > 49/192 − 1/768 = 65/256 > 1/4 > 0 = P(M). No λ_max(T) bound used."
> "3. **Pin-preserving exact-field robustness (explicit hypotheses).** For F̃ = F + E with E(M) = E(S) = 0, ‖DE‖_∞ ≤ η_R := 1/(8192R³) … the corrected tube cost 12(η_R/R)(R²/τ)h² - the V3 integration had printed the under-counted (η_R/12)(R²/τ)h²; the correct, larger bound still fits comfortably, since 12η_R/R ≤ 12/8192 ≪ 2…"
> "**Residual Palm-transfer premise (open).** To infer P^MS_{r,t,b,κ}(D_r^c) = O(r³) uniformly on compact positive mark sets one still needs, in one finite-r side-24 coordinate system: (i)-(v)…"

and the F.6 status map row:
> "Deterministic capture/escape (F.4) | repaired under explicit hypotheses | side-24 normal-form verification".

---

## 1. Item (A): R0 - compatibility statement

### 1.1 What the AO assembly consumes

Per (L5) and C031 §5 (theorem-grade assembly "conditional on: … and the R0/γ-LOC architecture inherited by every AO statement"), the consumption is exactly:

- **(C-i)** Lemma LB0 (C017): convert the window-saddle + separatrix event into D(M) ≠ S - needs the elder-rule pairing D(·) well-defined, i.e. ascending separatrices of index-1 points terminate at local maxima (no saddle-saddle heteroclinics), plus nondegenerate criticals and distinct critical values (Bulinskaya, (L6) §2.1).
- **(C-ii)** The C018/C019 flow instruments' terminal typing, and the C019 arch interpretation (filament = D(M) witness) - same content.
- **(C-iii)** Lemma UB0 §1: "By the Morse-Smale merge mechanics (Assumption MS, master source §S3-S4), on the failure event exactly one of two exchange geometries realizes" - the α/β dichotomy presupposes the a.s. Morse-Smale merge tree.
- **(C-iv)** Every AO(y) qualification verdict (adjacency + terminal-value comparison) is well-defined only if branch terminals exist and are maxima a.s.

In one line: **a.s. Morse-Smale structure of the (pair-Palm conditioned) periodized BF gradient flow on the compact side-24 torus, specifically excluding saddle-saddle heteroclinic orbits.**

### 1.2 What SARD-G now provides, and scope match

Per (L6)+(L7), at **specialist-reviewed-conditional** grade (organizationally distinct third-provider review, verdict bound byte-exact to manuscript sha256 `4a77f3b7…`):

- a.s. no saddle-saddle heteroclinic connections for the **exact normalized periodized Bargmann-Fock field on the side-24 torus** - precisely (C-i)-(C-iv)'s dynamical content; "the theorem is only about saddle-saddle connections" matches the C015 equivalence MS-Sard ⇔ no heteroclinic saddle connections (MS Shift §1).
- a.s. nondegenerate criticals and a.s. distinct critical values (§2.1 Bulinskaya package) - the Morse data.
- The torus is compact, so MS Shift's qualifier "restricted to any compact window" is subsumed.

**Conditioned-fiber transfer (the AO statements run under the pair-Palm law).** Three independent coverages, none new to this discharge: (i) MS Shift §2 - the Palm law is absolutely continuous w.r.t. the pinned law with density ∝ W₂, so null sets coincide, and the pinned law is the regular conditional Gaussian; (ii) SARD-G's own slicing re-runs verbatim inside the pin-annihilating Cameron-Martin subspace: Proposition 6.1's nonvanishing uses test functions supported near an interior orbit point disjoint from {p, q}, and the pin set {M, S} (resp. {M, S, y*}) is finite, so supports avoid it and the dense direction family can be chosen in the codimension-6 (resp. 9) fiber CM space; (iii) the in-program MS-Sard route executed the shift-Fubini directly in the conditioned fiber (det T = 1.058×10⁻³⁶ ≠ 0 at the representative configuration, C015) and remains in the record as an independent architecture-grade proof of the same conclusion. The transfer is routine and is not among the KIMI clarifications.

### 1.3 Residual conditionality, and whether it touches the AO usage

The verdict's three execution-level clarifications:
- **Clarification A** - complete the §8 measurable-selection writeup, or replace by the finite-dimensional universal moduli space + parametric transversality route (Abraham-Robbin);
- **Clarification B** - make the chart validity-set restriction explicit in the §10 disintegration display;
- **Clarification C** - display the endpoint-atom coefficient formulas, or add the Hirsch-Pugh-Shub citation with the local-manifold variation explicitly bounded.

Assessment: "both repairable in principle, neither substantive" (L7); flagged point 4 was "RESOLVED INTERNALLY" by the manuscript's own §2.1. **None touches the AO usage**: the AO assembly consumes only the conclusion (a.s. MS + no saddle-saddle connections), and A-C are bookkeeping completions of the proof's covering/disintegration formalism, not hypotheses on the flow. The old in-program residual (OBL-R0-UNIF, uniformity of the 18×18 nondegeneracy input to the MS-Sard route) was discharged separately at bookkeeping grade (R0UNIF Part A, continuity + compactness + universal interpolation; mp anchors 9.7267×10⁻²¹ / 3.7283×10⁻²³ at the two rungs - cited, not recomputed here) and is moot under the SARD-G route.

**Verdict (A): CLOSED** - grade upgraded from "derived-architecture (residual uniformity step)" to **specialist-reviewed-conditional**; conditional only on clarifications A-C (execution-level), which do not intersect the AO consumption.

---

## 2. Item (B): γ-LOC tube-local at the UB0-stated scope

### 2.0 Decomposition

γ-LOC (L3) = clause (i) locality/measurability + clause (ii) terminal-value stabilization, which itself splits: (ii-a) the other branch terminates; (ii-b) the terminal is a maximum; (ii-c) the terminal value is > b with probability 1 − O(exp).

### 2.1 Clause (i) - locality/measurability - PROVED (derived-standard; same formality class as Clarification A)

**Lemma (germ-measurability of the qualification verdict).** On the a.s. Morse-Smale event (Item (A) + Bulinskaya), for any index-1 critical point y of the conditioned field in the O(r) pair neighborhood with f(y) ∈ (b−ℓ, b), and either ascending branch β: the events {β terminates at x_M} and {β terminates at a maximum z with f(z) > b} are measurable w.r.t. σ(f | N), N = (O(r) pair neighborhood) ∪ (ascent tube of β up to its terminal).

*Proof.* β̄ = β ∪ {z} is compact (termination proved in §2.2). (a) Launch data: y and its unstable eigendirection are C¹ functions of the 2-jet of f at y (implicit function theorem at a nondegenerate critical point), and y lies in the pair neighborhood (rim disk δ_w = √(2ℓ/stiff) ≈ 0.005, resp. the arch filament at |y| ≈ 0.010-0.012 - both O(r) at r = 0.05, per UB0 §3). (b) Finite-time segment: on any finite horizon T the orbit is a C⁰ function of ∇f restricted to any tube around β([0,T]) (Grönwall along a compact regular segment; ‖∇f‖ bounded below off the critical set). (c) Terminal capture: z is a nondegenerate maximum with f(z) > b (resp. = b for x_M); choose ε > 0 with z the unique critical point in B(z, 2ε) and f > b on B(z, 2ε) - convergence to z is then determined by f|B(z,2ε), and there exists T < ∞ with β([T,∞)) ⊂ B(z,ε). (d) The event is the countable union, over rational T, rational polygonal tubes, and rational ε-balls, of open C¹ conditions on f restricted to those fixed rational neighborhoods - the standard rational-chart device (identical in form to SARD-G §8 / MS Shift Lemma B), hence σ(f|N)-measurable. The strict value comparison f(z) > b is read off f|B(z,ε) ⊂ N. ∎

The only non-explicit sub-step is the measurable-selection bookkeeping of (d) - the *same* standard consequence KIMI-AUD-022 names as Clarification A for SARD-G; clause (i) therefore stands at the same conditional grade as (A), and no higher formality is consumed by UB0, which needs exactly this σ-algebra ("measurable with respect to the field on the union of (i) the O(r) pair neighborhood and (ii) the other branch's ascent tube up to its terminal" - verbatim match).

### 2.2 Clauses (ii-a), (ii-b) - termination at a maximum - PROVED

On the compact torus, for a gradient flow of a Morse field: the ω-limit of any orbit is nonempty, compact, connected, and consists of critical points and heteroclinic connections among them; with **no saddle-saddle connections** (Item (A)) and **distinct critical values** (Bulinskaya), the ω-limit of an ascending branch β is a single critical point z. z cannot be index 0 (f strictly increases along β from f(y) > b − ℓ) nor index 1 (that would be a saddle-saddle heteroclinic orbit). Hence **z is a local maximum**, f(z) > b − ℓ. ∎ (This is also exactly what the C018/C019 instruments assume, per (L5)(ii).)

Terminating at a torus image of x_M (value exactly b) fails the strict inequality "> b"; as UB0 §4 records, reaching it "requires descending below the running height - impossible under ascent" **once running height > b** - so this sub-case routes into clause (ii-c) and is not separately closable by dynamics alone (the stable manifold of an image of x_M is open; the exclusion is the running-height input, not a null-set argument).

### 2.3 Clause (ii-c) - the stochastic core - what the proved machinery gives

**F.4 deterministic skeleton (machine-verified in certificate S1, 64 exact-rational checks at R = 1, 2, 5, 50, τ at floor).** Re-verified independently: cone flux s4 = 257/1024 + 257/(1024K) + 1/(2048K²) + ε₀ = 0.251038 < 1/3; strip energy s5 = 0.875153 < 9/10; inward-tube adverse budget 1 + 1/K + 1/(4K²) + 6ε₀ + 4R²/(Kτ) = 1.000246 < 2 (against the leading −4(R²/τ)h²); terminal ledger 49/192 − 1/768 = 65/256 > 1/4 > 0 = P(M); pin-preserving costs 3η_R/R (cone), 4η_R/(3R) (strip), and the **corrected** tube cost 12(η_R/R)(R²/τ)h² with ratio 12η_R/R ≤ 12/8192 (equality at R = 1) < 2 − 1.000246; η_R = δ₀/16 identity; the withdrawn V2 claim h(−1/2+δ₀) ≥ δ₀ regression-pinned false (δ₀ − δ₀² < δ₀). Interpretation for γ-LOC: in the side-24 normal form, the endpoint saddle's outward branch **escapes the conditioned zone with a uniform scaled height margin above the M-pin level** (the ledger 65/256), the inward branch is captured by M with no threshold, and both persist under pin-preserving C¹ perturbations up to η_R = 1/(8192R³). This is the deterministic template of (ii-c).

**C022 far-field stabilization (recomputed in certificate S2-S4 from the exact periodized kernel: spectral lattice (π/12)ℤ², masses e^{−|k|²/2}, truncation |k| ≤ 30, tail ≤ 4.4×10⁻¹⁸⁵ < 10⁻⁶⁰, mpmath dps 60/80).** Certified (on-grid maxima at 720 angles × dr = 0.03 polar per pin, argmax refined in mp dps 60; envelopes rigorous and monotone beyond):

| quantity (max over one-point jet components f, ∂x, ∂y, ∂xx, ∂xy, ∂yy) | 6-pin r=0.05 | 9-pin r=0.05 | 6-pin r=0.025 | 9-pin r=0.025 |
|---|---|---|---|---|
| variance deviation, d ≥ 3 | 8.90e-2 | 1.03e-1 | 9.25e-2 | 1.01e-1 |
| **value-variance deviation, d ≥ 3** | **1.91e-2** | **2.24e-2** | **2.01e-2** | **2.13e-2** |
| variance deviation, d ≥ 5 | 3.84e-6 | 4.54e-6 | 4.22e-6 | 4.37e-6 |
| value-law TV (testbed pin values), d ≥ 3 | 6.58e-2 | 1.54e-1 | 6.77e-2 | 1.32e-1 |
| value-law TV, d = 5 | **7.70e-5** | 2.24e-4 | **8.14e-5** | 1.85e-4 |
| rigorous envelope < 1e-4 beyond d = (variance / mean) | 6.1 / 6.7 | 6.6 / 7.1 | 6.4 / 7.0 | 6.9 / 7.5 |

Read against (L2): the "2.2% at d ≥ 3" is **confirmed as the value-variance (rigidity) channel** (certified max 2.24% vs ledger 2.2% - consistent with C030's own "v = 0.974 ↔ TV 2.2%", given their coarser station set); "< 1e-4 by d = 5" is **confirmed for the 6-pin value law** (7.7e-5 / 8.1e-5) and for all variance quantities (≤ 4.6e-6), marginally exceeded for the 9-pin value TV (2.2e-4); "enveloped beyond 6" is **confirmed** for the 6-pin variance envelope (d ≥ 6.1) and approximately for the others (≤ 7.5). One honest discrepancy, documented in the transcript: read as a full one-point-law TV including the conditional mean at the testbed pin values, the deviation at d ≥ 3 is 6.6%-31%, not 2.2% - driven by the null-mode/needle response of the mean to the ℓ pin-difference (‖G6⁻¹v‖₁ = 65560.02 / 518320.01, matching C022's dual weights 6.556e4 / 5.183e5, which is exactly this amplification). This discrepancy is **non-load-bearing**: C031 G1/R3′ explicitly removed the additive-TV usage ("theorem-grade far-term positivity needs no TV subtraction; Lemma FD is repositioned as the quantitative channel only"), and C032 repositioned Lemma FD out of the load path.

### 2.4 Discharge attempt and the exact remaining gap

Given §2.1-§2.3, the only unproved clause is (ii-c): the probability, under the 6/9-pin conditional law, that the outward branch of an arch/rim window saddle is captured below b (terminal maximum value ∈ (b−ℓ, b]) is O(exp). The F.4 machinery supplies the deterministic confinement + ledger margin + robustness radius for the **endpoint** saddle of the side-24 normal form; Lemma FD (certified above) supplies the far-zone stabilization from d ≈ 3-5 outward. What neither supplies is the **near/moderate-zone** (d ≲ 3) uniform tube margin at the **third-saddle** arch/rim configuration, transferred to a conditional-measure tail.

> **The gap, in one sentence:** γ-LOC(ii-c) lacks exactly a uniform C¹-over-the-ascent-tube margin between the pin-conditioned nominal field and the capture-below-b boundary for the third-saddle arch/rim escape configuration in the near/moderate zone d ≲ 3 (where Lemma FD's stabilization does not reach and the conditional mean's needle response is order-one-tenths of σ), together with the transfer of that margin to a 1 − O(exp) Gaussian tube-tail bound under the conditional measure - F.4's machine-verified margins being proved, as the supplement's own F.6 map records, for the endpoint-saddle normal form with "side-24 normal-form verification" still outstanding, and the Palm-transfer premise (F.4 residual (i)-(v)) covering the capture side only.

The measured support remains as UB0 §4 and the C021 addendum record it (m* profile 4.6-9.7σ above b at the probes; esc = 0 in 11,200 + 28,800 + 6,400 pooled compatible-ensemble flows), so the architecture label's empirical basis is unchanged; what is isolated above is the precise analytic step separating the label from a proof.

### 2.5 Falsifier (named, verbatim from UB0 §6 + addendum)

> "a rim/arch flow whose other branch terminates below b at an off-image maximum (none in 11,200 pooled compatible-ensemble fringe flows)" - addendum A.4: "none in 11,200 (C020) + 28,800 (C021) + 6,400 (ballfix) pooled flows"; and "a violation of the γ-LOC tube-locality by long-range conditioning leakage (constrained by the far-decorrelation probes; formal bound carried as OBL-FAR-DECAY)".

For item (A), the KIMI-AUD-022 falsifier binds: "exhibit a saddle-saddle connection chart where the curve density vanishes identically on the regular segment …; a compactly supported finite-order distribution on the torus whose kernel embedding vanishes but which is nonzero …; an error in the adjoint identity or the δp formula …; or a standard-reference counterexample showing the chart covering/measurable selection fails in this setting."

**Verdict (B): PARTIAL** - clauses (i), (ii-a), (ii-b) discharged at the grade of (A); clause (ii-c) OPEN with the gap isolated in §2.4.

---

## 3. Hypothesis list (everything consumed beyond the cited artifacts)

(H1) Bulinskaya/(H2′) package: a.s. nondegenerate criticals, distinct critical values, finitely many criticals per compact window (standard; restated in SARD-G §2.1).
(H2) Gaussian measure theory: Cameron-Martin quasi-invariance, RKHS injectivity for the periodized BF spectral weights e^{−2π²|k|²/L²} > 0 (KIMI receipt 1, VERIFIED), pin-conditioned disintegration, Palm ∝ pinned·W₂ null-set equivalence (MS Shift §2).
(H3) SARD-G manuscript at KIMI-AUD-022 verdict (L7), clarifications A-C execution-level.
(H4) Compactness of the side-24 torus; gradient-flow ω-limit structure for Morse fields (Palis-de Melo class).
(H5) Grönwall/IFT regularity for finite-time orbit segments and nondegenerate critical points (clause (i) proof).
(H6) Rational-chart measurable selection of the same class as Clarification A (clause (i), shared formality).
(H7) The exact periodized kernel as specified: lattice (π/12)ℤ², masses e^{−|k|²/2}, truncation |k| ≤ 30 (tail certified < 10⁻⁶⁰; spectral moments λ₂ = 1, λ₄ = 3, λ₂₂ = 1 to < 10⁻⁶⁰).
(H8) Testbed constants: b = 6/5, ℓ = r³/6, r ∈ {0.05, 0.025}, pins M = (−r/2, 0), S = (r/2, 0), y* = (−0.063, 0.012) / (−0.030, 0.007), v* = b − ℓ/2.
No measured quantity is consumed by the proofs of (A), §2.1, §2.2; measured values are quoted only as support/falsifier status per the ledger's own labels.

## 4. Dependency table (name - sha256 - role)

| artifact | sha256 (16) | role |
|---|---|---|
| C031_LBRATE_Integration.md | e7998ef0d17d951b | ledger items §6.1-6.2 (L1); AO assembly consumption |
| C022 Observed Update.json | 9bc0647b885931ba | γ-LOC corollary (L2); dual weights 6.556e4/5.183e5; gram9 1.321e-12 |
| Lemma UB0.md | c6a90469b6e43c59 | γ-LOC exact scope (L3); falsifier |
| Lemma UB0 Addendum.md | 224438cb740565ff | corrected rim picture; pooled-flow falsifier counts |
| R0UNIF and LBRestate.md | 0e9bb486c2dec103 | OBL-R0-UNIF discharge (cited anchors) |
| MS Shift R0 Closure.md | 6575e2bb5178e122 | R0 statement/consumption (L5); fiber transfer |
| SARD-G manuscript (WORKING-SOURCE 19242B) | 4a77f3b7ed390c0e | (A)'s theorem (L6); byte-match to KIMI bound input |
| KIMI-AUD-022.md | 8efcd937502b67ab | specialist verdict (L7); self-hash b00f0721… verified |
| SIDE24_GAP_FILL_SUPPLEMENT.md | 33609b70d8c4d970 | F.4 machinery (L8); F.6 residue |
| lb3_certificate.py (this discharge) | 6daa149cb5a7580a | fail-closed certificate |
| transcript_normal.txt = transcript_O.txt | 96c90d791376fdaa | certified transcript, byte-identical across -O |

## 5. Reproduction

```
cd /mnt/agents/output/lb3
python3 lb3_certificate.py > t1.txt    # exit 0
python3 -O lb3_certificate.py > t2.txt # exit 0
cmp t1.txt t2.txt                      # byte-identical
```
Runtime ≈ 55 s per mode. Fail-closed: any check failure prints `CHECK FAILED: …` and raises `SystemExit(1)`; `LB3_CERTIFICATE_PASS` is printed only after all 64 S1 exact-rational checks, S2 kernel certificates, S3 Gram eigencerts (λ_min > 10³×residual; margins 10⁴⁶-10⁵⁰), S4 stabilization thresholds, and S5 hash bindings pass.
