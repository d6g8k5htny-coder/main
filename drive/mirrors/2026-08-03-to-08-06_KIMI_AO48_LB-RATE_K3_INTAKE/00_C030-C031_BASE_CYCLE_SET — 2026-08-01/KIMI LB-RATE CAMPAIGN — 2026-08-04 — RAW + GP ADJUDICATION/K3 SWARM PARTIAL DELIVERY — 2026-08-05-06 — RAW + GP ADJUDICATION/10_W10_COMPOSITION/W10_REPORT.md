# W10 — DEPENDENCY MINIMIZATION: the lower-bound event inclusion, the per-channel minimal lemmas, and the WP-scaling question

**Workstream:** K3 / W10 (dependency-minimization), SIDE24/q0 lower-rate repair.
**Date:** 2026-08-05. **Certificate:** `w10_sanity.py` (fail-closed, `SystemExit`, no bare asserts; normal and `-O` transcripts byte-identical), transcripts `t_normal.txt` / `t_O.txt`.
**Inputs (read-only; nothing in the certificate tree was modified):** manuscript §2.3–2.8, §3.2–3.11, §6 (`SIDE24_pre_peer_review/manuscript.md`); C024 Corrections+Freeze; C025 TerminalHeight; C026/C027 Foundation; C028 FarAscent; C029 BasinPersistence; C030 CountingLemmas; C031_LBRATE_Integration (e7998ef0…); KIMI-THM-023 v1.0 (ba3974c0…) and v1.1 (noncontrolling HOLD draft); KIMI-DER-025 (7e76c548…); K3 master work order; K3 `00_READ_FIRST.md` (controlling facts 1–6); Q0_MASTER.md; CL-AUD-S2DER010 (the estimand audit of record for p_r / q_r^adj / η_r).
**Grade discipline:** the manuscript is context, not authority. Per K3 controlling fact 1, the old WP "certified upper bound" (missing square root) and everything derived from it (I_cs ≈ 1.30e-5 as a bound, the ub rung table, the 5.5e-3·r^1.6 envelope as a *bound*, the ub branch of the constant ledger) are **invalid as upper bounds** and are used below only as *scaling evidence*, never as bounds. The two-sided estimate I_true ≈ 4.7602e-6 is derived-on-grid evidence, not a bound and not a refutation.

---

## 0. Executive answer

**Does the assembly need WP = O(r³)? NO.** The WP channel enters the composition only as an *absolute additive probability loss* inside the reliability factor AO; the r³ rate of the final bound is supplied entirely by the Λ-side window factor (ℓ = r³/6). The assembly's exact requirement on WP is:

> **(WP-min)** an explicit, *valid* (genuinely upper-bounding) modulus E_WP(r), defined and certified on the whole interval (0, r₀], with E_WP(r) → 0 as r → 0 and E_WP(r₀) inside the AO budget.

Any E_WP(r) = C·r^α with **any α > 0** suffices; α ≈ 1.6 (or even α ≈ 0.8) changes the final constant c only through the evaluated-at-r₀ number, not through the architecture. What WP = O(r³) would *additionally* buy is only (i) the ornamental remainder display "(1 − O(r³))" and (ii) a bounded per-r³ coefficient ledger. What it is *not* needed for: the theorem 1 − q(r,6/5) ≥ c·r³ itself, in any of the honest composition forms of §6. The genuinely load-bearing WP requirements are **validity** (a real upper bound — the property the missing-sqrt defect destroyed), **interval coverage** (no rung gaps on (0, r₀]), and **smallness at r₀ against the AO budget** (§5.3: headroom ×1460 at the measured far tier at r₀ = 0.05; at the theorem far tier the budget is ~0.0896·P₀ and r₀ must shrink with P₀ — arithmetic in §5.4).

The companion finding: "WP = o(1) relative to the window mass" is **false and irrelevant** — the observed truth estimate scales like r^{1.4–1.6} while the window mass is ℓ = Θ(r³), so the count *grows* relative to ℓ (truth/r³ ≈ 0.11 → 11.4 across the rungs). The assembly never consumes that ratio.

---

## 1. The estimands, stated precisely (the K3 firewall, satisfied explicitly)

**Field and law.** f = exact normalized periodized Bargmann–Fock field on T²₂₄ (spectral masses e^{−|k|²/2} on (π/12)Z², Var f = 1), b = 6/5, ℓ = r³/6. A.s. Morse with distinct critical values (Bulinskaya package; full spectral support); a.s. Morse–Smale is the SARD-G conclusion (specialist-reviewed-conditional grade; used conclusion-only). The **typed pair-Palm law** P⁰_r is the regular conditional law given the corrected pin basis (manuscript (2.7)) at the pair (M, S), M = (−r/2, 0), S = (r/2, 0), values f(M) = b, f(S) = b − ℓ, ∇f = 0 at both, with RN weight W_r = |det H_M det H_S|·1_typed and normalizer Z_r ≍ r² (manuscript Preliminary 2.5, Thm 3.11, Prop. 3.20 — certified). The **9-pin law** used on the AO side conditions additionally on the jet at the arch station Y = M + r·(−0.76, 0.24) with f(Y) = μ_t (the 6-pin gradient-conditioned mean, (μ_t − b)/ℓ → −0.4999290, C026 exact rational −3476069/6953125), ∇f(Y) = 0; the C024 estimand's ratio E[W₃1_typed | 9 pins] / E[W₂1_typed | 𝒥₆] is exactly the third-point Palm correction connecting the two laws. The AO-side witness level is v* := b − ℓ/2 (mid-window; C025 R1 floor).

**The four estimands (object cards).** For a Morse realization write D(M) for the elder-rule death partner of the maximum M (superlevel merge tree), and A = A_f(M, S) for the *direct gradient-adjacency* event (a separatrix of S has ω-limit M).

| estimand | definition | object |
|---|---|---|
| p_r | P⁰_r{ D(M) = S } | merge-tree pairing probability of the typed pair |
| a_r | P⁰_r{ A = 1 } | direct-adjacency probability |
| q_r^adj | P⁰_r{ D(M) = S ∣ A = 1 } | adjacent-pairing probability |
| η_r | P⁰_r{ D(M) = S, A = 0 } | pairing *without* direct adjacency |

Exact partition (CL-AUD-S2DER010, Finding C2/R10, verified algebra):

    p_r = a_r · q_r^adj + η_r,        0 ≤ η_r ≤ 1 − a_r,
    1 − p_r = (1 − a_r) + a_r(1 − q_r^adj) − η_r = P{A=0, D≠S} + P{A=1, D≠S}.

**Identification for this campaign:** the theorem target's q(r, 6/5) **is p_r** (the typed merge-tree pairing probability). Manuscript Def. 2.8's phrasing ("ascending trajectories reach a maximum of height > b which elder-rule pairs with it (equivalently, the pairing succeeds)") mixes the adjacency-type event into the definition; the load-bearing estimand is the merge-tree one, and manuscript Lemma 3.5 (exhaustion trichotomy) is exactly the statement that, off the SARD-G-null set, the merge-tree pairing is determined by the ω-limits of the two unstable separatrices. Every use below is of p_r.

**The η_r channel — where it does and does not enter.** (i) The lower-bound assembly of §2–§3 lower-bounds 1 − p_r **directly**, by exhibiting a sub-event of {D(M) ≠ S}; it never conditions on A and never passes through q_r^adj, so **no η_r loss is incurred anywhere in the lower-bound chain**, and no typed-to-adjacent transfer is performed. (ii) The identity p_r = a_r q_r^adj + η_r shows the same lower bound is *simultaneously* valid for 1 − a_r q_r^adj − η_r — i.e. any consumer who re-reads the theorem in adjacent terms must retain η_r explicitly (the work-order firewall item 9; gate G10). (iii) The intensity-level identity (CL-AUD R10) I_r·q_r^adj = J_r·a_r·q_r^adj = J_r·(p_r − η_r) — with I_r the typed candidate-pair intensity and J_r the adjacency-filtered intensity — is where η_r is genuinely load-bearing: **on the upper side** (the adjacency-filtered first moment equals the complete persistence first moment iff J_r η_r vanishes). The ratified upper chain is outside this campaign; we record the identity so the successor assembly cannot silently drop η_r. (iv) Measurability caveat (CL-AUD C2): Theorem-grade measurability exists for the persistence mark D(M) = S (manuscript Prop. 3.20) but was never separately established for A_f(M, S); the assembly of §2 is therefore deliberately built from ω-limit events (trajectory convergence to critical points — measurable under the a.s. Morse–Smale conclusion) and merge-tree events, never from the bare adjacency mark. The arch-side "non-adjacency" channel (§2.4) concerns Y's branch reaching M, an ω-limit event — a *different* adjacency from A_f(M, S); the two must not be conflated (firewall).

---

## 2. The event inclusion, reconstructed from first principles

### 2.1 The pairing event and its complement

Under P⁰_r, on the a.s.-Morse–Smale full-measure set, the negative-gradient flow's unstable separatrices from any saddle converge to maxima, and the merge tree is well-defined. The pairing event is E_pair = { D(M) = S }: M's superlevel component, born at f(M) = b, first merges with a strictly older component (one whose maximum has height > b) exactly at the saddle S (height b − ℓ); the persistence of the pair is ℓ.

**Exhaustion (manuscript Lemma 3.5, restated for the maximum side).** Exactly one of the following holds:

- **F0 (degenerate).** A saddle–saddle connection or degenerate critical point is involved in the relevant merges. Probability 0 (Bulinskaya + SARD-G, conclusion-only). Channel name: **R0**.
- **F1 (preemption / window pass).** M's component merges with an older component at a saddle Y of height h_Y ∈ (b − ℓ, b), strictly above S. Then D(M) = Y ≠ S; the realized pair (M, Y) has persistence b − h_Y < ℓ.
- **F2 (non-connection).** S does not border M's component at all (both of S's separatrices reach maxima outside M's component).
- **F3 (younger merge at S / terminal ≤ b).** S borders M's component, but the component on S's other side has maximum height ≤ b: then that younger component's maximum pairs with S, M survives, D(M) ≠ S.
- **S (success).** S borders M's component and the other side's maximum has height > b: D(M) = S.

F0–F3 are the *complete* failure modes of the elder-rule pairing of the typed pair; {D(M) ≠ S} = F0 ∪ F1 ∪ F2 ∪ F3 (disjointness not needed). This is the B-side of the inclusion; the C024-class "selection inequality" (manuscript Lemma 3.6) is the union bound over these modes.

### 2.2 The constructive sub-event (the arch qualifying event)

The lower bound does not estimate P(F1 ∪ F2 ∪ F3) from above; it *exhibits* a controlled sub-event of F1. Fix the arch region Ũ ⊂ R² (the two lobes ±23° off the rear axis behind M at |ỹ| ≈ 0.75–0.8; C025 qualification map, C027 exact-limit lobe geometry), and for y = M + r·ỹ define the **qualifying event** Q(y):

    Q(y) = { Y := y is a critical point, typed saddle (det H_Y < 0),
             height f(Y) ∈ (b − ℓ, b)  [the window],
             branch₁: one unstable separatrix of Y has ω-limit M,
             branch₂: the other separatrix terminates at a maximum of height > b }.

**Inclusion lemma (mountain-pass/elder-rule, the B1–B4 reduction of C029 read forward).** On the Morse–Smale set, Q(y) ⇒ F1: at level h_Y ∈ (b−ℓ, b), Y merges the component containing M (born at b) with the component of an older maximum (born > b); the elder rule kills M's component, so D(M) = Y ≠ S. Hence

    { Σ_y 1_{Q(y)} ≥ 1 }  ⊆  F1  ⊆  { D(M) ≠ S } = {pairing fails}.

The reduction's hypotheses: R1 monotone ascent (gradient flow increases f — Established); B1 the branch trajectories stay at height ≥ v* := b − ℓ/2 once above it (monotonicity); B3 the mountain-pass theorem conditional on a.s.-Morse (Established-Math class); B4 exit treated separately. No adjacency mark A_f(M, S) appears: the branch conditions are ω-limit events.

### 2.3 Bonferroni factorization (the C024 reduction)

Let N_qual = Σ over critical points y of 1_{Q(y)}. Then

    1 − q(r, b) = P⁰_r{ D(M) ≠ S } ≥ P⁰_r{ N_qual ≥ 1 } ≥ E⁰_r[N_qual] − ½ E⁰_r[N_qual(N_qual − 1)].

Campbell/Palm disintegration at the third point y (the C024 frozen estimand, verbatim):

    E⁰_r[N_qual] = ∫_{arch} Λ(y) · AO(y) dA(y),
    Λ(y) = φ₂(∇f(y)=0 | 𝒥₆, with mean) · [Φ((b−μ_t)/s_t) − Φ((b−ℓ−μ_t)/s_t)]
           · E[W₃ 1_typed | 9 pins @ v* = clip(μ_t)] / E[W₂ 1_typed | 𝒥₆],

with AO(y) = P⁰_r(branch₁ ∧ branch₂ | the 9-pin law with Y typed in the window at y) — the **reliability** of the window saddle as a genuine pairing failure. The pair term E[N(N−1)] = O(r⁶) is registered at C022 through the marked near-diagonal law (ND′) plus far decorrelation (program grade; the C091 quarantine applies only to literature-only justifications, not to the internal certificate — Q0_MASTER §4.4). Since E[N_qual] = Θ(r³), the relative Bonferroni cost is O(r³).

### 2.4 The reliability decomposition (C025 R1–R4 + C029 B1–B4, re-derived)

AO-failure = {branch₁ fails} ∪ {branch₂ fails}:

- **branch₁ (BP-failure, capture of the near branch by M).** B1–B4: BP-failure ⊆ {exit from B_ρr(M)} ∪ {diversion to a maximum m′ ≠ M in B_ρr}. Diversion with trajectory height ≥ v* forces (B3, mountain pass) a saddle of height ≥ v* − ε between m′ and M, which is one of:
  (a) a **window-class saddle** (height in (b−ℓ, b)) in the pass zone — channel **WP**, P ≤ E⁰_r[N_ws(B₃ ∖ collars)] (Markov);
  (b) an **above-b saddle** near the pinned pair — channel **above-b**, P ≤ E⁰_r[N_crit(f ≥ b, B_{3r}(M))];
  (c) a **component split along the mean ridge** with no extra critical point — channel **ridge split**, excluded deterministically against the certified conditional-mean census plus fluctuation control.
- **branch₂ (terminal ≤ b).** R2: P(terminal ≤ b) ≤ E⁰_r[N_maxband(B_{d₀})] + far band-termination, where the **near term** counts maxima in the height band (b − ℓ/2, b] inside B_{d₀} (d₀ = 5), and the **far term** is the probability that the trajectory crosses ∂B_{d₀} in the band and then terminates ≤ b. R1 monotone ascent ⇒ far-zone entry at height ≥ b − ℓ/2. The far term is bounded at *fixed scale* (r-free): by R4/OBL-FAR-ASCENT(δ₀) with band monotonicity, far ≤ p̄(δ₀) for all ℓ/2 ≤ δ₀; the **selection/disintegration lemma** (KIMI-DER-009 + DER-009b premise discharge) removes the Rice multiplicity 2.43 and gives far ≤ sup_{ξ∈band} p̄(ξ) =: far_single; the R3′ conditional-barrier transfer (C031 G1, 36× margin) routes the C028 positivity *through* the 9-pin conditioning so **no TV subtraction** appears at theorem grade (Lemma FD repositioned as the quantitative channel only).

Union bound (exactly the C031/manuscript (3.12) ledger, now derived):

    AO(y) ≥ AO⁰ − [R2] − [WP] − [above-b] − [ridge split] − [exit],
    AO⁰ = 1 − far_single,

uniformly in y over the arch (the channels' bounds are y-independent at their stated grades: the R2/WP/above-b counts are over fixed regions around the pair; the ridge census is global in B_{2r}(M); exit and far are branch events).

### 2.5 The composition

    1 − q(r, 6/5) ≥ (∫_arch Λ dA) · inf_arch AO · (1 − O(r³))
                  ≥ c_Λ(r) · r³ · ( AO⁰ − Σ channels ) · (1 − O(r³)),

the probability factorization being: [window-saddle first moment over the arch] × [conditional reliability] − [pair term]. This is the entire architecture. **Every channel except far_single is consumed only as an absolute, additive, vanishing probability; far_single is the unique fixed-constant input; c_Λ·r³ is the unique source of the rate.**

---

## 3. Per-channel minimal sufficiency (the load-bearing table)

"Minimal grade" is relative to the target theorem 1 − q ≥ c·r³ ∀ r ∈ (0, r₀] with explicit c, r₀. The liminf-only requirement is one column left.

| # | channel | consumed as | minimal for **liminf** (1−q)/r³ ≥ AO⁰·c_Λ^∞ | minimal for **explicit-r₀ theorem** | currently available (honest grade, post-quarantine) |
|---|---|---|---|---|---|
| 1 | **R2 near maxima** E⁰[N_maxband(B₅)] | absolute loss | o(1) | explicit modulus; O(r³) with constant available: ≤ 0.1759·r³ (exact-Fraction check S7) | derived-structure + measured constants (ρ_mx(1.2) = 0.043685 derived, E-term 1.41350; G-F1 two-route 9.5%); KR validity + collar exclusion remain named rigidity-class formalities (C-4). Misprint (9−6.25)→(9−2.25) registered; corrected reading reproduces 2.111·(ℓ/2) (S7). |
| 2 | **WP window-pass** E⁰[N_ws(B₃∖collars)] | absolute loss | o(1) (unquantified) | explicit valid modulus E_WP(r) ↓ 0 on (0, r₀], **any rate α > 0**, inside the §5 budget | **OPEN** (K3 W2/W3/W4). Old ub invalid (missing sqrt, K3 fact 1). Scaling evidence (not bounds): truth est. decade slope 1.453, ub-table slope 1.557, truth/r³ grows ×103 over the decade (S2). |
| 3 | **above-b saddles** E⁰[N_crit(f ≥ b, B_{3r})] | absolute loss | o(1) | explicit modulus; available at ≪ 10⁻³·r³, superexponentially strengthening (E/tol 1.7e-23 at r=0.05 → 2.5e-506 at r=0.0125) | CLOSED at the localized form (LB-1; rigorous bound carries the sqrt — *unaffected* by the K3 defect, K3 fact 1). Interval coverage between rungs via the strengthening chain; sub-0.0125 needs the same formality class as WP. |
| 4 | **ridge split** | absolute loss | o(1) | 0 at certified rungs, or explicit modulus at all r ≤ r₀ | CLOSED at the two frozen rungs r ∈ {0.05, 0.025} (LB-2 mean census {M, yy, S} in B_{2r}, Kantorovich-hardened by DER-027c, uniqueness balls 0.106/0.080); r-uniform enclosure = mechanical strengthening, not yet executed. |
| 5 | **exit** (B4) | absolute loss | o(1) | explicit modulus; claimed 1 − O(exp) via γ-LOC | measured 0 (1750/1750 per rung, mp-exact instrument, negmass 2e-15) — measured grade only; theorem-grade form conditional on P-NMZ-γ (C-1). |
| 6 | **far** (far_single) | **fixed constant** < 1 | fixed p₀ < 1 (r-free) | fixed explicit p₀ < 1, or positive-but-non-explicit complement for the ∃-form | measured sup p̄ = 0.0334 (ν_T instrument, C025) with selection lemma (DER-009/009b, unconditional at R3′/C028 grade) removing the 2.43 multiplicity; theorem-grade p̄ ≤ 1 − 0.089569·P₀, P₀ > 0 by the Gaussian support theorem (C028), P₀ not explicitly floored (OBL-P0-FLOOR optional). |
| 7 | **R0 degenerate** | exclusion | P = 0 | P = 0 | SARD-G a.s.-Morse–Smale, specialist-reviewed-conditional; used conclusion-only. |
| 8 | **Λ-side window** ∫_arch Λ dA | the rate source | c_Λ^∞ > 0 (limit) | explicit c_Λ > 0 uniform on (0, r₀] | measured C*(0.05/0.025) = 0.9728/0.946 (rung-tagged measurements, never truth constants — Q0 binding note); derived-on-grid limit C*_∞ = 0.9091 ± 0.038 ± 0.009; certified-tier floor M_floor = 0.9001 conditional on H-B3 (C-7; full-grid transcripts running, W8); station-density formality C-6. S1–S6 factor limits exact (C026 rationals; S3 corrected to −0.4999290). |
| 9 | **Bonferroni pair term** | remainder | o(r³) | O(r⁶) at program grade | registered via (ND′) marked near-diagonal law + far decorrelation (program-grade closed internally; external audit status named). |
| — | Palm/disintegration architecture | the law itself | Z_r ≍ r², exact RN weight | same | certified (manuscript Thm 3.11, Prop. 3.20; FOUNDATION grade). |

**Uniform reading of the table.** Exactly one channel (far) requires a *fixed constant*; exactly one channel (Λ) supplies the *rate*; everything else is required only to **vanish with an explicit modulus and interval coverage**. This is the dependency minimization proper: no channel except Λ needs any specific power of r, and no AO channel needs a rate at all for the liminf form.

---

## 4. The Λ-side factor: what it actually must be

**Minimal statement.** ∫_arch Λ(y; r) dA ≥ c_Λ · r³ for all r ∈ (0, r₀], with **any** explicit c_Λ > 0. The value 0.946 is not required; c_Λ enters the final constant only as the multiplicative factor c = AO_lb · c_Λ. Three honest tiers:

- **measured tier:** c_Λ = C*(0.025) = 0.946 (rung-tagged measurement; the binding Q0/C092 rule: never a truth constant);
- **derived-on-grid tier:** c_Λ = C*_∞ = 0.9091 (exact-limit factors, quadrature + shell errors quantified; the grid→continuum station-density argument is the named formality C-6);
- **certified tier:** c_Λ = M_floor = 0.9001 conditional on the single named premise H-B3 (regional third-derivative constant; rung-stability machine-checked; full-grid transcripts running at W8).

Also required of the Λ-side, independently of the constant: (i) the r³ scaling itself — the window factor carries ℓ = r³/6 and the remaining factors have *exact positive limits* (C026), so the rate is structural, not measured; (ii) the factor localization (C024 estimand) and the typing/window conventions exactly as frozen; (iii) the Bonferroni pair term o(r³) (§3 row 9). Finite-r corrections of C*(r) to C*_∞ are O(r) (measured slope ≈ 1.4·r) — so even at the Λ side the honest relative remainder of the assembly is O(r), not O(r³); see §5.5.

---

## 5. The WP question, in full

### 5.1 Where WP sits and why no rate is needed

WP enters only inside AO ≥ AO⁰ − [R2] − [WP] − … The final claim 1 − q ≥ c·r³ is equivalent to: Λ-side supplies c_Λ·r³, and AO(r) ≥ a₀ > 0 on (0, r₀]. Since AO⁰ = 0.9666 (measured tier) or ≥ 0.0896·P₀ (theorem tier) is a *constant*, and R2 = O(r³), above-b = superexp-small, ridge = 0 at rungs, exit = measured 0, **the entire small-r burden on WP is WP(r) → 0**. No power, no relation to r³, no relation to ℓ.

### 5.2 The three sufficiency levels and what each buys

| level | WP statement | buys | costs |
|---|---|---|---|
| L1 | WP(r) → 0 (unquantified, e.g. via a dominated-convergence or compactness argument) | the **liminf theorem**: liminf (1−q)/r³ ≥ AO⁰·c_Λ^∞ | no explicit r₀, no explicit c |
| L2 | WP(r) ≤ C·r^α on (0, r₀], C, α > 0 explicit, **valid** | the **explicit theorem** 1 − q ≥ c·r³ ∀ r ≤ r₀ with c = c_Λ·(AO⁰ − C r₀^α − …) displayed | needs a certified uniform bound; the exponent is immaterial, the constant and coverage are not |
| L3 | WP(r) ≤ C·r³ | L2 **plus** the ornament "(1 − O(r³))" and a bounded per-r³ coefficient ledger (the 0.3889-style display) | strongest bound; **not needed for the target theorem** |

A fourth, weaker-looking level also works: L0, a **fixed reliability loss** WP(r) ≤ κ for all small r with κ < AO⁰ − a₀ — the assembly tolerates even a non-vanishing WP if it is small (budget arithmetic below). This is the sense in which the architecture is maximally forgiving.

### 5.3 Budget arithmetic (certified in `w10_sanity.py`, checks S4–S5b)

At the measured far tier (AO⁰ = 0.9666), targeting a₀ = 0.90 at r₀ = 0.05: the WP budget is 0.0666 − 0.1759·r₀³ = **0.06658**. Against the scaling-evidence envelope 5.5e-3·r^1.6 (used here only as the observed *size*): WP(0.05) ≈ 4.56e-5, i.e. **headroom ×1461**. Even the ×175 pointwise inflation exhibited at the witness point by the sqrt restoration (2.30247e-2 vs 1.31310e-4) applied uniformly to the envelope would give ≈ 8.0e-3 — still ×8 inside budget; and any smaller r₀ recovers margin polynomially. The falsified form WP = 0.213·r³ and the r^1.6 envelope differ in the final constant at r₀ = 0.05 by **|Δc| = 1.79e-5** (S5b). The limit constant AO⁰·c_Λ = 0.9144036 (measured c_Λ) is insensitive to the WP form altogether (S4: c(r₀) = 0.914340 … 0.914403 across r₀ = 0.05 … 0.0025).

### 5.4 The one place WP's modulus is genuinely load-bearing: the theorem far tier

At theorem grade, C028 gives far_single ≤ 1 − 0.089569·P₀ with P₀ > 0 **not explicitly floored** (G-D4b honest null: P₀ < 3.2e-4 at 95%, plausibly 1e-4–1e-6). Then AO⁰ ≥ 0.089569·P₀ is tiny, and *every* o(1) channel must be driven below it by shrinking r₀. With the observed envelope constant: WP(r₀) ≤ 0.089569·P₀/2 requires

    r₀ ≤ ( 0.089569·P₀ / (2·5.5e-3) )^{1/1.6}  =  0.0117 (P₀ = 1e-4),   6.6e-4 (P₀ = 1e-6)   [check S4].

So for the pure-theorem composition the **explicit modulus** (not the exponent) is load-bearing: it sets r₀ against P₀. For the ∃-form ("there exist c, r₀ > 0") even this is absorbed: P₀ > 0 by the support theorem and WP(r) → 0 suffice jointly.

### 5.5 What sub-cubic WP does to the remainder display

If WP(r) = Θ(r^α), α < 3 (evidence: truth slope 1.453, S2), then honestly

    1 − q ≥ c_Λ · AO⁰ · r³ · (1 − O(r^{α})) · (1 − O(r))  [Λ-side finite-r correction is O(r), §4]

— the "(1 − O(r³))" of manuscript (3.12) is **not** an honest display at WP-truth grade; the binding relative corrections are O(r) (Λ) and O(r^{α}) (WP). The target theorem is unaffected; the ornament is. (The upper chain's (1 + O(r³)) is outside scope; matching remainders two-sided would need WP = O(r³), but no statement of this campaign requires that match.)

### 5.6 What W2/W3/W4 must therefore deliver (minimal lemma, precisely)

> **Lemma WP-min (sufficient).** There exist explicit C, α, r₀ > 0 and a fail-closed certificate such that for all r ∈ (0, r₀]: E⁰_r[N_ws(B₃ ∖ collars)] ≤ C·r^α, with the bound **valid** (a genuine upper bound at every point of the integration domain — the CS chain with the square root present, or a sharper correlation-aware bound), and **covering the interval** (either a continuous-modulus argument, a monotone-majorant, or rung certificates plus an inter-rung stability inequality). Any α > 0 and any C with C·r₀^α ≲ 0.06 (measured tier) resp. ≲ 0.04·P₀ (theorem tier) closes the channel.

Not required: α = 3; α ≥ any particular value; smallness of C beyond the budget; any relation of WP to ℓ.

### 5.7 The invalid-ub interaction (risk register for the assembly)

The sqrt restoration makes the *generic* CS bound larger (at the witness: ×175). Two sub-risks: (i) the valid CS exponent could be worse than 1.6 — the window factor enters as √(P_window) ~ √(ℓ/s_eff) instead of (ℓ/s_eff), which with s_eff ~ r²-class effective-jet variance could pull the exponent toward ~0.8 — **still sufficient** (any α > 0); (ii) if the valid generic bound fails to vanish at all (α ≤ 0) on some sub-region, the CS route is dead there and a correlation-aware bound (the type∩window joint structure — the manuscript's own observation that the window condition forces det H > 0 on the −y side) is required; the assembly's requirement is unchanged. This is W2/W3/W4's adjudication; W10's output is that **the assembly survives any vanishing valid answer and needs nothing sharper**.

---

## 6. The honest composition forms and their exact hypothesis sets

Shared base hypotheses (all forms): **H-F** (field: exact normalized periodized BF on T²₂₄, b = 6/5, a.s. Morse — Established/Bulinskaya); **H-MS** (a.s. Morse–Smale — SARD-G conclusion; currently specialist-reviewed-conditional); **H-Palm** (typed pair-Palm disintegration, Z_r ≍ r², exact RN weight — certified); **H-Bonf** (marked pair term o(r³); program-grade O(r⁶) via (ND′)).

### Form A — unconditional explicit theorem (the strongest honest target)

*Statement.* ∃ explicit c, r₀ > 0: 1 − q(r, 6/5) ≥ c·r³ for all r ∈ (0, r₀], with the constant ledger displayed.

*Hypothesis set beyond base:* **A-Λ** (§4 certified tier: c_Λ ≥ 0.9001 uniform on (0, r₀] — needs H-B3 discharged and the C-6 station-density formality); **A-far** (selection lemma DER-009/009b + C028 barrier + R3′ transfer — gives far_single ≤ 1 − 0.089569·P₀; for an *explicit* c additionally an explicit P₀ floor (OBL-P0-FLOOR), else c is explicit up to the P₀ factor); **A-R2** (row 1 at derived-structure grade + the C-4 KR/collar formalities); **A-WP** (Lemma WP-min, §5.6); **A-aboveb** (row 3 closed + interval coverage); **A-ridge** (row 4: r-uniform enclosure of the frozen-rung census); **A-exit** (γ-LOC terminal clause — premise P-NMZ-γ (C-1), or any other explicit exit modulus); plus H-MS upgraded from specialist-reviewed-conditional to proved for full unconditionality. *Constant delivered:* c = 0.089569·P₀·c_Λ·(1 − o(1))-class — positive, small, honest.

### Form B — existential theorem ("∃ c, r₀ > 0", no explicit constants)

*Statement.* There exist c, r₀ > 0 with 1 − q ≥ c·r³ on (0, r₀].

*Hypothesis set beyond base:* A-Λ weakened to **c_Λ > 0 exists** (positive exact limits C026 + any neighborhood-uniformity formality); A-far weakened to **P₀ > 0** (support theorem — free); WP/R2/above-b/ridge/exit each **o(1) with no quantification**; H-MS as in A. This is the minimal *theorem-shaped* statement: every channel needs only to vanish, every constant only to be positive. The WP sub-cubic scaling is wholly invisible to Form B.

### Form C — liminf statement (the most robust to the WP defect)

*Statement.* liminf_{r↓0} (1 − q(r,6/5))/r³ ≥ AO⁰·c_Λ^∞, with the right side at the tier of its inputs (0.9144036 measured-tier; 0.8700… at c_Λ-floor 0.9001 — S4).

*Hypothesis set beyond base:* c_Λ^∞ > 0 as a **limit** (derived-on-grid 0.9091 + convergence; no interval uniformity); every AO channel **→ 0 along r → 0** (no modulus, no coverage); far at its fixed bound; H-Bonf as o(r³). No rung-table, envelope, or sub-0.0025 tail (C-3) is needed at all: the WP channel contributes only its limit 0. **This is the form whose hypothesis set is smallest while still displaying the 0.9144-class coefficient** (at measured/floor tier for c_Λ and far).

### Form D — mixed-tier / certified-rung statements (what is deliverable today)

*D1 (measured-grade bound, C031 §5 lineage):* 1 − q ≥ 0.87·r³ at measured grade — every term carrying a derived-structure ceiling or explicit-construction floor, measured inputs labeled (far union-bound 0.081 tier; superseded in sharpness by the selection-lemma tier).
*D2 (per-rung certified statement):* at each certified rung r ∈ {0.05, …, 0.0025}, 1 − q(r) ≥ c(r)·r³ with c(r) displayed from the rung's ledger — a finite set of point statements, honestly not r-uniform (interval gaps declared; C-3 names the sub-0.0025 tail).
*D3 (conditional theorem):* Form A with its named conditional set carried explicitly: C-1 (P-NMZ-γ), C-3 (WP sub-0.0025 tail — *only needed for interval coverage below the last rung, not for the limit*), C-4 (KR validity + collar exclusion), C-6 (Λ station density), C-7 (H-B3), C-5 (measured-grade inputs, labeled, never theorem constants).

**Tier discipline (binding):** the proved constant (Form A, when its hypotheses land), the limiting coefficient (Form C), the measured anchor (0.87·r³ D1), and the direct numerical estimate (C* ≈ 0.96 ± 0.06, C021) are four distinct objects; the proved floor exceeding the measured floor (0.9144 > 0.87, selection lemma vs union bound) is consistent and must remain un-conflated (K3 controlling fact 3; manuscript §3.11 firewall).

---

## 7. Answers to the mandate, compressed

1. **Event inclusion:** §2. Failure modes F0–F3 are exhaustive (trichotomy); the lower bound is constructive via the arch qualifying event Q(y) ⊆ F1 ⊆ {D(M) ≠ S}; the factorization is [Λ first moment] × [AO reliability] − [Bonferroni pair]; every estimand and the η_r channel are in §1 (assembly incurs no η_r loss; any adjacent-form restatement must carry η_r = P{D=S, A=0} explicitly; the upper-side identity I_r q_r^adj = J_r(p_r − η_r) is where η_r is load-bearing, out of scope but firewall-recorded).
2. **Per-channel minima:** §3 table. Only far needs a fixed constant; only Λ supplies a rate; all other channels need only explicit vanishing moduli with interval coverage. **WP = O(r³) is not needed** — §5: L1 (o(1)) buys the liminf form, L2 (C·r^α, any α > 0, valid + covering) buys the explicit theorem, L3 (O(r³)) buys only the remainder ornament; L0 (fixed small κ) would even suffice. Observed α ≈ 1.45–1.6 (truth/ub-table slopes, S2) costs |Δc| ≤ 1.8e-5 at r₀ = 0.05 and nothing in the limit (S5b, S4).
3. **Λ-side:** a positive r-uniform lower constant; tiers 0.946 (measured) / 0.9091 (derived-on-grid) / 0.9001 (certified floor, H-B3-conditional); the value only scales c (§4).
4. **Composition forms:** §6, Forms A/B/C/D with exact hypothesis sets.
5. **Certificates:** read-only respected; the only computation is the fail-closed arithmetic sanity `w10_sanity.py` (exit 0 normal and `-O`, transcripts byte-identical; checks S1–S7 as labeled above). No new numerics beyond transcription checks of printed source constants.

## 8. Falsifiers for this report

- A demonstration that WP enters the composition normalized by r³ or ℓ (would revive the O(r³) requirement; the C024/C031 ledgers say otherwise — absolute additive).
- A valid WP bound that fails to vanish (α ≤ 0) *and* exceeds the §5.3 budget at every r₀ — kills Forms A/D3 at the measured tier; Form C survives unless WP ↛ 0.
- A refutation of the inclusion Q(y) ⊆ F1 (e.g. a Morse–Smale realization where the arch configuration leaves D(M) = S) — kills the constructive sub-event; would force the non-constructive F1∪F2∪F3 estimation route.
- A refutation of the selection lemma's premise or of R3′ (reinstates the 2.43 multiplicity and/or the TV subtraction — then the theorem-tier far term needs c > 0.59, beyond the C028 barrier class: the measured tier survives, the theorem tier's AO⁰ collapses).
- SARD-G's conclusion failing (R0 > 0): F0 is no longer null and the merge-tree estimand itself needs repair.

---
*End of W10 report. Hash block below.*
