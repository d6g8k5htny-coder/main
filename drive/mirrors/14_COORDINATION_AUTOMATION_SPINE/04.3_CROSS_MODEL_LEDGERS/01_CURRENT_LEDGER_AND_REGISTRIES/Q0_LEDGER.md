# Q0 LEDGER — CONSOLIDATED CORRECTION, KILL, SUPERSESSION, AND PRE-REGISTRATION RECORD

**Discipline:** append-only. Nothing here is ever edited; corrections to corrections are appended.
**Scope:** every correction ledger, adjudication, kill, supersession, reconciliation, discharge, and cycle pre-registration freeze from the 488-file archive, verbatim, in chronological order — followed by the consolidation-event entries (E-CONS-1..6) and the duplicate/twin tables.
**Companions:** Q0_MASTER.md (frozen canon), q0_machine.json (registries, contracts, data, 488-file manifest), q0_verify.py (executable verification layer).

## PART A — CORRECTION AND ADJUDICATION SPINE (verbatim, chronological)


---

<!-- BEGIN SOURCE: C024_Corrections_and_Freeze.md | sha256:0df10fa670a3 | status:ledger-record -->

# C024 — Corrections of the reviewer's own record + freeze: independent estimator verification and factor localization of the r³ law
**Date:** 2026-07-10 · **Author:** the in-loop reviewer (Claude), correcting its own prior-turn statements per supersede-never-overwrite.

## Part 0 — Corrections (filed before any computation)

**E1 — SUPERSEDED: my C023-turn summary of the upper side.** I stated Lemma MODZONE gave "a rigorous O(r³) ceiling on the moderate zone … modulo (ND′)" and concluded "upper rate: proven modulo (ND′)." Wrong on both counts against the C023 record: (i) C023 proved the exact moderate-zone count DIVERGES (γ = 1.882, four rungs); the per-rung masses 2.07/7.31·r³ grow in r³ units; the r-uniform upper side is OPEN (OBL-BETA-RELEVANCE). (ii) The (ND′) dependency was my own import, not C023's. Corrected status: positivity proven (C022); per-rung envelopes rigorous at the two anchors; BOTH r-uniform directions of Θ(r³) open — upper via terminal selectivity, lower via the rate of ∫Λ·AO.

**E2 — VOID: my "Fix it"-turn evidence.** The scaling probe (ratio 6.05 vs 4.0) used the slice-at-b intensity — the estimand retired by C020 G-U2 (the needle). The "0.3σ/0.0σ centered" numbers used raw-f moments, not the gradient-conditioned (μ_t, s_t). Both void. The centering conclusion stands only via C022's C2 certificate (P_win = 1.000000, correct object). My "the anisotropic blow-up is not clean" is UNSUPPORTED-BY-EVIDENCE (undetermined), and is hereby withdrawn as a finding.

## Part 1 — Freeze: independent Λ-exact reimplementation (verification instrument)

New implementation from scratch: mp-Hermite analytic-BF covariances (dps 40), mp Schur conditioning, float64+eigh-clip MC for Hessian factors. No code shared with c019/c020/c021 instruments. Estimand definitions copied from Lemma UB0 §5 verbatim: Λ(y) = φ₂(∇f(y)=0 | 𝒥₆, with mean) · [Φ((b−μ_t)/s_t) − Φ((b−ℓ−μ_t)/s_t)] · E[W₃1_typed | 9 pins @ v*=clip(μ_t)] / E[W₂1_typed | 𝒥₆]; typing (detM>0, trM<0, detS<0, detY<0); window (b−ℓ, b); pins ((b,0,0),(b−ℓ,0,0),(v*,0,0)).

**Anchors and pass bands (recorded values of the program; deterministic unless marked MC):**
| ID | object | station (r; y) | recorded | band |
|---|---|---|---|---|
| A1 | s_t/ℓ | 0.05; (−0.063, 0.012) | 0.0676 | ±0.0015 |
| A2 | P_win | same | 1.000000 | ±1e−4 |
| A3 | s_t/ℓ | 0.025; (−0.030, 0.007) | 0.0864 | ±0.0015 |
| A4 | s_t/ℓ | 0.025; (−0.0315, 0.006) | 0.0670 | ±0.0015 |
| A5 | C1 λ_min(Σ_∇) | 0.05 y*; 0.025 y* | 1.156e−6; 8.939e−8 | rel ±2% |
| A6 | C3 λ_min(6-pin Hess 6×6) | both y* | 2.603e−10; 4.068e−12 | rel ±5% |
| A7 | C4 λ_min(9-pin Hess 9×9 @v*) | both y* | 1.948e−13; 3.095e−14 | rel ±5% |
| A8 | C5 λ_min(9-pin Gram) | both y* | 1.321e−12; 3.333e−14 | rel ±5% |
| A9 (MC) | E[W₂1_typed \| 6] | r = 0.025 | 0.00202046 ± 2e−6 | ±1% |
| A10 (MC) | Λ | 0.025; (−0.0315, 0.006) | 0.090989 | rel ±6% |

Adjudication rule: any deterministic-anchor failure ⇒ STOP, convention audit, report discrepancy as the finding; Phase 2 conclusions blocked. A9/A10 in (6%, 12%] ⇒ FAILED-AS-WRITTEN with diagnosis; > 12% ⇒ instrument-discrepancy alarm.

## Part 2 — Freeze: factor localization at the fixed rescaled station
Station ỹ* = (−0.76, 0.24) in (y − M)/r coordinates — identically the C022 r=0.05 y* and the C021 r=0.025 sanity point. Rungs r ∈ {0.05, 0.025, 0.0125} (the 0.0125 rung is NEW, labeled as such). Report per rung: φ₂, det Σ_∇∇, Mahalanobis q, (μ_t−b)/ℓ, s_t/ℓ, P_win, den = E[W₂1|6], num = E[W₃1_sad|9], num_max = E[W₃1_max|9] (the β-side object), R_H = num/den, Λ; the EXACT (Schur, no MC) 9-pin conditional mean Hessians and their dets.
**Planted gate G-D1:** den(r=0.05) = 0.00808 ± 8% — the r² law predicted from the recorded r=0.025 anchor via the mechanism |det H_M| ≈ κr·|f_ss| forced by the ∇-pin divided differences (f_tt(M) = −(r/2)f_ttt + O(r²) with f_ttt cubic-pinned). Failure kills the mechanistic account, not the data.
**Stability gate G-D2′:** per-factor log₂-slopes consistent across the two octaves to ±0.35.
All component exponents are RECORDED DISCOVERY (no committed split); indicative hypothesis H1 (recorded, not gated): R_H ~ r¹-class via the shared depletion of all three dets.

## Part 3 — Freeze: needle-sweep profiles (L2 inputs)
μ_t(y) along both axis directions through ỹ*, ±5 steps of 0.1r, three rungs; report (μ_t−b)/ℓ and the max sweep slope in ℓ-per-r units; derived ribbon width ℓ/|∇μ_t|. Recorded discovery.

## Part 4 — Deliverable
Lemma LB-RATE architecture package: the reduction, the L1/L2/L3 certificate families with computed per-rung inputs, the honest open core, and the terminal-height-law correspondence between the two frontiers (classified as a structural correspondence of estimand families, NOT an isomorphism; prior-art search pending — no network in this environment). No claim of closure.


<!-- END SOURCE: C024_Corrections_and_Freeze.md -->


---

<!-- BEGIN SOURCE: As_Corridor_Mechanism_KILL.md | sha256:78c05824875b | status:ledger-record -->

# Corridor_Mechanism_KILL — kill-registry entry: the O(1)-halo deployment of the LB0 mechanism

**Registry class: FAILED-AS-WRITTEN / SUPERSEDED-IN-SCOPE** (no rescue). Date 2026-07-09.
Scope chain: C017 (assembly) → C018 (kill evidence) → C019 (replacement architecture measured).
What is killed is the **deployment region**, not the mechanism: Lemma LB0 remains **Proved** and the
exact conditional Kac–Rice computation behind c₁ remains **Computed**; both are retained as
correct-but-misdeployed and re-enter corridor-locally.

## 1. The claim as written (C017)

C017 assembled: 1 − q ≥ c₁ · c₂ · r³ with
  c₁ = 0.113 = (1/6) ∫_{Ω_halo} λ^{pinned}(y) dA  over the opposite-halfplane halo at O(1) distances,
  c₂ = inf_{Ω_halo} (P_adj · P_older),   field-estimated c₂ ≈ 0.93,
yielding 1 − q ≥ 0.105·r³ "at field c₂", with the rigorous c₂ > 0 left as the sole obligation
(OBL-LB-ADJ). The deployment presumed the trigger mass and the separatrix connection both live in the
O(1) halo.

## 2. The kill evidence

- **C018 (A-side):** the pair-Palm ascent-adjacency A(y) ≡ 0 at every measured O(1) station (24-point
  polar grid, ρ ∈ [0.8, 4.0], all θ; trigger-weighted ⟨A⟩ = 0): the ascent connection to M does not
  exist at O(1) — c₂(Ω_halo) = 0 as a measured fact. P-L2 adjudicated branch (iii): corridor-scale
  collapse. The C017 assembled number 0.105·r³ is therefore **vacuous as deployed** (0.113 × 0 = 0).
- **C018 corollary:** the Palm-corrected trigger coefficient over the halo is c₁^{Palm} = 0.0643 ±
  0.0008 (P-L1), vs the pinned 0.113 — the pinned/Palm weighting-scope error, first caught here,
  foreshadowing OBL-UB-PALM.
- **C019 (λ-side):** the window-density factor independently annihilates the intermediate zone
  (Mahalanobis q = 416 at (0.2, 0.025); log φ ~ −10⁶..−10¹⁰ on the axis; λ ~ e⁻²⁰⁰ at inner rungs) —
  even if ascent were present, no trigger mass reaches from the corridor to O(1): a **death valley**
  between the arch (τ ≲ 2r) and the O(1) zone, with λ recovering to ~0.005 only at ρ ≳ 0.8 where A = 0.
- **Corridor scale correction:** C018's committed r^{3/2}-scale probes (P-L4/P-L5-band/P-L6-band) all
  FAILED-AS-WRITTEN; C019 measured the true support: an arch/filament at τ ≈ 0.75r–1.5r behind/over M —
  the scale is **O(r)**, not O(1) and not r^{3/2}.

## 3. What survives, and where it goes

| object | status | redeployment |
|---|---|---|
| Lemma LB0 (window-saddle + separatrices ⟹ D(M) ≠ S) | **Proved**, untouched | applied on the arch, where both separatrix conditions hold with measured probability 0.992–1.000 |
| exact conditional Kac–Rice machinery behind c₁ | **Computed**, correct | becomes the factor-1/factor-2 algebra of the lam3 disintegration (C019) |
| c₁ = 0.113 (pinned, halo) | superseded-in-scope | replaced by the Palm arch integral: (1/6)∫λ·A·O dA = **1.10 ± 0.09** (E1_extended, C019) |
| c₂ = inf(P_adj·P_older) over Ω_halo | killed (measured 0) | replaced by the measured on-arch product A·O = 0.992–1.000; the rigorous positivity obligation migrates |
| OBL-LB-ADJ (prove c₂ > 0 on Ω_halo) | superseded | → **OBL-LB-ARCH**: prove ∫_arch λ·(A·O) dA > 0 (positivity, not sharpness) — strictly easier: A = 1 on the arch is a separatrix-identity statement (the filament IS the D(M)-witness locus), reducible to LB0 + R0 + a nondegeneracy at one arch point |
| Bonferroni step (OBL-C017-2) | discharged | LB_Bonferroni.md (this delivery) |

## 4. Reconciliation with the standing record

- **C010** (field, moderate r): the G-trigger lower bound 0.18·r³ = (1/6)·p_∞·0.93 is a *gain-channel
  subset* of the arch count; 0.18 < 1.10 ✓ (a weaker bound, not a contradiction). C010's preemption-
  given-trigger 0.93 is the moderate-r shadow of the on-arch A·O ≈ 1.
- **C013** (Lemma ADJ): pair-Palm-weighted adjacency 1.0000 at all rungs — consistent with (and
  explained by) the corridor-local A ≡ 1; C013's adjacency was measured *at pair scale*, i.e., inside
  the corridor, never at O(1).
- **C008** (τ-shell law): outermost-shell dominance of the pinned annulus count — the pinned shadow of
  the same outward mass migration whose Palm completion is the arch; the C008 annulus (τ ≤ r/2) captures
  the arch's inner flank only, which is why C_annW = 0.188 under-counts the Palm mass ×~6 and C_strip
  ×~25 (→ OBL-UB-PALM).
- **C016**: the O(1)-zone R(ρ) profile (suppressed ρ < 1, peak ρ ≈ 1.5) probed 10× outside the arch;
  no overlap, no conflict; its "asymptotic far intensity = unconditional" conclusion stands for the far
  channel.
- **q(r) field law**: c_eff(0.7) ≈ 1.31, c_eff(0.4) ≈ 1.7 bracket the asymptotic 1.10 from above as
  moderate-r laws should. The arch coefficient is the first *asymptotic-slice* measurement of the sharp
  constant.

## 5. Registry lines (verbatim for the ledger)

- KILL: C017 regional deployment {c₁(halo,pinned) × c₂(halo)} — FAILED-AS-WRITTEN (c₂ measured 0;
  trigger mass absent at O(1) under Palm). Evidence: C018_Observed_Update (P-L2 branch iii, ⟨A⟩ = 0),
  C019_Observed_Update (death valley; arch).
- KILL: corridor scale r^{3/2} (C018 committed P-L4/5-band/6-band) — FAILED-AS-WRITTEN; measured scale
  O(r) (arch at 0.75r–1.5r).
- RETAIN: LB0 [Proved]; conditional-KR machinery [Computed]; C013 adjacency [Measured, in-corridor].
- SUPERSEDE: c₁ 0.113(pinned,halo) → 0.0643 (Palm, halo; C018 P-L1, no load) → arch coefficient 1.10
  (Palm, corridor-local; C019 E1_extended) — scope chain explicit, nothing overwritten.
- NEW OBLIGATIONS: OBL-LB-ARCH (positivity proof), OBL-UB-PALM (upper-bound constants under Palm),
  OBL-C019-1/2, OBL-LBB-1, OBL-R0-UNIF (see Registry_Reconciliation.md).


<!-- END SOURCE: As_Corridor_Mechanism_KILL.md -->


---

<!-- BEGIN SOURCE: C014_KILL_ThirdPoint_Channel.md | sha256:efba1d5483b6 | status:ledger-record -->

# C014-KILL: The Third-Point Max Failure Channel Vanishes as r → 0 (moderate-r contamination corrected)
**Artifact class:** kill-registry entry + correction (corrections supersede, do not overwrite). **Chain:** freeze C014p2 724b2636…8a7b; validated instrument joint_sim3.py; confirming runs at r ∈ {0.2, 0.1, 0.05}, tilts m ∈ {±1/2, ±3/2}.

## 1. What triggered it
The C014 part-2 Palm integrator returned num3 = 0 (empty numerator) at both tilts, while the validated failure fraction J was clearly nonzero. Instead of reporting Φ = 0, the empty numerator was treated as a diagnostic. Radial analysis of the partners resolved it.

## 2. Root cause (measured, not argued)
Points x_M = (−r/2, 0), x_S = (+r/2, 0) sit at separation r, which in r²-units is **1/r**. At r = 0.2 that is exactly **5 r²**. The C014 partner search validity extended to 10 r² = 2r, so **M lay inside the third-point search ball**. Measured max-partner distances: median **5.0 r² at both tilts, i.e. exactly M**. The "max third-point" was the pair's own maximum; "S-separatrix branch captured by the max-partner" was substantially "branch reaches M" = **adjacency, misclassified as failure**. The P-K2 gate passed only because both the part-1 and the corrected part-3 instruments carry the identical contamination — agreement between two instruments does not certify freedom from a shared physical mis-scoping.

## 3. Decisive small-r test (M excluded)
With the search restricted below M (validity < 8 r², partner ≥ r/4 from M):
| r | M at | max-partner fraction | location |
|---|---|---|---|
| 0.2 | 5 r² | 0.09 / 0.21 | 5.0 r² = **M** (contamination) |
| 0.1 | 10 r² | 0.013 / 0.016 | 7–8 r² (M-basin edge) |
| 0.05 | 20 r² | **0.000 / 0.000** | none |
Exhaustive whole-gap sweep at r = 0.05 (tilts ±1/2, ±3/2; radii to 16 r²): **max third-points at frac ≤ 0.0003**, the lone residual at 13 r² (M-basin outskirt). As the scales separate, the nearest non-S critical is always a **saddle or a min**. Mechanism: the natural nearby criticals of a tilted saddle are the transverse cubic's second root (a saddle) and mins; a genuine extra max requires higher-order structure and is O(r^{>0})-suppressed.

## 4. What is KILLED / SUPERSEDED
- **C014 J = 0.089 / 0.199 as a "deep-tail failure constant" → SUPERSEDED.** It is an r = 0.2 M-contamination artifact; the genuine small-r third-point failure fraction through this channel is **→ 0** (only max-partners produce branch-capture failure, and max third-points vanish).
- **"J = p_max exactly" → re-scoped.** True as an r = 0.2 identity, but p_max there was M's contamination fraction; p_max → 0 at small r, so it is not a deep-tail law.
- **C014 part-1 "deep-tail adjacency ≈ 0.23–0.28" revision → RETRACTED.** adjM was deflated by the spurious partner-vs-M competition for the same M-region events. **C013's clean pair-adjacency ≈ 0.92 on the |μ_S| ≤ r set (gated D-A1′) is REINSTATED** as the operative B2 tail-adjacency number.

## 5. What SURVIVES
- **The joint instrument (partner sweep + two-phase germ flow) is validated** — the P-K2 regression gate passes (batched J = 0.089/0.199 vs part-1 0.091/0.198, |diff| ≤ 0.002). It is a correct tool that was run at a contaminated regime; the analytic-seed partner search (part-2) was separately a bug and is retired in favor of the angular sweep.
- **The two-sided LOWER BOUND is unchanged.** Its positive constant was never from a third-point channel — it comes from the C010 field-level preemption measurement P(D≠S | G∈(0,ℓ)) = 0.93–1.00 × the inner window count, giving 1 − q ≥ ~0.18 r³. That stands untouched.
- **Positive structural content:** at small r the third point, when present, is a saddle or a min — neither disrupts the elder-rule S–M pairing (a saddle does not trap an ascending separatrix; a min is H0-superlevel-inert). So **third points do not generically break the pairing**; adjacency is robust. This *reinforces* that the genuine failures are the r-scale channels C010 measured (window-saddle ≈ 64%, W₂ ≈ 26%), not an r²-third-point disruption.

## 6. Corrected inner-law status
E_Palm[N_fail-inner] is set by the r-scale mechanism (C010), not a third-point factor. The inner window count C_innerW = 11.9 (C012) is multiplied by the field-measured preemption, not by a third-point J. The two-sided statement is exactly as before the C014 attempt: **0.18 r³ ≲ 1 − q ≤ [2(C_annW + C_strip) + C_far(L,b)] r³ (1+o(1))**; C014's contribution is a *null* refinement (no extra channel), logged.

## 7. Epistemic labels
Finding (max third-point vanishes as r → 0): **[Verified-numerically]**, robust across tilts/radii, with a clear suppression mechanism; a proof would bound the conditional intensity of a max in the r²-annulus as O(r^{k>0}). The kill of the C014 J/adjacency numbers: **[Retracted]**, root-caused. C013 reinstatement: **[Verified]** (unchanged, D-A1′-gated).


<!-- END SOURCE: C014_KILL_ThirdPoint_Channel.md -->


---

<!-- BEGIN SOURCE: C032_TC_Supersession_Package.md | sha256:8a76c5586e55 | status:ledger-record -->

# C032 — Lemma TC: the Terminal-Counting Supersession; the Mean-Rim Discovery; the AB Negative Result
**Freeze:** 1487c20c…2cae. **Provenance:** C031's G2 selection-obstacle analysis exposed that selection is unnecessary — the C025 step "terminal ≥ v* = b − ℓ/2 ≥ b − δ₀" discarded the ℓ-scale start-height information.

## Lemma TC (established at derived-structure + station-certified grade, modulo R0)
Modulo R0 (terminal of an ascent is a local max — the architecture item every AO statement already inherits): {outward terminal ≤ b} ⊆ {separatrix loop: outward branch terminates at M} ∪ {∃ local max with value ∈ [v*, b] on T²₂₄ ∖ collar(M)}. Hence, with the band width 0.49993·ℓ (m = −0.4999290, C026):

P(terminal ≤ b | 9 pins) ≤ P(sep-loop) + **2.1·(ℓ/2)** [B₅; C030 KR-MB] + **ρ_mx(1.2)·0.49993·ℓ·(L² − π·25)·E_sup** [exterior] = P(sep-loop) + **2.03·r³**

with ρ_mx(1.2) = 0.043685 DERIVED (C030 G-F1 two-route) and **E_sup = 1.0235 station-certified this cycle (G-H1 PASS):** conditional/unconditional band-max intensity ratio ∈ [0.98, 1.014] at d ∈ {5, 6, 8, 12} × 2 angles × 2 rungs (deviations = MC noise; v = 1 − O(1e-8) by d = 5, pull monotone → exact-zero underflow by d = 8); torus periodization ≤ e^{−72}, negligible. The sep-loop event is FOLDED into the flow-topology named family. The magic is the start height: the outward branch launches at mid-window, so only an ℓ/2-thin terminal band can fail — the far term is O(r³) **by counting alone**, no crossing selection, no Palm disintegration, no barrier in the load path.

## Supersession ledger (corrections supersede, never overwrite)
- **OBL-FAR-COMPOSE → CLOSED-BY-SUPERSESSION** (registered at C031 §4; both routes moot for the assembly; the Rice constant 2.43 remains as context).
- **C028 far-ascent theorem (p̄(1/5) ≤ 1 − c) and C031 R3′ (conditional-barrier transfer, 36× margin) → STANDALONE results:** correct, gate-passed, no longer load-bearing for AO.
- **Lemma FD → repositioned:** exterior-transfer evidence (now superseded in that role by the direct G-H1 intensity ratios) + the quantitative channel.
- **OBL-P0-FLOOR → moot for the assembly** (standalone-optional).
- **Reconciliation:** the measured p̄(0.05) = 0.0334 and the Rice-composed 0.081 were honest bounds for the δ₀-framed event; the framing was lossy because δ₀ was fixed while the true start height is b − ℓ/2. No prior number is wrong; the architecture is superseded by a sharper inclusion.

## The mean-rim discovery (vacuity check FAILED-AS-WRITTEN → finding)
The freeze's sup-of-mean check (m₉ ≤ b, max at M) FAILED: **m₉ exceeds b at every station outside the pin cluster** — 1.2007 at d = 0.2, rising to a rim of **1.40–1.57 at d = 1.3** (anisotropic: higher toward the arch side, angle 2.35), decaying to 0.0003 by d = 5 (C031 table). The check's premise (my pre-freeze narrative) was wrong; the instrument was right. The 9-pin cluster sits on the shoulder of a mean bump — the MEAN side of the same jet-cluster physics whose VARIANCE side is the C030 horizon. Consequences: (i) the above-b-saddle O(1) count is explained (the mean itself is above b in the zone); (ii) AO ≈ 1 acquires its mechanism (the outward ascent climbs mean-uphill; qualification is the default; capture 1.0000 explained at mean-field level); (iii) the sep-loop event requires descending the rim back to exactly b at M — strongly disfavored (evidence for the named item, not a closure); (iv) band-maxima in the rim zone require downward fluctuations against tiny v — the C030 transition-zone numbers are conservative.

## The AB negative result (G-H2; pre-registered P1 CONFIRMED, band hit)
Zone-integrated crude count E[N_sad(f > b, 0.15 < |x| < 1.5) | 9] = **0.0325 (r = 0.05) / 0.0260 (r = 0.025), rung ratio 1.25 ∈ [0.5, 2]** — O(1) in r, not ≈ 8. Station profile: exact-zero underflow for d ≤ 0.35 (rigidity: mean 1.2007-pinned above the threshold's reach... the field there is deterministically ABOVE b — no saddle-below), 1e-12 at d = 0.7, growing to 1e-2 at d = 1.3 where the rim's fluctuations live. **The counting route for the above-b-saddle diversion channel is DEAD** (my pre-freeze mechanism story — gap ∝ r·d² — is superseded by the rim mechanism); the channel remains in the flow-topology family with this quantitative context: any closure must use the pass-connectivity/flow structure, not saddle counting.

## The new AO assembly and LB-RATE
**AO ≥ 1 − WP(0.213·r³) − TC(2.03·r³) − [flow-topology named family] − [KR validity + collar exclusion] , modulo R0**
= **1 − 2.24·r³ − named**. At r = 0.025: AO ≥ 0.999965 − named; at r = 0.05: ≥ 0.999720 − named (G-H3 PASS: ≪ the superseded 0.081; consistent with the direct C021 AO ≈ 1).
**LB-RATE:** 1 − q ≥ C*(r)·r³·(1 − 2.24·r³)·(1 − O(r³)) ≈ **0.946·r³ at r = 0.025 (measured grade)** — the assembly now MATCHES the direct two-sided measurement C* ≈ 0.96 ± 0.06 instead of undershooting at 0.87; limit form 0.9091·r³·(1 − o(1)) with the named set.

## The revised named set (11 items; net −2, one family consolidated)
1. R0 / MS–Sard covering (architecture; carries TC's "terminal is a max"). 2. γ-LOC tube-local (architecture). 3. (ND′) chain note (upper side). 4. KR validity + collar exclusion (WP, KR-MB, TC-exterior). 5. **Flow-topology in the rigid zone** (consolidated family: ridge split + separatrix loop + above-b-saddle diversion; quantitative context: crude count ~0.03 r-free; mean-rim disfavors all three). 6. OBL-BETA-RELEVANCE (upper side, out of scope). 7. 9-pin certificate tightening. 8. Collar residual |ỹ| < 0.15. 9. Station-density/on-grid class (C027 grid; R3′ slab; G-H1/G-H2 stations). 10. Λ-side grid→continuum formality. 11. Prior-art searches (network-gated; now ALSO covering the mean-rim structure — analytic-kernel interpolant theory is a known field; NO novelty language used).
Removed: OBL-FAR-COMPOSE (superseded), OBL-P0-FLOOR (moot-for-assembly, standalone-optional).

## Gate ledger
G-H1 PASS (10 stations, both rungs, ratio band + monotonicity). G-H2: P1 CONFIRMED at band center (ratio 1.25); **sup-of-mean vacuity check FAILED-AS-WRITTEN → the mean-rim finding registered** (premise wrong, instrument right); tail quadrature 7 nodes × 6k draws per station, 20 stations. G-H3 PASS (both consistency clauses). Scope: BF, b = 6/5, L = 24, arch station; two angles per distance (angular sweep optional).


<!-- END SOURCE: C032_TC_Supersession_Package.md -->


---

<!-- BEGIN SOURCE: v2patch_Correction_Addendum.md | sha256:9736f35dd0c5 | status:ledger-record -->

# Correction Addendum to window_lemma_v2_patch (C009) — supersedes in part, never overwrites
1. **Section I (Lemma I): RESOLVED**, β = 5 at derived+verified grade; see Lemma_I_Package.md. The candidate β = 4 was an underestimate (M-side determinant drag adds one power). The sympy verification plan proposed in the patch was executed and extended far beyond (certified exact table, gates, two-weighting measurement).
2. **Section I (Localization): SUPERSEDED in its exp(−c/r²) form.** Conditional-at-fixed-w only; the Palm-averaged shell law is polynomial r^{2γ+1} (C008). All downstream assembly conclusions survive; the exp claim is retired to the kill-log class of R3.
3. **Section A′: SCOPE-CORRECTED.** Valid off the vertical strips through M and S; on the strips the count is polynomial (measured tail 2.95 on [r/2, 4r], geometric decay beyond, far-floor crossover at absolute O(1)). The κ-pinned t-floor makes the off-strip suppression degeneracy-proof; the strips expose the μ-flat direction and are now an explicit measured component of the peak zone.
4. **Assembly line replaced:** "C r^β (inner)" → the seven-zone table of Lemma_I_Package.md §3; conclusion 1 − q = O(r³) unchanged, now with every polynomial component verified or measured.


<!-- END SOURCE: v2patch_Correction_Addendum.md -->


---

<!-- BEGIN SOURCE: v2patch_Correction_Addendumaw.md | sha256:c5a246a96726 | status:ledger-record -->

# Correction Addendum to window_lemma_v2_patch (C009) — supersedes in part, never overwrites
1. **Section I (Lemma I): RESOLVED**, β = 5 at derived+verified grade; see Lemma_I_Package.md. The candidate β = 4 was an underestimate (M-side determinant drag adds one power). The sympy verification plan proposed in the patch was executed and extended far beyond (certified exact table, gates, two-weighting measurement).
2. **Section I (Localization): SUPERSEDED in its exp(−c/r²) form.** Conditional-at-fixed-w only; the Palm-averaged shell law is polynomial r^{2γ+1} (C008). All downstream assembly conclusions survive; the exp claim is retired to the kill-log class of R3.
3. **Section A′: SCOPE-CORRECTED.** Valid off the vertical strips through M and S; on the strips the count is polynomial (measured tail 2.95 on [r/2, 4r], geometric decay beyond, far-floor crossover at absolute O(1)). The κ-pinned t-floor makes the off-strip suppression degeneracy-proof; the strips expose the μ-flat direction and are now an explicit measured component of the peak zone.
4. **Assembly line replaced:** "C r^β (inner)" → the seven-zone table of Lemma_I_Package.md §3; conclusion 1 − q = O(r³) unchanged, now with every polynomial component verified or measured.

## Entry 5 (C011)
**Lemma B1 KILLED** (band event is typical: IVT on ∂B_{3r} with f(M) = b inside and transverse dips below b−ℓ; the exp(−c/r²) claim is off by everything — kill registry K-B1). **Corollary B2 replaced by loop absorption:** A_loop = {M₂ ∈ C at (b−ℓ)+} ⊆ {a window saddle exists}, since the max-min pass between M and M₂ lies in (b−ℓ, b); the loop term vanishes from the assembly, which simplifies to 1 − q ≤ E[N_w] + P(A₀) = O(r³). Field-level confirmation of the loop channel: C010 records, died-below-S ∧ f(M₂)-∉-window, rate 0.0014 at r = 0.7. See Obligations_Registry_Discharge.md §8.


<!-- END SOURCE: v2patch_Correction_Addendumaw.md -->


---

<!-- BEGIN SOURCE: Obligations_Registry_Discharge.md | sha256:72adf3eb07b9 | status:ledger-record -->

# The Obligation Registry and Formal Discharges (C011)
**Artifact class:** proof-grade consolidation. **Chain:** C011 snapshot a7b84d5a…4f91; companion ledger hashed. This document enumerates every open formal obligation across the Theorem A chain (FlatSaddle U1–U5, v2-patch R0/H/G/P/B1–B2, C008 general-γ transfers), discharges what is dischargeable at proof grade with the certified constants as named inputs, registers one kill with its replacement, and updates the master dependency graph.

## 0. Registry summary

| obligation | source | status after C011 |
|---|---|---|
| (U1) conditional Kac–Rice | FlatSaddle §5 | **DISCHARGED** (§2) |
| (U2) L^{2+ε} remainders | FlatSaddle §5 | **DISCHARGED** (§3) |
| (U3) tail-density Gram | FlatSaddle §5 | DISCHARGED C009: det = (1/540)r¹⁸; ξ-part via Lemma H |
| (U4) Palm-base uniformity | FlatSaddle §5 | **DISCHARGED** (§4) |
| (U5) adjacency ≤ 1 | FlatSaddle §5 | closed by definition (upper bound) |
| R0/R2 separatrix + merge | v2 patch | **DISCHARGED** (§1) |
| Lemma G far-zone uniformity | v2 patch | **DISCHARGED** (§5) |
| Lemma H | v2 patch | **PROVED for BF/positive spectral density** (§6); general (H2′) cited-with-conditions |
| Lemma P finite write-out | v2 patch QED-plan | OPEN, finite; precise sympy spec registered (§7) |
| Lemma B1 (band) | v2 patch | **KILLED** (§8); replaced by B1′ |
| Corollary B2 (loop) | v2 patch | **REPLACED**: A_loop absorbed into the window count (§8) |
| general-γ (U1)–(U4) | C008 | transfer by §§2–4 verbatim with τ = r^γ (all inputs certified bi-graded) |

## 1. R0/R2: separatrices and merge identification — discharged

(a) *A.s. Morse.* Under Lemma H the joint law of (∇f, ∇²f)(y) is nondegenerate for every y; by the Bulinskaya-type argument (Azaïs–Wschebor), a.s. no point has ∇f = 0 with det ∇²f = 0, and f is a.s. C² Morse on the torus. Critical values are a.s. pairwise distinct: for each pair type, Kac–Rice over pairs (y₁, y₂) with the codimension-one constraint f(y₁) = f(y₂) has vanishing expected count under Lemma-H nondegeneracy of (∇f(y₁), ∇f(y₂), f(y₁) − f(y₂)).
(b) *Separatrices.* At a nondegenerate saddle, the ∇f-flow's unstable manifold is one-dimensional with two branches (stable-manifold theorem); on a compact surface every forward orbit's ω-limit is a critical point (Łojasiewicz not needed — Morse suffices), of strictly higher value. A.s. no saddle-to-saddle connection: fix the countable family of saddle pairs; a connection is destroyed by an arbitrarily small Cameron–Martin shift in a direction altering the value gap along the connecting orbit; conditional on the orthogonal complement, the CM coordinate's law is absolutely continuous and the connection holds on a Lebesgue-null set of it, so P = 0. Hence both branches a.s. terminate at maxima: M₂ is well-defined.
(c) *R2.* Passing the single nondegenerate critical value f(S) = b−ℓ attaches a 1-cell at S to the sublevel-complement; the attaching endpoints are the two ascending-separatrix germs, so the superlevel components joined (or self-joined) at S are exactly those of the two terminal maxima. Merge identification proved.

## 2. (U1): conditional Kac–Rice — discharged

The pair conditioning is on six linear functionals L(f) = a; the conditional field is m_a + h with h Gaussian, covariance K minus a rank-6 correction. For counts over any region R excluding shrinking balls B_{ρ}(M) ∪ B_{ρ}(S): (i) paths a.s. C² (analytic kernel); (ii) the conditional law of ∇f(y) is nondegenerate for y ∉ {M, S} — degeneracy would make ∇f(y) an a.s. linear function of six functionals, excluded by Lemma H — with the degeneration rate as y → x_S given *exactly* by the certified gradient-block minors: Var_t Var_s − Cts² = τ⁴r²z_s⁴(1 + O(r + τ/r)) > 0 off the axis, with axis-collision integrability established by the C007-B cutoff-insensitivity record; (iii) E|det ∇²f| moments finite (Gaussian). Azaïs–Wschebor's conditional Kac–Rice then applies on every such R, and the ρ ↓ Cr² limit is dominated by the (U2) moment bounds. The three-point/two-scale application is precisely the C007/C008 assembly; its validity input is the C009 Gram (1/540)r¹⁸.

## 3. (U2): L^{2+ε} uniformity — discharged

Every O_P(·) remainder in (F1)–(F4) and in the certified mean/covariance expansions is an explicit polynomial (degree ≤ 4) in a fixed Gaussian vector of derivatives of order ≤ 4 at ≤ 3 points, with covariance entries given by the certified series, uniformly bounded on the Palm compact; λ₈ < ∞ (H1+) covers the fourth-order entries. By Gaussian hypercontractivity, ‖P(X)‖_{2+ε} ≤ C(deg, ε)‖P(X)‖₂, and the ‖·‖₂ scalings are the certified coefficients termwise. Uniformity over the tail set follows from the bounded conditional densities of (U3). ∎

## 4. (U4): Palm-base uniformity — discharged

Conditional means are affine in a = (b, 0, b−ℓ, 0, 0, 0); conditional covariances are a-independent. Every ledger constant is an integral of continuous-in-(b, κ) integrands against Gaussian envelopes independent of (b, κ) on compacts; dominated convergence gives continuity and uniform integrability. ∎

## 5. Lemma G: far-zone window count, uniform — discharged

E[N_w(far)] = ∫_{|y−x₀|≥δ} E[|det∇²f(y)| 1_win | ∇f(y)=0, pair] p_{∇f(y)|pair}(0) dy. Two inputs: (a) v(δ) := inf_{far} Var(f(y) | ∇f(y), pair functionals) > 0 — the variance is continuous in y, strictly positive pointwise (an eight-functional degeneracy is excluded by Lemma H), and the domain is compact (torus), so the inf is attained and positive; hence the conditional window probability is ≤ ℓ/√(2πv(δ)) uniformly. (b) The remaining integrand integrates to at most the conditional total-critical-point intensity, finite and continuous by the same nondegeneracy. Hence E[N_w(far)] ≤ C(δ, L)·ℓ, uniform on b-compacts. ∎

## 6. Lemma H for the testbed class — proved

For BF the spectral density is everywhere positive. A linear relation Σ c_{j,α} ∂^α f(x_j) = 0 a.s. forces Σ c_{j,α}(ik)^α e^{ik·x_j} = 0 for a.e. k; the left side is an entire function of k (finite sum of polynomials times distinct exponentials), which vanishes identically only if all c = 0 (linear independence of {k^α e^{ik·x_j}} for distinct x_j). Hence joint nondegeneracy of any finite derivative family at distinct points. For general (H2′), the same holds whenever the spectral measure is not supported on a real-analytic variety annihilating such a family; adopted as the stated form of (H2′). ∎

## 7. Lemma P: the one remaining finite computation

Open, with the exact spec now registered: compute (sympy, exact) the 6×6 covariance of the on-axis jet coefficients (c_{t³}, c_{t⁴}, c_{t²s}, c_{t³s}, c_{ts²}·grade, …) conditional on the four on-axis pin constraints {h, h_t at ±r/2} = 0, verify the r → 0 limit is positive-definite (the two-point analogue of the C009 (1/540) computation), yielding the pinned variance bounds sd(h) ≤ Cρ³, sd(h_t) ≤ Cρ²·ρ on the strip used by the off-strip A′ argument. Registered as the C012 candidate; the assembly's polynomial terms do not depend on it (only the off-strip exponential grade does).

## 8. The B1 kill and what replaces it

**Kill (K-B1, IVT argument).** B1 claimed P(∃ y ∈ ∂B_{3r}: f(y) ∈ (b−ℓ, b]) ≤ C exp(−c/r²). False: f(M) = b with M's window component of O(1) diameter forces f > b−ℓ somewhere on ∂B_{3r} with probability ~ 1, the transverse arcs dip below b−ℓ, and continuity on the circle forces window values at the crossovers. The band event is *typical*. Root cause class: a containment bound whose complementary event was analyzed at fixed generic geometry (the μ ≍ 1 transverse suppression) — the same μ-flat-direction blindness as the R3/Localization kills.

**Replacement (B1′, and B2 absorbed).** By R2, A_loop = {the S-passage is a self-merge} = {M₂ ∈ C at (b−ℓ)+}. If so, M and M₂ are connected in {f > b−ℓ}, and the max-min pass between them is attained (a.s.) at a saddle with value in (b−ℓ, b) — strictly below b because at level b−ε the components of M and M₂ are disjoint germs of their maxima. Hence **A_loop ⊆ {a window saddle exists} ⊆ {N_w ≥ 1}**: the loop event is *absorbed into the window count already bounded at O(r³)*; no separate band/loop budget exists in the assembly. Field-level confirmation: the C010 died-below-S ∧ f(M₂)-∉-window subpopulation (the predicted loop signature, partner elder in all members) is observed at rate 0.0014 at r = 0.7 — one event, exactly the Θ(ℓ)-class rarity expected.

**Corrected assembly.** 1 − q ≤ P(N_w ≥ 1) + P(A₀) ≤ E[N_w] + P(A₀) with E[N_w] = O(r³) by the seven-zone table (C009) and A₀ (a second critical point of value ≥ b inside C) a window-free O(ℓ)-type event bounded by the same counting at the level b. The exponential-zone lemmas (off-strip A′, Lemma P) now affect only interior refinements, not the rate.

## 9. Master dependency graph after C011

PROVED: R0/R2; (U1)–(U5) [with (U3)'s exact constant]; Lemma G; Lemma H (testbed class); B1′/loop absorption. DERIVED + VERIFIED: inner r⁵ law; γ-family shell law; annulus r³ boundary-dominance; the r³ assembly; Lemma LB preemption. MEASURED: window constants (0.51 core, 0.83–0.85 boundary, 0.24 strip ratio, acceptance profiles); failure taxonomy; p∞(b) ≈ 1.19 first estimate; loop-channel existence. OPEN: Lemma P finite computation (spec §7); the c·ℓ deep-scaling verification of Lemma LB (importance-sampled instrument); the exact-constant/two-sided inner law (B2 adjacency, with its three target constants); saddle-channel constant c_saddle.


<!-- END SOURCE: Obligations_Registry_Discharge.md -->


---

<!-- BEGIN SOURCE: LemmaP_Discharge.md | sha256:082940072912 | status:ledger-record -->

# Lemma P, Discharged Exactly: Cubic Pinning, the Slab Scope, and the Octave Law (C012)
**Artifact class:** final formal-obligation discharge. **Chain:** C012 snapshot 3473ca55…426c; companion ledger hashed. With this document the obligation registry of C011 §0 has no remaining OPEN formal entries; what remains open in the program is research (constants, two-sided law), not proof debt.

## 1. The limiting σ-algebra (the mechanism, made exact)

As r → 0 the six pair functionals {f, f_t, f_s at (±r/2, 0)} converge, through their symmetric/antisymmetric divided differences, to the pinning of **{f, f_t, f_tt, f_ttt, f_s, f_ts}(x₀)** — value, full t-jet to *third* order, and the transverse 1-jet. This is the cubic pinning: the on-axis field is observed to cubic order, so the on-axis residual opens at the quartic; the transverse direction is observed only to first order, so the transverse residual opens at f_ss — which is why the slab is polynomial and only the off-slab is exponential.

## 2. Certified variance laws (two-route: hand algebra committed first, engine authoritative)

All conditional variances proven **even in r** (P6: the conditioning set {±r/2} is r ↔ −r invariant; every odd-kr coefficient vanishes identically in the bi-graded output). Leading laws, exact:

| quantity | law | hand route | engine |
|---|---|---|---|
| Var(f_t(τ,0) \| A), on-axis | (2/3)τ⁶ | CondVar(f_tttt \| f, f_tt) = 105 − 81 = 24; ÷36 | 2/3 ✓ |
| Var(f(τ,0) \| A), on-axis | τ⁸/24 | same 24; ÷(4!)² | 1/24 ✓ |
| Var(f_t(0,σ) \| A), transverse | σ⁴/2 | CondVar(f_tss \| f_t, f_ttt) = 3 − 1 = 2; ÷4 | 1/2 ✓ |
| Var(f_s(0,σ) \| A), transverse | 2σ² | CondVar(f_ss \| f, f_tt) = 2 | 2 ✓ |
| general direction, Var(f_t) | (0,4)-coeff = z_s²(z_s² + 4z_t²)/2 | — | exact; vanishes on-axis, handing to τ⁶ |

Consequences used downstream, now at proof grade: on-axis sd(f_t) ~ τ³ against the mean floor κτ² gives the exp(−c/τ²) axis suppression; on-axis sd(f) ~ τ⁴ against the fold mean κτ³ gives the value-side suppression; the anisotropic 1/(|sinθ|+τ) degeneracy of 2-jet-only conditioning is removed exactly where claimed.

## 3. P5: the limiting nondegeneracy — exact

The conditional covariance of the six lowest unpinned jet coordinates (f_ss, f_tss, f_sss, f_ttss, f_ttts, f_tttt) given the six pinned functionals has diagonal **(2, 2, 6, 6, 6, 24)** and determinant **13824 > 0**, with all leading principal minors positive. This is the "limiting 6×6 covariance nondegenerate" of the v2 QED-plan, discharged with an exact integer. The diagonal's entries are precisely the constants that appeared independently in the certified S-centered table (Var(f_ss|A) = 2), in P3's input, and in P1/P2's input — three-route consistency.

## 4. Scope sharpening of A′ (registered refinement of the C009 correction)

Because the transverse residual opens at the *free* f_ss (variance 2), the exponential zone is **{|t| ≥ Cr} ∩ {τ ≤ δ}** — off the vertical *slab*, not merely off the two strips: the mid-line t ≈ 0 belongs to the polynomial zone as well. The C009/C010 strip measurements covered t ∈ [0, r] (mirror-symmetric to [−r, 0]), so the measured polynomial coverage is exactly the slab; no gap remains between the measured polynomial zone and the now-proven exponential zone.

## 5. The octave law (derivation catching up to a measurement)

In the deep slab (s = τ ≫ r): p_{f_t}(0) ~ 1/τ² (transverse σ⁴/2-law), p_{f_s}(0) ~ e^{−μ²/4}/(√2τ) (free-f_ss channel), value-window factor ℓ/(σ²-scale) ~ ℓ/τ², slab width ~ τ; assembling per octave: **count(τ-octave) ≍ ℓ/τ³**, geometric ratio (1/2)³ = **0.125 per octave**. The C010 measured ratios 0.08–0.14 bracket it. A law derived after its own measurement, matching it — registered as the bonus result of the cycle.

## 6. Status

Lemma P: **DISCHARGED** (variance laws exact and two-route; nondegeneracy exact; r-parity exact). The obligation registry is now clear of proof debt: every lemma in the Theorem A chain is proved, derived-with-certified-inputs-and-verified, or measured, with the open items being research objects (constants, the deep-r LB scaling, the two-sided inner law).


<!-- END SOURCE: LemmaP_Discharge.md -->


---

<!-- BEGIN SOURCE: Registry_Reconciliation_C020.md | sha256:5e699b607098 | status:ledger-record -->

# Registry_Reconciliation — obligations ledger and record hygiene (as of C020, 2026-07-09)

Supersedes Registry_Reconciliation.md (C019 edition); companion to Constants_Table_Canonical_C020.md,
Lemma_UB0.md, and C020_Observed_Update.json. Supersede-never-overwrite: nothing below deletes an archive
entry; every change is a scoped supersession.

## 1. Obligations ledger (complete, current)

| ID | content | status | discharge / carrier |
|---|---|---|---|
| MS-Sard / R0 | a.s. Morse–Smale for the conditioned flow | derived-architecture | MS_Shift_R0_Closure.md; residue → OBL-R0-UNIF |
| OBL-R0-UNIF | uniformity of the 18×18 nondegeneracy + constant chase | OPEN (bookkeeping grade) | carried |
| OBL-UB-PALM | re-derive upper-bound channel constants under Palm weighting | **DISCHARGED (C020)** — by *replacement*, not re-derivation: the pinned zone-bound route (annulus/strip/far constants) is retired; the two-sided constant is one Palm integral, E[N_qual] = ∫Λ·AO dA, with the β channel measured dead and the ledger within budget. The pinned constants become scaffolding-only (table note). | Lemma_UB0.md; C020_Observed_Update |
| OBL-C019-1 | second independent λ method at arch scale; E1 band [0.9, 1.3] adjudication | **DISCHARGED (C020)** — G-U1 two-covariance-model concordance (ratios 0.989–1.003, 6 stations); the unified constant 0.970 lies inside the committed band; the slice value 1.097 is superseded in scope (estimand change, G-U2-driven). | C020 gates |
| OBL-C019-2 | MID other-branch mechanism; marginal-typed polish artifact | OPEN (instrument hygiene) | carried |
| OBL-LBB-1 | two-point measurement at d ∈ {0.02, 0.05, 0.1} on the arch | OPEN | carried (not executed in C020; the O(r⁶) pair bound rests on the C018 d ≥ 0.2 values + the LB mechanism) |
| OBL-LB-ARCH | rigorous positivity of ∫Λ·AO dA | OPEN — the sole remaining lower-bound obligation; basis strengthened by C020 (Λ-exact estimator, two-model concordance, rim map) | Corridor_Mechanism_KILL §3; Lemma_UB0 §6 |
| **OBL-BETA-FAR** | rigorous ceiling for the far-zone window-edge-terminal β channel (∫_far Λ_max·B_far) | **NEW (C020)** | route: adjacency-type decay + the measured support edge (Gmin = 0.0062 = 300ℓ) |
| **OBL-LB-RESTATE** | restate Lemma LB's p_∞ mechanism as moderate-r; asymptotic lower bound carried by α alone | **NEW (C020)** | fork branch (iii) consequence |
| **OBL-RIM-REFINE** | rim quadrature densification + Pre-map refinement (~2% interpolation systematic) | **NEW (C020)** | optional escape-radius study attached |
| **OBL-CLIP-ARCH** | quantify the ≤ 2% arch systematic from the retired clip margins (re-run two arch rows margin-free) | **NEW (C020)** | error-budget hygiene |
| OBL-FAR-DECAY | quantitative off-pair decorrelation (γ-LOC formalization) | OPEN | carried; now load-bearing for Lemma_UB0's architecture step |
| κ_β (gain-channel multiplier) | — | **MOOT (C020)** under fork branch (iii): c_β ≤ 2.6×10⁻⁸; no multiplier enters the constant | C020_Observed_Update |
| File-5 B2-positivity ghost debt | — | DISCHARGED (C015) | LemmaP_Discharge_2.md |
| OBL-C017-1 / OBL-C017-2 / OBL-LB-ADJ | — | DISCHARGED / DISCHARGED / SUPERSEDED (C018–C019, unchanged) | prior registry |

## 2. FAILED-AS-WRITTEN ledger additions (C020; recorded without rescue)

- **G-C1 (continuity, 3.52σ at ε = 0.03):** the committed monotone upper-tail convention + 1σ z-binning
  bias the strata-assembled bulk CDF against the direct run. Both records stand; the small-ε ladder
  (load-bearing) has zero tail contribution and is unaffected.
- **G-U2 (window flatness):** the flatness premise itself was false (needle σ_t = 0.068ℓ at the arch,
  10⁻⁴ℓ near M). Drove the in-cycle estimator supersession (slice → Λ-exact). The gate as committed was
  mis-designed; recorded as such.
- **P-U2 (correction ledger, inner budget):** blown by the 372-coefficient alarm; resolved into (a) the
  genuine rim channel (+0.143, unified into α) and (b) the null-rigidity incident (below).

## 3. Instrument incident record (C020)

**Null-rigidity off-manifold conditioning (near-M zone).** The 9-pin Gram at δ ≈ 0.003–0.005 from M has
λ_min = 2.2×10⁻¹⁷ with null functional (f(M) − f(y))/√2; the retired 0.02ℓ interior clip margin displaced
the factor-2 height 57 null-sd off the compatible manifold, inflating E[W₃1] ×4×10⁴ and manufacturing
22σ far-field artifacts (measured; linear response 0.4σ-of-mountain per null-sd). Exposed by physical
forensics (f = 21–23 at escape positions). Repair: margin-free v* = clip(μ_t, window). All near-M records
superseded by compatible-ensemble reruns (*_v2 files); distorted records retained. Arch rows unaffected
(≤ 2% residual, budgeted, → OBL-CLIP-ARCH). Class: v2.1-lesson recurrence (near-degenerate conditioning
+ off-manifold target = silent ensemble distortion).

## 4. Reconciliation notes carried unchanged

C014 part-2 dual-account resolution (operative: KILL account (B)); C014p2 snapshot-absence flag;
C015/C016/C017 date-field corrections — all as in the C019 registry, unchanged.


<!-- END SOURCE: Registry_Reconciliation_C020.md -->


---

<!-- BEGIN SOURCE: Registry_Reconciliation_C021.md | sha256:aae592046fc2 | status:ledger-record -->

# Registry_Reconciliation — obligations ledger and record hygiene (as of C021, 2026-07-09)

Supersedes Registry_Reconciliation_C020.md; companion to Constants_Table_Canonical_C021.md,
Lemma_UB0.md + Addendum A, C021_Observed_Update.json. Supersede-never-overwrite throughout.

## 1. Obligations ledger (complete, current)

| ID | content | status | discharge / carrier |
|---|---|---|---|
| OBL-LB-ARCH | rigorous positivity of ∫Λ·AO dA | **OPEN — the single load-bearing item to proof-grade Theorem A.** Basis now maximal: Λ-exact estimator, two-model concordance at two rungs, whole-support qualification, clean gates. Reduction on file: LB0 + R0 + finite-dimensional Gaussian-positivity certificates + one open-tube-event step. | Lemma_UB0 §6; Corridor_Mechanism_KILL §3 |
| OBL-FAR-DECAY | quantitative off-pair decorrelation (γ-LOC) | OPEN (proof grade) | Lemma_UB0 §4 |
| OBL-BETA-FAR | rigorous far-zone β ceiling | OPEN (proof grade) | C020 registry |
| OBL-R0-UNIF | 18×18 nondegeneracy uniformity + constant chase | OPEN (bookkeeping) | MS_Shift_R0_Closure §6 |
| OBL-CLIP-ARCH | quantify the retired-clip arch systematic | **DISCHARGED (C021)**: margin-free re-runs, ratios 1.0009/1.0000 → ≤ 0.1%; r = 0.05 bar tightens to ±0.055 | c021_cliparch.json |
| OBL-RIM-REFINE | rim quadrature/Pre densification | **DOWNGRADED to optional (C021)**: rim share 0.026 and vanishing ~r^2.5 — refinement no longer moves C\*; structural interest only | P-V2 branch (ii) |
| OBL-LB-RESTATE | restate Lemma LB's p_∞ as moderate-r | OPEN (writing task) | carried |
| OBL-LBB-1 | two-point at d ∈ {0.02, 0.05, 0.1} | OPEN | carried |
| OBL-C019-2 | MID mechanism / polish artifact hygiene | OPEN | carried |
| **OBL-NEEDLE-TREND** | classify the peak needle-ratio drift 0.068 → 0.086 (constant vs r-trend) | **NEW (C021, non-load-bearing)**: one third-rung datum suffices | G-V3 note |
| **OBL-LOOP-LOCATE** | exhibit or bound-away the loop population (unmeasured at both rungs) | **NEW (C021, bookkeeping)**: the C011 absorption currently bounds an unpopulated event | Lemma_UB0 Addendum A §A.3 |

Previously discharged (unchanged): OBL-UB-PALM, OBL-C019-1, OBL-C017-1/2, OBL-LB-ADJ (superseded),
File-5 ghost debt; κ_β moot.

## 2. Fork outcomes of record (C021)

P-V1 → **branch (i)**: |Δ| = 2.5% ≤ 10% — constant stable across the octave; the frozen
measurement-completeness interpretation is adopted verbatim. P-V2 → branch (ii): rim ~r^2.5.
P-V3 → branch (i): β dead, edge deepens (894ℓ).

## 3. Incident/caveat closures (C021)

- **C020 isM-ball caveat → CLOSED by re-measurement** (c020_ballfix.py, 4 points, same seeds):
  the r = 0.05 "loop core" was ball-asserted; measured Pre = 0.81–1.00. C\*(0.05) supersede chain:
  0.9702 (as-adjudicated) → 0.9728 (ballfix-corrected); both stand. Lemma UB0 §3 dichotomy retired
  as measured geometry (Addendum A); qual/raw ≥ 0.9997 at both rungs.
- **Clean gate ledger**: C021 is the program's first cycle with every committed gate passing
  (6/6). Recorded as a protocol-health datum, not a relaxation of the FAILED-AS-WRITTEN norm.

## 4. Measurement-completeness statement (of record)

Per P-V1(i): no open item on the Theorem A path is a measurement. Submission path as experimental
mathematics with the pre-registered verification record: **CLEAR** (assembly task: manuscript
unification). Path to proof grade: OBL-LB-ARCH (load-bearing) + OBL-FAR-DECAY + OBL-BETA-FAR +
OBL-R0-UNIF. Theorem B's program is unaffected by this cycle (its own ledger stands as of C015).

## 5. Reconciliation notes carried unchanged
C014 dual-account resolution; C014p2 snapshot-absence flag; date-field corrections — as in the
C019/C020 registries.


<!-- END SOURCE: Registry_Reconciliation_C021.md -->


---

<!-- BEGIN SOURCE: Registry_Reconciliation_C022.md | sha256:dbdc8905309f | status:ledger-record -->

# Registry_Reconciliation — obligations ledger and record hygiene (as of C022, 2026-07-09)

Supersedes Registry_Reconciliation_C021.md; companion to Constants_Table_Canonical_C022.md,
Lemma_LB_ARCH_Package.md, Lemma_FD_Package.md, R0UNIF_and_LBRestate.md,
C022_Observed_Update.json. Supersede-never-overwrite throughout.

## 1. Obligations ledger (complete, current)

| ID | content | status | discharge / carrier |
|---|---|---|---|
| OBL-LB-ARCH | rigorous positivity of ∫Λ·AO dA | **CLOSED (C022)** at derived-and-verified; two cited-standard steps flagged inline (Bogachev 3.6.1 small-ball at CM points; Morse–Smale basin continuity). Five certificates (margins 10^38–10^61, both rungs) + explicit RKHS barrier with frozen-pin rigidity (ε₀ ≈ 0.008 macroscopic) + CM support step. | Lemma_LB_ARCH_Package.md |
| OBL-FAR-DECAY | quantitative off-pair decorrelation (γ-LOC) | **far-field core DISCHARGED (C022)** derived-and-verified (19-row exact tables to d = 6, both rungs, envelope beyond); tube-local part remains architecture, scope unchanged | Lemma_FD_Package.md §2 |
| OBL-BETA-FAR | rigorous far-zone β ceiling | **fixed-radius core DISCHARGED (C022)**: P(β-far) ≤ 12.41·r³ at d₀ = 0.3, rung-independent, all r ≤ 0.05; per-rung three-piece cover complete (no gap at r = 0.05). Residual re-scoped → OBL-BETA-MODZONE | Lemma_FD_Package.md §3 |
| **OBL-BETA-MODZONE** | r-uniform moderate-zone (2r, d₀) sharpening: replace Cauchy–Schwarz Hessian ceiling by the near-fold \|det H\| suppression | **NEW (C022)** — the successor residual, precisely scoped; per-rung numbers already rigorous (84.2 / 334.9 ·r³) | Lemma_FD §3 scoping |
| OBL-R0-UNIF | 18×18 nondegeneracy uniformity | **DISCHARGED (C022)**: continuity + compactness + universal interpolation off the analytic null set; mp anchors 9.73e−21 / 3.73e−23 | R0UNIF_and_LBRestate.md Part A |
| OBL-LB-RESTATE | scope Lemma LB to moderate-r | **DISCHARGED (C022)**: supersede-in-scope note; asymptotic chain of custody = LB0 → R0 → LB-ARCH → FD | Part B |
| **OBL-LB-RATE** | rigorous sharp lower rate via C012/C013 limit objects | **NEW (C022, optional)** — not required for Θ(r³) | LB-ARCH §6 |
| OBL-NEEDLE-TREND | needle-ratio drift 0.068 → 0.086 | carried (non-load-bearing; one third-rung datum) | C021 |
| OBL-LOOP-LOCATE | exhibit/bound-away loop population | carried (bookkeeping) | UB0 Addendum A |
| OBL-LBB-1 | two-point at d ∈ {0.02, 0.05, 0.1} | carried | — |
| OBL-C019-2 | MID mechanism / polish hygiene | carried | — |

Previously discharged (unchanged): OBL-UB-PALM, OBL-C019-1, OBL-CLIP-ARCH, OBL-C017-1/2,
OBL-LB-ADJ (superseded), File-5 ghost debt; OBL-RIM-REFINE optional; κ_β moot.

## 2. C022 record notes

- **Construction dead end (not a registry kill):** the 15/18-pin barrier variant rejected-as-built
  (max|w| = 1.27×10¹⁴, evaluability destroyed; natural y*-Hessian max-typed). The 9-pin +
  shaping-bump route is the construction of record.
- **Instrument incident (closed same-session):** first r = 0.025 barrier replication ran with the
  unpatched module-global ℓ (S-pin f-err −1.82×10⁻⁵ = exactly the ℓ discrepancy). Root-caused,
  fixed, re-run clean; both runs retained in the record.
- **Honest-scoping note:** the FD moderate-zone bound is per-rung rigorous but not r-uniform
  as-written (crude ceiling ~d⁻⁴ vs true Λ-scale mass); the gap is now a *named, scoped*
  obligation rather than an implicit one.

## 3. Program state (of record)

**Theorem A:** measurement-complete (C021) **+** lower-side positivity closed at derived grade
**+** rigorous O(r³) upper envelope (C022). The Θ(r³) statement holds with exponent at
proof-adjacent grade and constant measured. Remaining to full proof grade: OBL-BETA-MODZONE,
basin-continuity formalization (if referee-demanded), γ-LOC tube-local, manuscript unification.
**Theorem B:** unaffected; its ledger stands as of C015.
**Submission paths:** experimental-mathematics package — CLEAR (assembly only); methodology
paper — CLEAR; exact-lemma cluster — CLEAR; proof-grade Theorem A — the four items above.

## 4. Reconciliation notes carried unchanged
C014 dual-account resolution; C014p2 snapshot-absence flag; date-field corrections; C020 ballfix
supersede chain (0.9702 → 0.9728) — as in the C020/C021 registries.


<!-- END SOURCE: Registry_Reconciliation_C022.md -->


---

<!-- BEGIN SOURCE: Registry_Reconciliation_C023.md | sha256:5240408aa1df | status:ledger-record -->

# Registry_Reconciliation — obligations ledger and record hygiene (as of C023, 2026-07-09)

Supersedes Registry_Reconciliation_C022.md; companion to Constants_Table_Canonical_C023.md,
Lemma_MODZONE_CountLaw_Package.md, C023_Observed_Update.json. Supersede-never-overwrite.

## 1. Kill registry — new entries (FAILED-AS-WRITTEN, no rescue)

| ID | route | verdict |
|---|---|---|
| **P-W1 (C023)** | r-uniform moderate zone via the sup-decoupled certified ceiling c_sharp | branch (iii): γ ≈ 1.95 over four rungs (I = 272.5/902.6/4000/15758). Post-mortem recorded: sup-decoupling + gradient-density mean-term omission (C022/C023 ceilings remain valid, loose). |
| **P-W2 (C023b)** | r-uniform moderate zone via the **exact** Kac–Rice band-critical count | branch (iii): γ = 1.882, rms 0.022 (I_exact = 12.43/43.83/161.7/621.9). The divergence is exact truth: **no count-based route can deliver O(r³) in the moderate zone.** |

## 2. Obligations ledger (movements)

| ID | status | notes |
|---|---|---|
| OBL-BETA-MODZONE | **SUPERSEDED (C023)** | both registered routes killed; the estimand was wrong (count vs relevance) |
| **OBL-BETA-RELEVANCE** | **NEW (C023)** — the single mathematical frontier of Theorem A's r-uniform upper side | prove P(f(T) ∈ band ∧ ρ_T ∈ (2r, d₀)) ≤ C·r³ via terminal selectivity. Deterministic anchor: Σ_maxima 1{terminal} = 1 (unique S⁺ branch). Candidate mechanisms recorded: slab-traversal geometry (ribbon width ~ℓ/(κ_g d) over length d − r/2), crest decorrelation, reverse-barrier AO upper bound. Must reproduce the measured support edge 894ℓ. Includes the B1-mechanism audit item. |
| OBL-LB-ARCH, FAR-DECAY core, BETA-FAR fixed-d₀ core, R0-UNIF, LB-RESTATE | closed (C022), unchanged | fixed-d₀ far constant C(0.3) = 12.41 unaffected by the C023 corrections |
| OBL-LB-RATE (opt), OBL-NEEDLE-TREND, OBL-LOOP-LOCATE, OBL-LBB-1, OBL-C019-2, γ-LOC tube-local | carried unchanged | |

## 3. Archive corrections (scope-precise; supersede-in-scope)

1. **File-1 §5.2 (raw-law reading) + File-3 Proposition 5.2** — the claimed O(ℓ) annulus band
   count via Pinning-scaling suppression exp(−c/τ²): **OPEN-CONTRA-INDICATED** for the raw-count
   reading. Empirical basis (exact mp, four rungs): raw transverse gradient sd ~ κ_g·d (not τ³);
   crest Mahalanobis q_min ≈ 0.72 constant; E[N_ann | pair] = Θ(r^{−1.9})·ℓ exactly. Scope
   caveat: the chained / midpoint-2-jet-removed variant conditions differently and is not per se
   disproven — but no chaining rescues the raw expectation. The July-5 assembly's annulus term
   must be re-routed through OBL-BETA-RELEVANCE.
2. **Lemma B1 circle-band mechanism** — flagged: perpendicular circle stations are polynomially,
   not exponentially, suppressed under the raw law. Conclusion possibly rescuable via the
   multi-station product; audit carried inside OBL-BETA-RELEVANCE.
3. Positive replacement: the C020–C023 certified structure (near rigidity 8×10⁻⁴², corridor
   2.6×10⁻⁸, fixed-d₀ far 12.41·r³, exact per-rung moderate masses 2.07/7.31·r³) **supersedes**
   the contra-indicated annulus route at the anchor rungs with stronger, exact results.

## 4. C023 record notes

- Fork discipline: the sup-form freeze (P-W1) was adjudicated on its literal frozen statistic
  before the successor was frozen (C023b) — no silent upgrade; both snapshots hashed
  pre-compute; all adjudication data deterministic and archived.
- The gradient-density mean-term omission in C022's Lemma-FD table is a **looseness note, not a
  correctness incident** (the omitted factor is ≤ 1; all inequalities preserved).
- New certified constants promoted to the canonical table: ĉ(2) = 6.60×10⁻³ (0.5% four-rung),
  c∞ ≈ 0.14·d^{−2.9}, q_min = 0.720–0.727, I_exact table, per-rung all-in envelopes 15.5/20.7.

## 5. Program state (of record)

**Theorem A:** exponent two-sided at proof-adjacent grade (C022); **rigorous all-in per-rung
envelopes at both anchors (15.5 / 20.7 in r³ units)** with the sharp constant measured
(C\* = 0.96 ± 0.05). Remaining to r-uniform proof grade: **OBL-BETA-RELEVANCE** (the frontier),
basin-continuity formalization (if demanded), γ-LOC tube-local, manuscript unification.
**Theorem B:** unaffected; ledger stands as of C015.
**Submission paths:** experimental-mathematics package, methodology paper, exact-lemma cluster —
all CLEAR (assembly only); the C023 structural discovery (the Mahalanobis ridge + exact count
law + archive correction) is itself a publishable experimental-mathematics unit.


<!-- END SOURCE: Registry_Reconciliation_C023.md -->


---

<!-- BEGIN SOURCE: QaRegistry_Reconciliation.md | sha256:b61b26f80c45 | status:ledger-record -->

# Registry_Reconciliation — obligations ledger and record hygiene (as of C019, 2026-07-09)

Companion to Constants_Table_Canonical_07-09-26.md and the four lemma packages delivered this cycle.
Supersede-never-overwrite: nothing below deletes an archive entry; every change is a scoped supersession.

## 1. Obligations ledger (complete, current)

| ID | content | status | discharge / carrier |
|---|---|---|---|
| MS-Sard / R0 | a.s. Morse–Smale for the conditioned flow | **derived-architecture** | MS_Shift_R0_Closure.md (C015 18×18 input); residue → OBL-R0-UNIF |
| OBL-R0-UNIF | uniformity of the 18×18 nondegeneracy off the analytic null set + constant chase in the C¹ manifold citation | OPEN (bookkeeping grade) | §6 of MS_Shift_R0_Closure.md |
| OBL-C017-1 | Palm-vs-pinned correction of the halo trigger c₁ | **DISCHARGED** (C018 P-L1: c₁^{Palm} = 0.0643 ± 0.0008) | C018_Observed_Update |
| OBL-C017-2 | Bonferroni second-moment control | **DISCHARGED** | LB_Bonferroni.md (E[N(N−1)] = O(r⁶)); residue → OBL-LBB-1 |
| OBL-LBB-1 | two-point measurement at d ∈ {0.02, 0.05, 0.1} on the arch | OPEN (C020 freeze item) | LB_Bonferroni §5 |
| OBL-LB-ADJ | rigorous c₂ > 0 on the O(1) halo | **SUPERSEDED** (halo deployment killed; c₂(halo) measured 0) | → OBL-LB-ARCH |
| OBL-LB-ARCH | rigorous positivity of ∫_arch λ·A·O dA (not sharpness) | OPEN — the sole remaining lower-bound obligation; reduced to LB0 + R0 + one-point nondegeneracy on the filament | Corridor_Mechanism_KILL §3 |
| OBL-UB-PALM | re-derive upper-bound channel constants (annulus/strip/far) under Palm weighting; cover τ ∈ (r/2, 2r) | OPEN — **the program's next load-bearing item**; exponent unaffected | C019_Observed_Update; Constants table ⚠ rows |
| OBL-C019-1 | C020 freeze: arch-adapted grid; second independent λ method at arch scale; falsifiable E1 band [0.9, 1.3] | OPEN | C019_Observed_Update |
| OBL-C019-2 | MID other-branch mechanism; marginal-typed terminal-at-y polish artifact | OPEN (instrument hygiene) | C019_Observed_Update |
| File-5 B2-positivity ghost debt | — | **DISCHARGED by composition** (C015 cycle, LemmaP-general + B2-comp addendum) | LemmaP_Discharge_2.md |

## 2. Reconciliation of the two C014-part-2 accounts

The archive contains two part-2 narratives: (A) C014p2_Observed_Update.json (third-point channel as a
live failure mode, joint constants at r = 1/5) and (B) C014_KILL_ThirdPoint_Channel.md (the channel as
a moderate-r artifact: M itself sat inside the search ball at r = 0.2). **Operative account: (B)**, per
the D-A1′-gated M-excluded small-r re-run (max-third-point fraction ≤ 3×10⁻⁴ at r = 0.05). Account (A)
is retained verbatim as the FAILED-AS-WRITTEN record of the artifact, scope-tagged "moderate-r, ball
uncorrected"; its joint constants are demoted to phenomenology (see P-J6 row in the canonical table).
Chain: C013 (0.92 tail adjacency) → C014p1 (0.23–0.28, superseded) → C014-KILL (C013 reinstated) —
explicit and closed.

## 3. Record-hygiene items (logged, non-blocking)

- **C014p2 snapshot absence**: the archive holds the chain hash 724b2636… but the frozen snapshot JSON
  itself is ABSENT from the project store. The update+KILL pair fixes the content; the absence is
  logged so no future cycle claims a freeze it cannot exhibit. Action: none possible retroactively;
  flagged.
- **Date-field errors**: C015/C016/C017 update JSONs carry date 2026-07-06; the C017 verification
  session established the correct dates (C015/C016: 2026-07-06 execution confirmed; C017: 2026-07-09).
  Corrections live here; originals untouched.
- **C016 "≈14" flag**: one prose line in the C016 narrative quotes the enhancement peak as "≈14"; the
  frozen data gives R(1.5) ≈ 5. The data value governs; prose slip logged.
- **LB0 citation debt**: the LB0 proof text cites R0 and R2 informally; formal pointers are now
  MS_Shift_R0_Closure.md (R0) and the C015 reduction note (R2/Sard step). Bonferroni citation →
  LB_Bonferroni.md. LB0's Proved status is unchanged.
- **numpy 2.x**: np.trapz removed; all instruments now use np.trapezoid (silent-failure risk for any
  future re-run of old scripts; noted in instrument headers going forward).

## 4. The C019 instrument-chain lesson (for the methods paper)

The frozen G1′ pin gate functioned as the sole arbiter through two successive, differently-caused
instrument failures — (i) float64 amplification of kriging weights at cond > 1/ε (v1, 16/40 points,
errors up to 5×10²¹), then (ii) a 10⁻⁶-scale analytic-vs-discrete model mismatch exposed only after mp
whitening (v2.1) — before any science was read. The v2.2 resolution (exact coefficient-space projection:
condition the *simulator's own* Gaussian coordinates, then run stock dynamics) is the general pattern:
**when the estimand is a conditional law of a simulated Gaussian object, condition in the object's exact
coordinates rather than kriging an idealized continuum model onto it.** The same lesson produced lam3:
the frozen λ estimator was unbiased with empty effective support at shell scale (importance collapse,
CI [0, 0] false confidence); the exact disintegration computes the identical estimand with the
small-variance factor in closed form. Both replacement steps were identity-verified against the frozen
forms where those operate (v1/v2/v2.2 flow concordance; lam3 vs frozen-λ vs C018 at the G2′ point,
1.2σ/2.0σ).

## 5. Standing deliverable map (this delivery)

C018 set: snapshot + observed update + c018_flow.py + trigger files + 9 flow JSONs + g4_honest + pl5 +
pl6. C019 set: snapshot (sha256 fccf253f…) + observed update + instrument chain (4 flow versions +
lam3 + drivers) + 39-point committed grid + 33-point committed λ + arch data (ridge scan, quadrature,
micro-scans, 8 arch flows, halvings) + results assembly. Lemma packages: MS_Shift_R0_Closure,
LB_Bonferroni, Lemma_P_old, Corridor_Mechanism_KILL. Tables: Constants_Table_Canonical_07-09-26.
This file. All copied to outputs.


<!-- END SOURCE: QaRegistry_Reconciliation.md -->


---

<!-- BEGIN SOURCE: C091_CORRECTION_LEDGER.md | sha256:b0f806b3ff5c | status:ledger-record -->

# C091 CORRECTION LEDGER

**Date:** 2026-07-16  
**Discipline:** append-only. A correction changes the live dependency graph; it does not erase the historical artifact.

## E-C091-1 — Upper interpolation gate over-strengthened at C090

**Prior form:** require a bound on \(\sup|U''|\).  
**Finding:** the linear-interpolation remainder has \((r-a)(r-b)\le0\), so only negative curvature can lift \(U\) above its endpoint chord.  
**Live successor:** `UPPER_ONE_SIDED_INTERPOLATION`, requiring \(U''\ge-M\).  
**Effect on theorem:** reduces numerical burden; no theorem is weakened.

## E-C091-2 — H4 one-step Markov product killed

**Prior form:** use \(q_{step}^{n}\) for an \(n\)-checkpoint corridor.  
**Finding:** exact six-pin multivariate Gaussian diagnostics give corridor probabilities larger than \(q_{step}^{n}\) by factors up to about 3.35 in the tested skeletons.  
**Classification:** structural probability error, not a rounding issue.  
**Live successor:** `H4_PALM_ORTHANT`, using finite-dimensional Gaussian Chernoff bounds, shifted Palm determinant moments, and explicit path entropy.  
**Effect on theorem:** the C089 far coefficient \(0.0199\) returns to OPEN until re-certified.

## E-C091-3 — Bonferroni near-diagonal mark mismatch

**Prior form:** cite spatial critical-point repulsion to assert a two-point intensity per unit height squared bounded by \(C_{nd}d\), then factor \(\ell^2\).  
**Finding:** shrinking-window second-moment theory shows that nearby critical values can cost only one explicit height-window factor; same-type saddle repulsion is unmarked and does not by itself restore \(\ell^2\).  
**Classification:** incomplete source-to-estimand transfer. The old claim is quarantined, not declared impossible.  
**Live successor:** `BONFERRONI_MARKED_TWO_SCALE`, combining

\[
K_{ss}(d)\le C_{rep}d^3\log(e/d)
\]

with a bounded density for

\[
A=(u_1+u_2)/2,
\qquad
Z=(u_2-u_1)/d^3.
\]

**Effect on theorem:** lower decimal tiers are conditional on the new uniform constants. The one-point \(\Lambda\) machinery remains live.

## E-C091-4 — Exact torus transfer discharged in declared matrix scopes

**Prior status:** exact periodized ensemble defined, downstream matrix transfer open.  
**Finding:** explicit Neumann/Schur perturbation bounds preserve the local and far covariance floors with large margins.  
**Live status:** local and far-station transfer CLOSED for derivative order \(\le4\), pin dimension \(\le9\), target dimension \(\le6\), local radius \(3\), and far distance interval \([5,12]\).  
**Residual:** global integrals must partition their domains and declare any larger matrix/order use.

## E-C091-5 — Upper uncertainty polarity fixed

**Prior form:** a symmetric \(\pm0.02\) assembly band was attached to an upper theorem.  
**Finding:** upper claims consume only the positive side.  
**Live rule:** use \(+0.02\) unless the inputs are proved to be already one-sided upper limits.  
**Residual:** confidence level, multiplicity, sample size, and construction of the band remain unlocated.

## Status delta

```text
CLOSED:
    exact torus local transfer
    exact torus far-station transfer
    one-sided interpolation theorem
    upper uncertainty semantics
    Gaussian corridor Chernoff theorem
    Palm-weighted Chernoff theorem
    planar d^3 critical-value-gap law

KILLED:
    H4 q_step^n product rule

QUARANTINED:
    pointwise C_nd d claim as a uniform per-height^2 Bonferroni discharge

OPEN:
    six-pin marked saddle-repulsion constants
    H4 numerical path assembly
    upper shape certificate
    upper uncertainty calibration

BLOCKED FROM PROMOTION:
    finite lower 0.84
    asymptotic lower 0.8501
    continuous upper 0.99/1.01
```


<!-- END SOURCE: C091_CORRECTION_LEDGER.md -->


---

<!-- BEGIN SOURCE: C093_LOOSE_ENDS_DISPOSITION.md | sha256:79c5f727bade | status:ledger-record -->

# C093 LOOSE-ENDS DISPOSITION REGISTER

**Rule:** every item has one terminal disposition and one explicit effect on
the C092 core. An unnamed `OPEN` status is prohibited in the active release.

| ID | Item | Final disposition | Successor/owner | Effect on C092 core |
|---|---|---|---|---|
| LE-01 | Exact torus covariance | CORE-CLOSED | C092 | Canonical model |
| LE-02 | Local/far periodization transfer | CORE-CLOSED in declared scope | C092 | None |
| LE-03 | Fixed-L rate theorem | CORE-CLOSED at program grade | C092 | Canonical theorem |
| LE-04 | Qualitative q0=1 theorem | CORE-CLOSED | C092 | Canonical corollary |
| LE-05 | R0/SARD-G internal dependency | CORE-CLOSED | C092 | None |
| LE-06 | Independent SARD-G review | EXTERNAL-REVIEW-TRACK | Q0-REFEREE | No core dependency |
| LE-07 | Source-level marked ND audit | EXTERNAL-REVIEW-TRACK | Q0-REFEREE | No core dependency |
| LE-08 | Interval replacement of GRID moduli | DISPOSITIONED-SUCCESSOR | Q0-REFEREE | No core dependency |
| LE-09 | Full-radius analytic nine-pin atlas | DISPOSITIONED-SUCCESSOR | Q0-REFEREE | No core dependency |
| LE-10 | C089 finite lower 0.8501 | SUPERSEDED as finite theorem | Q0-SHARP | No core dependency |
| LE-11 | C089 upper 0.97 through r=.05 | KILLED by endpoint | None | No core dependency |
| LE-12 | H4 q_step^n product | KILLED by correlated-Gaussian counterexample | None | No core dependency |
| LE-13 | First-interceptor sharpened upper | DISPOSITIONED-SUCCESSOR | Q0-SHARP | No core dependency |
| LE-14 | Infinite-volume and critical-height scaling | DISPOSITIONED-SUCCESSOR | Q0-IV | No core dependency |
| LE-15 | Persistence-density Theorem B | DISPOSITIONED-SUCCESSOR | Q0-B | No core dependency |
| LE-16 | LLM empirical deployment | DISPOSITIONED-SUCCESSOR | Q0-LLM | No core dependency |
| LE-17 | Production LLM guarantee | NOT-CLAIMED | Q0-LLM | No core dependency |
| LE-18 | Scale-free alias TRUTH_CONST | KILLED/RETIRED | None | Replaced by rung tag |
| LE-19 | Absolute output paths | RELEASE-CLOSED | C093 | Portable scripts |
| LE-20 | Exact software environment | RELEASE-CLOSED | C093 | Pinned environment |
| LE-21 | Standalone builder/verifier | RELEASE-CLOSED | C093 | Included |
| LE-22 | Declared-file versus ZIP-entry count | RELEASE-CLOSED | C093 | Explicitly recorded |
| LE-23 | Exact ZIP attestation | RELEASE-CLOSED | C093 | Detached hash record |
| LE-24 | Fresh-extraction execution | RELEASE-CLOSED | C093 | All checks pass |
| LE-25 | Unowned items | CLOSED: ZERO | C093 | Release gate |

`DISPOSITIONED-SUCCESSOR` means scientifically unfinished but not unfinished
within C092/C093. `NOT-CLAIMED` is a completed boundary decision, not an
omission.

A successor can reopen C092 only through a versioned correction node containing
the failed core step, mathematical evidence, source and artifact hashes, the
downstream invalidation set, and a replacement contract.

**Unowned dispositions: 0.**


<!-- END SOURCE: C093_LOOSE_ENDS_DISPOSITION.md -->


---

<!-- BEGIN SOURCE: C094_E_LEDGER.md | sha256:86dc0dbdf55d | status:ledger-record -->

# C094 E-LEDGER — PASS 1

**Policy:** append-only. A failure, near-loss, or blocked import is preserved
with evidence, cause, downstream effect, and terminal disposition.

---

## E-C094-1 — Prescribed C093 verification order mutates a manifested file

**Evidence:** `C094_FRESH_STATE_VERIFICATION.json`.

**Observed sequence:**

```text
verify before audit: PASS
audit:               PASS, but rewrites Q0_C093_AUDIT_REPORT.json
verify after audit:  FAIL, hash and size mismatch
```

**Cause:** `audit_q0_c093_release.py` writes its current shallow audit result
over the deep-execution audit report included in the immutable manifest.

**Downstream impact:** release-pipeline reproducibility only; no mathematical
claim changes.

**Disposition:** `ERRATUM`. In every C094+ release, immutable-manifest
verification precedes writers, or the audit writes to a new versioned output
path.

---

## E-C094-2 — UB-G upper coefficient rounded in the wrong direction

**Evidence:** `C094_UBG_ADJUDICATION.json`.

**Finding:**

\[
(0.66+2.82)/0.80=4.35,
\]

while the frozen display was \(4.3\). The unrounded near/exterior assembly is
approximately \(4.3444138731\).

**Cause:** downward decimal rounding of an upper bound, with no displayed
offset certificate.

**Downstream impact:** `UB_G`, the two-sided rate display, endpoint arithmetic,
claim-language files, deprecation aliases, and both theorem hashes.

**Disposition:** `CORRECTION`. Replacement upper coefficient: \(4.35\).

---

## E-C094-3 — The \(0.8411\) lower coefficient lost its dimensions

**Evidence:** `C094_LOWER_COEFFICIENT_ADJUDICATION.json`.

**Finding:** the source writes

\[
1-q\ge C^*(r)r^3AO(r)(1-O(r^3)),
\qquad C^*(r)\ge0.8411.
\]

It does not license

\[
1-q\ge0.8411r^3
\]

without propagating the subunit AO and Bonferroni factors.

**Cause:** a first-moment coefficient was promoted to a final probability
coefficient.

**Downstream impact:** `LOWER_RATE_PG`, `RATE_PROGRAM_GRADE`, every “two-sided
closed theorem” wording.

**Disposition:** `CORRECTION`. Preserve \(C^*(r)\ge0.8411\); move finite lower
probability claims to Q0-SHARP.

---

## E-C094-4 — C091 same-name files lacked per-file provenance rows

**Evidence:** `C094_C091_C093_PROVENANCE.json`.

**Finding:** ten same-name artifacts had changed or release-specific bytes:
four portability edits, one control-character repair, one diagnostic
re-execution, and four identities; seven complete C091 artifacts were
archive-only.

**Cause:** C093 documented a portability pass globally but did not provide a
per-file same-name delta ledger.

**Downstream impact:** provenance clarity only.

**Disposition:** `ERRATUM-CLOSED` by the C094 table.

---

## E-C094-5 — Randomized orthant diagnostics changed at identical byte count

**Evidence:** same-size `gaussian_corridor_certificate_report.json` comparison.

**Finding:** twelve scalar leaves changed, exclusively
`orthant_probability_diagnostic` and `orthant_over_naive_product`. Maximum
relative change was approximately \(4.18\times10^{-7}\). The adjudication,
Chernoff bounds, covariance data, counterexample labels, and kill decision
were byte-equivalent as JSON values.

**Cause:** numerical multivariate-normal CDF diagnostics were re-executed
without a frozen integration state.

**Downstream impact:** none on theorem inputs; reproducibility metadata was
incomplete.

**Disposition:** `DIAGNOSTIC-REEXECUTION`. Future diagnostics record all
algorithmic seeds/state or are explicitly treated as non-hashed displays.

---

## E-C094-6 — “Linear repulsion is known” conflated three different claims

**Evidence:** `C094_CITATION_PROVENANCE.json`.

**Finding:**

- Ladgham–Lachièze-Rey describe typed extrema/saddle hard repulsion with three
  additional powers at the factorial-moment level;
- Azaïs–Delmas give aggregate neutrality in dimension two;
- neither fact alone supplies the pair-Palm two-height marked-window law.

**Cause:** spatial/type correlation, marked height density, and conditioned
six-pin transfer were collapsed into one citation label.

**Downstream impact:** Q0-REFEREE, BR-REP, BR-MARK, Q0-B.

**Disposition:** `CORRECTION-REGISTERED`; no citation-only mark transfer is
permitted.

---

## E-C094-7 — Nicolaescu identifier correction was stated too broadly

**Evidence:** `C094_CITATION_PROVENANCE.json`.

**Finding:** both IDs are real and refer to different papers:

```text
1101.5990  Critical sets of random smooth functions on compact manifolds
1209.0639  Random Morse functions and spectral geometry
```

**Cause:** an operator-line correction treated the IDs as globally
interchangeable.

**Downstream impact:** bibliography only.

**Disposition:** `ADJUDICATED-CONFLICT`. Match identifier to title and claimed
content.

---

## E-C094-8 — Rung minimum was consumed as a domain infimum

**Evidence:** `C094_IMPORTED_PRIORS_RECHECK.json`.

**Finding:** the C091 allowed-product values at registered rungs exceed 80,
but the full-domain limit is

\[
79.9889203915211757\ldots
\]

and the function crosses 80 at

\[
r=0.0018360362779673286\ldots.
\]

**Cause:** sampled-rung validation of a uniformly quantified interval claim.

**Downstream impact:** Q0-SHARP finite lower gate and verifier v5.

**Disposition:** `CORRECTION`. `DOMAIN-INFIMUM` is a first-class active gate.

---

## E-C094-9 — Exact six-pin factorization was nearly lost as an unowned import

**Evidence:** `C094_SIX_PIN_FACTORIZATION.json`.

**Finding:** for the exact periodized field,

\[
m_L(x,y)=k_L(y)m_L(x,0)
\]

follows exactly from product-kernel structure and parity-block decoupling.

**Cause:** the operator-line result was absent from C093 successor inputs.

**Downstream impact:** BR-MARK and Theorem B design.

**Disposition:** `IMPORTED-AFTER-REVERIFICATION`, grade `DERIVED-EXACT`.
It reshapes but does not close BR-MARK.

---

## E-C094-10 — Two operator-line measurements lack immutable source artifacts

**Items:**

- height-universality / flatness of \(q\) in \(b\);
- \(b/4\,H_{xx}(M)\) Richardson candidate.

**Cause:** no raw table, exact claim statement, or source hash exists in the
available C091/C093/Q0-X artifacts.

**Downstream impact:** Q0-IV, Q0-B, Q0-SHARP.

**Disposition:** `BLOCKED-EXTERNAL`. The unblock condition is receipt of the
immutable source plus independent rerun.

---

## E-C094-11 — C093 declared a two-sided core whose own source retained losses

**Evidence:** E-C094-2 and E-C094-3.

**Finding:** the semantic v4 graph could pass because it encoded theorem
statements, not the arithmetic assembly and propagation formulas behind them.

**Cause:** missing `ASSEMBLY`, `DOMAIN-INFIMUM`, and source-formula
cross-check gates.

**Downstream impact:** C094 corrected core; Q0-LLM verifier v5.

**Disposition:** `CORRECTION`. Live C094 root is the upper-rate theorem;
verifier v5 must include the three named gates.

---

## E-C094-12 — C093 release-close terminology overstated Pass-1 reproducibility

**Finding:** the frozen ZIP is immutable and verifies before mutation, but the
documented four-command sequence is not itself idempotent.

**Cause:** a writer was placed before hash verification.

**Downstream impact:** C093 closeout wording only.

**Disposition:** `ERRATUM`. C093 bytes remain immutable; C094 carries the
replacement procedure.

---

**Pass-1 E-ledger entries:** 12  
**Homeless entries:** 0


<!-- END SOURCE: C094_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C094_UBG_ADJUDICATION.md | sha256:ab273594faa6 | status:ledger-record -->

# C094 UB-G DISPLAY ADJUDICATION

**Item:** Directive v2.1 §2.2  
**Disposition:** **CORRECTION branch**  
**Frozen C093 archive:** unchanged  
**Affected claim IDs:** `UB_G`, `RATE_PROGRAM_GRADE`, `Q0_LIMIT`

## Finding

At 60 decimal digits,

\[
\frac{0.030449}6(576-9\pi)(1.014)
=
2.8185310984873743106224.
\]

Using the registered near coefficient \(0.657\),

\[
\frac{0.657+2.8185310984873743106}{0.80}
=
4.344413873109217888277999.
\]

Using the displayed rounded inputs,

\[
\frac{0.66+2.82}{0.80}=4.35.
\]

The frozen display \(4.3\) is therefore a downward rounding by \(0.05\) from
the displayed-input assembly and by
\(0.0444138731092178883\) from the unrounded-input
assembly. No C092 correction-ledger entry or displayed certificate licenses
that downward movement.

## Correction

Replace every live occurrence of the upper coefficient \(4.3\) by \(4.35\)
in the C094 successor release.

The corrected program-grade theorem is

\[
\boxed{
0.8411r^3
\le
1-q(r,6/5)
\le
4.35r^3,
\qquad
0<r\le0.025,
\quad L=24.
}
\]

The inherited COL and Gamma-kill conditions are unchanged. Relative to the
unrounded near/exterior inputs, \(4.35\) leaves coefficient margin

\[
4.35-4.344413873109217888278
=
0.005586126890782111722001.
\]

At \(r=0.025\), this is absolute probability margin

\[
8.728323266847049565626e-8,
\]

far above the recorded \(e^{-92}\) and \(10^{-46}\)-class station terms.
The derived COL \(O(r^3)\) backstop remains part of the same program-grade
condition set.

## Corrected endpoint

At \(r=0.025\),

\[
1-q\le4.35(0.025)^3
=
0.00006796875,
\]

hence

\[
q\ge0.99993203125.
\]

The lower side is unchanged, so

\[
q\le0.9999868578125.
\]

## Root replacement

```text
RATE_PROGRAM_GRADE
old: e24569bfedfac0d02a30ea606bcf271ad9a06650b2e935f8bdd81aa917ef5d48
new: 2521f134f94c61c7f7b27917f1ca591224acc89fb3010c1f802ae2ca52138d08

Q0_LIMIT
old: 0c643c9d26fdda9e87065b040c6d55d91c932a8748f4a5eb223f1babf10cb4c6
new: 8974f6cb3252fccd16b1e2af5a978b55090ee451bccb1f183667822bf44cd01b
```

The statement \(q(r,6/5)\to1\) is unchanged; its hash changes because its
`UB_G` dependency changed.

## Amendment classification

```text
type: CORRECTION
cause: non-conservative downward rounding of an upper bound
review status: internally re-derived at 60 dps
frozen C093 bytes: untouched
replacement release: C094 successor-era release
```


<!-- END SOURCE: C094_UBG_ADJUDICATION.md -->


---

<!-- BEGIN SOURCE: C094_LOWER_COEFFICIENT_ADJUDICATION.md | sha256:03c98a500ce8 | status:ledger-record -->

# C094 LOWER-COEFFICIENT DIMENSIONAL ADJUDICATION

**Directive row:** §2.3(3)  
**Disposition:** `ADJUDICATED-CONFLICT` → **CORRECTION**  
**Frozen C092/C093 archives:** untouched

## Source-level formula

The lower-side source states

\[
C^*(r)\ge0.8411
\]

and then assembles

\[
1-q(r,6/5)
\ge
C^*(r)r^3\,AO(r)\,(1-O(r^3)).
\]

It separately records

\[
AO(r)\ge1-2.24r^3-2e^{-111}-\text{named residues}.
\]

Therefore \(0.8411\) is a lower coefficient for the qualifying first moment,
not automatically the final defect probability.

At \(r=0.025\), even before Bonferroni and the other named residues,

\[
0.8411\left(1-2.24(0.025)^3\right)
=
0.8410705615
<
0.8411.
\]

The displayed finite lower theorem cannot follow from the displayed inputs.

## Correction

Withdraw the live core claim

\[
1-q(r,6/5)\ge0.8411r^3.
\]

Preserve the correctly typed statement

\[
C^*(r)\ge0.8411,
\qquad0<r\le0.025,
\]

at the inherited program/verification grade.

Preserve the rung-tagged measured coefficient \(0.946\) at \(r=0.025\).

Move every finite probability lower sharpening to `Q0-SHARP`, where it
requires the complete chain

```text
BONFERRONI
→ FAR_DECORRELATION
→ BR-REP
→ BR-MARK
→ SIX_PIN_PALM_UNIFORMITY
→ exact AO and loss propagation
→ full-domain infimum certificate
```

## Corrected live core

The corrected C094 core theorem is one-sided:

\[
\boxed{
0\le1-q(r,6/5)\le4.35r^3,
\qquad0<r\le0.025,
\quad L=24,
}
\]

at the same inherited program/verification grade.

It still implies

\[
\boxed{q(r,6/5)\to1}.
\]

## Root replacement

```text
old two-sided root:
e24569bfedfac0d02a30ea606bcf271ad9a06650b2e935f8bdd81aa917ef5d48

new upper-rate root:
3d6d82aed7661c388528755c8bec12ad73a991a76995275915e88c2bbcf2d0e5

old Q0_LIMIT root:
0c643c9d26fdda9e87065b040c6d55d91c932a8748f4a5eb223f1babf10cb4c6

new Q0_LIMIT root:
772f8b41b2eb13a0dfec415f415ff74049efbed33783ed24af46f6cdcbc38d73
```


<!-- END SOURCE: C094_LOWER_COEFFICIENT_ADJUDICATION.md -->


---

<!-- BEGIN SOURCE: RECONCILIATION_REPORT_C094.md | sha256:3d9dca12252a | status:ledger-record -->

# RECONCILIATION REPORT C094

**Pass:** 1  
**Status:** PASS  
**Blocking UB-G item:** resolved by CORRECTION  
**Frozen C091/C093 bytes:** unchanged

## Corrected live boundary

\[
\boxed{0\le1-q(r,6/5)\le4.35r^3,\quad0<r\le0.025,\quad L=24}
\]

\[
\boxed{\lim_{r\downarrow0}q(r,6/5)=1}
\]

The coefficient `0.8411` is preserved as the qualifying first-moment coefficient, not as an unpropagated finite defect coefficient.

## Objective results

| Objective | Result |
|---|---|
| 2.1 fresh state | PASS-WITH-ERRATUM; exact environment match; C091 valid |
| 2.2 UB-G | CORRECTION: 4.3 → 4.35 |
| 2.3 cross-line | 11 rows, no averaging |
| 2.4 provenance | 0 unclassified diffs |
| 2.5 completeness | 0 unowned items |
| 2.6 lineage | v4 subsumes validated v2/v3 surface; v3 remains importable |

## LE-01…LE-25 reconciliation

| ID | Item | C094 disposition | Owner/effect |
|---|---|---|---|
| LE-01 | Exact torus covariance | RETAINED-CLOSED | C094 corrected core |
| LE-02 | Local/far periodization transfer | RETAINED-CLOSED-IN-SCOPE | C094 corrected core |
| LE-03 | Fixed-L rate theorem | CORRECTED | two-sided C093 root superseded; upper-only 4.35 root live |
| LE-04 | Qualitative q0=1 theorem | RE-DERIVED | new Q0_LIMIT root; substance unchanged |
| LE-05 | R0/SARD-G internal dependency | RETAINED | Q0-REFEREE external review |
| LE-06 | Independent SARD-G review | OWNED | Q0-REFEREE |
| LE-07 | Source-level marked ND audit | OWNED | Q0-REFEREE |
| LE-08 | Interval replacement of GRID moduli | OWNED | Q0-REFEREE |
| LE-09 | Full-radius analytic nine-pin atlas | OWNED | Q0-REFEREE |
| LE-10 | C089 finite lower 0.8501 | SUPERSEDED | Q0-SHARP; exact loss propagation required |
| LE-11 | C089 upper 0.97 through r=.05 | KILLED | no owner needed |
| LE-12 | H4 q_step^n product | KILLED | Q0-SHARP replacement H4-PATH |
| LE-13 | First-interceptor sharpened upper | OWNED | Q0-SHARP |
| LE-14 | Infinite-volume and critical-height scaling | OWNED | Q0-IV |
| LE-15 | Persistence-density Theorem B | OWNED | Q0-B |
| LE-16 | LLM empirical deployment | OWNED-EXTENDED | Q0-LLM tracks 4.1–4.7 |
| LE-17 | Production LLM guarantee | NOT-CLAIMED | Q0-LLM terminal boundary retained |
| LE-18 | Scale-free alias TRUTH_CONST | KILLED/RETIRED | rung tag retained |
| LE-19 | Absolute output paths | RETAINED-CLOSED | C091→C093 portability rows audited |
| LE-20 | Exact software environment | REVERIFIED-EXACT | C094 environment match |
| LE-21 | Standalone builder/verifier | ERRATUM | audit writer order corrected in C094 |
| LE-22 | Declared-file vs ZIP-entry count | RETAINED | C091 and C093 counts verified |
| LE-23 | Exact ZIP attestation | RETAINED | frozen ZIPs unchanged |
| LE-24 | Fresh-extraction execution | PASS-WITH-ERRATUM | first three pass; writer-before-verifier defect filed |
| LE-25 | Unowned items | CLOSED-ZERO | C094 charter owns 40 items |

## Pass-1 constitutional audit

- Rule 5: H4 probability product remains killed.
- Rule 6: the 80 rung target is corrected by the full-domain infimum 79.9889203915… .
- Rule 7: the finite 0.8411 lower display is withdrawn.
- Rule 8: the upper display is rounded conservatively to 4.35.
- Rule 11: every numerical claim in this pass has an executed artifact.
- Rule 13: coverage remains first-class in the verifier lineage.
- Rule 14: C091/C093 archives are untouched; every change is appended in C094.

## Completeness

- LE rows reconciled: **25**
- cross-line rows: **11**
- E-ledger entries: **12**
- unowned items: **0**
- silent losses: **0**

<!-- END SOURCE: RECONCILIATION_REPORT_C094.md -->


---

<!-- BEGIN SOURCE: C094_CROSSLINE_RECON.md | sha256:560f87dad336 | status:ledger-record -->

# C094 Cross-Line Reconciliation

**Rule:** no import without independent re-derivation or re-measurement; conflicts are adjudicated, never averaged.

| ID | Item | Disposition | Grade | Owner | Result | Evidence |
|---|---|---|---|---|---|---|
| XL-01 | UB-G upper 4.3 vs 4.35 | ADJUDICATED-CONFLICT | CERTIFIED-ARITHMETIC | C094 CORRECTION | 4.3 rejected; 4.35 is the conservative replacement. New roots filed. | `C094_UBG_ADJUDICATION.json` |
| XL-02 | Six-pin transverse conditional-mean factorization | IMPORTED-AFTER-REVERIFICATION | DERIVED-EXACT | Q0-SHARP / Q0-B | m_L(x,y)=k_L(y)m_L(x,0) for the exact periodized field; mean-Hessian identities follow. BR-MARK reshaped, not closed. | `C094_SIX_PIN_FACTORIZATION.json` |
| XL-03 | Lower coefficient 0.8411 dimensional status | ADJUDICATED-CONFLICT | DERIVED-EXACT CONTRACT AUDIT | C094 CORRECTION / Q0-SHARP | 0.8411 retained as C*(r) first-moment coefficient; finite defect lower display withdrawn pending AO/Bonferroni propagation. | `C094_LOWER_COEFFICIENT_ADJUDICATION.json` |
| XL-04 | Height-universality / q flat in b | REGISTERED-SUCCESSOR-ITEM | MEASURED-CANDIDATE | Q0-IV / Q0-B | Raw operator-line data and immutable source hash are not present in the available C093/C091 artifacts; registered BLOCKED-EXTERNAL until supplied and independently rerun. | `C094_PORTFOLIO_CHARTER.json` |
| XL-05 | PALM-SUSC/STAB-UB Stage-0 priors and de-truncation artifact | IMPORTED-AFTER-REVERIFICATION | MEASURED-PROXY | Q0-IV | At b=.2, wrap 0.2433→0.0433→0.00333→0 for L=24,48,96,192; mean area L192/L96=0.93886. Single-value-pin proxy only. | `C094_IMPORTED_PRIORS_RECHECK.json` |
| XL-06 | Selected-set Campbell coefficient lambda | IMPORTED-AFTER-REVERIFICATION | MEASURED-PROXY | Q0-IV / Q0-LLM | Cross-fitted moving-selector lambda rechecks to 4.5087–4.8462; registered as selection-amplification instrument. | `C094_IMPORTED_PRIORS_RECHECK.json` |
| XL-07 | b/4 H_xx(M) Richardson candidate | REGISTERED-SUCCESSOR-ITEM | HYPOTHESIS | Q0-SHARP | Exact candidate statement and raw Richardson table are absent from available artifacts. BLOCKED-EXTERNAL pending source; must be promoted by independent derivation or killed. | `C094_PORTFOLIO_CHARTER.json` |
| XL-08 | Citation content fixes | IMPORTED-AFTER-REVERIFICATION | ESTABLISHED-SOURCE-CHECK | Q0-REFEREE | Ladgham typed law has three additional powers; Azaïs–Delmas aggregate N=2 is neutral; Nicolaescu IDs disambiguated by title. | `C094_CITATION_PROVENANCE.json` |
| XL-09 | Repulsion vs marked-window error family | ADJUDICATED-CONFLICT | DERIVED-SOURCE-AUDIT | Q0-REFEREE / Q0-SHARP / Q0-B | Spatial/type correlation and pair-Palm two-height marked density are separate claims; no citation-only transfer is licensed. | `C094_CITATION_PROVENANCE.json` |
| XL-10 | Bonferroni allowed-product domain infimum | IMPORTED-AFTER-REVERIFICATION | CERTIFIED-ARITHMETIC | Q0-SHARP / verifier v5 | Full-domain infimum is 79.9889203915 as r→0; allowed=80 crossing at r≈0.00183603628. Rung minimum is invalid. | `C094_IMPORTED_PRIORS_RECHECK.json` |
| XL-11 | C093 verification order | ADJUDICATED-CONFLICT | CERTIFIED-REPRODUCTION | C094 release engineering | Audit script mutates manifested report; verify-before-writer or versioned audit output required. | `C094_FRESH_STATE_VERIFICATION.json` |

## Governing outcomes

- corrected upper coefficient: **4.35**
- finite lower `0.8411 r^3`: withdrawn from the live C094 core;
- exact six-pin mean factorization: imported at derived-exact grade;
- Bonferroni domain-infimum target: **79.988920391521175719575325883263398928955399268802**;
- every measured import remains explicitly Stage-0/proxy unless its full pair-Palm design is executed.

<!-- END SOURCE: C094_CROSSLINE_RECON.md -->


---

<!-- BEGIN SOURCE: C094_C091_C093_PROVENANCE.md | sha256:b923e7b0e5a5 | status:ledger-record -->

# C094 C091 → C093 Inter-Release Provenance Audit

| File | Classification | C091 SHA | C093 SHA | Cause / disposition |
|---|---|---|---|---|
| `Q0_C091_PROOF_GATE_REDUCTION.md` | CONTROL-CHARACTER-REPAIR | `128ab070189f` | `10e1f4e1fa73` | hidden form-feed interpretation after LaTeX \frac caused split 'rac' lines; C093 repaired the rendered equations; semantic equations restored; no theorem-status change |
| `C091_CORRECTION_LEDGER.md` | IDENTICAL | `b0f806b3ff5c` | `b0f806b3ff5c` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `q0_c091_canonical_contract.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_contract_checker.py` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_contract_check_report.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_gate_reduction.py` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_gate_reduction_report.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `periodized_bf_matrix_transfer.py` | PORTABILITY-EDIT | `91cb008e45df` | `3b1c44449f52` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `periodized_bf_matrix_transfer_report.json` | IDENTICAL | `75f423ed03bd` | `75f423ed03bd` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `gaussian_corridor_certificate.py` | PORTABILITY-EDIT | `d8bd3f12b341` | `46f373df22bb` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `gaussian_corridor_certificate_report.json` | DIAGNOSTIC-REEXECUTION | `b6a70cfa846b` | `8b874396913a` | SciPy multivariate-normal orthant diagnostics are numerical algorithm outputs without a frozen random integration state; re-execution changed only diagnostic CDF leaves; all Chernoff candidates, covariance inputs, counterexample labels, invalid-product verdicts, and H4 replacement gates are unchanged |
| `palm_weighted_gaussian_chernoff.py` | PORTABILITY-EDIT | `e2a308393bcd` | `6a32d466d028` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `palm_weighted_gaussian_chernoff_report.json` | IDENTICAL | `1a46553e7b57` | `1a46553e7b57` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `bf_two_critical_value_gap.py` | PORTABILITY-EDIT | `d0324cdc9b9a` | `0a0b5e24bebe` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `bf_two_critical_value_gap_report.json` | IDENTICAL | `acf6691a68d9` | `acf6691a68d9` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `q0_c091_corridor_product_audit.png` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_bonferroni_budget.png` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |

## Same-size corridor report

- changed scalar leaves: **12**
- only orthant diagnostic fields changed: **True**
- adjudication object unchanged: **True**
- maximum relative diagnostic shift: **4.174e-07**

The exact Gaussian/Chernoff inputs and every terminal adjudication remain unchanged. The varying fields are explicitly labeled diagnostics in the report; they are not theorem inputs.

## Governance conclusion

All changed same-name files are now ledgered. The C091 bundle remains the immutable authority for its own bytes; the C093 variants are new-release portability or repair variants, not silent overwrites.

<!-- END SOURCE: C094_C091_C093_PROVENANCE.md -->


---

<!-- BEGIN SOURCE: C094_CITATION_PROVENANCE.md | sha256:8a0ecc9bc901 | status:ledger-record -->

# C094 Citation Provenance and Content Adjudication

## Verified primary sources

### Ladgham–Lachièze-Rey

`arXiv:2209.04150`, *Local repulsion of planar Gaussian critical points*.

The paper distinguishes the aggregate critical-point process from typed
subprocesses. Its discussion states that extrema and saddle subprocesses have
**three additional orders of magnitude** of local repulsion. The Q0
prior-art surface must not describe this typed law as merely linear.

### Azaïs–Delmas

`arXiv:1911.02300`, *Mean number and correlation function of critical points
of isotropic Gaussian fields and some results on GOE random matrices*.

The abstract states:

```text
N > 2: attraction
N = 2: neutrality
N = 1: repulsion
```

Strong maxima–minima repulsion is separate. This paper does not license a
claim of aggregate planar repulsion.

### Nicolaescu disambiguation

```text
arXiv:1101.5990
Critical sets of random smooth functions on compact manifolds

arXiv:1209.0639
Random Morse functions and spectral geometry
```

Both identifiers are valid, but for different papers. The v3.2 sentence that
pairs `1209.0639` with *Random Morse functions and spectral geometry* is
consistent. A citation that attaches `1209.0639` to *Critical sets...* must be
corrected to `1101.5990`.

## Q0-REFEREE rule

Spatial/type repulsion and the pair-Palm two-height marked-window density are
distinct claims. No external citation discharges `BR-MARK` without an
explicit mark-transfer theorem.

## Frozen-release scan

The C093 theorem contract and core verifier do not embed these citations. The
retained C091 audit cites Ladgham–Lachièze-Rey bibliographically. Therefore
the citation corrections alter the successor referee package and the
v3.2-line prior-art wording, not the frozen C093 executable theorem graph.


<!-- END SOURCE: C094_CITATION_PROVENANCE.md -->


---

<!-- BEGIN SOURCE: C094_VERIFIER_LINEAGE.md | sha256:147eef066d0e | status:ledger-record -->

# C094 Q0-X Intake and Verifier-Lineage Continuity

| Feature | v2 | v3 | v4 | Continuity route |
|---|---:|---:|---:|---|
| proof DAG validation, cycle/supersession checks | ✓ | ✓ | ✓ | imports Grade/ClaimNode/ProofGraph from v3 |
| incremental descendant invalidation and Merkle certificates | ✓ | ✓ | ✓ | v3 dependency + SemanticProofGraph Merkle roots |
| H0 widest-path grounding and exact forest compression | ✓ | ✓ | ✓ | imports SupportEdge/widest_path_grounding from v3 |
| anchored universal-kernel RKHS mismatch | ✓ | ✓ | ✓ | imports anchored_rkhs_mismatch from v3 |
| anisotropic Q-SARD tube bound and effective rank | ✓ | ✓ | ✓ | imports anisotropic_sard_tube_bound/effective_transverse_rank |
| CoverageCertificate | — | ✓ | ✓ | direct import from v3 |
| SelectionAwareRiskLedger and coverage-limited rank | — | ✓ | ✓ | direct import from v3 |
| semantic theorem-contract gates | — | — | ✓ | SemanticProofGraph.validate |
| MARK and COMPOSITION gates | — | — | ✓ | SemanticProofGraph.validate |

## Result

- v4 subsumes every validated v2/v3 component in the directive.
- `q0_llm_verifier_v3.py` remains a direct import dependency of v4.
- the clean-extraction v4 validation passed.
- Q0-X v2 is registered as the experiment and theorem-design source for Q0-IV and Q0-LLM.
- v5 additions are new C094 scope, not retroactive claims about v4.

<!-- END SOURCE: C094_VERIFIER_LINEAGE.md -->


---

<!-- BEGIN SOURCE: C095_E_LEDGER.md | sha256:adee50bb40d2 | status:ledger-record -->

# C095 E-LEDGER

**Cycle:** C095  
**Policy:** append-only; supersede-never-overwrite.

---

## E-C095-1 — Gate Framework v1.0 mixed status, assumption role, and warrant

**Finding:** `Primitive`, `Proven`, `Open`, and `Killed` were presented as one
linear epistemic ladder.

**Cause:** unlike concepts were compressed into a convenient ordering.

**Impact:** dependency-grade rules became contradictory and difficult to
implement mechanically.

**Disposition:** `CORRECTION`. Master v1.1 separates:

```text
claim status
assumption role
warrant class
reasoning mode
```

---

## E-C095-2 — Composition witness list omitted UNION_BOUND

**Finding:** v1.0 required a witness for unions while its closed witness list
did not contain `UNION_BOUND`; the worked example also suggested disjointness.

**Cause:** products, unions, and monotone assemblies were treated as one
operation.

**Impact:** an implementation could incorrectly require independence for a
union bound or accept a product under a generic “assembly” label.

**Disposition:** `CORRECTION`. v1.1 uses operation-specific witnesses and adds
`UNION_BOUND`, `MONOTONE-ASSEMBLY`, and `INTERVAL-ARITHMETIC`.

---

## E-C095-3 — The standalone master omitted C094-mandated numerical gates

**Finding:** Master v1.0 contained 13 gates but omitted:

```text
ASSEMBLY
DOMAIN-INFIMUM
BAND-PROVENANCE
```

**Evidence:** E-C094-2, E-C094-3, E-C094-5, E-C094-8, and E-C094-11.

**Disposition:** `CORRECTION`. Master v1.1 contains 16 gates plus the SCHEMA
preflight.

---

## E-C095-4 — COMMON-MODE and HEURISTIC-BRIDGE retrodiction evidence was not self-contained

**Finding:** v1.0 Appendix A described two examples for each new gate but did
not identify frozen failure IDs, artifact hashes, or the earlier catch
mechanism.

**Impact:** the gates' logic is implementable, but the meta-claim that they
passed the framework's own retrodiction protocol could not be independently
verified from the accessible release.

**Disposition:** `PROVISIONAL`. Both gates are implemented. Their
framework-level retrodiction grade remains provisional until the evidence
package is supplied.

---

## E-C095-5 — Initial v5 ASSEMBLY implementation did not consume named residual terms

**Finding:** the first v5 draft stored `residual_terms` but did not require
those terms to appear in the input map and arithmetic expression.

**Catch mechanism:** self-application of Gate 14 to the current UB-G object.

**Impact:** the draft could have repeated the exact failure it was designed to
catch.

**Disposition:** `CORRECTION`. v5 now fails when any required or residual term
is absent or unused.

---

## E-C095-6 — C094 4.35 UB-G correction still lacked a full residual coefficient

**Finding:** the C094 base coefficient

\[
(0.657+2.8185310984\ldots)/0.80
=
4.3444138731\ldots
\]

does not itself include the positive Gamma and collar residuals named by the
source.

The rounded display inputs

\[
(0.66+2.82)/0.80=4.35
\]

leave zero display-input margin.

**Impact:** the uniform decimal \(4.35r^3\) is not promotable until

\[
\sup_{0<r\le0.025}
\frac{\Gamma(r)+\operatorname{Collar}(r)}{r^3}
\]

is explicitly budgeted in the assembly.

**Disposition:** `CORRECTION`. Structural UB-G remains Derived. The decimal
upper and q₀ consequence are `Proven-Modulo UB_G_RESIDUAL_UNIFORM`.

---

## E-C095-7 — The current Q0 contract fails the verifier built to audit it

**Evidence:** `q0_llm_verifier_v5_validation.json`.

**Observed gates:**

```text
ASSEMBLY
UB_G_RESIDUAL_UNIFORM
```

**Disposition:** expected and preserved. No grandfather exception is created.

---

## E-C095-8 — A single regime-of-validity field was insufficient

**Finding:** establishment regime alone does not state where a result is being
deployed.

**Disposition:** `CORRECTION`. F-1 in v1.1 records both establishment and
deployment regimes and requires bridges for mismatches.

---

## E-C095-9 — COMMON-MODE and HEURISTIC-BRIDGE lacked mechanical schema fields

**Disposition:** v1.1 adds F-4 and F-5; v5 implements the corresponding
certificates and grade caps.

---

## E-C095-10 — Exact six-pin covariance structure was hidden inside a generic 6×6 Schur complement

**Finding:** parity and the product kernel split the conditional covariance
exactly into a 4×4 even block and a 2×2 odd block.

**Result:**

\[
C_6(X,Y)
=
k(x-x')k(y-y')
-
k(y)k(y')e(x)^TE^{-1}e(x')
-
k'(y)k'(y')o(x)^TO^{-1}o(x').
\]

**Disposition:** `DERIVED-EXACT`. BR-MARK is substantially simplified but not
closed.

---

## E-C095-11 — Gate count was treated as a presentation invariant

**Finding:** v1.0's “13 gates + 1 field” count conflicted with frozen C094
requirements.

**Disposition:** `CORRECTION`. The count is an output of the failure record.
v1.1 has 16 numbered gates, one preflight, and five field groups.

---

## E-C095-12 — Uniform numerical claims need both certificate type and domain coverage

**Finding:** a certificate value alone cannot distinguish an analytic
infimum from a sampled rung minimum.

**Disposition:** `CORRECTION`. v5 records certificate type, full domain,
extremum direction, proof hash, and the certified extremum value.

---

**Entries:** 12  
**Homeless entries:** 0


<!-- END SOURCE: C095_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C095_GATE_FRAMEWORK_REVIEW.md | sha256:acddf457d17a | status:ledger-record -->

# C095 Review of Gate Framework Master v1.0

**Disposition:** `SUPERSEDED-BY-v1.1-NOT-OVERWRITTEN`

The restriction-drop meta-rule, proof-object discipline, and first 11 semantic gates are retained. The file is not executable as written because three C094-mandated numerical gates are absent and two new gates lack the metadata needed for mechanical checking.

## Findings

| ID | Severity | Topic | Finding | Repair |
|---|---|---|---|---|
| GF-R01 | MATERIAL | Grade/status conflation | Primitive, Proven, Open, and Killed are not points on one total warrant ladder. Primitive is a role; Open/Killed are statuses. | Split assumption role, warrant grade, claim status, and reasoning mode into separate fields. |
| GF-R02 | MATERIAL | Dependency rule ambiguity | The constitutional rule first says only four grades may be dependencies, then allows explicit Open/Conjecture dependencies. | Use a promotion matrix: unresolved dependencies may appear only as explicit hypotheses; a proved implication is Proven-Modulo, otherwise the dependent claim is capped at Plausible. |
| GF-R03 | MATERIAL | Composition witness incompleteness | UNION_BOUND is omitted from the closed witness list even though the text requires a witness for unions. A union bound does not require disjointness or independence. | Add UNION_BOUND, MONOTONE_ASSEMBLY, and INTERVAL_ARITHMETIC; make witness requirements operation-specific. |
| GF-R04 | MATERIAL | Missing C094 numerical gates | The file omits ASSEMBLY, DOMAIN-INFIMUM, and BAND-PROVENANCE, which were made mandatory by frozen C094 failures. | Promote them as gates 14–16 in v1.1. |
| GF-R05 | MATERIAL | Regime field is one-sided | A single regime-of-validity field records establishment but does not explicitly record the deployment regime being checked. | F-1 becomes an establishment/deployment pair with a named bridge for every mismatch. |
| GF-R06 | MATERIAL | COMMON-MODE schema support | Gate 12 cannot be mechanical without instrument IDs, shared components, error-channel declarations, and an independence or external-crosscheck certificate. | Add verification-mode and common-mode-certificate fields. |
| GF-R07 | MATERIAL | HEURISTIC-BRIDGE schema support | Gate 13 cannot be mechanical because the proof object does not carry reasoning mode, load-bearing status, or bridge ID. | Add reasoning-mode, load-bearing, and bridge fields. |
| GF-R08 | MATERIAL | Retrodiction evidence completeness | Appendix A claims two-case retrodiction for COMMON-MODE and HEURISTIC-BRIDGE but gives no frozen failure IDs or hashes. | Retain both gates provisionally; cap framework-level promotion until a hashed retrodiction table is supplied. |
| GF-R09 | MATERIAL | PROVISIONAL semantics | The file says PROVISIONAL caps grade but does not define the cap uniformly. | A provisional gate blocks promotion to Proven/Certified; the claim may remain Derived or Plausible according to the failing gate's explicit cap. |
| GF-R10 | IMPORTANT | Coverage mechanics | Coverage is not checkable without a declared failure universe, chart family, and uncovered-mass certificate. | Add a coverage certificate schema. |
| GF-R11 | IMPORTANT | Band construction | Uncertainty side is present, but method, sample size, confidence, multiplicity, algorithmic tolerance, and seed are not. | Add F-3 uncertainty provenance and Gate 16. |
| GF-R12 | IMPORTANT | Assembly arithmetic | COMPOSITION and POLARITY do not recompute a displayed numerical assembly or enforce conservative rounding. | Add F-2 assembly expression and Gate 14. |
| GF-R13 | IMPORTANT | Uniform-domain claims | DOMAIN and ENDPOINT do not prove an infimum/supremum between sampled rungs. | Add a full-domain extremum certificate and Gate 15. |
| GF-R14 | IMPORTANT | Threshold types | The statement that every threshold must be Derived excludes legitimate normative or user-specified decision thresholds. | Separate theorem/certificate thresholds from policy thresholds; policy thresholds are Primitive-Normative and cannot be cited as mathematical facts. |
| GF-R15 | IMPORTANT | Hash semantics | A content hash is only meaningful if canonical serialization and all load-bearing metadata are included. | Hash canonical proof-object serialization plus dependency hashes. |
| GF-R16 | IMPORTANT | Schema completeness | The file says a missing field is a gate failure but does not define a schema preflight. | Add a mandatory unnumbered SCHEMA preflight before Gate 1. |

## Accepted core

- restriction-drop as organizing lens, explicitly not a theorem
- atomic proof objects and dependency graphs
- supersede-never-overwrite
- freeze-before-execute
- preserve every failure
- retrodiction instead of forward-selected scoring
- survivorship limitation
- separation of exact and deployment checks
- weakest-link discipline
- gates DOMAIN through COVERAGE as core semantic gates

## v1.1 structure

The successor specification contains:

- one mandatory `SCHEMA` preflight;
- 16 numbered gates;
- five schema field groups;
- explicit status/warrant/reasoning-mode separation;
- operation-specific composition witnesses;
- conservative numerical assembly and full-domain certificate gates;
- provisional evidence status for COMMON-MODE and HEURISTIC-BRIDGE until their claimed retrodiction cases are supplied by ID and hash.

<!-- END SOURCE: C095_GATE_FRAMEWORK_REVIEW.md -->


---

<!-- BEGIN SOURCE: C095_GATE_RETRODICTION_AUDIT.md | sha256:484f57803c53 | status:ledger-record -->

# C095 Gate Retrodiction Audit

**Frozen failure universe:** 19 entries

| Gate | Status | Hits | Silent fraction | Disposition |
|---|---|---:|---:|---|
| ASSEMBLY | QUALIFIED | 2 | 0.895 | POLARITY knows the direction but v4 did not evaluate the arithmetic expression or enforce display rounding. |
| DOMAIN-INFIMUM | QUALIFIED | 2 | 0.895 | DOMAIN checks metadata containment and ENDPOINT checks boundaries; neither proves the interior extremum of a numerically defined function. |
| BAND-PROVENANCE | QUALIFIED-WITH-SCOPE-CLARIFICATION | 2 | 0.895 | POLARITY guards side, not the construction or reproducibility of the band/tolerance. Gate scope includes stochastic numerical uncertainty. |
| COMMON-MODE | PROVISIONAL-EVIDENCE-BLOCKED | 0 | 1.000 | Implement the gate because its logic is sound; do not call the gate retrodiction-validated until the evidence package is supplied. |
| HEURISTIC-BRIDGE | PROVISIONAL-EVIDENCE-BLOCKED | 0 | 1.000 | Implement the gate and cap unbridged claims; keep the meta-validation claim provisional. |

## Qualified gates

### ASSEMBLY

- E-C094-2: upper bound rounded down from 4.35 to 4.3.
- E-C094-3: first-moment coefficient displayed before AO and Bonferroni losses.

### DOMAIN-INFIMUM

- E-C094-8: rung minimum exceeded 80 while the domain infimum was 79.9889203915… .
- C089-U-DOMAIN: discrete rung data was used toward a continuous interval claim without an extremum certificate.

### BAND-PROVENANCE

- E-C091-5: unexplained ±0.02 theorem band.
- E-C094-5: randomized orthant diagnostic changed because algorithmic state was not frozen.

## Provisional gates

COMMON-MODE and HEURISTIC-BRIDGE are implemented because their logical obligations are useful and low-cost. Their v1.0 claims of completed retrodiction remain provisional: the accessible release contains no named failure IDs and hashes for the examples asserted in Appendix A.

## Survivorship limitation

This audit covers only recorded failures. It cannot measure failures that were never detected and does not prove completeness of the gate set.

<!-- END SOURCE: C095_GATE_RETRODICTION_AUDIT.md -->


---

<!-- BEGIN SOURCE: C095_UBG_RESIDUAL_ADJUDICATION.md | sha256:467e4d8e839b | status:ledger-record -->

# C095 UB-G Residual-Uniformity Adjudication

**Amendment type:** CORRECTION  
**Affected claims:** `UPPER_RATE_PROGRAM_GRADE`, `Q0_LIMIT`  
**Frozen C094 bytes:** untouched

## Gate finding

The C094 base assembly is

\[
\frac{0.657+
2.8185310984873743106223995278378565957688812716057680797457}{0.80}
=
4.344413873109217888277999409797320744711101589507210099682125.
\]

The coefficient margin to \(4.35\) is

\[
4.35-
4.344413873109217888277999409797320744711101589507210099682125
=
0.005586126890782111722000590202679255288898410492789900317875.
\]

But the source-level inequality also names positive residuals:

\[
\Gamma(r)+\operatorname{Collar}(r).
\]

Those residuals were not inputs to the C094 arithmetic expression. Gate 14
therefore fails by construction.

The rounded display inputs

\[
(0.66+2.82)/0.80=4.35
\]

leave **zero** displayed-input margin.

## Exact replacement

The live structural inequality is

\[
\boxed{
D(r)
\le
\frac{C_{\rm near}(r)+C_{\rm ext}(r)}
{P_{\rm typed}(r)}r^3
+
\Gamma(r)
+
\operatorname{Collar}(r).
}
\]

A uniform \(4.35r^3\) display requires the additional certificate

\[
\sup_{0<r\le0.025}
\left[
\frac{C_{\rm near}(r)+C_{\rm ext}(r)}
{P_{\rm typed}(r)}
+
\frac{\Gamma(r)+\operatorname{Collar}(r)}{r^3}
\right]
\le4.35.
\]

No such complete arithmetic object is present in the available C091–C094
artifacts.

## Status

```text
structural UB-G decomposition:
    DERIVED

uniform decimal 4.35:
    BLOCKED on UB_G_RESIDUAL_UNIFORM

q0 limit derived from the cubic UB-G rate:
    PROVEN-MODULO UB_G_RESIDUAL_UNIFORM
```

The unblock artifact is a full-domain Gamma/collar coefficient certificate or
a newly assembled larger coefficient with every residual present and rounded
up.


<!-- END SOURCE: C095_UBG_RESIDUAL_ADJUDICATION.md -->


---

<!-- BEGIN SOURCE: C096_E_LEDGER.md | sha256:488e8c2775ee | status:ledger-record -->

# C096 E-LEDGER

**Cycle:** C096  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** uploaded Gate Framework files, executable kernels, legacy registry,
PDF, UB-G residual source, and BR-MARK continuation.

---

## E-C096-1 — Two different artifacts were both named Master v1.1

**Finding.** The uploaded `gate-framework-master-v1_1.md` and the C095
`GATE_FRAMEWORK_MASTER_v1_1.md` were independently current, but they specified
different gate counts and different machine contracts.

**Cause.** Parallel development lines reused the same version identifier.

**Impact.** “Use v1.1” was not a well-defined instruction.

**Disposition.** `CORRECTION`. Both files are preserved. Their compatible
content is reconciled in `GATE_FRAMEWORK_MASTER_v1_2_RECONCILED.md`.

---

## E-C096-2 — Metadata consistency had been overstated as mathematical verification

**Finding.** A gate can verify that tags agree while all tags are wrong.

**Source repair imported.** The uploaded v1.1 explicitly separates a
mechanical shell from a mathematical core and states that gates certify
tag-consistency, not tag-to-content fidelity.

**Disposition.** `CORRECTION-RETAINED`. Master v1.2 makes the shell/core
distinction constitutional.

---

## E-C096-3 — Conditions could disappear through “fully dependable” dependencies

**Finding.** Primitive and Proven-Modulo objects were called fully dependable,
but no graph rule propagated their unresolved condition sets.

**Failure shape.**

```text
H -> lemma graded Proven-Modulo -> parent graded Proven
```

could erase `H` if the parent read only the lemma's grade.

**Disposition.** `CORRECTION`. Gate Kernel 2.0 propagates complete condition
sets. A nonconditional parent inherits every unresolved condition or fails.

---

## E-C096-4 — COMMON-MODE's old provisional ceiling was a no-op

**Finding.** “Ceiling = current grade” preserves an already inflated
`Proven` label.

**Disposition.** `CORRECTION`. The default agreement-only ceiling is
`Derived`. A stricter project policy may lower it.

---

## E-C096-5 — Witness nodes were graded but not operation typed

**Finding.** The 1.x schema could use a union-bound witness for a probability
product.

**Cause.** `is_combination=True` did not identify the operation.

**Disposition.** `CORRECTION`. Kernel 2.0 records both operation and witness
type. A witness must be compatible with the exact operation.

---

## E-C096-6 — UNION_BOUND was absent from the closed witness lexicon

**Finding.** The framework invoked a union bound while requiring
“disjoint events” and omitting `UNION_BOUND` from the witness list.

**Disposition.** `CORRECTION`. A union bound requires neither disjointness nor
independence. It is an explicit witness type.

---

## E-C096-7 — Regime-of-validity recorded only establishment, not deployment

**Finding.** A transfer gate needs both regimes to decide whether a
restriction was dropped.

**Disposition.** `CORRECTION`. F-1 is now an establishment/deployment pair.

---

## E-C096-8 — Content hashes were labels rather than hashes

**Finding.** Values such as `C#1` and `T#1` were accepted as content hashes.
Changing the statement while retaining the label did not stale a parent.

**Disposition.** `CORRECTION`. Kernel 2.0 recomputes canonical SHA-256 over
every load-bearing field and verifies every pinned edge.

---

## E-C096-9 — DOMAIN skipped missing restrictions

**Finding.** Gate Kernel v1.2 passed a quantified parent when a dependency had
no domain or omitted one parent parameter.

**Evidence.** `V12-DOMAIN-EMPTY-DEPENDENCY` and
`V12-DOMAIN-MISSING-PARAMETER`.

**Disposition.** `CORRECTION`. Missing domain metadata fails a quantified
parent.

---

## E-C096-10 — MARK could be self-attested by the parent

**Finding.** A parent could copy its required marks into its supplied-marks
field even when every support node was unmarked.

**Disposition.** `CORRECTION`. Marks for a non-leaf claim are collected from
the evidence cone or a typed mark-transfer node.

---

## E-C096-11 — High-grade but untyped nodes bridged arbitrary regimes

**Finding.** In v1.2, any sufficiently graded change-of-measure or transfer
edge could bridge any source and target.

**Disposition.** `CORRECTION`. Every bridge declares kind, source regime,
target regime, scope domain, transferred marks, evidence, and falsifier.

---

## E-C096-12 — COMMON-MODE was a Boolean attestation

**Finding.** `error_independence_arg=true` passed without instrument IDs,
shared components, tested error channel, or source hash.

**Disposition.** `CORRECTION`. A first-class common-mode certificate is now
required.

---

## E-C096-13 — Provisional ceilings did not propagate

**Finding.** The registry recorded `C_const` as provisional/Derived but
admitted a Proven parent because it read the dependency's declared grade.

**Disposition.** `CORRECTION`. Effective warrant ceilings propagate through
every nonconditional edge.

---

## E-C096-14 — PRECEDENCE ignored a dead root

**Finding.** A killed root with no descendants passed because only
descendants were checked.

**Disposition.** `CORRECTION`. The root is included in its own reachable cone.

---

## E-C096-15 — CORE CLOSURE checked Open but not Conjecture or debt

**Finding.** A closed core could retain a conditional Conjecture.

**Disposition.** `CORRECTION`. Closed cores require empty condition sets, no
Open/Plausible/Conjecture nodes, no provisional gates, and no open active
gates.

---

## E-C096-16 — Registry admission bypassed graph validation

**Finding.** Gate Kernel v1.2 admitted a live claim graded Killed as:

```text
PASS at Killed
```

**Disposition.** `CORRECTION`. Kernel 2.0 separates archive, candidate, and
promotion admission after strict graph validation.

---

## E-C096-17 — Registry loading silently returned a partial graph

**Finding.** `Registry.load` ignored admission refusals. A malformed claim
could disappear while the loader returned the remaining registry.

**Disposition.** `CORRECTION`. Loading is all-or-nothing and raises with the
complete structural error set.

---

## E-C096-18 — Schema version was ignored

**Finding.** `graph_from_json` accepted arbitrary schema labels.

**Disposition.** `CORRECTION`. Kernel 2.0 rejects every schema except
`gate-kernel/2.0`.

---

## E-C096-19 — Topological order silently omitted cycles

**Finding.** A cyclic graph produced an incomplete order rather than an error.

**Disposition.** `CORRECTION`. Cycle detection is explicit and topological
ordering raises.

---

## E-C096-20 — Gate Kernel v1.2 used a hard-coded output directory

**Finding.**

```text
/mnt/user-data/outputs/q0_registry.json
```

caused the shipped demo to fail outside its original environment.

**Disposition.** `CORRECTION`. Kernel 2.0 uses CLI paths or paths supplied by
the caller.

---

## E-C096-21 — Primary verdict conflated storage and theorem promotion

**Finding.** The demo printed `PASS at Proven-Modulo` while separately listing
an Open dependency and a provisional coefficient.

**Disposition.** `CORRECTION`. Root reports now distinguish:

```text
archive validity
shell validity
conditional promotability
unconditional promotability
closability
condition set
effective warrant
```

---

## E-C096-22 — The uploaded registry was not the Q0 theorem registry

**Finding.** It uses \(C=0.4127\), \(r\le1\), \(b\le3\), and generic
inner/far/boundary placeholders, which do not match the frozen Q0 program.

**Disposition.** `AUTHORITY-CORRECTION`. It is retained as a legacy schema
demonstration. It is never cited as theorem authority.

---

## E-C096-23 — `R0` had a dangerous cross-domain name collision

**Finding.** In the uploaded registry, `R0` means an unbounded remainder.
In the Q0 program, `R0` means the Gaussian-Sard/Morse-Smale condition.

**Disposition.** `CORRECTION`. The migrated demo uses
`PAIRING_REMAINDER_CONDITION`. Bare `R0` is prohibited in that registry.

---

## E-C096-24 — Source grades in the legacy registry were unsupported

**Finding.** `inner_bound`, `far_bound`, and `bdry_bound` were labeled Proven
without proof artifacts. `C_const` claimed a closed form but supplied only
0.4127.

**Disposition.** `DOWNGRADE`. They are explicit Open/Plausible or
Open/Measured conditions in the migrated archive.

---

## E-C096-25 — Endpoint Booleans were mistaken for endpoint certificates

**Finding.** `true` was stored beside endpoint expressions, with no executed
calculation or immutable evidence object.

**Disposition.** `CORRECTION`. `ENDPOINT-FIDELITY` remains open in the
migration.

---

## E-C096-26 — The PDF was an obsolete v1.0 artifact

**Finding.** The 15-page PDF retains the old “mechanically checkable” wording
and old schema, while the accompanying Markdown is v1.1.

**Disposition.** `SUPERSEDED`. The PDF is provenance-only. A clean Master v1.2
PDF has been generated.

---

## E-C096-27 — The old PDF had weak navigation and object warnings

**Finding.** It had no bookmarks or page numbers and emitted wrong-pointing
object warnings for objects 7 0, 16 0, and 46 0.

**Disposition.** `CORRECTION`. The new PDF has a table of contents, bookmarks,
page numbers, and a normalized object structure.

---

## E-C096-28 — The absolute Gamma ceiling cannot prove a cubic rate at zero

**Finding.** The source bound is

\[
D(r)\le C_{\rm base}r^3+e^{-92}+\operatorname{Collar}(r).
\]

A positive constant cannot be absorbed into \(Cr^3\) on
\(0<r\le r_0\).

**Impact.** The printed UB-G source does not, by itself, prove \(q(r)\to1\).

**Disposition.** `CORRECTION`. The q₀ implication is
`PROVEN-MODULO GAMMA-DENSITY` and an explicit collar coefficient.

---

## E-C096-29 — The collar had an \(O(r^3)\) shape but no usable coefficient

**Finding.** The near-diagonal backstop gives, per radius-\(2r\) collar,

\[
\frac{16\pi}{3}C_{\rm nd}r^3.
\]

The retrieved record does not give a \(C_{\rm nd}\) small enough to fit the
4.35 budget.

**Disposition.** `OPEN-COEFFICIENT`. Structural cubic order remains Derived.

---

## E-C096-30 — Uniform six-pin mark density fails in the pair-scaled chart

**Finding.** The exact limiting Gaussian mark law was computed for
\((A,Z)\). On tested \(O(r)\)-scaled midpoints,

\[
\det\Sigma_{A,Z}\asymp r^8\text{ to }r^{10},
\]

and the Gaussian density supremum grows approximately \(r^{-4}\) to
\(r^{-5}\).

**Impact.** A global factorization into “spatial repulsion constant × uniformly
bounded mark-density constant” is not valid in the near-pair chart.

**Disposition.** `HYPOTHESIS-KILLED-AS-UNIFORM`. BR-MARK is reshaped around a
joint typed spatial-marked Kac–Rice bound with regional charts and Palm
determinant depletion.

---

## E-C096-31 — The framework had two candidate notions of uniformity

**Finding.** Uploaded v1.1 staged a general quantifier-order UNIFORMITY gate;
C095 established a numerical DOMAIN-INFIMUM gate.

**Disposition.** `RECONCILIATION`. Gate 15 establishes numerical
UNIFORMITY/DOMAIN-EXTREMUM. General logical quantifier-order uniformity remains
a staged extension.

---

## E-C096-32 — The 1.x kernel line cannot enforce its own conceptual master

**Evidence.** Seventeen adversarial tests reproduced seventeen shell or
registry escapes.

**Disposition.** `SUPERSEDED`. Gate Kernel 2.0 is a breaking schema revision,
not a patch-level edit.

---

## E-C096-33 — Gate Kernel 2.0 passed the independent replacement battery

**Evidence.** Thirty-six preregistered positive and negative cases passed.

**Disposition.** `CERTIFIED-SHELL-PROTOTYPE`. This grade applies to executable
contract behavior, not to the mathematical truth of arbitrary supplied
certificates.

---

**Entries:** 33  
**Homeless entries:** 0


<!-- END SOURCE: C096_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C096_ALL_FILES_DEEP_REVIEW.md | sha256:91156c9bace6 | status:ledger-record -->

# C096 Deep Review of the Uploaded Gate Framework File Set

## 0. Scope and result

This review covers every newly supplied artifact:

1. `gate-framework-master-v1_1.md`;
2. `# THE GATE FRAMEWORK — MASTER….pdf`;
3. `gate kernel v1_1.py`;
4. `gate kernel v1_2.py`;
5. `q0 registry.json`.

It also reconciles them against the already frozen C095 verifier line.

The file set contains two distinct kinds of value:

- a strong conceptual framework, especially the uploaded Markdown v1.1;
- an executable shell prototype whose implementation does not yet enforce
  several of the framework's own strongest rules.

The conceptual framework is retained and strengthened. The 1.x kernel line is
superseded by a breaking 2.0 schema rather than patched in place.

The resulting authority stack is:

```text
conceptual specification:
    GATE_FRAMEWORK_MASTER_v1_2_RECONCILED.md

executable shell:
    gate_kernel_v2_0.py

legacy demonstration registry:
    q0 registry.json
    preserved, not theorem authority

typed migration:
    q0_registry_v2_0.json
    archive-valid, not candidate- or promotion-valid

v1.0 PDF:
    immutable provenance only
```

---

# 1. Uploaded Master v1.1 Markdown

## 1.1 What it gets exactly right

### Shell/core honesty

The file's most important move is the explicit statement that the proof object
is a type and the gates are typing rules. The machine can verify that declared
tags are mutually compatible; it cannot infer that the tags faithfully state
the mathematics.

This avoids the central category error in many automated-verification systems:
metadata consistency is not theorem truth.

The corresponding division of labor is also correct:

```text
gates:
    classify and block structural misuse

operating disciplines:
    detect numerical/content error and establish tag fidelity
```

That should remain the public explanatory center of the framework.

### Bridges are nodes

The file correctly identifies the largest v1.0 escape hatch: writing the word
“Independence,” “change of measure,” or “transfer certificate” in a metadata
field does not establish the witness.

Making every bridge a first-class graded node is the right abstraction. It
turns a bridge from an attestation into a dependency whose own warrant, source,
scope, and failure modes are visible.

### Retrodiction discipline

The v1.1 rule that a candidate gate must survive COMMON-MODE analysis of its
own evidence is an excellent self-application. Two examples derived from the
same misconception are not independent witnesses.

### Domain transfer

The software-latency example is useful because it shows that the framework is
not merely Q0 terminology with renamed nouns. Warm-cache versus cold-start law,
p50 versus p95 rung, and configuration-specific model tags are real transfer
axes.

## 1.2 Remaining conceptual defects

### Warrant and conditionality are still partially conflated

The uploaded v1.1 calls Primitive and Proven-Modulo “fully dependable.” That is
safe only when conditional debts propagate.

A primitive hypothesis can support a proved implication, but not an
unconditional theorem. Likewise, a Proven-Modulo dependency with conditions
\(H_1,\ldots,H_m\) cannot support a Proven parent unless those conditions have
been discharged. Otherwise conditions disappear through the graph.

Master v1.2 therefore treats condition sets as first-class graph data.

### COMMON-MODE's old ceiling can preserve an inflated grade

The uploaded rule says COMMON-MODE PROVISIONAL has ceiling “the claim's current
grade.” If the claim has already been labeled Proven, this ceiling does
nothing. The reconciled default cap is Derived.

### Witness operations need types

The uploaded composition lexicon does not list `UNION_BOUND`, even while its
worked example uses a union bound. It also says “union bound with disjoint
events,” although disjointness is unnecessary.

A witness now specifies both:

```text
operation:
    PRODUCT | UNION | AFFINE-ASSEMBLY | ...

witness type:
    INDEPENDENCE | UNION-BOUND | EXACT-ALGEBRA | ...
```

A union witness cannot license a product.

### F-1 needs two regimes

A reusable result needs an establishment regime and a deployment regime.
Without both, a transfer gate has nothing concrete to compare.

### Canonical hashing was underspecified

The uploaded file activates content hashes but does not define canonical
serialization. A user-supplied label such as `C#1` cannot detect mutation.

Kernel 2.0 hashes deterministic JSON containing every load-bearing field,
including pinned dependency edges.

### Three numerical gates are now earned

The uploaded v1.1 correctly staged new gates conservatively. Later frozen
failures now establish:

- ASSEMBLY;
- numerical UNIFORMITY / DOMAIN-EXTREMUM;
- BAND-PROVENANCE.

These are not convenience additions. They are required by concrete failures
that the first 13 gates did not mechanically catch.

---

# 2. PDF review

## 2.1 Identity

The PDF is not the uploaded Markdown v1.1. It is the earlier v1.0 text:

```text
13 gates + 1 schema field
mechanically checkable
free-text bridge-era schema
old grade rule
old PROVISIONAL reference
```

It must therefore be cited as v1.0 provenance, never as the current master.

## 2.2 Visual inspection

The document rendered successfully into 15 page images. It is visually
legible, text-based, and uses embedded monospaced fonts.

The layout is dense but stable:

- no clipped paragraphs;
- no black squares;
- no overlapping blocks;
- tables remain readable;
- page numbering is absent;
- no diagrams or figures require separate extraction.

The contact sheet shows a consistent single-column technical-manuscript layout.
Page 15 contains only the final adoption note and end marker, leaving large
unused white space.

## 2.3 Structural inspection

```text
pages:       15
size:        612 × 792 pt
encrypted:   no
forms:       none
attachments: none
annotations: none
outline:     none
scanned:     no
```

The lack of bookmarks is a usability defect in a long specification.

Parsers emit warnings for wrong pointing objects:

```text
7 0
16 0
46 0
```

The PDF remains openable, but the next generated version should be normalized
rather than copied from this object structure.

---

# 3. Gate Kernel v1.1

## 3.1 Strengths

v1.1 is a compact and readable prototype. It already contains:

- the 13-gate vector;
- constitutional dependency checks;
- stale-edge labels;
- a useful demo graph;
- explicit scope language saying the core cannot be checked;
- clear regression examples for missing composition, measure mismatch, and a
  stale pinned hash.

It compiles and executes successfully.

## 3.2 Structural limitations

It is not a registry implementation:

- no graph-wide well-formedness validation;
- no cycle rejection;
- no strict serialization;
- no schema version;
- no canonical content hash;
- no all-or-nothing load;
- no root-level admission modes.

Its gate shells also inherit the deeper 1.x issues documented below for v1.2.

v1.1 is best retained as the pedagogical prototype from which the registry
line developed.

---

# 4. Gate Kernel v1.2

## 4.1 Improvements over v1.1

v1.2 adds genuine engineering value:

- `validate_graph`;
- JSON serialization;
- a hash-indexed registry;
- leaf-up admission;
- cycle checks;
- malformed-graph demonstration;
- round-trip intent.

The code compiles.

## 4.2 Portability failure

Running the file as shipped fails at:

```text
/mnt/user-data/outputs/q0_registry.json
```

because the directory is hard-coded and absent in the current environment.

A library-quality executable must accept a path argument or use a path relative
to the working directory/script.

## 4.3 Red-team result

Seventeen frozen adversarial tests were executed. Every one reproduced the
target implementation gap:

```text
tests:                       17
vulnerabilities reproduced: 17
secure rejections:           0
harness errors:              0
```

This does not mean the kernel has no useful checks. It means its current shell
is materially weaker than the uploaded v1.1 contract.

## 4.4 Detailed implementation findings

### DOMAIN silently skips missing information

The gate deliberately treats absent dependency domains as nonviolations. It
also checks only parameters that happen to exist in the dependency.

Therefore a parent quantified over \(r,b\) can pass using:

- a dependency with no domain;
- a dependency that declares only \(r\).

This contradicts the framework's “name the restriction” discipline.

### MARK can be satisfied by the parent itself

The gate computes

```text
required_marks - parent.supplied_marks
```

rather than collecting marks from evidence dependencies.

A parent can therefore write both required and supplied marks even when every
support node is unmarked.

### Bridge nodes are graded but untyped

Any sufficiently graded `CHANGE_OF_MEASURE` edge can bridge any measure pair.
The node does not declare:

- source measure;
- target measure;
- source conditioning depth;
- target conditioning depth;
- domain scope.

The same problem affects model and mark transfers.

Grading a bridge is necessary, but not sufficient. It must also be typed.

### COMPOSITION does not know the operation

The parent stores only `is_combination=True`. A union-bound node can therefore
license a product, and an exact-algebra witness can license an unrelated
statistical assembly.

### COMMON-MODE is a Boolean attestation

The two fields

```text
error_independence_arg
external_crosscheck
```

carry no instrument IDs, shared components, tested error channel, or evidence
hash. Setting a Boolean to true passes the gate.

### Provisional ceilings do not propagate

The registry records a provisional ceiling for `C_const`, but parent
admissibility reads the dependency's declared grade rather than the recorded
ceiling.

A Proven parent can therefore depend on a Proven-but-provisional node and
remain Proven.

### Hash drift is label drift, not content drift

`content_hash` is arbitrary input. The kernel never recomputes it from the
claim.

A statement can be materially changed while preserving the same label and all
parents remain hash-consistent.

### PRECEDENCE ignores the root itself

Reachability starts from descendants. A killed root with no children passes.

### CORE CLOSURE checks only `Grade.OPEN`

A closed core can retain a conditional Conjecture and pass because Conjecture
is not Open.

### Registry admission skips graph validation

`Registry.admit` checks duplicate hashes, unresolved dependencies, and cycles,
but does not invoke the status/grade checks in `validate_graph`.

A live claim graded Killed was admitted as:

```text
PASS at Killed
```

### Registry loading is silently partial

`Registry.load` calls `admit` and ignores its return value. A malformed claim
can be dropped while the function returns an apparently valid smaller
registry.

### Schema version is ignored

`graph_from_json` never checks the `schema` value.

### Topological ordering silently drops cycle nodes

The helper returns an incomplete order rather than raising.

### Verdict language conflates storage and promotion

The demonstration prints:

```text
T_main PASS at Proven-Modulo
```

while also reporting:

```text
Open dependency: R0
conditional debts: C_const, R0
C_const ceiling: Derived
```

A registry must distinguish archival storage, conditional theorem status, and
unconditional promotion.

---

# 5. `q0 registry.json`

## 5.1 It is a demonstration, not the Q0 theorem registry

The registry contains:

```text
C = 0.4127
r ∈ [0,1]
b ∈ [0,3]
generic inner/far/boundary claims
```

These are not the constants or domains of the frozen Q0 Rate Program.

The file must not be used as a mathematical source of truth.

## 5.2 Critical R0 collision

The registry uses `R0` for:

> an unbounded remainder in a pairing-failure expansion.

The Q0 mathematical program uses `R0` for the Gaussian-Sard/Morse-Smale
condition.

This collision is severe enough that the migrated registry renames the demo
object:

```text
PAIRING_REMAINDER_CONDITION
```

## 5.3 Unsupported source grades

The registry labels inner, far, and boundary bounds Proven but supplies no
proof objects, constants, or source IDs.

`C_const` claims “closed form + numerical value” but supplies only 0.4127.

The migration therefore preserves the claims as explicit open conditions rather
than silently endorsing their grades.

## 5.4 Orphaned fixtures

`pinned_input` is disconnected. `cm_node` is attached to the root even though
none of the root's support nodes has the Gaussian-pinned measure.

The migrated bridge and pinned input remain as disconnected historical
fixtures.

## 5.5 Missing theorem machinery

The root lacks:

- a coefficient assembly;
- a full-domain coefficient certificate;
- recomputed endpoint evidence;
- a common-mode certificate for C;
- evidence hashes for the region bounds.

The migrated graph is therefore:

```text
archive-valid:   yes
candidate-valid: no
promotable:      no
```

That is an honest terminal result for the file as supplied.

---

# 6. Reconciled specification and kernel

## 6.1 Master v1.2

The new master combines:

- the uploaded v1.1 shell/core honesty;
- graded bridge nodes;
- condition propagation;
- canonical hashes;
- strict registry semantics;
- the three retrodiction-qualified numerical gates.

## 6.2 Gate Kernel 2.0

The schema revision is intentionally major because the changes are breaking.

Kernel 2.0 implements:

- separate status, warrant, assumption role, and reasoning mode;
- strict parameter domains;
- establishment and deployment regimes;
- evidence-cone mark checking;
- typed bridges;
- operation-specific witness nodes;
- canonical SHA-256;
- pinned dependency hashes;
- all-or-nothing registry loading;
- explicit archive/candidate/promotion admission;
- provisional ceiling propagation;
- condition-set propagation;
- ASSEMBLY;
- DOMAIN-INFIMUM/UNIFORMITY;
- BAND-PROVENANCE;
- no hard-coded output path.

## 6.3 Independent validation

The 2.0 battery contains 36 cases. All 36 returned the preregistered outcome.

The battery includes every reproduced v1.2 escape and positive controls for:

- typed measure/model bridges;
- operation-compatible product witnesses;
- independent common-mode certificates;
- explicit conditional theorem use;
- conservative assemblies;
- full-domain extrema;
- complete band provenance;
- second-domain matrix perturbation;
- clean registry round-trip.

---

# 7. Final dispositions

| Artifact | Final disposition |
|---|---|
| PDF v1.0 | provenance-only; regenerate cleanly |
| uploaded Master v1.1 | conceptually authoritative input; superseded by merged v1.2 |
| kernel v1.1 | pedagogical prototype |
| kernel v1.2 | superseded executable; preserve red-team record |
| q0 registry.json | legacy demo; not theorem authority |
| Master v1.2 | current conceptual specification |
| Gate Kernel 2.0 | current executable shell |
| q0_registry_v2_0.json | strict archival migration; not promotable |

---

# 8. Next research actions

The file-set review changes the order of successor work.

First, all Q0-REFEREE and Q0-SHARP contract objects should be migrated into
Gate Kernel 2.0 so that conditions, bridges, assemblies, and residuals cannot
disappear.

Second, `UB_G_RESIDUAL_UNIFORM` remains the immediate theorem gate. No decimal
upper coefficient should be republished until Gamma and collar residuals occur
inside the actual assembly expression.

Third, BR-MARK should consume the exact six-pin covariance factorization
already derived in C095 and produce typed source/target mark-transfer
certificates rather than prose bridges.

Fourth, the framework should receive a clean v1.2 PDF with:

- normalized object structure;
- bookmarks;
- page numbers;
- a table of contents;
- no stale v1.0 wording.

Fifth, COMMON-MODE and HEURISTIC-BRIDGE should retain operational enforcement
while their framework-level retrodiction evidence remains provisional.

**End of C096 deep review.**


<!-- END SOURCE: C096_ALL_FILES_DEEP_REVIEW.md -->


---

<!-- BEGIN SOURCE: C096_Q0_REGISTRY_MIGRATION.md | sha256:86387971d400 | status:ledger-record -->

# C096 Q0 Registry Migration

**Source:** `q0 registry.json`  
**Target:** `q0_registry_v2_0.json`  
**Source bytes:** untouched

## Authority decision

The uploaded registry is a **Gate Kernel demonstration**, not the source-of-truth theorem registry for the frozen q0 Rate Program. Its coefficient, domain, generic region claims, and use of `R0` do not match the canonical program objects.

## Critical name collision

The legacy registry's `R0` means an unbounded remainder in a toy pairing-failure expansion. In the mathematical Q0 program, `R0` names the Gaussian-Sard/Morse-Smale condition. The migrated object is therefore:

```text
PAIRING_REMAINDER_CONDITION
```

and never bare `R0`.

## Finding table

| ID | Finding | Migration repair |
|---|---|---|
| REG-01 | The legacy ID R0 denotes an unbounded expansion remainder, not the canonical Q0 Gaussian-Sard/Morse-Smale condition R0. | Renamed PAIRING_REMAINDER_CONDITION. |
| REG-02 | The coefficient 0.4127, domain r∈(0,1], b∈[0,3], and three generic region bounds do not match the frozen Q0 theorem. | Registry metadata labels the graph as a legacy schema demonstration. |
| REG-03 | inner_bound, far_bound, and bdry_bound are labeled Proven without proof/evidence artifacts or numerical statements. | Migrated as explicit open/Plausible conditions rather than nonconditional theorem support. |
| REG-04 | C_const says 'closed form + numerical value' but provides only 0.4127 and an agreement-based tag with no independent-error certificate. | Migrated as an open Measured condition. |
| REG-05 | pinned_input is disconnected, while cm_node is attached to T_main despite no Gaussian-pinned support edge requiring the bridge. | Both are retained as disconnected fixtures; the bridge no longer launders into the root. |
| REG-06 | The union witness says the regions are disjoint, although a union bound requires neither disjointness nor independence. | Typed as an exact UNION_BOUND witness. |
| REG-07 | Endpoint truth is stored as Boolean values rather than as recomputable endpoint certificates. | ENDPOINT-FIDELITY remains an active open gate. |
| REG-08 | T_main prints PASS at Proven-Modulo under v1.2 while R0 is open and C_const is provisional. | The v2.0 root report separates archive validity, candidate shell validity, conditional promotion, and unconditional promotion. |
| REG-09 | Human labels such as C#1 are not cryptographic content hashes. | Every target claim and edge uses canonical SHA-256. |
| REG-10 | No coefficient assembly or full-domain uniformity certificate is present. | COEFFICIENT-ASSEMBLY-SOURCE and UNIFORM-COEFFICIENT-CERTIFICATE remain open blockers. |

## Root disposition

- structural archive admission: **True**
- candidate admission: **False**
- promoted admission: **False**

The migrated root has five explicit conditions and remains blocked by:

- `DOMAIN-INFIMUM`
- `COEFFICIENT-ASSEMBLY-SOURCE`
- `UNIFORM-COEFFICIENT-CERTIFICATE`
- `ENDPOINT-FIDELITY`

This is a successful migration outcome. It preserves the demonstration without presenting it as completed mathematics.

<!-- END SOURCE: C096_Q0_REGISTRY_MIGRATION.md -->


---

<!-- BEGIN SOURCE: C096_UBG_RESIDUAL_SOURCE_AUDIT.md | sha256:0c41ec46c12c | status:ledger-record -->

# C096 UB-G Residual Source Audit

**Grade:** `DERIVED-EXACT-IMPLICATION-AUDIT`  
**Source status affected:** fixed-volume upper-rate display and the q₀ corollary  
**Frozen source files:** unchanged

## 1. What the source actually proves as written

The upper-side source explicitly assembles

\[
D(r)
\le
C_{\rm base}r^3
+
e^{-92}
+
\operatorname{Collar}(r),
\]

where

\[
C_{\rm base}
=
\frac{0.657+
2.8185310984873743106223995278378565957688812716057680797457}
{0.80}
=
4.344413873109217888277999409797320744711101589507210099682125.
\]

The source describes the Gamma contribution as \(e^{-92}\), not as
\(C_\Gamma r^3\).

Therefore the printed inequality implies only

\[
\limsup_{r\downarrow0}D(r)
\le
e^{-92}
+
\limsup_{r\downarrow0}\operatorname{Collar}(r).
\]

A positive constant does not vanish when \(r\to0\). The step

\[
C_{\rm base}r^3+e^{-92}+\operatorname{Collar}(r)
\Longrightarrow
4.3r^3
\]

or \(4.35r^3\) on the entire punctured interval \(0<r\le r_0\) is not valid
without an additional \(r\)-dependent Gamma theorem.

## 2. Exact crossover

\[
e^{-92}
=
1.108939019312136379459597534352117145641980176873064965893702126853638866722979505118611312e-40.
\]

Relative to the exact unrounded margin

\[
4.35-C_{\rm base}
=
0.00558612689078211172200059020267925528889841049278990031787500000000000000000000000000000013,
\]

the absolute Gamma ceiling fits only when

\[
r
\ge
0.0000000000002707690093883541437041039891222756067232764080067681042304164283779642167156826243612075574.
\]

Thus even ignoring the collar, the absolute station ceiling cannot be absorbed
uniformly on a neighborhood punctured at zero.

The source's \(e^{-92}\) remains useful as a station/rung-scale diagnostic.
It is not a proof of \(O(r^3)\).

## 3. Collar reduction

The source's analytic backstop is

\[
\lambda_2(y,z)
\le
C_{\rm nd}|y-z|.
\]

For one disk of radius \(2r\),

\[
\int_0^{2r}
C_{\rm nd}t\,(2\pi t)\,dt
=
\frac{16\pi}3 C_{\rm nd}r^3.
\]

For two collars, the conservative coefficient is

\[
\frac{32\pi}3 C_{\rm nd}.
\]

Numerically,

\[
\frac{16\pi}3
=
16.755160819145563938467431377490682049051570130001,
\qquad
\frac{32\pi}3
=
33.510321638291127876934862754981364098103140260001.
\]

The source establishes the **shape** \(O(r^3)\), but the retrieved record does
not provide an explicit \(C_{\rm nd}\) that fits the narrow \(4.35\) budget.

Even with \(C_\Gamma=0\), the two-collar version would require

\[
C_{\rm nd}
\le
0.00016669869513872497927699661528163731222277464685051.
\]

No such certificate has been located.

## 4. Minimal Gamma repair

Let \(G_r\) be the gain of the maximum reached by the second branch of the
canonical saddle under the typed pair-Palm law.

The event is

\[
\Gamma_r=\{0<G_r<\ell\},
\qquad
\ell=\frac{r^3}6.
\]

A uniform density theorem

\[
\sup_{0<r\le r_0}
\sup_{u\in[0,\ell_0]}
p_{G_r}(u)
\le
M_\Gamma
<\infty
\]

implies

\[
P(\Gamma_r)
\le
M_\Gamma\ell
=
\frac{M_\Gamma}6r^3.
\]

Together with the collar backstop,

\[
D(r)
\le
\left[
C_{\rm base}
+
\frac{M_\Gamma}6
+
N_{\rm collar}\frac{16\pi}3C_{\rm nd}
\right]r^3.
\]

This is the exact missing bridge from a tiny fixed-r station probability to a
uniform cubic rate.

## 5. Final disposition

```text
structural UB-G decomposition:
    DERIVED

absolute Gamma station ceiling e^-92:
    STATION/RUNG-SCALE CERTIFICATE

uniform Gamma O(r^3):
    OPEN — GAMMA-DENSITY

collar O(r^3):
    DERIVED STRUCTURE, explicit coefficient unresolved

uniform decimal 4.35:
    BLOCKED

q(r)->1 from UB-G:
    PROVEN-MODULO GAMMA-DENSITY and explicit collar backstop
```

The q₀ conclusion may still be true. The printed \(e^{-92}\) ceiling is not
the theorem that proves it.


<!-- END SOURCE: C096_UBG_RESIDUAL_SOURCE_AUDIT.md -->


---

<!-- BEGIN SOURCE: C096_BR_MARK_LIMIT_ADJUDICATION.md | sha256:218d6eeffd98 | status:ledger-record -->

# C096 BR-MARK Limit Adjudication

**Jet reduction:** `DERIVED-EXACT`  
**Scaling evidence:** `MEASURED-DIAGNOSTIC`  
**BR-MARK:** `OPEN-RESHAPED`

## Exact limit

For candidate critical points at midpoint $m$, separation direction $e$, transverse direction $n$, and separation $d\downarrow0$, the two gradient constraints converge to
$D_e f=D_e^2 f=D_n f=D_eD_n f=0$. The marks converge to
$A_d\to f(m)$ and $Z_d\to-D_e^3f(m)/12$.

## Two-rung scaling

| scaled midpoint | angle | Var(A) exponent | Var(Z) exponent | density blow-up exponent |
|---|---:|---:|---:|---:|
| ['0', '2.25'] | 0.0000 | 6.001 | 1.998 | 4.000 |
| ['0', '2.25'] | 1.5708 | 6.111 | 0.932 | 4.001 |
| ['2', '2'] | 0.0000 | 6.003 | 1.988 | 3.994 |
| ['2', '2'] | 1.5708 | 7.998 | 1.996 | 4.999 |
| ['3', '1'] | 0.0000 | 6.279 | 1.855 | 3.931 |
| ['3', '1'] | 1.5708 | 7.093 | 0.940 | 4.473 |

The tested pair-scaled charts exhibit Gaussian density growth roughly $r^{-4}$, with a special chart near $r^{-5}$. The largest tested density was `1.372007214e9`. Fixed-physical controls were stable across rungs.

## Adjudication

The uniform planar-style hypothesis $\sup_{r,m,e}\|p_{A,Z}^{(6\mathrm{pin})}\|_\infty<\infty$ is false on the tested near-pair charts. This does not kill Bonferroni. It kills the factorization `spatial repulsion constant × globally bounded mark-density constant` as a uniform architecture.

The replacement is a regional and joint bound: exterior bounded-density chart; pair-scaled joint typed spatial-marked Kac--Rice chart; dedicated pin collars. Determinant Palm depletion and spatial repulsion must be allowed to cancel the mark-density singularity before extracting a scalar constant.

<!-- END SOURCE: C096_BR_MARK_LIMIT_ADJUDICATION.md -->


---

<!-- BEGIN SOURCE: C098_E_LEDGER.md | sha256:5930aa0dd0d8 | status:ledger-record -->

# C098 E-LEDGER

**Cycle:** C098  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** Gamma direct-gain upper route and its Q0 dependency cone.

---

## E-C098-1 — Incorrect inherited C096 ZIP hash in the C097 freeze

**Finding.** `C097_FREEZE.json` recorded

```text
735913ddc3c6ecb5ff16411da21b028d4589f6fa1c63f983f81601629a3a8373
```

as the C096 release ZIP hash.

The immutable ZIP and `C096_ATTESTATION.json` agree on

```text
735913ddf4dd2a3dfd3bf5c076ce798e13711f16f600b3d71aa7b7046e4fc188
```

**Cause.** Transcription of the detached release hash during the C097 freeze.

**Impact.** No source or release byte changed. The C097 mathematical artifacts
remain intact, but the inherited provenance field was wrong.

**Disposition.** `CORRECTION`. C098 freezes the recomputed hash and preserves
the incorrect C097 record.

---

## E-C098-2 — The selected-maximum density route was unnecessarily load-bearing

**Prior route.** C097 selected the maximum reached by the second branch and
reduced Gamma anti-concentration to seven chart/Malliavin/Palm obligations.

**Finding.** On Gamma,

\[
f(S)=b-\ell,
\qquad
0<f(X^\dagger)-f(S)<\ell,
\]

so

\[
b-\ell<f(X^\dagger)<b.
\]

Therefore

\[
\Gamma_r
\subset
\{N_{\max}((b-\ell,b))\ge1\}.
\]

**Impact.** The upper bound can use

\[
P(\Gamma_r)
\le
E N_{\max}((b-\ell,b))
\]

without selecting \(X^\dagger\) as a differentiable functional.

**Disposition.** `SUPERSEDED-AS-LOAD-PATH`. The C097 density theorem is not
false; it is retained as a standalone sharper route.

---

## E-C098-3 — Gamma now has an exact same-measure Kac–Rice count object

**Finding.** The correct integrand keeps the maximum–saddle pair determinant
weight and the third-point maximum mark inside one typed pair-Palm object:

\[
\lambda_{\max}^{MS}(x,u;r)
=
p_{(f,\nabla f)(x)\mid J_6}(u,0)
\frac{
E[W_{MS}W_x^{\max}\mid J_6,f(x)=u,\nabla f(x)=0]
}{
E[W_{MS}\mid J_6]
}.
\]

**Impact.** No Gaussian-pinned coefficient is silently deployed under
pair-Palm. No mark or conditioning law is dropped.

**Disposition.** `DERIVED-EXACT`.

---

## E-C098-4 — Exterior maximum-window transfer is qualitatively closed

**Finding.** In the corrected pair frame,

\[
W_{MS}=r^2\widehat W_r,
\qquad
E[W_{MS}\mid J_6]=r^2z_r,
\qquad
z_r\to z_0>0.
\]

Outside a fixed ball around the collapsing pair, the exact conditional
Gaussian family extends continuously to \(r=0\), the third-point
value/gradient covariance has a uniform floor, and the scaled
determinant-weighted Hessian moment is bounded.

Therefore

\[
\sup\lambda_{\max}^{MS}(x,u;r)<\infty
\]

on the exterior, and integration over a window of width \(r^3/6\) gives a
finite \(C_{\rm ext}^{\max}r^3\) bound.

**Disposition.** `CLOSED-QUALITATIVELY`. An explicit exterior decimal remains
open.

---

## E-C098-5 — Station atlas validates the pair-weight and exterior mechanism

**Finding.** The independently assembled determinant-Palm normalizers were:

```text
r=0.05:   0.00808417960496
r=0.025:  0.00201529242071
r=0.0125: 0.000503621144509
```

Their \(r^{-2}\)-scaled values are approximately:

```text
3.23367
3.22447
3.22318
```

matching the archived \(3.230979\)-class limit.

Across twelve exterior station/rung combinations, the ratio of the
pair-Palm maximum-window intensity to the stationary maximum-height benchmark
was between approximately `0.98467` and `1.01030`.

**Disposition.** `MEASURED-DIAGNOSTIC`. This validates the mechanism but is not
a continuum interval certificate.

---

## E-C098-6 — The count route cannot preserve the inherited 4.35 total

**Finding.** The stationary all-torus maximum-window benchmark is

\[
\frac{24^2\rho_{\max}(1.2)}6
=
4.19376
\]

in units of \(r^3\).

This is the Gamma count alone, before the existing interceptor coefficient and
collar residue.

**Impact.** Global maximum counting is appropriate for a qualitative finite
cubic rate and \(q_0\), not for preserving the narrow 4.35 total display.

**Disposition.** `NO-DECIMAL-PROMOTION`. The 4.35 claim remains blocked.

---

## E-C098-7 — Q0 conditional dependency cone reduced

**Prior C098 v1 conditions:**

```text
GLOBAL_INTERCEPTOR_CUBIC_UNIFORM
COLLAR_CUBIC_EXPLICIT
GAMMA_MAXCOUNT_NEAR_CUBIC
GAMMA_MAXCOUNT_EXTERIOR_CUBIC
```

**After exterior closure:**

```text
GLOBAL_INTERCEPTOR_CUBIC_UNIFORM
COLLAR_CUBIC_EXPLICIT
GAMMA_MAXCOUNT_NEAR_CUBIC
```

The updated root hashes are:

```text
UB_G_CUBIC_MODULO_MAXCOUNT_v2
ca35af0d5b8920aa6824d0899f99e097cebb8cf299685757c4f4db6760cae00e

Q0_LIMIT_MODULO_MAXCOUNT_v2
df1597967aa1cb7666a55c74bb85d176ea0647195356f920f6e87290f61911f7
```

**Disposition.** `PROVEN-MODULO THREE CONDITIONS`.

---

**Entries:** 7  
**Homeless entries:** 0


<!-- END SOURCE: C098_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C098_GAMMA_MAXCOUNT_SUPERSESSION.md | sha256:b8f0b48d7002 | status:ledger-record -->

# C098 Gamma Maximum-Window Count Supersession

**Deterministic reduction:** `PROVEN`  
**Kac–Rice identity:** `DERIVED-EXACT`  
**Finite station atlas:** `MEASURED-DIAGNOSTIC`  
**Uniform cubic theorem:** `PROVEN-MODULO`

## 1. Load-bearing observation

The direct-gain event was defined as follows:

- the canonical saddle has value
  \[
  f(S)=b-\ell,
  \qquad
  \ell=\frac{r^3}{6};
  \]
- its second ascending branch reaches a local maximum \(X^\dagger\);
- the gain satisfies
  \[
  0<f(X^\dagger)-f(S)<\ell.
  \]

Therefore

\[
b-\ell<f(X^\dagger)<b.
\]

Let

\[
W_r=(b-\ell,b)
\]

and let \(N_{\max}(W_r;D)\) count local maxima in \(D\) whose values lie in
\(W_r\). Then, pathwise,

\[
\boxed{
\Gamma_r
\subset
\{N_{\max}(W_r;\mathbb T_{24}^2)\ge1\}.
}
\]

Consequently,

\[
\boxed{
P^{\rm Palm}(\Gamma_r)
\le
E^{\rm Palm}N_{\max}(W_r;\mathbb T_{24}^2).
}
\]

This reduction does not select the terminal maximum as a differentiable
functional. It does not need:

- a chart cover for \(X^\dagger\);
- an inverse-Hessian moment for \(X^\dagger\);
- a Malliavin divergence estimate for \(G_r\);
- a chart-switching residual.

The count deliberately includes every maximum in the height window, whether
or not that maximum is reached by the branch. The overcount is conservative.

## 2. Exact typed pair-Palm Kac–Rice object

Let \(\gamma_{r,b}\) be the exact periodized BF Gaussian law conditioned on

\[
f(M)=b,\quad \nabla f(M)=0,
\]

\[
f(S)=b-\ell,\quad \nabla f(S)=0.
\]

Define

\[
W_{MS}
=
|\det H_M\det H_S|
\,\mathbf 1_{\{H_M\prec0\}}
\,\mathbf 1_{\{\det H_S<0\}},
\]

and

\[
Z_{MS}
=
E_{\gamma_{r,b}}W_{MS}.
\]

The typed maximum–saddle pair-Palm law is

\[
dP^{MS}_{r,b}
=
\frac{W_{MS}}{Z_{MS}}\,d\gamma_{r,b}.
\]

At a third point \(x\), put

\[
W_x^{\max}
=
|\det H_x|\mathbf 1_{\{H_x\prec0\}}.
\]

The exact marked intensity is

\[
\boxed{
\lambda_{\max}^{MS}(x,u;r)
=
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\frac{
E_\gamma[
W_{MS}W_x^{\max}
\mid
J_6,f(x)=u,\nabla f(x)=0
]
}{
E_\gamma[W_{MS}\mid J_6]
}.
}
\]

Hence

\[
\boxed{
E^{MS}_{r,b}N_{\max}(W_r;D)
=
\int_D
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx.
}
\]

No Gaussian-pinned quantity is silently inserted under pair-Palm. The
determinant weight and all three critical-point type marks remain inside the
same Kac–Rice object.

## 3. Regional theorem obligations

Use the exact partition

\[
\mathbb T_{24}^2
=
\mathcal C_r
\cup
\mathcal N_r
\cup
\mathcal E,
\]

where

\[
\mathcal C_r
=
B(M,2r)\cup B(S,2r),
\]

\[
\mathcal N_r
=
B_3\setminus\mathcal C_r,
\]

\[
\mathcal E
=
\mathbb T_{24}^2\setminus B_3.
\]

### `GAMMA-MAXCOUNT-COLLAR-CUBIC`

Since maxima are critical points,

\[
N_{\max}(W_r;\mathcal C_r)
\le
N_{\rm crit}(\mathcal C_r).
\]

The existing COL architecture supplies the \(O(r^3)\) shape. An explicit
coefficient remains required.

### `GAMMA-MAXCOUNT-NEAR-CUBIC`

Prove

\[
\int_{\mathcal N_r}
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx
\le
C_{\rm near}^{\max}r^3
\]

uniformly for \(0<r\le0.025\).

This is a compact three-point Kac–Rice problem. It retains the collapsing
six-pin geometry but removes branch-selection charts and selected-Hessian
inverse moments.

### `GAMMA-MAXCOUNT-EXTERIOR-CUBIC`

Prove a finite exact transfer factor for the exterior maximum-height
intensity:

\[
\int_{\mathcal E}
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx
\le
C_{\rm ext}^{\max}r^3.
\]

The final coefficient is

\[
\boxed{
C_\Gamma
=
C_{\rm col}^{\max}
+
C_{\rm near}^{\max}
+
C_{\rm ext}^{\max}.
}
\]

## 4. Station-atlas execution

`C098_GAMMA_MAXCOUNT_BENCHMARK.py` evaluates the exact pair-Palm Kac–Rice
integrand using:

- exact periodized covariance;
- 80-decimal covariance assembly;
- high-precision covariance eigenpairs;
- three-node Gauss–Legendre integration over \(u\in(b-\ell,b)\);
- frozen Monte Carlo seeds;
- 500,000 pair-weight samples per rung;
- 80,000 triple-Hessian samples per station.

The pair determinant-weight normalizers were:

| \(r\) | measured normalizer | archived anchor | relative difference |
|---:|---:|---:|---:|
| 0.05 | 0.00808417960496 | 0.00808 | 0.0517% |
| 0.025 | 0.00201529242071 | 0.00202046 | -0.2558% |
| 0.0125 | 0.000503621144509 |  |  |

The two archived anchors are reproduced within the frozen Monte Carlo
uncertainty.

Across all exterior stations and all three rungs, the ratio to the stationary
maximum-height benchmark was

\[
0.984673226
\le
\frac{\lambda_{\max}^{MS}}{\rho_{\max}(1.2)}
\le
1.010298403.
\]

Pair-scaled and fixed-near stations were strongly anisotropic and mostly
suppressed. No sampled station outside the two radius-\(2r\) collars exceeded
the stationary benchmark by more than approximately \(1.1\%\).

This is not a continuum certificate. It is a frozen falsification and
localization atlas for the interval proof.

## 5. Coefficient scale

The recorded stationary maximum-height density is

\[
\rho_{\max}(1.2)=0.043685.
\]

For a full window of width \(\ell=r^3/6\) over area \(24^2=576\), the
stationary benchmark is

\[
\boxed{
\frac{576\,\rho_{\max}(1.2)}6
=
4.19376.
}
\]

The earlier nine-pin terminal count was \(2.03r^3\) for approximately a
half-window; linear full-window scaling gives the comparison value
\(4.06r^3\). That is a different conditioned law and is not used as a MEASURE
bridge.

The counting route is therefore naturally a **qualitative cubic-rate route**.
It should not be forced into the tiny residual margin left by the inherited
\(4.35\) total display.

## 6. Supersession of the C097 route

The C097 selected-gain density theorem is not false. It remains a valid,
sharper anti-concentration program.

For the Gamma upper bound, however, the following C097 objects become
non-load-bearing:

```text
GAIN-CHART-COVER
GAIN-RESIDUAL-VARIANCE-FLOOR
GAIN-HESSIAN-INVERSE-MOMENT
GAIN-KERNEL-DERIVATIVE-BOUND
GAIN-MALLIAVIN-DIVERGENCE
GAIN-PALM-FIBER-WEIGHT
GAIN-CHART-RESIDUAL
```

They are replaced in the weakest-link path by:

```text
GAMMA-MAXCOUNT-COLLAR-CUBIC
GAMMA-MAXCOUNT-NEAR-CUBIC
GAMMA-MAXCOUNT-EXTERIOR-CUBIC
```

The exact selected-value calculus remains useful for other problems and for a
future sharp Gamma constant.

## 7. Status

```text
Gamma event-to-count reduction:
    PROVEN

exact pair-Palm count formula:
    DERIVED-EXACT

station atlas:
    MEASURED-DIAGNOSTIC

uniform Gamma O(r^3):
    PROVEN-MODULO three regional count obligations

C097 selected-density route:
    STANDALONE / NON-LOAD-BEARING FOR THIS UPPER BOUND

4.35 total coefficient:
    BLOCKED

qualitative q0 route:
    substantially simplified
```


<!-- END SOURCE: C098_GAMMA_MAXCOUNT_SUPERSESSION.md -->


---

<!-- BEGIN SOURCE: C101_E_LEDGER.md | sha256:8194eafd8d05 | status:ledger-record -->

# C101 E-LEDGER

**Cycle:** C101  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** `GLOBAL_INTERCEPTOR_CUBIC_UNIFORM` and qualitative \(q_0\).

---

## E-C101-1 — The old typing division was unnecessary for a qualitative rate

**Finding.** The original numerical UB-G assembly estimated a Gaussian-pinned
saddle count and divided by an approximately \(0.80\) typing probability.

**Repair.** C101 works directly with the exact determinant-weighted typed
pair-Palm Kac–Rice intensity. Pair typing remains inside the numerator and
normalizer.

**Disposition.** `SUPERSEDED-FOR-QUALITATIVE-RATE`. No independent typing
residue remains in this route.

---

## E-C101-2 — Exact \(\eta^6\) three-determinant factor

Under the third value and gradient constraints, each leading Hessian
determinant contains an exact \(\eta^2\) factor. Therefore

\[
\det D^2P(M)\det D^2P(S)\det D^2P(x)
=
\eta^6R.
\]

**Disposition.** `DERIVED-EXACT`.

---

## E-C101-3 — Singular window-critical count is integrable

The C099 value/gradient density has scale \(t^{-6}\). The physical triple
determinant product is \(t^6\eta^6=r^6\), and the pair-Palm normalizer is
\(r^2z_r\).

Thus the per-height intensity is bounded by

\[
Cr^4t^{-6}.
\]

After the \(r^3/6\) height window and polar integration,

\[
Cr^7\int_{(\sqrt{15}/2)r}^{\delta}t^{-5}\,dt
\le C'r^3.
\]

**Disposition.** `DERIVED-EXACT-EXISTENCE`.

---

## E-C101-4 — The predicted pointwise upper scale was not sharp

The diagnostic fixed-\(\eta\) saddle intensities decreased under rung
refinement rather than approaching the conservative \(r^{-2}\) pointwise
upper scale.

**Cause.** Pair typing and geometry impose additional depletion on the sampled
charts.

**Impact.** None on the upper theorem; the analytic scale was only an upper
bound.

**Disposition.** `PRESERVED-NONSHARPNESS`. No asymptotic equality is claimed.

---

## E-C101-5 — Exterior stationary benchmark reproduced

At physical distance \(5\), the measured window-saddle coefficient divided by
\(\rho_{\rm sad}(1.2)/6\) lay in approximately

```text
[0.99742, 1.00972]
```

over three rungs.

**Disposition.** `MEASURED-DIAGNOSTIC`.

---

## E-C101-6 — Independent exact arithmetic checker passed

Nine rational generic cases were solved from the raw affine station equations,
without importing the production symbolic derivation. Every case verified:

```text
pair station/value equations
eta^2 factor in each determinant
eta^6 factor in the product
nonpositive third determinant
polar integral identity
```

**Disposition.** `CERTIFIED-ARITHMETIC`.

---

## E-C101-7 — Final successor residual condition closed

The global interceptor count, Gamma count, and collar residues now all have
finite cubic coefficients under the exact typed pair-Palm law.

**Disposition.** `CORE-CLOSED-INTERNAL-PROGRAM-GRADE`.

---

## E-C101-8 — Qualitative rate and limit receive new roots

```text
Q0_CUBIC_RATE_EXISTENCE_C101
f92e4aae4c5b6cf7a3a34c41a43642f14f129543d03eceb82d73a4e77f49100f

Q0_LIMIT_C101
5ed20eafac7cffa923518fce06cae56822c0a2a62b2d8061e5cd90163ca47996
```

Both roots are shell-valid, unconditionally promotable within the internal Q0
program-grade scope, and have empty successor condition sets.

---

## E-C101-9 — Numerical constants remain withdrawn

The existence theorem does not restore:

```text
4.35 upper
0.8411 finite lower
0.84 finite lower
0.99 / 1.01 sharpened upper
```

**Disposition.** `NOT-CLAIMED`.

---

## E-C101-10 — Initial fresh extraction omitted transitive executable dependencies

**Finding.** The first C101 ZIP passed manifest verification, but its disposable
fresh-extraction audit failed because the release inventory omitted:

```text
C095_SIX_PIN_COVARIANCE_FACTORIZATION.py
q0 registry.json
q0_registry_v2_0.json
```

The C098 Kac–Rice instrument imports the C095 covariance implementation, and
the Gate Kernel 2.0 validation battery exercises both registry fixtures.

**Cause.** The initial inventory followed theorem dependencies but did not walk
the complete executable import/fixture dependency graph.

**Impact.** The first ZIP was not release-closed and is not authoritative.

**Disposition.** `CORRECTION`. The three transitive dependencies were added,
the manifest and ZIP were rebuilt, and true fresh extraction then passed.

---

## E-C101-11 — Audit initially required its own not-yet-created output

**Finding.** The first deep-audit preflight listed
`C101_RELEASE_AUDIT.json` as a required input before the audit had written it.

**Impact.** A valid first execution was incorrectly reported as missing a
required file.

**Disposition.** `CORRECTION`. The audit output remains in the release
inventory but is excluded from the audit's pre-execution required-input list.

---

## E-C101-12 — Claim-language checker used brittle exact strings

**Finding.** The first checker searched for wording that differed from the
actual, stronger not-claimed paragraph, even though the manuscript explicitly
withdrew the 4.3/4.35 upper displays and finite lower displays.

**Disposition.** `CORRECTION`. The checker now tests the semantic withdrawal
phrases appearing in the canonical theorem file.

---

## E-C101-13 — Large diagnostics are frozen falsifiers, not theorem-grade audit dependencies

**Finding.** Re-executing every large Monte Carlo diagnostic inside every deep
audit caused environment/runtime instability while adding no theorem-grade
warrant; the exact symbolic derivations and independent arithmetic checker are
the load-bearing computations.

**Disposition.** `PROTOCOL-CORRECTION`. The deep audit re-executes all exact
proof instruments and the compact C098 anchor, and validates the schema and
integrity of the larger frozen C099–C101 diagnostics. Full diagnostic reruns
remain available as optional reproductions under their frozen seeds.

---

**Entries:** 13  
**Homeless entries:** 0


<!-- END SOURCE: C101_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C102_E_LEDGER.md | sha256:723c8a48ddd3 | status:ledger-record -->

# C102 E-LEDGER

## E-C102-1 — Freeze inherited a pre-final C101 ZIP hash

The first C102 freeze was written before the C101 packaging-failure entries were appended and the C101 ZIP rebuilt. It pinned the then-current ZIP hash. No C102 mathematical execution occurred under that freeze.

**Disposition:** `CORRECTION`. `C102_FREEZE_v2.json` supersedes the first freeze and pins the final C101 bundle and detached attestation. The first freeze is preserved.

**Entries:** 1  
**Homeless entries:** 0


<!-- END SOURCE: C102_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C103_E_LEDGER.md | sha256:64851b5e9aa0 | status:ledger-record -->

# C103 E-LEDGER

**Cycle:** C103  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** Q0-B near-diagonal persistence-density theorem.

---

## E-C103-1 — The approximate fold lock was unnecessary for the density Jacobian

The first C103 freeze inherited the older formulation

\[
\ell=\frac{\kappa r^3}{6}(1+o(1)).
\]

For the persistence-density pushforward, define the observed normalized gap
mark exactly by

\[
\boxed{
\kappa_{\rm gap}=\frac{6(f(M)-f(S))}{r^3}.
}
\]

Then

\[
\ell=\frac{\kappa_{\rm gap}r^3}{6}
\]

is an identity.

**Disposition:** `SHARPENING`. `C103_FREEZE_v2.json` supersedes the first
freeze. Pair pinning remains relevant to the limiting mark law, but no
approximate delta-family step is needed for the Jacobian.

---

## E-C103-2 — Exact fold Jacobian and exponent closed

The exact transformation is

\[
r\,dr
=
\frac{6^{2/3}}{3}
\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
\]

More generally, radial measure \(r^{d-1}dr\) and lock power \(p\) give
lifetime exponent

\[
\frac{d}{p}-1.
\]

At \(d=2,p=3\), this is \(-1/3\).

**Disposition:** `DERIVED-EXACT`.

---

## E-C103-3 — Fixed-mark selection could not support an integrated theorem

C101/C102 proved selection at

\[
b=\frac65,\qquad\kappa=1.
\]

A persistence-density coefficient over a birth and modulus window integrates
over \(b\) and \(\kappa\). A theorem at one mark point has measure zero in
that integral and cannot be silently promoted to compact-mark uniformity.

**Disposition:** `DOMAIN/UNIFORMITY-CORRECTION`.

---

## E-C103-4 — Contact neutrality closed at program grade

The exact corrected-frame power ledger is

\[
r^3\text{ mark Jacobian}
\times
r^{-5}\text{ six-pin density}
\times
r^2\text{ typed determinant moment}
=
r^0.
\]

Thus the unadjacent typed candidate-pair intensity has a finite positive
limit.

The adjacency factor converges by scaled-field convergence and
Morse–Smale/SARD-G structural stability. It is positive because the canonical
fold has an open trapping configuration with positive Gaussian small-ball
probability.

**Disposition:** `PROGRAM-GRADE-PROVEN`.

**Boundary:** external specialist acceptance of SARD-G remains pending.

---

## E-C103-5 — Compact-mark algebra and selection uniformity closed

The C099–C101 symbolic identities were rederived with arbitrary
\(\kappa>0\):

- the third-maximum determinant remains a negative sum of squares;
- every window-critical Hessian determinant retains an \(\eta^2\) factor;
- the three-determinant product retains \(\eta^6\);
- the collar collision product retains a \(\rho^2\) zero.

All additional \(b,\kappa\) factors are uniformly bounded on compact
\(B\times K\subset\mathbb R\times(0,\infty)\).

Therefore

\[
\sup_{\theta,b,\kappa}
(1-q(r,\theta,b,\kappa))
\le C_{B,K}r^3.
\]

**Disposition:** `PROGRAM-GRADE-PROVEN`.

---

## E-C103-6 — Modulus tails and off-fold separation closed

The corrected-frame Gaussian density gives

\[
I_r(\theta,b,\kappa)
\le
C(1+\kappa^m)e^{-c\kappa^2}.
\]

Consequently,

\[
\kappa^{-2/3}I_r
\]

is integrable near both \(0\) and \(\infty\).

Pair separations outside a fixed local chart have bounded density per unit
lifetime and contribute

\[
O(1)=o(\ell^{-1/3}).
\]

**Disposition:** `PROGRAM-GRADE-PROVEN`.

---

## E-C103-7 — Theorem B mathematical core obtained new roots

The Gate Kernel v2.0 contract reports both compact-mark and full-\(\kappa\)
roots as shell-valid, closable, and unconditionally promotable at internal
program grade:

```text
THEOREM_B_COMPACT_MARK_C103_v3
25ea3f354d485d8fe72669f64f1fd8687794b5034ce211a9994dd1fc01e50b59

THEOREM_B_FULL_KAPPA_C103_v3
b1be71324ec28c3bafea0263a0fad6b212d48906ad110bc32b5af34c90054d55
```

The full mathematical statement is

\[
\nu_B(\ell)
=
C_*\ell^{-1/3}(1+o(1)),
\qquad
0<C_*<\infty,
\]

for a compact birth-height window \(B\), under the exact model and the
program-grade SARD-G layer.

**Disposition:** `MATHEMATICAL-CORE-CLOSED-PROGRAM-GRADE`.

No numerical value of \(C_*\) is claimed.

---

## E-C103-8 — The pre-registered held-out simulation failed

The primary frozen gate used a periodic four-neighbor vertex filtration at
grid \(256\) and the window \([0.005,0.05]\).

Observed cumulative slope:

\[
0.1487,
\]

with field-bootstrap 95% interval

\[
[0.1370,0.1616].
\]

The target was

\[
\frac23.
\]

Every registered confidence interval at both grids excluded \(2/3\) in the
same downward direction. The frozen kill signal triggered.

**Disposition:** `FAILED-AS-WRITTEN`. The result may not be relabeled a pass.

---

## E-C103-9 — A discretization layer was detected but did not fully explain the failure

For grid spacing \(h=24/n\),

```text
n=192:
    q25(lifetime)/h^2 = 0.571

n=256:
    q25(lifetime)/h^2 = 0.654
```

A large population of bars therefore collapses at an \(h^2\)-class scale,
consistent with grid/interpolation artifacts contaminating the cumulative
count.

However, exploratory windows above the visible \(h^2\) layer still produced
slopes well below \(2/3\).

**Disposition:** `INSTRUMENT-OR-THEORY-UNRESOLVED`.

The discrete proxy does not automatically falsify the continuum theorem, but
it blocks empirical completion.

---

## E-C103-10 — Q0-B is not terminally closed

The mathematical derivation is internally closed at program grade, but the
project’s pre-registered held-out validation gate failed.

A new obligation is registered:

```text
CONTINUUM_PERSISTENCE_VALIDATION
```

The repair must use a continuum-consistent persistence approximation and
coupled refinement.

**Disposition:** `NOT TERMINAL`.

---

**Entries:** 10  
**Homeless entries:** 0


<!-- END SOURCE: C103_E_LEDGER.md -->


---

<!-- BEGIN SOURCE: C104_E_LEDGER.md | sha256:938965bb573d | status:ledger-record -->

# C104 E-LEDGER

**Cycle:** C104  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** continuum-consistent validation and Q0-B project close.

---

## E-C104-1 — The first full execution exceeded the notebook wrapper limit

The initial C104 implementation used pure-Python union-find and a
field-by-field bootstrap likelihood. The visible execution wrapper interrupted
the run before completion.

**Diagnosis:** computational architecture, not mathematical failure.

**Repair:**

- compiled periodic Freudenthal persistence with Numba;
- replaced repeated likelihood scans with field-level sufficient statistics;
- retained the frozen field counts, grids, seeds, cutoffs, and pass criteria.

**Disposition:** `INSTRUMENT-OPTIMIZATION`. The experimental specification was
not changed.

---

## E-C104-2 — The frozen primary continuum-consistent gate survives

For the frozen primary pair \(256\to512\) and \(U=0.2\):

\[
\widehat\alpha=-0.48144,
\]

with 95% field-bootstrap interval

\[
[-0.71150,-0.24375].
\]

The interval contains

\[
-\frac13.
\]

There were 543 refinement-stable bars and zero failed bootstrap replicates.

**Disposition:** `SURVIVES`.

---

## E-C104-3 — The frozen kill signal does not trigger

Not every valid confidence interval excludes \(-1/3\), and the finest primary
estimate moves toward \(-1/3\) relative to the preceding resolution pair.

**Disposition:** `KILL-SIGNAL-FALSE`.

---

## E-C104-4 — Refinement behavior is directionally consistent

Mean interpolation error decreased:

```text
128->256: 0.05482
192->384: 0.02495
256->512: 0.01418
```

The refinement-stable bar fraction increased:

```text
128->256: 0.5555
192->384: 0.6804
256->512: 0.7336
```

**Disposition:** `MEASURED-SUPPORT`.

---

## E-C104-5 — A secondary finite-window tension is preserved

At \(256\to512\) with \(U=0.3\),

\[
\widehat\alpha=-0.50047
\]

with interval

\[
[-0.63855,-0.37905],
\]

which excludes \(-1/3\).

**Disposition:** `PRESERVED-LIMITATION`.

This does not replace the frozen primary gate, but it forbids describing the
experiment as a precise confirmation of the exponent.

---

## E-C104-6 — The C103 failure remains valid for its proxy

C103's four-neighbor raw cumulative-count test failed and triggered its kill
signal. C104 does not relabel that result.

The C104 replacement used a different pre-registered measurement object:

- Freudenthal PL persistence;
- nested-grid coupling;
- interpolation-error bounds;
- diagram matching;
- refinement-stability filtering;
- left-truncated likelihood.

**Disposition:** `SUPERSEDED-AS-VALIDATION-INSTRUMENT`, not erased.

---

## E-C104-7 — Continuum-persistence validation obligation discharged

The pre-registered primary gate survives and the kill signal is false.

**Disposition:** `CONTINUUM_PERSISTENCE_VALIDATION CLOSED`.

The result is diagnostic support, not a theorem proof.

---

## E-C104-8 — Q0-B reaches terminal external-review disposition

The mathematical core is internally closed at program grade, the held-out
primary validation survives, and the only named terminal dependency is
independent SARD-G specialist review.

New roots:

```text
THEOREM_B_FULL_KAPPA_C104
6f9d5b233ec7551e25ff7c51a0e49af6de7a318837488b3174cdfcdb6b07a6ca

Q0_B_PROJECT_CLOSE_C104
5771999a8d61dfa3c933b2131e6dbdc919c69465949371cc9327a8fa564bcc8c
```

**Disposition:** `EXTERNAL-REVIEW-TRACK`.

---

## E-C104-9 — Numerical constant remains unclaimed

No value of

\[
C_*
\]

is extracted from the simulation or promoted into the theorem.

**Disposition:** `NOT-CLAIMED`.

---

**Entries:** 9  
**Homeless entries:** 0


<!-- END SOURCE: C104_E_LEDGER.md -->


## PART B — CYCLE PRE-REGISTRATION FREEZES C025–C047 (verbatim)

Each freeze was hashed before computation; adjudication lives in the corresponding Observed_Update record in q0_machine.json.


---

<!-- BEGIN SOURCE: C024_Corrections_and_Freeze.md | sha256:0df10fa670a3 | status:pre-registration-freeze -->

# C024 — Corrections of the reviewer's own record + freeze: independent estimator verification and factor localization of the r³ law
**Date:** 2026-07-10 · **Author:** the in-loop reviewer (Claude), correcting its own prior-turn statements per supersede-never-overwrite.

## Part 0 — Corrections (filed before any computation)

**E1 — SUPERSEDED: my C023-turn summary of the upper side.** I stated Lemma MODZONE gave "a rigorous O(r³) ceiling on the moderate zone … modulo (ND′)" and concluded "upper rate: proven modulo (ND′)." Wrong on both counts against the C023 record: (i) C023 proved the exact moderate-zone count DIVERGES (γ = 1.882, four rungs); the per-rung masses 2.07/7.31·r³ grow in r³ units; the r-uniform upper side is OPEN (OBL-BETA-RELEVANCE). (ii) The (ND′) dependency was my own import, not C023's. Corrected status: positivity proven (C022); per-rung envelopes rigorous at the two anchors; BOTH r-uniform directions of Θ(r³) open — upper via terminal selectivity, lower via the rate of ∫Λ·AO.

**E2 — VOID: my "Fix it"-turn evidence.** The scaling probe (ratio 6.05 vs 4.0) used the slice-at-b intensity — the estimand retired by C020 G-U2 (the needle). The "0.3σ/0.0σ centered" numbers used raw-f moments, not the gradient-conditioned (μ_t, s_t). Both void. The centering conclusion stands only via C022's C2 certificate (P_win = 1.000000, correct object). My "the anisotropic blow-up is not clean" is UNSUPPORTED-BY-EVIDENCE (undetermined), and is hereby withdrawn as a finding.

## Part 1 — Freeze: independent Λ-exact reimplementation (verification instrument)

New implementation from scratch: mp-Hermite analytic-BF covariances (dps 40), mp Schur conditioning, float64+eigh-clip MC for Hessian factors. No code shared with c019/c020/c021 instruments. Estimand definitions copied from Lemma UB0 §5 verbatim: Λ(y) = φ₂(∇f(y)=0 | 𝒥₆, with mean) · [Φ((b−μ_t)/s_t) − Φ((b−ℓ−μ_t)/s_t)] · E[W₃1_typed | 9 pins @ v*=clip(μ_t)] / E[W₂1_typed | 𝒥₆]; typing (detM>0, trM<0, detS<0, detY<0); window (b−ℓ, b); pins ((b,0,0),(b−ℓ,0,0),(v*,0,0)).

**Anchors and pass bands (recorded values of the program; deterministic unless marked MC):**
| ID | object | station (r; y) | recorded | band |
|---|---|---|---|---|
| A1 | s_t/ℓ | 0.05; (−0.063, 0.012) | 0.0676 | ±0.0015 |
| A2 | P_win | same | 1.000000 | ±1e−4 |
| A3 | s_t/ℓ | 0.025; (−0.030, 0.007) | 0.0864 | ±0.0015 |
| A4 | s_t/ℓ | 0.025; (−0.0315, 0.006) | 0.0670 | ±0.0015 |
| A5 | C1 λ_min(Σ_∇) | 0.05 y*; 0.025 y* | 1.156e−6; 8.939e−8 | rel ±2% |
| A6 | C3 λ_min(6-pin Hess 6×6) | both y* | 2.603e−10; 4.068e−12 | rel ±5% |
| A7 | C4 λ_min(9-pin Hess 9×9 @v*) | both y* | 1.948e−13; 3.095e−14 | rel ±5% |
| A8 | C5 λ_min(9-pin Gram) | both y* | 1.321e−12; 3.333e−14 | rel ±5% |
| A9 (MC) | E[W₂1_typed \| 6] | r = 0.025 | 0.00202046 ± 2e−6 | ±1% |
| A10 (MC) | Λ | 0.025; (−0.0315, 0.006) | 0.090989 | rel ±6% |

Adjudication rule: any deterministic-anchor failure ⇒ STOP, convention audit, report discrepancy as the finding; Phase 2 conclusions blocked. A9/A10 in (6%, 12%] ⇒ FAILED-AS-WRITTEN with diagnosis; > 12% ⇒ instrument-discrepancy alarm.

## Part 2 — Freeze: factor localization at the fixed rescaled station
Station ỹ* = (−0.76, 0.24) in (y − M)/r coordinates — identically the C022 r=0.05 y* and the C021 r=0.025 sanity point. Rungs r ∈ {0.05, 0.025, 0.0125} (the 0.0125 rung is NEW, labeled as such). Report per rung: φ₂, det Σ_∇∇, Mahalanobis q, (μ_t−b)/ℓ, s_t/ℓ, P_win, den = E[W₂1|6], num = E[W₃1_sad|9], num_max = E[W₃1_max|9] (the β-side object), R_H = num/den, Λ; the EXACT (Schur, no MC) 9-pin conditional mean Hessians and their dets.
**Planted gate G-D1:** den(r=0.05) = 0.00808 ± 8% — the r² law predicted from the recorded r=0.025 anchor via the mechanism |det H_M| ≈ κr·|f_ss| forced by the ∇-pin divided differences (f_tt(M) = −(r/2)f_ttt + O(r²) with f_ttt cubic-pinned). Failure kills the mechanistic account, not the data.
**Stability gate G-D2′:** per-factor log₂-slopes consistent across the two octaves to ±0.35.
All component exponents are RECORDED DISCOVERY (no committed split); indicative hypothesis H1 (recorded, not gated): R_H ~ r¹-class via the shared depletion of all three dets.

## Part 3 — Freeze: needle-sweep profiles (L2 inputs)
μ_t(y) along both axis directions through ỹ*, ±5 steps of 0.1r, three rungs; report (μ_t−b)/ℓ and the max sweep slope in ℓ-per-r units; derived ribbon width ℓ/|∇μ_t|. Recorded discovery.

## Part 4 — Deliverable
Lemma LB-RATE architecture package: the reduction, the L1/L2/L3 certificate families with computed per-rung inputs, the honest open core, and the terminal-height-law correspondence between the two frontiers (classified as a structural correspondence of estimand families, NOT an isomorphism; prior-art search pending — no network in this environment). No claim of closure.


<!-- END SOURCE: C024_Corrections_and_Freeze.md -->


---

<!-- BEGIN SOURCE: C025_Freeze.md | sha256:0f8cb3dcb337 | status:pre-registration-freeze -->

# C025 — Freeze: Restructuring the Terminal-Height Frontier (AO-uniformity ∧ β-relevance) into r-free + computable parts
**Date:** 2026-07-10. **Honest scope:** this cycle does NOT close AO-uniformity or β-relevance. It (a) registers a reduction that removes the r-uniformity from the hard core, (b) certifies the computable parts numerically at three rungs on the C024-verified independent instrument, and (c) measures, for the first time, the shared object itself — the unconditional terminal-height law ν_T.

## 1. The reduction (registered architecture; equivalence links classified)
For the arch configuration (9 pins: M max at b, S saddle at b−ℓ, y saddle at v* = μ_t ≈ b−ℓ/2), write AO-failure ⊆ {non-adjacency} ∪ {outward-branch terminal ≤ b}. Then:

**(R1) [definitional/monotone]** {terminal > b} ⊇ {the outward ascending trajectory ever reaches height b}. Ascent ⇒ the trajectory's height is ≥ v* = b − ℓ/2 at every point, including any far-zone entry. So failure of the terminal condition ⇒ the trajectory terminates at a max with value in the band (v*, b], of width ≤ ℓ/2.
**(R2) [total probability + union bound, rigorous]** P(terminal ≤ b) ≤ E[N_maxband(B_{d₀}(pair)) | 9 pins] + P(far-zone ascent from height ≥ b−ℓ/2 band-terminates), for any fixed d₀.
**(R3) [Lemma FD, proven, quantitative]** at distance ≥ d₀ = 3 the conditional field is within TV-2.2% of unconditional (10⁻⁴ by d=5); the far term is bounded by the unconditional quantity + TV slack.
**(R4) [NEW NAMED OBLIGATION — OBL-FAR-ASCENT(δ₀), r-FREE]** for one fixed δ₀ > 0: p̄(δ₀) := P(an ascending trajectory of the *unconditional* field, at a point of height b−δ₀, terminates at a max ≤ b) ≤ 1 − c. Monotonicity in the band gives: for all ℓ/2 ≤ δ₀, the far band-termination ≤ p̄(δ₀). **No r appears.** Closure route: the C022 barrier technique at fixed scale (one explicit field + C² ball + Cameron–Martin mass — no degenerating pins).
**Assembly:** AO ≥ 1 − P(non-adj) − E[N_maxband(B_{d₀})] − p̄(δ₀) − TV(d₀). The same decomposition read on the divergent off-arch moderate population gives the β-relevance suppression: off-arch window saddles fail α-qualification via the *adjacency* term (mean-field mechanism to be mapped this cycle).

## 2. Committed gates
- **G-A1 (structural, abort-on-fail):** the 9-pin conditional mean field m₉ interpolates exactly: |∇m₉| at M, S, y ≤ 1e-10 and values (b, b−ℓ, v*) to 1e-12.
- **P-A1 (arch adjacency, mean level):** on m₉, the M-ward branch of y's unstable direction is captured by M at all three rungs r ∈ {0.05, 0.025, 0.0125}, with rescaled path geometry r-stable (capture; path in rescaled units within 0.1 across rungs); the fluctuation-dominance ratio min_path[transverse well scale / σ₉] r-stable within ×1.5. Adversarial: instability across rungs ⇒ the mean-dominance route to adjacency fails and the C013-class extension is blocked — register and stop that branch.
- **P-A2 (the qualification map, recorded + one committed clause):** over a polar station grid of the moderate annulus (τ/r ∈ {0.75, 1.25, 1.75} from M, 16 angles, r = 0.025): record P_win, Λ, and the mean-field branch classification. Committed clause: the in-window (P_win > 0.5) station set is angularly localized (≤ 50% of angles at each radius). The disqualifier taxonomy off-arch is DISCOVERY.
- **P-A3 (near band-max):** the max-typed channel over the in-window stations satisfies Σ Λ_max ≤ 0.02 · Σ Λ_sad at r = 0.025.
- **P-A4 (the terminal-height law, unconditional, r-free):** committed: P(terminal ≤ b | start height b − 0.05) ≤ 0.5. Recorded: ν_T histograms and the band profile P(terminal ∈ (h₀, b]) for δ₀ ∈ {0.2, 0.1, 0.05, 0.02}, uniform-at-height and |∇f|-weighted variants. Instrument: spectral BF synthesis on a 512² torus, discrete steepest-ascent basins; vacuity check: ≥ 3000 starts per window; caveats (grid basins; passage-measure weighting) recorded in-line.

## 3. Labels committed in advance
Everything produced is architecture + certification + measurement. The rigorous closure set after this cycle, if gates pass: {C013-class fluctuation lemma for 9-pin adjacency} + {Kac–Rice band-max bound over B_{d₀}} + {OBL-FAR-ASCENT(δ₀) via fixed-scale barrier} — each named, none claimed closed. Prior-art search on terminal/basin-height laws: REQUIRED before any novelty statement; not performable here (no network); no novelty ranking is made.


<!-- END SOURCE: C025_Freeze.md -->


---

<!-- BEGIN SOURCE: C026_Freeze.md | sha256:6ac9a33c2bfd | status:pre-registration-freeze -->

# C026 — Freeze: Foundation Cycle. Exact-Series Derivation of the S1–S6 Limiting Objects in a Regularizing Frame
**Date:** 2026-07-10. **Purpose:** convert the C024 factor localization from numerically-certified to derived: exact rational leading coefficients for every S-quantity at the fixed rescaled station ỹ* = (−19/25, 6/25), via the divided-difference regularizing frame in which all conditional quantities are ANALYTIC at r = 0 (so leading Taylor coefficients are rigorously the limits, given two exact nonsingularity checks). **Not claimed:** neighborhood uniformity (coefficients are analytic in ỹ; a second-station spot-check is included; the uniform statement remains a named formalization step). No novelty claims (prior-art search unavailable).

## Method (registered)
Pins p = (f, ∇f) at M, S [+ y for the 9-frame] degenerate as r → 0. Regularize: with the monomial interpolation bases B₆ = {1, x, y, x², xy, x³} and B₉ = {1, x, y, x², xy, y², x³, x²y, xy²} (unisolvence = exact nonsingularity of the rational Vandermonde-Hermite V̂ — CHECK-1), define û = D_c⁻¹ V̂⁻¹ D_g⁻¹ p (D_g = gradient 1/r scaling, D_c = diag(r^{|a|})). Then û → (D^a f(M)/a!) and the transformed Gram G_reg(r) is a REGULAR series with G₀ = G_reg(0) an exact one-point jet Gram (nonsingularity — CHECK-2, exact rational determinant). Neumann inversion order-by-order in exact Fractions; all conditional quantities (Σ_∇∇, mg, μ_t, s_t, mean Hessians) as exact regular series. Internal regularity assertion: every G_reg entry must have nonnegative Laurent offset (structural gate G-B1). Engine tie-in gate G-B2: truncated entry series must reproduce the C024-verified mp covariances to rel 1e-8 at r ∈ {0.05, 0.0125}.

## Committed planted-truth gates (series evaluated at the three rungs vs the C024 mp-exact stored values)
For each derived quantity Q ∈ {det Σ_∇∇, q, (μ_t−b)/ℓ, s_t/ℓ, mean detM/S/Y}: |series(r)/mp(r) − 1| ≤ 1e-3 at r = 0.0125 and ≤ 1e-2 at r = 0.05 (truncation-limited). Vacuity: leading coefficients stable under order N → N+4 to 1e-12 rel.
Exact-limit bands (from the C024 three-rung trends): c₆ := lim det Σ_∇∇/r⁶ ∈ [0.0210, 0.0214]; q∞ ∈ [10.40, 10.43]; lim s_t/ℓ ∈ [0.0664, 0.0672]; lim(μ_t−b)/ℓ = −1/2 exactly at leading order; mean-det limits: c_M ∈ [7.75, 7.85], c_S ∈ [−18.05, −17.85], c_Y ∈ [−13.9, −13.7].
α+β−1: report exact vanishing order (identity-candidate); no identity claim beyond computed orders.
Limiting-law constants (Part 4): den/r² → exact-limit Gaussian law (means/cov from series) + MC ⇒ band [3.20, 3.28]; num/r⁶ band [1900, 1990]; R_H/r⁴ band [590, 615]; Λ/r limit band [3.40, 3.58].

## Gate-design standard (codified after the three C025 design-lesson-4 incidents)
(1) Every threshold must be DERIVED: from a physical scale (ℓ, r, capture radii) or from the estimator's sampling distribution (never single-instance statistics vs percent bands). (2) Deterministic claims get mp-exactness gates separate from float-deployment gates. (3) Any FAILED gate: diagnose the VARIABLE and the ESTIMATOR before the physics. This standard is registry-grade as of C026.


<!-- END SOURCE: C026_Freeze.md -->


---

<!-- BEGIN SOURCE: C027_Freeze.md | sha256:f08334432099 | status:pre-registration-freeze -->

# C027 — Freeze: Foundation II. The Coefficient Field, the Derived Intensity Constant C*_∞, the Convergence-Radius Certificate, and the num∞ Lemma
**Date:** 2026-07-10. **Purpose:** finish the foundation. Four targets: (T1) the exact-limit coefficient functions (c₆, q∞, m, s̃, c_M, c_S, c_Y, typed∞, λ)(ỹ) on a rational station grid over the moderate region — the neighborhood-uniformity item at derived-on-grid grade; (T2) the derived leading-order intensity constant C*_∞ = ∫λ(ỹ)dỹ (λ = φ₂c·P_win∞·num∞/den∞ with den∞ = 3.230979 closed-form), with quadrature and boundary systematics quantified; (T3) an explicit Neumann convergence-radius certificate r₀ for the regular frame (exact coefficient norms k ≤ N, geometric tail majorant stated as hypothesis and checked against observed decay); (T4) the num∞ concentration mini-lemma written with its exact ingredients, including the previously unchecked trM∞ < 0 condition.

## Integrand definition (registered)
Per station ỹ (rational): 6-pin block at NM=17/K=13 gives leads of detΣ (assert order 6), qnum (order 6), muN (order 9 else P_win∞=0: window escapes), s2N (order 12). m = 6·lead(muN)/lead(detΣ); s̃² = 36·lead(s2N)/lead(detΣ); P_win∞ = Φ(−m/s̃) − Φ((−1−m)/s̃). 9-pin block at NM=10/K=4 with v* = b + (r³/6)·clip(m,−1,0): leads of the three mean dets (order 2; higher order ⇒ coefficient 0). typed∞ ⇔ c_M>0 ∧ c_S<0 ∧ c_Y<0 ∧ trM∞<0 (trM∞ = the order-0 lead of mh[0]+mh[2]). num∞ = |c_M c_S c_Y|·1_typed∞. Boundary flag when any typed margin is within 10% of its local scale. Domain: x̃∈[−3,2], ỹ∈[1/5,2], step 1/5, y-symmetry doubled; M-collar |ỹ|<1/4 and S-collar |ỹ−(1,0)|<1/4 excluded (C021 ballfix context recorded: ~0.3% of C* at r=0.05); on-axis line contributes 0 (measured exact axis kill + Hessian kill, C025 map; V̂₉ collinear-degeneracy noted). Lobe refinement x̃∈[−1.6,−0.2], ỹ∈[1/10,9/10] at step 1/10 replaces its coarse cells.

## Committed gates
- **G-C1 (reduction validity):** the sweep machinery at reduced (NM,K) reproduces λ(ỹ*) = 3.4755 within 1e-3 relative.
- **G-C2 (off-ridge function identity):** at two C025-map stations (τ/r=0.75, θ=157° and τ/r=1.25, θ=157°, r=0.025), the series' finite-r components (φ₂, P_win, three mean dets) match a fresh mp station() run to rel ≤ 1e-6.
- **G-C3 (quadrature):** lobe-box integral, coarse vs refined, differs ≤ 3%.
- **G-C4 (the constant):** C*_∞ ∈ [0.80, 1.05]; recorded against C021's measured C*(0.05)=0.9728, C*(0.025)=0.946 with O(r) corrections acknowledged; no supersession claim in either direction — consistency assessment only.
- **G-C5 (domain adequacy):** outer-boundary ring mass ≤ 1% of the integral, else extend.
- **G-C6 (radius):** certificate r₀ ≥ 0.05 targeted; whatever is provable is recorded.
- Vacuity: station count with flags reported; boundary-flagged mass quantified as the systematic.

## Labels committed
T1: derived-on-grid + analyticity-in-ỹ (explicit modulus step named open). T2: derived-at-leading-order with stated systematics; the measured C* stands as the finite-r truth. T3: certified modulo the stated tail-majorant hypothesis with observed-decay verification. T4: lemma with exact ingredients; any failed ingredient reported. Scope: BF, b=6/5, fixed pair frame. No novelty claims (no prior-art access).


<!-- END SOURCE: C027_Freeze.md -->


---

<!-- BEGIN SOURCE: C028_Freeze.md | sha256:4a343a18f396 | status:pre-registration-freeze -->

# C028 — Freeze: OBL-FAR-ASCENT(δ₀) via the Fixed-Scale Ramp Barrier
**Date:** 2026-07-10. **Obligation (quoted from the C025 registered architecture, R4):** "for one fixed δ₀ > 0: p̄(δ₀) := P(an ascending trajectory of the *unconditional* field, at a point of height b−δ₀, terminates at a max ≤ b) ≤ 1 − c. Monotonicity in the band gives: for all ℓ/2 ≤ δ₀, the far band-termination ≤ p̄(δ₀). No r appears. Closure route: the C022 barrier technique at fixed scale (one explicit field + C² ball + Cameron–Martin mass — no degenerating pins)." **Target grade:** the C022-precedent grade — theorem with explicit barrier, exact CM norm, and support-theorem positivity of the ball mass — PLUS a measured value of the ball mass (labeled Measured) and an end-to-end MC validation of the deterministic implication. The composition into AO (R2/R3) is C025's registered architecture, not re-derived here.

## The statement to be proved (this cycle)
Fix δ₀ = 1/5, b = 6/5, v := b − δ₀ = 1. By stationarity condition at the origin: f(0) = v. Claim: P(the maximal gradient-ascent trajectory of f from 0 terminates at a critical point of value ≤ b | f(0) = v) ≤ 1 − c with c = e^{−‖ũ₀‖²_H/2}·P₀ > 0, where ũ₀ = u₀ − v·K(·) is the explicit barrier reduced to the conditional Cameron–Martin space and P₀ = P(‖F̃‖_{C¹(D̄)} ≤ η) > 0 for the conditioned field F̃ = f − f(0)K(·) — positivity by the Gaussian support theorem on C¹(D̄) (0 lies in the support; the ball is open).

## The deterministic ramp lemma (to be certified with explicit constants)
Let D = [0, T] × [−w, w]. If h ∈ C¹(D̄) satisfies (i) ∂₁h ≥ a > 0 on D̄, (ii) |∂₂h| ≤ κ·a on D̄ with κT < w, and (iii) h(0) = v, then the ascent trajectory of h from 0 has no critical point in D, remains in D until exiting through {x₁ = T}, and exits with height ≥ v + aT. If additionally aT ≥ δ₀ + m₀ (m₀ > 0), every field agreeing with h on D̄ has its ascent from 0 reach height ≥ b + m₀ before any possible termination; by monotone ascent the terminal value then exceeds b. Applied to f = u₀ + g with ‖g‖_{C¹(D̄)} ≤ η: conditions hold with a = a₀ − η, κa = β₀ + η, where a₀ = min_D̄ ∂₁u₀ and β₀ = max_D̄ |∂₂u₀|.

## Construction (committed form; numeric values derived in-cycle, thresholds rule-bound)
u₀ = Σᵢ cᵢ K(·−xᵢ), nodes on the ray x₂ = 0 (with flanking nodes for transverse control), coefficients solved exactly (mp) so u₀ interpolates a linear ramp v + s·x₁ at the nodes with u₀(0) = v exact. Slab parameters (T, w, s) chosen so the certified margins satisfy: η := min(a₀ − a_req, slack terms)/2 with a_req from κT < w and (a₀−η)T ≥ δ₀ + m₀. ‖ũ₀‖²_H exact by the kernel quadratic form (two routes: c′Gc and the pinned-value form).

## Committed gates
- **G-D1 (interpolation exactness):** |u₀(xᵢ) − targetᵢ| ≤ 1e-25 (mp), all nodes; u₀(0) = v to 1e-25.
- **G-D2 (certified slab bounds, rigorous grid+Lipschitz):** a₀ and β₀ certified on D̄ by a grid of spacing h_g with the rigorous fill term h_g·L₂ (L₂ = certified sup of second derivatives of u₀ via Σ|cᵢ|·sup|K^(α)|, exact He-extrema): a₀_cert = min_grid ∂₁u₀ − h_g·L₂·√2/2 > 0; β₀_cert likewise; cone κT < w and gain (a₀_cert−η)T ≥ δ₀ + m₀ must hold with the DERIVED η > 0. Any failure ⇒ redesign is a NEW freeze (no silent parameter tuning).
- **G-D3 (CM norm, two routes):** c′Gc vs the value-form agree to 1e-20 relative (mp).
- **G-D4 (end-to-end MC validation):** on ≥ 300 synthesized conditioned fields with ‖F̃‖_{C¹(D̄)} ≤ η (rejection via the C025 circulant instrument, grid-evaluated norm), the discrete ascent of u₀-shifted… i.e. of f = ũ₀ + F̃ + vK from 0 reaches height ≥ b in 100% of cases (any failure ⇒ the deterministic lemma or its certification is wrong — abort and diagnose). Recorded: measured P₀ with CI; the assembled measured c.
- **Vacuity:** fill terms strictly positive fraction of margins (≥ 25% slack); MC acceptance count ≥ 300.

## Labels committed
Theorem-grade: p̄(δ₀) < 1 with c > 0 (explicit barrier, exact CM norm, support-theorem positivity) — matching the C022 grade on record. Numeric c: c ≥ e^{−‖ũ₀‖²/2}·P₀ with P₀ Measured (MC, grid C¹-norm proxy caveat recorded). No novelty claims (no network; prior-art on ascent/barrier arguments unperformable). BF testbed, b = 6/5. The R2/R3 composition and E[N_maxband] remain C025's separate items.

## SUPERSESSION v2 (design v1 G-D2-feasibility FAILED-AS-WRITTEN: value-node spacing 0.4 ⟹ Σ|cᵢ| ≈ 110, β₀ = 0.54, fill 0.16, gain 0.145 < 0.25; recorded)
v2 construction: mixed RKHS constraints L = {u₀(0,0)=v; ∂₁u₀=s at (0,0),(0.35,0),(0.7,0); u₀(0.7,0)=v+0.7s} with s, slab (T,w) chosen at build; minimum-norm interpolant via the exact mixed Gram (cov_mp machinery). Lipschitz fill via per-order certified suprema B(α) = sup|K^(α)| (dense 1D He-scan + monotone tails), L₂ = Σ|cᵢ|·B(αᵢ+2nd-order). All other gates and rules unchanged; the corrected gain inequality a(1−κ²)T ≥ δ₀ + m₀ governs.

## SUPERSESSION v3 (v2 FAILED: near-redundant value pin at (0.7,0) ⟹ representer coefficients ±1746; recorded)
v3: constraints {u₀(0,0)=v; ∂₁u₀=s at (0,0),(0.35,0),(0.7,0)} only. Lipschitz route upgraded (strict improvement, rigorous): sup_x|D^α u₀| ≤ ‖u₀‖_H·√(cov(α,α)(0)) by the reproducing bound — √3 for ∂₁₁,∂₂₂ and 1 for ∂₁₂ — replacing the coefficient-sum bound. All gates/rules otherwise unchanged.

## SUPERSESSION v4 (v3 FAILED: cone κT=0.39 > w=0.28 at β₀=0.4227; β₀ scales with s so steepening cannot restore κ<w/T; recorded)
v4: constraints {u₀(0,0)=v; ∂₁u₀=s at (0,0),(1/4,0),(1/2,0); ∂₂u₀=0 at (1/4,±3/10)} with s=1, slab T=1/2, w=3/10. Rationale: transverse flattening pins cut β₀ directly (the operative ratio κ≈β₀/a₀ was s-invariant). Reproducing-bound Lipschitz route retained. All gate rules unchanged.


<!-- END SOURCE: C028_Freeze.md -->


---

<!-- BEGIN SOURCE: C029_Freeze.md | sha256:ddd8596f894c | status:pre-registration-freeze -->

# C029 — Freeze: Basin-Persistence Reduced via Mountain-Pass + Direct Capture Measurement
**Date:** 2026-07-10. **Honest scope:** this cycle does NOT close the basin-persistence lemma. It (a) registers a reduction of BP-failure to Λ-class objects plus named pieces, (b) measures capture directly on the true 9-pin-conditioned field for the first time, at float-safe rungs.

## Registered reduction (equivalence links classified)
Setting: arch configuration, 9 pins (M max at b; S saddle at b−ℓ; y saddle at v* ≈ b−ℓ/2), M-ward ascending branch of y. BP-failure := {the branch does not terminate at M}.
**(B1) [monotone, definitional]** the branch's height ≥ v* at every point.
**(B2) [dichotomy, rigorous]** BP-failure ⊆ {terminates at a max m′ ≠ M inside B_ρ·r(pair)} ∪ {exits B_ρ·r}.
**(B3) [mountain-pass, Established-Math conditional on a.s.-Morse regularity]** if the branch terminates at m′ ≠ M inside the ball and m′, M lie in one component of {f > v*−ε}, the pass between them is a critical point with value > v*−ε inside the ball: a **window-class saddle** (value ≤ b case — Λ-suppressed at O(r³) by the proven intensity machinery integrated over the ball) **or an above-b saddle near the pinned pair** (rigidity-suppressed; C022-class extension, NAMED). If different components: the branch's own component at level v*−ε is disjoint from M's — excluded by the connectivity of the pinned structure along the mean ridge (NAMED deterministic piece, mean-field-certified skeleton C025).
**(B4) [exit channel]** exits with height ≥ v* feed R2/R3 + C028 for terminal > b, but adjacency fails ⟹ P(exit) must itself vanish; mean-flow fans (C025: 27/27 over ±20°) say exits are fluctuation-driven — MEASURED this cycle.
**Assembly:** P(BP-fail) ≤ O(r³)[window-pass] + P(above-b saddle near pair)[named] + P(exit)[measured] + component-split[named]. The lemma's hard core is thereby the two NAMED pieces, both rigidity-class.

## Committed gates
- **G-E1 (instrument):** the conditioned-field synthesizer (9-pin exact Schur mean + patch covariance) reproduces pins on the grid: |f(pin)−val| ≤ 1e-6·ℓ and |∇f(pin)| grid-consistent; conditional variance PSD after clip with negative-mass ≤ 1e-8 of trace.
- **P-E1 (capture, committed):** at r = 0.15 and r = 0.10 (float-safe rungs; smaller rungs REQUIRE the regularized-frame instrument — named), capture fraction over ≥ 200 field-draws × M-ward launches ≥ 0.8, and non-decreasing as r decreases (adversarial: a decreasing trend ⟹ the fluctuation picture is wrong ⟹ register and stop).
- **P-E2 (recorded):** per-field counts of extra maxima with value > v* in B_ρ·r and of grid-saddle-class cells in the window band; exit fraction; diversion fraction. DISCOVERY.
- **Vacuity:** ≥ 200 accepted fields per rung; launch set ≥ 6 per field.

## Labels
Architecture + measurement. Named after this cycle: {above-b-saddle rigidity near the pair (C022-class)}, {component-connectivity along the mean ridge}, {sub-0.1 rungs via regularized-frame synthesis}. No novelty claims (no network). BF, b = 6/5.


<!-- END SOURCE: C029_Freeze.md -->


---

<!-- BEGIN SOURCE: C030_Freeze.md | sha256:ae20f4e3a239 | status:pre-registration-freeze -->

# C030 — Freeze: the Two Counting Lemmas (WP and KR-MB) + the Density-Floor Mini-Lemma
**Date:** 2026-07-10. **Purpose:** convert C029-B3's main channel and C025-R2's near term to derived grade.

## Lemma WP (window-pass count, second-order Palm)
E[N_ws(A) | 9 pins] for A ⊆ B_3: intensity λ₉ws(y′) = φ∇(0|9)·φ_f(b | 9, ∇=0)·ℓ·E[|det H|1_sad | 9, ∇=0, f=b]·(1+O(ℓ)) — the band factor extracted analytically (window width ℓ), the rest a 9-pin Schur + typed Hessian expectation at y′. Two zones: in-ball B_2.5r∖collars (expect double suppression: area r² × band ℓ) and fixed zone 0.5 ≤ |y′| ≤ 3 (expect → unconditional per Lemma FD). Deliverable: station table at rungs, zone bounds C_wp-in·r⁵-class + C_wp-far·r³.

## Lemma KR-MB (band-max count)
Same structure, max-typing, band (v*, b] width ℓ/2, over B_3. Fixed-zone core: the UNCONDITIONAL band-max density for BF, derived: at a point, ∇f ⟂ (f, H); H | f=u has mean −uI, Cov: Var(Hxx|f)=Var(Hyy|f)=2 independent, Hxy~N(0,1); ρ_mx(u) = (1/2π)·E[|det H|·1_{H≺0} | f=u]·φ(u). Assembly: E[N_maxband(B_3)|9] ≤ [near, station-measured] + ρ_mx(b)·(ℓ/2)·Area(0.5–3)·(1+TV 2.2%) + [intermediate sup × area].

## Mini-lemma DF (density floor/ceiling, shared)
v(y′) := Var(f(y′) | 9 pins, ∇f(y′)=0) at dist ≥ δ from pins: values at stations; FD limit v → 1 − 0 = 1 at separation (f ⟂ own ∇). Ceiling on φ_f(b|·) = 1/√(2πv)·e^{...} follows.

## Committed gates
- **G-F1 (unconditional density, two routes):** route (a) the conditional-Gaussian reduction + 2e6-draw MC of E[|det|1_max|f=u]; route (b) direct counting on ≥40 synthesized unconditional fields (torus L=24, N=512; synthesis gates: Var within derived band, K(1) within band), band (1.1, 1.2], density = count/(area·width). Agreement within max(10%, 3·combined se).
- **G-F2 (machinery + scaling):** at fixed stations d ∈ {1, 2, 3}: λ₉ws(r=0.05)/λ₉ws(r=0.025) ∈ [6.5, 9.5] (the ℓ ratio 8); at d = 3: λ₉ws within 3× of the unconditional saddle-band analog (FD-consistency; wide band for MC + TV).
- **G-F3 (assembly consistency):** the KR-MB near+zone assembly must be consistent with P-A3's measured 8.9e-9 relative channel and with C029's zero extra maxima (predicted P(≥1) ≤ E[N_in-ball-class] ≪ 1/500).
- Vacuity: MC draws ≥ 2e4 per station; ≥ 6 stations; fields ≥ 40 route (b).

## Labels
WP and KR-MB: derived-structure with measured constants (the Kac–Rice regularity/validity conditions and the collar exclusion argument are NAMED formalities, rigidity-class). DF: exact-at-stations. No novelty claims (no network). BF, b = 6/5.


<!-- END SOURCE: C030_Freeze.md -->


---

<!-- BEGIN SOURCE: C031_Freeze.md | sha256:e165821b1479 | status:pre-registration-freeze -->

# C031 — Freeze: the R2/R3 Integration Write-up (LB-RATE assembly)
**Date:** 2026-07-10. **Deliverable:** the integration document assembling Lemma LB-RATE from cycle artifacts, every term carrying its exact epistemic grade, every named formality listed in place, every grade claim cited to a cycle artifact by hash.

## Adjudication rules (pre-committed)
1. No grade inflation: a term's grade is the weakest link in its dependency chain; measured stays measured; derived-on-grid stays on-grid.
2. Composition gaps discovered during assembly are REGISTERED as named obligations, not papered over. Two candidate gaps identified pre-freeze and to be adjudicated: (G1) additive-TV vacuity of the AO assembly at theorem grade when c ≪ TV; (G2) entry-point selection/multiplicity over the far circle (analytic-kernel determinism makes naive Palm restarts delicate).
3. R3′ (conditional-barrier transfer) may be CLOSED this cycle iff the pre-committed gate passes: at d₀ = 5, sup over the slab stations of [|m₉|_{C¹} + ‖ũ₀‖·√(1−v_grad)] ≤ 0.1307 (the unallocated half of C028's feasible η), with station density (spacing ≤ 0.25 on the slab) + analyticity as the named on-grid formality, matching C027's grade class. Exit-height slack must also clear: 0.175 − sup|m₉| > 0.
4. Verification constants computed this cycle: (V1) Rice up-crossing count of level v* on ∂B₅ (derived: unit-variance, λ₂ = 1 isotropic BF tangential process); (V2) the R3′ station table {m₉, ∇m₉, v, v_gx, v_gy, ‖m₉‖_H} exact (mp).
5. The measured-grade assembly is reported with the honest constants; the theorem-grade assembly is reported CONDITIONAL on the named set, with the two composition registrations explicit.


<!-- END SOURCE: C031_Freeze.md -->


---

<!-- BEGIN SOURCE: C032_Freeze.md | sha256:1487c20c4518 | status:pre-registration-freeze -->

# C032 — Freeze: Lemma TC (terminal-counting supersession) + the AB counting-route adjudication
**Date:** 2026-07-10. **Provenance:** C031's G2 analysis (the selection obstacle) exposed that selection is unnecessary: R1 monotonicity gives terminal height ≥ v* = b − ℓ/2·(1+o(1)), so a failing terminal (≤ b) is a local max with value in the band [v*, b] — an ℓ/2-thin band, countable globally.

## Lemma TC (statement to be established)
Modulo R0 (a.s. Morse + flow-genericity; terminal of an ascent is a max — architecture item, already named and inherited): {outward-branch terminal ≤ b} ⊆ {terminal = M} ∪ {∃ local max of f with value ∈ [v*, b] on T_L² ∖ collar(M)}. Hence P(terminal ≤ b | 9 pins) ≤ P(sep-loop) + E[N_maxband(T² ∖ collar(M)) | 9 pins], where:
- E[N_maxband(B₅ ∖ collars) | 9] ≤ 2.1·(ℓ/2) (C030 Lemma KR-MB, B₅ extension per C031) — derived-structure + measured constants.
- E[N_maxband(T² ∖ B₅) | 9] ≤ ρ_mx(1.2)·0.49993·ℓ·(L² − π·25)·E_sup, with ρ_mx(1.2) = 0.043685 DERIVED (C030 G-F1) and E_sup = sup over the exterior of the conditional/unconditional band-max intensity ratio — measured at stations this cycle (G-H1).
- {terminal = M} = both separatrices of y flow to M (separatrix loop): a flow-topology event in the rigid zone; FOLDED into the existing named family (ridge split / connectivity), renamed **"flow-topology in the rigid zone."**
- Torus periodization of the plane BF kernel at L = 24: corrections ≤ e^{−72} at the relevant distances; recorded, negligible.

**Supersession (corrections supersede, never overwrite):** Lemma TC replaces the C025 R2/R3/R4 far assembly in the AO load path. Consequences to be recorded: OBL-FAR-COMPOSE → CLOSED-BY-SUPERSESSION (both routes moot for the assembly); C028's far-ascent theorem and C031's R3′ → STANDALONE results (correct, no longer load-bearing); Lemma FD → repositioned (exterior intensity-transfer evidence + quantitative channel); OBL-P0-FLOOR → moot for the assembly. The reconciliation note (why p̄(δ₀) was lossy: fixed δ₀ vs the ℓ/2 start height) is part of the record.

## The AB adjudication (above-b-saddle channel: counting route)
Pre-registered prediction P1: the crude count E[N_sad(f > b, horizon zone) | 9] is **O(1) in r** (rung ratio ∈ [0.5, 2], NOT ≈ 8), because the mean surface's gap below b scales as O(r)·d² while the fluctuation σ is r-free — if so, the counting route for this channel is DEAD and the channel stays in the flow-topology named family, sharpened with the quantitative evidence. Alternative outcome P2: if the count scales as r-positive-power and is small, the channel CLOSES by counting. Either outcome is adjudicated as measured; the losing route is registered.

## Gates
- **G-H1 (exterior transfer):** at stations d ∈ {5, 6, 8, 12} × 2 angles, r = 0.025 (plus d = 5 at r = 0.05): band-max intensity ratio λ₉/λ_unc ∈ [0.95, 1.05] at every station with d ≥ 5, and pull √(1−v_grad) nonincreasing in d within station tolerance. E_sup := max ratio + 0.01 margin.
- **G-H2 (AB stations):** ≥ 16 stations over d ∈ {0.2, 0.35, 0.7, 1.0, 1.3} at 2 rungs; tail quadrature over u ∈ [b, b+6σ], ≥ 5 nodes × ≥ 5k draws; report per-station λ_sad>b and the zone-integrated count; adjudicate P1 vs P2 by the rung ratio.
- **G-H3 (assembly consistency):** the TC bound must be ≪ the superseded far term 0.081 (it replaces a crude bound with a sharper one for the same event); AO assembly consistent with C021's direct AO ≈ 1; r³ scaling of TC by construction (∝ ℓ).
- Vacuity: station counts as above; the sup-of-mean check over AB stations (m₉ ≤ b, attained at M).


<!-- END SOURCE: C032_Freeze.md -->


---

<!-- BEGIN SOURCE: C033_Freeze.md | sha256:026952147494 | status:pre-registration-freeze -->

# C033 — Freeze: Foundation-Enforcement Audit of Lemma TC (AUDIT-TC)
**Date:** 2026-07-10. **Purpose:** adversarial verification of C032's load-bearing structure BEFORE C034 builds on it. The mean-rim discovery implies the 9-pin mean surface crosses the terminal band [v*, b] on rings around the pin cluster; the C030 near-zone stations (2 points) may have straddled such structure. This is the program's recurring sparse-station failure class — audit it.

## Audit items and pre-committed gates
- **AU-0 (chain integrity):** recompute sha256 of the archived C028–C032 freezes/packages; all must match the recorded values. Gate: exact match, else halt.
- **AU-1 (the mean-gradient kill):** fine radial profiles of m₉(d), |∇m₉(d)|, var_g(d) = min-eig gradient conditional variance, for d ∈ [0.05, 0.6] (step ≤ 0.025) × 3 angles, r = 0.025. Gate G-K1: at every profiled point OUTSIDE the pin collars (dist > 2r from M, S, y) where m₉ ∈ [v* − 10ℓ, b + 10ℓ] (the band ± guard), the kill exponent |∇m₉|²/(2·var_g) ≥ 30 (φ_∇ ≤ e⁻³⁰). This is the mechanism protecting TC's interior: critical points cannot form on the rigid zone away from mean-critical points.
- **AU-2 (no band-valued mean-critical points outside collars):** locate where |∇m₉| is small (< 0.02) over d ∈ [0.05, 2.2] × 8 angles: at every such point, m₉ ∉ [v* − 10ℓ, b + 10ℓ]. Expected: small-gradient loci = the rim crest (m ≈ 1.4–1.6, far above band) only. Gate G-K2: pass iff no band-valued small-gradient point outside collars.
- **AU-3 (dip-ring localization):** the mean's band-crossing at the M-side must lie INSIDE the 2r collar (predicted d_ring ≈ 0.25r). Gate G-K3: the innermost profiled point (d = 0.05 = 2r) already has m₉ ABOVE b + 10ℓ or the crossing is bracketed inside d < 2r by a dedicated intra-collar mini-profile (d ∈ [0.1r, 2r]); either way the ring is collar-interior, carried by the existing named item.
- **AU-4 (B₅ extension margin):** band-max intensity ratio λ₉/λ_unc at d ∈ {3.5, 4, 4.5} × 2 angles: gate G-K4: all ≤ 2.0 (the margin used in the C030→C031 B₅ assembly), expected 1.05–1.4 interpolating the measured 1.9 (d = 3) → 1.0 (d = 5).
- **Failure protocol:** any gate failure ⟹ TC is corrected/superseded THIS cycle before any forward construction; the audit finding is registered either way.

## If clean: MF-seed (forward load, same data)
The AU-1/AU-2 profiles ARE the input to Lemma MF (mean-flow shadowing): record the mean-flow skeleton facts they establish (sign structure of radial ∇m₉, rim crest location, M-basin geometry at mean level) as the C034 seed. No MF claims graded this cycle.


<!-- END SOURCE: C033_Freeze.md -->


---

<!-- BEGIN SOURCE: C034_Freeze.md | sha256:b665c84b343a | status:pre-registration-freeze -->

# C034 — Freeze: DEF-AUDIT (the Definitions & Derivatives Canon)
**Date:** 2026-07-10. **Purpose:** comprehension enforcement before MF construction. Deliverable: a single canonical document defining every object, term, zone, grade, and convention in current use, with (i) load-bearing definitions quoted verbatim from archived artifacts, (ii) a derivative-identity battery proving the covariance machinery implements one consistent convention complete for every derivative-level object the program uses, (iii) an ambiguity register fixing precedence wherever cycles drifted or stale numbers persist in archived documents.

## Pre-committed gates
- **G-D1 (identity battery):** B1 symmetry/stationarity relations; B2 finite-difference consistency of cov_mp across ALL derivative orders in program use (outputs |a| ≤ 2, totals |a+c| ≤ 4), tolerance 1e-10 at mp h = 1e-8; B3 closed-form BF values at coincidence (Var f = 1, Var ∂f = 1, Cov(f, ∂²ᵢᵢf) = −1, Var ∂²ᵢᵢf = 3, Cov(∂²ₓₓ, ∂²ᵧᵧ) = 1, Var ∂ₓᵧ = 1, odd-order zeros), tolerance 1e-25; B4 conditional-derivative commutation (∇ of m₉ via FD equals E[∇f | pins] from the code) at 3 test points incl. one rim and one valley point, tolerance 1e-8; B5 two-route conditional variance (schur vs direct 1 − kᵀG⁻¹k) at 2 points, tolerance 1e-20. ALL must pass or the machinery is corrected before the canon issues.
- **G-D2 (closure):** every term appearing in the canon's term index has a definition entry; zero undefined references.
- **G-D3 (ambiguity register):** every identified drift/stale-number/overloaded symbol gets an explicit precedence ruling; anything unresolvable from the archive is flagged as a GAP, never silently resolved.


<!-- END SOURCE: C034_Freeze.md -->


---

<!-- BEGIN SOURCE: C035_Freeze.md | sha256:89fad14eb8c9 | status:pre-registration-freeze -->

# C035 — Freeze: Lemma MF (mean-flow skeleton, shadowing certificates, and the positivity architecture)
**Date:** 2026-07-10. **Target:** the flow-topology named family (post-TC content: sep-loop = {outward terminal = M}; M-side escape = {M-side terminal = above-b max ≠ M}). **Key reframe (registered):** Theorem A's lower half needs inf_r AO ≥ c_AO > 0, NOT AO → 1; the intra-cluster segment (where mean tilt ~ ℓ/r is dominated by cluster-scale fluctuation — MEAN-SHADOWING FAILS THERE, stated up front) is therefore a POSITIVITY problem: exhibit a robust witness in the conditional CM space, apply openness + support theorem, transfer uniformity via the C026 regularizing frame. Candidate witness: the conditional mean m₉ itself (its flow topology is computable). Outside the cluster (d > 2r), shadowing is available on the rim side per the C033 kill map.

## Architecture (three sub-lemmas + composition)
- **MF-1 (cluster positivity):** the rescaled cluster event {M-side separatrix of y → M; outward branch exits east with height reaching > b} has conditional probability ≥ c₀ > 0 uniformly in small r. Route: witness = m₉'s own flow if its skeleton is clean (G-M1); openness margins quantified; support theorem + C026 limit convergence = the NAMED write-up.
- **MF-2 (annulus shadowing):** along the outward mean separatrix from cluster exit to its b-crossing, min q_shadow := |∇m₉|²/(2·var_g) ≥ 30 station-certified ⟹ the true flow follows the mean path's height profile up to superexp failure (on-grid class).
- **MF-3 (sep-loop closure, corollary):** shadowed outward branch exceeds b (C033: the arch-ray mean crosses b at d ≈ 0.11); monotone ascent then forbids terminal = M (value b). P(sep-loop) ≤ ε_shadow + cluster-complement.
- **Composition:** AO ≥ c₀·(1 − ε_shadow) − 2.03·r³ (TC guards the band-max channel), modulo R0 + named.

## Pre-committed gates
- **G-M1 (skeleton):** RK integration of ẋ = ∇m₉/|∇m₉| from y ± ε·e₊(H̄_y), ε = 0.2r: the minus/M-side branch enters the M-collar (dist < r/2); the plus/outward branch crosses m₉ = b on the east/rim side without entering any pin collar. Any stall at a non-pin mean-critical point is a FINDING (registered, not smoothed).
- **G-M2 (shadowing):** at ≥ 12 points along the outward path with d > 2r up to the b-crossing + 3 beyond: q_shadow ≥ 30 everywhere.
- **G-M3 (rescaled scale check):** Var(f | 9)/ℓ² at 4 cluster stations ∈ [0.05, 20] (the cluster problem is O(1)-Gaussian in rescaled units — confirming positivity, not shadowing, is the right tool there).
- **G-M4 (launch confinement seed):** mean Hessian at y: unstable eigenvector angle to the y→M axis reported; Var(H(y) | 9) computed; the induced angular scatter σ_ang and the ±20° fan tail exponent reported (seed for the write-up; no pass/fail).
- Vacuity: both branches must actually move (|∇m₉(x₀)| > 0); path lengths and endpoints logged.


<!-- END SOURCE: C035_Freeze.md -->


---

<!-- BEGIN SOURCE: C036_Freeze.md | sha256:4278a6eb6332 | status:pre-registration-freeze -->

# C036 — Freeze: Lemma MF-C (the shadowing inequality with explicit constants; rung-2 skeleton; tube certification)
**Date:** 2026-07-10. **Registered refinement of C035 (pre-committed):** C035's composed statement "P(flow-topology failure) ≤ C·(e⁻¹¹¹ + e⁻³⁹)" used pointwise stall exponents as composed exponents. The honest composition is: cone certificate over the launch segment (d_y ≤ h₀) + tube-exit bound beyond, where tube exit is DIRECTIONAL (tolerance |∇m|/2 ⟹ exponent q/4) and carries covering multiplicity N and a second-derivative remainder. This cycle computes the true composed exponent. Sufficiency floor (registered): ANY composed bound ≤ 1e-3 suffices for the assembly (Theorem A's lower half needs AO ≥ c > 0); the target is better, the floor is what matters.

## The inequality (to be established at derived-structure grade)
For the tube T around the computed mean paths (radius ρ(x), outside pin collars, beyond the handoff d_y ≥ h₀), with net spacing δ:
P(true trajectory exits T or stalls) ≤ P(launch outside cone) + N(T, δ)·sup_net exp(−q(x)/4) + P(sup_T |D²g| > ε_min/(2δ)) + P(sup_T |g| > height margin at the b-crossing),
each term explicit: cone = e⁻¹¹¹ (C035 G-M4, r-stable); the D²-term exponent = (ε_min/(2δ))²/(2·var_H,max) with var_H measured on the tube; the value term = Gaussian with v measured at the crossing. Grade: Gaussian tails + union + Taylor remainder = derived-structure with station-measured variances; tube sups over stations = the on-grid class.

## Gates
- **G-N1 (rung 2):** the r = 0.05 skeleton reproduces the topology: M-side branch captured_M, outward branch crossed_b with no collar contact.
- **G-N2 (handoff):** on both rungs, q(x) ≥ 120 at every path/flank station with d_y ≥ h₀ := 0.5r (so q/4 ≥ 30 beyond the cone's jurisdiction). Flank stations at ±ρ within factor 10 of centerline q.
- **G-N3 (rigidity of D²):** var_H := max Hessian-component conditional variance ≤ 1e-3 at all tube stations with d ≤ 0.5 (so δ = 5e-3 gives a D²-exponent ≥ (ε_min/0.01)²/2e-3 with ε_min from the profile).
- **G-N4 (value margin):** √v ≤ 5e-3 at the b-crossing ± 0.05 (margin 0.02 ≥ 4σ... adjudicate with the actual σ).
- **G-N5 (composed exponent):** assemble E_comp := min(111, min-over-net q/4 − ln N, D²-exponent − ln 2, value-exponent) and report; PASS if E_comp ≥ 25; if 7 ≤ E_comp < 25, PASS-AT-FLOOR (sufficient, stated); if < 7, the lemma fails as architected and the positivity fallback (C035 freeze) is reactivated.
- Vacuity: station counts; both rungs' paths logged; the C035 mislabel's corrected sampling (full OUTWARD-path profile this time) verified by endpoint identity.


<!-- END SOURCE: C036_Freeze.md -->


---

<!-- BEGIN SOURCE: C037_Freeze.md | sha256:a6c30777704c | status:pre-registration-freeze -->

# C037 — Freeze: the Λ-side grid→continuum closure (named item 10)
**Date:** 2026-07-10. **Deliverables:** (Q) a rigorized quadrature bound for C*_∞ from measured first/second difference moduli of the archived 730-station λ-field (midpoint-rule second-order assembly per nested-grid region), cross-checked against C027's Richardson estimate; (U) r-uniformity via a per-station finite-r sweep (r = 0.025 → 0.003125) against the C026 exact limits, establishing single-signed O(r) approach with measured slopes and the assembled uniform lower bound on C*(r) for r ≤ 0.025. Instrument validation precedes all claims.

## Gates
- **G-Q0 (instrument):** the limit evaluator reproduces λ(ỹ*) = 3.4755 and ≥ 2 archived station values to ≤ 1e-6 relative.
- **G-Q1 (quadrature):** discrete sup|∇λ| and sup|second differences| measured from the archived grid per region (with the q-kill wall region isolated); assembled second-order bound E_quad; consistency: E_quad ∈ [0.2, 5]× the Richardson residual; the rigorized statement uses max(E_quad, Richardson).
- **G-U1 (r-approach):** at ≥ 6 of 8 representative stations: λ_r − λ_∞ single-signed (positive expected), Richardson ratio (λ_{2r}−λ_r)/(λ_r−λ_{r/2}) ∈ [1.6, 2.4], slope K_station = (λ_r−λ_∞)/r stable within 30% across rungs.
- **G-U2 (uniform bound):** assembled: for 0 < r ≤ 0.025, C*(r) ≥ C*_∞ − E_quad − 0.009 (shell) + min(0, slope-term) with the slope-term's sign from G-U1; positive margin over 0.85 required.
- **Closure semantics (pre-committed):** PASS on all gates closes named item 10 INTO the on-grid class (item 9) — the Λ-side carries no separate formality beyond the single station-density class shared program-wide; the analytic-in-ỹ modulus and the r-interpolation between sampled rungs are exactly that class. FAIL on any gate: the item stays open with the failure registered.


<!-- END SOURCE: C037_Freeze.md -->


---

<!-- BEGIN SOURCE: C038_Freeze.md | sha256:2a031bf7c30c | status:pre-registration-freeze -->

# C038 — Freeze: Lemma UB-G (global interceptor counting; the upper side of Θ(r³))
**Date:** 2026-07-10. **Architecture (the TC insight applied upward):** the defect event {M's homological death partner ≠ S | 6-pin typed} decomposes as {∃ interceptor: a saddle with value ∈ (b−ℓ, b) merging M's component before S} ∪ {no interceptor, S merges M with a YOUNGER component (across-max < b), partner deferred deeper} ∪ [UB0 §5 residue — G-B0]. The first is bounded by the GLOBAL window-saddle count under the 6-pin law — no zone decomposition, superseding the C023 moderate-zone route and its divergence (which is bypassed, not resolved in place). The second is a rim-kill statement under the 6-pin law (across-S = the east sector). OBL-BETA-RELEVANCE closes-by-supersession iff all gates pass INCLUDING the taxonomy check.

## Gates
- **G-B0 (taxonomy):** the UB0 §5 failure modes are exactly the two above (plus collar-class residue). Any additional mode found in the retrieval is carried explicitly or the closure is PARTIAL.
- **G-B1 (typing floor):** P(typed | 6 pins) = P(detM>0, trM<0, detS<0) at r ∈ {0.05, 0.025} ≥ 0.05, and stable across rungs (limit-law behavior: mean Hessians O(r) → the floor is the r→0 fluctuation-law constant).
- **G-B2 (near zone, 6-pin):** window-saddle intensity stations on B₃∖collars under the 6-pin law at both rungs; zone-assembled count ≤ C·ℓ with rung ratio of the r³-normalized mass ∈ [0.5, 2] (r-FREE coefficient — the direct refutation of the moderate-zone growth).
- **G-B3 (exterior, 6-pin):** intensity ratio λ₆/λ_unc at d ∈ {5, 8} × 2 angles ∈ [0.9, 1.1]; E_sup6 := max + 0.01.
- **G-B4 (across-younger rim kill):** the 6-pin mean profile eastward from S: m₆ crosses b and rises; the across-component-max-below-b event carries exponent ≥ 30 at the profile's peak-vicinity stations (E_ay = (m₆ − b)²/(2v₆) at the best station).
- **Assembly:** 1 − q ≤ [G-B2 + collar named] + [ρ_sad(1.2)·0.9999ℓ·(L²−π·9)·E_sup6]/P_typed-handling + [e^{−E_ay}] + [G-B0 residue], with the typed conditioning handled by the floor (E[N·1_typed|6]/P(typed) bound) — target: an explicit C_UB with C_UB·r³ ≥ the measured 1 − q ≈ 0.96·r³ (consistency) and C_UB finite r-free (the theorem-shape).


<!-- END SOURCE: C038_Freeze.md -->


---

<!-- BEGIN SOURCE: C039_Freeze.md | sha256:0702fefa87b1 | status:pre-registration-freeze -->

# C039 — Freeze: the Rigidity & Formalities Document (Lemmas KR-V, COL, GRID)
**Date:** 2026-07-10. **Deliverable:** one artifact consolidating the formality classes both halves of Theorem A lean on, with each stated as a precise lemma at its honest grade, plus the cycle's computations: the conditional-gradient floor table and the near-diagonal vanishing verification at the pinned law.

## The three lemmas
- **KR-V (Kac–Rice validity under the pinned laws):** (i) a.s. C^∞ paths (BF analytic — Established-Math); (ii) Σ_∇(x) nonsingular on T²∖{pins}: the Gaussian kernel's strict positive definiteness ⟹ derivative functionals at distinct points are linearly independent ⟹ Var(a·∇f(x) | pins) = dist²_H(a·∇k_x, span(pins)) > 0 for x off the pin set (Established-Math + short derived argument); quantitative floors = the measured var_g table (this cycle, compiled from C033/C035/C036/C038 station data); (iii) a.s. nondegeneracy of critical points: Bulinskaya-class, cited (same hypothesis family as R0's finite verification).
- **COL (collar counting — closure candidate):** the exclusion-radius route is REJECTED up front (it costs O(r): P(λ_min(H_typed) ≲ r) ~ r — the wrong tool; recorded as a warning). The correct route: E[#extra critical points in a pin collar | pins] = ∫_collar λ₁(u) du with λ₁ the conditional critical-intensity at offset u from the pinned critical point; the (ND′)-chain near-diagonal law λ₂ ≤ C_nd|Δ| (C024 Bonferroni verbatim, degenerate-merger Jacobian) gives λ₁(u) ≤ C·|u| ⟹ collar count ≤ C′·r³. **Gate G-R2 verifies the vanishing law AT the pinned configuration** (the ND derivation's setting vs the 9-pin law is the transfer being tested).
- **GRID (the on-grid class, consolidated):** analytic-in-x conditional quantities on δ-nets; sup ≤ net-max + δ·(measured modulus); one lemma, one moduli table spanning C027/R3′/C033/MF/C037/C038 usages; the per-use measured moduli are the class's irreducible content — the item REMAINS as one precisely-stated lemma.

## Gates
- **G-R1 (floor table):** min conditional gradient variance per zone per law compiled from archived JSONs; all > 0; table complete for every zone any count uses.
- **G-R2 (near-diagonal vanishing at the pinned law):** λ₁(u) at u ∈ {0.2r, 0.5r, 1r, 2r} × 2 angles × 2 rungs from M under the 9-pin law (all-critical-type intensity); log-log power fit α with PASS iff α ≥ 1 (± fit tolerance 0.2) at both rungs; assembled collar count ≤ C_col·r³ with C_col reported. FAIL ⟹ COL stays named with the failure registered.
- **G-R3 (closure semantics):** all-pass ⟹ the "KR validity + collar exclusion" register item CLOSES (KR-V into Established-Math + measured floors; COL into the (ND′) chain — the (ND′) note's scope is EXTENDED and recorded); named set → 7. The collar-residual shell item (|ỹ| < 0.15, Λ-integral) is a DIFFERENT item and is untouched.


<!-- END SOURCE: C039_Freeze.md -->


---

<!-- BEGIN SOURCE: C040_Freeze.md | sha256:9a934ec9716e | status:pre-registration-freeze -->

# C040 — Freeze: Manuscript Assembly (the Theorem A Rate Master Document) + Seam Audit
**Date:** 2026-07-10. **Deliverables:** (1) the standalone master document: the two-sided rate statement with complete dependency trace — every lemma with statement, grade, cycle, hash; the canonical constants table; the named set; the consolidated supersession and corrections ledgers; scope and open problems. (2) The machine-readable dependency graph (JSON, knowledge-chain format). (3) The seam audit, gates below — assembly IS the audit.

## Gates
- **G-S1 (hash chain):** recompute sha256 for every freeze/package/lemma document C028–C039 in the archive; all must match the recorded values (extends C033 AU-0 by seven cycles).
- **G-S2 (constants coherence):** for each headline constant, grep the archive: the constant must appear consistently; retired constants (1.10, 6.583, 0.8212-class intermediates) must appear ONLY in historical/ruled contexts (C024 corrections, C034 canon, C037 precedence). Any live-context conflict = seam, registered.
- **G-S3 (dependency closure):** the built graph has no orphan dependencies: every edge terminates in a lemma node, an Established-Math citation, or a named-set item.
- **G-S4 (grade propagation):** the two-sided statement's modulo-set equals the union of named dependencies over all paths in the graph — computed from the graph, compared to the C038 statement; any silently dropped condition = seam.
- **G-S5 (supersession integrity):** every node marked superseded/standalone/retired is non-load-bearing (no edge from the theorem node reaches it except through history annotations).


<!-- END SOURCE: C040_Freeze.md -->


---

<!-- BEGIN SOURCE: C041_Freeze.md | sha256:aab4d5268e63 | status:pre-registration-freeze -->

# C041 — Freeze: Lemma SARD-G (the R0 crux closure candidate)
**Date:** 2026-07-10. **Target:** the named Gaussian–Sard step of the C015 R0 reduction. **Pre-committed grade ceiling:** CLOSURE CANDIDATE — derived skeleton + verified core identity — pending (a) the C015 interface check (G-X0), (b) a dedicated adversarial audit cycle before R0's register status changes (the TC→C033 pattern), (c) prior-art (parametric transversality for random fields is a known genre; NO novelty language).

## The architecture (five steps)
1. **Mismatch functional.** For a saddle pair (p, q) with a candidate connection: the forward separatrix of p, stopped at its first crossing of a fixed small section Σ transverse to W^s(q); mismatch = signed transverse offset on Σ. Finite-time, C¹ in Cameron–Martin perturbations (smooth dependence of ODE flows). Connection ⟺ mismatch = 0.
2. **The derivative's Riesz representative.** D_h[mismatch] = ⟨h, G⟩_H with G = ∫₀ᵀ a(t)·∇₂k(·, γ(t)) dt + [endpoint jet atoms at p] — a(t) the adjoint of the transverse variational flow. (Derived: reproducing property + variational ODE.)
3. **G ≢ 0, always.** K̂ > 0 (Bochner) ⟹ the kernel embedding is injective on compactly supported distributions of finite order; G = 0 would force a curve measure with continuous nonvanishing density (a(t) ≠ 0: the adjoint weight is an exponential, never zero) to equal a finite atomic jet distribution — impossible by support. [Established-Math + derived.]
4. **Countable-dense selection + 1-D CM Fubini.** {h_j} dense in the CM ball; every zero configuration is j-regular for some j (⟨h_j, G⟩ ≠ 0); along the h_j-line the j-regular zeros are isolated ⟹ Lebesgue-null in the CM coordinate, whose conditional law is a nondegenerate 1-D Gaussian ⟹ per-(pair, branch, j) probability 0; countable union (critical points a.s. locally finite; a.s. Morse) ⟹ P(∃ saddle-saddle connection) = 0. [Standard-technique write-up.]
5. **Interface:** step 4's conclusion must equal the C015 named step verbatim (G-X0), with the finite nondegeneracy C015 already verified mapping onto step 2's endpoint-jet structure.

## Gates
- **G-X0 (interface):** the retrieved C015 named-step statement is implied by steps 1–4 as written; any residue is registered.
- **G-X1 (core identity, three-route):** on an explicit synthetic analytic two-saddle field with a near-connection: (route A) FD of the actual perturbed separatrix offset at the section under f + εh; (route B) the variational-ODE prediction ∫ a·∇h(γ); (route C) the RKHS pairing ⟨h, G⟩ with h a kernel translate. A ≡ B ≡ C within a tolerance band that carries the quantified endpoint residual (|∇h(p)|·launch sensitivity). PASS: pairwise agreement ≤ 5% + residual accounting.
- **G-X2 (G ≠ 0 numerically):** ‖G‖²_H > 0 computed for the test configuration via kernel quadrature (vacuity: the discretized norm must exceed 10× its own discretization-error estimate).
- **Register semantics:** all-pass ⟹ R0's register entry gains "SARD-G closure candidate registered (C041); audit owed"; R0 itself remains OPEN until the audit cycle passes.


<!-- END SOURCE: C041_Freeze.md -->


---

<!-- BEGIN SOURCE: C042_Freeze.md | sha256:06588d96bd44 | status:pre-registration-freeze -->

# C042 — Freeze: AUDIT-SARDG (adversarial audit of the R0 closure candidate)
**Date:** 2026-07-10. **Template:** TC→C033. **Checklist = the C041 pre-listed items.** Verdict semantics (pre-committed): all-pass ⟹ SARD-G status "AUDITED closure candidate — plumbing write-up + prior-art remaining"; R0 stays OPEN (annotated); master v1.0 still unchanged (prior-art gates v1.1). Any structural hole ⟹ SARD-G demoted with the hole registered.

## Gates
- **G-A0 (chain):** recompute C040–C041 artifact hashes; exact match.
- **G-A1 (engineered heteroclinic):** the exact-connection field f₀ = −y²(1−2x)/2 + 3x² − 2x³ (saddles p = (0,0), q = (1,0); connection along y = 0), broken by μ·y·b(x): the two-sided defect D(μ) at the section x = 0.75 (forward interior flow from p's unstable manifold vs backward interior flow from q's stable manifold) crosses zero with |dD/dμ| > 0 (FD).
- **G-A2 (self-pairing — the decisive identity):** at μ = 0.02: assemble G (curve nodes with adjoint weights + endpoint atoms at BOTH p and q); perturb the field by ε·h_G (G evaluated as an explicit kernel-quadrature function) and verify FD[D]/ε = ‖G‖²_H within 10% (kernel-gram norm; band covers assembly discretization + FD error).
- **G-A3 (q-side atoms):** the backward-flow analog of G-X1b at q: local FD vs the analytic saddle-response formula, ≤ 5%.
- **G-A4 (structural adjudications, written):** AU-1 the injectivity re-derivation with a(t)-nonvanishing explicit (scalar transverse adjoint = exponential, never zero; the numerical weights all-positive as the instantiation); AU-2 the step-4 plumbing enumerated (chart covering by pair-persistence intervals with rational sections; degenerate-ε sweep; measurable section selection; flow measurability; per-j regularity measurability; union bookkeeping) with the adjudication: fixable-plumbing vs structural-hole, per item.


<!-- END SOURCE: C042_Freeze.md -->


---

<!-- BEGIN SOURCE: C043_Freeze.md | sha256:2902623a9a6e | status:pre-registration-freeze -->

# C043 — Freeze: Closeout (route-B substitution; L10 refinement; plumbing write-up)
**Date:** 2026-07-10. **Purpose:** finish the remaining tractable mathematics before Master v2.0.
## Gates
- **G-C1 (route-B on the audit testbed):** the variational-ODE evaluation of D_h[two-sided defect] with both endpoint-atom compositions, for an explicit kernel-translate h, vs FD of the fully perturbed defect on the C042 engineered-heteroclinic configuration (μ = 0.02): agreement ≤ 5%. PASS ⟹ the SARD-G instrument item CLOSES by adopting route-B as the canonical pairing evaluator (explicit-G quadrature demoted to illustrative, its s^{ν−1} tail law recorded in the lemma).
- **G-C2 (L10 refinement):** sup of second differences at h = 1/20 sampled at the ≥ 8 largest-curvature L10-own stations; E_L10′ = (h²/24)·sup·area; updated E_hier′ and the rigorized lower tier; PASS if lower tier ≥ 0.84 (target 0.85; report honestly either way).
- **G-C3 (plumbing):** the six-item step-4 measurability write-up issued (chart covering; degenerate-ε; section selection; flow measurability; per-j regularity; union bookkeeping) — each with its one-paragraph argument and Established-Math citations; this closes the "plumbing write-up" obligation into the SARD-G record (R0 still gated on prior-art).


<!-- END SOURCE: C043_Freeze.md -->


---

<!-- BEGIN SOURCE: C046_Freeze.md | sha256:7629190f3139 | status:pre-registration-freeze -->

# C046 — Freeze: Independent Write-Up Review of SARD-G + Prior-Art Follow-Ups
**Date:** 2026-07-12. **Two components, pre-committed before execution.**

## Component A: the write-up review (referee-mode, of Master v3.0 §8.3 AS WRITTEN)
Distinct from the C042 audit (which tested the mathematics computationally): this review reads the ARGUMENT as a hostile referee, hunting for logical gaps, unstated hypotheses, circularity, and imprecision. Pre-committed checklist:
- **RC1 (Step 1):** Is D well-defined and C¹ where claimed? Does "Connection ⟺ D = 0" hold as stated? Are the section/crossing hypotheses explicit?
- **RC2 (Step 2):** Does the Riesz representation actually hold — is D CM-differentiable with a BOUNDED derivative (does G's curve integral converge as an H-element, given the infinite time-length of a heteroclinic)? Are the atom formulas' hypotheses (simple eigenvalues) stated?
- **RC3 (Step 3):** Is the injectivity computation airtight (Paley–Wiener step, membership of G's distribution in the compactly-supported finite-order class, the support-separation argument)? What exactly does "nondegenerate connecting segment" exclude?
- **RC4 (Step 4):** Does the countable-dense/Fubini logic close — per-chart C¹, isolated j-regular zeros, measurability, the chart-covering completeness (the hardest item)?
- **Verdict semantics (pre-committed):** findings classified STRUCTURAL (a hole: SARD-G demoted, hole registered) vs PRECISION (statable fixes: incorporated into a revised §8.3 THIS cycle, then SARD-G promotes to "proven at program grade — derived + verified + audited + reviewed; external referee review remains for publication," and R0's condition entry updates accordingly in Master v3.1). No middle grade.

## Component B: prior-art follow-ups (the search's own recommendations)
- **FB1:** careful read of Lerario–Stecconi's applications companion arXiv:1906.04444 (flagged: could it extend jet-transversality to any global statement?).
- **FB2:** targeted residual queries on the two incomplete-search flags: Melnikov/separatrix-breaking for random fields (Item 1c residual); Bargmann–Fock/persistence-pairing-specific work (Item 2 residual).
- **Semantics:** any surfaced close match ⟹ the §9.2 ruling updates and the register changes; clean follow-ups ⟹ the incomplete-search flags upgrade to "confirmed by targeted follow-up (date)" — still not exhaustive, stated as such.

## Deliverables
C046_Review.md (the review, findings, verdict); Master v3.1 (v3.0 with: revised §8.3 incorporating precision findings; the R0 promotion per verdict; the follow-up results folded into §9.2; ledger updates). Gates: G-R1 = every checklist item adjudicated with reasoning shown; G-R2 = v3.1 differs from v3.0 ONLY in the pre-declared sections (§8.2/§8.3/§9.1/§9.2/§10/§12 + version header); G-R3 = chain extended.


<!-- END SOURCE: C046_Freeze.md -->


---

<!-- BEGIN SOURCE: C047_Freeze.md | sha256:a3bea42865fa | status:pre-registration-freeze -->

# C047 — Freeze: Calibrated Claim Language (the distribution-unblocking pass)
**Date:** 2026-07-12. **Deliverables:** (1) the Claim Language Annex — per prior-art item: ruling restated, the paper-ready calibrated sentences, the citation sentences, the in-text qualifications, and the FORBIDDEN phrasings that the ruling does not license; plus the global methodology-disclosure block. (2) Master v3.2 — v3.1 with exactly three anchored edits (header novelty line → annex pointer; §9.2 consequence line → annex pointer; §12 + version lines). The Annex travels WITH the master; claims live in the Annex, verification lives in the master.

## Pre-committed calibration rules (drafting is constrained by these, set before drafting)
- **CR1 (traceability):** every claim sentence carries its ruling tag [Item N: RULING] inline; zero claim sentences without a ruling source.
- **CR2 (strength ceiling):** NO-CLOSE-MATCH-FOUND licenses at most "to our knowledge, … has not previously appeared" / "to our knowledge, the first …" — never bare "novel," "the first," "unprecedented," "solves," "settles," "long-standing open problem" (nothing in the record shows the problem was ever POSED as open).
- **CR3 (nearest-neighbor rule):** every claim sentence names the closest prior work in the same breath, so the reader sees exactly what is being distinguished from what.
- **CR4 (flags in-text):** Item 8's incomplete-search flag and Item 2's confirmed-but-not-exhaustive status appear IN the claim sentences, not in a footnote; the 2026-preprint provisionality (Item 7) likewise.
- **CR5 (KNOWN = cite only):** Item 6 and every KNOWN component receives citation sentences and ZERO claim language.
- **CR6 (split rule):** KNOWN-IN-PART items state the split explicitly: cite the known part first, then the qualified claim for the unlocated part.
- **CR7 (verification disclosure):** the SARD-G presentation sentence discloses the numerical-verification and internal-audit apparatus as reproducible strengths, without substituting them for the proof.

## Gates
- **G-L1:** mechanical scan — every line beginning "CLAIM" in the Annex contains "[Item" and "to our knowledge" (or an explicit flag phrase); count of violations must be 0.
- **G-L2:** forbidden-token scan over the Annex and v3.2 — bare tokens {novel, the first, unprecedented, breakthrough, settles, long-standing} may occur ONLY inside FORBIDDEN blocks or negated contexts; scan output inspected, violations 0.
- **G-L3:** v3.2 differs from v3.1 in exactly the three pre-declared regions.


<!-- END SOURCE: C047_Freeze.md -->


## PART C — CONSOLIDATION-EVENT ENTRIES (new, 2026-07-19)

### E-CONS-1 — 488-file archive consolidated to 4 files
**Action:** the complete project archive (488 files, 50 MB) is superseded-for-use by {Q0_MASTER.md, Q0_LEDGER.md, q0_machine.json, q0_verify.py}. Every original file has a manifest row in q0_machine.json with sha256, byte count, disposition, and content home. Zero orphans (classifier gate, mirroring G-S3). C092/C093 frozen archives are untouched at their recorded hashes; this is a successor release under the standing amendment policy, not an edit.
**Disposition:** `SUPERSESSION` (release-engineering; no mathematical claim changes).

### E-CONS-2 — q0_llm_verifier_v3.py reconstructed as a DERIVED shim
**Finding:** v3 is absent from the 488-file archive while v4 imports eleven symbols from it; the v5 22/22 validation was therefore irreproducible from the archive as shipped.
**Action:** a shim was reconstructed: eight symbols re-exported verbatim from q0_llm_verifier_v2.py (which the C094 lineage table records as their validated origin), and the three v3 additions (CoverageCertificate, SelectionAwareRiskLedger, coverage_limited_rank_requirement) implemented exactly from the frozen written spec (Q0_C092_FINAL_MASTER §11.5–11.6 and the COVERAGE gate).
**Behavioral certificate:** v4 validation all_expected_checks_pass=true; v5 validation 22/22 with zero case-level divergence from the archived q0_llm_verifier_v5_validation.json; second-domain root hash reproduces the C095 frozen value 1f2c349058aa638ac7dcaf09079e3bf74007010e688ecf8f7333765a3c9eebd5 exactly.
**Grade:** `DERIVED` (reconstruction), not `CERTIFIED`. The original v3 bytes remain unlocated; if recovered, diff against the shim and append the result here.

### E-CONS-3 — Page-image bundles: image layer dropped, text retained
**Finding:** all 37 files with .pdf extension are ZIP page-image bundles (JPEG pages + extracted text + manifest), 41.6 MB of images carrying 0.59 MB of text. 17 bundles are byte-content-redundant with live .md/.py sources (mapping in the manifest); 20 carry unique text (chiefly the six-file ancestor manuscript set).
**Action:** the 20 unique texts are embedded verbatim in Q0_MASTER.md Part VI with per-bundle provenance headers; the image layers are dropped. OCR control characters (\x02 for hyphen) were normalized; this is the only textual transformation applied anywhere in the consolidation.
**Disposition:** `SUPERSESSION` (rendering layer only).

### E-CONS-4 — Legacy space-named registry fixture preserved
**Finding:** the Gate Kernel 2.0 validation case `legacy_schema_rejected` requires the file name `q0 registry.json` (with space); the archive carries it renamed q0_registry.json, causing a spurious 35/36 on re-run.
**Action:** q0_verify.py materializes the fixture under its original name during selftest; q0_machine.json records the mapping. 36/36 restored. The registry itself remains DEMONSTRATION-ONLY authority with the known R0 name collision (E-C096-22/23).

### E-CONS-5 — Hard-coded /mnt/data paths shimmed at run time
**Finding:** several pre-C096 scripts hard-code /mnt/data (the defect Gate Kernel 2.0 fixed for itself in E-C096-20).
**Action:** q0_verify.py embeds every script byte-verbatim (hashes in the manifest match the originals) and applies a documented path rewrite only in the materialized run copy. Original bytes are never modified.

### E-CONS-6 — Exact duplicates and md/json twins recorded
Five byte-identical pairs and thirteen md/json twin pairs exist in the archive; primaries carry the content, duplicates carry manifest rows with `duplicate-of`. Twin .json sides are all embedded in q0_machine.json; twin .md sides live in Q0_MASTER.md or here per their disposition. Full tables in the manifest.

**END OF LEDGER — append below this line only.**
