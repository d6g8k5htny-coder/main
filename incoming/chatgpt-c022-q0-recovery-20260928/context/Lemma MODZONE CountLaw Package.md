# Lemma MODZONE-CL — the exact moderate-zone band-critical count law, the Mahalanobis ridge, and the re-scoping of the β moderate zone

**Cycle:** C023 / C023b · **Date:** 2026-07-09
**Freezes:** C023_MODZONE_snapshot.json sha256 `94e4372b2287766a9ab53124a00ef014fc338073578ede91be51f1ea73e7d089`;
C023b_Exact_snapshot.json sha256 `f0d08becb81081f367f2e5ccf56daaa610a6a2a5add5db88c956e85f72af065e`
**Obligation addressed:** OBL-BETA-MODZONE → adjudicated FAILED-AS-WRITTEN on both registered
routes (P-W1, P-W2) and **superseded** by OBL-BETA-RELEVANCE (§6). The cycle's positive residue
is an exact, four-rung-certified count law (§2–3), a structural discovery about the raw Palm law
(§4), and two archive-correction flags (§5).
**Data:** c023_modzone_r{0.05,0.025,0.0125,0.00625}.json, c023b_exact_r{…}.json,
c023b_adjudication_raw.json, c023c_mahalanobis.json.

---

## §1. The two adjudications (both branch iii, recorded without rescue)

**P-W1 (sup-decoupled sharpening).** Frozen statistic: c_sharp = φ_∇(0; no mean term) ·
sup_v(f-density) · sup_v E[|det H| | f = v, ∇ = 0, pins]. Four-rung integrals
I_sup = 272.5 / 902.6 / 4000 / 15758 at r = 0.05 / 0.025 / 0.0125 / 0.00625; successive
exponents 1.73, 2.15, 1.98. **Branch (iii): γ ≈ 1.95 ≥ 0.3 — FAILED-AS-WRITTEN.** Post-mortem
(recorded, not rescued): (a) the sup-decoupling; (b) the gradient-density factor omitted the
conditional-mean term e^{−½μ_g'Σ⁻¹μ_g} — a **valid but loose** ceiling (factor ≤ 1). The same
omission is present in the C022 Lemma-FD table; those entries remain valid ceilings (inequality
direction preserved) and the fixed-d₀ constants stand; recorded as a looseness note.

**P-W2 (exact Kac–Rice count).** Successor statistic: the exact band-critical intensity
c̄(d) = ⟨θ⟩[ φ_∇(0; **with** mean term) · (1/ℓ)∫_band φ_f(v)·E(|det H| ∣ f = v, ∇ = 0, pins) dv ],
so that ∫ c̄·2πd dd · ℓ **equals** E[# band criticals in the annulus | 6-pin Palm] (Kac–Rice
identity — no inequality except maxima ≤ all types). Refined log-spaced grids, log-trapezoid:

| r | I_exact = E[#]/ℓ over (2r, 0.3] |
|---|---|
| 0.05 | **12.43** |
| 0.025 | **43.83** |
| 0.0125 | **161.7** |
| 0.00625 | **621.9** |

Power fit **γ = 1.882, residual rms 0.022** (successive 1.818, 1.884, 1.943).
**Branch (iii) — FAILED-AS-WRITTEN for the total-count route.** This is now known to be a
property of the truth, not of a bound: the exact expected moderate-zone band-critical count in
ℓ-units diverges like r^{−1.9}.

## §2. The exact count law (positive residue; derived-and-verified)

The intensity separates into an inner self-similar profile and an outer fixed profile:

- **Inner collapse.** c̄(s·r)·r⁴ → ĉ(s) with a four-rung collapse to 0.5% at the inner edge:
  ĉ(2) = 6.74/6.63/6.61/**6.60** ×10⁻³ across the rungs; cross-rung agreement at matching s to
  3 digits (e.g. s = 2.2: 2.44×10⁻³ at both r = 0.0125 and 0.00625). ĉ(s) falls ≈ s^{−10}:
  samples (2, 6.6e−3), (2.2, 2.44e−3), (2.5, 7.44e−4).
- **Outer profile.** At fixed d, c̄ converges (r ≤ d/4) to c∞(d): 109.5 / 33.1 / 14.4 / 4.70 at
  d = 0.10 / 0.15 / 0.20 / 0.30 — local slope ≈ d^{−2.9}, c∞ ≈ 0.14·d^{−2.9}.
- **Assembly.** I_exact(r) ≈ 2π∫ĉ(s)s ds·r^{−2} + outer ≈ **0.019·r^{−2}** + O((2r)^{−0.9});
  verification at r = 0.00625: 0.019/3.9×10⁻⁵ ≈ 487 plus outer ≈ 130 gives 617 ≈ 621.9 ✓.

**Per-rung salvage (rigorous numbers at the anchor rungs).** Because I_exact is exact, the
moderate-zone β ceiling per rung is: r = 0.05: ≤ 12.43ℓ = **2.07·r³**; r = 0.025: ≤ 43.8ℓ =
**7.31·r³**. Hence the complete per-rung upper envelope at the two flagship rungs:
(1 − q)/r³ ≤ 0.96 + 12.41 (far, d₀ = 0.3) + 2.07 / 7.31 (moderate, exact) + β_near
(≤ 2.6×10⁻⁸ + 8×10⁻⁴²) + ledger ≈ **15.5 / 20.7** — rigorous at each anchor. Only the r → 0
uniformity of the moderate term is open, and §1 proves it cannot come from counting.

## §3. Why the count diverges: the Mahalanobis ridge (structural discovery)

Adversarial verification of the mechanism (c023c_mahalanobis.json), exact mp, three rungs:

- **q_min(d) := min_θ μ_g'Σ_∇∇⁻¹μ_g ≈ 0.720–0.727, constant** — rung-collapse 0.2–0.9% at
  fixed s ∈ {2, 3, 4}, flat in s, minimizer θ* ≈ 1.31–1.49 rad (a crest-line tilted slightly
  M-ward of perpendicular). The suppression factor along the crest is e^{−q/2} ≈ **0.70 per
  station** — polynomial physics; there is **no exp(−c/τ²) mechanism in the raw 6-pin Palm law**.
- **Transverse gradient sd is linear:** sd(f_⊥′ | pair) ≈ κ_g·d with κ_g ≈ 2.7/r·…, i.e.
  sd_gy(s·r) ≈ 2.72·r·g(s), doubling across rungs at fixed s and growing ~d within a rung.
  The archived τ³ transverse-gradient scaling holds only for the 2-jet-removed residual /
  on-axis; the raw law's transverse gradient freedom is first-order (the free Hessians at the
  pins rotate the gradient at rate ~d — as an analytic-kernel Taylor argument predicts).
- On-axis the archived scalings are confirmed: sd(f | pair)_axis ~ τ⁴, exit mean ~ κτ³/3,
  longitudinal sd_gx ~ τ^{2.6–3}: the on-axis physics is as File-1 §5.1–5.2 describes. The
  failure of suppression is strictly a transverse/off-axis phenomenon.

**Retro-consistency.** 45,000+ measured flows across C020/C021 produced **zero** β events
(terminal support edge 894ℓ) despite this abundant band-critical population. Conclusion: β
suppression is a **relevance/selectivity** phenomenon (the one S⁺ branch almost never selects a
band-max as its terminal), not a scarcity phenomenon. The deterministic key: the S⁺ ascending
branch is unique, hence Σ_maxima 1{branch terminates at m} = 1 always — so the correct estimand
is the **terminal-height law** P(f(T) ∈ band ∧ ρ_T ∈ zone), not a count.

## §4. What the divergent population is

The band-critical mass sits on the crest-line where the conditional mean-gradient's Mahalanobis
norm is O(1): the mean surface's level-b contour through the pair, along which the mean value
stays within O(ℓ)-to-O(d³) of b while both the value sd (perpendicular ~d²) and the transverse
gradient sd (~d) are polynomially large. A β-mod event would force the **entire branch** into
the closed ℓ-band (ascent + terminal-in-band ⟹ f ∈ (b − ℓ, b] along the whole branch), i.e. an
ascending path of length ≥ d − r/2 inside an ℓ-slab; near-critical band points are abundant on
the crest, so path-rigidity/entropy arguments fail in the inner moderate zone. For **fixed** d₀,
the path-rigidity route re-derives an O(ℓ) bound (noted; consistent with Lemma FD's fixed-d₀
ceiling); it is the (2r, d₀) inner-moderate zone that requires selectivity.

## §5. Archive-correction flags (supersede-never-overwrite; scope-precise)

1. **File-1 §5.2 (raw-law reading) and File-3 Proposition 5.2** claim the annulus band count is
   O(ℓ) via the Pinning-scaling suppression exp(−c/τ²) with sd(f_s | pair) ≤ Cτ³. The exact
   four-rung computation above **contra-indicates the raw-count reading**: the raw transverse
   gradient sd is ~d, the crest Mahalanobis is O(1), and E[N_ann | pair] = Θ(r^{−1.9})·ℓ.
   The July-5 status "PROVEN-MODULO: B1 chaining" for the annulus/loop suppression is
   **downgraded to OPEN-CONTRA-INDICATED for the raw-count reading**. Scope caveat, recorded
   honestly: the File-1/File-3 route conditions differently (midpoint 2-jet removed and chained
   separately), so this is an archive **correction/scope flag**, not necessarily a disproof of
   the chained version; but no chaining can rescue a statement about the raw expectation
   E[N_ann | pair] itself, which is now exactly computed and divergent in ℓ-units.
2. **Lemma B1's circle-band mechanism** (loop absorption via exp(−c/r²) on the 3r-circle): the
   perpendicular stations of the circle are only polynomially suppressed under the raw law
   (mean deficit ~ μd²/2 against sd ~ d²: ratio O(1)). B1's conclusion may survive via the
   multi-station product (many near-independent O(ℓ/r²)-cheap stations), but its stated
   mechanism is flagged; carried as a scoped audit item inside OBL-BETA-RELEVANCE.

## §6. The superseding obligation

**OBL-BETA-RELEVANCE** (successor of OBL-BETA-MODZONE, registered): prove
P(f(T) ∈ (b − ℓ, b] ∧ ρ_T ∈ (2r, d₀)) ≤ C·r³ — equivalently an O(ℓ) bound on the *S-adjacent*
band-maximum intensity — by a selectivity mechanism. Candidate mechanisms recorded at freeze
of the successor (not adjudicated here): (i) slab-traversal geometry: the branch must ascend a
length-(d − r/2) path inside the ℓ-slab whose transverse ribbon width is ~ℓ/|∇m_⊥| ~ ℓ/(κ_g d);
(ii) crest-decorrelation: the crest-line's field decorrelates along its length at the kernel
scale, making the joint in-slab event multiplicatively costly; (iii) the exact terminal-height
law via the AO-machinery of Lemma LB-ARCH run in reverse (upper barrier). The measured truth
this must explain: terminal support edge = 894ℓ (C021, 20k+ flows/rung).

## §7. Labels

§1: adjudications, FAILED-AS-WRITTEN ×2, recorded per protocol. §2: derived-and-verified
(exact KR identity + deterministic mp arithmetic + four-rung collapse). §3: derived-and-verified
(three-rung collapse 0.2–0.9%). §4: derived (mechanism narrative on top of §2–3 facts).
§5: adjudicated contra-indication with explicit scope caveats. §6: registered obligation.
