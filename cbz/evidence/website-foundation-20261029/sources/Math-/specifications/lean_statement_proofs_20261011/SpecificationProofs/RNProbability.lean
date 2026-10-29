import Specifications.RN

/-!
# RN: kernel-checked proofs of the abstract probability `Prop`s (N1)-(N5) (statement module
# `Specifications.RN`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem below has as its TYPE a registered `Prop` constant of `UniversalLaw.Spec.RN`
(statement packet `specifications/lean_statements_20261010`, commit pinned there). A
kernel-checked proof of such a `Prop` is evidence about that `Prop` only: it does not make the
informal Layer 0 claim (RN_COUNT_INTERFACE.md, Theorem N) accepted, reviewed or "verified",
does not change its LANDING_CLAIMS disposition (HOLD_WITH_DOMAIN, transcribed by the register),
says nothing about any count of the Gaussian model, and does not touch the statement register
(the statement module is not edited here). Helper lemmas live in the `Aux` namespace and are not
inventory items.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only (checked by
`#print axioms`, see the run receipt). No `sorry`, `native_decide`, `decide` on `ℝ`, or custom
axioms.

Covered (five `Prop`s): `HolderCountBoundN1`, `HolderConsequenceN2`,
`ConditionalMeanIdentityN3`, `UniformCounterexampleN4`, `ExponentialTailLayerCakeN5`.
-/

namespace UniversalLaw.Proofs.RNProbability

open MeasureTheory UniversalLaw.Spec.RN

namespace Aux

/-- `N · 1_E = 1_E · N` pointwise. -/
theorem mul_indicator_one_eq {Ω : Type} (Ev : Set Ω) (N : Ω → ℕ) (ω : Ω) :
    (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω = Ev.indicator (fun ω => (N ω : ℝ)) ω := by
  by_cases h : ω ∈ Ev
  · simp [h]
  · simp [h]

end Aux

/-- Proves: `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3` (register claims
`rn-count-interface`, `math.rn-count-interface`).
Method: unfold `ProbabilityTheory.cond = (P E)⁻¹ • P.restrict E`, `integral_smul_measure`,
`integral_indicator`, cancel `P(E) · P(E)⁻¹`.
Does not establish: the informal Theorem N, the bounded-conditional-mean equivalence, the lower
comparison `q ≥ c r^3`, anything about the Gaussian model, review or acceptance. -/
theorem conditionalMeanIdentityN3 : ConditionalMeanIdentityN3 := by
  intro Ω _ P _ Ev N hEv hpos _hint
  simp_rw [Aux.mul_indicator_one_eq]
  rw [integral_indicator hEv, ProbabilityTheory.cond, integral_smul_measure, smul_eq_mul,
    ENNReal.toReal_inv, ← mul_assoc, mul_inv_cancel₀ hpos.ne', one_mul]

/-- Proves: `UniversalLaw.Spec.RN.HolderCountBoundN1` (register claims `rn-count-interface`,
`math.rn-count-interface`).
Method: Mathlib's Hölder inequality `integral_mul_le_Lp_mul_Lq_of_nonneg` with the conjugate
exponent `q = p/(p-1)`, `MemLp` of `N` recovered from the integrability of `N^p`, and
`∫ 1_E^q = P(E)`.
Does not establish: the informal Theorem N, that any Gaussian count satisfies the moment
hypothesis, any rate in `r`, review or acceptance. -/
theorem holderCountBoundN1 : HolderCountBoundN1 := by
  intro Ω _ P _ Ev N p hEv hp hint
  have hpq : p.HolderConjugate (Real.conjExponent p) := Real.HolderConjugate.conjExponent hp
  have hp0 : 0 < p := by linarith
  have hq0 : 0 < Real.conjExponent p := hpq.symm.pos
  have hNmeas : AEStronglyMeasurable (fun ω => (N ω : ℝ)) P := by
    have h1 : AEStronglyMeasurable (fun ω => (N ω : ℝ) ^ p) P := hint.aestronglyMeasurable
    have h2 : (fun ω => (N ω : ℝ)) = fun ω => ((N ω : ℝ) ^ p) ^ p⁻¹ := by
      funext ω; rw [Real.rpow_rpow_inv (Nat.cast_nonneg _) hp0.ne']
    rw [h2]
    exact (Real.continuous_rpow_const (inv_nonneg.mpr hp0.le)).comp_aestronglyMeasurable h1
  have hfLp : MemLp (fun ω => (N ω : ℝ)) (ENNReal.ofReal p) P := by
    have hne : ENNReal.ofReal p ≠ 0 := by
      rw [ne_eq, ENNReal.ofReal_eq_zero]; exact not_le.mpr hp0
    have hnt : ENNReal.ofReal p ≠ ⊤ := ENNReal.ofReal_ne_top
    rw [← memLp_norm_rpow_iff hNmeas hne hnt, ENNReal.div_self hne hnt, memLp_one_iff_integrable]
    have : (fun ω => ‖(N ω : ℝ)‖ ^ (ENNReal.ofReal p).toReal) = fun ω => (N ω : ℝ) ^ p := by
      funext ω
      rw [Real.norm_of_nonneg (Nat.cast_nonneg _), ENNReal.toReal_ofReal hp0.le]
    rw [this]
    exact hint
  have hgLp : MemLp (Ev.indicator fun _ => (1 : ℝ)) (ENNReal.ofReal (Real.conjExponent p)) P :=
    memLp_indicator_const _ hEv 1 (Or.inr (measure_ne_top P Ev))
  have hfnn : 0 ≤ᵐ[P] fun ω => (N ω : ℝ) := ae_of_all _ (fun ω => Nat.cast_nonneg _)
  have hgnn : 0 ≤ᵐ[P] Ev.indicator fun _ => (1 : ℝ) :=
    ae_of_all _ (fun ω => Set.indicator_nonneg (fun _ _ => zero_le_one) ω)
  have key := integral_mul_le_Lp_mul_Lq_of_nonneg hpq hfnn hgnn hfLp hgLp
  have hind : (fun a => Ev.indicator (fun _ => (1 : ℝ)) a ^ Real.conjExponent p) =
      Ev.indicator (fun _ => (1 : ℝ)) := by
    funext a
    by_cases h : a ∈ Ev
    · simp [h]
    · simp [h, Real.zero_rpow hq0.ne']
  rw [hind, integral_indicator_const (1 : ℝ) hEv, measureReal_def, smul_eq_mul, mul_one] at key
  have h1q : 1 / Real.conjExponent p = 1 - 1 / p := by
    rw [one_div, one_div, hpq.one_sub_inv]
  rw [h1q] at key
  exact key

/-- Proves: `UniversalLaw.Spec.RN.HolderConsequenceN2` (register claims `rn-count-interface`,
`math.rn-count-interface`).
Method: `holderCountBoundN1`, then monotonicity of `x ↦ x^(1/p)` and `x ↦ x^(1-1/p)` on
`[0, ∞)` and the rpow algebra `(A r^(-β))^(1/p) (C r^3)^(1-1/p) = A^(1/p) C^(1-1/p)
r^((3(p-1)-β)/p)`.
Does not establish: the informal Theorem N, that the exponent can be rounded up to `3`, that
any Gaussian count satisfies the hypotheses, review or acceptance. -/
theorem holderConsequenceN2 : HolderConsequenceN2 := by
  intro Ω _ P _ Ev N p A β C r hEv hp hA hC hr hint hmom hprob
  have h1 := holderCountBoundN1 Ω P Ev N p hEv hp hint
  have hp0 : 0 < p := by linarith
  have h1p : 0 ≤ 1 / p := by positivity
  have h1p' : 0 ≤ 1 - 1 / p := by
    rw [sub_nonneg, div_le_one hp0]; exact hp.le
  have hI : 0 ≤ ∫ ω, (N ω : ℝ) ^ p ∂P :=
    integral_nonneg (fun ω => Real.rpow_nonneg (Nat.cast_nonneg _) p)
  have h2 : (∫ ω, (N ω : ℝ) ^ p ∂P) ^ (1 / p) ≤ (A * r ^ (-β)) ^ (1 / p) :=
    Real.rpow_le_rpow hI hmom h1p
  have h3 : (P Ev).toReal ^ (1 - 1 / p) ≤ (C * r ^ 3) ^ (1 - 1 / p) :=
    Real.rpow_le_rpow ENNReal.toReal_nonneg hprob h1p'
  have hrb : 0 ≤ r ^ (-β) := Real.rpow_nonneg hr.le _
  have hr3 : 0 ≤ r ^ 3 := pow_nonneg hr.le 3
  have hexp : r ^ (-β * (1 / p)) * r ^ ((3 : ℝ) * (1 - 1 / p)) = r ^ ((3 * (p - 1) - β) / p) := by
    rw [← Real.rpow_add hr]; congr 1; field_simp; ring
  calc ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P
      ≤ (∫ ω, (N ω : ℝ) ^ p ∂P) ^ (1 / p) * (P Ev).toReal ^ (1 - 1 / p) := h1
    _ ≤ (A * r ^ (-β)) ^ (1 / p) * (C * r ^ 3) ^ (1 - 1 / p) :=
        mul_le_mul h2 h3 (Real.rpow_nonneg ENNReal.toReal_nonneg _)
          (Real.rpow_nonneg (mul_nonneg hA hrb) _)
    _ = A ^ (1 / p) * r ^ (-β * (1 / p)) * (C ^ (1 - 1 / p) * r ^ ((3 : ℝ) * (1 - 1 / p))) := by
        rw [Real.mul_rpow hA hrb, Real.mul_rpow hC hr3, ← Real.rpow_mul hr.le,
          ← Real.rpow_natCast r 3, ← Real.rpow_mul hr.le]
        push_cast
        ring
    _ = A ^ (1 / p) * C ^ (1 - 1 / p) * r ^ ((3 * (p - 1) - β) / p) := by
        rw [← hexp]; ring

namespace Aux

/-- The uniform law of `Iic a` for `0 ≤ a ≤ 1` is `a`. -/
theorem uniformLaw_Iic {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    (uniformLaw (Set.Iic a)).toReal = a := by
  rw [uniformLaw, Measure.restrict_apply measurableSet_Iic]
  have : Set.Iic a ∩ Set.Icc 0 1 = Set.Icc 0 a := by
    ext x
    simp only [Set.mem_inter_iff, Set.mem_Iic, Set.mem_Icc]
    constructor
    · rintro ⟨h1, h2, _⟩; exact ⟨h2, h1⟩
    · rintro ⟨h1, h2⟩; exact ⟨h2, h1, h2.trans ha1⟩
  rw [this, Real.volume_Icc, sub_zero, ENNReal.toReal_ofReal ha0]

/-- `∫ N_r dU = ceil(log(1/r)) · U(E_r)`. -/
theorem integral_counterCount (r : ℝ) :
    ∫ u, counterCount r u ∂uniformLaw =
      (⌈Real.log (1 / r)⌉₊ : ℝ) * (uniformLaw (Set.Iic (r ^ 3))).toReal := by
  unfold counterCount
  rw [integral_const_mul, integral_indicator_const (1 : ℝ) measurableSet_Iic, measureReal_def,
    smul_eq_mul, mul_one]

/-- `∫ N_r^p dU = ceil(log(1/r))^p · U(E_r)` for `p ≠ 0`. -/
theorem integral_counterCount_rpow (r : ℝ) {p : ℝ} (hp : p ≠ 0) :
    ∫ u, counterCount r u ^ p ∂uniformLaw =
      (⌈Real.log (1 / r)⌉₊ : ℝ) ^ p * (uniformLaw (Set.Iic (r ^ 3))).toReal := by
  have : (fun u => counterCount r u ^ p) =
      (Set.Iic (r ^ 3)).indicator (fun _ => (⌈Real.log (1 / r)⌉₊ : ℝ) ^ p) := by
    funext u
    unfold counterCount
    by_cases h : u ∈ Set.Iic (r ^ 3)
    · simp [h]
    · simp [h, Real.zero_rpow hp]
  rw [this, integral_indicator_const _ measurableSet_Iic, measureReal_def, smul_eq_mul, mul_comm]

/-- The two displays of (N4) for a fixed `0 < r ≤ e^(-1)`. -/
theorem n4_fixed (r : ℝ) (hr : 0 < r) (hre : r ≤ Real.exp (-1)) :
    (uniformLaw (Set.Iic (r ^ 3))).toReal = r ^ 3 ∧
    (∫ u, counterCount r u ∂uniformLaw) / r ^ 3 = (⌈Real.log (1 / r)⌉₊ : ℝ) := by
  have hr3 : 0 < r ^ 3 := pow_pos hr 3
  have hr1 : r ≤ 1 := hre.trans (Real.exp_lt_one_iff.mpr (by norm_num)).le
  have hr31 : r ^ 3 ≤ 1 := pow_le_one₀ hr.le hr1
  have hmeas : (uniformLaw (Set.Iic (r ^ 3))).toReal = r ^ 3 := uniformLaw_Iic hr3.le hr31
  refine ⟨hmeas, ?_⟩
  rw [integral_counterCount, hmeas, mul_div_assoc, div_self hr3.ne', mul_one]

/-- `log (1/r) ≥ 1` for `0 < r ≤ e^(-1)`. -/
theorem one_le_log_one_div {r : ℝ} (hr : 0 < r) (hre : r ≤ Real.exp (-1)) :
    1 ≤ Real.log (1 / r) := by
  rw [Real.le_log_iff_exp_le (by positivity), le_one_div (Real.exp_pos _) hr, one_div,
    ← Real.exp_neg]
  exact hre

/-- `r^3 = exp(-3 log(1/r))` for `r > 0`. -/
theorem pow_three_eq_exp {r : ℝ} (hr : 0 < r) :
    r ^ 3 = Real.exp (-(3 * Real.log (1 / r))) := by
  have : r = Real.exp (-Real.log (1 / r)) := by
    rw [Real.exp_neg, Real.exp_log (by positivity), one_div, inv_inv]
  conv_lhs => rw [this]
  rw [← Real.exp_nat_mul]
  congr 1
  push_cast
  ring

/-- `∫_0^∞ exp(-s/B) ds = B` for `B > 0`. -/
theorem integral_exp_neg_div_Ioi_zero {B : ℝ} (hB : 0 < B) :
    ∫ s in Set.Ioi (0 : ℝ), Real.exp (-s / B) = B := by
  have h := integral_comp_mul_left_Ioi (fun x => Real.exp (-x)) 0 (inv_pos.mpr hB)
  simp only [mul_zero, inv_inv, smul_eq_mul, integral_exp_neg_Ioi_zero, mul_one] at h
  have hfun : (fun s : ℝ => Real.exp (-s / B)) = fun x => Real.exp (-(B⁻¹ * x)) := by
    funext s; congr 1; ring
  rw [hfun]
  exact h

end Aux

/-- Proves: `UniversalLaw.Spec.RN.UniformCounterexampleN4` (register claims
`rn-count-interface`, `math.rn-count-interface`).
Method: `U(E_r) = r^3` from `volume.restrict` on `Icc 0 1`; `∫ N_r dU = ceil(log(1/r)) r^3`
by `integral_indicator_const`; the limit from `log ∘ inv → ∞` on `𝓝[>] 0` and
`x ≤ ⌈x⌉₊`; the moment bound with the explicit constant `A_p = 2^n n!`, `n = ⌈p⌉₊`, via
`⌈t⌉₊^p ≤ (2t)^n` for `t = log(1/r) ≥ 1` and `t^n ≤ n! e^t`.
Does not establish: the source's sharper bound `sup_(t>=1) e^(-3t)(t+1)^p`, the one-`p`
sharpness construction, anything about the Gaussian field, the informal Theorem N, review or
acceptance. -/
theorem uniformCounterexampleN4 : UniformCounterexampleN4 := by
  refine ⟨fun r hr hre => Aux.n4_fixed r hr hre, ?_, ?_⟩
  · have hev : (fun r : ℝ => (⌈Real.log (1 / r)⌉₊ : ℝ)) =ᶠ[nhdsWithin (0 : ℝ) (Set.Ioi 0)]
        fun r => (∫ u, counterCount r u ∂uniformLaw) / r ^ 3 := by
      have h1 : ∀ᶠ r in nhdsWithin (0 : ℝ) (Set.Ioi 0), 0 < r := self_mem_nhdsWithin
      have h2 : ∀ᶠ r in nhdsWithin (0 : ℝ) (Set.Ioi 0), r < Real.exp (-1) :=
        nhdsWithin_le_nhds (Iio_mem_nhds (Real.exp_pos _))
      filter_upwards [h1, h2] with r hr hre
      exact ((Aux.n4_fixed r hr hre.le).2).symm
    refine Filter.Tendsto.congr' hev ?_
    refine Filter.tendsto_atTop_mono (fun r => Nat.le_ceil (Real.log (1 / r))) ?_
    have : (fun r : ℝ => Real.log (1 / r)) = Real.log ∘ (fun r : ℝ => r⁻¹) := by
      funext r; simp [one_div]
    rw [this]
    exact Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
  · intro p hp
    refine ⟨2 ^ ⌈p⌉₊ * ((⌈p⌉₊).factorial : ℝ), ?_⟩
    intro r hr hre
    have hp0 : p ≠ 0 := by linarith
    rw [Aux.integral_counterCount_rpow r hp0, (Aux.n4_fixed r hr hre).1]
    set n := ⌈p⌉₊ with hn
    set t := Real.log (1 / r) with ht
    have ht1 : 1 ≤ t := Aux.one_le_log_one_div hr hre
    have hc1 : (1 : ℝ) ≤ (⌈t⌉₊ : ℝ) := by
      exact_mod_cast Nat.one_le_ceil_iff.mpr (by linarith)
    have hc2 : (⌈t⌉₊ : ℝ) ≤ 2 * t := by
      have := Nat.ceil_lt_add_one (by linarith : (0 : ℝ) ≤ t); linarith
    have hr3 : r ^ 3 = Real.exp (-(3 * t)) := Aux.pow_three_eq_exp hr
    have hcp : (⌈t⌉₊ : ℝ) ^ p ≤ (⌈t⌉₊ : ℝ) ^ n := by
      rw [← Real.rpow_natCast]
      exact Real.rpow_le_rpow_of_exponent_le hc1 (Nat.le_ceil p)
    have hcn : (⌈t⌉₊ : ℝ) ^ n ≤ (2 * t) ^ n := pow_le_pow_left₀ (by linarith) hc2 n
    have hexp : t ^ n ≤ (n.factorial : ℝ) * Real.exp t := by
      have := Real.pow_div_factorial_le_exp t (by linarith) n
      rw [div_le_iff₀ (by positivity)] at this
      linarith
    have hexp2 : Real.exp (-(2 * t)) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    calc (⌈t⌉₊ : ℝ) ^ p * r ^ 3 ≤ (2 * t) ^ n * Real.exp (-(3 * t)) := by
          rw [hr3]
          exact mul_le_mul (hcp.trans hcn) le_rfl (Real.exp_pos _).le (by positivity)
      _ = 2 ^ n * (t ^ n * Real.exp (-(3 * t))) := by rw [mul_pow]; ring
      _ ≤ 2 ^ n * ((n.factorial : ℝ) * Real.exp t * Real.exp (-(3 * t))) := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          exact mul_le_mul_of_nonneg_right hexp (Real.exp_pos _).le
      _ = 2 ^ n * (n.factorial : ℝ) * Real.exp (-(2 * t)) := by
          have : Real.exp t * Real.exp (-(3 * t)) = Real.exp (-(2 * t)) := by
            rw [← Real.exp_add]; congr 1; ring
          rw [mul_assoc ((n.factorial : ℝ)), this]; ring
      _ ≤ 2 ^ n * (n.factorial : ℝ) := mul_le_of_le_one_right (by positivity) hexp2

/-- Proves: `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5` (register claims
`rn-count-interface`, `math.rn-count-interface`).
Method: with `t₀ = B log(A/q) ≥ 0`, the pointwise bound `N 1_E ≤ t₀ 1_E + (N - t₀)⁺`; the
layer-cake formula `Integrable.integral_eq_integral_meas_lt` for `(N - t₀)⁺`; the tail
hypothesis at `t = s + t₀`; and `∫_0^∞ A e^(-(s+t₀)/B) ds = A B e^(-t₀/B) = B q`
(`integral_comp_mul_left_Ioi`, `integral_exp_neg_Ioi_zero`).
Does not establish: the source's intermediate identity `∫_0^∞ min(q, A e^(-t/B)) dt`, that
the logarithmic loss is attained, anything about the Gaussian model, the informal Theorem N,
review or acceptance. -/
theorem exponentialTailLayerCakeN5 : ExponentialTailLayerCakeN5 := by
  intro Ω _ P _ Ev N A B hEv hA hB hq hint htail
  set q : ℝ := (P Ev).toReal with hqdef
  set t₀ : ℝ := B * Real.log (A / q) with ht₀
  have hq1 : q ≤ 1 := by
    rw [hqdef, ← measureReal_def]; exact measureReal_le_one
  have hAq : 1 ≤ A / q := by rw [le_div_iff₀ hq]; linarith
  have hlog : 0 ≤ Real.log (A / q) := Real.log_nonneg hAq
  have ht₀0 : 0 ≤ t₀ := mul_nonneg hB.le hlog
  -- pointwise: N 1_E ≤ t₀ 1_E + (N - t₀)⁺
  have hpt : ∀ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ≤
      Ev.indicator (fun _ => t₀) ω + max ((N ω : ℝ) - t₀) 0 := by
    intro ω
    by_cases h : ω ∈ Ev
    · simp only [Set.indicator_of_mem h, mul_one]
      have := le_max_left ((N ω : ℝ) - t₀) 0
      linarith
    · simp only [Set.indicator_of_notMem h, mul_zero, zero_add]
      exact le_max_right _ _
  have hint_t : Integrable (Ev.indicator (fun _ => t₀)) P := (integrable_const t₀).indicator hEv
  have hint_pos : Integrable (fun ω => max ((N ω : ℝ) - t₀) 0) P :=
    (hint.sub (integrable_const t₀)).pos_part
  have step1 : ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P ≤
      t₀ * q + ∫ ω, max ((N ω : ℝ) - t₀) 0 ∂P := by
    calc ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P
        ≤ ∫ ω, (Ev.indicator (fun _ => t₀) ω + max ((N ω : ℝ) - t₀) 0) ∂P := by
          apply integral_mono_of_nonneg
          · exact ae_of_all _ (fun ω => mul_nonneg (Nat.cast_nonneg _)
              (Set.indicator_nonneg (fun _ _ => zero_le_one) ω))
          · exact hint_t.add hint_pos
          · exact ae_of_all _ hpt
      _ = t₀ * q + ∫ ω, max ((N ω : ℝ) - t₀) 0 ∂P := by
          rw [integral_add hint_t hint_pos, integral_indicator_const _ hEv, measureReal_def,
            smul_eq_mul, mul_comm]
  -- layer cake for (N - t₀)⁺
  have hnn : 0 ≤ᵐ[P] fun ω => max ((N ω : ℝ) - t₀) 0 := ae_of_all _ (fun ω => le_max_right _ _)
  have hlc := hint_pos.integral_eq_integral_meas_lt hnn
  have hbound : ∀ s ∈ Set.Ioi (0 : ℝ), P.real {a | s < max ((N a : ℝ) - t₀) 0} ≤
      A * Real.exp (-t₀ / B) * Real.exp (-s / B) := by
    intro s hs
    have hs' : 0 < s := hs
    have hsub : {a | s < max ((N a : ℝ) - t₀) 0} ⊆ {ω | (s + t₀) < (N ω : ℝ)} := by
      intro a ha
      have ha' : s < max ((N a : ℝ) - t₀) 0 := ha
      show s + t₀ < (N a : ℝ)
      rcases lt_max_iff.mp ha' with h | h
      · linarith
      · exact absurd h (not_lt.mpr hs'.le)
    calc P.real {a | s < max ((N a : ℝ) - t₀) 0} ≤ P.real {ω | (s + t₀) < (N ω : ℝ)} :=
          measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ A * Real.exp (-(s + t₀) / B) := htail (s + t₀) (by linarith)
      _ = A * Real.exp (-t₀ / B) * Real.exp (-s / B) := by
          rw [mul_assoc, ← Real.exp_add]; congr 2; ring
  have hint_g : IntegrableOn (fun s : ℝ => A * Real.exp (-t₀ / B) * Real.exp (-s / B))
      (Set.Ioi 0) := by
    have h0 := exp_neg_integrableOn_Ioi 0 (inv_pos.mpr hB)
    have h2 : (fun s : ℝ => A * Real.exp (-t₀ / B) * Real.exp (-s / B)) =
        fun s => A * Real.exp (-t₀ / B) * Real.exp (-B⁻¹ * s) := by
      funext s; congr 2; ring
    rw [h2]
    exact h0.const_mul _
  have step2 : ∫ ω, max ((N ω : ℝ) - t₀) 0 ∂P ≤ B * q := by
    rw [hlc]
    have hexp : Real.exp (-t₀ / B) = q / A := by
      have h1 : -t₀ / B = -Real.log (A / q) := by rw [ht₀]; field_simp
      rw [h1, Real.exp_neg, Real.exp_log (by positivity), inv_div]
    calc ∫ s in Set.Ioi 0, P.real {a | s < max ((N a : ℝ) - t₀) 0}
        ≤ ∫ s in Set.Ioi 0, A * Real.exp (-t₀ / B) * Real.exp (-s / B) := by
          apply integral_mono_of_nonneg
          · exact ae_of_all _ (fun s => measureReal_nonneg)
          · exact hint_g
          · exact ae_restrict_of_forall_mem measurableSet_Ioi hbound
      _ = A * Real.exp (-t₀ / B) * ∫ s in Set.Ioi 0, Real.exp (-s / B) := integral_const_mul _ _
      _ = A * Real.exp (-t₀ / B) * B := by rw [Aux.integral_exp_neg_div_Ioi_zero hB]
      _ = B * q := by
          rw [hexp]
          have hA0 : A ≠ 0 := by linarith
          field_simp
  calc ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P
      ≤ t₀ * q + ∫ ω, max ((N ω : ℝ) - t₀) 0 ∂P := step1
    _ ≤ t₀ * q + B * q := by linarith [step2]
    _ = B * q * (1 + Real.log (A / q)) := by rw [ht₀]; ring

end UniversalLaw.Proofs.RNProbability
