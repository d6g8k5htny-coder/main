import Specifications.Side24

/-!
# SIDE24 interface proofs: the `m = 1` cone moment identity, the trace moments and the law of
`R^2 = x^2 + y^2` over `ConeMomentInterface` (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here), applied to its own parameter `I : ConeMomentInterface`. A kernel-checked
proof of such a `Prop` is evidence about that `Prop` only: it does not make the informal Layer 0
note (SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) reviewed, accepted
or "verified", it does not identify the formal definitions with the informal objects (that
alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`, and it
promotes, closes, discharges or reclassifies nothing. Nothing here constructs an inhabitant of
`ConeMomentInterface` or derives its law fields from any field; the theorems hold for every
inhabitant and use only the law fields `A_law`, `sxy_law` and the instance `isProb`.

Provenance: the three `Prop`s proved here were selected by the round-4 triage (workflow
`wf_a3531fd8-891`) of the registered `Prop`s still without a theorem; each proof was written and
verified in its own lane and the lane files were merged into this module, with the helpers the
lanes proved twice removed (one copy kept, call sites repointed) and the lanes' tags kept. Helper
lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux`, tagged `c1_`, `tm_`, `rl_`; the gate
inventories only the three theorems outside that namespace. The mathematics:

* The common step: a map whose image measure under a probability measure is a non-degenerate
  Gaussian (or a product of them) is a.e.-measurable, because at this Mathlib revision the image
  measure of a non-a.e.-measurable map under a nonzero measure is a Dirac measure
  (`Measure.map_of_not_aemeasurable_of_ne_zero`), which charges a singleton, while the Gaussian
  law has `NullSingletonClass` (`c1_aemeasurable_of_map_eq_gaussian` for `A`,
  `tm_aemeasurable_triple` for `(s, x, y)`). Integrals are then transported along the laws by
  `integral_map`, and probabilities by `Measure.map_apply_of_aemeasurable`.
* `ConeMomentD1Identity I` (`c1_`): the negative half of the second moment of
  `gaussianReal 0 (8/3)` is `(8/3)/2 = 4/3` (`gaussianReal_map_neg`, `integral_add_compl`,
  `integral_Ici_eq_integral_Ioi`, `variance_fun_id_gaussianReal`).
* `TraceMoments I` (`tm_`): the first marginal of `sxy_law` is `gaussianReal 0 (5/3)`
  (`Measure.map_fst_prod`); `∫ t^2 = 5/3`, `∫ t^4 = 3 (5/3)^2 = 25/3` (`iteratedDeriv_mgf_zero`
  with the explicit fourth derivative of `exp (c t^2/2)` at `0`), `∫ exp(-t^2/2) = √(3/8)`
  (`integral_gaussianReal_eq_integral_smul`, `integral_gaussian`).
* `RayleighSquareLaw I` (`rl_`): the second marginal of `sxy_law` is
  `gaussianReal 0 1 ⊗ gaussianReal 0 1` (`Measure.map_snd_prod`); the measure of the closed disk of
  radius `√a` is computed by hand (Mathlib has no Rayleigh or chi-squared law): `prod_withDensity`,
  `withDensity_apply`, `ofReal_integral_eq_lintegral_ofReal`, polar coordinates
  (`integral_comp_polarCoord_symm`, `setIntegral_prod_mul`), the angular integral `2π` and the
  radial integral by the fundamental theorem of calculus; the right-hand side
  `∫_0^a exp(-z/2)/2 dz = 1 - exp(-a/2)` is the same elementary integral.

Remark recorded by the lanes, for the maintainers (not edited here): the proof-packet README row
"Side24 interface statements" gives as its informal reason that the Gaussian moment identities
"need the conditional law `A_u | V_u = 0` built in Mathlib's Gaussian measure theory"; for the three
`Prop`s proved here the interface's law fields suffice. This is a remark about an informal reason,
not about any claim; no register label changes.
-/

namespace UniversalLaw.Proofs.Side24Interfaces

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

/-! ## `c1_`: the `m = 1` cone moment identity (`ConeMomentD1Identity`) -/

/-! ### (a) A map with a non-degenerate Gaussian law is a.e.-measurable -/

/-- If `Measure.map A P` is the Gaussian `gaussianReal 0 v` with `v ≠ 0`, then `A` is
a.e.-measurable: otherwise the image measure would be a Dirac measure
(`Measure.map_of_not_aemeasurable_of_ne_zero`), giving a singleton of measure one, whereas the
Gaussian charges no singleton (`nullSingletonClass_gaussianReal`). -/
theorem c1_aemeasurable_of_map_eq_gaussian {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → ℝ) {v : NNReal} (hv : v ≠ 0)
    (h : Measure.map A P = gaussianReal 0 v) : AEMeasurable A P := by
  by_contra hA
  obtain ⟨c, hc⟩ : ∃ c : ℝ, Measure.map A P = Measure.dirac c :=
    ⟨_, Measure.map_of_not_aemeasurable_of_ne_zero hA (IsProbabilityMeasure.ne_zero P)⟩
  rw [hc] at h
  have : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have h1 : Measure.dirac c {c} = gaussianReal 0 v {c} := by rw [h]
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton c), measure_singleton] at h1
  exact one_ne_zero h1

/-! ### (b) Second moments of the centred Gaussian -/

/-- The integrand `a ↦ 1_{Iio 0}(a) a^2` is measurable. -/
theorem c1_measurable_indicator_sq :
    Measurable (fun a : ℝ => Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a) :=
  (measurable_id.pow_const 2).indicator measurableSet_Iio

/-- The second moment of `gaussianReal 0 v` is `v`, from Mathlib's
`variance_fun_id_gaussianReal` rewritten with `variance_eq_integral` (the mean is `0`). -/
theorem c1_second_moment_gaussianReal (v : NNReal) :
    ∫ t, t ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral (by fun_prop)] at h
  simpa using h

/-- `a ↦ a^2` is integrable against `gaussianReal 0 v` (`memLp_id_gaussianReal` at `p = 2`). -/
theorem c1_integrable_sq_gaussianReal (v : NNReal) :
    Integrable (fun a : ℝ => a ^ 2) (gaussianReal 0 v) := by
  have := (memLp_id_gaussianReal (μ := 0) (v := v) 2).integrable_sq
  simpa using this

/-- Symmetry of the centred Gaussian: the second moment over `Iio 0` equals the second moment
over `Ioi 0`, by `gaussianReal_map_neg` and a change of variables `a ↦ -a` (`integral_map`),
using `neg_lt_zero` and `neg_sq` pointwise. -/
theorem c1_setIntegral_sq_Iio_eq_Ioi (v : NNReal) :
    ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) =
      ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) := by
  set γ := gaussianReal 0 v with hγ
  have hneg : γ.map (fun x => -x) = γ := by rw [hγ, gaussianReal_map_neg, neg_zero]
  calc ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂γ
      = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂γ :=
        (integral_indicator measurableSet_Iio).symm
    _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(γ.map (fun x => -x)) := by
        rw [hneg]
    _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (-a) ∂γ :=
        integral_map measurable_neg.aemeasurable c1_measurable_indicator_sq.aestronglyMeasurable
    _ = ∫ a, Set.indicator (Set.Ioi (0 : ℝ)) (fun a => a ^ 2) a ∂γ := by
        congr 1; funext a
        simp only [Set.indicator, Set.mem_Iio, Set.mem_Ioi, neg_lt_zero, neg_sq]
    _ = ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂γ := integral_indicator measurableSet_Ioi

/-- The negative half of the second moment of `gaussianReal 0 v`, `v ≠ 0`, is `v / 2`:
`integral_add_compl` over `Iio 0` and its complement `Ici 0`, `integral_Ici_eq_integral_Ioi`
(the Gaussian charges no singleton), the symmetry lemma, and the full second moment `v`. -/
theorem c1_setIntegral_sq_Iio (v : NNReal) (hv : v ≠ 0) :
    ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) / 2 := by
  have : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have hsplit := integral_add_compl (μ := gaussianReal 0 v) (s := Set.Iio (0 : ℝ))
    measurableSet_Iio (c1_integrable_sq_gaussianReal v)
  rw [Set.compl_Iio, integral_Ici_eq_integral_Ioi, ← c1_setIntegral_sq_Iio_eq_Ioi,
    c1_second_moment_gaussianReal] at hsplit
  linarith

/-! ### (c) Transport along the law of `A` -/

/-- For a map `A` with law `gaussianReal 0 (8/3)` under a probability measure `P`, the integral
of `1_{Iio 0}(A) A^2` equals the set integral of `a ↦ a^2` over `Iio 0` against
`gaussianReal 0 (8/3)`: a.e.-measurability of `A` from its law, then `integral_map` and
`integral_indicator`. Stated for an abstract `(Ω, P, A)` so that the interface's own instance
fields `I.mΩ`, `I.isProb` are supplied explicitly at the point of use. -/
theorem c1_integral_indicator_eq {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → ℝ)
    (hlaw : Measure.map A P = gaussianReal 0 (8 / 3 : NNReal)) :
    ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (A ω) ∂P =
      ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 (8 / 3 : NNReal)) := by
  have hv : ((8 : NNReal) / 3) ≠ 0 := by norm_num
  have hA : AEMeasurable A P := c1_aemeasurable_of_map_eq_gaussian P A hv hlaw
  have e : ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (A ω) ∂P =
      ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(Measure.map A P) :=
    (integral_map hA c1_measurable_indicator_sq.aestronglyMeasurable).symm
  rw [e, hlaw, integral_indicator measurableSet_Iio]

/-! ## `tm_`: the trace moments (`TraceMoments`) -/

/-! ### (a) Calculus: the fourth derivative of `exp (c t^2 / 2)` at `0` -/

/-- Derivative of `t ↦ exp (c t^2 / 2)`: `c t exp (c t^2 / 2)` (chain rule through
`HasDerivAt.exp`). -/
theorem tm_hasDerivAt_exp_sq (c t : ℝ) :
    HasDerivAt (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) (c * t * Real.exp (c * t ^ 2 / 2)) t := by
  have h1 : HasDerivAt (fun t : ℝ => c * t ^ 2 / 2)
      (c * (((2 : ℕ) : ℝ) * t ^ (2 - 1) * 1) / 2) t :=
    (((hasDerivAt_id' t).fun_pow 2).const_mul c).div_const 2
  have h2 := h1.exp
  convert h2 using 1
  push_cast
  ring

/-- The fourth derivative of `t ↦ exp (c t^2 / 2)` at `0` is `3 c^2`: four explicit
`HasDerivAt` steps, chained through `iteratedDeriv_succ` and `iteratedDeriv_one`. -/
theorem tm_iteratedDeriv_four (c : ℝ) :
    iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) 0 = 3 * c ^ 2 := by
  have d1 : deriv (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      fun t => c * t * Real.exp (c * t ^ 2 / 2) := by
    funext t
    exact (tm_hasDerivAt_exp_sq c t).deriv
  have d2 : deriv (fun t : ℝ => c * t * Real.exp (c * t ^ 2 / 2)) =
      fun t => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have h := ((hasDerivAt_id' t).const_mul c).mul (tm_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => c * t * Real.exp (c * t ^ 2 / 2)) _ t := h
    rw [h'.deriv]
    ring
  have d3 : deriv (fun t : ℝ => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2)) =
      fun t => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have h := ((((hasDerivAt_id' t).fun_pow 2).const_mul (c ^ 2)).const_add c).mul
      (tm_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2)) _ t := h
    rw [h'.deriv]
    push_cast
    ring
  have d4 : deriv (fun t : ℝ => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2)) =
      fun t => (3 * c ^ 2 + 6 * c ^ 3 * t ^ 2 + c ^ 4 * t ^ 4) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have ha := ((hasDerivAt_id' t).const_mul (3 * c ^ 2)).add
      (((hasDerivAt_id' t).fun_pow 3).const_mul (c ^ 3))
    have h := ha.mul (tm_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2))
      _ t := h
    rw [h'.deriv]
    simp only [Pi.add_apply]
    push_cast
    ring
  have e4 : iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 3 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  have e3 : iteratedDeriv 3 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 2 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  have e2 : iteratedDeriv 2 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 1 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  rw [e4, e3, e2, iteratedDeriv_one, d1, d2, d3, d4]
  simp

/-! ### (b) Moments of the centred real Gaussian `gaussianReal 0 v`

The second moment `∫ t^2 = v` is `c1_second_moment_gaussianReal` above (the `tm_` copy was
removed at integration). -/

/-- Fourth moment of `N(0, v)`: `∫ t^4 = 3 v^2`, from the fourth derivative at `0` of the
moment generating function `t ↦ exp (v t^2 / 2)` (`iteratedDeriv_mgf_zero`,
`mgf_fun_id_gaussianReal`, `tm_iteratedDeriv_four`). -/
theorem tm_fourth_moment (v : NNReal) :
    ∫ t, t ^ 4 ∂(gaussianReal 0 v) = 3 * (v : ℝ) ^ 2 := by
  have h := iteratedDeriv_mgf_zero (X := fun x : ℝ => x) (μ := gaussianReal 0 v) (by simp) 4
  rw [mgf_fun_id_gaussianReal] at h
  simp only [zero_mul, zero_add] at h
  rw [tm_iteratedDeriv_four] at h
  simpa using h.symm

/-- Exponential moment of `N(0, 5/3)`: `∫ exp (-t^2/2) = sqrt (3/8)`. The density
`(sqrt (10 π / 3))⁻¹ exp (-3 t^2 / 10)` times `exp (-t^2/2)` is
`(sqrt (10 π / 3))⁻¹ exp (-(4/5) t^2)`, whose Lebesgue integral is `sqrt (π / (4/5))`
(`integral_gaussian`); the square roots combine to `sqrt (3/8)`. -/
theorem tm_exp_moment :
    ∫ t, Real.exp (-(t ^ 2) / 2) ∂(gaussianReal 0 (5 / 3 : NNReal)) = Real.sqrt (3 / 8) := by
  rw [integral_gaussianReal_eq_integral_smul (by norm_num)]
  have hpt : ∀ x : ℝ, gaussianPDFReal 0 (5 / 3 : NNReal) x • Real.exp (-(x ^ 2) / 2) =
      (Real.sqrt (10 * Real.pi / 3))⁻¹ * Real.exp (-(4 / 5) * x ^ 2) := by
    intro x
    rw [gaussianPDFReal_def, smul_eq_mul, mul_assoc, ← Real.exp_add]
    push_cast
    congr 2
    · congr 1
      ring
    · ring
  simp_rw [hpt]
  rw [integral_const_mul, integral_gaussian, ← Real.sqrt_inv, ← Real.sqrt_mul (by positivity)]
  congr 1
  field_simp
  ring

/-! ### (c) The marginal law of `s` from the interface's product law -/

/-- The triple `(s, x, y)` is a.e.-measurable: its image measure under `P` is the product
Gaussian law (`sxy_law`); were the triple not a.e.-measurable, `Measure.map` would be a Dirac
mass (`Measure.map_of_not_aemeasurable_of_ne_zero`), which gives a singleton measure `1`,
whereas the product of non-degenerate Gaussians has `NullSingletonClass`. -/
theorem tm_aemeasurable_triple (I : ConeMomentInterface) :
    @AEMeasurable _ _ _ I.mΩ (fun ω => (I.s ω, I.x ω, I.y ω)) I.P := by
  let := I.mΩ
  have := I.isProb
  have hv : ((5 : NNReal) / 3) ≠ 0 := by norm_num
  have : NullSingletonClass (gaussianReal 0 (5 / 3 : NNReal)) := nullSingletonClass_gaussianReal hv
  by_contra hT
  obtain ⟨c, hc⟩ : ∃ c : ℝ × ℝ × ℝ,
      Measure.map (fun ω => (I.s ω, I.x ω, I.y ω)) I.P = Measure.dirac c :=
    ⟨_, Measure.map_of_not_aemeasurable_of_ne_zero hT (IsProbabilityMeasure.ne_zero I.P)⟩
  have h := I.sxy_law
  rw [hc] at h
  have h1 : Measure.dirac c {c} =
      ((gaussianReal 0 (5 / 3)).prod ((gaussianReal 0 1).prod (gaussianReal 0 1))) {c} := by
    rw [h]
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton c), measure_singleton] at h1
  exact one_ne_zero h1

/-- `s` is a.e.-measurable, as the first component of the a.e.-measurable triple. -/
theorem tm_aemeasurable_s (I : ConeMomentInterface) : @AEMeasurable _ _ _ I.mΩ I.s I.P :=
  (tm_aemeasurable_triple I).fst

/-- The law of `s` is `N(0, 5/3)`: the first marginal of the product law `sxy_law`
(`AEMeasurable.map_map_of_aemeasurable`, `Measure.map_fst_prod`; the composite
`Prod.fst ∘ (s, x, y)` is `s`). -/
theorem tm_law_s (I : ConeMomentInterface) :
    @Measure.map _ _ I.mΩ _ I.s I.P = gaussianReal 0 (5 / 3 : NNReal) := by
  let := I.mΩ
  have := I.isProb
  have hT := tm_aemeasurable_triple I
  have h := AEMeasurable.map_map_of_aemeasurable (measurable_fst.aemeasurable) hT
  rw [I.sxy_law, Measure.map_fst_prod, measure_univ, one_smul] at h
  exact h.symm

/-! ### (d) Transport of an integral along `s` to the marginal law -/

/-- For a measurable `f : ℝ → ℝ`, `∫ f (s ω) dP = ∫ f t d(gaussianReal 0 (5/3))`
(`integral_map` along the a.e.-measurable `s`, then `tm_law_s`). -/
theorem tm_integral_comp_s (I : ConeMomentInterface) (f : ℝ → ℝ) (hf : Measurable f) :
    ∫ ω, f (I.s ω) ∂I.P = ∫ t, f t ∂(gaussianReal 0 (5 / 3 : NNReal)) := by
  let := I.mΩ
  have := I.isProb
  rw [← tm_law_s I]
  exact (integral_map (tm_aemeasurable_s I) hf.aestronglyMeasurable).symm

/-! ## `rl_`: the law of `R^2` (`RayleighSquareLaw`) -/

/-! ### (a) Elementary integrals -/

/-- The right-hand side of `RayleighSquareLaw`: `∫_0^a exp (-z/2)/2 dz = 1 - exp (-a/2)`, by
the fundamental theorem of calculus with antiderivative `z ↦ -exp (-z/2)`. -/
theorem rl_rhs_integral (a : ℝ) :
    ∫ z in (0 : ℝ)..a, Real.exp (-z / 2) / 2 = 1 - Real.exp (-a / 2) := by
  have hderiv : ∀ z ∈ Set.uIcc (0 : ℝ) a,
      HasDerivAt (fun z : ℝ => -Real.exp (-z / 2)) (Real.exp (-z / 2) / 2) z := by
    intro z _
    have h1 : HasDerivAt (fun z : ℝ => -z / 2) (-1 / 2) z :=
      ((hasDerivAt_id' z).neg).div_const 2
    have h2 := h1.exp.neg
    convert h2 using 1
    ring
  have hint : IntervalIntegrable (fun z : ℝ => Real.exp (-z / 2) / 2) volume 0 a :=
    (by fun_prop : Continuous fun z : ℝ => Real.exp (-z / 2) / 2).intervalIntegrable 0 a
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint]
  simp
  ring

/-- The radial integral of the planar Gaussian density over the disk of radius `sqrt a`:
`∫_0^{sqrt a} r (2π)⁻¹ exp (-r^2/2) dr = (2π)⁻¹ (1 - exp (-a/2))`, by the fundamental
theorem of calculus with antiderivative `r ↦ -(2π)⁻¹ exp (-r^2/2)`. -/
theorem rl_radial_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ r in (0 : ℝ)..Real.sqrt a, r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)) =
      (2 * Real.pi)⁻¹ * (1 - Real.exp (-a / 2)) := by
  have hderiv : ∀ r ∈ Set.uIcc (0 : ℝ) (Real.sqrt a),
      HasDerivAt (fun r : ℝ => -((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))
        (r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) r := by
    intro r _
    have h1 : HasDerivAt (fun r : ℝ => -(r ^ 2) / 2)
        (-(((2 : ℕ) : ℝ) * r ^ (2 - 1) * 1) / 2) r :=
      (((hasDerivAt_id' r).fun_pow 2).neg).div_const 2
    have h2 := (h1.exp.const_mul ((2 * Real.pi)⁻¹)).neg
    convert h2 using 1
    push_cast
    ring
  have hcont : Continuous fun r : ℝ => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)) := by
    fun_prop
  have hint : IntervalIntegrable (fun r : ℝ => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))
      volume 0 (Real.sqrt a) :=
    hcont.intervalIntegrable 0 (Real.sqrt a)
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint, Real.sq_sqrt ha]
  simp
  ring

/-- The angular integral: `∫_{Ioo (-π) π} 1 = 2π`. -/
theorem rl_angular_integral : ∫ _θ in Set.Ioo (-Real.pi) Real.pi, (1 : ℝ) = 2 * Real.pi := by
  rw [setIntegral_const, smul_eq_mul, mul_one, measureReal_def, Real.volume_Ioo,
    ENNReal.toReal_ofReal (by linarith [Real.pi_pos])]
  ring

/-! ### (b) The planar standard Gaussian density -/

/-- The product of two standard Gaussian densities is the planar density
`(2π)⁻¹ exp (-(x^2 + y^2)/2)`. -/
theorem rl_pdf_mul (x y : ℝ) :
    gaussianPDFReal 0 1 x * gaussianPDFReal 0 1 y =
      (2 * Real.pi)⁻¹ * Real.exp (-(x ^ 2 + y ^ 2) / 2) := by
  simp only [gaussianPDFReal_def, NNReal.coe_one, mul_one, sub_zero]
  have hs : (Real.sqrt (2 * Real.pi))⁻¹ * (Real.sqrt (2 * Real.pi))⁻¹ =
      (2 * Real.pi)⁻¹ := by
    rw [← mul_inv, Real.mul_self_sqrt (by positivity)]
  calc (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-x ^ 2 / 2) *
        ((Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-y ^ 2 / 2))
      = ((Real.sqrt (2 * Real.pi))⁻¹ * (Real.sqrt (2 * Real.pi))⁻¹) *
          (Real.exp (-x ^ 2 / 2) * Real.exp (-y ^ 2 / 2)) := by ring
    _ = (2 * Real.pi)⁻¹ * Real.exp (-(x ^ 2 + y ^ 2) / 2) := by
      rw [hs, ← Real.exp_add]
      congr 2
      ring

/-- The closed disk `{p | p.1^2 + p.2^2 ≤ a}` is measurable. -/
theorem rl_measurableSet_disk (a : ℝ) :
    MeasurableSet {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} :=
  measurableSet_le (by fun_prop) measurable_const

/-- The set `{r | r^2 ≤ a}` is measurable. -/
theorem rl_measurableSet_radial (a : ℝ) : MeasurableSet {r : ℝ | r ^ 2 ≤ a} :=
  measurableSet_le (by fun_prop) measurable_const

/-- In polar coordinates the disk integrand `p.1 • 1_D (density ∘ polarCoord.symm) p` is the
radial function `1_{r^2 ≤ a} (r (2π)⁻¹ exp (-r^2/2))` of `p.1` times the constant `1` of
`p.2` (the form `setIntegral_prod_mul` separates). -/
theorem rl_polar_integrand (a : ℝ) (p : ℝ × ℝ) :
    p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2) (polarCoord.symm p) =
      Set.indicator {r : ℝ | r ^ 2 ≤ a}
        (fun r => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) p.1 *
        (fun _ : ℝ => (1 : ℝ)) p.2 := by
  have hsq : (p.1 * Real.cos p.2) ^ 2 + (p.1 * Real.sin p.2) ^ 2 = p.1 ^ 2 := by
    rw [mul_pow, mul_pow, ← mul_add, Real.cos_sq_add_sin_sq, mul_one]
  simp only [polarCoord_symm_apply, Set.indicator, Set.mem_ofPred_eq, rl_pdf_mul, hsq,
    smul_eq_mul, mul_one]
  split_ifs <;> simp

/-- For `a ≥ 0`, `Ioi 0 ∩ {r | r^2 ≤ a} = Ioc 0 (sqrt a)`. -/
theorem rl_Ioi_inter_eq (a : ℝ) (ha : 0 ≤ a) :
    Set.Ioi (0 : ℝ) ∩ {r : ℝ | r ^ 2 ≤ a} = Set.Ioc 0 (Real.sqrt a) := by
  ext r
  simp only [Set.mem_inter_iff, Set.mem_Ioi, Set.mem_ofPred_eq, Set.mem_Ioc]
  constructor
  · rintro ⟨h0, hr⟩
    exact ⟨h0, (Real.le_sqrt h0.le ha).2 hr⟩
  · rintro ⟨h0, hr⟩
    exact ⟨h0, (Real.le_sqrt h0.le ha).1 hr⟩

/-- The Bochner integral of the planar standard Gaussian density over the closed disk of
radius `sqrt a` is `1 - exp (-a/2)`: polar coordinates (`integral_comp_polarCoord_symm`),
separation of variables (`setIntegral_prod_mul`), the angular integral `2π` and the radial
integral `(2π)⁻¹ (1 - exp (-a/2))`. -/
theorem rl_disk_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ p in {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a},
        gaussianPDFReal 0 1 p.1 * gaussianPDFReal 0 1 p.2 = 1 - Real.exp (-a / 2) := by
  have h1 : ∫ p in {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a},
      gaussianPDFReal 0 1 p.1 * gaussianPDFReal 0 1 p.2 =
      ∫ p, Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2) p :=
    (integral_indicator (rl_measurableSet_disk a)).symm
  have h2 : ∫ p, Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
      (fun q => gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2) p =
      ∫ p in polarCoord.target, p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2) (polarCoord.symm p) :=
    (integral_comp_polarCoord_symm _).symm
  have h3 : ∫ p in polarCoord.target,
      p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2) (polarCoord.symm p) =
      ∫ p in Set.Ioi (0 : ℝ) ×ˢ Set.Ioo (-Real.pi) Real.pi,
        Set.indicator {r : ℝ | r ^ 2 ≤ a}
          (fun r => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) p.1 *
        (fun _ : ℝ => (1 : ℝ)) p.2 := by
    rw [polarCoord_target]
    exact integral_congr_ae (Filter.Eventually.of_forall (fun p => rl_polar_integrand a p))
  have h4 := setIntegral_prod_mul (μ := (volume : Measure ℝ)) (ν := (volume : Measure ℝ))
    (Set.indicator {r : ℝ | r ^ 2 ≤ a}
      (fun r => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))))
    (fun _ : ℝ => (1 : ℝ)) (Set.Ioi (0 : ℝ)) (Set.Ioo (-Real.pi) Real.pi)
  have h5 : ∫ r in Set.Ioi (0 : ℝ), Set.indicator {r : ℝ | r ^ 2 ≤ a}
      (fun r => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) r =
      (2 * Real.pi)⁻¹ * (1 - Real.exp (-a / 2)) := by
    rw [setIntegral_indicator (rl_measurableSet_radial a), rl_Ioi_inter_eq a ha,
      ← intervalIntegral.integral_of_le (Real.sqrt_nonneg a), rl_radial_integral a ha]
  rw [h1, h2, h3]
  refine h4.trans ?_
  rw [h5, rl_angular_integral]
  field_simp

/-- The measure of the closed disk of radius `sqrt a` under the standard planar Gaussian
`gaussianReal 0 1 ⊗ gaussianReal 0 1` is `1 - exp (-a/2)`: the product of the two
`withDensity` measures is `withDensity` of the product density (`prod_withDensity`), whose
value on the (measurable) disk is the `lintegral` of the density there (`withDensity_apply`),
which is `ENNReal.ofReal` of the Bochner integral `rl_disk_integral`
(`ofReal_integral_eq_lintegral_ofReal`). -/
theorem rl_disk_measure (a : ℝ) (ha : 0 ≤ a) :
    ((gaussianReal 0 1).prod (gaussianReal 0 1)) {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} =
      ENNReal.ofReal (1 - Real.exp (-a / 2)) := by
  have hD := rl_measurableSet_disk a
  rw [gaussianReal_of_var_ne_zero 0 one_ne_zero,
    prod_withDensity (measurable_gaussianPDF 0 1) (measurable_gaussianPDF 0 1),
    ← Measure.volume_eq_prod ℝ ℝ, withDensity_apply _ hD, ← rl_disk_integral a ha]
  have hint : Integrable (fun p : ℝ × ℝ => gaussianPDFReal 0 1 p.1 * gaussianPDFReal 0 1 p.2)
      (volume : Measure (ℝ × ℝ)) :=
    (integrable_gaussianPDFReal 0 1).mul_prod (integrable_gaussianPDFReal 0 1)
  have hnn : 0 ≤ᵐ[(volume : Measure (ℝ × ℝ)).restrict
      {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a}]
      (fun p : ℝ × ℝ => gaussianPDFReal 0 1 p.1 * gaussianPDFReal 0 1 p.2) :=
    Filter.Eventually.of_forall
      (fun p => mul_nonneg (gaussianPDFReal_nonneg _ _ _) (gaussianPDFReal_nonneg _ _ _))
  rw [ofReal_integral_eq_lintegral_ofReal hint.integrableOn hnn]
  refine setLIntegral_congr_fun hD (fun p _ => ?_)
  simp only [gaussianPDF]
  rw [ENNReal.ofReal_mul (gaussianPDFReal_nonneg _ _ _)]

/-! ### (c) The interface: measurability and the law of `(x, y)`

A.e.-measurability of the triple `(s, x, y)` is `tm_aemeasurable_triple` above (the `rl_` copy
was removed at integration). -/

/-- The pair `(x, y)` is a.e.-measurable, as the second component of the triple. -/
theorem rl_aemeasurable_xy (I : ConeMomentInterface) :
    @AEMeasurable _ _ _ I.mΩ (fun ω => (I.x ω, I.y ω)) I.P :=
  (tm_aemeasurable_triple I).snd

/-- The law of `(x, y)` is `gaussianReal 0 1 ⊗ gaussianReal 0 1`: the second marginal of the
product law `sxy_law` (`AEMeasurable.map_map_of_aemeasurable`, `Measure.map_snd_prod`; the
composite `Prod.snd ∘ (s, x, y)` is `(x, y)`). -/
theorem rl_law_xy (I : ConeMomentInterface) :
    @Measure.map _ _ I.mΩ _ (fun ω => (I.x ω, I.y ω)) I.P =
      (gaussianReal 0 1).prod (gaussianReal 0 1) := by
  let := I.mΩ
  have := I.isProb
  have hT := tm_aemeasurable_triple I
  have h := AEMeasurable.map_map_of_aemeasurable (measurable_snd.aemeasurable) hT
  rw [I.sxy_law, Measure.map_snd_prod, measure_univ, one_smul] at h
  exact h.symm

/-- The probability of `{x^2 + y^2 ≤ a}` under `P` is the Gaussian measure of the closed disk:
the event is measurable, so `Measure.map_apply_of_aemeasurable` applies to the law `rl_law_xy`. -/
theorem rl_prob_eq_disk_measure (I : ConeMomentInterface) (a : ℝ) :
    I.P {ω | I.x ω ^ 2 + I.y ω ^ 2 ≤ a} =
      ((gaussianReal 0 1).prod (gaussianReal 0 1)) {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} := by
  let := I.mΩ
  have := I.isProb
  rw [← rl_law_xy I,
    Measure.map_apply_of_aemeasurable (rl_aemeasurable_xy I) (rl_measurableSet_disk a)]
  rfl

end Aux

/-! ## The `m = 1` cone moment identity for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.ConeMomentD1Identity` (claim `side24-coefficient`).
Method: transport along the law field `I.A_law` (`Aux.c1_integral_indicator_eq`, applied to
`I.Ω`, `I.mΩ`, `I.P`, `I.isProb`, `I.A`; it uses `integral_map` after a.e.-measurability of
`I.A` from `Aux.c1_aemeasurable_of_map_eq_gaussian`, the Dirac-versus-`NullSingletonClass`
argument), then the negative half of the second moment of
`gaussianReal 0 (8/3)` is `(8/3)/2` (`Aux.c1_setIntegral_sq_Iio`, from `integral_add_compl`,
`integral_Ici_eq_integral_Ioi`, the symmetry `gaussianReal_map_neg` and the variance
`variance_fun_id_gaussianReal`), and `push_cast`/`norm_num` give `4/3 = coneMomentD1`.
Does not establish: the informal identity `D_1 = E[A^2 1{A<0}] = 4/3` of the source as the
source's object (the alignment of `ConeMomentInterface`, its field `A_law` and `coneMomentD1`
with the source's `A ~ N(0, 8/3)` and `D_1` is PENDING_INDEPENDENT_REVIEW); that `D_1` is the
parent's cone moment `D_u` of (15.1) for the reference law; that `ConeMomentInterface` has an
inhabitant, or that its laws are the conditional law `A | V = 0` of the reference field; any
enclosure of the coefficient; `ConeMomentD2Identity`, `TraceMoments` or `RayleighSquareLaw`;
the disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the
register, not derived here; nonauthor review or acceptance of
SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem coneMomentD1Identity (I : ConeMomentInterface) :
    UniversalLaw.Spec.Side24.ConeMomentD1Identity I := by
  unfold UniversalLaw.Spec.Side24.ConeMomentD1Identity coneMomentD1
  rw [@Aux.c1_integral_indicator_eq I.Ω I.mΩ I.P I.isProb I.A I.A_law,
    Aux.c1_setIntegral_sq_Iio (8 / 3 : NNReal) (by norm_num)]
  push_cast
  norm_num

/-! ## The Section 1 trace moments for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.TraceMoments` (claim `side24-coefficient`).
Method: the law of `I.s` is `gaussianReal 0 (5/3)` (`Aux.tm_law_s`: the first marginal of the
interface's product law `sxy_law`, after `Aux.tm_aemeasurable_triple` derives a.e.-measurability
of the triple from the law by the Dirac-versus-`NullSingletonClass` argument); each of the three
integrals is transported to that law by `integral_map` (`Aux.tm_integral_comp_s`) and computed
there: `∫ t^2 = 5/3` (`Aux.c1_second_moment_gaussianReal`, shared with `ConeMomentD1Identity`,
from `variance_fun_id_gaussianReal`),
`∫ t^4 = 3 (5/3)^2 = 25/3` (`Aux.tm_fourth_moment`, from `iteratedDeriv_mgf_zero` and the
explicit fourth derivative `Aux.tm_iteratedDeriv_four`), and `∫ exp (-t^2/2) = sqrt (3/8)`
(`Aux.tm_exp_moment`, from `integral_gaussianReal_eq_integral_smul` and `integral_gaussian`).
Does not establish: the informal Section 1 display of the source as the source's object (the
alignment of the interface fields `s`, `P`, `sxy_law` and of `gaussianReal` with the source's
`s ~ N(0, 5/3)` is PENDING_INDEPENDENT_REVIEW); that any inhabitant of `ConeMomentInterface`
exists, or that its laws are the conditional law `A | V = 0` of the reference field (the laws are
hypothesis fields of the given `I`, not derived here); the main-side constants
`UniversalLaw.Side24.fourth_moment_of_trace` and `exponential_moment_of_trace`, which
kernel-check `3 (5/3)^2 = 25/3` and `1 + 5/3 = 8/3` on main and are neither imported nor
re-stated here (the same arithmetic is discharged inline by `ring`); `ConeMomentD2Identity`,
`RayleighSquareLaw` or any cone moment; any enclosure of the coefficient; the statement module's
own `Does not claim:` text (the statement packet is not edited here); the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here;
nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem traceMoments (I : ConeMomentInterface) : UniversalLaw.Spec.Side24.TraceMoments I := by
  unfold UniversalLaw.Spec.Side24.TraceMoments
  refine ⟨?_, ?_, ?_⟩
  · have e : ∫ ω, I.s ω ^ 2 ∂I.P = ∫ t, t ^ 2 ∂(gaussianReal 0 (5 / 3 : NNReal)) :=
      Aux.tm_integral_comp_s I (fun t => t ^ 2) (by fun_prop)
    rw [e, Aux.c1_second_moment_gaussianReal]
    push_cast
    ring
  · have e : ∫ ω, I.s ω ^ 4 ∂I.P = ∫ t, t ^ 4 ∂(gaussianReal 0 (5 / 3 : NNReal)) :=
      Aux.tm_integral_comp_s I (fun t => t ^ 4) (by fun_prop)
    rw [e, Aux.tm_fourth_moment]
    push_cast
    ring
  · have e : ∫ ω, Real.exp (-(I.s ω ^ 2) / 2) ∂I.P =
        ∫ t, Real.exp (-(t ^ 2) / 2) ∂(gaussianReal 0 (5 / 3 : NNReal)) :=
      Aux.tm_integral_comp_s I (fun t => Real.exp (-(t ^ 2) / 2)) (by fun_prop)
    rw [e, Aux.tm_exp_moment]

/-! ## The law of `R^2` for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.RayleighSquareLaw` (claim `side24-coefficient`).
Method: for `a ≥ 0`, `P {x^2 + y^2 ≤ a}` is the measure of the closed disk of radius `sqrt a`
under `gaussianReal 0 1 ⊗ gaussianReal 0 1` (`Aux.rl_prob_eq_disk_measure`, from the law
`Aux.rl_law_xy` of `(x, y)` obtained as the second marginal of the interface's product law
`sxy_law`, after `Aux.tm_aemeasurable_triple`, shared with `TraceMoments`, derives
a.e.-measurability of the triple from the
law by the Dirac-versus-`NullSingletonClass` argument); that disk measure is
`ENNReal.ofReal (1 - exp (-a/2))` (`Aux.rl_disk_measure`: `prod_withDensity`,
`withDensity_apply`, `ofReal_integral_eq_lintegral_ofReal`, then polar coordinates
`integral_comp_polarCoord_symm` and `setIntegral_prod_mul` in `Aux.rl_disk_integral`, with the
angular integral `2π` and the radial integral by the fundamental theorem of calculus); the
right-hand side `∫_0^a exp (-z/2)/2 dz` is the same number `1 - exp (-a/2)`
(`Aux.rl_rhs_integral`), and `ENNReal.toReal_ofReal` closes.
Does not establish: the informal Section 1 display "The variable R^2 has density e^(-z/2)/2" of
the source as the source's object, nor its review and acceptance (the alignment of the interface
fields `x`, `y`, `P`, `sxy_law` and of `gaussianReal` with the source's `x, y ~ N(0, 1)` is
PENDING_INDEPENDENT_REVIEW); from the Prop's own `Does not claim:` line, "the law (statement
only)": the law is proved here only for the formal objects of the Prop, as the distribution
function `P(R^2 ≤ a)` for `a ≥ 0`, not as a density and not for the source's variable; and
"measurability of the event (if the event is not measurable, `P` is its outer measure)":
measurability of the formal event `{x^2 + y^2 ≤ a}` is proved here (`Aux.rl_measurableSet_disk`,
`Measure.map_apply_of_aemeasurable`), so the outer-measure reading does not arise for this
event, but nothing is established about any other event; that any inhabitant of
`ConeMomentInterface` exists, or that its laws are the conditional law `A | V = 0` of the
reference field (the laws are hypothesis fields of the given `I`, not derived here);
`ConeMomentD2Identity` or any cone moment; any enclosure of the coefficient; the statement
module's own text (the statement packet is not edited here); the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here;
nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem rayleighSquareLaw (I : ConeMomentInterface) : UniversalLaw.Spec.Side24.RayleighSquareLaw I := by
  unfold UniversalLaw.Spec.Side24.RayleighSquareLaw
  intro a ha
  rw [Aux.rl_prob_eq_disk_measure I a, Aux.rl_disk_measure a ha, Aux.rl_rhs_integral,
    ENNReal.toReal_ofReal]
  have : Real.exp (-a / 2) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  linarith

end UniversalLaw.Proofs.Side24Interfaces
