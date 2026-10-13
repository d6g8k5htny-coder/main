import Specifications.Side24

/-!
# SIDE24 interface proofs: the law of `R^2 = x^2 + y^2` (`RayleighSquareLaw I`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux` (prefix `rl_`). The route,
for an arbitrary inhabitant `I : ConeMomentInterface` (existence of an inhabitant is neither
asserted nor used): (1) the triple `(s, x, y)` is a.e.-measurable, because at this Mathlib
revision the image measure of a non-a.e.-measurable map under a nonzero measure is a Dirac
measure (`Measure.map_of_not_aemeasurable_of_ne_zero`), which charges a singleton, while the
product Gaussian law `sxy_law` has `NullSingletonClass`; (2) the law of `(x, y)` is the
product `gaussianReal 0 1 ⊗ gaussianReal 0 1` (second marginal of `sxy_law`,
`AEMeasurable.map_map_of_aemeasurable`, `Measure.map_snd_prod`); (3) the event
`{x^2 + y^2 ≤ a}` is measurable, so `Measure.map_apply_of_aemeasurable` identifies its
probability with the Gaussian measure of the closed disk of radius `sqrt a`; (4) that disk
measure is computed by hand (Mathlib has no Rayleigh or chi-squared law): the product Gaussian
is `volume.withDensity` of the planar density `(2π)⁻¹ exp (-(p.1^2 + p.2^2)/2)`
(`gaussianReal_of_var_ne_zero`, `prod_withDensity`, `withDensity_apply`), the `lintegral` is a
Bochner integral of a nonnegative integrable function (`ofReal_integral_eq_lintegral_ofReal`),
polar coordinates (`integral_comp_polarCoord_symm`, exactly as Mathlib's
`integral_gaussian_sq_complex`) and `setIntegral_prod_mul` separate the angle, whose integral
over `Ioo (-π) π` is `2π`, from the radial integral `∫_0^{sqrt a} r exp (-r^2/2) dr =
1 - exp (-a/2)` (fundamental theorem of calculus); (5) the right-hand side
`∫_0^a exp (-z/2)/2 dz = 1 - exp (-a/2)` is the same elementary integral. Only the field
`sxy_law` and the instance `isProb` of the interface are used; the fields `A`, `A_law` play no
role.
-/

namespace UniversalLaw.Proofs.Side24Interfaces

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

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
`∫_0^{sqrt a} r (2π)⁻¹ exp (-r^2/2) dr = (2π)⁻¹ (1 - exp (-a/2))`, by the fundamental theorem
of calculus with antiderivative `r ↦ -(2π)⁻¹ exp (-r^2/2)`. -/
theorem rl_radial_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ r in (0 : ℝ)..Real.sqrt a, r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)) =
      (2 * Real.pi)⁻¹ * (1 - Real.exp (-a / 2)) := by
  have hderiv : ∀ r ∈ Set.uIcc (0 : ℝ) (Real.sqrt a),
      HasDerivAt (fun r : ℝ => -((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))
        (r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) r := by
    intro r _
    have h1 : HasDerivAt (fun r : ℝ => -(r ^ 2) / 2) (-(((2 : ℕ) : ℝ) * r ^ (2 - 1) * 1) / 2) r :=
      (((hasDerivAt_id' r).fun_pow 2).neg).div_const 2
    have h2 := (h1.exp.const_mul ((2 * Real.pi)⁻¹)).neg
    convert h2 using 1
    push_cast
    ring
  have hint : IntervalIntegrable (fun r : ℝ => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))
      volume 0 (Real.sqrt a) :=
    (by fun_prop : Continuous fun r : ℝ => r * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))).intervalIntegrable
      0 (Real.sqrt a)
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
  have hs : (Real.sqrt (2 * Real.pi))⁻¹ * (Real.sqrt (2 * Real.pi))⁻¹ = (2 * Real.pi)⁻¹ := by
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
  have h3 : ∫ p in polarCoord.target, p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
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
  have hnn : 0 ≤ᵐ[(volume : Measure (ℝ × ℝ)).restrict {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a}]
      (fun p : ℝ × ℝ => gaussianPDFReal 0 1 p.1 * gaussianPDFReal 0 1 p.2) :=
    Filter.Eventually.of_forall
      (fun p => mul_nonneg (gaussianPDFReal_nonneg _ _ _) (gaussianPDFReal_nonneg _ _ _))
  rw [ofReal_integral_eq_lintegral_ofReal hint.integrableOn hnn]
  refine setLIntegral_congr_fun hD (fun p _ => ?_)
  simp only [gaussianPDF]
  rw [ENNReal.ofReal_mul (gaussianPDFReal_nonneg _ _ _)]

/-! ### (c) The interface: measurability and the law of `(x, y)` -/

/-- The triple `(s, x, y)` is a.e.-measurable: its image measure under `P` is the product
Gaussian law (`sxy_law`); were the triple not a.e.-measurable, `Measure.map` would be a Dirac
mass (`Measure.map_of_not_aemeasurable_of_ne_zero`), which gives a singleton measure `1`,
whereas the product of non-degenerate Gaussians has `NullSingletonClass`. -/
theorem rl_aemeasurable_triple (I : ConeMomentInterface) :
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

/-- The pair `(x, y)` is a.e.-measurable, as the second component of the triple. -/
theorem rl_aemeasurable_xy (I : ConeMomentInterface) :
    @AEMeasurable _ _ _ I.mΩ (fun ω => (I.x ω, I.y ω)) I.P :=
  (rl_aemeasurable_triple I).snd

/-- The law of `(x, y)` is `gaussianReal 0 1 ⊗ gaussianReal 0 1`: the second marginal of the
product law `sxy_law` (`AEMeasurable.map_map_of_aemeasurable`, `Measure.map_snd_prod`; the
composite `Prod.snd ∘ (s, x, y)` is `(x, y)`). -/
theorem rl_law_xy (I : ConeMomentInterface) :
    @Measure.map _ _ I.mΩ _ (fun ω => (I.x ω, I.y ω)) I.P =
      (gaussianReal 0 1).prod (gaussianReal 0 1) := by
  let := I.mΩ
  have := I.isProb
  have hT := rl_aemeasurable_triple I
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
  rw [← rl_law_xy I, Measure.map_apply_of_aemeasurable (rl_aemeasurable_xy I) (rl_measurableSet_disk a)]
  rfl

end Aux

/-! ## The law of `R^2` for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.RayleighSquareLaw` (claim `side24-coefficient`).
Method: for `a ≥ 0`, `P {x^2 + y^2 ≤ a}` is the measure of the closed disk of radius `sqrt a`
under `gaussianReal 0 1 ⊗ gaussianReal 0 1` (`Aux.rl_prob_eq_disk_measure`, from the law
`Aux.rl_law_xy` of `(x, y)` obtained as the second marginal of the interface's product law
`sxy_law`, after `Aux.rl_aemeasurable_triple` derives a.e.-measurability of the triple from the
law by the Dirac-versus-`NullSingletonClass` argument); that disk measure is
`ENNReal.ofReal (1 - exp (-a/2))` (`Aux.rl_disk_measure`: `prod_withDensity`,
`withDensity_apply`, `ofReal_integral_eq_lintegral_ofReal`, then polar coordinates
`integral_comp_polarCoord_symm` and `setIntegral_prod_mul` in `Aux.rl_disk_integral`, with the
angular integral `2π` and the radial integral by the fundamental theorem of calculus); the
right-hand side `∫_0^a exp (-z/2)/2 dz` is the same number `1 - exp (-a/2)`
(`Aux.rl_rhs_integral`), and `ENNReal.toReal_ofReal` closes.
Does not establish: the informal Section 1 display "The variable R^2 has density e^(-z/2)/2" of
the source as the source's object (the alignment of the interface fields `x`, `y`, `P`,
`sxy_law` and of `gaussianReal` with the source's `x, y ~ N(0, 1)` is
PENDING_INDEPENDENT_REVIEW); the density statement itself, which the Prop writes only through
the distribution function `P(R^2 ≤ a)` for `a ≥ 0`; measurability of the event is proved here
for the given formal event, but the Prop's own `Does not claim:` reservation (if the event were
not measurable, `P` would be its outer measure) is a statement about the Prop, not about the
source; that any inhabitant of `ConeMomentInterface` exists, or that its laws are the
conditional law `A | V = 0` of the reference field (the laws are hypothesis fields of the given
`I`, not derived here); `ConeMomentD2Identity` or any cone moment; any enclosure of the
coefficient; the statement module's own text (the statement packet is not edited here); the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not
derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem rayleighSquareLaw (I : ConeMomentInterface) :
    UniversalLaw.Spec.Side24.RayleighSquareLaw I := by
  unfold UniversalLaw.Spec.Side24.RayleighSquareLaw
  intro a ha
  rw [Aux.rl_prob_eq_disk_measure I a, Aux.rl_disk_measure a ha, Aux.rl_rhs_integral,
    ENNReal.toReal_ofReal]
  have : Real.exp (-a / 2) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  linarith

end UniversalLaw.Proofs.Side24Interfaces

-- SCRATCH COPY ONLY (integrator strips): axioms of every theorem, then the constant audit.
#print axioms UniversalLaw.Proofs.Side24Interfaces.rayleighSquareLaw
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_rhs_integral
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_radial_integral
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_angular_integral
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_pdf_mul
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_measurableSet_disk
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_measurableSet_radial
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_polar_integrand
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_Ioi_inter_eq
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_disk_integral
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_disk_measure
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_aemeasurable_triple
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_aemeasurable_xy
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_law_xy
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.rl_prob_eq_disk_measure
-- SCRATCH COPY ONLY: list every constant this module adds, with kind and Prop-ness.
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut rows : Array String := #[]
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp ci.type)
    rows := rows.push s!"{kind} prop={isP} unsafe={ci.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"TOTAL {rows.size}"
