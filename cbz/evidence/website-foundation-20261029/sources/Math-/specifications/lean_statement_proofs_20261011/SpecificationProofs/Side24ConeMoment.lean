import Specifications.Side24
import SpecificationProofs.Side24Interfaces
import SpecificationProofs.Side24Elementary

/-!
# SIDE24 cone moment: the `m = 2` identity `ConeMomentD2Identity` over `ConeMomentInterface`
(statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The one theorem outside the `Aux` namespace has as its type exactly the registered `Prop`
`UniversalLaw.Spec.Side24.ConeMomentD2Identity` of `Specifications/Side24.lean` (statement packet
`specifications/lean_statements_20261010`, which is not modified here), applied to its own
parameter `I : ConeMomentInterface`. A kernel-checked proof of that `Prop` is evidence about that
`Prop` only, for the formal objects it names (`ConeMomentInterface`, `transverseMatrix`,
`NegDefCone`, `coneIntegrand`, Mathlib's `gaussianReal` and Bochner integral): it is not evidence
about the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1, register row
`side24-coefficient`), which it does not make reviewed, accepted or "verified"; it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW); it does not enclose the coefficient `c_{d,24}`; and it promotes,
closes, discharges or reclassifies nothing. Nothing here constructs an inhabitant of
`ConeMomentInterface` or derives its law fields; the theorem holds for every inhabitant and uses
only the law field `sxy_law` and the instances `mΩ`, `isProb` (the latter through
`tm_aemeasurable_triple`); the fields `A` and `A_law` are not used.

Provenance: round-5 lane for `ConeMomentD2Identity`, selected by the round-4 triage. Helper lemmas
live in `UniversalLaw.Proofs.Side24ConeMoment.Aux` with the tag `cm_`; the gate inventories only
the theorem outside that namespace. The module reuses, without restating them, the public helpers
of the round-4 module `SpecificationProofs.Side24Interfaces` (`tm_aemeasurable_triple`,
`tm_fourth_moment`, `tm_exp_moment`, `c1_second_moment_gaussianReal`,
`c1_integrable_sq_gaussianReal`, `rl_pdf_mul`, `rl_measurableSet_radial`, `rl_Ioi_inter_eq`,
`rl_angular_integral`) and the inventory theorem `elementaryConeIntegral` of
`SpecificationProofs.Side24Elementary`. The mathematics:

* Cone identification (`cm_cone_iff`): `NegDefCone (s, x, y)` holds iff `s < 0` and
  `x^2 + y^2 < s^2`. Forward: `det (-A) = det A = s^2 - x^2 - y^2 > 0` (`Matrix.PosDef.det_pos`,
  `cm_det_transverse`) and the diagonal entries `-s - x`, `-s + x` are positive
  (`Matrix.PosDef.diag_pos`). Converse: `-A` is symmetric and its quadratic form at `v = (a, b)`
  is `-s (a^2 + b^2) - x (a^2 - b^2) - 2 y a b`, which is positive for `v ≠ 0` by the Lagrange
  identity `(x (a^2 - b^2) + 2 y a b)^2 + (2 x a b - y (a^2 - b^2))^2 = (x^2 + y^2) (a^2 + b^2)^2`
  (`Matrix.PosDef.of_dotProduct_mulVec_pos`, `cm_quad_pos`). Hence `coneIntegrand` is the
  indicator of `{s < 0 ∧ x^2 + y^2 < s^2}` times `(s^2 - x^2 - y^2)^2`
  (`cm_coneIntegrand_eq`), a measurable function bounded by `s^4`.
* Transport and Fubini: the triple `(s, x, y)` is a.e.-measurable (`tm_aemeasurable_triple`), so
  `integral_map` and the law field `sxy_law` move the integral to
  `N(0, 5/3) ⊗ (N(0, 1) ⊗ N(0, 1))`; the integrand is integrable there (bound `s^4`,
  `memLp_id_gaussianReal'`, `Integrable.comp_fst`), so `integral_prod` gives an iterated integral.
* Inner integral (`cm_inner`): for `s ≥ 0` the slice is `0`; for `s < 0` the slice equals the
  closed-disk integrand `1{x^2 + y^2 ≤ s^2} (s^2 - x^2 - y^2)^2` (equal on the circle, where the
  square vanishes), whose planar Gaussian integral is `a^2 - 4a + 8 - 8 exp (-a/2)` at `a = s^2`
  (`cm_gauss2_integral`): the product of the two `withDensity` laws is `withDensity` of the
  product density (`prod_withDensity`, `integral_withDensity_eq_integral_toReal_smul`), polar
  coordinates (`integral_comp_polarCoord_symm`, `setIntegral_prod_mul`), the angular integral
  `2π`, and the radial integral by the substitution `z = r^2`
  (`intervalIntegral.integral_comp_mul_deriv`) onto `elementaryConeIntegral`.
* Outer integral: the bracket `t^4 - 4t^2 + 8 - 8 exp (-t^2/2)` is even and integrable under
  `N(0, 5/3)`, so its integral over `t < 0` is half its full integral (`gaussianReal_map_neg`,
  `integral_add_compl`, `integral_Ici_eq_integral_Ioi`; `cm_setIntegral_Iio_even`), and the full
  integral is `25/3 - 4 (5/3) + 8 - 8 sqrt (3/8)` from the moments `∫ t^4 = 25/3`,
  `∫ t^2 = 5/3`, `∫ exp (-t^2/2) = sqrt (3/8)` (`cm_integral_bracket`).

Remark recorded by the lane, for the maintainers: for the formal objects, the proof follows the
shape of the source's Section 1 computation for `m = 2`: the cone as `s < -R`, in the form
`s < 0 ∧ x^2 + y^2 < s^2` (`cm_cone_iff`); the truncated integral of `(a - z)^2` against the
density `e^(-z/2)/2` (`cm_radial_integral`, evaluated by `elementaryConeIntegral`, whose
registered anchor is the source's display of that integral); symmetry in `s`
(`cm_setIntegral_Iio_even`); and the three moments of `s` (`cm_integral_bracket`). It does not
invoke the registered `RayleighSquareLaw` or any statement of the law of `R^2`: the
`R^2`-density integral is obtained inline, by polar coordinates and the substitution `z = r^2`
(`cm_disk_integral`, `cm_radial_integral`), and is then evaluated by `elementaryConeIntegral`.
It also writes out steps the source does not write out (the transport along `sxy_law`, Fubini with
its integrability, and the closed disk in place of the open one, which changes nothing because
the integrand vanishes on the circle `x^2 + y^2 = s^2`). It is therefore not a cross-check of the
source's Rayleigh cone integration by a different route. Being authored by Anthropic / Claude at
organizational-independence credit 0, it is also not the nonauthor review that the source asks
for. That these formal steps are the source's steps is not established (alignment
PENDING_INDEPENDENT_REVIEW). This is a remark about a route, not about any claim; no register
label changes.
-/

namespace UniversalLaw.Proofs.Side24ConeMoment

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

open UniversalLaw.Proofs.Side24Interfaces.Aux

/-! ## `cm_` (a): the negative-definite cone of the `m = 2` transverse matrix -/

/-- `det [[s + x, y], [y, s - x]] = s^2 - x^2 - y^2` (`Matrix.det_fin_two_of`). -/
theorem cm_det_transverse (s x y : ℝ) :
    (transverseMatrix s x y).det = s ^ 2 - x ^ 2 - y ^ 2 := by
  unfold transverseMatrix
  rw [Matrix.det_fin_two_of]
  ring

/-- Forward direction of the cone characterisation: positive definiteness of `-A` gives `s < 0`
(the two diagonal entries `-s - x`, `-s + x` are positive, `Matrix.PosDef.diag_pos`) and
`x^2 + y^2 < s^2` (`det (-A) = det A > 0`, `Matrix.PosDef.det_pos`, `Matrix.det_neg`). -/
theorem cm_cone_forward (s x y : ℝ) (h : NegDefCone (s, x, y)) :
    s < 0 ∧ x ^ 2 + y ^ 2 < s ^ 2 := by
  unfold NegDefCone at h
  have hdet := h.det_pos
  rw [Matrix.det_neg, cm_det_transverse] at hdet
  have hdiag := h.diag_pos (i := 0)
  have hdiag' := h.diag_pos (i := 1)
  simp [transverseMatrix] at hdiag hdiag'
  simp only [Fintype.card_fin, even_two, Even.neg_pow, one_pow, one_mul] at hdet
  constructor
  · linarith
  · linarith

/-- A nonzero vector of `Fin 2 → ℝ` has positive squared norm `v 0 ^ 2 + v 1 ^ 2`. -/
theorem cm_sq_add_sq_pos (v : Fin 2 → ℝ) (hv : v ≠ 0) : 0 < v 0 ^ 2 + v 1 ^ 2 := by
  by_contra hle
  rw [not_lt] at hle
  have h0 : v 0 = 0 := by nlinarith [sq_nonneg (v 0), sq_nonneg (v 1)]
  have h1 : v 1 = 0 := by nlinarith [sq_nonneg (v 0), sq_nonneg (v 1)]
  apply hv
  funext i
  fin_cases i
  · exact h0
  · exact h1

/-- The quadratic form of `-A` at `(a, b) ≠ 0` is positive inside the cone: by the Lagrange
identity `(x (a^2 - b^2) + 2 y a b)^2 + (2 x a b - y (a^2 - b^2))^2 = (x^2 + y^2) (a^2 + b^2)^2`,
`|x (a^2 - b^2) + 2 y a b| ≤ sqrt (x^2 + y^2) (a^2 + b^2) < -s (a^2 + b^2)` (`nlinarith`). -/
theorem cm_quad_pos (s x y a b : ℝ) (hs : s < 0) (hr : x ^ 2 + y ^ 2 < s ^ 2)
    (hab : 0 < a ^ 2 + b ^ 2) :
    0 < -s * (a ^ 2 + b ^ 2) - x * (a ^ 2 - b ^ 2) - 2 * y * a * b := by
  have key : (x * (a ^ 2 - b ^ 2) + 2 * y * a * b) ^ 2 + (2 * x * a * b - y * (a ^ 2 - b ^ 2)) ^ 2 =
      (x ^ 2 + y ^ 2) * (a ^ 2 + b ^ 2) ^ 2 := by ring
  by_contra hle
  rw [not_lt] at hle
  have hL : -s * (a ^ 2 + b ^ 2) ≤ x * (a ^ 2 - b ^ 2) + 2 * y * a * b := by linarith
  have hc : 0 < -s * (a ^ 2 + b ^ 2) := by nlinarith
  have h1 : (-s * (a ^ 2 + b ^ 2)) ^ 2 ≤ (x * (a ^ 2 - b ^ 2) + 2 * y * a * b) ^ 2 := by
    nlinarith
  have h2 : (x ^ 2 + y ^ 2) * (a ^ 2 + b ^ 2) ^ 2 < s ^ 2 * (a ^ 2 + b ^ 2) ^ 2 := by
    have : 0 < (a ^ 2 + b ^ 2) ^ 2 := by positivity
    nlinarith
  nlinarith [sq_nonneg (2 * x * a * b - y * (a ^ 2 - b ^ 2))]

/-- Converse direction of the cone characterisation: for `s < 0` and `x^2 + y^2 < s^2`, `-A` is
symmetric and its quadratic form is positive on nonzero vectors
(`Matrix.PosDef.of_dotProduct_mulVec_pos`, `cm_quad_pos`). -/
theorem cm_cone_backward (s x y : ℝ) (hs : s < 0) (hr : x ^ 2 + y ^ 2 < s ^ 2) :
    NegDefCone (s, x, y) := by
  unfold NegDefCone
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [transverseMatrix]
  · intro v hv
    have hab := cm_sq_add_sq_pos v hv
    have := cm_quad_pos s x y (v 0) (v 1) hs hr hab
    simp [transverseMatrix, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
    nlinarith

/-- The formal cone `NegDefCone q` is `q.1 < 0 ∧ q.2.1^2 + q.2.2^2 < q.1^2` (both directions
above). A statement about the formal definition only. -/
theorem cm_cone_iff (q : ℝ × ℝ × ℝ) :
    NegDefCone q ↔ q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2 :=
  ⟨fun h => cm_cone_forward q.1 q.2.1 q.2.2 h,
    fun h => cm_cone_backward q.1 q.2.1 q.2.2 h.1 h.2⟩

/-! ## `cm_` (b): the planar Gaussian integral of the disk integrand -/

/-- The radial integral `∫_0^{sqrt a} r (a - r^2)^2 (2π)⁻¹ exp (-r^2/2) dr =
(2π)⁻¹ (a^2 - 4a + 8 - 8 exp (-a/2))`: the substitution `z = r^2`
(`intervalIntegral.integral_comp_mul_deriv`) turns it into `(2π)⁻¹` times the interval integral
of `ElementaryConeIntegral`, which `Side24Elementary.elementaryConeIntegral` evaluates. -/
theorem cm_radial_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ r in (0 : ℝ)..Real.sqrt a,
        r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2))) =
      (2 * Real.pi)⁻¹ * (a ^ 2 - 4 * a + 8 - 8 * Real.exp (-a / 2)) := by
  have hderiv : ∀ r ∈ Set.uIcc (0 : ℝ) (Real.sqrt a),
      HasDerivAt (fun r : ℝ => r ^ 2) (2 * r) r := by
    intro r _
    simpa using hasDerivAt_pow 2 r
  have hcv := intervalIntegral.integral_comp_mul_deriv (a := 0) (b := Real.sqrt a)
    (f := fun r : ℝ => r ^ 2) (f' := fun r => 2 * r)
    (g := fun z : ℝ => (2 * Real.pi)⁻¹ * ((a - z) ^ 2 * Real.exp (-z / 2) / 2)) hderiv
    (by fun_prop) (by fun_prop)
  beta_reduce at hcv
  have hpt : (fun r : ℝ => r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))) =
      fun r => ((fun z : ℝ => (2 * Real.pi)⁻¹ * ((a - z) ^ 2 * Real.exp (-z / 2) / 2)) ∘
        (fun r : ℝ => r ^ 2)) r * (2 * r) := by
    funext r
    simp only [Function.comp]
    ring
  rw [hpt, hcv, Real.sq_sqrt ha, show ((0 : ℝ)) ^ 2 = 0 by norm_num,
    intervalIntegral.integral_const_mul,
    UniversalLaw.Proofs.Side24Elementary.elementaryConeIntegral a ha]

/-- In polar coordinates the disk integrand `p.1 • 1_D (a - |q|^2)^2 φ(q.1) φ(q.2)` at
`q = polarCoord.symm p` is the radial function `1_{r^2 ≤ a} r (a - r^2)^2 (2π)⁻¹ exp (-r^2/2)` of
`p.1` times the constant `1` of `p.2` (`rl_pdf_mul`, `cos^2 + sin^2 = 1`). -/
theorem cm_polar_integrand (a : ℝ) (p : ℝ × ℝ) :
    p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2 *
          (gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2)) (polarCoord.symm p) =
      Set.indicator {r : ℝ | r ^ 2 ≤ a}
        (fun r => r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))) p.1 *
        (fun _ : ℝ => (1 : ℝ)) p.2 := by
  have hsq : (p.1 * Real.cos p.2) ^ 2 + (p.1 * Real.sin p.2) ^ 2 = p.1 ^ 2 := by
    rw [mul_pow, mul_pow, ← mul_add, Real.cos_sq_add_sin_sq, mul_one]
  have hsq' : a - (p.1 * Real.cos p.2) ^ 2 - (p.1 * Real.sin p.2) ^ 2 = a - p.1 ^ 2 := by
    rw [sub_sub, hsq]
  simp only [polarCoord_symm_apply, Set.indicator, Set.mem_ofPred_eq, rl_pdf_mul, hsq, hsq',
    smul_eq_mul, mul_one]
  split_ifs <;> simp

/-- The Lebesgue integral over the plane of `1{|q|^2 ≤ a} (a - |q|^2)^2 φ(q.1) φ(q.2)` (`φ` the
standard Gaussian density) is `a^2 - 4a + 8 - 8 exp (-a/2)`: polar coordinates
(`integral_comp_polarCoord_symm`), separation of variables (`setIntegral_prod_mul`), the angular
integral `2π` (`rl_angular_integral`) and the radial integral `cm_radial_integral`. -/
theorem cm_disk_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ p, Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2 *
          (gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2)) p =
      a ^ 2 - 4 * a + 8 - 8 * Real.exp (-a / 2) := by
  have h2 : ∫ p, Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2 *
          (gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2)) p =
      ∫ p in polarCoord.target, p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2 *
          (gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2)) (polarCoord.symm p) :=
    (integral_comp_polarCoord_symm _).symm
  have h3 : ∫ p in polarCoord.target, p.1 • Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a}
        (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2 *
          (gaussianPDFReal 0 1 q.1 * gaussianPDFReal 0 1 q.2)) (polarCoord.symm p) =
      ∫ p in Set.Ioi (0 : ℝ) ×ˢ Set.Ioo (-Real.pi) Real.pi,
        Set.indicator {r : ℝ | r ^ 2 ≤ a}
          (fun r => r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))) p.1 *
        (fun _ : ℝ => (1 : ℝ)) p.2 := by
    rw [polarCoord_target]
    exact integral_congr_ae (Filter.Eventually.of_forall (fun p => cm_polar_integrand a p))
  have h4 := setIntegral_prod_mul (μ := (volume : Measure ℝ)) (ν := (volume : Measure ℝ))
    (Set.indicator {r : ℝ | r ^ 2 ≤ a}
      (fun r => r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))))
    (fun _ : ℝ => (1 : ℝ)) (Set.Ioi (0 : ℝ)) (Set.Ioo (-Real.pi) Real.pi)
  have h5 : ∫ r in Set.Ioi (0 : ℝ), Set.indicator {r : ℝ | r ^ 2 ≤ a}
      (fun r => r * ((a - r ^ 2) ^ 2 * ((2 * Real.pi)⁻¹ * Real.exp (-(r ^ 2) / 2)))) r =
      (2 * Real.pi)⁻¹ * (a ^ 2 - 4 * a + 8 - 8 * Real.exp (-a / 2)) := by
    rw [setIntegral_indicator (rl_measurableSet_radial a), rl_Ioi_inter_eq a ha,
      ← intervalIntegral.integral_of_le (Real.sqrt_nonneg a), cm_radial_integral a ha]
  rw [h2, h3]
  refine h4.trans ?_
  rw [h5, rl_angular_integral]
  field_simp

/-- Under the standard planar Gaussian `gaussianReal 0 1 ⊗ gaussianReal 0 1`, the integral of
`1{|q|^2 ≤ a} (a - |q|^2)^2` is `a^2 - 4a + 8 - 8 exp (-a/2)`: the product of the two
`withDensity` laws is `withDensity` of the product density (`prod_withDensity`), the integral
against it is the Lebesgue integral of the density times the integrand
(`integral_withDensity_eq_integral_toReal_smul`, `toReal_gaussianPDF`), and `cm_disk_integral`. -/
theorem cm_gauss2_integral (a : ℝ) (ha : 0 ≤ a) :
    ∫ p, Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ a} (fun q => (a - q.1 ^ 2 - q.2 ^ 2) ^ 2) p
        ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) =
      a ^ 2 - 4 * a + 8 - 8 * Real.exp (-a / 2) := by
  have hmeas : Measurable (fun z : ℝ × ℝ => gaussianPDF 0 1 z.1 * gaussianPDF 0 1 z.2) :=
    ((measurable_gaussianPDF 0 1).comp measurable_fst).mul
      ((measurable_gaussianPDF 0 1).comp measurable_snd)
  rw [gaussianReal_of_var_ne_zero 0 one_ne_zero,
    prod_withDensity (measurable_gaussianPDF 0 1) (measurable_gaussianPDF 0 1),
    ← Measure.volume_eq_prod ℝ ℝ,
    integral_withDensity_eq_integral_toReal_smul hmeas
      (Filter.Eventually.of_forall (fun _ => ENNReal.mul_lt_top gaussianPDF_lt_top gaussianPDF_lt_top)),
    ← cm_disk_integral a ha]
  congr 1
  funext p
  simp only [ENNReal.toReal_mul, toReal_gaussianPDF, Set.indicator, Set.mem_ofPred_eq, smul_eq_mul]
  split_ifs <;> ring

/-! ## `cm_` (c): the bracket `t^4 - 4t^2 + 8 - 8 exp (-t^2/2)` under `N(0, 5/3)` -/

/-- `t ↦ t^4` is integrable against `gaussianReal 0 v` (`memLp_id_gaussianReal'` at `p = 4`,
`MemLp.integrable_norm_pow`, `|t|^4 = t^4`). -/
theorem cm_integrable_pow_four (v : NNReal) :
    Integrable (fun t : ℝ => t ^ 4) (gaussianReal 0 v) := by
  have h := (memLp_id_gaussianReal' (μ := 0) (v := v) ((4 : ℕ) : ENNReal) (by simp)).integrable_norm_pow
    (by norm_num)
  refine h.congr (Filter.Eventually.of_forall fun x => ?_)
  change ‖x‖ ^ 4 = x ^ 4
  rw [Real.norm_eq_abs, pow_abs, abs_of_nonneg (by positivity)]

/-- `t ↦ exp (-t^2/2)` is integrable against `gaussianReal 0 v`: measurable and bounded by `1`
under a probability measure (`Integrable.mono'`). -/
theorem cm_integrable_exp (v : NNReal) :
    Integrable (fun t : ℝ => Real.exp (-(t ^ 2) / 2)) (gaussianReal 0 v) := by
  refine (integrable_const (1 : ℝ)).mono' (by fun_prop : Measurable fun t : ℝ =>
    Real.exp (-(t ^ 2) / 2)).aestronglyMeasurable (Filter.Eventually.of_forall fun t => ?_)
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
  have : 0 ≤ t ^ 2 := sq_nonneg t
  linarith

/-- The bracket `t^4 - 4t^2 + 8 - 8 exp (-t^2/2)` is integrable against `gaussianReal 0 v`
(sums and scalar multiples of integrable functions). -/
theorem cm_integrable_bracket (v : NNReal) :
    Integrable (fun t : ℝ => t ^ 4 - 4 * t ^ 2 + 8 - 8 * Real.exp (-(t ^ 2) / 2))
      (gaussianReal 0 v) :=
  ((((cm_integrable_pow_four v).sub ((c1_integrable_sq_gaussianReal v).const_mul 4)).add
    (integrable_const 8)).sub ((cm_integrable_exp v).const_mul 8))

/-- The integral of the bracket against `gaussianReal 0 (5/3)` is
`25/3 - 20/3 + 8 - 8 sqrt (3/8)`: linearity, then the moments `∫ t^4 = 3 (5/3)^2`
(`tm_fourth_moment`), `∫ t^2 = 5/3` (`c1_second_moment_gaussianReal`) and
`∫ exp (-t^2/2) = sqrt (3/8)` (`tm_exp_moment`). -/
theorem cm_integral_bracket :
    ∫ t, (t ^ 4 - 4 * t ^ 2 + 8 - 8 * Real.exp (-(t ^ 2) / 2)) ∂(gaussianReal 0 (5 / 3 : NNReal)) =
      25 / 3 - 20 / 3 + 8 - 8 * Real.sqrt (3 / 8) := by
  have h4 := cm_integrable_pow_four (5 / 3 : NNReal)
  have h2 := c1_integrable_sq_gaussianReal (5 / 3 : NNReal)
  have he := cm_integrable_exp (5 / 3 : NNReal)
  have hA : Integrable (fun t : ℝ => t ^ 4 - 4 * t ^ 2) (gaussianReal 0 (5 / 3 : NNReal)) :=
    h4.sub (h2.const_mul 4)
  have hB : Integrable (fun t : ℝ => t ^ 4 - 4 * t ^ 2 + 8) (gaussianReal 0 (5 / 3 : NNReal)) :=
    hA.add (integrable_const 8)
  have hC : Integrable (fun t : ℝ => 8 * Real.exp (-(t ^ 2) / 2))
      (gaussianReal 0 (5 / 3 : NNReal)) :=
    he.const_mul 8
  have hD : Integrable (fun t : ℝ => 4 * t ^ 2) (gaussianReal 0 (5 / 3 : NNReal)) :=
    h2.const_mul 4
  rw [integral_sub hB hC, integral_add hA (integrable_const 8), integral_sub h4 hD,
    integral_const_mul, integral_const_mul, integral_const,
    tm_fourth_moment, c1_second_moment_gaussianReal, tm_exp_moment]
  simp
  norm_num

/-- For an even, measurable, integrable `h` and `v ≠ 0`, the integral of `1_{Iio 0} h` against
`gaussianReal 0 v` is half the integral of `h`: the change of variables `t ↦ -t`
(`gaussianReal_map_neg`, `integral_map`) equates the integrals over `Iio 0` and `Ioi 0`, and
`integral_add_compl` with `integral_Ici_eq_integral_Ioi` (no atoms) adds them to the whole. -/
theorem cm_setIntegral_Iio_even (v : NNReal) (hv : v ≠ 0) (h : ℝ → ℝ) (hm : Measurable h)
    (heven : ∀ t, h (-t) = h t) (hint : Integrable h (gaussianReal 0 v)) :
    ∫ t, Set.indicator (Set.Iio 0) h t ∂(gaussianReal 0 v) =
      (1 / 2) * ∫ t, h t ∂(gaussianReal 0 v) := by
  have : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  set γ := gaussianReal 0 v with hγ
  have hneg : γ.map (fun x => -x) = γ := by rw [hγ, gaussianReal_map_neg, neg_zero]
  have hsym : ∫ t in Set.Iio (0 : ℝ), h t ∂γ = ∫ t in Set.Ioi (0 : ℝ), h t ∂γ := by
    calc ∫ t in Set.Iio (0 : ℝ), h t ∂γ
        = ∫ t, Set.indicator (Set.Iio (0 : ℝ)) h t ∂γ :=
          (integral_indicator measurableSet_Iio).symm
      _ = ∫ t, Set.indicator (Set.Iio (0 : ℝ)) h t ∂(γ.map (fun x => -x)) := by rw [hneg]
      _ = ∫ t, Set.indicator (Set.Iio (0 : ℝ)) h (-t) ∂γ :=
          integral_map measurable_neg.aemeasurable
            (hm.indicator measurableSet_Iio).aestronglyMeasurable
      _ = ∫ t, Set.indicator (Set.Ioi (0 : ℝ)) h t ∂γ := by
          congr 1
          funext t
          simp only [Set.indicator, Set.mem_Iio, Set.mem_Ioi, neg_lt_zero, heven]
      _ = ∫ t in Set.Ioi (0 : ℝ), h t ∂γ := integral_indicator measurableSet_Ioi
  have hsplit := integral_add_compl (μ := γ) (s := Set.Iio (0 : ℝ)) measurableSet_Iio hint
  rw [Set.compl_Iio, integral_Ici_eq_integral_Ioi, ← hsym] at hsplit
  rw [integral_indicator measurableSet_Iio]
  linarith

/-! ## `cm_` (d): the cone integrand on the product space -/

/-- `coneIntegrand q` is the indicator of `{q.1 < 0 ∧ q.2.1^2 + q.2.2^2 < q.1^2}` applied to
`(q.1^2 - q.2.1^2 - q.2.2^2)^2` (`cm_cone_iff`, `cm_det_transverse`). -/
theorem cm_coneIntegrand_eq (q : ℝ × ℝ × ℝ) :
    coneIntegrand q = Set.indicator {q : ℝ × ℝ × ℝ | q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2}
      (fun q => (q.1 ^ 2 - q.2.1 ^ 2 - q.2.2 ^ 2) ^ 2) q := by
  unfold coneIntegrand
  simp only [Set.indicator, Set.mem_ofPred_eq, cm_cone_iff, cm_det_transverse]

/-- `coneIntegrand` is measurable: by `cm_coneIntegrand_eq` it is a polynomial times the
indicator of a measurable set (`measurableSet_lt`, `Measurable.indicator`). -/
theorem cm_measurable_coneIntegrand : Measurable coneIntegrand := by
  have e : coneIntegrand = fun q => Set.indicator
      {q : ℝ × ℝ × ℝ | q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2}
      (fun q => (q.1 ^ 2 - q.2.1 ^ 2 - q.2.2 ^ 2) ^ 2) q := funext cm_coneIntegrand_eq
  rw [e]
  have h1 : MeasurableSet {q : ℝ × ℝ × ℝ | q.1 < 0} :=
    measurableSet_lt measurable_fst measurable_const
  have h2 : MeasurableSet {q : ℝ × ℝ × ℝ | q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2} :=
    measurableSet_lt (by fun_prop) (by fun_prop)
  exact Measurable.indicator (by fun_prop) (h1.inter h2)

/-- `|coneIntegrand q| ≤ q.1^4`: inside the cone `0 ≤ q.1^2 - q.2.1^2 - q.2.2^2 ≤ q.1^2`,
outside it the integrand is `0`. -/
theorem cm_coneIntegrand_bound (q : ℝ × ℝ × ℝ) : ‖coneIntegrand q‖ ≤ q.1 ^ 4 := by
  rw [cm_coneIntegrand_eq]
  by_cases h : q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2
  · rw [Set.indicator_of_mem
      (show q ∈ {q : ℝ × ℝ × ℝ | q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2} from h), Real.norm_eq_abs,
      abs_of_nonneg (sq_nonneg _)]
    have h0 : 0 ≤ q.1 ^ 2 - q.2.1 ^ 2 - q.2.2 ^ 2 := by linarith [h.2]
    have h1 : q.1 ^ 2 - q.2.1 ^ 2 - q.2.2 ^ 2 ≤ q.1 ^ 2 := by
      nlinarith [sq_nonneg q.2.1, sq_nonneg q.2.2]
    calc (q.1 ^ 2 - q.2.1 ^ 2 - q.2.2 ^ 2) ^ 2 ≤ (q.1 ^ 2) ^ 2 := pow_le_pow_left₀ h0 h1 2
      _ = q.1 ^ 4 := by ring
  · rw [Set.indicator_of_notMem
      (show q ∉ {q : ℝ × ℝ × ℝ | q.1 < 0 ∧ q.2.1 ^ 2 + q.2.2 ^ 2 < q.1 ^ 2} from h), norm_zero]
    positivity

/-- `coneIntegrand` is integrable against `N(0, 5/3) ⊗ (N(0, 1) ⊗ N(0, 1))`: it is dominated by
`q ↦ q.1^4`, integrable as a function of the first coordinate (`cm_integrable_pow_four`,
`Integrable.comp_fst`, `Integrable.mono'`). -/
theorem cm_integrable_coneIntegrand :
    Integrable coneIntegrand ((gaussianReal 0 (5 / 3 : NNReal)).prod
      ((gaussianReal 0 1).prod (gaussianReal 0 1))) :=
  Integrable.mono' ((cm_integrable_pow_four (5 / 3 : NNReal)).comp_fst _)
    cm_measurable_coneIntegrand.aestronglyMeasurable
    (Filter.Eventually.of_forall cm_coneIntegrand_bound)

/-- For `s < 0` the slice `p ↦ coneIntegrand (s, p)` is the closed-disk integrand
`1{|p|^2 ≤ s^2} (s^2 - |p|^2)^2` (on the circle `|p|^2 = s^2` both sides vanish). -/
theorem cm_coneIntegrand_neg (s : ℝ) (p : ℝ × ℝ) (hs : s < 0) :
    coneIntegrand (s, p) = Set.indicator {q : ℝ × ℝ | q.1 ^ 2 + q.2 ^ 2 ≤ s ^ 2}
      (fun q => (s ^ 2 - q.1 ^ 2 - q.2 ^ 2) ^ 2) p := by
  rw [cm_coneIntegrand_eq]
  simp only [Set.indicator, Set.mem_ofPred_eq]
  split_ifs with h1 h2 h2
  · rfl
  · exact absurd h1.2.le h2
  · have he : p.1 ^ 2 + p.2 ^ 2 = s ^ 2 := le_antisymm h2 (not_lt.mp (fun h => h1 ⟨hs, h⟩))
    have he' : s ^ 2 - p.1 ^ 2 - p.2 ^ 2 = 0 := by linarith
    rw [he']
    norm_num
  · rfl

/-- For `s ≥ 0` the slice `p ↦ coneIntegrand (s, p)` vanishes (the cone needs `s < 0`). -/
theorem cm_coneIntegrand_nonneg (s : ℝ) (p : ℝ × ℝ) (hs : 0 ≤ s) : coneIntegrand (s, p) = 0 := by
  rw [cm_coneIntegrand_eq, Set.indicator_of_notMem]
  simp only [Set.mem_ofPred_eq, not_and]
  intro h
  linarith

/-- The inner integral: `∫ coneIntegrand (s, ·) d(N(0, 1) ⊗ N(0, 1))` is
`1_{Iio 0}(s) (s^4 - 4s^2 + 8 - 8 exp (-s^2/2))` (`cm_coneIntegrand_neg` with
`cm_gauss2_integral` at `a = s^2`, and `cm_coneIntegrand_nonneg`). -/
theorem cm_inner (s : ℝ) :
    ∫ p, coneIntegrand (s, p) ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) =
      Set.indicator (Set.Iio 0)
        (fun t : ℝ => t ^ 4 - 4 * t ^ 2 + 8 - 8 * Real.exp (-(t ^ 2) / 2)) s := by
  rcases lt_or_ge s 0 with hs | hs
  · rw [Set.indicator_of_mem (Set.mem_Iio.mpr hs)]
    refine (integral_congr_ae (Filter.Eventually.of_forall
      (fun p => cm_coneIntegrand_neg s p hs))).trans ?_
    rw [cm_gauss2_integral (s ^ 2) (sq_nonneg s)]
    ring
  · rw [Set.indicator_of_notMem (by simpa using hs)]
    refine (integral_congr_ae (Filter.Eventually.of_forall
      (fun p => cm_coneIntegrand_nonneg s p hs))).trans ?_
    simp

end Aux

/-! ## The `m = 2` cone moment identity for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.ConeMomentD2Identity` (claim `side24-coefficient`), for every inhabitant `I : ConeMomentInterface`.
Method: transport along the law field `I.sxy_law` (`integral_map` after `Side24Interfaces.Aux.tm_aemeasurable_triple`), Fubini (`integral_prod`, with integrability from the bound `|coneIntegrand q| ≤ q.1^4`, `Aux.cm_integrable_coneIntegrand`), the cone identification `NegDefCone (s, x, y) ↔ s < 0 ∧ x^2 + y^2 < s^2` and `det A = s^2 - x^2 - y^2` (`Aux.cm_cone_iff`, `Aux.cm_det_transverse`), the inner planar Gaussian integral `s^4 - 4s^2 + 8 - 8 exp (-s^2/2)` for `s < 0` and `0` for `s ≥ 0` (`Aux.cm_inner`: `prod_withDensity`, polar coordinates `integral_comp_polarCoord_symm`, the substitution `z = r^2` onto `Side24Elementary.elementaryConeIntegral`), the symmetry halving of the even bracket under `gaussianReal 0 (5/3)` (`Aux.cm_setIntegral_Iio_even`), and the moments `25/3`, `5/3`, `sqrt (3/8)` (`Aux.cm_integral_bracket`, from `Side24Interfaces.Aux.tm_fourth_moment`, `c1_second_moment_gaussianReal`, `tm_exp_moment`).
Does not establish: the informal identity `D_2 = E[det(A)^2 1{A<0}] = (1/2)[25/3 - 20/3 + 8 - 8 sqrt(3/8)]` of the source as the source's object (the alignment of `ConeMomentInterface`, `transverseMatrix`, `NegDefCone`, `coneIntegrand` and `gaussianReal` with the source's objects is PENDING_INDEPENDENT_REVIEW); the source's Rayleigh integration route or its reduction of the cone to `s < -R` as the source's statements (the helper `Aux.cm_cone_iff` characterises the formal `NegDefCone` only, and `RayleighSquareLaw` is not used); integrability of the sample-space integrand under `I.P` as a separate statement (the integral is transported by `integral_map`, which needs only a.e.-measurability); that `D_2` is the parent's cone moment `D_u` of (15.1) for the reference law; that `ConeMomentInterface` has an inhabitant, or that its laws are the conditional law `A | V = 0` of the reference field; the value `29/6 - sqrt 6` (`ConeMomentD2Algebra`, not used here); any enclosure of the coefficient; the disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem coneMomentD2Identity (I : ConeMomentInterface) : UniversalLaw.Spec.Side24.ConeMomentD2Identity I := by
  unfold UniversalLaw.Spec.Side24.ConeMomentD2Identity
  let := I.mΩ
  have := I.isProb
  calc ∫ ω, coneIntegrand (I.s ω, I.x ω, I.y ω) ∂I.P
      = ∫ q, coneIntegrand q ∂(Measure.map (fun ω => (I.s ω, I.x ω, I.y ω)) I.P) :=
        (integral_map (UniversalLaw.Proofs.Side24Interfaces.Aux.tm_aemeasurable_triple I)
          Aux.cm_measurable_coneIntegrand.aestronglyMeasurable).symm
    _ = ∫ q, coneIntegrand q ∂((gaussianReal 0 (5 / 3 : NNReal)).prod
          ((gaussianReal 0 1).prod (gaussianReal 0 1))) := by rw [I.sxy_law]
    _ = ∫ s, ∫ p, coneIntegrand (s, p) ∂((gaussianReal 0 1).prod (gaussianReal 0 1))
          ∂(gaussianReal 0 (5 / 3 : NNReal)) :=
        integral_prod _ Aux.cm_integrable_coneIntegrand
    _ = ∫ s, Set.indicator (Set.Iio 0)
          (fun t : ℝ => t ^ 4 - 4 * t ^ 2 + 8 - 8 * Real.exp (-(t ^ 2) / 2)) s
          ∂(gaussianReal 0 (5 / 3 : NNReal)) := by
        congr 1
        funext s
        exact Aux.cm_inner s
    _ = (1 / 2) * ∫ t, (t ^ 4 - 4 * t ^ 2 + 8 - 8 * Real.exp (-(t ^ 2) / 2))
          ∂(gaussianReal 0 (5 / 3 : NNReal)) :=
        Aux.cm_setIntegral_Iio_even (5 / 3 : NNReal) (by norm_num) _ (by fun_prop)
          (fun t => by simp only [neg_sq, show (-t) ^ 4 = t ^ 4 by ring])
          (Aux.cm_integrable_bracket _)
    _ = (1 / 2) * (25 / 3 - 20 / 3 + 8 - 8 * Real.sqrt (3 / 8)) := by
        rw [Aux.cm_integral_bracket]

end UniversalLaw.Proofs.Side24ConeMoment
