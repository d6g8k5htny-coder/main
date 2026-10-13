import Specifications.Side24

/-!
# SIDE24 interface proofs: the Section 1 trace moments (`TraceMoments`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux` (tag `tm_`). The route: the
interface field `sxy_law` identifies the law of the triple `(s, x, y)` with a product of
non-degenerate Gaussians; a map whose image measure under a nonzero measure is such a law is
a.e.-measurable (otherwise `Measure.map` is a Dirac mass, which charges a singleton, while the
product Gaussian does not: `Measure.map_of_not_aemeasurable_of_ne_zero` versus
`NullSingletonClass`). The first marginal of the product law is then `gaussianReal 0 (5/3)`
(`AEMeasurable.map_map_of_aemeasurable`, `Measure.map_fst_prod`), and `integral_map` transports
each of the three integrals to that law, where they are computed: `E t^2 = 5/3` from
`variance_fun_id_gaussianReal`; `E t^4 = 3 (5/3)^2 = 25/3` from `iteratedDeriv_mgf_zero` with
the explicit fourth derivative of `exp (c t^2 / 2)` at `0`; `E exp (-t^2/2) = sqrt (3/8)` from
`integral_gaussianReal_eq_integral_smul` and `integral_gaussian`. Only the joint law and the
probability-measure instance of the interface are used; nothing here constructs an inhabitant
of `ConeMomentInterface`.
-/

namespace UniversalLaw.Proofs.Side24Interfaces

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

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

/-! ### (b) Moments of the centred real Gaussian `gaussianReal 0 v` -/

/-- Second moment of `N(0, v)`: `∫ t^2 = v`, from `variance_fun_id_gaussianReal` rewritten
through `variance_eq_integral` (the mean is `0`). -/
theorem tm_second_moment (v : NNReal) : ∫ t, t ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral (by fun_prop)] at h
  simpa using h

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

end Aux

/-! ## The Section 1 trace moments for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.TraceMoments` (claim `side24-coefficient`).
Method: the law of `I.s` is `gaussianReal 0 (5/3)` (`Aux.tm_law_s`: the first marginal of the
interface's product law `sxy_law`, after `Aux.tm_aemeasurable_triple` derives a.e.-measurability
of the triple from the law by the Dirac-versus-`NullSingletonClass` argument); each of the three
integrals is transported to that law by `integral_map` (`Aux.tm_integral_comp_s`) and computed
there: `∫ t^2 = 5/3` (`Aux.tm_second_moment`, from `variance_fun_id_gaussianReal`),
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
    rw [e, Aux.tm_second_moment]
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

end UniversalLaw.Proofs.Side24Interfaces

-- VERIFIER SCRATCH COPY ONLY (not the deliverable).
#check @UniversalLaw.Proofs.Side24Interfaces.traceMoments
#print UniversalLaw.Spec.Side24.TraceMoments
#print UniversalLaw.Proofs.Side24Interfaces.traceMoments
#print axioms UniversalLaw.Proofs.Side24Interfaces.traceMoments
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_hasDerivAt_exp_sq
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_iteratedDeriv_four
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_second_moment
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_fourth_moment
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_exp_moment
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_aemeasurable_triple
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_aemeasurable_s
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_law_s
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.tm_integral_comp_s

example : ∀ I : UniversalLaw.Spec.Side24.ConeMomentInterface, UniversalLaw.Spec.Side24.TraceMoments I :=
  UniversalLaw.Proofs.Side24Interfaces.traceMoments

open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Interfaces.traceMoments | throwError "missing theorem"
  let ty := ci.type
  match ty with
  | .forallE n (.const ``UniversalLaw.Spec.Side24.ConeMomentInterface []) (.app (.const ``UniversalLaw.Spec.Side24.TraceMoments []) (.bvar 0)) bi =>
      logInfo s!"EXACT_TYPE_OK binder={n} explicit={bi.isExplicit}"
  | _ => logInfo s!"EXACT_TYPE_MISMATCH {ty}"
  match ci with
  | .thmInfo _ => logInfo "KIND thm"
  | _ => logInfo "KIND not-thm (REFUSED)"
  for n in [``UniversalLaw.Spec.Side24.TraceMoments, ``UniversalLaw.Spec.Side24.ConeMomentInterface] do
    match env.getModuleIdxFor? n with
    | some idx => logInfo s!"PROVENANCE {n} <- {env.header.moduleNames[idx.toNat]!}"
    | none => logInfo s!"PROVENANCE {n} <- CURRENT FILE (REFUSED)"
  let mut rows : Array String := #[]
  for (n, c) in env.constants.map₂.toList do
    let kind := match c with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp c.type)
    rows := rows.push s!"AUDIT {kind} prop={isP} unsafe={c.isUnsafe} partial={c.isPartial} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"AUDIT TOTAL {rows.size}"
