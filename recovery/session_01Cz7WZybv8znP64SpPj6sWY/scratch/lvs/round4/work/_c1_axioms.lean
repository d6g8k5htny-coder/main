import Specifications.Side24

/-!
# SIDE24 interface proofs: the `m = 1` cone moment identity (`ConeMomentD1Identity I`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux` (prefix `c1_`). The route,
for an arbitrary inhabitant `I : ConeMomentInterface` (existence of an inhabitant is neither
asserted nor used): (1) `I.A` is a.e.-measurable, because at this Mathlib revision the image
measure of a non-a.e.-measurable map under a nonzero measure is a Dirac measure
(`Measure.map_of_not_aemeasurable_of_ne_zero`), which charges a singleton, while the Gaussian
`gaussianReal 0 (8/3)` has `NullSingletonClass`; (2) `integral_map` transports the integral of
the indicator integrand along the law `I.A_law` to `gaussianReal 0 (8/3)`, and
`integral_indicator` makes it the set integral of `a ↦ a^2` over `Iio 0`; (3) the symmetry
`gaussianReal_map_neg` (with `neg_zero`) and a second `integral_map` along `x ↦ -x` give
`∫_{Iio 0} a^2 = ∫_{Ioi 0} a^2`; (4) `integral_add_compl` with `Set.compl_Iio` and
`integral_Ici_eq_integral_Ioi` splits the full second moment, which equals the variance `8/3`
by `variance_fun_id_gaussianReal` and `variance_eq_integral`; so the negative half is
`(8/3)/2 = 4/3 = coneMomentD1`. Only the field `A_law` and the instance `isProb` of the
interface are used; the fields `s`, `x`, `y`, `sxy_law` play no role.

Nothing here changes any register label; the disposition of `side24-coefficient`
(REVIEWED_SCOPED) is transcribed in the register, not derived here.
-/

namespace UniversalLaw.Proofs.Side24Interfaces

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

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

end UniversalLaw.Proofs.Side24Interfaces

#print axioms UniversalLaw.Proofs.Side24Interfaces.coneMomentD1Identity
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_aemeasurable_of_map_eq_gaussian
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_measurable_indicator_sq
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_second_moment_gaussianReal
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_integrable_sq_gaussianReal
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_setIntegral_sq_Iio_eq_Ioi
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_setIntegral_sq_Iio
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_integral_indicator_eq
