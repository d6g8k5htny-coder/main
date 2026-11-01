import Mathlib.Probability.HasCondDistrib
import Mathlib.MeasureTheory.Constructions.Cylinders

/-!
# A field-valued regression kernel, specified at every pin value

`GaussianRegressionKernel` builds the canonical conditional law for a finite-dimensional
residual `Fin k → ℝ`. The tether proposal (main `docs/ELDER_EVENT_TETHER_20261031.md`, §3 T2
and §7 step 1) needs the same object for a *field*: a residual that is a whole function
`T → ℝ` on an arbitrary index set `T`, with the product σ-algebra. This module supplies it.

* `reconstructionKernel` is, for any measurable reconstruction `g`, the kernel
  `x ↦ (law R).map (g (x, ·))`. It is defined at every observation `x`, including
  observations of probability zero such as prescribed pins.
* `independent_reconstruction_hasCondDistrib`: if the observation `X` and the residual `R` are
  independent, then `g (X, R)` has that kernel as a conditional law given `X`.
* `fieldRegressionKernel` and `fieldRegression_hasCondDistrib` specialize this to
  `f = R + mY + c (X - mX)`, with residual field `R : Ω → T → ℝ`, a measurable
  correction `c : (Fin n → ℝ) → T → ℝ` and mean field `mY : T → ℝ`. This is the shape of
  display (12) in `PERSISTENCE_BRIDGE.md` §4.1.
* `indepFun_pi_of_finite`: independence of `X` from the whole residual field follows from
  independence of `X` from every finite restriction `R|_S`, through the π-system of
  measurable cylinders.

What this does NOT establish: that the residual of an actual Gaussian field is independent of
the pin data (for each finite family that needs joint Gaussianity and zero cross-covariance,
supplied elsewhere in finite dimension only); any continuity of the kernel in the pins; any
statement about a smooth-function space or its Borel σ-algebra (the product σ-algebra on
`T → ℝ` is coarser); the elder indicator or its measurability; any rate. Scientific effect
NONE; credit 0; OBL OPEN.
-/

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace WholeFieldRegression

/-! ### Inlined from Math- `frontiers/determinant_moment_20261025/ResearchFormalCoreR1/GaussianRegressionKernel.lean`
(blob `face5de8`, Math- main `19a55750`), unchanged except for the namespace. -/

/-- Copying the first coordinate and retaining an independent second coordinate. -/
lemma regression_compProd_id_const {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SFinite μ] [SFinite ν] :
    μ ⊗ₘ (Kernel.id ×ₖ Kernel.const α ν) =
      (μ.prod ν).map (fun p : α × β => (p.1, p)) := by
  ext s hs
  rw [Measure.compProd_apply hs, Measure.map_apply (by fun_prop) hs,
    Measure.prod_apply (s := (fun p : α × β => (p.1, p)) ⁻¹' s) (hs.preimage (by fun_prop))]
  simp_rw [Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply, Measure.dirac_prod]
  congr with x
  rw [Measure.map_apply (by fun_prop) (measurable_prodMk_left hs)]
  rfl

/-- An independent residual paired with the observation has the explicit product kernel. -/
lemma regression_pair_hasCondDistrib {Ω α β : Type*} [MeasurableSpace Ω]
    [MeasurableSpace α] [MeasurableSpace β] {P : Measure Ω} [IsProbabilityMeasure P]
    {X : Ω → α} {R : Ω → β} (hX : AEMeasurable X P) (hR : AEMeasurable R P)
    (hXR : IndepFun X R P) :
    HasCondDistrib (fun ω => (X ω, R ω)) X
      (Kernel.id ×ₖ Kernel.const α (P.map R)) P := by
  refine ⟨hX.prodMk (hX.prodMk hR), ?_⟩
  calc
    P.map (fun ω => (X ω, (X ω, R ω))) =
        (P.map (fun ω => (X ω, R ω))).map (fun p : α × β => (p.1, p)) := by
      rw [AEMeasurable.map_map_of_aemeasurable (by fun_prop) (hX.prodMk hR)]
      rfl
    _ = ((P.map X).prod (P.map R)).map (fun p : α × β => (p.1, p)) := by
      rw [hXR.map_prod_eq_prod_map_map hX hR]
    _ = (P.map X) ⊗ₘ (Kernel.id ×ₖ Kernel.const α (P.map R)) :=
      (regression_compProd_id_const (P.map X) (P.map R)).symm


/-- The conditional kernel of a measurable reconstruction from an independent residual. -/
noncomputable def reconstructionKernel {α β E : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [MeasurableSpace E] (ν : Measure β) (g : α × β → E) : Kernel α E :=
  ((Kernel.id : Kernel α α) ×ₖ Kernel.const α ν).map g

theorem reconstructionKernel_isMarkov {α β E : Type*} [MeasurableSpace α]
    [MeasurableSpace β] [MeasurableSpace E] (ν : Measure β) [IsProbabilityMeasure ν]
    {g : α × β → E} (hg : Measurable g) :
    IsMarkovKernel (reconstructionKernel ν g) := by
  unfold reconstructionKernel
  exact Kernel.IsMarkovKernel.map _ hg

/-- The kernel's literal value at every observation: a pushforward of the residual law. -/
lemma reconstructionKernel_apply {α β E : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [MeasurableSpace E] (ν : Measure β) [SFinite ν] {g : α × β → E} (hg : Measurable g)
    (x : α) :
    reconstructionKernel ν g x = ν.map (fun r => g (x, r)) := by
  rw [reconstructionKernel, Kernel.map_apply _ hg, Kernel.prod_apply, Kernel.id_apply,
    Kernel.const_apply, Measure.dirac_prod, Measure.map_map hg (by fun_prop)]
  rfl

/-- An independent residual reconstructed measurably has the explicit conditional kernel. -/
theorem independent_reconstruction_hasCondDistrib {Ω α β E : Type*} [MeasurableSpace Ω]
    [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace E]
    {P : Measure Ω} [IsProbabilityMeasure P] {X : Ω → α} {R : Ω → β}
    (hX : AEMeasurable X P) (hR : AEMeasurable R P) (hXR : IndepFun X R P)
    {g : α × β → E} (hg : Measurable g) :
    HasCondDistrib (fun ω => g (X ω, R ω)) X (reconstructionKernel (P.map R) g) P :=
  (regression_pair_hasCondDistrib hX hR hXR).comp_left hg

/-! ### The field specialization -/

/-- Field reconstruction: residual field plus mean field plus a pin-dependent correction. -/
def fieldReconstruction {n : ℕ} {T : Type*} (c : (Fin n → ℝ) → T → ℝ)
    (mX : Fin n → ℝ) (mY : T → ℝ) (p : (Fin n → ℝ) × (T → ℝ)) : T → ℝ :=
  fun t => p.2 t + mY t + c (p.1 - mX) t

lemma fieldReconstruction_measurable {n : ℕ} {T : Type*} {c : (Fin n → ℝ) → T → ℝ}
    (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ) :
    Measurable (fieldReconstruction c mX mY) := by
  refine measurable_pi_iff.mpr (fun t => ?_)
  have h2 : Measurable fun p : (Fin n → ℝ) × (T → ℝ) => p.2 t :=
    (measurable_pi_apply t).comp measurable_snd
  have h3 : Measurable fun p : (Fin n → ℝ) × (T → ℝ) => c (p.1 - mX) t :=
    (measurable_pi_apply t).comp (hc.comp (measurable_fst.sub_const mX))
  exact (h2.add_const (mY t)).add h3

/-- The field regression kernel, specified at every pin value `x`. -/
noncomputable def fieldRegressionKernel {n : ℕ} {T : Type*} (ν : Measure (T → ℝ))
    (c : (Fin n → ℝ) → T → ℝ) (mX : Fin n → ℝ) (mY : T → ℝ) :
    Kernel (Fin n → ℝ) (T → ℝ) :=
  reconstructionKernel ν (fieldReconstruction c mX mY)

lemma fieldRegressionKernel_apply {n : ℕ} {T : Type*} (ν : Measure (T → ℝ)) [SFinite ν]
    {c : (Fin n → ℝ) → T → ℝ} (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ)
    (x : Fin n → ℝ) :
    fieldRegressionKernel ν c mX mY x =
      ν.map (fun r t => r t + mY t + c (x - mX) t) :=
  reconstructionKernel_apply ν (fieldReconstruction_measurable hc mX mY) x

/-- Display (12) in shape: given independence of pins and residual field, the field
`R + mY + c (X - mX)` has the field regression kernel as a conditional law given `X`. -/
theorem fieldRegression_hasCondDistrib {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {n : ℕ} {T : Type*}
    {X : Ω → Fin n → ℝ} {R : Ω → T → ℝ}
    (hX : AEMeasurable X P) (hR : AEMeasurable R P) (hXR : IndepFun X R P)
    {c : (Fin n → ℝ) → T → ℝ} (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ) :
    HasCondDistrib (fun ω t => R ω t + mY t + c (X ω - mX) t) X
      (fieldRegressionKernel (P.map R) c mX mY) P :=
  independent_reconstruction_hasCondDistrib hX hR hXR
    (fieldReconstruction_measurable hc mX mY)

/-! ### From finite restrictions to the whole field -/

/-- Independence of the pins from the whole residual field (product σ-algebra on `T → ℝ`)
follows from independence from every finite restriction `R|_S`: the measurable cylinders form
a generating π-system. This is the "countable determining family" step of (12), stated for
the product σ-algebra; it says nothing about a Borel σ-algebra on a smooth-function space. -/
theorem indepFun_pi_of_finite {Ω α : Type*} [MeasurableSpace Ω] [mα : MeasurableSpace α]
    {P : Measure Ω} [IsProbabilityMeasure P] {T : Type*}
    {X : Ω → α} {R : Ω → T → ℝ} (hX : Measurable X) (hR : Measurable R)
    (hfin : ∀ S : Finset T, IndepFun X (fun ω => S.restrict (R ω)) P) :
    IndepFun X R P := by
  rw [IndepFun_iff_Indep]
  refine IndepSets.indep (p1 := {s | MeasurableSet[MeasurableSpace.comap X mα] s})
    (p2 := Set.preimage R '' measurableCylinders (fun _ : T => ℝ))
    hX.comap_le hR.comap_le (@MeasurableSpace.isPiSystem_measurableSet Ω (MeasurableSpace.comap X mα))
    (isPiSystem_measurableCylinders.comap R)
    (@MeasurableSpace.generateFrom_measurableSet Ω (MeasurableSpace.comap X mα)).symm
    (by rw [← generateFrom_measurableCylinders, MeasurableSpace.comap_generateFrom]) ?_
  rw [IndepSets_iff]
  rintro t1 t2 ht1 ⟨C, hC, rfl⟩
  obtain ⟨s', hs', rfl⟩ := ht1
  obtain ⟨S, U, hU, rfl⟩ := (mem_measurableCylinders C).1 hC
  have key := (indepFun_iff_measure_inter_preimage_eq_mul.1 (hfin S)) s' U hs' hU
  simpa [cylinder, Set.preimage_preimage] using key

/-- Combined form: finite-restriction independence gives the field regression kernel. -/
theorem fieldRegression_hasCondDistrib_of_finite {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {n : ℕ} {T : Type*}
    {X : Ω → Fin n → ℝ} {R : Ω → T → ℝ} (hX : Measurable X) (hR : Measurable R)
    (hfin : ∀ S : Finset T, IndepFun X (fun ω => S.restrict (R ω)) P)
    {c : (Fin n → ℝ) → T → ℝ} (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ) :
    HasCondDistrib (fun ω t => R ω t + mY t + c (X ω - mX) t) X
      (fieldRegressionKernel (P.map R) c mX mY) P :=
  fieldRegression_hasCondDistrib hX.aemeasurable hR.aemeasurable
    (indepFun_pi_of_finite hX hR hfin) hc mX mY

end WholeFieldRegression
