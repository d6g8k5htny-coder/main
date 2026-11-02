import SourceNormalForm

noncomputable section
set_option backward.isDefEq.respectTransparency false
open Filter Set Matrix SourceCoordinates MorseCongruence ChartedSpace IsManifold SourceNormalForm
open scoped Topology ContDiff Manifold Matrix.Norms.Frobenius
namespace SourceCriticalEvent
attribute [local instance] I4TorusC4.actualChartedSpace

/-- Canonical evaluation is a norm-preserving equivalence, not merely a CLE. -/
def oneReal : E 1 ≃ₗᵢ[ℝ] ℝ where
  toLinearEquiv := (PiLp.equivOfUnique 2 ℝ (fun _ : Fin 1 => ℝ)).toLinearEquiv
  norm_map' v := by
    change ‖v (0 : Fin 1)‖ = ‖v‖
    apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
    rw [Real.norm_eq_abs, sq_abs, EuclideanSpace.real_norm_sq_eq]
    simp

@[simp] theorem oneReal_apply (v : E 1) : oneReal v = v 0 := rfl

theorem oneReal_symm (r : ℝ) : oneReal.symm r = EuclideanSpace.single 0 r := by
  apply oneReal.injective
  rw [oneReal.apply_symm_apply, oneReal_apply]
  simp

section Transport
variable {A B F M : Type*} [NormedAddCommGroup A] [NormedSpace ℝ A]
  [NormedAddCommGroup B] [NormedSpace ℝ B]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [TopologicalSpace M]

/-- Ordinary product isometry. Energy preservation is proved separately below. -/
def prodIso (a : A ≃ₗᵢ[ℝ] B) : A × F ≃ₗᵢ[ℝ] B × F where
  toLinearEquiv := a.toLinearEquiv.prodCongr (LinearEquiv.refl ℝ F)
  norm_map' z := by
    change max ‖a z.1‖ ‖z.2‖ = max ‖z.1‖ ‖z.2‖
    rw [a.norm_map]

@[simp] theorem prodIso_apply (a : A ≃ₗᵢ[ℝ] B) (z : A × F) :
    prodIso a z = (a z.1, z.2) := rfl

@[simp] theorem prodIso_symm_apply (a : A ≃ₗᵢ[ℝ] B) (z : B × F) :
    (prodIso a).symm z = (a.symm z.1, z.2) := rfl

theorem prodIso_quadratic (a : A ≃ₗᵢ[ℝ] B) (z : A × F) :
    MorseCone.quadratic (prodIso a z) = MorseCone.quadratic z := by
  simp [MorseCone.quadratic, a.norm_map]

theorem prodIso_energy (a : A ≃ₗᵢ[ℝ] B) (z : A × F) :
    ‖(prodIso a z).1‖^2 + ‖(prodIso a z).2‖^2 = ‖z.1‖^2 + ‖z.2‖^2 := by
  simp [a.norm_map]

theorem prodIso_radius (a : A ≃ₗᵢ[ℝ] B) (z : A × F) (ρ : ℝ) :
    prodIso a z ∈ MorseCone.radiusBall ρ ↔ z ∈ MorseCone.radiusBall ρ := by
  simp [MorseCone.radiusBall, a.norm_map]

/-- Transport every field of D, using the full coordinate homeomorphism. -/
def transport (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    MorseCone.NormalForm M B F where
  f := D.f
  p := D.p
  c := D.c
  ρ := D.ρ
  hρ := D.hρ
  chart := D.chart.trans (prodIso a).toHomeomorph.toOpenPartialHomeomorph
  center_mem := ⟨D.center_mem, Set.mem_univ _⟩
  center_eq := by
    change prodIso a (D.chart D.p) = 0
    rw [D.center_eq]
    exact map_zero _
  ball_target := by
    intro z hz
    change z ∈ Set.univ ∧ (prodIso a).symm z ∈ D.chart.target
    refine ⟨Set.mem_univ _, D.ball_target ?_⟩
    have h := prodIso_radius a ((prodIso a).symm z) D.ρ
    rw [(prodIso a).apply_symm_apply] at h
    exact h.mp hz
  equation := by
    intro x hx
    change D.f x = D.c + MorseCone.quadratic (prodIso a (D.chart x))
    rw [prodIso_quadratic]
    exact D.equation x hx.1

theorem transport_fields (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    (transport D a).f = D.f ∧ (transport D a).p = D.p ∧
      (transport D a).c = D.c ∧ (transport D a).ρ = D.ρ := ⟨rfl,rfl,rfl,rfl⟩

theorem transport_chart (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    (transport D a).chart = D.chart.trans (prodIso a).toHomeomorph.toOpenPartialHomeomorph := rfl

theorem transport_source (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    (transport D a).chart.source = D.chart.source := by
  change D.chart.source ∩ D.chart ⁻¹' Set.univ = D.chart.source
  simp

theorem transport_target (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) (z : B × F) :
    z ∈ (transport D a).chart.target ↔ (prodIso a).symm z ∈ D.chart.target := by
  change (z ∈ Set.univ ∧ (prodIso a).symm z ∈ D.chart.target) ↔ _
  simp

theorem transport_apply (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) (x : M) :
    (transport D a).chart x = prodIso a (D.chart x) := rfl

theorem transport_symm (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) (z : B × F) :
    (transport D a).chart.symm z = D.chart.symm ((prodIso a).symm z) := rfl

theorem transport_roundtrips (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    (∀ x ∈ (transport D a).chart.source,
      (transport D a).chart.symm ((transport D a).chart x) = x) ∧
    (∀ z ∈ (transport D a).chart.target,
      (transport D a).chart ((transport D a).chart.symm z) = z) :=
  ⟨fun _ hx => (transport D a).chart.left_inv hx,
   fun _ hz => (transport D a).chart.right_inv hz⟩

theorem transport_closedLevel (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    MorseCone.closedLevel (transport D a) = MorseCone.closedLevel D := rfl

theorem transport_oldSet (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    MorseCone.oldSet (transport D a) = MorseCone.oldSet D := rfl

theorem transport_patchSet (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B) :
    MorseCone.patchSet (transport D a) = MorseCone.patchSet D := by
  ext x
  change (x.val ∈ (transport D a).chart.source ∧
    (transport D a).chart x.val ∈ MorseCone.radiusBall D.ρ) ↔
    (x.val ∈ D.chart.source ∧ D.chart x.val ∈ MorseCone.radiusBall D.ρ)
  rw [transport_source, transport_apply, prodIso_radius]

/-- C2 on the actual source and inverse target, not on all ambient points. -/
theorem transport_C2 {H : Type*} [NormedAddCommGroup H] [NormedSpace ℝ H]
    [ChartedSpace H M] (D : MorseCone.NormalForm M A F) (a : A ≃ₗᵢ[ℝ] B)
    (hf : ContMDiffOn 𝓘(ℝ,H) 𝓘(ℝ,A × F) 2 D.chart D.chart.source)
    (hi : ContMDiffOn 𝓘(ℝ,A × F) 𝓘(ℝ,H) 2 D.chart.symm D.chart.target) :
    ContMDiffOn 𝓘(ℝ,H) 𝓘(ℝ,B × F) 2 (transport D a).chart (transport D a).chart.source ∧
    ContMDiffOn 𝓘(ℝ,B × F) 𝓘(ℝ,H) 2 (transport D a).chart.symm (transport D a).chart.target := by
  have hQ : ContMDiff 𝓘(ℝ,A × F) 𝓘(ℝ,B × F) 2 (prodIso (F := F) a) :=
    (prodIso (F := F) a).toContinuousLinearEquiv.toContinuousLinearMap.contMDiff
  have hQi : ContMDiff 𝓘(ℝ,B × F) 𝓘(ℝ,A × F) 2 (prodIso (F := F) a).symm :=
    (prodIso (F := F) a).symm.toContinuousLinearEquiv.toContinuousLinearMap.contMDiff
  constructor
  · rw [transport_source]
    exact hQ.comp_contMDiffOn hf
  · exact hi.comp hQi.contMDiffOn (fun z hz => (transport_target D a z).mp hz)

end Transport

def oneProd (F : Type*) [NormedAddCommGroup F] [NormedSpace ℝ F] :
    E 1 × F ≃ₗᵢ[ℝ] ℝ × F := prodIso oneReal

theorem transport_one_foot {M F : Type*} [TopologicalSpace M] [T1Space M]
    [NormedAddCommGroup F] [NormedSpace ℝ F] (D : MorseCone.NormalForm M (E 1) F) :
    ∃ a b : ZerothHomotopy (MorseCone.oldSet D), ∀ q,
      MorseOpenGluing.Foot (MorseCone.oldSet D) (MorseCone.patchSet D) q ↔ q=a ∨ q=b := by
  simpa only [transport_oldSet, transport_patchSet] using MorseCone.one_foot (transport D oneReal)

/-- This is an Event on the original literal inclusion; equal classes remain neutral. -/
def transport_one_event {M F : Type*} [TopologicalSpace M] [T1Space M]
    [NormedAddCommGroup F] [NormedSpace ℝ F] (D : MorseCone.NormalForm M (E 1) F) :
    ElementaryHistory.Event (MorseOpenGluing.inclusionMap (MorseCone.oldSet D)) :=
  MorseCone.one_event (transport D oneReal)

theorem actual_source_critical_event {n : ℕ} {L : ℝ} [Fact (0 < L)]
    (omega : I4Source.SourceMode n → ℝ)
    (hmajor : Summable (fun m => I4Source.modeMajorant n L m * |omega m|))
    (x0 : X n) (hdet : (PhysicalSpectral.hessian (L := L) omega x0).det ≠ 0)
    (hcrit : fderiv ℝ (I4Source.sourceField n L omega) x0 = 0) :
    ∃ (k : ℕ) (hk : k ≤ n) (P : E n ≃L[ℝ] X n) (s : Sym n → Sym n)
      (e : OpenPartialHomeomorph (E n) (E n))
      (D : MorseCone.NormalForm (I4Torus.SourceTorus n L) (E k) (E (n-k))),
      let J := PhysicalSpectral.signSym n k
      let M := TaylorSymLift.sourceSymMatrix (L := L) omega x0 P.toContinuousLinearMap
      M 0 = J ∧ (∀ u, e u = theta J s M u) ∧
      e 0 = 0 ∧ e.symm 0 = 0 ∧ 0 ∈ e.source ∧ 0 ∈ e.target ∧
      HasFDerivAt e (ContinuousLinearMap.id ℝ (E n)) 0 ∧
      ContDiffOn ℝ 2 e e.source ∧ ContDiffOn ℝ 2 e.symm e.target ∧
      D.f = I4Torus.torusSourceField n L omega ∧
      D.p = I4Torus.torusProjection n L x0 ∧ D.c = I4Source.sourceField n L omega x0 ∧
      D.chart = (liftChart (L := L) x0).trans (physicalChart hk x0 P e) ∧
      (∀ q, D.chart q = split hk (e (P.symm (liftChart (L := L) x0 q - x0)))) ∧
      (∀ z, D.chart.symm z = I4Torus.torusProjection n L (x0 + P (e.symm ((split hk).symm z)))) ∧
      ContMDiffOn 𝓘(ℝ, X n) 𝓘(ℝ, E k × E (n-k)) 2 D.chart D.chart.source ∧
      ContMDiffOn 𝓘(ℝ, E k × E (n-k)) 𝓘(ℝ, X n) 2 D.chart.symm D.chart.target ∧
      (k = 1 → (∃ a b : ZerothHomotopy (MorseCone.oldSet D), ∀ q,
        MorseOpenGluing.Foot (MorseCone.oldSet D) (MorseCone.patchSet D) q ↔ q=a ∨ q=b) ∧
        Nonempty (ElementaryHistory.Event (MorseOpenGluing.inclusionMap (MorseCone.oldSet D)))) := by
  obtain ⟨k, hk, P, s, e, D, hM, he, he0, hi0, hs, ht, hd, heC, hiC,
    hDf, hDp, hDc, hchart, hforward, hinverse, hDC, hDiC⟩ :=
    actual_source_normal_form omega hmajor x0 hdet hcrit
  refine ⟨k, hk, P, s, e, D, hM, he, he0, hi0, hs, ht, hd, heC, hiC,
    hDf, hDp, hDc, hchart, hforward, hinverse, hDC, hDiC, ?_⟩
  intro hk1
  subst k
  exact ⟨transport_one_foot D, ⟨transport_one_event D⟩⟩

end SourceCriticalEvent
