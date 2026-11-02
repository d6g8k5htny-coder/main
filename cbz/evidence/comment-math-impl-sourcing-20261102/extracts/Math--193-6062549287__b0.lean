import DeterministicSourceC4
import CapFirstExit
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Calculus.ContDiff.Basic

noncomputable section
open I4Foundation I4Weights I4Source I4Torus I4DeterministicC4
open scoped BigOperators RealInnerProductSpace
namespace I1PhysicalFrame

abbrev E (n : ℕ) := EuclideanSpace ℝ (Fin n)

def frame (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) :
    (ℝ × E m) ≃L[ℝ] (Fin (m+1) → ℝ) :=
  let e1 := (LinearEquiv.refl ℝ ℝ).prodCongr (WithLp.linearEquiv 2 ℝ (Fin m → ℝ))
  let e2 := e1.trans (Fin.consLinearEquiv ℝ (fun _ : Fin (m+1) => ℝ))
  let e3 := e2.trans (WithLp.linearEquiv 2 ℝ (Fin (m+1) → ℝ)).symm
  let e4 := e3.trans b.repr.symm.toLinearEquiv
  (e4.trans (WithLp.linearEquiv 2 ℝ (Fin (m+1) → ℝ))).toContinuousLinearEquiv

def affineFrame (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (z0 : Fin (m+1) → ℝ) : (ℝ × E m) ≃ₜ (Fin (m+1) → ℝ) :=
  (frame m b).toHomeomorph.trans (Homeomorph.addLeft z0)

theorem frame_apply (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (t : ℝ) (y : E m) :
    frame m b (t,y) = WithLp.ofLp (b.repr.symm (WithLp.toLp 2 (Fin.cons t (WithLp.ofLp y)))) := by
  rfl

theorem affine_apply (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (z0 : Fin (m+1) → ℝ) (t : ℝ) (y : E m) :
    affineFrame m b z0 (t,y) = z0 + frame m b (t,y) := by
  rfl

theorem affine_midpoint (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (z0 : Fin (m+1) → ℝ) : affineFrame m b z0 (0,0) = z0 := by
  rw [affine_apply]
  change z0 + frame m b 0 = z0
  simp

theorem physical_energy (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (z0 : Fin (m+1) → ℝ) (t : ℝ) (y : E m) :
    ‖WithLp.toLp 2 (affineFrame m b z0 (t,y) - z0)‖ ^ 2 = t^2 + ‖y‖^2 := by
  rw [affine_apply, add_sub_cancel_left, frame_apply, WithLp.toLp_ofLp,
    b.repr.symm.norm_map, EuclideanSpace.real_norm_sq_eq]
  simp [Fin.sum_univ_succ, EuclideanSpace.real_norm_sq_eq]

theorem frame_axial (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) :
    frame m b (1,0) = WithLp.ofLp (b 0) := by
  rw [frame_apply]
  have he : WithLp.toLp 2 (Fin.cons (1:ℝ) (WithLp.ofLp (0:E m))) =
      EuclideanSpace.single (0:Fin (m+1)) 1 := by
    ext i
    refine Fin.cases ?_ (fun j => ?_) i <;> simp
  rw [he, b.repr_symm_single]

theorem frame_transverse (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (y : E m) : frame m b (0,y) = WithLp.ofLp (∑ i, y i • b i.succ) := by
  rw [frame_apply, ← b.sum_repr_symm]
  congr 1
  simp [Fin.sum_univ_succ]

theorem axial_norm (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) :
    ‖WithLp.toLp 2 (frame m b (1,0))‖ = 1 := by
  rw [frame_axial, WithLp.toLp_ofLp]
  exact b.norm_eq_one 0

theorem transverse_norm (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (y : E m) : ‖WithLp.toLp 2 (frame m b (0,y))‖ = ‖y‖ := by
  have h := physical_energy m b 0 0 y
  simp only [affine_apply, zero_add, sub_zero] at h
  nlinarith [norm_nonneg (WithLp.toLp 2 (frame m b (0,y))), norm_nonneg y]

theorem axial_transverse_orthogonal (m : ℕ)
    (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) (y : E m) :
    inner ℝ (WithLp.toLp 2 (frame m b (1,0))) (WithLp.toLp 2 (frame m b (0,y))) = 0 := by
  rw [frame_axial, frame_transverse, WithLp.toLp_ofLp, WithLp.toLp_ofLp]
  simp [inner_sum, inner_smul_right, b.inner_eq_ite, eq_comm]

theorem source_derivative_transfer (m : ℕ)
    (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) (z0 : Fin (m+1) → ℝ)
    {L : ℝ} (hL : L ≠ 0) (coeff : SourceMode (m+1) → ℝ)
    (hcoeff : Summable (fun k => modeMajorant (m+1) L k * |coeff k|))
    (j : ℕ) (hj : j ≤ 4) (q : ℝ × E m) (v : Fin j → (ℝ × E m)) :
    iteratedFDeriv ℝ j (sourceField (m+1) L coeff ∘ affineFrame m b z0) q v =
      iteratedFDeriv ℝ j (sourceField (m+1) L coeff) (affineFrame m b z0 q)
        (fun i => frame m b (v i)) := by
  have hs := sourceField_contDiff_of_majorant (m+1) hL coeff hcoeff
  have hh : ContDiff ℝ 4 (fun x : Fin (m+1) → ℝ => sourceField (m+1) L coeff (z0+x)) :=
    hs.comp (contDiff_const.add contDiff_id)
  have he : sourceField (m+1) L coeff ∘ affineFrame m b z0 =
      (fun x => sourceField (m+1) L coeff (z0+x)) ∘ (frame m b).toContinuousLinearMap := by
    funext x
    rcases x with ⟨t,y⟩
    rw [Function.comp_apply, affine_apply]
    rfl
  rw [he, (frame m b).toContinuousLinearMap.iteratedFDeriv_comp_right hh q
    (by exact_mod_cast hj), iteratedFDeriv_comp_add_left]
  rcases q with ⟨t,y⟩
  simp only [ContinuousMultilinearMap.compContinuousLinearMap_apply,
    ContinuousLinearEquiv.coe_coe, affine_apply]

theorem source_third_blocks (m : ℕ)
    (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) (z0 : Fin (m+1) → ℝ)
    {L : ℝ} (hL : L ≠ 0) (coeff : SourceMode (m+1) → ℝ)
    (hcoeff : Summable (fun k => modeMajorant (m+1) L k * |coeff k|))
    (q : ℝ × E m) (v : Fin 3 → E m) (axial : Fin 3 → Bool) :
    iteratedFDeriv ℝ 3 (sourceField (m+1) L coeff ∘ affineFrame m b z0) q
      (fun i => if axial i then (1,0) else (0,v i)) =
    iteratedFDeriv ℝ 3 (sourceField (m+1) L coeff) (affineFrame m b z0 q)
      (fun i => if axial i then WithLp.ofLp (b 0) else WithLp.ofLp (∑ k, v i k • b k.succ)) := by
  rw [source_derivative_transfer m b z0 hL coeff hcoeff 3 (by omega)]
  congr 1
  funext i
  cases axial i <;> simp [frame_axial, frame_transverse]

theorem source_fourth_blocks (m : ℕ)
    (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) (z0 : Fin (m+1) → ℝ)
    {L : ℝ} (hL : L ≠ 0) (coeff : SourceMode (m+1) → ℝ)
    (hcoeff : Summable (fun k => modeMajorant (m+1) L k * |coeff k|))
    (q : ℝ × E m) (v : Fin 4 → E m) (axial : Fin 4 → Bool) :
    iteratedFDeriv ℝ 4 (sourceField (m+1) L coeff ∘ affineFrame m b z0) q
      (fun i => if axial i then (1,0) else (0,v i)) =
    iteratedFDeriv ℝ 4 (sourceField (m+1) L coeff) (affineFrame m b z0 q)
      (fun i => if axial i then WithLp.ofLp (b 0) else WithLp.ofLp (∑ k, v i k • b k.succ)) := by
  rw [source_derivative_transfer m b z0 hL coeff hcoeff 4 (by omega)]
  congr 1
  funext i
  cases axial i <;> simp [frame_axial, frame_transverse]

theorem source_torus_pullback (m : ℕ)
    (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1))) (z0 : Fin (m+1) → ℝ)
    (L : ℝ) [Fact (0 < L)] (coeff : SourceMode (m+1) → ℝ) (q : ℝ × E m) :
    (sourceField (m+1) L coeff ∘ affineFrame m b z0) q =
      torusSourceField (m+1) L coeff (CapFirstExit.torusCover L (m+1) (affineFrame m b z0 q)) := by
  exact (torusSourceField_pullback (m+1) L coeff (affineFrame m b z0 q)).symm

end I1PhysicalFrame

