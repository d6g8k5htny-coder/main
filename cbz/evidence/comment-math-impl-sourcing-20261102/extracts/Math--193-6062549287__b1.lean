import PhysicalFrame

noncomputable section
open I1PhysicalFrame I4Foundation I4Weights I4Source I4Torus
open scoped BigOperators RealInnerProductSpace
namespace PhysicalFrameTests

abbrev std (m : ℕ) := EuclideanSpace.basisFun (Fin (m+1)) ℝ

example (m : ℕ) (t : ℝ) (y : E m) : frame m (std m) (t,y) = Fin.cons t (WithLp.ofLp y) := by
  rw [frame_apply]
  rfl

example (m : ℕ) (z0 : Fin (m+1) → ℝ) : affineFrame m (std m) z0 (0,0) = z0 :=
  affine_midpoint _ _ _

-- The actual two axial pin images have the supplied midpoint, for an arbitrary physical basis.
example (m : ℕ) (b : OrthonormalBasis (Fin (m+1)) ℝ (E (m+1)))
    (z0 : Fin (m+1) → ℝ) (r : ℝ) :
    (1/2:ℝ) • (affineFrame m b z0 (-r/2,0) + affineFrame m b z0 (r/2,0)) = z0 := by
  have hn : frame m b (-r/2,0) = -frame m b (r/2,0) := by
    rw [show ((-r/2,0) : ℝ × E m) = -(r/2,0) by simp [neg_div]]
    exact (frame m b).map_neg _
  rw [affine_apply,affine_apply,hn]
  ext i
  simp only [Pi.smul_apply,Pi.add_apply,Pi.neg_apply,smul_eq_mul]
  ring

example (t : ℝ) (y : E 0) : frame 0 (std 0) (t,y) 0 = t := by
  rw [frame_apply]
  rfl

example : frame 0 (std 0) (0,(0:E 0)) = 0 := by simp

example (b : OrthonormalBasis (Fin 3) ℝ (E 3)) (z0 : Fin 3 → ℝ) (y : E 2) :
    ‖WithLp.toLp 2 (affineFrame 2 b z0 (2,y) - z0)‖^2 = 4 + ‖y‖^2 := by
  simpa only [show (2:ℝ)^2 = 4 by norm_num] using physical_energy 2 b z0 2 y

example (b : OrthonormalBasis (Fin 3) ℝ (E 3)) : frame 2 b (1,0) = WithLp.ofLp (b 0) :=
  frame_axial _ _

example (b : OrthonormalBasis (Fin 3) ℝ (E 3)) (y : E 2) :
    frame 2 b (0,y) = WithLp.ofLp (y 0 • b 1 + y 1 • b 2) := by
  simpa [Fin.sum_univ_succ] using frame_transverse 2 b y

example (b : OrthonormalBasis (Fin 3) ℝ (E 3)) (y : E 2) :
    inner ℝ (WithLp.toLp 2 (frame 2 b (1,0))) (WithLp.toLp 2 (frame 2 b (0,y))) = 0 :=
  axial_transverse_orthogonal _ _ _

-- The constructed equivalence preserves PHYSICAL Euclidean energy, not the product sup norm.
example : ‖WithLp.toLp 2 (frame 1 (std 1) (1,EuclideanSpace.single 0 1))‖^2 ≠
    ‖((1:ℝ),(EuclideanSpace.single 0 (1:ℝ) : E 1))‖^2 := by
  intro he
  have h := physical_energy 1 (std 1) 0 1 (EuclideanSpace.single 0 1)
  simp only [affine_apply, zero_add, sub_zero] at h
  norm_num at h
  have hp : ‖((1:ℝ),(EuclideanSpace.single 0 (1:ℝ) : E 1))‖^2 = (1:ℝ) := by
    norm_num [Prod.norm_def]
  rw [hp] at he
  linarith

private theorem single_majorant (n : ℕ) (L : ℝ) (m : SourceMode n) :
    Summable (fun k => modeMajorant n L k * |((Pi.single m (1:ℝ)) : SourceMode n → ℝ) k|) := by
  have hs : Summable (fun k : SourceMode n => if k=m then modeMajorant n L m else 0) := by
    exact (hasSum_ite_eq m (modeMajorant n L m)).summable
  convert hs using 1
  funext k
  by_cases h : k=m <;> simp [h]

-- A nonconstant actual paired cosine mode; derivatives stay on the SAME coefficients and field.
def nonconstant : SourceMode 2 := some (⟨fun _ => 1,⟨0,by norm_num,
  fun j hj => (Fin.not_lt_zero j hj).elim⟩⟩,false)

example (q : ℝ × E 1) (v : Fin 2 → (ℝ × E 1)) :
    iteratedFDeriv ℝ 2 (sourceField 2 1 (Pi.single nonconstant 1) ∘ affineFrame 1 (std 1) 0) q v =
    iteratedFDeriv ℝ 2 (sourceField 2 1 (Pi.single nonconstant 1)) (affineFrame 1 (std 1) 0 q)
      (fun i => frame 1 (std 1) (v i)) :=
  source_derivative_transfer _ _ _ (by norm_num) _ (single_majorant 2 1 nonconstant) 2 (by omega) q v

example (q : ℝ × E 1) (y : E 1) :
    iteratedFDeriv ℝ 3 (sourceField 2 1 (Pi.single nonconstant 1) ∘ affineFrame 1 (std 1) 0) q
      ![(0,y),(1,0),(1,0)] =
    iteratedFDeriv ℝ 3 (sourceField 2 1 (Pi.single nonconstant 1)) (affineFrame 1 (std 1) 0 q)
      ![WithLp.ofLp (y 0 • std 1 1),WithLp.ofLp (std 1 0),WithLp.ofLp (std 1 0)] := by
  convert source_third_blocks 1 (std 1) 0 (by norm_num : (1:ℝ) ≠ 0)
    (Pi.single nonconstant 1) (single_majorant 2 1 nonconstant) q (fun _ => y) ![false,true,true] using 1 <;>
    congr 1 <;> funext i <;> fin_cases i <;> simp

example (q : ℝ × E 1) (y : E 1) :
    iteratedFDeriv ℝ 4 (sourceField 2 1 (Pi.single nonconstant 1) ∘ affineFrame 1 (std 1) 0) q
      ![(0,y),(1,0),(1,0),(1,0)] =
    iteratedFDeriv ℝ 4 (sourceField 2 1 (Pi.single nonconstant 1)) (affineFrame 1 (std 1) 0 q)
      ![WithLp.ofLp (y 0 • std 1 1),WithLp.ofLp (std 1 0),WithLp.ofLp (std 1 0),WithLp.ofLp (std 1 0)] := by
  convert source_fourth_blocks 1 (std 1) 0 (by norm_num : (1:ℝ) ≠ 0)
    (Pi.single nonconstant 1) (single_majorant 2 1 nonconstant) q (fun _ => y) ![false,true,true,true] using 1 <;>
    congr 1 <;> funext i <;> fin_cases i <;> simp

example (b : OrthonormalBasis (Fin 2) ℝ (E 2)) (z0 : Fin 2 → ℝ)
    (coeff : SourceMode 2 → ℝ) (q : ℝ × E 1) :
    (sourceField 2 1 coeff ∘ affineFrame 1 b z0) q =
      torusSourceField 2 1 coeff (CapFirstExit.torusCover 1 2 (affineFrame 1 b z0 q)) :=
  by
    let _ : Fact (0 < (1:ℝ)) := ⟨by norm_num⟩
    exact source_torus_pullback _ _ _ _ _ _

end PhysicalFrameTests

