import Specifications.Lifetime

/-!
A source-scoped countermodel for the abstract pair-law interface.
The field, elder interface and frame are parameters; no existence of them is
claimed here. The constant law is permitted by the existing interface but is
not asserted to be the geometric elder probability of any Gaussian field.
-/

namespace CBZ.LifetimePairLawCountermodel
open UniversalLaw.Spec.Lifetime

def oneLaw (L : ℝ) (G : GaussianFieldInterface 2 L)
    (I : SuperlevelBarInterface 2 L) : PairLawInterface 2 L G I where
  p := fun _ _ _ _ => 1
  p_nonneg := by intros; norm_num
  p_le_one := by intros; norm_num

theorem oneLaw_upper (L : ℝ) (G : GaussianFieldInterface 2 L)
    (I : SuperlevelBarInterface 2 L) (bLo bHi kLo kHi : ℝ) :
    ParentUniformSelectionA 2 L G I (oneLaw L G I) bLo bHi kLo kHi := by
  intro _ _ _ _ _
  refine ⟨1, by norm_num, 0, ?_⟩
  intro r hr hrs b hb k hk R
  simp [oneLaw]

theorem oneLaw_refutes_twoSided (L : ℝ) (G : GaussianFieldInterface 2 L)
    (I : SuperlevelBarInterface 2 L) (hL : 0 < L)
    (R : OrthonormalBasis (Fin 2) ℝ (E 2)) :
    ¬ PlanarTwoSidedRateE4 L G I (oneLaw L G I) 0 0 1 1 := by
  intro hypothetical
  have upper := oneLaw_upper L G I 0 0 1 1
  obtain ⟨c, hc, C, rStar, hrs, bound⟩ :=
    hypothetical upper hL (by norm_num) (by norm_num) (by norm_num)
  have hr : 0 < rStar / 2 := by linarith
  have hle : rStar / 2 ≤ rStar := by linarith
  have hbound : c * (rStar / 2) ^ 3 ≤ 0 := by
    simpa [oneLaw] using (bound (rStar / 2) hr hle 0 (by simp) 1 (by simp) R).1
  have hpositive : 0 < c * (rStar / 2) ^ 3 := mul_pos hc (pow_pos hr 3)
  linarith

#print axioms oneLaw_upper
#print axioms oneLaw_refutes_twoSided

end CBZ.LifetimePairLawCountermodel
